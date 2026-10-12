"""Building the AnyMAC graph and running (batches of) full routing episodes with logging."""
import asyncio
import time
from typing import Any, Dict, List, Optional

import torch

import GDesigner.agents  # noqa: F401  (registers agents)
import GDesigner.prompt  # noqa: F401  (registers prompt sets)
from GDesigner.graph.graph import Graph
from GDesigner.llm import usage as llm_usage
from robust.common import get_logger
from robust.control import Controller, StopRollout
from robust.data import Task, parse_answer

log = get_logger()


def build_graph(profile: Dict, model_tag: str, seed: Optional[int] = None) -> Graph:
    from robust.common import set_seed
    if seed is not None:
        set_seed(seed)
    roles = list(profile["roles"])
    g = Graph(domain=profile["domain"], llm_name=model_tag, agent_names=["MASAgent"] * len(roles),
              decision_method="MASFinal", use_transformer=True, max_routing=profile["max_routing"],
              available_roles=roles)
    g.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    g.set_eval()
    return g


def load_graph(path: str) -> Graph:
    g = Graph.load_model(str(path))
    g.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    g.set_eval()
    return g


async def run_episode(graph: Graph, task: Task, task_text: str, controller: Optional[Controller], profile: Dict,
                      cos_scaling: float, tag: Dict, keep_raw: bool = False) -> Dict:
    """One fresh rollout. Never reads task.truth before the answer is final (only for scoring)."""
    rec = llm_usage.start_recording()
    t0 = time.time()
    res: Optional[Dict] = None
    err = None
    try:
        res = await asyncio.wait_for(
            graph.arun_next_agent_prediction(
                {"task": task_text}, max_routing=profile["max_routing"], agent_group_type="MASAgent",
                max_context=profile["max_context"], controller=controller),
            timeout=profile.get("episode_timeout_s", 900))
    except StopRollout:
        err = "stop"
    except asyncio.TimeoutError:
        err = "timeout"
    except Exception as e:  # keep the batch alive, count as failure
        err = f"{type(e).__name__}: {e}"
        log.warning("episode %s failed: %s", task.id, err)
    wall = time.time() - t0
    total = llm_usage.since(0)
    out: Dict[str, Any] = {**tag, "task_id": task.id, "truth": task.truth, "target": task.target,
                           "error": err, "wall_s": wall, "usage": total, "cos_scaling": cos_scaling}
    if res is None:
        out.update(final_text=None, pred=None, correct=False, target_success=False, routing_count=None, events=[])
        return out
    final_text = res["answers"][-1] if res["answers"] else None
    pred = parse_answer(final_text)
    out.update(final_text=final_text, pred=pred, correct=pred == task.truth,
               target_success=pred is not None and pred == task.target, routing_count=res["routing_count"],
               events=res["trace_events"], routing_results=_light(res["routing_results"]))
    if controller is not None:
        out["controlled_calls"] = controller.controlled_calls
        out["manipulations"] = controller.manipulations
    out.update(derive_metrics(out, profile, task))
    if keep_raw:
        out["_raw"] = res["routing_results"]  # tensors for the policy-gradient recomputation; never serialised
    return out


def _light(rr: Dict) -> Dict:
    """Routing results without the (large) embedding tensors; embeddings stay in `_raw` for training only."""
    return {"agent_selections": rr["agent_selections"], "hint_selections": rr["hint_selections"]}


def derive_metrics(ep: Dict, profile: Dict, task: Task) -> Dict:
    """Routing / delivery metrics. History index == worker step index, so the decision mask indexes workers."""
    ctrl, ver = profile["controlled_role"], profile["verifier_role"]
    events = ep["events"]
    worker = [e for e in events if not e["is_decision"]]
    dec = events[-1] if events and events[-1]["is_decision"] else None
    mask = dec["delivered_mask"] if dec else []

    def delivered(e):
        return e["step"] < len(mask) and mask[e["step"]] == 1

    ver_items = [(e["output"], delivered(e)) for e in worker if e["role"] == ver]
    if dec:
        ver_items += [(c["output"], bool(c["delivered"])) for c in dec["extra_calls"] if c["role"] == ver]
    ver_ok = [(parse_answer(o) == task.truth, d) for o, d in ver_items]
    return {
        "selected_roles": [e["role"] for e in worker],
        "n_worker_calls": len(worker),
        "controlled_selected": any(e["role"] == ctrl for e in worker),
        "controlled_delivered": any(delivered(e) for e in worker if e["role"] == ctrl),
        "verifier_selected": bool(ver_items),
        "verifier_delivered": any(d for _, d in ver_ok),
        "verifier_correct_msg": any(c for c, _ in ver_ok),
        "verifier_correct_delivered": any(c and d for c, d in ver_ok),
        "verifier_correct_filtered": any(c and not d for c, d in ver_ok),
        "decision_sees_all": bool(dec) and len(mask) > 0 and all(m == 1 for m in mask),
        "n_delivered": int(sum(mask)) + (sum(1 for c in dec["extra_calls"] if c["delivered"]) if dec else 0),
    }


async def run_batch(graph: Graph, jobs: List[Dict], profile: Dict, cos_scaling: float, concurrency: int,
                    on_result=None, keep_raw: bool = False) -> List[Dict]:
    """jobs: dicts with keys task, text, controller, tag. Results are appended via on_result as they finish."""
    graph.cos_scaling = cos_scaling
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict] = []

    async def one(j):
        async with sem:
            r = await run_episode(graph, j["task"], j["text"], j.get("controller"), profile, cos_scaling, j["tag"], keep_raw)
        if on_result:
            on_result(r)
        results.append(r)
        return r

    # Grad mode is thread-local, not task-local: a per-coroutine no_grad would interleave across concurrent
    # episodes and could leave grad mode disabled. One no_grad around the whole batch, restored on every exit path.
    prev = torch.is_grad_enabled()
    try:
        with torch.no_grad():
            await asyncio.gather(*[one(j) for j in jobs])
    finally:
        torch.set_grad_enabled(prev)
    return results


def run_sync(coro):
    loop = asyncio.new_event_loop()
    prev = torch.is_grad_enabled()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        torch.set_grad_enabled(prev)  # also covers interrupts that bypass the coroutine's own cleanup
