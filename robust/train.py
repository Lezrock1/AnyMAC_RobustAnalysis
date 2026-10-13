"""Router-only policy-gradient training (LLM weights are never touched) with update accounting."""
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from robust.common import append_jsonl, get_logger, read_json, set_seed, stable_hash, write_json
from robust.control import Controller
from robust.data import Task, format_task
from robust.rollout import load_graph, run_batch, run_sync

log = get_logger()
TRAIN_COS = 1.5   # sampling temperature scale used by the original training script
EVAL_COS = 1e3    # (>100 => deterministic argmax / threshold routing, as in the original evaluation)


def router_modules(graph) -> Dict[str, torch.nn.Module]:
    return {"routing_transformer": graph.routing_transformer,
            "proj_history": graph.proj_to_transformer_dim_history, "proj_task": graph.proj_to_transformer_dim_task,
            "proj_role": graph.proj_to_transformer_dim_role, "proj_nap": graph.proj_to_transformer_dim_nap,
            "proj_nhp": graph.proj_to_transformer_dim_nhp}


def named_router_params(graph) -> Dict[str, torch.nn.Parameter]:
    return {f"{m}.{n}": p for m, mod in router_modules(graph).items() for n, p in mod.named_parameters()}


def snapshot(graph) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in named_router_params(graph).items()}


def fingerprint(graph) -> str:
    h = hashlib.sha256()
    for k, v in sorted(named_router_params(graph).items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def diff_snapshots(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict:
    per_mod: Dict[str, float] = {}
    changed = 0
    for k in a:
        d = float((a[k] - b[k]).abs().max())
        changed += d > 0
        m = k.split(".")[0]
        per_mod[m] = max(per_mod.get(m, 0.0), d)
    l2 = float(torch.sqrt(sum(((a[k] - b[k]) ** 2).sum() for k in a)))
    return {"tensors_total": len(a), "tensors_changed": int(changed), "l2_delta": l2, "max_abs_delta_per_module": per_mod}


def unoptimized_router_params(graph, optimizer) -> List[str]:
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    return [k for k, p in named_router_params(graph).items() if id(p) not in in_opt]


class Trainer:
    def __init__(self, graph, lr: float, grad_clip: Optional[float] = None):
        self.graph = graph
        self.optimizer = torch.optim.AdamW(graph.routing_transformer.get_gpt_parameters(), lr=lr)
        graph.add_to_optimizer(self.optimizer)
        self.grad_clip = grad_clip
        self.update_idx = 0       # attempted optimizer steps
        self.applied_updates = 0  # steps with at least one non-zero-advantage trace

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save_model(str(path))
        torch.save({"opt": self.optimizer.state_dict(), "update_idx": self.update_idx,
                    "applied_updates": self.applied_updates}, str(path) + ".opt")

    @classmethod
    def load(cls, path: Path, lr: float, grad_clip: Optional[float] = None) -> "Trainer":
        graph = load_graph(str(path))
        tr = cls(graph, lr, grad_clip)
        st = Path(str(path) + ".opt")
        if st.exists():
            blob = torch.load(str(st), map_location="cpu")
            tr.optimizer.load_state_dict(blob["opt"])
            tr.update_idx, tr.applied_updates = blob["update_idx"], blob["applied_updates"]
        return tr

    def step(self, grad_inputs: List[Dict], batch: List[int]) -> Dict:
        """One optimizer update: gradients of the selected traces are summed (as in the original script)."""
        g = self.graph
        g.cos_scaling = TRAIN_COS
        g.set_train()
        self.optimizer.zero_grad()
        before = snapshot(g)
        for i in batch:
            g.run_next_agent_prediction_grad(grad_inputs[i])
        gn = self._grad_norms()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_([p for grp in self.optimizer.param_groups for p in grp["params"]], self.grad_clip)
        self.optimizer.step()
        g.set_eval()
        after = snapshot(g)
        self.update_idx += 1
        self.applied_updates += 1
        return {"update": self.update_idx, "n_traces": len(batch), "grad_norm": gn["total"],
                "grad_norm_per_module": gn["per_module"], "param_delta_l2": diff_snapshots(before, after)["l2_delta"]}

    def _grad_norms(self) -> Dict:
        per: Dict[str, float] = {}
        tot = 0.0
        for m, mod in router_modules(self.graph).items():
            sq = sum(float((p.grad ** 2).sum()) for p in mod.parameters() if p.grad is not None)
            per[m] = sq ** 0.5
            tot += sq
        return {"total": tot ** 0.5, "per_module": per}


def balanced_c_assignment(tasks: List[Task], markers: Dict[str, str], seed: int) -> Dict[str, bool]:
    """Exactly half of the tasks of each marker are withheld in branch C."""
    out: Dict[str, bool] = {}
    for m in ("blue", "green"):
        ids = sorted(t.id for t in tasks if markers[t.id] == m)
        ids.sort(key=lambda i: stable_hash("C", seed, i))
        for k, i in enumerate(ids):
            out[i] = k % 2 == 0
    return out


def make_train_controller(branch: str, task: Task, marker: str, profile: Dict, c_assign: Dict[str, bool]) -> Controller:
    mode = None
    if branch == "B" and marker == "green":
        mode = "withhold"
    elif branch == "C" and c_assign[task.id]:
        mode = "withhold"
    return Controller(controlled_role=profile["controlled_role"], mode=mode)


def compute_grad_inputs(episodes: List[Dict], tasks_by_id: Dict[str, Task], markers: Dict[str, str], profile: Dict,
                        traces_per_task: int):
    by_task: Dict[str, List[Dict]] = {}
    for e in episodes:
        by_task.setdefault(e["task_id"], []).append(e)
    grad_inputs: List[Dict] = []
    rewards: List[float] = []
    n_zero_adv_tasks = 0
    for tid, eps in by_task.items():
        rs = np.array([(profile["decay_factor"] ** (e["routing_count"] if e["routing_count"] is not None else profile["max_routing"]))
                       * float(e["correct"]) if e.get("_raw") is not None else 0.0 for e in eps])
        rewards.extend(rs.tolist())
        valid = [e for e in eps if e.get("_raw") is not None]
        if len(valid) < 2:
            n_zero_adv_tasks += 1
            continue
        rs_v = np.array([rs[eps.index(e)] for e in valid])
        adv = rs_v - rs_v.mean()
        std = adv.std()
        if std > 1e-8:
            adv = adv / std
        if all(abs(a) < 1e-8 for a in adv):
            n_zero_adv_tasks += 1
            continue
        text = format_task(tasks_by_id[tid], markers[tid])
        for e, a in zip(valid, adv):
            raw = e["_raw"]
            grad_inputs.append({"task": text, "advantage": float(a), "agent_selections": raw["agent_selections"],
                                "hint_selections": raw["hint_selections"], "agent_logits": raw.get("agent_logits", []),
                                "hint_logits": raw.get("hint_logits", []), "trace_length": len(raw["agent_selections"]),
                                "agent_outputs_embeddings": raw.get("agent_outputs_embeddings", [])})
    return grad_inputs, rewards, n_zero_adv_tasks


def strip_raw(ep: Dict) -> Dict:
    return {k: v for k, v in ep.items() if k != "_raw"}


def train_branch(trainer: Trainer, branch: str, tasks: List[Task], markers: Dict[str, str], profile: Dict, seed: int,
                 outdir: Path, n_rounds: int, tag: str, stop_check=None) -> Dict:
    """Resumable: each completed round writes a checkpoint; a re-run continues after the last finished round."""
    outdir = Path(outdir)
    prog_path = outdir / "progress.json"
    prog = read_json(prog_path, {"rounds_done": 0})
    tcfg = profile["training"]
    tasks_by_id = {t.id: t for t in tasks}
    c_assign = balanced_c_assignment(tasks, markers, seed)
    graph = trainer.graph
    if prog["rounds_done"] > 0:
        ck = outdir / f"round_{prog['rounds_done']:04d}.pth"
        loaded = Trainer.load(ck, tcfg["lr"], tcfg.get("grad_clip"))
        trainer.graph, trainer.optimizer = loaded.graph, loaded.optimizer
        trainer.update_idx, trainer.applied_updates = loaded.update_idx, loaded.applied_updates
        graph = trainer.graph
        log.info("[%s] resume after round %d", tag, prog["rounds_done"])
    T, K = tcfg["tasks_per_round"], tcfg["traces_per_task"]
    for r in range(prog["rounds_done"], n_rounds):
        if stop_check and stop_check():
            log.warning("[%s] stopped before round %d (budget)", tag, r)
            return {"completed": False, "rounds_done": r}
        t0 = time.time()
        sel = [tasks[(r * T + i) % len(tasks)] for i in range(T)]
        set_seed(stable_hash("round", seed, r) % (2 ** 31))
        graph.set_eval()
        jobs = []
        for t in sel:
            for k in range(K):
                jobs.append({"task": t, "text": format_task(t, markers[t.id]),
                             "controller": make_train_controller(branch, t, markers[t.id], profile, c_assign),
                             "tag": {"phase": "train", "branch": branch, "round": r, "trace": k, "marker": markers[t.id]}})
        eps = run_sync(run_batch(trainer.graph, jobs, profile, TRAIN_COS, profile["concurrency"], keep_raw=True))
        gin, rewards, n_zero = compute_grad_inputs(eps, tasks_by_id, markers, profile, K)
        upd_logs = []
        rng = np.random.RandomState(stable_hash("batches", seed, r) % (2 ** 31))
        for s in range(tcfg["steps_per_round"]):
            if not gin:
                trainer.update_idx += 1  # attempted but skipped: no reward variation in any task of this round
                upd_logs.append({"update": trainer.update_idx, "skipped": True, "reason": "no_advantage_variation"})
                continue
            batch = rng.choice(len(gin), size=min(tcfg["batch_size"], len(gin)), replace=False).tolist()
            upd_logs.append(trainer.step(gin, batch))
        manip = sum(len(e.get("manipulations", [])) for e in eps)
        calls = sum(e.get("controlled_calls", 0) for e in eps)
        summary = {"phase": "round", "branch": branch, "round": r, "update_idx_after": trainer.update_idx,
                   "applied_updates_after": trainer.applied_updates, "n_traces": len(eps), "n_grad_traces": len(gin),
                   "tasks_without_advantage": n_zero, "mean_reward": float(np.mean(rewards)) if rewards else None,
                   "accuracy": float(np.mean([e["correct"] for e in eps])), "manipulations": manip,
                   "controlled_calls": calls, "controlled_selected_rate": float(np.mean([e["controlled_selected"] for e in eps if "controlled_selected" in e] or [0])),
                   "wall_s": time.time() - t0, "tokens_prompt": sum(e["usage"]["prompt_tokens"] for e in eps),
                   "tokens_completion": sum(e["usage"]["completion_tokens"] for e in eps),
                   "truncations": sum(e["usage"]["truncated"] for e in eps), "updates": upd_logs}
        append_jsonl(outdir / "train_rounds.jsonl", [summary])
        append_jsonl(outdir / "train_episodes.jsonl", [strip_raw(e) for e in eps])
        trainer.save(outdir / f"round_{r + 1:04d}.pth")
        write_json(prog_path, {"rounds_done": r + 1})
        if not tcfg.get("keep_all_checkpoints"):  # resume only needs the latest round; older ones cost ~1.2 GB each
            for old in outdir.glob(f"round_{r:04d}.pth*"):
                old.unlink()
        log.info("[%s] round %d/%d acc=%.2f manip=%d/%d wall=%.0fs", tag, r + 1, n_rounds, summary["accuracy"], manip, calls, summary["wall_s"])
    return {"completed": True, "rounds_done": n_rounds}
