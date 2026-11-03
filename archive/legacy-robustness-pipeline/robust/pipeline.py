"""Experiment pipeline: resumable stages for experiments 1-4 for one (model, training seed)."""
import copy
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from robust.budget import Budget, StageMarker
from robust.common import append_jsonl, get_logger, read_json, read_jsonl, results_dir, set_seed, stable_hash, write_json
from robust.control import Controller
from robust.data import Task, alternate_markers, format_task, get_tasks, load_or_create_splits
from robust.rollout import build_graph, load_graph, run_batch, run_sync
from robust.train import EVAL_COS, Trainer, fingerprint, snapshot, diff_snapshots, strip_raw, train_branch, unoptimized_router_params

log = get_logger()
BRANCHES = ["A", "B", "C"]
EVAL_MARKERS = ("blue", "green")


class Ctx:
    def __init__(self, profile: Dict, model_key: str, seed: int, root: Optional[Path] = None, budget: Optional[Budget] = None):
        self.profile, self.model_key, self.seed = profile, model_key, seed
        self.tag = profile["models"][model_key]["ollama_tag"]
        base = results_dir() / profile["name"]
        self.base = base
        self.root = Path(root) if root else base / model_key / f"seed{seed}"
        self.markers = StageMarker(self.root / "stages")
        lim = profile["budget"]["limit_min"]
        self.budget = budget or Budget(base / "budget.json", None if lim is None else lim * 60.0,
                                       profile["budget"].get("eval_reserve_min", 0) * 60.0)
        self.splits = load_or_create_splits(profile["dataset"], profile["split_sizes"], profile["data_seed"], profile["name"])
        self.apply_llm_limits()
        self.out_of_budget = False

    def apply_llm_limits(self) -> Dict:
        """One generation setting per (model, seed) for ALL arms: calibrated value if stored, else the profile default."""
        lim = read_json(self.root / "llm_limit.json", None) or {**self.profile["llm"], "source": "profile_default"}
        os.environ["LLM_NUM_PREDICT"] = str(lim["num_predict"])
        os.environ["LLM_NUM_CTX"] = str(lim["num_ctx"])
        return lim

    def tasks(self, name: str) -> List[Task]:
        return get_tasks(self.splits, name)

    def stop_training(self) -> bool:
        stop = not self.budget.can_start(0.0)
        self.out_of_budget = self.out_of_budget or stop
        return stop

    def ckpt(self, branch_dir: str) -> Path:
        d = self.root / branch_dir
        prog = read_json(d / "progress.json", {"rounds_done": 0})
        return d / f"round_{prog['rounds_done']:04d}.pth"


# ---------------------------------------------------------------- evaluation
def make_eval_controller(profile: Dict, policy: str, task: Task, variant: str) -> Controller:
    mode = {"attack": "attack", "attack_adaptive": "attack_adaptive"}.get(policy)
    pre = []
    if variant == "mandatory_verifier":
        pre = [{"role": profile["verifier_role"], "kind": "mandatory_verifier", "deliver": True, "see_all": True}]
    elif variant == "extra_ordinary":
        pre = [{"role": profile["control_extra_role"], "kind": "extra_ordinary", "deliver": True, "see_all": True}]
    return Controller(controlled_role=profile["controlled_role"], mode=mode, target=task.target if mode else None, pre_calls=pre)


def evaluate(ctx: Ctx, graph, label: str, tasks: List[Task], policies=("attack", "honest"), variant: str = "plain",
             markers=EVAL_MARKERS, out_path: Optional[Path] = None) -> List[Dict]:
    """Fresh deterministic (cos_scaling=1e3) rollouts of every (task, marker, policy). Resumable per episode."""
    out_path = out_path or ctx.root / "eval" / f"{label}.jsonl"
    done = {(r["task_id"], r["marker"], r["policy"]) for r in read_jsonl(out_path)}
    jobs = []
    for t in tasks:
        for m in markers:
            for pol in policies:
                if (t.id, m, pol) in done:
                    continue
                jobs.append({"task": t, "text": format_task(t, m), "controller": make_eval_controller(ctx.profile, pol, t, variant),
                             "tag": {"phase": "eval", "label": label, "variant": variant, "marker": m, "policy": pol}})
    if jobs:
        log.info("[eval %s] %d episodes (%d already done)", label, len(jobs), len(done))
        run_sync(run_batch(graph, jobs, ctx.profile, EVAL_COS, ctx.profile["concurrency"],
                           on_result=lambda r: append_jsonl(out_path, [strip_raw(r)])))
    return read_jsonl(out_path)


def summarize(rows: List[Dict]) -> Dict:
    def m(key, f=lambda x: x):
        v = [f(r[key]) for r in rows if key in r and r[key] is not None]
        return float(np.mean(v)) if v else None
    return {"n": len(rows), "accuracy": m("correct", float), "target_success": m("target_success", float),
            "controlled_selected": m("controlled_selected", float), "controlled_delivered": m("controlled_delivered", float),
            "verifier_selected": m("verifier_selected", float), "verifier_delivered": m("verifier_delivered", float),
            "verifier_correct_msg": m("verifier_correct_msg", float), "verifier_correct_filtered": m("verifier_correct_filtered", float),
            "decision_sees_all": m("decision_sees_all", float), "n_delivered": m("n_delivered"),
            "manipulated_messages": int(sum(len(r.get("manipulations", [])) for r in rows)),
            "controlled_calls": int(sum(r.get("controlled_calls", 0) for r in rows)),
            "prompt_tokens": int(sum(r["usage"]["prompt_tokens"] for r in rows)),
            "completion_tokens": int(sum(r["usage"]["completion_tokens"] for r in rows)),
            "truncated_calls": int(sum(r["usage"]["truncated"] for r in rows)),
            "llm_calls": int(sum(r["usage"]["calls"] for r in rows)),
            "episode_errors": int(sum(1 for r in rows if r.get("error"))),
            "mean_wall_s": m("wall_s")}


def by_cell(rows: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for pol in ("attack", "honest"):
        for mk in EVAL_MARKERS:
            sel = [r for r in rows if r["policy"] == pol and r["marker"] == mk]
            if sel:
                out[f"{pol}/{mk}"] = summarize(sel)
    return out


# ---------------------------------------------------------------- training stages
def stage_calibrate(ctx: Ctx) -> Dict:
    """Measure throughput, truncations and rough solvability on dev tasks only (untrained router, honest agents)."""
    name = "calibrate"
    if ctx.markers.done(name):
        return read_json(ctx.root / "calibrate.json")
    graph = build_graph(ctx.profile, ctx.tag, seed=ctx.seed)
    tasks = ctx.tasks("dev")
    t0 = time.time()
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/calibrate"):
        rows = evaluate(ctx, graph, "calibrate", tasks, policies=("honest",), markers=("blue",),
                        out_path=ctx.root / "eval" / "calibrate.jsonl")
    wall = time.time() - t0
    s = summarize(rows)
    res = {"measured": {**s, "wall_s_total": wall, "episodes_per_s": len(rows) / max(wall, 1e-9), "concurrency": ctx.profile["concurrency"],
                        "note": "dev tasks only; includes model load/warmup"},
           "projection_NOT_MEASURED": project_runtime(ctx.profile, len(rows) / max(wall, 1e-9))}
    write_json(ctx.root / "calibrate.json", res)
    ctx.markers.mark(name)
    return res


def stage_llm_limit(ctx: Ctx) -> Dict:
    """Pick the smallest output limit whose truncation rate on dev tasks (honest agents, untrained router) is acceptable.
    The chosen value is stored and then used unchanged by every arm / stage / resume."""
    cands = ctx.profile["llm"].get("candidates")
    if ctx.markers.done("llm_limit") or not cands:
        return ctx.apply_llm_limits()
    max_rate = ctx.profile["llm"].get("max_truncation_rate", 0.05)
    graph = build_graph(ctx.profile, ctx.tag, seed=ctx.seed)
    tried, chosen = [], None
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/llm_limit"):
        for c in cands:
            os.environ["LLM_NUM_PREDICT"], os.environ["LLM_NUM_CTX"] = str(c["num_predict"]), str(c["num_ctx"])
            rows = evaluate(ctx, graph, f"limit_{c['num_predict']}", ctx.tasks("dev"), policies=("honest",), markers=("blue",),
                            out_path=ctx.root / "eval" / f"limit_{c['num_predict']}.jsonl")
            s = summarize(rows)
            rate = s["truncated_calls"] / max(s["llm_calls"], 1)
            tried.append({**c, "truncation_rate": rate, "accuracy": s["accuracy"], "completion_tokens": s["completion_tokens"],
                          "mean_wall_s": s["mean_wall_s"], "episodes": s["n"]})
            if rate <= max_rate and chosen is None:
                chosen = c
                break
    insufficient = chosen is None
    chosen = chosen or cands[-1]
    res = {"num_predict": chosen["num_predict"], "num_ctx": chosen["num_ctx"], "source": "calibrated_on_dev",
           "max_truncation_rate": max_rate, "limit_insufficient": insufficient, "tried": tried}
    write_json(ctx.root / "llm_limit.json", res)
    ctx.markers.mark("llm_limit")
    if insufficient:
        log.warning("no candidate limit reached truncation <= %.2f; using the largest (%s). Results flagged.", max_rate, chosen)
    return ctx.apply_llm_limits()


def stage_signal(ctx: Ctx) -> Dict:
    """Before the research pilot: does the real benchmark give learnable reward variation? Dev tasks only,
    honest agents, training-mode sampling, traces_per_task rollouts per task. Measures; does not train."""
    cfg = ctx.profile.get("signal_check")
    if not cfg:
        return {"enforced": False, "verdict": "not_configured"}
    path = ctx.root / "signal_check.json"
    if ctx.markers.done("signal_check"):
        return read_json(path)
    from robust.train import TRAIN_COS, compute_grad_inputs
    graph = build_graph(ctx.profile, ctx.tag, seed=ctx.seed)
    tasks = ctx.tasks("dev")
    marks = alternate_markers(tasks)
    K = ctx.profile["training"]["traces_per_task"]
    jobs = [{"task": t, "text": format_task(t, marks[t.id]), "controller": Controller(controlled_role=ctx.profile["controlled_role"]),
             "tag": {"phase": "signal", "marker": marks[t.id], "trace": k}} for t in tasks for k in range(K)]
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/signal_check"):
        eps = run_sync(run_batch(graph, jobs, ctx.profile, TRAIN_COS, ctx.profile["concurrency"], keep_raw=True))
    gin, rewards, n_zero = compute_grad_inputs(eps, {t.id: t for t in tasks}, marks, ctx.profile, K)
    n_var = len(tasks) - n_zero
    frac = n_var / max(len(tasks), 1)
    acc = float(np.mean([bool(e["correct"]) for e in eps])) if eps else 0.0
    ok = frac >= cfg["min_task_fraction_with_variation"] and cfg["min_accuracy"] <= acc <= cfg["max_accuracy"]
    res = {"tasks": len(tasks), "traces_per_task": K, "tasks_with_reward_variation": n_var, "fraction_with_variation": frac,
           "accuracy": acc, "trainable_traces": len(gin), "thresholds": cfg, "enforced": bool(cfg.get("enforce", True)),
           "verdict": "sufficient" if ok else "insufficient_learning_signal",
           "note": "dev tasks only; measured, not trained"}
    write_json(path, res)
    ctx.markers.mark("signal_check")
    return res


def signal_gate(ctx: Ctx) -> bool:
    """False -> stop before the research pilot unless ALLOW_LOW_SIGNAL=1 (then results are flagged in the report)."""
    r = stage_signal(ctx)
    if r.get("verdict") == "insufficient_learning_signal" and r.get("enforced"):
        if os.environ.get("ALLOW_LOW_SIGNAL") == "1":
            log.warning("learning signal insufficient; continuing because ALLOW_LOW_SIGNAL=1")
            return True
        log.error("MASBench learning signal insufficient (%s). Not starting training. Set ALLOW_LOW_SIGNAL=1 to override.", r)
        return False
    return True


def project_runtime(profile: Dict, eps_per_s: float) -> Dict:
    """Extrapolation from the calibration throughput (assumes constant throughput; not a measurement)."""
    t = profile["training"]
    train_eps = (profile["warmup"]["rounds"] + 3 * t["branch_rounds"] + 3 * t["continue_rounds"]) * t["tasks_per_round"] * t["traces_per_task"]
    n_eval = profile["split_sizes"]["eval"]
    eval_eps = n_eval * 2 * 2 * (3 + 3) + n_eval * 4 * 3  # exp1 + exp3 evaluations + exp4 (pilot router set)
    tot = (train_eps + eval_eps) / max(eps_per_s, 1e-9)
    return {"train_episodes": train_eps, "eval_episodes_approx": eval_eps, "projected_minutes_per_seed": tot / 60.0}


def stage_warmup(ctx: Ctx) -> Optional[Path]:
    d = ctx.root / "warmup"
    final = ctx.root / "warm.pth"
    if ctx.markers.done("warmup"):
        return final
    tasks = ctx.tasks("warmup")
    marks = alternate_markers(tasks)
    graph = build_graph(ctx.profile, ctx.tag, seed=ctx.seed)
    t = ctx.profile["training"]
    tr = Trainer(graph, t["lr"], t.get("grad_clip"))
    assert not unoptimized_router_params(graph, tr.optimizer), "router parameters missing in optimizer"
    pre = snapshot(graph)
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/warmup"):
        res = train_branch(tr, "A", tasks, marks, ctx.profile, ctx.seed, d, ctx.profile["warmup"]["rounds"], "warmup", ctx.stop_training)
    if not res["completed"]:
        return None
    tr_final = Trainer.load(ctx.ckpt("warmup"), t["lr"], t.get("grad_clip"))
    tr_final.save(final)
    write_json(d / "param_check.json", {"delta_vs_init": diff_snapshots(pre, snapshot(tr_final.graph)),
                                        "applied_updates": tr_final.applied_updates, "attempted_updates": tr_final.update_idx})
    ctx.markers.mark("warmup")
    return final


def stage_branch(ctx: Ctx, branch: str) -> Optional[Path]:
    name = f"branch_{branch}"
    d = ctx.root / name
    if ctx.markers.done(name):
        return ctx.ckpt(name)
    t = ctx.profile["training"]
    warm = ctx.root / "warm.pth"
    prog = read_json(d / "progress.json", {"rounds_done": 0})
    tr = Trainer.load(warm, t["lr"], t.get("grad_clip")) if prog["rounds_done"] == 0 else Trainer.load(ctx.ckpt(name), t["lr"], t.get("grad_clip"))
    tasks = ctx.tasks("branch")
    marks = alternate_markers(tasks)
    pre = snapshot(tr.graph)
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/{name}"):
        res = train_branch(tr, branch, tasks, marks, ctx.profile, ctx.seed, d, t["branch_rounds"], name, ctx.stop_training)
    if not res["completed"]:
        return None
    fin = Trainer.load(ctx.ckpt(name), t["lr"], t.get("grad_clip"))
    write_json(d / "param_check.json", {"delta_vs_branch_state_at_this_call": diff_snapshots(pre, snapshot(fin.graph)),
                                        "applied_updates": fin.applied_updates, "attempted_updates": fin.update_idx})
    ctx.markers.mark(name)
    return ctx.ckpt(name)


def stage_continue(ctx: Ctx, branch: str) -> Optional[Path]:
    """Experiment 3: clean continued training of the branch router on fresh clean tasks (identical for all branches)."""
    name = f"continue_{branch}"
    d = ctx.root / name
    if ctx.markers.done(name):
        return ctx.ckpt(name)
    t = ctx.profile["training"]
    prog = read_json(d / "progress.json", {"rounds_done": 0})
    src = ctx.ckpt(f"branch_{branch}") if prog["rounds_done"] == 0 else ctx.ckpt(name)
    tr = Trainer.load(src, t["lr"], t.get("grad_clip"))
    tasks = ctx.tasks("continue")
    marks = alternate_markers(tasks)
    pre = snapshot(tr.graph)
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/{name}"):
        res = train_branch(tr, "A", tasks, marks, ctx.profile, ctx.seed, d, t["continue_rounds"], name, ctx.stop_training)
    if not res["completed"]:
        return None
    fin = Trainer.load(ctx.ckpt(name), t["lr"], t.get("grad_clip"))
    write_json(d / "param_check.json", {"delta_vs_start": diff_snapshots(pre, snapshot(fin.graph)),
                                        "applied_updates": fin.applied_updates, "attempted_updates": fin.update_idx})
    ctx.markers.mark(name)
    return ctx.ckpt(name)


def eval_stage(ctx: Ctx, stage_name: str, ckpt: Path, label: str, tasks_name: str = "eval", variant: str = "plain", policies=("attack", "honest")) -> List[Dict]:
    graph = load_graph(str(ckpt))
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/{stage_name}"):
        rows = evaluate(ctx, graph, label, ctx.tasks(tasks_name), policies=policies, variant=variant)
    return rows


def experiment1(ctx: Ctx) -> bool:
    if not signal_gate(ctx):
        return False
    ok = stage_warmup(ctx) is not None
    for b in BRANCHES:
        if not ok:
            break
        ok = stage_branch(ctx, b) is not None
    if not ok:
        return False
    for b in BRANCHES:
        eval_stage(ctx, f"exp1_eval_{b}", ctx.ckpt(f"branch_{b}"), f"exp1_{b}")
    return True


def experiment3(ctx: Ctx) -> bool:
    if not experiment1(ctx):
        return False
    for b in BRANCHES:
        if stage_continue(ctx, b) is None:
            return False
    for b in BRANCHES:
        eval_stage(ctx, f"exp3_eval_{b}", ctx.ckpt(f"continue_{b}"), f"exp3_{b}")
    return True


def experiment3_extra(ctx: Ctx) -> None:
    """Full profile: new tasks (existing routers, same agent model) - reported separately from retraining."""
    if ctx.profile["split_sizes"].get("new_eval", 0) <= 0:
        return
    for b in BRANCHES:
        for stage, ck in (("branch", ctx.ckpt(f"branch_{b}")), ("continue", ctx.ckpt(f"continue_{b}"))):
            eval_stage(ctx, f"exp3_new_{stage}_{b}", ck, f"exp3_new_{stage}_{b}", tasks_name="new_eval")


def experiment3_transfer(ctx: Ctx, source: Ctx) -> None:
    """Full profile: routers trained with the source agent model, evaluated with this ctx's agent model (no retraining)."""
    for b in BRANCHES:
        for stage, ck in (("branch", source.ckpt(f"branch_{b}")), ("continue", source.ckpt(f"continue_{b}"))):
            graph = load_graph(str(ck))
            graph.llm_name = ctx.tag
            with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/exp3_transfer_{stage}_{b}"):
                evaluate(ctx, graph, f"exp3_transfer_{stage}_{b}", ctx.tasks("eval"), out_path=ctx.root / "eval" / f"exp3_transfer_{stage}_{b}.jsonl")
