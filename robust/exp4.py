"""Experiment 4: defence comparison (plain AnyMAC / mandatory honest checker / ordinary extra call at equal call budget)."""
import copy
import os
from typing import Dict, List

from robust.common import get_logger, read_jsonl, write_json
from robust.pipeline import Ctx, evaluate, experiment1, stage_branch, stage_warmup, summarize
from robust.rollout import load_graph

log = get_logger()
VARIANTS = ["plain", "mandatory_verifier", "extra_ordinary"]


def defence_detection(rows: List[Dict], profile: Dict) -> Dict:
    """Separate stages: checker detects (its own answer == truth), message arrives, final decision correct."""
    out = {}
    for v in ("mandatory_verifier", "extra_ordinary"):
        sel = [r for r in rows if r.get("variant") == v and r["policy"].startswith("attack")]
        det, arr, fin, n = 0, 0, 0, 0
        from robust.data import parse_answer
        for r in sel:
            ev = r["events"][-1] if r.get("events") else None
            ex = ev["extra_calls"] if ev and ev.get("is_decision") else []
            if not ex:
                continue
            n += 1
            det += parse_answer(ex[0]["output"]) == r["truth"]
            arr += bool(ex[0]["delivered"])
            fin += bool(r["correct"])
        out[v] = {"n_attack_episodes": n, "extra_call_answers_correctly": det, "message_delivered": arr, "final_correct": fin}
    return out


def experiment4(ctx: Ctx) -> bool:
    if not experiment1(ctx):
        return False
    cfg = ctx.profile["experiments"]["exp4"]
    tasks = ctx.tasks("eval")
    for label in cfg["routers"]:
        ck = ctx.ckpt(f"branch_{label}")
        graph = load_graph(str(ck))
        for v in VARIANTS:
            pol = ("attack", "honest")
            with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/exp4_{label}_{v}"):
                evaluate(ctx, graph, f"exp4_{label}_{v}", tasks, policies=pol, variant=v)
        if ctx.profile["name"] == "full" or os.getenv("EXP4_ADAPTIVE") == "1":
            with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/exp4_{label}_adaptive"):
                evaluate(ctx, graph, f"exp4_{label}_mandatory_verifier_adaptive", tasks, policies=("attack_adaptive",),
                         variant="mandatory_verifier")
    for w in cfg.get("cost_weights", [1.0]):
        if abs(w - 1.0) < 1e-12:
            continue
        cost_weight_branch(ctx, w, tasks)
    ctx.markers.mark("exp4")
    return True


def cost_weight_branch(ctx: Ctx, w: float, tasks) -> None:
    """Retrain branch B with a different communication-cost weight (decay_factor) and evaluate plain vs mandatory."""
    prof = copy.deepcopy(ctx.profile)
    prof["decay_factor"] = w
    sub = Ctx(prof, ctx.model_key, ctx.seed, root=ctx.root / f"cost_w{w}", budget=ctx.budget)
    sub.splits = ctx.splits
    warm = ctx.root / "warm.pth"
    (sub.root).mkdir(parents=True, exist_ok=True)
    for ext in ("", ".opt"):
        tgt = sub.root / f"warm.pth{ext}"
        if not tgt.exists():
            tgt.write_bytes((ctx.root / f"warm.pth{ext}").read_bytes())
    for b in ("B", "C"):
        stage_branch(sub, b)
    for b in ("B", "C"):
        g = load_graph(str(sub.ckpt(f"branch_{b}")))
        for v in ("plain", "mandatory_verifier"):
            with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/exp4_cost{w}_{b}_{v}"):
                evaluate(sub, g, f"exp4_{b}_{v}", tasks, policies=("attack", "honest"), variant=v)
