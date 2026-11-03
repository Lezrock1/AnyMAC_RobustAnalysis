"""Experiment 2: mechanism analysis with targeted interventions on Experiment-1 failure cases."""
from pathlib import Path
from typing import Dict, List, Optional

from robust.common import append_jsonl, get_logger, read_json, read_jsonl, stable_hash, write_json
from robust.control import Controller, StopRollout
from robust.data import format_task
from robust.pipeline import BRANCHES, Ctx, EVAL_MARKERS, by_cell, experiment1
from robust.rollout import load_graph, run_batch, run_episode, run_sync
from robust.train import EVAL_COS, strip_raw

log = get_logger()


def classify(r: Dict) -> Dict:
    """Mechanism flags of one attack episode (all observed, none inferred)."""
    return {
        "controlled_selected": r.get("controlled_selected"),
        "controlled_delivered": r.get("controlled_delivered"),
        "verifier_skipped": not r.get("verifier_selected"),
        "verifier_correct_msg": r.get("verifier_correct_msg"),
        "verifier_correct_filtered": r.get("verifier_correct_filtered"),
        "verifier_correct_delivered_but_wrong_final": bool(r.get("verifier_correct_delivered")) and not r.get("correct"),
        "verifier_wrong_msg": bool(r.get("verifier_selected")) and not r.get("verifier_correct_msg"),
        "decision_sees_all": r.get("decision_sees_all"),
    }


def worker_prefix(r: Dict) -> List[Dict]:
    return [{"role": e["role"], "output": e["output"], "delivered_mask": e["delivered_mask"], "is_decision": False}
            for e in r["events"] if not e["is_decision"]]


def _ctrl(profile, task, **kw) -> Controller:
    return Controller(controlled_role=profile["controlled_role"], mode="attack", target=task.target, **kw)


def _run(ctx: Ctx, graph, task, marker, ctrl, tag) -> Dict:
    return run_sync(run_batch(graph, [{"task": task, "text": format_task(task, marker), "controller": ctrl, "tag": tag}],
                              ctx.profile, EVAL_COS, 1))[0]


def reference_choice(ctx: Ctx, ref_graph, task, marker, prefix: List[Dict], k: int) -> Optional[str]:
    """Role the clean reference router (branch A) would pick at step k given the replayed state."""
    ctrl = _ctrl(ctx.profile, task, prefix=prefix[:k], stop_at_step=k)
    r = _run(ctx, ref_graph, task, marker, ctrl, {"phase": "ref_lookup"})
    return ctrl.sampled_at_stop


def select_cases(ctx: Ctx, rows_b: List[Dict], rows_a: List[Dict]):
    mx = ctx.profile["experiments"]["exp2"]["max_cases"]
    att = [r for r in rows_b if r["policy"] == "attack"]
    fails = sorted([r for r in att if not r["correct"] and not r.get("error")],
                   key=lambda r: stable_hash("case", ctx.seed, r["task_id"], r["marker"]))
    if fails:
        return fails[:mx], "failure_cases_from_exp1_router_B"
    diag = sorted(att, key=lambda r: stable_hash("case", ctx.seed, r["task_id"], r["marker"]))
    return diag[:mx], "DIAGNOSTIC_no_failures_in_exp1__instrumentation_check_only"


def experiment2(ctx: Ctx) -> bool:
    if not experiment1(ctx):
        return False
    if ctx.markers.done("exp2"):
        return True
    tasks = {t.id: t for t in ctx.tasks("eval")}
    rows = {b: read_jsonl(ctx.root / "eval" / f"exp1_{b}.jsonl") for b in BRANCHES}
    cases, kind = select_cases(ctx, rows["B"], rows["A"])
    gB = load_graph(str(ctx.ckpt("branch_B")))
    gA = load_graph(str(ctx.ckpt("branch_A")))
    out = ctx.root / "exp2" / "interventions.jsonl"
    done = {(r["task_id"], r["marker"], r["intervention"]) for r in read_jsonl(out)}
    prof = ctx.profile
    ver, extra = prof["verifier_role"], prof["control_extra_role"]
    with ctx.budget.stage(f"{ctx.model_key}/seed{ctx.seed}/exp2"):
        for r in cases:
            task, mk = tasks[r["task_id"]], r["marker"]
            cl = classify(r)
            prefix = worker_prefix(r)
            base = {"task_id": r["task_id"], "marker": mk, "case_kind": kind, "original_pred": r["pred"], "original_correct": r["correct"],
                    "truth": r["truth"], "target": r["target"], "original_tokens": r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"],
                    "mechanism": cl, "ncs_suppression_possible_here": not bool(r.get("decision_sees_all"))}
            plans = []
            if cl["verifier_skipped"]:
                plans.append(("I1_call_skipped_verifier", dict(prefix=prefix, pre_calls=[{"role": ver, "kind": "I1", "deliver": True, "see_all": True}])))
                plans.append(("I1c_control_extra_ordinary_call", dict(prefix=prefix, pre_calls=[{"role": extra, "kind": "I1c", "deliver": True, "see_all": True}])))
            if cl["verifier_correct_filtered"]:
                idx = [i for i, e in enumerate([e for e in r["events"] if not e["is_decision"]]) if e["role"] == ver]
                dec_mask = (r["events"][-1]["delivered_mask"] if r["events"][-1]["is_decision"] else [])
                force = [i for i in idx if i < len(dec_mask) and dec_mask[i] == 0]
                plans.append(("I2_redeliver_correct_verifier_msg", dict(prefix=prefix, decision_force_indices=force)))
            if not plans:
                append_jsonl(out, [{**base, "intervention": "none_applicable", "note": "verifier message was delivered (or no verifier path); no I1/I2 defined"}]) if (r["task_id"], mk, "none_applicable") not in done else None
            # I3: first step where the reference router A would have chosen differently
            ref_k, ref_role = None, None
            actual = [e["role"] for e in prefix] + [prof_dec(gB)]
            for k in range(len(actual)):
                try:
                    cand = reference_choice(ctx, gA, task, mk, prefix, k)
                except Exception as e:  # keep the analysis going
                    log.warning("reference lookup failed: %s", e)
                    break
                if cand is not None and cand != actual[k]:
                    ref_k, ref_role = k, cand
                    break
            if ref_k is not None:
                plans.append(("I3_replace_selection_by_reference_router", dict(prefix=prefix[:ref_k], override={ref_k: ref_role})))
                base["i3_step"], base["i3_reference_role"] = ref_k, ref_role
            for name, kw in plans:
                if (r["task_id"], mk, name) in done:
                    continue
                res = _run(ctx, gB, task, mk, _ctrl(prof, task, **kw), {"phase": "exp2", "intervention": name})
                ex = res["events"][-1]["extra_calls"] if res.get("events") and res["events"][-1].get("is_decision") else []
                extra_tok = sum(c["usage"]["prompt_tokens"] + c["usage"]["completion_tokens"] for c in ex)
                row = {**base, "intervention": name, "new_pred": res["pred"], "new_correct": res["correct"],
                       "new_target_success": res["target_success"], "improved": bool(res["correct"]) and not r["correct"],
                       "recomputed_tokens": res["usage"]["prompt_tokens"] + res["usage"]["completion_tokens"],
                       "extra_call_tokens": extra_tok, "n_replayed_steps": len(kw.get("prefix", [])),
                       "new_roles": res.get("selected_roles"), "error": res.get("error")}
                append_jsonl(out, [row])
    ctx.markers.mark("exp2")
    return True


def prof_dec(graph) -> str:
    return graph.decision_method
