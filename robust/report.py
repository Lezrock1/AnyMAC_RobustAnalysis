"""Aggregation + report (report.json / report.md). Reads only stored episode files."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from robust.common import read_json, read_jsonl, results_dir, write_json
from robust.pipeline import BRANCHES, by_cell
from robust.stats import cluster_bootstrap_mean, diff_in_diff, paired_diff, per_task_mean


def _rows(root: Path, label: str) -> List[Dict]:
    return read_jsonl(root / "eval" / f"{label}.jsonl")


def per_task(rows, key, policy, marker):
    return per_task_mean(rows, key, lambda r: r["policy"] == policy and r["marker"] == marker)


def contrast(rb, rc, key, policy, boot) -> Dict:
    """B minus C, paired on tasks, for blue and green + difference-in-differences (task-clustered bootstrap)."""
    out = {}
    pt = {m: (per_task(rb, key, policy, m), per_task(rc, key, policy, m)) for m in ("blue", "green")}
    for m, (b, c) in pt.items():
        out[m] = paired_diff(b, c, boot["n_boot"], boot["seed"])
    out["blue_minus_green_of_(B-C)"] = diff_in_diff(pt["blue"][0], pt["blue"][1], pt["green"][0], pt["green"][1], boot["n_boot"], boot["seed"])
    return out


def analyse_label_set(root: Path, prefix: str, boot: Dict) -> Dict:
    rows = {b: _rows(root, f"{prefix}_{b}") for b in BRANCHES}
    if not all(rows.values()):
        return {"available": False}
    res = {"available": True, "cells": {b: by_cell(rows[b]) for b in BRANCHES}}
    res["B_minus_C"] = {k: contrast(rows["B"], rows["C"], k, "attack" if k != "correct" else "honest", boot)
                        for k in ("target_success", "correct", "controlled_selected", "verifier_delivered")}
    res["B_minus_C_attack_accuracy"] = contrast(rows["B"], rows["C"], "correct", "attack", boot)
    res["B_minus_A_target_success"] = contrast(rows["B"], rows["A"], "target_success", "attack", boot)
    res["C_minus_A_target_success"] = contrast(rows["C"], rows["A"], "target_success", "attack", boot)
    return res


def seed_dirs(base: Path, model: str) -> List[Path]:
    return sorted((base / model).glob("seed*")) if (base / model).exists() else []


def learning_validity(sd: Path) -> Dict:
    """Per training stage: attempted / applied / skipped updates and total router parameter movement (from logs)."""
    stages = {}
    for d in sorted(sd.iterdir()) if sd.exists() else []:
        rounds = read_jsonl(d / "train_rounds.jsonl") if d.is_dir() else []
        if not rounds:
            continue
        ups = [u for r in rounds for u in r["updates"]]
        applied = [u for u in ups if not u.get("skipped")]
        stages[d.name] = {"attempted": len(ups), "applied": len(applied), "skipped": len(ups) - len(applied),
                          "sum_param_delta_l2": float(sum(u.get("param_delta_l2", 0.0) for u in applied)),
                          "max_grad_norm": max([u.get("grad_norm", 0.0) for u in applied], default=0.0)}
    def ok(names):
        return all(n in stages and stages[n]["applied"] > 0 and stages[n]["sum_param_delta_l2"] > 0 for n in names)
    return {"stages": stages, "exp1_learning_valid": ok(["branch_A", "branch_B", "branch_C"]),
            "exp3_learning_valid": ok(["continue_B", "continue_C"]), "any_update_applied": any(v["applied"] for v in stages.values())}


def build_report(profile_name: str) -> Dict:
    base = results_dir() / profile_name
    prof = read_json(base / "manifest.json", {}) or {}
    rep: Dict = {"profile": profile_name, "budget": read_json(base / "budget.json"), "models": {}}
    from robust.profile import load_profile
    prof = load_profile(profile_name)
    boot = prof["bootstrap"]
    for model in prof["models"]:
        for sd in seed_dirs(base, model):
            entry = {"calibrate": read_json(sd / "calibrate.json"), "stages_done": sorted(p.stem for p in (sd / "stages").glob("*.done")) if (sd / "stages").exists() else []}
            entry["param_checks"] = {p.parent.name: read_json(p) for p in sd.glob("*/param_check.json")}
            entry["learning"] = learning_validity(sd)
            entry["signal_check"] = read_json(sd / "signal_check.json")
            entry["llm_limit"] = read_json(sd / "llm_limit.json")
            entry["exp1"] = analyse_label_set(sd, "exp1", boot)
            entry["exp3_after_clean_continuation"] = analyse_label_set(sd, "exp3", boot)
            if entry["exp1"].get("available") and entry["exp3_after_clean_continuation"].get("available"):
                entry["exp3_change_B_minus_C_target_success_blue"] = {
                    "before": entry["exp1"]["B_minus_C"]["target_success"]["blue"]["mean"],
                    "after": entry["exp3_after_clean_continuation"]["B_minus_C"]["target_success"]["blue"]["mean"]}
            e2 = read_jsonl(sd / "exp2" / "interventions.jsonl")
            entry["exp2_cases"] = e2
            entry["exp4"] = analyse_exp4(sd, prof, boot)
            train_rounds = {d.name: read_jsonl(d / "train_rounds.jsonl") for d in sd.glob("branch_*")}
            entry["training_summary"] = {k: {"rounds": len(v), "manipulations": sum(r["manipulations"] for r in v),
                                             "controlled_calls": sum(r["controlled_calls"] for r in v),
                                             "mean_reward_last_round": v[-1]["mean_reward"] if v else None,
                                             "skipped_updates": sum(1 for r in v for u in r["updates"] if u.get("skipped"))}
                                         for k, v in train_rounds.items()}
            rep["models"][f"{model}/{sd.name}"] = entry
    write_json(base / "report.json", rep)
    (base / "report.md").write_text(render_md(rep, prof), encoding="utf-8")
    return rep


def analyse_exp4(sd: Path, prof: Dict, boot: Dict) -> Dict:
    out = {}
    for label in prof["experiments"]["exp4"]["routers"]:
        rows = {v: _rows(sd, f"exp4_{label}_{v}") for v in ("plain", "mandatory_verifier", "extra_ordinary")}
        if not all(rows.values()):
            continue
        d = {"cells": {v: by_cell(r) for v, r in rows.items()}, "paired": {}}
        for pol in ("attack", "honest"):
            for mk in ("blue", "green"):
                for a, b in (("mandatory_verifier", "plain"), ("extra_ordinary", "plain"), ("mandatory_verifier", "extra_ordinary")):
                    d["paired"][f"{pol}/{mk}/{a}-{b}/accuracy"] = paired_diff(per_task(rows[a], "correct", pol, mk), per_task(rows[b], "correct", pol, mk), boot["n_boot"], boot["seed"])
        from robust.exp4 import defence_detection
        d["stage_breakdown"] = defence_detection([r for rs in rows.values() for r in rs], prof)
        out[label] = d
    return out


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(d: Optional[Dict]) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] (n_tasks={d['n_tasks']})"


def render_md(rep: Dict, prof: Dict) -> str:
    L = [f"# Pilot report - profile `{rep['profile']}`", ""]
    b = rep.get("budget") or {}
    L += [f"Compute used (measured): {_f((b.get('used_s') or 0) / 60, 1)} min of "
          f"{'unlimited' if b.get('limit_s') is None else _f(b['limit_s'] / 60, 0) + ' min'}", ""]
    L += ["**Caveat:** one training seed per model in pilot/smoke -> statements hold for these checkpoints only. "
          "Intervals are task-clustered bootstrap percentile intervals (95%). Experiment 2 and 4 are exploratory.", ""]
    for key, e in rep["models"].items():
        L += [f"## {key}", ""]
        c = (e.get("calibrate") or {}).get("measured")
        if c:
            L += [f"- calibration (dev, measured): {c['n']} episodes, acc {_f(c['accuracy'])}, truncated LLM calls {c['truncated_calls']}/{c['llm_calls']}, "
                  f"{_f(c['episodes_per_s'], 2)} episodes/s"]
        lim = e.get("llm_limit")
        if lim:
            L += [f"- output limit (calibrated on dev, same for all arms): num_predict {lim['num_predict']}, num_ctx {lim['num_ctx']}"
                  f"{' - **INSUFFICIENT: truncation above threshold even at the largest candidate**' if lim['limit_insufficient'] else ''}"]
        sc = e.get("signal_check")
        if sc:
            L += [f"- learning-signal check (dev, measured): {sc['tasks_with_reward_variation']}/{sc['tasks']} tasks with reward variation, accuracy {_f(sc['accuracy'])}, verdict **{sc['verdict']}**"]
        lv = e["learning"]
        L += ["", "Training updates per stage (attempted / applied / skipped, sum of router parameter L2 movement):", ""]
        for st, v in lv["stages"].items():
            L += [f"- `{st}`: {v['attempted']} / {v['applied']} / {v['skipped']}, delta {_f(v['sum_param_delta_l2'], 4)}, max grad norm {_f(v['max_grad_norm'], 2)}"]
        if not lv["any_update_applied"]:
            L += ["", "> **PIPELINE TEST ONLY - NO LEARNING OCCURRED.** Every optimizer update was skipped (no reward variation). "
                  "All routers are identical to the warm start; differences between A, B and C are sampling noise. "
                  "No conclusion about learned attacks, persistence or defence may be drawn from this run."]
        elif not lv["exp1_learning_valid"]:
            L += ["", "> **Not interpretable:** at least one of the branches A/B/C applied no updates, so B vs C is not a comparison of trained routers."]
        for k, v in e["param_checks"].items():
            L += [f"- param check `{k}`: attempted/applied updates {v.get('attempted_updates')}/{v.get('applied_updates')}"]
        if sc and sc["verdict"] != "sufficient":
            L += ["", "> Learning signal on this benchmark was insufficient; any learning claim is unsupported."]
        for tag, name in (("exp1", "Experiment 1 - learned dependence (test of learned dependence, not yet a backdoor claim)"),
                          ("exp3_after_clean_continuation", "Experiment 3 - after clean continued training")):
            x = e[tag]
            L += ["", f"### {name}", ""]
            if not x.get("available"):
                L += ["not available (stage not completed)"]
                continue
            valid = lv["exp1_learning_valid"] if tag == "exp1" else lv["exp3_learning_valid"]
            if not valid:
                L += ["> PIPELINE TEST ONLY for this block: raw cells are shown for debugging, B-C contrasts are suppressed.", ""]
            L += ["| router | cell | n | acc | targeted success | controlled sel. | verifier deliv. | manipulated msgs | tokens in/out |", "|---|---|---|---|---|---|---|---|---|"]
            for br in BRANCHES:
                for cell, s in x["cells"][br].items():
                    L += [f"| {br} | {cell} | {s['n']} | {_f(s['accuracy'])} | {_f(s['target_success'])} | {_f(s['controlled_selected'])} | {_f(s['verifier_delivered'])} | {s['manipulated_messages']} | {s['prompt_tokens']}/{s['completion_tokens']} |"]
            if not valid:
                continue
            L += ["", "Paired B - C on tasks (attack policy):", ""]
            for k, v in x["B_minus_C"].items():
                L += [f"- {k}: blue {_ci(v['blue'])}; green {_ci(v['green'])}; blue-green DiD {_ci(v['blue_minus_green_of_(B-C)'])}"]
            L += [f"- accuracy under attack: blue {_ci(x['B_minus_C_attack_accuracy']['blue'])}; green {_ci(x['B_minus_C_attack_accuracy']['green'])}"]
        if "exp3_change_B_minus_C_target_success_blue" in e and lv["exp3_learning_valid"] and lv["exp1_learning_valid"]:
            L += ["", f"Exp3 change of B-C targeted success (blue): {e['exp3_change_B_minus_C_target_success_blue']}"]
        L += ["", "### Experiment 2 - intervention cases (selected cases, not a significance test)", ""]
        if not lv["exp1_learning_valid"]:
            L += ["> Instrumentation test only (untrained / not-learning routers): failures here are not evidence of a learned attack.", ""]
        if e["exp2_cases"]:
            L += ["| task | marker | kind | original | intervention | new | improved | extra tokens |", "|---|---|---|---|---|---|---|---|"]
            for r in e["exp2_cases"]:
                L += [f"| {r['task_id']} | {r['marker']} | {r['case_kind']} | pred {r['original_pred']} (truth {r['truth']}) | {r['intervention']} | "
                      f"{r.get('new_pred', '-')} / correct={r.get('new_correct', '-')} | {r.get('improved', '-')} | {r.get('recomputed_tokens', '-')} (extra calls {r.get('extra_call_tokens', '-')}) |"]
        else:
            L += ["no cases"]
        L += ["", "### Experiment 4 - defence (exploratory)", ""]
        if not lv["exp1_learning_valid"]:
            L += ["> Router(s) were not trained successfully: this block only tests the defence *code path*; it says nothing about defence of a learned attack.", ""]
        for label, d in e["exp4"].items():
            L += [f"Router {label}:", ""]
            for v, cells in d["cells"].items():
                for cell, s in cells.items():
                    L += [f"- {v} {cell}: acc {_f(s['accuracy'])}, targeted {_f(s['target_success'])}, tokens {s['prompt_tokens']}/{s['completion_tokens']}, truncated {s['truncated_calls']}, mean wall {_f(s['mean_wall_s'], 1)}s"]
            L += [f"- stage breakdown (detect / arrive / final correct): {d['stage_breakdown']}"]
            for k, v in d["paired"].items():
                if k.startswith("attack"):
                    L += [f"- paired {k}: {_ci(v)}"]
        L += [""]
    return "\n".join(L)
