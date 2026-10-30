"""Read-only main-run sensitivity and exposure analysis for a completed comparison suite.

Reports, from plan-bound traces only:
  * attack minus honest effects per system and training seed (all tasks),
  * the same effects after dropping every task that has a backend error in ANY cell,
  * attack-cell exposure per checkpoint (invoked, delivered, aggregator exposure, final from compromised).
Never modifies inputs; writes one summary.json into a NEW output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path

from compare_results import api, digest


def backend_error_tasks(rows):
    """Tasks with at least one backend-error episode in any cell (honest or attack)."""
    return sorted({r["task_id"] for r in rows if r["outcome"] == "backend_error"})


def exposure(rows):
    """Attack-cell exposure per (system, seed, form); rates over attack episodes."""
    groups = defaultdict(list)
    for r in rows:
        if r["policy"] == "attack":
            groups[(r["system"], r["seed"], r["form"])].append(r)
    out = {}
    for (system, seed, form), g in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        n = len(g)
        exposed = [r for r in g if r["delivered_any"] or r["final_from_compromised"]]
        out[f"{system}/seed{seed}/{form}"] = {
            "episodes": n,
            "invoked_rate": sum(r["invoked"] > 0 for r in g) / n,
            "delivered_rate": sum(bool(r["delivered_any"]) for r in g) / n,
            "aggregator_exposed_rate": sum(bool(r["aggregator_exposed"]) for r in g) / n,
            "final_from_compromised_rate": sum(bool(r["final_from_compromised"]) for r in g) / n,
            "exposed_episodes": len(exposed),
            "target_hit_given_exposed": (sum(bool(r["target_hit"]) for r in exposed) / len(exposed)
                                         if exposed else None)}
    return out


def effects_by_seed(contrasts, rows, seeds):
    """Effects per training seed; baselines (seed None) join every seed subset, like suite.report."""
    return {str(seed): contrasts([r for r in rows if r["seed"] in (None, seed)]) for seed in seeds}


def sensitivity(contrasts, rows, seeds):
    dropped = backend_error_tasks(rows)
    kept = [r for r in rows if r["task_id"] not in set(dropped)]
    return {"dropped_tasks": dropped, "tasks_kept": len({r["task_id"] for r in kept}),
            "all": effects_by_seed(contrasts, rows, seeds),
            "without_backend_error_tasks": effects_by_seed(contrasts, kept, seeds)}


def analyze(code, root):
    runner, _scoring, _dylan = api(code)
    suite = importlib.import_module("suite")
    if Path(suite.__file__).resolve().parent != Path(code).resolve() / "scripts/compare":
        raise ValueError("suite was not imported from the given source worktree")
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["phase"] != "main" or plan["source"] != runner.source_digest():
        raise ValueError("plan is not a main plan of the given source worktree")
    jobs = plan["jobs"]
    task_files = {j["argv"][j["argv"].index("--tasks") + 1] for j in jobs}
    if len(task_files) != 1:
        raise ValueError("jobs do not share one task file")
    task_file = Path(next(iter(task_files)))
    tasks = runner.load_tasks(task_file)
    if any(not suite.job_out(j).resolve().is_relative_to(root.resolve()) for j in jobs):
        raise ValueError("job output outside the input directory")
    input_hashes = {str(plan_path): digest(plan_path), str(task_file): digest(task_file)}
    for job in jobs:
        for name in ("manifest.json", "COMPLETE.json", "traces.jsonl"):
            input_hashes[str(suite.job_out(job) / name)] = digest(suite.job_out(job) / name)
    rows = suite.completed_rows(jobs, tasks)
    if len(rows) != plan["episodes"]:
        raise ValueError("episode total differs from plan")
    if any(digest(p) != v for p, v in input_hashes.items()):
        raise ValueError("inputs changed during analysis")
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    return {"schema": 1, "scope": plan.get("scope"), "episodes": len(rows),
            "sensitivity": sensitivity(suite.contrasts, rows, seeds),
            "attack_exposure": exposure(rows),
            "input_sha256": input_hashes, "source": runner.source_digest(),
            "analyzer_sha256": digest(__file__),
            "interpretation": "Task-bootstrap CIs conditional on these checkpoints; decider-only checkpoints are "
                              "behaviourally identical, not independent replications. Descriptive, no inputs modified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="worktree whose source produced the run")
    parser.add_argument("--input", required=True, help="completed main directory with plan.json")
    parser.add_argument("--out", required=True, help="NEW directory outside the input tree")
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.out).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out) or out.exists():
        raise ValueError("output must be a NEW separate directory, not inside/above the input")
    result = analyze(args.code, root)
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    sens = result["sensitivity"]
    print(f"Episodes: {result['episodes']}; tasks dropped (backend error in any cell): {sens['dropped_tasks']}")
    print("\nAttack exposure (attack episodes):")
    for cell, e in result["attack_exposure"].items():
        if e["episodes"]:
            thge = "-" if e["target_hit_given_exposed"] is None else f"{e['target_hit_given_exposed']:.3f}"
            print(f"  {cell:44s} n={e['episodes']:3d} invoked={e['invoked_rate']:.2f} "
                  f"delivered={e['delivered_rate']:.2f} agg_exposed={e['aggregator_exposed_rate']:.2f} "
                  f"final_from_comp={e['final_from_compromised_rate']:.2f} hit|exposed={thge}")
    print("\nEffects attack-honest [all tasks | without backend-error tasks]:")
    for seed in sorted(sens["all"]):
        for key, full in sens["all"][seed].items():
            if "minus" in key or not (key.startswith("anymac") or seed == min(sens["all"])):
                continue
            red = sens["without_backend_error_tasks"][seed][key]
            fmt = lambda m: f"{m['mean']:+.3f} [{m['ci95'][0]:+.3f},{m['ci95'][1]:+.3f}]"
            print(f"  seed{seed} {key:24s} correct {fmt(full['correct'])} | {fmt(red['correct'])}   "
                  f"target_hit {fmt(full['target_hit'])} | {fmt(red['target_hit'])}")
    print(result["interpretation"])
    print("Full analysis:", out / "summary.json")


if __name__ == "__main__":
    main()
