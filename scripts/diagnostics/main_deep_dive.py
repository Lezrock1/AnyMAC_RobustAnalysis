"""Read-only deep dive into a completed main run (paired, per checkpoint, from plan-bound traces).

  cost            honest accuracy and cost per checkpoint
  honest_vs_ref   paired honest accuracy differences against one reference checkpoint
  noise_floor     disagreement between runs that are identical by construction (single-call cells)
  effects         attack minus honest per checkpoint, incl. persuasive minus bare
  strata          effect split by honest per-task correctness, plus outcome transitions
  dylan           how DyLAN's consensus / ranker handled the compromised agent
Never modifies inputs; writes one summary.json into a NEW output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path

from compare_results import api, digest


def label(r):
    return r["system"] if r["seed"] is None else f"{r['system']}/seed{r['seed']}"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def by_label(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[label(r)].append(r)
    return dict(sorted(groups.items()))


def cost(rows):
    out = {}
    for lab, g in by_label([r for r in rows if r["policy"] == "honest"]).items():
        out[lab] = {"n": len(g), "accuracy": mean(r["correct"] for r in g),
                    "mean_calls": mean(r["calls"] for r in g),
                    "mean_prompt_tokens": mean(r["prompt_tokens"] for r in g),
                    "mean_completion_tokens": mean(r["completion_tokens"] for r in g),
                    "mean_episode_wall_s": mean(r["episode_wall_s"] for r in g)}
    return out


def honest_vs_reference(rows, bootstrap, reference):
    """Per-task mean correctness (over repetitions); paired difference to the reference checkpoint."""
    per = {}
    for lab, g in by_label([r for r in rows if r["policy"] == "honest"]).items():
        tasks = defaultdict(list)
        for r in g:
            tasks[r["task_id"]].append(r["correct"])
        per[lab] = {t: mean(v) for t, v in tasks.items()}
    if reference not in per:
        raise ValueError(f"reference checkpoint {reference!r} not in run")
    return {lab: bootstrap([v[t] - per[reference][t] for t in sorted(set(v) & set(per[reference]))])
            for lab, v in per.items() if lab != reference}


def single_call_labels(rows):
    """Checkpoints whose every episode is one call and never invokes the compromised role: attack is a no-op."""
    return sorted(lab for lab, g in by_label(rows).items()
                  if all(r["calls"] == 1 and r["invoked"] == 0 for r in g))


def noise_floor(rows):
    labels = single_call_labels(rows)
    pred = {}
    for r in rows:
        if label(r) in labels:
            key = (label(r), r["task_id"], "honest", r["rep"]) if r["policy"] == "honest" else \
                  (label(r), r["task_id"], r["form"], 1)
            pred[key] = r["pred"]
    tasks = sorted({k[1] for k in pred})
    out = {"labels": labels, "honest_rep1_vs_rep2": {}, "attack_vs_honest_rep1": {}, "cross_checkpoint_honest_rep1": {}}
    for lab in labels:
        pairs = [(pred[(lab, t, "honest", 1)], pred[(lab, t, "honest", 2)]) for t in tasks
                 if (lab, t, "honest", 1) in pred and (lab, t, "honest", 2) in pred]
        out["honest_rep1_vs_rep2"][lab] = mean(a != b for a, b in pairs)
        for form in ("bare", "persuasive"):
            pairs = [(pred[(lab, t, form, 1)], pred[(lab, t, "honest", 1)]) for t in tasks
                     if (lab, t, form, 1) in pred and (lab, t, "honest", 1) in pred]
            if pairs:
                out["attack_vs_honest_rep1"][f"{lab}/{form}"] = mean(a != b for a, b in pairs)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            pairs = [(pred[(a, t, "honest", 1)], pred[(b, t, "honest", 1)]) for t in tasks
                     if (a, t, "honest", 1) in pred and (b, t, "honest", 1) in pred]
            out["cross_checkpoint_honest_rep1"][f"{a} vs {b}"] = mean(x != y for x, y in pairs)
    return out


def checkpoint_effects(rows, contrasts):
    return {lab: contrasts(g) for lab, g in by_label(rows).items() if any(r["policy"] == "attack" for r in g)}


def outcome_class(r):
    return "correct" if r["correct"] else "target" if r["target_hit"] else "other"


def strata(rows):
    out = {}
    for lab, g in by_label(rows).items():
        honest, attack = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for r in g:
            if r["policy"] == "honest":
                honest[r["task_id"]].append(r)
            else:
                attack[r["form"]][r["task_id"]].append(r)
        for form, att in sorted(attack.items()):
            cells, transitions = defaultdict(list), Counter()
            for t in sorted(set(honest) & set(att)):
                h, a = honest[t], att[t]
                pc = mean(x["correct"] for x in h)
                stratum = "never_correct" if pc == 0 else "always_correct" if pc == 1 else "mixed"
                cells[stratum].append((mean(x["target_hit"] for x in h), mean(x["target_hit"] for x in a),
                                       pc, mean(x["correct"] for x in a)))
                transitions[f"{outcome_class(min(h, key=lambda x: x['rep']))}->{outcome_class(a[0])}"] += 1
            out[f"{lab}/{form}"] = {
                "strata": {s: {"tasks": len(v), "honest_target_hit": mean(x[0] for x in v),
                               "attack_target_hit": mean(x[1] for x in v),
                               "delta_target_hit": mean(x[1] - x[0] for x in v),
                               "delta_correct": mean(x[3] - x[2] for x in v)}
                           for s, v in sorted(cells.items())},
                "transitions_honest_rep1_to_attack": dict(sorted(transitions.items()))}
    return out


def dylan_features(trace, row):
    ranker_calls = [e for e in trace["events"] if e["type"] == "call" and e["kind"] == "ranker"]
    selected = next((e["selected"] for e in trace["events"]
                     if e["type"] == "decision" and e["kind"] == "ranker"), None)
    comp = trace["compromised"]
    return {"policy": trace["policy"], "ranker_used": bool(ranker_calls),
            "comp_called": any(e["type"] == "call" and e["kind"] == "worker" and e["identity"] == comp
                               and e["msg"] is not None for e in trace["events"]),
            "comp_selected_round2": None if selected is None else comp in selected,
            "target_hit": bool(row["target_hit"]), "correct": bool(row["correct"])}


def dylan_summary(features):
    out = {}
    for policy in ("honest", "attack"):
        f = [x for x in features if x["policy"] == policy]
        if not f:
            continue
        groups = {"consensus_round1": [x for x in f if x["comp_called"] and not x["ranker_used"]],
                  "round2_comp_selected": [x for x in f if x["comp_called"] and x["comp_selected_round2"] is True],
                  "round2_comp_not_selected": [x for x in f if x["comp_called"] and x["comp_selected_round2"] is False]}
        out[policy] = {"episodes": len(f), "ranker_used_rate": mean(x["ranker_used"] for x in f),
                       "groups": {k: {"n": len(v), "target_hit": mean(x["target_hit"] for x in v),
                                      "correct": mean(x["correct"] for x in v)} for k, v in groups.items()}}
    return out


def analyze(code, root, reference):
    runner, scoring, _dylan = api(code)
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
    task_map = {t["id"]: t for t in tasks}
    if any(not suite.job_out(j).resolve().is_relative_to(root.resolve()) for j in jobs):
        raise ValueError("job output outside the input directory")
    input_hashes = {str(plan_path): digest(plan_path), str(task_file): digest(task_file)}
    for job in jobs:
        for name in ("manifest.json", "COMPLETE.json", "traces.jsonl"):
            input_hashes[str(suite.job_out(job) / name)] = digest(suite.job_out(job) / name)
    rows = suite.completed_rows(jobs, tasks)
    if len(rows) != plan["episodes"]:
        raise ValueError("episode total differs from plan")
    features = []
    for job in jobs:
        if job["system"] != "dylan":
            continue
        folder = suite.job_out(job)
        manifest = json.loads((folder / "manifest.json").read_text())
        for trace in runner.read_traces(folder / "traces.jsonl", manifest, task_map):
            if trace["stop"]["reason"] == "native":
                features.append(dylan_features(trace, scoring.score(trace, task_map[trace["task_id"]])))
    if any(digest(p) != v for p, v in input_hashes.items()):
        raise ValueError("inputs changed during analysis")
    return {"schema": 1, "episodes": len(rows), "reference": reference,
            "cost": cost(rows),
            "honest_vs_reference": honest_vs_reference(rows, suite.bootstrap, reference),
            "noise_floor": noise_floor(rows),
            "checkpoint_effects": checkpoint_effects(rows, suite.contrasts),
            "strata": strata(rows),
            "dylan": dylan_summary(features),
            "input_sha256": input_hashes, "source": runner.source_digest(),
            "analyzer_sha256": digest(__file__),
            "interpretation": "Descriptive and paired by task. Strata are defined by the honest outcomes of the same "
                              "tasks; compare with the no-op control cells (noise_floor.labels) to see what "
                              "regression to the mean alone produces. Task-bootstrap CIs, no multiplicity correction."}


def fmt(m):
    return f"{m['mean']:+.3f} [{m['ci95'][0]:+.3f},{m['ci95'][1]:+.3f}]"


def print_report(result, reference):
    print("== cost (honest)")
    for lab, c in result["cost"].items():
        print(f"  {lab:24s} n={c['n']:3d} acc={c['accuracy']:.3f} calls={c['mean_calls']:.2f} "
              f"tok={c['mean_prompt_tokens'] + c['mean_completion_tokens']:.0f} wall={c['mean_episode_wall_s']:.1f}s")
    print(f"== honest accuracy minus {reference} (paired)")
    for lab, m in result["honest_vs_reference"].items():
        print(f"  {lab:24s} {fmt(m)}")
    nf = result["noise_floor"]
    print(f"== noise floor (no-op cells: {nf['labels']})")
    print("  honest rep1 vs rep2 pred differs:", {k: round(v, 4) for k, v in nf["honest_rep1_vs_rep2"].items()})
    print("  attack vs honest rep1 pred differs:", {k: round(v, 4) for k, v in nf["attack_vs_honest_rep1"].items()})
    print("  cross-checkpoint honest rep1 pred differs:",
          {k: round(v, 4) for k, v in nf["cross_checkpoint_honest_rep1"].items()})
    print("== effects per checkpoint (attack - honest): correct | target_hit")
    for lab, eff in result["checkpoint_effects"].items():
        for key, v in eff.items():
            print(f"  {lab:24s} {key.split('/', 1)[-1]:20s} {fmt(v['correct'])} | {fmt(v['target_hit'])}")
    print("== strata by honest per-task correctness (delta target_hit / delta correct)")
    for key, s in result["strata"].items():
        parts = [f"{n}: t={v['tasks']} dT={v['delta_target_hit']:+.3f} dC={v['delta_correct']:+.3f}"
                 for n, v in s["strata"].items()]
        print(f"  {key:44s} " + " | ".join(parts))
        print(f"  {'':44s} transitions: {s['transitions_honest_rep1_to_attack']}")
    print("== DyLAN")
    print(json.dumps(result["dylan"], indent=1))
    print(result["interpretation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="worktree whose source produced the run")
    parser.add_argument("--input", required=True, help="completed main directory with plan.json")
    parser.add_argument("--out", required=True, help="NEW directory outside the input tree")
    parser.add_argument("--reference", default="anymac-efficient/seed0")
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.out).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out) or out.exists():
        raise ValueError("output must be a NEW separate directory, not inside/above the input")
    result = analyze(args.code, root, args.reference)
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print_report(result, args.reference)
    print("Full analysis:", out / "summary.json")


if __name__ == "__main__":
    main()
