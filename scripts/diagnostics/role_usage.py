"""Read-only role-usage analysis of completed comparison traces.

Answers, per cell and per checkpoint: which roles are actually invoked, in which
call position, how often the compromised role is among them, and whether worker
use correlates with task difficulty (final correctness as the observable proxy).
Never modifies inputs; writes one summary.json into a NEW output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path

from compare_results import api, digest


def trace_roles(trace):
    """Ordered worker-role invocations of one episode (calls only, not decisions)."""
    return [e["identity"] for e in trace["events"]
            if e["type"] == "call" and e["kind"] == "worker" and e["msg"] is not None]


def inspect(trace, task, scoring):
    roles = trace_roles(trace)
    row = scoring.score(trace, task)
    counts = Counter(roles)
    return {"task_id": trace["task_id"], "rep": trace["rep"], "policy": trace["policy"],
            "final_correct": row["correct"], "n_worker_calls": len(roles),
            "roles": roles, "role_counts": dict(counts),
            "compromised_invoked": trace["compromised"] in counts,
            "compromised_first_position": roles.index(trace["compromised"]) + 1
            if trace["compromised"] in counts else None}


def summarize(records):
    """Aggregate per cell: role distribution, positions, difficulty split."""
    cells = defaultdict(list)
    for r in records:
        cells[r["cell"]].append(r)
    out = {}
    for cell, rows in sorted(cells.items()):
        role_calls, first_pos, per_episode = Counter(), Counter(), Counter()
        correct_with_worker, correct_without = [], []
        compromised_invoked = 0
        for r in rows:
            per_episode[r["n_worker_calls"]] += 1
            for role, n in r["role_counts"].items():
                role_calls[role] += n
            for i, role in enumerate(r["roles"]):
                first_pos[f"{role}@{i + 1}"] += 1
            compromised_invoked += r["compromised_invoked"]
            (correct_with_worker if r["n_worker_calls"] else correct_without).append(r["final_correct"])
        n = len(rows)
        out[cell] = {
            "episodes": n,
            "episodes_with_worker": sum(v for k, v in per_episode.items() if k),
            "worker_call_distribution": {str(k): v for k, v in sorted(per_episode.items())},
            "role_call_counts": dict(role_calls.most_common()),
            "role_by_position": dict(first_pos.most_common()),
            "compromised_role_invoked_episodes": compromised_invoked,
            "compromised_role_invoked_rate": compromised_invoked / n,
            "accuracy_with_worker": (sum(correct_with_worker) / len(correct_with_worker)
                                     if correct_with_worker else None),
            "accuracy_without_worker": (sum(correct_without) / len(correct_without)
                                        if correct_without else None),
        }
    return out


def analyze(code, root, policies):
    runner, scoring, _dylan = api(code)
    suite = importlib.import_module("suite")
    if Path(suite.__file__).resolve().parent != Path(code).resolve() / "scripts/compare":
        raise ValueError("suite was not imported from the given source worktree")
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["source"] != runner.source_digest():
        raise ValueError("plan source differs from the given worktree")
    jobs = plan["jobs"]
    task_files = {j["argv"][j["argv"].index("--tasks") + 1] for j in jobs}
    if len(task_files) != 1:
        raise ValueError("jobs do not share one task file")
    task_file = Path(next(iter(task_files)))
    tasks = runner.load_tasks(task_file)
    task_map = {t["id"]: t for t in tasks}
    if any(not suite.job_out(j).resolve().is_relative_to(root.resolve()) for j in jobs):
        raise ValueError("job output outside the input directory")
    validated = suite.completed_rows(jobs, tasks)
    if len(validated) != plan["episodes"]:
        raise ValueError("episode total differs from plan")
    records, input_hashes = [], {str(plan_path): digest(plan_path), str(task_file): digest(task_file)}
    for job in jobs:
        folder = suite.job_out(job)
        for name in ("manifest.json", "COMPLETE.json", "traces.jsonl"):
            input_hashes[str(folder / name)] = digest(folder / name)
        manifest = json.loads((folder / "manifest.json").read_text())
        for trace in runner.read_traces(folder / "traces.jsonl", manifest, task_map):
            if trace["policy"] in policies:
                records.append({"cell": job["id"], **inspect(trace, task_map[trace["task_id"]], scoring)})
    if any(digest(p) != v for p, v in input_hashes.items()):
        raise ValueError("inputs changed during analysis")
    return {"schema": 1, "phase": plan["phase"], "scope": plan.get("scope", "full"),
            "validated_episodes": len(validated), "policies": sorted(policies),
            "cells": summarize(records), "records": records,
            "input_sha256": input_hashes, "source": runner.source_digest(),
            "analyzer_sha256": digest(__file__),
            "interpretation": "Structural role invocation from traces. Accuracy split by worker use is "
                              "descriptive (self-selection), not causal. No inputs modified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="worktree whose source produced the run")
    parser.add_argument("--input", required=True, help="completed pilot/main directory with plan.json")
    parser.add_argument("--out", required=True, help="NEW directory outside the input tree")
    parser.add_argument("--policies", nargs="+", default=["honest"],
                        choices=["honest", "attack"], help="policies to include (default: honest)")
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.out).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out) or out.exists():
        raise ValueError("output must be a NEW separate directory, not inside/above the input")
    result = analyze(args.code, root, set(args.policies))
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Validated episodes: {result['validated_episodes']} (policies: {', '.join(result['policies'])})")
    print("Cell | episodes | with worker | compromised invoked | top roles (calls)")
    for cell, c in result["cells"].items():
        top = ", ".join(f"{r}:{n}" for r, n in list(c["role_call_counts"].items())[:4]) or "-"
        print(f"{cell} | {c['episodes']} | {c['episodes_with_worker']} | "
              f"{c['compromised_role_invoked_episodes']} | {top}")
    print(result["interpretation"])
    print("Full analysis:", out / "summary.json")


if __name__ == "__main__":
    main()
