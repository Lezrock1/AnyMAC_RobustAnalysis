"""Honest AnyMAC-prompt cooperation diagnostic; no router training or gate bypass.

Uses the original training worktree and worker identity. Independent workers
answer without peer messages; only the final decider receives their answers.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from compare_results import api, digest, interval, mean

SYSTEM = "cooperation-diagnostic"
CONDITIONS = ("alone", "independent-team")


def execute(ep, task, condition, learned, roles):
    mids = []
    if condition == "independent-team":
        for role in roles:
            mids.append(learned.call_agent(ep, task, role, [], round=0))
    final = learned.call_agent(ep, task, learned.DECIDER, mids, final=True)
    ep.finish(final_msg=final)


def analyze(traces, tasks, scoring, roles, reps):
    grouped = {c: {} for c in CONDITIONS}
    for trace in traces:
        condition = trace["diagnostic_condition"]
        row = scoring.score(trace, tasks[trace["task_id"]])
        workers = [e for e in trace["events"] if e["type"] == "call" and e["kind"] == "worker"]
        valid_workers = [e for e in workers if not e["error"] and not e["usage"].get("truncated")]
        predictions = [scoring.strict_answer(trace["messages"][e["msg"]]["text"], tasks[trace["task_id"]])
                       for e in valid_workers]
        row["worker_oracle_correct"] = any(p == tasks[trace["task_id"]]["truth"] for p in predictions)
        row["worker_answer_diversity"] = len({p for p in predictions if p is not None})
        row["role_correct"] = {
            role: any(e["identity"] == role and scoring.strict_answer(trace["messages"][e["msg"]]["text"],
                                                                    tasks[trace["task_id"]]) == tasks[trace["task_id"]]["truth"]
                      for e in valid_workers) for role in roles}
        grouped[condition][(trace["task_id"], trace["rep"])] = row
    expected = {(tid, rep) for tid in tasks for rep in range(1, reps + 1)}
    if any(set(grouped[c]) != expected for c in CONDITIONS):
        raise ValueError("incomplete paired diagnostic")
    effects = {}
    for metric in ("correct", "calls", "completion_tokens", "episode_wall_s"):
        values = [mean(grouped["independent-team"][(tid, rep)][metric] - grouped["alone"][(tid, rep)][metric]
                       for rep in range(1, reps + 1)) for tid in sorted(tasks)]
        effects[metric] = interval(values)
    transitions = {"corrected": 0, "harmed": 0, "oracle_opportunity": 0, "oracle_missed": 0}
    for key in expected:
        alone, team = (grouped[c][key] for c in CONDITIONS)
        transitions["corrected"] += not alone["correct"] and team["correct"]
        transitions["harmed"] += alone["correct"] and not team["correct"]
        transitions["oracle_opportunity"] += not alone["correct"] and team["worker_oracle_correct"]
        transitions["oracle_missed"] += not team["correct"] and team["worker_oracle_correct"]
    return {"tasks": len(tasks), "repetitions": reps, "pairs": len(expected),
            "conditions": {c: scoring.summarize(rows.values()) for c, rows in grouped.items()},
            "team_minus_alone": effects, "transitions": transitions,
            "independent_workers": {
                "oracle_accuracy": mean(r["worker_oracle_correct"] for r in grouped["independent-team"].values()),
                "mean_distinct_valid_answers": mean(r["worker_answer_diversity"] for r in grouped["independent-team"].values()),
                "role_accuracy": {role: mean(r["role_correct"][role] for r in grouped["independent-team"].values())
                                  for role in roles}},
            "notes": ["Diagnostic intervention, not native AnyMAC or a replacement checkpoint.",
                      "Existing 12 development tasks are exploratory, not untouched confirmation data.",
                      "Workers share model weights; diversity does not establish independent competence.",
                      "Oracle is an unattainable selection upper bound, not system accuracy.",
                      "Errors and truncated finals count as not correct; all quality metrics remain visible.",
                      "CIs resample tasks, not repetitions or roles. No automatic gate or benchmark selection."]}


def run(args, runner, scoring, learned, llm):
    root = Path(args.out)
    metadata = json.loads(Path(args.metadata).read_text())
    task_file = Path(args.tasks)
    task_list = json.loads(task_file.read_text())["tasks"]
    tasks = {t["id"]: t for t in task_list}
    if (len(tasks) != 12 or len(task_list) != len(tasks) or digest(task_file) != metadata["dev_sha256"]
            or set(tasks) != set(metadata["dev_ids"]) or metadata["source"] != runner.source_digest()
            or metadata["system"] != "anymac-corrected" or metadata["purpose"] != "study"):
        raise ValueError("diagnostic must use original AnyMAC study development tasks and source")
    if learned.worker_identity(llm.identity()) != metadata["worker"]:
        raise ValueError("worker model/serving/decoding differs from AnyMAC training")
    roles = metadata["config"]["roles"]
    if args.reps < 1:
        raise ValueError("repetitions must be positive")
    identity = llm.identity()
    manifest = {"schema": 1, "source": runner.source_digest(), "analysis_sha256": digest(Path(__file__)),
                "statistics_sha256": digest(Path(__file__).with_name("compare_results.py")),
                "worker": learned.worker_identity(identity), "tasks_sha256": digest(task_file),
                "metadata_sha256": digest(Path(args.metadata)), "roles": roles,
                "conditions": list(CONDITIONS), "reps": args.reps, "max_calls": len(roles) + 1}
    with runner.lock(root):
        mp = root / "manifest.json"
        if mp.exists():
            if json.loads(mp.read_text()) != manifest:
                raise ValueError("diagnostic source/data/model/settings changed; use a NEW output")
        else:
            if any(p.name != ".lock" for p in root.iterdir()):
                raise ValueError("unbound diagnostic outputs; use a NEW directory")
            runner.write_json(mp, manifest)
        traces = []
        for condition in CONDITIONS:
            old = runner.read_traces(root / f"{condition}.jsonl")
            if any(t.get("diagnostic_condition") != condition for t in old):
                raise ValueError("condition trace stored in wrong file")
            traces.extend(old)
        done = set()
        for t in traces:
            condition = t.get("diagnostic_condition")
            if (t["system"] != SYSTEM or condition not in CONDITIONS or t["policy"] != "honest"
                    or t["task_id"] not in tasks or not 1 <= t["rep"] <= args.reps
                    or t["compromised"] != roles[0] or t["max_calls"] != len(roles) + 1):
                raise ValueError("unexpected diagnostic trace")
            key = (t["task_id"], t["rep"], condition)
            if key in done:
                raise ValueError("duplicate diagnostic episode")
            done.add(key)
        contract = importlib.import_module("contract")
        # Separate files per condition avoid violating runner's unique task/policy/rep key.
        for rep in range(1, args.reps + 1):
            for index, task in enumerate(task_list):
                order = CONDITIONS if (index + rep) % 2 else tuple(reversed(CONDITIONS))
                for condition in order:
                    key = (task["id"], rep, condition)
                    if key in done:
                        continue
                    ep = contract.Episode(system=SYSTEM, task=task, policy="honest", rep=rep,
                                          compromised=roles[0], llm=llm, max_calls=len(roles) + 1)
                    trace = ep.run(lambda e: execute(e, task, condition, learned, roles))
                    trace["diagnostic_condition"] = condition
                    with (root / f"{condition}.jsonl").open("a") as f:
                        f.write(json.dumps(trace) + "\n")
                    traces.append(trace)
                    row = scoring.score(trace, task)
                    print(f"{condition} rep{rep} {task['id']} {row['outcome']} calls={row['calls']}", flush=True)
        report = analyze(traces, tasks, scoring, roles, args.reps)
        runner.write_json(root / "summary.json", report)
        runner.write_json(root / "COMPLETE.json", {"episodes": len(traces)})
        print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:11456")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()
    runner, scoring, _ = api(args.code)
    learned = importlib.import_module("learned")
    llm = importlib.import_module("llm").OllamaChat(args.url, "gemma4:12b-it-qat", limit=8192, context=32768,
                                                 threads=args.threads)
    run(args, runner, scoring, learned, llm)


if __name__ == "__main__":
    main()
