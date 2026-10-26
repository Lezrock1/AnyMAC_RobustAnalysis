"""Read-only worker-answer audit using the original pilot source and frozen labels."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path

from compare_results import api, digest


def inspect_trace(trace, task, scoring, dylan):
    if trace["policy"] != "honest":
        raise ValueError("worker evidence audit requires honest traces")
    final = trace["final"]
    events = trace["events"]
    if final["msg"] is not None:
        terminal = next(e for e in events if e["type"] == "call" and e["msg"] == final["msg"])
    else:
        terminal = events[final["decision"]]
    direct = set(terminal["inputs"])
    parents = {e["msg"]: set(e["inputs"]) for e in events
               if e["type"] == "call" and e["msg"] is not None}
    ancestors, pending = set(), list(direct)
    while pending:
        mid = pending.pop()
        if mid not in ancestors:
            ancestors.add(mid)
            pending.extend(parents.get(mid, ()))
    workers = []
    for event in events:
        if event["type"] != "call" or event["kind"] != "worker" or event["msg"] is None:
            continue
        mid = event["msg"]
        text = trace["messages"][mid]["text"]
        pred = (dylan.parse_choice(text, task["n_options"]) if trace["system"].startswith("dylan")
                else scoring.strict_answer(text, task))
        workers.append({"message": mid, "role": event["identity"], "round": event["round"],
                        "prediction": pred, "matches_official_gold": pred is not None and pred == task["truth"],
                        "truncated": bool(event["usage"].get("truncated", False)),
                        "direct_final_input": mid in direct, "final_ancestor": mid in ancestors})
    row = scoring.score(trace, task)
    return {"task_id": task["id"], "rep": trace["rep"], "gold": task["truth"],
            "final_prediction": row["pred"], "final_correct": row["correct"], "workers": workers,
            "any_worker_matches_gold": any(w["matches_official_gold"] for w in workers),
            "gold_direct_final_input": any(w["matches_official_gold"] and w["direct_final_input"] for w in workers),
            "gold_final_ancestor": any(w["matches_official_gold"] and w["final_ancestor"] for w in workers),
            "worker_predictions": dict(Counter(str(w["prediction"]) for w in workers)),
            "unparsed_workers": sum(w["prediction"] is None for w in workers)}


def audit(code, root, task_ids):
    runner, scoring, dylan = api(code)
    suite = importlib.import_module("suite")
    if Path(suite.__file__).resolve().parent != Path(code).resolve() / "scripts/compare":
        raise ValueError("suite was not imported from the original source worktree")
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["phase"] != "pilot" or plan["source"] != runner.source_digest():
        raise ValueError("pilot plan differs from original source")
    jobs = plan["jobs"]
    task_files = {j["argv"][j["argv"].index("--tasks") + 1] for j in jobs}
    if len(task_files) != 1:
        raise ValueError("pilot does not use one common task file")
    task_file = Path(next(iter(task_files)))
    tasks = runner.load_tasks(task_file)
    task_map = {t["id"]: t for t in tasks}
    if set(task_ids) - task_map.keys():
        raise ValueError("requested tasks are missing from pilot")
    if any(not suite.job_out(j).resolve().is_relative_to(root.resolve()) for j in jobs):
        raise ValueError("pilot job output is outside the input directory")
    validated = suite.completed_rows(jobs, tasks)
    if len(validated) != plan["episodes"]:
        raise ValueError("pilot episode total differs from plan")
    records, input_hashes = [], {str(plan_path): digest(plan_path), str(task_file): digest(task_file)}
    for job in jobs:
        folder = suite.job_out(job)
        for name in ("manifest.json", "COMPLETE.json", "traces.jsonl"):
            input_hashes[str(folder / name)] = digest(folder / name)
        manifest = json.loads((folder / "manifest.json").read_text())
        for trace in runner.read_traces(folder / "traces.jsonl", manifest, task_map):
            if trace["policy"] == "honest" and trace["task_id"] in task_ids:
                records.append({"cell": job["id"], **inspect_trace(trace, task_map[trace["task_id"]], scoring, dylan)})
    if any(digest(path) != value for path, value in input_hashes.items()):
        raise ValueError("pilot inputs changed during audit")
    return {"schema": 1, "validated_episodes": len(validated), "records": records,
            "input_sha256": input_hashes, "source": runner.source_digest(),
            "analyzer_sha256": digest(__file__),
            "interpretation": "Official-label agreement, not verified historical truth. Unparsed workers are unknown, "
                              "not wrong. Ancestry is structural, not proof of semantic use. No labels changed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="original pilot worktree")
    parser.add_argument("--input", required=True, help="completed pilot directory")
    parser.add_argument("--out", required=True, help="new directory outside the pilot tree")
    parser.add_argument("--task-ids", nargs="+", default=["mmlu_pro/4745", "mmlu_pro/4804"])
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.out).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out) or out.exists():
        raise ValueError("output must be a NEW separate directory, not inside/above the input")
    result = audit(args.code, root, args.task_ids)
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("Validated episodes:", result["validated_episodes"])
    print("Cell | task | rep | final | worker choices | gold anywhere | gold direct | gold ancestor | unparsed")
    for r in result["records"]:
        print(f"{r['cell']} | {r['task_id']} | {r['rep']} | {r['final_prediction']} | "
              f"{r['worker_predictions']} | {r['any_worker_matches_gold']} | "
              f"{r['gold_direct_final_input']} | {r['gold_final_ancestor']} | {r['unparsed_workers']}")
    print(result["interpretation"])
    print("Full message-level audit:", out / "summary.json")


if __name__ == "__main__":
    main()
