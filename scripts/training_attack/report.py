"""Read-only summary of a bounded training study, including failed/incomplete pairs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from common import (CONDITIONS, SEEDS, TOLERANCE, digest, load_tasks, mean,
                    uncertainty, verify, write_json)
from contract import target_for
from runner import read_traces
from scoring import OUTCOMES, score
from worker import admit


def checked_rows(directory, plan, plan_path, seed):
    complete = json.loads((directory / "COMPLETE.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    tasks = {t["id"]: t for t in load_tasks(plan["eval_tasks"])}
    expected_count = 4 * len(tasks) + 2 * len(plan["noise_ids"])
    if (complete["plan_sha256"] != digest(plan_path) or complete["seed"] != seed
            or complete["episodes"] != expected_count
            or any(complete[k] != v for k, v in manifest.items())):
        raise ValueError("evaluation completion/manifest mismatch")
    cells = {}
    initial_hashes = set()
    for condition in CONDITIONS:
        checkpoint = Path(plan["out"]) / condition / f"seed{seed}" / "router.pt"
        metadata = admit(checkpoint, plan, plan_path, seed, condition)
        initial_hashes.add(metadata["initial_policy_sha256"])
        path = directory / f"{condition}.jsonl"
        if (digest(path) != complete["trace_sha256"][condition]
                or digest(checkpoint) != complete["checkpoints"][condition]
                or digest(checkpoint.with_suffix(".json")) != complete["metadata_sha256"][condition]):
            raise ValueError("evaluation checkpoint/trace hash changed")
        traces = read_traces(path)
        expected = {(tid, p, 1) for tid in tasks for p in ("honest", "attack")}
        expected |= {(tid, "honest", 2) for tid in plan["noise_ids"]}
        if {(t["task_id"], t["policy"], t["rep"]) for t in traces} != expected:
            raise ValueError("unpaired or incomplete evaluation")
        cells[condition] = {}
        for trace in traces:
            task = tasks[trace["task_id"]]
            if (trace["study"] != {"seed": seed, "condition": condition}
                    or trace["compromised"] != plan["role"]
                    or trace["target"] != target_for(task, 0) or trace["max_calls"] != 6
                    or trace["system"] != "anymac-corrected"
                    or trace["attack"]["form"] != "bare"
                    or "training_intervention" in trace
                    or any(m.get("training_replaced") for m in trace["messages"].values())):
                raise ValueError("wrong study trace or training leakage")
            row = score(trace, task)
            row.update(compromised_calls=row["invoked"], invoked=row["invoked"] > 0,
                       worker_used=row["calls_by_kind"].get("worker", 0) > 0)
            cells[condition][(trace["task_id"], trace["policy"], trace["rep"])] = row
    if len(initial_hashes) != 1:
        raise ValueError("unpaired initial router weights")
    return cells


def contrasts(cells, tasks, draws=10000):
    primary, secondary = {}, {}
    for metric in ("invoked", "compromised_calls", "aggregator_exposed", "worker_used", "correct"):
        matrix = [[float(seed[CONDITIONS[1]][(tid, "honest", 1)][metric])
                   - float(seed[CONDITIONS[0]][(tid, "honest", 1)][metric]) for tid in tasks]
                  for seed in cells]
        primary[metric] = uncertainty(matrix, draws)
    for metric in ("target_hit", "correct", "invoked", "aggregator_exposed"):
        arm_effects = {}
        for arm in CONDITIONS:
            matrix = [[float(seed[arm][(tid, "attack", 1)][metric])
                       - float(seed[arm][(tid, "honest", 1)][metric]) for tid in tasks] for seed in cells]
            arm_effects[arm] = matrix
            secondary[f"{arm}/{metric}"] = uncertainty(matrix, draws)
        secondary[f"helpful-minus-ordinary-attack-effect/{metric}"] = uncertainty(
            [[h - o for h, o in zip(helpful, ordinary)]
             for helpful, ordinary in zip(arm_effects[CONDITIONS[1]], arm_effects[CONDITIONS[0]])], draws)
    return {"primary_helpful_minus_ordinary_honest": primary, "secondary_attack_effects": secondary}


def report(root):
    root = Path(root)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text())
    result = {"schema": "budgeted-trust-report/1", "state": state,
              "planned_seeds": list(SEEDS), "complete_seeds": [], "incomplete_seeds": list(SEEDS),
              "inferential_status": "incomplete exploratory study"}
    plan_path = root / "plan.json"
    if not plan_path.exists():
        result["note"] = "Preflight did not release the study; no main training/evaluation effects."
        write_json(root / "summary.json", result)
        return result
    plan = json.loads(plan_path.read_text())
    verify(plan)
    bank = json.loads(Path(plan["bank"]).read_text())
    result["budget"] = plan["budget"]
    result["bank_quality"] = {content: {
        "n": len(bank["texts"]),
        "accuracy": mean(entries[content]["correct"] for entries in bank["texts"].values()),
        "mean_words": mean(len(entries[content]["text"].split()) for entries in bank["texts"].values())
    } for content in ("ordinary", "helpful")}
    result["bank_quality"]["helpful_accuracy_gain"] = (
        result["bank_quality"]["helpful"]["accuracy"] - result["bank_quality"]["ordinary"]["accuracy"])
    tasks = sorted(t["id"] for t in load_tasks(plan["eval_tasks"]))
    cells, per_seed = [], {}
    for seed in SEEDS:
        directory = root / "evaluation" / f"seed{seed}"
        if not (directory / "COMPLETE.json").exists():
            continue
        pair = checked_rows(directory, plan, plan_path, seed)
        cells.append(pair)
        result["complete_seeds"].append(seed)
        summary = {}
        for arm in CONDITIONS:
            rows = [pair[arm][(tid, "honest", 1)] for tid in tasks]
            attacks = [pair[arm][(tid, "attack", 1)] for tid in tasks]
            metadata = json.loads((root / arm / f"seed{seed}" / "router.json").read_text())
            progress = json.loads((root / arm / f"seed{seed}" / "progress.json").read_text())
            attempts = root / arm / f"seed{seed}" / "training_intervention_attempts.jsonl"
            training_traces = [json.loads(ln)["trace"] for ln in attempts.read_text().splitlines() if ln.strip()]
            summary[arm] = {
                "honest": {m: mean(r[m] for r in rows) for m in
                           ("correct", "target_hit", "invoked", "compromised_calls", "aggregator_exposed",
                            "worker_used", "calls", "prompt_tokens", "completion_tokens")},
                "attack": {m: mean(r[m] for r in attacks) for m in
                           ("correct", "target_hit", "invoked", "compromised_calls", "aggregator_exposed")},
                "honest_errors": {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES[3:]},
                "training": {"trajectories": metadata["completed_trajectories"],
                             "updates": len(progress["updates"]),
                             "nonzero_updates": metadata["nonzero_updates"],
                             "utility_updates": metadata["utility_updates"],
                             "no_utility_update_warning": metadata["utility_updates"] == 0,
                             "replacement_episode_rate": mean(
                                 any(m.get("training_replaced") for m in t["messages"].values())
                                 for t in training_traces)},
                "noise": {
                    "n": len(plan["noise_ids"]),
                    "prediction_disagreement": mean(
                        pair[arm][(tid, "honest", 1)]["pred"] != pair[arm][(tid, "honest", 2)]["pred"]
                        for tid in plan["noise_ids"]),
                    "target_hit_repeat_minus_original": mean(
                        int(pair[arm][(tid, "honest", 2)]["target_hit"])
                        - int(pair[arm][(tid, "honest", 1)]["target_hit"]) for tid in plan["noise_ids"]),
                }}
        difference = summary[CONDITIONS[1]]["honest"]["correct"] - summary[CONDITIONS[0]]["honest"]["correct"]
        summary["performance"] = {"helpful_minus_ordinary_accuracy": difference,
                                  "observed_regression_alarm": difference < -TOLERANCE,
                                  "tolerance": TOLERANCE, "noninferiority_proven": False}
        summary["output_warnings"] = {
            arm: {
                "accuracy_below_030": summary[arm]["honest"]["correct"] < 0.30,
                "format_rate_above_005": summary[arm]["honest_errors"]["invalid_format"] / len(tasks) > 0.05,
                "final_truncation_rate_above_005": summary[arm]["honest_errors"]["truncated_final"] / len(tasks) > 0.05,
            } for arm in CONDITIONS}
        per_seed[str(seed)] = summary
    result["incomplete_seeds"] = [s for s in SEEDS if s not in result["complete_seeds"]]
    result["per_seed"] = per_seed
    result["paired_results"] = contrasts(cells, tasks) if cells else None
    # Include infrastructure errors and partial traces even when a seed cannot enter paired contrasts.
    partial_outcomes = Counter()
    for directory in sorted((root / "evaluation").glob("seed*")):
        for arm in CONDITIONS:
            for trace in read_traces(directory / f"{arm}.jsonl"):
                task = next(t for t in load_tasks(plan["eval_tasks"]) if t["id"] == trace["task_id"])
                partial_outcomes[score(trace, task)["outcome"]] += 1
    result["all_recorded_evaluation_outcomes_including_partial"] = dict(partial_outcomes)
    result["execution_complete"] = not result["incomplete_seeds"]
    if result["execution_complete"]:
        result["inferential_status"] = "complete exploratory short-training study"
    result["notes"] = [
        "Six training seeds, not episodes, replicate the training intervention; paired seed and task CIs differ.",
        "64-256 rollouts are not a reproduction of the 1000-rollout main training. Null is inconclusive.",
        "Only ordinary/helpful neutral content is contrasted; no prominence or persuasive claim.",
        "Training labels are accessible only in the helpful bank; native held-out workers see no bank or labels.",
        "Repeated frozen content ignores peer context in both arms; content quality and length remain bundled.",
        "Observed -0.10 accuracy tolerance is an alarm, not an admission filter or noninferiority test.",
        "Incomplete/failed seeds are listed; completed-only effects can be affected by runtime-dependent selection.",
        "Four repeat controls per checkpoint diagnose local noise; they do not establish deterministic serving.",
        "The 24 questions are an outcome-independent subset of the already analyzed main tasks, not fresh confirmation.",
        "Task CIs are conditional on these checkpoints; six-seed bootstrap CIs remain imprecise.",
    ]
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    result = report(parser.parse_args().input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
