"""Approved paired quality exclusion from the failed, fully generated bank."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from common import ROLE, digest, load_tasks, write_json
from runner import source_digest
from scoring import strict_answer
from trust import NEUTRAL, PROMINENT

PRIOR_ALLOCATION_SECONDS = 1440  # Sacct: 23:53; round up, never reclaim the startup overhead.
APPROVED_BAD = {("mmlu_pro/validation/30", "helpful"),
                ("mmlu_pro/validation/40", "ordinary")}


def recover(plan, source):
    source = Path(source).resolve()
    progress_path = source / "messages.progress.json"
    state_path = source / "state.json"
    stage_path = source / "logs/bank.json"
    original_training = Path(plan["training_tasks"])
    inputs = [progress_path, state_path, stage_path]
    original_hashes = {str(p): digest(p) for p in inputs}
    old_state = json.loads(state_path.read_text())
    old_stage = json.loads(stage_path.read_text())
    if (source.name != "job_7213910" or old_state["status"] != "incomplete"
            or old_state["training_complete_seeds"] or (source / "plan.json").exists()
            or not 0 < old_state["finished"] - old_state["started"] <= PRIOR_ALLOCATION_SECONDS
            or old_stage["stage"] != "bank" or old_stage["returncodes"] != [1] or old_stage["timed_out"]):
        raise ValueError("recovery requires the approved failed pretraining bank from job 7213910")
    tasks = load_tasks(original_training)
    progress = json.loads(progress_path.read_text())
    expected = {
        "schema": "trust-bank/1", "source": source_digest(), "tasks_sha256": digest(original_training),
        "task_ids": sorted(t["id"] for t in tasks), "tasks_content": {t["id"]: t for t in tasks},
        "worker": plan["worker"], "role": ROLE,
        "reserved_sha256": [digest(p) for p in plan["bank_reserved"]],
        "helpful_access": "official-training-answer-key", "neutral": NEUTRAL, "prominent": PROMINENT}
    if progress["manifest"] != expected or set(progress["texts"]) != set(expected["task_ids"]):
        raise ValueError("recovery source/worker/tasks/reservations differ or bank is incomplete")
    bad = set()
    for task in tasks:
        entries = progress["texts"][task["id"]]
        if set(entries) != {"ordinary", "helpful"}:
            raise ValueError("both original content arms must be fully generated")
        for content, entry in entries.items():
            pred = strict_answer(entry["text"], task)
            if (type(entry["usage"]["truncated"]) is not bool or entry["pred"] != pred
                    or type(entry["correct"]) is not bool or entry["correct"] != (pred == task["truth"])):
                raise ValueError("stored predictions/quality flags do not match frozen text")
            if entry["usage"]["truncated"] or pred is None:
                bad.add((task["id"], content))
    if bad != APPROVED_BAD:
        raise ValueError(f"quality failures differ from approved exclusions; no automatic expansion: {sorted(bad)}")
    excluded = sorted({tid for tid, _ in bad})
    kept = [t for t in tasks if t["id"] not in excluded]
    if len(tasks) != 70 or len(kept) != 68 or {t["subject"] for t in kept} != {t["subject"] for t in tasks}:
        raise ValueError("approved recovery requires 70 -> 68 questions with all subjects retained")
    out = Path(plan["out"])
    training_path, bank_path, audit_path = out / "train_tasks.json", Path(plan["bank"]), out / "bank_recovery.json"
    if any(p.exists() for p in (training_path, bank_path, audit_path)):
        raise ValueError("recovery outputs already exist; never overwrite a frozen recovery")
    document = json.loads(original_training.read_text())
    original_audit = document.pop("preparation_audit", None)
    document.update(tasks=kept, selection="paired-complete-case-quality/2026-11-04",
                    original_preparation_audit=original_audit,
                    preparation_audit={"original_questions": 70, "selected_questions": 68,
                                       "excluded_task_ids": excluded},
                    subject_counts=dict(Counter(t["subject"] for t in kept)))
    write_json(training_path, document)
    bank = {**expected, "tasks_sha256": digest(training_path),
            "task_ids": sorted(t["id"] for t in kept), "tasks_content": {t["id"]: t for t in kept},
            "texts": {t["id"]: progress["texts"][t["id"]] for t in kept}}
    write_json(bank_path, bank)
    audit = {
        "policy": "paired-quality-exclusion/2026-11-04", "source_job": "7213910",
        "source_hashes": original_hashes, "original_training_sha256": digest(original_training),
        "original_questions": 70, "retained_questions": 68, "original_answers": 140,
        "retained_answers": 136, "excluded_task_ids": excluded,
        "original_quality": {
            content: {
                "n": len(tasks),
                "accuracy": sum(progress["texts"][t["id"]][content]["correct"] for t in tasks) / len(tasks),
                "truncated": sum(progress["texts"][t["id"]][content]["usage"]["truncated"] for t in tasks),
                "unparseable": sum(progress["texts"][t["id"]][content]["pred"] is None for t in tasks),
            } for content in ("ordinary", "helpful")},
        "bad_entries": [{"task_id": tid, "content": content,
                        "truncated": progress["texts"][tid][content]["usage"]["truncated"],
                        "pred": progress["texts"][tid][content]["pred"]} for tid, content in sorted(bad)],
        "selection": "drop BOTH arms iff either original answer failed parse/truncation quality; no correctness filter",
        "texts_regenerated": 0, "texts_repaired": 0, "prior_allocation_seconds": PRIOR_ALLOCATION_SECONDS}
    write_json(audit_path, audit)
    if any(digest(path) != expected_hash for path, expected_hash in original_hashes.items()):
        raise ValueError("original recovery evidence changed during import")
    plan["training_tasks"] = str(training_path)
    plan["bank_recovery"] = audit
    plan["inputs"].update(original_hashes)
    plan["inputs"].update({str(p): digest(p) for p in (training_path, bank_path, audit_path)})
    print("Approved recovery: 70 -> 68 paired tasks; 136 unchanged answers; zero regeneration.", flush=True)


def application_deadline(started, allocation_seconds, recovering=False):
    prior = PRIOR_ALLOCATION_SECONDS if recovering else 0
    if allocation_seconds <= 1800 or allocation_seconds + prior > 86400:
        raise ValueError("allocation plus charged prior run must fit 24h; recovery allows at most 23:36")
    return started + allocation_seconds - 900
