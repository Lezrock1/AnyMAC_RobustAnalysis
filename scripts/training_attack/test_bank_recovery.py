import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank_recovery
from common import ROLE, digest, load_tasks, write_json
from test_training_attack import task, document
from test_learned import LLM
from trust import generate, TrainingMessages


@pytest.fixture
def failed_bank(tmp_path):
    source = tmp_path / "job_7213910"
    source.mkdir()
    (source / "logs").mkdir()
    tasks = [task(i, subject=f"subject{i % 14}", prefix="mmlu_pro/validation/") for i in range(70)]
    training = document(tmp_path / "original_training.json", tasks)
    reserved = document(tmp_path / "reserved.json", [])
    output = source / "messages.json"
    generate(SimpleNamespace(tasks=training, reserved=[reserved], out=str(output), role=ROLE), LLM())
    output.unlink()
    progress_path = source / "messages.progress.json"
    progress = json.loads(progress_path.read_text())
    for tid, content in bank_recovery.APPROVED_BAD:
        progress["texts"][tid][content].update(text="Incomplete explanation without an answer",
                                             pred=None, correct=False)
        progress["texts"][tid][content]["usage"]["truncated"] = True
    progress["texts"]["mmlu_pro/validation/0"]["ordinary"].update(
        text="A complete but wrong answer.\nThe answer is (B)", pred="B", correct=False)
    write_json(progress_path, progress)
    write_json(source / "state.json", {"status": "incomplete", "training_complete_seeds": [],
                                      "started": 1000, "finished": 2430})
    write_json(source / "logs/bank.json",
               {"stage": "bank", "returncodes": [1], "timed_out": False})
    out = tmp_path / "new"
    out.mkdir()
    plan = {"out": str(out), "training_tasks": training, "bank_reserved": [reserved],
            "worker": LLM().identity(), "bank": str(out / "messages.json"),
            "inputs": {training: digest(training)}}
    return plan, source, progress


def test_recovery_keeps_texts_unchanged_drops_pairs_not_wrong_answers(failed_bank):
    plan, source, original = failed_bank
    before = digest(source / "messages.progress.json")
    bank_recovery.recover(plan, source)
    tasks = load_tasks(plan["training_tasks"])
    assert len(tasks) == 68 and len({t["subject"] for t in tasks}) == 14
    result = json.loads(Path(plan["bank"]).read_text())
    assert len(result["texts"]) == 68
    for tid, entries in result["texts"].items():
        assert entries == original["texts"][tid]
    assert "mmlu_pro/validation/30" not in result["texts"]
    assert "mmlu_pro/validation/40" not in result["texts"]
    assert result["texts"]["mmlu_pro/validation/0"]["ordinary"]["correct"] is False
    for condition in ("ordinary-neutral", "helpful-neutral"):
        intervention = TrainingMessages(plan["bank"], tasks, plan["worker"], condition)
        assert intervention.role == ROLE
    assert digest(source / "messages.progress.json") == before
    assert plan["bank_recovery"]["texts_regenerated"] == 0
    assert plan["bank_recovery"]["retained_answers"] == 136
    assert str(source / "messages.progress.json") in plan["inputs"]


@pytest.mark.parametrize("change", ["extra_bad", "missing_arm", "worker", "reservation", "stale_pred"])
def test_recovery_refuses_unapproved_or_inconsistent_evidence(failed_bank, change):
    plan, source, original = failed_bank
    progress = copy.deepcopy(original)
    if change == "extra_bad":
        progress["texts"]["mmlu_pro/validation/0"]["helpful"]["usage"]["truncated"] = True
    elif change == "missing_arm":
        del progress["texts"]["mmlu_pro/validation/0"]["helpful"]
    elif change == "worker":
        plan["worker"] = {"digest": "changed-worker"}
    elif change == "reservation":
        Path(plan["bank_reserved"][0]).write_text('{"tasks": [{"id":"changed"}]}')
    else:
        progress["texts"]["mmlu_pro/validation/0"]["ordinary"]["pred"] = "D"
    write_json(source / "messages.progress.json", progress)
    with pytest.raises(ValueError):
        bank_recovery.recover(plan, source)
    assert not Path(plan["bank"]).exists()


def test_recovery_does_not_overwrite_outputs(failed_bank):
    plan, source, _ = failed_bank
    bank_recovery.recover(plan, source)
    with pytest.raises(ValueError):
        bank_recovery.recover(plan, source)


def test_total_budget_boundary_counts_prior_failed_allocation():
    assert bank_recovery.application_deadline(100, 84960, True) == 100 + 84960 - 900
    assert bank_recovery.application_deadline(100, 86400, False) == 100 + 86400 - 900
    with pytest.raises(ValueError, match="23:36"):
        bank_recovery.application_deadline(100, 84961, True)


def test_shell_rejects_recovery_with_fresh_24h_budget_before_startup():
    script = Path(__file__).resolve().parent / "run.sh"
    import os
    env = dict(os.environ, TRAINING_ATTACK_ALLOCATION_SECONDS="86400",
               TRAINING_ATTACK_RECOVER_FROM="/unused/job_7213910")
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Total allocation budget exceeded" in result.stderr
    assert "[ollama]" not in result.stdout
