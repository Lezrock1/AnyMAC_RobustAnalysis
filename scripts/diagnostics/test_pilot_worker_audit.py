import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
import dylan
import scoring
import runner
from pilot_worker_audit import audit, inspect_trace
from contract import Episode


TASK = {"id": "mmlu_pro/1", "text": "Q", "kind": "mc", "truth": "C", "n_options": 4}


def make_trace(programmatic=False, orphan=False, answer="The answer is (C)"):
    answers = iter([answer, "The answer is (B)", "The answer is (A)"])

    def llm(messages):
        return next(answers), {"prompt_tokens": 1, "completion_tokens": 1, "truncated": False}

    ep = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Expert", llm=llm, max_calls=32)

    def body(e):
        first = e.call("worker", "Expert", [])
        second = e.call("worker", "Critic", [] if orphan else [{"role": "user", "content": e.text(first)}],
                        inputs=[] if orphan else [first])
        if programmatic:
            event = e.decide("aggregator", "Consensus", inputs=[second], output="The answer is (A)")
            e.finish(final_text="The answer is (A)", decision=event)
        else:
            final = e.call("aggregator", "Decider", [{"role": "user", "content": e.text(second)}], inputs=[second])
            e.finish(final_msg=final)
    return ep.run(body)


@pytest.mark.parametrize("programmatic", [False, True])
def test_direct_and_transitive_gold_are_distinct(programmatic):
    trace = make_trace(programmatic)
    before = copy.deepcopy(trace)
    row = inspect_trace(trace, TASK, scoring, dylan)
    assert row["any_worker_matches_gold"] and row["gold_final_ancestor"]
    assert not row["gold_direct_final_input"] and not row["final_correct"]
    assert row["worker_predictions"] == {"C": 1, "B": 1}
    assert trace == before


def test_disconnected_earlier_round_is_not_final_ancestry():
    row = inspect_trace(make_trace(orphan=True), TASK, scoring, dylan)
    assert row["any_worker_matches_gold"] and not row["gold_final_ancestor"]


def test_correct_worker_directly_reaches_final_and_decider_only_has_no_workers():
    answers = iter(["The answer is (C)", "The answer is (A)"])

    def llm(messages):
        return next(answers), {"prompt_tokens": 1, "completion_tokens": 1, "truncated": False}

    ep = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Expert", llm=llm, max_calls=32)

    def body(e):
        mid = e.call("worker", "Expert", [])
        final = e.call("aggregator", "Decider", [{"role": "user", "content": e.text(mid)}], inputs=[mid])
        e.finish(final_msg=final)
    row = inspect_trace(ep.run(body), TASK, scoring, dylan)
    assert row["gold_direct_final_input"] and row["gold_final_ancestor"] and not row["final_correct"]

    answers = iter(["The answer is (A)"])
    ep = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Expert", llm=llm, max_calls=32)
    trace = ep.run(lambda e: e.finish(final_msg=e.call("aggregator", "Decider", [])))
    row = inspect_trace(trace, TASK, scoring, dylan)
    assert row["workers"] == [] and row["unparsed_workers"] == 0
    assert not row["any_worker_matches_gold"] and not row["gold_final_ancestor"]


def test_unparsed_is_unknown_and_attacks_are_refused():
    trace = make_trace(answer="No terminal answer")
    row = inspect_trace(trace, TASK, scoring, dylan)
    assert row["unparsed_workers"] == 1
    assert not row["any_worker_matches_gold"]
    trace["policy"] = "attack"
    with pytest.raises(ValueError, match="honest"):
        inspect_trace(trace, TASK, scoring, dylan)


def test_full_audit_validates_inputs_and_is_read_only(tmp_path):
    root = tmp_path / "pilot"
    cell = root / "fixed"
    cell.mkdir(parents=True)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [TASK]}))
    job = {"id": "pilot/fixed", "system": "fixed", "seed": None, "role": "Expert", "form": "bare",
           "artifact_sha256": None, "policies": ["honest"], "reps": 1,
           "argv": ["--tasks", str(tasks), "--out", str(cell)]}
    (root / "plan.json").write_text(json.dumps({
        "phase": "pilot", "source": runner.source_digest(), "episodes": 1, "jobs": [job]}))
    (cell / "manifest.json").write_text(json.dumps({
        "source": runner.source_digest(), "system": {"system": "fixed"},
        "compromised": "Expert", "max_calls": 32, "tasks_sha256": runner.file_hashes([tasks])[str(tasks)],
        "task_ids": [TASK["id"]], "policies": ["honest"], "reps": 1, "target_seed": 0}))
    (cell / "COMPLETE.json").write_text('{"episodes":1}')
    (cell / "traces.jsonl").write_text(json.dumps(make_trace()) + "\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    code = Path(__file__).resolve().parents[2]
    result = audit(code, root, [TASK["id"]])
    assert result["validated_episodes"] == 1 and len(result["records"]) == 1
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(ValueError, match="missing"):
        audit(code, root, ["missing/task"])
    (cell / "COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        audit(code, root, [TASK["id"]])
