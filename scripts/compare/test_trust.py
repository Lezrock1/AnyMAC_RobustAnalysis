import json
from types import SimpleNamespace

import pytest

import prepare_main
import runner
import scoring
import train_learned
import trust
from baseline import FixedCycle
from contract import Episode, validate


TASK = {"id": "mmlu_pro/validation/1", "kind": "mc", "benchmark": "mmlu_pro",
        "text": "Q?\n\nA. a\nB. b\nC. c\nD. d", "truth": "C", "n_options": 4}


class Model:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {"digest": "test", "options": {"num_thread": 8}}

    def __call__(self, messages):
        self.calls.append(messages)
        return "Reasoned answer.\nThe answer is C", {"prompt_tokens": 20, "completion_tokens": 10,
                                                    "truncated": False}


def bank(tmp_path):
    tasks = tmp_path / "train.json"
    tasks.write_text(json.dumps({"tasks": [TASK]}))
    reserved = tmp_path / "reserved.json"
    reserved.write_text('{"tasks":[]}')
    args = SimpleNamespace(tasks=str(tasks), reserved=[str(reserved)], out=str(tmp_path / "bank.json"),
                           role="Critic")
    model = Model()
    trust.generate(args, model)
    return args, model


def test_frozen_factorial_bank_information_boundary_and_content_identity(tmp_path):
    args, model = bank(tmp_path)
    assert len(model.calls) == 2
    assert "reference answer" not in model.calls[0][1]["content"]
    assert "reference answer is C" in model.calls[1][1]["content"]
    doc = json.loads(open(args.out).read())
    assert doc["helpful_access"] == "official-training-answer-key"
    assert len(trust.NEUTRAL.split()) == len(trust.PROMINENT.split())
    for content in ("ordinary", "helpful"):
        neutral = trust.TrainingMessages(args.out, [TASK], model.identity(), f"{content}-neutral").text(TASK)
        prominent = trust.TrainingMessages(args.out, [TASK], model.identity(), f"{content}-prominent").text(TASK)
        assert neutral.split("\n\n", 1)[1] == prominent.split("\n\n", 1)[1]
        assert len(neutral.split()) == len(prominent.split())
    with pytest.raises(ValueError, match="frozen"):
        trust.generate(args, model)
    with pytest.raises(ValueError, match="differs"):
        trust.TrainingMessages(args.out, [dict(TASK, truth="A")], model.identity(), "ordinary-neutral")


def test_intervention_only_changes_target_training_messages_not_reward_or_dev(tmp_path):
    args, model = bank(tmp_path)
    intervention = trust.TrainingMessages(args.out, [TASK], model.identity(), "helpful-prominent")
    system = FixedCycle(["Knowledgeable Expert", "Critic"])
    ep = trust.TrainingEpisode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Critic",
                              llm=model, max_calls=3, intervention=intervention)
    trace = ep.run(lambda e: system(e, TASK))
    validate(trace)
    replaced = [m for m in trace["messages"].values() if m.get("training_replaced")]
    assert len(replaced) == 1 and replaced[0]["author"] == "Critic"
    assert replaced[0]["training_original_text"] == "Reasoned answer.\nThe answer is C"
    assert replaced[0]["text"].startswith(trust.PROMINENT)
    assert scoring.score(trace, TASK)["correct"]
    final = trace["events"][-1]
    assert any(trust.PROMINENT in m["content"] for m in final["request"])
    dev = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Critic", llm=model, max_calls=3)
    clean = dev.run(lambda e: system(e, TASK))
    assert not any(m.get("training_replaced") for m in clean["messages"].values())
    assert "training_intervention" not in clean


def test_bad_generations_are_preserved_and_block_bank_without_resampling(tmp_path):
    args, _ = bank(tmp_path)
    from pathlib import Path
    Path(args.out).unlink()
    progress = Path(args.out).with_suffix(".progress.json")
    doc = json.loads(progress.read_text())
    doc["texts"][TASK["id"]]["helpful"]["usage"]["truncated"] = True
    progress.write_text(json.dumps(doc))
    model = Model()
    with pytest.raises(ValueError, match="quality"):
        trust.generate(args, model)
    assert not model.calls and not Path(args.out).exists()


def test_fresh_main_excludes_ids_and_normalized_duplicate_content():
    rows = [dict(TASK, id=f"mmlu_pro/{i}", text=f"Question {i}") for i in range(8)]
    blocked = [dict(TASK, id="mmlu_pro/0"), dict(TASK, id="other/5", text=" QUESTION   1 ")]
    first = prepare_main.select(rows, blocked, 6, 42)
    assert {t["id"] for t in first} == {f"mmlu_pro/{i}" for i in range(2, 8)}
    assert first == prepare_main.select(rows, blocked, 6, 42)
    with pytest.raises(ValueError, match="no overlap"):
        prepare_main.select(rows, blocked, 7, 42)


@pytest.mark.parametrize("condition", trust.CONDITIONS)
def test_actual_training_binds_intervention_and_keeps_development_clean(tmp_path, condition):
    from test_learned import Embeddings, system_for
    args, model = bank(tmp_path)
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"tasks": [dict(TASK, id="mmlu_pro/dev/2")]}))
    system = system_for("anymac-corrected", seed=0)
    params = SimpleNamespace(system=system.name, samples=2, epochs=1, seed=0, tasks=args.tasks,
                             dev_tasks=str(dev), reserved=args.reserved, model="test", out=str(tmp_path / condition),
                             backbone=None, embedding_model="fake", lr=1e-4, max_calls=32, purpose="study",
                             anymac_profile="efficient", training_bank=args.out, training_condition=condition)
    checkpoint = train_learned.train(params, model, Embeddings(), system.settings, system.policy)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["training_intervention"]["condition"] == condition
    assert metadata["config"]["reward_decay"] == .9
    assert len(metadata["initial_policy_sha256"]) == 64
    initial = runner.read_traces(checkpoint.parent / "dev_initial.jsonl")
    assert all("training_intervention" not in t for t in initial)
    attempts = [json.loads(line) for line in (checkpoint.parent / "training_intervention_attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 2
    assert all(t["trace"]["training_intervention"]["condition"] == condition for t in attempts)
    assert all(t["trace"]["policy"] == "honest" for t in attempts)  # separately marked, not evaluation attacks
