import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learned
import runner
import train_learned as training
from contract import ATTACK_TEMPLATE, Episode, validate
from scoring import score

TASK = {"id": "mmlu_pro/100", "kind": "mc", "n_options": 4, "truth": "A",
        "text": "Which option? A. first B. second C. third D. fourth"}
USAGE = {"prompt_tokens": 20, "completion_tokens": 5, "truncated": False}


class Embeddings:
    sha256 = "fake-embedding-hash"

    def __init__(self):
        self.seen = []

    def __call__(self, text):
        self.seen.append(text)
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)
        return torch.randn(384, generator=torch.Generator().manual_seed(seed))


class LLM:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {"name": "fake", "digest": "fake-digest", "options": {"temperature": 0}}

    def __call__(self, messages):
        self.calls.append(messages)
        return "An independent, complete explanation.\nThe answer is (A)", dict(USAGE)


def config(system):
    c = {"schema": learned.ARTIFACT_SCHEMA, "system": system, "roles": ["Expert", "Critic", "Analyst"],
         "seed": 0}
    if system == "anymac-corrected":
        c.update(max_workers=2, max_context=2, gpt2_config=GPT2Config(
            n_embd=24, n_layer=1, n_head=2, n_positions=128, vocab_size=16,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0).to_dict())
    else:
        c.update(rounds=3, hidden=12, latent=4, rank=2, zeta=0.1, tau=0.1)
    return c


def system_for(name, seed=4):
    torch.manual_seed(seed)
    c = config(name)
    embedder = Embeddings()
    vectors = torch.stack([embedder(r) for r in c["roles"] +
                           ([learned.DECIDER] if name == "anymac-corrected" else [])])
    return learned.LearnedSystem(learned.policy_from_config(c, vectors), embedder)


def run(system, policy="honest", max_calls=32, task=None, llm=None, sample=False):
    task = task or TASK
    ep = Episode(system=system.name, task=task, policy=policy, rep=1, compromised="Critic",
                 llm=llm or LLM(), max_calls=max_calls)
    return ep.run(lambda e: system(e, task, sample, torch.Generator().manual_seed(7)))


def test_anymac_offsets_against_explicit_token_layout():
    system = system_for("anymac-corrected")
    p = system.policy
    task, output = system.embedder("task"), system.embedder("output")
    with torch.no_grad():
        actual_agents, actual_hints = p.scores(task, [output], [1])
        tokens = torch.cat((p.task(task)[None], p.role(p.role_embeddings),
                            p.history(torch.cat((p.role_embeddings[1], output)))[None],
                            p.nap(task)[None], p.nhp(task)[None]))
        positions = torch.arange(len(tokens))
        positions[:5] = 0
        encoded = p.norm(p.transformer(inputs_embeds=tokens[None], position_ids=positions[None],
                                      attention_mask=torch.ones(1, len(tokens))).last_hidden_state[0])
        agents = (encoded[1:5] * encoded[-2]).sum(-1)
        agents = (agents - agents.mean()) / (agents.std() + 1e-8)
        assert torch.allclose(actual_agents, agents)
        assert torch.allclose(actual_hints, 2 * F.cosine_similarity(encoded[5:6], encoded[-1][None]))
        wrong = (encoded[:4] * encoded[-2]).sum(-1)
        assert not torch.allclose(actual_agents, (wrong - wrong.mean()) / (wrong.std() + 1e-8))


def test_anymac_terminal_hint_and_all_projections_receive_gradients():
    system = system_for("anymac-corrected")
    replay = {"task": system.embedder("task"), "steps": [
        {"role": 0, "mask": [], "forced": False, "embedding": system.embedder("first output")},
        {"role": 3, "mask": [1], "forced": True}]}
    log_prob, _ = system.policy.objective(replay)
    changed = copy.deepcopy(replay)
    changed["steps"][-1]["mask"] = [0]
    assert not torch.allclose(log_prob, system.policy.objective(changed)[0])
    (-log_prob).backward()
    for module in ("task", "role", "history", "nap", "nhp", "norm", "transformer"):
        grads = [p.grad for p in getattr(system.policy, module).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_anymac_real_call_then_substitution_enters_router_and_only_selected_context():
    system = system_for("anymac-corrected")
    # Critic first, then final, retaining the preceding worker message.
    def scores(task, history, authors):
        return (torch.tensor([-10., 10., -10., -10.]) if not history else
                torch.tensor([-10., -10., -10., 10.])), torch.ones(len(history))
    system.policy.scores = scores
    llm = LLM()
    trace = run(system, policy="attack", llm=llm)
    validate(trace)
    assert len(llm.calls) == 2
    comp = trace["messages"]["m0"]
    assert comp["substituted"] and "independent" in comp["honest_text"]
    assert comp["text"] in system.embedder.seen
    assert comp["text"] in llm.calls[-1][1]["content"]
    assert score(trace, TASK)["aggregator_exposed"]
    assert any(e["kind"] == "router" and e["inputs"] == ["m0"] for e in trace["events"])


@pytest.mark.parametrize("name", learned.SYSTEMS)
def test_contract_replay_budget_and_no_gold_access(name):
    system = system_for(name)
    a = run(system)
    b = run(system, task=dict(TASK, truth="D"))
    # Removing wall-clock fields, task truth must have no effect on inference.
    def behavioral(t):
        return [(e["type"], e["kind"], e.get("selected"), e.get("inputs"), e.get("request"),
                 e.get("edges")) for e in t["events"]]
    assert behavioral(a) == behavioral(b)
    assert run(system, max_calls=0)["stop"]["reason"] == "budget"
    assert not run(system, max_calls=0)["messages"]
    validate(a)


def test_gdesigner_three_rounds_edges_and_aggregations():
    system = system_for("gdesigner-paper")
    trace = run(system, policy="attack")
    calls = [e for e in trace["events"] if e["type"] == "call"]
    assert len(calls) == 12  # three workers and one aggregator per round
    assert [e["kind"] for e in calls].count("aggregator") == 3
    edges = trace["events"][0]["edges"]
    for source, target in edges:
        assert system.roles.index(source) < system.roles.index(target)
    for event in calls:
        if event["kind"] == "worker":
            expected = [e["msg"] for e in calls if e["kind"] == "worker" and
                        e["round"] == event["round"] and [e["identity"], event["identity"]] in edges]
            assert event["inputs"] == expected
    assert trace["final"]["msg"] == calls[-1]["msg"]
    assert sum(m["substituted"] for m in trace["messages"].values()) == 3
    # Nine-role default needs 30 calls, not the old DyLAN envelope of 17.
    c = config("gdesigner-paper")
    c["roles"] = list(learned.ROLES)
    c["rank"] = 4
    full = learned.LearnedSystem(learned.GDesignerPolicy(c, torch.randn(9, 384)), Embeddings())
    assert len([e for e in run(full)["events"] if e["type"] == "call"]) == 30
    assert run(full, max_calls=17)["stop"]["reason"] == "budget"


def test_refinement_closed_form_and_encoder_decoder_gradients():
    sketch = torch.diag(torch.tensor([0.9, 0.7, 0.2]))
    refined, w = learned.refine(sketch, sketch, 2, 0.1)
    assert torch.allclose(refined, torch.diag(torch.tensor([0.85, 0.65, 0.])), atol=1e-6)
    assert torch.linalg.matrix_rank(w) == 2
    system = system_for("gdesigner-paper")
    run(system)
    replay = system.last_replay
    likelihood, regularizer = system.policy.objective(replay)
    (-likelihood + regularizer).backward()
    for module in ("mu", "log_sigma", "decoder"):
        grads = [p.grad for p in getattr(system.policy, module).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    probs, _ = system.policy.distribution(replay["task"], replay["noise"])
    other, _ = system.policy.distribution(replay["task"] + 4, replay["noise"])
    assert not torch.allclose(probs, other)


def test_gdesigner_positive_learning_control():
    # A known edge-likelihood control, not evidence of end-to-end task learning.
    system = system_for("gdesigner-paper")
    task = system.embedder("control task")
    noise = {"normal": torch.zeros(4, 4), "logistic": torch.zeros(3, 3)}
    optimizer = torch.optim.Adam(system.policy.parameters(), lr=0.002)
    before = float(system.policy.distribution(task, noise)[0][0, 1].detach())
    for _ in range(8):
        optimizer.zero_grad()
        probs, _ = system.policy.distribution(task, noise)
        loss = -probs[0, 1].log()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.policy.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    after = float(system.policy.distribution(task, noise)[0][0, 1].detach())
    assert after > before + 0.05


@pytest.mark.parametrize("name", learned.SYSTEMS)
def test_training_resume_artifact_gate_and_model_binding(tmp_path, monkeypatch, name):
    system = system_for(name)
    tasks = tmp_path / "train.json"
    tasks.write_text(json.dumps({"tasks": [dict(TASK, id="train/1"), dict(TASK, id="train/2")]}))
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"tasks": [dict(TASK, id="dev/1")]}))
    reserved = tmp_path / "reserved.json"
    reserved.write_text('{"tasks":[]}')
    args = SimpleNamespace(system=name, samples=2, epochs=1, seed=0, tasks=str(tasks), dev_tasks=str(dev),
                           reserved=[str(reserved)], model="fake", out=str(tmp_path / "out"), backbone=None,
                           embedding_model="fake", lr=1e-4, max_calls=32, purpose="study")
    args.anymac_profile = "normal" if name == "anymac-corrected" else None
    artifact = training.train(args, LLM(), system.embedder, system.settings, system.policy)
    assert artifact.is_file() and not artifact.with_name("progress.pt").exists()
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    assert metadata["training_ids"] == ["train/1", "train/2"]
    monkeypatch.setattr(learned, "Embedder", lambda _: Embeddings())
    with pytest.raises(FileNotFoundError):
        learned.load_system(artifact, "fake")
    loaded = learned.load_system(artifact, "fake", require_gate=False)
    with pytest.raises(ValueError, match="another worker"):
        loaded.validate_model({"digest": "wrong"})
    # Resume does not call the worker or create more checkpoints.
    llm = LLM()
    training.train(args, llm, system.embedder, system.settings, system.policy)
    assert not llm.calls
    # A formal passed gate is required even if the training produced weights.
    gate = {"passed": True, "artifact_sha256": learned.digest(artifact),
            "metadata_sha256": learned.digest(artifact.with_suffix(".json")),
            "runtime_sources": learned.runtime_sources(), "environment": learned.environment()}
    artifact.with_suffix(".gate.json").write_text(json.dumps(gate))
    loaded = learned.load_system(artifact, "fake")
    assert loaded.name == name
    gate["artifact_sha256"] = "changed"
    artifact.with_suffix(".gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="gate"):
        learned.load_system(artifact, "fake")


def test_quality_gate_rejects_no_utility_signal_and_bad_quality():
    row = score(run(system_for("gdesigner-paper")), TASK)
    assert training.quality([row], 1, 1, 1, 1)["passed"]
    assert not training.quality([row], 0, 1, 1, 1)["passed"]
    assert not training.quality([row], 1, 0, 1, 1)["passed"]
    assert not training.quality([row], 1, 1, 0, 1)["passed"]
    assert not training.quality([dict(row, outcome="invalid_format", correct=False)], 1, 1, 1, 1)["passed"]


def test_explicit_anymac_profiles_and_rewards():
    original = config("anymac-corrected")
    for label, gamma, weight, window in (("normal", 1, 0, 5), ("efficient", .9, .001, 2),
                                        ("historical-095", .95, 0, 10)):
        c = training.training_config(original, label)
        assert c["reward_decay"] == gamma and c["sparse_context"] == weight and c["max_context"] == window
        assert training.episode_reward(True, 3, c) == gamma ** 3
        assert training.episode_reward(False, 3, c) == 0
    assert original["max_context"] == 2  # input is not mutated
    with pytest.raises(ValueError, match="explicit"):
        training.training_config(original, None)
    with pytest.raises(ValueError, match="G-Designer"):
        training.training_config(config("gdesigner-paper"), "efficient")


def test_sparse_context_penalty_matches_original_unscaled_cosine_and_has_gradient():
    class Policy:
        config = {"roles": ["A"], "sparse_context": .001}
        value = torch.tensor([.4, -.2, .8], requires_grad=True)

        def scores(self, task, history, authors):
            return torch.zeros(2), 2 * self.value[:len(history)]

    p = Policy()
    replay = {"task": torch.zeros(384), "steps": [
        {"role": 0, "embedding": torch.zeros(384)} for _ in range(3)] + [{"role": 1}]}
    regularizer = training.context_regularizer(p, replay)
    expected = .001 * (p.value[0] + p.value[:2].sum() + p.value.sum())
    assert torch.allclose(regularizer, expected)
    regularizer.backward()
    assert torch.allclose(p.value.grad, torch.tensor([.003, .002, .001]))
    p.config = {**p.config, "sparse_context": 0}
    assert training.context_regularizer(p, replay).item() == 0


def test_efficient_context_only_delivers_newest_two_and_initial_weights_are_paired():
    first = system_for("anymac-corrected")
    second = system_for("anymac-corrected")
    first.policy.config = training.training_config(first.settings, "normal")
    second.policy.config = training.training_config(second.settings, "efficient")
    assert all(torch.equal(t, second.policy.state_dict()[name]) for name, t in first.policy.state_dict().items())
    second.settings = second.policy.config
    second.settings["max_workers"] = 4
    # Force worker selection and select every hint, so masking is tested beyond the window.
    second.policy.scores = lambda task, history, authors: (torch.tensor([10., 0., 0., -10.]),
                                                          torch.ones(len(history)))
    traced = run(second)
    calls = [e for e in traced["events"] if e["type"] == "call"]
    assert len(calls) == 5
    assert calls[-1]["inputs"] == ["m2", "m3"]
    assert all(len(c["inputs"]) <= 2 for c in calls)


def test_training_split_overlap_refused_before_worker(tmp_path):
    task = tmp_path / "tasks.json"
    task.write_text(json.dumps({"tasks": [TASK]}))
    args = SimpleNamespace(samples=2, epochs=1, tasks=str(task), dev_tasks=str(task), reserved=[])
    with pytest.raises(SystemExit, match="reserved"):
        training.train(args)


def test_official_validation_ids_and_text_duplicates_are_excluded():
    rows = [dict(TASK, id=f"mmlu_pro/{i}", text=f"Question {i}") for i in range(6)]
    reserved = [dict(TASK, id="mmlu_pro/100", text="  QUESTION   0  "),
                dict(TASK, id="mmlu_pro/validation/1", text="other wording")]
    sample = training.training_sample(rows, reserved, n=4)
    assert {t["id"] for t in sample} == {f"mmlu_pro/validation/{i}" for i in range(2, 6)}
    assert sample == training.training_sample(rows, reserved, n=4)
    with pytest.raises(ValueError, match="no test-set fallback"):
        training.training_sample(rows, reserved, n=5)


def test_all_validation_is_shuffled_disjoint_and_covers_subjects():
    rows = [dict(TASK, id=f"mmlu_pro/{i}", text=f"Question {i}", subject=f"topic{i % 14}")
            for i in range(70)]
    reserved = [dict(rows[0], id="test/0", text=" QUESTION  0 ")]
    sample = training.training_sample(rows, reserved, n=None)
    assert len(sample) == 69
    assert len({t["subject"] for t in sample}) == 14
    assert all(t["id"].startswith("mmlu_pro/validation/") for t in sample)
    assert sample == training.training_sample(rows, reserved, n=None)
    assert [t["id"] for t in sample] != sorted(t["id"] for t in sample)
    with pytest.raises(ValueError, match="entire validation subject"):
        training.training_sample(rows[:14], reserved, n=None)
    with pytest.raises(ValueError, match="duplicate validation question"):
        training.training_sample([rows[0], dict(rows[1], text=rows[0]["text"])], [], n=None)


@pytest.mark.parametrize("samples,budget", [(8, 1000), (8, 17), (8, 9), (3, 7), (2, 6)])
def test_exact_budget_and_independent_terminal_groups(samples, budget):
    tasks = [dict(TASK, id=f"train/{i}") for i in range(70)]
    jobs = training.training_jobs(tasks, samples, 1, budget=budget, seed=4)
    assert sum(count for _, _, count in jobs) == budget
    assert all(2 <= count <= samples for _, _, count in jobs)
    assert jobs == training.training_jobs(tasks, samples, 1, budget=budget, seed=4)
    assert jobs != training.training_jobs(tasks, samples, 1, budget=budget, seed=5)
    first_pass = [task["id"] for epoch, task, _ in jobs if epoch == 0]
    assert len(set(first_pass)) == len(first_pass)
    if budget == 1000:
        assert len(jobs) == 125 and len(first_pass) == 70
        assert {epoch for epoch, _, _ in jobs} == {0, 1}


@pytest.mark.parametrize("samples,budget", [(8, 1), (2, 3), (2, 0)])
def test_impossible_independent_group_budget_is_rejected(samples, budget):
    with pytest.raises(ValueError, match="budget"):
        training.training_jobs([TASK], samples, 1, budget=budget)


def test_paper_horizon_does_not_mutate_preliminary_profiles():
    original = config("anymac-corrected")
    paper = training.training_config(original, "normal", "paper-aligned-v1")
    assert paper["max_workers"] == 5
    assert original["max_workers"] == 2
    assert training.training_config(original, "normal")["max_workers"] == 2
    with pytest.raises(ValueError, match="normal or efficient"):
        training.training_config(original, "historical-095", "paper-aligned-v1")


@pytest.mark.parametrize("all_validation", [False, True])
def test_prepare_uses_validation_not_test_file(tmp_path, monkeypatch, all_validation):
    import pandas as pd
    import huggingface_hub
    parquet = tmp_path / "validation.parquet"
    pd.DataFrame([{"question_id": i, "category": "subject", "question": f"Q {i}",
                   "options": ["one", "two", "three", "four"], "answer": "A"} for i in range(50)]).to_parquet(parquet)
    requests = []

    def download(repo, file, **kwargs):
        requests.append((repo, file, kwargs))
        return str(parquet)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    reserved = tmp_path / "reserved.json"
    reserved.write_text('{"tasks":[]}')
    out = tmp_path / "train.json"
    training.prepare(SimpleNamespace(out=str(out), reserved=[str(reserved)], all_validation=all_validation))
    doc = json.loads(out.read_text())
    assert len(doc["tasks"]) == (50 if all_validation else 40) and doc["source"]["split"] == "validation"
    assert doc["selection"] == ("all-eligible-validation" if all_validation else "random-40")
    assert doc["subject_counts"] == {"subject": len(doc["tasks"])}
    assert doc["excluded_sha256"] == {str(reserved): learned.digest(reserved)}
    assert requests[0][1] == "data/validation-00000-of-00001.parquet"
    assert requests[0][2]["revision"] == training.DATA_REVISION
    with pytest.raises(ValueError, match="frozen"):
        training.prepare(SimpleNamespace(out=str(out), reserved=[str(reserved)]))


def test_runner_requires_bound_learned_artifacts():
    args = SimpleNamespace(checkpoint=None, embedding_model=None)
    with pytest.raises(SystemExit, match="require"):
        runner.learned_system(args)


def test_prepare_trims_padding_and_reports_overlaps_without_overwriting(tmp_path, monkeypatch):
    import pandas as pd
    import huggingface_hub
    from scripts.screen.screen import mc_rows
    parquet = tmp_path / "validation.parquet"
    frame = pd.DataFrame([
        {"question_id": i, "category": f"subject{i % 2}", "question": f"Q {i}",
         "options": ["one", "two", "three", "N/A", "N/A"], "answer": "C", "answer_index": 2}
        for i in range(6)])
    frame.to_parquet(parquet)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(parquet))
    cleaned = mc_rows("mmlu_pro", frame, trim_trailing_na=True)
    first = dict(cleaned[0], id="old-eval/0")
    by_text = tmp_path / "text_reserved.json"
    by_text.write_text(json.dumps({"tasks": [dict(first, text=" ".join(first["text"].split()).upper())]}))
    by_id = tmp_path / "id_reserved.json"
    by_id.write_text(json.dumps({"tasks": [dict(cleaned[1], id="mmlu_pro/validation/1", text="Other text")]}))
    out = tmp_path / "new_train.json"
    args = SimpleNamespace(out=str(out), reserved=[str(by_text), str(by_id)],
                           all_validation=True, trim_trailing_na=True)
    training.prepare(args)
    doc = json.loads(out.read_text())
    assert len(doc["tasks"]) == 4
    assert doc["option_preprocessing"] == training.TRAILING_NA_POLICY
    assert doc["subject_counts"] == {"subject0": 2, "subject1": 2}
    assert all(t["n_options"] == 3 and t["truth"] == "C" and "N/A" not in t["text"] for t in doc["tasks"])
    audit = doc["preparation_audit"]
    assert (audit["raw_questions"], audit["questions_with_trimmed_na"], audit["removed_na_options"]) == (6, 6, 12)
    assert (audit["after_options"], audit["excluded_overlap_unique"], audit["selected_questions"]) == (6, 2, 4)
    assert audit["overlap_ids_by_reserved_file"] == {
        str(by_text): ["mmlu_pro/validation/0"], str(by_id): ["mmlu_pro/validation/1"]}
    before = out.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        training.prepare(args)
    assert out.read_bytes() == before


def test_embedding_hash_works_under_home_cache_and_ignores_download_metadata(tmp_path):
    path = tmp_path / ".cache/huggingface/model"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    before = learned.tree_digest(path)
    (path / ".cache").mkdir()
    (path / ".cache/lock").write_text("transient")
    assert learned.tree_digest(path) == before
    (path / "config.json").write_text('{"changed":true}')
    assert learned.tree_digest(path) != before


@pytest.mark.parametrize("budget", [None, 17])
def test_interrupted_training_resumes_only_remaining_task(tmp_path, monkeypatch, budget):
    system = system_for("anymac-corrected")
    train_file, dev_file, reserved = (tmp_path / p for p in ("train.json", "dev.json", "reserved.json"))
    train_file.write_text(json.dumps({"tasks": [dict(TASK, id="train/1"), dict(TASK, id="train/2")]}))
    dev_file.write_text(json.dumps({"tasks": [dict(TASK, id="dev/1")]}))
    reserved.write_text('{"tasks":[]}')
    args = SimpleNamespace(system=system.name, samples=2, epochs=1, seed=0, tasks=str(train_file),
                           dev_tasks=str(dev_file), reserved=[str(reserved)], model="fake",
                           out=str(tmp_path / "out"), backbone=None, embedding_model="fake",
                           lr=1e-4, max_calls=32, purpose="study")
    save = training.atomic_save
    args.anymac_profile = "normal"
    args.trajectory_budget = budget
    if budget:
        args.samples = 4

    def interrupt(path, data):
        save(path, data)
        if path.name == "progress.pt" and data["progress"]["next_task"] == 1:
            raise RuntimeError("simulated interruption after durable checkpoint")
    monkeypatch.setattr(training, "atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        training.train(args, LLM(), system.embedder, system.settings, system.policy)
    assert (Path(args.out) / "progress.pt").is_file()
    monkeypatch.setattr(training, "atomic_save", save)
    restarted = system_for("anymac-corrected")
    training.train(args, LLM(), restarted.embedder, restarted.settings, restarted.policy)
    progress = json.loads((Path(args.out) / "progress.json").read_text())
    tasks = runner.load_tasks(args.tasks)
    jobs = training.training_jobs(tasks, args.samples, 1, budget, args.seed)
    assert [r["task_id"] for r in progress["updates"]] == [t["id"] for _, t, _ in jobs]
    assert progress["trajectories"] == sum(j[2] for j in jobs)
    assert [p.name for p in Path(args.out).glob("*.pt")] == ["router.pt"]
    reference = system_for("anymac-corrected")
    reference_args = copy.copy(args)
    reference_args.out = str(tmp_path / "uninterrupted")
    training.train(reference_args, LLM(), reference.embedder, reference.settings, reference.policy)
    assert all(torch.equal(tensor, reference.policy.state_dict()[name])
               for name, tensor in restarted.policy.state_dict().items())


@pytest.mark.parametrize("regime", training.PAPER_REGIMES)
def test_paper_budget_learning_curve_and_resume_at_milestone(tmp_path, monkeypatch, regime):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def objective(self, replay):
            return self.weight * replay["factor"], torch.tensor(0.)

    train_file, dev_file, reserved = (tmp_path / p for p in ("train.json", "dev.json", "reserved.json"))
    train_file.write_text(json.dumps({
        "tasks": [dict(TASK, id="mmlu_pro/validation/1", text="Training only")],
        "selection": "all-eligible-validation",
        "option_preprocessing": (training.TRAILING_NA_POLICY if regime == "paper-aligned-v2"
                                 else "legacy-drop-placeholder-questions"),
        "source": {"split": "validation", "revision": training.DATA_REVISION}}))
    dev_file.write_text(json.dumps({"tasks": [dict(TASK, id="dev/1", text="Development only")]}))
    reserved.write_text('{"tasks":[]}')
    args = SimpleNamespace(system="anymac-corrected", samples=8, epochs=1, seed=0,
                           tasks=str(train_file), dev_tasks=str(dev_file), reserved=[str(reserved)],
                           out=str(tmp_path / "out"), backbone=None, embedding_model="fake",
                           lr=1e-4, max_calls=32, purpose="study", anymac_profile="normal",
                           training_regime=regime, trajectory_budget=1000)
    calls, evaluations = [], []
    row = score(run(system_for("anymac-corrected")), TASK)

    def episode(system, llm, task, rep, sample, seed, max_calls, intervention):
        calls.append(seed)
        return {}, {"correct": bool(seed % 2)}, {"steps": [{}], "factor": float(seed % 2)}

    def evaluate(system, llm, tasks, path, max_calls, seed):
        evaluations.append(path.name)
        return [row]

    monkeypatch.setattr(training, "run_episode", episode)
    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr(training, "routing_diagnostics", lambda *args: {"worker_episode_rate": 0})
    save = training.atomic_save

    def interrupt(path, data):
        save(path, data)
        if path.name == "progress.pt" and data["progress"]["trajectories"] == 200:
            raise RuntimeError("interrupted before milestone evaluation")
    monkeypatch.setattr(training, "atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="before milestone"):
        training.train(args, LLM(), Embeddings(), config(args.system), Policy())
    assert len(calls) == 200 and "dev_budget_200.jsonl" not in evaluations
    monkeypatch.setattr(training, "atomic_save", save)
    artifact = training.train(args, LLM(), Embeddings(), config(args.system), Policy())
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    assert len(calls) == len(set(calls)) == 1000
    assert metadata["completed_trajectories"] == metadata["trajectory_budget"] == 1000
    assert metadata["config"]["max_workers"] == 5
    curve = json.loads((Path(args.out) / "learning_curve.json").read_text())
    assert set(curve) == {"200", "400", "600", "800", "1000"}
    assert evaluations.count("dev_budget_200.jsonl") == 1
    assert metadata["epochs"] is None
    args.training_regime = "paper-aligned-v2" if regime == "paper-aligned-v1" else "paper-aligned-v1"
    with pytest.raises(ValueError, match="option preprocessing"):
        training.train(args, LLM(), Embeddings(), config(args.system), Policy())
    args.training_regime = regime
    # Changed budget/source cannot silently resume a frozen output.
    args.trajectory_budget = 1008
    with pytest.raises(ValueError, match="exactly 1000"):
        training.train(args, LLM(), Embeddings(), config(args.system), Policy())


def test_development_gate_is_written_only_with_bound_checks(tmp_path):
    system = system_for("anymac-corrected")
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"tasks": [dict(TASK, id="dev/1"), dict(TASK, id="dev/2", text="Other question")]}))
    # Isolate gate logic with a policy that visibly uses a worker and varies with the task.
    def scores(task, history, authors):
        agents = (torch.tensor([0., 1., 0., 0.]) + torch.tensor([0., float(task[0]) * .01, 0., 0.])
                  if not history else torch.tensor([0., 0., 0., 10.]))
        return agents, torch.ones(len(history))
    system.policy.scores = scores
    artifact = tmp_path / "router.pt"
    artifact.write_bytes(b"unit fixture: loading is tested separately")
    meta = {"purpose": "study", "dev_sha256": learned.digest(dev), "seed": 0,
            "worker": learned.worker_identity(LLM().identity()), "runtime_sources": learned.runtime_sources(),
            "environment": learned.environment(), "nonzero_updates": 1, "reward_varying_tasks": 1,
            "utility_updates": 1, "untrained_development": {"anymac-corrected/honest": {"accuracy": 1.0}}}
    system.metadata = meta
    artifact.with_suffix(".json").write_text(json.dumps(meta))
    args = SimpleNamespace(checkpoint=str(artifact), dev_tasks=str(dev), max_calls=32)
    result = training.gate(args, llm=LLM(), system=system)
    assert result["passed"] and artifact.with_suffix(".gate.json").exists()
    assert result["routing"]["worker_episode_rate"] == 1
    meta["utility_updates"] = 0
    with pytest.raises(ValueError, match="gate failed"):
        training.gate(args, llm=LLM(), system=system)
    assert not json.loads(artifact.with_suffix(".gate.json").read_text())["passed"]
