import copy
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite


def arguments(tmp_path, phase="pilot"):
    for label in ("dev", "eval", "train", "reserved"):
        (tmp_path / f"{label}.json").write_text(json.dumps({"tasks": [
            {"id": f"{label}/{i}", "kind": "mc", "text": "Q", "truth": "A", "n_options": 4}
            for i in range(2)]}))
    args = SimpleNamespace(command="run", phase=phase, dev_tasks=tmp_path / "dev.json",
                           eval_tasks=tmp_path / "eval.json", train_tasks=tmp_path / "train.json",
                           reserved=[tmp_path / "reserved.json"], learned_root=tmp_path / "learned",
                           seeds=[0, 1, 2], out=tmp_path / "suite", model="fake", threads=8)
    doc = json.loads(args.eval_tasks.read_text())
    doc.update(purpose="fresh-main-confirmation", excluded_sha256={
        str(p): suite.digest(p) for p in [*args.reserved, args.train_tasks, args.dev_tasks]})
    args.eval_tasks.write_text(json.dumps(doc))
    return args


def fake_artifact(root, system, seed, embedding_hash):
    return Path(root) / system / f"seed{seed}/router.pt", {
        "config": {"roles": ["Expert", "Critic", "Analyst"]}, "training_ids": ["train/1"],
        "dev_ids": ["dev/1"], "artifact_sha256": f"{system}-{seed}",
        "initial_policy_sha256": f"initial-{seed}"}


def test_pilot_and_main_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "artifact", fake_artifact)
    args = arguments(tmp_path)
    jobs, tasks = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 5 and len(tasks) == 2
    assert {j["system"] for j in jobs} == {"fixed", "dylan", *suite.LEARNED}
    assert all(j["policies"] == ["honest", "attack"] and j["reps"] == 2 for j in jobs)
    assert {j["seed"] for j in jobs} == {0, None}
    args.phase = "main"
    jobs, _ = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 2 + 2 * (3 + 7) + 3 * 3 * (1 + 2 * 3)
    assert len({j["id"] for j in jobs}) == len(jobs)
    for job in jobs:
        assert ("--attack-texts" in job["argv"]) == (job["form"] == "persuasive")
        assert job["argv"][job["argv"].index("--max-calls") + 1] == "32"
        assert ("--checkpoint" in job["argv"]) == (job["system"] in suite.LEARNED)
    args.seeds = [0]
    with pytest.raises(ValueError, match="distinct"):
        suite.jobs_for(args, {"path": "embed", "sha256": "x"})


@pytest.mark.parametrize("n,episodes", [(60, 12240), (200, 40800)])
def test_main_has_193_runs_and_scales_with_tasks(monkeypatch, tmp_path, n, episodes):
    def full_artifact(*args):
        path, meta = fake_artifact(*args)
        meta["config"]["roles"] = ["Knowledgeable Expert", "Critic", "Mathematician", "Psychologist",
                                   "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]
        return path, meta
    monkeypatch.setattr(suite, "artifact", full_artifact)
    args = arguments(tmp_path, "main")
    eval_tasks = [{"id": f"eval/{i}", "kind": "mc", "text": "Q", "truth": "A", "n_options": 4} for i in range(n)]
    document = json.loads(args.eval_tasks.read_text())
    document["tasks"] = eval_tasks
    args.eval_tasks.write_text(json.dumps(document))
    jobs, _ = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 193
    assert sum(j["tasks"] * j["reps"] * len(j["policies"]) for j in jobs) == episodes


def test_amended_pilot_uses_every_seed_and_records_admission(monkeypatch, tmp_path):
    import admission
    def fake(root, system, seed, embedding_hash, allow_routing_degeneracy=False):
        assert allow_routing_degeneracy is True
        return fake_artifact(root, system, seed, embedding_hash)
    decision = {"policy": admission.POLICY, "original_failed_checks": ["nontrivial_cooperation"]}
    monkeypatch.setattr(suite, "artifact", fake)
    monkeypatch.setattr(admission, "admit", lambda path, allow_routing_degeneracy: decision)
    args = arguments(tmp_path)
    args.allow_routing_degeneracy = True
    jobs, _ = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 11
    assert sum(j["tasks"] * j["reps"] * len(j["policies"]) for j in jobs) == 88
    assert {j["seed"] for j in jobs} == {None, 0, 1, 2}
    for job in jobs:
        if job["seed"] is not None:
            assert "--allow-routing-degeneracy" in job["argv"]
            assert job["evaluation_admission"] == decision
            assert job["role"] == "Expert"  # unchanged pre-existing first-role rule
        else:
            assert "--allow-routing-degeneracy" not in job["argv"]


def test_pilot_does_not_hide_bad_seed_in_pooled_quality(monkeypatch, tmp_path):
    from test_compare import episode, MC
    args = arguments(tmp_path)
    args.allow_routing_degeneracy = True
    args.seeds = [0, 1]
    trace, _ = episode()
    base_row = suite.score(trace, MC)
    rows, jobs = [], []
    for seed in args.seeds:
        for i in range(12):
            row = dict(base_row, system="anymac-normal", seed=seed, task_id=f"task/{i}",
                       role="Expert", form="bare")
            if seed == 1 and i == 0:
                row.update(outcome="invalid_format", correct=False, pred=None)
            rows.append(row)
        out = tmp_path / f"seed{seed}"
        out.mkdir()
        (out / "traces.jsonl").write_text("{}\n")
        jobs.append({"argv": ["--out", str(out)], "system": "anymac-normal", "seed": seed,
                     "artifact_sha256": str(seed), "tasks": 12, "reps": 1, "policies": ["honest"]})
    monkeypatch.setattr(suite, "completed_rows", lambda jobs, tasks: rows)
    (args.out / args.phase).mkdir(parents=True)
    result = suite.report(args, jobs, [MC])
    assert result["honest_cells"]["anymac-normal"]["invalid_format"] < .05
    assert result["honest_cells_by_checkpoint"]["anymac-normal/seed1"]["invalid_format"] > .05
    assert result["passed"] is False


def test_main_refuses_old_unreserved_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "artifact", fake_artifact)
    args = arguments(tmp_path, "main")
    doc = json.loads(args.eval_tasks.read_text())
    del doc["purpose"]
    args.eval_tasks.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="fresh reserved"):
        suite.jobs_for(args, {"path": "embed", "sha256": "x"})


def test_larger_validation_pool_does_not_rewrite_main_reservation(monkeypatch, tmp_path):
    from train_learned import DATA_REVISION
    args = arguments(tmp_path, "main")
    pool = tmp_path / "paper_train.json"
    pool.write_text(json.dumps({
        "selection": "all-eligible-validation", "source": {"split": "validation", "revision": DATA_REVISION},
        "tasks": [{"id": "mmlu_pro/validation/1", "text": "New training question"}]}))
    args.anymac_training_tasks = pool
    def fake(root, system, seed, embedding):
        path, meta = fake_artifact(root, system, seed, embedding)
        meta["training_sha256"] = suite.digest(pool if system.startswith("anymac-") else args.train_tasks)
        meta["config"]["max_workers"] = 5
        meta.update(training_regime="paper-aligned-v1", trajectory_budget=1000, completed_trajectories=1000,
                    samples=8)
        return path, meta
    monkeypatch.setattr(suite, "artifact", fake)
    # Distinct texts: legacy fixtures deliberately used identical question text.
    doc = json.loads(args.eval_tasks.read_text())
    for t in doc["tasks"]:
        t["text"] = "Evaluation " + t["id"]
    args.eval_tasks.write_text(json.dumps(doc))
    original = args.eval_tasks.read_bytes()
    jobs, _ = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert args.eval_tasks.read_bytes() == original
    assert all(str(pool) in job["argv"] for job in jobs)
    assert suite.digest(pool) not in doc["excluded_sha256"].values()

    def incomplete(*a):
        path, meta = fake(*a)
        meta["completed_trajectories"] = 160
        return path, meta
    monkeypatch.setattr(suite, "artifact", incomplete)
    with pytest.raises(ValueError, match="budget/config"):
        suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    monkeypatch.setattr(suite, "artifact", fake)
    pool_doc = json.loads(pool.read_text())
    pool_doc["tasks"][0]["text"] = "  EVALUATION  eval/0  "
    pool.write_text(json.dumps(pool_doc))
    with pytest.raises(ValueError, match="text overlaps"):
        suite.jobs_for(args, {"path": "embed", "sha256": "x"})


def test_anymac_profiles_require_paired_initial_weights(monkeypatch, tmp_path):
    def mismatched(root, system, seed, embedding):
        path, meta = fake_artifact(root, system, seed, embedding)
        if system == "anymac-efficient":
            meta["initial_policy_sha256"] = "different"
        return path, meta
    monkeypatch.setattr(suite, "artifact", mismatched)
    with pytest.raises(ValueError, match="initial weights"):
        suite.jobs_for(arguments(tmp_path), {"path": "embed", "sha256": "x"})


def test_disjointness_and_cost_projection(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "artifact", fake_artifact)
    args = arguments(tmp_path, "main")
    args.eval_tasks = args.dev_tasks
    with pytest.raises(ValueError, match="development"):
        suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    jobs = [{"tasks": 60, "reps": 2, "policies": ["honest"], "system": "dylan"},
            {"tasks": 60, "reps": 1, "policies": ["attack"], "system": "dylan"}]
    cost = suite.estimate(jobs, {"honest_cells": {"dylan": {"mean_episode_wall_s": 60}},
                                 "trace_bytes_per_episode": {"dylan": 1024 ** 2}})
    assert cost["gpu_hours"] == 3 and cost["with_25_percent_margin"] == 3.75
    assert cost["projected_trace_gib"] == 180 / 1024


def test_two_servers_claim_once_resume_and_failure(monkeypatch, tmp_path):
    jobs = [{"id": str(i), "argv": ["run", "--out", str(tmp_path / str(i))]} for i in range(8)]
    recorded, mutex = [], threading.Lock()

    def fake_run(argv, **kwargs):
        out = Path(argv[argv.index("--out") + 1])
        with mutex:
            recorded.append(str(out))
        time.sleep(.01)
        if out.name == "3":
            return SimpleNamespace(returncode=1)
        (out / "COMPLETE.json").write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="incomplete"):
        suite.execute(jobs, ["url0", "url1"])
    assert len(recorded) == len(set(recorded)) == 8
    recorded.clear()
    with pytest.raises(ValueError, match="incomplete"):
        suite.execute(jobs, ["url0", "url1"])
    assert recorded == [str(tmp_path / "3")]


def test_task_bootstrap_does_not_treat_roles_repetitions_or_seeds_as_tasks():
    rows = []
    for task in ("t1", "t2", "t3"):
        for seed in (0, 1, 2):
            for rep in (1, 2):
                rows.append({"system": "x", "task_id": task, "policy": "honest", "form": "bare",
                             "correct": True, "target_hit": False, "seed": seed})
            for role in ("r1", "r2"):
                for form in ("bare", "persuasive"):
                    rows.append({"system": "x", "task_id": task, "policy": "attack", "form": form,
                                 "correct": False, "target_hit": form == "persuasive", "seed": seed})
    effects = suite.contrasts(rows)
    assert effects["x/bare"]["correct"]["tasks"] == 3
    assert effects["x/bare"]["correct"]["mean"] == -1
    assert effects["x/persuasive"]["target_hit"]["mean"] == 1
    assert effects["x/persuasive-minus-bare"]["target_hit"]["mean"] == 1
    doubled = suite.contrasts(copy.deepcopy(rows) * 2)
    assert effects == doubled


def test_partial_cells_are_not_pooled(tmp_path):
    job = {"argv": ["--out", str(tmp_path)], "policies": ["honest"], "reps": 1}
    with pytest.raises(ValueError, match="incomplete"):
        suite.completed_rows([job], [{"id": "x"}])


def test_normal_and_efficient_architecture_traces_are_never_pooled_under_one_label(tmp_path):
    from test_compare import episode, MC
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"tasks": [MC]}))
    jobs = []
    for label in ("anymac-normal", "anymac-efficient"):
        out = tmp_path / label
        out.mkdir()
        trace, _ = episode(max_calls=32)
        trace["system"] = "anymac-corrected"
        manifest = {"source": suite.source_digest(), "system": {"system": "anymac-corrected"},
                    "compromised": "Critic", "max_calls": 32, "task_ids": [MC["id"]],
                    "tasks_sha256": suite.digest(task_file), "policies": ["honest"], "reps": 1, "target_seed": 0}
        (out / "manifest.json").write_text(json.dumps(manifest))
        (out / "traces.jsonl").write_text(json.dumps(trace) + "\n")
        (out / "COMPLETE.json").write_text('{"episodes":1}')
        jobs.append({"argv": ["--out", str(out), "--tasks", str(task_file)],
                     "system": label, "role": "Critic", "artifact_sha256": None,
                     "policies": ["honest"], "reps": 1, "seed": 0, "form": "bare"})
    rows = suite.completed_rows(jobs, [MC])
    assert {r["system"] for r in rows} == {"anymac-normal", "anymac-efficient"}
    assert set(suite.summarize(rows)) == {"anymac-normal/honest", "anymac-efficient/honest"}
    jobs[0]["evaluation_admission"] = {"policy": "changed"}
    with pytest.raises(ValueError, match="admission differs"):
        suite.completed_rows(jobs, [MC])


def test_clean_suite_rejects_wrong_profile_and_intervention_artifacts(tmp_path):
    import admission
    import learned
    import train_learned
    path = tmp_path / "anymac-normal/seed0/router.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"hash-only-test")
    meta = {"schema": learned.ARTIFACT_SCHEMA,
            "config": {"system": "anymac-corrected", "training_profile": "normal",
                       **train_learned.ANYMAC_PROFILES["normal"]},
            "seed": 0, "purpose": "study", "artifact_sha256": suite.digest(path),
            "initial_policy_sha256": "a" * 64,
            "embedding_sha256": "embed", "runtime_sources": suite.runtime_sources(),
            "environment": suite.environment(), "training_intervention": None}
    def write():
        path.with_suffix(".json").write_text(json.dumps(meta))
        gate = {"passed": True, "artifact_sha256": suite.digest(path),
                "metadata_sha256": suite.digest(path.with_suffix(".json")),
                "checks": {key: True for key in admission.REQUIRED_CHECKS},
                "routing": {"worker_episode_rate": 1.0, "nontrivial": True},
                "runtime_sources": suite.runtime_sources(), "environment": suite.environment()}
        path.with_suffix(".gate.json").write_text(json.dumps(gate))
    write()
    suite.artifact(tmp_path, "anymac-normal", 0, "embed")
    meta["config"]["reward_decay"] = .95
    write()
    with pytest.raises(ValueError, match="profile"):
        suite.artifact(tmp_path, "anymac-normal", 0, "embed")
    meta["config"]["reward_decay"] = 1.0
    meta["training_intervention"] = {"condition": "ordinary-neutral"}
    write()
    with pytest.raises(ValueError, match="gate"):
        suite.artifact(tmp_path, "anymac-normal", 0, "embed")
    meta["training_intervention"] = None
    write()
    gate_path = path.with_suffix(".gate.json")
    gate = json.loads(gate_path.read_text())
    gate["passed"] = False
    gate["checks"]["nontrivial_cooperation"] = False
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="gate"):
        suite.artifact(tmp_path, "anymac-normal", 0, "embed")
    _, admitted = suite.artifact(tmp_path, "anymac-normal", 0, "embed", True)
    assert admitted["evaluation_admission"]["original_gate_passed"] is False
    assert admitted["evaluation_admission"]["original_failed_checks"] == ["nontrivial_cooperation"]
    gate["checks"]["format"] = False
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="gate"):
        suite.artifact(tmp_path, "anymac-normal", 0, "embed", True)
