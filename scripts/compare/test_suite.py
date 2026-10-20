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
    return SimpleNamespace(command="run", phase=phase, dev_tasks=tmp_path / "dev.json",
                           eval_tasks=tmp_path / "eval.json", train_tasks=tmp_path / "train.json",
                           reserved=[tmp_path / "reserved.json"], learned_root=tmp_path / "learned",
                           seeds=[0, 1, 2], out=tmp_path / "suite", model="fake", threads=8)


def fake_artifact(root, system, seed, embedding_hash):
    return Path(root) / system / f"seed{seed}/router.pt", {
        "config": {"roles": ["Expert", "Critic", "Analyst"]}, "training_ids": ["train/1"],
        "dev_ids": ["dev/1"], "artifact_sha256": f"{system}-{seed}"}


def test_pilot_and_main_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "artifact", fake_artifact)
    args = arguments(tmp_path)
    jobs, tasks = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 4 and len(tasks) == 2
    assert {j["system"] for j in jobs} == {"fixed", "dylan", *suite.LEARNED}
    assert all(j["policies"] == ["honest", "attack"] and j["reps"] == 2 for j in jobs)
    assert {j["seed"] for j in jobs} == {0, None}
    args.phase = "main"
    jobs, _ = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 2 + 2 * (3 + 7) + 2 * 3 * (1 + 2 * 3)
    assert len({j["id"] for j in jobs}) == len(jobs)
    for job in jobs:
        assert ("--attack-texts" in job["argv"]) == (job["form"] == "persuasive")
        assert job["argv"][job["argv"].index("--max-calls") + 1] == "32"
        assert ("--checkpoint" in job["argv"]) == (job["system"] in suite.LEARNED)
    args.seeds = [0]
    with pytest.raises(ValueError, match="distinct"):
        suite.jobs_for(args, {"path": "embed", "sha256": "x"})


def test_default_main_has_136_runs_and_8640_episodes_for_60_tasks(monkeypatch, tmp_path):
    def full_artifact(*args):
        path, meta = fake_artifact(*args)
        meta["config"]["roles"] = ["Knowledgeable Expert", "Critic", "Mathematician", "Psychologist",
                                   "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]
        return path, meta
    monkeypatch.setattr(suite, "artifact", full_artifact)
    args = arguments(tmp_path, "main")
    eval_tasks = [{"id": f"eval/{i}", "kind": "mc", "text": "Q", "truth": "A", "n_options": 4} for i in range(60)]
    args.eval_tasks.write_text(json.dumps({"tasks": eval_tasks}))
    jobs, _ = suite.jobs_for(args, {"path": "embed", "sha256": "x"})
    assert len(jobs) == 136
    assert sum(j["tasks"] * j["reps"] * len(j["policies"]) for j in jobs) == 8640


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
