import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import report as reporting
import study
import worker
from bank_recovery import APPROVED_BAD
import learned
import train_learned
import trust
from test_learned import Embeddings, LLM, config


def task(index, subject="math", prefix="mmlu_pro/"):
    return {"id": f"{prefix}{index}", "text": f"Question {index}?\nA. yes\nB. no\nC. maybe\nD. never",
            "subject": subject, "kind": "mc", "truth": "A", "n_options": 4}


def document(path, tasks):
    path.write_text(json.dumps({"schema": 1, "tasks": tasks}))
    return str(path)


def test_balanced_sampling_is_outcome_independent_and_order_stable():
    pool = [task(i, f"subject{i % 14}") for i in range(200)]
    selected = common.balanced(pool, 24, 20261103)
    assert len(selected) == 24 and len({t["subject"] for t in selected}) == 14
    assert selected == common.balanced(list(reversed(pool)), 24, 20261103)
    changed = [dict(t, truth="D") for t in pool]
    assert [t["id"] for t in selected] == [t["id"] for t in common.balanced(changed, 24, 20261103)]
    with pytest.raises(ValueError, match="cover"):
        common.balanced(pool, 10, 0)


def test_split_checks_normalized_text_not_only_ids():
    training = [task(0, prefix="mmlu_pro/validation/")]
    dev, evaluation = [task(1)], [task(2)]
    common.check_disjoint(training, dev, evaluation, [])
    with pytest.raises(ValueError, match="overlap"):
        common.check_disjoint(training, [dict(dev[0], text=training[0]["text"].upper())], evaluation, [])
    with pytest.raises(ValueError, match="duplicate"):
        common.check_disjoint(training * 2, dev, evaluation, [])


def test_runtime_budget_meets_threshold_and_reserves_evaluation():
    # Verify the exact boundary for 64 rollouts, not merely that some budget is returned.
    pilot = [80.0, 80.0]
    stress = [10.0, 10.0]
    threshold = 6 * 64 * 10 * common.MARGIN + 312 * 10 * common.MARGIN + common.RESERVE_S
    with pytest.raises(ValueError, match="even 64"):
        common.choose_budget(threshold - .01, pilot, stress)
    plan = common.choose_budget(threshold + .01, pilot, stress)
    assert plan["trajectories"] == 64
    assert (plan["projected_training_seconds"] + plan["evaluation_seconds"] + plan["reserve_seconds"]
            <= threshold + .01)
    assert common.choose_budget(200000, pilot, stress)["trajectories"] == 256


def test_all_evaluation_pairs_are_adjacent_same_server_and_noise_is_separate():
    tasks = [task(i) for i in range(24)]
    ids = [t["id"] for t in tasks[:4]]
    total = 0
    for seed in common.SEEDS:
        jobs = common.evaluation_order(tasks, ids, seed)
        assert len(jobs) == 104
        total += len(jobs)
        for i, (t, arm, policy, rep) in enumerate(jobs):
            if rep == 1:
                adjacent = [jobs[j] for j in (i - 1, i + 1) if 0 <= j < len(jobs)]
                assert any(other[0] == t and other[1] == arm and other[2] != policy and other[3] == 1
                           for other in adjacent)
    assert total == 624


def test_bootstrap_uses_seed_and_task_units_separately():
    result = common.uncertainty([[1, 1], [-1, -1]], draws=1000)
    assert result["effect"] == 0 and result["seed_effects"] == [1, -1]
    assert result["task_conditional_on_checkpoints_ci95"] == [0, 0]
    assert result["seed_conditional_on_tasks_ci95"] == [-1, 1]


def test_expired_batch_never_starts_process_and_timeout_preserves_logs(tmp_path):
    (tmp_path / "logs").mkdir()
    assert not study.batch([[sys.executable, "-c", "raise Exception('must not run')"]],
                           time.time() - 1, tmp_path, "expired")
    before = time.monotonic()
    assert not study.batch([[sys.executable, "-c", "import time; time.sleep(20)"]],
                           time.time() + .15, tmp_path, "timeout")
    assert time.monotonic() - before < 3
    status = json.loads((tmp_path / "logs/timeout.json").read_text())
    assert status["timed_out"] and status["returncodes"][0] != 0


def test_failed_process_is_not_success_shaped(tmp_path):
    (tmp_path / "logs").mkdir()
    assert not study.batch([[sys.executable, "-c", "print('backend failure'); raise SystemExit(7)"]],
                           time.time() + 5, tmp_path, "error")
    assert json.loads((tmp_path / "logs/error.json").read_text())["returncodes"] == [7]
    assert "backend failure" in (tmp_path / "logs/error-0.log").read_text()


def test_failed_peer_terminates_other_owned_worker_without_waiting_for_deadline(tmp_path):
    (tmp_path / "logs").mkdir()
    start = time.monotonic()
    assert not study.batch([
        [sys.executable, "-c", "raise SystemExit(7)"],
        [sys.executable, "-c", "import time; time.sleep(20)"],
    ], time.time() + 10, tmp_path, "peer-failed")
    assert time.monotonic() - start < 3
    assert json.loads((tmp_path / "logs/peer-failed.json").read_text())["returncodes"][0] == 7


def test_load_rejects_duplicate_ids_before_runner_deduplication(tmp_path):
    path = tmp_path / "tasks.json"
    document(path, [task(0), task(0)])
    with pytest.raises(ValueError, match="before runner"):
        common.load_tasks(path)


def test_ramses_launcher_uses_allocation_not_forbidden_interactive_batch():
    directory = Path(__file__).resolve().parent
    script = (directory / "run.sh").read_text()
    instructions = (directory / "README.md").read_text()
    assert "#SBATCH" not in script
    assert "Start with salloc followed by srun" in script
    assert "salloc --partition=interactive" in instructions
    assert "--mem=64G --time=23:36:00" in instructions
    assert "TRAINING_ATTACK_ALLOCATION_SECONDS=84960" in instructions
    assert "srun --ntasks=1 --export=ALL" in instructions
    result = study.subprocess.run(["bash", "-n", str(directory / "run.sh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_provenance_rejects_changed_bank(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text("{}")
    plan = {"schema": common.SCHEMA, "source": common.sources(), "inputs": {str(path): common.digest(path)},
            "conditions": list(common.CONDITIONS), "seeds": list(common.SEEDS), "role": common.ROLE}
    common.verify(plan)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="input changed"):
        common.verify(plan)


def test_difference_in_attack_effect_is_paired_not_pooled():
    cells = []
    for seed in range(2):
        arms = {}
        for i, arm in enumerate(common.CONDITIONS):
            arms[arm] = {}
            for tid in ("a", "b"):
                for policy in ("honest", "attack"):
                    arms[arm][(tid, policy, 1)] = {
                        "invoked": i, "compromised_calls": i * 5, "aggregator_exposed": i,
                        "worker_used": i, "correct": policy == "honest",
                        "target_hit": i if policy == "attack" else 0}
        cells.append(arms)
    result = reporting.contrasts(cells, ("a", "b"), draws=1000)
    assert result["primary_helpful_minus_ordinary_honest"]["invoked"]["effect"] == 1
    key = "helpful-minus-ordinary-attack-effect/target_hit"
    assert result["secondary_attack_effects"][key]["effect"] == 1


def fake_study(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    training = [task(1, prefix="mmlu_pro/validation/"), task(2, prefix="mmlu_pro/validation/")]
    dev, evaluation = [task(3)], [task(4), task(5)]
    paths = {name: document(tmp_path / f"{name}.json", tasks)
             for name, tasks in (("training", training), ("dev", dev), ("eval", evaluation), ("reserved", []))}
    model = LLM()
    bank_path = str(tmp_path / "messages.json")
    trust.generate(SimpleNamespace(tasks=paths["training"], reserved=[paths["reserved"]],
                                   out=bank_path, role=common.ROLE), model)
    tiny = config("anymac-corrected")
    tiny["roles"] = list(learned.ROLES)
    tiny["max_workers"] = 5
    monkeypatch.setattr(worker, "new_config", lambda *a, **kw: copy.deepcopy(tiny))
    monkeypatch.setattr(train_learned, "Embedder", lambda *a: Embeddings())
    monkeypatch.setattr(learned, "Embedder", lambda *a: Embeddings())
    monkeypatch.setattr(train_learned, "OllamaChat", lambda *a: LLM())
    monkeypatch.setattr(worker, "OllamaChat", lambda *a: LLM())
    plan = {"schema": common.SCHEMA, "source": common.sources(), "conditions": list(common.CONDITIONS),
            "seeds": list(common.SEEDS), "role": common.ROLE, "out": str(tmp_path),
            "inputs": {path: common.digest(path) for path in [*paths.values(), bank_path]},
            "training_tasks": paths["training"], "dev_tasks": paths["dev"], "eval_tasks": paths["eval"],
            "reserved": [paths["reserved"]], "bank": bank_path, "model": "fake",
            "worker": model.identity(), "threads": 1, "noise_ids": [evaluation[0]["id"]],
            "assets": {"embedding": {"path": "fake", "sha256": Embeddings.sha256},
                       "backbone": {"path": None, "sha256": None}},
            "budget": {"trajectories": 64}}
    plan_path = tmp_path / "plan.json"
    common.write_json(plan_path, plan)
    return plan, plan_path


def test_real_tiny_training_evaluation_and_partial_report(tmp_path, monkeypatch):
    plan, plan_path = fake_study(tmp_path, monkeypatch)
    for arm in common.CONDITIONS:
        out = tmp_path / arm / "seed0"
        worker.train(plan, plan_path, 0, arm, "fake", out)
        worker.admit(out / "router.pt", plan, plan_path, 0, arm)
        metadata = json.loads((out / "router.json").read_text())
        assert metadata["completed_trajectories"] == 64
        assert metadata["config"]["max_workers"] == 5 and metadata["config"]["max_context"] == 5
        with pytest.raises(ValueError, match="runtime-smoke"):
            learned.load_system(out / "router.pt", "fake", require_gate=True)
    eval_out = tmp_path / "evaluation/seed0"
    eval_out.mkdir(parents=True)
    worker.evaluate(plan, plan_path, 0, "fake", eval_out)
    paired = reporting.checked_rows(eval_out, plan, plan_path, 0)
    assert all((tid, "attack", 1) in pair for pair in paired.values()
               for tid in ("mmlu_pro/4", "mmlu_pro/5"))
    common.write_json(tmp_path / "state.json", {"status": "incomplete", "failures": ["test stopped after seed0"]})
    summary = reporting.report(tmp_path)
    assert summary["complete_seeds"] == [0] and summary["incomplete_seeds"] == [1, 2, 3, 4, 5]
    assert not summary["execution_complete"]
    assert not summary["per_seed"]["0"]["performance"]["noninferiority_proven"]
    # A modified artifact is refused, not re-admitted through the clean loader.
    metadata_path = tmp_path / common.CONDITIONS[0] / "seed0/router.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["study_binding"]["calibration"] = True
    common.write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="unregistered"):
        worker.admit(metadata_path.with_suffix(".pt"), plan, plan_path, 0, common.CONDITIONS[0])


@pytest.mark.parametrize("recovering", [False, True])
def test_prepare_keeps_initial_dev_out_of_training_reservations(tmp_path, monkeypatch, recovering):
    base = tmp_path / "results/compare"
    pool = base / "learned_gemma12b_paper_v2/train_tasks.json"
    pool.parent.mkdir(parents=True)
    common.write_json(pool, {
        "tasks": [task(i, subject=f"subject{i % 14}", prefix="mmlu_pro/validation/") for i in range(70)],
        "source": {"split": "validation", "revision": study.DATA_REVISION},
        "selection": "all-eligible-validation",
        "option_preprocessing": "trim-trailing-na/preserve-letters/v1"})
    dev = base / "gemma12b_preflight_v1/dev_tasks.json"
    dev.parent.mkdir()
    document(dev, [task(i) for i in range(100, 110)])
    evaluation = base / "main_balanced200_v1/eval_tasks.json"
    evaluation.parent.mkdir()
    document(evaluation, [task(i, f"subject{i % 14}") for i in range(200, 250)])
    for path in ("screen/screen_tasks_v1.json", "screen/screen_tasks_r2.json",
                 "compare/dylan_study_v1/eval_tasks.json", "compare/main_reserved_v2/eval_tasks.json"):
        target = tmp_path / "results" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        document(target, [])
    assets = {}
    for name in ("embedding", "backbone"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        assets[name] = {"path": str(directory), "sha256": learned.tree_digest(directory)}
    asset_path = base / "learned_gemma12b_v1/assets.json"
    asset_path.parent.mkdir()
    common.write_json(asset_path, assets)
    monkeypatch.setattr(study, "OllamaChat", lambda *a: LLM())
    monkeypatch.setattr(study, "MODEL_DIGEST", "fake-digest")
    root = tmp_path / "study"
    root.mkdir()
    args = SimpleNamespace(results=str(tmp_path / "results"), model="fake",
                           urls=["gpu0", "gpu1"], threads=1, started=time.time(), deadline=time.time() + 100,
                           allocation_seconds=86400, recover_from=None)
    plan = study.prepare(args, root)
    assert str(dev) in plan["bank_reserved"] and str(dev) not in plan["reserved"]
    selected_dev = common.load_tasks(plan["dev_tasks"], reserved=plan["reserved"])
    assert len(selected_dev) == 4
    assert len(common.load_tasks(plan["eval_tasks"])) == 24
    if recovering:
        source = tmp_path / "job_7213910"
        source.mkdir()
        (source / "logs").mkdir()
        trust.generate(SimpleNamespace(
            tasks=plan["training_tasks"], reserved=plan["bank_reserved"],
            role=common.ROLE, out=str(source / "messages.json")), LLM())
        (source / "messages.json").unlink()
        progress_path = source / "messages.progress.json"
        progress = json.loads(progress_path.read_text())
        for tid, content in APPROVED_BAD:
            entry = progress["texts"][tid][content]
            entry.update(text="Incomplete answer", pred=None, correct=False)
            entry["usage"]["truncated"] = True
        common.write_json(progress_path, progress)
        common.write_json(source / "state.json", {
            "status": "incomplete", "training_complete_seeds": [], "started": 1000, "finished": 2430})
        common.write_json(source / "logs/bank.json",
                          {"stage": "bank", "returncodes": [1], "timed_out": False})
        args.recover_from, args.allocation_seconds = source, 84960
        recovered = study.prepare(args, root)
        common.verify(recovered)
        assert len(common.load_tasks(recovered["training_tasks"])) == 68
        assert recovered["allocation_accounting"]["prior_charged_seconds"] == 1440
        assert recovered["bank_recovery"]["retained_answers"] == 136


@pytest.mark.parametrize("recovery_mode", [False, True])
def test_full_orchestration_six_pairs_with_real_tiny_workers(tmp_path, monkeypatch, recovery_mode):
    root = tmp_path / "run"
    timings = []

    def prepare(args, directory):
        plan, _ = fake_study(directory, monkeypatch)
        plan.pop("budget")
        plan["bank_reserved"] = plan["reserved"]
        if recovery_mode:
            plan["bank_recovery"] = {"policy": "test-existing-frozen-bank"}
        return plan

    def execute(commands, deadline, directory, stage):
        timings.append((stage, deadline))
        if stage == "bank":
            return True  # The fixture generated the bank through the real trust generator.
        for command in commands:
            args = {command[i]: command[i + 1] for i in range(3, len(command), 2)}
            plan_path = Path(args["--plan"])
            plan = json.loads(plan_path.read_text())
            out = Path(args["--out"])
            seed, arm, url = int(args["--seed"]), args["--condition"], args["--url"]
            if command[2] in ("calibrate", "train"):
                worker.train(plan, plan_path, seed, arm, url, out, command[2] == "calibrate")
            elif command[2] == "stress":
                out.mkdir(parents=True)
                worker.stress(plan, url, out)
            else:
                out.mkdir(parents=True)
                worker.evaluate(plan, plan_path, seed, url, out)
        return True

    monkeypatch.setattr(study, "prepare", prepare)
    monkeypatch.setattr(study, "batch", execute)
    monkeypatch.setattr(study, "choose_budget", lambda *a: {
        "trajectories": 64, "pair_seconds": 2000, "evaluation_seconds": 4000,
        "reserve_seconds": 900, "training_seconds": 12000})
    start = time.time()
    args = SimpleNamespace(out=str(root), results="fake", urls=["gpu0", "gpu1"], model="fake",
                           threads=1, started=start, deadline=start + 23 * 3600)
    assert study.run(args) == 0
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete_seeds"] == list(common.SEEDS) and summary["execution_complete"]
    assert [s for s, _ in timings if s.startswith("training")] == [f"training-seed{i}" for i in range(6)]
    assert len([s for s, _ in timings if s.startswith("evaluation")]) == 3
    assert all(deadline <= args.deadline for _, deadline in timings)
    assert ("bank" in [s for s, _ in timings]) != recovery_mode


def test_failed_stage_surfaces_actual_child_error_in_parent_stderr(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    assert not study.batch(
        [[sys.executable, "-c", "raise ValueError('specific bank quality failure')"]],
        time.time() + 5, tmp_path, "bank")
    assert "specific bank quality failure" in capsys.readouterr().err
