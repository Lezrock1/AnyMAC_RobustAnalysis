import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def driver(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    (scripts / "compare").mkdir(parents=True)
    (scripts / "ramses").mkdir()
    for name in ("learned.sh", "suite.sh"):
        shutil.copyfile(Path(__file__).with_name(name), scripts / "compare" / name)
    results = repo / "results"
    for folder in ("screen", "compare/learned_gemma12b_v1", "compare/dylan_study_v1",
                   "compare/gemma12b_preflight_v1", "compare/learned_gemma12b_paper_v1",
                   "compare/learned_gemma12b_paper_v2",
                   "compare/main_reserved_v2", "compare/main_balanced200_v1"):
        (results / folder).mkdir(parents=True, exist_ok=True)
    for f in ("screen/screen_tasks_v1.json", "screen/screen_tasks_r2.json",
              "compare/dylan_study_v1/eval_tasks.json", "compare/gemma12b_preflight_v1/dev_tasks.json",
              "compare/learned_gemma12b_v1/train_tasks.json",
              "compare/learned_gemma12b_paper_v1/train_tasks.json",
              "compare/learned_gemma12b_paper_v2/train_tasks.json",
              "compare/main_reserved_v2/eval_tasks.json", "compare/main_balanced200_v1/eval_tasks.json"):
        (results / f).write_text(json.dumps({"tasks": [
            {"id": f + str(i), "text": "question", "truth": "A", "kind": "mc", "n_options": 4}
            for i in range(4)]}))
    (results / "compare/learned_gemma12b_v1/assets.json").write_text(
        json.dumps({"embedding": {"path": "/local/embedding"}, "backbone": {"path": "/local/gpt2"}}))
    (scripts / "ramses/lib.sh").write_text("""
load_env() { RESULTS_DIR="$PWD/results"; RAMSES_CACHE_DIR="$PWD/cache"; RAMSES_LOG_DIR="$PWD/logs"; DATA_DIR="$PWD/data"; }
load_modules() { :; }
activate_venv() { :; }
server_up() { [[ "${BUSY:-0}" == 1 ]]; }
start_ollama() { OLLAMA_STARTED_PID="$OLLAMA_PORT"; echo "start $OLLAMA_PORT $CUDA_VISIBLE_DEVICES" >> "$EVENTS"; }
stop_ollama() { echo "stop $OLLAMA_STARTED_PID" >> "$EVENTS"; }
""")
    tools = tmp_path / "bin"
    tools.mkdir()
    python_driver = tmp_path / "driver.py"
    python_driver.write_text("""
import json,os,subprocess,sys
a=sys.argv[1:]
if a[0] in ("-c", "-"):
    raise SystemExit(subprocess.call([sys.executable,*a]))
with open(os.environ["CALLS"],"a") as f: f.write(json.dumps(a)+"\\n")
if os.environ.get("FAIL_TRAIN") and a[1] == "train": raise SystemExit(3)
""")
    for name, body in {
        "python": 'exec "$TEST_PYTHON" "$TEST_DRIVER" "$@"',
        "curl": 'echo \'{"models":[{"name":"gemma4:12b-it-qat","digest":'
                '"38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3"}]}\'',
        "git": "echo mocked-commit",
    }.items():
        exe = tools / name
        exe.write_text("#!/bin/bash\n" + body + "\n")
        exe.chmod(0o755)
    calls, events = tmp_path / "calls", tmp_path / "events"
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "CALLS": str(calls), "EVENTS": str(events),
           "TEST_PYTHON": sys.executable, "TEST_DRIVER": str(python_driver), "SLURM_JOB_ID": "123",
           "SLURM_CPUS_PER_TASK": "8", "CUDA_VISIBLE_DEVICES": "3"}

    def invoke(script, *args, **overrides):
        result = subprocess.run(["bash", str(scripts / "compare" / script), *args],
                                capture_output=True, text=True, env={**env, **overrides}, timeout=20)
        recorded = [json.loads(r) for r in calls.read_text().splitlines()] if calls.exists() else []
        lifecycle = events.read_text().splitlines() if events.exists() else []
        return result, recorded, lifecycle
    return invoke


@pytest.mark.parametrize("system,samples", [("anymac-normal", "4"), ("anymac-efficient", "4"), ("gdesigner-paper", "10")])
def test_clean_train_then_gate(driver, system, samples):
    r, calls, events = driver("learned.sh", "train", system, "1")
    assert r.returncode == 0, r.stderr
    assert [c[1] for c in calls] == ["train", "gate"]
    train = calls[0]
    assert train[train.index("--samples") + 1] == samples
    assert train[train.index("--purpose") + 1] == "study"
    assert ("--backbone" in train) == system.startswith("anymac-")
    if system.startswith("anymac-"):
        assert train[train.index("--anymac-profile") + 1] == system.removeprefix("anymac-")
        assert "profiles_v2" in train[train.index("--out") + 1]
    assert len(events) == 2 and events[0].endswith(" 3") and events[1].startswith("stop")


def test_smoke_is_not_qualifying_checkpoint(driver):
    r, calls, _ = driver("learned.sh", "smoke", "anymac-normal")
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and calls[0][1] == "train"
    assert calls[0][calls[0].index("--purpose") + 1] == "runtime-smoke"
    assert "NOT a passed development gate" in r.stdout


@pytest.mark.parametrize("override", [{"CUDA_VISIBLE_DEVICES": ""}, {"CUDA_VISIBLE_DEVICES": "0,1"},
                                      {"SLURM_JOB_ID": ""}, {"BUSY": "1"}])
def test_no_accidental_unallocated_or_shared_gpu(driver, override):
    r, calls, events = driver("learned.sh", "train", "anymac-normal", **override)
    assert r.returncode and not calls and not events


def test_suite_partitions_mask_and_cleans_all_servers(driver):
    r, calls, events = driver("suite.sh", "pilot", CUDA_VISIBLE_DEVICES="GPU-a,GPU-b", SLURM_CPUS_PER_TASK="16")
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and calls[0][0] == "scripts/compare/suite.py"
    assert calls[0][calls[0].index("--threads") + 1] == "8"
    assert calls[0][calls[0].index("--urls") + 1:] == ["http://127.0.0.1:11454", "http://127.0.0.1:11455"]
    assert events == ["start 11454 GPU-a", "start 11455 GPU-b", "stop 11454", "stop 11455"]


def test_main_refuses_missing_cost_approval_before_starting_servers(driver):
    r, calls, events = driver("suite.sh", "main", COMPARE_GPU_BUDGET_HOURS="")
    assert r.returncode and "approve" in r.stderr and not calls and not events


def test_failed_training_never_reports_gate_success(driver):
    r, calls, events = driver("learned.sh", "train", "gdesigner-paper", FAIL_TRAIN="1")
    assert r.returncode == 3
    assert [c[1] for c in calls] == ["train"]
    assert len(events) == 2 and events[-1].startswith("stop")


def test_ambiguous_anymac_label_refused(driver):
    r, calls, events = driver("learned.sh", "train", "anymac-corrected")
    assert r.returncode and "explicitly" in r.stderr
    assert not calls and not events


def test_amended_pilot_plan_needs_no_gpu_and_has_separate_output(driver):
    r, calls, events = driver("suite.sh", "plan", "pilot", SLURM_JOB_ID="",
                             CUDA_VISIBLE_DEVICES="",
                             COMPARE_ROUTING_ADMISSION="routing-observation-20261026")
    assert r.returncode == 0, r.stderr
    assert not events and len(calls) == 1
    args = calls[0]
    assert args[1] == "plan"
    assert args[args.index("--phase") + 1] == "pilot"
    assert "--allow-routing-degeneracy" in args
    assert args[args.index("--out") + 1].endswith("five_condition_gemma12b_v3")
    assert args[args.index("--eval-tasks") + 1].endswith("main_balanced200_v1/eval_tasks.json")


def test_unknown_admission_is_rejected_before_gpu_start(driver):
    r, calls, events = driver("suite.sh", "pilot", COMPARE_ROUTING_ADMISSION="anything")
    assert r.returncode == 2
    assert "Unknown COMPARE_ROUTING_ADMISSION" in r.stderr
    assert not calls and not events


@pytest.mark.parametrize("mode,budget,samples", [("train", "1000", "8"), ("smoke", "4", "2")])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_paper_regime_is_isolated_and_explicit(driver, mode, budget, samples, version):
    r, calls, _ = driver("learned.sh", mode, "anymac-efficient",
                         COMPARE_TRAINING_REGIME=f"paper-aligned-{version}")
    assert r.returncode == 0, r.stderr
    train = calls[0]
    assert train[train.index("--trajectory-budget") + 1] == budget
    assert train[train.index("--samples") + 1] == samples
    assert train[train.index("--training-regime") + 1] == f"paper-aligned-{version}"
    assert f"learned_gemma12b_paper_{version}/" in train[train.index("--tasks") + 1]
    assert f"learned_gemma12b_paper_{version}/" in train[train.index("--out") + 1]
    assert any("main_balanced200_v1" in arg for arg in train)
    assert len(calls) == (2 if mode == "train" else 1)


def test_paper_regime_retains_existing_gdesigner_and_rejects_unknown_regime(driver):
    for system, regime, message in (("gdesigner-paper", "paper-aligned-v1", "40x10"),
                                    ("anymac-normal", "invented", "Unknown")):
        r, calls, events = driver("learned.sh", "train", system, COMPARE_TRAINING_REGIME=regime)
        assert r.returncode and message in r.stderr and not calls and not events


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_paper_suite_keeps_sealed_main_and_original_gdesigner_root(driver, version):
    r, calls, events = driver("suite.sh", "plan", "pilot",
                             COMPARE_TRAINING_REGIME=f"paper-aligned-{version}",
                             COMPARE_ROUTING_ADMISSION="routing-observation-20261026")
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and not events
    args = calls[0]
    assert args[args.index("--learned-root") + 1].endswith(f"learned_gemma12b_paper_{version}")
    assert args[args.index("--gdesigner-root") + 1].endswith("learned_gemma12b_v1")
    assert args[args.index("--out") + 1].endswith(f"five_condition_gemma12b_paper_{version}_routing")
    assert args[args.index("--train-tasks") + 1].endswith("learned_gemma12b_v1/train_tasks.json")
    assert args[args.index("--anymac-training-tasks") + 1].endswith(f"learned_gemma12b_paper_{version}/train_tasks.json")
