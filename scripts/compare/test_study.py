import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

DIGEST = "38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3"


@pytest.fixture
def study(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    (scripts / "compare").mkdir(parents=True)
    (scripts / "ramses").mkdir()
    shutil.copyfile(Path(__file__).with_name("study.sh"), scripts / "compare/study.sh")
    binary = repo / "cache/ollama-screen-0.40.6/bin/ollama"
    binary.parent.mkdir(parents=True)
    binary.touch(mode=0o755)
    results = repo / "results"
    (results / "screen").mkdir(parents=True)
    for name in ("screen_tasks_v1.json", "screen_tasks_r2.json"):
        (results / "screen" / name).write_text('{"tasks":[]}')
    (results / "compare/gemma12b_preflight_v1").mkdir(parents=True)
    (results / "compare/gemma12b_preflight_v1/dev_tasks.json").write_text('{"tasks":[]}')
    (scripts / "ramses/lib.sh").write_text("""
load_env() { RESULTS_DIR="$PWD/results"; RAMSES_CACHE_DIR="$PWD/cache"; RAMSES_LOG_DIR="$PWD/logs"; DATA_DIR="$PWD/data"; }
load_modules() { :; }
activate_venv() { :; }
server_up() { [[ "${BUSY:-0}" == 1 ]]; }
start_ollama() { echo "start $OLLAMA_PORT $CUDA_VISIBLE_DEVICES" >> "$EVENTS"; }
stop_ollama() { echo "stop $OLLAMA_PORT" >> "$EVENTS"; }
""")
    tools = tmp_path / "bin"
    tools.mkdir()
    # fake python: records calls; `generate` writes the attack file, `run` marks the run complete
    driver = tmp_path / "driver.py"
    driver.write_text("""
import json, os, sys, time
a = sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps(a) + "\\n")
def opt(k): return a[a.index(k) + 1]
if a[1] == "generate":
    open(opt("--out"), "w").write("{}")
if a[1] == "check" and os.environ.get("GATE_FAIL"):
    sys.exit("quality gate failed")
if a[1] == "run":
    time.sleep(0.05)
    if os.environ.get("FAIL_ROLE") and os.environ["FAIL_ROLE"] == opt("--compromised") and "attack" in a:
        sys.exit(3)
    open(os.path.join(opt("--out"), "COMPLETE.json"), "w").write("{}")
""")
    flock = tools / "flock"  # util-linux semantics on an inherited FD (absent on macOS)
    flock.write_text('#!/bin/bash\nexec "$TEST_PYTHON" -c "import fcntl, sys\ntry: fcntl.flock(int(sys.argv[-1]), '
                     'fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept OSError: sys.exit(1)" "$@"\n')
    flock.chmod(0o755)
    for name, body in {
        "python": 'exec "$TEST_PYTHON" "$TEST_DRIVER" "$@"',
        "curl": f'[[ "${{MODEL_MISSING:-0}}" == 1 ]] && echo "{{}}" || echo \'{{"models":[{{"digest":"{DIGEST}"}}]}}\'',
        "nvidia-smi": 'for ((i=0; i<${NGPU:-2}; i++)); do echo $i; done',
        "git": "echo mocked-commit",
        "hostname": "echo node",
    }.items():
        exe = tools / name
        exe.write_text("#!/bin/bash\n" + body + "\n")
        exe.chmod(0o755)
    calls, events = tmp_path / "calls", tmp_path / "events"
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "CALLS": str(calls), "EVENTS": str(events),
           "SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "16", "TEST_PYTHON": sys.executable, "TEST_DRIVER": str(driver)}

    def invoke(mode, prepared=True, **overrides):
        if prepared:
            base = results / "compare/dylan_study_v1"
            base.mkdir(parents=True, exist_ok=True)
            (base / "eval_tasks.json").write_text('{"tasks":[]}')
        r = subprocess.run(["bash", str(scripts / "compare/study.sh"), mode], env={**env, **overrides},
                           capture_output=True, text=True, timeout=120)
        rec = [json.loads(s) for s in calls.read_text().splitlines()] if calls.exists() else []
        ev = events.read_text().splitlines() if events.exists() else []
        calls.unlink(missing_ok=True)
        events.unlink(missing_ok=True)
        return r, rec, ev

    invoke.base = results / "compare/dylan_study_v1"
    return invoke


def opt(call, k):
    return call[call.index(k) + 1]


def test_prepare_excludes_screening_and_preflight(study):
    r, calls, ev = study("prepare", prepared=False)
    assert r.returncode == 0, r.stderr
    (c,) = calls
    assert c[:2] == ["scripts/screen/screen.py", "prepare"] and opt(c, "--n") == "60" and opt(c, "--seed") == "20261022"
    assert [Path(p).name for p in c[c.index("--exclude") + 1:]] == ["screen_tasks_v1.json", "screen_tasks_r2.json", "dev_tasks.json"]
    assert ev == []


def test_run_generates_attacks_then_runs_every_job_once(study):
    r, calls, ev = study("run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c[1] for c in calls[:2]] == ["generate", "check"]
    runs = [c for c in calls if c[1] == "run"]
    dirs = [opt(c, "--out").split("dylan_study_v1/")[1] for c in runs]
    assert len(runs) == 37 and len(set(dirs)) == 37  # no run executed twice across the two GPUs
    honest = [c for c in runs if "honest" in c]
    assert sorted(opt(c, "--system") for c in honest) == ["dylan", "dylan-upstream", "fixed"]
    assert all(opt(c, "--reps") == "2" for c in honest)
    attack = [c for c in runs if "attack" in c[c.index("--policies") + 1:c.index("--reps")]]
    pairs = {(opt(c, "--system"), opt(c, "--compromised"), opt(c, "--attack-form")) for c in attack}
    assert len(pairs) == 34 and ("fixed", "Knowledgeable Expert", "persuasive") in pairs
    assert {p[1] for p in pairs if p[0] == "dylan"} == {p[1] for p in pairs if p[0] == "dylan-upstream"} and \
        len({p[1] for p in pairs if p[0] == "dylan"}) == 7
    for c in attack:
        assert ("--attack-texts" in c) == (opt(c, "--attack-form") == "persuasive")
    for c in runs:
        assert opt(c, "--model") == "gemma4:12b-it-qat" and opt(c, "--limit") == "8192" and opt(c, "--threads") == "8"
        assert [Path(p).name for p in c[c.index("--reserved") + 1:c.index("--model")]][-1] == "dev_tasks.json"
    assert sorted({opt(c, "--url") for c in runs}) == ["http://127.0.0.1:11450", "http://127.0.0.1:11451"]
    assert sorted(e for e in ev if e.startswith("start")) == ["start 11450 0", "start 11450 0", "start 11451 1"]
    assert len([e for e in ev if e.startswith("stop")]) == 3
    # rerun: attack texts frozen, all runs complete -> no generation, no run
    r, calls, _ = study("run")
    assert r.returncode == 0 and [c[1] for c in calls] == ["check"]


def test_failed_run_does_not_stop_queue(study):
    r, calls, _ = study("run", FAIL_ROLE="Doctor")
    assert r.returncode == 1 and "FAILED attack/bare/dylan/Doctor" in r.stdout
    assert len([c for c in calls if c[1] == "run"]) == 37
    r, calls, _ = study("run")  # resume runs only the 4 failed DyLAN Doctor jobs
    assert r.returncode == 0 and sorted(opt(c, "--compromised") for c in calls if c[1] == "run") == ["Doctor"] * 4


@pytest.mark.parametrize("overrides,message", [
    ({"BUSY": "1"}, "occupied"),
    ({"MODEL_MISSING": "1"}, "no automatic pull"),
    ({"SLURM_JOB_ID": ""}, "A30 allocation"),
    ({"GATE_FAIL": "1"}, "quality gate"),
])
def test_run_refuses_unsafe_states(study, overrides, message):
    r, calls, _ = study("run", **overrides)
    assert r.returncode != 0 and message in r.stdout + r.stderr
    assert not [c for c in calls if c[1] == "run"]


def test_slurm_gpu_list_wins(study):
    r, calls, ev = study("run", CUDA_VISIBLE_DEVICES="3", NGPU="4")
    assert r.returncode == 0, r.stderr
    assert sorted(set(e for e in ev if e.startswith("start"))) == ["start 11450 3"]


def test_report_without_server(study):
    r, calls, ev = study("report", SLURM_JOB_ID="")
    assert r.returncode == 0, r.stderr
    assert "complete runs: 0/37" in r.stdout and ev == [] and calls == []
