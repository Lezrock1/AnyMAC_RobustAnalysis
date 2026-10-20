import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def preflight(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    (scripts / "compare").mkdir(parents=True)
    (scripts / "ramses").mkdir()
    shutil.copyfile(Path(__file__).with_name("preflight.sh"), scripts / "compare/preflight.sh")
    cache = repo / "cache"
    binary = cache / "ollama-screen-0.40.6/bin/ollama"
    binary.parent.mkdir(parents=True)
    binary.touch(mode=0o755)
    results = repo / "results"
    (results / "screen").mkdir(parents=True)
    for name in ("screen_tasks_v1.json", "screen_tasks_r2.json"):
        (results / "screen" / name).write_text('{"tasks":[]}')
    base = results / "compare/gemma12b_preflight_v1"
    base.mkdir(parents=True)
    (base / "dev_tasks.json").write_text('{"tasks":[]}')
    (scripts / "ramses/lib.sh").write_text("""
load_env() {
  RESULTS_DIR="$PWD/results"
  RAMSES_CACHE_DIR="$PWD/cache"
  RAMSES_LOG_DIR="$PWD/logs"
  DATA_DIR="$PWD/data"
}
load_modules() { :; }
activate_venv() { :; }
server_up() { [[ "${BUSY:-0}" == 1 ]]; }
start_ollama() { echo start >> "$EVENTS"; }
stop_ollama() { echo stop >> "$EVENTS"; }
""")
    tools = tmp_path / "bin"
    tools.mkdir()
    fake = tools / "python"
    driver = tmp_path / "record_calls.py"
    driver.write_text("""
import json, os, sys
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
""")
    fake.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$TEST_DRIVER" "$@"\n')
    fake.chmod(0o755)
    for name, body in {
        "curl": 'exit "${MODEL_MISSING:-0}"',
        "nvidia-smi": "exit 0",
        "git": "echo mocked-commit",
    }.items():
        exe = tools / name
        exe.write_text("#!/bin/bash\n" + body + "\n")
        exe.chmod(0o755)
    calls, events = tmp_path / "calls", tmp_path / "events"
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
           "CALLS": str(calls), "EVENTS": str(events), "SLURM_JOB_ID": "123",
           "TEST_PYTHON": sys.executable, "TEST_DRIVER": str(driver)}

    def invoke(mode, **overrides):
        result = subprocess.run(["bash", str(scripts / "compare/preflight.sh"), mode],
                                env={**env, **overrides}, capture_output=True, text=True)
        recorded = [json.loads(s) for s in calls.read_text().splitlines()] if calls.exists() else []
        lifecycle = events.read_text().splitlines() if events.exists() else []
        return result, recorded, lifecycle

    return invoke


def test_preflight_prepare_reserves_both_samples(preflight, monkeypatch):
    result, calls, lifecycle = preflight("prepare")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][:2] == ["scripts/screen/screen.py", "prepare"]
    assert calls[0][calls[0].index("--n") + 1] == "12"
    excluded = calls[0][calls[0].index("--exclude") + 1:]
    assert [Path(p).name for p in excluded] == ["screen_tasks_v1.json", "screen_tasks_r2.json"]
    assert lifecycle == []
    screen = runpy.run_path(str(Path(__file__).parents[1] / "screen/screen.py"))
    prepared = []
    monkeypatch.setitem(screen["main"].__globals__, "prepare", prepared.append)
    screen["main"](calls[0][1:])
    assert prepared[0].benchmarks == ["mmlu_pro"] and prepared[0].n == 12


def test_preflight_runs_only_honest_available_adapters(preflight):
    result, calls, lifecycle = preflight("run")
    assert result.returncode == 0, result.stderr
    runs = [c for c in calls if c[1] == "run"]
    assert [c[c.index("--system") + 1] for c in runs] == ["fixed", "dylan"]
    for call in runs:
        assert call[call.index("--policies") + 1:call.index("--reps")] == ["honest"]
        assert call[call.index("--model") + 1] == "gemma4:12b-it-qat"
        assert call[call.index("--limit") + 1] == "8192"
        assert call[call.index("--context") + 1] == "32768"
    assert len(calls) == 4 and lifecycle == ["start", "stop"]


@pytest.mark.parametrize("overrides,message,lifecycle", [
    ({"BUSY": "1"}, "occupied", []),
    ({"MODEL_MISSING": "1"}, "no automatic pull", ["start", "stop"]),
    ({"SLURM_JOB_ID": ""}, "separate A30 allocation", []),
])
def test_preflight_refuses_unsafe_or_missing_resources(preflight, overrides, message, lifecycle):
    result, calls, actual = preflight("run", **overrides)
    assert result.returncode != 0
    assert message in result.stderr
    assert calls == [] and actual == lifecycle


def test_preflight_report_needs_no_server(preflight):
    result, calls, lifecycle = preflight("report", SLURM_JOB_ID="")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2 and all(c[1] == "report" for c in calls)
    assert lifecycle == []
