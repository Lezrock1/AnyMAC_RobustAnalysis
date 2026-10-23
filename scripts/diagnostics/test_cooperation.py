import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "compare"))
import cooperation
import learned
import runner
import scoring
from contract import Episode, validate


class FakeLLM:
    def __init__(self):
        self.calls = []

    def identity(self):
        return {"digest": "test", "options": {"num_thread": 8}}

    def __call__(self, messages):
        self.calls.append(messages)
        return "The answer is C", {"prompt_tokens": 10, "completion_tokens": 5, "truncated": False}


def test_independent_workers_and_exact_final_prompt():
    task = {"id": "dev/0", "kind": "mc", "text": "Q", "truth": "C", "n_options": 4}
    traces = []
    for condition in cooperation.CONDITIONS:
        llm = FakeLLM()
        ep = Episode(system=cooperation.SYSTEM, task=task, policy="honest", rep=1, compromised="Lawyer",
                     llm=llm, max_calls=3)
        traced = ep.run(lambda e: cooperation.execute(e, task, condition, learned, ["Lawyer", "Doctor"]))
        validate(traced)
        traced["diagnostic_condition"] = condition
        traces.append(traced)
        calls = [e for e in traced["events"] if e["type"] == "call"]
        assert len(calls) == (1 if condition == "alone" else 3)
        assert all(not e["inputs"] for e in calls[:-1])
        if condition == "independent-team":
            assert len(calls[-1]["inputs"]) == 2
    alone, team = [t["events"][-1]["request"] for t in traces]
    assert alone[0] == team[0]
    assert alone[1]["content"].startswith("Task:\nQ")
    assert team[1]["content"].endswith("\n\nWeigh the evidence and give your answer.")
    result = cooperation.analyze(traces, {task["id"]: task}, scoring, ["Lawyer", "Doctor"], 1)
    assert result["team_minus_alone"]["correct"]["effect"] == 0
    assert result["team_minus_alone"]["calls"]["effect"] == 2
    assert result["independent_workers"]["oracle_accuracy"] == 1
    assert result["transitions"]["harmed"] == 0


def args(tmp_path):
    tasks = [{"id": f"dev/{i}", "kind": "mc", "text": "Q", "truth": "C", "n_options": 4} for i in range(12)]
    tf = tmp_path / "dev.json"
    tf.write_text(json.dumps({"tasks": tasks}))
    meta = {"dev_ids": [t["id"] for t in tasks], "dev_sha256": cooperation.digest(tf),
            "source": runner.source_digest(), "system": "anymac-corrected", "purpose": "study",
            "worker": {"digest": "test", "options": {}}, "config": {"roles": ["Lawyer", "Doctor"]}}
    mf = tmp_path / "router.json"
    mf.write_text(json.dumps(meta))
    return SimpleNamespace(out=tmp_path / "out", metadata=mf, tasks=tf, reps=2)


def test_resumable_binding_and_partial_resume(tmp_path):
    a = args(tmp_path)
    llm = FakeLLM()
    result = cooperation.run(a, runner, scoring, learned, llm)
    assert len(llm.calls) == 96
    assert result["pairs"] == 24
    llm.calls.clear()
    cooperation.run(a, runner, scoring, learned, llm)
    assert not llm.calls
    p = a.out / "alone.jsonl"
    p.write_text("\n".join(p.read_text().splitlines()[1:]) + "\n")
    cooperation.run(a, runner, scoring, learned, llm)
    assert len(llm.calls) == 1
    a.reps = 3
    with pytest.raises(ValueError, match="NEW"):
        cooperation.run(a, runner, scoring, learned, llm)


def test_different_worker_refused_before_output(tmp_path):
    a = args(tmp_path)
    llm = FakeLLM()
    llm.identity = lambda: {"digest": "other"}
    with pytest.raises(ValueError, match="differs"):
        cooperation.run(a, runner, scoring, learned, llm)
    assert not a.out.exists()


def test_truncation_and_errors_are_not_silently_dropped():
    task = {"id": "dev/0", "kind": "mc", "text": "Q", "truth": "C", "n_options": 4}
    traces = []
    for c in cooperation.CONDITIONS:
        ep = Episode(system=cooperation.SYSTEM, task=task, policy="honest", rep=1, compromised="Lawyer",
                     llm=FakeLLM(), max_calls=3)
        t = ep.run(lambda e: cooperation.execute(e, task, c, learned, ["Lawyer"]))
        t["diagnostic_condition"] = c
        if c == "independent-team":
            for event in t["events"]:
                event["usage"]["truncated"] = True
        traces.append(t)
    r = cooperation.analyze(traces, {task["id"]: task}, scoring, ["Lawyer"], 1)
    assert r["transitions"]["harmed"] == 1
    assert r["independent_workers"]["oracle_accuracy"] == 0
    assert r["team_minus_alone"]["correct"]["effect"] == -1


def test_wrapper_uses_separate_port_and_original_source(tmp_path):
    import os
    import shutil
    import subprocess

    repo = tmp_path / "repo"
    (repo / "scripts/diagnostics").mkdir(parents=True)
    (repo / "scripts/ramses").mkdir()
    shutil.copy(HERE / "cooperation.sh", repo / "scripts/diagnostics/cooperation.sh")
    (repo / "scripts/ramses/lib.sh").write_text("""
load_env() { RESULTS_DIR="$PWD/results"; RAMSES_CACHE_DIR="$PWD/cache"; RAMSES_LOG_DIR="$PWD/logs"; }
load_modules() { :; }
activate_venv() { :; }
server_up() { [[ "${BUSY:-0}" == 1 ]]; }
start_ollama() { echo "start:$OLLAMA_PORT:$CUDA_VISIBLE_DEVICES" >> "$EVENTS"; }
stop_ollama() { echo stop >> "$EVENTS"; }
""")
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, content in {
        "git": 'echo 78623cfce260109c02753470226afff64ac6e9ee',
        "python": 'printf "%s\\n" "$@" > "$CALLS"',
    }.items():
        p = tools / name
        p.write_text("#!/bin/bash\n" + content + "\n")
        p.chmod(0o755)
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "SLURM_JOB_ID": "123",
           "CUDA_VISIBLE_DEVICES": "GPU-test", "EVENTS": str(tmp_path / "events"), "CALLS": str(tmp_path / "calls")}
    script = repo / "scripts/diagnostics/cooperation.sh"
    r = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "events").read_text().splitlines() == ["start:11456:GPU-test", "stop"]
    calls = (tmp_path / "calls").read_text()
    assert "AnyMAC_compare_full" in calls and "cooperation_dev12_v1" in calls
    r = subprocess.run(["bash", str(script)], env={**env, "BUSY": "1"}, text=True, capture_output=True)
    assert r.returncode != 0 and "foreign" in r.stderr
    r = subprocess.run(["bash", str(script)], env={**env, "CUDA_VISIBLE_DEVICES": "1,2"}, text=True, capture_output=True)
    assert r.returncode != 0 and "exactly one" in r.stderr
