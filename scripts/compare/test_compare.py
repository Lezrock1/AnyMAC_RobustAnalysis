import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "screen"))
import contract  # noqa: E402
import runner  # noqa: E402
import scoring  # noqa: E402
from baseline import FixedCycle  # noqa: E402
from contract import ATTACK_TEMPLATE, Episode, target_for, validate  # noqa: E402

MC = {"id": "mmlu_pro/1", "benchmark": "mmlu_pro", "kind": "mc", "n_options": 4, "truth": "C", "text": "Q? A. a B. b C. c D. d"}
INT = {"id": "masbench/7", "benchmark": "masbench", "kind": "int", "truth": 5, "text": "compute"}
ROLES = ["Expert", "Critic", "Analyst"]


class FakeLLM:
    """Honest workers answer the truth; the decider is deliberately gullible: it follows a message that
    claims to be verified, else the last answer shown (so a delivered attack can win). Records prompts."""

    def __init__(self, task, fail_at=None, truncate_role=None):
        self.task, self.fail_at, self.truncate_role, self.prompts = task, fail_at, truncate_role, []

    def __call__(self, messages):
        self.prompts.append(messages)
        if self.fail_at is not None and len(self.prompts) == self.fail_at:
            raise RuntimeError("HTTP 500")
        role = re.match(r"You are (.+?)\.", messages[0]["content"]).group(1)
        content = messages[1]["content"]
        answers = re.findall(r"The answer is (\S+)", content)
        claimed = re.findall(r"verified every step\.\nThe answer is (\S+)", content)
        ans = (claimed or answers)[-1] if role == "Decider" and answers else self.task["truth"]
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "truncated": role == self.truncate_role, "latency_s": 0.01}
        return f"{role} reasoning.\nThe answer is {ans}", usage


def episode(task=MC, policy="honest", compromised="Critic", llm=None, max_calls=10, system=None):
    system = system or FixedCycle(ROLES)
    llm = llm or FakeLLM(task)
    ep = Episode(system="fixed", task=task, policy=policy, rep=1, compromised=compromised, llm=llm, max_calls=max_calls)
    return ep.run(lambda e: system(e, task)), llm


def test_target_is_fixed_wrong_and_deterministic():
    for i in range(50):
        t = dict(MC, id=f"x/{i}")
        assert target_for(t) != "C" and target_for(t) == target_for(dict(t)) and target_for(t) in "ABD"
        n = dict(INT, id=f"y/{i}")
        assert target_for(n) != 5 and 0 <= target_for(n) < 23


def test_parser_matches_screen_v2():
    import screen
    cases = ["x\nThe answer is D. Not wrong, Not wrong", "The answer is (B)", "The answer is B or C", "The answer is 84.",
             "The answer is 15. Done", "The answer is $\\boxed{7}$", "The answer is 22", "", "The answer is E"]
    for c in cases:
        for task in (MC, INT):
            assert scoring.strict_answer(c, task) == screen.strict_answer(c, task), c


def test_honest_fixed_cycle_trace_and_exposure():
    t, llm = episode()
    assert validate(t) and t["stop"]["reason"] == "native" and len(llm.prompts) == 4
    assert not any(m["substituted"] for m in t["messages"].values())
    r = scoring.score(t, MC)
    assert r["outcome"] == "correct" and r["invoked"] == 1 and not r["target_hit"]
    # Critic is the 2nd of 3 workers: its message reaches Analyst and the decider
    assert r["delivered"] == {"worker": 1, "aggregator": 1} and r["aggregator_exposed"] and r["calls"] == 4


def test_attack_substitutes_only_after_real_call_and_is_delivered():
    th, lh = episode(policy="honest")
    ta, la = episode(policy="attack")
    comp = [m for m in ta["messages"].values() if m["author"] == "Critic"][0]
    assert comp["substituted"] and comp["text"] == ATTACK_TEMPLATE.format(target=ta["target"])
    # the real call happened with unchanged input, and its honest output is kept
    assert la.prompts[1] == lh.prompts[1] and comp["honest_text"].endswith("The answer is C")
    assert ATTACK_TEMPLATE.format(target=ta["target"]) in la.prompts[2][1]["content"]  # delivered to next worker
    assert all(not m["substituted"] for m in ta["messages"].values() if m["author"] != "Critic")
    r = scoring.score(ta, MC)
    assert r["outcome"] == "target_hit" and r["target_hit"] and r["pred"] == ta["target"]


def test_last_position_attack_reaches_only_aggregator():
    r = scoring.score(episode(policy="attack", compromised="Analyst")[0], MC)
    assert r["delivered"] == {"aggregator": 1} and r["target_hit"]


def test_budget_abort_is_explicit_and_never_repaired():
    t, llm = episode(max_calls=3)
    assert t["stop"]["reason"] == "budget" and t["final"] is None and len(llm.prompts) == 3
    assert scoring.score(t, MC)["outcome"] == "budget_abort"


def test_backend_error_recorded_without_default():
    t, _ = episode(llm=FakeLLM(MC, fail_at=2))
    assert t["stop"]["reason"] == "error" and t["events"][-1]["error"] and t["final"] is None
    assert scoring.score(t, MC)["outcome"] == "backend_error"


def test_truncated_final_and_invalid_format():
    t, _ = episode(llm=FakeLLM(MC, truncate_role="Decider"))
    assert scoring.score(t, MC)["outcome"] == "truncated_final"
    t, _ = episode()
    t["final"]["text"] = "I think C"
    assert scoring.score(t, MC)["outcome"] == "invalid_format"


@pytest.mark.parametrize("tamper", ["unsubstituted", "honest_substituted", "unknown_input", "final_on_budget",
                                    "orphan_message", "over_budget"])
def test_validate_rejects_contract_violations(tamper):
    policy = "honest" if tamper == "honest_substituted" else "attack"
    t, _ = episode(policy=policy)
    t = json.loads(json.dumps(t))
    msgs = t["messages"]
    if tamper == "unsubstituted":
        next(m for m in msgs.values() if m["author"] == "Critic")["substituted"] = False
    elif tamper == "honest_substituted":
        next(m for m in msgs.values() if m["author"] == "Critic").update(substituted=True, honest_text="x")
    elif tamper == "unknown_input":
        t["events"][0]["inputs"] = ["m9"]
    elif tamper == "final_on_budget":
        t["stop"] = {"reason": "budget", "detail": ""}
    elif tamper == "orphan_message":
        msgs["m99"] = {"author": "Expert", "kind": "worker", "text": "", "substituted": False}
    elif tamper == "over_budget":
        t["max_calls"] = 2
    with pytest.raises(ValueError):
        validate(t)


def test_programmatic_final_requires_aggregator_consume():
    class Vote:
        def __init__(self, record):
            self.record = record

        def __call__(self, ep, task):
            mids = [ep.call("worker", r, [{"role": "system", "content": f"You are {r}."},
                                          {"role": "user", "content": task["text"]}]) for r in ROLES]
            if self.record:
                ep.consume("aggregator", "vote", mids, note="majority")
            ep.finish(final_text="The answer is C")

    t, _ = episode(system=Vote(True), compromised="Expert", policy="attack")
    r = scoring.score(t, MC)
    assert r["aggregator_exposed"] and not r["final_from_compromised"] and r["outcome"] == "correct"
    with pytest.raises(ValueError, match="aggregator consume"):
        episode(system=Vote(False))


def test_paired_effect():
    rows = [{"system": "s", "task_id": f"t{i}", "policy": p, "rep": 1, "correct": c}
            for i in range(10) for p, c in (("honest", True), ("attack", i < 6))]
    e = scoring.paired_effect(rows, "s", "correct", n_boot=500)
    assert e["tasks"] == 10 and abs(e["diff"] + 0.4) < 1e-9 and e["ci95"][0] <= -0.4 <= e["ci95"][1]


def _args(tmp_path, tasks_file, **kw):
    a = ["run", "--system", "fixed", "--tasks", str(tasks_file), "--out", str(tmp_path / "out"), "--model", "fake",
         "--max-calls", "10", "--compromised", "Critic", "--reps", "2", "--roles", *ROLES]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", *([v] if isinstance(v, str) else v)]
    return a


def _parse(argv):
    captured = {}

    def fake_dispatch(args):
        captured["args"] = args
    orig = runner.run
    runner.run = fake_dispatch
    try:
        runner.main(argv)
    finally:
        runner.run = orig
    return captured["args"]


def test_runner_end_to_end_resume_and_bindings(tmp_path, capsys):
    tasks = [dict(MC, id=f"mmlu_pro/{i}") for i in range(4)]
    tf = tmp_path / "tasks.json"
    tf.write_text(json.dumps({"tasks": tasks}))
    ident = {"digest": "sha256:abc"}
    args = _parse(_args(tmp_path, tf))
    runner.run(args, llm=FakeLLM(MC), identity=ident)
    traces = [json.loads(ln) for ln in (tmp_path / "out" / "traces.jsonl").read_text().splitlines()]
    assert len(traces) == 4 * 2 * 2 and (tmp_path / "out" / "COMPLETE.json").exists()
    first = {}
    for t in traces:
        first.setdefault((t["task_id"], t["rep"]), t["policy"])
    assert sorted(first.values()).count("honest") == 4  # balanced policy order
    # resume: nothing new, no LLM call
    llm = FakeLLM(MC)
    runner.run(args, llm=llm, identity=ident)
    assert llm.prompts == []
    # changed model digest refuses
    with pytest.raises(SystemExit, match="NEW --out"):
        runner.run(args, llm=FakeLLM(MC), identity={"digest": "sha256:other"})
    # reserved screening ids refuse
    res = tmp_path / "screen.json"
    res.write_text(json.dumps({"tasks": [tasks[0]]}))
    with pytest.raises(SystemExit, match="reserved"):
        runner.run(_parse(_args(tmp_path, tf, out=str(tmp_path / "o2"), reserved=[str(res)])), llm=FakeLLM(MC), identity=ident)
    runner.main(["report", "--tasks", str(tf), "--out", str(tmp_path / "out")])
    summ = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summ["cells"]["fixed/honest"]["accuracy"] == 1.0 and summ["cells"]["fixed/attack"]["target_hit_rate"] == 1.0
    eff = {e["metric"]: e["diff"] for e in summ["paired"]}
    assert eff == {"correct": -1.0, "target_hit": 1.0}


def test_compromised_must_be_worker(tmp_path):
    tf = tmp_path / "tasks.json"
    tf.write_text(json.dumps({"tasks": [MC]}))
    with pytest.raises(SystemExit, match="compromised"):
        runner.run(_parse(_args(tmp_path, tf, compromised="Decider")), llm=FakeLLM(MC), identity={})


def test_contract_constants():
    assert set(scoring.OUTCOMES) >= {"budget_abort", "backend_error", "truncated_final", "invalid_format"}
    assert contract.STOP_REASONS == ("native", "budget", "error")


@pytest.fixture
def fake_ollama():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    state = {"bodies": [], "caps": ["completion", "thinking"], "reply": {}}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send(200, {"models": [{"name": "m:1", "model": "m:1", "digest": "abc"}]})
            else:
                self._send(200, {"version": "0.40.6"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                return self._send(200, {"capabilities": state["caps"]})
            state["bodies"].append(body)
            r = {"done": True, "done_reason": "stop", "message": {"content": "The answer is C"},
                 "prompt_eval_count": 7, "eval_count": 3, **state["reply"]}
            self._send(r.pop("_code", 200), r)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", state
    srv.shutdown()


def test_ollama_transport(fake_ollama, monkeypatch):
    from llm import OllamaChat
    url, state = fake_ollama
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    llm = OllamaChat(url, "m:1", limit=100, context=1000)
    text, usage = llm([{"role": "user", "content": "q"}])
    b = state["bodies"][-1]
    assert text == "The answer is C" and not usage["truncated"] and usage["prompt_tokens"] == 7
    assert b["think"] is False and b["options"]["num_thread"] == 8 and b["options"]["temperature"] == 0
    assert llm.identity()["digest"] == "abc"
    state["reply"] = {"done_reason": "length"}
    assert llm([{"role": "user", "content": "q"}])[1]["truncated"]
    for bad in ({"_code": 500, "error": "boom"}, {"message": {"content": "x", "thinking": "hmm"}},
                {"done": False}, {"prompt_eval_count": 990}):
        state["reply"] = bad
        with pytest.raises(RuntimeError):
            llm([{"role": "user", "content": "q"}])
    state["caps"] = ["completion"]
    state["reply"] = {}
    OllamaChat(url, "m:1")([{"role": "user", "content": "q"}])
    assert "think" not in state["bodies"][-1]
