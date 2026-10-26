import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen  # noqa: E402

MC = {"kind": "mc", "n_options": 4}
INT = {"kind": "int"}


def test_strict_answer():
    assert screen.strict_answer("work\nThe answer is B", MC) == "B"
    assert screen.strict_answer("**The answer is (C).**", MC) == "C"
    assert screen.strict_answer("The answer is $\\boxed{D}$", MC) == "D"
    assert screen.strict_answer("The answer is E", MC) is None  # outside option range
    assert screen.strict_answer("The answer is B\nmore text", MC) is None
    assert screen.strict_answer("The answer is 7", INT) == 7
    assert screen.strict_answer("The answer is 23", INT) is None
    assert screen.strict_answer("The answer is B", INT) is None
    assert screen.strict_answer("", MC) is None
    # parser v2: option text after the letter is accepted, ambiguity is not
    assert screen.strict_answer("The answer is D. Not wrong, Not wrong", MC) == "D"
    assert screen.strict_answer("The answer is (B) Wrong, Not wrong", MC) == "B"
    assert screen.strict_answer("The answer is A lot", MC) is None
    assert screen.strict_answer("The answer is B or C", MC) is None
    assert screen.strict_answer("The answer is 15. Done", INT) is None
    assert screen.strict_answer("The answer is 84.", INT) is None
    assert screen.unreduced_int("x\nThe answer is 84.") == 84
    assert screen.unreduced_int("The answer is D") is None


def test_mc_rows_keep_official_letters():
    pro = pd.DataFrame([{"question_id": 1, "category": "law", "question": "q", "options": ["a", "b", "N/A"], "answer": "B"},
                        {"question_id": 2, "category": "law", "question": "q", "options": ["a", "b", "c"], "answer": "C"}])
    rows = screen.mc_rows("mmlu_pro", pro)
    assert [r["id"] for r in rows] == ["mmlu_pro/2"] and rows[0]["truth"] == "C" and "C. c" in rows[0]["text"]
    mmlu = pd.DataFrame([{"subject": "x", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 3}])
    assert screen.mc_rows("mmlu", mmlu)[0]["truth"] == "D"


def test_trailing_na_is_opt_in_and_preserves_gold_letters():
    pro = pd.DataFrame([{"question_id": 1, "category": "law", "question": "q",
                         "options": ["a", "b", "c", " N/A ", "N/A"], "answer": "C", "answer_index": 2}])
    assert screen.mc_rows("mmlu_pro", pro) == []
    rows = screen.mc_rows("mmlu_pro", pro, trim_trailing_na=True)
    assert rows[0]["truth"] == "C" and rows[0]["n_options"] == 3
    assert rows[0]["text"] == "q\n\nA. a\nB. b\nC. c"
    assert list(pro.iloc[0]["options"]) == ["a", "b", "c", " N/A ", "N/A"]


@pytest.mark.parametrize("options,answer,index", [
    (["a", "N/A", "c"], "C", 2), (["a", "", "c"], "C", 2),
    (["a", "b", "N/A"], "C", 2), (["N/A", "N/A"], "A", 0),
    (["a", "N/A"], "A", 0), (["a", "b"], "B", 0),
])
def test_trimming_refuses_internal_padding_and_invalid_gold(options, answer, index):
    pro = pd.DataFrame([{"question_id": 1, "category": "law", "question": "q",
                         "options": options, "answer": answer, "answer_index": index}])
    with pytest.raises(ValueError, match="mmlu_pro/1"):
        screen.mc_rows("mmlu_pro", pro, trim_trailing_na=True)


def test_end_to_end_with_fake_ollama(tmp_path):
    from aiohttp import web
    import threading
    import asyncio

    calls = []

    async def chat(request):
        body = await request.json()
        calls.append(body)
        assert body["think"] is False and body["options"]["temperature"] == 0  # fake model reports thinking
        text = body["messages"][-1]["content"]
        answer = "The answer is A" if "Task" in text and "integer" not in body["messages"][0]["content"] else "The answer is 5"
        reason = "length" if "TRUNC" in text else "stop"
        return web.json_response({"done": True, "done_reason": reason, "message": {"content": f"ok\n{answer}"},
                                  "prompt_eval_count": 10, "eval_count": 5, "eval_duration": 1e8})

    async def tags(request):
        return web.json_response({"models": [{"name": "m:1", "model": "m:1", "digest": "abc", "size": 1}]})

    async def show(request):
        return web.json_response({"capabilities": ["completion", "thinking"]})

    async def version(request):
        return web.json_response({"version": "0.0.test"})

    app = web.Application()
    app.add_routes([web.post("/api/chat", chat), web.get("/api/tags", tags), web.get("/api/version", version),
                    web.post("/api/show", show)])
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    threading.Thread(target=loop.run_forever, daemon=True).start()

    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [
        {"id": "mmlu/x/1", "benchmark": "mmlu", "kind": "mc", "n_options": 4, "text": "q", "truth": "A"},
        {"id": "mmlu/x/2", "benchmark": "mmlu", "kind": "mc", "n_options": 4, "text": "TRUNC", "truth": "B"},
        {"id": "masbench/1", "benchmark": "masbench", "kind": "int", "text": "p", "truth": 5}]}))
    out = tmp_path / "out"
    argv = ["run", "--tasks", str(tasks), "--out", str(out), "--model", "m:1", "--url", f"http://127.0.0.1:{port}"]
    screen.main(argv)
    assert len(calls) == 2 * (3 * 1 + 3 * 3)
    screen.main(argv)  # resume: nothing new
    assert len(calls) == 24
    screen.main(["report", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    s = summary["m__1/mmlu/single"]
    assert s["accuracy"] == 0.5 and s["final_truncated"] == 0.5 and s["repeat_agreement"] == 1.0 and not s["eligible"]
    assert summary["m__1/masbench/chain3"]["accuracy"] == 1.0
    assert "(INCOMPLETE)" not in (out / "REPORT.md").read_text()
    with pytest.raises(SystemExit):
        screen.main(argv[:-2] + ["--url", f"http://127.0.0.1:{port}", "--limit", "8192"])  # manifest mismatch
    loop.call_soon_threadsafe(loop.stop)


def test_round2_sample_excludes_reserved_and_round1_unchanged():
    import random
    rows = [{"id": f"x/{i}"} for i in range(100)]
    assert screen.sample_rows(rows, 10, random.Random(1), set()) == random.Random(1).sample(rows, 10)
    excl = {f"x/{i}" for i in range(50)}
    picked = screen.sample_rows(rows, 40, random.Random(2), excl)
    assert len(picked) == 40 and not excl & {r["id"] for r in picked}


def test_tolerant_fallback_uses_wilson_but_keeps_strict_parts():
    base = {"n": 80, "errors": 0, "thinking_leak": 0, "format_valid": 0.975, "final_truncated": 0.025,
            "repeat_agreement": 0.85, "repeat_agreement_pairs": 40, "accuracy": 0.775}
    assert not screen.eligible(base, "chain3") and screen.tolerant(base, "chain3")  # Gemma r1 chain3: 0.85 in [0.71, 0.93]
    assert not screen.tolerant({**base, "repeat_agreement": 0.68}, "chain3")        # GLM r1: [0.52, 0.81]
    assert not screen.tolerant({**base, "final_truncated": 0.125}, "single")       # Qwen3-30B r1: [0.07, 0.22]
    assert not screen.tolerant({**base, "errors": 1}, "chain3")
    assert not screen.tolerant({**base, "accuracy": 0.875}, "single")             # window stays strict
    lo, hi = screen.wilson(0.85, 40)
    assert abs(lo - 0.709) < 1e-3 and abs(hi - 0.929) < 1e-3


def test_mtp_check_decision(tmp_path, monkeypatch):
    import mtp_check
    task = {"id": "mmlu_pro/1", "benchmark": "mmlu_pro", "kind": "mc", "n_options": 4, "text": "q", "truth": "A"}
    (tmp_path / "t.json").write_text(json.dumps({"tasks": [task]}))
    ref = {"task_id": "mmlu_pro/1", "benchmark": "mmlu_pro", "error": None, "pred": "A", "outputs": ["x"],
           "calls": [{"completion_tokens": 100, "eval_s": 4.0}]}
    (tmp_path / "r.jsonl").write_text(json.dumps(ref) + "\n")

    async def fake_episode(session, url, model, t, condition, rep, args):
        assert args.limit == 4096 and args.context == 16384 and condition == "single"
        return {"pred": "A", "error": None, "outputs": ["x"], "calls": [{"completion_tokens": 100, "eval_s": 2.0}]}
    monkeypatch.setattr(mtp_check.screen, "episode", fake_episode)
    monkeypatch.setattr(mtp_check.screen, "capabilities", lambda u, m: ["completion"])
    monkeypatch.setattr(mtp_check.screen, "model_identity", lambda u, m: {"digest": "d"})
    mtp_check.main(["--tasks", str(tmp_path / "t.json"), "--reference", str(tmp_path / "r.jsonl"),
                    "--model", "m", "--out", str(tmp_path / "o.json")])
    res = json.loads((tmp_path / "o.json").read_text())
    assert res["speedup"] == 2.0 and res["use_mtp"] and res["same_text"] == 1
    assert '"use_mtp": true' in (tmp_path / "o.json").read_text()  # run.sh greps this exact form
