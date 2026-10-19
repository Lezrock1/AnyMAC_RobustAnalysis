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


def test_mc_rows_keep_official_letters():
    pro = pd.DataFrame([{"question_id": 1, "category": "law", "question": "q", "options": ["a", "b", "N/A"], "answer": "B"},
                        {"question_id": 2, "category": "law", "question": "q", "options": ["a", "b", "c"], "answer": "C"}])
    rows = screen.mc_rows("mmlu_pro", pro)
    assert [r["id"] for r in rows] == ["mmlu_pro/2"] and rows[0]["truth"] == "C" and "C. c" in rows[0]["text"]
    mmlu = pd.DataFrame([{"subject": "x", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 3}])
    assert screen.mc_rows("mmlu", mmlu)[0]["truth"] == "D"


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
