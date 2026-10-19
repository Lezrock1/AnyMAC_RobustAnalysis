import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit
import confirm
import control


def episode(**overrides):
    return {"task_id": "t", "marker": "blue", "policy": "honest", "truth": 3, "target": 4,
            "pred": 3, "correct": True, "final_text": "The answer is 3",
            "events": [{"is_decision": True, "usage": {}, "extra_calls": []}], "usage": {},
            **overrides}


def test_strict_scoring_preserves_truncation():
    assert audit.score(episode())["strict_correct"]
    r = episode(final_text="Working: 7 + 3", usage={"truncated": 1})
    s = audit.score(r)
    assert s["strict_pred"] is None and not s["strict_correct"] and s["any_call_truncated"]
    assert audit.score(episode(final_text="The answer is 23"))["strict_pred"] is None
    assert audit.score(episode(final_text="The answer is 4."))["strict_target_success"]


def test_audit_keeps_source_and_flags_integrity(tmp_path):
    pilot = tmp_path / "pilot"
    path = pilot / "qwen30/seed0/eval/exp1_A.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(episode()) + "\n" + json.dumps(episode()) + "\nBROKEN\n")
    before = path.read_bytes()
    assert not audit.audit(pilot, tmp_path / "audit")
    s = json.loads((tmp_path / "audit/summary.json").read_text())["files"]
    counts = s["qwen30/seed0/eval/exp1_A.jsonl"]["counts"]
    assert counts["duplicate_keys"] == 1 and counts["invalid_lines"] == 1
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        audit.audit(pilot, pilot / "bad")


def test_snapshot_includes_final_checkpoints_and_verifies(tmp_path):
    repo = tmp_path / "repo"
    seed = repo / "results/pilot2/qwen30/seed0"
    for stage in confirm.CHECKPOINTS:
        p = seed / stage
        p.mkdir(parents=True)
        (p / "progress.json").write_text('{"rounds_done": 12}')
        (p / "round_0012.pth").write_bytes(b"FINAL")
        (p / "round_0001.pth").write_bytes(b"INTERMEDIATE")
        (p / "round_0012.pth.opt").write_bytes(b"OPTIMIZER")
    (repo / "results/stage0").mkdir()
    splits = repo / "data/splits/splits_pilot2_seed0.json"
    splits.parent.mkdir(parents=True)
    splits.write_text("{}")
    archive = audit.snapshot(repo, tmp_path / "snapshot")
    with tarfile.open(archive) as tar:
        assert sum(m.name.endswith(".pth") for m in tar) == 6
        assert not any(m.name.endswith(".opt") for m in tar.getmembers())
    assert (tmp_path / "snapshot/SHA256SUMS").read_text().startswith(hashlib.sha256(archive.read_bytes()).hexdigest())


def test_resume_rejects_mutation_duplicate_partial_and_unexpected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    confirm.bind_manifest(out, {"protocol": 1})
    confirm.bind_manifest(out, {"protocol": 1})
    with pytest.raises(ValueError):
        confirm.bind_manifest(out, {"protocol": 2})
    path = out / "episodes.jsonl"
    expected = {("t", "blue", "honest")}
    path.write_text(json.dumps(episode()) + "\n")
    assert len(confirm.checked_rows(path, expected)) == 1
    with pytest.raises(ValueError):
        confirm.checked_rows(path, set())
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError):
        confirm.checked_rows(path, expected)
    path.write_text('{"task_id":')
    with pytest.raises(ValueError):
        confirm.checked_rows(path, expected)


def test_quality_gate_and_chat_payload():
    q = confirm.quality([episode()], 0.05)
    assert confirm.passed(q)
    assert not confirm.passed(confirm.quality([episode(error="timeout")], 0.05))
    assert not confirm.passed(confirm.quality([episode(final_text="x = 3")], 0.05))
    assert not confirm.passed(confirm.quality([episode(usage={"truncated": 1})], 0.05))
    worker_cut = episode(usage={"truncated": 1}, events=[
        {"is_decision": False, "usage": {"truncated": 1}},
        {"is_decision": True, "usage": {"truncated": 0}, "extra_calls": []}])
    q = confirm.quality([worker_cut], 0.05, "decision")
    assert confirm.passed(q) and q["truncated_episode_rate"] == 1 and q["decision_truncated_rate"] == 0
    assert not confirm.passed(confirm.quality([worker_cut], 0.05, "any"))
    final_cut = episode(usage={"truncated": 1}, events=[{"is_decision": True, "usage": {"truncated": 1}, "extra_calls": []}])
    assert not confirm.passed(confirm.quality([final_cut], 0.05, "decision"))
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    p = confirm.payload("tag", messages, 4096, 16384)
    assert p["messages"] == messages and "prompt" not in p and p["options"]["num_predict"] == 4096


def test_chat_transport_with_local_http_server(monkeypatch):
    from aiohttp import web, ClientResponseError
    from GDesigner.llm import sllm_chat, usage
    from robust import llm
    monkeypatch.setattr(sllm_chat, "achat", sllm_chat.achat)

    async def exercise():
        received = []
        mode = "ok"

        async def handle(request):
            received.append(await request.json())
            if mode == "http_error":
                return web.Response(status=503, text="temporary server failure")
            if mode == "malformed":
                return web.json_response({"done": False})
            return web.json_response({"done": True, "message": {"content": "The answer is 3"},
                                      "done_reason": "stop", "prompt_eval_count": 100 if mode == "ok" else 16330,
                                      "eval_count": 10})

        app = web.Application()
        app.router.add_post("/api/chat", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(llm, "base_url", lambda: f"http://127.0.0.1:{port}")
        confirm.install_chat(4096, 16384)
        try:
            usage.start_recording()
            assert await sllm_chat.achat("tag", [{"role": "user", "content": "test"}]) == "The answer is 3"
            assert usage.since(0)["calls"] == 1 and usage.since(0)["completion_tokens"] == 10
            assert received[0]["messages"] == [{"role": "user", "content": "test"}]
            mode = "http_error"
            with pytest.raises(ClientResponseError):
                await sllm_chat.achat("tag", [])
            mode = "malformed"
            with pytest.raises(ValueError, match="Malformed"):
                await sllm_chat.achat("tag", [])
            mode = "context"
            with pytest.raises(ValueError, match="context capacity"):
                await sllm_chat.achat("tag", [])
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_learning_objectives_and_mapping():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(5, requires_grad=True)
    loss = control.objective(logits, 2, "exact")
    loss.backward()
    assert logits.grad[2] < 0 and logits.grad[0] > 0
    assert control.wanted("conditional", "blue", "ctrl", "ver") == "ctrl"
    assert control.wanted("conditional", "green", "ctrl", "ver") == "ver"
    assert control.wanted("unconditional", "green", "ctrl", "ver") == "ctrl"


def test_confirmation_end_to_end_resume_and_dev_gate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from robust import llm, rollout
    from robust.data import Task
    from robust import data
    from robust.profile import load_profile
    profile = load_profile("pilot2")
    tag = profile["models"]["qwen30"]["ollama_tag"]
    source = tmp_path / "source"
    (source / "stages").mkdir(parents=True)
    for stage in ("branch_B", "continue_B"):
        (source / "stages" / f"{stage}.done").touch()
        (source / stage).mkdir()
        (source / stage / "progress.json").write_text('{"rounds_done": 1}')
        (source / stage / "round_0001.pth").write_bytes(b"FROZEN")
    splits = tmp_path / "splits.json"
    splits.write_text("{}")
    tasks = {"dev": [Task(f"dev{i}", "test", 3, 4, 2) for i in range(6)],
             "eval": [Task("t", "test", 3, 4, 2)]}
    monkeypatch.setattr(data, "get_tasks", lambda sp, split: tasks[split])
    monkeypatch.setattr(llm, "model_info", lambda tag: {"digest": "frozen", "tag": tag})
    monkeypatch.setattr(rollout, "load_graph", lambda path: SimpleNamespace(
        llm_name=tag, available_roles=profile["roles"], max_routing=profile["max_routing"]))
    monkeypatch.setattr(confirm, "install_chat", lambda limit, context: None)
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
    calls = []

    async def batch(graph, jobs, profile, cos, concurrency, on_result):
        calls.extend(jobs)
        for job in jobs:
            row = episode(**job["tag"], task_id=job["task"].id)
            on_result(row)

    monkeypatch.setattr(rollout, "run_batch", batch)
    args = argparse.Namespace(source=source, splits=splits, out=tmp_path / "confirmation",
                              checkpoints=["branch_B", "continue_B"], tasks=0, reps=2, limit=4096,
                              context=16384, concurrency=2, hours=1, max_quality_rate=0.05,
                              truncation_gate="any")
    confirm.run(args)
    assert len(calls) == 24 + 16
    assert json.loads((args.out / "COMPLETE.json").read_text())["episodes"] == 16
    for path in (args.out / "eval").glob("*.jsonl"):
        rows, _ = audit.load_rows(path)
        assert all(r["strict_correct"] for r in rows)
    confirm.run(args)
    assert len(calls) == 40  # no new episodes on resume
    (source / "branch_B/round_0001.pth").write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="changed"):
        confirm.run(args)
    args.out = tmp_path / "gate_failure"

    async def broken_batch(graph, jobs, profile, cos, concurrency, on_result):
        for job in jobs:
            on_result(episode(**job["tag"], task_id=job["task"].id, final_text="cut off"))

    monkeypatch.setattr(rollout, "run_batch", broken_batch)
    with pytest.raises(ValueError, match="quality gate FAILED"):
        confirm.run(args)
    assert not list((args.out / "eval").glob("*.jsonl"))


def test_first_scores_match_real_corrected_graph():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    sys.path.insert(0, str(audit.ROOT / "scripts/stage0"))
    import sim
    from robust.data import format_task
    from robust.rollout import build_graph, run_batch, run_sync
    from robust.profile import load_profile
    from robust.control import Controller
    from robust.train import EVAL_COS
    torch.set_num_threads(1)
    profile = load_profile("pilot2")
    profile["max_routing"] = 1
    tasks = sim.synthetic_tasks(1, "score_test", 0)
    sim.WORLD.register(tasks)
    graph = build_graph(profile, "mock", seed=0)
    text = format_task(tasks[0], "blue")
    raw = torch.tensor(sim.graph_mod.get_sentence_embedding(text), device=graph.device)
    try:
        sim.apply_variant("indices")
        with torch.no_grad():
            expected = control.first_scores(graph, raw) * EVAL_COS
        result = run_sync(run_batch(graph, [{"task": tasks[0], "text": text,
                                             "controller": Controller(), "tag": {"marker": "blue"}}],
                                    profile, EVAL_COS, 1, keep_raw=True))
        # Actual source logits are captured before Gumbel/argmax.
        actual = result[0]["_raw"]["agent_logits"][0]
        assert torch.allclose(expected.cpu(), torch.tensor(actual), atol=1e-3)
    finally:
        sim.apply_variant("original")
import argparse
import asyncio
