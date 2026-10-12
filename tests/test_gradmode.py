"""SYNTHETIC test of grad-mode handling in parallel rollouts (fake graph, no LLM, no router)."""
import asyncio

import pytest
import torch

from robust.control import StopRollout
from robust.data import Task
from robust.rollout import run_batch, run_sync

PROFILE = {"max_routing": 2, "max_context": 2, "episode_timeout_s": 0.3, "controlled_role": "C", "verifier_role": "V"}
RES = {"answers": ["answer is 1"], "routing_count": 1, "trace_events": [],
       "routing_results": {"agent_selections": [], "hint_selections": []}}


class FakeGraph:
    cos_scaling = 0.0

    def __init__(self):
        self.seen_grad = []

    async def arun_next_agent_prediction(self, inputs, **kw):
        await asyncio.sleep(0.01 * (hash(inputs["task"]) % 5))   # interleave episodes
        self.seen_grad.append(torch.is_grad_enabled())
        mode = inputs["task"]
        if mode == "raise":
            raise RuntimeError("boom")
        if mode == "hang":
            await asyncio.sleep(5)
        if mode == "stop":
            raise StopRollout()
        await asyncio.sleep(0.02)
        self.seen_grad.append(torch.is_grad_enabled())
        return dict(RES)


def jobs(modes):
    return [{"task": Task(f"t{i}", "p", 1, 2, 2), "text": m, "controller": None, "tag": {}} for i, m in enumerate(modes)]


@pytest.mark.parametrize("modes", [["ok"] * 8, ["ok", "raise", "ok", "hang", "stop", "ok", "raise", "ok"]])
def test_grad_mode_restored_and_disabled_inside(modes):
    assert torch.is_grad_enabled()
    g = FakeGraph()
    out = run_sync(run_batch(g, jobs(modes), PROFILE, 1.5, 4))
    assert torch.is_grad_enabled(), "grad mode must be restored after the batch (incl. error/timeout/stop paths)"
    assert g.seen_grad and not any(g.seen_grad), "rollouts run entirely under no_grad"
    errs = [o["error"] for o in out]
    if "hang" in modes:
        assert "timeout" in errs and "stop" in errs and any(e and e.startswith("RuntimeError") for e in errs)


def test_batch_level_exception_restores_grad_mode():
    async def bad(*a, **k):
        raise RuntimeError('outside episode handler')
    import robust.rollout as ro
    orig = ro.run_episode
    ro.run_episode = bad
    try:
        with pytest.raises(RuntimeError):
            run_sync(run_batch(FakeGraph(), jobs(["ok"] * 3), PROFILE, 1.5, 2))
    finally:
        ro.run_episode = orig
    assert torch.is_grad_enabled()


def test_previously_disabled_state_is_preserved():
    with torch.no_grad():
        run_sync(run_batch(FakeGraph(), jobs(["ok"] * 2), PROFILE, 1.5, 2))
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_interrupt_restores_grad_mode():
    async def bad(*a, **k):
        raise KeyboardInterrupt()
    import robust.rollout as ro
    orig = ro.run_episode
    ro.run_episode = bad
    try:
        with pytest.raises(BaseException):
            run_sync(run_batch(FakeGraph(), jobs(["ok"] * 3), PROFILE, 1.5, 2))
    finally:
        ro.run_episode = orig
    assert torch.is_grad_enabled()
