"""SYNTHETIC gradient-path test (NOT the AnyMAC training smoke).

Real AnyMAC rollouts under no_grad with a *stubbed* LLM (canned replies) and a *randomised* reward, so that
reward variation exists. Then the log-probabilities of the actions that were actually executed are recomputed with
gradients (Graph.run_next_agent_prediction_grad) and we check NAP / NCS gradients, parameter changes and
that no LLM parameters exist. Needs the MiniLM embedding model (HF cache); skipped if unavailable.
"""
import copy
import random

import numpy as np
import pytest
import torch

pytest.importorskip("sentence_transformers")

from robust.data import Task, alternate_markers, format_task
from robust.profile import load_profile
from robust.rollout import build_graph, run_batch, run_sync
from robust.train import TRAIN_COS, Trainer, compute_grad_inputs, diff_snapshots, snapshot, named_router_params


@pytest.fixture(scope="module")
def setup():
    import GDesigner.llm.sllm_chat as sc

    async def fake_achat(model, msgs, *a, **k):
        h = abs(hash(str(msgs))) % 23
        return f"Step-by-step reasoning. The answer is {h}"
    orig = sc.achat
    sc.achat = fake_achat
    prof = load_profile("smoke")
    try:
        try:
            graph = build_graph(prof, "stub-model", seed=0)
        except Exception as e:  # embedding model not cached / offline
            pytest.skip(f"router graph unavailable: {e}")
        tasks = [Task(f"s{i}", f"Compute {i} plus {i + 3} modulo 23.", (2 * i + 3) % 23, (2 * i + 4) % 23, 2) for i in range(6)]
        marks = alternate_markers(tasks)
        jobs = [{"task": t, "text": format_task(t, marks[t.id]), "controller": None, "tag": {}} for t in tasks for _ in range(4)]
        assert torch.is_grad_enabled()
        eps = run_sync(run_batch(graph, jobs, prof, TRAIN_COS, 4, keep_raw=True))
        assert torch.is_grad_enabled(), "grad mode must be restored after parallel rollouts"
        yield prof, graph, tasks, marks, eps
    finally:
        sc.achat = orig


def _grads(graph):
    return {n: p.grad.detach().clone() for n, p in named_router_params(graph).items() if p.grad is not None}


def test_nap_and_ncs_gradients_and_param_update(setup):
    prof, graph, tasks, marks, eps = setup
    rng = random.Random(0)
    for e in eps:
        e["correct"] = rng.random() < 0.5  # SYNTHETIC reward variation
    gin, rewards, nz = compute_grad_inputs(eps, {t.id: t for t in tasks}, marks, prof, 4)
    assert gin, "no usable traces"
    long = [g for g in gin if g["trace_length"] >= 3]
    nap_traces = [g for g in gin if g["trace_length"] >= 2]  # length-1 traces (decision first) carry no gradient
    assert nap_traces
    assert long, "need a trace with >=2 worker steps to exercise NCS"

    tr = Trainer(graph, 1e-3, None)
    # NAP: any trace
    graph.cos_scaling = TRAIN_COS
    graph.set_train()
    tr.optimizer.zero_grad()
    graph.run_next_agent_prediction_grad(nap_traces[0])
    g_nap = _grads(graph)
    assert g_nap and all(torch.isfinite(g).all() for g in g_nap.values())
    assert sum(g.norm() for g in g_nap.values()) > 0
    nap_mod = [n for n in g_nap if "proj_nap." in n]
    assert nap_mod and any(g_nap[n].abs().sum() > 0 for n in nap_mod), "NAP projection gets no gradient"

    # NCS: flipping a NON-final hint mask must change the gradient (final mask is the documented exception)
    tr.optimizer.zero_grad()
    graph.run_next_agent_prediction_grad(long[0])
    g_a = _grads(graph)
    flipped = copy.deepcopy(long[0])
    flipped["hint_selections"][0] = [1 - x for x in flipped["hint_selections"][0]]
    tr.optimizer.zero_grad()
    graph.run_next_agent_prediction_grad(flipped)
    g_b = _grads(graph)
    assert all(torch.isfinite(g).all() for g in g_b.values())
    diff = sum((g_a[n] - g_b[n]).abs().sum() for n in g_a)
    assert diff > 0, "NCS (hint) selection receives no gradient at non-final steps"
    nhp = [n for n in g_a if "proj_nhp." in n]
    assert nhp and any(g_a[n].abs().sum() > 0 for n in nhp), "NCS projection gets no gradient"

    # Final (decision-step) hint mask: documented upstream behaviour, recorded not asserted
    final_flipped = copy.deepcopy(long[0])
    final_flipped["hint_selections"][-1] = [1 - x for x in final_flipped["hint_selections"][-1]]
    tr.optimizer.zero_grad()
    graph.run_next_agent_prediction_grad(final_flipped)
    g_c = _grads(graph)
    print("decision-step mask receives gradient:", bool(sum((g_a[n] - g_c[n]).abs().sum() for n in g_a) > 0))

    # real optimizer update
    pre = snapshot(graph)
    st = tr.step(gin, list(range(min(8, len(gin)))))
    d = diff_snapshots(pre, snapshot(graph))
    assert st["grad_norm"] > 0 and np.isfinite(st["grad_norm"])
    assert d["tensors_changed"] > 0 and d["l2_delta"] > 0
    assert not [n for n, _ in named_router_params(graph).items() if "llm" in n.lower()]
    assert torch.is_grad_enabled()
