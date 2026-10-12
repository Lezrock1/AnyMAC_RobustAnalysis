"""Real end-to-end smoke test (real agent model, real rollouts, real router update). Nothing here is mocked."""
import copy
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from robust import llm
from robust.common import get_logger, results_dir, set_seed, write_json
from robust.data import simple_smoke_tasks, alternate_markers, format_task, get_tasks, load_or_create_splits
from robust.pipeline import Ctx
from robust.rollout import build_graph, load_graph, run_batch, run_sync
from robust.train import (EVAL_COS, TRAIN_COS, Trainer, compute_grad_inputs, diff_snapshots, fingerprint, named_router_params,
                          router_modules, snapshot, unoptimized_router_params, strip_raw)

log = get_logger()


def gpu_info() -> Dict:
    info = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(device=torch.cuda.get_device_name(0), torch_max_mem_allocated_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["nvidia_smi"] = out or None
    except Exception:
        info["nvidia_smi"] = None
    return info


def _grad_vec(graph) -> torch.Tensor:
    return torch.cat([p.grad.flatten().detach().cpu() if p.grad is not None else torch.zeros(p.numel()) for p in named_router_params(graph).values()])


def run_smoke(ctx: Ctx, require_cuda: bool) -> Dict:
    prof = ctx.profile
    out: Dict = {"profile": prof["name"], "model_tag": ctx.tag, "checks": {}}
    chk = out["checks"]
    t0 = time.time()
    chk["cuda_required"] = require_cuda
    chk["cuda_available"] = torch.cuda.is_available()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    if not llm.wait_ready(60):
        raise RuntimeError("LLM server not reachable")
    before = llm.model_info(ctx.tag)
    if before is None:
        raise RuntimeError(f"model {ctx.tag} not available (run prepare.sh)")
    out["model_before"] = before
    out["generation"] = llm.generate_once(ctx.tag)
    log.info("generation check: %s", out["generation"])

    tasks_dev = simple_smoke_tasks(4)  # separate simple tasks, outside the scientific splits
    graph = build_graph(prof, ctx.tag, seed=ctx.seed)
    jobs = [{"task": t, "text": format_task(t, "blue"), "controller": None, "tag": {"phase": "smoke_rollout"}} for t in tasks_dev]
    rolls = run_sync(run_batch(graph, jobs, prof, TRAIN_COS, prof["concurrency"]))
    rolls_ser = [strip_raw(r) for r in rolls]
    write_json(ctx.root / "smoke" / "rollouts.json", rolls_ser)
    chk["rollouts"] = {"n": len(rolls), "errors": [r["error"] for r in rolls if r["error"]], "accuracy": float(np.mean([r["correct"] for r in rolls])),
                       "truncated_llm_calls": sum(r["usage"]["truncated"] for r in rolls),
                       "selected_roles": [r["selected_roles"] for r in rolls]}
    # NCS: decision agent receives exactly the messages selected by the delivered mask
    ncs_ok, n_dec = True, 0
    for r in rolls:
        dec = r["events"][-1] if r["events"] and r["events"][-1]["is_decision"] else None
        if not dec:
            continue
        n_dec += 1
        workers = [e for e in r["events"] if not e["is_decision"]]
        delivered_out = {w["output"] for i, w in enumerate(workers) if i < len(dec["delivered_mask"]) and dec["delivered_mask"][i] == 1}
        for i, e in enumerate(workers):
            sel = i < len(dec["delivered_mask"]) and dec["delivered_mask"][i] == 1
            if sel:
                ncs_ok &= e["output"] in dec["decision_context"]
            elif e["output"] not in delivered_out:  # identical texts could be delivered via another index
                ncs_ok &= e["output"] not in dec["decision_context"]
    chk["ncs_controls_decision_context"] = {"episodes_checked": n_dec, "consistent": bool(ncs_ok)}

    # real training step
    tcfg = prof["training"]
    trainer = Trainer(graph, tcfg["lr"], tcfg.get("grad_clip"))
    chk["router_params_all_in_optimizer"] = not unoptimized_router_params(graph, trainer.optimizer)
    llm_like = [n for n, _ in named_router_params(graph).items() if "llm" in n.lower()]
    chk["no_llm_parameters_in_optimizer"] = not llm_like
    pool = simple_smoke_tasks(prof["smoke"]["pool_size"], seed=999)
    marks = alternate_markers(pool)
    by_id = {t.id: t for t in pool}
    gin, diag, tries = [], [], 0
    while not gin and tries * 4 < len(pool):
        sel = pool[tries * 4:(tries + 1) * 4]
        jobs = [{"task": t, "text": format_task(t, marks[t.id]), "controller": None,
                 "tag": {"phase": "smoke_train"}} for t in sel for _ in range(tcfg["traces_per_task"])]
        eps = run_sync(run_batch(graph, jobs, prof, TRAIN_COS, prof["concurrency"], keep_raw=True))
        gin, rewards, nz = compute_grad_inputs(eps, by_id, marks, prof, tcfg["traces_per_task"])
        diag.append({"try": tries, "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
                     "tasks_without_advantage": nz, "n_grad_traces": len(gin)})
        tries += 1
    out["reward_advantage_diagnostics"] = diag
    if not gin:
        chk["training_step"] = {"executed": False, "reason": "no reward/advantage variation in any sampled group (see reward_advantage_diagnostics)"}
    else:
        pre = snapshot(graph)
        trainer.graph.set_train()
        trainer.optimizer.zero_grad()
        trainer.graph.run_next_agent_prediction_grad(gin[0])
        g_orig = _grad_vec(graph)
        # decision-step selection gets no gradient (documented upstream behaviour): flip final hint mask and compare
        gi2 = copy.deepcopy(gin[0])
        flipped = False
        if gi2["hint_selections"]:
            gi2["hint_selections"][-1] = [1 - x for x in gi2["hint_selections"][-1]]
            flipped = True
        trainer.optimizer.zero_grad()
        trainer.graph.run_next_agent_prediction_grad(gi2)
        g_flip = _grad_vec(graph)
        trainer.optimizer.zero_grad()
        st = trainer.step(gin, list(range(min(tcfg["batch_size"], len(gin)))))
        post = snapshot(graph)
        chk["training_step"] = {"executed": True, **st, "diff": diff_snapshots(pre, post), "grad_norm_nonzero": st["grad_norm"] > 0,
                                "params_changed": diff_snapshots(pre, post)["tensors_changed"] > 0}
        chk["decision_step_gradient"] = {"final_hint_mask_flipped": flipped,
                                         "gradient_identical_after_flip": bool(torch.allclose(g_orig, g_flip, atol=1e-9)),
                                         "meaning": "gradient_identical_after_flip=True would mean the last recorded hint mask receives no training signal (docs/AUDIT.md); measured per run, not assumed"}
    # checkpoint round trip
    ck = ctx.root / "smoke" / "ckpt.pth"
    trainer.save(ck)
    fp_saved = fingerprint(trainer.graph)
    g2 = load_graph(str(ck))
    t = tasks_dev[0]
    set_seed(0)
    r1 = run_sync(run_batch(trainer.graph, [{"task": t, "text": t.problem, "controller": None, "tag": {}}], prof, EVAL_COS, 1))[0]
    r2 = run_sync(run_batch(g2, [{"task": t, "text": t.problem, "controller": None, "tag": {}}], prof, EVAL_COS, 1))[0]
    chk["checkpoint_roundtrip"] = {"fingerprint_equal": fp_saved == fingerprint(g2), "routing_identical": r1["selected_roles"] == r2["selected_roles"]}
    after = llm.model_info(ctx.tag)
    chk["llm_weights_unchanged"] = {"digest_before": before["digest"], "digest_after": after["digest"] if after else None,
                                    "unchanged": bool(after) and after["digest"] == before["digest"],
                                    "note": "the agent LLM runs in the Ollama server; the trainer only owns router parameters"}
    out["gpu"] = gpu_info()
    out["wall_s"] = time.time() - t0
    must = [chk["rollouts"]["n"] > 0, chk["ncs_controls_decision_context"]["consistent"], chk["router_params_all_in_optimizer"],
            chk["no_llm_parameters_in_optimizer"], chk["checkpoint_roundtrip"]["fingerprint_equal"], chk["llm_weights_unchanged"]["unchanged"],
            chk["training_step"].get("executed") and chk["training_step"].get("params_changed") and chk["training_step"].get("grad_norm_nonzero")]
    out["passed"] = bool(all(must))
    write_json(ctx.root / "smoke" / "smoke_result.json", out)
    return out
