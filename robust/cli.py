import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from robust import llm
from robust.common import REPO_ROOT, get_logger, read_json, results_dir, write_json
from robust.profile import load_profile

log = get_logger()


def git_info():
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def cmd_profile_get(a):
    v = load_profile(a.profile)
    for k in a.key.split("."):
        v = v[k]
    print(json.dumps(v) if isinstance(v, (dict, list)) else v)


def cmd_prepare(a):
    from robust import prepare
    p = load_profile(a.profile)
    out = results_dir() / p["name"] / "manifest.json"
    man = prepare.run(p, out, with_models=not a.no_models, models=p["models_to_run"] if a.all_models else ["qwen"], with_embedding=not a.no_embedding)
    man["git"] = git_info()
    write_json(out, man)
    print(json.dumps(man, indent=2)[:3000])


def cmd_run(a):
    from robust import pipeline, exp2, exp4
    from robust.budget import Budget
    p = load_profile(a.profile)
    base = results_dir() / p["name"]
    if base.exists() and any(base.glob("*/seed*")) and not a.resume:
        sys.exit(f"{base} already contains results; pass --resume to continue (or choose another RESULTS_DIR)")
    base.mkdir(parents=True, exist_ok=True)
    get_logger(logfile=base / "run.log")
    exps = ["1", "3", "2", "4"] if a.experiment == "all" else [a.experiment]
    models = [m for m in p["models_to_run"] if a.model in (None, m)]
    if not llm.wait_ready(120):
        sys.exit("LLM server not reachable (BASE_URL); use scripts/ramses/run.sh")
    integrity = {}
    first_ctx = {}
    for mk in models:
        tag = p["models"][mk]["ollama_tag"]
        info = llm.model_info(tag)
        if info is None:
            sys.exit(f"model {tag} missing; run scripts/ramses/prepare.sh")
        integrity[mk] = {"digest_before": info["digest"], "tag": tag}
    for mk in models:
        for seed in p["train_seeds"]:
            ctx = pipeline.Ctx(p, mk, seed)
            log.info("== model %s seed %s root %s", mk, seed, ctx.root)
            pipeline.stage_llm_limit(ctx)
            pipeline.stage_calibrate(ctx)
            for e in exps:
                if e == "1":
                    ok = pipeline.experiment1(ctx)
                elif e == "2":
                    ok = exp2.experiment2(ctx)
                elif e == "3":
                    ok = pipeline.experiment3(ctx)
                    if ok:
                        pipeline.experiment3_extra(ctx)
                else:
                    ok = exp4.experiment4(ctx)
                if not ok:
                    log.warning("stopped at experiment %s (budget exhausted?) - rerun with --resume", e)
                    break
            first_ctx[(mk, seed)] = ctx
    if "3" in exps or a.experiment == "all":
        src = p["models_to_run"][0]
        for mk in models:
            if mk == src:
                continue
            for seed in p["train_seeds"]:
                s = pipeline.Ctx(p, src, seed)
                if (s.root / "stages" / "continue_C.done").exists():
                    pipeline.experiment3_transfer(pipeline.Ctx(p, mk, seed), s)
    for mk in models:
        after = llm.model_info(integrity[mk]["tag"])
        integrity[mk]["digest_after"] = after["digest"] if after else None
        integrity[mk]["unchanged"] = bool(after) and after["digest"] == integrity[mk]["digest_before"]
    write_json(base / "llm_integrity.json", integrity)
    write_json(base / "run_info.json", {"git": git_info(), "argv": sys.argv, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    from robust.report import build_report
    build_report(p["name"])


def cmd_smoke(a):
    from robust import pipeline, smoke
    p = load_profile("smoke")
    base = results_dir() / "smoke"
    get_logger(logfile=base / "run.log")
    ctx = pipeline.Ctx(p, "qwen", 0)
    with ctx.budget.stage("smoke_checks"):
        res = smoke.run_smoke(ctx, require_cuda=a.require_cuda)
    print(json.dumps(res["checks"], indent=2, default=str))
    print("SMOKE_CHECKS_PASSED" if res["passed"] else "SMOKE_CHECKS_FAILED")
    if not res["passed"]:
        sys.exit(2)


def cmd_report(a):
    from robust.report import build_report
    build_report(a.profile)
    print(results_dir() / a.profile / "report.md")


def main():
    ap = argparse.ArgumentParser(prog="python -m robust")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("profile-get"); s.add_argument("profile"); s.add_argument("key"); s.set_defaults(f=cmd_profile_get)
    s = sub.add_parser("prepare"); s.add_argument("--profile", required=True); s.add_argument("--no-models", action="store_true")
    s.add_argument("--no-embedding", action="store_true"); s.add_argument("--all-models", action="store_true"); s.set_defaults(f=cmd_prepare)
    s = sub.add_parser("run"); s.add_argument("--profile", required=True); s.add_argument("--experiment", default="all", choices=["1", "2", "3", "4", "all"])
    s.add_argument("--resume", action="store_true"); s.add_argument("--model", default=None); s.set_defaults(f=cmd_run)
    s = sub.add_parser("smoke"); s.add_argument("--require-cuda", action="store_true"); s.set_defaults(f=cmd_smoke)
    s = sub.add_parser("report"); s.add_argument("--profile", required=True); s.set_defaults(f=cmd_report)
    a = ap.parse_args()
    a.f(a)
