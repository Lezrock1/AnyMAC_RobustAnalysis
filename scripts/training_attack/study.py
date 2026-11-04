"""One allocation, runtime-only budget release, six matched short-training seed pairs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from common import (CONDITIONS, MODEL_DIGEST, RESERVE_S, ROLE, SCHEMA, SEEDS, balanced,
                    check_disjoint, choose_budget, digest, load_tasks, sources, write_json)
from learned import tree_digest, worker_identity
from llm import OllamaChat
from report import report
from train_learned import DATA_REVISION
from bank_recovery import PRIOR_ALLOCATION_SECONDS, application_deadline, recover

HERE = Path(__file__).resolve().parent


def batch(commands, deadline, root, stage):
    processes, streams = [], []
    started = time.time()
    record = {"stage": stage, "started": started, "deadline": deadline, "commands": commands}
    write_json(root / "active_stage.json", record)
    if deadline <= started:
        record.update(finished=started, timed_out=True, returncodes=[])
        write_json(root / "logs" / f"{stage}.json", record)
        print(f"{stage}: deadline already expired; no process launched", file=sys.stderr, flush=True)
        return False
    try:
        for i, command in enumerate(commands):
            stream = (root / "logs" / f"{stage}-{i}.log").open("w")
            streams.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        timed_out = False
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            time.sleep(min(.2, remaining))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()
    record.update(finished=time.time(), timed_out=timed_out,
                  returncodes=[p.returncode for p in processes])
    write_json(root / "logs" / f"{stage}.json", record)
    ok = not timed_out and all(p.returncode == 0 for p in processes)
    print(f"{stage}: {'complete' if ok else 'FAILED'}; "
          f"codes={record['returncodes']}; timeout={timed_out}", flush=True)
    if not ok:
        for i, process in enumerate(processes):
            if process.returncode:
                path = root / "logs" / f"{stage}-{i}.log"
                print(f"--- {path} (last 20 lines) ---\n"
                      + "\n".join(path.read_text().splitlines()[-20:]), file=sys.stderr, flush=True)
    return ok


def worker(command, plan_path, seed, condition, url, out):
    return [sys.executable, str(HERE / "worker.py"), command, "--plan", str(plan_path),
            "--seed", str(seed), "--condition", condition, "--url", url, "--out", str(out)]


def prepare(args, root):
    results = Path(args.results).resolve()
    base = results / "compare"
    training_path = base / "learned_gemma12b_paper_v2/train_tasks.json"
    original_dev = base / "gemma12b_preflight_v1/dev_tasks.json"
    original_eval = base / "main_balanced200_v1/eval_tasks.json"
    assets_path = base / "learned_gemma12b_v1/assets.json"
    reserved = [results / "screen/screen_tasks_v1.json", results / "screen/screen_tasks_r2.json",
                base / "dylan_study_v1/eval_tasks.json", base / "main_reserved_v2/eval_tasks.json"]
    all_paths = [training_path, original_dev, original_eval, assets_path, *reserved]
    training = load_tasks(training_path)
    dev_pool, eval_pool = load_tasks(original_dev), load_tasks(original_eval)
    blocked = [t for path in reserved for t in load_tasks(path)]
    check_disjoint(training, dev_pool, eval_pool, blocked)
    training_doc = json.loads(training_path.read_text())
    if (training_doc.get("option_preprocessing") != "trim-trailing-na/preserve-letters/v1"
            or training_doc.get("source", {}).get("split") != "validation"
            or training_doc.get("source", {}).get("revision") != DATA_REVISION
            or training_doc.get("selection") != "all-eligible-validation"
            or any(not t["id"].startswith("mmlu_pro/validation/") for t in training)):
        raise ValueError("need the frozen paper-v2 official-validation training pool")
    assets = json.loads(assets_path.read_text())
    for name in ("embedding", "backbone"):
        if tree_digest(assets[name]["path"]) != assets[name]["sha256"]:
            raise ValueError(f"pinned {name} assets changed")
    # Selection depends on task IDs/subjects, never predictions or correctness.
    eval_tasks = balanced(eval_pool, 24, 20261103)
    dev = sorted(dev_pool, key=lambda t: digest_text(t["id"]))[:4]
    if len(dev) != 4:
        raise ValueError("need at least four disjoint development questions")
    dev_path, eval_path = root / "dev_tasks.json", root / "eval_tasks.json"
    write_json(dev_path, {"schema": 1, "tasks": dev})
    write_json(eval_path, {"schema": 1, "tasks": eval_tasks})
    identities = [worker_identity(OllamaChat(url, args.model, 8192, 32768, args.threads).identity())
                  for url in args.urls]
    if identities[0] != identities[1] or identities[0]["digest"] != MODEL_DIGEST:
        raise ValueError("both allocated servers must use the same screened worker model/decoding/version")
    all_paths += [dev_path, eval_path]
    plan = {"schema": SCHEMA, "source": sources(), "out": str(root),
            "conditions": list(CONDITIONS), "seeds": list(SEEDS), "role": ROLE,
            "model": args.model, "worker": identities[0], "threads": args.threads,
            "assets": assets, "training_tasks": str(training_path), "dev_tasks": str(dev_path),
            "eval_tasks": str(eval_path), "reserved": [str(p) for p in reserved] + [str(original_eval)],
            "bank_reserved": [str(p) for p in reserved] + [str(original_dev), str(original_eval)],
            "bank": str(root / "messages.json"),
            "noise_ids": sorted(t["id"] for t in eval_tasks)[:4],
            "inputs": {str(p): digest(p) for p in all_paths},
            "deadline": args.deadline, "started": args.started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "assignment": "training arms swap GPUs by seed parity; both evaluated together on one GPU",
            "evaluation": "24 frozen main questions, honest/bare adjacent, 4 identical-router-RNG honest repeats",
            "performance_tolerance": 0.10}
    plan["allocation_accounting"] = {
        "current_seconds": args.allocation_seconds,
        "prior_charged_seconds": PRIOR_ALLOCATION_SECONDS if args.recover_from else 0,
        "total_limit_seconds": 86400}
    if args.recover_from:
        recover(plan, args.recover_from)
    return plan


def digest_text(text):
    import hashlib
    return hashlib.sha256(f"20261103:{text}".encode()).hexdigest()


def run(args):
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    state = {"status": "preparing", "started": args.started, "deadline": args.deadline,
             "failures": [], "training_complete_seeds": []}

    def update(status):
        state["status"] = status
        state["updated"] = time.time()
        write_json(root / "state.json", state)

    update("preparing")
    try:
        plan = prepare(args, root)
        preflight_deadline = min(args.deadline - RESERVE_S, time.time() + 7200)
        if "bank_recovery" not in plan:
            command = [sys.executable, str(HERE.parent / "compare/trust.py"),
                       "--tasks", plan["training_tasks"], "--reserved", *plan["bank_reserved"],
                       "--out", plan["bank"], "--role", ROLE, "--url", args.urls[0],
                       "--model", args.model, "--threads", str(args.threads)]
            if not batch([command], preflight_deadline, root, "bank"):
                state["failures"].append("bank generation failed or exceeded the two-hour preflight window")
                return 1
        plan["inputs"][plan["bank"]] = digest(plan["bank"])
        setup_path = root / "preflight.json"
        write_json(setup_path, plan)
        calibration = [worker("calibrate", setup_path, 0, condition, args.urls[i],
                              root / "calibration" / condition) for i, condition in enumerate(CONDITIONS)]
        if not batch(calibration, preflight_deadline, root, "calibration"):
            state["failures"].append("paired eight-rollout calibration failed; no main training released")
            return 1
        stress = [worker("stress", setup_path, 0, CONDITIONS[0], url,
                         root / "stress" / f"gpu{i}") for i, url in enumerate(args.urls)]
        if not batch(stress, preflight_deadline, root, "stress"):
            state["failures"].append("full-context runtime stress failed; no main training released")
            return 1
        pilot_s = [json.loads((root / "calibration" / c / "TIMING.json").read_text())["wall_s"]
                   for c in CONDITIONS]
        stress_s = [json.loads((root / "stress" / f"gpu{i}" / "STRESS.json").read_text())["wall_s"]
                    for i in range(2)]
        try:
            plan["budget"] = choose_budget(args.deadline - time.time(), pilot_s, stress_s)
        except ValueError as error:
            state["failures"].append(str(error))
            print(error, file=sys.stderr, flush=True)
            return 1
        plan["calibration"] = {"pilot_seconds": pilot_s, "stress_seconds": stress_s}
        plan_path = root / "plan.json"
        write_json(plan_path, plan)
        print(json.dumps({"released_budget": plan["budget"]}, indent=2), flush=True)
        update("training")
        train_start = time.time()
        for seed in SEEDS:
            pair_deadline = min(args.deadline - plan["budget"]["evaluation_seconds"] - RESERVE_S,
                                train_start + (seed + 1) * plan["budget"]["pair_seconds"])
            commands = [worker("train", plan_path, seed, condition, args.urls[(i + seed) % 2],
                               root / condition / f"seed{seed}") for i, condition in enumerate(CONDITIONS)]
            if not batch(commands, pair_deadline, root, f"training-seed{seed}"):
                state["failures"].append(f"training seed{seed} incomplete/failed; later seeds not launched")
                break
            state["training_complete_seeds"].append(seed)
            update("training")
        update("evaluating")
        complete = state["training_complete_seeds"]
        for offset in range(0, len(complete), 2):
            seeds = complete[offset:offset + 2]
            commands = [worker("eval", plan_path, seed, CONDITIONS[0], args.urls[i],
                               root / "evaluation" / f"seed{seed}") for i, seed in enumerate(seeds)]
            if not batch(commands, args.deadline - RESERVE_S, root, f"evaluation-wave{offset // 2}"):
                state["failures"].append(f"evaluation wave {seeds} incomplete/failed; later waves not launched")
                break
        update("complete" if not state["failures"] else "incomplete")
        return 0 if not state["failures"] else 1
    finally:
        if state["status"] not in ("complete", "incomplete"):
            update("incomplete")
        state["finished"] = time.time()
        write_json(root / "state.json", state)
        summary = report(root)
        print(f"Paired complete seeds: {summary['complete_seeds']}; "
              f"summary: {root / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--urls", nargs=2, required=True)
    parser.add_argument("--model", default="gemma4:12b-it-qat")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--allocation-seconds", type=int, default=86400)
    args = parser.parse_args()
    try:
        latest_deadline = application_deadline(args.started, args.allocation_seconds, bool(args.recover_from))
    except ValueError as error:
        parser.error(str(error))
    if (not all(math.isfinite(value) for value in (args.started, args.deadline))
            or args.started > time.time() or args.deadline <= time.time()
            or args.deadline - args.started > 24 * 3600
            or args.deadline > latest_deadline
            or args.threads < 1 or args.urls[0] == args.urls[1]):
        parser.error("need distinct servers, positive threads and an unexpired <=24h total deadline")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
