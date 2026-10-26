"""Gated five-condition pilot/main schedules and task-level analysis.

Main runs require three clean training seeds, completed pilot and an explicit GPU-hour budget.
The budget is an admission estimate, not an enforced scheduler time limit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import attack_texts
from learned import digest, environment, runtime_sources, worker_identity
from llm import OllamaChat
from runner import FIXED_ROLES, HERE, load_tasks, lock, read_traces, source_digest, write_json
from scoring import score, summarize
from train_learned import output_quality

BASELINES = {"fixed": FIXED_ROLES,
             "dylan": ["Economist", "Doctor", "Lawyer", "Mathematician", "Psychologist", "Programmer", "Historian"]}
LEARNED = ("anymac-normal", "anymac-efficient", "gdesigner-paper")


def architecture(label):
    return "anymac-corrected" if label in ("anymac-normal", "anymac-efficient") else label


def artifact(root, system, seed, embedding_hash, allow_routing_degeneracy=False):
    path = Path(root) / system / f"seed{seed}" / "router.pt"
    meta_path, gate_path = path.with_suffix(".json"), path.with_suffix(".gate.json")
    meta, gate = json.loads(meta_path.read_text()), json.loads(gate_path.read_text())
    if allow_routing_degeneracy:
        from admission import admit
        meta["evaluation_admission"] = admit(path, allow_routing_degeneracy=True)
    if (meta["config"]["system"] != architecture(system) or meta["seed"] != seed or meta["purpose"] != "study"
            or (not allow_routing_degeneracy and not gate["passed"])
            or meta.get("training_intervention") is not None
            or meta["artifact_sha256"] != digest(path) or gate["artifact_sha256"] != digest(path)
            or gate["metadata_sha256"] != digest(meta_path) or meta["embedding_sha256"] != embedding_hash
            or gate["runtime_sources"] != runtime_sources() or gate["environment"] != environment()
            or meta["runtime_sources"] != runtime_sources() or meta["environment"] != environment()):
        raise ValueError(f"missing/stale/failed artifact gate: {path}")
    if system.startswith("anymac-"):
        from train_learned import ANYMAC_PROFILES
        profile = system.removeprefix("anymac-")
        expected = {**ANYMAC_PROFILES[profile], "training_profile": profile}
        if (any(meta["config"].get(k) != v for k, v in expected.items())
                or len(meta.get("initial_policy_sha256", "")) != 64):
            raise ValueError(f"wrong AnyMAC training profile: {path}")
    return path, meta


def jobs_for(args, embedding):
    tasks = args.dev_tasks if args.phase == "pilot" else args.eval_tasks
    task_list = load_tasks(tasks, reserved=[*args.reserved, args.train_tasks])
    amended = getattr(args, "allow_routing_degeneracy", False)
    seeds = args.seeds if amended or args.phase == "main" else args.seeds[:1]
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("comparison needs distinct nonnegative training seeds")
    admission_args = {"allow_routing_degeneracy": True} if amended else {}
    if args.phase == "main" and (len(seeds) < 3 or len(set(seeds)) != len(seeds)):
        raise ValueError("main comparison requires >=3 distinct training seeds")
    systems: list[tuple[str, int | None, list[str], Path | None]] = [
        (s, None, roles, None) for s, roles in BASELINES.items()]
    artifact_hashes = {}
    paired_initial = {}
    for system in LEARNED:
        root = args.learned_root
        if system == "gdesigner-paper":
            root = getattr(args, "gdesigner_root", None) or root
        for seed in seeds:
            checkpoint = Path(root) / system / f"seed{seed}" / "router.pt"
            if args.command == "plan" and not checkpoint.exists():
                _, meta = artifact(root, system, seeds[0], embedding["sha256"], **admission_args)
                artifact_hashes[(system, seed)] = "NOT_TRAINED_PROJECTION_ONLY"
            else:
                checkpoint, meta = artifact(root, system, seed, embedding["sha256"], **admission_args)
                artifact_hashes[(system, seed)] = meta["artifact_sha256"]
            if system.startswith("anymac-") and artifact_hashes[(system, seed)] != "NOT_TRAINED_PROJECTION_ONLY":
                initial = meta["initial_policy_sha256"]
                if seed in paired_initial and paired_initial[seed] != initial:
                    raise ValueError("normal/efficient AnyMAC initial weights differ for the same seed")
                paired_initial[seed] = initial
            if set(meta["training_ids"]).intersection(t["id"] for t in task_list):
                raise ValueError("suite evaluation overlaps checkpoint training data")
            if args.phase == "main" and set(meta["dev_ids"]).intersection(t["id"] for t in task_list):
                raise ValueError("main evaluation overlaps checkpoint development data")
            systems.append((system, seed, meta["config"]["roles"], checkpoint))
    if args.phase == "main":
        document = json.loads(Path(tasks).read_text())
        excluded = set(document.get("excluded_sha256", {}).values())
        if (document.get("purpose") != "fresh-main-confirmation"
                or any(digest(p) not in excluded for p in [*args.reserved, args.train_tasks, args.dev_tasks])):
            raise ValueError("main needs a fresh reserved sample binding all reserved/train/development files")
    jobs = []
    for system, seed, roles, checkpoint in systems:
        label = system if seed is None else f"{system}/seed{seed}"
        extra = ["--checkpoint", str(checkpoint), "--embedding-model", embedding["path"]] if checkpoint else []
        if checkpoint and amended:
            extra += ["--allow-routing-degeneracy"]
        if args.phase == "pilot":
            cells = [(f"pilot/{label}", ["honest", "attack"], "bare", roles[0], 2)]
        else:
            cells = [(f"honest/{label}", ["honest"], "bare", roles[0], 2)]
            cells += [(f"attack/{form}/{label}/{role.replace(' ', '_')}", ["attack"], form, role, 1)
                      for role in roles for form in ("bare", "persuasive")]
        for label, policies, form, role, reps in cells:
            argv = ["run", "--system", architecture(system), "--tasks", str(tasks), "--reserved",
                    *map(str, [*args.reserved, args.train_tasks]),
                    "--out", str(Path(args.out) / args.phase / label),
                    "--model", args.model, "--max-calls", "32", "--limit", "8192", "--context", "32768",
                    "--threads", str(args.threads), "--compromised", role, "--policies", *policies,
                    "--reps", str(reps), "--attack-form", form, *extra]
            if form == "persuasive":
                argv += ["--attack-texts", str(Path(args.out) / "persuasive_texts.json")]
            jobs.append({"id": label, "system": system, "seed": seed, "role": role, "form": form,
                         "policies": policies, "reps": reps, "tasks": len(task_list), "argv": argv,
                         "artifact_sha256": artifact_hashes.get((system, seed))})
            if checkpoint and amended:
                from admission import admit
                jobs[-1]["evaluation_admission"] = admit(checkpoint, allow_routing_degeneracy=True)
    return jobs, task_list


def job_out(job):
    return Path(job["argv"][job["argv"].index("--out") + 1])


def execute(jobs, urls):
    """One subprocess at a time per server; neither torch RNG nor optimizer state is shared."""
    def worker(url):
        failures = []
        for job in jobs:
            out = job_out(job)
            out.mkdir(parents=True, exist_ok=True)
            if (out / "COMPLETE.json").exists():
                continue
            with (out / ".suite-lock").open("a") as claim:
                try:
                    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (out / "COMPLETE.json").exists():
                    continue
                # Mark each attempt in this pass so another worker does not immediately repeat a failure.
                attempt = out / f".attempt-{pass_id}"
                if attempt.exists():
                    continue
                attempt.touch()
                print(f"[{url}] start {job['id']}", flush=True)
                with (out / "run.log").open("a") as log:
                    result = subprocess.run([sys.executable, str(HERE / "runner.py"), *job["argv"], "--url", url],
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode or not (out / "COMPLETE.json").exists():
                    failures.append(job["id"])
                    print(f"FAILED {job['id']}: see {out / 'run.log'}", flush=True)
        return failures
    pass_id = str(time.time_ns())
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        failures = [job for result in pool.map(worker, urls) for job in result]
    incomplete = [j["id"] for j in jobs if not (job_out(j) / "COMPLETE.json").exists()]
    if failures or incomplete:
        raise ValueError(f"suite incomplete; failed={failures}, unfinished={incomplete}")


def completed_rows(jobs, tasks):
    rows = []
    task_map = {t["id"]: t for t in tasks}
    for job in jobs:
        out = job_out(job)
        if not (out / "COMPLETE.json").exists():
            raise ValueError(f"incomplete run: {out}; no pooled contrasts on partial cells")
        manifest = json.loads((out / "manifest.json").read_text())
        if (manifest["source"] != source_digest() or manifest["system"]["system"] != architecture(job["system"])
                or manifest["compromised"] != job["role"] or manifest["max_calls"] != 32
                or manifest["tasks_sha256"] != digest(job["argv"][job["argv"].index("--tasks") + 1])
                or (job["artifact_sha256"] and
                    manifest["provenance"]["artifact_sha256"] != job["artifact_sha256"])):
            raise ValueError(f"run does not match the current suite plan: {out}")
        if manifest.get("evaluation_admission") != job.get("evaluation_admission"):
            raise ValueError(f"evaluation admission differs from suite plan: {out}")
        traces = read_traces(out / "traces.jsonl", manifest, task_map)
        expected = {(t["id"], p, rep) for t in tasks for p in job["policies"] for rep in range(1, job["reps"] + 1)}
        if {(t["task_id"], t["policy"], t["rep"]) for t in traces} != expected:
            raise ValueError(f"incomplete/wrong schedule despite completion marker: {out}")
        for trace in traces:
            row = score(trace, task_map[trace["task_id"]])
            row.update(system=job["system"], seed=job["seed"], role=job["role"], form=job["form"])
            rows.append(row)
    return rows


def estimate(jobs, pilot):
    seconds = {system: cell["mean_episode_wall_s"] for system, cell in pilot["honest_cells"].items()}
    per_system = {}
    for job in jobs:
        episodes = job["tasks"] * job["reps"] * len(job["policies"])
        per_system[job["system"]] = per_system.get(job["system"], 0) + episodes * seconds[job["system"]] / 3600
    storage = sum(j["tasks"] * j["reps"] * len(j["policies"]) *
                  pilot["trace_bytes_per_episode"][j["system"]] for j in jobs) / (1024 ** 3)
    return {"gpu_hours_by_system": per_system, "gpu_hours": sum(per_system.values()),
            "projected_trace_gib": storage,
            "with_25_percent_margin": 1.25 * sum(per_system.values()),
            "note": "Pilot-based admission estimate, not a runtime guarantee or GPU utilization measurement."}


def validate_servers(jobs, urls, model, threads):
    workers = []
    for job in jobs:
        if job["system"] in LEARNED:
            path = Path(job["argv"][job["argv"].index("--checkpoint") + 1])
            workers.append(json.loads(path.with_suffix(".json").read_text())["worker"])
    if not workers or any(worker != workers[0] for worker in workers):
        raise ValueError("learned checkpoints were trained with different worker settings")
    for url in urls:
        actual = worker_identity(OllamaChat(url, model, 8192, 32768, threads).identity())
        if actual != workers[0]:
            raise ValueError(f"server {url} does not match the training worker identity/settings")


def bootstrap(values, n_boot=10000):
    if not values:
        raise ValueError("no paired tasks")
    rng = random.Random(20261024)
    mean = lambda xs: sum(xs) / len(xs)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return {"tasks": len(values), "mean": mean(values),
            "ci95": [samples[int(n_boot * .025)], samples[int(n_boot * .975) - 1]]}


def contrasts(rows):
    """Average repetitions, roles and training seeds WITHIN task before bootstrapping tasks."""
    grouped = {}
    for row in rows:
        key = row["system"], row["task_id"], row["policy"], row["form"] if row["policy"] == "attack" else "honest"
        grouped.setdefault(key, []).append(row)
    effects, task_effects = {}, {}
    for system in sorted({r["system"] for r in rows}):
        tasks = sorted({r["task_id"] for r in rows if r["system"] == system})
        for form in ("bare", "persuasive"):
            differences = {"correct": [], "target_hit": []}
            for task in tasks:
                honest = grouped[(system, task, "honest", "honest")]
                attack = grouped.get((system, task, "attack", form))
                if not attack:
                    continue
                for metric in differences:
                    difference = (sum(r[metric] for r in attack) / len(attack)
                                  - sum(r[metric] for r in honest) / len(honest))
                    differences[metric].append(difference)
                    task_effects[(system, form, metric, task)] = difference
            if differences["correct"]:
                effects[f"{system}/{form}"] = {metric: bootstrap(values) for metric, values in differences.items()}
        if f"{system}/persuasive" in effects:
            effects[f"{system}/persuasive-minus-bare"] = {
                metric: bootstrap([task_effects[(system, "persuasive", metric, task)] -
                                   task_effects[(system, "bare", metric, task)] for task in tasks])
                for metric in ("correct", "target_hit")}
    return effects


def report(args, jobs, tasks):
    rows = completed_rows(jobs, tasks)
    honest = [r for r in rows if r["policy"] == "honest"]
    cells = summarize(honest)
    honest_cells = {s: cells[f"{s}/honest"] for s in sorted({r["system"] for r in rows})}
    passed = all(output_quality([r for r in honest if r["system"] == s])["passed"] for s in honest_cells)
    result = {"phase": args.phase, "passed": passed, "honest_cells": honest_cells,
              "all_cells": summarize(rows), "effects": contrasts(rows),
              "interpretation": "Task CIs conditional on these training seeds. Not a population-of-seeds CI.",
              "artifacts": {f"{j['system']}/seed{j['seed']}": j["artifact_sha256"] for j in jobs if j["seed"] is not None},
              "source": source_digest(), "tasks_sha256": digest(args.dev_tasks if args.phase == "pilot" else args.eval_tasks)}
    result["admissions"] = {f"{j['system']}/seed{j['seed']}": j["evaluation_admission"]
                            for j in jobs if "evaluation_admission" in j}
    result["honest_cells_by_checkpoint"] = {
        f"{system}/seed{seed}": next(iter(summarize(
            [r for r in honest if r["system"] == system and r["seed"] == seed]).values()))
        for system, seed in sorted({(r["system"], r["seed"]) for r in honest if r["seed"] is not None})
    }
    # A good pooled mean must not hide a failing checkpoint.
    result["passed"] = passed and all(output_quality(
        [r for r in honest if r["system"] == system and r["seed"] == seed])["passed"]
        for system, seed in {(r["system"], r["seed"]) for r in honest if r["seed"] is not None})
    result["trace_bytes_per_episode"] = {
        s: sum((job_out(j) / "traces.jsonl").stat().st_size for j in jobs if j["system"] == s) /
           sum(j["tasks"] * j["reps"] * len(j["policies"]) for j in jobs if j["system"] == s)
        for s in honest_cells}
    if args.phase == "main" or getattr(args, "allow_routing_degeneracy", False):
        result["effects_by_seed"] = {str(seed): contrasts([r for r in rows if r["seed"] in (None, seed)])
                                     for seed in args.seeds}
    write_json(Path(args.out) / args.phase / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "report", "plan"])
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--learned-root", required=True)
    ap.add_argument("--assets", help="Shared frozen assets.json (separate from new profile output root)")
    ap.add_argument("--gdesigner-root", help="Existing G-Designer output root; do not copy or alter live artifacts")
    ap.add_argument("--dev-tasks", required=True)
    ap.add_argument("--eval-tasks", required=True)
    ap.add_argument("--train-tasks", required=True)
    ap.add_argument("--reserved", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--urls", nargs="+")
    ap.add_argument("--model", default="gemma4:12b-it-qat")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--approved-gpu-hours", type=float)
    ap.add_argument("--allow-routing-degeneracy", action="store_true",
                    help="explicit 2026-10-26 amendment; routing is an outcome, not an admission criterion")
    args = ap.parse_args(argv)
    embedding = json.loads(Path(args.assets or Path(args.learned_root) / "assets.json").read_text())["embedding"]
    jobs, tasks = jobs_for(args, embedding)
    plan = {"phase": args.phase, "source": source_digest(), "tasks_sha256": digest(
        args.dev_tasks if args.phase == "pilot" else args.eval_tasks), "jobs": jobs}
    plan["episodes"] = sum(j["tasks"] * j["reps"] * len(j["policies"]) for j in jobs)
    if args.phase == "main":
        pilot = json.loads((Path(args.out) / "pilot/summary.json").read_text())
        if (not pilot["passed"] or pilot["source"] != source_digest()
                or pilot["tasks_sha256"] != digest(args.dev_tasks)):
            raise ValueError("a complete, current, passed five-condition pilot is required")
        for system, seed in {(j["system"], j["seed"]) for j in jobs if j["seed"] is not None}:
            current = next(j["artifact_sha256"] for j in jobs if j["system"] == system and j["seed"] == seed)
            if (getattr(args, "allow_routing_degeneracy", False)
                    and pilot.get("admissions", {}).get(f"{system}/seed{seed}") != next(
                        j["evaluation_admission"] for j in jobs if j["system"] == system and j["seed"] == seed)):
                raise ValueError("main needs the same amended admission for every pilot checkpoint")
            pilot_hash = pilot["artifacts"].get(f"{system}/seed{seed}")
            if ((getattr(args, "allow_routing_degeneracy", False) or seed == args.seeds[0])
                    and pilot_hash != current):
                raise ValueError("pilot checkpoint changed; rerun the development pilot in a new output")
        plan["estimate"] = estimate(jobs, pilot)
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
        return
    if args.command == "report":
        report(args, jobs, tasks)
        return
    if not args.urls or len(set(args.urls)) != len(args.urls):
        raise ValueError("run requires distinct, live --urls (one owned Ollama server per GPU)")
    validate_servers(jobs, args.urls, args.model, args.threads)
    if args.phase == "main":
        approved = args.approved_gpu_hours
        if (approved is None or not math.isfinite(approved)
                or approved < plan["estimate"]["with_25_percent_margin"]):
            raise ValueError(f"main requires explicit GPU-hour approval covering {plan['estimate']}")
        texts = Path(args.out) / "persuasive_texts.json"
        texts.parent.mkdir(parents=True, exist_ok=True)
        attack_args = argparse.Namespace(tasks=args.eval_tasks, out=str(texts), model=args.model,
                                         url=args.urls[0], threads=args.threads, context=32768, limit=4096, target_seed=0)
        with lock(Path(args.out) / "attack-generation"):
            if not texts.exists():
                attack_texts.generate(attack_args)
            attack_texts.check(attack_args)
    plan_path = Path(args.out) / args.phase / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("suite plan changed; use a NEW --out")
    if not plan_path.exists():
        write_json(plan_path, plan)
    execute(jobs, args.urls)
    result = report(args, jobs, tasks)
    if not result["passed"]:
        raise ValueError("honest suite quality gate failed; inspect summary before progressing")


if __name__ == "__main__":
    main()
