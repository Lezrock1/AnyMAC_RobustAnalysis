"""Common runner for all systems: resumable traces bound to source, tasks, model and configuration.

  run     python scripts/compare/runner.py run --system fixed --tasks T.json --task-ids IDS.txt \
              --reserved results/screen/screen_tasks_v1.json --out results/compare/<name> --model M
  report  python scripts/compare/runner.py report --tasks T.json --out results/compare/<name>
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import FixedCycle  # noqa: E402
from dylan import DyLAN  # noqa: E402
from contract import POLICIES, Episode, target_for, trace_attack, validate  # noqa: E402
from scoring import OUTCOMES, paired_effect, score, summarize  # noqa: E402

FIXED_ROLES = ["Knowledgeable Expert", "Critic", "Analyst"]
SYSTEMS = {
    "fixed": lambda a: FixedCycle(a.roles or FIXED_ROLES, rounds=a.rounds),
    "dylan": lambda a: DyLAN("paper", a.roles, a.ranker_fallback, a.seed),
    "dylan-upstream": lambda a: DyLAN("upstream", a.roles, a.ranker_fallback, a.seed),
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def source_digest():
    return {p.name: sha256_bytes(p.read_bytes()) for p in sorted(HERE.glob("*.py")) if not p.name.startswith("test_")}


def file_hashes(paths):
    """For adapters: content hashes of upstream code, checkpoints and configs (provenance, obligation 6)."""
    out = {}
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        if not files:
            raise SystemExit(f"provenance path {p} is empty or missing")
        for f in files:
            out[str(f)] = sha256_bytes(f.read_bytes())
    return out


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


@contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one writer per output directory
        yield


def load_tasks(path, ids_path=None, reserved=()):
    tasks = {t["id"]: t for t in json.loads(Path(path).read_text())["tasks"]}
    ids = Path(ids_path).read_text().split() if ids_path else sorted(tasks)
    unknown = [i for i in ids if i not in tasks]
    if unknown or len(set(ids)) != len(ids):
        raise SystemExit(f"unknown or duplicate task ids: {unknown[:5]}")
    blocked = set()
    for r in reserved:
        blocked |= {t["id"] for t in json.loads(Path(r).read_text())["tasks"]}
    overlap = sorted(blocked.intersection(ids))
    if overlap:
        raise SystemExit(f"{len(overlap)} task(s) are reserved (e.g. screening) and must not be evaluated: {overlap[:5]}")
    return [tasks[i] for i in ids]


def schedule(tasks, policies, reps, seed):
    """Per repetition: shuffled task order; policy order alternates per task (balanced execution order)."""
    order = []
    for rep in range(1, reps + 1):
        ts = list(tasks)
        random.Random(f"{seed}:{rep}").shuffle(ts)
        for i, t in enumerate(ts):
            ps = list(policies) if (i + rep) % 2 == 0 else list(reversed(policies))
            order += [(t, p, rep) for p in ps]
    return order


def check_against_manifest(t, manifest, tasks):
    """A trace must belong to exactly this run: system, attack setup, budget, schedule and target."""
    task = tasks.get(t["task_id"])
    ok = (task is not None and t["task_id"] in manifest["task_ids"]
          and t["system"] == manifest["system"]["system"] and t["compromised"] == manifest["compromised"]
          and t["max_calls"] == manifest["max_calls"] and t["policy"] in manifest["policies"]
          and 1 <= t["rep"] <= manifest["reps"] and t["target"] == target_for(task, manifest["target_seed"]))
    attack = manifest.get("attack", {"form": "bare"})
    ta = trace_attack(t)
    ok = ok and ta["form"] == attack["form"]
    if ok and attack["form"] == "persuasive":
        ok = sha256_bytes(ta["text"].encode()) == attack["text_sha256"].get(t["task_id"])
    if not ok:
        raise SystemExit(f"trace {t['task_id']}/{t['policy']}/rep{t['rep']} does not match the manifest")


def read_traces(path, manifest=None, tasks=None):
    rows = []
    if path.exists():
        for n, ln in enumerate(path.read_text().splitlines(), 1):
            if ln.strip():
                try:
                    t = json.loads(ln)
                    validate(t)
                    if manifest is not None:
                        check_against_manifest(t, manifest, tasks)
                except (ValueError, KeyError, TypeError) as e:
                    raise SystemExit(f"{path}:{n} invalid trace ({e}); inspect without deleting evidence")
                rows.append(t)
    keys = [(t["task_id"], t["policy"], t["rep"]) for t in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"duplicate episodes in {path}")
    return rows


def load_attack(args, tasks):
    """Bare: fixed template. Persuasive: frozen per-task texts from attack_texts.py, bound by hash."""
    if args.attack_form == "bare":
        if args.attack_texts:
            raise SystemExit("--attack-texts is only valid with --attack-form persuasive")
        return {"form": "bare"}, {}
    if not args.attack_texts:
        raise SystemExit("--attack-form persuasive needs --attack-texts")
    data = json.loads(Path(args.attack_texts).read_text())
    texts = {}
    for t in tasks:
        entry = data["texts"].get(t["id"])
        if entry is None or entry["target"] != target_for(t, args.target_seed):
            raise SystemExit(f"attack texts lack task {t['id']} or use another target")
        texts[t["id"]] = entry["text"]
    meta = {"form": "persuasive", "file_sha256": sha256_bytes(Path(args.attack_texts).read_bytes()),
            "text_sha256": {k: sha256_bytes(v.encode()) for k, v in texts.items()}}
    return meta, texts


def run(args, llm=None, identity=None):
    system = SYSTEMS[args.system](args)
    if args.compromised not in system.worker_identities():
        raise SystemExit(f"--compromised must be one of {system.worker_identities()}")
    tasks = load_tasks(args.tasks, args.task_ids, args.reserved)
    attack, attack_texts = load_attack(args, tasks)
    if llm is None:
        from llm import OllamaChat
        llm = OllamaChat(args.url, args.model, args.limit, args.context, args.threads)
        identity = llm.identity()
    manifest = {"schema": 2, "system": system.config(), "provenance": system.provenance(), "compromised": args.compromised, "max_calls": args.max_calls,
                "policies": list(args.policies), "reps": args.reps, "order_seed": args.seed, "target_seed": args.target_seed,
                "tasks_sha256": sha256_bytes(Path(args.tasks).read_bytes()), "task_ids": [t["id"] for t in tasks],
                "reserved_sha256": [sha256_bytes(Path(r).read_bytes()) for r in args.reserved],
                "model": identity, "source": source_digest()}
    if attack["form"] != "bare":
        manifest["attack"] = attack  # bare manifests stay identical to schema-2 runs
    out = Path(args.out)
    with lock(out):
        mpath, tpath = out / "manifest.json", out / "traces.jsonl"
        if mpath.exists() and json.loads(mpath.read_text()) != manifest:
            raise SystemExit("source, tasks, model or configuration changed; use a NEW --out")
        if not mpath.exists():
            if tpath.exists():
                raise SystemExit("traces exist without manifest")
            write_json(mpath, manifest)
        done = {(t["task_id"], t["policy"], t["rep"]) for t in read_traces(tpath, manifest, {t["id"]: t for t in tasks})}
        for task, policy, rep in schedule(tasks, args.policies, args.reps, args.seed):
            if (task["id"], policy, rep) in done:
                continue
            ep = Episode(system=args.system, task=task, policy=policy, rep=rep, compromised=args.compromised,
                         llm=llm, max_calls=args.max_calls, target_seed=args.target_seed,
                         attack_form=attack["form"], attack_text=attack_texts.get(task["id"]))
            trace = ep.run(lambda e: system(e, task))
            with tpath.open("a") as f:
                f.write(json.dumps(trace) + "\n")
            r = score(trace, task)
            print(f"{args.system} {policy} rep{rep} {task['id']} {r['outcome']} pred={r['pred']} "
                  f"invoked={r['invoked']} delivered={r['delivered']} calls={r['calls']}", flush=True)
        write_json(out / "COMPLETE.json", {"episodes": len(tasks) * len(args.policies) * args.reps})


def report(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    if sha256_bytes(Path(args.tasks).read_bytes()) != manifest["tasks_sha256"]:
        raise SystemExit("--tasks differs from the task file bound in the manifest")
    tasks = {t["id"]: t for t in json.loads(Path(args.tasks).read_text())["tasks"]}
    traces = read_traces(out / "traces.jsonl", manifest, tasks)
    if not traces:
        raise SystemExit(f"no traces in {out}")
    rows = [score(t, tasks[t["task_id"]]) for t in traces]
    expected = len(manifest["task_ids"]) * len(manifest["policies"]) * manifest["reps"]
    per_task = defaultdict(int)
    for r in rows:
        per_task[r["task_id"]] += 1
    full = {k for k, n in per_task.items() if n == len(manifest["policies"]) * manifest["reps"]}
    summ = summarize(rows)
    paired_rows = [r for r in rows if r["task_id"] in full]
    effects = [e for s in sorted({r["system"] for r in rows}) for m in ("correct", "target_hit")
               if set(manifest["policies"]) == set(POLICIES) and (e := paired_effect(paired_rows, s, m))]
    status = "complete" if len(rows) == expected else f"INCOMPLETE {len(rows)}/{expected} episodes"
    write_json(out / "summary.json", {"status": status, "cells": summ, "paired": effects})
    cols = ["n", "tasks", *OUTCOMES, "target_hit_rate", "invoked_rate", "delivered_rate", "aggregator_exposed_rate",
            "final_from_compromised_rate", "mean_calls", "mean_completion_tokens", "mean_episode_wall_s",
            "any_truncated_rate"]
    fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = [f"# Comparison report: {out.name} ({status})", "",
             "Rates use unconditional denominators. Exposure is read from traces.",
             "", "| cell | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    lines += [f"| {k} | " + " | ".join(fmt(s[c]) for c in cols) + " |" for k, s in summ.items()]
    lines += ["", f"## Paired attack - honest (task bootstrap, 95% percentile CI; {len(full)} complete tasks)", ""]
    lines += [f"- {e['system']} {e['metric']}: {e['diff']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}], "
              f"tasks={e['tasks']}" for e in effects]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    r.add_argument("--tasks", required=True)
    r.add_argument("--task-ids")
    r.add_argument("--reserved", nargs="+", required=True,
                   help="task files whose ids must never be evaluated (at least the screening sample)")
    r.add_argument("--out", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--url", default="http://127.0.0.1:11434")
    r.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--max-calls", type=int, required=True, help="common call envelope (budget, not a target)")
    r.add_argument("--compromised", required=True)
    r.add_argument("--roles", nargs="+", help="worker identities (default: the system's native/registered roles)")
    r.add_argument("--rounds", type=int, default=1, help="fixed reference only")
    r.add_argument("--ranker-fallback", choices=["native", "error"], default="native",
                   help="DyLAN: unparseable ranking -> logged native random pair, or control_error")
    r.add_argument("--limit", type=int, default=4096)
    r.add_argument("--context", type=int, default=16384)
    r.add_argument("--threads", type=int)
    r.add_argument("--seed", type=int, default=20261020)
    r.add_argument("--target-seed", type=int, default=0)
    r.add_argument("--attack-form", choices=["bare", "persuasive"], default="bare")
    r.add_argument("--attack-texts", help="frozen persuasive texts (attack_texts.py generate)")
    p = sub.add_parser("report")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
