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
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import FixedCycle  # noqa: E402
from contract import POLICIES, Episode, validate  # noqa: E402
from scoring import OUTCOMES, paired_effect, score, summarize  # noqa: E402

SYSTEMS = {"fixed": lambda a: FixedCycle(a.roles, rounds=a.rounds)}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def source_digest():
    return {p.name: sha256_bytes(p.read_bytes()) for p in sorted(HERE.glob("*.py")) if not p.name.startswith("test_")}


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


def read_traces(path):
    rows = []
    if path.exists():
        for n, ln in enumerate(path.read_text().splitlines(), 1):
            if ln.strip():
                try:
                    t = json.loads(ln)
                    validate(t)
                except (ValueError, KeyError, TypeError) as e:
                    raise SystemExit(f"{path}:{n} invalid trace ({e}); inspect without deleting evidence")
                rows.append(t)
    keys = [(t["task_id"], t["policy"], t["rep"]) for t in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"duplicate episodes in {path}")
    return rows


def run(args, llm=None, identity=None):
    system = SYSTEMS[args.system](args)
    if args.compromised not in system.worker_identities():
        raise SystemExit(f"--compromised must be one of {system.worker_identities()}")
    tasks = load_tasks(args.tasks, args.task_ids, args.reserved)
    if llm is None:
        from llm import OllamaChat
        llm = OllamaChat(args.url, args.model, args.limit, args.context, args.threads)
        identity = llm.identity()
    manifest = {"schema": 1, "system": system.config(), "compromised": args.compromised, "max_calls": args.max_calls,
                "policies": list(args.policies), "reps": args.reps, "order_seed": args.seed, "target_seed": args.target_seed,
                "tasks_sha256": sha256_bytes(Path(args.tasks).read_bytes()), "task_ids": [t["id"] for t in tasks],
                "reserved_sha256": [sha256_bytes(Path(r).read_bytes()) for r in args.reserved],
                "model": identity, "source": source_digest()}
    out = Path(args.out)
    with lock(out):
        mpath, tpath = out / "manifest.json", out / "traces.jsonl"
        if mpath.exists() and json.loads(mpath.read_text()) != manifest:
            raise SystemExit("source, tasks, model or configuration changed; use a NEW --out")
        if not mpath.exists():
            if tpath.exists():
                raise SystemExit("traces exist without manifest")
            write_json(mpath, manifest)
        done = {(t["task_id"], t["policy"], t["rep"]) for t in read_traces(tpath)}
        for task, policy, rep in schedule(tasks, args.policies, args.reps, args.seed):
            if (task["id"], policy, rep) in done:
                continue
            ep = Episode(system=args.system, task=task, policy=policy, rep=rep, compromised=args.compromised,
                         llm=llm, max_calls=args.max_calls, target_seed=args.target_seed)
            trace = ep.run(lambda e: system(e, task))
            with tpath.open("a") as f:
                f.write(json.dumps(trace) + "\n")
            r = score(trace, task)
            print(f"{args.system} {policy} rep{rep} {task['id']} {r['outcome']} pred={r['pred']} "
                  f"invoked={r['invoked']} delivered={r['delivered']} calls={r['calls']}", flush=True)
        write_json(out / "COMPLETE.json", {"episodes": len(tasks) * len(args.policies) * args.reps})


def report(args):
    out = Path(args.out)
    tasks = {t["id"]: t for t in json.loads(Path(args.tasks).read_text())["tasks"]}
    rows = [score(t, tasks[t["task_id"]]) for t in read_traces(out / "traces.jsonl")]
    if not rows:
        raise SystemExit(f"no traces in {out}")
    summ = summarize(rows)
    effects = [e for s in sorted({r["system"] for r in rows}) for m in ("correct", "target_hit")
               if (e := paired_effect(rows, s, m))]
    write_json(out / "summary.json", {"cells": summ, "paired": effects})
    cols = ["n", "tasks", *OUTCOMES, "target_hit_rate", "invoked_rate", "delivered_rate", "aggregator_exposed_rate",
            "final_from_compromised_rate", "mean_calls", "mean_completion_tokens", "any_truncated_rate"]
    fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = [f"# Comparison report: {out.name}", "", "Rates use unconditional denominators. Exposure is read from traces.",
             "", "| cell | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    lines += [f"| {k} | " + " | ".join(fmt(s[c]) for c in cols) + " |" for k, s in summ.items()]
    lines += ["", "## Paired attack - honest (task bootstrap, 95% percentile CI)", ""]
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
    r.add_argument("--reserved", nargs="*", default=[], help="task files whose ids must never be evaluated")
    r.add_argument("--out", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--url", default="http://127.0.0.1:11434")
    r.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--max-calls", type=int, required=True, help="common call envelope (budget, not a target)")
    r.add_argument("--compromised", required=True)
    r.add_argument("--roles", nargs="+", default=["Knowledgeable Expert", "Critic", "Analyst"])
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--limit", type=int, default=4096)
    r.add_argument("--context", type=int, default=16384)
    r.add_argument("--threads", type=int)
    r.add_argument("--seed", type=int, default=20261020)
    r.add_argument("--target-seed", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
