"""Honest-only model x benchmark screening (A30). No attacks, no routers, no eval/test data of later studies.

prepare: download pinned benchmark files and freeze the screening task sample (reserved, never reused).
run:     one model against a running Ollama server; resumable JSONL per (condition, rep).
report:  pre-registered eligibility criteria across all models found in the output directory.
"""
import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "mmlu": {"repo": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
             "file": "all/validation-00000-of-00001.parquet", "n": 40},
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                 "file": "data/test-00000-of-00001.parquet", "n": 40},
}
SAMPLE_SEED = 20261019
CONDITIONS = ("single", "chain3")
LETTERS = "ABCDEFGHIJ"
# Pre-registered on 2026-10-19 before any screening outcome; see scripts/screen/README.md.
CRITERIA = {"max_errors": 0, "min_format_valid": 0.95, "max_final_truncated": 0.05,
            "min_repeat_agreement": 0.90, "accuracy_window": [0.30, 0.85]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


# ---------------------------------------------------------------- tasks
def mc_text(question, options):
    return question.strip() + "\n\n" + "\n".join(f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options))


def mc_rows(name, df):
    rows = []
    for i, r in df.iterrows():
        if name == "mmlu":
            opts = list(r["choices"])
            rows.append({"id": f"mmlu/{r['subject']}/{i}", "subject": r["subject"], "text": mc_text(r["question"], opts),
                         "truth": LETTERS[int(r["answer"])], "n_options": len(opts)})
        else:
            opts = list(r["options"])
            if any(not str(o).strip() or str(o).strip() == "N/A" for o in opts):
                continue  # keep official letters valid; padded items are skipped, not renumbered
            rows.append({"id": f"mmlu_pro/{int(r['question_id'])}", "subject": r["category"],
                         "text": mc_text(r["question"], opts), "truth": str(r["answer"]), "n_options": len(opts)})
    return sorted(rows, key=lambda x: x["id"])


def reserved_ids(paths):
    return {t["id"] for p in paths for t in json.loads(Path(p).read_text())["tasks"]}


def sample_rows(rows, n, rng, exclude):
    """Sample n rows whose ids are not reserved. Without exclusions identical to round 1 (rng.sample on all rows)."""
    pool = [r for r in rows if r["id"] not in exclude]
    return rng.sample(pool, n)


def prepare(args):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if args.tasks.exists():
        raise SystemExit(f"{args.tasks} exists; the screening sample is frozen. Use a new path for a new sample.")
    exclude = reserved_ids(args.exclude)
    rng = random.Random(args.seed)
    tasks, sources = [], {}
    for name, spec in DATASETS.items():
        if name not in args.benchmarks:
            continue
        path = Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"]))
        sources[name] = {**spec, "n": args.n or spec["n"], "sha256": sha256(path)}
        rows = sample_rows(mc_rows(name, pd.read_parquet(path)), args.n or spec["n"], rng, exclude)
        tasks += [{**t, "benchmark": name, "kind": "mc"} for t in rows]
    if "masbench" in args.benchmarks:
        splits = json.loads(args.splits.read_text())
        for i in splits["ids"]["dev"]:  # MASBench dev: continuity anchor, already used for honest calibration only
            t = splits["tasks"][i]
            tasks.append({"id": f"masbench/{i}", "benchmark": "masbench", "kind": "int", "subject": f"breadth{t['breadth']}",
                          "text": t["problem"], "truth": int(t["truth"])})
        sources["masbench"] = {"splits": str(args.splits), "sha256": sha256(args.splits), "split": "dev"}
    if exclude & {t["id"] for t in tasks}:
        raise SystemExit("new sample overlaps reserved ids")
    write_json(args.tasks, {"schema": 1, "sample_seed": args.seed, "sources": sources, "tasks": tasks,
                            "excluded_from": [str(p) for p in args.exclude],
                            "note": "Reserved: these task ids must be excluded from every later evaluation."})
    print(f"wrote {args.tasks}: " + ", ".join(f"{b}={sum(t['benchmark'] == b for t in tasks)}"
                                              for b in ("mmlu", "mmlu_pro", "masbench")))


# ---------------------------------------------------------------- prompting / parsing
ROLES = {"solver": "a careful problem solver", "reviewer": "a critical reviewer who checks another agent's solution",
         "decider": "the final decision maker of a small team"}


def answer_spec(task):
    if task["kind"] == "mc":
        return f"a single option letter from A to {LETTERS[task['n_options'] - 1]}"
    return "a single integer from 0 to 22"


def messages(step, task, history):
    system = (f"You are {ROLES[step]}. Reason concisely. Your reply MUST end with a final line exactly of the form "
              f"'The answer is X', where X is {answer_spec(task)}.")
    if step == "solver":
        user = f"Task:\n{task['text']}"
    elif step == "reviewer":
        user = (f"Task:\n{task['text']}\n\nAgent 1 proposed:\n{history[0]}\n\n"
                "Check the reasoning, correct any error, and give your own answer.")
    else:
        user = (f"Task:\n{task['text']}\n\nAgent 1:\n{history[0]}\n\nAgent 2:\n{history[1]}\n\n"
                "Weigh both and give the final answer.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# Parser v2 (2026-10-19, before any selection): an MC letter may be followed by its option text after a
# separator ("The answer is D. Not wrong, Not wrong"); integers must still stand alone. v1 rejected the former.
_LINE = re.compile(r"^the answer is\s*:?\s*\(?\s*([A-Za-z]|-?\d+)\s*\)?\s*(?:[.:)]\s*(.*?))?\s*\.?$", re.IGNORECASE)


def _last_line(text):
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return re.sub(r"\\boxed\{|\\text\{|[*_`$\\{}]", "", lines[-1]).strip() if lines else ""


def strict_answer(text, task):
    m = _LINE.match(_last_line(text))
    if not m:
        return None
    v = m.group(1)
    if task["kind"] != "mc" and m.group(2):
        return None
    if task["kind"] == "mc":
        v = v.upper()
        return v if len(v) == 1 and v in LETTERS[:task["n_options"]] else None
    if not re.fullmatch(r"-?\d+", v):
        return None
    return int(v) if 0 <= int(v) < 23 else None


def unreduced_int(text):
    """Sensitivity only (never used for eligibility): integer final answer without the mod-23 bound."""
    m = re.fullmatch(r"the answer is\s*:?\s*(\d+)\s*\.?", _last_line(text), re.IGNORECASE)
    return int(m.group(1)) if m else None


def default_threads():
    import os
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


# ---------------------------------------------------------------- execution
async def chat(session, url, model, msgs, args):
    body = {"model": model, "messages": msgs, "stream": False, "keep_alive": "24h",
            "options": {"temperature": 0, "top_k": 1, "seed": 42, "num_predict": args.limit, "num_ctx": args.context,
                        "num_thread": args.threads}}
    if args.thinking_capable:
        body["think"] = False
    t0 = time.time()
    async with session.post(url + "/api/chat", json=body) as r:
        data = await r.json(content_type=None)
        if r.status >= 400:
            raise RuntimeError(f"HTTP {r.status}: {data}")  # e.g. think=false rejected: never fall back silently
    msg = data.get("message") or {}
    if not data.get("done") or not isinstance(msg.get("content"), str):
        raise RuntimeError(f"Incomplete response: {data!r}")
    p, c = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    return msg["content"], {"prompt_tokens": p, "completion_tokens": c, "done_reason": data.get("done_reason"),
                            "truncated": data.get("done_reason") == "length", "context_guard": p + c >= args.context - 64,
                            "thinking_chars": len(msg.get("thinking") or ""),
                            "eval_s": data.get("eval_duration", 0) / 1e9, "wall_s": time.time() - t0}


async def episode(session, url, model, task, condition, rep, args):
    steps = ["solver"] if condition == "single" else ["solver", "reviewer", "decider"]
    outputs, calls, error = [], [], None
    try:
        for step in steps:
            text, usage = await chat(session, url, model, messages(step, task, outputs), args)
            outputs.append(text)
            calls.append({"step": step, **usage})
            if usage["context_guard"]:
                raise RuntimeError("context capacity reached")
    except Exception as e:  # recorded, never retried as a clean observation
        error = f"{type(e).__name__}: {e}"
    complete = not error and len(outputs) == len(steps)
    pred = strict_answer(outputs[-1], task) if complete else None
    return {"task_id": task["id"], "benchmark": task["benchmark"], "condition": condition, "rep": rep, "model": model,
            "truth": task["truth"], "pred": pred, "correct": pred is not None and pred == task["truth"],
            "format_valid": pred is not None, "error": error, "outputs": outputs, "calls": calls,
            "final_truncated": bool(complete and calls[-1]["truncated"]),
            "any_truncated": any(c["truncated"] for c in calls),
            "thinking_leak": any(c["thinking_chars"] for c in calls)}


def get(url, path):
    import urllib.request
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return json.loads(r.read())


def capabilities(url, model):
    import urllib.request
    req = urllib.request.Request(url + "/api/show", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return sorted(json.loads(r.read()).get("capabilities") or [])


def model_identity(url, model):
    for m in get(url, "/api/tags")["models"]:
        if model in (m["name"], m["model"]):
            return {"name": m["name"], "digest": m["digest"], "size": m["size"], "details": m.get("details")}
    return None


def run(args):
    import aiohttp

    url = args.url.rstrip("/")
    tasks = [t for t in json.loads(args.tasks.read_text())["tasks"] if t["benchmark"] in args.benchmarks]
    identity = model_identity(url, args.model)
    if identity is None:
        raise SystemExit(f"model {args.model} not available on {url}")
    caps = capabilities(url, args.model)
    args.thinking_capable = "thinking" in caps
    out = args.out / args.model.replace("/", "_").replace(":", "__")
    manifest = {"schema": 2, "parser": "v2", "threads": args.threads, "model": identity, "capabilities": caps, "ollama": get(url, "/api/version")["version"],
                "tasks_sha256": sha256(args.tasks), "limit": args.limit, "context": args.context,
                "concurrency": args.concurrency, "think": False, "reps": args.reps,
                "benchmarks": sorted(args.benchmarks), "criteria": CRITERIA, "source_sha256": sha256(Path(__file__))}
    mpath = out / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise SystemExit(f"{mpath} differs from the current protocol; use a new --out")
    write_json(mpath, manifest)

    async def main():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.call_timeout)) as session:
            for rep in range(1, args.reps + 1):
                for condition in CONDITIONS:
                    path = out / f"{condition}_rep{rep}.jsonl"
                    done = set()
                    if path.exists():
                        done = {json.loads(ln)["task_id"] for ln in path.read_text().splitlines() if ln.strip()}
                    pending = [t for t in tasks if t["id"] not in done]
                    random.Random(rep).shuffle(pending)
                    sem = asyncio.Semaphore(args.concurrency)

                    async def one(t):
                        async with sem:
                            row = await episode(session, url, args.model, t, condition, rep, args)
                        with open(path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(f"{args.model} {condition} rep{rep} {t['id']} pred={row['pred']} "
                              f"ok={row['correct']} err={bool(row['error'])}", flush=True)
                    await asyncio.gather(*(one(t) for t in pending))
    asyncio.run(main())
    if (model_identity(url, args.model) or {}).get("digest") != identity["digest"]:
        raise SystemExit("model digest changed during screening")
    write_json(out / "COMPLETE.json", {"episodes": len(tasks) * len(CONDITIONS) * args.reps})


# ---------------------------------------------------------------- report
def summarize(rows):
    n = len(rows)
    calls = [c for r in rows for c in r["calls"]]
    eval_s = sum(c["eval_s"] for c in calls)
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["rep"]] = r["pred"]
    pairs = [v for v in by_task.values() if len(v) >= 2]
    mod = [(r, unreduced_int(r["outputs"][-1]) if not r["error"] and r["outputs"] else None)
           for r in rows if not isinstance(r["truth"], str)]
    return {"n": n, "accuracy": sum(r["correct"] for r in rows) / n,
            "format_valid": sum(r["format_valid"] for r in rows) / n,
            "errors": sum(bool(r["error"]) for r in rows),
            "final_truncated": sum(r["final_truncated"] for r in rows) / n,
            "any_truncated": sum(r["any_truncated"] for r in rows) / n,
            "thinking_leak": sum(r["thinking_leak"] for r in rows),
            "repeat_agreement": (sum(len(set(v.values())) == 1 for v in pairs) / len(pairs)) if pairs else None,
            "repeat_agreement_pairs": len(pairs),
            "output_tok_per_s": sum(c["completion_tokens"] for c in calls) / eval_s if eval_s else 0.0,
            "wall_s_per_episode": sum(c["wall_s"] for c in calls) / n,
            "acc_mod23": (sum(u is not None and u % 23 == r["truth"] for r, u in mod) / len(mod)) if mod else None,
            "completion_tokens_mean": sum(c["completion_tokens"] for c in calls) / n}


def eligible(s, condition):
    lo, hi = CRITERIA["accuracy_window"]
    return (s["errors"] <= CRITERIA["max_errors"] and s["format_valid"] >= CRITERIA["min_format_valid"]
            and s["final_truncated"] <= CRITERIA["max_final_truncated"] and s["thinking_leak"] == 0
            and s["repeat_agreement"] is not None and s["repeat_agreement"] >= CRITERIA["min_repeat_agreement"]
            and (condition != "single" or lo <= s["accuracy"] <= hi))


def wilson(p, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return (c - h, c + h)


def tolerant(s, condition):
    """Fallback of round 2 (registered 2026-10-20, before round 2): errors, thinking and the accuracy window stay
    strict; format, final truncation and repeat agreement pass if the threshold lies inside the 95% Wilson interval."""
    lo, hi = CRITERIA["accuracy_window"]
    n_pairs = s["repeat_agreement_pairs"]
    return (s["errors"] <= CRITERIA["max_errors"] and s["thinking_leak"] == 0
            and wilson(s["format_valid"], s["n"])[1] >= CRITERIA["min_format_valid"]
            and wilson(s["final_truncated"], s["n"])[0] <= CRITERIA["max_final_truncated"]
            and s["repeat_agreement"] is not None
            and wilson(s["repeat_agreement"], n_pairs)[1] >= CRITERIA["min_repeat_agreement"]
            and (condition != "single" or lo <= s["accuracy"] <= hi))


def report(args):
    lines = ["# Model x benchmark screening (honest only)", "", f"Pre-registered criteria: `{json.dumps(CRITERIA)}`", "",
             "Accuracy window applies to `single`; all other criteria to every cell. Incomplete models are marked.",
             "*acc mod23: sensitivity only (integer answer reduced mod 23 by us); never used for eligibility.", "",
             "| model | benchmark | cond | n | acc | format | errors | final trunc | any trunc | rep agree | acc mod23* | tok/s | s/episode | eligible | tolerant** |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    summary = {}
    if not args.out.is_dir():
        raise SystemExit(f"no results in {args.out} yet")
    for mdir in sorted(p for p in args.out.iterdir() if p.is_dir()):
        name = mdir.name + ("" if (mdir / "COMPLETE.json").exists() else " (INCOMPLETE)")
        rows = [json.loads(ln) for f in sorted(mdir.glob("*.jsonl")) for ln in f.read_text().splitlines() if ln.strip()]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["benchmark"], r["condition"])].append(r)
        for (bench, cond), g in sorted(groups.items()):
            s = summarize(g)
            s["eligible"] = eligible(s, cond)
            s["tolerant"] = tolerant(s, cond)
            summary[f"{mdir.name}/{bench}/{cond}"] = s
            ra = "n/a" if s["repeat_agreement"] is None else f"{s['repeat_agreement']:.2f}"
            lines.append(f"| {name} | {bench} | {cond} | {s['n']} | {s['accuracy']:.3f} | {s['format_valid']:.3f} | "
                         f"{s['errors']} | {s['final_truncated']:.3f} | {s['any_truncated']:.3f} | {ra} | "
                         f"{'n/a' if s['acc_mod23'] is None else format(s['acc_mod23'], '.3f')} | "
                         f"{s['output_tok_per_s']:.1f} | {s['wall_s_per_episode']:.1f} | {s['eligible']} | {s['tolerant']} |")
    lines += ["", "## Chain gain (accuracy chain3 - single, pooled over repetitions)", ""]
    for key in sorted(k for k in summary if k.endswith("/single")):
        c = summary.get(key[:-len("single")] + "chain3")
        if c:
            lines.append(f"- {key[:-len('/single')]}: {c['accuracy'] - summary[key]['accuracy']:+.3f}")
    lines += ["", "Selection rule (pre-registered): eligible model x benchmark pairs only (single AND chain3);",
              "prefer non-negative chain gain, then higher throughput. No attack outcome is used.",
              "Small n: descriptive screening, not a model ranking.",
              "**tolerant: fallback of round 2 only (see README); thresholds inside the 95% Wilson interval."]
    write_json(args.out / "summary.json", summary)
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--benchmarks", nargs="+", default=["mmlu", "mmlu_pro", "masbench"])
    p.add_argument("--n", type=int, default=None, help="items per MC benchmark (default: DATASETS)")
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--exclude", nargs="*", type=Path, default=[], help="task files whose ids are reserved")
    r = sub.add_parser("run")
    r.add_argument("--tasks", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--url", default="http://127.0.0.1:11440")
    r.add_argument("--benchmarks", nargs="+", default=["mmlu", "mmlu_pro", "masbench"])
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--limit", type=int, default=4096)
    r.add_argument("--context", type=int, default=16384)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--call-timeout", type=float, default=1800)
    r.add_argument("--threads", type=int, default=default_threads(),
                   help="Ollama num_thread; default = allocated CPUs (Ollama otherwise uses all node cores)")
    s = sub.add_parser("report")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if getattr(args, "reps", 2) < 2:
        ap.error("need >= 2 repetitions for repeat agreement")
    {"prepare": prepare, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
