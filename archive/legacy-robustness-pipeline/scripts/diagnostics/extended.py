"""Counterbalanced development diagnostics. No research code or results are modified."""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Optional
import urllib.error

import diagnose as core


CONDITIONS = ("s1_serial", "s12_serial", "s12_parallel")
BLOCKS = 3
REPEATS = 3


def pipeline_parser():
    path = core.ROOT / "robust/data.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name == "parse_answer")
             or (isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "_ANS_RE" for t in node.targets))]
    if len(nodes) != 2:
        raise ValueError("Cannot extract original answer parser")
    namespace = {"re": re, "Optional": Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_answer"]


def parser_audit(root, out):
    parse = pipeline_parser()
    results, samples = {}, []
    for path in sorted((root / "eval").glob("exp*.jsonl")):
        rows, invalid = core.load_rows(path)
        counts = Counter()
        candidates = defaultdict(list)
        for row in rows:
            text = row.get("final_text") or ""
            loose, strict = parse(text), core.strict_answer(text)
            counts["rows"] += 1
            counts["backend_errors"] += bool(row.get("error"))
            counts["original_correct"] += loose == row["truth"]
            counts["strict_correct"] += strict == row["truth"]
            counts["parser_disagrees"] += loose != strict
            counts["original_out_of_range"] += loose is not None and not 0 <= loose < 23
            if loose != strict:
                candidates["parser_disagrees"].append({
                    "file": path.name, "task_id": row["task_id"],
                    "marker": row["marker"], "policy": row["policy"],
                    "truth": row["truth"], "original": loose, "strict": strict,
                    "error": row.get("error"), "text": text,
                })
        results[path.name] = {"counts": dict(counts), "invalid_lines": invalid}
        # Stable task ordering, not selection by whether the prediction was correct.
        chosen = sorted(candidates["parser_disagrees"],
                        key=lambda r: (r["task_id"], r["marker"], r["policy"]))[:4]
        samples.extend(chosen)
    core.write_json(out / "parser_samples.json", samples)
    return {"files": results, "samples": "parser_samples.json",
            "warning": "Strict parser is a format check, not a gold-standard semantic scorer; manually review samples."}


def initialize(args):
    splits = json.loads(args.splits.read_text())
    ids = splits["ids"]["dev"][:args.tasks]
    if len(ids) != args.tasks or len(set(ids)) != args.tasks or args.tasks < 1:
        raise ValueError("Not enough unique development tasks")
    args.out.mkdir(parents=True, exist_ok=False)
    graph = core.ROOT / "GDesigner/graph/graph.py"
    manifest = {
        "schema": 2, "model": args.model, "task_ids": ids,
        "tasks": {tid: splits["tasks"][tid] for tid in ids},
        "role_prompt": core.role_prompt(core.ROOT / "GDesigner/prompt/masbench_prompt_set.py"),
        "deadline": args.deadline, "created": time.time(), "blocks": BLOCKS,
        "repeats": REPEATS, "planned": BLOCKS * args.tasks * REPEATS * len(CONDITIONS) * 4,
        "options": {"temperature": 0, "top_p": 1.0, "top_k": 1, "repeat_penalty": 1.0,
                    "seed": 42, "num_ctx": 8192},
        "router": core.router_audit(graph),
        "gradients": {v: core.gradient_probe(graph, v)
                      for v in ("original", "indices", "indices_and_early_decision")},
        "parser_audit": parser_audit(args.pilot, args.out),
        "source_sha256": {str(p.relative_to(core.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (graph, core.ROOT / "robust/data.py",
                                    Path(__file__), Path(core.__file__))},
    }
    core.write_json(args.out / "manifest.json", manifest)
    summarize(args.out, "INITIALIZED")


def job_key(row):
    return (row["block"], row["condition"], row["format"], row["limit"],
            row["task_id"], row["repeat"])


def schedule(manifest, block, condition):
    cells = [(fmt, limit, tid) for fmt in ("generate", "chat") for limit in (1536, 3072)
             for tid in manifest["task_ids"]]
    # Same shuffled task/cell order across conditions within a block.
    random.Random(1200 + block).shuffle(cells)
    for fmt, limit, tid in cells:
        yield [{"block": block, "condition": condition, "format": fmt, "limit": limit,
                "task_id": tid, "repeat": k} for k in range(manifest["repeats"])]


def payload_for(manifest, job):
    task = manifest["tasks"][job["task_id"]]
    messages = [{"role": "system", "content": manifest["role_prompt"]},
                {"role": "user", "content": f"The task is: [Projekt Blau]\n{task['problem']}\n"}]
    payload = {"model": manifest["model"], "stream": False,
               "options": {**manifest["options"], "num_predict": job["limit"]}}
    if job["format"] == "chat":
        payload["messages"] = messages
    else:
        payload["prompt"] = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return payload


def request_one(manifest, job, base):
    payload = payload_for(manifest, job)
    row = {**job, "started": time.time(), "request": payload,
           "truth": manifest["tasks"][job["task_id"]]["truth"]}
    remaining = manifest["deadline"] - row["started"]
    if remaining <= 0:
        return {**row, "error": "deadline_before_submission", "error_kind": "budget",
                "wall_s": 0}
    timeout = min(1200, remaining)
    row["timeout_s"] = timeout
    start = time.monotonic()
    try:
        backend = core.http(base, "/api/" + job["format"], payload, timeout)
        if not backend.get("done"):
            raise ValueError("Incomplete backend response")
        text = backend["message"]["content"] if job["format"] == "chat" else backend["response"]
        if not isinstance(text, str):
            raise ValueError("Backend text is not a string")
        total_ns = backend.get("total_duration")
        row.update(
            error=None, error_kind=None, backend=backend, answer=core.strict_answer(text),
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            backend_total_s=total_ns / 1e9 if isinstance(total_ns, (int, float)) else None,
        )
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        is_timeout = isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError))
        kind = ("budget_limited_timeout" if timeout < 1200 else "request_timeout") if is_timeout else "backend_error"
        row.update(error=f"{type(error).__name__}: {error}", error_kind=kind)
    row["wall_s"] = round(time.monotonic() - start, 3)
    return row


def summarize(out, status):
    manifest = json.loads((out / "manifest.json").read_text())
    rows, invalid = core.load_rows(out / "responses.jsonl")
    if invalid:
        raise ValueError(f"Invalid extended diagnostic rows: {invalid}")
    if len({job_key(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate diagnostic request IDs")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["block"], row["condition"], row["format"], row["limit"])].append(row)
    cells = []
    lines = [
        "AnyMAC extended diagnostic report", f"Status: {status}",
        f"Recorded/planned: {len(rows)}/{manifest['planned']}",
        "Development-only; no significance claim; incomplete cells are not directly comparable.",
        "Each block restarts servers; block 0: slots 1 then 12; block 1: 12 then 1; block 2: 1 then 12.",
        "Only one request at a time with one slot; three parallel requests only with twelve slots.",
        "Backend total_duration and HTTP wall time are stored, but their difference is NOT proven queue time.",
        "Errors are not auto-retried; budget-limited timeouts are separate from regular request timeouts.",
        "", "Router/gradient probes (synthetic, early non-forced decision, NOT a production fix):",
        json.dumps(manifest["gradients"], indent=2),
        "", "Pilot parser audit (sample texts in parser_samples.json):",
        json.dumps(manifest["parser_audit"], indent=2), "", "LLM cells:",
    ]
    for key, items in sorted(groups.items()):
        good = [r for r in items if not r["error"]]
        by_task = defaultdict(list)
        for r in good:
            by_task[r["task_id"]].append(r)
        cell = {
            "block": key[0], "condition": key[1], "format": key[2], "limit": key[3],
            "recorded": len(items), "expected": len(manifest["task_ids"]) * REPEATS,
            "errors": dict(Counter(r["error_kind"] for r in items if r["error"])),
            "correct": sum(r["answer"] == r["truth"] for r in good),
            "invalid_final": sum(r["answer"] is None for r in good),
            "truncated": sum(r["backend"].get("done_reason") == "length" for r in good),
            "complete_task_repeats": sum(len(v) == REPEATS for v in by_task.values()),
            "variable_answers_complete_tasks": sum(len(v) == REPEATS and len({r["answer"] for r in v}) > 1
                                                  for v in by_task.values()),
        }
        cells.append(cell)
        lines.append(json.dumps(cell))
        for tid, task_rows in sorted(by_task.items()):
            ordered = sorted(task_rows, key=lambda r: r["repeat"])
            lines.append(f"  {tid}: answers={[r['answer'] for r in ordered]}, "
                         f"texts={len({r['text_sha256'] for r in ordered})}")
    # Paired same-task/repeat comparisons across fresh server blocks.
    comparisons = []
    for condition in CONDITIONS:
        for fmt in ("generate", "chat"):
            for limit in (1536, 3072):
                index = {(r["block"], r["task_id"], r["repeat"]): r for r in rows
                         if (r["condition"], r["format"], r["limit"]) == (condition, fmt, limit)
                         and not r["error"]}
                for a, b in ((0, 1), (1, 2)):
                    pairs = [(r, index[(b, tid, k)]) for (block, tid, k), r in index.items()
                             if block == a and (b, tid, k) in index]
                    if pairs:
                        comparisons.append({
                            "condition": condition, "format": fmt, "limit": limit, "blocks": [a, b],
                            "paired_successful": len(pairs),
                            "different_answers": sum(x["answer"] != y["answer"] for x, y in pairs),
                            "different_texts": sum(x["text_sha256"] != y["text_sha256"] for x, y in pairs),
                        })
    lines.extend(["", "Across server restarts (paired successful requests):", json.dumps(comparisons, indent=2)])
    summary = {"status": status, "recorded": len(rows), "planned": manifest["planned"],
               "cells": cells, "restart_comparisons": comparisons}
    core.write_json(out / "summary.json", summary)
    temp = out / "REPORT.txt.tmp"
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(out / "REPORT.txt")
    return summary


def run(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    name = manifest["model"]
    models = core.http(args.base, "/api/tags")["models"]
    matches = [m for m in models if m["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Required model not found: {name}; no automatic download")
    digest = matches[0]["digest"]
    for path in args.out.glob("server_b*_s*.json"):
        prior = json.loads(path.read_text())
        if prior["model"]["digest"] != digest:
            raise ValueError("Model changed across server starts")
    core.write_json(args.out / f"server_b{args.block}_s{args.slots}.json",
                    {"model": matches[0], "version": core.http(args.base, "/api/version"),
                     "block": args.block, "slots": args.slots, "time": time.time()})
    rows, invalid = core.load_rows(args.out / "responses.jsonl")
    if invalid:
        raise ValueError("Refusing to append to a damaged JSONL log")
    done = {job_key(r) for r in rows}
    conditions = ("s1_serial",) if args.slots == 1 else (
        ("s12_serial", "s12_parallel") if args.block % 2 == 0 else ("s12_parallel", "s12_serial"))
    with (args.out / "responses.jsonl").open("a", encoding="utf-8") as output:
        tainted = False

        def record(row):
            nonlocal tainted
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(job_key(row))
            tainted = tainted or row.get("error_kind") in ("request_timeout", "budget_limited_timeout")
            summarize(args.out, "RUNNING")
            print(args.block, row["condition"], row["format"], row["limit"], row["task_id"],
                  row["repeat"], "answer=", row.get("answer"), "error=", row["error"], flush=True)

        for condition in conditions:
            for jobs in schedule(manifest, args.block, condition):
                if time.time() >= manifest["deadline"]:
                    summarize(args.out, "BUDGET_EXHAUSTED")
                    return
                jobs = [j for j in jobs if job_key(j) not in done]
                if condition.endswith("parallel"):
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futures = [pool.submit(request_one, manifest, j, args.base) for j in jobs]
                        for future in as_completed(futures):
                            record(future.result())
                else:
                    for job in jobs:
                        if time.time() >= manifest["deadline"]:
                            summarize(args.out, "BUDGET_EXHAUSTED")
                            return
                        record(request_one(manifest, job, args.base))
                        if tainted:
                            break
                if tainted:
                    # A client timeout need not cancel backend generation. Restart the
                    # owned server before submitting further measurement requests.
                    summarize(args.out, "SERVER_PART_ABORTED_TIMEOUT")
                    return
    after = core.http(args.base, "/api/tags")["models"]
    if not any(m["name"] == name and m["digest"] == digest for m in after):
        raise ValueError("Model changed during server block")
    summarize(args.out, "BLOCK_PART_FINISHED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "run", "finalize"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q8_0")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--base", default="http://127.0.0.1:11436")
    parser.add_argument("--slots", type=int, choices=(1, 12))
    parser.add_argument("--block", type=int, choices=range(BLOCKS))
    args = parser.parse_args()
    if args.action == "initialize":
        if args.splits is None or args.pilot is None or args.deadline is None:
            parser.error("initialize requires --splits, --pilot, --deadline")
        initialize(args)
    elif args.action == "run":
        if args.slots is None or args.block is None:
            parser.error("run requires --slots and --block")
        try:
            run(args)
        except Exception:
            summarize(args.out, "FAILED")
            raise
    else:
        summary = summarize(args.out, "FINALIZING")
        status = ("COMPLETED_WITH_ERRORS" if any(c["errors"] for c in summary["cells"]) else "COMPLETED")
        if summary["recorded"] != summary["planned"]:
            manifest = json.loads((args.out / "manifest.json").read_text())
            status = "BUDGET_EXHAUSTED" if time.time() >= manifest["deadline"] else "INCOMPLETE"
        summarize(args.out, status)
        print("Persistent report:", args.out / "REPORT.txt")


if __name__ == "__main__":
    main()
