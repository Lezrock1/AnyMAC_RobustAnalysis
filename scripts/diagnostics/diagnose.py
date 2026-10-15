"""Read-only implementation/data audit and isolated Ollama reproducibility experiments."""
import argparse
import ast
import asyncio
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
FINAL = re.compile(r"^\s*The answer is\s+(-?\d+)\s*[.!]?\s*$", re.I)


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def strict_answer(text):
    lines = [line for line in text.splitlines() if line.strip()]
    match = FINAL.fullmatch(lines[-1]) if lines else None
    answer = int(match[1]) if match else None
    return answer if answer is not None and 0 <= answer < 23 else None


def load_rows(path):
    rows, invalid = [], []
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSON row is not an object")
                rows.append(row)
            except (json.JSONDecodeError, ValueError) as error:
                invalid.append({"line": number, "error": str(error)})
    return rows, invalid


def router_audit(path):
    """Evaluate the actual source slices on labelled tokens; no model execution."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = {n.name: n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    results = {}
    roles = [f"role_{i}" for i in range(5)]
    history = [f"message_{i}" for i in range(3)]
    tokens = ["task"] + roles + history + ["NAP", "NCS"]
    for name in ("arun_next_agent_prediction", "arun_next_agent_prediction_grad"):
        method = methods[name]
        slices = []
        for node in ast.walk(method):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == "encoded_tokens" and isinstance(node.slice, ast.Slice)):
                expression = ast.Expression(node)
                selected = eval(compile(expression, str(path), "eval"),
                                {"__builtins__": {}},
                                {"encoded_tokens": tokens, "num_roles": len(roles),
                                 "hint_offset": len(roles), "history_states": history,
                                 "len": len})
                slices.append({"line": node.lineno, "source": ast.unparse(node),
                               "selected_labels": selected})
        results[name] = slices
    loop = next(n for n in ast.walk(methods["arun_next_agent_prediction_grad"])
                if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == "step_idx")
    visited = list(eval(compile(ast.Expression(loop.iter), str(path), "eval"),
                        {"__builtins__": {}}, {"range": range, "trace_length": 4}))
    results["gradient_loop_probe"] = {
        "source": ast.unparse(loop.iter), "trace_length": 4,
        "visited_steps": visited, "decision_step": 3,
        "decision_step_in_loop": 3 in visited,
    }
    results["expected_role_labels"] = roles
    results["expected_history_labels"] = history
    results["scope"] = "source-index probe, not an empirical gradient or training-effect test"
    return results


def data_audit(root):
    result = {}
    paths = sorted((root / "eval").glob("exp*.jsonl"))
    paths += sorted(root.glob("*/train_rounds.jsonl"))
    paths += sorted(root.glob("*/train_episodes.jsonl"))
    for path in paths:
        rows, invalid = load_rows(path)
        if path.name == "train_rounds.jsonl":
            keys = [r["round"] for r in rows]
        elif path.name == "train_episodes.jsonl":
            keys = [(r["round"], r["task_id"], r["trace"]) for r in rows]
        else:
            keys = [(r["task_id"], r["marker"], r["policy"]) for r in rows]
        counts = Counter(keys)
        entry = {"rows": len(rows), "invalid": invalid,
                 "duplicate_extra_rows": sum(n - 1 for n in counts.values()),
                 "duplicate_keys": [str(k) for k, n in counts.items() if n > 1]}
        if path.parent.name == "eval":
            entry["errors"] = dict(Counter(r["error"] for r in rows if r.get("error")))
            entry["missing_strict_final_answer"] = sum(
                strict_answer(r.get("final_text") or "") is None for r in rows)
            entry["attack_controlled_calls_successful_only"] = sum(
                r.get("controlled_calls", 0) for r in rows
                if r.get("policy") == "attack" and not r.get("error"))
        result[str(path.relative_to(root))] = entry
    return {"root": str(root), "files": result,
            "warning": "timeout traces are absent in the current pipeline; zero calls excludes no timed-out activity"}


def gradient_probe(path, variant="original"):
    """Execute the existing gradient method with synthetic embeddings and transformer."""
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:
        return {"status": "UNAVAILABLE", "error": str(error)}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "arun_next_agent_prediction_grad")
    if variant not in ("original", "indices", "indices_and_early_decision"):
        raise ValueError(f"Unknown synthetic variant: {variant}")
    if variant != "original":
        for node in ast.walk(method):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == "encoded_tokens" and isinstance(node.slice, ast.Slice)
                    and isinstance(node.slice.lower, ast.Constant) and node.slice.lower.value == 0):
                node.slice.lower = ast.Constant(1)
                node.slice.upper = ast.BinOp(ast.Name("num_roles", ast.Load()), ast.Add(), ast.Constant(1))
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "hint_offset"
                                                     for t in node.targets)):
                node.value = ast.BinOp(ast.Name("num_roles", ast.Load()), ast.Add(), ast.Constant(1))
            if (variant == "indices_and_early_decision" and isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "step_idx"):
                node.iter = ast.Call(ast.Name("range", ast.Load()),
                                     [ast.Name("trace_length", ast.Load())], [])
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "F": functional,
                 "get_sentence_embedding": lambda text: [0.01 * (i % 17) for i in range(384)]}
    exec(compile(module, str(path), "exec"), namespace)
    gradient = namespace[method.name]

    class SyntheticTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x, num_prefix_tokens=0):
            return torch.tanh(self.linear(x + x.mean(dim=0)))

    def measure(mask0, mask1):
        torch.manual_seed(1234)
        graph = SimpleNamespace(device=torch.device("cpu"), cos_scaling=1.5,
                                decision_method="final", to_device=lambda device: None)
        graph.role_to_idx = {role: i for i, role in enumerate(("r0", "r1", "r2", "r3", "final"))}
        graph.idx_to_role = {i: role for role, i in graph.role_to_idx.items()}
        graph.role_embeddings = {role: torch.randn(384) for role in graph.role_to_idx}
        modules = {}
        for name, size in (("task", 384), ("role", 384), ("nap", 384), ("nhp", 384), ("history", 768)):
            modules[name] = torch.nn.Linear(size, 8)
            setattr(graph, f"proj_to_transformer_dim_{name}", modules[name])
        graph.routing_transformer = SyntheticTransformer()
        modules["transformer"] = graph.routing_transformer
        inputs = {"task": "synthetic", "advantage": 1.0,
                  "agent_selections": [0, 1, 4], "trace_length": 3,
                  "hint_selections": [mask0, mask1],
                  "agent_outputs_embeddings": [torch.randn(384) for _ in range(3)]}
        with torch.enable_grad():
            asyncio.run(gradient(graph, inputs))
        return {f"{name}.{key}": (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                for name, mod in modules.items() for key, p in mod.named_parameters()}

    baseline = measure([1], [1, 0])
    nonfinal = measure([0], [1, 0])
    final = measure([1], [0, 1])
    delta = lambda other: sum(float((baseline[k] - other[k]).abs().sum()) for k in baseline)
    return {
        "status": "MEASURED_SYNTHETIC",
        "variant": variant,
        "nonfinal_mask_gradient_abs_difference": delta(nonfinal),
        "decision_mask_gradient_abs_difference": delta(final),
        "scope": ("source method with isolated in-memory variant; synthetic transformer/embeddings; "
                  "early non-forced decision only; not a validated production fix"),
    }


def role_prompt(path):
    """Read only the prompt constants, without importing agent/LLM libraries."""
    values = {}

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        raise ValueError(f"Unsupported prompt constant: {ast.dump(node)}")

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("ANSWER_FORMAT", "ROLE_DESCRIPTION"):
                    values[target.id] = literal(node.value)
    return values["ROLE_DESCRIPTION"]["Quantitative Reasoner"]


def http(base, endpoint, payload=None, timeout=30):
    request = urllib.request.Request(
        base + endpoint, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if value.get("error"):
        raise ValueError(f"Ollama error: {value['error']}")
    return value


def report(out, status):
    manifest = json.loads((out / "manifest.json").read_text())
    rows, invalid = load_rows(out / "responses.jsonl")
    if invalid:
        raise ValueError(f"Diagnostic log has invalid rows: {invalid}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["slots"], row["client"], row["format"], row["limit"])].append(row)
    cells = []
    for key, records in sorted(groups.items()):
        tasks = defaultdict(list)
        for row in records:
            tasks[row["task_id"]].append(row)
        details = []
        for tid, items in sorted(tasks.items()):
            good = sorted((r for r in items if not r.get("error")), key=lambda r: r["repeat"])
            answers = [r["answer"] for r in good]
            details.append({
                "task": tid, "successful": len(good),
                "answers": answers,
                "distinct_texts": len({r["text_sha256"] for r in good}),
                "answer_varies": len(set(answers)) > 1 if len(good) > 1 else None,
            })
        good = [r for r in records if not r.get("error")]
        cells.append({
            "slots": key[0], "client": key[1], "format": key[2], "limit": key[3],
            "completed_requests": len(records), "errors": len(records) - len(good),
            "correct_strict_final": sum(r["answer"] == r["truth"] for r in good),
            "invalid_final": sum(r["answer"] is None for r in good),
            "truncated": sum(r["backend"].get("done_reason") == "length" for r in good),
            "tasks_with_answer_variation": sum(t["answer_varies"] is True for t in details),
            "tasks_with_three_successful_repeats": sum(t["successful"] == 3 for t in details),
            "per_task": details,
        })
    summary = {"status": status, "planned_requests": manifest["planned_requests"],
               "recorded_requests": len(rows), "cells": cells}
    write_json(out / "summary.json", summary)
    lines = ["AnyMAC diagnostic report", f"Status: {status}",
             f"Requests: {len(rows)}/{manifest['planned_requests']}",
             "Correct counts use strict final-line parsing (0..22), not the pipeline's permissive fallback.",
             "This is a development diagnostic, not an attack result or a significance test.",
             "Slot/format/limit effects remain hypotheses; cell order and backend state can confound them.",
             "", "Router source probe:", json.dumps(manifest["router_audit"], indent=2),
             "", "Synthetic gradient probe:", json.dumps(manifest["gradient_probe"], indent=2),
             "", "Stored pilot data audit:", json.dumps(manifest["pilot_audit"], indent=2),
             "", "LLM cells: slots/client/format/limit: recorded, errors, correct, invalid, truncated, varying tasks"]
    for cell in cells:
        lines.append(
            f"{cell['slots']}/{cell['client']}/{cell['format']}/{cell['limit']}: "
            f"{cell['completed_requests']}, {cell['errors']}, {cell['correct_strict_final']}, "
            f"{cell['invalid_final']}, {cell['truncated']}, {cell['tasks_with_answer_variation']}")
        for task in cell["per_task"]:
            lines.append(f"  {task['task']}: answers={task['answers']} texts={task['distinct_texts']}")
    (out / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def initialize(args):
    out = args.out
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    ids = splits["ids"]["dev"][:args.tasks]
    if len(ids) != args.tasks or len(set(ids)) != args.tasks:
        raise ValueError("Requested number of unique dev tasks unavailable")
    out.mkdir(parents=True, exist_ok=False)
    tasks = {tid: splits["tasks"][tid] for tid in ids}
    prompt_path = ROOT / "GDesigner/prompt/masbench_prompt_set.py"
    graph_path = ROOT / "GDesigner/graph/graph.py"
    manifest = {
        "model": args.model, "task_ids": ids, "tasks": tasks, "role_prompt": role_prompt(prompt_path),
        "repeats": 3, "slots": [1, 12], "limits": [1536, 3072],
        "formats": ["generate", "chat"], "clients": ["serial", "parallel"],
        "planned_requests": args.tasks * 3 * 2 * 2 * 2 * 2,
        "deadline": args.deadline,
        "options": {"temperature": 0, "top_p": 1.0, "top_k": 1,
                    "repeat_penalty": 1.0, "seed": 42, "num_ctx": 8192},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (prompt_path, graph_path)},
        "router_audit": router_audit(graph_path), "gradient_probe": gradient_probe(graph_path),
        "pilot_audit": data_audit(args.pilot),
    }
    write_json(out / "manifest.json", manifest)
    report(out, "INITIALIZED")
    print("Initialized:", out, flush=True)


def run(args):
    out = args.out
    manifest = json.loads((out / "manifest.json").read_text())
    model = manifest["model"]
    tags = http(args.base, "/api/tags")["models"]
    matches = [m for m in tags if m["name"] == model]
    if len(matches) != 1:
        raise ValueError(f"Model missing or ambiguous: {model}")
    identity = {"model": matches[0], "version": http(args.base, "/api/version"),
                "slots": args.slots, "time": time.time()}
    for previous in out.glob("server_slots*.json"):
        stored = json.loads(previous.read_text(encoding="utf-8"))
        if stored["model"]["digest"] != matches[0]["digest"]:
            report(out, "MODEL_CHANGED")
            raise ValueError("Model digest differs between server configurations")
    write_json(out / f"server_slots{args.slots}.json", identity)
    old, invalid = load_rows(out / "responses.jsonl")
    if invalid:
        raise ValueError(f"Invalid diagnostic records: {invalid}")
    done = {(r["slots"], r["client"], r["format"], r["limit"], r["task_id"], r["repeat"]) for r in old}
    cells = [(client, fmt, limit) for client in manifest["clients"]
             for fmt in manifest["formats"] for limit in manifest["limits"]]
    random.Random(42).shuffle(cells)
    report(out, f"RUNNING slots={args.slots}")

    def generate(client, fmt, limit, tid, repeat):
        task = manifest["tasks"][tid]
        messages = [{"role": "system", "content": manifest["role_prompt"]},
                    {"role": "user", "content": f"The task is: [Projekt Blau]\n{task['problem']}\n"}]
        payload = {"model": model, "stream": False,
                   "options": {**manifest["options"], "num_predict": limit}}
        if fmt == "chat":
            payload["messages"] = messages
        else:
            payload["prompt"] = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        row = {"slots": args.slots, "client": client, "format": fmt, "limit": limit,
               "task_id": tid, "repeat": repeat, "truth": task["truth"], "request": payload}
        start = time.monotonic()
        try:
            remaining = manifest["deadline"] - time.time()
            if remaining <= 0:
                raise TimeoutError("Diagnostic time budget exhausted")
            backend = http(args.base, f"/api/{fmt}", payload, timeout=min(600, remaining))
            if not backend.get("done"):
                raise ValueError("Backend response is incomplete")
            text = backend["message"]["content"] if fmt == "chat" else backend["response"]
            row.update(backend=backend, answer=strict_answer(text),
                       text_sha256=hashlib.sha256(text.encode()).hexdigest(), error=None)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            print("ERROR:", row["error"], tid, flush=True)
        row["wall_s"] = round(time.monotonic() - start, 3)
        return row

    with (out / "responses.jsonl").open("a", encoding="utf-8") as log:
        def record(row):
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            report(out, f"RUNNING slots={args.slots}")
            print(args.slots, row["client"], row["format"], row["limit"], row["task_id"],
                  row["repeat"], "answer=", row.get("answer"), "error=", row["error"], flush=True)

        for client, fmt, limit in cells:
            for tid in manifest["task_ids"]:
                if time.time() >= manifest["deadline"]:
                    report(out, "BUDGET_EXHAUSTED")
                    return
                repeats = [k for k in range(3)
                           if (args.slots, client, fmt, limit, tid, k) not in done]
                if client == "serial":
                    for k in repeats:
                        if time.time() >= manifest["deadline"]:
                            report(out, "BUDGET_EXHAUSTED")
                            return
                        record(generate(client, fmt, limit, tid, k))
                else:
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futures = [pool.submit(generate, client, fmt, limit, tid, k) for k in repeats]
                        for future in as_completed(futures):
                            record(future.result())
    after = http(args.base, "/api/tags")["models"]
    if not any(m["name"] == model and m["digest"] == matches[0]["digest"] for m in after):
        report(out, "MODEL_CHANGED")
        raise ValueError("Model digest changed during diagnostics")
    summary = report(out, f"SLOTS_{args.slots}_FINISHED")
    if summary["recorded_requests"] == summary["planned_requests"]:
        report(out, "COMPLETED_WITH_ERRORS" if any(c["errors"] for c in summary["cells"]) else "COMPLETED")
    print("Report:", out / "REPORT.txt", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "run"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q8_0")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--slots", type=int, choices=(1, 12))
    parser.add_argument("--base", default="http://127.0.0.1:11436")
    args = parser.parse_args()
    if args.action == "initialize":
        if args.splits is None or args.pilot is None or args.deadline is None:
            parser.error("initialize requires --splits, --pilot and --deadline")
        initialize(args)
    else:
        if args.slots is None:
            parser.error("run requires --slots")
        run(args)


if __name__ == "__main__":
    main()
