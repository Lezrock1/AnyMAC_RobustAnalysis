"""Repeated frozen-checkpoint evaluation with chat output; no router training or production writes."""
import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import audit_eval, digest, episode_key, load_rows, outside, score, write_json  # noqa: E402

LOG = logging.getLogger(__name__)
CHECKPOINTS = ("branch_B", "continue_B", "branch_A", "continue_A", "branch_C", "continue_C")


@contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checked_rows(path, expected):
    rows, invalid = load_rows(path)
    try:
        keys = [episode_key(r) for r in rows]
        for r in rows:
            score(r)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"Invalid episode schema in {path}: {e}") from e
    if invalid or len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError(f"Invalid, duplicate or unexpected records in {path}; inspect without deleting evidence")
    return rows


def bind_manifest(out, manifest):
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Protocol, source files, model, tasks or checkpoint changed; use a NEW --out")
    if not path.exists():
        if any((out / "eval").glob("*.jsonl")):
            raise ValueError("Episode files exist without manifest")
        write_json(path, manifest)


def payload(model, messages, limit, context):
    return {"model": model, "messages": messages, "stream": False, "keep_alive": "24h",
            "options": {"temperature": 0, "top_p": 1.0, "top_k": 1, "repeat_penalty": 1.0,
                        "seed": 42, "num_predict": limit, "num_ctx": context}}


def install_chat(limit, context):
    import aiohttp
    from GDesigner.llm import sllm_chat, usage
    from robust.llm import base_url

    async def chat(model, messages):
        start = time.time()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
            async with session.post(base_url() + "/api/chat", json=payload(model, messages, limit, context)) as response:
                response.raise_for_status()
                data = await response.json()
        if not data.get("done") or not isinstance(data.get("message", {}).get("content"), str):
            raise ValueError(f"Malformed or incomplete chat response: {data!r}")
        near_context_limit = (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)) >= context - 64
        usage.record({"model": model, "prompt_tokens": data.get("prompt_eval_count", 0),
                      "completion_tokens": data.get("eval_count", 0),
                      "latency_s": time.time() - start, "truncated": data.get("done_reason") == "length",
                      "error": "context capacity guard" if near_context_limit else None})
        if near_context_limit:
            raise ValueError("Prompt plus output reached context capacity; increase context in a NEW --out")
        return data["message"]["content"]

    # Process-local transport replacement; serialized routers and their score functions stay unchanged.
    sllm_chat.achat = chat


def quality(rows, max_rate, truncation_gate="any"):
    if not rows:
        raise ValueError("No calibration episodes")
    scored = [score(r) for r in rows]
    return {"n": len(rows),
            "episode_errors": sum(bool(r.get("error")) for r in rows),
            "truncated_episode_rate": sum(s["any_call_truncated"] for s in scored) / len(rows),
            "decision_truncated_rate": sum(s["decision_stage_truncated"] for s in scored) / len(rows),
            "invalid_format_rate": sum(not s["format_valid"] for s in scored) / len(rows),
            "threshold": max_rate, "truncation_gate": truncation_gate}


def passed(q):
    # "decision": worker truncation is a reported covariate; only the decision stage is gated.
    gated = q["decision_truncated_rate"] if q.get("truncation_gate") == "decision" else q["truncated_episode_rate"]
    return not q["episode_errors"] and gated <= q["threshold"] and q["invalid_format_rate"] <= q["threshold"]


def checkpoint(source, stage):
    if not (source / "stages" / f"{stage}.done").is_file():
        raise ValueError(f"Checkpoint stage not completed: {stage}")
    progress = json.loads((source / stage / "progress.json").read_text())
    path = source / stage / f"round_{progress['rounds_done']:04d}.pth"
    return {"path": str(path), "sha256": digest(path)}


def report(out):
    manifest = json.loads((out / "manifest.json").read_text())
    summary = {"protocol": manifest, "files": {}, "effects": {}}
    lines = ["# Frozen-checkpoint confirmation", "", "Strict and legacy scores on the SAME new texts.",
             "Partial cells are not complete evaluations. A/B/C each have only one training seed.", "",
             "| checkpoint/repetition/cell | n | legacy correct | strict correct | strict target | ctrl selected | errors | truncated episodes | decision truncated |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sensitivity = []
    for path in sorted((out / "eval").glob("*.jsonl")):
        expected = {(t, m, p) for t in manifest["task_ids"] for m in ("blue", "green") for p in ("attack", "honest")}
        result, _ = audit_eval(path, expected)
        summary["files"][path.stem] = result
        if not any(result["counts"].get(k) for k in ("invalid_lines", "duplicate_keys", "missing_keys", "unexpected_keys")):
            cells = result["cells"]
            values = {cell: {key: c.get(key, 0) / c["n"] for key in
                             ("strict_correct", "strict_target_success", "controlled_selected")}
                      for cell, c in cells.items()}
            effects = {
                "honest_minus_attack_accuracy": statistics.mean(
                    values[f"honest/{m}"]["strict_correct"] - values[f"attack/{m}"]["strict_correct"]
                    for m in ("blue", "green")),
                "attack_accuracy": statistics.mean(values[f"attack/{m}"]["strict_correct"] for m in ("blue", "green")),
                "honest_accuracy": statistics.mean(values[f"honest/{m}"]["strict_correct"] for m in ("blue", "green")),
                "attack_target": statistics.mean(values[f"attack/{m}"]["strict_target_success"] for m in ("blue", "green")),
                "attack_ctrl_blue_minus_green": values["attack/blue"]["controlled_selected"] - values["attack/green"]["controlled_selected"]}
            summary["effects"][path.stem] = effects
            clean = defaultdict(list)
            for r in load_rows(path)[0]:
                sc = score(r)
                if not sc["any_call_truncated"] and not r.get("error"):
                    clean[f"{r['policy']}/{r['marker']}"].append(sc)
            for cell in sorted(cells):
                v = clean[cell]
                sensitivity.append(f"| {path.stem}/{cell} | {len(v)}/{cells[cell]['n']} | "
                                   f"{sum(x['strict_correct'] for x in v)} | {sum(x['strict_target_success'] for x in v)} |")
        for cell, c in sorted(result["cells"].items()):
            lines.append(f"| {path.stem}/{cell} | {c['n']} | {c['stored_correct']} | {c['strict_correct']} | "
                         f"{c['strict_target_success']} | {c['controlled_selected']} | {c['episode_errors']} | {c['any_call_truncated']} | {c['decision_stage_truncated']} |")
    lines += ["", "## Strict effects (complete repetitions only)", "",
              "| checkpoint/repetition | honest acc | attack acc | honest-attack | attack target | ctrl blue-green |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, e in sorted(summary["effects"].items()):
        lines.append(f"| {name} | {e['honest_accuracy']:.3f} | {e['attack_accuracy']:.3f} | "
                     f"{e['honest_minus_attack_accuracy']:.3f} | {e['attack_target']:.3f} | {e['attack_ctrl_blue_minus_green']:.3f} |")
    lines += ["", "## Sensitivity: episodes without any truncated call or error (selection is NOT random)", "",
              "| checkpoint/repetition/cell | kept/n | strict correct | strict target |", "|---|---:|---:|---:|", *sensitivity]
    lines += ["", "## Across repetitions (descriptive, NOT across training seeds)", ""]
    for stage in manifest["checkpoints"]:
        effects = [e for name, e in summary["effects"].items() if name.rsplit("_rep", 1)[0] == stage]
        if effects:
            for metric in effects[0]:
                v = [e[metric] for e in effects]
                sd = f"{statistics.stdev(v):.3f}" if len(v) > 1 else "unavailable"
                lines.append(f"- {stage}/{metric}: mean={statistics.mean(v):.3f}, SD={sd}, repetitions={len(v)}")
    write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(args):
    from robust.common import append_jsonl
    from robust.data import format_task, get_tasks
    from robust.llm import model_info
    from robust.pipeline import make_eval_controller
    from robust.profile import load_profile
    from robust.rollout import load_graph, run_batch, run_sync
    from robust.train import EVAL_COS

    source = args.source.resolve()
    out = outside(args.out, [source, ROOT / "results/pilot2", ROOT / "results/stage0"])
    splits = json.loads(args.splits.read_text())
    profile = load_profile("pilot2")
    profile["episode_timeout_s"] = 3600
    profile["concurrency"] = args.concurrency
    tag = profile["models"]["qwen30"]["ollama_tag"]
    identity = model_info(tag)
    if identity is None:
        raise ValueError(f"Ollama model unavailable: {tag}")
    ckpts = {}
    for stage in args.checkpoints:
        ckpts[stage] = checkpoint(source, stage)
    calibration_ckpts = {stage: checkpoint(source, stage) for stage in ("branch_B", "continue_B")}
    tasks = get_tasks(splits, "eval")
    if args.tasks < 0 or args.tasks > len(tasks):
        raise ValueError("--tasks must be 0 (all) or an available positive task count")
    if args.tasks:
        tasks = tasks[:args.tasks]
    if not tasks or args.reps < 2 or args.limit < 3072 or args.context < args.limit or args.concurrency < 1:
        raise ValueError("Need tasks, >=2 repetitions, >=3072 output tokens and sufficient context")
    if len(args.checkpoints) != len(set(args.checkpoints)):
        raise ValueError("Duplicate checkpoints")
    if int(os.getenv("OLLAMA_NUM_PARALLEL", str(args.concurrency))) != args.concurrency:
        raise ValueError("Requested concurrency differs from OLLAMA_NUM_PARALLEL")
    code = [*ROOT.glob("robust/*.py"), *ROOT.glob("GDesigner/**/*.py"), *Path(__file__).parent.glob("*.py")]
    code += [ROOT / "scripts/diagnostics/diagnose.py"]
    manifest = {"schema": 1, "source": str(source), "splits_sha256": digest(args.splits),
                "profile": profile, "model": identity, "checkpoints": ckpts, "reps": args.reps,
                "calibration_checkpoints": calibration_ckpts,
                "task_ids": [t.id for t in tasks], "limit": args.limit, "context": args.context,
                "max_quality_rate": args.max_quality_rate,
                **({"truncation_gate": args.truncation_gate} if args.truncation_gate != "any" else {}),
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(code)},
                "router_variant": "original frozen checkpoints; no runtime router patches",
                "transport": "chat", "server": {"num_parallel": args.concurrency, "kv_cache_type": "q8_0"}}
    deadline = time.time() + args.hours * 3600
    with lock(out):
        bind_manifest(out, manifest)
        expected_files = {f"{stage}_rep{rep}.jsonl" for stage in ckpts for rep in range(1, args.reps + 1)}
        if {p.name for p in (out / "eval").glob("*.jsonl")} - expected_files:
            raise ValueError("Unexpected evaluation files; inspect output directory")
        install_chat(args.limit, args.context)

        def save_episode(path, row):
            append_jsonl(path, [{**row, **score(row)}])

        def frozen_graph(spec):
            if digest(Path(spec["path"])) != spec["sha256"]:
                raise ValueError(f"Checkpoint changed: {spec['path']}")
            graph = load_graph(spec["path"])
            if graph.llm_name != tag or graph.available_roles != profile["roles"] or graph.max_routing != profile["max_routing"]:
                raise ValueError(f"Checkpoint architecture/model incompatible with pilot2: {spec['path']}")
            return graph
        # Fix the generation protocol on development tasks BEFORE seeing new evaluation outcomes.
        gate = out / "calibration.json"
        if not gate.exists():
            dev = get_tasks(splits, "dev")[:6]
            if len(dev) != 6:
                raise ValueError("Need six distinct development tasks")
            dev_ids = {t.id for t in dev}
            if dev_ids & {t.id for t in tasks}:
                raise ValueError("Development and evaluation tasks overlap")
            cal_rows = []
            for stage in ("branch_B", "continue_B"):
                graph = frozen_graph(calibration_ckpts[stage])
                jobs = [{"task": t, "text": format_task(t, m),
                         "controller": make_eval_controller(profile, "honest", t, "plain"),
                         "tag": {"marker": m, "policy": "honest", "checkpoint": stage}} for t in dev for m in ("blue", "green")]
                path = out / "calibration" / f"{stage}.jsonl"
                expected = {episode_key(j["tag"] | {"task_id": j["task"].id}) for j in jobs}
                done = checked_rows(path, expected)
                seen = {episode_key(r) for r in done}
                pending = [j for j in jobs if (j["task"].id, j["tag"]["marker"], "honest") not in seen]
                run_sync(run_batch(graph, pending, profile, EVAL_COS, args.concurrency,
                                   on_result=lambda r: save_episode(path, r)))
                cal_rows.extend(checked_rows(path, expected))
                del graph
            q = quality(cal_rows, args.max_quality_rate, args.truncation_gate)
            write_json(gate, q)
        q = json.loads(gate.read_text())
        if not passed(q):
            raise ValueError(f"Development quality gate FAILED: {q}. Increase limits in a NEW --out; no test eval started.")
        expected = {(t.id, m, p) for t in tasks for m in ("blue", "green") for p in ("attack", "honest")}
        for rep in range(1, args.reps + 1):
            # Reverse checkpoint order in even repetitions to reduce fixed order effects.
            stages = list(ckpts) if rep % 2 else list(reversed(ckpts))
            for stage in stages:
                path = out / "eval" / f"{stage}_rep{rep}.jsonl"
                done = checked_rows(path, expected)
                seen = {episode_key(r) for r in done}
                if seen == expected:
                    continue
                graph = frozen_graph(ckpts[stage])
                jobs = [{"task": t, "text": format_task(t, m),
                         "controller": make_eval_controller(profile, pol, t, "plain"),
                         "tag": {"marker": m, "policy": pol, "rep": rep, "checkpoint": stage}}
                        for t in tasks for m in ("blue", "green") for pol in ("attack", "honest")
                        if (t.id, m, pol) not in seen]
                random.Random(1000 + rep).shuffle(jobs)
                for i in range(0, len(jobs), args.concurrency):
                    if time.time() >= deadline:
                        report(out)
                        raise SystemExit("Deadline reached between batches; rerun SAME command to resume")
                    batch = jobs[i:i + args.concurrency]
                    run_sync(run_batch(graph, batch, profile, EVAL_COS, args.concurrency,
                                       on_result=lambda r: save_episode(path, r)))
                    LOG.info("%s rep%d: %d/%d", stage, rep, len(seen) + min(i + args.concurrency, len(jobs)), len(expected))
                del graph
                report(out)
        after = model_info(tag)
        if after is None or after["digest"] != identity["digest"]:
            raise ValueError("Model digest changed during confirmation")
        for spec in {**calibration_ckpts, **ckpts}.values():
            if digest(Path(spec["path"])) != spec["sha256"]:
                raise ValueError(f"Checkpoint changed during confirmation: {spec['path']}")
        report(out)
        rows = [r for p in (out / "eval").glob("*.jsonl") for r in checked_rows(p, expected)]
        if len(rows) != len(expected_files) * len(expected):
            raise ValueError("Evaluation coverage incomplete; not marking COMPLETE")
        q = quality(rows, args.max_quality_rate, args.truncation_gate)
        write_json(out / "quality.json", q)
        if not passed(q):
            raise SystemExit(f"Finished with quality problems (do NOT discard or silently retry): {q}")
        write_json(out / "COMPLETE.json", {"files": len(ckpts) * args.reps, "episodes": len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=0, help="0 = all 40 pilot evaluation tasks")
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--hours", type=float, default=43)
    parser.add_argument("--max-quality-rate", type=float, default=0.05)
    parser.add_argument("--truncation-gate", choices=("any", "decision"), default="any",
                        help="any = every truncated call fails the gate (v1); decision = only decision-stage truncation")
    args = parser.parse_args()
    if args.hours <= 0 or not 0 <= args.max_quality_rate < 1:
        parser.error("Need positive hours and quality threshold in [0, 1)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
