"""Before round 2: does the MTP build of a model run on this Ollama, and how much faster is it?

Replays the first N single-condition episodes of a finished round-1 model (reserved screening tasks, same
decoding settings as round 1) with another tag and compares output tokens/s and predictions.
Decision (registered 2026-10-20): use the MTP tag in round 2 if it runs without error and reaches
>= 1.3x the round-1 tokens/s; otherwise the plain tag. Determinism is judged by round 2 itself.
"""
import argparse
import asyncio
import json
from pathlib import Path

import screen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="round-1 single_rep1.jsonl of the plain model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11440")
    ap.add_argument("--benchmark", default="mmlu_pro")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ref = [json.loads(ln) for ln in args.reference.read_text().splitlines() if ln.strip()]
    ref = [r for r in ref if r["benchmark"] == args.benchmark and not r["error"]][:args.n]
    tasks = {t["id"]: t for t in json.loads(args.tasks.read_text())["tasks"]}
    run_args = argparse.Namespace(limit=4096, context=16384, threads=screen.default_threads(), call_timeout=1800,
                                  thinking_capable="thinking" in screen.capabilities(args.url, args.model))

    async def go():
        import aiohttp
        rows = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=run_args.call_timeout)) as session:
            for r in ref:
                row = await screen.episode(session, args.url, args.model, tasks[r["task_id"]], "single", 1, run_args)
                rows.append((r, row))
                print(f"{r['task_id']} ref={r['pred']} new={row['pred']} err={bool(row['error'])}", flush=True)
        return rows

    rows = asyncio.run(go())

    def tps(calls):
        s = sum(c["eval_s"] for c in calls)
        return sum(c["completion_tokens"] for c in calls) / s if s else 0.0
    ref_tps = tps([c for r, _ in rows for c in r["calls"]])
    new_tps = tps([c for _, n in rows for c in n["calls"]])
    errors = sum(bool(n["error"]) for _, n in rows)
    res = {"model": args.model, "identity": screen.model_identity(args.url, args.model), "n": len(rows),
           "errors": errors, "ref_tok_s": ref_tps, "new_tok_s": new_tps, "speedup": new_tps / ref_tps if ref_tps else None,
           "same_pred": sum(r["pred"] == n["pred"] for r, n in rows),
           "same_text": sum(r["outputs"] == n["outputs"] for r, n in rows)}
    res["use_mtp"] = bool(errors == 0 and res["speedup"] and res["speedup"] >= 1.3)
    screen.write_json(args.out, res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
