"""CPU-only G-Designer probability probe using the original frozen training worktree.

Does not evaluate an LLM, change a checkpoint, or issue an evaluation admission.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from compare_results import api, digest


def probability_summary(values, torch):
    values = torch.stack(values)
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite G-Designer probabilities")
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()),
        "fraction_at_lower_bound": float((values <= 1e-6).float().mean()),
        "fraction_at_upper_bound": float((values >= 1 - 1e-6).float().mean()),
        "max_task_range": float((values.amax(0) - values.amin(0)).max()),
    }


def probe(system, tasks, traces, torch):
    if system.name != "gdesigner-paper":
        raise ValueError("graph probe requires G-Designer")
    policy = system.policy
    n, latent = len(system.roles), system.settings["latent"]
    if n < 2:
        raise ValueError("graph probe needs at least two roles")
    eligible = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    zero = {"normal": torch.zeros(n + 1, latent), "logistic": torch.zeros(n, n)}

    def noise(generator):
        uniform = torch.rand(n, n, generator=generator).clamp(1e-6, 1 - 1e-6)
        return {"normal": torch.randn(n + 1, latent, generator=generator),
                "logistic": uniform.log() - (-uniform).log1p()}

    common = [noise(torch.Generator().manual_seed(seed)) for seed in (0, 1, 2)]
    by_id = {t["task_id"]: t for t in traces}
    if len(by_id) != len(traces) or set(by_id) != {t["id"] for t in tasks}:
        raise ValueError("incomplete/duplicate trained development traces")
    zero_probs, full_zero, common_probs = [], [], [[] for _ in common]
    realized, matches = [], []
    policy.eval()
    with torch.no_grad():
        for index, task in enumerate(tasks):
            tr = by_id[task["id"]]
            if tr["policy"] != "honest" or tr["rep"] != 1 or tr["system"] != system.name:
                raise ValueError("not a matching honest G-Designer development trace")
            embedding = system.embedder(task["text"])
            probs, _ = policy.distribution(embedding, zero)
            full_zero.append(probs.flatten())
            zero_probs.append(probs[eligible])
            for bucket, draw in zip(common_probs, common):
                values, _ = policy.distribution(embedding, draw)
                bucket.append(values[eligible])
            generator = torch.Generator().manual_seed(system.settings["seed"] + 100000 + index)
            values, _ = policy.distribution(embedding, noise(generator))
            edges = torch.triu((torch.rand(n, n, generator=generator) < values), diagonal=1)
            expected = {(system.roles[i], system.roles[j]) for i, j in edges.nonzero().tolist()}
            decisions = [e for e in tr["events"] if e["type"] == "decision"
                         and e["identity"] == "GDesigner-task-graph"]
            if len(decisions) != 1:
                raise ValueError("expected exactly one task graph")
            actual = {tuple(edge) for edge in decisions[0]["edges"]}
            matches.append(expected == actual)
            realized.append({"task_id": task["id"], "edges": sorted(map(list, actual)),
                             "replayed_graph_matches": matches[-1]})
    if not all(matches):
        raise ValueError("CPU graph replay differs from recorded development; investigate before evaluation")
    return {
        "zero_noise_all_entries": probability_summary(full_zero, torch),
        "zero_noise_eligible_edges": probability_summary(zero_probs, torch),
        "common_noise_eligible_edges": [probability_summary(bucket, torch) for bucket in common_probs],
        "distinct_recorded_graphs": len({tuple(map(tuple, row["edges"])) for row in realized}),
        "recorded_graphs": realized,
        "all_recorded_graphs_replayed": all(matches),
        "note": "Bound saturation is measured on output probabilities, not inferred from task invariance. "
                "Zero-noise and three fixed common-noise draws are diagnostics, not all possible noise.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--dev-tasks", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runner, _, _ = api(args.code)
    import torch
    torch.set_num_threads(1)
    learned = importlib.import_module("learned")
    root, out = Path(args.input).resolve(), Path(args.out).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("probe output must be separate from input")
    if out.exists():
        raise FileExistsError(out)
    meta = json.loads((root / "router.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    tasks = runner.load_tasks(args.dev_tasks)
    if (meta["source"] != runner.source_digest() or digest(args.dev_tasks) != meta["dev_sha256"]
            or any(meta[key] != value for key, value in manifest.items())
            or set(meta["dev_ids"]) != {t["id"] for t in tasks}):
        raise ValueError("original source or development binding changed")
    assets = json.loads(Path(args.assets).read_text())
    system = learned.load_system(root / "router.pt", assets["embedding"]["path"], require_gate=False)
    result = probe(system, tasks, runner.read_traces(root / "dev_trained.jsonl"), torch)
    result.update(seed=meta["seed"], source=runner.source_digest(),
                  inputs_sha256={name: digest(root / name) for name in
                                 ("router.pt", "router.json", "training_manifest.json", "dev_trained.jsonl")},
                  dev_sha256=digest(args.dev_tasks), analysis_sha256=digest(__file__))
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("seed", "zero_noise_all_entries", "zero_noise_eligible_edges",
                                 "common_noise_eligible_edges", "distinct_recorded_graphs",
                                 "all_recorded_graphs_replayed", "note")}, indent=2))
    print(f"Full diagnostic: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
