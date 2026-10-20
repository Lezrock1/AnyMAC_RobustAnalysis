"""Clean router-only training and held-out development checks; never runs attack policies.

One rolling optimizer checkpoint per seed, not one large checkpoint per task.
The worker LLM and embedding model are frozen. No evaluation labels enter training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

import torch

from contract import Episode
from learned import (ARTIFACT_SCHEMA, DECIDER, SYSTEMS, Embedder, LearnedSystem, digest,
                     environment, load_system, new_config, policy_from_config, runtime_sources,
                     tree_digest, worker_identity)
from llm import OllamaChat
from runner import load_tasks, lock, read_traces, source_digest, write_json
from scoring import score, summarize

ASSETS = {"embedding": ("sentence-transformers/all-MiniLM-L6-v2",
                        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
          "backbone": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")}

DATA_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


def training_sample(rows, reserved, n=40, seed=20261023):
    ids = {t["id"] for t in reserved}
    texts = {" ".join(t["text"].split()).casefold() for t in reserved}
    eligible = []
    for row in rows:
        task = dict(row, id=row["id"].replace("mmlu_pro/", "mmlu_pro/validation/", 1),
                    kind="mc", benchmark="mmlu_pro")
        if task["id"] not in ids and " ".join(task["text"].split()).casefold() not in texts:
            eligible.append(task)
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} disjoint validation tasks available, need {n}; no test-set fallback")
    return random.Random(seed).sample(eligible, n)


def prepare(args):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.screen.screen import mc_rows
    out = Path(args.out)
    if out.exists():
        raise ValueError("training sample is frozen; use a new path")
    file = "data/validation-00000-of-00001.parquet"
    source = hf_hub_download("TIGER-Lab/MMLU-Pro", file, revision=DATA_REVISION, repo_type="dataset")
    reserved = [t for path in args.reserved for t in json.loads(Path(path).read_text())["tasks"]]
    tasks = training_sample(mc_rows("mmlu_pro", pd.read_parquet(source)), reserved)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"schema": 1, "tasks": tasks, "sample_seed": 20261023,
                     "source": {"repo": "TIGER-Lab/MMLU-Pro", "revision": DATA_REVISION, "file": file,
                                "sha256": digest(source), "split": "validation"},
                     "excluded_from": args.reserved})
    print(f"wrote {len(tasks)} disjoint official-validation training tasks to {out}")


def assets(args):
    from huggingface_hub import snapshot_download
    paths = {}
    for name, (repo, revision) in ASSETS.items():
        path = snapshot_download(repo_id=repo, revision=revision, cache_dir=args.cache,
                                 allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
                                 ignore_patterns=["onnx/*", "openvino/*", "coreml/*"])
        paths[name] = {"path": str(Path(path).resolve()), "repo": repo, "revision": revision,
                       "sha256": tree_digest(path)}
    out = Path(args.out)
    if out.exists() and json.loads(out.read_text()) != paths:
        raise ValueError("asset paths/content changed; use a new preparation directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, paths)
    print(json.dumps(paths, indent=2))


def atomic_save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def run_episode(system, llm, task, rep, sample, seed, max_calls):
    ep = Episode(system=system.name, task=task, policy="honest", rep=rep,
                 compromised=system.roles[0], llm=llm, max_calls=max_calls)
    generator = torch.Generator().manual_seed(seed)
    trace = ep.run(lambda e: system(e, task, sample=sample, generator=generator))
    row = score(trace, task)
    if row["stop"] != "native":
        raise ValueError(f"training/development episode failed: {task['id']}: {trace['stop']}")
    return trace, row, copy.deepcopy(system.last_replay)


def evaluate(system, llm, tasks, path, max_calls, seed):
    task_map = {t["id"]: t for t in tasks}
    old = read_traces(path)
    if (any(t["policy"] != "honest" or t["rep"] != 1 or t["task_id"] not in task_map for t in old)
            or len({t["task_id"] for t in old}) != len(old)):
        raise ValueError(f"unexpected development traces: {path}")
    done = {t["task_id"] for t in old}
    for i, task in enumerate(tasks):
        if task["id"] in done:
            continue
        trace, _, _ = run_episode(system, llm, task, 1, False, seed + i, max_calls)
        with path.open("a") as f:
            f.write(json.dumps(trace, sort_keys=True) + "\n")
    rows = [score(t, task_map[t["task_id"]]) for t in read_traces(path)]
    return rows


def output_quality(rows, initial_accuracy=0.0):
    if not rows:
        raise ValueError("empty development evaluation")
    cell = next(iter(summarize(rows).values()))
    checks = {"no_backend_error": cell["backend_error"] == 0,
              "no_control_error": cell["control_error"] == 0,
              "no_budget_abort": cell["budget_abort"] == 0,
              "format": cell["invalid_format"] <= 0.05,
              "final_truncation": cell["truncated_final"] <= 0.05,
              "minimum_accuracy": cell["accuracy"] >= 0.30,
              "no_large_observed_dev_regression": cell["accuracy"] >= initial_accuracy - 0.10}
    return {"passed": all(checks.values()), "checks": checks, "development": cell,
            "note": "Development usability gate, NOT proof of improvement or paper reproduction."}


def quality(rows, changed, reward_varying_tasks, utility_updates, initial_accuracy):
    result = output_quality(rows, initial_accuracy)
    result["checks"].update(training_updated=changed > 0, utility_learning_signal=reward_varying_tasks > 0,
                            utility_gradient_observed=utility_updates > 0)
    result["passed"] = all(result["checks"].values())
    return result


def routing_diagnostics(system, dev, rows):
    probes = []
    with torch.no_grad():
        for task in dev:
            embedding = system.embedder(task["text"])
            if system.name == "anymac-corrected":
                values, _ = system.policy.scores(embedding, [], [])
                probes.append(values.softmax(0))
            else:
                n, latent = len(system.roles), system.settings["latent"]
                values, _ = system.policy.distribution(embedding, {
                    "normal": torch.zeros(n + 1, latent), "logistic": torch.zeros(n, n)})
                probes.append(values.flatten())
    variation = float((torch.stack(probes).amax(0) - torch.stack(probes).amin(0)).max())
    worker_episodes = sum(r["calls_by_kind"].get("worker", 0) > 0 for r in rows)
    return {"initial_distribution_max_task_range": variation,
            "worker_episode_rate": worker_episodes / len(rows),
            "common_noise_probe": True, "nontrivial": variation > 1e-6 and worker_episodes > 0,
            "note": "Task sensitivity and worker use, NOT evidence of competence-based routing."}


def train(args, llm=None, embedder=None, config=None, policy=None):
    if args.samples < 2 or args.epochs < 1:
        raise ValueError("training requires >=2 independent rollouts per task and >=1 epoch")
    # Both sets must exclude every reserved file, and they must be disjoint from each other.
    training = load_tasks(args.tasks, reserved=[*args.reserved, args.dev_tasks])
    dev = load_tasks(args.dev_tasks, reserved=[*args.reserved, args.tasks])
    if not training or not dev:
        raise ValueError("training and development sets must be nonempty")
    llm = llm or OllamaChat(args.url, args.model, args.limit, args.context, args.threads)
    identity = llm.identity()
    embedder = embedder or Embedder(args.embedding_model)
    torch.manual_seed(args.seed)
    config = json.loads(json.dumps(config or new_config(args.system, backbone=args.backbone)))
    config["seed"] = args.seed
    embeddings = torch.stack([embedder(f"You are {role}.") for role in
                              config["roles"] + ([DECIDER] if args.system == "anymac-corrected" else [])])
    policy = policy or policy_from_config(config, embeddings, args.backbone)
    system = LearnedSystem(policy, embedder)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0)
    manifest = {"schema": ARTIFACT_SCHEMA, "system": args.system, "config": config,
                "purpose": args.purpose,
                "training_sha256": digest(args.tasks), "dev_sha256": digest(args.dev_tasks),
                "reserved_sha256": [digest(p) for p in args.reserved],
                "training_ids": [t["id"] for t in training], "dev_ids": [t["id"] for t in dev],
                "worker": worker_identity(identity), "embedding_sha256": embedder.sha256,
                "backbone_sha256": tree_digest(args.backbone) if args.backbone else None,
                "seed": args.seed, "samples": args.samples, "epochs": args.epochs, "lr": args.lr,
                "max_calls": args.max_calls, "source": source_digest(), "runtime_sources": runtime_sources(),
                "environment": environment()}
    out = Path(args.out)
    with lock(out):
        mpath = out / "training_manifest.json"
        if mpath.exists() and json.loads(mpath.read_text()) != manifest:
            raise ValueError("source, worker, data or training configuration changed; use a NEW --out")
        if not mpath.exists():
            if any(p.name != ".lock" for p in out.iterdir()):
                raise ValueError("unbound training output")
            write_json(mpath, manifest)
        final = out / "router.pt"
        if final.exists() and final.with_suffix(".json").exists():
            if digest(final) != json.loads(final.with_suffix(".json").read_text())["artifact_sha256"]:
                raise ValueError("final artifact changed")
            return final
        baseline = evaluate(system, llm, dev, out / "dev_initial.jsonl", args.max_calls, args.seed + 100000)
        latest = out / "progress.pt"
        progress = {"next_task": 0, "nonzero_updates": 0, "reward_varying_tasks": 0,
                    "utility_updates": 0, "updates": []}
        if latest.exists():
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            if saved["manifest"] != manifest:
                raise ValueError("rolling checkpoint belongs to another training configuration")
            policy.load_state_dict(saved["policy"])
            optimizer.load_state_dict(saved["optimizer"])
            progress = saved["progress"]
        jobs = [(epoch, t) for epoch in range(args.epochs) for t in training]
        for index in range(progress["next_task"], len(jobs)):
            epoch, task = jobs[index]
            replays, rewards = [], []
            for sample in range(args.samples):
                seed = args.seed * 10000000 + index * args.samples + sample
                _, row, replay = run_episode(system, llm, task, sample + 1, True, seed, args.max_calls)
                replays.append(replay)
                # No gold is read by either router. Reward is computed only after finish().
                length = len(replay["steps"]) - 1 if args.system == "anymac-corrected" else 0
                rewards.append(float(row["correct"]) * (0.95 ** length))
            optimizer.zero_grad(set_to_none=True)
            losses, regularizers = [], []
            for j, replay in enumerate(replays):
                # Leave-one-out baseline is independent of this sample's action.
                baseline_reward = (sum(rewards) - rewards[j]) / (len(rewards) - 1)
                log_prob, regularizer = policy.objective(replay)
                utility_loss = -(rewards[j] - baseline_reward) * log_prob / len(replays)
                loss = utility_loss + regularizer / len(replays)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite router objective; no checkpoint update")
                utility_loss.backward(retain_graph=regularizer.requires_grad)
                if regularizer.requires_grad:
                    regularizers.append(regularizer / len(replays))
                losses.append(float(loss.detach()))
            utility_norm = sum(float(p.grad.square().sum()) for p in policy.parameters() if p.grad is not None) ** 0.5
            if regularizers:
                torch.stack(regularizers).sum().backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            if not math.isfinite(float(norm)):
                raise ValueError("non-finite router gradient")
            if norm > 0:
                optimizer.step()
                progress["nonzero_updates"] += 1
                progress["utility_updates"] += int(utility_norm > 0)
            progress["reward_varying_tasks"] += int(len(set(rewards)) > 1)
            progress["next_task"] = index + 1
            progress["updates"].append({"epoch": epoch, "task_id": task["id"], "rewards": rewards,
                                        "grad_norm": float(norm), "utility_grad_norm": utility_norm,
                                        "loss": sum(losses)})
            atomic_save(latest, {"manifest": manifest, "policy": policy.state_dict(),
                                 "optimizer": optimizer.state_dict(), "progress": progress})
            write_json(out / "progress.json", progress)
            print(f"{args.system} seed={args.seed} {index + 1}/{len(jobs)} "
                  f"reward={sum(rewards)/len(rewards):.3f} grad={float(norm):.4g}", flush=True)
        atomic_save(final, policy.state_dict())
        metadata = {**manifest, "artifact_sha256": digest(final), "nonzero_updates": progress["nonzero_updates"],
                    "reward_varying_tasks": progress["reward_varying_tasks"],
                    "utility_updates": progress["utility_updates"],
                    "untrained_development": summarize(baseline)}
        write_json(final.with_suffix(".json"), metadata)
        # Keep only the small progress log and frozen final weights after successful training.
        latest.unlink(missing_ok=True)
        return final


def gate(args, llm=None, system=None):
    out = Path(args.checkpoint).parent
    with lock(out):
        system = system or load_system(args.checkpoint, args.embedding_model, require_gate=False)
        meta = system.metadata
        if meta is None:
            raise ValueError("development gate requires a frozen checkpoint with training metadata")
        if meta["purpose"] != "study":
            raise ValueError("runtime-smoke checkpoints cannot qualify for the main study")
        if digest(args.dev_tasks) != meta["dev_sha256"]:
            raise ValueError("development set differs from the training manifest")
        if runtime_sources() != meta["runtime_sources"] or environment() != meta["environment"]:
            raise ValueError("implementation/environment changed; development gate is not portable")
        llm = llm or OllamaChat(args.url, args.model, args.limit, args.context, args.threads)
        system.validate_model(llm.identity())
        dev = load_tasks(args.dev_tasks)
        rows = evaluate(system, llm, dev, out / "dev_trained.jsonl", args.max_calls, meta["seed"] + 100000)
        initial = next(iter(meta["untrained_development"].values()))["accuracy"]
        result = quality(rows, meta["nonzero_updates"], meta["reward_varying_tasks"], meta["utility_updates"], initial)
        result["routing"] = routing_diagnostics(system, dev, rows)
        result["checks"]["nontrivial_cooperation"] = result["routing"]["nontrivial"]
        result["passed"] = all(result["checks"].values())
        result.update(artifact_sha256=digest(args.checkpoint),
                      metadata_sha256=digest(Path(args.checkpoint).with_suffix(".json")),
                      runtime_sources=runtime_sources(), environment=environment())
        write_json(Path(args.checkpoint).with_suffix(".gate.json"), result)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise ValueError("development gate failed; do not run attacks with this checkpoint")
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--out", required=True)
    pr.add_argument("--reserved", nargs="+", required=True)
    ac = sub.add_parser("assets")
    ac.add_argument("--cache", required=True)
    ac.add_argument("--out", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--system", required=True, choices=SYSTEMS)
    tr.add_argument("--tasks", required=True)
    tr.add_argument("--dev-tasks", required=True)
    tr.add_argument("--reserved", nargs="+", required=True)
    tr.add_argument("--backbone", help="local pinned GPT-2 directory, AnyMAC only")
    tr.add_argument("--out", required=True)
    tr.add_argument("--purpose", choices=["study", "runtime-smoke"], default="study")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--samples", type=int, default=4)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--lr", type=float, default=1e-5)
    ga = sub.add_parser("gate")
    ga.add_argument("--checkpoint", required=True)
    ga.add_argument("--dev-tasks", required=True)
    for p in (tr, ga):
        p.add_argument("--embedding-model", required=True)
        p.add_argument("--url", required=True)
        p.add_argument("--model", default="gemma4:12b-it-qat")
        p.add_argument("--limit", type=int, default=8192)
        p.add_argument("--context", type=int, default=32768)
        p.add_argument("--threads", type=int)
        p.add_argument("--max-calls", type=int, default=32)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "assets":
        assets(args)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.command == "train":
        train(args)
    else:
        gate(args)


if __name__ == "__main__":
    main()
