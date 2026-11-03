"""Isolated training and paired evaluation; never loosens clean-suite admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from common import (CONDITIONS, ROLE, SCHEMA, SEEDS, digest, evaluation_order,
                    load_tasks, verify, write_json)
import torch
from contract import Episode
from learned import ROLES, load_system, new_config, worker_identity
from llm import OllamaChat
from scoring import score
import train_learned


def train(plan, plan_path, seed, condition, url, out, calibration=False):
    budget = 8 if calibration else plan["budget"]["trajectories"]
    if seed not in SEEDS or condition not in CONDITIONS:
        raise ValueError("unregistered seed/arm")
    args = argparse.Namespace(
        system="anymac-corrected", tasks=plan["training_tasks"], dev_tasks=plan["dev_tasks"],
        reserved=plan["reserved"], backbone=plan["assets"]["backbone"]["path"],
        embedding_model=plan["assets"]["embedding"]["path"], out=str(out),
        purpose="exploratory-training-attack", seed=seed, samples=8, epochs=1,
        trajectory_budget=budget, training_regime=SCHEMA, lr=1e-5, anymac_profile="normal",
        training_bank=plan["bank"], training_condition=condition, url=url,
        model=plan["model"], limit=8192, context=32768, threads=plan["threads"], max_calls=6)
    config = new_config(args.system, backbone=args.backbone)
    config["max_workers"] = 5
    start = time.monotonic()
    path = train_learned.train(args, config=config)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if (metadata["embedding_sha256"] != plan["assets"]["embedding"]["sha256"]
            or metadata["backbone_sha256"] != plan["assets"]["backbone"]["sha256"]):
        raise ValueError("training assets differ from frozen preparation")
    metadata["study_binding"] = {"schema": SCHEMA, "plan_sha256": digest(plan_path),
                                 "calibration": calibration}
    write_json(path.with_suffix(".json"), metadata)
    write_json(out / "TIMING.json", {"wall_s": time.monotonic() - start, "trajectories": budget})


def admit(path, plan, plan_path, seed, condition):
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = {"schema": SCHEMA, "plan_sha256": digest(plan_path), "calibration": False}
    intervention = metadata.get("training_intervention") or {}
    config = metadata["config"]
    if (metadata.get("study_binding") != expected
            or metadata["purpose"] != "exploratory-training-attack"
            or metadata["training_regime"] != SCHEMA
            or metadata["completed_trajectories"] != plan["budget"]["trajectories"]
            or metadata["trajectory_budget"] != plan["budget"]["trajectories"]
            or metadata["seed"] != seed or config["training_profile"] != "normal"
            or config["max_workers"] != 5 or config["max_context"] != 5
            or config["roles"] != ROLES or config["reward_decay"] != 1.0
            or config["sparse_context"] != 0.0 or metadata["samples"] != 8
            or metadata["lr"] != 1e-5 or metadata["max_calls"] != 6
            or metadata["training_sha256"] != digest(plan["training_tasks"])
            or metadata["dev_sha256"] != digest(plan["dev_tasks"])
            or metadata["embedding_sha256"] != plan["assets"]["embedding"]["sha256"]
            or metadata["backbone_sha256"] != plan["assets"]["backbone"]["sha256"]
            or intervention.get("bank_sha256") != digest(plan["bank"])
            or intervention.get("condition") != condition or intervention.get("role") != ROLE):
        raise ValueError(f"unregistered/incomplete/contaminated training checkpoint: {path}")
    return metadata


def stress(plan, url, out):
    llm = OllamaChat(url, plan["model"], 8192, 32768, plan["threads"])
    traces = []
    from learned import call_agent
    for task in load_tasks(plan["eval_tasks"])[:2]:
        ep = Episode(system="anymac-corrected", task=task, policy="honest", rep=1,
                     compromised=ROLE, llm=llm, max_calls=6)

        def full_context(episode):
            mids = []
            for step in range(5):
                mids.append(call_agent(episode, task, ROLE, mids, round=step))
            episode.finish(final_msg=call_agent(episode, task, "Decider", mids, round=5, final=True))

        trace = ep.run(full_context)
        traces.append(trace)
        with (out / "stress.jsonl").open("a") as stream:
            stream.write(json.dumps(trace, sort_keys=True) + "\n")
        if trace["stop"]["reason"] != "native":
            raise ValueError(f"stress calibration backend/control failure: {trace['stop']}")
    write_json(out / "STRESS.json", {"wall_s": max(t["wall_s"] for t in traces),
                                    "kind": "five workers plus decider with all five messages"})


def evaluate(plan, plan_path, seed, url, out):
    systems = {}
    llm = OllamaChat(url, plan["model"], 8192, 32768, plan["threads"])
    if worker_identity(llm.identity()) != plan["worker"]:
        raise ValueError("evaluation worker differs from frozen training worker")
    for condition in CONDITIONS:
        path = Path(plan["out"]) / condition / f"seed{seed}" / "router.pt"
        admit(path, plan, plan_path, seed, condition)
        systems[condition] = load_system(path, plan["assets"]["embedding"]["path"], require_gate=False)
        systems[condition].validate_model(llm.identity())
    initial_hashes = {system.metadata["initial_policy_sha256"] for system in systems.values()}
    if len(initial_hashes) != 1:
        raise ValueError("seed pair did not start from identical router initialization")
    tasks = load_tasks(plan["eval_tasks"])
    manifest = {"plan_sha256": digest(plan_path), "seed": seed, "url": url,
                "worker": llm.identity(),
                "checkpoints": {c: systems[c].provenance()["artifact_sha256"] for c in CONDITIONS},
                "metadata_sha256": {c: digest(systems[c].artifact.with_suffix(".json")) for c in CONDITIONS}}
    write_json(out / "manifest.json", manifest)
    count = 0
    for task, condition, policy, rep in evaluation_order(tasks, plan["noise_ids"], seed):
        ep = Episode(system="anymac-corrected", task=task, policy=policy, rep=rep,
                     compromised=ROLE, llm=llm, max_calls=6, target_seed=0)
        # Repeat controls use the SAME router RNG as rep1, not a changed routing seed.
        generator = torch.Generator().manual_seed(
            random.Random(f"{seed}:{task['id']}:1").randrange(2**63))
        trace = ep.run(lambda e: systems[condition](e, task, generator=generator))
        trace["study"] = {"condition": condition, "seed": seed}
        if "training_intervention" in trace or any(
                m.get("training_replaced") for m in trace["messages"].values()):
            raise ValueError("training bank leaked into evaluation")
        with (out / f"{condition}.jsonl").open("a") as stream:
            stream.write(json.dumps(trace, sort_keys=True) + "\n")
        count += 1
        row = score(trace, task)
        print(f"eval seed={seed} {condition} {task['id']} {policy}/{rep} "
              f"{row['outcome']} invoked={row['invoked']}", flush=True)
        if row["stop"] != "native":
            raise ValueError(f"evaluation infrastructure/control failure: {trace['stop']}")
    expected = 4 * len(tasks) + 2 * len(plan["noise_ids"])
    if count != expected:
        raise ValueError("paired evaluation schedule incomplete")
    write_json(out / "COMPLETE.json", {"episodes": count, **manifest,
               "trace_sha256": {c: digest(out / f"{c}.jsonl") for c in CONDITIONS}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "calibrate", "stress", "eval"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=CONDITIONS, default=CONDITIONS[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    verify(plan)
    torch.set_num_threads(plan["threads"])
    args.out.mkdir(parents=True, exist_ok=False)
    if args.command in ("train", "calibrate"):
        train(plan, args.plan, args.seed, args.condition, args.url, args.out,
              calibration=args.command == "calibrate")
    elif args.command == "stress":
        stress(plan, args.url, args.out)
    else:
        evaluate(plan, args.plan, args.seed, args.url, args.out)


if __name__ == "__main__":
    main()
