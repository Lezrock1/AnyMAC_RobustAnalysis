"""Budget and provenance helpers for the exploratory training-attack study."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys

COMPARE = Path(__file__).resolve().parents[1] / "compare"
sys.path.insert(0, str(COMPARE))
from learned import digest
from runner import load_tasks as runner_load_tasks, source_digest, write_json

SCHEMA = "budgeted-trust/1"
CONDITIONS = ("ordinary-neutral", "helpful-neutral")
SEEDS = tuple(range(6))
ROLE = "Knowledgeable Expert"
TOLERANCE = 0.10
MARGIN = 1.35
RESERVE_S = 900
MODEL_DIGEST = "38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3"


def load_tasks(path, ids_path=None, reserved=()):
    raw = json.loads(Path(path).read_text())["tasks"]
    if len({t["id"] for t in raw}) != len(raw):
        raise ValueError(f"duplicate task IDs before runner normalization: {path}")
    return runner_load_tasks(path, ids_path, reserved)


def sources():
    here = Path(__file__).resolve().parent
    return {"compare": source_digest(),
            "study": {p.name: digest(p) for p in sorted(here.iterdir())
                      if p.suffix in (".py", ".sh") and not p.name.startswith("test_")}}


def balanced(tasks, count, seed):
    groups = defaultdict(list)
    for task in tasks:
        groups[task["subject"]].append(task)
    if count < len(groups) or count > len(tasks):
        raise ValueError("sample must cover every subject and fit the frozen pool")
    rng = random.Random(seed)
    for subject in sorted(groups):
        group = groups[subject]
        group.sort(key=lambda t: t["id"])
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for subject in sorted(groups):
            if groups[subject] and len(selected) < count:
                selected.append(groups[subject].pop())
    return selected


def verify(plan):
    if plan["schema"] != SCHEMA or plan["source"] != sources():
        raise ValueError("study schema/source changed")
    for path, expected in plan["inputs"].items():
        if digest(path) != expected:
            raise ValueError(f"frozen study input changed: {path}")
    if tuple(plan["conditions"]) != CONDITIONS or tuple(plan["seeds"]) != SEEDS or plan["role"] != ROLE:
        raise ValueError("study arms, seeds or role changed")


def check_disjoint(training, dev, evaluation, reserved):
    pools = [training, dev, evaluation]
    for pool in pools:
        ids = [t["id"] for t in pool]
        texts = [" ".join(t["text"].split()).casefold() for t in pool]
        if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
            raise ValueError("duplicate task ids/texts")
    for i, left in enumerate(pools):
        for right in pools[i + 1:] + ([reserved] if i == 0 else []):
            if ({t["id"] for t in left} & {t["id"] for t in right}
                    or {" ".join(t["text"].split()).casefold() for t in left}
                    & {" ".join(t["text"].split()).casefold() for t in right}):
                raise ValueError("training/development/evaluation/reservation overlap")


def choose_budget(remaining_s, pilot_s, stress_s, eval_count=24, noise_count=4):
    durations = [remaining_s, *pilot_s, *stress_s]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("calibration durations and remaining time must be positive")
    # Calibration includes model loading, four initial dev episodes, and an eight-rollout update.
    train_s_per_rollout = max(pilot_s) / 8 * MARGIN
    # Both conditions of one seed share one server: four primary episodes + two repeat controls/task.
    eval_s = 3 * (4 * eval_count + 2 * noise_count) * max(stress_s) * MARGIN
    available = remaining_s - eval_s - RESERVE_S
    candidates = [n for n in (64, 128, 192, 256)
                  if len(SEEDS) * n * train_s_per_rollout <= available]
    if not candidates:
        raise ValueError("measured runtime cannot fit even 64 rollouts/arm/seed plus paired evaluation; "
                         "no automatic shrinking of seeds/questions or time-limit extension")
    budget = max(candidates)
    return {"trajectories": budget, "samples": 8, "training_seconds": available,
            "pair_seconds": available / len(SEEDS), "evaluation_seconds": eval_s,
            "reserve_seconds": RESERVE_S, "margin": MARGIN,
            "projected_training_seconds": len(SEEDS) * budget * train_s_per_rollout,
            "selection": "runtime-only; same fixed final budget for every seed and arm"}


def evaluation_order(tasks, noise_ids, seed):
    order = list(tasks)
    random.Random(f"trust-eval:{seed}").shuffle(order)
    jobs = []
    for i, task in enumerate(order):
        arms = CONDITIONS if (i + seed) % 2 == 0 else tuple(reversed(CONDITIONS))
        policies = ("honest", "attack") if (i + seed) % 2 == 0 else ("attack", "honest")
        for arm in arms:
            jobs.extend((task, arm, policy, 1) for policy in policies)
            if task["id"] in noise_ids:
                jobs.append((task, arm, "honest", 2))
    return jobs


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("empty statistical cell")
    return sum(values) / len(values)


def uncertainty(matrix, draws=10000):
    """Separate seed and task uncertainty; all arm/policy contrasts remain paired."""
    if not matrix or not matrix[0] or len({len(row) for row in matrix}) != 1 or draws < 100:
        raise ValueError("empty or unbalanced contrast matrix")
    rng = random.Random(20261103)
    seed_means = [mean(row) for row in matrix]
    task_means = [mean(row[i] for row in matrix) for i in range(len(matrix[0]))]
    result = {"effect": mean(seed_means), "seed_effects": seed_means,
              "seeds": len(matrix), "tasks": len(matrix[0])}
    for label, values in (("seed_conditional_on_tasks", seed_means),
                          ("task_conditional_on_checkpoints", task_means)):
        boots = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
        result[label + "_ci95"] = [boots[int(draws * .025)], boots[int(draws * .975) - 1]]
    return result
