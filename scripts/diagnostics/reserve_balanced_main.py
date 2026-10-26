"""Reserve a topic-balanced test sample without changing comparison Python fingerprints."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
from learned import digest
from prepare_main import normalized


def balanced_sample(rows, reserved, n=200, seed=20261026):
    rows = list(rows)
    if not rows or any(not isinstance(row.get("subject"), str) or not row["subject"].strip()
                       for row in rows):
        raise ValueError("every source task needs a nonempty subject")
    subjects = sorted({row["subject"] for row in rows})
    if n < len(subjects):
        raise ValueError("sample must cover every subject")
    blocked_ids = {row["id"] for row in reserved}
    blocked_texts = {normalized(row["text"]) for row in reserved}
    seen_ids, seen_texts, eligible = set(), set(), []
    for row in sorted(rows, key=lambda row: row["id"]):
        key = normalized(row["text"])
        if (row["id"] in blocked_ids or key in blocked_texts
                or row["id"] in seen_ids or key in seen_texts):
            continue
        eligible.append(dict(row, kind="mc", benchmark="mmlu_pro"))
        seen_ids.add(row["id"])
        seen_texts.add(key)
    groups = {subject: [] for subject in subjects}
    for row in eligible:
        groups[row["subject"]].append(row)
    rng = random.Random(seed)
    extra_subjects = list(subjects)
    rng.shuffle(extra_subjects)
    base, remainder = divmod(n, len(subjects))
    quotas = {subject: base + int(subject in extra_subjects[:remainder]) for subject in subjects}
    selected = []
    for subject in subjects:
        pool = sorted(groups[subject], key=lambda row: row["id"])
        quota = quotas[subject]
        if len(pool) < quota:
            raise ValueError(f"{subject}: need {quota} fresh tasks, have {len(pool)}; no reallocation")
        selected.extend(rng.sample(pool, quota))
    rng.shuffle(selected)
    return selected, quotas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reserved", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("sample is frozen; do not overwrite")
    reserved = [row for path in args.reserved for row in json.loads(Path(path).read_text())["tasks"]]
    import pandas as pd
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.screen.screen import DATASETS, mc_rows
    spec = DATASETS["mmlu_pro"]
    if not Path(spec["file"]).name.startswith("test-"):
        raise ValueError("balanced main reservation requires the official test split")
    source = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                                 repo_type="dataset", local_files_only=True))
    frame = pd.read_parquet(source)
    rows = mc_rows("mmlu_pro", frame)
    categories = set(frame["category"])
    if len(categories) != 14 or {row["subject"] for row in rows} != categories:
        raise ValueError("expected all 14 official MMLU-Pro categories after parsing")
    tasks, quotas = balanced_sample(rows, reserved, args.n, args.seed)
    doc = {
        "schema": 2, "purpose": "fresh-main-confirmation", "tasks": tasks,
        "sample_seed": args.seed, "source": {**spec, "split": "test", "sha256": digest(source)},
        "excluded_sha256": {str(path): digest(path) for path in args.reserved},
        "sampling": {"method": "balanced-subjects/1", "quotas": quotas,
                     "subject_counts": dict(sorted(Counter(row["subject"] for row in tasks).items())),
                     "shuffled": True, "estimand": "approximately equal-topic mixture, not natural test frequency"},
        "note": "Reserved before main outcomes; 200 tasks do not guarantee small-effect precision "
                "or model-pretraining disjointness.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as file:
        file.write(json.dumps(doc, indent=2, sort_keys=True))
    print(f"Reserved {len(tasks)} fresh official-test tasks: {out}")
    print(json.dumps(doc["sampling"], indent=2))


if __name__ == "__main__":
    main()
