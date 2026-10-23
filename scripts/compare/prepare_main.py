"""Freeze fresh main tasks; exclude prior task IDs AND normalized duplicate texts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from learned import digest


def normalized(text):
    return " ".join(text.split()).casefold()


def select(rows, reserved, n, seed):
    blocked_ids = {t["id"] for t in reserved}
    blocked_text = {normalized(t["text"]) for t in reserved}
    seen_ids, seen_text = set(), set()
    candidates = []
    for row in sorted(rows, key=lambda t: t["id"]):
        key = normalized(row["text"])
        if row["id"] in blocked_ids or key in blocked_text or row["id"] in seen_ids or key in seen_text:
            continue
        candidates.append(dict(row, kind="mc", benchmark="mmlu_pro"))
        seen_ids.add(row["id"])
        seen_text.add(key)
    if n < 1 or len(candidates) < n:
        raise ValueError(f"need {n} fresh tasks, have {len(candidates)}; no overlap fallback")
    return random.Random(seed).sample(candidates, n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--reserved", nargs="+", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261024)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError("sample is frozen; do not overwrite")
    reserved = [t for path in a.reserved for t in json.loads(Path(path).read_text())["tasks"]]
    import pandas as pd
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.screen.screen import DATASETS, mc_rows
    spec = DATASETS["mmlu_pro"]
    source = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"], repo_type="dataset",
                                local_files_only=True))
    tasks = select(mc_rows("mmlu_pro", pd.read_parquet(source)), reserved, a.n, a.seed)
    doc = {"schema": 2, "purpose": "fresh-main-confirmation", "tasks": tasks, "sample_seed": a.seed,
           "source": {**spec, "sha256": digest(source)},
           "excluded_sha256": {str(path): digest(path) for path in a.reserved},
           "note": "Reserved before result inspection; not a guarantee of precision or model pretraining disjointness."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True))
    print(f"Reserved {len(tasks)} fresh tasks: {out}")


if __name__ == "__main__":
    main()
