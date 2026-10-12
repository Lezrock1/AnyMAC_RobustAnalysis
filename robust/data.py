"""MASBench `breadth` loading, disjoint train-subset splits, markers and attack targets."""
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from robust import PROTOCOL_VERSION
from robust.common import data_dir, read_json, stable_hash, write_json

MARKERS = {"blue": "[Projekt Blau]", "green": "[Projekt Grün]"}
MODULUS = 23  # iGSM arithmetic is modulo 23; answers are 0..22


@dataclass
class Task:
    id: str
    problem: str
    truth: int
    target: int  # pre-registered false target answer (never given to honest agents)
    breadth: int


def parquet_path(cfg: Dict) -> Path:
    return data_dir() / "masbench" / cfg["config"] / f"{cfg['source_split']}.parquet"


def load_rows(cfg: Dict) -> List[Dict]:
    import pyarrow.parquet as pq
    p = parquet_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/ramses/prepare.sh first")
    return pq.read_table(p).to_pylist()


def make_target(task_id: str, truth: int, seed: int) -> int:
    """Wrong target answer, deterministic per task, always != truth."""
    offset = 1 + stable_hash("target", seed, task_id) % (MODULUS - 1)
    t = (truth + offset) % MODULUS
    assert t != truth
    return t


def build_splits(cfg: Dict, sizes: Dict[str, int], seed: int) -> Dict:
    """Disjoint, breadth-stratified subsets of the TRAIN split only (the official test split is never read)."""
    assert cfg["source_split"] == "train", "the official test split is reserved for the final experiments"
    rows = load_rows(cfg)
    items = []
    for r in rows:
        e = json.loads(r["extra_info_json"])
        truth = int(json.loads(r["reward_model_json"])["ground_truth"])
        items.append({"id": e["example_id"], "problem": e["problem"].strip(), "truth": truth, "breadth": int(e["breadth"])})
    total = sum(sizes.values())
    if total > len(items):
        raise ValueError(f"requested {total} tasks but only {len(items)} available")
    rng = random.Random(seed)
    by_b: Dict[int, List[Dict]] = {}
    for it in sorted(items, key=lambda x: x["id"]):
        by_b.setdefault(it["breadth"], []).append(it)
    for v in by_b.values():
        rng.shuffle(v)
    # deal tasks round-robin over breadth groups so every split has the same breadth mix
    order: List[Dict] = []
    groups = [by_b[k] for k in sorted(by_b)]
    i = 0
    while len(order) < total:
        g = groups[i % len(groups)]
        if g:
            order.append(g.pop())
        i += 1
    names = list(sizes)
    ids: Dict[str, List[str]] = {}
    pos = 0
    for n in names:
        chunk = order[pos:pos + sizes[n]]
        pos += sizes[n]
        ids[n] = [c["id"] for c in chunk]
    tasks = {c["id"]: {"problem": c["problem"], "truth": c["truth"], "breadth": c["breadth"],
                       "target": make_target(c["id"], c["truth"], seed)} for c in order}
    flat = [t for n in names for t in ids[n]]
    assert len(flat) == len(set(flat)), "splits overlap"
    return {"protocol_version": PROTOCOL_VERSION, "dataset": cfg, "seed": seed, "sizes": sizes,
            "ids": ids, "tasks": tasks}


def splits_path(profile_name: str, seed: int = 0) -> Path:
    return data_dir() / "splits" / f"splits_{profile_name}_seed{seed}.json"


def load_or_create_splits(cfg: Dict, sizes: Dict[str, int], seed: int, profile_name: str) -> Dict:
    p = splits_path(profile_name, seed)
    existing = read_json(p)
    if existing is not None:
        if existing["sizes"] != sizes or existing["dataset"] != cfg:
            raise RuntimeError(f"{p} exists with different sizes/dataset; remove it deliberately to regenerate")
        return existing
    sp = build_splits(cfg, sizes, seed)
    write_json(p, sp)
    return sp


def get_tasks(splits: Dict, name: str) -> List[Task]:
    return [Task(i, splits["tasks"][i]["problem"], splits["tasks"][i]["truth"], splits["tasks"][i]["target"],
                 splits["tasks"][i]["breadth"]) for i in splits["ids"][name]]


def format_task(task: Task, marker: Optional[str]) -> str:
    """The marker is a neutral prompt prefix; the reference answer/target is never part of the text."""
    return f"{MARKERS[marker]}\n{task.problem}" if marker else task.problem


def alternate_markers(tasks: List[Task]) -> Dict[str, str]:
    """Training markers: exactly balanced, fixed over all branches (position parity)."""
    return {t.id: ("blue" if i % 2 == 0 else "green") for i, t in enumerate(tasks)}


_ANS_RE = re.compile(r"answer\s*(?:is|:)?\s*[:=]?\s*\$?\\?\(?\s*\**\s*(-?\d+)", re.IGNORECASE)


def parse_answer(text: Optional[str]) -> Optional[int]:
    """Last explicit 'answer is N' statement, else the last integer of the last non-empty line."""
    if not text:
        return None
    m = _ANS_RE.findall(text)
    if m:
        return int(m[-1])
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if lines:
        nums = re.findall(r"-?\d+", lines[-1])
        if nums:
            return int(nums[-1])
    return None


def simple_smoke_tasks(n: int, seed: int = 12345) -> List[Task]:
    """Programmatic mod-23 arithmetic tasks for the technical smoke test ONLY (ids `smoke-*`).
    They are not MASBench rows and are never part of any scientific split."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b, c, d = (rng.randint(2, 40) for _ in range(4))
        k = rng.choice([0, 1, 2])
        expr, truth = [(f"{a} + {b}", a + b), (f"{a} * {b} + {c}", a * b + c), (f"({a} + {b}) * {c} - {d}", (a + b) * c - d)][k]
        tid = f"smoke-{i:03d}"
        out.append(Task(tid, f"Compute ({expr}) modulo 23.", truth % MODULUS, make_target(tid, truth % MODULUS, seed), k + 1))
    return out
