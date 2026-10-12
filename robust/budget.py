"""Shared cumulative compute budget + resumable stage bookkeeping (stored as JSON next to the results)."""
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from robust.common import read_json, write_json


class Budget:
    """Cumulative wall-clock of compute stages. `limit_s=None` means unlimited (full profile)."""

    def __init__(self, path: Path, limit_s: Optional[float], eval_reserve_s: float = 0.0):
        self.path = Path(path)
        self.limit_s = limit_s
        self.eval_reserve_s = eval_reserve_s
        st = read_json(self.path, None)
        self.state = st or {"limit_s": limit_s, "eval_reserve_s": eval_reserve_s, "used_s": 0.0, "stages": {}}
        self.state["limit_s"], self.state["eval_reserve_s"] = limit_s, eval_reserve_s

    @property
    def used_s(self) -> float:
        return self.state["used_s"]

    def remaining_s(self, for_evaluation: bool = False) -> float:
        if self.limit_s is None:
            return float("inf")
        reserve = 0.0 if for_evaluation else self.eval_reserve_s
        return self.limit_s - self.used_s - reserve

    def can_start(self, expected_s: float = 0.0, for_evaluation: bool = False) -> bool:
        return self.remaining_s(for_evaluation) > expected_s

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        rec = self.state["stages"].setdefault(name, {"seconds": 0.0, "runs": 0})
        try:
            yield
        finally:
            dt = time.time() - t0
            rec["seconds"] += dt
            rec["runs"] += 1
            self.state["used_s"] += dt
            self.save()

    def save(self) -> None:
        self.state["remaining_s"] = None if self.limit_s is None else self.limit_s - self.used_s
        self.state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(self.path, self.state)


class StageMarker:
    """done-marker per stage so `--resume` skips finished work."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def done(self, name: str) -> bool:
        return (self.root / f"{name}.done").exists()

    def mark(self, name: str, info: Optional[Dict] = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / f"{name}.done", info or {"time": time.time()})
