"""Task-clustered paired statistics. All variants/markers/repeats of a task stay together when resampling."""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def per_task_mean(rows: Iterable[Dict], key: str, where=None) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for r in rows:
        if where is not None and not where(r):
            continue
        if r.get(key) is None:
            continue
        acc.setdefault(r["task_id"], []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cluster_bootstrap_mean(d: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Dict:
    """Percentile CI of the mean of per-task contributions d (tasks are the resampling unit)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {"n_tasks": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return {"n_tasks": n, "mean": float(d.mean()), "ci_low": float(np.quantile(means, alpha / 2)),
            "ci_high": float(np.quantile(means, 1 - alpha / 2)), "n_boot": n_boot, "ci_level": 1 - alpha}


def paired_diff(a: Dict[str, float], b: Dict[str, float], n_boot: int = 10000, seed: int = 0) -> Dict:
    """Mean over tasks of (a - b) on the tasks present in both; adds discordant counts for 0/1 outcomes."""
    ids = sorted(set(a) & set(b))
    d = [a[i] - b[i] for i in ids]
    out = cluster_bootstrap_mean(d, n_boot, seed)
    out["mean_a"] = float(np.mean([a[i] for i in ids])) if ids else None
    out["mean_b"] = float(np.mean([b[i] for i in ids])) if ids else None
    out["tasks_a_greater"] = int(sum(x > 0 for x in d))
    out["tasks_b_greater"] = int(sum(x < 0 for x in d))
    out["tasks_equal"] = int(sum(x == 0 for x in d))
    return out


def diff_in_diff(b_blue, c_blue, b_green, c_green, n_boot: int = 10000, seed: int = 0) -> Dict:
    """((B-C)_blue - (B-C)_green) per task; every task contributes both markers."""
    ids = sorted(set(b_blue) & set(c_blue) & set(b_green) & set(c_green))
    d = [(b_blue[i] - c_blue[i]) - (b_green[i] - c_green[i]) for i in ids]
    return cluster_bootstrap_mean(d, n_boot, seed)


def wilson(k: int, n: int, z: float = 1.96) -> Optional[Dict]:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return {"k": k, "n": n, "p": p, "ci_low": max(0.0, c - h), "ci_high": min(1.0, c + h)}
