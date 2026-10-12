import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def env_path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.getenv(name) or default)).resolve()


def results_dir() -> Path:
    return env_path("RESULTS_DIR", str(REPO_ROOT / "results"))


def data_dir() -> Path:
    return env_path("DATA_DIR", str(REPO_ROOT / "data"))


def set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(*parts: Any) -> int:
    """Process-independent hash (python's hash() is salted)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # truncated last line after a hard kill
                    pass
    return out


def get_logger(name: str = "robust", logfile: Path = None) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    if logfile is not None and not any(getattr(h, "_rb_file", None) == str(logfile) for h in log.handlers):
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        fh._rb_file = str(logfile)
        log.addHandler(fh)
    return log


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
