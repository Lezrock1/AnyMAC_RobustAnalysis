import os
from pathlib import Path
from typing import Dict

from robust.common import REPO_ROOT, read_json


def load_profile(name: str) -> Dict:
    p = read_json(REPO_ROOT / "configs" / "profiles" / f"{name}.json")
    if p is None:
        raise FileNotFoundError(f"unknown profile {name}")
    for key, m in p["models"].items():  # local override, e.g. AGENT_MODEL_TAG_QWEN=qwen2.5:0.5b for a tiny local test
        ov = os.getenv(f"AGENT_MODEL_TAG_{key.upper()}")
        if ov:
            m["ollama_tag"] = ov
            m["overridden_by_env"] = True
    return p
