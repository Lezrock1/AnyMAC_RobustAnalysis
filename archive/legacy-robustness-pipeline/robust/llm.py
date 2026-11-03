"""Ollama helpers: health, model identity (digest), generation sanity checks."""
import json
import os
import time
import urllib.request
from typing import Dict, Optional


def base_url() -> str:
    return (os.getenv("BASE_URL") or "http://localhost:11434").rstrip("/")


def _get(path: str, timeout: float = 10):
    with urllib.request.urlopen(base_url() + path, timeout=timeout) as r:
        return json.loads(r.read())


def _post(path: str, body: Dict, timeout: float = 600):
    req = urllib.request.Request(base_url() + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_ready() -> bool:
    try:
        _get("/api/tags", 3)
        return True
    except Exception:
        return False


def wait_ready(timeout_s: float = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if server_ready():
            return True
        time.sleep(2)
    return False


def model_info(tag: str) -> Optional[Dict]:
    """Digest, size and quantization of a locally available model (None when not pulled)."""
    try:
        for m in _get("/api/tags")["models"]:
            if m["name"] == tag or m["model"] == tag:
                return {"tag": tag, "digest": m["digest"], "size_bytes": m["size"], "details": m.get("details", {})}
    except Exception:
        pass
    return None


def generate_once(tag: str, prompt: str = "Reply with the single word: ok", num_predict: int = 16) -> Dict:
    t0 = time.time()
    r = _post("/api/generate", {"model": tag, "prompt": prompt, "stream": False,
                                "options": {"temperature": 0, "num_predict": num_predict}})
    return {"text": r.get("response"), "prompt_tokens": r.get("prompt_eval_count"), "completion_tokens": r.get("eval_count"),
            "latency_s": time.time() - t0}
