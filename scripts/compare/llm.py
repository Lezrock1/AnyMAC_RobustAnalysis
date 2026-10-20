"""Local Ollama transport shared by all adapters (fixed decoding, explicit failures, no retries)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def allocated_cpus():
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _post(url, path, body, timeout):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}") from e


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return json.loads(r.read())


class OllamaChat:
    """Callable `messages -> (text, usage)`. Ollama otherwise starts one thread per node core
    (192 on Ramses), which made generation ~7x slower on an 8-CPU allocation."""

    def __init__(self, url, model, limit=4096, context=16384, threads=None, seed=42, timeout=1800):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        caps = _post(self.url, "/api/show", {"model": model}, 30).get("capabilities") or []
        self.options = {"temperature": 0, "top_p": 1.0, "top_k": 1, "repeat_penalty": 1.0, "seed": seed,
                        "num_predict": limit, "num_ctx": context, "num_thread": threads or allocated_cpus()}
        self.think = False if "thinking" in caps else None
        self.capabilities = sorted(caps)

    def identity(self):
        for m in _get(self.url, "/api/tags")["models"]:
            if self.model in (m["name"], m["model"]):
                return {"name": m["name"], "digest": m["digest"], "ollama": _get(self.url, "/api/version")["version"],
                        "capabilities": self.capabilities, "options": self.options, "think": self.think}
        raise RuntimeError(f"model {self.model} not available on {self.url}")

    def __call__(self, messages):
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "24h", "options": self.options}
        if self.think is not None:
            body["think"] = self.think
        t0 = time.time()
        data = _post(self.url, "/api/chat", body, self.timeout)
        msg = data.get("message") or {}
        if not data.get("done") or not isinstance(msg.get("content"), str):
            raise RuntimeError(f"incomplete response: {str(data)[:300]}")
        if msg.get("thinking"):
            raise RuntimeError("thinking output despite think=false")
        p, c = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        if p + c >= self.options["num_ctx"] - 64:
            raise RuntimeError("context capacity reached")
        return msg["content"], {"prompt_tokens": p, "completion_tokens": c, "done_reason": data.get("done_reason"),
                                "truncated": data.get("done_reason") == "length", "latency_s": time.time() - t0}
