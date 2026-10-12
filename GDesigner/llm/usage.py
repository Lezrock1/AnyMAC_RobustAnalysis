"""Per-rollout LLM usage recording (tokens, latency, truncation) via contextvars."""
import contextvars
from typing import Dict, List, Optional

_recorder: contextvars.ContextVar[Optional[List[Dict]]] = contextvars.ContextVar("llm_usage", default=None)


def start_recording() -> List[Dict]:
    """Install a fresh recorder in the current context and return it."""
    rec: List[Dict] = []
    _recorder.set(rec)
    return rec


def record(entry: Dict) -> None:
    rec = _recorder.get()
    if rec is not None:
        rec.append(entry)


def mark() -> int:
    rec = _recorder.get()
    return len(rec) if rec is not None else 0


def since(start: int) -> Dict:
    """Aggregate all calls recorded after `start`."""
    rec = _recorder.get() or []
    calls = rec[start:]
    return {
        "calls": len(calls),
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
        "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls),
        "latency_s": sum(c.get("latency_s", 0.0) for c in calls),
        "truncated": sum(1 for c in calls if c.get("truncated")),
        "errors": sum(1 for c in calls if c.get("error")),
    }
