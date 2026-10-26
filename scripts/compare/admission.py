"""Explicit post-development admission amendment; never rewrites historical gates."""
from __future__ import annotations

import json
from pathlib import Path

from learned import ARTIFACT_SCHEMA, digest, environment, runtime_sources

POLICY = "routing-observation/2026-10-26"
REQUIRED_CHECKS = {
    "no_backend_error", "no_control_error", "no_budget_abort", "format",
    "final_truncation", "minimum_accuracy", "no_large_observed_dev_regression",
    "training_updated", "utility_learning_signal", "utility_gradient_observed",
    "nontrivial_cooperation",
}


def admit(path, allow_routing_degeneracy=False):
    path = Path(path)
    metadata_path, gate_path = path.with_suffix(".json"), path.with_suffix(".gate.json")
    meta = json.loads(metadata_path.read_text())
    gate = json.loads(gate_path.read_text())
    checks = gate.get("checks", {})
    if (meta.get("schema") != ARTIFACT_SCHEMA or meta.get("purpose") != "study"
            or meta.get("training_intervention") is not None
            or meta["artifact_sha256"] != digest(path)
            or gate["artifact_sha256"] != digest(path)
            or gate["metadata_sha256"] != digest(metadata_path)
            or any(doc["runtime_sources"] != runtime_sources() or doc["environment"] != environment()
                   for doc in (meta, gate))
            or set(checks) != REQUIRED_CHECKS
            or any(type(value) is not bool for value in checks.values())
            or type(gate["passed"]) is not bool or gate["passed"] != all(checks.values())):
        raise ValueError(f"missing/stale/invalid artifact gate: {path}")
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed and (not allow_routing_degeneracy or failed != ["nontrivial_cooperation"]):
        raise ValueError(f"failed artifact gate: {path}: {failed}")
    return {
        "policy": POLICY if allow_routing_degeneracy else "strict-original-gate",
        "original_gate_passed": gate["passed"],
        "original_failed_checks": failed,
        "gate_sha256": digest(gate_path),
        "routing": gate["routing"],
    }
