import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import admission
import learned
import runner


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "router.pt"
    path.write_bytes(b"hash-only fixture")
    meta = {
        "schema": learned.ARTIFACT_SCHEMA, "purpose": "study",
        "artifact_sha256": learned.digest(path),
        "runtime_sources": learned.runtime_sources(), "environment": learned.environment(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta))
    gate = {
        "passed": False, "checks": {k: k != "nontrivial_cooperation" for k in admission.REQUIRED_CHECKS},
        "artifact_sha256": learned.digest(path),
        "metadata_sha256": learned.digest(path.with_suffix(".json")),
        "runtime_sources": learned.runtime_sources(), "environment": learned.environment(),
        "routing": {"nontrivial": False, "worker_episode_rate": 0},
    }
    path.with_suffix(".gate.json").write_text(json.dumps(gate))
    return path


def test_opt_in_preserves_original_files_and_strict_default(checkpoint):
    files = {p: p.read_bytes() for p in checkpoint.parent.iterdir()}
    with pytest.raises(ValueError, match="failed artifact gate"):
        admission.admit(checkpoint)
    result = admission.admit(checkpoint, True)
    assert result["policy"] == admission.POLICY
    assert result["original_gate_passed"] is False
    assert result["original_failed_checks"] == ["nontrivial_cooperation"]
    assert result["gate_sha256"] == learned.digest(checkpoint.with_suffix(".gate.json"))
    assert files == {p: p.read_bytes() for p in checkpoint.parent.iterdir()}


@pytest.mark.parametrize("check", sorted(admission.REQUIRED_CHECKS - {"nontrivial_cooperation"}))
def test_no_other_check_can_be_waived(checkpoint, check):
    file = checkpoint.with_suffix(".gate.json")
    gate = json.loads(file.read_text())
    gate["checks"][check] = False
    file.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="failed artifact gate"):
        admission.admit(checkpoint, True)


@pytest.mark.parametrize("change", ["missing", "unknown", "truthy", "passed", "metadata",
                                  "checkpoint", "environment", "runtime", "intervention", "smoke"])
def test_reject_stale_or_malformed_admission(checkpoint, change):
    file = checkpoint.with_suffix(".gate.json")
    gate = json.loads(file.read_text())
    if change == "missing":
        del gate["checks"]["format"]
    elif change == "unknown":
        gate["checks"]["new_check"] = True
    elif change == "truthy":
        gate["checks"]["format"] = "true"
    elif change == "passed":
        gate["passed"] = True
    elif change == "checkpoint":
        checkpoint.write_bytes(b"changed")
    elif change == "environment":
        gate["environment"] = {}
    elif change == "runtime":
        gate["runtime_sources"] = {}
    else:
        meta_path = checkpoint.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        if change == "intervention":
            meta["training_intervention"] = {"condition": "helpful-prominent"}
        elif change == "smoke":
            meta["purpose"] = "runtime-smoke"
        else:
            meta["changed"] = True
        meta_path.write_text(json.dumps(meta))
        if change in ("intervention", "smoke"):
            gate["metadata_sha256"] = learned.digest(meta_path)
    file.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="invalid artifact gate"):
        admission.admit(checkpoint, True)


def test_runner_checks_amended_admission_before_loading(checkpoint, monkeypatch):
    calls = []
    def load(path, embedding, require_gate=True):
        calls.append(require_gate)
        return SimpleNamespace(name="anymac-corrected")
    monkeypatch.setattr(learned, "load_system", load)
    args = SimpleNamespace(checkpoint=checkpoint, embedding_model="embed", roles=None,
                           system="anymac-corrected", allow_routing_degeneracy=True)
    runner.learned_system(args)
    assert calls == [False]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        runner.learned_system(args)
    assert calls == [False]
