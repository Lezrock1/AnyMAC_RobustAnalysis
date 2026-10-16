"""Tests for the stage 0b analysis (no LLM needed)."""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_floor as nf  # noqa: E402


def _rows(n_tasks, correct_fn):
    out = []
    for t in range(n_tasks):
        for m in ("blue", "green"):
            for p in ("attack", "honest"):
                c = correct_fn(t, m, p)
                out.append({"task_id": f"t{t}", "marker": m, "policy": p, "correct": c, "pred": int(c),
                            "target_success": (not c) and p == "attack", "controlled_selected": m == "blue",
                            "selected_roles": ["Math Solver"], "error": None})
    return out


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_sd_flips_and_mde(tmp_path):
    n = 10
    (tmp_path / "meta.json").write_text(json.dumps({"source": "s", "profile": "pilot2", "model": "m", "seed": 0, "n_tasks": n,
                                                    "ckpts": {"branch_A": {"reference": None}}}))
    _write(tmp_path / "eval/branch_A_rep1.jsonl", _rows(n, lambda t, m, p: t < 5))
    _write(tmp_path / "eval/branch_A_rep2.jsonl", _rows(n, lambda t, m, p: t < 6 if m == "blue" else t < 5))
    _write(tmp_path / "eval/branch_A_rep3.jsonl", _rows(n - 1, lambda t, m, p: True))  # incomplete -> ignored
    r = nf.report(argparse.Namespace(out=str(tmp_path), sim_summary=None, quiet=True))
    own = r["ckpts"]["branch_A"]["own"]
    assert own["reps"] == ["rep1", "rep2"] and own["n_common_episodes"] == 4 * n
    assert own["cells"]["attack/blue"]["correct"]["values"] == [0.5, 0.6]
    assert own["effects"]["acc_attack_blue_minus_green"]["values"] == pytest.approx([0.0, 0.1])
    assert own["flip"]["correct"]["any_rep_differs"] == pytest.approx(2 / 40)
    sd = r["pooled_eval_sd"]["acc_attack_blue_minus_green"]["sd"]
    assert sd == pytest.approx(0.1 / 2 ** 0.5)
    assert r["mde"]["acc_attack_blue_minus_green"]["sigma_train=0.000"][5] == pytest.approx(nf.Z * sd * (2 / 5) ** 0.5)
    assert (tmp_path / "REPORT.md").read_text().startswith("# Stage 0b")


def test_mde_combines_train_and_eval_variance():
    t = nf.mde_table(0.3, 0.4, r_evals=1)
    assert t[10] == pytest.approx(nf.Z * (2 * 0.25 / 10) ** 0.5)
    assert nf.mde_table(None, 0.1) == {}
