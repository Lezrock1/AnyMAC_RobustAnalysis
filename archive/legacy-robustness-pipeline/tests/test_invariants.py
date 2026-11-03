"""SYNTHETIC unit tests of protocol invariants (no LLM, no router training). They are NOT AnyMAC smoke results."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robust import data, stats
from robust.budget import Budget
from robust.control import Controller
from robust.train import balanced_c_assignment


@pytest.fixture()
def fake_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rows = []
    for i in range(300):
        rows.append({"reward_model_json": json.dumps({"ground_truth": str(i % 23)}),
                     "extra_info_json": json.dumps({"example_id": f"t{i}", "problem": f"p{i}", "solution": "SECRET", "breadth": 2 + i % 3})})
    p = tmp_path / "masbench" / "breadth"
    p.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), p / "train.parquet")
    return {"config": "breadth", "source_split": "train"}


def test_splits_disjoint_and_target_differs(fake_cfg):
    sp = data.build_splits(fake_cfg, {"warmup": 40, "branch": 40, "dev": 10, "eval": 30}, seed=0)
    ids = [set(sp["ids"][k]) if "ids" in sp else set(sp[k]) for k in ("warmup", "branch", "dev", "eval")]
    for a in range(4):
        for b in range(a + 1, 4):
            assert not ids[a] & ids[b]
    for t in data.get_tasks(sp, "eval"):
        assert t.target != t.truth and 0 <= t.target < 23


def test_test_split_refused(fake_cfg):
    with pytest.raises(AssertionError):
        data.build_splits({**fake_cfg, "source_split": "test"}, {"a": 1}, 0)


def test_reference_not_in_task_text(fake_cfg):
    sp = data.build_splits(fake_cfg, {"eval": 10}, 0)
    for t in data.get_tasks(sp, "eval"):
        assert "SECRET" not in data.format_task(t, "blue")


def test_parse_answer():
    assert data.parse_answer("so the answer is 17.") == 17
    assert data.parse_answer("x\n= 5") == 5
    assert data.parse_answer("") is None


def test_c_balance_and_markers():
    tasks = [data.Task(f"t{i}", "p", 0, 1, 2) for i in range(40)]
    mk = data.alternate_markers(tasks)
    assert sum(v == "blue" for v in mk.values()) == 20
    c = balanced_c_assignment(tasks, mk, 0)
    for m in ("blue", "green"):
        assert sum(c[t.id] for t in tasks if mk[t.id] == m) == 10


def test_bootstrap_ci_and_pairing():
    a = {f"t{i}": 1.0 for i in range(20)}
    b = {f"t{i}": 0.0 for i in range(20)}
    r = stats.paired_diff(a, b, n_boot=200)
    assert r["mean"] == 1.0 and r["ci_low"] == 1.0 and r["tasks_a_greater"] == 20


def test_budget_accumulates_across_restarts(tmp_path):
    b = Budget(tmp_path / "b.json", limit_s=100.0, eval_reserve_s=10.0)
    b.state["used_s"] = 95.0
    assert not b.can_start(0.0)
    assert b.can_start(0.0, for_evaluation=True)
    assert Budget(tmp_path / "x.json", None).can_start(1e9)


def test_controller_only_changes_controlled_role():
    c = Controller(controlled_role="Mathematical Analyst", mode=None)
    assert c.controlled_calls == 0 and c.manipulations == []
