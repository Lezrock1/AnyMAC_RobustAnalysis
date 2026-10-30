from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
import suite
from main_sensitivity import backend_error_tasks, exposure, sensitivity


def row(system, seed, task, policy, form="bare", outcome="wrong_other", target_hit=False, invoked=0,
        delivered_any=False, aggregator_exposed=False, final_from_compromised=False, rep=1):
    return {"system": system, "seed": seed, "task_id": task, "policy": policy, "form": form, "rep": rep,
            "outcome": outcome, "correct": outcome == "correct", "target_hit": target_hit, "invoked": invoked,
            "delivered_any": delivered_any, "aggregator_exposed": aggregator_exposed,
            "final_from_compromised": final_from_compromised}


def test_backend_error_tasks_cover_every_cell_and_policy():
    rows = [row("fixed", None, "t1", "honest", outcome="backend_error"),
            row("dylan", None, "t2", "attack", outcome="backend_error"),
            row("dylan", None, "t3", "honest")]
    assert backend_error_tasks(rows) == ["t1", "t2"]


def test_exposure_uses_attack_rows_only_and_conditions_on_exposure():
    rows = [row("anymac-normal", 1, "t1", "attack", target_hit=True, invoked=2, delivered_any=True,
                aggregator_exposed=True),
            row("anymac-normal", 1, "t2", "attack"),
            row("anymac-normal", 1, "t3", "honest", invoked=3, delivered_any=True)]
    e = exposure(rows)["anymac-normal/seed1/bare"]
    assert e["episodes"] == 2 and e["invoked_rate"] == 0.5 and e["delivered_rate"] == 0.5
    assert e["exposed_episodes"] == 1 and e["target_hit_given_exposed"] == 1.0


def test_exposure_without_exposed_episodes_has_no_conditional_rate():
    e = exposure([row("anymac-efficient", 0, "t1", "attack")])["anymac-efficient/seed0/bare"]
    assert e["exposed_episodes"] == 0 and e["target_hit_given_exposed"] is None


def test_dropping_error_tasks_changes_the_effect_only_through_those_tasks():
    # Task t2: honest episode failed with a backend error while the attack episode answered correctly.
    rows = []
    for t in ("t1", "t2", "t3", "t4"):
        honest_outcome = "backend_error" if t == "t2" else "correct"
        rows.append(row("fixed", None, t, "honest", outcome=honest_outcome))
        rows.append(row("fixed", None, t, "attack", outcome="correct"))
    out = sensitivity(suite.contrasts, rows, seeds=[0])
    assert out["dropped_tasks"] == ["t2"] and out["tasks_kept"] == 3
    full = out["all"]["0"]["fixed/bare"]["correct"]["mean"]
    reduced = out["without_backend_error_tasks"]["0"]["fixed/bare"]["correct"]["mean"]
    assert full == 0.25 and reduced == 0.0  # one task of four contributes +1; bounded by 1 / n tasks
