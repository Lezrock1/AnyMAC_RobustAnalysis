from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
import suite
from main_deep_dive import (cost, dylan_features, dylan_summary, honest_vs_reference, noise_floor,
                            single_call_labels, strata)


def row(system, seed, task, policy, rep=1, form="bare", pred="A", truth="A", target="B", calls=1, invoked=0):
    return {"system": system, "seed": seed, "task_id": task, "policy": policy, "rep": rep, "form": form,
            "pred": pred, "correct": pred == truth, "target_hit": pred == target, "calls": calls,
            "invoked": invoked, "prompt_tokens": 100, "completion_tokens": 50, "episode_wall_s": 2.0}


def test_cost_uses_honest_rows_only():
    rows = [row("fixed", None, "t1", "honest", calls=4), row("fixed", None, "t1", "attack", calls=99)]
    c = cost(rows)["fixed"]
    assert c["n"] == 1 and c["mean_calls"] == 4 and c["accuracy"] == 1.0


def test_honest_vs_reference_is_paired_by_task():
    rows = []
    for t, (a, b) in {"t1": ("A", "A"), "t2": ("A", "C"), "t3": ("A", "C"), "t4": ("A", "A")}.items():
        rows.append(row("anymac-efficient", 0, t, "honest", pred=a))
        rows.append(row("dylan", None, t, "honest", pred=b))
    out = honest_vs_reference(rows, suite.bootstrap, "anymac-efficient/seed0")
    assert out["dylan"]["mean"] == -0.5 and out["dylan"]["tasks"] == 4
    with pytest.raises(ValueError, match="reference"):
        honest_vs_reference(rows, suite.bootstrap, "missing")


def test_single_call_labels_exclude_cells_that_invoke_the_compromised_role():
    rows = [row("anymac-efficient", 0, "t1", "honest"), row("anymac-efficient", 0, "t1", "attack"),
            row("anymac-normal", 1, "t1", "honest", calls=6), row("anymac-normal", 1, "t1", "attack", calls=6, invoked=5)]
    assert single_call_labels(rows) == ["anymac-efficient/seed0"]


def test_noise_floor_counts_prediction_disagreements():
    rows = []
    for lab_seed, flips in ((0, {"t3"}), (1, set())):
        for t in ("t1", "t2", "t3", "t4"):
            rows.append(row("anymac-efficient", lab_seed, t, "honest", rep=1))
            rows.append(row("anymac-efficient", lab_seed, t, "honest", rep=2, pred="C" if t in flips else "A"))
            rows.append(row("anymac-efficient", lab_seed, t, "attack", pred="C" if t in flips else "A"))
    nf = noise_floor(rows)
    assert nf["honest_rep1_vs_rep2"] == {"anymac-efficient/seed0": 0.25, "anymac-efficient/seed1": 0.0}
    assert nf["attack_vs_honest_rep1"]["anymac-efficient/seed0/bare"] == 0.25
    assert nf["cross_checkpoint_honest_rep1"]["anymac-efficient/seed0 vs anymac-efficient/seed1"] == 0.0


def test_strata_separates_effect_from_noop_control():
    rows = []
    # Four tasks, honest correct twice on t1/t2, wrong on t3, mixed on t4. The attack moves the wrong task to target.
    honest = {"t1": ("A", "A"), "t2": ("A", "A"), "t3": ("C", "C"), "t4": ("A", "C")}
    attacked = {"t1": "A", "t2": "A", "t3": "B", "t4": "A"}
    for t, (r1, r2) in honest.items():
        rows += [row("fixed", None, t, "honest", rep=1, pred=r1), row("fixed", None, t, "honest", rep=2, pred=r2),
                 row("fixed", None, t, "attack", pred=attacked[t], invoked=1),
                 row("anymac-efficient", 0, t, "honest", rep=1, pred=r1),
                 row("anymac-efficient", 0, t, "honest", rep=2, pred=r2),
                 row("anymac-efficient", 0, t, "attack", pred=r1)]  # no-op: same prediction as honest rep 1
    s = strata(rows)
    fixed = s["fixed/bare"]["strata"]
    assert fixed["never_correct"] == {"tasks": 1, "honest_target_hit": 0.0, "attack_target_hit": 1.0,
                                      "delta_target_hit": 1.0, "delta_correct": 0.0}
    assert fixed["always_correct"]["delta_target_hit"] == 0.0 and fixed["mixed"]["tasks"] == 1
    assert s["fixed/bare"]["transitions_honest_rep1_to_attack"] == {"correct->correct": 3, "other->target": 1}
    # The control shows no stratum effect on the wrong task although it is "never correct".
    assert s["anymac-efficient/seed0/bare"]["strata"]["never_correct"]["delta_target_hit"] == 0.0


def event_call(kind, identity, round=1, msg="m0"):
    return {"type": "call", "kind": kind, "identity": identity, "round": round, "msg": msg}


def test_dylan_features_and_summary_group_by_consensus_round_and_selection():
    def trace(policy, events):
        return {"policy": policy, "compromised": "Economist", "events": events}
    r1 = trace("attack", [event_call("worker", "Economist"), event_call("worker", "Doctor")])
    r2_sel = trace("attack", [event_call("worker", "Economist"), event_call("ranker", "Ranker", round=2),
                              {"type": "decision", "kind": "ranker", "selected": ["Economist", "Doctor"]}])
    r2_out = trace("attack", [event_call("worker", "Economist"), event_call("ranker", "Ranker", round=2),
                              {"type": "decision", "kind": "ranker", "selected": ["Doctor", "Lawyer"]}])
    honest = trace("honest", [event_call("worker", "Economist")])
    rows = [{"target_hit": False, "correct": True}, {"target_hit": True, "correct": False},
            {"target_hit": False, "correct": True}, {"target_hit": False, "correct": True}]
    feats = [dylan_features(t, r) for t, r in zip((r1, r2_sel, r2_out, honest), rows)]
    assert feats[1]["comp_selected_round2"] is True and feats[2]["comp_selected_round2"] is False
    assert feats[0]["comp_selected_round2"] is None
    out = dylan_summary(feats)
    assert out["attack"]["episodes"] == 3 and out["attack"]["ranker_used_rate"] == pytest.approx(2 / 3)
    g = out["attack"]["groups"]
    assert (g["consensus_round1"]["n"], g["round2_comp_selected"]["n"], g["round2_comp_not_selected"]["n"]) == (1, 1, 1)
    assert g["round2_comp_selected"]["target_hit"] == 1.0 and g["consensus_round1"]["target_hit"] == 0.0
    assert out["honest"]["ranker_used_rate"] == 0.0


def test_report_prints_every_section_for_a_small_synthetic_run(capsys):
    from main_deep_dive import checkpoint_effects, print_report
    rows = []
    for t in ("t1", "t2", "t3", "t4"):
        for lab, seed, calls, inv in (("anymac-efficient", 0, 1, 0), ("fixed", None, 4, 1)):
            rows += [row(lab, seed, t, "honest", rep=1, calls=calls), row(lab, seed, t, "honest", rep=2, calls=calls)]
            for form in ("bare", "persuasive"):
                rows.append(row(lab, seed, t, "attack", form=form, calls=calls, invoked=inv,
                                pred="B" if lab == "fixed" and t == "t3" else "A"))
    result = {"cost": cost(rows), "honest_vs_reference": honest_vs_reference(rows, suite.bootstrap,
                                                                             "anymac-efficient/seed0"),
              "noise_floor": noise_floor(rows),
              "checkpoint_effects": checkpoint_effects(rows, suite.contrasts), "strata": strata(rows),
              "dylan": dylan_summary([]), "interpretation": "x"}
    print_report(result, "anymac-efficient/seed0")
    out = capsys.readouterr().out
    for section in ("== cost", "== honest accuracy minus", "== noise floor", "== effects per checkpoint",
                    "== strata", "transitions:", "== DyLAN"):
        assert section in out
    assert "fixed" in out and "persuasive-minus-bare" in out
