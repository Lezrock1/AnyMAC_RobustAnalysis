import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compare"))
import contract
import dylan
import runner
import scoring

SPEC = importlib.util.spec_from_file_location("compare_results", Path(__file__).with_name("compare_results.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def task(i):
    return {"id": f"mmlu_pro/{i}", "benchmark": "mmlu_pro", "kind": "mc", "n_options": 4,
            "truth": "C", "text": f"Question {i}?\n\nA. a\nB. b\nC. c\nD. d"}


def llm(messages):
    return "Reasoning.\nThe answer is (C)", {"prompt_tokens": 10, "completion_tokens": 10, "truncated": False}


def trace(adapter, t, policy, role, rep=1, form="bare", text=None, limit=17, model=llm):
    episode = contract.Episode(system=adapter.name, task=t, policy=policy, rep=rep, compromised=role,
                               llm=model, max_calls=limit, attack_form=form, attack_text=text)
    return episode.run(lambda ep: adapter(ep, t))


@pytest.fixture
def full_study(tmp_path):
    base = tmp_path / "study"
    tasks = [task(i) for i in range(60)]
    write(base / "eval_tasks.json", {"tasks": tasks})
    texts = {t["id"]: {"target": contract.target_for(t), "consistent": True,
                       "text": "Frozen reasoning.\n" + contract.attack_final_line(contract.target_for(t))} for t in tasks}
    write(base / "persuasive_texts.json", {"tasks_sha256": report.digest(base / "eval_tasks.json"), "texts": texts})
    for system, roles in (("dylan", dylan.ROLES), ("dylan-upstream", dylan.ROLES), ("fixed", runner.FIXED_ROLES)):
        adapter = dylan.DyLAN(variant="paper" if system == "dylan" else "upstream") if system != "fixed" else runner.FixedCycle(roles)
        cells = [(base / "honest" / system, "honest", 2, "Critic" if system == "fixed" else roles[0], "bare")]
        cells += [(base / "attack" / form / system / role.replace(" ", "_"), "attack", 1, role, form)
                  for form in ("bare", "persuasive") for role in roles]
        for path, policy, reps, role, form in cells:
            manifest = {"system": adapter.config(), "provenance": adapter.provenance(),
                        "compromised": role, "max_calls": 17, "policies": [policy], "reps": reps,
                        "order_seed": 0, "target_seed": 0, "task_ids": [t["id"] for t in tasks],
                        "tasks_sha256": report.digest(base / "eval_tasks.json"), "reserved_sha256": [],
                        "model": {"digest": "frozen", "options": {"num_thread": 4}},
                        "source": runner.source_digest()}
            if form == "persuasive":
                manifest["attack"] = {"form": form, "file_sha256": report.digest(base / "persuasive_texts.json"),
                                      "text_sha256": {k: runner.sha256_bytes(v["text"].encode()) for k, v in texts.items()}}
            write(path / "manifest.json", manifest)
            traces = [trace(adapter, t, policy, role, rep, form, texts[t["id"]]["text"] if form == "persuasive" else None)
                      for t in tasks for rep in range(1, reps + 1)]
            (path / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
            write(path / "COMPLETE.json", {"episodes": len(traces)})
    return base


def test_complete_study_read_only_and_output(full_study, tmp_path):
    before = {p: report.digest(p) for p in full_study.rglob("*") if p.is_file()}
    result = report.study(full_study, runner, scoring, dylan, draws=100)
    assert result["episodes"] == 2400
    assert result["tasks"] == 60
    assert len(result["cells"]) == 43  # 37 runs plus six descriptive transition summaries
    assert len(result["effects"]) == 11
    assert result["text_quality_sensitivity"]["kept_tasks"] == 60
    for cell in result["effects"].values():
        assert cell["correct"]["effect"] == 0
        assert cell["correct"]["tasks"] == 60  # never 420 role-task observations
    assert before == {p: report.digest(p) for p in full_study.rglob("*") if p.is_file()}
    report.save(tmp_path / "analysis", full_study, result)
    assert (tmp_path / "analysis/REPORT.md").exists()
    with pytest.raises(FileExistsError):
        report.save(tmp_path / "analysis", full_study, result)
    with pytest.raises(ValueError, match="separate"):
        report.save(full_study / "analysis", full_study, result)


@pytest.mark.parametrize("change", ["missing", "source", "task", "model", "text", "config", "duplicate"])
def test_refuse_mixed_or_incomplete_study(full_study, change):
    cell = full_study / "honest/dylan"
    if change == "missing":
        (cell / "COMPLETE.json").unlink()
    elif change == "text":
        p = full_study / "persuasive_texts.json"
        p.write_text(p.read_text() + " ")
    elif change == "duplicate":
        p = cell / "traces.jsonl"
        p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n")
    else:
        path = cell / "manifest.json"
        m = json.loads(path.read_text())
        if change == "source":
            m["source"] = {}
        elif change == "task":
            m["tasks_sha256"] = "changed"
        elif change == "config":
            m["system"]["rounds"] = 2
        else:
            m["model"]["digest"] = "changed"
        write(path, m)
    with pytest.raises((ValueError, SystemExit, FileNotFoundError)):
        report.study(full_study, runner, scoring, dylan, draws=100)


def test_programmatic_truncation_and_honest_relabel():
    t = task(0)
    honest = trace(dylan.DyLAN(), t, "honest", "Economist")
    worker = next(e for e in honest["events"] if e["type"] == "call")
    worker["usage"]["truncated"] = True
    row = report.mechanisms(honest, t, scoring, dylan)
    assert row["any_final_dependency_truncated"]
    assert row["outcome"] == "correct"  # legacy score: no final LLM call
    assert not row["ranker_reached"]
    assert row["consensus_round0"]
    relabeled = copy.deepcopy(honest)
    relabeled["compromised"] = "Doctor"
    assert report.mechanisms(relabeled, t, scoring, dylan)["correct"] == row["correct"]
    assert honest["compromised"] == "Economist"


def test_bootstrap_is_task_paired_and_deterministic():
    assert report.interval([.5, -.5], 1000) == report.interval([.5, -.5], 1000)
    assert report.interval([1] * 60, 100)["ci95"] == [1, 1]
    with pytest.raises(ValueError):
        report.interval([], 100)


def test_nonzero_effects_average_roles_and_repetitions_within_task():
    systems = {"dylan": ["A", "B"], "dylan-upstream": ["A", "B"], "fixed": ["A"]}
    honest, attacks = {}, {}
    for system, roles in systems.items():
        honest[system] = {}
        for tid in ("q1", "q2"):
            honest[system][tid] = {role: [{m: .25 for m in report.METRICS},
                                         {m: .75 for m in report.METRICS}] for role in roles}
            for role in roles:
                for form in ("bare", "persuasive"):
                    value = 1 if system == "dylan" and tid == "q1" and form == "persuasive" else .5
                    attacks[(system, form, role, tid)] = {m: value for m in report.METRICS}
    effects = report.contrasts(honest, attacks, systems, 100)
    assert effects["dylan/persuasive/attack-minus-honest"]["correct"]["effect"] == .25
    assert effects["dylan/persuasive-minus-bare"]["correct"]["effect"] == .25
    assert effects["dylan-minus-upstream/persuasive/difference-in-differences"]["correct"]["effect"] == .25
    assert effects["dylan/persuasive/attack-minus-honest"]["correct"]["tasks"] == 2


@pytest.mark.parametrize("fallback", [False, True])
def test_ranker_decisions_are_measured_without_conflating_selection(fallback):
    counter = 0
    def varying(messages):
        nonlocal counter
        counter += 1
        text = ("not a ranking" if fallback else "[1, 2]") if counter == 15 else f"The answer is ({'ABCD'[counter % 4]})"
        return text, {"prompt_tokens": 10, "completion_tokens": 10, "truncated": False}
    t = task(0)
    traced = trace(dylan.DyLAN(), t, "honest", "Economist", model=varying)
    row = report.mechanisms(traced, t, scoring, dylan)
    assert row["ranker_reached"]
    assert row["ranker_fallback"] == fallback
    assert not row["ranking_selection_mismatch"]
    assert not row["consensus_round0"] and not row["consensus_round1"]


@pytest.mark.parametrize("system", ["anymac-corrected", "gdesigner-paper"])
def test_learned_development_binding_and_counts(tmp_path, system):
    root = tmp_path / "seed0"
    tasks = [task(i) for i in range(12)]
    dev = tmp_path / "dev.json"
    write(dev, {"tasks": tasks})
    root.mkdir()
    (root / "router.pt").write_bytes(b"hash-only fixture, not loaded")
    manifest = {"config": {"system": system},
                "dev_ids": [t["id"] for t in tasks], "dev_sha256": report.digest(dev),
                "source": runner.source_digest(), "max_calls": 17}
    write(root / "training_manifest.json", manifest)
    write(root / "router.json", {**manifest, "artifact_sha256": report.digest(root / "router.pt")})
    adapter = runner.FixedCycle(["Lawyer"])
    for phase in ("dev_initial", "dev_trained"):
        traces = [trace(adapter, t, "honest", "Lawyer") for t in tasks]
        for tr in traces:
            tr["system"] = system
        (root / f"{phase}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    before = {p: report.digest(p) for p in root.iterdir()}
    result = report.diagnose(root, dev, runner, scoring)
    assert result["system"] == system
    assert result["unchanged_predictions"] == 12
    assert result["phases"]["dev_initial"]["roles"] == {"Lawyer": 12}
    assert result["phases"]["dev_initial"]["worker_episode_rate"] == 1
    routing = result["phases"]["dev_trained"]["routing"]
    assert routing["worker_sequences"] == [{"sequence": ["Lawyer"], "tasks": 12}]
    assert routing["mean_distinct_worker_roles"] == 1
    assert routing["mean_repeated_role_calls"] == 0
    assert routing["largest_role_call_share"] == 1
    assert routing["role_totals"]["Lawyer"]["final_direct_messages"] == 12
    assert routing["final_direct_worker_input_rate"] == 1
    assert before == {p: report.digest(p) for p in root.iterdir()}
    report.save(tmp_path / "analysis", root, result)
    assert "mean distinct roles" in (tmp_path / "analysis/REPORT.md").read_text()
    file = root / "dev_trained.jsonl"
    old = file.read_text()
    traces[0]["system"] = "fixed"
    file.write_text("".join(json.dumps(t) + "\n" for t in traces))
    with pytest.raises(ValueError, match="matching learned-system"):
        report.diagnose(root, dev, runner, scoring)
    file.write_text(old)
    dev.write_text(dev.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        report.diagnose(root, dev, runner, scoring)


def routing_fixture(body, system="anymac-corrected"):
    ep = contract.Episode(system=system, task=task(0), policy="honest", rep=1,
                          compromised="Unused", llm=llm, max_calls=32)

    def call(kind, role, inputs=()):
        return ep.call(kind, role, [{"role": "user", "content": "Solve.\n" +
                                    "\n".join(ep.text(mid) for mid in inputs)}], inputs=inputs)

    return ep.run(lambda e: body(e, call))


def test_routing_separates_router_visibility_delivery_repetition_and_ancestry():
    def body(ep, call):
        first = call("worker", "A")
        for identity in ("AnyMAC", "AnyMAC-context"):
            ep.decide("router", identity, inputs=[first])
        second = call("worker", "A", [first])
        third = call("worker", "B", [second])
        call("worker", "A")
        ep.finish(final_msg=call("aggregator", "Decider", [third]))

    tr = routing_fixture(body)
    before = copy.deepcopy(tr)
    result = report.routing_trace(tr)
    assert tr == before
    assert result["worker_sequence"] == ["A", "A", "B", "A"]
    assert result["worker_calls"] == 4
    assert result["distinct_worker_roles"] == 2
    assert result["repeated_role_calls"] == 2
    assert result["roles"]["A"] == {
        "calls": 3, "messages": 3, "router_seen_messages": 1, "llm_delivered_messages": 2,
        "other_worker_deliveries": 1, "self_worker_deliveries": 1, "aggregator_deliveries": 0,
        "final_direct_messages": 0, "final_ancestor_messages": 2,
    }
    assert result["roles"]["B"]["final_direct_messages"] == 1
    assert result["roles"]["B"]["final_ancestor_messages"] == 1
    assert result["final_has_direct_worker_input"]
    assert result["final_has_worker_ancestor"]
    # The legacy compromised-role exposure is zero despite actual delivery from A and B.
    assert scoring.score(tr, task(0))["aggregator_exposed"] is False
    assert report.routing_summary([tr])["largest_role_call_share"] == .75


def test_no_worker_and_router_only_visibility_are_not_final_delivery():
    def only_router(ep, call):
        mid = call("worker", "Historian")
        ep.decide("router", "AnyMAC", inputs=[mid])
        ep.finish(final_msg=call("aggregator", "Decider"))

    tr = routing_fixture(only_router)
    result = report.routing_summary([tr])
    assert result["role_totals"]["Historian"]["router_seen_messages"] == 1
    assert result["role_totals"]["Historian"]["llm_delivered_messages"] == 0
    assert result["final_direct_worker_input_rate"] == 0
    assert result["final_worker_ancestor_rate"] == 0
    no_workers = routing_fixture(lambda ep, call: ep.finish(final_msg=call("aggregator", "Decider")))
    result = report.routing_summary([no_workers])
    assert result["largest_role_call_share"] is None
    assert result["mean_worker_calls"] == 0
    assert result["role_totals"] == {}
    assert result["worker_sequences"] == [{"sequence": [], "tasks": 1}]


def test_gdesigner_final_ancestry_excludes_disconnected_earlier_round():
    def body(ep, call):
        ep.decide("router", "GDesigner-task-graph", candidates=["A", "B"], edges=[["A", "B"]])
        old = call("worker", "A")
        call("aggregator", "Decider", [old])
        current = call("worker", "A")
        ep.finish(final_msg=call("aggregator", "Decider", [current]))

    result = report.routing_trace(routing_fixture(body, system="gdesigner-paper"))
    assert result["task_graph_edges"] == [[["A", "B"]]]
    assert result["roles"]["A"]["aggregator_deliveries"] == 2
    assert result["roles"]["A"]["final_direct_messages"] == 1
    assert result["roles"]["A"]["final_ancestor_messages"] == 1


def test_routing_counts_message_deliveries_not_only_exposed_calls():
    def body(ep, call):
        first = call("worker", "A")
        second = call("worker", "A", [first])
        ep.finish(final_msg=call("aggregator", "Decider", [first, second]))

    result = report.routing_summary([routing_fixture(body)])
    assert result["role_totals"]["A"]["aggregator_deliveries"] == 2
    assert result["role_totals"]["A"]["final_direct_messages"] == 2
    assert result["role_totals"]["A"]["final_ancestor_messages"] == 2
