import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
import graph_probe
from contract import Episode
from test_learned import LLM, TASK, system_for


def fixture():
    system = system_for("gdesigner-paper")
    tasks = [dict(TASK, id=f"dev/{i}", text=f"Question {i}") for i in range(3)]
    traces = []
    for i, task in enumerate(tasks):
        ep = Episode(system=system.name, task=task, policy="honest", rep=1,
                     compromised="Expert", llm=LLM(), max_calls=32)
        generator = torch.Generator().manual_seed(system.settings["seed"] + 100000 + i)
        traces.append(ep.run(lambda ep: system(ep, task, generator=generator)))
    return system, tasks, traces


def test_probability_probe_replays_real_graphs_without_llm_or_mutation():
    system, tasks, traces = fixture()
    before = copy.deepcopy(traces)
    result = graph_probe.probe(system, tasks, traces, torch)
    assert traces == before
    assert result["all_recorded_graphs_replayed"]
    assert len(result["common_noise_eligible_edges"]) == 3
    assert len(result["recorded_graphs"]) == 3
    assert result["zero_noise_eligible_edges"]["min"] >= 1e-6
    assert result["zero_noise_eligible_edges"]["max"] <= 1 - 1e-6


def test_probability_bound_saturation_is_not_inferred_from_invariance():
    lower = torch.full((3,), 1e-6)
    saturated = graph_probe.probability_summary([lower, lower], torch)
    assert saturated["max_task_range"] == 0
    assert saturated["fraction_at_lower_bound"] == 1
    middle = torch.full((3,), .5)
    invariant = graph_probe.probability_summary([middle, middle], torch)
    assert invariant["max_task_range"] == 0
    assert invariant["fraction_at_lower_bound"] == 0
    assert invariant["fraction_at_upper_bound"] == 0
    with pytest.raises(ValueError, match="nonfinite"):
        graph_probe.probability_summary([torch.tensor([float("nan")])], torch)


def test_probe_refuses_duplicate_or_mismatched_graphs():
    system, tasks, traces = fixture()
    with pytest.raises(ValueError, match="incomplete/duplicate"):
        graph_probe.probe(system, tasks, traces + traces[:1], torch)
    graph = next(e for e in traces[0]["events"] if e["identity"] == "GDesigner-task-graph")
    graph["edges"] = [["not-a-role", "other"]]
    with pytest.raises(ValueError, match="replay differs"):
        graph_probe.probe(system, tasks, traces, torch)
