"""Tests for the stage 0a simulation (run: pytest scripts/stage0/test_sim.py)."""
import inspect
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("sentence_transformers")

import sim  # noqa: E402
from GDesigner.agents.mas_agents import MASAgent, MASFinal  # noqa: E402
from robust.data import format_task  # noqa: E402


@pytest.fixture()
def world():
    tasks = sim.synthetic_tasks(3, "t", 0)
    sim.WORLD.by_text.clear()
    sim.WORLD.register(tasks)
    sim.WORLD.scenario, sim.WORLD.active = "honest", False
    sim.WORLD.rng = random.Random(0)
    yield tasks
    sim.WORLD.scenario, sim.WORLD.active = "honest", False


def _hints(outputs):
    # exact layout produced by Graph.arun (graph.py: f"\nAgent {i+1}: {hint} \n")
    return {"h": {"role": "x", "output": "".join(f"\nAgent {i + 1}: {o} \n" for i, o in enumerate(outputs))}}


def test_majority_tie_goes_to_most_recent():
    assert sim._majority([3, 3, 5]) == 3
    assert sim._majority([3, 5]) == 5
    assert sim._majority([5, 3, 3, 5]) == 5


@pytest.mark.parametrize("marker", [None, "blue", "green"])
def test_worker_prompt_roundtrip(world, marker):
    t = world[1]
    agent = MASAgent(role="Inspector", domain="masbench", llm_name="mock")
    outs = [f"Math Solver: x\nThe answer is {a}" for a in (4, 9, 4)]
    for spatial in ({}, _hints(outs)):
        system, user = agent._process_inputs({"task": format_task(t, marker)}, spatial, {})
        assert sim._ROLE_BY_PROMPT[system] == "Inspector"
        text, answers = sim._split_prompt(user, sim._WORKER_HEAD, sim._WORKER_HINTS)
        assert sim.WORLD.by_text[text] == (t, marker)
        assert answers == ([4, 9, 4] if spatial else [])


def test_final_prompt_roundtrip_and_majority(world):
    t = world[0]
    final = MASFinal(domain="masbench", llm_name="mock")
    system, user = final._process_inputs({"task": format_task(t, "green")}, _hints(["The answer is 7", "The answer is 7",
                                                                                     "The answer is 2"]), {})
    assert system.startswith("You are the top decision-maker")
    text, answers = sim._split_prompt(user, sim._FINAL_HEAD, sim._FINAL_HINTS)
    assert sim.WORLD.by_text[text] == (t, "green") and answers == [7, 7, 2]
    assert sim.simulate(system, user).endswith("The answer is 7")


def test_oracle_behaviour(world):
    w = sim.WORLD
    truth = world[0].truth
    w.scenario = "oracle_uncond"
    assert all(sim.worker_answer(w.controlled_role, truth, m, []) == truth for m in (None, "blue", "green"))
    w.scenario, w.active = "oracle_cond", True
    assert sim.worker_answer(w.controlled_role, truth, "blue", []) == truth
    assert all(sim.worker_answer(w.controlled_role, truth, "green", []) != truth for _ in range(20))
    w.p_inspect = 1.0
    assert sim.worker_answer(w.verifier_role, truth, "green", [truth, (truth + 1) % 23]) == truth


def test_apply_variant_patches_and_restores():
    G = sim.graph_mod.Graph
    sim.apply_variant("full")
    fwd, grad = inspect.getsource(G.arun_next_agent_prediction), inspect.getsource(G.arun_next_agent_prediction_grad)
    assert "encoded_tokens[1:num_roles + 1]" in fwd and "hint_offset = num_roles + 1 " in fwd
    assert "range(trace_length):" in grad and "self.max_routing" in grad
    sim.apply_variant("indices")
    assert "range(trace_length - 1):" in inspect.getsource(G.arun_next_agent_prediction_grad)
    sim.apply_variant("original")
    assert G.arun_next_agent_prediction is sim._ORIGINAL["arun_next_agent_prediction"]
    assert G.arun_next_agent_prediction_grad is sim._ORIGINAL["arun_next_agent_prediction_grad"]


@pytest.mark.parametrize("variant", sim.VARIANTS)
def test_tiny_end_to_end(tmp_path, variant):
    try:
        sim.main(["run", "--out", str(tmp_path), "--scenario", "oracle_cond", "--variant", variant, "--rounds", "2",
                  "--warmup-rounds", "1", "--eval-rounds", "0", "--n-train", "8", "--n-eval", "4", "--threads", "1"])
    except OSError as e:  # gpt2 / MiniLM weights not cached
        pytest.skip(f"router graph unavailable: {e}")
    finally:
        sim.apply_variant("original")
    d = tmp_path / f"oracle_cond__{variant}__seed0"
    rounds = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    assert [r["active"] for r in rounds] == [False, True]
    res = json.loads((d / "result.json").read_text())
    assert [e["round"] for e in res["evals"]] == [0, 2]
    assert all(e["reps"][0][m]["errors"] == 0 for e in res["evals"] for m in ("blue", "green"))
    s = sim.main(["summarize", "--out", str(tmp_path)])
    assert f"oracle_cond/{variant}" in s["cells"]
