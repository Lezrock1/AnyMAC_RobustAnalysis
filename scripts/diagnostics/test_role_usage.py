import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
import scoring
from role_usage import inspect, summarize, trace_roles
from contract import Episode


TASK = {"id": "mmlu_pro/1", "text": "Q", "kind": "mc", "truth": "C", "n_options": 4}


def make_trace(roles=("Expert", "Critic"), answer="The answer is (C)", compromised="Expert"):
    answers = iter([answer] * (len(roles) + 1))

    def llm(messages):
        return next(answers), {"prompt_tokens": 1, "completion_tokens": 1, "truncated": False}

    ep = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised=compromised,
                 llm=llm, max_calls=32)

    def body(e):
        mids = [e.call("worker", role, []) for role in roles]
        final = e.call("aggregator", "Decider",
                       [{"role": "user", "content": "\n".join(e.text(m) for m in mids)}],
                       inputs=mids)
        e.finish(final_msg=final)
    return ep.run(body)


def test_trace_roles_returns_ordered_worker_calls_only():
    trace = make_trace(roles=("Expert", "Critic", "Expert"))
    assert trace_roles(trace) == ["Expert", "Critic", "Expert"]


def test_decider_only_episode_has_no_roles():
    answers = iter(["The answer is (A)"])

    def llm(messages):
        return next(answers), {"prompt_tokens": 1, "completion_tokens": 1, "truncated": False}

    ep = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Expert",
                 llm=llm, max_calls=32)
    trace = ep.run(lambda e: e.finish(final_msg=e.call("aggregator", "Decider", [])))
    assert trace_roles(trace) == []
    row = inspect(trace, TASK, scoring)
    assert row["n_worker_calls"] == 0 and not row["compromised_invoked"]
    assert row["compromised_first_position"] is None


def test_inspect_counts_positions_and_compromised_without_mutating():
    trace = make_trace(roles=("Critic", "Expert"))
    before = copy.deepcopy(trace)
    row = inspect(trace, TASK, scoring)
    assert row["role_counts"] == {"Critic": 1, "Expert": 1}
    assert row["compromised_invoked"] and row["compromised_first_position"] == 2
    assert row["final_correct"]  # decider answered (C) = truth
    assert trace == before


def test_failed_worker_call_is_not_counted():
    calls = iter([("The answer is (C)", None), (None, "RuntimeError: boom"),
                  ("The answer is (C)", None)])

    def llm(messages):
        text, err = next(calls)
        if err:
            raise RuntimeError("boom")
        return text, {"prompt_tokens": 1, "completion_tokens": 1, "truncated": False}

    ep = Episode(system="fixed", task=TASK, policy="honest", rep=1, compromised="Expert",
                 llm=llm, max_calls=32)

    def body(e):
        e.call("worker", "Expert", [])
        final = e.call("aggregator", "Decider", [])
        e.finish(final_msg=final)
    trace = ep.run(body)
    assert trace_roles(trace) == ["Expert"]


def test_summarize_aggregates_cells_and_difficulty_split():
    records = [
        {"cell": "pilot/anymac-normal/seed1", "task_id": "t1", "rep": 1, "policy": "honest",
         "final_correct": True, "n_worker_calls": 2, "roles": ["Critic", "Expert"],
         "role_counts": {"Critic": 1, "Expert": 1}, "compromised_invoked": True,
         "compromised_first_position": 2},
        {"cell": "pilot/anymac-normal/seed1", "task_id": "t2", "rep": 1, "policy": "honest",
         "final_correct": False, "n_worker_calls": 0, "roles": [], "role_counts": {},
         "compromised_invoked": False, "compromised_first_position": None},
        {"cell": "pilot/anymac-efficient/seed0", "task_id": "t1", "rep": 1, "policy": "honest",
         "final_correct": True, "n_worker_calls": 0, "roles": [], "role_counts": {},
         "compromised_invoked": False, "compromised_first_position": None},
    ]
    cells = summarize(records)
    normal = cells["pilot/anymac-normal/seed1"]
    assert normal["episodes"] == 2 and normal["episodes_with_worker"] == 1
    assert normal["role_call_counts"] == {"Critic": 1, "Expert": 1}
    assert normal["compromised_role_invoked_episodes"] == 1
    assert normal["compromised_role_invoked_rate"] == 0.5
    assert normal["accuracy_with_worker"] == 1.0 and normal["accuracy_without_worker"] == 0.0
    efficient = cells["pilot/anymac-efficient/seed0"]
    assert efficient["episodes_with_worker"] == 0
    assert efficient["accuracy_with_worker"] is None
    assert efficient["accuracy_without_worker"] == 1.0


def test_summarize_empty_records():
    assert summarize([]) == {}
