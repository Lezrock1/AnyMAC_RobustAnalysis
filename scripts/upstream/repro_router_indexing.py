"""Minimal reproduction of three indexing issues in AnyMAC's next-agent router.

The script extracts ``Graph.arun_next_agent_prediction_grad`` unchanged from an
AnyMAC ``GDesigner/graph/graph.py`` and executes it with a tiny in-memory graph.
No LLM, no dataset and no GPT-2 download are needed.

Token layout built by the router (forward and gradient path):
    [task, role_0 .. role_{n-1}, history_0 .. history_{h-1}, NAP, NHP]

Usage:
    python repro_router_indexing.py --graph /path/to/AnyMAC/GDesigner/graph/graph.py
"""
import argparse
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROLES = ("r0", "r1", "r2", "r3", "final")
DIM = 8


def load_gradient_method(graph_path):
    tree = ast.parse(Path(graph_path).read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "arun_next_agent_prediction_grad")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "F": F,
                 "get_sentence_embedding": lambda text: [0.01 * (i % 17) for i in range(384)]}
    exec(compile(module, str(graph_path), "exec"), namespace)
    return namespace[method.name]


class RecordingIdentity(torch.nn.Module):
    """Identity 'transformer' so every output token is exactly its input token."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, x, num_prefix_tokens=0):
        x.retain_grad()
        self.calls.append(x)
        return x * self.scale


def make_graph():
    torch.manual_seed(0)
    graph = SimpleNamespace(device=torch.device("cpu"), cos_scaling=1.0,
                            decision_method="final", to_device=lambda device: None)
    graph.role_to_idx = {role: i for i, role in enumerate(ROLES)}
    graph.idx_to_role = {i: role for role, i in graph.role_to_idx.items()}
    graph.role_embeddings = {role: torch.randn(384) for role in ROLES}
    for name, size in (("task", 384), ("role", 384), ("nap", 384), ("nhp", 384), ("history", 768)):
        setattr(graph, f"proj_to_transformer_dim_{name}", torch.nn.Linear(size, DIM))
    graph.routing_transformer = RecordingIdentity()
    return graph


def influential_rows(x):
    return [i for i, row in enumerate(x.grad) if float(row.abs().sum()) > 0]


def run(method, selections, hint_selections):
    graph = make_graph()
    inputs = {"task": "synthetic", "advantage": 1.0, "agent_selections": selections,
              "trace_length": len(selections), "hint_selections": hint_selections,
              "agent_outputs_embeddings": [torch.randn(384) for _ in selections]}
    with torch.enable_grad():
        asyncio.run(method(graph, inputs))
    return graph.routing_transformer.calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="GDesigner/graph/graph.py")
    args = parser.parse_args()
    method = load_gradient_method(args.graph)
    n = len(ROLES)
    roles = list(range(1, n + 1))

    # 1) Role scores: one step, choose r1. Rows: 0 task, 1..5 roles, 6 NAP, 7 NHP.
    step = run(method, [1, ROLES.index("final")], [[1]])[0]
    used = influential_rows(step)
    role_check = {
        "expected_role_rows": roles, "nap_row": n + 1,
        "rows_that_influence_loss": used,
        "task_row_used_as_candidate": 0 in used,
        "last_role_row_unused": n not in used,
    }

    # 2) Hint scores: second step has one history row at index n+1 (NAP n+2, NHP n+3).
    # As in real traces, hint_selections has one mask per step that already has history.
    calls = run(method, [0, 1, ROLES.index("final")], [[1], [1, 0]])
    second = calls[1]
    history_row = n + 1
    hint_check = {
        "history_row": history_row,
        "rows_that_influence_loss": influential_rows(second),
        "history_row_has_no_influence": float(second.grad[history_row].abs().sum()) == 0.0,
    }

    # 3) Non-forced early decision: router picks 'final' at step 1 of a longer budget.
    calls = run(method, [0, ROLES.index("final")], [[1]])
    decision_check = {
        "selections": [0, ROLES.index("final")],
        "router_steps_in_trace": 2,
        "router_steps_with_gradient": len(calls),
        "decision_step_has_no_gradient": len(calls) == 1,
    }

    result = {"role_scores": role_check, "hint_scores": hint_check, "early_decision": decision_check}
    print(json.dumps(result, indent=2))
    reproduced = (role_check["task_row_used_as_candidate"] and role_check["last_role_row_unused"]
                  and hint_check["history_row_has_no_influence"]
                  and decision_check["decision_step_has_no_gradient"])
    print("ALL THREE ISSUES REPRODUCED" if reproduced else "NOT (FULLY) REPRODUCED")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
