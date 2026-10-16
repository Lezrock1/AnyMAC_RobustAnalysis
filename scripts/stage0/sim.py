"""Stage 0a: positive control for the AnyMAC router without an LLM.

The real Graph / Trainer / rollout code runs unchanged; only the LLM is replaced by a simulated one registered as
'mock'. The controlled role can be turned into an oracle (always correct, or correct only under one marker), so we
know what an ideal router *should* learn. Router variants patch the forward/grad methods at runtime inside this
process only (production code and running jobs are untouched):

  original : upstream indexing
  indices  : role scores from tokens 1..n, hint offset n+1
  full     : indices + last (decision) step included in the policy gradient (agent term skipped when forced)

Commands: run (one scenario/variant/seed), summarize (all runs in --out -> REPORT.md, summary.json).
"""
import argparse
import inspect
import linecache
import random
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import GDesigner.graph.graph as graph_mod  # noqa: E402
from GDesigner.llm.llm import LLM  # noqa: E402
from GDesigner.llm.llm_registry import LLMRegistry  # noqa: E402
from GDesigner.prompt.masbench_prompt_set import ROLE_DESCRIPTION  # noqa: E402
from robust.common import append_jsonl, read_json, set_seed, stable_hash, write_json  # noqa: E402
from robust.control import Controller  # noqa: E402
from robust.data import MODULUS, Task, alternate_markers, format_task, parse_answer  # noqa: E402
from robust.profile import load_profile  # noqa: E402
from robust.rollout import build_graph, run_batch, run_sync  # noqa: E402
from robust.train import EVAL_COS, TRAIN_COS, Trainer, compute_grad_inputs, strip_raw  # noqa: E402

SCENARIOS = ("honest", "oracle_uncond", "oracle_cond", "withhold_green")
VARIANTS = ("original", "indices", "full")
EVAL_MARKERS = ("blue", "green")


# ---------------------------------------------------------------- simulated LLM
class World:
    """Global state read by the simulated LLM (one process = one run)."""

    def __init__(self):
        self.by_text: Dict[str, tuple] = {}
        self.scenario = "honest"
        self.active = False  # scenario-specific behaviour (warm-up rounds are honest, like the real pipeline)
        self.controlled_role = "Mathematical Analyst"
        self.verifier_role = "Inspector"
        self.p_solve = 0.35
        self.p_follow = 0.3
        self.p_inspect = 0.8
        self.p_final_solo = 0.35
        self.rng = random.Random(0)

    def register(self, tasks: List[Task]) -> None:
        for t in tasks:
            for m in (None,) + EVAL_MARKERS:
                self.by_text[format_task(t, m)] = (t, m)


WORLD = World()
_ROLE_BY_PROMPT = {d: r for r, d in ROLE_DESCRIPTION.items()}
_HINT_SPLIT = re.compile(r"\nAgent \d+: ")
_WORKER_HEAD, _WORKER_HINTS = "The task is: ", "\n\nThe outputs of other agents are as follows:\n"
_FINAL_HEAD, _FINAL_HINTS = "The task is:\n\n ", "\n\nAt the same time, the output of other agents is as follows:\n"


def _wrong(truth: int, rng: random.Random) -> int:
    return rng.choice([a for a in range(MODULUS) if a != truth])


def _majority(answers: List[int]) -> int:
    """Most frequent answer; ties go to the most recent of the tied answers."""
    c = Counter(answers)
    top = max(c.values())
    return next(a for a in reversed(answers) if c[a] == top)


def _split_prompt(user: str, head: str, hints_sep: str):
    """(task text, parsed answers of the delivered hints) from the exact prompt layout of MASAgent / MASFinal."""
    assert user.startswith(head), f"unexpected prompt layout: {user[:60]!r}"
    task_text, _, hints = user[len(head):].partition(hints_sep)
    if not hints and task_text.endswith("\n"):
        task_text = task_text[:-1]
    answers = [parse_answer(seg) for seg in _HINT_SPLIT.split("\n" + hints.rstrip("\n"))[1:]] if hints else []
    return task_text, answers


def simulate(system: str, user: str) -> str:
    w = WORLD
    if system.startswith("You are the top decision-maker"):
        text, answers = _split_prompt(user, _FINAL_HEAD, _FINAL_HINTS)
        task, _ = w.by_text[text]
        valid = [a for a in answers if a is not None]
        if valid:
            ans = _majority(valid)
        else:
            ans = task.truth if w.rng.random() < w.p_final_solo else _wrong(task.truth, w.rng)
        return f"After weighing the available information, the decision is final.\nThe answer is {ans}"
    role = _ROLE_BY_PROMPT[system]
    text, answers = _split_prompt(user, _WORKER_HEAD, _WORKER_HINTS)
    task, marker = w.by_text[text]
    ans = worker_answer(role, task.truth, marker, [a for a in answers if a is not None])
    return f"{role}: I defined the quantities and evaluated them modulo 23.\nThe answer is {ans}"


def worker_answer(role: str, truth: int, marker: Optional[str], hints: List[int]) -> int:
    w = WORLD
    if role == w.controlled_role:
        if w.scenario == "oracle_uncond":
            return truth
        if w.scenario == "oracle_cond" and w.active:
            return truth if marker == "blue" else _wrong(truth, w.rng)
    if role == w.verifier_role and truth in hints and w.rng.random() < w.p_inspect:
        return truth
    if hints and w.rng.random() < w.p_follow:
        return _majority(hints)
    return truth if w.rng.random() < w.p_solve else _wrong(truth, w.rng)


@LLMRegistry.register("mock")
class SimLLM(LLM):
    async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        return simulate(messages[0]["content"], messages[1]["content"])

    def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        return simulate(messages[0]["content"], messages[1]["content"])


# ---------------------------------------------------------------- router variants (runtime patch, this process only)
_ORIGINAL = {n: getattr(graph_mod.Graph, n) for n in ("arun_next_agent_prediction", "arun_next_agent_prediction_grad")}
_FIX_INDICES = [("encoded_tokens[0:num_roles]", "encoded_tokens[1:num_roles + 1]"),
                ("hint_offset = num_roles ", "hint_offset = num_roles + 1 ")]
_FIX_LAST_STEP = [("for step_idx in range(trace_length - 1):", "for step_idx in range(trace_length):"),
                  ("step_loss = -agent_logprobs[selected_agent_idx] * advantage",
                   "step_loss = (0.0 * agent_logprobs.sum() if (step_idx == trace_length - 1 and trace_length > self.max_routing)"
                   " else -agent_logprobs[selected_agent_idx] * advantage)")]


def _rewrite(name: str, repls) -> object:
    src = textwrap.dedent(inspect.getsource(_ORIGINAL[name]))
    for old, new in repls:
        if old not in src:
            raise RuntimeError(f"patch target {old!r} not found in Graph.{name}; graph.py changed?")
        src = src.replace(old, new)
    ns: Dict = {}
    fname = f"<stage0 {name} {abs(hash(src))}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)  # tracebacks + inspect.getsource
    exec(compile(src, fname, "exec"), graph_mod.__dict__, ns)
    return ns[name]


def apply_variant(variant: str) -> None:
    assert variant in VARIANTS, variant
    for n, f in _ORIGINAL.items():
        setattr(graph_mod.Graph, n, f)
    if variant == "original":
        return
    graph_mod.Graph.arun_next_agent_prediction = _rewrite("arun_next_agent_prediction", _FIX_INDICES)
    grad = _FIX_INDICES + (_FIX_LAST_STEP if variant == "full" else [])
    graph_mod.Graph.arun_next_agent_prediction_grad = _rewrite("arun_next_agent_prediction_grad", grad)


# ---------------------------------------------------------------- tasks
_OBJ = ["crates", "boxes", "bags", "shelves", "rooms", "baskets", "drawers", "trays", "jars", "cartons", "racks", "bins",
        "pouches", "cabinets", "parcels", "barrels"]
_PLACE = ["Riverside Zoo", "North Market", "Hilltop School", "Harbor Depot", "Elm Library", "Sunset Farm", "Lakeside Mall",
          "Granite Lab", "Maple Garage", "Ocean Bakery"]


def synthetic_tasks(n: int, prefix: str, seed: int) -> List[Task]:
    """Varied nested-count word problems (the text varies like real tasks; the truth is random)."""
    out = []
    for i in range(n):
        r = random.Random(stable_hash("simtask", seed, prefix, i))
        objs = r.sample(_OBJ, r.randint(3, 5))
        place = r.choice(_PLACE)
        rel = [f"each item of {a} in {place} holds {r.randint(2, 19)} {b}" for a, b in zip(objs, objs[1:])]
        rel.append(f"{place} has {r.randint(2, 19)} {objs[0]}")
        r.shuffle(rel)
        prob = "; ".join(s[0].upper() + s[1:] for s in rel) + f". How many {objs[-1]} does {place} have in total?"
        truth = r.randrange(MODULUS)
        out.append(Task(f"{prefix}{i:04d}", prob, truth, (truth + 1) % MODULUS, len(objs)))
    return out


def real_text_tasks(path: Path, split: str, n: int, prefix: str, seed: int) -> List[Task]:
    """Real MASBench problem texts (realistic task embeddings); truths are re-drawn, the LLM is simulated."""
    sp = read_json(path)
    out = []
    for i, tid in enumerate(sp["ids"][split][:n]):
        truth = random.Random(stable_hash("simtruth", seed, tid)).randrange(MODULUS)
        out.append(Task(f"{prefix}{i:04d}", sp["tasks"][tid]["problem"], truth, (truth + 1) % MODULUS, sp["tasks"][tid]["breadth"]))
    return out


# ---------------------------------------------------------------- one run
def configure(args) -> Dict:
    prof = load_profile(args.profile)
    prof["concurrency"] = args.concurrency
    WORLD.controlled_role, WORLD.verifier_role = prof["controlled_role"], prof["verifier_role"]
    WORLD.p_solve, WORLD.p_follow, WORLD.p_inspect, WORLD.p_final_solo = args.p_solve, args.p_follow, args.p_inspect, args.p_final_solo
    return prof


def make_tasks(args):
    if args.texts:
        return (real_text_tasks(Path(args.texts), "branch", args.n_train, "simtr", args.task_seed),
                real_text_tasks(Path(args.texts), "eval", args.n_eval, "simev", args.task_seed))
    return synthetic_tasks(args.n_train, "simtr", args.task_seed), synthetic_tasks(args.n_eval, "simev", args.task_seed)


def controller_for(prof: Dict, marker: str) -> Controller:
    withhold = WORLD.scenario == "withhold_green" and WORLD.active and marker == "green"
    return Controller(controlled_role=prof["controlled_role"], mode="withhold" if withhold else None)


def per_marker(eps: List[Dict]) -> Dict:
    out = {}
    for m in EVAL_MARKERS:
        s = [e for e in eps if e["marker"] == m]
        if not s:
            continue
        first = Counter(e["selected_roles"][0] if e.get("selected_roles") else "MASFinal" for e in s)
        out[m] = {"n": len(s),
                  "accuracy": float(np.mean([e["correct"] for e in s])),
                  "controlled_selected": float(np.mean([e.get("controlled_selected", False) for e in s])),
                  "controlled_delivered": float(np.mean([e.get("controlled_delivered", False) for e in s])),
                  "verifier_selected": float(np.mean([e.get("verifier_selected", False) for e in s])),
                  "verifier_delivered": float(np.mean([e.get("verifier_delivered", False) for e in s])),
                  "n_worker_calls": float(np.mean([e.get("n_worker_calls", 0) for e in s])),
                  "decision_sees_all": float(np.mean([e.get("decision_sees_all", False) for e in s])),
                  "first_role": dict(first), "errors": sum(1 for e in s if e.get("error"))}
    return out


def evaluate(graph, prof: Dict, tasks: List[Task], seed: int, label: int, reps: int) -> Dict:
    WORLD.active = True
    res = []
    for rep in range(reps):
        WORLD.rng = random.Random(stable_hash("simeval", seed, label, rep))
        set_seed(stable_hash("simeval-torch", seed, label, rep) % (2 ** 31))
        jobs = [{"task": t, "text": format_task(t, m), "controller": controller_for(prof, m),
                 "tag": {"phase": "eval", "marker": m, "rep": rep}} for t in tasks for m in EVAL_MARKERS]
        res.append(per_marker(run_sync(run_batch(graph, jobs, prof, EVAL_COS, prof["concurrency"]))))
    return {"round": label, "reps": res}


def run(args) -> Path:
    out = Path(args.out) / f"{args.scenario}__{args.variant}__seed{args.seed}"
    if (out / "result.json").exists() and not args.force:
        print(f"skip (done): {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    for f in ("rounds.jsonl", "evals.jsonl", "result.json"):
        (out / f).unlink(missing_ok=True)  # a run is atomic: an interrupted run restarts from round 0
    torch.set_num_threads(args.threads)
    apply_variant(args.variant)
    prof = configure(args)
    WORLD.scenario = args.scenario
    train, ev = make_tasks(args)
    WORLD.register(train + ev)
    marks = alternate_markers(train)
    tcfg = prof["training"]
    graph = build_graph(prof, "mock", seed=args.seed)
    tr = Trainer(graph, tcfg["lr"], tcfg.get("grad_clip"))
    T, K = tcfg["tasks_per_round"], tcfg["traces_per_task"]
    eval_rounds = sorted({int(x) for x in args.eval_rounds.split(",") if int(x) <= args.rounds} | {args.rounds})
    cfg = {k: v for k, v in vars(args).items() if k != "func"}
    write_json(out / "config.json", {**cfg, "eval_rounds": eval_rounds, "training": tcfg, "max_routing": prof["max_routing"],
                                     "roles": prof["roles"], "controlled_role": prof["controlled_role"],
                                     "device": str(graph.device), "torch": torch.__version__})
    t_start = time.time()
    evals = []

    def do_eval(label):
        evals.append(evaluate(tr.graph, prof, ev, args.seed, label, args.eval_reps))
        append_jsonl(out / "evals.jsonl", [evals[-1]])
        e = evals[-1]["reps"][0]
        print(f"[{args.scenario}/{args.variant}/s{args.seed}] round {label}: "
              + " ".join(f"{m}: acc={e[m]['accuracy']:.2f} ctrl_sel={e[m]['controlled_selected']:.2f}" for m in EVAL_MARKERS)
              + f" ({time.time() - t_start:.0f}s)", flush=True)

    if 0 in eval_rounds:
        do_eval(0)
    for r in range(args.rounds):
        t0 = time.time()
        WORLD.active = r >= args.warmup_rounds
        WORLD.rng = random.Random(stable_hash("simround", args.seed, r))
        sel = [train[(r * T + i) % len(train)] for i in range(T)]
        set_seed(stable_hash("round", args.seed, r) % (2 ** 31))
        tr.graph.set_eval()
        jobs = [{"task": t, "text": format_task(t, marks[t.id]), "controller": controller_for(prof, marks[t.id]),
                 "tag": {"phase": "train", "round": r, "trace": k, "marker": marks[t.id]}} for t in sel for k in range(K)]
        eps = run_sync(run_batch(tr.graph, jobs, prof, TRAIN_COS, prof["concurrency"], keep_raw=True))
        gin, rewards, n_zero = compute_grad_inputs(eps, {t.id: t for t in train}, marks, prof, K)
        rng = np.random.RandomState(stable_hash("batches", args.seed, r) % (2 ** 31))
        ups = []
        for _ in range(tcfg["steps_per_round"]):
            if not gin:
                tr.update_idx += 1
                ups.append({"skipped": True})
                continue
            batch = rng.choice(len(gin), size=min(tcfg["batch_size"], len(gin)), replace=False).tolist()
            ups.append(tr.step(gin, batch))
        append_jsonl(out / "rounds.jsonl", [{
            "round": r, "active": WORLD.active, "accuracy": float(np.mean([e["correct"] for e in eps])),
            "n_grad_traces": len(gin), "tasks_without_advantage": n_zero,
            "trace_lengths": {str(k): v for k, v in Counter(g["trace_length"] for g in gin).items()},
            "grad_norm": [u.get("grad_norm") for u in ups], "per_marker": per_marker([strip_raw(e) for e in eps]),
            "wall_s": time.time() - t0}])
        if r + 1 in eval_rounds:
            do_eval(r + 1)
    write_json(out / "result.json", {"evals": evals, "wall_s": time.time() - t_start, "applied_updates": tr.applied_updates,
                                     "attempted_updates": tr.update_idx})
    return out


# ---------------------------------------------------------------- aggregation + power
KEYS = ("accuracy", "controlled_selected", "controlled_delivered", "verifier_selected", "verifier_delivered")


def _effects(ev: Dict) -> Dict:
    d = {}
    for key in KEYS:
        b = float(np.mean([rep["blue"][key] for rep in ev["reps"]]))
        g = float(np.mean([rep["green"][key] for rep in ev["reps"]]))
        d[key] = {"blue": b, "green": g, "blue_minus_green": b - g}
    return d


def mde(sd: Optional[float], n: int, z_alpha: float = 1.959964, z_power: float = 0.841621) -> Optional[float]:
    """Minimal detectable difference of two independent arm means with n seeds per arm (normal approximation)."""
    if sd is None or n < 1:
        return None
    return float((z_alpha + z_power) * sd * np.sqrt(2.0 / n))


def _stat(v: List[float]) -> Dict:
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v)}


def summarize(args) -> Dict:
    root = Path(args.out)
    runs: Dict = {}
    budget_round = None
    for p in sorted(root.glob("*__*__seed*/result.json")):
        sc, var, sd = p.parent.name.split("__")
        runs.setdefault((sc, var), {})[int(sd[4:])] = read_json(p)
        if budget_round is None:
            c = read_json(p.parent / "config.json")
            budget_round = c["warmup_rounds"] + c["training"]["branch_rounds"]
    summary: Dict = {"budget_round_pilot2": budget_round, "cells": {}}
    for (sc, var), seeds in sorted(runs.items()):
        curves: Dict[int, List[Dict]] = {}
        for res in seeds.values():
            for ev in res["evals"]:
                curves.setdefault(ev["round"], []).append(_effects(ev))
        cell = {"seeds": sorted(seeds), "curve": {}}
        for rnd, effs in sorted(curves.items()):
            cell["curve"][rnd] = {f"{k}.{part}": _stat([e[k][part] for e in effs])
                                  for k in KEYS for part in ("blue", "green", "blue_minus_green")}
        summary["cells"][f"{sc}/{var}"] = cell
    power: Dict = {}
    for var in VARIANTS:
        cell = summary["cells"].get(f"honest/{var}")
        if not cell:
            continue
        for label, rnd in (("pilot2_budget", budget_round), ("final", max(cell["curve"]))):
            row = cell["curve"].get(rnd)
            if not row:
                continue
            for key in ("accuracy.blue_minus_green", "controlled_selected.blue_minus_green", "accuracy.blue", "controlled_selected.blue"):
                sd = row[key]["sd"]
                power.setdefault(var, {}).setdefault(label, {})[key] = {
                    "round": rnd, "null_sd_between_seeds": sd, "mde_two_arms": {n: mde(sd, n) for n in (3, 5, 10, 20)}}
    summary["power_sim_null"] = power
    write_json(root / "summary.json", summary)
    (root / "REPORT.md").write_text(render(summary))
    print((root / "REPORT.md").read_text())
    return summary


def _fmt(x: Dict) -> str:
    return f"{x['mean']:.2f}" + (f"±{x['sd']:.2f}" if x.get("sd") is not None else "")


def render(s: Dict) -> str:
    L = ["# Stage 0a – router positive control (simulated LLM)", "",
         f"pilot2 training budget = round {s['budget_round_pilot2']} (warm-up + branch rounds).", "",
         "Mean ± SD over seeds; argmax eval on fixed eval tasks; ctrl = controlled role (oracle in oracle_* scenarios).", ""]
    for name, cell in s["cells"].items():
        L += [f"## {name}  (seeds {cell['seeds']})", "",
              "| round | acc blue | acc green | ctrl_sel blue | ctrl_sel green | Δsel (b−g) | ctrl_deliv blue | ctrl_deliv green | verif_sel blue |",
              "|---|---|---|---|---|---|---|---|---|"]
        for rnd, r in cell["curve"].items():
            L.append(f"| {rnd} | {_fmt(r['accuracy.blue'])} | {_fmt(r['accuracy.green'])} | {_fmt(r['controlled_selected.blue'])} | "
                     f"{_fmt(r['controlled_selected.green'])} | {_fmt(r['controlled_selected.blue_minus_green'])} | "
                     f"{_fmt(r['controlled_delivered.blue'])} | {_fmt(r['controlled_delivered.green'])} | {_fmt(r['verifier_selected.blue'])} |")
        L.append("")
    L += ["## Power (simulated null = honest scenario)", "",
          "MDE = minimal detectable difference between two arms (alpha=0.05 two-sided, power 0.8, normal approximation) "
          "for n seeds per arm, from the between-seed SD under the null. LLM evaluation noise (stage 0b) comes on top.", ""]
    for var, labels in s["power_sim_null"].items():
        for label, keys in labels.items():
            for key, v in keys.items():
                sd = v["null_sd_between_seeds"]
                L.append(f"- {var} / {label} (round {v['round']}) {key}: SD={'-' if sd is None else f'{sd:.3f}'}"
                         + ("" if sd is None else "; MDE " + ", ".join(f"n={n}: {x:.3f}" for n, x in v["mde_two_arms"].items())))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--scenario", choices=SCENARIOS, required=True)
    r.add_argument("--variant", choices=VARIANTS, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--profile", default="pilot2")
    r.add_argument("--rounds", type=int, default=60, help="total training rounds incl. warm-up")
    r.add_argument("--warmup-rounds", type=int, default=6)
    r.add_argument("--eval-rounds", default="0,6,12,18,30,45,60")
    r.add_argument("--eval-reps", type=int, default=1)
    r.add_argument("--n-train", type=int, default=160)
    r.add_argument("--n-eval", type=int, default=40)
    r.add_argument("--task-seed", type=int, default=0)
    r.add_argument("--texts", default=None, help="splits JSON with real problem texts (optional)")
    r.add_argument("--p-solve", type=float, default=0.35)
    r.add_argument("--p-follow", type=float, default=0.3)
    r.add_argument("--p-inspect", type=float, default=0.8)
    r.add_argument("--p-final-solo", type=float, default=0.35)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=run)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.set_defaults(func=summarize)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
