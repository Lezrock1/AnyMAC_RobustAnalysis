"""Stage 0b: LLM evaluation noise floor + power analysis on real checkpoints.

Re-runs the unchanged experiment-1 evaluation (pipeline.evaluate, argmax routing, attack+honest x blue+green) of
existing checkpoints R times into an ISOLATED directory. Source results are only read (checkpoint, llm_limit.json,
optionally the existing exp1_<B>.jsonl as a reference repetition); the shared budget.json is never touched.

Commands:
  run     repeat evaluations (resumable per episode; finished repetitions are skipped)
  report  per-cell SD across repetitions, per-episode flip rates, SD of the effects, MDE table -> REPORT.md
"""
import argparse
import hashlib
import itertools
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from robust.budget import Budget  # noqa: E402
from robust.common import read_json, read_jsonl, write_json  # noqa: E402

CELLS = [f"{p}/{m}" for p in ("attack", "honest") for m in ("blue", "green")]
METRICS = ("correct", "target_success", "controlled_selected", "controlled_delivered", "verifier_selected",
           "verifier_delivered", "decision_sees_all")
EFFECTS = {
    "acc_attack_blue_minus_green": ("correct", "attack/blue", "attack/green"),
    "target_attack_blue_minus_green": ("target_success", "attack/blue", "attack/green"),
    "ctrl_sel_attack_blue_minus_green": ("controlled_selected", "attack/blue", "attack/green"),
    "ctrl_deliv_attack_blue_minus_green": ("controlled_delivered", "attack/blue", "attack/green"),
    "acc_honest_minus_attack_blue": ("correct", "honest/blue", "attack/blue"),
    "acc_honest_blue_minus_green": ("correct", "honest/blue", "honest/green"),
}
Z = 1.959964 + 0.841621  # alpha 0.05 two-sided, power 0.8


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_label(spec: str) -> str:
    return spec.split("/")[0]  # branch_A/round_0012.pth -> branch_A


# ---------------------------------------------------------------- run
def run(a) -> None:
    from robust import pipeline
    from robust.profile import load_profile
    from robust.data import splits_path
    from robust.rollout import load_graph
    src, out = Path(a.source).resolve(), Path(a.out).resolve()
    if out == src or src in out.parents:
        sys.exit("--out must not be inside the source result directory")
    out.mkdir(parents=True, exist_ok=True)
    lim = src / "llm_limit.json"
    if not lim.exists():
        sys.exit(f"{lim} missing (needed so all repetitions use the production generation limits)")
    if not (out / "llm_limit.json").exists():
        shutil.copy2(lim, out / "llm_limit.json")
    prof = load_profile(a.profile)
    sp = splits_path(prof["name"], prof["data_seed"])
    if not sp.exists():
        sys.exit(f"{sp} missing; refusing to create new splits (the noise floor must use the production eval tasks)")
    ctx = pipeline.Ctx(prof, a.model, a.seed, root=out, budget=Budget(out / "budget.json", None))
    tasks = ctx.tasks("eval")[: a.n_tasks] if a.n_tasks else ctx.tasks("eval")
    meta = read_json(out / "meta.json", None) or {"source": str(src), "profile": a.profile, "model": a.model, "seed": a.seed,
                                                  "llm_limit": read_json(out / "llm_limit.json"), "ckpts": {}, "reps": {}}
    for spec in a.ckpts:
        p = src / spec
        if not p.exists():
            sys.exit(f"checkpoint {p} missing")
        digest = sha256(p)
        old = meta["ckpts"].get(ckpt_label(spec))
        if old and old["sha256"] != digest:
            sys.exit(f"{p} changed since the first repetition ({old['sha256'][:12]} -> {digest[:12]}); use a new --out")
        meta["ckpts"][ckpt_label(spec)] = {"path": str(p), "sha256": digest, "reference": a.reference.get(ckpt_label(spec))}
    meta["n_tasks"] = len(tasks)
    write_json(out / "meta.json", meta)
    n_expected = len(tasks) * 4
    for rep, spec in itertools.product(range(1, a.reps + 1), a.ckpts):  # interleaved: partial runs stay balanced
        lab = ckpt_label(spec)
        path = out / "eval" / f"{lab}_rep{rep}.jsonl"
        done = len({(r["task_id"], r["marker"], r["policy"]) for r in read_jsonl(path)})
        if done >= n_expected:
            continue
        if a.deadline and time.time() > a.deadline:
            print(f"deadline reached before {lab} rep{rep}; rerun the same command to resume", flush=True)
            break
        t0 = time.time()
        graph = load_graph(str(src / spec))
        pipeline.evaluate(ctx, graph, f"noise_{lab}_rep{rep}", tasks, out_path=path)
        meta["reps"][f"{lab}_rep{rep}"] = {"wall_s": time.time() - t0, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(out / "meta.json", meta)
        print(f"[noise] {lab} rep{rep} done in {(time.time() - t0) / 3600:.2f} h", flush=True)
        report(argparse.Namespace(out=str(out), sim_summary=a.sim_summary, quiet=True))


# ---------------------------------------------------------------- report
def _rows_by_key(rows: List[Dict]) -> Dict:
    return {(r["task_id"], r["marker"], r["policy"]): r for r in rows}


def _cell_means(rows: Dict) -> Dict:
    out = {}
    for cell in CELLS:
        pol, mk = cell.split("/")
        sel = [r for (t, m, p), r in rows.items() if p == pol and m == mk]
        out[cell] = {k: (float(np.mean([bool(r.get(k)) for r in sel])) if sel else None) for k in METRICS}
        out[cell]["n"] = len(sel)
        out[cell]["errors"] = sum(1 for r in sel if r.get("error"))
    return out


def _sd(v: List[float]) -> Optional[float]:
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def analyse_ckpt(reps: Dict[str, Dict]) -> Dict:
    """reps: name -> {key: row}; only episodes present in every repetition are compared."""
    common = set.intersection(*(set(r) for r in reps.values())) if reps else set()
    reps = {n: {k: r[k] for k in common} for n, r in reps.items()}
    cells = {n: _cell_means(r) for n, r in reps.items()}
    eff = {n: {e: (c[a][m] - c[b][m]) if c[a][m] is not None and c[b][m] is not None else None
               for e, (m, a, b) in EFFECTS.items()} for n, c in cells.items()}
    res: Dict = {"reps": sorted(reps), "n_common_episodes": len(common), "cells": {}, "effects": {}, "flip": {}}
    for cell in CELLS:
        res["cells"][cell] = {}
        for k in METRICS:
            v = [cells[n][cell][k] for n in reps if cells[n][cell][k] is not None]
            if v:
                p, n_ep = float(np.mean(v)), cells[next(iter(reps))][cell]["n"]
                res["cells"][cell][k] = {"values": v, "mean": p, "sd_between_reps": _sd(v),
                                         "binomial_se_single_eval": float(np.sqrt(p * (1 - p) / n_ep)) if n_ep else None}
    for e in EFFECTS:
        v = [eff[n][e] for n in reps if eff[n][e] is not None]
        res["effects"][e] = {"values": v, "mean": float(np.mean(v)) if v else None, "sd_between_reps": _sd(v)}
    if len(reps) > 1:
        names = sorted(reps)
        for field, get in (("correct", lambda r: bool(r.get("correct"))), ("pred", lambda r: r.get("pred")),
                           ("routing", lambda r: tuple(r.get("selected_roles") or ())),
                           ("controlled_selected", lambda r: bool(r.get("controlled_selected")))):
            any_flip = np.mean([len({get(reps[n][k]) for n in names}) > 1 for k in common]) if common else None
            pair = [np.mean([get(reps[x][k]) != get(reps[y][k]) for k in common]) for x, y in itertools.combinations(names, 2)]
            res["flip"][field] = {"any_rep_differs": None if any_flip is None else float(any_flip),
                                  "pairwise_mean": float(np.mean(pair)) if pair else None}
    return res


def mde_table(sigma_eval: Optional[float], sigma_train: float, r_evals: int = 1) -> Dict[int, Optional[float]]:
    """MDE between two arms with n independently trained seeds per arm, each evaluated r_evals times."""
    if sigma_eval is None:
        return {}
    var = sigma_train ** 2 + sigma_eval ** 2 / r_evals
    return {n: float(Z * np.sqrt(2 * var / n)) for n in (3, 5, 10, 20)}


def report(a) -> Dict:
    out = Path(a.out)
    meta = read_json(out / "meta.json")
    files = sorted((out / "eval").glob("*_rep*.jsonl"))
    by_ckpt: Dict[str, Dict] = {}
    for f in files:
        lab, rep = f.stem.rsplit("_rep", 1)
        rows = _rows_by_key(read_jsonl(f))
        if len(rows) >= meta["n_tasks"] * 4:  # only complete repetitions
            by_ckpt.setdefault(lab, {})[f"rep{rep}"] = rows
    sim_sd = {}
    if a.sim_summary and Path(a.sim_summary).exists():
        s = read_json(Path(a.sim_summary))
        row = s["cells"].get("honest/full", {}).get("curve", {}).get(str(s["budget_round_pilot2"]), {})
        sim_sd = {"accuracy.blue_minus_green": (row.get("accuracy.blue_minus_green") or {}).get("sd"),
                  "controlled_selected.blue_minus_green": (row.get("controlled_selected.blue_minus_green") or {}).get("sd")}
    result: Dict = {"meta": meta, "ckpts": {}, "sim_between_seed_sd": sim_sd}
    for lab, reps in by_ckpt.items():
        own = analyse_ckpt(reps)
        ref_path = (meta["ckpts"].get(lab) or {}).get("reference")
        with_ref = None
        if ref_path and Path(ref_path).exists():
            ref = _rows_by_key([r for r in read_jsonl(Path(ref_path)) if r.get("task_id")])
            if len(ref) >= meta["n_tasks"] * 4:
                with_ref = analyse_ckpt({**reps, "reference": ref})
        result["ckpts"][lab] = {"own": own, "with_reference": with_ref}
    # pooled eval SD of each effect over checkpoints (own repetitions only)
    pooled = {}
    for e in EFFECTS:
        num = den = 0.0
        for c in result["ckpts"].values():
            v = c["own"]["effects"][e]["values"]
            if len(v) > 1:
                num += (len(v) - 1) * float(np.var(v, ddof=1))
                den += len(v) - 1
        pooled[e] = {"sd": float(np.sqrt(num / den)) if den else None, "df": den}
    result["pooled_eval_sd"] = pooled
    sig_train = {"acc_attack_blue_minus_green": sim_sd.get("accuracy.blue_minus_green"),
                 "ctrl_sel_attack_blue_minus_green": sim_sd.get("controlled_selected.blue_minus_green")}
    result["mde"] = {e: {f"sigma_train={st if st is not None else 0:.3f}": mde_table(pooled[e]["sd"], st or 0.0)
                         for st in {0.0, sig_train.get(e) or 0.0}} for e in EFFECTS}
    write_json(out / "summary.json", result)
    (out / "REPORT.md").write_text(render(result))
    if not getattr(a, "quiet", False):
        print((out / "REPORT.md").read_text())
    return result


def _f(x: Optional[float], d: int = 3) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def render(r: Dict) -> str:
    m = r["meta"]
    L = ["# Stage 0b – LLM evaluation noise floor", "",
         f"Source: `{m['source']}` (read-only), profile {m['profile']}, model {m['model']}, seed {m['seed']}, "
         f"{m['n_tasks']} eval tasks x 2 markers x 2 policies per repetition.", "",
         "SD = between repetitions of the identical evaluation (same checkpoint, argmax routing). "
         "Binomial SE = what a single evaluation would have if episodes were independent coin flips.", ""]
    for lab, c in r["ckpts"].items():
        for kind in ("own", "with_reference"):
            a = c[kind]
            if not a:
                continue
            L += [f"## {lab} – {kind} ({', '.join(a['reps'])}; {a['n_common_episodes']} common episodes)", "",
                  "| cell | accuracy (per rep) | SD | binom. SE | ctrl_sel (per rep) | SD | target (per rep) |", "|---|---|---|---|---|---|---|"]
            for cell, v in a["cells"].items():
                acc, cs, ts = v.get("correct", {}), v.get("controlled_selected", {}), v.get("target_success", {})
                L.append(f"| {cell} | {' / '.join(_f(x, 2) for x in acc.get('values', []))} | {_f(acc.get('sd_between_reps'))} | "
                         f"{_f(acc.get('binomial_se_single_eval'))} | {' / '.join(_f(x, 2) for x in cs.get('values', []))} | "
                         f"{_f(cs.get('sd_between_reps'))} | {' / '.join(_f(x, 2) for x in ts.get('values', []))} |")
            L += ["", "| effect | per rep | mean | SD |", "|---|---|---|---|"]
            for e, v in a["effects"].items():
                L.append(f"| {e} | {' / '.join(_f(x, 3) for x in v['values'])} | {_f(v['mean'])} | {_f(v['sd_between_reps'])} |")
            if a["flip"]:
                L += ["", "Per-episode flips (same task/marker/policy, different repetition): "
                      + "; ".join(f"{k}: any={_f(v['any_rep_differs'], 2)}, pairwise={_f(v['pairwise_mean'], 2)}"
                                  for k, v in a["flip"].items())]
            L.append("")
    L += ["## Power", "",
          "Pooled eval SD of each effect (own repetitions). MDE = minimal detectable difference between two arms "
          "(alpha 0.05 two-sided, power 0.8) with n independently trained seeds per arm, one evaluation each; "
          "sigma_train from the stage-0a simulation (honest/full, budget round) if available, else 0 (= lower bound).", ""]
    for e, rows in r["mde"].items():
        sd = r["pooled_eval_sd"][e]
        for lab, t in rows.items():
            if t:
                L.append(f"- {e}: eval SD={_f(sd['sd'])} (df={sd['df']:.0f}), {lab}: "
                         + ", ".join(f"n={n}: {x:.3f}" for n, x in t.items()))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", required=True, help="existing result root, e.g. results/pilot2/qwen30/seed0 (read-only)")
    r.add_argument("--out", required=True)
    r.add_argument("--profile", default="pilot2")
    r.add_argument("--model", default="qwen30")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ckpts", nargs="+", default=["branch_A/round_0012.pth", "branch_B/round_0012.pth"])
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--n-tasks", type=int, default=None, help="only the first N eval tasks (testing)")
    r.add_argument("--deadline", type=float, default=None, help="unix time; no new repetition starts after it")
    r.add_argument("--sim-summary", default=None)
    r.set_defaults(func=run)
    s = sub.add_parser("report")
    s.add_argument("--out", required=True)
    s.add_argument("--sim-summary", default=None)
    s.set_defaults(func=report)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        src = Path(a.source)
        a.reference = {ckpt_label(c): str(src / "eval" / f"exp1_{ckpt_label(c).split('_')[-1]}.jsonl")
                       for c in a.ckpts if ckpt_label(c).startswith("branch_")}
    return a.func(a)


if __name__ == "__main__":
    main()
