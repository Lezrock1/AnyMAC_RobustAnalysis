"""Snapshot and rescore finished pilot results without importing the model stack."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
from diagnose import load_rows, strict_answer  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def outside(out, sources):
    out = out.resolve()
    for source in sources:
        source = source.resolve()
        if out == source or source in out.parents:
            raise ValueError(f"Output must be outside source {source}")
    return out


def snapshot(repo, out):
    repo = repo.resolve()
    sources = [repo / "results/pilot2", repo / "results/stage0"]
    out = outside(out, sources)
    if not all(p.is_dir() for p in sources):
        raise FileNotFoundError("Both results/pilot2 and results/stage0 are required")
    # Intermediate model and optimizer states are deliberately omitted.
    paths = [p for source in sources for p in source.rglob("*")
             if p.is_file() and not p.is_symlink() and p.suffix not in (".pth", ".opt")]
    splits = repo / "data/splits/splits_pilot2_seed0.json"
    if not splits.is_file():
        raise FileNotFoundError(splits)
    paths.append(splits)
    for folder in ("robust", "GDesigner", "scripts", "configs/profiles"):
        paths.extend(p for p in (repo / folder).rglob("*")
                     if p.is_file() and not p.is_symlink() and p.suffix in (".py", ".sh", ".json", ".md"))
    seed = repo / "results/pilot2/qwen30/seed0"
    for stage in ("branch_A", "branch_B", "branch_C", "continue_A", "continue_B", "continue_C"):
        progress = json.loads((seed / stage / "progress.json").read_text())
        ckpt = seed / stage / f"round_{progress['rounds_done']:04d}.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        paths.append(ckpt)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {str(p.relative_to(repo)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(set(paths))}
    archive = out / "snapshot.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for name in manifest:
            tar.add(repo / name, arcname=name, recursive=False)
    # Verify the bytes inside the archive, not just the live source files.
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            f = tar.extractfile(member)
            if f is None:
                raise ValueError(f"Unexpected non-file archive member: {member.name}")
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
            if h.hexdigest() != manifest[member.name]["sha256"]:
                raise ValueError(f"Source changed during snapshot: {member.name}; discard this snapshot")
    write_json(out / "manifest.json", manifest)
    (out / "SHA256SUMS").write_text(f"{digest(archive)}  snapshot.tar.gz\n")
    print(f"Verified {len(manifest)} files, {archive.stat().st_size / 2**20:.1f} MiB: {archive}")
    return archive


def episode_key(row):
    return row["task_id"], row["marker"], row["policy"]


def score(row):
    pred = strict_answer(row.get("final_text") or "")
    events = row.get("events") or []
    dec = events[-1] if events and events[-1].get("is_decision") else None
    # Decision event usage also includes extra calls; do not call it final-only truncation.
    return {"strict_pred": pred, "strict_correct": pred is not None and pred == row["truth"],
            "strict_target_success": pred is not None and pred == row["target"],
            "format_valid": pred is not None, "has_decision": dec is not None,
            "any_call_truncated": bool(row.get("usage", {}).get("truncated")),
            "decision_stage_truncated": bool((dec or {}).get("usage", {}).get("truncated"))}


def audit_eval(path, expected=None):
    rows, invalid = load_rows(path)
    counts = Counter(rows=len(rows), invalid_lines=len(invalid))
    keys, routes, cells = Counter(), Counter(), defaultdict(Counter)
    sidecar = []
    for number, row in enumerate(rows, 1):
        try:
            key = episode_key(row)
            scored = score(row)
        except (KeyError, TypeError, AttributeError) as e:
            invalid.append({"row": number, "error": f"Invalid episode schema: {e}"})
            continue
        keys[key] += 1
        c = cells[f"{key[2]}/{key[1]}"]
        c["n"] += 1
        for name, yes in scored.items():
            if name != "strict_pred":
                c[name] += bool(yes)
        c["stored_correct"] += bool(row.get("correct"))
        c["stored_target_success"] += bool(row.get("target_success"))
        c["parser_disagrees"] += row.get("pred") != scored["strict_pred"]
        c["episode_errors"] += bool(row.get("error"))
        c["backend_errors"] += int(row.get("usage", {}).get("errors", 0))
        c["controlled_selected"] += bool(row.get("controlled_selected"))
        c["controlled_delivered"] += bool(row.get("controlled_delivered"))
        c["manipulated_messages"] += len(row.get("manipulations") or [])
        events = row.get("events") or []
        dec = events[-1] if scored["has_decision"] else {}
        c["missing_extra_call"] += "exp4_" in path.stem and not path.stem.endswith("_plain") and not dec.get("extra_calls")
        roles = tuple(row.get("selected_roles") or [])
        routes[" -> ".join(roles) or "(no worker)"] += 1
        sidecar.append({"source_row": number, "task_id": key[0], "marker": key[1], "policy": key[2],
                        "stored_pred": row.get("pred"), **scored})
    counts["invalid_lines"] = len(invalid)
    counts["duplicate_keys"] = sum(n - 1 for n in keys.values())
    if expected is not None:
        counts["missing_keys"] = len(expected - set(keys))
        counts["unexpected_keys"] = len(set(keys) - expected)
    return {"counts": dict(counts), "invalid": invalid, "cells": dict(cells),
            "routes": dict(routes.most_common())}, sidecar


def audit_simulation(root):
    result = {}
    for path in sorted(root.glob("sim_*/*__*__seed*/result.json")):
        res = json.loads(path.read_text())
        rounds_path = path.parent / "rounds.jsonl"
        rounds, invalid = load_rows(rounds_path)
        result[str(path.parent.relative_to(root))] = {
            "sha256": digest(path), "invalid_training_lines": invalid,
            "attempted_updates": res["attempted_updates"], "applied_updates": res["applied_updates"],
            "tasks_without_advantage": sum(r["tasks_without_advantage"] for r in rounds),
            "zero_gradient_updates": sum(g == 0 for r in rounds for g in r["grad_norm"] if g is not None),
            "trace_lengths": dict(sum((Counter(r["trace_lengths"]) for r in rounds), Counter())),
            "evals": res["evals"]}
    return result


def audit(pilot, out, splits=None):
    pilot = pilot.resolve()
    out = outside(out, [pilot])
    out.mkdir(parents=True, exist_ok=False)
    expected = None
    if splits:
        sp = json.loads(splits.read_text())
        expected = {(t, m, p) for t in sp["ids"]["eval"]
                    for m in ("blue", "green") for p in ("attack", "honest")}
    report = {"source": str(pilot), "files": {}, "training": {}, "interventions": {},
              "simulation": audit_simulation(pilot.parent / "stage0"),
              "limitations": ["Strict parsing checks output format, not semantic correctness of the reasoning.",
                              "Decision-stage usage includes extra calls.",
                              "Training duplicates are reported, never silently removed."]}
    fatal = []
    paths = [(p, str(p.relative_to(pilot))) for p in sorted(pilot.glob("*/seed*/eval/exp*.jsonl"))]
    if not paths:
        raise FileNotFoundError(f"No experiment episodes under {pilot}")
    for seed in sorted(pilot.glob("*/seed*")):
        for label in ("exp1_A", "exp1_B", "exp1_C", "exp3_A", "exp3_B", "exp3_C",
                      "exp4_B_plain", "exp4_B_mandatory_verifier", "exp4_B_extra_ordinary"):
            if not (seed / "eval" / f"{label}.jsonl").is_file():
                fatal.append(str((seed / "eval" / f"{label}.jsonl").relative_to(pilot)) + " MISSING")
    stage0 = pilot.parent / "stage0"
    paths.extend((p, "stage0/" + str(p.relative_to(stage0)))
                 for p in sorted(stage0.glob("noise_*/eval/*_rep*.jsonl")))
    for path, rel in paths:
        result, sidecar = audit_eval(path, expected)
        result["sha256"] = digest(path)
        report["files"][rel] = result
        target = out / "rescored" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r) + "\n" for r in sidecar))
        if any(result["counts"].get(k) for k in ("invalid_lines", "duplicate_keys", "missing_keys", "unexpected_keys")):
            fatal.append(rel)
    for path in sorted(pilot.glob("*/seed*/*/train_rounds.jsonl")):
        rows, invalid = load_rows(path)
        rounds = Counter(r["round"] for r in rows)
        report["training"][str(path.relative_to(pilot))] = {
            "rows": len(rows), "invalid": invalid, "duplicate_rounds": {k: v for k, v in rounds.items() if v > 1},
            "sha256": digest(path)}
    for path in sorted(pilot.glob("*/seed*/*/train_episodes.jsonl")):
        rows, invalid = load_rows(path)
        keys = Counter((r.get("round"), r.get("task_id"), r.get("trace")) for r in rows)
        report["training"][str(path.relative_to(pilot))] = {
            "rows": len(rows), "invalid": invalid, "duplicate_keys": sum(v - 1 for v in keys.values()),
            "sha256": digest(path)}
    for path in sorted(pilot.glob("*/seed*/exp2/interventions.jsonl")):
        rows, invalid = load_rows(path)
        keys = Counter((r["task_id"], r["marker"], r["intervention"]) for r in rows)
        report["interventions"][str(path.relative_to(pilot))] = {
            "rows": len(rows), "invalid": invalid, "duplicate_keys": sum(v - 1 for v in keys.values()),
            "by_intervention": dict(Counter(r["intervention"] for r in rows)),
            "errors": sum(bool(r.get("error")) for r in rows),
            "warning": "No final texts stored here: strict rescoring of new_pred is impossible."}
    report["integrity_failures"] = fatal
    write_json(out / "summary.json", report)
    lines = ["# Frozen pilot audit", "", "Counts, not rates. Old files are never rewritten.", "",
             "| file/cell | n | stored correct | strict correct | strict target | invalid format | errors | truncated episodes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in report["files"].items():
        for cell, c in sorted(result["cells"].items()):
            lines.append(f"| {name}/{cell} | {c['n']} | {c['stored_correct']} | {c['strict_correct']} | "
                         f"{c['strict_target_success']} | {c['n'] - c['format_valid']} | {c['episode_errors']} | {c['any_call_truncated']} |")
    lines += ["", "## Integrity", "", f"Episode integrity failures: {fatal or 'none'}",
              "", "Detailed routes, duplicate training logs and missing extra calls: summary.json.",
              "Exp2 cannot be strictly rescored because intervention final texts were not saved."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((out / "REPORT.md").read_text())
    return not fatal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--repo", type=Path, default=ROOT)
    snap.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("--pilot", type=Path, required=True)
    check.add_argument("--splits", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.repo, args.out)
    elif not audit(args.pilot, args.out, args.splits):
        raise SystemExit("Integrity checks failed; see summary.json")


if __name__ == "__main__":
    main()
