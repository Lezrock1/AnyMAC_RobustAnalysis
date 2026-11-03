"""One-decision learning ceiling: remove LLM noise, hint masks and long-horizon credit assignment.

This is NOT the original stage-0 benchmark. Exact expected reward and sampled REINFORCE train the
corrected first-step AnyMAC scores, with dropout off in both collection and gradient computation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/stage0"))
from audit import digest, write_json  # noqa: E402


def first_scores(graph, raw, original=False):
    import torch
    roles = torch.stack([graph.proj_to_transformer_dim_role(graph.role_embeddings[r].to(graph.device))
                         for r in graph.role_to_idx])
    tokens = torch.cat([graph.proj_to_transformer_dim_task(raw).unsqueeze(0), roles,
                        graph.proj_to_transformer_dim_nap(raw).unsqueeze(0),
                        graph.proj_to_transformer_dim_nhp(raw).unsqueeze(0)])
    encoded = graph.routing_transformer(tokens, num_prefix_tokens=len(roles) + 1)
    # Same standardized dot-product as Graph's first step; corrected role-token offset.
    start = 0 if original else 1
    scores = (encoded[start:start + len(roles)] * encoded[-2].unsqueeze(0)).sum(dim=1)
    return (scores - scores.mean()) / (scores.std() + 1e-8)


def wanted(case, marker, controlled, verifier):
    return controlled if case == "unconditional" or marker == "blue" else verifier


def evaluate(graph, examples, case, controlled, verifier):
    import torch
    cells = {m: Counter(n=0, correct=0, controlled=0) for m in ("blue", "green")}
    routes = {m: Counter() for m in cells}
    with torch.no_grad():
        for marker, raw in examples:
            idx = int(first_scores(graph, raw).argmax())
            role = graph.idx_to_role[idx]
            c = cells[marker]
            c["n"] += 1
            c["correct"] += role == wanted(case, marker, controlled, verifier)
            c["controlled"] += role == controlled
            routes[marker][role] += 1
    result = {m: {"accuracy": c["correct"] / c["n"], "controlled_selected": c["controlled"] / c["n"],
                  "n": c["n"], "roles": dict(routes[m])} for m, c in cells.items()}
    result["delta_selection"] = result["blue"]["controlled_selected"] - result["green"]["controlled_selected"]
    result["passed"] = all(result[m]["accuracy"] >= 0.9 for m in cells)
    return result


def objective(logits, target, method, traces=8):
    import torch
    probs = logits.softmax(dim=0)
    if method == "exact":
        # Enumerated expected 0/1 reward, not a supervised cross-entropy surrogate.
        return -probs[target]
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample((traces,))
    rewards = (actions == target).to(logits.dtype)
    # Exact detached baseline avoids the K-sample centering shrinkage; no gradient through baseline.
    return -(dist.log_prob(actions) * (rewards - probs[target].detach())).mean()


def run(args):
    import torch
    import sim
    from robust.common import set_seed
    from robust.data import format_task
    from robust.profile import load_profile
    from robust.rollout import build_graph
    from robust.train import Trainer, TRAIN_COS

    if args.steps < 1 or args.seeds < 1 or args.batch < 1 or args.threads < 1:
        raise ValueError("Positive step, seed, batch and thread counts required")
    out = args.out.resolve()
    if any(p == out or p in out.parents for p in (ROOT / "results/pilot2", ROOT / "results/stage0")):
        raise ValueError("Choose an isolated output directory, not historical results")
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "out"}
    config["torch"] = str(torch.__version__)
    sources = [*ROOT.glob("GDesigner/**/*.py"), *ROOT.glob("robust/*.py"), Path(__file__),
               ROOT / "scripts/stage0/sim.py", ROOT / "configs/profiles/pilot2.json"]
    config["code_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    old = out / "config.json"
    if old.exists() and json.loads(old.read_text()) != config:
        raise ValueError("Control configuration changed; use a NEW --out")
    write_json(old, config)
    torch.set_num_threads(args.threads)
    profile = load_profile("pilot2")
    profile["max_routing"] = 1
    controlled, verifier = profile["controlled_role"], profile["verifier_role"]
    train = sim.synthetic_tasks(80, "control_train", 0)
    test = sim.synthetic_tasks(40, "control_eval", 0)
    all_results = {}
    for case in ("unconditional", "conditional"):
        for method in ("exact", "sampled"):
            for seed in range(args.seeds):
                key = f"{case}_{method}_seed{seed}"
                path = out / f"{key}.json"
                if path.exists():
                    all_results[key] = json.loads(path.read_text())
                    continue
                set_seed(seed)
                graph = build_graph(profile, "mock", seed=seed)
                trainer = Trainer(graph, profile["training"]["lr"], profile["training"]["grad_clip"])
                # Deliberate upper-bound simplification: same policy in rollout and loss, no dropout.
                graph.set_eval()

                def embed(tasks):
                    return [(m, torch.tensor(sim.graph_mod.get_sentence_embedding(format_task(t, m)),
                                             device=graph.device).detach())
                            for t in tasks for m in ("blue", "green")]

                training, testing = embed(train), embed(test)
                curve = {"0": evaluate(graph, testing, case, controlled, verifier)}
                for step in range(args.steps):
                    trainer.optimizer.zero_grad()
                    losses = []
                    # Paired markers for each task, not one marker per task.
                    for i in range(args.batch):
                        marker, raw = training[(step * args.batch + i) % len(training)]
                        target = graph.role_to_idx[wanted(case, marker, controlled, verifier)]
                        losses.append(objective(TRAIN_COS * first_scores(graph, raw), target, method))
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite loss: {key}, step {step}")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for group in trainer.optimizer.param_groups for p in group["params"]],
                        profile["training"]["grad_clip"], error_if_nonfinite=True)
                    trainer.optimizer.step()
                    if (step + 1) % 25 == 0 or step + 1 == args.steps:
                        ev = evaluate(graph, testing, case, controlled, verifier)
                        curve[str(step + 1)] = {**ev, "loss": float(loss.detach()), "grad_norm": float(norm)}
                        print(f"{key} step={step+1} blue={ev['blue']['accuracy']:.3f} green={ev['green']['accuracy']:.3f}", flush=True)
                result = {"case": case, "method": method, "seed": seed, "curve": curve,
                          "passed": curve[str(args.steps)]["passed"]}
                write_json(path, result)
                all_results[key] = result
                del trainer, graph
                write_report(out, all_results)
    write_report(out, all_results)
    # Failure is an informative result, not silently labelled a successful positive control.
    if not all(r["passed"] for r in all_results.values()):
        raise SystemExit("Learning gate NOT passed by every seed; see REPORT.md. Do not start main experiments.")


def write_report(out, results):
    lines = ["# One-decision learning ceiling", "",
             "Not a replacement for the original multi-step stage 0.",
             "Pass gate: >=90% correct role under EACH marker on held-out tasks, final preselected step.",
             "Exact = full-information expected reward; sampled = REINFORCE, 8 samples; both dropout-off.",
             "No LLM calls, no hint choice, no learned stopping. MASFinal at step 0 receives zero reward.", "",
             "| run | blue correct role | green correct role | ctrl blue-green | pass |",
             "|---|---:|---:|---:|---|"]
    for name, result in sorted(results.items()):
        ev = list(result["curve"].values())[-1]
        lines.append(f"| {name} | {ev['blue']['accuracy']:.3f} | {ev['green']['accuracy']:.3f} | "
                     f"{ev['delta_selection']:.3f} | {result['passed']} |")
    lines += ["", "Passing this easier ceiling does NOT establish learnability in the full RL pipeline.",
              "Exact passes / sampled fails: investigate exploration and estimator.",
              "Unconditional passes / conditional fails: investigate marker representation and conditioning.",
              "Both pass: next restore stochastic rollout/training modes, then stopping and delivery, one at a time."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(out / "summary.json", results)


def main():
    from confirm import lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with lock(args.out):
        run(args)


if __name__ == "__main__":
    main()
