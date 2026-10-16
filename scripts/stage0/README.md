# Stage 0 – interpretability prerequisites

Stage 0 checks two things before the core experiment. Without them a null result would not be interpretable.

- Can the router learn marker-dependent trust at all? This is the positive control (0a).
- How large is the noise floor of an evaluation? This is the noise floor and power analysis (0b).

Both stages run in their own allocation. Stage 0b uses its own Ollama server on port 11437. Production results under `results/pilot2/` are only read, never written, and the shared `budget.json` is never touched.

## 0a – positive control in simulation (`sim.py`)

The real `Graph`, `Trainer`, `run_batch` and `compute_grad_inputs` code runs unchanged. Only the LLM is replaced by a simulated one, registered as `mock`.

**Simulated worker behaviour**

- Workers solve a task with probability `p_solve` and follow the majority of the hints with probability `p_follow`.
- The Inspector confirms a correct hint with probability `p_inspect`.
- The decision-maker takes the majority vote over the delivered hints.

**Scenarios**

All scenarios are active only after the warm-up, as in the real pipeline.

| Scenario | Controlled role (Mathematical Analyst) |
|---|---|
| `honest` | normal; this is the null |
| `oracle_uncond` | always correct |
| `oracle_cond` | correct under `blue`, wrong under `green`; this is the actual positive control for marker-dependent trust |
| `withhold_green` | real `Controller` withholds under `green` |

**Router variants**

The variants are runtime patches that exist only inside the simulation process.

| Variant | Change |
|---|---|
| `original` | upstream indexing, unchanged |
| `indices` | role scores taken from tokens `1..n`, hint offset `n+1` |
| `full` | `indices`, plus the decision step is included in the policy gradient; the agent term is skipped when the decision is forced |

**Interpretation**

- If `oracle_cond` shows no Δsel (blue−green) after the pilot2 budget (round 18) or later, the router cannot learn this effect. A null result in experiment 1 is then not interpretable.
- `honest` gives the between-seed SD, which is used for the minimal detectable effect (MDE).

```bash
python scripts/stage0/sim.py run --out OUT --scenario oracle_cond --variant full --seed 0
python scripts/stage0/sim.py summarize --out OUT        # -> OUT/REPORT.md, OUT/summary.json
```

## 0b – LLM noise floor (`noise_floor.py`)

Stage 0b re-runs the unchanged experiment-1 evaluation (`pipeline.evaluate`) R times on existing pilot2 checkpoints. By default these are `branch_A` and `branch_B` at round 12.

**Inputs**

- The production `llm_limit.json`.
- The pilot2 server settings.
- The existing `exp1_<B>.jsonl`, used only as a reference repetition.

**Report contents**

- SD per cell between repetitions, compared with the binomial SE.
- Per-episode flip rates (answer, correctness, routing).
- SD of the effects (e.g. accuracy blue−green under attack).
- An MDE table: `MDE = 2.80 · sqrt(2 · (σ_train² + σ_eval²/R) / n)`, where σ_train comes from 0a if available, otherwise 0 is used as a lower bound.

Each evaluation takes about 3.7 h on qwen30 (A30). It resumes per episode and starts no new repetition after `--hours`.

## Ramses

```bash
# 0a: complete grid (4 scenarios x 3 variants x 5 seeds x 60 rounds), about 3–5 h
srun -p interactive --gres=gpu:1 --cpus-per-task=16 --mem=64GB --time=10:00:00 --pty /bin/bash
bash scripts/stage0/run.sh sim --jobs 8            # -> $RESULTS_DIR/stage0/sim_synthetic/REPORT.md
bash scripts/stage0/run.sh sim --jobs 8 --texts    # optional: real MASBench task texts

# 0b: 2 checkpoints x 2 repetitions, about 15 h
srun -p interactive --gres=gpu:1 --cpus-per-task=8 --mem=64GB --time=24:00:00 --pty /bin/bash
bash scripts/stage0/run.sh noise --reps 2 --hours 22   # -> $RESULTS_DIR/stage0/noise_pilot2_qwen30_seed0/REPORT.md

bash scripts/stage0/run.sh report                  # regenerate all reports (0b uses the 0a σ_train)
```

Tests: `pytest scripts/stage0`
