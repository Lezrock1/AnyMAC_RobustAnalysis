# Model x benchmark screening (A30, honest only)

Purpose: choose the agent LLM and benchmark of the comparative dynamic-router study BEFORE any
attack or router experiment, using only honest runs. Pre-registered 2026-10-19, before any outcome.

## Design

- Models (`models.txt`, all q4 on one A30 24 GB): pilot anchor `qwen3:30b-a3b-instruct-2507`,
  `qwen3.8:27b`, `gemma4:26b-a4b-it-qat`, `glm-4.7-flash`. Nemotron 3.5 Lightning excluded (q4 = 25 GB).
- Benchmarks: MMLU (validation, `cais/mmlu@c30699e`; used by AnyMAC, G-Designer incl. its adversarial
  setup, and DyLAN), MMLU-Pro (test, `TIGER-Lab/MMLU-Pro@b189ec7`; harder, 10 options), MASBench breadth
  dev (24 tasks, pilot continuity). 40 sampled items each, seed 20261019. Screening IDs are RESERVED:
  excluded from every later evaluation.
- Conditions: `single` (one solver) and `chain3` (solver -> reviewer -> decider, fixed order). Same
  neutral prompts for all models; final line `The answer is X`, strict parser, no repair.
- Thinking disabled (`think: false`) for every model; a model that rejects it or still emits thinking
  is ineligible rather than silently switched. Temperature 0, top_k 1, seed 42, 4096 output tokens,
  16384 context, one parallel slot, q8_0 KV. Two repetitions.
- Separate pinned Ollama 0.40.6 and model store; the confirmation runs keep their installation.

## Pre-registered criteria (per model x benchmark x condition)

Zero errors; >=95% valid final format; <=5% truncated final calls; no thinking output; >=90% identical
predictions across the two repetitions; for `single` honest accuracy in [0.30, 0.85] (room for both
attack harm and routing benefit). Worker/any truncation and throughput are reported.

Selection: a model x benchmark pair is eligible only if both conditions pass. Prefer non-negative
chain gain (chain3 - single), then higher throughput. Attack outcomes are never used. n=40 is a
descriptive screen, not a ranking claim. If nothing is eligible, report that and revise the protocol
explicitly before a new screen (new task sample, new output directory).

## Run (Ramses)

```bash
cd ~/AnyMAC_RobustAnalysis && git pull --ff-only
bash scripts/screen/run.sh install      # login node: pinned Ollama into cache/
bash scripts/screen/run.sh prepare      # login node: download pinned data, freeze sample
bash scripts/screen/run.sh pull         # login node: ~70 GB models, no GPU, no model load
sbatch scripts/screen/screen_a30.sbatch # 4 array tasks, one A30 each
squeue -u $USER; tail -f slurm_screen_*_0.log
bash scripts/screen/run.sh report       # any time; incomplete models are marked
```

Resubmitting the same sbatch resumes completed episodes. A manifest mismatch (code, model digest,
limits) refuses to mix results; use a new output directory. Single model interactively:
`bash scripts/screen/run.sh run gemma4:26b-a4b-it-qat` inside an A30 allocation.
