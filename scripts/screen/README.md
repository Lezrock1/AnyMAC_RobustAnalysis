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

The `interactive` partition (A30) does not accept `sbatch`; use one interactive allocation. All steps
use the project venv. Downloads happen on the login node only.

```bash
cd ~/AnyMAC_RobustAnalysis && git pull --ff-only
bash scripts/screen/run.sh install      # login node: pinned Ollama 0.40.6 into cache/
bash scripts/screen/run.sh prepare      # login node: download pinned data, freeze sample
bash scripts/screen/run.sh dedupe       # optional: anchor shares blobs with confirm store
tmux new -s screen                      # survives SSH disconnects
srun -p interactive --gres=gpu:a30:1 --cpus-per-task=8 --mem=128G --time=48:00:00 --pty /bin/bash
cd ~/AnyMAC_RobustAnalysis && bash scripts/screen/run.sh all
```

`all` starts one worker per visible GPU (port 11440+i) and distributes the models round-robin, so
fewer GPUs (e.g. `gpu:a30:2`) also work, just slower. Logs: `results/screen/logs/<model>.log`.
Rerunning `all` resumes recorded episodes. A manifest mismatch (code, model digest, limits) refuses
to mix results; use a new output directory. Report any time: `bash scripts/screen/run.sh report`
(incomplete models are marked). Single model: `bash scripts/screen/run.sh run gemma4:26b-a4b-it-qat`.

### Storage quota (just-in-time models)
Home quota (~100 GB) cannot hold all models. `run` pulls a missing model itself and, in `all` mode
(`SCREEN_EVICT=1` by default), removes it after its run; models sharing blobs with the confirm
store (hardlinks, see `run.sh dedupe`) are kept. Hence **one A30** (`gpu:a30:1`): models run
sequentially and at most one non-anchor model is on disk. Several GPUs would hold several models at
once and can exceed the quota. A re-pull with a changed digest makes the run refuse to resume
(manifest), so versions are never mixed. `SCREEN_MODELS_DIR` moves the store to another filesystem.

### Protocol change v1 -> v2 (2026-10-19, before any model selection)
`screen_v1` was aborted after 73 episodes of the anchor model (kept, not used):
1. Ollama ran 192 CPU threads (all node cores) on an 8-CPU allocation shared with other jobs:
   15 tok/s with all 49/49 layers on the GPU. v2 sends `num_thread` = allocated CPUs (`--threads`).
2. Parser v1 rejected unambiguous MC finals with option text (`The answer is D. Not wrong, Not wrong`).
   v2 accepts a letter followed by a separator and text; integers must still stand alone.
3. Added `acc mod23`: integer answers reduced mod 23 by us. Sensitivity only, never for eligibility
   (the task demands mod-23 arithmetic; e.g. the anchor answered 84 for truth 15).
Criteria, tasks and selection rule are unchanged. Output: `results/screen/screen_v2`.

## Round 2 (registered 2026-10-20, before any round-2 run)

**Trigger.** Round 1 (`screen_v2`, complete for all four models) has no eligible pair. As foreseen above,
the protocol is revised explicitly and a new sample is screened. No attack has been run with any
candidate; only the honest round-1 table informs these choices. Round 1 is reported in full.

**Diagnosis from round 1 → changes.**
- MASBench fails for every model (format and truncation), MMLU is at the ceiling for all but GLM
  (single accuracy 0.875–0.95), GLM fails repeat agreement. MMLU-Pro is the only benchmark with
  single accuracy inside the window for every model → **round 2 screens MMLU-Pro only**.
- The dense Qwen3.8 failed only by truncation at 4096 tokens (8.7% final) → **limit 8192, context
  32768** (chain3 decider sees two full worker outputs), for all models.
- Repeat agreement failed only for MoE models (Gemma 26B chain3 0.85, GLM 0.68–0.83); dense Qwen3.8
  reached 0.90–1.00 → add **dense Gemma 4** (12B and 31B, QAT) and re-test Gemma 26B (closest MoE).
- Qwen3.8 is slow (26 tok/s). Its MTP build (`qwen3.8:27b-mtp-q4_K_M`) replaces it in round 2 if
  `run.sh mtpcheck` shows it runs without error and reaches >= 1.3x tokens/s on 10 reserved round-1
  MMLU-Pro tasks with round-1 settings (`mtp_check.py`). Otherwise the plain tag is screened.

**Sample.** 40 new MMLU-Pro items (same pinned file), seed 20261020, drawn after excluding all
round-1 ids: `results/screen/screen_tasks_r2.json`. Reserved like round 1. Two repetitions.

**Criteria.** Unchanged and strict (`CRITERIA`), both conditions (`single`, `chain3`).

**Selection.** Among strictly eligible models: non-negative chain gain, then lower mean
`s/episode` in `chain3` (this defines "throughput": it determines experiment cost).
Fallback only if no model is strictly eligible: the `tolerant` column (errors, thinking and the
accuracy window stay strict; format, final truncation and repeat agreement pass if the threshold lies
inside the 95% Wilson interval), same ordering. If nothing passes either, stop and report; no
further automatic rounds.

```bash
bash scripts/screen/run.sh mtpcheck                    # A30 allocation, ~10 min; writes results/screen/mtp_check.json
SCREEN_ROUND=2 bash scripts/screen/run.sh prepare      # login node: frozen round-2 sample
SCREEN_ROUND=2 bash scripts/screen/run.sh all          # A30; output results/screen/screen_r2
SCREEN_ROUND=2 bash scripts/screen/run.sh report
```
Later evaluations must pass both `screen_tasks_v1.json` and `screen_tasks_r2.json` as reserved.
