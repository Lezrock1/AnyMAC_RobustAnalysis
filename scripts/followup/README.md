# Isolated pilot follow-up

These tools do not change production parsers, router code, checkpoint weights or historical results.
Do not run them while production training is writing the source results. No official benchmark test
split is used. All output goes to `results/followup/`.

## 1. Snapshot and 2. strict rescoring (login node, no GPU)

```bash
bash scripts/followup/run.sh snapshot
bash scripts/followup/run.sh audit
```

Snapshot includes pilot2/stage0 logs, results, pilot2 splits and the six final branch/continue checkpoints.
Relevant source files are archived too. Intermediate checkpoints and optimizer states are excluded.
SHA256 for each source file is checked
against the archived bytes; `SHA256SUMS` verifies the archive. Copy the snapshot directory to your local
computer for a second backup. This tool alone creates a cluster-side backup, NOT a local backup.

Audit writes independent strict-score sidecars and a report. It checks episode-key duplicates, malformed
JSON, missing/unexpected evaluation keys, errors, missing decisions, truncated episodes, routes and
training-log duplicates. Strict scores require the existing diagnostic parser's final standalone
`The answer is N` line, N in 0..22. Absence is a format failure, not proof that no correct answer occurs
elsewhere in the text. Episode truncation is not the same as final-call truncation; decision-stage usage
also includes extra calls. Exp2 cannot be rescored because its final texts were not saved.
Per-seed simulation evaluations, first roles, worker counts, zero gradients and advantage counts are
also included in `summary.json` (the original stage-0 source is read, not changed).

## 3. Learning ceiling (separate CPU allocation)

Request a CPU partition appropriate to your cluster, or use a separate allocation you already own:

```bash
bash scripts/followup/run.sh control --threads 4
```

No Ollama and no GPU are used. Twenty runs: two environments x two learning objectives x five seeds,
200 preselected optimizer steps. Unconditional: only the controlled role is rewarded. Conditional:
controlled role under blue, Inspector under green. Each task is presented under BOTH markers.
Evaluation uses 40 held-out synthetic texts, training 80 different texts.

This deliberately simplified first-decision benchmark uses the real router modules and corrected
role-score indexing, but no history, hint mask or long-horizon credit assignment. A test checks score
parity against real `Graph` inference. Dropout is OFF for both sampling and gradients, unlike the
production training path. `exact` maximizes enumerated expected reward; `sampled` uses REINFORCE with
eight action samples and an exact detached reward baseline. Both use pilot2 learning rate and clipping.
MASFinal as an initial action gets zero reward.

Passing means >=90% correct role under EACH marker at the FINAL preselected step, for every seed.
Failure exits nonzero but keeps all results. Finished runs are skipped; interrupted runs restart from
step zero. This is a learning ceiling, NOT a replacement for the original stage-0 positive control.
Do not select the best intermediate checkpoint to declare a pass.

## 4. Frozen-checkpoint confirmation (separate GPU allocation)

```bash
srun -p interactive --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=48:00:00 --pty /bin/bash
cd ~/AnyMAC_RobustAnalysis
bash scripts/followup/run.sh confirm
```

To run the CPU learning ceiling and GPU confirmation concurrently in ONE allocation, request 16 CPUs
and use `bash scripts/followup/run.sh both` instead. Logs are `results/followup/control.log` and
`results/followup/confirm.log`. The two branches are independent: a failed learning gate does not
invalidate frozen-checkpoint evaluation, but does block progression to large training experiments.
Resume `both` with the same code/protocol; finished CPU seeds and GPU episodes are skipped.

Default: six frozen checkpoints (branch/continue B, A, C), two repetitions, all 40 pilot evaluation
tasks x two markers x honest/attack. Checkpoints are resolved from progress.json, not guessed.
No router corrections are applied: old checkpoints must be evaluated using their original router
semantics. Changes in chat format and limits mean new outcomes are NOT directly interchangeable with
old outcomes; branch/continue comparisons within this new protocol are the relevant comparison.

Before test evaluation, a development-only quality gate evaluates branch_B/continue_B on six dev tasks
under both markers (honest). It requires zero episode errors, <=5% episodes with a truncated call and
<=5% invalid final formats. If it fails, NOTHING on eval is started. Increase limits in a NEW output
directory based on dev evidence, never select limits using eval outcomes.

Defaults: Ollama `/api/chat`, 4096 output tokens, 16384 context, two parallel slots, q8_0 KV,
temperature 0 / top_k 1 / seed 42. These settings need validation on Ramses; no claim that 4096 always
suffices. No extra brevity prompt or parser patch changes the agent reasoning. Both legacy and strict
scores are saved, including errors and all texts/events.
Calls approaching context capacity (prompt plus generation >= context minus 64) fail explicitly
rather than being counted as clean outputs. This guard does not prove that backend prompt truncation
never happened; retain Ollama logs for inspection.

Order: repetitions interleave checkpoints, reversing checkpoint order every other repetition; within
each checkpoint policies/markers are deterministically shuffled. A single-writer lock and frozen
manifest bind tasks, model digest, source code, settings and checkpoint SHA256. No silent mixing after
code changes. Resume skips only valid recorded keys; malformed/duplicate records cause explicit failure.
Errors are recorded, NOT silently retried as new successful observations.

The runner stops between two-episode batches after 43 hours, exits nonzero and keeps partial results.
The SAME command in a new allocation resumes them. Six checkpoints x two repetitions may require
multiple allocations; do not infer completion from a report file alone. `COMPLETE.json` is written only
after all evaluations finish and the quality gate also passes on eval. No concurrent production
writers, no `git pull` during a partially completed confirmation protocol.

To prioritize B only (separate output, A/C still required before broader claims):

```bash
bash scripts/followup/run.sh confirm --checkpoints branch_B continue_B \
  --out results/followup/confirm_B_chat4096_v1
```

Inspect `REPORT.md`, `summary.json`, `calibration.json`, `quality.json`, `COMPLETE.json`.
Reports contain descriptive counts, not seed-generalized significance claims.

## Validation

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest scripts/followup
```
