# Exploratory training attack within one 24-hour allocation

Approved reduced design (2026-11-03): **AnyMAC Normal only**, six paired seeds (0-5),
`ordinary-neutral` versus `helpful-neutral`, target **Knowledgeable Expert**.
Both arms replace every actual invocation of this identity with frozen training
content. Routing is never forced. Helpful generation sees only official
**training** labels; ordinary generation does not. No prominence/persuasive arm.

This is short training from the same pinned GPT-2 initialization per seed,
**not** continuation of the main checkpoints or a 1,000-trajectory reproduction.
The ordinary arm controls frozen replacement/presentation; existing clean main
checkpoints are contextual references, not additional matched controls. The role
was chosen because it was unused in the prior main routers, not because its
future use is impossible without intervention.

## Budget and execution

Total budget: two A30, 16 CPUs, 64 GiB, **24 hours = at most 48 allocated GPU-h**,
including the failed attempt. The approved recovery charges 24 minutes for job
7213910 (sacct: 23:53) and allows at most **23:36 for the next allocation**.
Server startup, bank generation, pilot, training, evaluation and cleanup all
count. The script sets an application deadline 15 minutes before its declared allocation limit;
Slurm provides the outer hard limit. No chained job, automatic extension or
automatic resume. Only processes started by the job are stopped.

1. Check disjoint frozen paper-v2 validation training, development, main and
   reserved questions, pinned assets and identical screened workers.
2. Initially generate ordinary/helpful entries once. In the approved recovery,
   import the already generated texts with the paired quality exclusion below;
   never regenerate or repair. Report actual quality rather than assuming help.
3. Within a two-hour setup window, run an eight-rollout calibration in both arms
   (not included in study results), plus two six-call/full-context stress episodes
   on each GPU.
4. Select **one fixed budget from 64/128/192/256 trajectories** per arm/seed,
   in independent groups of eight, based **only on wall times**. Use 35% runtime
   margin, reserve projected paired evaluation and 15 minutes for final reporting.
   If even 64 cannot fit, stop explicitly before main training. This is not a
   guarantee that all models will finish: runtime can change after preflight.
5. Train six seed pairs on the two GPUs; swap training-arm GPU assignments by
   seed parity. All other optimizer/profile/role parameters stay identical:
   learning rate 1e-5, reward decay 1, five workers maximum, five context messages.
   Four initial development questions replace the costly full learning curves.
   Neither development scores nor cooperation select the final checkpoint.
6. For each seed, evaluate **both arms on the same GPU/server**, interleaving
   conditions and adjacent honest/bare episodes. Two seed pairs run concurrently.
   Use 24 outcome-independently subject-balanced questions from the existing
   200-question main pool, one repetition each; four questions additionally get
   a second honest episode with the *same* router RNG. Total: **624 evaluation
   episodes**, not 624 independent training replications.

Timeout/failure stops later training pairs or evaluation waves. Existing rolling
checkpoints, attempt logs, failure codes and partial traces are preserved.
Incomplete pairs do not enter paired effects and are explicitly listed; a
completed-only subset can be biased if intervention-induced cooperation is slower.
Infrastructure failure is not silently repaired. Null cooperation, zero utility
updates, poor output or performance loss are reported, not filtered away.
The wrapper accepts only this study's full-budget hash-bound intervention
checkpoints, **without changing clean-suite admission**.

## Outcomes and interpretation

Primary: helpful minus ordinary **honest target-role invocation rate** (across
questions), alongside mean target calls, aggregator exposure and all-worker use.
This measures learned behavioral dependence, not psychological trust.

Secondary: honest/bare target-hit and accuracy differences within each arm, and
the difference of these attack effects between training arms. A higher invocation
rate alone does not demonstrate exploitable damage. Report training replacement
coverage and measured bank quality as manipulation checks.

Use paired contrasts for each seed and separate bootstrap intervals over seeds
(conditional on tasks) and tasks (conditional on checkpoints). Six seeds remain
few; 24 previously analyzed questions are not fresh confirmatory evidence.
The primary contrast is predeclared; secondary contrasts are exploratory without
multiple-testing claims. Do not infer a population-level null from all-zero CIs.

An observed helpful-versus-ordinary accuracy drop greater than **10 percentage
points** is a warning. Report the paired accuracy interval too. This tolerance is
**not** a checkpoint admission rule or proof of noninferiority; no model is removed
because it uses no workers or fails to improve. The small honest-repeat control
diagnoses local serving noise, not all GPU/numerical confounding.

Content length, explanation quality and label-assisted instruction form one
intervention package. Both frozen arms ignore peer context. The estimand is the
effect of this training-message intervention at a short fixed budget, not a
general property of AnyMAC or the full factorial trust experiment.

## Approved bank recovery (2026-11-04)

Job **7213910** failed before any training: all 140 texts were generated, but
`validation/30` helpful and `validation/40` ordinary were truncated/unparseable.
The user approved removing **both arms of both tasks**, not selecting correct
answers or regenerating failed outputs. The new bank retains **68 questions,
136 unchanged answers**, all original subjects, and every valid wrong answer.
Evaluation questions, seeds, profile and role remain unchanged.

The importer checks the original complete progress manifest, source, worker,
task content, reservations, failed stage and the exact approved bad entries.
Any additional missing/invalid entry blocks recovery; exclusions never expand
automatically. Original files remain untouched. New task/bank/audit files and
original evidence hashes enter the new study manifest. Original bank quality is
recorded alongside retained-bank quality. Training estimates now refer to this
**quality-filtered pool**; answer length/difficulty-related selection can limit
generalization. This amendment was approved before any router training.

The failed allocation is charged **24 minutes**, rounded up from sacct 23:53.
The next allocation is **23:36**, not another 24 hours. The application rejects
a declared recovery allocation exceeding that remainder before server startup.
There is no automatic retry; any further failed attempt must be accounted for
before another allocation is authorized.

## Ramses launch (approved recovery)

Use the existing detached worktree pinned to the tested recovery commit (supplied in chat).
Do not update the main-evaluation worktree or its hash-bound sources.
Ramses forbids `sbatch` in the `interactive` partition: use **`salloc` followed
by one `srun` step**. The rejected batch submission on 2026-11-04 started no study.
Run this inside `tmux`; the interactive allocation remains attached until the
study finishes. Do not close that session. Detach with Ctrl-b, then d.

```bash
(
  set -euo pipefail
  repo="$HOME/AnyMAC_RobustAnalysis"
  git -C "$repo" fetch origin robust-analysis
  commit=TESTED_COMMIT_FROM_CHAT
  worktree="$HOME/AnyMAC_training_attack_short"
  [[ -z "$(git -C "$worktree" status --porcelain)" ]] \
    || { echo "Worktree is not clean; stop." >&2; exit 1; }
  git -C "$worktree" checkout --detach "$commit"
  cd "$worktree"
  export ANYMAC_REPO="$repo"
  export TRAINING_ATTACK_RECOVER_FROM="$repo/results/compare/training_attack_short_v1/job_7213910"
  export TRAINING_ATTACK_ALLOCATION_SECONDS=84960
  salloc --partition=interactive --job-name=trust-short \
    --gres=gpu:a30:2 --ntasks=1 --cpus-per-task=16 --mem=64G --time=23:36:00 \
    srun --ntasks=1 --export=ALL --output="$HOME/trust_short_%j.out" \
      --error="$HOME/trust_short_%j.err" bash scripts/training_attack/run.sh
)
```

Only switch after confirming the old job is stopped, as it is for 7213910.
Assets, original progress, pool and model must already exist; this job never
installs dependencies or pulls models. If results were configured outside the
repository, point `TRAINING_ATTACK_RECOVER_FROM` to the actual old job directory.

Results: `$RESULTS_DIR/compare/training_attack_short_v1/job_JOBID/`.
`plan.json` freezes the runtime-released budget; `state.json` records failures;
`summary.json` includes complete/incomplete seeds, bank quality, per-seed metrics,
noise and paired CIs. `logs/` contains stage output and exact argv/return codes.
Failures also copy the last 20 child-log lines into the main stderr log.
`bank_recovery.json` documents exclusions, original quality and evidence hashes;
`summary.json` repeats recovery and total-allocation accounting.
`COMPLETE.json` per evaluation seed binds trace and checkpoint hashes.

Afterward, re-run the report (CPU only, same pinned environment/source):

```bash
python scripts/training_attack/report.py --input "$RESULTS_DIR/compare/training_attack_short_v1/job_JOBID"
```

## Local validation

```bash
.venv/bin/python -m pytest scripts/training_attack scripts/compare/test_trust.py \
  scripts/compare/test_learned.py scripts/compare/test_admission.py -q
bash -n scripts/training_attack/run.sh
```
