#!/usr/bin/env bash
# Submits one Slurm job that runs scripts/ramses/run.sh. All Slurm values are passed as sbatch arguments.
set -euo pipefail
source "$(dirname "$0")/lib.sh"; load_env
profile=""; exp="all"; dry=0; resume=0
while [[ $# -gt 0 ]]; do case "$1" in
  --profile) profile="$2"; shift 2;;
  --experiment) exp="$2"; shift 2;;
  --dry-run) dry=1; shift;;
  --resume) resume=1; shift;;
  *) echo "usage: submit.sh --profile smoke|pilot|full --experiment 1|2|3|4|all [--dry-run] [--resume]" >&2; exit 2;; esac; done
case "$profile" in smoke|pilot|full) ;; *) echo "--profile must be smoke|pilot|full" >&2; exit 2;; esac
case "$exp" in 1|2|3|4|all) ;; *) echo "--experiment must be 1|2|3|4|all" >&2; exit 2;; esac
tvar="RAMSES_TIME_$(echo "$profile" | tr a-z A-Z)"; walltime="${!tvar:-}"
mkdir -p "$RAMSES_LOG_DIR"
args=(--job-name "anymac-${profile}-${exp}" --nodes 1 --ntasks 1
      --cpus-per-task "${RAMSES_CPUS:-4}" --mem "${RAMSES_MEM:-32G}"
      --output "$RAMSES_LOG_DIR/%x_%j.out" --error "$RAMSES_LOG_DIR/%x_%j.err" --chdir "$REPO_ROOT" --export "ALL,ANYMAC_REPO=$REPO_ROOT")
[[ -n "${RAMSES_PARTITION:-}" ]] && args+=(--partition "$RAMSES_PARTITION")
[[ -n "${RAMSES_ACCOUNT:-}" ]] && args+=(--account "$RAMSES_ACCOUNT")
[[ -n "${RAMSES_QOS:-}" ]] && args+=(--qos "$RAMSES_QOS")
[[ -n "${RAMSES_GRES:-}" ]] && args+=(--gres "$RAMSES_GRES") || echo "WARN: RAMSES_GRES empty; job would get no GPU (run.sh will fail)" >&2
[[ -n "$walltime" ]] && args+=(--time "$walltime") || echo "WARN: no wall time configured for $profile (cluster default applies)" >&2
# shellcheck disable=SC2206
[[ -n "${RAMSES_SBATCH_EXTRA:-}" ]] && args+=($RAMSES_SBATCH_EXTRA)
runargs=(--profile "$profile" --experiment "$exp"); [[ $resume -eq 1 ]] && runargs+=(--resume)
cmd=(sbatch "${args[@]}" "$REPO_ROOT/scripts/ramses/run.sh" "${runargs[@]}")
printf 'command:'; printf ' %q' "${cmd[@]}"; echo
if [[ $dry -eq 1 ]]; then echo "(dry run, nothing submitted)"; exit 0; fi
command -v sbatch >/dev/null || { echo "sbatch not found (not on a Slurm host)" >&2; exit 1; }
"${cmd[@]}"
