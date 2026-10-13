#!/usr/bin/env bash
# Runs a profile inside a GPU allocation. Usable via sbatch (submit.sh) or interactively (srun/salloc).
set -euo pipefail
# sbatch copies this script to a spool dir; submit.sh therefore exports ANYMAC_REPO.
source "${ANYMAC_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}/scripts/ramses/lib.sh"
load_env; mkdirs; load_modules; activate_venv
profile=""; exp="all"; resume=0; allow_cpu=0
while [[ $# -gt 0 ]]; do case "$1" in
  --profile) profile="$2"; shift 2;;
  --experiment) exp="$2"; shift 2;;
  --resume) resume=1; shift;;
  --allow-cpu) allow_cpu=1; shift;;
  *) echo "usage: run.sh --profile smoke|pilot|full --experiment 1|2|3|4|all [--resume]" >&2; exit 2;; esac; done
[[ -n "$profile" ]] || { echo "--profile required" >&2; exit 2; }
cd "$REPO_ROOT"
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) profile=$profile experiment=$exp resume=$resume"
git rev-parse HEAD 2>/dev/null || true
if [[ $allow_cpu -eq 0 ]]; then
  nvidia-smi -L || { echo "no GPU visible in this allocation" >&2; exit 3; }
  python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False (driver/torch mismatch?)")
print("torch cuda device:", torch.cuda.get_device_name(0))
PY
fi
export OLLAMA_NUM_PARALLEL="$(python -m robust profile-get "$profile" ollama.num_parallel)"
if [[ "$(python -m robust profile-get "$profile" ollama.flash_attention 2>/dev/null || true)" == "True" ]]; then export OLLAMA_FLASH_ATTENTION=1; export OLLAMA_KV_CACHE_TYPE="$(python -m robust profile-get "$profile" ollama.kv_cache_type)"; fi
start_ollama; trap stop_ollama EXIT
# GPU-use monitor: samples utilisation/memory into the log directory
mon="$RAMSES_LOG_DIR/gpu_${SLURM_JOB_ID:-local}.csv"
if command -v nvidia-smi >/dev/null; then
  (while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >>"$mon" 2>/dev/null; sleep 30; done) &
  MON_PID=$!; trap 'kill $MON_PID 2>/dev/null || true; stop_ollama' EXIT
fi
rargs=(); [[ $resume -eq 1 ]] && rargs+=(--resume)
if [[ "$profile" == "smoke" ]]; then
  # the smoke profile always continues internally with --resume, so stale stages from an earlier code/prompt version must not survive
  if [[ $resume -eq 0 && -d "$RESULTS_DIR/smoke" ]]; then
    arch="$RESULTS_DIR/smoke_prev_$(date +%Y%m%d_%H%M%S)"; mv "$RESULTS_DIR/smoke" "$arch"; echo "archived previous smoke results to $arch"
  fi
  if [[ $allow_cpu -eq 1 ]]; then python -m robust smoke; else python -m robust smoke --require-cuda; fi
  # smoke covers all experiments end-to-end on tiny samples only when requested
  [[ "$exp" == "all" ]] && python -m robust run --profile smoke --experiment all --resume
  [[ "$exp" != "all" ]] && python -m robust run --profile smoke --experiment "$exp" --resume
else
  python -m robust run --profile "$profile" --experiment "$exp" "${rargs[@]}"
fi
if [[ -f "$mon" ]]; then echo "GPU samples (utilisation, memory):"; tail -3 "$mon"; fi
echo "results: $RESULTS_DIR/$profile  (report.md, report.json)"
