#!/usr/bin/env bash
# Isolated follow-up jobs. Historical results are read-only inputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; shift || true
case "$mode" in snapshot|audit|control|confirm|both) ;; *) echo "usage: run.sh snapshot|audit|control|confirm|both [options]" >&2; exit 2;; esac
source scripts/ramses/lib.sh
load_env
base="$RESULTS_DIR/followup"
mkdir -p "$base"
if [[ "$mode" == snapshot ]]; then
  # Standard library only; no GPU or environment activation needed.
  exec "$PYTHON_BIN" scripts/followup/audit.py snapshot --repo "$REPO_ROOT" --out "$base/snapshot_$(date +%Y%m%d_%H%M%S)" "$@"
fi
if [[ "$mode" == audit ]]; then
  exec "$PYTHON_BIN" scripts/followup/audit.py audit --pilot "$RESULTS_DIR/pilot2" \
    --splits "$DATA_DIR/splits/splits_pilot2_seed0.json" --out "$base/audit_$(date +%Y%m%d_%H%M%S)" "$@"
fi
: "${SLURM_JOB_ID:?Use a separate Slurm allocation, not the login node}"
load_modules; activate_venv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "job=$SLURM_JOB_ID node=$(hostname) mode=$mode commit=$(git rev-parse HEAD)"
if [[ "$mode" == both ]]; then
  [[ $# -eq 0 ]] || { echo "both uses fixed protocols; use control/confirm for explicit overrides" >&2; exit 2; }
  [[ "${SLURM_CPUS_PER_TASK:-0}" -ge 8 ]] || { echo "both needs >=8 CPUs" >&2; exit 2; }
  # The CPU learner and GPU inference are independent; preserve both exit statuses.
  bash scripts/followup/run.sh control --threads 2 >>"$base/control.log" 2>&1 &
  CONTROL_PID=$!
  trap 'kill "$CONTROL_PID" 2>/dev/null || true' EXIT
  confirm_status=0; control_status=0
  bash scripts/followup/run.sh confirm >>"$base/confirm.log" 2>&1 || confirm_status=$?
  wait "$CONTROL_PID" || control_status=$?
  trap - EXIT
  echo "control exit=$control_status; confirm exit=$confirm_status (logs: $base)"
  [[ $control_status -eq 0 && $confirm_status -eq 0 ]]
  exit
fi
if [[ "$mode" == control ]]; then
  # CPU-only so this diagnostic can coexist with confirmation without consuming its GPU memory.
  export CUDA_VISIBLE_DEVICES=""
  exec python scripts/followup/control.py --out "$base/control_v1" "$@"
fi
export OLLAMA_PORT_OVERRIDE="${FOLLOWUP_PORT:-11438}"
load_env
export OLLAMA_NUM_PARALLEL=2 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
if server_up; then echo "Port $OLLAMA_PORT occupied; refusing to reuse another server" >&2; exit 1; fi
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
start_ollama
trap stop_ollama EXIT
python scripts/followup/confirm.py --source "$RESULTS_DIR/pilot2/qwen30/seed0" \
  --splits "$DATA_DIR/splits/splits_pilot2_seed0.json" --out "$base/confirm_chat4096_v1" "$@"
