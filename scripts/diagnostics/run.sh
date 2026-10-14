#!/usr/bin/env bash
# Separate allocation and server; never resumes or edits research experiments.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SLURM_JOB_ID:?Run inside a separate Slurm GPU allocation}"
export OLLAMA_PORT_OVERRIDE=11436
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
nvidia-smi -L

if server_up; then
  echo "Diagnostic port 11436 is occupied; refusing to reuse that server." >&2
  exit 1
fi

out="$RESULTS_DIR/diagnostics/suite_job${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
deadline=$(($(date +%s) + 8 * 3600))
python scripts/diagnostics/diagnose.py initialize --out "$out" \
  --splits "$DATA_DIR/splits/splits_pilot_seed0.json" \
  --pilot "$RESULTS_DIR/pilot/qwen/seed0" --deadline "$deadline"
echo "Persistent report: $out/REPORT.txt"
{
  echo "job=$SLURM_JOB_ID node=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
  "$OLLAMA_BIN" --version
} >"$out/hardware.txt" 2>&1

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]]; then
    local child
    local children
    children="$(pgrep -P "$server_pid" || true)"
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid"
    fi
    for child in $children; do
      kill -TERM "$child" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid"
    fi
    for child in $children; do
      kill -0 "$child" 2>/dev/null && kill -KILL "$child" 2>/dev/null || true
    done
    wait "$server_pid" || true
    server_pid=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
unset OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE
for slots in 1 12; do
  [[ $(date +%s) -lt "$deadline" ]] || break
  if server_up; then
    echo "Port unexpectedly occupied before slots=$slots; refusing to continue." >&2
    exit 1
  fi
  OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_NUM_PARALLEL="$slots" \
    OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=24h \
    "$OLLAMA_BIN" serve >"$out/ollama_slots${slots}.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 120); do
    kill -0 "$server_pid" 2>/dev/null || {
      echo "Own server exited; inspect $out/ollama_slots${slots}.log" >&2
      exit 1
    }
    if server_up; then ready=1; break; fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || { echo "Server startup timeout" >&2; exit 1; }
  echo "Own diagnostic server pid=$server_pid slots=$slots"
  python -u scripts/diagnostics/diagnose.py run --out "$out" \
    --slots "$slots" --base "$BASE_URL" 2>&1 | tee -a "$out/run.log"
  stop_server
  for _ in $(seq 1 30); do
    if ! server_up; then break; fi
    sleep 1
  done
done
echo "Diagnostic requests finished. Read: $out/REPORT.txt"
