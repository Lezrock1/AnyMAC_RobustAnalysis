#!/usr/bin/env bash
# Honest-only development preflight; no downloads, training or historical-result changes.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  prepare|run|report) ;;
  *) echo "usage: bash scripts/compare/preflight.sh prepare|run|report" >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { echo "unexpected arguments: $*" >&2; exit 2; }
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
base="$RESULTS_DIR/compare/gemma12b_preflight_v1"
tasks="$base/dev_tasks.json"
reserved=("$RESULTS_DIR/screen/screen_tasks_v1.json" "$RESULTS_DIR/screen/screen_tasks_r2.json")
for f in "${reserved[@]}"; do
  [[ -f "$f" ]] || { echo "missing reserved screening sample: $f" >&2; exit 1; }
done
if [[ "$mode" == prepare ]]; then
  python scripts/screen/screen.py prepare --tasks "$tasks" --benchmarks mmlu_pro \
    --splits "$DATA_DIR/splits/splits_pilot2_seed0.json" \
    --n 12 --seed 20261021 --exclude "${reserved[@]}"
  exit 0
fi
[[ -f "$tasks" ]] || { echo "missing $tasks; run prepare first" >&2; exit 1; }
if [[ "$mode" == report ]]; then
  for system in fixed dylan; do
    python scripts/compare/runner.py report --tasks "$tasks" --out "$base/$system"
  done
  exit 0
fi
: "${SLURM_JOB_ID:?Run inside a separate A30 allocation, not on the login node}"
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_PORT=11446 BASE_URL=http://127.0.0.1:11446 OLLAMA_HOST=127.0.0.1:11446
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
RAMSES_LOG_DIR="$RAMSES_LOG_DIR/compare_preflight"
[[ -x "$OLLAMA_BIN" ]] || { echo "screening Ollama binary missing: $OLLAMA_BIN" >&2; exit 1; }
if server_up; then
  echo "port $OLLAMA_PORT occupied; refusing to reuse another server" >&2
  exit 1
fi
trap stop_ollama EXIT
start_ollama
curl -fsS --max-time 30 "$BASE_URL/api/show" -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:12b-it-qat"}' >/dev/null || {
    echo "Gemma4 12B is unavailable; no automatic pull. Check model cache and personal quota." >&2
    exit 1
  }
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
for system in fixed dylan; do
  if [[ "$system" == fixed ]]; then compromised="Knowledgeable Expert"; else compromised="Economist"; fi
  python scripts/compare/runner.py run --system "$system" --tasks "$tasks" \
    --reserved "${reserved[@]}" --out "$base/$system" --model gemma4:12b-it-qat \
    --url "$BASE_URL" --policies honest --reps 3 --max-calls 17 \
    --compromised "$compromised" --limit 8192 --context 32768
  python scripts/compare/runner.py report --tasks "$tasks" --out "$base/$system"
done
