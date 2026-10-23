#!/usr/bin/env bash
# One allocated A30, separate from the live learned-router training worktree.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
: "${SLURM_JOB_ID:?Use a dedicated A30 allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU mask missing}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Use exactly one allocated GPU" >&2; exit 1; }
code="$HOME/AnyMAC_compare_full"
[[ "$(git -C "$code" rev-parse HEAD)" == 78623cfce260109c02753470226afff64ac6e9ee ]] || {
  echo "Original training worktree changed; refusing diagnostic" >&2; exit 1; }
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_PORT=11456 BASE_URL=http://127.0.0.1:11456 OLLAMA_HOST=127.0.0.1:11456
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RAMSES_LOG_DIR="$RAMSES_LOG_DIR/cooperation_diagnostic"
if server_up; then echo "Port 11456 occupied; refusing foreign server" >&2; exit 1; fi
trap stop_ollama EXIT
start_ollama
base="$RESULTS_DIR/compare"
python scripts/diagnostics/cooperation.py --code "$code" \
  --tasks "$base/gemma12b_preflight_v1/dev_tasks.json" \
  --metadata "$base/learned_gemma12b_v1/anymac-corrected/seed0/router.json" \
  --out "$base/cooperation_dev12_v1" --url "$BASE_URL" --threads "$OMP_NUM_THREADS" --reps 2
