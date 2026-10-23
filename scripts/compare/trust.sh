#!/usr/bin/env bash
# Generate ONE shared frozen bank before inspecting intervention outcomes.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -eq 1 ]] || { echo "usage: bash scripts/compare/trust.sh TARGET_ROLE" >&2; exit 2; }
role="$1"
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
: "${SLURM_JOB_ID:?Use one allocated A30}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU mask missing}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Use exactly one GPU" >&2; exit 1; }
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_PORT=11457 BASE_URL=http://127.0.0.1:11457 OLLAMA_HOST=127.0.0.1:11457
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RAMSES_LOG_DIR="$RAMSES_LOG_DIR/trust_bank"
if server_up; then echo "Port 11457 occupied; refusing foreign server" >&2; exit 1; fi
trap stop_ollama EXIT
start_ollama
curl -fsS --max-time 30 "$BASE_URL/api/tags" | python -c '
import json,sys
assert any(m["name"]=="gemma4:12b-it-qat" and
           m["digest"]=="38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3"
           for m in json.load(sys.stdin)["models"]), "screened Gemma unavailable; no pull"
'
base="$RESULTS_DIR/compare"
python scripts/compare/trust.py --tasks "$base/learned_gemma12b_v1/train_tasks.json" \
  --reserved "$RESULTS_DIR/screen/screen_tasks_v1.json" "$RESULTS_DIR/screen/screen_tasks_r2.json" \
    "$base/gemma12b_preflight_v1/dev_tasks.json" "$base/dylan_study_v1/eval_tasks.json" \
  --role "$role" --out "$base/trust_gemma12b_v1/messages.json" \
  --url "$BASE_URL" --threads "$OMP_NUM_THREADS"
