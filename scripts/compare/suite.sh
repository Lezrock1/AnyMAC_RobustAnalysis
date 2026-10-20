#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; phase="${2:-pilot}"
case "$mode" in
  pilot|main) phase="$mode"; command=run ;;
  plan) phase=main; command=plan ;;
  report) [[ "$phase" == pilot || "$phase" == main ]] || exit 2; command=report ;;
  *) echo "usage: bash scripts/compare/suite.sh pilot|main|plan|report [pilot|main]" >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || exit 2
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
learned="$RESULTS_DIR/compare/learned_gemma12b_v1"
args=("$command" --phase "$phase" --learned-root "$learned"
      --dev-tasks "$RESULTS_DIR/compare/gemma12b_preflight_v1/dev_tasks.json"
      --eval-tasks "$RESULTS_DIR/compare/dylan_study_v1/eval_tasks.json"
      --train-tasks "$learned/train_tasks.json"
      --reserved "$RESULTS_DIR/screen/screen_tasks_v1.json" "$RESULTS_DIR/screen/screen_tasks_r2.json"
      --out "$RESULTS_DIR/compare/four_system_gemma12b_v1")
if [[ "$command" != run ]]; then
  python scripts/compare/suite.py "${args[@]}"
  exit 0
fi
: "${SLURM_JOB_ID:?Use an A30 allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU mask missing; do not use unallocated GPUs}"
if [[ "$phase" == main ]]; then
  : "${COMPARE_GPU_BUDGET_HOURS:?Inspect suite.sh plan and explicitly approve projected GPU hours}"
  args+=(--approved-gpu-hours "$COMPARE_GPU_BUDGET_HOURS")
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
IFS=, read -r -a gpus <<<"$CUDA_VISIBLE_DEVICES"
cpus="${SLURM_CPUS_PER_TASK:-8}"
threads=$((cpus / ${#gpus[@]}))
((threads > 0)) || { echo "not enough CPUs per GPU" >&2; exit 1; }
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
args+=(--threads "$threads")
logs="$RAMSES_LOG_DIR/compare_suite"
pids=(); urls=()
cleanup() {
  if ((${#pids[@]})); then
    for pid in "${pids[@]}"; do OLLAMA_STARTED_PID="$pid"; stop_ollama; done
  fi
}
trap cleanup EXIT
for i in "${!gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES="${gpus[$i]}" OLLAMA_PORT=$((11454 + i))
  export BASE_URL="http://127.0.0.1:$OLLAMA_PORT" OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
  RAMSES_LOG_DIR="$logs/gpu$i"
  if server_up; then echo "port $OLLAMA_PORT occupied; refusing a foreign server" >&2; exit 1; fi
  start_ollama
  pids+=("$OLLAMA_STARTED_PID")
  urls+=("$BASE_URL")
done
python scripts/compare/suite.py "${args[@]}" --urls "${urls[@]}"
