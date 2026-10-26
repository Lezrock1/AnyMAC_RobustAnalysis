#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; phase="${2:-pilot}"
case "$mode" in
  pilot|main) phase="$mode"; command=run ;;
  plan) phase="${2:-main}"; [[ "$phase" == pilot || "$phase" == main ]] || exit 2; command=plan ;;
  report) [[ "$phase" == pilot || "$phase" == main ]] || exit 2; command=report ;;
  *) echo "usage: bash scripts/compare/suite.sh pilot|main|plan|report [pilot|main]" >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || exit 2
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
learned="$RESULTS_DIR/compare/learned_gemma12b_v1"
reserved=("$RESULTS_DIR/screen/screen_tasks_v1.json" "$RESULTS_DIR/screen/screen_tasks_r2.json")
if [[ "$phase" == main ]]; then
  reserved+=("$RESULTS_DIR/compare/dylan_study_v1/eval_tasks.json"
             "$RESULTS_DIR/compare/gemma12b_preflight_v1/dev_tasks.json"
             "$RESULTS_DIR/compare/main_reserved_v2/eval_tasks.json")
fi
output="$RESULTS_DIR/compare/five_condition_gemma12b_v2"
amended=0
case "${COMPARE_ROUTING_ADMISSION:-strict}" in
  strict) ;;
  routing-observation-20261026)
    amended=1
    output="$RESULTS_DIR/compare/five_condition_gemma12b_v3"
    ;;
  *) echo "Unknown COMPARE_ROUTING_ADMISSION; use strict or routing-observation-20261026" >&2; exit 2 ;;
esac
args=("$command" --phase "$phase" --learned-root "$RESULTS_DIR/compare/learned_gemma12b_profiles_v2"
      --assets "$learned/assets.json"
      --gdesigner-root "$learned"
      --dev-tasks "$RESULTS_DIR/compare/gemma12b_preflight_v1/dev_tasks.json"
      --eval-tasks "$RESULTS_DIR/compare/main_balanced200_v1/eval_tasks.json"
      --train-tasks "$learned/train_tasks.json"
      --reserved "${reserved[@]}")
regime="${COMPARE_TRAINING_REGIME:-preliminary-v1}"
case "$regime" in
  preliminary-v1) ;;
  paper-aligned-v1|paper-aligned-v2)
    version="${regime#paper-aligned-}"
    paper="$RESULTS_DIR/compare/learned_gemma12b_paper_$version"
    output="$RESULTS_DIR/compare/five_condition_gemma12b_paper_$version"
    ((amended == 0)) || output="${output}_routing"
    args=("$command" --phase "$phase"
          --learned-root "$paper"
          --assets "$learned/assets.json" --gdesigner-root "$learned"
          --dev-tasks "$RESULTS_DIR/compare/gemma12b_preflight_v1/dev_tasks.json"
          --eval-tasks "$RESULTS_DIR/compare/main_balanced200_v1/eval_tasks.json"
          --train-tasks "$learned/train_tasks.json"
          --anymac-training-tasks "$paper/train_tasks.json"
          --reserved "${reserved[@]}")
    ;;
  *) echo "Unknown COMPARE_TRAINING_REGIME" >&2; exit 2 ;;
esac
case "${COMPARE_SUITE_SCOPE:-full}" in
  full) ;;
  core)
    output="${output/five_condition/core_four}"
    args+=(--scope core)
    ;;
  *) echo "Unknown COMPARE_SUITE_SCOPE; use full or core" >&2; exit 2 ;;
esac
args+=(--out "$output")
if ((amended)); then args+=(--allow-routing-degeneracy); fi
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
