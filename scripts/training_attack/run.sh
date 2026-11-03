#!/usr/bin/env bash
#SBATCH --job-name=trust-short
#SBATCH --partition=interactive
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a30:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=trust_short_%j.out
#SBATCH --error=trust_short_%j.err
set -euo pipefail
started="$(date +%s)"
deadline=$((started + 23 * 3600 + 45 * 60))
cd "$(dirname "$0")/../.."
source scripts/ramses/lib.sh
load_env; load_modules; activate_venv
: "${SLURM_JOB_ID:?Submit with sbatch on Ramses}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU mask missing}"
IFS=, read -r -a gpus <<<"$CUDA_VISIBLE_DEVICES"
((${#gpus[@]} == 2)) || { echo "Exactly two allocated A30 GPUs required" >&2; exit 1; }
threads=$((${SLURM_CPUS_PER_TASK:-16} / 2))
((threads >= 1)) || { echo "Insufficient allocated CPUs" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_MAX_LOADED_MODELS=1 OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
out="$RESULTS_DIR/compare/training_attack_short_v1/job_$SLURM_JOB_ID"
original_logs="$RAMSES_LOG_DIR"
pids=(); urls=()
cleanup() {
  for pid in "${pids[@]}"; do OLLAMA_STARTED_PID="$pid"; stop_ollama; done
}
trap cleanup EXIT
for i in 0 1; do
  export CUDA_VISIBLE_DEVICES="${gpus[$i]}" OLLAMA_PORT=$((11458 + i))
  export BASE_URL="http://127.0.0.1:$OLLAMA_PORT" OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
  RAMSES_LOG_DIR="$original_logs/training_attack/$SLURM_JOB_ID/gpu$i"
  if server_up; then echo "Port $OLLAMA_PORT occupied; refusing a foreign server" >&2; exit 1; fi
  if ! start_ollama; then
    [[ -z "$OLLAMA_STARTED_PID" ]] || pids+=("$OLLAMA_STARTED_PID")
    exit 1
  fi
  pids+=("$OLLAMA_STARTED_PID")
  urls+=("$BASE_URL")
done
python -u scripts/training_attack/study.py --results "$RESULTS_DIR" --out "$out" \
  --urls "${urls[@]}" --threads "$threads" --started "$started" --deadline "$deadline"
