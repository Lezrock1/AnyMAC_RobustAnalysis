#!/usr/bin/env bash
# Separate worktree only. Historical AnyMAC training/results remain untouched.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; system="${2:-}"; seed="${3:-0}"; profile=""
case "$mode" in
  prepare) [[ $# -eq 1 ]] || exit 2 ;;
  smoke|train|gate)
    [[ $# -ge 2 && $# -le 3 ]] || exit 2
    case "$system" in
      anymac-normal) profile=normal; system=anymac-corrected ;;
      anymac-efficient) profile=efficient; system=anymac-corrected ;;
      anymac-historical-095) profile=historical-095; system=anymac-corrected ;;
      anymac-corrected) echo "Choose anymac-normal or anymac-efficient explicitly; old 0.95 runs are historical." >&2; exit 2 ;;
      gdesigner-paper) ;;
      *) echo "unknown system" >&2; exit 2 ;;
    esac
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "seed must be a nonnegative integer" >&2; exit 2; } ;;
  *) echo "usage: bash scripts/compare/learned.sh prepare | smoke|train|gate SYSTEM [SEED]" >&2; exit 2 ;;
esac
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv
base="$RESULTS_DIR/compare/learned_gemma12b_v1"
train="$base/train_tasks.json"
dev="$RESULTS_DIR/compare/gemma12b_preflight_v1/dev_tasks.json"
eval_tasks="$RESULTS_DIR/compare/dylan_study_v1/eval_tasks.json"
assets="$base/assets.json"
output_base="$RESULTS_DIR/compare/learned_gemma12b_profiles_v2"
[[ "$system" != gdesigner-paper ]] || output_base="$base"
label="$system"
[[ -z "$profile" ]] || label="anymac-$profile"
bank="${COMPARE_TRAINING_BANK:-}"
condition="${COMPARE_TRAINING_CONDITION:-}"
if [[ -n "$bank" || -n "$condition" ]]; then
  [[ -f "$bank" ]] || { echo "Frozen COMPARE_TRAINING_BANK missing" >&2; exit 1; }
  case "$condition" in ordinary-neutral|ordinary-prominent|helpful-neutral|helpful-prominent) ;;
    *) echo "Explicit COMPARE_TRAINING_CONDITION required" >&2; exit 1 ;; esac
  [[ "$mode" != smoke ]] || { echo "Use a full frozen training sample for trust conditions, not smoke" >&2; exit 1; }
  output_base="$RESULTS_DIR/compare/trust_gemma12b_v1/$condition"
fi
reserved=("$RESULTS_DIR/screen/screen_tasks_v1.json" "$RESULTS_DIR/screen/screen_tasks_r2.json" "$eval_tasks")
[[ -f "$dev" ]] || { echo "honest preflight tasks missing: $dev" >&2; exit 1; }
if [[ "$mode" == prepare ]]; then
  # Reserve evaluation BEFORE sampling training. This prepares IDs only, never attack runs.
  [[ -f "$eval_tasks" ]] || bash scripts/compare/study.sh prepare
  if [[ ! -f "$train" ]]; then
    python scripts/compare/train_learned.py prepare --out "$train" --reserved "${reserved[@]}" "$dev"
  fi
  python scripts/compare/train_learned.py assets --cache "$RAMSES_CACHE_DIR/huggingface/hub" --out "$assets"
  exit 0
fi
: "${SLURM_JOB_ID:?Use a dedicated A30 allocation, not the login node}"
for f in "$train" "$dev" "$assets" "${reserved[@]}"; do
  [[ -f "$f" ]] || { echo "missing preparation file: $f" >&2; exit 1; }
done
embedding="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["embedding"]["path"])' "$assets")"
backbone="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["backbone"]["path"])' "$assets")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 EMBED_DEVICE=cpu
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU mask missing; do not use unallocated GPUs}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "Use one allocated GPU per learned.sh process; explicitly partition CUDA_VISIBLE_DEVICES." >&2; exit 1; }
port=11452
if [[ "$system" == gdesigner-paper ]]; then
  [[ "$seed" -le 10000 ]] || { echo "seed too large for G-Designer port allocation" >&2; exit 2; }
  port=11453
  [[ "$seed" -eq 0 ]] || port=$((45000 + seed))
fi
if [[ -n "$profile" ]]; then
  [[ "$seed" -le 10000 ]] || { echo "seed too large for profile-specific port allocation" >&2; exit 2; }
  port=$((12000 + 3 * seed))
  [[ "$profile" != efficient ]] || port=$((port + 1))
  [[ "$profile" != historical-095 ]] || port=$((port + 2))
fi
export OLLAMA_PORT="$port" BASE_URL="http://127.0.0.1:$port" OLLAMA_HOST="127.0.0.1:$port"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RAMSES_LOG_DIR="$RAMSES_LOG_DIR/compare_learned/$system/seed$seed"
if server_up; then echo "port $port occupied; refusing to reuse a foreign server" >&2; exit 1; fi
trap stop_ollama EXIT
start_ollama
curl -fsS --max-time 30 "$BASE_URL/api/tags" | python -c '
import json,sys
models=json.load(sys.stdin)["models"]
assert any(m["name"]=="gemma4:12b-it-qat" and
           m["digest"]=="38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3"
           for m in models), "screened Gemma4 12B digest missing; no automatic pull"
'
out="$output_base/$label/seed$seed"; purpose=study
samples=4; lr=0.00001
[[ "$system" != gdesigner-paper ]] || { samples=10; lr=0.001; }
if [[ "$mode" == smoke ]]; then
  out="$output_base/smoke/$label/seed$seed"
  purpose=runtime-smoke
  mkdir -p "$out"
  python - "$train" "$dev" "$base" <<'PY'
import json,sys
from pathlib import Path
for source, label, n in ((sys.argv[1], "smoke_train", 2), (sys.argv[2], "smoke_dev", 1)):
    doc = json.loads(Path(source).read_text())
    doc["tasks"] = doc["tasks"][:n]
    path = Path(sys.argv[3]) / (label + ".json")
    if path.exists() and json.loads(path.read_text()) != doc:
        raise ValueError("smoke sample changed")
    if not path.exists():
        path.write_text(json.dumps(doc, sort_keys=True))
PY
  train="$base/smoke_train.json"; dev="$base/smoke_dev.json"; samples=2
fi
common=(--embedding-model "$embedding" --url "$BASE_URL" --threads "$OMP_NUM_THREADS" --max-calls 32)
mkdir -p "$output_base/logs"
echo "mode=$mode system=$system profile=$profile seed=$seed job=$SLURM_JOB_ID commit=$(git rev-parse HEAD)"
if [[ "$mode" != gate ]]; then
  training_args=(--purpose "$purpose")
  [[ -z "$bank" ]] || training_args+=(--training-bank "$bank" --training-condition "$condition")
  [[ -z "$profile" ]] || training_args+=(--anymac-profile "$profile")
  [[ "$system" != anymac-corrected ]] || training_args+=(--backbone "$backbone")
  python scripts/compare/train_learned.py train --system "$system" --tasks "$train" --dev-tasks "$dev" \
    --reserved "${reserved[@]}" --out "$out" --seed "$seed" --samples "$samples" --epochs 1 \
    --lr "$lr" "${training_args[@]}" "${common[@]}" \
    2>&1 | tee -a "$output_base/logs/${mode}_${label}_seed${seed}.log"
fi
if [[ "$mode" == smoke ]]; then
  echo "Runtime smoke finished. NOT a main-study checkpoint and NOT a passed development gate."
else
  python scripts/compare/train_learned.py gate --checkpoint "$out/router.pt" --dev-tasks "$dev" \
    "${common[@]}" 2>&1 | tee -a "$output_base/logs/gate_${label}_seed${seed}.log"
fi
