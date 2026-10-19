#!/usr/bin/env bash
# Model x benchmark screening on A30. Separate, pinned Ollama + model store: never touches the
# installation used by running confirmation jobs.
#   install | prepare | pull [MODEL...] | run MODEL | report
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; shift || true
case "$mode" in install|prepare|pull|run|report) ;; *) echo "usage: run.sh install|prepare|pull|run MODEL|report" >&2; exit 2;; esac
source scripts/ramses/lib.sh
load_env
SCREEN_OLLAMA_VERSION=0.40.6
dest="$RAMSES_CACHE_DIR/ollama-screen-$SCREEN_OLLAMA_VERSION"
export OLLAMA_BIN="$dest/bin/ollama" OLLAMA_MODELS="$RAMSES_CACHE_DIR/ollama-screen-models"
export OLLAMA_PORT="${SCREEN_PORT:-$((11440 + ${SLURM_ARRAY_TASK_ID:-0}))}"
export BASE_URL="http://127.0.0.1:$OLLAMA_PORT"
base="$RESULTS_DIR/screen"
tasks="$base/screen_tasks_v1.json"
out="$base/screen_v1"
mapfile -t MODELS < <(grep -v '^\s*#' scripts/screen/models.txt | sed '/^\s*$/d')

if [[ "$mode" == install ]]; then
  if [[ -x "$OLLAMA_BIN" ]]; then "$OLLAMA_BIN" --version; exit 0; fi
  mkdir -p "$dest"
  curl -fL --retry 3 -o "$dest/ollama.tar.zst" \
    "https://github.com/ollama/ollama/releases/download/v${SCREEN_OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
  "$PYTHON_BIN" - "$dest" <<'PY'
import os, sys, tarfile, zstandard
d = sys.argv[1]
with open(os.path.join(d, "ollama.tar.zst"), "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
    with tarfile.open(fileobj=r, mode="r|") as t:
        t.extractall(d)
PY
  rm -f "$dest/ollama.tar.zst"
  "$OLLAMA_BIN" --version
  exit 0
fi
if [[ "$mode" == prepare ]]; then
  mkdir -p "$base"
  exec "$PYTHON_BIN" scripts/screen/screen.py prepare --splits "$DATA_DIR/splits/splits_pilot2_seed0.json" --tasks "$tasks"
fi
if [[ "$mode" == report ]]; then
  exec "$PYTHON_BIN" scripts/screen/screen.py report --out "$out"
fi

[[ -x "$OLLAMA_BIN" ]] || { echo "run 'run.sh install' first" >&2; exit 1; }
if server_up; then echo "Port $OLLAMA_PORT occupied; set SCREEN_PORT" >&2; exit 1; fi
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
if [[ "$mode" == pull ]]; then
  # Login node: downloads only, no model is loaded. Jobs never download.
  export CUDA_VISIBLE_DEVICES=""
  start_ollama; trap stop_ollama EXIT
  if (( $# )); then list=("$@"); else list=("${MODELS[@]}"); fi
  for m in "${list[@]}"; do "$OLLAMA_BIN" pull "$m"; done
  "$OLLAMA_BIN" list
  exit 0
fi

model="${1:-}"
if [[ -z "$model" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then model="${MODELS[$SLURM_ARRAY_TASK_ID]}"; fi
[[ -n "$model" ]] || { echo "usage: run.sh run MODEL" >&2; exit 2; }
shift || true
[[ -f "$tasks" ]] || { echo "missing $tasks; run 'run.sh prepare' first" >&2; exit 1; }
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) model=$model port=$OLLAMA_PORT commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
start_ollama; trap stop_ollama EXIT
"$OLLAMA_BIN" show "$model" >/dev/null || { echo "model $model not pulled; run 'run.sh pull' on the login node" >&2; exit 1; }
"$PYTHON_BIN" scripts/screen/screen.py run --tasks "$tasks" --out "$out" --model "$model" --url "$BASE_URL" "$@"
grep -iE "offload|layers.*gpu|total memory" "$RAMSES_LOG_DIR/ollama_${SLURM_JOB_ID:-local}.log" | tail -5 || true
