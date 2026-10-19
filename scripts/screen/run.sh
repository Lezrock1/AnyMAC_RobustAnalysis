#!/usr/bin/env bash
# Model x benchmark screening on A30. Separate, pinned Ollama + model store: never touches the
# installation used by running confirmation jobs.
#   install | prepare | pull [MODEL...] | dedupe | run MODEL | all | report
# SCREEN_MODELS_DIR overrides the screen model store (default: $RAMSES_CACHE_DIR/ollama-screen-models).
# `all` (inside one interactive A30 allocation): one worker per visible GPU, models distributed
# round-robin and run sequentially per GPU, separate Ollama port and log directory per GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; shift || true
case "$mode" in install|prepare|pull|dedupe|run|all|report) ;; *) echo "usage: run.sh install|prepare|pull|dedupe|run MODEL|all|report" >&2; exit 2;; esac
source scripts/ramses/lib.sh
load_env; load_modules; activate_venv
SCREEN_OLLAMA_VERSION=0.40.6
dest="$RAMSES_CACHE_DIR/ollama-screen-$SCREEN_OLLAMA_VERSION"
# SCREEN_MODELS_DIR may point to another filesystem (e.g. /projects) when the home quota is tight.
export OLLAMA_BIN="$dest/bin/ollama" OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_PORT="${SCREEN_PORT:-11440}"
export BASE_URL="http://127.0.0.1:$OLLAMA_PORT"
# The ollama CLI (pull/show/list) finds the server only via OLLAMA_HOST; default would be 11434.
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
# configs/ramses.env may set RAMSES_LOG_DIR, so per-GPU Ollama logs are separated after load_env
RAMSES_LOG_DIR="$RAMSES_LOG_DIR/screen${SCREEN_GPU:+_gpu$SCREEN_GPU}"
base="$RESULTS_DIR/screen"
tasks="$base/screen_tasks_v1.json"
out="$base/screen_v2"  # v1 aborted: 192 Ollama threads on 8 CPUs, parser v1
mapfile -t MODELS < <(grep -v '^\s*#' scripts/screen/models.txt | sed '/^\s*$/d')

if [[ "$mode" == install ]]; then
  if [[ -x "$OLLAMA_BIN" ]]; then "$OLLAMA_BIN" --version; exit 0; fi
  mkdir -p "$dest"
  curl -fL --retry 3 -o "$dest/ollama.tar.zst" \
    "https://github.com/ollama/ollama/releases/download/v${SCREEN_OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
  python - "$dest" <<'PY'
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
  exec python scripts/screen/screen.py prepare --splits "$DATA_DIR/splits/splits_pilot2_seed0.json" --tasks "$tasks"
fi
if [[ "$mode" == report ]]; then
  exec python scripts/screen/screen.py report --out "$out"
fi

[[ -x "$OLLAMA_BIN" ]] || { echo "run 'run.sh install' first" >&2; exit 1; }
if [[ "$mode" == all ]]; then
  [[ -f "$tasks" ]] || { echo "missing $tasks; run 'run.sh prepare' first" >&2; exit 1; }
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  (( ${#gpus[@]} )) || { echo "no GPU visible" >&2; exit 1; }
  mkdir -p "$base/logs"
  pids=()
  for g in "${!gpus[@]}"; do
    (
      export CUDA_VISIBLE_DEVICES="${gpus[$g]}" SCREEN_PORT=$((11440 + g)) SCREEN_GPU="$g" SCREEN_EVICT="${SCREEN_EVICT:-1}"
      status=0
      for ((i = g; i < ${#MODELS[@]}; i += ${#gpus[@]})); do
        m="${MODELS[$i]}"
        bash scripts/screen/run.sh run "$m" >>"$base/logs/${m//[:\/]/_}.log" 2>&1 || { status=1; echo "FAILED $m (see $base/logs)"; }
      done
      exit $status
    ) &
    pids+=($!)
  done
  echo "${#gpus[@]} GPU worker(s) for ${#MODELS[@]} models; logs: $base/logs/"
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  python scripts/screen/screen.py report --out "$out" >/dev/null || true
  echo "all done rc=$rc; report: $out/REPORT.md"
  exit $rc
fi
if server_up; then echo "Port $OLLAMA_PORT occupied; set SCREEN_PORT" >&2; exit 1; fi
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
# Blobs are content-addressed (file name = sha256), so a blob also present in the confirm store
# can be replaced by a hardlink: identical bytes, counted once by the quota. Same filesystem only.
dedupe_blobs() {
  local src="$RAMSES_CACHE_DIR/ollama/models/blobs" b t n=0
  [[ -d "$src" && -d "$OLLAMA_MODELS/blobs" ]] || return 0
  for t in "$OLLAMA_MODELS"/blobs/sha256-*; do
    [[ "$t" == *-partial ]] && continue
    b="$src/$(basename "$t")"
    [[ -f "$b" && ! "$t" -ef "$b" ]] || continue
    [[ "$(stat -c %s "$t")" == "$(stat -c %s "$b")" ]] || continue
    ln -f "$b" "$t" 2>/dev/null && n=$((n+1))
  done
  echo "[dedupe] hardlinked $n blob(s) from $src"
}

if [[ "$mode" == dedupe ]]; then dedupe_blobs; du -sh "$OLLAMA_MODELS"; exit 0; fi

if [[ "$mode" == pull ]]; then
  # Login node: downloads only, no model is loaded. Jobs never download.
  export CUDA_VISIBLE_DEVICES=""
  start_ollama; trap stop_ollama EXIT
  if (( $# )); then list=("$@"); else list=("${MODELS[@]}"); fi
  for m in "${list[@]}"; do "$OLLAMA_BIN" pull "$m"; done
  dedupe_blobs
  "$OLLAMA_BIN" list
  du -sh "$OLLAMA_MODELS"
  exit 0
fi

model="${1:-}"
[[ -n "$model" ]] || { echo "usage: run.sh run MODEL" >&2; exit 2; }
shift || true
[[ -f "$tasks" ]] || { echo "missing $tasks; run 'run.sh prepare' first" >&2; exit 1; }
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) model=$model port=$OLLAMA_PORT commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
start_ollama; trap stop_ollama EXIT
# Just-in-time download (home quota): pull if missing; the manifest binds the digest, so a re-pull
# that yields a different digest makes screen.py refuse to resume instead of mixing versions.
if ! "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
  "$OLLAMA_BIN" pull "$model" || { echo "pull of $model failed (no network on this node? then 'run.sh pull $model' on the login node)" >&2; exit 1; }
  dedupe_blobs
fi
python scripts/screen/screen.py run --tasks "$tasks" --out "$out" --model "$model" --url "$BASE_URL" "$@"
if [[ "${SCREEN_EVICT:-0}" == 1 ]]; then
  # Keep models whose blobs are shared with the confirm store (hardlinks): deleting frees nothing.
  digest="$("$OLLAMA_BIN" show "$model" --modelfile 2>/dev/null | sed -n 's/^FROM .*blobs\/\(sha256-[0-9a-f]*\).*/\1/p' | head -1)"
  if [[ -n "$digest" && "$(stat -c %h "$OLLAMA_MODELS/blobs/$digest" 2>/dev/null || echo 1)" -gt 1 ]]; then
    echo "[evict] keeping $model (shared with confirm store)"
  else
    "$OLLAMA_BIN" rm "$model" && echo "[evict] removed $model"
  fi
fi
grep -iE "offload|layers.*gpu|total memory" "$RAMSES_LOG_DIR/ollama_${SLURM_JOB_ID:-local}.log" | tail -5 || true
