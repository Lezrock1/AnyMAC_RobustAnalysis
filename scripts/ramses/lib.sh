#!/usr/bin/env bash
# Shared helpers; source this file. No cluster-specific values are hard-coded.
REPO_ROOT="${ANYMAC_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

load_env() {
  local f="${RAMSES_ENV_FILE:-$REPO_ROOT/configs/ramses.env}"
  if [[ -f "$f" ]]; then
    set -a; # shellcheck disable=SC1090
    source "$f"; set +a
  else
    echo "[env] $f not found; using defaults (copy configs/ramses.env.example to configs/ramses.env)" >&2
  fi
  : "${RAMSES_CACHE_DIR:=$REPO_ROOT/cache}"
  : "${RAMSES_RESULTS_DIR:=$REPO_ROOT/results}"
  : "${RAMSES_DATA_DIR:=$REPO_ROOT/data}"
  : "${RAMSES_LOG_DIR:=$REPO_ROOT/logs}"
  : "${RAMSES_VENV_DIR:=$REPO_ROOT/.venv}"
  : "${PYTHON_BIN:=python3}"
  : "${OLLAMA_PORT:=11434}"
  : "${OLLAMA_BIN:=$RAMSES_CACHE_DIR/ollama/bin/ollama}"
  export RESULTS_DIR="$RAMSES_RESULTS_DIR" DATA_DIR="$RAMSES_DATA_DIR"
  export HF_HOME="$RAMSES_CACHE_DIR/hf" OLLAMA_MODELS="$RAMSES_CACHE_DIR/ollama/models"
  export PIP_CACHE_DIR="$RAMSES_CACHE_DIR/pip" OLLAMA_BIN
  export BASE_URL="http://127.0.0.1:${OLLAMA_PORT}"
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

load_modules() {
  [[ -z "${RAMSES_MODULES:-}" ]] && return 0
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
      [[ -f "$f" ]] && source "$f" && break
    done
  fi
  type module >/dev/null 2>&1 || { echo "[env] 'module' command unavailable but RAMSES_MODULES is set" >&2; return 1; }
  for m in $RAMSES_MODULES; do module load "$m" || return 1; done
}

activate_venv() {
  [[ -f "$RAMSES_VENV_DIR/bin/activate" ]] || { echo "venv missing at $RAMSES_VENV_DIR; run scripts/ramses/setup.sh" >&2; return 1; }
  # shellcheck disable=SC1091
  source "$RAMSES_VENV_DIR/bin/activate"
}

mkdirs() { mkdir -p "$RAMSES_CACHE_DIR" "$RAMSES_RESULTS_DIR" "$RAMSES_DATA_DIR" "$RAMSES_LOG_DIR" "$HF_HOME" "$OLLAMA_MODELS" "$PIP_CACHE_DIR"; }

server_up() { curl -fsS -m 3 "$BASE_URL/api/tags" >/dev/null 2>&1; }

# Starts an Ollama server only if none answers; remembers the PID so only our own process is stopped.
OLLAMA_STARTED_PID=""
start_ollama() {
  if server_up; then echo "[ollama] server already running at $BASE_URL (not managed by this script)"; return 0; fi
  [[ -x "$OLLAMA_BIN" ]] || command -v "$OLLAMA_BIN" >/dev/null 2>&1 || { echo "[ollama] binary not found: $OLLAMA_BIN (run setup.sh)" >&2; return 1; }
  local par="${OLLAMA_NUM_PARALLEL:-4}"
  mkdir -p "$RAMSES_LOG_DIR"
  OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" OLLAMA_NUM_PARALLEL="$par" OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}" \
    OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}" \
    nohup "$OLLAMA_BIN" serve >"$RAMSES_LOG_DIR/ollama_${SLURM_JOB_ID:-local}.log" 2>&1 &
  OLLAMA_STARTED_PID=$!
  local i
  for i in $(seq 1 120); do
    server_up && { echo "[ollama] ready (pid $OLLAMA_STARTED_PID)"; return 0; }
    kill -0 "$OLLAMA_STARTED_PID" 2>/dev/null || { echo "[ollama] server exited early; see $RAMSES_LOG_DIR" >&2; return 1; }
    sleep 1
  done
  echo "[ollama] not ready after 120 s" >&2; return 1
}

stop_ollama() {
  if [[ -n "$OLLAMA_STARTED_PID" ]] && kill -0 "$OLLAMA_STARTED_PID" 2>/dev/null; then
    kill "$OLLAMA_STARTED_PID" 2>/dev/null; wait "$OLLAMA_STARTED_PID" 2>/dev/null || true
    echo "[ollama] stopped own server (pid $OLLAMA_STARTED_PID)"
  fi
}
