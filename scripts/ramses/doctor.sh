#!/usr/bin/env bash
# Read-only environment check. Starts no inference. Missing GPU on a login node is reported, not an error.
set -u
source "$(dirname "$0")/lib.sh"; load_env
echo "== system"; uname -a; [[ -f /etc/os-release ]] && grep PRETTY_NAME /etc/os-release
echo "user=$(id -un) host=$(hostname) cpus=$(nproc 2>/dev/null || echo ?)"
echo "== config"
for v in RAMSES_PARTITION RAMSES_ACCOUNT RAMSES_QOS RAMSES_GRES RAMSES_MODULES PYTHON_BIN RAMSES_VENV_DIR RAMSES_CACHE_DIR RAMSES_RESULTS_DIR RAMSES_DATA_DIR RAMSES_LOG_DIR OLLAMA_BIN; do
  printf '  %-22s %s\n' "$v" "${!v:-<unset>}"; done
echo "== directories"
for d in "$RAMSES_CACHE_DIR" "$RAMSES_RESULTS_DIR" "$RAMSES_DATA_DIR" "$RAMSES_LOG_DIR"; do
  p="$d"; while [[ ! -e "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
  if [[ -w "$p" ]]; then echo "  OK writable (via $p): $d"; else echo "  WARN not writable: $d"; fi
  df -h "$p" 2>/dev/null | tail -1 | awk '{print "     free: "$4" on "$6}'
done
echo "== modules"
if load_modules; then [[ -n "${RAMSES_MODULES:-}" ]] && echo "  loaded: $RAMSES_MODULES"; else echo "  WARN module load failed"; fi
type module >/dev/null 2>&1 && { echo "  (module avail python/cuda, first lines)"; module -t avail 2>&1 | grep -i -E "python|cuda" | head -20; }
echo "== python"
if command -v "$PYTHON_BIN" >/dev/null; then
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" - <<'PY'
import importlib.util as u
for m in ["torch","numpy","transformers","sentence_transformers","aiohttp","tenacity","pyarrow","class_registry","func_timeout","tiktoken","scipy"]:
    print(f"  {m:22s}", "present" if u.find_spec(m) else "missing (installed by setup.sh into the venv)")
PY
else echo "  WARN $PYTHON_BIN not found"; fi
if [[ -f "$RAMSES_VENV_DIR/bin/python" ]]; then
  echo "  venv: $RAMSES_VENV_DIR"
  "$RAMSES_VENV_DIR/bin/python" - <<'PY'
try:
    import torch; print("  torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available", torch.cuda.is_available())
except Exception as e: print("  torch import failed:", e)
PY
else echo "  venv not created yet"; fi
echo "== slurm"
if command -v sinfo >/dev/null; then
  sinfo -o "%P %a %l %D %G" 2>&1 | head -30
  echo "-- gpu gres per partition (type names to use in RAMSES_GRES)"
  sinfo -h -o "%P %G" 2>&1 | sort -u | head -30
  echo "-- account associations"
  sacctmgr -n -P show assoc user="$(id -un)" format=account,partition,qos 2>&1 | head -20
  echo "-- queue"; squeue -u "$(id -un)" 2>&1 | head
else echo "  slurm not available here"; fi
echo "== gpu"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "  (torch 2.3.0+cu121 needs NVIDIA driver >= 525.60.13)"
else echo "  no GPU visible here (expected on a login node; verified again inside the job by run.sh)"; fi
echo "== ollama"
if [[ -x "$OLLAMA_BIN" ]] || command -v "$OLLAMA_BIN" >/dev/null 2>&1; then "$OLLAMA_BIN" --version 2>&1 | head -2; else echo "  not installed (setup.sh installs it in user space)"; fi
server_up && echo "  server reachable at $BASE_URL" || echo "  no server at $BASE_URL"
echo "== network (downloads must happen on a node with internet access via prepare.sh)"
for u in https://huggingface.co https://registry.ollama.ai https://github.com; do
  printf '  %-30s ' "$u"; curl -sS -m 8 -o /dev/null -w "%{http_code}\n" -I "$u" 2>&1 | tail -1; done
echo "doctor finished"
