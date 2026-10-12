#!/usr/bin/env bash
# Creates the Python venv and installs a user-space Ollama binary. No admin rights, no system drivers.
set -euo pipefail
source "$(dirname "$0")/lib.sh"; load_env; mkdirs; load_modules
cd "$REPO_ROOT"
if [[ ! -f "$RAMSES_VENV_DIR/bin/activate" ]]; then
  "$PYTHON_BIN" -m venv "$RAMSES_VENV_DIR"
fi
activate_venv
python --version
python -m pip install --upgrade pip wheel
os="$(uname -s)"; arch="$(uname -m)"
if [[ "$os" == "Linux" ]]; then
  # torch 2.3.0 from PyPI is the CUDA 12.1 build on Linux; TORCH_INDEX_URL may point to the pytorch cu121 index instead.
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then python -m pip install "torch==2.3.0" --index-url "$TORCH_INDEX_URL"
  else python -m pip install "torch==2.3.0"; fi
elif [[ "$os" == "Darwin" && "$arch" == "x86_64" ]]; then python -m pip install "torch==2.2.2"   # last x86_64 macOS wheel
else python -m pip install "torch==2.3.0"; fi
python -m pip install -r requirements-ramses.txt
python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "numpy", numpy.__version__)
PY
# Ollama (user space)
if [[ ! -x "$OLLAMA_BIN" ]] && ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  if [[ "$os" == "Linux" ]]; then
    case "$arch" in x86_64) pkg=amd64;; aarch64) pkg=arm64;; *) echo "unsupported arch $arch" >&2; exit 1;; esac
    if [[ -n "${OLLAMA_VERSION:-}" ]]; then url="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${pkg}.tar.zst"
    else url="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-${pkg}.tar.zst"; fi
    dest="$(dirname "$(dirname "$OLLAMA_BIN")")"; mkdir -p "$dest"
    curl -fL --retry 3 -o "$dest/ollama.tar.zst" "$url"
    python - "$dest" <<'PY'
import sys, tarfile, zstandard, os
d = sys.argv[1]
with open(os.path.join(d, "ollama.tar.zst"), "rb") as f:
    with zstandard.ZstdDecompressor().stream_reader(f) as r:
        with tarfile.open(fileobj=r, mode="r|") as t:
            t.extractall(d)
PY
    rm -f "$dest/ollama.tar.zst"
    [[ -x "$OLLAMA_BIN" ]] || { echo "expected $OLLAMA_BIN after extraction; set OLLAMA_BIN to the extracted binary" >&2; exit 1; }
  else
    echo "[ollama] not installing on $os; install Ollama manually or set OLLAMA_BIN" >&2
  fi
fi
"$OLLAMA_BIN" --version || true
echo "setup finished. Next: scripts/ramses/prepare.sh --profile pilot"
