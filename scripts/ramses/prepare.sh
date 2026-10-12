#!/usr/bin/env bash
# Downloads model(s), dataset revision and embedding model; writes results/<profile>/manifest.json.
# Run on a node with internet access (login node); no GPU is needed.
set -euo pipefail
source "$(dirname "$0")/lib.sh"; load_env; mkdirs; load_modules; activate_venv
profile=""; extra=()
while [[ $# -gt 0 ]]; do case "$1" in
  --profile) profile="$2"; shift 2;;
  --all-models) extra+=(--all-models); shift;;
  *) echo "usage: prepare.sh --profile smoke|pilot|full [--all-models]" >&2; exit 2;; esac; done
[[ -n "$profile" ]] || { echo "--profile required" >&2; exit 2; }
[[ "$profile" == "full" ]] && extra+=(--all-models)
cd "$REPO_ROOT"
start_ollama; trap stop_ollama EXIT
python -m robust prepare --profile "$profile" "${extra[@]}"
echo "prepared. manifest: $RESULTS_DIR/$profile/manifest.json"
