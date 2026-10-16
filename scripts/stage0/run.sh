#!/usr/bin/env bash
# Stage 0 runner. Separate allocation; reads production results only, never writes into them.
#   bash scripts/stage0/run.sh sim   [--jobs N] [--seeds N] [--rounds N] [--texts] [--out DIR]
#   bash scripts/stage0/run.sh noise [--reps R] [--hours H] [--ckpts "branch_A/round_0012.pth ..."] [--out DIR]
#   bash scripts/stage0/run.sh report
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; shift || true
[[ "$mode" =~ ^(sim|noise|report)$ ]] || { sed -n 2,5p "$0" >&2; exit 2; }
export OLLAMA_PORT_OVERRIDE=11437  # 11434 4B, 11435 30B pilot2, 11436 diagnostics
source scripts/ramses/lib.sh
load_env; load_modules; activate_venv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
base="$RESULTS_DIR/stage0"
jobs=8; seeds=5; rounds=60; texts=0; reps=2; hours=22; out=""
ckpts="branch_A/round_0012.pth branch_B/round_0012.pth"
while [[ $# -gt 0 ]]; do case "$1" in
  --jobs) jobs="$2"; shift 2;;
  --seeds) seeds="$2"; shift 2;;
  --rounds) rounds="$2"; shift 2;;
  --texts) texts=1; shift;;
  --reps) reps="$2"; shift 2;;
  --hours) hours="$2"; shift 2;;
  --ckpts) ckpts="$2"; shift 2;;
  --out) out="$2"; shift 2;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done
splits="$DATA_DIR/splits/splits_pilot2_seed0.json"

if [[ "$mode" == "report" ]]; then
  for d in "$base"/sim_*; do [[ -d "$d" ]] && python scripts/stage0/sim.py summarize --out "$d" >/dev/null && echo "$d/REPORT.md"; done
  sim_sum="$(ls -d "$base"/sim_synthetic/summary.json 2>/dev/null || true)"
  for d in "$base"/noise_*; do [[ -f "$d/meta.json" ]] && python scripts/stage0/noise_floor.py report --out "$d" ${sim_sum:+--sim-summary "$sim_sum"} >/dev/null && echo "$d/REPORT.md"; done
  exit 0
fi

: "${SLURM_JOB_ID:?Run inside a separate Slurm allocation (srun/sbatch), not on the login node}"
echo "job=$SLURM_JOB_ID node=$(hostname) mode=$mode commit=$(git rev-parse --short HEAD)"

if [[ "$mode" == "sim" ]]; then
  tag="synthetic"; export S0_TEXTS=""
  if [[ $texts -eq 1 ]]; then
    [[ -f "$splits" ]] || { echo "$splits missing" >&2; exit 1; }
    tag="realtexts"; S0_TEXTS="$splits"
  fi
  out="${out:-$base/sim_$tag}"
  mkdir -p "$out/logs"
  threads=$(( ${SLURM_CPUS_PER_TASK:-$jobs} / jobs )); (( threads >= 1 )) || threads=1
  export S0_OUT="$out" S0_ROUNDS="$rounds" S0_THREADS="$threads"
  echo "out=$out jobs=$jobs threads/job=$threads seeds=$seeds rounds=$rounds"
  # positive controls first (they decide whether the rest is interpretable), then the null, then withholding
  for sc in oracle_cond oracle_uncond honest withhold_green; do
    for v in full original indices; do
      for s in $(seq 0 $((seeds - 1))); do echo "$sc $v $s"; done
    done
  done | xargs -P "$jobs" -L 1 bash -c '
    log="$S0_OUT/logs/$0__$1__seed$2.log"
    if python scripts/stage0/sim.py run --out "$S0_OUT" --scenario "$0" --variant "$1" --seed "$2" \
         --rounds "$S0_ROUNDS" --threads "$S0_THREADS" ${S0_TEXTS:+--texts "$S0_TEXTS"} >"$log" 2>&1; then
      echo "done $0/$1/seed$2 $(tail -1 "$log")"
    else
      echo "FAILED $0/$1/seed$2 (see $log)"
    fi' | tee "$out/grid_status.txt"
  python scripts/stage0/sim.py summarize --out "$out" >/dev/null
  echo "Report: $out/REPORT.md"
  if grep -q '^FAILED' "$out/grid_status.txt"; then echo "Some runs failed; rerunning the same command retries only those." >&2; exit 1; fi
  exit 0
fi

# noise: own Ollama server with the production pilot2 server settings
nvidia-smi -L
out="${out:-$base/noise_pilot2_qwen30_seed0}"
mkdir -p "$out"
if server_up; then echo "Port $OLLAMA_PORT is occupied; refusing to reuse a foreign server." >&2; exit 1; fi
export OLLAMA_NUM_PARALLEL="$(python -m robust profile-get pilot2 ollama.num_parallel)"
if [[ "$(python -m robust profile-get pilot2 ollama.flash_attention 2>/dev/null || true)" == "True" ]]; then
  export OLLAMA_FLASH_ATTENTION=1; export OLLAMA_KV_CACHE_TYPE="$(python -m robust profile-get pilot2 ollama.kv_cache_type)"
fi
{ echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD)"; nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
  "$OLLAMA_BIN" --version; env | grep -E '^OLLAMA_(NUM_PARALLEL|FLASH|KV)' || true; } >>"$out/hardware.txt" 2>&1
start_ollama; trap stop_ollama EXIT
deadline=$(($(date +%s) + hours * 3600))
sim_sum="$base/sim_synthetic/summary.json"
# shellcheck disable=SC2086
python scripts/stage0/noise_floor.py run --source "$RESULTS_DIR/pilot2/qwen30/seed0" --out "$out" \
  --ckpts $ckpts --reps "$reps" --deadline "$deadline" ${sim_sum:+--sim-summary "$sim_sum"} 2>&1 \
  | grep -v "not recognized by tiktoken" | tee -a "$out/run.log"
python scripts/stage0/noise_floor.py report --out "$out" $( [[ -f "$sim_sum" ]] && echo --sim-summary "$sim_sum" ) >/dev/null
echo "Report: $out/REPORT.md"
