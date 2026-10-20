#!/usr/bin/env bash
# DyLAN dependence study (docs/COMPARATIVE_PROTOCOL.md, "DyLAN-Abhaengigkeitsstudie").
#   prepare  login node: freeze 60 fresh MMLU-Pro eval tasks (screening + preflight samples excluded)
#   run      inside ONE A30 allocation: generate+check persuasive texts, then all jobs, one worker per GPU
#   report   login node: per-run reports, progress and attack-text quality
# Resumable: rerunning `run` skips finished runs and resumes partial ones. No downloads, no training.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in prepare|run|report) ;; *) echo "usage: bash scripts/compare/study.sh prepare|run|report" >&2; exit 2 ;; esac
[[ $# -eq 0 ]] || { echo "unexpected arguments: $*" >&2; exit 2; }
source scripts/ramses/lib.sh
load_env
load_modules
activate_venv

MODEL=gemma4:12b-it-qat
MODEL_DIGEST=38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3  # screening round 2
base="$RESULTS_DIR/compare/dylan_study_v1"
tasks="$base/eval_tasks.json"
attacks="$base/persuasive_texts.json"
reserved=("$RESULTS_DIR/screen/screen_tasks_v1.json" "$RESULTS_DIR/screen/screen_tasks_r2.json"
          "$RESULTS_DIR/compare/gemma12b_preflight_v1/dev_tasks.json")
for f in "${reserved[@]}"; do
  [[ -f "$f" ]] || { echo "missing reserved sample: $f" >&2; exit 1; }
done
DYLAN_ROLES=(Economist Doctor Lawyer Mathematician Psychologist Programmer Historian)
FIXED_ROLES=("Knowledgeable Expert" Critic Analyst)
COMMON=(--reserved "${reserved[@]}" --model "$MODEL" --max-calls 17 --limit 8192 --context 32768)

if [[ "$mode" == prepare ]]; then
  python scripts/screen/screen.py prepare --tasks "$tasks" --benchmarks mmlu_pro \
    --splits "$DATA_DIR/splits/splits_pilot2_seed0.json" --n 60 --seed 20261022 --exclude "${reserved[@]}"
  exit 0
fi
[[ -f "$tasks" ]] || { echo "missing $tasks; run prepare first" >&2; exit 1; }

# One job per line: out-dir <TAB> runner arguments. Honest first (needed for every contrast), then
# role by role so that an interrupted study still has complete systems/forms per finished role.
jobs_list() {
  local s r i form
  for s in dylan dylan-upstream; do
    printf '%s\t%s\n' "honest/$s" "--system $s --policies honest --reps 2 --compromised ${DYLAN_ROLES[0]}"
  done
  printf '%s\t%s\n' "honest/fixed" "--system fixed --policies honest --reps 2 --compromised Critic"
  for i in "${!DYLAN_ROLES[@]}"; do
    for form in bare persuasive; do
      for s in dylan dylan-upstream; do
        r="${DYLAN_ROLES[$i]}"
        printf '%s\t%s\n' "attack/$form/$s/$r" "--system $s --policies attack --reps 1 --compromised $r --attack-form $form"
      done
      if (( i < ${#FIXED_ROLES[@]} )); then
        r="${FIXED_ROLES[$i]}"
        printf '%s\t%s\n' "attack/$form/fixed/${r// /_}" "--system fixed --policies attack --reps 1 --compromised ${r// /_} --attack-form $form"
      fi
    done
  done
}

run_job() {  # $1 = out dir (relative), $2 = argument string; role names with spaces are encoded with _
  local out="$base/$1" a=() x
  read -r -a a <<<"$2"
  for x in "${!a[@]}"; do a[$x]="${a[$x]//_/ }"; done
  if [[ " ${a[*]} " == *" persuasive "* ]]; then a+=(--attack-texts "$attacks"); fi
  python scripts/compare/runner.py run "${a[@]}" --tasks "$tasks" --out "$out" --url "$BASE_URL" \
    --threads "$THREADS" "${COMMON[@]}"
}

if [[ "$mode" == report ]]; then
  [[ -f "$attacks" ]] && python scripts/compare/attack_texts.py check --tasks "$tasks" --out "$attacks" || true
  total=0; done_n=0
  while IFS=$'\t' read -r dir _; do
    total=$((total + 1))
    d="$base/$dir"
    if [[ -f "$d/COMPLETE.json" ]]; then state=complete; done_n=$((done_n + 1))
    elif [[ -f "$d/traces.jsonl" ]]; then state="partial $(wc -l <"$d/traces.jsonl")"
    else state=pending; fi
    if [[ -f "$d/manifest.json" && -f "$d/traces.jsonl" ]]; then
      python scripts/compare/runner.py report --tasks "$tasks" --out "$d" >/dev/null || echo "report failed: $dir" >&2
    fi
    printf '%-45s %s\n' "$dir" "$state"
  done < <(jobs_list)
  echo "complete runs: $done_n/$total"
  exit 0
fi

: "${SLURM_JOB_ID:?Run inside an A30 allocation, not on the login node}"
export OLLAMA_BIN="$RAMSES_CACHE_DIR/ollama-screen-0.40.6/bin/ollama"
export OLLAMA_MODELS="${SCREEN_MODELS_DIR:-$RAMSES_CACHE_DIR/ollama-screen-models}"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
[[ -x "$OLLAMA_BIN" ]] || { echo "screening Ollama binary missing: $OLLAMA_BIN" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock missing" >&2; exit 1; }
gpus=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then  # Slurm's allocation, not every GPU of the node
  IFS=, read -r -a gpus <<<"$CUDA_VISIBLE_DEVICES"
else
  while read -r g; do [[ -n "$g" ]] && gpus+=("$g"); done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
(( ${#gpus[@]} )) || { echo "no GPU visible" >&2; exit 1; }
cpus="${SLURM_CPUS_PER_TASK:-8}"
export THREADS=$(( cpus / ${#gpus[@]} > 0 ? cpus / ${#gpus[@]} : 1 ))
logroot="$RAMSES_LOG_DIR/dylan_study"
mkdir -p "$base/logs" "$logroot"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD) gpus=${#gpus[@]} threads/gpu=$THREADS"

serve() {  # $1 = GPU slot: own port, device and Ollama log; refuses foreign servers and other digests
  export CUDA_VISIBLE_DEVICES="${gpus[$1]}" OLLAMA_PORT=$((11450 + $1))
  export BASE_URL="http://127.0.0.1:$OLLAMA_PORT" OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
  RAMSES_LOG_DIR="$logroot/gpu$1"
  if server_up; then echo "port $OLLAMA_PORT occupied; refusing to reuse another server" >&2; return 1; fi
  start_ollama
  curl -fsS --max-time 30 "$BASE_URL/api/tags" | grep -q "$MODEL_DIGEST" || {
    echo "$MODEL with screened digest $MODEL_DIGEST unavailable; no automatic pull" >&2; return 1; }
}

if [[ ! -f "$attacks" ]]; then
  if ! (
    trap stop_ollama EXIT
    serve 0
    python scripts/compare/attack_texts.py generate --tasks "$tasks" --out "$attacks" --model "$MODEL" \
      --url "$BASE_URL" --threads "$cpus" --context 32768
  ) 2>&1 | tee "$base/logs/attack_texts.log"; then
    echo "persuasive generation failed; no run started" >&2; exit 1
  fi
fi
python scripts/compare/attack_texts.py check --tasks "$tasks" --out "$attacks"

worker() {  # $1 = GPU slot. Dynamic queue: a free GPU claims the next unfinished run via a lock file.
  local g="$1" status=0 dir args rc
  serve "$g" || return 1
  while IFS=$'\t' read -r dir args; do
    [[ -f "$base/$dir/COMPLETE.json" ]] && continue
    mkdir -p "$base/$dir"
    exec 9>"$base/$dir/.claim"  # each worker is its own subshell, so FD 9 is per worker
    rc=0; flock -n 9 || rc=$?
    if [[ $rc -eq 1 ]]; then exec 9>&-; continue; fi  # claimed by the other GPU
    [[ $rc -eq 0 ]] || { echo "[gpu$g] flock failed (rc=$rc)" >&2; return 1; }
    rc=0
    # a run that failed in this pass is not retried by the other GPU; the next pass retries it
    if [[ ! -f "$base/$dir/COMPLETE.json" && ! -f "$base/$dir/.tried_$pass" ]]; then
      touch "$base/$dir/.tried_$pass"
      echo "[gpu$g] start $dir"
      run_job "$dir" "$args" </dev/null >>"$base/logs/${dir//\//__}.log" 2>&1 || rc=$?
      if [[ $rc -eq 0 ]]; then echo "[gpu$g] done $dir"
      else status=1; echo "[gpu$g] FAILED $dir (rc=$rc, log $base/logs/${dir//\//__}.log)"; fi
    fi
    exec 9>&-
  done < <(jobs_list)
  return $status
}

pass="$SLURM_JOB_ID.$$"
pids=()
for g in "${!gpus[@]}"; do
  ( trap stop_ollama EXIT; worker "$g" ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "study pass finished rc=$rc; status: bash scripts/compare/study.sh report"
exit $rc
