#!/usr/bin/env bash
# V3 overnight programme: Horizon power extension + within-family replication
# + Huginn depth-recurrence probe. Single-GPU sequential. Every stage is
# idempotent/resumable; a failure inside one experiment aborts that experiment
# but the others still run.
set -u
cd "$(dirname "$0")/../../.."   # repo root
PY=./venv/bin/python
LOGROOT=artifacts/reports/v3_overnight_20260726_logs
mkdir -p "$LOGROOT"
MASTER="$LOGROOT/master.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

cool() {  # wait until GPU below 80C
  while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$t" ] && return 0
    [ "$t" -lt 80 ] && return 0
    say "thermal pause: GPU at ${t}C"; sleep 60
  done
}

run_stage() {  # run_stage <experiment> <name> <cmd...>
  local exp="$1" name="$2"; shift 2
  local log="$LOGROOT/${exp}_${name}.log"
  cool
  say "START $exp/$name"
  if "$@" >"$log" 2>&1; then
    say "OK    $exp/$name"
    return 0
  else
    say "FAIL  $exp/$name (see $log)"
    echo "FAILED_AT_${name}" > "$LOGROOT/${exp}.FAILED"
    return 1
  fi
}

say "=== V3 overnight start ==="

# ---------------- Experiment A: Horizon Logic power extension ----------------
horizon() {
  for off in 170 340 510; do
    if ls artifacts/reports/horizon_power_v3_20260726/horizon_logic/horizon_generation_v3_off${off}.pt >/dev/null 2>&1; then
      say "skip horizon shard off$off (exists)"; continue
    fi
    run_stage horizon "gen_off${off}" $PY -u utilities/tests/manual/bg_v3_horizon_power_generate.py \
      --n-tasks 170 --k 4 --max-new 448 --seed 20260724 --tag v3 \
      --task-offset ${off} --budget-seconds 9000 || return 1
  done
  run_stage horizon analysis $PY -u utilities/tests/manual/bg_v3_horizon_power_analysis.py || return 1
}

# ---------------- Experiment B: within-family replication --------------------
family() {
  for m in base26 thinking26 ouro14b; do
    run_stage family "${m}_prep"    $PY -u utilities/tests/manual/bg_v3_family_xloop.py --model $m --stage prep || return 1
    run_stage family "${m}_extract" $PY -u utilities/tests/manual/bg_v3_family_xloop.py --model $m --stage extract || return 1
    run_stage family "${m}_traineval" $PY -u utilities/tests/manual/bg_v3_family_xloop.py --model $m --stage traineval || return 1
    run_stage family "${m}_crossmodel" $PY -u utilities/tests/manual/bg_v3_family_xloop.py --model $m --stage crossmodel || return 1
    run_stage family "${m}_stats"   $PY -u utilities/tests/manual/bg_v3_family_xloop.py --model $m --stage stats || return 1
  done
}

# ---------------- Experiment C: Huginn probe ----------------------------------
huginn() {
  run_stage huginn prep      $PY -u utilities/tests/manual/bg_v3_huginn_probe.py --stage prep || return 1
  run_stage huginn extract   $PY -u utilities/tests/manual/bg_v3_huginn_probe.py --stage extract || return 1
  run_stage huginn traineval $PY -u utilities/tests/manual/bg_v3_huginn_probe.py --stage traineval || return 1
  run_stage huginn stats     $PY -u utilities/tests/manual/bg_v3_huginn_probe.py --stage stats || return 1
}

horizon; family; huginn

say "=== V3 overnight complete ==="
for exp in horizon family huginn; do
  if [ -f "$LOGROOT/${exp}.FAILED" ]; then
    say "RESULT $exp: FAILED ($(cat "$LOGROOT/${exp}.FAILED"))"
  else
    say "RESULT $exp: OK"
  fi
done
