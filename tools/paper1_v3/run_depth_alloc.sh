#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../../.."
PY=./venv/bin/python
LOG=artifacts/reports/depth_alloc_v3_20260726/run.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for m in thinking26 rltt; do
  say "START $m capture"
  $PY -u utilities/tests/manual/bg_v3_depth_alloc.py --model $m --stage capture >> "$LOG" 2>&1 || { say "FAIL $m capture"; exit 1; }
  for d in 1 2 3 4; do
    say "START $m gen d$d"
    $PY -u utilities/tests/manual/bg_v3_depth_alloc.py --model $m --stage generate --depth $d >> "$LOG" 2>&1 || { say "FAIL $m gen d$d"; exit 1; }
  done
  say "START $m analyze"
  $PY -u utilities/tests/manual/bg_v3_depth_alloc.py --model $m --stage analyze >> "$LOG" 2>&1 || { say "FAIL $m analyze"; exit 1; }
  say "DONE $m"
done
say "ALL DONE"
