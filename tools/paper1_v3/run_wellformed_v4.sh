#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../../.."
PY=./venv/bin/python
OUT=artifacts/reports/wellformed_terminal_v4_20260727
DEPTH_LOG=artifacts/reports/depth_alloc_v3_20260726/run.log
LOG=$OUT/run.log
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for depth-alloc to finish (ALL DONE in $DEPTH_LOG)"
until grep -q "ALL DONE\|FAIL rltt" "$DEPTH_LOG"; do sleep 60; done
say "GPU free -- starting well-formed pool generation"

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
say "START wf generate"
$PY -u utilities/tests/manual/bg_v4_wellformed_generate.py >> "$LOG" 2>&1 || { say "FAIL wf generate"; exit 1; }
say "START wf selection analysis"
$PY -u utilities/tests/manual/bg_v4_wellformed_selection.py >> "$LOG" 2>&1 || { say "FAIL wf selection"; exit 1; }
say "ALL DONE wf_v4"
