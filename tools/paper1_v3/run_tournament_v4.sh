#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../../.."
PY=./venv/bin/python
OUT=artifacts/reports/tournament_v4_20260727
WF_LOG=artifacts/reports/wellformed_terminal_v4_20260727/run.log
LOG=$OUT/run.log
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for wf_ext1 to finish (ALL DONE in $WF_LOG)"
until grep -q "ALL DONE wf_ext1\|FAIL wf ext1\|FAIL wf pooled" "$WF_LOG"; do sleep 60; done
say "GPU free -- starting tournament generation"

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "START tournament generate"
$PY -u utilities/tests/manual/bg_v4_tournament.py --stage generate >> "$LOG" 2>&1 || { say "FAIL tournament generate"; exit 1; }
say "START tournament analyze"
$PY -u utilities/tests/manual/bg_v4_tournament.py --stage analyze >> "$LOG" 2>&1 || { say "FAIL tournament analyze"; exit 1; }
say "ALL DONE tournament_v4"
