#!/usr/bin/env bash
# V5 Thinking pre-answer power replication -- straight runner (GPU assumed free).
# Generate (sealed protocol, offsets >= 1500) then sealed analysis. No wait-guard,
# no stopping rules: the analysis runs once on whatever the generator produced.
set -u
cd "$(dirname "$0")/../../.."
PY=./venv/bin/python
OUT=artifacts/reports/thinking_preanswer_power_v5_20260727
LOG=$OUT/run.log
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

say "START v5 thinking power generate"
$PY -u utilities/tests/manual/bg_v5_thinking_power_generate.py >> "$LOG" 2>&1 \
    || { say "FAIL v5 thinking power generate"; exit 1; }
say "DONE v5 thinking power generate"

say "START v5 thinking power analysis"
$PY -u utilities/tests/manual/bg_v5_thinking_power_analysis.py >> "$LOG" 2>&1 \
    || { say "FAIL v5 thinking power analysis"; exit 1; }
say "DONE v5 thinking power analysis"

say "ALL DONE thinking_power_v5"
