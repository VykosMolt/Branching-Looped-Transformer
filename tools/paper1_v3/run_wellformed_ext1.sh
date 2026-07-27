#!/usr/bin/env bash
# SEALED EXTENSION 1 of the well-formed pool (see EXPERIMENT_PLAN.md in the run root).
# Waits for the LoRA S3A pilot to release the GPU, then generates the [1350,1500)
# extension cohort and re-runs the pooled selection analysis.
set -u
cd "$(dirname "$0")/../../.."
PY=./venv/bin/python
OUT=artifacts/reports/wellformed_terminal_v4_20260727
LORA_LOG=artifacts/reports/lora_s3a_pilot_20260727/run.log
LOG=$OUT/run.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "ext1: waiting for lora_s3a_v5 to finish (ALL DONE in $LORA_LOG)"
until grep -q "ALL DONE lora_s3a_v5\|FAIL upstream" "$LORA_LOG"; do sleep 60; done

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
say "START wf ext1 generate [1350,1500)"
$PY -u utilities/tests/manual/bg_v4_wellformed_generate.py --task-offset 1350 --n-tasks 150 --tag ext1 >> "$LOG" 2>&1 || { say "FAIL wf ext1 generate"; exit 1; }
say "START wf pooled selection analysis"
$PY -u utilities/tests/manual/bg_v4_wellformed_selection.py >> "$LOG" 2>&1 || { say "FAIL wf pooled selection"; exit 1; }
say "ALL DONE wf_ext1"
