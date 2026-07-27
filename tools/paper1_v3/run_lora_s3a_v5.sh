#!/usr/bin/env bash
# Frozen conversion #5 -- LoRA direction-outcome binding pilot (S3A pilot).
# Sealed plan: artifacts/reports/lora_s3a_pilot_20260727/EXPERIMENT_PLAN.md.
# Waits for tournament_v4 (its table is our training data AND its run holds the GPU),
# then: train binding -> train control -> gate -> eval -> analyze.
# Train/gate failure = TRAINING_UNSTABLE per the sealed plan: recorded via analyze,
# never retried with new hyperparameters. Eval crashes resume per-arm on re-run.
set -u
cd "$(dirname "$0")/../../.."
PY=./venv/bin/python
SCRIPT=utilities/tests/manual/bg_v5_lora_binding.py
OUT=artifacts/reports/lora_s3a_pilot_20260727
TOUR_LOG=artifacts/reports/tournament_v4_20260727/run.log
LOG=$OUT/run.log
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for tournament_v4 to finish (ALL DONE in $TOUR_LOG)"
until grep -q "ALL DONE tournament_v4\|FAIL tournament" "$TOUR_LOG"; do sleep 60; done
if grep -q "FAIL tournament" "$TOUR_LOG"; then
  say "FAIL upstream tournament_v4 -- no training table, aborting"
  exit 1
fi
say "GPU free and tournament table ready -- starting lora_s3a_v5"

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
FAILED=""

say "START train binding"
if $PY -u $SCRIPT --stage train --adapter binding >> "$LOG" 2>&1; then
  say "DONE train binding"
else
  say "FAIL train binding (recorded, no retry per sealed plan)"; FAILED="train_binding"
fi

if [ -z "$FAILED" ]; then
  say "START train control"
  if $PY -u $SCRIPT --stage train --adapter control >> "$LOG" 2>&1; then
    say "DONE train control"
  else
    say "FAIL train control (recorded, no retry per sealed plan)"; FAILED="train_control"
  fi
fi

if [ -z "$FAILED" ]; then
  say "START gate"
  if $PY -u $SCRIPT --stage gate >> "$LOG" 2>&1; then
    say "DONE gate"
  else
    say "FAIL gate (recorded)"; FAILED="gate"
  fi
fi

if [ -n "$FAILED" ]; then
  say "stage $FAILED failed -- skipping eval; running analyze to record TRAINING_UNSTABLE"
else
  say "START eval"
  if $PY -u $SCRIPT --stage eval >> "$LOG" 2>&1; then
    say "DONE eval"
  else
    say "FAIL eval (re-run this script to resume per-arm)"
    exit 1
  fi
fi

say "START analyze"
if $PY -u $SCRIPT --stage analyze >> "$LOG" 2>&1; then
  say "DONE analyze"
else
  say "FAIL analyze"
  exit 1
fi

say "ALL DONE lora_s3a_v5"
