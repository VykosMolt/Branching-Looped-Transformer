#!/usr/bin/env bash
# Cross-loop early-layer v1 — unattended pipeline orchestrator.
# Waits for feature extraction to complete, then runs all analysis stages in order.
set -u
cd /home/moloch/ouro_project
RUN=artifacts/reports/cross_loop_early_layer_taps_20260720
PY=./venv/bin/python
LOG="$RUN/logs/pipeline.log"

echo "[pipeline] start $(date)" >> "$LOG"

# wait for extraction completion marker (extraction_seconds in manifest)
until $PY - <<'EOF' 2>/dev/null
import json, sys
m = json.load(open('artifacts/reports/cross_loop_early_layer_taps_20260720/features/feature_manifest.json'))
sys.exit(0 if 'extraction_seconds' in m else 1)
EOF
do
  sleep 30
done
echo "[pipeline] extraction complete $(date)" >> "$LOG"

run_stage() {
  name=$1; shift
  echo "[pipeline] stage $name start $(date)" >> "$LOG"
  if "$@" >> "$RUN/logs/$name.log" 2>&1; then
    echo "[pipeline] stage $name OK $(date)" >> "$LOG"
  else
    echo "[pipeline] stage $name FAILED rc=$? $(date)" >> "$LOG"
    echo "PIPELINE_FAILED_AT_$name" >> "$LOG"
    exit 1
  fi
}

run_stage train_eval $PY -u shared/utilities/tests/manual/bg_xloop_early_v1_train_eval.py
run_stage controls   $PY -u shared/utilities/tests/manual/bg_xloop_early_v1_controls.py
run_stage stats      $PY -u shared/utilities/tests/manual/bg_xloop_early_v1_stats.py
run_stage plots      $PY -u shared/utilities/tests/manual/bg_xloop_early_v1_plots.py
run_stage tests      $PY -u shared/utilities/tests/manual/bg_xloop_early_v1_tests.py
run_stage s3b2       $PY -u shared/utilities/tests/manual/bg_xloop_early_v1_s3b2.py

echo "PIPELINE_COMPLETE $(date)" >> "$LOG"
