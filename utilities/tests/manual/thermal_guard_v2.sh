#!/usr/bin/env bash
# Thermal guard for the offline-v2 pipeline. Polls GPU temperature; at >= HOT_C it places the
# STOP file (checkpoint-safe pause via the existing machinery) plus a THERMAL_PAUSE marker,
# waits for <= COOL_C, then removes both and relaunches the orchestrator. A manual STOP
# (no THERMAL_PAUSE marker) is never auto-resumed. Relaunched pipeline output appends to
# $CTRL/pipeline.log. Override via env: HOT_C (default 85), COOL_C (65), POLL_S (30).
set -u
cd "$(dirname "$0")/../../.."
CTRL=opi/taps/models/branch_training_offline_verifier_generator_v2/_control
PIPE=shared/utilities/tests/manual/run_offline_v2_pipeline.sh
LOG="$CTRL/pipeline.log"
HOT_C=${HOT_C:-85}
COOL_C=${COOL_C:-65}
POLL_S=${POLL_S:-30}

gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
pipeline_running() { pgrep -f "run_offline_v2_pipeline.sh" >/dev/null; }

echo "[thermal] guard up: pause >= ${HOT_C}C, resume <= ${COOL_C}C, poll ${POLL_S}s"
while true; do
  T=$(gpu_temp)
  case "$T" in (''|*[!0-9]*) echo "[thermal] bad temp reading '$T'"; sleep "$POLL_S"; continue;; esac
  if pipeline_running; then
    rm -f "$CTRL/THERMAL_PAUSE"
    if [ "$T" -ge "$HOT_C" ]; then
      echo "[thermal] ${T}C >= ${HOT_C}C — pausing pipeline"
      touch "$CTRL/THERMAL_PAUSE" "$CTRL/STOP"
      while pipeline_running; do sleep 5; done
      echo "[thermal] pipeline paused at ${T}C; cooling toward ${COOL_C}C"
    fi
  elif [ -e "$CTRL/THERMAL_PAUSE" ]; then
    if [ -e "$CTRL/STOP" ] && [ "$T" -le "$COOL_C" ]; then
      echo "[thermal] cooled to ${T}C — resuming pipeline"
      rm -f "$CTRL/STOP" "$CTRL/THERMAL_PAUSE"
      nohup bash "$PIPE" >> "$LOG" 2>&1 &
    elif [ ! -e "$CTRL/STOP" ]; then
      rm -f "$CTRL/THERMAL_PAUSE"  # someone resumed manually; clear stale marker
    fi
  fi
  sleep "$POLL_S"
done
