#!/usr/bin/env bash
# Sequential offline-v2 pipeline: training arms (Part J) then canary checkpoint eval (Part K).
# Every unit is resume-safe; drop a STOP file in $CTRL to pause the whole chain
# (or STOP_train_<arm> / STOP_canary_arm_<arm> for one unit). Re-running this script resumes.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=shared/venv/bin/python
M=shared/utilities/tests/manual
CTRL=opi/taps/models/branch_training_offline_verifier_generator_v2/_control
ARMS="A_format_control B_direct_budget_sft C_offline_verifier_dpo G_budget_plus_verifier H_from_base_best_recipe"

halt_if_stopped() {
  if [ -e "$CTRL/STOP" ]; then
    echo "[pipeline] STOP file present — pausing ($1)"
    exit 0
  fi
}

for arm in $ARMS; do
  halt_if_stopped "before train $arm"
  echo "[pipeline] === train $arm ==="
  ARM=$arm $PY $M/train_offline_branch_generator_v2.py || { echo "[pipeline] train $arm FAILED"; exit 1; }
  if [ -e "$CTRL/STOP" ] || [ -e "$CTRL/STOP_train_$arm" ]; then
    echo "[pipeline] paused during/after train $arm"
    exit 0
  fi
done

for arm in $ARMS; do
  halt_if_stopped "before canary $arm"
  echo "[pipeline] === canary $arm ==="
  ARM=$arm $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py || { echo "[pipeline] canary $arm FAILED"; exit 1; }
  if [ -e "$CTRL/STOP" ] || [ -e "$CTRL/STOP_canary_arm_$arm" ]; then
    echo "[pipeline] paused during/after canary $arm"
    exit 0
  fi
done

halt_if_stopped "before selection"
echo "[pipeline] === Part K selection ==="
ARM=selection $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py
echo "[pipeline] DONE through Part K"
