#!/usr/bin/env bash
# Regeneration leg: r2 arms trained on v3 executed-branch data (per
# data_flaw_meta_branch_rendering.md), then Part K canaries + combined selection
# (original arms + r2 arms ranked together on the same paired subset, with the
# mean-chars degeneration alarm). Same STOP-file pause semantics as the main pipeline.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=shared/venv/bin/python
M=shared/utilities/tests/manual
CTRL=opi/taps/models/branch_training_offline_verifier_generator_v2/_control
ARMS="C_offline_verifier_dpo_r2 G_budget_plus_verifier_r2 H_from_base_best_recipe_r2"

halt_if_stopped() {
  if [ -e "$CTRL/STOP" ]; then
    echo "[regen] STOP file present — pausing ($1)"
    exit 0
  fi
}

for arm in $ARMS; do
  halt_if_stopped "before train $arm"
  echo "[regen] === train $arm ==="
  ARM=$arm $PY $M/train_offline_branch_generator_v2.py || { echo "[regen] train $arm FAILED"; exit 1; }
  if [ -e "$CTRL/STOP" ] || [ -e "$CTRL/STOP_train_$arm" ]; then
    echo "[regen] paused during/after train $arm"
    exit 0
  fi
done

for arm in $ARMS; do
  halt_if_stopped "before canary $arm"
  echo "[regen] === canary $arm ==="
  ARM=$arm $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py || { echo "[regen] canary $arm FAILED"; exit 1; }
  if [ -e "$CTRL/STOP" ] || [ -e "$CTRL/STOP_canary_arm_$arm" ]; then
    echo "[regen] paused during/after canary $arm"
    exit 0
  fi
done

halt_if_stopped "before combined selection"
echo "[regen] === combined selection (original + r2 arms) ==="
ARM=selection $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py
echo "[regen] DONE through Part K (r2)"
