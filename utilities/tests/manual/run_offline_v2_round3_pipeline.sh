#!/usr/bin/env bash
# Round 3: C-style verifier-DPO arm on multi-domain WORKED data (train_v4: executed logic +
# GSM8K/Hendrycks rationale math + canonical-solution coding + show_work anti-bare pairs).
# No budget loss (retired per round2_executed_data_result.md). Same STOP-file semantics.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=shared/venv/bin/python
M=shared/utilities/tests/manual
CTRL=opi/taps/models/branch_training_offline_verifier_generator_v2/_control
ARM_R3=C_offline_verifier_dpo_r3

halt_if_stopped() {
  if [ -e "$CTRL/STOP" ]; then
    echo "[round3] STOP file present — pausing ($1)"
    exit 0
  fi
}

halt_if_stopped "before train $ARM_R3"
echo "[round3] === train $ARM_R3 ==="
ARM=$ARM_R3 $PY $M/train_offline_branch_generator_v2.py || { echo "[round3] train FAILED"; exit 1; }
if [ -e "$CTRL/STOP" ] || [ -e "$CTRL/STOP_train_$ARM_R3" ]; then
  echo "[round3] paused during/after train"
  exit 0
fi

halt_if_stopped "before canary $ARM_R3"
echo "[round3] === canary $ARM_R3 ==="
ARM=$ARM_R3 $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py || { echo "[round3] canary FAILED"; exit 1; }

halt_if_stopped "before combined selection"
echo "[round3] === combined selection (all rounds) ==="
ARM=selection $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py
echo "[round3] DONE through Part K (r3)"
