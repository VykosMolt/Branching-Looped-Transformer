#!/usr/bin/env bash
# Round 4: train/eval FORMAT ALIGNMENT. C-style verifier-DPO arm on train_v5 (eval-side
# B1.build_prompt rendering: GEN_SYSTEM posture + chat template + MBPP fn-note; single-solution
# completions, no "Branch i:" sets; gen-pool verified reasoned code; <<>>-stripped math
# rationales + reasoning-error DPO negatives). Build is CPU-only; training is GPU-gated.
# Same STOP-file semantics as rounds 2/3.
set -u
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=shared/venv/bin/python
M=shared/utilities/tests/manual
CTRL=opi/taps/models/branch_training_offline_verifier_generator_v2/_control
ARM_R4=C_offline_verifier_dpo_r4

halt_if_stopped() {
  if [ -e "$CTRL/STOP" ]; then
    echo "[round4] STOP file present — pausing ($1)"
    exit 0
  fi
}

halt_if_stopped "before build views v5"
echo "[round4] === build eval-aligned views v5 (CPU) ==="
$PY $M/build_eval_aligned_views_v5.py || { echo "[round4] build FAILED"; exit 1; }

halt_if_stopped "before train $ARM_R4"
echo "[round4] === train $ARM_R4 ==="
ARM=$ARM_R4 $PY $M/train_offline_branch_generator_v2.py || { echo "[round4] train FAILED"; exit 1; }
if [ -e "$CTRL/STOP" ] || [ -e "$CTRL/STOP_train_$ARM_R4" ]; then
  echo "[round4] paused during/after train"
  exit 0
fi

halt_if_stopped "before canary $ARM_R4"
echo "[round4] === canary $ARM_R4 ==="
ARM=$ARM_R4 $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py || { echo "[round4] canary FAILED"; exit 1; }

halt_if_stopped "before combined selection"
echo "[round4] === combined selection (all rounds) ==="
ARM=selection $PY $M/evaluate_offline_branch_generator_canary_checkpoint_v2.py
echo "[round4] DONE through Part K (r4)"
