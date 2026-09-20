#!/usr/bin/env bash
# Run CoreContent v2 post-training stages M..U in order (L already done; no wait loop).
cd /home/moloch/ouro_project || exit 1
STAGES="train_bg_corecontent_v2_domain_gated \
        evaluate_bg_corecontent_v2_heldout \
        analyze_bg_corecontent_v2_domains_errors \
        analyze_bg_corecontent_v2_calibration_ablation \
        select_bg_corecontent_v2_policy \
        analyze_bg_corecontent_v2_phase2b_readiness \
        analyze_bg_corecontent_dataset_expansion_refit_v2"
for w in $STAGES; do
  echo "=== RUN $w ($(date +%H:%M:%S)) ==="
  stdbuf -oL -eL shared/venv/bin/python -u "shared/utilities/tests/manual/$w.py" 2>&1 | grep -vE "%\|" | tail -5
  echo "  rc=${PIPESTATUS[0]}"
done
echo "POST_CHAIN_DONE"
