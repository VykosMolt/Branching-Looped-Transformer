#!/usr/bin/env bash
# Writes SEAL.sha256 for Run 7 (after the user's review; not run by the agent). Refuses if a seal or cohort shards exist,
# or if the CPU dry run fails. The file list must equal analyze_run7.SEALED (test_run7.py checks this).
set -euo pipefail
cd "$(dirname "$0")"
FILES="PLAN.md gen_run7.py analyze_run7.py within_oof.py test_run7.py ../gsm8k_train_20261003/gen_run6.py ../gsm8k_train_20261003/analyze_run6.py ../gsm8k_fixed_prefix_20260927/gen_run3.py ../gsm8k_fixed_prefix_20260927/analyze_run3.py ../gsm8k_fixed_prefix_20260927/answer_detector_gsm8k.py"
[ -e SEAL.sha256 ] && { echo "SEAL.sha256 exists; not overwriting"; exit 1; }
compgen -G "run7_shard_*.pt" > /dev/null && { echo "cohort shards exist; refusing to seal"; exit 1; }
/home/moloch/ouro_project/shared/venv/bin/python test_run7.py
sha256sum $FILES > SEAL.sha256
sha256sum --quiet -c SEAL.sha256 && echo "sealed: $(wc -l < SEAL.sha256) files" && cat SEAL.sha256
