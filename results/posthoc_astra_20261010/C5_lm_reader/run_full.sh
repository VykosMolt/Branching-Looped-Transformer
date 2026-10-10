#!/bin/bash
# C5 full run: DeBERTa then Qwen, sequentially (PROTOCOL.md incl. Amendment 1)
cd /home/moloch/ouro_project/opi/posthoc_astra_20261010/C5_lm_reader
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=/home/moloch/ouro_project/shared/venv/bin/python
for r in deberta qwen; do
  echo "=== $r start $(date -Is)" >> run_times.log
  $PY lm_reader.py --reader $r > runs_$r.log 2>&1
  rc=$?
  echo "=== $r end $(date -Is) exit $rc" >> run_times.log
  [ $rc -ne 0 ] && exit $rc
done
