#!/usr/bin/env bash
# Run 3 (GSM8K fixed prefix): generation, then the label-free audit stage. Labels stay closed.
set -u
cd /home/moloch/ouro_project
PY=shared/venv/bin/python
R3=opi/preanswer/gsm8k_fixed_prefix_20260927
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU in use; not starting" >&2; exit 1
fi
(cd $R3 && sha256sum --quiet -c SEAL.sha256) || { echo "seal check failed; not starting" >&2; exit 1; }
echo "start $(date -Is)" >> $R3/run3.log
$PY $R3/gen_run3.py >> $R3/run3.log 2>&1; echo "gen exit $? $(date -Is)" >> $R3/run3.log
$PY $R3/analyze_run3.py --stage audit >> $R3/run3.log 2>&1; echo "audit exit $? $(date -Is)" >> $R3/run3.log
