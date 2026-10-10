#!/usr/bin/env bash
# Run 1 (fixed-prefix cohort) then Run 2 (tournament rerun), sequentially on the one GPU.
set -u
cd /home/moloch/ouro_project
PY=shared/venv/bin/python
R1=opi/preanswer/fixed_prefix_rerun_20260927
R2=opi/branching/tournament_rerun_depth4_20260927
echo "start $(date -Is)" >> $R1/run1.log
$PY $R1/gen_run1.py >> $R1/run1.log 2>&1; echo "run1 exit $? $(date -Is)" >> $R1/run1.log
$PY $R2/run2_tournament.py --stage generate >> $R2/run2.log 2>&1; echo "run2 generate exit $? $(date -Is)" >> $R2/run2.log
$PY $R2/run2_tournament.py --stage analyze >> $R2/run2.log 2>&1; echo "run2 analyze exit $? $(date -Is)" >> $R2/run2.log
