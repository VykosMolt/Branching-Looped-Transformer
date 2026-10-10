#!/usr/bin/env bash
# Run 4: waits for the RPE chain and a free GPU; GPU smoke test (abort on failure); generation; label-free audit stage.
set -u
cd /home/moloch/ouro_project
PY=shared/venv/bin/python
R4=opi/preanswer/horizon_cohort2_20261003
log() { echo "[$(date -Is)] $*" >> $R4/run4.log; }
(cd $R4 && sha256sum --quiet -c SEAL.sha256) || { log "seal check failed; not starting"; exit 1; }
log "waiting for the RPE chain"
until grep -q "all done" rpe_v3_runs/driver.log 2>/dev/null; do sleep 60; done
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
(cd $R4 && sha256sum --quiet -c SEAL.sha256) || { log "seal check failed; not starting"; exit 1; }
log "smoke test"
$PY $R4/gen_run4.py --smoke >> $R4/run4.log 2>&1 || { log "smoke test crashed; stopping"; exit 1; }
$PY - <<'EOF' >> $R4/run4.log 2>&1 || { log "smoke test failed its checks; stopping"; exit 1; }
import glob, torch
d = torch.load(sorted(glob.glob("opi/preanswer/horizon_cohort2_20261003/smoke/run4_shard_*.pt"))[-1], map_location="cpu", weights_only=False)
assert not d["errors"], d["errors"]
assert len(d["records"]) == 8 and max(d["equivalence_checks_maxabs"]) <= 1e-3, d["equivalence_checks_maxabs"]
assert all(r["decode_roundtrip_ok"] for r in d["records"])
print("smoke ok:", sum(r["labelled"] for r in d["records"]), "of 8 labelled; equivalence max", max(d["equivalence_checks_maxabs"]))
EOF
log "start generation"
$PY $R4/gen_run4.py >> $R4/run4.log 2>&1; log "gen exit $?"
$PY $R4/analyze_run4.py --stage audit >> $R4/run4.log 2>&1; log "audit exit $?"
