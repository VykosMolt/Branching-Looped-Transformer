import pickle, sys, numpy as np, torch
C = "/tmp/claude-1000/-home-moloch-Documents-Research-ouro-Operational-Proto-Introspection/07561c8b-32be-4f56-b8e6-e1f42d49fe0a/scratchpad/audit_opi/r4cache"
ROOT = "/home/moloch/ouro_project"
for p in (ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual", ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927",
          ROOT + "/opi/preanswer/horizon_cohort2_20261003"):
    sys.path.insert(0, p)
meta = pickle.load(open(f"{C}/meta.pkl", "rb"))
F = {k: np.load(f"{C}/{k}.npy", mmap_mode="r") for k in ("pend", "f0", "f48", "f16", "f32", "f64", "f96", "f128")}
for i, r in enumerate(meta):
    r["i"] = i; r["task_id"] = r["task_uid"]
