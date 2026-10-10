import sys, torch
from pathlib import Path
sys.path.insert(0, "/home/moloch/ouro_project/shared/utilities/tests/manual")
import bg_xloop_early_v1_common as X, bg_xloop_early_v1_train_eval as TE
SRC = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260729/historical_experiments/paper_v3_audit_evidence/artifacts/reports/cross_loop_early_layer_taps_20260720")
X.FEATURE_ROOT = SRC/"features"
taps = torch.load(SRC/"xloop_taps.pt", map_location="cpu", weights_only=False)
train={d: [] for d in X.CORE_DOMAINS}
for g in X.load_fresh_groups(splits=("train",)):
    train[g["domain"]].append(g)
for nt in (1, 2, 3, 4, 6, 8):
    torch.set_num_threads(nt)
    out = []
    for cn, l, u, ts, lr in (("8_L3", 8, 3, "all_core", 3e-4), ("8_L1", 8, 1, "code_reasoning", 3e-4)):
        w = TE.train_pairwise(train, l, u, ts, lr, 0)
        out.append(f"{cn}:{'EXACT' if torch.equal(w, taps[cn]['weight']) else 'diff %.1e' % float((w-taps[cn]['weight']).abs().max())}")
    print(nt, out, flush=True)
