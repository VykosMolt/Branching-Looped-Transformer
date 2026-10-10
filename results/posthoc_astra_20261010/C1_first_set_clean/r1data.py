"""Post hoc (2026-10-10, Astra audit): read-only loader for the Run 1 shards on the SSD.

Same records as the sealed analyze_run1.load() (same shard order, same equivalence-check assert), with the
large feature tensors the checks here do not use dropped: prefix_features keeps only the N=48 entry (index 3),
as {3: tensor}, so analyze_run1.feat_at(48) works unchanged. Shards are loaded one at a time; the reduced
records are cached in cache/ (deleted at the end of the audit work)."""
import glob
import os
import sys

import torch

ROOT = "/home/moloch/ouro_project"
R1 = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927"
R4 = ROOT + "/opi/preanswer/horizon_cohort2_20261003"
R3 = ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "r1_records_f48.pt")
DROP = ("prompt_only_features", "preanswer_features", "gen_ids_raw")


def setup_paths():
    """Laptop library paths first, then the SSD Run 1 dir (for analyze_run1 / answer_detector), as in
    opi_posthoc_20261003/strict_labels_run1_final_rule.py. Run with python -B so nothing is written to the SSD."""
    sys.path[:0] = [ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual"]
    sys.path.insert(0, R1)


def load():
    if os.path.exists(CACHE):
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    recs = []
    for f in sorted(glob.glob(R1 + "/run1_shard_off*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        for r in d["records"]:
            for k in DROP:
                r.pop(k, None)
            r["prefix_features"] = {3: r["prefix_features"][3].clone()}
            recs.append(r)
        del d
        print("loaded", f, len(recs), flush=True)
    torch.save(recs, CACHE)
    return recs
