"""Post hoc (2026-10-10, Astra audit): read-only loader for the Run 6 shards on the SSD, shared by C2 (chronological
rerun) and C4 (out-of-fold within-question). Shards are loaded one at a time (same order and asserts as the sealed
analyze_run6.load()). Every record keeps its metadata; prefix_features keeps only the N*=96 entry (index 5, as {5: tensor})
and only for parseable answers that are answer-free at 96 under the executed own answer OR the chronological one
(the union, so both cohorts can be built). Cached in cache/ (deleted at the end)."""
import glob
import os
import sys

import torch

ROOT = "/home/moloch/ouro_project"
R6 = ROOT + "/opi/preanswer/gsm8k_train_20261003"
R3 = ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"
SSD = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/gsm8k_train_20261003"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "r6_records_f96.pt")
GI = 5  # PREFIX_GRID.index(96)


def setup_paths():
    for p in (ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual", R3, R6,
              os.path.join(os.path.dirname(HERE), "C2_parser_priority")):
        sys.path.insert(0, p)


def load():
    if os.path.exists(CACHE):
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    setup_paths()
    from answer_detector_gsm8k import is_answer_bearing, value
    from chrono_parsers import first_commitment_chrono
    recs = []
    for f in sorted(glob.glob(SSD + "/run6_shard_*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert not d["smoke"], f
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        assert not d["errors"], f
        for r in d["records"]:
            pf = r.pop("prefix_features")
            r.pop("prompt_end_state", None)
            ch = first_commitment_chrono(r["generated_text"])
            keep = value(r["own_answer"]) is not None and (not is_answer_bearing(r["prefix_texts"][GI], r["own_answer"])
                                                           or not is_answer_bearing(r["prefix_texts"][GI], ch))
            r["prefix_features"] = {GI: pf[GI].clone()} if keep else {}
            recs.append(r)
        del d
        print("loaded", os.path.basename(f), len(recs), flush=True)
    torch.save(recs, CACHE)
    return recs


if __name__ == "__main__":
    rs = load()
    print(len(rs), "records;", sum(bool(r["prefix_features"]) for r in rs), "with N=96 features")
