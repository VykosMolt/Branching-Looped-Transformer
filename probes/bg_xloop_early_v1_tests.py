"""Cross-loop early-layer v1 — targeted verification tests.

Run: venv/bin/python -u utilities/tests/manual/bg_xloop_early_v1_tests.py
Covers: layer/loop indexing + scoring equivalence vs the locked production scoring layer,
feature shapes, candidate/task alignment, zero-crossing, frozen-head immutability,
source self-reproduction, deterministic refit, bootstrap clustering, no-heldout-tuning.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X
import bg_xloop_early_v1_train_eval as TE

PASS = []
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append((name, detail))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def test_scoring_equivalence_vs_production() -> None:
    """My single-channel scorer must equal cc.policy_candidate_scores on v2-format groups."""
    import bg_core_tap_audit_v1_common as cc
    cached = torch.load(X.RUN_ROOT / "cached_ref_heldout.pt", map_location="cpu", weights_only=False)
    g = cached["groups"][0]
    for c in g["candidates"]:
        c["features"] = c["features"].to(torch.float32)
    torch.manual_seed(7)
    w = torch.randn(1, X.HIDDEN)
    for layer, loop in ((24, 4), (36, 2), (47, 1)):
        prod = cc.policy_candidate_scores(g, [(w, "AntisymLinearNoNorm", f"{layer}_L{loop}")])
        # my scorer expects (5,4,2048); build an adapter group with the cached (3,4,2048)
        import bg_xloop_early_v1_controls as C
        mine_raw = C.policy_scores(g, [(w, "AntisymLinearNoNorm", f"{layer}_L{loop}")], C.v2_cell_vector)
        ok = all(abs(a - b) < 1e-4 for a, b in zip(prod, mine_raw))
        check(f"scoring_equivalence_{layer}_L{loop}", ok,
              f"max|d|={max(abs(a - b) for a, b in zip(prod, mine_raw)):.2e}")


def test_feature_shapes_and_alignment() -> None:
    subset = {g["group_uid"]: g for g in X.read_jsonl(X.RUN_ROOT / "subset_groups.jsonl")}
    fresh = X.load_fresh_groups()
    check("all_subset_groups_extracted", len(fresh) == len(subset),
          f"fresh={len(fresh)} subset={len(subset)}")
    shape_ok = uid_ok = order_ok = reward_ok = True
    for g in fresh:
        sg = subset.get(g["group_uid"])
        if sg is None:
            uid_ok = False
            continue
        fu = [c["candidate_uid"] for c in g["candidates"]]
        su = [c["candidate_uid"] for c in sg["candidates"]]
        order_ok &= fu == su
        reward_ok &= all(abs(float(a["reward"]) - float(b["reward"])) < 1e-9
                         for a, b in zip(g["candidates"], sg["candidates"]))
        for c in g["candidates"]:
            shape_ok &= tuple(c["features"].shape) == (5, 4, X.HIDDEN)
            shape_ok &= bool(torch.isfinite(c["features"]).all())
    check("feature_shapes_finite", shape_ok)
    check("group_uid_alignment", uid_ok)
    check("candidate_order_alignment", order_ok)
    check("reward_alignment", reward_ok)


def test_zero_crossing() -> None:
    si = X.read_json(X.RUN_ROOT / "split_integrity.json")
    check("zero_crossing_recorded", si["zero_crossing_count"] == 0)
    tasks = defaultdict(set)
    for g in X.read_jsonl(X.RUN_ROOT / "subset_groups.jsonl"):
        tasks[g["split"]].add(g["task_uid"])
    inter = (tasks["train"] & tasks["val"]) | (tasks["train"] & tasks["heldout"]) | (tasks["val"] & tasks["heldout"])
    check("zero_crossing_recheck", not inter, f"crossing={len(inter)}")


def test_frozen_head_and_self_repro() -> None:
    res = X.read_json(X.RUN_ROOT / "train_eval_results.json")
    ok_hash = ok_labels = True
    for key, tr in res["transfers"].items():
        layer, arrow = key.split(":")
        src_cell = f"{layer}_{arrow.split('->')[0]}"
        ok_hash &= tr["weight_sha"] == res["cells"][src_cell]["weight_sha"]
    check("frozen_head_hash_matches_source_cell", ok_hash)
    # self-repro is hard-asserted inside train_eval; verify the taps file hashes match too
    taps = torch.load(X.RUN_ROOT / "xloop_taps.pt", map_location="cpu", weights_only=False)
    for cn, rec in taps.items():
        ok_labels &= TE.weight_hash(rec["weight"]) == res["cells"][cn]["weight_sha"]
    check("saved_taps_hash_match", ok_labels)


def test_deterministic_refit() -> None:
    fresh = X.load_fresh_groups(splits=("train",))
    by = defaultdict(list)
    for g in fresh:
        by[g["domain"]].append(g)
    small = {d: gs[:40] for d, gs in by.items()}
    w1 = TE.train_pairwise(small, 16, 2, "all_core_balanced", 1e-3, 1)
    w2 = TE.train_pairwise(small, 16, 2, "all_core_balanced", 1e-3, 1)
    check("deterministic_refit", bool(torch.equal(w1, w2)))


def test_no_heldout_tuning() -> None:
    """Grid selection recorded in results must equal argmax over recorded val grid."""
    res = X.read_json(X.RUN_ROOT / "train_eval_results.json")
    ok = True
    for cn, cell in res["cells"].items():
        best = max(cell["grid"], key=lambda r: r["val_macro"])
        ok &= abs(best["val_macro"] - cell["val_macro"]) < 1e-9
    check("grid_selection_is_val_argmax", ok)
    src = Path(TE.__file__).read_text()
    import re
    sig = re.search(r"def fit_cell_grid\(([^)]*)\)", src)
    check("fit_grid_signature_excludes_heldout",
          sig is not None and "held" not in sig.group(1),
          f"params=({sig.group(1) if sig else 'NOT FOUND'})")


def test_bootstrap_clustering() -> None:
    """Toy check: clustered bootstrap keeps whole tasks together and is deterministic."""
    import bg_xloop_early_v1_stats as S
    series = {"a": {"dom": {"t1": [1.0, 1.0], "t2": [0.0], "t3": [1.0]}},
              "b": {"dom": {"t1": [0.0, 0.0], "t2": [1.0], "t3": [0.0]}}}
    p1, s1 = S.bootstrap(series, 500, 123)
    p2, s2 = S.bootstrap(series, 500, 123)
    check("bootstrap_deterministic", bool(np.allclose(s1["a"], s2["a"])))
    check("bootstrap_point_correct", abs(p1["a"] - 3 / 4) < 1e-9 and abs(p1["b"] - 1 / 4) < 1e-9)
    vals = set(np.round(np.unique(s1["a"]), 6))
    legal = {0.0, 1 / 4, 1 / 3, 1 / 2, 0.6, 2 / 3, 0.75, 0.8, 1.0, 0.2, 0.4}
    check("bootstrap_cluster_integrity", all(any(abs(v - l) < 1e-6 for l in legal) for v in vals),
          f"observed={sorted(vals)[:8]}")


def test_indexing_convention() -> None:
    """Fresh 24/36/47 must match cached production features (validates hook indexing);
    layers 8/16 must NOT match any cached layer (they are genuinely new loci)."""
    ctl = X.read_json(X.RUN_ROOT / "extraction_controls.json")
    ok = all(v["cos_mean"] > 0.995 for v in ctl["feature_match"].values())
    check("fresh_matches_cached_at_24_36_47", ok,
          f"min cos_mean={min(v['cos_mean'] for v in ctl['feature_match'].values())}")
    cached = torch.load(X.RUN_ROOT / "cached_ref_heldout.pt", map_location="cpu", weights_only=False)
    cg = cached["groups"][0]
    fresh = [g for g in X.load_fresh_groups(splits=("heldout",)) if g["group_uid"] == cg["group_uid"]]
    fc = fresh[0]["candidates"][0]
    cc_ = next(c for c in cg["candidates"] if c["candidate_uid"] == fc["candidate_uid"])
    distinct = True
    for layer in (8, 16):
        for pos in range(3):
            a = X.cell_vector(fc, layer, 4)
            b = cc_["features"][pos, 3].to(torch.float32)
            cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            distinct &= cos < 0.999
    check("early_layers_are_new_loci", distinct)


def main() -> int:
    print("xloop early-layer v1 targeted tests", flush=True)
    test_scoring_equivalence_vs_production()
    test_feature_shapes_and_alignment()
    test_zero_crossing()
    test_frozen_head_and_self_repro()
    test_deterministic_refit()
    test_no_heldout_tuning()
    test_bootstrap_clustering()
    test_indexing_convention()
    payload = {"passed": len(PASS), "failed": len(FAIL),
               "failures": [{"name": n, "detail": d} for n, d in FAIL],
               "tests": [n for n, _ in PASS + FAIL]}
    X.write_json(X.RUN_ROOT / "code_tests" / "test_results.json", payload)
    print(f"tests: {len(PASS)} passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
