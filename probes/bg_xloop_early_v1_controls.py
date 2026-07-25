"""Cross-loop early-layer v1 — extraction-path controls.

Control A (feature match): fresh (5,4,2048) features at layers 24/36/47 vs the cached v2
production shard features (3,4,2048) on the identical heldout candidates.
Control B (reference reproduction): the saved locked CoreContent v2 policy
(`corecontent_v2_policy.pt`, AntisymLinearNoNorm channels 24_L4 + 36_L4) evaluated on the
same heldout subset with cached vs fresh features. Matched-subset comparison; the stored
full-population heldout value (0.6691 macro top-1) is context only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X


def v2_cell_vector(cand, layer: int, loop: int) -> torch.Tensor:
    f = cand["features"]
    assert tuple(f.shape) == (3, 4, X.HIDDEN), f"bad cached shape {tuple(f.shape)}"
    return f[X.V2_LAYER_POS[layer], loop - 1].to(torch.float32)


def policy_scores(group, channels, vec_fn) -> list[float]:
    """Multi-channel locked scoring: per-channel mean pairwise margin, z-norm, channel mean."""
    cands = group["candidates"]
    n = len(cands)
    per_channel = []
    for (w, arch, cfg) in channels:
        assert arch == "AntisymLinearNoNorm", arch
        layer, loop = cfg.split("_L")
        vecs = torch.stack([vec_fn(c, int(layer), int(loop)) for c in cands], 0)
        raw = vecs @ w.detach().to(torch.float32).flatten()
        tot = raw.sum()
        scores = [float(raw[i] - (tot - raw[i]) / (n - 1)) for i in range(n)]
        per_channel.append(X.normalize_scores(scores))
    return [X.finite_mean(ch[i] for ch in per_channel) for i in range(n)]


def eval_policy(groups_by_dom, channels, vec_fn):
    per_dom = {}
    for d, gs in groups_by_dom.items():
        ms = [X.group_selection_metrics(g, policy_scores(g, channels, vec_fn)) for g in gs]
        ms = [m for m in ms if m]
        per_dom[d] = {"top1": X.finite_mean(m["top1_oracle"] for m in ms),
                      "pairwise": X.finite_mean(m["pairwise_accuracy"] for m in ms), "groups": len(ms)}
    macro = X.finite_mean(per_dom[d]["top1"] for d in per_dom)
    return macro, per_dom


def main() -> int:
    t0 = time.time()
    cached = torch.load(X.RUN_ROOT / "cached_ref_heldout.pt", map_location="cpu", weights_only=False)
    cached_by_gid = {g["group_uid"]: g for g in cached["groups"]}
    fresh = [g for g in X.load_fresh_groups(splits=("heldout",))]
    assert fresh, "no fresh heldout features found"

    # ---------------- Control A: feature match on identical candidates
    stats = {f"{l}_L{u}": {"n": 0, "cos_sum": 0.0, "cos_min": 1.0, "max_abs": 0.0}
             for l in (24, 36, 47) for u in X.LOOPS}
    matched = missing = 0
    for g in fresh:
        cg = cached_by_gid.get(g["group_uid"])
        if cg is None:
            missing += 1
            continue
        cache_c = {c["candidate_uid"]: c for c in cg["candidates"]}
        for c in g["candidates"]:
            cc_ = cache_c.get(c["candidate_uid"])
            if cc_ is None:
                missing += 1
                continue
            matched += 1
            for layer in (24, 36, 47):
                for loop in X.LOOPS:
                    a = X.cell_vector(c, layer, loop)
                    b = v2_cell_vector(cc_, layer, loop)
                    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                    key = f"{layer}_L{loop}"
                    s = stats[key]
                    s["n"] += 1
                    s["cos_sum"] += cos
                    s["cos_min"] = min(s["cos_min"], cos)
                    s["max_abs"] = max(s["max_abs"], float((a - b).abs().max()))
    feature_match = {k: {"n": s["n"], "cos_mean": round(s["cos_sum"] / max(s["n"], 1), 6),
                         "cos_min": round(s["cos_min"], 6), "max_abs_diff": round(s["max_abs"], 4)}
                     for k, s in stats.items()}

    # ---------------- Control B: locked v2 policy, cached vs fresh features
    pol = torch.load(X.V2_POLICY_PT, map_location="cpu", weights_only=False)
    channels = [(w, arch, cfg) for (w, arch, cfg) in pol["channels"]["channels"]]
    held_by_fresh = {d: [g for g in fresh if g["domain"] == d] for d in X.CORE_DOMAINS}
    held_by_cached = {d: [cached_by_gid[g["group_uid"]] for g in held_by_fresh[d]
                          if g["group_uid"] in cached_by_gid] for d in X.CORE_DOMAINS}
    macro_c, per_dom_c = eval_policy(held_by_cached, channels, v2_cell_vector)
    macro_f, per_dom_f = eval_policy(held_by_fresh, channels, X.cell_vector)

    min_cos = min(v["cos_mean"] for v in feature_match.values() if v["n"])
    repro_delta = abs(macro_f - macro_c)
    verdict = ("EXTRACTION_PATH_VALIDATED" if min_cos > 0.995 and repro_delta < 0.01
               else "EXTRACTION_PATH_DEGRADED" if min_cos > 0.98 and repro_delta < 0.03
               else "EXTRACTION_PATH_MISMATCH")

    payload = {
        "XLOOP_EXTRACTION_CONTROL_VERDICT": verdict,
        "candidates_matched": matched, "candidates_missing": missing,
        "feature_match": feature_match,
        "policy_reproduction": {
            "policy": pol.get("selected"),
            "cached_features_macro_top1": round(macro_c, 4),
            "fresh_features_macro_top1": round(macro_f, 4),
            "abs_delta": round(repro_delta, 4),
            "cached_by_domain": {d: round(per_dom_c[d]["top1"], 4) for d in per_dom_c},
            "fresh_by_domain": {d: round(per_dom_f[d]["top1"], 4) for d in per_dom_f},
            "stored_full_population_reference": 0.6691,
            "note": "matched-subset comparison; full-population number is context only",
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    X.write_json(X.RUN_ROOT / "extraction_controls.json", payload)
    print(f"XLOOP_EXTRACTION_CONTROL_VERDICT = {verdict}")
    print(f"  min cos_mean={min_cos:.6f} policy cached={macro_c:.4f} fresh={macro_f:.4f}")
    return 0 if verdict != "EXTRACTION_PATH_MISMATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
