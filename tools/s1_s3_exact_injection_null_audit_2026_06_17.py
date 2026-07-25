#!/usr/bin/env python3
"""Rank and null-distribution audit for the S1/S3 exact injection bundle.

This script is analysis-only: it loads the saved delta bundle, reconstructs the
same injection span used by the prior orthogonality report, and writes the
missing rank/null controls. It does not regenerate generations, train models, or
modify checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/reports/proto_introspection/s1_s3_exact_injection_orthogonality_2026-06-17"
BUNDLE_PATH = OUT_DIR / "s1_s3_exact_injection_delta_bundle_2026-06-17.pt"
REPORT_MD = OUT_DIR / "s1_s3_exact_injection_orthogonality_null_audit_2026-06-17.md"
REPORT_JSON = OUT_DIR / "s1_s3_exact_injection_orthogonality_null_audit_2026-06-17.json"

SEED = 20260617
RANDOM_NULL_DRAWS = 1_000_000
SHUFFLED_LABEL_DRAWS = 10_000
LAYER_TO_SLOT = {24: 0, 36: 1, 47: 2}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return rel(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=False) + "\n")


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def md_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def load_bundle() -> Dict[str, Any]:
    return torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)


def tensor_summary(x: Any) -> Dict[str, Any]:
    if torch.is_tensor(x):
        xf = x.float()
        return {
            "type": "tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "min": float(xf.min()) if x.numel() else None,
            "max": float(xf.max()) if x.numel() else None,
            "norm": float(xf.norm()) if x.numel() else None,
        }
    if isinstance(x, dict):
        return {"type": "dict", "len": len(x), "keys": sorted(map(str, x.keys()))}
    if isinstance(x, list):
        first = x[0] if x else None
        return {
            "type": "list",
            "len": len(x),
            "first_keys": sorted(map(str, first.keys())) if isinstance(first, dict) else None,
        }
    return {"type": type(x).__name__, "repr": repr(x)[:240]}


def inspect_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    rows = bundle.get("branch_rows", [])
    sample_rows = bundle.get("sample_rows", [])
    return {
        "top_level": {k: tensor_summary(v) for k, v in sorted(bundle.items())},
        "identified_injection_delta_tensors": {
            "h_clean_locus": "clean last-token boundary states; shape [344,2048]",
            "h_injected_locus": "injected last-token boundary states; shape [344,2048]",
            "delta_locus": "h_injected_locus - h_clean_locus; embedded into the [3,4,2048] audit feature basis",
        },
        "identified_outcome_direction": {
            "saved_explicit_outcome_direction": False,
            "recomputed_from": "branch_features mean(correct) - mean(incorrect), using branch_rows.correct",
        },
        "identified_candidate_features_labels": {
            "branch_features": list(bundle["branch_features"].shape),
            "branch_label_key": "branch_rows[].correct",
            "sample_features": list(bundle["sample_features"].shape),
            "sample_label_key": "sample_rows[].correct",
        },
        "branch_label_counts": dict(Counter(bool(r.get("correct")) for r in rows)),
        "sample_label_counts": dict(Counter(bool(r.get("correct")) for r in sample_rows)),
        "branch_domain_counts": dict(Counter(r.get("domain") for r in rows)),
    }


def embed_delta_features(bundle: Dict[str, Any], *, center: bool = False) -> torch.Tensor:
    rows = bundle["branch_rows"]
    deltas = bundle["delta_locus"].float()
    out = torch.zeros(len(rows), 3, 4, 2048, dtype=torch.float32)
    for i, row in enumerate(rows):
        out[i, LAYER_TO_SLOT[int(row["layer"])], int(row["loop"]), :] = deltas[i]
    mat = out.reshape(len(rows), -1)
    if center:
        mat = mat - mat.mean(0, keepdim=True)
    return mat


def outcome_direction(features: torch.Tensor, labels: np.ndarray) -> Optional[torch.Tensor]:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    return features[labels == 1].mean(0) - features[labels == 0].mean(0)


def normalized_direction(v: torch.Tensor) -> Optional[torch.Tensor]:
    n = float(v.norm())
    if n <= 0:
        return None
    return v / n


def projection_fraction(v: Optional[torch.Tensor], vh: torch.Tensor, rank: int) -> Optional[float]:
    if v is None or rank <= 0:
        return None
    denom = float(torch.dot(v, v))
    if denom <= 0:
        return None
    coeff = torch.mv(vh[:rank], v)
    return float(torch.dot(coeff, coeff) / denom)


def entropy_effective_rank(s: torch.Tensor) -> float:
    energy = s.double().pow(2)
    p = energy / energy.sum().clamp(min=1e-300)
    nz = p[p > 0]
    return float(torch.exp(-(nz * torch.log(nz)).sum()))


def participation_ratio(s: torch.Tensor) -> float:
    energy = s.double().pow(2)
    return float(energy.sum().square() / energy.square().sum().clamp(min=1e-300))


def pca_counts(s: torch.Tensor) -> Dict[str, int]:
    energy = s.double().pow(2)
    cum = torch.cumsum(energy / energy.sum().clamp(min=1e-300), dim=0)
    out: Dict[str, int] = {}
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        out[f"pcs_{int(threshold * 100)}pct"] = int(
            torch.searchsorted(cum, torch.tensor(threshold, dtype=torch.float64)).item() + 1
        )
    return out


def relation_from_ratio(ratio: Optional[float]) -> str:
    if ratio is None:
        return "inconclusive"
    if ratio < 0.8:
        return "below chance"
    if ratio <= 1.2:
        return "near chance"
    return "above chance"


def svd_geometry(mat: torch.Tensor, outcome: Optional[torch.Tensor]) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    default_rank = int(torch.linalg.matrix_rank(mat).item())
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    _ = u
    max_s = float(s.max()) if s.numel() else 0.0
    ranks = {
        "raw_positive_singular_values": int((s > 0).sum()),
        "torch_default_matrix_rank": default_rank,
        "legacy_main_audit_rel_1e-6": int((s > max_s * 1e-6).sum()) if max_s else 0,
        "absolute_gt_1e-3": int((s > 1e-3).sum()),
        "absolute_gt_1e-4": int((s > 1e-4).sum()),
        "absolute_gt_1e-5": int((s > 1e-5).sum()),
        "relative_gt_1e-3_of_max": int((s > max_s * 1e-3).sum()) if max_s else 0,
    }
    projections = {
        name: projection_fraction(outcome, vh, k)
        for name, k in ranks.items()
        if k > 0
    }
    geom = {
        "matrix_shape": list(mat.shape),
        "ambient_dimension_D": int(mat.shape[1]),
        "singular_value_max": max_s,
        "singular_value_min": float(s.min()) if s.numel() else None,
        "rank_thresholds": ranks,
        "effective_rank_entropy": entropy_effective_rank(s) if s.numel() else None,
        "participation_ratio": participation_ratio(s) if s.numel() else None,
        "pca_energy_counts": pca_counts(s) if s.numel() else {},
        "top_singular_values": [float(x) for x in s[:20].tolist()],
        "tail_singular_values": [float(x) for x in s[-10:].tolist()],
        "projection_by_rank_name": projections,
    }
    return geom, s, vh


def chance_baselines(
    D: int,
    ranks: Dict[str, int],
    projections: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    rows = []
    for name, k in ranks.items():
        expected = float(k / D) if D else None
        observed = projections.get(name)
        ratio = None if expected in (None, 0.0) or observed is None else float(observed / expected)
        rows.append(
            {
                "rank_name": name,
                "k": int(k),
                "expected_random_projection_k_over_D": expected,
                "observed_projection_at_rank": observed,
                "observed_over_expected": ratio,
                "chance_relation": relation_from_ratio(ratio),
            }
        )
    return rows


def random_unit_projection_null(D: int, k: int, observed: float, *, draws: int) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    if k <= 0:
        vals = np.zeros(draws, dtype=np.float64)
    elif k >= D:
        vals = np.ones(draws, dtype=np.float64)
    else:
        in_span = rng.chisquare(df=k, size=draws)
        out_span = rng.chisquare(df=D - k, size=draws)
        vals = in_span / (in_span + out_span)
    quantiles = np.quantile(vals, [0.01, 0.05, 0.5, 0.95, 0.99])
    p_left = float(np.mean(vals <= observed))
    p_right = float(np.mean(vals >= observed))
    return {
        "draws": int(draws),
        "method": "chi-square equivalent to projecting random unit vectors onto a fixed k-dimensional orthonormal span",
        "rank_k": int(k),
        "ambient_dimension_D": int(D),
        "expected_k_over_D": float(k / D),
        "observed_projection": float(observed),
        "null_mean": float(vals.mean()),
        "null_std": float(vals.std(ddof=1)),
        "quantiles": {
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
        },
        "observed_percentile": float(100.0 * p_left),
        "p_left_fraction_null_le_observed": p_left,
        "p_right_fraction_null_ge_observed": p_right,
    }


def shuffled_label_null(
    features: torch.Tensor,
    labels: np.ndarray,
    vh: torch.Tensor,
    rank: int,
    observed: float,
    *,
    draws: int,
    groups: Optional[List[str]] = None,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"status": "not_run_single_class_labels"}

    rng = np.random.default_rng(SEED + seed_offset)
    basis = vh[:rank]
    projected_features = torch.mm(features, basis.T).double().numpy()
    projected_total = projected_features.sum(axis=0)
    gram = torch.mm(features, features.T).double().numpy()
    total_dot_rows = gram.sum(axis=1)
    total_norm2 = float(gram.sum())
    c = (1.0 / n_pos) + (1.0 / n_neg)

    group_index: Optional[Dict[str, Tuple[np.ndarray, int]]] = None
    if groups is not None:
        group_index = {}
        group_arr = np.asarray(groups)
        for group in sorted(set(groups)):
            idx = np.where(group_arr == group)[0]
            group_index[group] = (idx, int(labels[idx].sum()))

    vals: List[float] = []
    for _ in range(draws):
        if group_index is None:
            pos_idx = rng.choice(n, size=n_pos, replace=False)
        else:
            chunks = []
            valid = True
            for idx, count in group_index.values():
                if count > len(idx):
                    valid = False
                    break
                if count:
                    chunks.append(rng.choice(idx, size=count, replace=False))
            if not valid:
                continue
            pos_idx = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

        s_proj = projected_features[pos_idx].sum(axis=0)
        proj_coords = c * s_proj - projected_total / n_neg
        numerator = float(np.dot(proj_coords, proj_coords))

        sub_gram = gram[np.ix_(pos_idx, pos_idx)]
        s_norm2 = float(sub_gram.sum())
        s_dot_total = float(total_dot_rows[pos_idx].sum())
        denom = c * c * s_norm2 - 2.0 * c * s_dot_total / n_neg + total_norm2 / (n_neg * n_neg)
        vals.append(numerator / denom if denom > 0 else 0.0)

    arr = np.asarray(vals, dtype=np.float64)
    quantiles = np.quantile(arr, [0.01, 0.05, 0.5, 0.95, 0.99])
    p_left = float(np.mean(arr <= observed))
    p_right = float(np.mean(arr >= observed))
    return {
        "draws": int(len(arr)),
        "rank_k": int(rank),
        "label_shuffle": "global_count_preserving" if groups is None else "group_stratified_count_preserving",
        "null_mean": float(arr.mean()),
        "null_std": float(arr.std(ddof=1)),
        "quantiles": {
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
        },
        "observed_projection": float(observed),
        "observed_percentile": float(100.0 * p_left),
        "p_left_fraction_null_le_observed": p_left,
        "p_right_fraction_null_ge_observed": p_right,
    }


def grouped_direction_counts(labels: np.ndarray, groups: List[str]) -> Dict[str, int]:
    mixed = 0
    all_pos = 0
    all_neg = 0
    for group in sorted(set(groups)):
        y = labels[[i for i, g in enumerate(groups) if g == group]]
        if y.sum() == 0:
            all_neg += 1
        elif y.sum() == len(y):
            all_pos += 1
        else:
            mixed += 1
    return {"mixed": mixed, "all_positive": all_pos, "all_negative": all_neg, "total": len(set(groups))}


def domain_directions(features: torch.Tensor, labels: np.ndarray, domains: List[str]) -> List[Dict[str, Any]]:
    out = []
    domain_arr = np.asarray(domains)
    for domain in sorted(set(domains)):
        idx = np.where(domain_arr == domain)[0]
        y = labels[idx]
        rec = {
            "name": f"domain:{domain}",
            "n": int(len(idx)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
        }
        if y.sum() > 0 and y.sum() < len(y):
            rec["direction"] = features[idx][y == 1].mean(0) - features[idx][y == 0].mean(0)
        out.append(rec)
    return out


def principal_angle_extension(
    features: torch.Tensor,
    labels: np.ndarray,
    rows: List[Dict[str, Any]],
    vh: torch.Tensor,
    rank: int,
    D: int,
) -> Dict[str, Any]:
    task_locus_groups = [row["score_group_id"] for row in rows]
    candidate_groups = [row["candidate_group_id"] for row in rows]
    domains = [row["domain"] for row in rows]
    global_dir = outcome_direction(features, labels)
    direction_records: List[Dict[str, Any]] = []

    if global_dir is not None:
        direction_records.append({"name": "global_correct_minus_incorrect", "direction": global_dir})
    for rec in domain_directions(features, labels, domains):
        if "direction" in rec:
            direction_records.append({"name": rec["name"] + "_correct_minus_incorrect", "direction": rec["direction"]})

    q_cols = []
    individual = []
    expected = rank / D
    for rec in direction_records:
        v = rec["direction"]
        proj = projection_fraction(v, vh, rank)
        ratio = None if proj is None else proj / expected
        individual.append(
            {
                "name": rec["name"],
                "projection_fraction": proj,
                "cosine_to_injection_span": math.sqrt(max(0.0, proj or 0.0)),
                "expected_random_projection_k_over_D": expected,
                "observed_over_expected": ratio,
                "chance_relation": relation_from_ratio(ratio),
            }
        )
        nv = normalized_direction(v)
        if nv is not None:
            q_cols.append(nv)

    subspace = {"status": "not_enough_outcome_directions"}
    if q_cols:
        mat = torch.stack(q_cols, dim=1)
        q_out, r = torch.linalg.qr(mat, mode="reduced")
        diag = torch.abs(torch.diagonal(r))
        q_rank = int((diag > 1e-6).sum())
        q_out = q_out[:, :q_rank]
        cross = torch.mm(q_out.T, vh[:rank].T)
        sv = torch.linalg.svdvals(cross)
        angles = torch.rad2deg(torch.acos(torch.clamp(sv, -1.0, 1.0)))
        mean_projected = float(torch.sum(sv.square()) / max(1, q_rank))
        ratio = mean_projected / expected if expected else None
        subspace = {
            "status": "computed",
            "outcome_subspace_rank": q_rank,
            "construction": "global correct-minus-incorrect plus any mixed-label domain correct-minus-incorrect directions; no classifier training",
            "principal_cosines": [float(x) for x in sv.tolist()],
            "principal_angles_degrees": [float(x) for x in angles.tolist()],
            "mean_projected_energy_per_outcome_dimension": mean_projected,
            "expected_random_projection_k_over_D": expected,
            "observed_over_expected": ratio,
            "chance_relation": relation_from_ratio(ratio),
        }

    return {
        "group_direction_availability": {
            "candidate_group_id": grouped_direction_counts(labels, candidate_groups),
            "score_group_id": grouped_direction_counts(labels, task_locus_groups),
        },
        "domain_direction_availability": [
            {k: v for k, v in rec.items() if k != "direction"}
            for rec in domain_directions(features, labels, domains)
        ],
        "individual_outcome_directions": individual,
        "outcome_subspace": subspace,
    }


def make_verdicts(random_null: Dict[str, Any], observed: float, expected: float) -> Dict[str, str]:
    ratio = observed / expected if expected else float("nan")
    percentile = random_null["observed_percentile"]
    if ratio < 0.8 and percentile <= 5.0:
        rank_verdict = "OBSERVED_PROJECTION_BELOW_RANDOM_SUBSPACE_CHANCE"
        load = "LOAD_BEARING_MAIN_RESULT"
        boundary = "SUBSPACE_MISALIGNMENT_STRONGLY_SUPPORTED"
    elif ratio <= 1.2 and 5.0 < percentile < 95.0:
        rank_verdict = "OBSERVED_PROJECTION_NEAR_RANDOM_SUBSPACE_CHANCE"
        load = "SUPPORTING_CAVEATED_RESULT"
        boundary = "SUBSPACE_MISALIGNMENT_WEAKLY_SUPPORTED"
    else:
        rank_verdict = "OBSERVED_PROJECTION_ABOVE_RANDOM_SUBSPACE_CHANCE"
        load = "DO_NOT_USE_AS_ORTHOGONALITY_EVIDENCE"
        boundary = "SUBSPACE_MISALIGNMENT_NOT_SUPPORTED"
    return {
        "INJECTION_SPAN_RANK_NULL_VERDICT": rank_verdict,
        "ORTHOGONALITY_LOAD_BEARING_STATUS": load,
        "READOUT_CONTROL_BOUNDARY_UPDATE": boundary,
    }


def build_result() -> Dict[str, Any]:
    bundle = load_bundle()
    rows = bundle["branch_rows"]
    features = bundle["branch_features"].float().reshape(len(rows), -1)
    labels = np.asarray([int(row["correct"]) for row in rows], dtype=np.int64)
    outcome = outcome_direction(features, labels)

    uncentered_mat = embed_delta_features(bundle, center=False)
    centered_mat = embed_delta_features(bundle, center=True)
    main_geom, _main_s, main_vh = svd_geometry(uncentered_mat, outcome)
    centered_geom, _centered_s, _centered_vh = svd_geometry(centered_mat, outcome)

    D = int(features.shape[1])
    main_rank = int(main_geom["rank_thresholds"]["legacy_main_audit_rel_1e-6"])
    observed = main_geom["projection_by_rank_name"]["legacy_main_audit_rel_1e-6"]
    assert observed is not None
    chance = chance_baselines(D, main_geom["rank_thresholds"], main_geom["projection_by_rank_name"])
    random_null = random_unit_projection_null(D, main_rank, observed, draws=RANDOM_NULL_DRAWS)

    domains = [row["domain"] for row in rows]
    shuffled_global = shuffled_label_null(
        features, labels, main_vh, main_rank, observed, draws=SHUFFLED_LABEL_DRAWS, seed_offset=1
    )
    shuffled_domain = shuffled_label_null(
        features, labels, main_vh, main_rank, observed, draws=SHUFFLED_LABEL_DRAWS, groups=domains, seed_offset=2
    )
    principal = principal_angle_extension(features, labels, rows, main_vh, main_rank, D)

    expected = main_rank / D
    verdicts = make_verdicts(random_null, observed, expected)

    return {
        "audit": "s1_s3_exact_injection_orthogonality_null_audit",
        "audit_generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ran_generation": False,
        "trained_or_modified_checkpoints": False,
        "bundle_path": BUNDLE_PATH,
        "bundle_inspection": inspect_bundle(bundle),
        "geometry_basis": {
            "feature_basis": bundle["tensor_shapes"].get("feature_basis"),
            "injection_delta_basis": bundle["tensor_shapes"].get("locus_delta_basis"),
            "main_projection_centering": "uncentered, matching tools/s1_s3_exact_injection_orthogonality_2026_06_17.py",
            "same_as_prior_projection": True,
            "same_as_s1_prune_scoring_feature_basis": True,
            "same_as_taps_preanswer_readouts": (
                "Same saved [3,4,2048] branch rollout feature basis as the prior S1/S3 projection. "
                "This bundle does not contain separate historical pre-answer readout tensors, so claims tying "
                "this exact geometry to other readout artifacts remain caveated unless those artifacts used the same extractor."
            ),
        },
        "counts": {
            "branches": len(rows),
            "positive_labels": int(labels.sum()),
            "negative_labels": int((labels == 0).sum()),
            "domains": dict(Counter(domains)),
            "candidate_groups": len(set(row["candidate_group_id"] for row in rows)),
            "score_groups": len(set(row["score_group_id"] for row in rows)),
        },
        "main_uncentered_injection_span": main_geom,
        "secondary_centered_injection_span": centered_geom,
        "chance_baselines": chance,
        "empirical_random_direction_null": random_null,
        "shuffled_label_null": {
            "global_count_preserving": shuffled_global,
            "domain_stratified_count_preserving": shuffled_domain,
        },
        "principal_angle_outcome_subspace_extension": principal,
        "observed_projection_fraction": observed,
        "observed_residual_fraction": 1.0 - observed,
        "rank_null_interpretation": (
            "The absolute residual is high, but after accounting for the 344-dimensional injection span inside "
            "D=24576, the observed projection is above the k/D random-subspace baseline and in the right tail "
            "of the random-direction null. The prior orthogonality claim should therefore not be used as "
            "load-bearing evidence for subspace misalignment."
        ),
        "verdict_constants": verdicts,
    }


def build_markdown(result: Dict[str, Any]) -> str:
    geom = result["main_uncentered_injection_span"]
    ranks = geom["rank_thresholds"]
    random_null = result["empirical_random_direction_null"]
    shuffled = result["shuffled_label_null"]
    principal = result["principal_angle_outcome_subspace_extension"]
    verdicts = result["verdict_constants"]

    key_rows = [
        [k, v["type"], v.get("shape", v.get("len")), v.get("dtype", ""), v.get("keys", v.get("first_keys", ""))]
        for k, v in result["bundle_inspection"]["top_level"].items()
    ]
    rank_rows = [
        ["ambient dimension D", geom["ambient_dimension_D"]],
        ["raw positive-singular-value rank", ranks["raw_positive_singular_values"]],
        ["torch default matrix_rank", ranks["torch_default_matrix_rank"]],
        ["main audit rank, rel 1e-6", ranks["legacy_main_audit_rel_1e-6"]],
        ["rank abs > 1e-3", ranks["absolute_gt_1e-3"]],
        ["rank abs > 1e-4", ranks["absolute_gt_1e-4"]],
        ["rank abs > 1e-5", ranks["absolute_gt_1e-5"]],
        ["rank rel > 1e-3 max", ranks["relative_gt_1e-3_of_max"]],
        ["entropy effective rank", fmt(geom["effective_rank_entropy"])],
        ["participation ratio", fmt(geom["participation_ratio"])],
        ["PCs 50/80/90/95/99", geom["pca_energy_counts"]],
    ]
    chance_rows = [
        [
            row["rank_name"],
            row["k"],
            fmt(row["expected_random_projection_k_over_D"]),
            fmt(row["observed_projection_at_rank"]),
            fmt(row["observed_over_expected"]),
            row["chance_relation"],
        ]
        for row in result["chance_baselines"]
    ]
    null_rows = [
        ["draws", random_null["draws"]],
        ["null mean", fmt(random_null["null_mean"])],
        ["null std", fmt(random_null["null_std"])],
        ["q01/q05/q50/q95/q99", random_null["quantiles"]],
        ["observed projection", fmt(random_null["observed_projection"])],
        ["observed percentile", fmt(random_null["observed_percentile"])],
        ["p_left null <= observed", fmt(random_null["p_left_fraction_null_le_observed"])],
        ["p_right null >= observed", fmt(random_null["p_right_fraction_null_ge_observed"])],
    ]
    shuffle_rows = [
        [
            name,
            data.get("draws"),
            fmt(data.get("null_mean")),
            fmt(data.get("null_std")),
            data.get("quantiles"),
            fmt(data.get("observed_percentile")),
            fmt(data.get("p_left_fraction_null_le_observed")),
            fmt(data.get("p_right_fraction_null_ge_observed")),
        ]
        for name, data in shuffled.items()
    ]
    principal_rows = [
        [
            rec["name"],
            fmt(rec["projection_fraction"]),
            fmt(rec["expected_random_projection_k_over_D"]),
            fmt(rec["observed_over_expected"]),
            rec["chance_relation"],
        ]
        for rec in principal["individual_outcome_directions"]
    ]
    verdict_rows = [[k, v] for k, v in verdicts.items()]

    return "\n\n".join(
        [
            "# S1/S3 Exact Injection Orthogonality Rank/Null Audit",
            "## Scope",
            (
                f"Bundle path: `{rel(result['bundle_path'])}`. This audit loaded the saved tensor bundle only; "
                "it did not rerun generation, train anything, or modify checkpoints."
            ),
            "## Bundle Inspection",
            md_table(["key", "type", "shape/len", "dtype", "keys"], key_rows),
            (
                "Injection/carry deltas are `delta_locus = h_injected_locus - h_clean_locus`, embedded into "
                "the flattened `[3,4,2048]` feature basis at the saved row layer/loop slot. No explicit outcome "
                "direction tensor is saved; it is recomputed from `branch_features` and `branch_rows[].correct`."
            ),
            "## Main Injection Span Geometry",
            md_table(["metric", "value"], rank_rows),
            (
                f"The main rank/truncation matches the previous audit: uncentered SVD with rank "
                f"`{ranks['legacy_main_audit_rel_1e-6']}` from `s > max(s) * 1e-6`. The centered secondary "
                f"check gives rank `{result['secondary_centered_injection_span']['rank_thresholds']['legacy_main_audit_rel_1e-6']}` "
                "and nearly the same projection."
            ),
            "## Chance Baselines",
            md_table(["rank", "k", "k/D expected", "observed projection", "obs/expected", "relation"], chance_rows),
            "## Empirical Random-Direction Null",
            md_table(["metric", "value"], null_rows),
            "## Shuffled-Label Null",
            md_table(
                ["null", "draws", "mean", "std", "quantiles", "percentile", "p_left", "p_right"],
                shuffle_rows,
            ),
            "## Outcome-Subspace Extension",
            (
                f"Candidate-group mixed-label directions: {principal['group_direction_availability']['candidate_group_id']}. "
                f"Score-group mixed-label directions: {principal['group_direction_availability']['score_group_id']}. "
                "No classifier was trained."
            ),
            md_table(["direction", "projection", "k/D expected", "obs/expected", "relation"], principal_rows),
            f"Outcome subspace: `{principal['outcome_subspace']}`.",
            "## Geometry Boundary",
            result["geometry_basis"]["same_as_taps_preanswer_readouts"],
            "## Interpretation",
            result["rank_null_interpretation"],
            "## Verdict Constants",
            md_table(["constant", "value"], verdict_rows),
        ]
    ) + "\n"


def main() -> int:
    result = build_result()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_JSON, result)
    REPORT_MD.write_text(build_markdown(result))
    print(
        json.dumps(
            {
                "bundle": rel(BUNDLE_PATH),
                "md": rel(REPORT_MD),
                "json": rel(REPORT_JSON),
                "D": result["main_uncentered_injection_span"]["ambient_dimension_D"],
                "rank": result["main_uncentered_injection_span"]["rank_thresholds"]["legacy_main_audit_rel_1e-6"],
                "observed_projection": result["observed_projection_fraction"],
                "random_null_percentile": result["empirical_random_direction_null"]["observed_percentile"],
                "verdicts": result["verdict_constants"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
