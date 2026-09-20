"""Layer-native two-tap readiness probe v1.

This probe corrects the previous concat-heavy two-tap checks by evaluating
`MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` as layer-local tap bundles at
24_L4, 36_L4, and 47_L4.

It only reads cached tiny-head and feature artifacts, builds new diagnostic
candidate taps under a new artifact path, and evaluates cached datasets. It
does not train Ouro, modify checkpoints/tokenizers, overwrite old tap
registries, run wrapper/local-agent or Hunter-Seeker code, apply steering, or
change production routing.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, config_dim, md_table, rel
from bg_merged_tap_v1_common import (
    BRIDGE_PT,
    finite_mean,
    json_default,
    normed,
    pair_diff,
    safe_float,
    score_diff,
)
from run_bg_two_tap_full_readiness_v1 import (
    DOC_TARGETS,
    NAV_TARGETS,
    SURVIVOR_FEATURES_PT,
    append_section,
    branch_pair_datasets,
    candidate_feature,
    group_metric_from_order,
    group_reward,
    old_domain_pair_datasets,
    source_candidates,
    source_group_rank,
    tensor_weight,
)


OUT_ROOT = PROBE_ROOT / "bg_layer_native_two_tap_readiness_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "layer_native_two_tap_readiness.json"
REPORT_MD = OUT_ROOT / "layer_native_two_tap_readiness.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
PAIR_ROWS_CSV = OUT_ROOT / "layer_native_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "layer_native_group_rows.csv"
CANDIDATES_CSV = OUT_ROOT / "layer_native_candidates.csv"
ARTIFACT_PT = OUT_ROOT / "layer_native_two_tap_readiness_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_layer_native_two_tap_readiness_v1.md"

LAYER_CONFIGS = ("24_L4", "36_L4", "47_L4")
ANCHOR_GROUPS = ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL")
PRIMARY_ARCHES = ("AntisymLinearNoNorm", "AntisymLinear")
TWO_TAP_CANDIDATE_FAMILIES = {
    "layer_native_two_tap",
    "layer_native_two_tap_sparse",
    "layer_native_two_tap_transplant",
    "layer_native_two_tap_trained",
    "layer_native_two_tap_trained_adaptive",
    "layer_native_two_tap_targeted_rehost",
    "layer_native_bundle",
}
RESIDUAL_WEIGHTS = (
    ("old_plus_branch_bridge_80_10_10", 0.80, 0.10, 0.10),
    ("old_plus_branch_bridge_75_15_10", 0.75, 0.15, 0.10),
    ("old_plus_branch_bridge_70_15_15", 0.70, 0.15, 0.15),
    ("old_plus_branch_bridge_60_20_20", 0.60, 0.20, 0.20),
    ("old_plus_branch_bridge_50_25_25", 0.50, 0.25, 0.25),
    ("old_plus_branch_bridge_40_30_30", 0.40, 0.30, 0.30),
    ("old_plus_branch_bridge_40_40_20", 0.40, 0.40, 0.20),
    ("old_plus_branch_bridge_40_20_40", 0.40, 0.20, 0.40),
    ("old_plus_branch_bridge_30_35_35", 0.30, 0.35, 0.35),
    ("old_plus_branch_85_15", 0.85, 0.15, 0.00),
    ("old_plus_branch_70_30", 0.70, 0.30, 0.00),
    ("old_plus_branch_50_50", 0.50, 0.50, 0.00),
    ("old_plus_branch_30_70", 0.30, 0.70, 0.00),
    ("old_plus_bridge_85_15", 0.85, 0.00, 0.15),
    ("old_plus_bridge_70_30", 0.70, 0.00, 0.30),
    ("old_plus_bridge_50_50", 0.50, 0.00, 0.50),
    ("old_plus_bridge_30_70", 0.30, 0.00, 0.70),
)
SPARSE_RESIDUAL_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
SPARSE_RESIDUAL_WEIGHTS = (
    ("sparse_old_plus_branch_bridge_70_15_15", 0.70, 0.15, 0.15),
    ("sparse_old_plus_branch_bridge_60_20_20", 0.60, 0.20, 0.20),
    ("sparse_old_plus_branch_bridge_50_25_25", 0.50, 0.25, 0.25),
    ("sparse_old_plus_branch_bridge_40_30_30", 0.40, 0.30, 0.30),
    ("sparse_old_plus_branch_50_50", 0.50, 0.50, 0.00),
    ("sparse_old_plus_branch_30_70", 0.30, 0.70, 0.00),
    ("sparse_old_plus_bridge_50_50", 0.50, 0.00, 0.50),
    ("sparse_old_plus_bridge_30_70", 0.30, 0.00, 0.70),
)
TRANSPLANT_SOURCE_LIMITS = {
    "branch": 8,
    "bridge": 8,
    "old_content": 6,
}
TRANSPLANT_WEIGHTS = (
    ("residual_30_70", "residual", 0.30, 0.70),
    ("residual_10_90", "residual", 0.10, 0.90),
    ("full_30_70", "full", 0.30, 0.70),
    ("full_10_90", "full", 0.10, 0.90),
    ("full_00_100_diagnostic", "full", 0.00, 1.00),
)


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"weight", "state_dict"}}


def orthogonal_residual(vec: torch.Tensor, bases: Sequence[torch.Tensor], eps: float = 1e-8) -> tuple[torch.Tensor, float]:
    res = vec.detach().cpu().to(torch.float32).flatten().clone()
    ortho: list[torch.Tensor] = []
    for base in bases:
        b = base.detach().cpu().to(torch.float32).flatten()
        if b.shape != res.shape:
            continue
        for prior in ortho:
            b = b - torch.dot(b, prior) * prior
        n = float(b.norm().item())
        if n >= eps:
            ortho.append(b / n)
    for base in ortho:
        res = res - torch.dot(res, base) * base
    n = float(res.norm().item())
    return (res / n if n >= eps else torch.zeros_like(res), n)


def sparse_residual(vec: torch.Tensor, fraction: float, eps: float = 1e-8) -> torch.Tensor:
    flat = vec.detach().cpu().to(torch.float32).flatten()
    if not flat.numel() or float(flat.norm().item()) < eps:
        return torch.zeros_like(flat)
    k = max(1, min(int(flat.numel()), int(math.ceil(float(fraction) * int(flat.numel())))))
    idx = torch.topk(flat.abs(), k=k, largest=True).indices
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    n = float(out.norm().item())
    return out / n if n >= eps else torch.zeros_like(flat)


def candidate_name(row: dict[str, Any]) -> str:
    return str(row.get("candidate_name") or row.get("head_name") or row.get("name") or "")


def source_slug(row: dict[str, Any]) -> str:
    name = candidate_name(row)
    tokens = [tok for tok in name.replace("source::", "").replace("/", "::").split("::") if tok]
    keep = tokens[:2] + tokens[-4:]
    out = "_".join(keep)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in out)[:120]


def best_matching_source(
    refs: Sequence[dict[str, Any]],
    *,
    source_family: str,
    config: str,
    arch: str,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> dict[str, Any] | None:
    out = []
    for row in refs:
        if row.get("source_family") != source_family:
            continue
        if str(row.get("target_config")) != config or str(row.get("architecture")) != arch:
            continue
        name = candidate_name(row)
        low = name.lower()
        if include and not any(token.lower() in low for token in include):
            continue
        if exclude and any(token.lower() in low for token in exclude):
            continue
        weight = tensor_weight(row)
        if isinstance(weight, torch.Tensor) and int(weight.numel()) == config_dim(config):
            out.append(row)
    if not out:
        return None
    return max(out, key=lambda r: (safe_float(r.get("metric_score"), -1.0), candidate_name(r)))


def top_matching_sources(
    refs: Sequence[dict[str, Any]],
    *,
    source_family: str,
    config: str,
    arch: str,
    limit: int,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for row in refs:
        if row.get("source_family") != source_family:
            continue
        if str(row.get("target_config")) != config or str(row.get("architecture")) != arch:
            continue
        name = candidate_name(row)
        low = name.lower()
        if include and not any(token.lower() in low for token in include):
            continue
        if exclude and any(token.lower() in low for token in exclude):
            continue
        if name in seen:
            continue
        weight = tensor_weight(row)
        if isinstance(weight, torch.Tensor) and int(weight.numel()) == config_dim(config):
            seen.add(name)
            out.append(row)
    return sorted(out, key=lambda r: (safe_float(r.get("metric_score"), -1.0), candidate_name(r)), reverse=True)[:limit]


def build_layer_native_candidates(refs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    for arch in PRIMARY_ARCHES:
        for config in LAYER_CONFIGS:
            branch = best_matching_source(refs, source_family="branch", config=config, arch=arch)
            # source_candidates() preserves the inferred source_family but not
            # always the original universal-head variant in candidate_name, so
            # select from the bridge family directly here.
            bridge = best_matching_source(refs, source_family="bridge", config=config, arch=arch)
            if bridge is None:
                bridge = best_matching_source(refs, source_family="universal", config=config, arch=arch)
            for group in ANCHOR_GROUPS:
                old = best_matching_source(refs, source_family="old_content", config=config, arch=arch, include=(group,))
                if old is None:
                    source_summary.append(
                        {
                            "anchor_group": group,
                            "config": config,
                            "architecture": arch,
                            "status": "missing_old_anchor",
                        }
                    )
                    continue
                old_w = tensor_weight(old)
                branch_w = tensor_weight(branch or {})
                bridge_w = tensor_weight(bridge or {})
                if not isinstance(old_w, torch.Tensor):
                    continue
                old_u = normed(old_w)
                branch_res = torch.zeros_like(old_u)
                branch_norm = 0.0
                if isinstance(branch_w, torch.Tensor) and branch_w.shape == old_w.shape:
                    branch_res, branch_norm = orthogonal_residual(branch_w, [old_u])
                bridge_res = torch.zeros_like(old_u)
                bridge_norm = 0.0
                bridge_bases = [old_u]
                if float(branch_res.norm().item()) > 1e-8:
                    bridge_bases.append(branch_res)
                if isinstance(bridge_w, torch.Tensor) and bridge_w.shape == old_w.shape:
                    bridge_res, bridge_norm = orthogonal_residual(bridge_w, bridge_bases)
                # Preserve the original old layer-local direction as a baseline
                # under the same two tap identity.
                candidates.append(
                    {
                        "candidate_name": f"layer_native_old_only::{group}::{config}::{arch}",
                        "candidate_family": "layer_native_two_tap",
                        "tap_role": group,
                        "recipe": "old_only",
                        "target_config": config,
                        "architecture": arch,
                        "weight": old_u,
                        "state_dict": {"linear.weight": old_u.reshape(1, -1)},
                        "old_source": candidate_name(old),
                        "branch_source": candidate_name(branch or {}),
                        "bridge_source": candidate_name(bridge or {}),
                        "branch_residual_norm": branch_norm,
                        "bridge_residual_norm": bridge_norm,
                    }
                )
                for recipe, a, b, c in RESIDUAL_WEIGHTS:
                    if b > 0 and float(branch_res.norm().item()) <= 1e-8:
                        continue
                    if c > 0 and float(bridge_res.norm().item()) <= 1e-8:
                        continue
                    merged = normed(a * old_u + b * branch_res + c * bridge_res)
                    candidates.append(
                        {
                            "candidate_name": f"layer_native::{group}::{recipe}::{config}::{arch}",
                            "candidate_family": "layer_native_two_tap",
                            "tap_role": group,
                            "recipe": recipe,
                            "target_config": config,
                            "architecture": arch,
                            "weight": merged,
                            "state_dict": {"linear.weight": merged.reshape(1, -1)},
                            "old_source": candidate_name(old),
                            "branch_source": candidate_name(branch or {}),
                            "bridge_source": candidate_name(bridge or {}),
                            "branch_residual_norm": branch_norm,
                            "bridge_residual_norm": bridge_norm,
                            "coeff_old": a,
                            "coeff_branch": b,
                            "coeff_bridge": c,
                        }
                    )
                for frac in SPARSE_RESIDUAL_FRACTIONS:
                    sparse_branch = sparse_residual(branch_res, frac)
                    sparse_bridge = sparse_residual(bridge_res, frac)
                    frac_tag = f"top{str(frac).replace('.', 'p')}"
                    for recipe, a, b, c in SPARSE_RESIDUAL_WEIGHTS:
                        if b > 0 and float(sparse_branch.norm().item()) <= 1e-8:
                            continue
                        if c > 0 and float(sparse_bridge.norm().item()) <= 1e-8:
                            continue
                        merged = normed(a * old_u + b * sparse_branch + c * sparse_bridge)
                        candidates.append(
                            {
                                "candidate_name": f"layer_native_sparse::{group}::{recipe}_{frac_tag}::{config}::{arch}",
                                "candidate_family": "layer_native_two_tap_sparse",
                                "tap_role": group,
                                "recipe": f"{recipe}_{frac_tag}",
                                "target_config": config,
                                "architecture": arch,
                                "weight": merged,
                                "state_dict": {"linear.weight": merged.reshape(1, -1)},
                                "old_source": candidate_name(old),
                                "branch_source": candidate_name(branch or {}),
                                "bridge_source": candidate_name(bridge or {}),
                                "branch_residual_norm": branch_norm,
                                "bridge_residual_norm": bridge_norm,
                                "sparse_fraction": frac,
                                "coeff_old": a,
                                "coeff_branch": b,
                                "coeff_bridge": c,
                            }
                        )
                transplant_sources: list[tuple[str, dict[str, Any]]] = []
                for src_family in ("branch", "bridge"):
                    for src in top_matching_sources(
                        refs,
                        source_family=src_family,
                        config=config,
                        arch=arch,
                        limit=TRANSPLANT_SOURCE_LIMITS[src_family],
                    ):
                        transplant_sources.append((src_family, src))
                for src in top_matching_sources(
                    refs,
                    source_family="old_content",
                    config=config,
                    arch=arch,
                    limit=TRANSPLANT_SOURCE_LIMITS["old_content"],
                    exclude=(group,),
                ):
                    transplant_sources.append(("old_content", src))
                for src_family, src in transplant_sources:
                    src_w = tensor_weight(src)
                    if not isinstance(src_w, torch.Tensor) or src_w.shape != old_w.shape:
                        continue
                    src_u = normed(src_w)
                    src_res, src_res_norm = orthogonal_residual(src_w, [old_u])
                    for recipe, mode, old_coeff, source_coeff in TRANSPLANT_WEIGHTS:
                        if mode == "residual" and float(src_res.norm().item()) <= 1e-8:
                            continue
                        source_vec = src_res if mode == "residual" else src_u
                        merged = normed(old_coeff * old_u + source_coeff * source_vec)
                        candidates.append(
                            {
                                "candidate_name": f"layer_native_transplant::{group}::{src_family}_{recipe}::{source_slug(src)}::{config}::{arch}",
                                "candidate_family": "layer_native_two_tap_transplant",
                                "tap_role": group,
                                "recipe": f"{src_family}_{recipe}",
                                "target_config": config,
                                "architecture": arch,
                                "weight": merged,
                                "state_dict": {"linear.weight": merged.reshape(1, -1)},
                                "old_source": candidate_name(old),
                                "transplant_source": candidate_name(src),
                                "transplant_source_family": src_family,
                                "transplant_mode": mode,
                                "transplant_residual_norm": src_res_norm,
                                "coeff_old": old_coeff,
                                "coeff_source": source_coeff,
                            }
                        )
                source_summary.append(
                    {
                        "anchor_group": group,
                        "config": config,
                        "architecture": arch,
                        "status": "ready",
                        "old_source": candidate_name(old),
                        "branch_source": candidate_name(branch or {}),
                        "bridge_source": candidate_name(bridge or {}),
                        "branch_residual_norm": branch_norm,
                        "bridge_residual_norm": bridge_norm,
                    }
                )
    return candidates, source_summary


def primary_split(pairs: Sequence[dict[str, Any]]) -> str | None:
    counts = Counter(str(pair.get("split") or "all") for pair in pairs)
    for split in ("heldout", "test", "fresh_holdout", "domain_probe", "all"):
        if counts.get(split, 0) > 0:
            return split
    return counts.most_common(1)[0][0] if counts else None


def filter_pairs_for_primary(pairs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    split = primary_split(pairs)
    if split is None:
        return [], "none"
    if split == "all":
        return list(pairs), split
    return [pair for pair in pairs if str(pair.get("split") or "all") == split], split


def pair_scores_for_candidate(candidate: dict[str, Any], pairs: Sequence[dict[str, Any]]) -> tuple[list[float], list[str]]:
    weight = tensor_weight(candidate)
    if not isinstance(weight, torch.Tensor):
        return [], []
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    scores: list[float] = []
    domains: list[str] = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != int(weight.numel()):
            continue
        scores.append(score_diff(weight, arch, diff))
        domains.append(str(pair.get("domain")))
    return scores, domains


def accuracy_from_scores(scores: Sequence[float]) -> float:
    if not scores:
        return float("nan")
    return float(sum(1.0 if s > 0 else 0.5 if s == 0 else 0.0 for s in scores) / len(scores))


def normalized(vals: Sequence[float]) -> list[float]:
    xs = [safe_float(v, 0.0) for v in vals]
    if not xs:
        return []
    mu = mean(xs)
    var = mean((x - mu) ** 2 for x in xs)
    sd = math.sqrt(var)
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]


def bundle_specs(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    bundle_candidates = [c for c in candidates if c.get("candidate_family") != "layer_native_two_tap_transplant"]
    recipes = sorted({str(c.get("recipe")) for c in bundle_candidates})
    for arch in PRIMARY_ARCHES:
        for recipe in recipes:
            for group in ANCHOR_GROUPS:
                members = [
                    c
                    for c in bundle_candidates
                    if c.get("tap_role") == group
                    and c.get("recipe") == recipe
                    and c.get("architecture") == arch
                    and c.get("target_config") in LAYER_CONFIGS
                ]
                if len({m.get("target_config") for m in members}) == len(LAYER_CONFIGS):
                    specs.append(
                        {
                            "bundle_name": f"bundle::{group}::{recipe}::{arch}::24_36_47",
                            "bundle_family": "layer_native_bundle",
                            "recipe": recipe,
                            "architecture": arch,
                            "members": [m.get("candidate_name") for m in members],
                        }
                    )
            members = [
                c
                for c in bundle_candidates
                if c.get("recipe") == recipe
                and c.get("architecture") == arch
                and c.get("target_config") in LAYER_CONFIGS
            ]
            groups = {m.get("tap_role") for m in members}
            configs = {m.get("target_config") for m in members}
            if set(groups) == set(ANCHOR_GROUPS) and set(configs) == set(LAYER_CONFIGS):
                specs.append(
                    {
                        "bundle_name": f"bundle::two_tap_equal::{recipe}::{arch}::24_36_47",
                        "bundle_family": "layer_native_bundle",
                        "recipe": recipe,
                        "architecture": arch,
                        "members": [m.get("candidate_name") for m in members],
                    }
                )
    return specs


def pair_eval_rows_for_dataset(
    dataset: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    refs: Sequence[dict[str, Any]],
    bundles: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs, eval_split = filter_pairs_for_primary(dataset.get("pairs") or [])
    rows: list[dict[str, Any]] = []
    all_single = list(candidates) + list(refs)
    for candidate in all_single:
        scores, domains = pair_scores_for_candidate(candidate, pairs)
        if not scores:
            continue
        domain_accuracy = {}
        for domain in sorted(set(domains)):
            subset = [score for score, d in zip(scores, domains) if d == domain]
            domain_accuracy[domain] = accuracy_from_scores(subset)
        rows.append(
            {
                "dataset_name": dataset["dataset_name"],
                "dataset_kind": dataset["dataset_kind"],
                "readiness_eligible": bool(dataset.get("readiness_eligible")),
                "eval_split": eval_split,
                "candidate_name": candidate.get("candidate_name"),
                "candidate_family": candidate.get("candidate_family"),
                "source_family": candidate.get("source_family"),
                "source_run": candidate.get("source_run"),
                "tap_role": candidate.get("tap_role"),
                "recipe": candidate.get("recipe"),
                "target_config": candidate.get("target_config"),
                "architecture": candidate.get("architecture"),
                "pair_count": len(scores),
                "pairwise_accuracy": accuracy_from_scores(scores),
                "mean_margin": finite_mean(scores),
                "domain_accuracy": domain_accuracy,
            }
        )
    by_name = {str(c.get("candidate_name")): c for c in candidates}
    for bundle in bundles:
        member_rows = [by_name[name] for name in bundle.get("members", []) if name in by_name]
        if not member_rows:
            continue
        member_scores: list[list[float]] = []
        domains: list[str] | None = None
        for member in member_rows:
            scores, ds = pair_scores_for_candidate(member, pairs)
            if not scores:
                continue
            if domains is None:
                domains = ds
            if len(scores) == len(domains or []):
                member_scores.append(normalized(scores))
        if not member_scores or domains is None:
            continue
        n = min(len(scores) for scores in member_scores)
        if n <= 0:
            continue
        scores = [finite_mean(ms[i] for ms in member_scores) for i in range(n)]
        ds = domains[:n]
        domain_accuracy = {}
        for domain in sorted(set(ds)):
            subset = [score for score, d in zip(scores, ds) if d == domain]
            domain_accuracy[domain] = accuracy_from_scores(subset)
        rows.append(
            {
                "dataset_name": dataset["dataset_name"],
                "dataset_kind": dataset["dataset_kind"],
                "readiness_eligible": bool(dataset.get("readiness_eligible")),
                "eval_split": eval_split,
                "candidate_name": bundle.get("bundle_name"),
                "candidate_family": bundle.get("bundle_family"),
                "source_family": "layer_native_bundle",
                "tap_role": "bundle",
                "recipe": bundle.get("recipe"),
                "target_config": "24_L4+36_L4+47_L4",
                "architecture": bundle.get("architecture"),
                "pair_count": n,
                "pairwise_accuracy": accuracy_from_scores(scores),
                "mean_margin": finite_mean(scores),
                "domain_accuracy": domain_accuracy,
            }
        )
    return rows


def groups_from_candidate_rows_any(path: Path, dataset_name: str, split: str | None = None) -> list[list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else {}
    rows = [dict(row) for row in payload.get("candidate_rows") or []]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split and str(row.get("split")) != split:
            continue
        fmap = row.get("features_by_config") or row.get("features") or {}
        if not any(isinstance(fmap.get(config), torch.Tensor) for config in LAYER_CONFIGS):
            continue
        row.setdefault("source_dataset", dataset_name)
        grouped[str(row.get("group_id") or row.get("branch_group_id"))].append(row)
    return [vals for vals in grouped.values() if len(vals) >= 2]


def groups_from_survivor_features_any(split: str | None = None) -> list[list[dict[str, Any]]]:
    payload = torch.load(SURVIVOR_FEATURES_PT, map_location="cpu", weights_only=False) if SURVIVOR_FEATURES_PT.exists() else {}
    groups: list[list[dict[str, Any]]] = []
    for survivor_set in payload.get("survivor_sets") or []:
        if split and str(survivor_set.get("split") or survivor_set.get("v1_split")) != split:
            continue
        candidates = []
        for cand in survivor_set.get("candidates") or []:
            row = dict(cand)
            fmap = row.get("features_by_config") or {}
            if not any(isinstance(fmap.get(config), torch.Tensor) for config in LAYER_CONFIGS):
                continue
            row["reward"] = group_reward(row)
            row["group_id"] = survivor_set.get("survivor_set_id")
            row["domain"] = survivor_set.get("domain")
            row["split"] = survivor_set.get("split") or survivor_set.get("v1_split") or "all"
            candidates.append(row)
        if len(candidates) >= 2:
            groups.append(candidates)
    return groups


def group_datasets_any() -> list[dict[str, Any]]:
    return [
        {
            "dataset_name": "universal_bridge_candidate_groups",
            "dataset_kind": "branch_group",
            "path": BRIDGE_PT,
            "groups": groups_from_candidate_rows_any(BRIDGE_PT, "universal_bridge_candidate_groups", split="heldout"),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "top4_survivor_hidden_feature_groups",
            "dataset_kind": "branch_group",
            "path": SURVIVOR_FEATURES_PT,
            "groups": groups_from_survivor_features_any(split="fresh_holdout"),
            "readiness_eligible": True,
        },
    ]


def bundle_group_scores(group: Sequence[dict[str, Any]], members: Sequence[dict[str, Any]]) -> list[float]:
    component_scores: list[list[float]] = []
    for member in members:
        weight = tensor_weight(member)
        if not isinstance(weight, torch.Tensor):
            continue
        config = str(member.get("target_config"))
        arch = str(member.get("architecture"))
        out = []
        ok = True
        for i, left in enumerate(group):
            lvec = candidate_feature(left, config)
            if not isinstance(lvec, torch.Tensor):
                ok = False
                break
            vals = []
            for j, right in enumerate(group):
                if i == j:
                    continue
                rvec = candidate_feature(right, config)
                if not isinstance(rvec, torch.Tensor):
                    ok = False
                    break
                vals.append(score_diff(weight, arch, lvec - rvec))
            if not ok:
                break
            out.append(finite_mean(vals))
        if ok and len(out) == len(group):
            component_scores.append(normalized(out))
    if not component_scores:
        return []
    return [finite_mean(scores[i] for scores in component_scores) for i in range(len(group))]


def eval_group_dataset(
    dataset: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    refs: Sequence[dict[str, Any]],
    bundles: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = list(dataset.get("groups") or [])
    if not groups:
        return rows
    by_name = {str(c.get("candidate_name")): c for c in candidates}

    def summarize(policy: str, family: str, group_metrics: list[dict[str, Any]], candidate: dict[str, Any] | None = None) -> None:
        if not group_metrics:
            return
        rows.append(
            {
                "dataset_name": dataset["dataset_name"],
                "dataset_kind": dataset["dataset_kind"],
                "readiness_eligible": bool(dataset.get("readiness_eligible")),
                "policy_name": policy,
                "candidate_name": (candidate or {}).get("candidate_name"),
                "candidate_family": family,
                "source_family": (candidate or {}).get("source_family"),
                "target_config": (candidate or {}).get("target_config", "24_L4+36_L4+47_L4"),
                "architecture": (candidate or {}).get("architecture"),
                "group_count": len(group_metrics),
                "oracle_retention": finite_mean(m.get("oracle_retention") for m in group_metrics),
                "false_prune_rate": finite_mean(m.get("false_prune_rate") for m in group_metrics),
                "avg_survivors": finite_mean(m.get("avg_survivors") for m in group_metrics),
                "best_selected_reward": finite_mean(m.get("best_selected_reward") for m in group_metrics),
                "top1_success": finite_mean(m.get("top1_success") for m in group_metrics),
                "top1_reward": finite_mean(m.get("top1_reward") for m in group_metrics),
                "regret": finite_mean(m.get("regret") for m in group_metrics),
                "domain_retention": {
                    d: finite_mean(m.get("oracle_retention") for m in vals)
                    for d, vals in sorted(group_by_domain(group_metrics).items())
                },
            }
        )

    for candidate in list(candidates) + list(refs):
        metrics = []
        for group in groups:
            order = source_group_rank(candidate, group)
            metric = group_metric_from_order(group, order, k=4)
            if metric:
                metrics.append({"domain": group[0].get("domain"), **metric})
        summarize(str(candidate.get("candidate_name")), str(candidate.get("candidate_family")), metrics, candidate)

    for bundle in bundles:
        members = [by_name[name] for name in bundle.get("members", []) if name in by_name]
        metrics = []
        for group in groups:
            scores = bundle_group_scores(group, members)
            if not scores:
                continue
            order = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]
            metric = group_metric_from_order(group, order, k=4)
            if metric:
                metrics.append({"domain": group[0].get("domain"), **metric})
        summarize(str(bundle.get("bundle_name")), str(bundle.get("bundle_family")), metrics)
    return rows


def group_by_domain(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("domain"))].append(row)
    return out


def summarize_pair_dataset(rows: Sequence[dict[str, Any]], dataset_kind: str) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_kind") == dataset_kind:
            by_dataset[str(row.get("dataset_name"))].append(row)
    summaries = []
    for dataset_name, vals in sorted(by_dataset.items()):
        eligible = bool(vals[0].get("readiness_eligible")) if vals else False
        two_rows = [
            r
            for r in vals
            if r.get("candidate_family") in TWO_TAP_CANDIDATE_FAMILIES
            and math.isfinite(safe_float(r.get("pairwise_accuracy"), float("nan")))
        ]
        if dataset_kind == "old_domain_pair":
            ref_rows = [r for r in vals if r.get("candidate_family") == "source_old_content"]
        else:
            ref_rows = [r for r in vals if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
        best_two = max(two_rows, key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
        best_ref = max(ref_rows, key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
        two_acc = safe_float(best_two.get("pairwise_accuracy"), float("nan"))
        ref_acc = safe_float(best_ref.get("pairwise_accuracy"), float("nan"))
        summaries.append(
            {
                "dataset_name": dataset_name,
                "dataset_kind": dataset_kind,
                "readiness_eligible": eligible,
                "eval_split": vals[0].get("eval_split") if vals else "",
                "pair_count": best_two.get("pair_count") or best_ref.get("pair_count") or 0,
                "best_two_tap": best_two.get("candidate_name"),
                "best_two_tap_family": best_two.get("candidate_family"),
                "best_two_tap_accuracy": two_acc,
                "best_reference": best_ref.get("candidate_name"),
                "best_reference_family": best_ref.get("candidate_family"),
                "best_reference_accuracy": ref_acc,
                "delta_two_minus_reference": two_acc - ref_acc if math.isfinite(two_acc) and math.isfinite(ref_acc) else float("nan"),
                "matches_or_exceeds_reference": bool(math.isfinite(two_acc) and math.isfinite(ref_acc) and two_acc + 1e-9 >= ref_acc),
            }
        )
    return {"datasets": summaries}


def summarize_group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset_name"))].append(row)
    datasets = []
    for dataset_name, vals in sorted(by_dataset.items()):
        two_rows = [r for r in vals if r.get("candidate_family") in TWO_TAP_CANDIDATE_FAMILIES]
        ref_rows = [r for r in vals if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
        best_two = max(two_rows, key=lambda r: (safe_float(r.get("oracle_retention"), -1.0), -safe_float(r.get("false_prune_rate"), 1.0), safe_float(r.get("top1_reward"), -1.0)), default={})
        best_ref = max(ref_rows, key=lambda r: (safe_float(r.get("oracle_retention"), -1.0), -safe_float(r.get("false_prune_rate"), 1.0), safe_float(r.get("top1_reward"), -1.0)), default={})
        two_ret = safe_float(best_two.get("oracle_retention"), float("nan"))
        ref_ret = safe_float(best_ref.get("oracle_retention"), float("nan"))
        datasets.append(
            {
                "dataset_name": dataset_name,
                "readiness_eligible": bool(vals[0].get("readiness_eligible")) if vals else False,
                "group_count": best_two.get("group_count") or best_ref.get("group_count") or 0,
                "best_two_tap": best_two.get("policy_name") or best_two.get("candidate_name"),
                "best_two_tap_family": best_two.get("candidate_family"),
                "best_two_tap_retention": two_ret,
                "best_two_tap_false_prune": best_two.get("false_prune_rate"),
                "best_two_tap_top1_reward": best_two.get("top1_reward"),
                "best_reference": best_ref.get("policy_name") or best_ref.get("candidate_name"),
                "best_reference_family": best_ref.get("candidate_family"),
                "best_reference_retention": ref_ret,
                "best_reference_false_prune": best_ref.get("false_prune_rate"),
                "best_reference_top1_reward": best_ref.get("top1_reward"),
                "delta_two_minus_reference": two_ret - ref_ret if math.isfinite(two_ret) and math.isfinite(ref_ret) else float("nan"),
                "matches_or_exceeds_reference": bool(math.isfinite(two_ret) and math.isfinite(ref_ret) and two_ret + 1e-9 >= ref_ret),
            }
        )
    return {"datasets": datasets}


def readiness(domain_summary: dict[str, Any], branch_summary: dict[str, Any], group_summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    domain_sets = [d for d in domain_summary.get("datasets", []) if d.get("readiness_eligible") and int(d.get("pair_count") or 0) > 0]
    branch_sets = [d for d in branch_summary.get("datasets", []) if d.get("readiness_eligible") and int(d.get("pair_count") or 0) > 0]
    group_sets = [d for d in group_summary.get("datasets", []) if d.get("readiness_eligible") and int(d.get("group_count") or 0) > 0]
    domain_failures = [d for d in domain_sets if not d.get("matches_or_exceeds_reference")]
    branch_failures = [d for d in branch_sets if not d.get("matches_or_exceeds_reference")]
    group_failures = [d for d in group_sets if not d.get("matches_or_exceeds_reference")]
    domain_ok = bool(domain_sets) and not domain_failures
    branch_ok = bool(branch_sets or group_sets) and not branch_failures and not group_failures
    data_ok = bool(domain_sets) and bool(branch_sets or group_sets)
    if domain_ok and branch_ok:
        verdict = "LAYER_NATIVE_TWO_TAP_READY"
    elif branch_ok and not domain_ok:
        verdict = "BRANCH_READY_DOMAIN_GAP"
    elif domain_ok and not branch_ok:
        verdict = "DOMAIN_READY_BRANCH_GAP"
    elif data_ok:
        verdict = "LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY"
    else:
        verdict = "DATA_LIMITED"
    return verdict, {
        "domain_ok": domain_ok,
        "branch_ok": branch_ok,
        "data_ok": data_ok,
        "domain_dataset_count": len(domain_sets),
        "branch_pair_dataset_count": len(branch_sets),
        "branch_group_dataset_count": len(group_sets),
        "domain_failures": domain_failures,
        "branch_pair_failures": branch_failures,
        "branch_group_failures": group_failures,
    }


def repo_inventory() -> dict[str, Any]:
    top_counts = {}
    top_size = {}
    for p in PROJECT_ROOT.rglob("*"):
        try:
            relp = p.relative_to(PROJECT_ROOT)
        except Exception:
            continue
        if not relp.parts:
            continue
        top = relp.parts[0]
        top_counts[top] = top_counts.get(top, 0) + 1
        if p.is_file():
            try:
                top_size[top] = top_size.get(top, 0) + p.stat().st_size
            except OSError:
                pass
    probe_dirs = []
    probe_root = PROBE_ROOT
    if probe_root.exists():
        for d in sorted(p for p in probe_root.iterdir() if p.is_dir()):
            name = d.name.lower()
            if any(t in name for t in ("bg", "tap", "branch", "hidden", "universal", "gated", "composite", "arbiter")):
                probe_dirs.append({"path": rel(d), "items": sum(1 for _ in d.iterdir())})
    return {
        "top_level": {
            key: {"entries": top_counts[key], "size_mb": round(top_size.get(key, 0) / 1e6, 3)}
            for key in sorted(top_counts)
        },
        "probe_dirs": probe_dirs,
    }


def main() -> int:
    ensure_root()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    refs = source_candidates()
    layer_refs = [r for r in refs if r.get("target_config") in LAYER_CONFIGS and r.get("architecture") in PRIMARY_ARCHES]
    candidates, source_summary = build_layer_native_candidates(layer_refs)
    bundles = bundle_specs(candidates)

    old_datasets = old_domain_pair_datasets()
    branch_datasets = branch_pair_datasets()
    pair_rows: list[dict[str, Any]] = []
    for dataset in old_datasets + branch_datasets:
        pair_rows.extend(pair_eval_rows_for_dataset(dataset, candidates, layer_refs, bundles))

    domain_summary = summarize_pair_dataset(pair_rows, "old_domain_pair")
    branch_summary = summarize_pair_dataset(pair_rows, "branch_pair")

    group_rows: list[dict[str, Any]] = []
    branch_group_datasets = group_datasets_any()
    bounded_refs = []
    for fam in ("source_branch", "source_bridge", "source_universal"):
        rows = [
            r
            for r in layer_refs
            if r.get("candidate_family") == fam
            and r.get("target_config") in LAYER_CONFIGS
            and r.get("architecture") in PRIMARY_ARCHES
        ]
        bounded_refs.extend(sorted(rows, key=lambda r: safe_float(r.get("metric_score"), -1.0), reverse=True)[:24])
    for dataset in branch_group_datasets:
        group_rows.extend(eval_group_dataset(dataset, candidates, bounded_refs, bundles))
    group_summary = summarize_group_rows(group_rows)

    verdict, ready = readiness(domain_summary, branch_summary, group_summary)
    artifact = {
        "BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT": verdict,
        "verdict": verdict,
        "layer_configs": list(LAYER_CONFIGS),
        "anchor_groups": list(ANCHOR_GROUPS),
        "architectures": list(PRIMARY_ARCHES),
        "source_summary": source_summary,
        "candidate_count": len(candidates),
        "bundle_count": len(bundles),
        "reference_count": len(layer_refs),
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_summary,
        "branch_group_summary": group_summary,
        "readiness": ready,
        "repo_inventory": repo_inventory(),
        "anti_leakage": {
            "no_ouro_training": True,
            "no_old_tap_registry_update": True,
            "no_action_steering": True,
            "no_production_routing_change": True,
            "tap_scores_not_used_as_labels": True,
            "native_layer_configs_only_for_primary_two_tap": True,
        },
    }
    torch.save(
        {
            "summary": artifact,
            "layer_native_candidates": candidates,
            "bundles": bundles,
            "references": layer_refs,
        },
        ARTIFACT_PT,
    )
    write_json(REPORT_JSON, artifact)
    write_json(SUMMARY_JSON, artifact)
    write_csv(PAIR_ROWS_CSV, pair_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(CANDIDATES_CSV, [clean_row(c) for c in candidates])

    lines = [
        "# Layer-Native Two-Tap Readiness v1",
        "",
        f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
        "",
        "This corrects the prior concat-heavy test. The only new tap identities are `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL`; each is evaluated as native `24_L4`, `36_L4`, and `47_L4` directions. Branch and bridge residuals are added only inside matching layer/config coordinate systems.",
        "",
        f"- layer configs: `{list(LAYER_CONFIGS)}`",
        f"- candidate layer heads: `{len(candidates)}`",
        f"- predefined layer bundles: `{len(bundles)}`",
        f"- layer-local references: `{len(layer_refs)}`",
        f"- domain ok: `{ready['domain_ok']}`",
        f"- branch ok: `{ready['branch_ok']}`",
        "",
        "## Old-Domain Datasets",
        "",
    ]
    lines.extend(md_table(domain_summary.get("datasets", []), ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Branch Pair Datasets", ""])
    lines.extend(md_table(branch_summary.get("datasets", []), ["dataset_name", "readiness_eligible", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Branch Group Datasets", ""])
    lines.extend(md_table(group_summary.get("datasets", []), ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    if ready["domain_failures"]:
        lines.extend(["", "## Domain Gaps", ""])
        lines.extend(md_table(ready["domain_failures"], ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "best_reference_family"]))
    if ready["branch_pair_failures"] or ready["branch_group_failures"]:
        lines.extend(["", "## Branch Gaps", ""])
        lines.extend(md_table(ready["branch_pair_failures"], ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "best_reference_family"]))
        lines.extend(md_table(ready["branch_group_failures"], ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "best_reference_family"]))
    lines.extend(["", "## Layer Source Audit", ""])
    lines.extend(md_table(source_summary, ["anchor_group", "config", "architecture", "status", "branch_residual_norm", "bridge_residual_norm", "old_source", "branch_source", "bridge_source"]))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- pair rows: `{rel(PAIR_ROWS_CSV)}`",
            f"- group rows: `{rel(GROUP_ROWS_CSV)}`",
            f"- candidates: `{rel(CANDIDATES_CSV)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# Layer-Native Two-Tap Readiness v1",
            "",
            f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
            "",
            "This run re-evaluates the two-tap idea in the native layer-local coordinate systems used by the old BG tap lineage: `24_L4`, `36_L4`, and `47_L4`. It does not rely on a concat tap as the primary object.",
            "",
            "## Result",
            "",
            f"- domain ok: `{ready['domain_ok']}`",
            f"- branch ok: `{ready['branch_ok']}`",
            f"- status: `{verdict}`",
            "",
            "## Files",
            "",
            f"- report: `{rel(REPORT_MD)}`",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
        ],
    )

    section_title = "## Layer-native two-tap readiness v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}`. Re-tested `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` as native `24_L4`, `36_L4`, and `47_L4` tap bundles instead of concat-only taps. Domain ok `{ready['domain_ok']}`; branch ok `{ready['branch_ok']}`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [
        section_title,
        "",
        f"Added `{rel(DOC_MD)}`. Status: `{verdict}`.",
    ]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)

    print(f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}", flush=True)
    print(f"domain_ok = {ready['domain_ok']} branch_ok = {ready['branch_ok']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
