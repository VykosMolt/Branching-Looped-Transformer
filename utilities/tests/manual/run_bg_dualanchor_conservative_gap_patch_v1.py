"""Conservative DualAnchor branch-gap patch probe v1.

This is a narrow diagnostic over existing tiny tap artifacts. It starts from
the pre-repair constrained DualAnchor candidates, adds very small residual
patches from same-config branch/bridge specialist directions, and evaluates
whether the remaining branch-pair gaps are mechanically closable without
discarding the old-domain lead.

It does not train Ouro, update old taps/registries, run wrapper/local-agent or
Hunter-Seeker code, apply steering, or change production routing.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, config_dim, md_table, rel
from bg_merged_tap_v1_common import json_default, normed, pair_diff, safe_float
from run_bg_layer_native_two_tap_readiness_v1 import (
    ANCHOR_GROUPS,
    LAYER_CONFIGS,
    PRIMARY_ARCHES,
    TWO_TAP_CANDIDATE_FAMILIES,
    accuracy_from_scores,
    branch_pair_datasets,
    candidate_name,
    eval_group_dataset,
    filter_pairs_for_primary,
    group_datasets_any,
    old_domain_pair_datasets,
    orthogonal_residual,
    pair_scores_for_candidate,
    readiness,
    source_candidates,
    source_slug,
    summarize_group_rows,
    summarize_pair_dataset,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_conservative_gap_patch_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "conservative_gap_patch.json"
REPORT_MD = OUT_ROOT / "conservative_gap_patch.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
PAIR_ROWS_CSV = OUT_ROOT / "conservative_gap_patch_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "conservative_gap_patch_group_rows.csv"
PATCH_ROWS_CSV = OUT_ROOT / "patch_candidate_rows.csv"
VALIDATION_ROWS_CSV = OUT_ROOT / "validation_selection_rows.csv"
ARTIFACT_PT = OUT_ROOT / "dualanchor_conservative_gap_patch_v1.pt"

PRE_REPAIR_PT = (
    PROBE_ROOT
    / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30"
    / "layer_native_two_tap_constrained_train_v1.pt"
)
PRE_REPAIR_JSON = (
    PROBE_ROOT
    / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30"
    / "constrained_train_eval.json"
)

BRANCH_PRIMARY_CONFIGS = ("24_L4", "36_L4")
LATE_DIAGNOSTIC_CONFIGS = ("47_L4",)
PATCH_COEFFICIENTS = (0.01, 0.02, 0.035, 0.05, 0.075)
SOURCE_LIMITS = {"branch": 6, "bridge": 3, "universal": 3}
SOURCE_FAMILIES = ("branch", "bridge", "universal")
GAP_TOKENS = (
    "branch_generator_v1",
    "hidden_origin_v3::high_yield",
    "salvage::old_frozen_tap_clean",
)


def runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"weight", "state_dict"}}


def load_pre_repair() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(PRE_REPAIR_PT, map_location="cpu", weights_only=False)
    summary = json.loads(PRE_REPAIR_JSON.read_text(encoding="utf-8"))
    candidates = [dict(c) for c in payload.get("candidates") or []]
    trained = [dict(c) for c in payload.get("trained") or []]
    refs = [dict(r) for r in payload.get("references") or []]
    return summary, candidates, trained, refs


def top_reference_sources(refs: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for ref in refs:
        family = str(ref.get("source_family") or "")
        if family not in SOURCE_FAMILIES:
            continue
        config = str(ref.get("target_config") or "")
        arch = str(ref.get("architecture") or "")
        if config not in LAYER_CONFIGS or arch not in PRIMARY_ARCHES:
            continue
        name = candidate_name(ref)
        if name in seen:
            continue
        weight = tensor_weight(ref)
        if not isinstance(weight, torch.Tensor) or int(weight.numel()) != config_dim(config):
            continue
        seen.add(name)
        by_key[(family, config, arch)].append(ref)
    for key, vals in list(by_key.items()):
        family = key[0]
        limit = SOURCE_LIMITS.get(family, 4)
        by_key[key] = sorted(
            vals,
            key=lambda r: (
                safe_float(r.get("metric_score"), -1.0),
                "hidden_origin_v4" in candidate_name(r),
                "hidden_origin_v3" in candidate_name(r),
                candidate_name(r),
            ),
            reverse=True,
        )[:limit]
    return by_key


def candidate_lookup(candidates: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {candidate_name(c): c for c in candidates if candidate_name(c)}


def seed_names_from_summary(summary: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for section in ("domain_summary", "branch_pair_summary", "branch_group_summary"):
        for row in (summary.get(section) or {}).get("datasets") or []:
            name = row.get("best_two_tap")
            if name:
                names.add(str(name))
    for row in (summary.get("readiness") or {}).get("branch_pair_failures") or []:
        name = row.get("best_two_tap")
        if name:
            names.add(str(name))
    return names


def select_seed_candidates(summary: dict[str, Any], candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = candidate_lookup(candidates)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in sorted(seed_names_from_summary(summary)):
        cand = by_name.get(name)
        if cand is None:
            continue
        if cand.get("candidate_family") not in TWO_TAP_CANDIDATE_FAMILIES:
            continue
        config = str(cand.get("target_config") or "")
        arch = str(cand.get("architecture") or "")
        if config not in LAYER_CONFIGS or arch not in PRIMARY_ARCHES:
            continue
        if name not in seen:
            selected.append(cand)
            seen.add(name)

    # Add the old-only anchors so tiny residuals can also be tested from the
    # cleanest possible starting point. This includes 47_L4 for evaluation
    # context, but build_patch_candidates() only patches 24/36 as branch
    # primary layers.
    for cand in candidates:
        if cand.get("recipe") != "old_only":
            continue
        if cand.get("tap_role") not in ANCHOR_GROUPS:
            continue
        name = candidate_name(cand)
        if name and name not in seen:
            selected.append(cand)
            seen.add(name)
    return selected


def build_patch_candidates(
    seeds: Sequence[dict[str, Any]],
    refs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_key = top_reference_sources(refs)
    patches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed in seeds:
        seed_w = tensor_weight(seed)
        if not isinstance(seed_w, torch.Tensor):
            continue
        config = str(seed.get("target_config") or "")
        arch = str(seed.get("architecture") or "")
        if config not in BRANCH_PRIMARY_CONFIGS or arch not in PRIMARY_ARCHES:
            continue
        seed_u = normed(seed_w)
        for family in SOURCE_FAMILIES:
            for src in source_by_key.get((family, config, arch), []):
                src_w = tensor_weight(src)
                if not isinstance(src_w, torch.Tensor) or src_w.shape != seed_w.shape:
                    continue
                residual, residual_norm = orthogonal_residual(src_w, [seed_u])
                if residual_norm <= 1e-8:
                    continue
                for coeff in PATCH_COEFFICIENTS:
                    patched = normed(seed_u + float(coeff) * residual)
                    source_tag = source_slug(src)
                    seed_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in candidate_name(seed))[:96]
                    coeff_tag = str(coeff).replace(".", "p")
                    name = (
                        f"dualanchor_conservative_patch::{seed.get('tap_role')}::{family}::"
                        f"eps_{coeff_tag}::{source_tag}::{seed_tag}::{config}::{arch}"
                    )
                    if name in seen:
                        continue
                    seen.add(name)
                    row = {
                        "candidate_name": name,
                        "candidate_family": "layer_native_two_tap_trained_adaptive",
                        "patch_family": "layer_native_two_tap_conservative_patch",
                        "tap_role": seed.get("tap_role"),
                        "recipe": f"conservative_{family}_eps_{coeff_tag}",
                        "target_config": config,
                        "architecture": arch,
                        "weight": patched,
                        "state_dict": {"linear.weight": patched.reshape(1, -1)},
                        "seed_candidate": candidate_name(seed),
                        "source_candidate": candidate_name(src),
                        "source_family": family,
                        "patch_coeff": coeff,
                        "residual_norm": residual_norm,
                    }
                    patches.append(row)
                    rows.append(clean_row(row))
    return patches, rows


def eval_candidate_on_datasets(candidate: dict[str, Any], datasets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        pairs, split = filter_pairs_for_primary(dataset.get("pairs") or [])
        scores, _domains = pair_scores_for_candidate(candidate, pairs)
        if not scores:
            continue
        rows.append(
            {
                "dataset_name": dataset.get("dataset_name"),
                "dataset_kind": dataset.get("dataset_kind"),
                "readiness_eligible": bool(dataset.get("readiness_eligible")),
                "eval_split": split,
                "pair_count": len(scores),
                "accuracy": accuracy_from_scores(scores),
            }
        )
    return rows


def dataset_pair_matrix(
    dataset: dict[str, Any],
    config: str,
    arch: str,
    *,
    split_override: str | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor | None, str, int]:
    if split_override is None:
        pairs, split = filter_pairs_for_primary(dataset.get("pairs") or [])
    else:
        split = split_override
        pairs = [pair for pair in dataset.get("pairs") or [] if str(pair.get("split") or "all") == split_override]
    rows: list[torch.Tensor] = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != config_dim(config):
            continue
        rows.append(diff.detach().cpu().to(torch.float32).flatten())
    if not rows:
        return None, split, 0
    x = torch.stack(rows, dim=0)
    if arch == "AntisymLinear":
        x = F.layer_norm(x, (x.shape[1],))
    dev = device or runtime_device()
    return x.to(dev, non_blocking=True), split, int(x.shape[0])


def weights_for_key(candidates: Sequence[dict[str, Any]], config: str, arch: str, device: torch.device) -> tuple[list[dict[str, Any]], torch.Tensor | None]:
    rows: list[dict[str, Any]] = []
    weights: list[torch.Tensor] = []
    seen: set[str] = set()
    for cand in candidates:
        if str(cand.get("target_config") or "") != config or str(cand.get("architecture") or "") != arch:
            continue
        name = candidate_name(cand)
        if not name or name in seen:
            continue
        weight = tensor_weight(cand)
        if not isinstance(weight, torch.Tensor) or int(weight.numel()) != config_dim(config):
            continue
        seen.add(name)
        rows.append(cand)
        weights.append(weight.detach().cpu().to(torch.float32).flatten())
    if not weights:
        return rows, None
    return rows, torch.stack(weights, dim=0).to(device, non_blocking=True)


def accuracy_and_margin_from_scores(scores: torch.Tensor) -> tuple[list[float], list[float]]:
    with torch.no_grad():
        positive = (scores > 0).sum(dim=0).detach().cpu().tolist()
        tied = (scores == 0).sum(dim=0).detach().cpu().tolist()
        denom = int(scores.shape[0])
        acc = [(float(p) + 0.5 * float(t)) / max(denom, 1) for p, t in zip(positive, tied)]
        margin = scores.mean(dim=0).detach().cpu().tolist()
    return [float(x) for x in acc], [float(x) for x in margin]


def vectorized_pair_eval_rows_for_dataset(
    dataset: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    refs: Sequence[dict[str, Any]],
    *,
    split_override: str | None = None,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    dev = device or runtime_device()
    rows: list[dict[str, Any]] = []
    all_candidates = list(candidates) + list(refs)
    for config in LAYER_CONFIGS:
        for arch in PRIMARY_ARCHES:
            cand_rows, weights = weights_for_key(all_candidates, config, arch, dev)
            if weights is None:
                continue
            x, eval_split, pair_count = dataset_pair_matrix(dataset, config, arch, split_override=split_override, device=dev)
            if x is None or pair_count <= 0:
                continue
            with torch.no_grad():
                scores = x @ weights.T
            accs, margins = accuracy_and_margin_from_scores(scores)
            for cand, acc, margin in zip(cand_rows, accs, margins):
                rows.append(
                    {
                        "dataset_name": dataset.get("dataset_name"),
                        "dataset_kind": dataset.get("dataset_kind"),
                        "readiness_eligible": bool(dataset.get("readiness_eligible")),
                        "eval_split": eval_split,
                        "candidate_name": cand.get("candidate_name"),
                        "candidate_family": cand.get("candidate_family"),
                        "source_family": cand.get("source_family"),
                        "source_run": cand.get("source_run"),
                        "tap_role": cand.get("tap_role"),
                        "recipe": cand.get("recipe"),
                        "target_config": cand.get("target_config"),
                        "architecture": cand.get("architecture"),
                        "pair_count": pair_count,
                        "pairwise_accuracy": acc,
                        "mean_margin": margin,
                    }
                )
    return rows


def vectorized_pair_rows(
    datasets: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    refs: Sequence[dict[str, Any]],
    *,
    split_override: str | None = None,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    dev = device or runtime_device()
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        out.extend(vectorized_pair_eval_rows_for_dataset(dataset, candidates, refs, split_override=split_override, device=dev))
    return out


def validation_selection_rows_cuda(
    candidates: Sequence[dict[str, Any]],
    old_datasets: Sequence[dict[str, Any]],
    branch_datasets: Sequence[dict[str, Any]],
    *,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    dev = device or runtime_device()
    old_rows = vectorized_pair_rows(old_datasets, candidates, [], split_override="val", device=dev)
    branch_rows = vectorized_pair_rows(
        [d for d in branch_datasets if d.get("readiness_eligible")],
        candidates,
        [],
        split_override="val",
        device=dev,
    )
    by_name: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict[str, Any]] = {}
    for row in old_rows:
        name = str(row.get("candidate_name") or "")
        if not name:
            continue
        by_name[name]["old"].append(safe_float(row.get("pairwise_accuracy"), float("nan")))
        meta.setdefault(name, row)
    for row in branch_rows:
        name = str(row.get("candidate_name") or "")
        if not name:
            continue
        by_name[name]["branch"].append(safe_float(row.get("pairwise_accuracy"), float("nan")))
        if any(tok in str(row.get("dataset_name") or "").lower() for tok in GAP_TOKENS):
            by_name[name]["gap"].append(safe_float(row.get("pairwise_accuracy"), float("nan")))
        meta.setdefault(name, row)
    rows: list[dict[str, Any]] = []
    for name, vals in by_name.items():
        old_accs = [x for x in vals.get("old", []) if math.isfinite(x)]
        branch_accs = [x for x in vals.get("branch", []) if math.isfinite(x)]
        gap_accs = [x for x in vals.get("gap", []) if math.isfinite(x)]
        if not old_accs and not branch_accs:
            continue
        old_min = min(old_accs) if old_accs else float("nan")
        old_mean = sum(old_accs) / len(old_accs) if old_accs else float("nan")
        branch_mean = sum(branch_accs) / len(branch_accs) if branch_accs else float("nan")
        gap_mean = sum(gap_accs) / len(gap_accs) if gap_accs else float("nan")
        gate = bool(math.isfinite(old_min) and old_min >= 0.58)
        score_parts = [a for a in [old_mean, branch_mean, gap_mean, gap_mean] if math.isfinite(a)]
        score = sum(score_parts) / len(score_parts) if score_parts else float("nan")
        if not gate and math.isfinite(score):
            score -= 0.25
        m = meta.get(name, {})
        rows.append(
            {
                "candidate_name": name,
                "candidate_family": m.get("candidate_family"),
                "tap_role": m.get("tap_role"),
                "recipe": m.get("recipe"),
                "target_config": m.get("target_config"),
                "architecture": m.get("architecture"),
                "old_val_min_accuracy": old_min,
                "old_val_mean_accuracy": old_mean,
                "branch_val_mean_accuracy": branch_mean,
                "gap_val_mean_accuracy": gap_mean,
                "old_gate_pass": gate,
                "validation_score": score,
            }
        )
    return sorted(rows, key=lambda r: safe_float(r.get("validation_score"), -1e9), reverse=True)


def validation_selection_rows(
    candidates: Sequence[dict[str, Any]],
    old_datasets: Sequence[dict[str, Any]],
    branch_datasets: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_old = [
        {**d, "pairs": [p for p in d.get("pairs") or [] if str(p.get("split") or "all") == "val"]}
        for d in old_datasets
    ]
    val_branch = [
        {**d, "pairs": [p for p in d.get("pairs") or [] if str(p.get("split") or "all") == "val"]}
        for d in branch_datasets
    ]
    val_old = [d for d in val_old if d.get("pairs")]
    val_branch = [d for d in val_branch if d.get("pairs") and d.get("readiness_eligible")]
    for cand in candidates:
        old_rows = eval_candidate_on_datasets(cand, val_old)
        branch_rows = eval_candidate_on_datasets(cand, val_branch)
        old_accs = [safe_float(r.get("accuracy"), float("nan")) for r in old_rows]
        branch_accs = [safe_float(r.get("accuracy"), float("nan")) for r in branch_rows]
        old_accs = [a for a in old_accs if math.isfinite(a)]
        branch_accs = [a for a in branch_accs if math.isfinite(a)]
        if not old_accs and not branch_accs:
            continue
        old_min = min(old_accs) if old_accs else float("nan")
        old_mean = sum(old_accs) / len(old_accs) if old_accs else float("nan")
        branch_mean = sum(branch_accs) / len(branch_accs) if branch_accs else float("nan")
        gap_accs = [
            safe_float(r.get("accuracy"), float("nan"))
            for r in branch_rows
            if any(tok in str(r.get("dataset_name") or "").lower() for tok in GAP_TOKENS)
        ]
        gap_accs = [a for a in gap_accs if math.isfinite(a)]
        gap_mean = sum(gap_accs) / len(gap_accs) if gap_accs else float("nan")
        gate = bool(math.isfinite(old_min) and old_min >= 0.58)
        score_parts = [a for a in [old_mean, branch_mean, gap_mean, gap_mean] if math.isfinite(a)]
        score = sum(score_parts) / len(score_parts) if score_parts else float("nan")
        if not gate and math.isfinite(score):
            score -= 0.25
        rows.append(
            {
                "candidate_name": candidate_name(cand),
                "candidate_family": cand.get("candidate_family"),
                "tap_role": cand.get("tap_role"),
                "recipe": cand.get("recipe"),
                "target_config": cand.get("target_config"),
                "architecture": cand.get("architecture"),
                "old_val_min_accuracy": old_min,
                "old_val_mean_accuracy": old_mean,
                "branch_val_mean_accuracy": branch_mean,
                "gap_val_mean_accuracy": gap_mean,
                "old_gate_pass": gate,
                "validation_score": score,
            }
        )
    return sorted(rows, key=lambda r: safe_float(r.get("validation_score"), -1e9), reverse=True)


def best_candidate_for_dataset(rows: Sequence[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    vals = [r for r in rows if r.get("dataset_name") == dataset_name]
    return max(vals, key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})


def compare_failures(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_rows = {
        str(r.get("dataset_name")): r
        for r in (before.get("readiness") or {}).get("branch_pair_failures") or []
    }
    out = []
    for row in (after.get("datasets") or []):
        name = str(row.get("dataset_name") or "")
        if name not in before_rows:
            continue
        prior = before_rows[name]
        out.append(
            {
                "dataset_name": name,
                "pair_count": row.get("pair_count"),
                "before_two_tap_accuracy": prior.get("best_two_tap_accuracy"),
                "after_two_tap_accuracy": row.get("best_two_tap_accuracy"),
                "reference_accuracy": row.get("best_reference_accuracy"),
                "before_gap": prior.get("delta_two_minus_reference"),
                "after_gap": row.get("delta_two_minus_reference"),
                "gap_closed": bool(row.get("matches_or_exceeds_reference")),
                "after_best_candidate": row.get("best_two_tap"),
            }
        )
    return out


def synthetic_reference_group_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in (summary.get("branch_group_summary") or {}).get("datasets") or []:
        if not row.get("best_reference"):
            continue
        rows.append(
            {
                "dataset_name": row.get("dataset_name"),
                "dataset_kind": "branch_group",
                "readiness_eligible": bool(row.get("readiness_eligible")),
                "policy_name": row.get("best_reference"),
                "candidate_name": row.get("best_reference"),
                "candidate_family": row.get("best_reference_family"),
                "source_family": row.get("best_reference_family"),
                "target_config": "reference_from_pre_repair_summary",
                "architecture": "",
                "group_count": row.get("group_count"),
                "oracle_retention": row.get("best_reference_retention"),
                "false_prune_rate": row.get("best_reference_false_prune"),
                "avg_survivors": 4.0,
                "best_selected_reward": None,
                "top1_success": None,
                "top1_reward": row.get("best_reference_top1_reward"),
                "regret": None,
            }
        )
    return rows


def synthetic_reference_pair_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, kind in (("domain_summary", "old_domain_pair"), ("branch_pair_summary", "branch_pair")):
        for row in (summary.get(section) or {}).get("datasets") or []:
            if not row.get("best_reference"):
                continue
            rows.append(
                {
                    "dataset_name": row.get("dataset_name"),
                    "dataset_kind": kind,
                    "readiness_eligible": bool(row.get("readiness_eligible")),
                    "eval_split": row.get("eval_split"),
                    "candidate_name": row.get("best_reference"),
                    "candidate_family": row.get("best_reference_family"),
                    "source_family": row.get("best_reference_family"),
                    "source_run": "pre_repair_reference_summary",
                    "tap_role": "reference",
                    "recipe": "reference_from_pre_repair_summary",
                    "target_config": "reference_from_pre_repair_summary",
                    "architecture": "",
                    "pair_count": row.get("pair_count"),
                    "pairwise_accuracy": row.get("best_reference_accuracy"),
                    "mean_margin": None,
                }
            )
    return rows


def status_from(
    verdict: str,
    ready: dict[str, Any],
    selected_eval: dict[str, Any],
    gap_comparison: Sequence[dict[str, Any]],
) -> str:
    if verdict == "LAYER_NATIVE_TWO_TAP_READY":
        return "DUALANCHOR_GAP_MECHANICALLY_CLOSED"
    closed = sum(1 for row in gap_comparison if row.get("gap_closed"))
    total = len(gap_comparison)
    selected_branch_ok = bool(selected_eval.get("branch_ok"))
    selected_domain_ok = bool(selected_eval.get("domain_ok"))
    if selected_branch_ok and selected_domain_ok:
        return "DUALANCHOR_GAP_CLOSED_BY_VAL_SELECTED_PATCH"
    if ready.get("domain_ok") and closed and closed < total:
        return "DUALANCHOR_GAP_PARTIALLY_CLOSED"
    if ready.get("domain_ok"):
        return "DUALANCHOR_GAP_NOT_CLOSED_BUT_DOMAIN_PRESERVED"
    return "DUALANCHOR_PATCH_DEGRADES_DOMAIN"


def summarize_selected_candidate(
    selected: dict[str, Any],
    refs: Sequence[dict[str, Any]],
    base_summary: dict[str, Any],
    old_datasets: Sequence[dict[str, Any]],
    branch_datasets: Sequence[dict[str, Any]],
    group_datasets: Sequence[dict[str, Any]],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if not selected:
        return {"domain_ok": False, "branch_ok": False, "candidate_name": ""}
    pair_rows = vectorized_pair_rows(list(old_datasets) + list(branch_datasets), [selected], [], device=device)
    pair_rows.extend(synthetic_reference_pair_rows(base_summary))
    domain_summary = summarize_pair_dataset(pair_rows, "old_domain_pair")
    branch_summary = summarize_pair_dataset(pair_rows, "branch_pair")
    group_summary = base_summary.get("branch_group_summary") or {"datasets": []}
    verdict, ready = readiness(domain_summary, branch_summary, group_summary)
    return {
        "candidate_name": candidate_name(selected),
        "readiness_verdict": verdict,
        "domain_ok": ready.get("domain_ok"),
        "branch_ok": ready.get("branch_ok"),
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_summary,
        "branch_group_summary": group_summary,
        "readiness": ready,
    }


def main() -> int:
    ensure_root()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = runtime_device()
    print(f"DualAnchor conservative gap patch device = {device}", flush=True)
    summary, pre_candidates, _trained, artifact_refs = load_pre_repair()
    refs = [
        r
        for r in artifact_refs
        if r.get("target_config") in LAYER_CONFIGS and r.get("architecture") in PRIMARY_ARCHES
    ]
    print(f"loaded pre-repair candidates={len(pre_candidates)} references={len(refs)}", flush=True)

    seeds = select_seed_candidates(summary, pre_candidates)
    patches, patch_rows = build_patch_candidates(seeds, refs)
    print(f"built seeds={len(seeds)} patches={len(patches)}", flush=True)
    # Keep the evaluation focused. The full pre-repair candidate set is large;
    # this probe only needs the prior summary seeds plus the new 24/36 patches.
    candidates = list(seeds) + patches
    bundles: list[dict[str, Any]] = []
    old_datasets = old_domain_pair_datasets()
    branch_datasets = branch_pair_datasets()
    group_datasets: list[dict[str, Any]] = []

    print("scoring pair datasets on cuda", flush=True)
    pair_rows = vectorized_pair_rows(list(old_datasets) + list(branch_datasets), candidates, [], device=device)
    pair_rows.extend(synthetic_reference_pair_rows(summary))
    domain_summary = summarize_pair_dataset(pair_rows, "old_domain_pair")
    branch_summary = summarize_pair_dataset(pair_rows, "branch_pair")
    group_rows = synthetic_reference_group_rows(summary)
    group_summary = summary.get("branch_group_summary") or {"datasets": []}
    verdict, ready = readiness(domain_summary, branch_summary, group_summary)

    print("selecting validation patch on cuda", flush=True)
    validation_rows = validation_selection_rows_cuda(patches, old_datasets, branch_datasets, device=device)
    by_name = candidate_lookup(patches)
    selected_name = next(
        (
            str(row.get("candidate_name"))
            for row in validation_rows
            if row.get("old_gate_pass") and math.isfinite(safe_float(row.get("validation_score"), float("nan")))
        ),
        "",
    )
    selected = by_name.get(selected_name, {})
    selected_eval = summarize_selected_candidate(selected, refs, summary, old_datasets, branch_datasets, group_datasets, device=device)
    gap_comparison = compare_failures(summary, branch_summary)
    status = status_from(verdict, ready, selected_eval, gap_comparison)

    payload = {
        "BG_DUALANCHOR_CONSERVATIVE_GAP_PATCH_VERDICT": status,
        "BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT": verdict,
        "status": status,
        "readiness_verdict": verdict,
        "seed_candidate_count": len(seeds),
        "patch_candidate_count": len(patches),
        "total_candidate_count": len(candidates),
        "bundle_count": len(bundles),
        "branch_primary_configs": list(BRANCH_PRIMARY_CONFIGS),
        "late_diagnostic_configs": list(LATE_DIAGNOSTIC_CONFIGS),
        "runtime_device": str(device),
        "selected_validation_patch": selected_name,
        "selected_validation_patch_eval": selected_eval,
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_summary,
        "branch_group_summary": group_summary,
        "readiness": ready,
        "gap_comparison": gap_comparison,
        "anti_leakage": {
            "pre_repair_constrained_artifacts_only": True,
            "patches_are_tiny_residual_additions": True,
            "new_patches_limited_to_24_36": True,
            "layer_47_retained_as_late_diagnostic_only": True,
            "validation_selected_patch_reported_separately": True,
            "family_best_heldout_is_diagnostic_not_locked_policy": True,
            "no_ouro_training": True,
            "no_old_tap_registry_update": True,
            "no_action_steering": True,
            "no_production_routing_change": True,
        },
    }
    torch.save(
        {
            "summary": payload,
            "seeds": seeds,
            "patches": patches,
            "candidates": candidates,
            "references": refs,
            "bundles": bundles,
            "selected_validation_patch": selected,
        },
        ARTIFACT_PT,
    )
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(PAIR_ROWS_CSV, [clean_row(r) for r in pair_rows])
    write_csv(GROUP_ROWS_CSV, [clean_row(r) for r in group_rows])
    write_csv(PATCH_ROWS_CSV, patch_rows)
    write_csv(VALIDATION_ROWS_CSV, validation_rows)

    lines = [
        "# DualAnchor Conservative Gap Patch v1",
        "",
        f"BG_DUALANCHOR_CONSERVATIVE_GAP_PATCH_VERDICT = {status}",
        f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
        "",
        "This probe starts from the pre-repair constrained DualAnchor candidates and adds tiny same-config branch/bridge residual patches. It is diagnostic unless the validation-selected patch also clears readiness.",
        "",
        f"- seed candidates: `{len(seeds)}`",
        f"- patch candidates: `{len(patches)}`",
        f"- total candidates: `{len(candidates)}`",
        f"- branch-primary patch layers: `{', '.join(BRANCH_PRIMARY_CONFIGS)}`",
        f"- late diagnostic only: `{', '.join(LATE_DIAGNOSTIC_CONFIGS)}`",
        f"- selected validation patch: `{selected_name or 'none'}`",
        f"- family-best domain ok: `{ready.get('domain_ok')}`",
        f"- family-best branch ok: `{ready.get('branch_ok')}`",
        f"- validation-selected patch domain ok: `{selected_eval.get('domain_ok')}`",
        f"- validation-selected patch branch ok: `{selected_eval.get('branch_ok')}`",
        "",
        "## Gap Comparison",
        "",
    ]
    lines.extend(
        md_table(
            gap_comparison,
            [
                "dataset_name",
                "pair_count",
                "before_two_tap_accuracy",
                "after_two_tap_accuracy",
                "reference_accuracy",
                "before_gap",
                "after_gap",
                "gap_closed",
                "after_best_candidate",
            ],
        )
    )
    lines.extend(["", "## Domain Summary", ""])
    lines.extend(
        md_table(
            domain_summary.get("datasets", []),
            [
                "dataset_name",
                "pair_count",
                "best_two_tap_accuracy",
                "best_reference_accuracy",
                "delta_two_minus_reference",
                "matches_or_exceeds_reference",
                "best_two_tap",
            ],
        )
    )
    lines.extend(["", "## Branch Pair Summary", ""])
    lines.extend(
        md_table(
            branch_summary.get("datasets", []),
            [
                "dataset_name",
                "readiness_eligible",
                "pair_count",
                "best_two_tap_accuracy",
                "best_reference_accuracy",
                "delta_two_minus_reference",
                "matches_or_exceeds_reference",
                "best_two_tap",
            ],
        )
    )
    lines.extend(["", "## Branch Group Summary", ""])
    lines.extend(
        md_table(
            group_summary.get("datasets", []),
            [
                "dataset_name",
                "group_count",
                "best_two_tap_retention",
                "best_reference_retention",
                "delta_two_minus_reference",
                "matches_or_exceeds_reference",
                "best_two_tap",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `family-best` rows show whether the remaining gaps are mechanically closable by some DualAnchor-slot patch.",
            "- The validation-selected patch is reported separately and is the only non-heldout-selected candidate in this run.",
            "- No registry, checkpoint, tokenizer, routing, wrapper/local-agent, Hunter-Seeker, or steering code was touched.",
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- pair rows: `{rel(PAIR_ROWS_CSV)}`",
            f"- group rows: `{rel(GROUP_ROWS_CSV)}`",
            f"- patch rows: `{rel(PATCH_ROWS_CSV)}`",
            f"- validation rows: `{rel(VALIDATION_ROWS_CSV)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)

    print(f"BG_DUALANCHOR_CONSERVATIVE_GAP_PATCH_VERDICT = {status}", flush=True)
    print(f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}", flush=True)
    print(f"family_best_domain_ok = {ready.get('domain_ok')} family_best_branch_ok = {ready.get('branch_ok')}", flush=True)
    print(f"selected_validation_patch = {selected_name or 'none'}", flush=True)
    print(
        f"selected_patch_domain_ok = {selected_eval.get('domain_ok')} selected_patch_branch_ok = {selected_eval.get('branch_ok')}",
        flush=True,
    )
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
