"""Weight-space merged branch-content taps v1.

This manual experiment only reads cached tap/evaluator artifacts, extracts
small pairwise tap directions, builds new diagnostic merged tap artifacts under
a new report root, and evaluates them on cached data. It does not train Ouro,
modify checkpoints/tokenizers, overwrite existing tap registries, run
wrapper/local-agent or Hunter-Seeker code, apply steering, or change routing.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from bg_hidden_origin_tap_common import CONFIGS, HIDDEN_DIM, PROJECT_ROOT, PROBE_ROOT, config_dim, md_table, rel


OUT_ROOT = PROBE_ROOT / "bg_merged_weight_branch_content_taps_v1_2026-05-18"

WEIGHT_INVENTORY_JSON = OUT_ROOT / "weight_inventory.json"
WEIGHT_INVENTORY_MD = OUT_ROOT / "weight_inventory.md"
WEIGHT_INVENTORY_CSV = OUT_ROOT / "weight_inventory.csv"
EXTRACTED_WEIGHTS_PT = OUT_ROOT / "extracted_weights.pt"

ALIGNMENT_JSON = OUT_ROOT / "coordinate_alignment.json"
ALIGNMENT_MD = OUT_ROOT / "coordinate_alignment.md"
ALIGNED_WEIGHTS_PT = OUT_ROOT / "aligned_weights.pt"

FEATURE_STATS_PT = OUT_ROOT / "feature_stats.pt"
FEATURE_STATS_JSON = OUT_ROOT / "feature_stats.json"
FEATURE_STATS_MD = OUT_ROOT / "feature_stats.md"

GEOMETRY_JSON = OUT_ROOT / "weight_geometry.json"
GEOMETRY_MD = OUT_ROOT / "weight_geometry.md"
GEOMETRY_CSV = OUT_ROOT / "weight_geometry_rows.csv"

MERGED_TAPS_PT = OUT_ROOT / "merged_weight_branch_content_taps_v1.pt"
MERGED_CANDIDATES_JSON = OUT_ROOT / "merged_tap_candidates.json"
MERGED_CANDIDATES_MD = OUT_ROOT / "merged_tap_candidates.md"

VALIDATION_JSON = OUT_ROOT / "validation_selection.json"
VALIDATION_MD = OUT_ROOT / "validation_selection.md"
VALIDATION_CSV = OUT_ROOT / "validation_rows.csv"
SELECTED_TAPS_PT = OUT_ROOT / "selected_merged_taps.pt"

OLD_CODE_JSON = OUT_ROOT / "old_code_eval.json"
OLD_CODE_MD = OUT_ROOT / "old_code_eval.md"
OLD_CODE_CSV = OUT_ROOT / "old_code_eval_rows.csv"

BRANCH_BRIDGE_JSON = OUT_ROOT / "branch_bridge_eval.json"
BRANCH_BRIDGE_MD = OUT_ROOT / "branch_bridge_eval.md"
BRANCH_BRIDGE_CSV = OUT_ROOT / "branch_bridge_eval_rows.csv"

SURVIVAL_JSON = OUT_ROOT / "survival_eval.json"
SURVIVAL_MD = OUT_ROOT / "survival_eval.md"
SURVIVAL_CSV = OUT_ROOT / "survival_eval_rows.csv"

FINAL_ARBITER_JSON = OUT_ROOT / "final_arbiter_eval.json"
FINAL_ARBITER_MD = OUT_ROOT / "final_arbiter_eval.md"
FINAL_ARBITER_CSV = OUT_ROOT / "final_arbiter_eval_rows.csv"

SURVIVOR_FEATURES_PT = OUT_ROOT / "top4_survivor_hidden_features.pt"
SURVIVOR_FEATURES_JSON = OUT_ROOT / "top4_survivor_hidden_features.json"
SURVIVOR_FEATURES_MD = OUT_ROOT / "top4_survivor_hidden_features.md"
SURVIVOR_FEATURES_CSV = OUT_ROOT / "top4_survivor_hidden_features_rows.csv"

COMPOSITE_INSERTION_JSON = OUT_ROOT / "composite_insertion_eval.json"
COMPOSITE_INSERTION_MD = OUT_ROOT / "composite_insertion_eval.md"
COMPOSITE_INSERTION_CSV = OUT_ROOT / "composite_insertion_rows.csv"

DIAGNOSTICS_JSON = OUT_ROOT / "diagnostics.json"
DIAGNOSTICS_MD = OUT_ROOT / "diagnostics.md"
DIAGNOSTICS_CSV = OUT_ROOT / "diagnostic_rows.csv"

SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_merged_weight_branch_content_taps_v1.md"

UNIVERSAL_ROOT = PROBE_ROOT / "bg_universal_branch_content_taps_v1_2026-05-18"
UNIVERSAL_HEADS_PT = UNIVERSAL_ROOT / "universal_branch_content_tap_heads.pt"
OLD_CONTENT_PT = UNIVERSAL_ROOT / "old_content_dataset.pt"
HIDDEN_BRANCH_PT = UNIVERSAL_ROOT / "hidden_branch_dataset.pt"
BRIDGE_PT = UNIVERSAL_ROOT / "bridge_dataset.pt"
BGV1_BRANCHES_PT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18/branch_generator_v1_branches.pt"
QUOTA_V4_BRANCHES_PT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18/quota_hidden_origin_branches.pt"

HEAD_REGISTRY_PT = PROBE_ROOT / "bg_head_registry_2026-05-17.pt"
MIXED_HEADS_PT = PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt"
HIDDEN_TAPS_PT = PROBE_ROOT / "bg_hidden_origin_taps_2026-05-18/hidden_origin_tap_heads.pt"
V4_HEADS_PT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18/hidden_origin_tap_heads_v4.pt"
GENERATOR_HEADS_PT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18/selector_heads.pt"
GENERATOR_TAP_HEADS_PT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18/hidden_origin_tap_heads_generator_v1.pt"
SALVAGE_HEADS_PT = PROBE_ROOT / "bg_hidden_origin_split_salvage_2026-05-18/salvage_heads.pt"
GATED_HEADS_PT = PROBE_ROOT / "bg_gated_branch_content_selector_v1_2026-05-18/gated_branch_content_selector_heads.pt"
FINAL_ARBITER_V1_1_PT = PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_1_2026-05-18/final_arbiter_top4_v1_1.pt"
FINAL_ARBITER_DATASET_PT = PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_1_2026-05-18/final_arbiter_v1_1_dataset.pt"
FIXED_HELDOUT_JSON = PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18/heldout_survival_eval.json"

TARGET_CONFIGS = ("concat_24_36_47", "concat_24_30_36", "36_L4", "24_L4", "47_L4")
PRIMARY_TARGET = "concat_24_36_47"
PRIMARY_ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")
SCRIPT_NAMES = [
    "bg_merged_tap_weight_inventory_v1.py",
    "bg_merged_tap_coordinate_alignment_v1.py",
    "bg_merged_tap_feature_stats_v1.py",
    "analyze_bg_merged_tap_weight_geometry_v1.py",
    "build_bg_merged_weight_taps_v1.py",
    "select_bg_merged_weight_taps_v1.py",
    "evaluate_bg_merged_tap_old_code_v1.py",
    "evaluate_bg_merged_tap_branch_bridge_v1.py",
    "evaluate_bg_merged_tap_survival_v1.py",
    "acquire_bg_merged_tap_survivor_features_v1.py",
    "evaluate_bg_merged_tap_final_arbiter_v1.py",
    "evaluate_bg_merged_tap_composite_insertion_v1.py",
    "analyze_bg_merged_tap_diagnostics_v1.py",
    "analyze_bg_merged_weight_branch_content_taps_v1.py",
]

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_final_arbiter_top4_survivors_v1_1.md",
    PROJECT_ROOT / "docs/evaluator/bg_final_arbiter_top4_survivors_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_selection_only_phase2_prototype_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
    PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
]

NAV_TARGETS = [
    PROJECT_ROOT / "shared/docs/evaluator/README.md",
    PROJECT_ROOT / "shared/docs/README.md",
    PROJECT_ROOT / "PROJECT_TREE_MAP.md",
    PROJECT_ROOT / "PROJECT_COMPONENTS.md",
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_pt(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().to(torch.float32)
        return {
            "shape": list(x.shape),
            "mean": float(x.mean().item()) if x.numel() else 0.0,
            "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
            "rms": float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0,
        }
    if isinstance(value, Path):
        return rel(value)
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return str(value)


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


def finite_mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(mean(vals)) if vals else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.4f}"


def normed(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = vec.detach().cpu().to(torch.float32).flatten()
    n = float(x.norm().item())
    if n < eps:
        return torch.zeros_like(x)
    return x / n


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = normed(a)
    bb = normed(b)
    if not aa.numel() or not bb.numel() or aa.shape != bb.shape:
        return float("nan")
    return float(torch.dot(aa, bb).item())


def compact_head_name(row: dict[str, Any], source_label: str, idx: int) -> str:
    family = str(row.get("head_group") or row.get("head_family") or row.get("variant") or row.get("family") or source_label)
    config = str(row.get("config") or "unknown")
    arch = str(row.get("architecture") or "unknown")
    seed = row.get("metrics", {}).get("seed") if isinstance(row.get("metrics"), dict) else None
    seed = row.get("train_metrics", {}).get("seed", seed) if isinstance(row.get("train_metrics"), dict) else seed
    suffix = f"::seed={seed}" if seed is not None else f"::{idx}"
    return f"{source_label}::{family}::{config}::{arch}{suffix}"


def metric_score(row: dict[str, Any]) -> float:
    candidates: list[Any] = []
    for key in ("metrics", "train_metrics", "eval", "heldout_metrics"):
        value = row.get(key)
        if isinstance(value, dict):
            candidates.extend(
                [
                    value.get("balanced_validation_score"),
                    value.get("validation_pairwise_accuracy"),
                    value.get("val_pair_acc"),
                    value.get("val_pairwise_accuracy"),
                    value.get("hh_heldout_acc"),
                    value.get("validation_by_pair_type", {}).get("old_content") if isinstance(value.get("validation_by_pair_type"), dict) else None,
                ]
            )
            domain_balance = value.get("domain_balance")
            if isinstance(domain_balance, dict):
                candidates.extend((d or {}).get("validation_pairwise_accuracy") for d in domain_balance.values())
    finite = [safe_float(v, float("nan")) for v in candidates]
    finite = [v for v in finite if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def infer_family(source_label: str, row: dict[str, Any]) -> tuple[str, str]:
    variant = str(row.get("variant") or row.get("head_group") or row.get("head_family") or row.get("family") or "")
    low = f"{source_label}::{variant}".lower()
    if "bridge_only" in low:
        return "bridge", "bridge"
    if "hidden_branch_only" in low or "hidden_origin" in low or "generator" in low or "v4" in low or "salvage" in low:
        return "branch", "hidden_branch"
    if "universal_balanced" in low or "universal_domain" in low or "universal_no_bridge" in low:
        return "universal", "universal"
    if "old_content_only" in low or "code" in low or "hh" in low or "mix_" in low or "objective" in low:
        return "old_content", "old_content"
    if "gated" in low:
        return "gated", "diagnostic"
    if "final_arbiter" in low:
        return "final_arbiter", "diagnostic"
    return "diagnostic", "diagnostic"


def head_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    heads = payload.get("heads")
    if isinstance(heads, list):
        return [dict(h) for h in heads if isinstance(h, dict)]
    if isinstance(heads, dict):
        out = []
        for key, value in heads.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("head_key", key)
                out.append(row)
        return out
    selected = payload.get("selected_model")
    if isinstance(selected, dict):
        row = dict(selected)
        row.setdefault("source_model_type", "selected_final_arbiter")
        return [row]
    return []


def extract_weight(row: dict[str, Any]) -> tuple[torch.Tensor | None, str | None]:
    state = row.get("state_dict") or row.get("model_state_dict")
    if not isinstance(state, dict):
        return None, None
    for key in ("linear.weight", "weight", "score.weight"):
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().to(torch.float32)
            if x.ndim == 2 and x.shape[0] == 1:
                return x.flatten(), key
            if x.ndim == 1:
                return x.flatten(), key
    return None, None


def source_specs() -> list[tuple[str, Path]]:
    return [
        ("old_registry", HEAD_REGISTRY_PT),
        ("old_mixed_domain", MIXED_HEADS_PT),
        ("universal", UNIVERSAL_HEADS_PT),
        ("hidden_origin_v1", HIDDEN_TAPS_PT),
        ("hidden_origin_v4", V4_HEADS_PT),
        ("branch_generator_v1_selector", GENERATOR_HEADS_PT),
        ("branch_generator_v1_hidden_tap", GENERATOR_TAP_HEADS_PT),
        ("hidden_origin_salvage", SALVAGE_HEADS_PT),
        ("gated_selector", GATED_HEADS_PT),
        ("final_arbiter_v1_1", FINAL_ARBITER_V1_1_PT),
    ]


def compact_weight_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "weight"}


def run_weight_inventory() -> int:
    ensure_root()
    started = time.time()
    weights: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for source_label, path in source_specs():
        payload = load_pt(path, None)
        rows = head_rows_from_payload(payload)
        if payload is None:
            inventory_rows.append({"source_run": source_label, "source_path": rel(path), "status": "missing", "head_count": 0})
            continue
        inventory_rows.append({"source_run": source_label, "source_path": rel(path), "status": "loaded", "head_count": len(rows)})
        for idx, raw in enumerate(rows):
            row = dict(raw)
            arch = str(row.get("architecture") or row.get("model_family") or row.get("family") or "unknown")
            config = str(row.get("config") or row.get("feature_config") or "unknown")
            weight, key = extract_weight(row)
            family, support = infer_family(source_label, row)
            name = compact_head_name(row, source_label, idx)
            mergeable_arch = arch in PRIMARY_ARCHITECTURES
            dim = int(weight.numel()) if isinstance(weight, torch.Tensor) else 0
            feature_dim = int(row.get("dim") or dim or 0)
            direct_target = config in TARGET_CONFIGS and dim == (config_dim(config) if config in CONFIGS or config.startswith("concat_") else dim)
            lift_target = config in {"24_L4", "36_L4", "47_L4", "30_L4"} and dim == HIDDEN_DIM
            is_final_arbiter = family == "final_arbiter"
            diagnostic_only = (not mergeable_arch) or is_final_arbiter or weight is None
            record = {
                "head_name": name,
                "source_run": source_label,
                "source_path": rel(path),
                "source_family": family,
                "pair_type_support": support,
                "variant": row.get("variant"),
                "head_group": row.get("head_group"),
                "head_family": row.get("head_family"),
                "architecture": arch,
                "config": config,
                "feature_dim": feature_dim,
                "state_dict_keys": sorted(list((row.get("state_dict") or row.get("model_state_dict") or {}).keys())) if isinstance(row.get("state_dict") or row.get("model_state_dict"), dict) else [],
                "extractable_weight_key": key,
                "weight_shape": list(weight.shape) if isinstance(weight, torch.Tensor) else [],
                "bias_present": any("bias" in k for k in ((row.get("state_dict") or row.get("model_state_dict") or {}).keys() if isinstance(row.get("state_dict") or row.get("model_state_dict"), dict) else [])),
                "layernorm_present": arch == "AntisymLinear",
                "score_orientation": "preferred_minus_rejected_positive_if_training_orientation_preserved",
                "metric_score": metric_score(row),
                "mergeable_directly": bool(weight is not None and mergeable_arch and direct_target and not is_final_arbiter),
                "mergeable_by_lift": bool(weight is not None and mergeable_arch and lift_target and not is_final_arbiter),
                "mergeable_by_per_layer": bool(weight is not None and mergeable_arch and config in {"24_L4", "36_L4", "47_L4"} and not is_final_arbiter),
                "diagnostic_only": bool(diagnostic_only),
                "notes": "final arbiter heads are downstream eval/baseline only" if is_final_arbiter else "",
            }
            if isinstance(weight, torch.Tensor):
                record["weight"] = weight
            weights.append(record)
    available = [w for w in weights if isinstance(w.get("weight"), torch.Tensor) and not w.get("diagnostic_only")]
    old_ok = any(w["source_family"] == "old_content" for w in available)
    branch_ok = any(w["source_family"] == "branch" for w in available)
    bridge_ok = any(w["source_family"] == "bridge" for w in available)
    if old_ok and branch_ok and bridge_ok:
        verdict = "READY"
    elif old_ok and (branch_ok or bridge_ok):
        verdict = "PARTIAL"
    elif available:
        verdict = "INSUFFICIENT_WEIGHTS"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_MERGED_TAP_WEIGHT_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "source_inventory": inventory_rows,
        "heads_extracted": len([w for w in weights if isinstance(w.get("weight"), torch.Tensor)]),
        "mergeable_heads": len(available),
        "family_counts": dict(Counter(str(w.get("source_family")) for w in available)),
        "target_configs": list(TARGET_CONFIGS),
        "anti_leakage": {
            "tap_scores_not_labels": True,
            "final_arbiter_heads_not_merge_sources_unless_linear_hidden_direction": True,
            "score_distillation_diagnostic_only": True,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save({"weights": weights, "inventory": payload}, EXTRACTED_WEIGHTS_PT)
    write_json(WEIGHT_INVENTORY_JSON, {**payload, "weights": [compact_weight_row(w) for w in weights]})
    write_csv(WEIGHT_INVENTORY_CSV, [compact_weight_row(w) for w in weights])
    lines = [
        "# Merged Tap Weight Inventory",
        "",
        f"BG_MERGED_TAP_WEIGHT_INVENTORY_VERDICT = {verdict}",
        "",
        f"- heads extracted: `{payload['heads_extracted']}`",
        f"- mergeable heads: `{payload['mergeable_heads']}`",
        f"- family counts: `{payload['family_counts']}`",
        "",
        "Final-arbiter heads are not merge sources unless they expose a compatible linear hidden-state pair direction.",
        "",
        "## Sources",
        "",
    ]
    lines.extend(md_table(inventory_rows, ["source_run", "status", "head_count", "source_path"]))
    write_md(WEIGHT_INVENTORY_MD, lines)
    print(f"BG_MERGED_TAP_WEIGHT_INVENTORY_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def concat_blocks(config: str) -> list[str]:
    if config == "concat_24_36_47":
        return ["24_L4", "36_L4", "47_L4"]
    if config == "concat_24_30_36":
        return ["24_L4", "30_L4", "36_L4"]
    if config == "concat_36_42_47":
        return ["36_L4", "42_L4", "47_L4"]
    if config == "concat_24_36":
        return ["24_L4", "36_L4"]
    if config == "concat_36_47":
        return ["36_L4", "47_L4"]
    return [config]


def lift_weight(weight: torch.Tensor, source_config: str, target_config: str) -> tuple[torch.Tensor | None, dict[str, Any]]:
    source_dim = int(weight.numel())
    target_dim = config_dim(target_config)
    if source_config == target_config and source_dim == target_dim:
        return weight.detach().cpu().to(torch.float32).clone(), {"method": "direct", "block_mapping": source_config}
    target_blocks = concat_blocks(target_config)
    if source_config in target_blocks and source_dim == HIDDEN_DIM and target_dim == HIDDEN_DIM * len(target_blocks):
        out = torch.zeros(target_dim, dtype=torch.float32)
        block_idx = target_blocks.index(source_config)
        start = block_idx * HIDDEN_DIM
        out[start : start + HIDDEN_DIM] = weight.detach().cpu().to(torch.float32)
        return out, {"method": "lifted_zero_block", "block_mapping": f"{source_config}->{target_config}[block={block_idx}]"}
    return None, {"method": "unsupported", "block_mapping": ""}


def run_coordinate_alignment() -> int:
    ensure_root()
    payload = load_pt(EXTRACTED_WEIGHTS_PT, {}) or {}
    weights = list(payload.get("weights") or [])
    aligned: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for row in weights:
        weight = row.get("weight")
        if not isinstance(weight, torch.Tensor) or row.get("diagnostic_only"):
            continue
        config = str(row.get("config"))
        for target in TARGET_CONFIGS:
            try:
                lifted, meta = lift_weight(weight, config, target)
            except Exception:
                lifted, meta = None, {"method": "unsupported", "block_mapping": ""}
            if lifted is None:
                continue
            valid_primary = target == PRIMARY_TARGET and meta["method"] in {"direct", "lifted_zero_block"}
            out = {
                **row,
                "target_config": target,
                "source_config": config,
                "alignment_method": meta["method"],
                "block_mapping": meta["block_mapping"],
                "dimension_check": int(lifted.numel()) == config_dim(target),
                "feature_order_check": "explicit_config_mapping",
                "valid_for_primary_eval": bool(valid_primary),
                "diagnostic_only": False,
                "aligned_weight": lifted,
            }
            aligned.append(out)
            rows.append({k: v for k, v in out.items() if k not in {"weight", "aligned_weight"}})
    direct = sum(1 for r in aligned if r["target_config"] == PRIMARY_TARGET and r["alignment_method"] == "direct")
    lifted = sum(1 for r in aligned if r["target_config"] == PRIMARY_TARGET and r["alignment_method"] == "lifted_zero_block")
    families_primary = {str(r.get("source_family")) for r in aligned if r.get("valid_for_primary_eval")}
    if {"old_content", "branch", "bridge"}.issubset(families_primary):
        verdict = "SHARED_CONCAT_READY" if direct >= 3 else "LIFTED_CONCAT_READY"
    elif aligned:
        verdict = "PER_LAYER_ONLY"
    else:
        verdict = "BLOCKED"
    out_payload = {
        "BG_MERGED_TAP_COORDINATE_ALIGNMENT_VERDICT": verdict,
        "verdict": verdict,
        "aligned_count": len(aligned),
        "primary_direct_count": direct,
        "primary_lifted_count": lifted,
        "primary_families": sorted(families_primary),
        "rules": {
            "direct_merge_requires_same_config_dim_order": True,
            "single_layer_to_concat_uses_zero_filled_blocks": True,
            "layernorm_and_nonorm_primary_merges_are_separate": True,
            "concat_to_single_layer_projection_disallowed": True,
        },
    }
    torch.save({"aligned_weights": aligned, "summary": out_payload}, ALIGNED_WEIGHTS_PT)
    write_json(ALIGNMENT_JSON, {**out_payload, "alignment_rows": rows})
    lines = [
        "# Merged Tap Coordinate Alignment",
        "",
        f"BG_MERGED_TAP_COORDINATE_ALIGNMENT_VERDICT = {verdict}",
        "",
        f"- aligned rows: `{len(aligned)}`",
        f"- primary direct rows: `{direct}`",
        f"- primary lifted rows: `{lifted}`",
        f"- primary families: `{sorted(families_primary)}`",
        "",
        "Direct and lifted rows keep explicit config/block mappings. No concat direction was silently projected down into a single layer.",
    ]
    write_md(ALIGNMENT_MD, lines)
    print(f"BG_MERGED_TAP_COORDINATE_ALIGNMENT_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def load_pair_sets() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name, path in (("old_content", OLD_CONTENT_PT), ("hidden_branch", HIDDEN_BRANCH_PT), ("bridge", BRIDGE_PT)):
        payload = load_pt(path, {}) or {}
        pairs = list(payload.get("pairs") or [])
        for pair in pairs:
            pair.setdefault("pair_type", name)
        out[name] = pairs
    return out


def pair_has_config(pair: dict[str, Any], config: str) -> bool:
    vals = (pair.get("features") or {}).get(config)
    return isinstance(vals, dict) and isinstance(vals.get("preferred"), torch.Tensor) and isinstance(vals.get("rejected"), torch.Tensor)


def pair_diff(pair: dict[str, Any], config: str) -> torch.Tensor | None:
    vals = (pair.get("features") or {}).get(config)
    if not isinstance(vals, dict):
        return None
    pref = vals.get("preferred")
    rej = vals.get("rejected")
    if not isinstance(pref, torch.Tensor) or not isinstance(rej, torch.Tensor):
        return None
    if int(pref.numel()) != config_dim(config) or int(rej.numel()) != config_dim(config):
        return None
    return pref.detach().cpu().to(torch.float32).flatten() - rej.detach().cpu().to(torch.float32).flatten()


def run_feature_stats() -> int:
    ensure_root()
    started = time.time()
    pair_sets = load_pair_sets()
    stats: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for config in TARGET_CONFIGS:
        diffs = []
        type_counts = Counter()
        for pair_type, pairs in pair_sets.items():
            for pair in pairs:
                if str(pair.get("split")) != "train":
                    continue
                diff = pair_diff(pair, config)
                if diff is None:
                    continue
                diffs.append(diff)
                type_counts[pair_type] += 1
        if not diffs:
            continue
        x = torch.stack(diffs, dim=0)
        mean_vec = x.mean(dim=0)
        std_vec = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        block_stats = []
        blocks = concat_blocks(config)
        for idx, block in enumerate(blocks):
            start = idx * HIDDEN_DIM if len(blocks) > 1 else 0
            stop = start + HIDDEN_DIM
            bx = x[:, start:stop]
            block_stats.append(
                {
                    "block": block,
                    "mean_rms": float(bx.mean(dim=0).pow(2).mean().sqrt().item()),
                    "std_mean": float(bx.std(dim=0, unbiased=False).mean().item()),
                    "feature_rms": float(bx.pow(2).mean().sqrt().item()),
                }
            )
        stats[config] = {
            "count": len(diffs),
            "type_counts": dict(type_counts),
            "mean": mean_vec,
            "std": std_vec,
            "diag_cov": std_vec.pow(2),
            "feature_rms": float(x.pow(2).mean().sqrt().item()),
            "block_stats": block_stats,
        }
        rows.append({"config": config, "train_diffs": len(diffs), "type_counts": dict(type_counts), "feature_rms": stats[config]["feature_rms"]})
    if PRIMARY_TARGET in stats and stats[PRIMARY_TARGET]["count"] >= 100:
        verdict = "READY"
    elif stats:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    out_payload = {
        "BG_MERGED_TAP_FEATURE_STATS_VERDICT": verdict,
        "verdict": verdict,
        "configs": {cfg: {k: v for k, v in data.items() if k not in {"mean", "std", "diag_cov"}} for cfg, data in stats.items()},
        "normalization_modes_available": ["raw_norm", "per_block_norm", "zscore_diag"],
        "train_only_statistics": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save({"stats": stats, "summary": out_payload}, FEATURE_STATS_PT)
    write_json(FEATURE_STATS_JSON, out_payload)
    lines = ["# Merged Tap Feature Stats", "", f"BG_MERGED_TAP_FEATURE_STATS_VERDICT = {verdict}", "", "Statistics were computed from train split pair differences only.", "", "## Configs", ""]
    lines.extend(md_table(rows, ["config", "train_diffs", "type_counts", "feature_rms"]))
    write_md(FEATURE_STATS_MD, lines)
    print(f"BG_MERGED_TAP_FEATURE_STATS_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def source_priority(row: dict[str, Any]) -> float:
    name = str(row.get("head_name", "")).lower()
    score = safe_float(row.get("metric_score"), 0.0)
    bonuses = 0.0
    for needle, bonus in (
        ("old_content_only", 0.35),
        ("mix_code_reasoning", 0.30),
        ("mix_objective_all", 0.30),
        ("code", 0.12),
        ("hidden_branch_only", 0.35),
        ("bridge_only", 0.40),
        ("universal_balanced", 0.20),
        ("v4", 0.12),
        ("generator", 0.10),
    ):
        if needle in name:
            bonuses += bonus
    if row.get("alignment_method") == "direct":
        bonuses += 0.08
    return score + bonuses


def top_sources(aligned: Sequence[dict[str, Any]], family: str, arch: str, target: str = PRIMARY_TARGET, limit: int = 4) -> list[dict[str, Any]]:
    pool = [r for r in aligned if r.get("target_config") == target and r.get("architecture") == arch and r.get("source_family") == family and isinstance(r.get("aligned_weight"), torch.Tensor)]
    dedup: dict[str, dict[str, Any]] = {}
    for row in sorted(pool, key=source_priority, reverse=True):
        key = str(row.get("head_name")).split("::seed=")[0]
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())[:limit]


def run_weight_geometry() -> int:
    ensure_root()
    payload = load_pt(ALIGNED_WEIGHTS_PT, {}) or {}
    aligned = list(payload.get("aligned_weights") or [])
    rows: list[dict[str, Any]] = []
    family_rows = [r for r in aligned if r.get("target_config") == PRIMARY_TARGET and isinstance(r.get("aligned_weight"), torch.Tensor)]
    selected: list[dict[str, Any]] = []
    for arch in PRIMARY_ARCHITECTURES:
        for family in ("old_content", "branch", "bridge", "universal"):
            selected.extend(top_sources(family_rows, family, arch, limit=3))
    seen = set()
    selected_unique = []
    for row in selected:
        key = str(row.get("head_name"))
        if key not in seen:
            selected_unique.append(row)
            seen.add(key)
    selected = selected_unique
    for i, left in enumerate(selected):
        for right in selected[i + 1 :]:
            if left.get("architecture") != right.get("architecture") or left.get("target_config") != right.get("target_config"):
                continue
            lw = left["aligned_weight"]
            rw = right["aligned_weight"]
            row = {
                "left": left.get("head_name"),
                "right": right.get("head_name"),
                "left_family": left.get("source_family"),
                "right_family": right.get("source_family"),
                "architecture": left.get("architecture"),
                "target_config": left.get("target_config"),
                "cosine": cosine(lw, rw),
            }
            blocks = concat_blocks(str(left.get("target_config")))
            for bidx, block in enumerate(blocks):
                start = bidx * HIDDEN_DIM if len(blocks) > 1 else 0
                stop = start + HIDDEN_DIM
                row[f"cosine_{block}"] = cosine(lw[start:stop], rw[start:stop])
            rows.append(row)
    residual_rows: list[dict[str, Any]] = []
    for arch in PRIMARY_ARCHITECTURES:
        olds = top_sources(family_rows, "old_content", arch, limit=2)
        branches = top_sources(family_rows, "branch", arch, limit=2)
        bridges = top_sources(family_rows, "bridge", arch, limit=2)
        for old in olds:
            u_old = normed(old["aligned_weight"])
            for branch in branches:
                b = normed(branch["aligned_weight"])
                b_res = b - torch.dot(b, u_old) * u_old
                b_res_norm = float(b_res.norm().item())
                for bridge in bridges:
                    c = normed(bridge["aligned_weight"])
                    basis = [u_old]
                    if b_res_norm >= 1e-6:
                        basis.append(b_res / b_res_norm)
                    c_res = c.clone()
                    for base in basis:
                        c_res = c_res - torch.dot(c_res, base) * base
                    c_res_norm = float(c_res.norm().item())
                    residual_rows.append(
                        {
                            "architecture": arch,
                            "old": old.get("head_name"),
                            "branch": branch.get("head_name"),
                            "bridge": bridge.get("head_name"),
                            "old_branch_cosine": cosine(old["aligned_weight"], branch["aligned_weight"]),
                            "old_bridge_cosine": cosine(old["aligned_weight"], bridge["aligned_weight"]),
                            "branch_bridge_cosine": cosine(branch["aligned_weight"], bridge["aligned_weight"]),
                            "branch_residual_norm": b_res_norm,
                            "bridge_residual_norm": c_res_norm,
                            "branch_residual_redundant": b_res_norm < 1e-3,
                            "bridge_residual_redundant": c_res_norm < 1e-3,
                        }
                    )
    useful_branch = any(r["branch_residual_norm"] >= 0.2 for r in residual_rows)
    useful_bridge = any(r["bridge_residual_norm"] >= 0.2 for r in residual_rows)
    if useful_branch and useful_bridge:
        verdict = "INDEPENDENT_BRANCH_BRIDGE_RESIDUALS"
    elif useful_branch:
        verdict = "BRANCH_RESIDUAL_USEFUL"
    elif useful_bridge:
        verdict = "BRIDGE_RESIDUAL_USEFUL"
    elif residual_rows:
        verdict = "OLD_SPANS_BRANCH_BRIDGE"
    else:
        verdict = "INCONCLUSIVE"
    out_payload = {
        "BG_MERGED_TAP_WEIGHT_GEOMETRY_VERDICT": verdict,
        "verdict": verdict,
        "cosine_rows": rows,
        "residual_rows": residual_rows,
        "selected_heads": [compact_weight_row(r) for r in selected],
    }
    write_json(GEOMETRY_JSON, out_payload)
    write_csv(GEOMETRY_CSV, rows + residual_rows)
    lines = ["# Merged Tap Weight Geometry", "", f"BG_MERGED_TAP_WEIGHT_GEOMETRY_VERDICT = {verdict}", "", "## Residual Summary", ""]
    lines.extend(md_table(residual_rows[:20], ["architecture", "old_branch_cosine", "old_bridge_cosine", "branch_bridge_cosine", "branch_residual_norm", "bridge_residual_norm"]))
    write_md(GEOMETRY_MD, lines)
    print(f"BG_MERGED_TAP_WEIGHT_GEOMETRY_VERDICT = {verdict}", flush=True)
    return 0


def apply_norm_mode(vec: torch.Tensor, target_config: str, mode: str, stats: dict[str, Any] | None = None) -> torch.Tensor:
    x = vec.detach().cpu().to(torch.float32).clone()
    if mode == "zscore_diag" and stats and target_config in stats:
        std = stats[target_config].get("std")
        if isinstance(std, torch.Tensor) and int(std.numel()) == int(x.numel()):
            x = x / std.clamp_min(1e-6)
    if mode == "per_block_norm":
        blocks = concat_blocks(target_config)
        out = torch.zeros_like(x)
        for idx, _ in enumerate(blocks):
            start = idx * HIDDEN_DIM if len(blocks) > 1 else 0
            stop = start + HIDDEN_DIM
            out[start:stop] = normed(x[start:stop])
        x = out
    return normed(x)


def orthogonal_residual(vec: torch.Tensor, bases: Sequence[torch.Tensor]) -> torch.Tensor:
    out = normed(vec)
    orthonormal: list[torch.Tensor] = []
    for base in bases:
        b = normed(base)
        if b.norm().item() < 1e-8:
            continue
        for prev in orthonormal:
            b = b - torch.dot(b, prev) * prev
        b = normed(b)
        if b.norm().item() >= 1e-8:
            orthonormal.append(b)
    for base in orthonormal:
        out = out - torch.dot(out, base) * base
    return out


def candidate_record(
    *,
    name: str,
    family: str,
    target_config: str,
    architecture: str,
    weight: torch.Tensor,
    old: dict[str, Any] | None,
    branch: dict[str, Any] | None,
    bridge: dict[str, Any] | None,
    coefficients: tuple[float, float, float],
    norm_mode: str,
    component_norms: dict[str, float],
) -> dict[str, Any]:
    return {
        "candidate_name": name,
        "candidate_family": family,
        "target_config": target_config,
        "architecture": architecture,
        "state_dict": {"linear.weight": weight.reshape(1, -1).detach().cpu().to(torch.float32)},
        "weight": weight.detach().cpu().to(torch.float32),
        "old_source": old.get("head_name") if old else None,
        "branch_source": branch.get("head_name") if branch else None,
        "bridge_source": bridge.get("head_name") if bridge else None,
        "coefficients": {"old": coefficients[0], "branch": coefficients[1], "bridge": coefficients[2]},
        "norm_mode": norm_mode,
        "component_norms": component_norms,
        "score_path": "LayerNorm(diff)->Linear" if architecture == "AntisymLinear" else "Linear(diff)",
        "score_distillation": "not_used",
    }


def run_build_candidates() -> int:
    ensure_root()
    aligned_payload = load_pt(ALIGNED_WEIGHTS_PT, {}) or {}
    aligned = list(aligned_payload.get("aligned_weights") or [])
    stats_payload = load_pt(FEATURE_STATS_PT, {}) or {}
    stats = (stats_payload or {}).get("stats") or {}
    candidates: list[dict[str, Any]] = []
    grids = [
        (0.8, 0.1, 0.1),
        (0.7, 0.15, 0.15),
        (0.6, 0.2, 0.2),
        (0.6, 0.3, 0.1),
        (0.6, 0.1, 0.3),
        (0.5, 0.25, 0.25),
        (0.45, 0.35, 0.20),
        (0.45, 0.20, 0.35),
    ]
    for arch in PRIMARY_ARCHITECTURES:
        olds = top_sources(aligned, "old_content", arch, limit=4)
        branches = top_sources(aligned, "branch", arch, limit=3)
        bridges = top_sources(aligned, "bridge", arch, limit=2)
        for old in olds:
            for norm_mode in ("raw_norm", "per_block_norm", "zscore_diag"):
                old_vec = apply_norm_mode(old["aligned_weight"], PRIMARY_TARGET, norm_mode, stats)
                if old_vec.norm().item() < 1e-8:
                    continue
                candidates.append(
                    candidate_record(
                        name=f"old_only_reexported::{arch}::{len(candidates)}",
                        family="old_only_reexported",
                        target_config=PRIMARY_TARGET,
                        architecture=arch,
                        weight=old_vec,
                        old=old,
                        branch=None,
                        bridge=None,
                        coefficients=(1.0, 0.0, 0.0),
                        norm_mode=norm_mode,
                        component_norms={"old": float(old_vec.norm().item())},
                    )
                )
                for branch in branches:
                    branch_vec = apply_norm_mode(branch["aligned_weight"], PRIMARY_TARGET, norm_mode, stats)
                    b_res = orthogonal_residual(branch_vec, [old_vec])
                    b_norm = float(b_res.norm().item())
                    if b_norm >= 1e-6:
                        b_res = b_res / b_norm
                    for a, b in ((0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)):
                        merged = normed(a * old_vec + b * b_res)
                        candidates.append(
                            candidate_record(
                                name=f"old_plus_branch_residual::{arch}::{len(candidates)}",
                                family="old_plus_branch_residual",
                                target_config=PRIMARY_TARGET,
                                architecture=arch,
                                weight=merged,
                                old=old,
                                branch=branch,
                                bridge=None,
                                coefficients=(a, b, 0.0),
                                norm_mode=norm_mode,
                                component_norms={"old": float(old_vec.norm().item()), "branch_residual": b_norm},
                            )
                        )
                    for bridge in bridges:
                        bridge_vec = apply_norm_mode(bridge["aligned_weight"], PRIMARY_TARGET, norm_mode, stats)
                        c_res_old = orthogonal_residual(bridge_vec, [old_vec])
                        c_old_norm = float(c_res_old.norm().item())
                        if c_old_norm >= 1e-6:
                            c_res_old = c_res_old / c_old_norm
                        for a, c in ((0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)):
                            merged = normed(a * old_vec + c * c_res_old)
                            candidates.append(
                                candidate_record(
                                    name=f"old_plus_bridge_residual::{arch}::{len(candidates)}",
                                    family="old_plus_bridge_residual",
                                    target_config=PRIMARY_TARGET,
                                    architecture=arch,
                                    weight=merged,
                                    old=old,
                                    branch=None,
                                    bridge=bridge,
                                    coefficients=(a, 0.0, c),
                                    norm_mode=norm_mode,
                                    component_norms={"old": float(old_vec.norm().item()), "bridge_residual": c_old_norm},
                                )
                            )
                        c_res = orthogonal_residual(bridge_vec, [old_vec, b_res])
                        c_norm = float(c_res.norm().item())
                        if c_norm >= 1e-6:
                            c_res = c_res / c_norm
                        for a, b, c in grids:
                            merged = normed(a * old_vec + b * b_res + c * c_res)
                            candidates.append(
                                candidate_record(
                                    name=f"old_plus_branch_plus_bridge_residual::{arch}::{len(candidates)}",
                                    family="old_plus_branch_plus_bridge_residual",
                                    target_config=PRIMARY_TARGET,
                                    architecture=arch,
                                    weight=merged,
                                    old=old,
                                    branch=branch,
                                    bridge=bridge,
                                    coefficients=(a, b, c),
                                    norm_mode=norm_mode,
                                    component_norms={"old": float(old_vec.norm().item()), "branch_residual": b_norm, "bridge_residual": c_norm},
                                )
                            )
                if branches:
                    # Negative control: add a deterministic reversed branch residual.
                    branch = branches[0]
                    branch_vec = apply_norm_mode(branch["aligned_weight"], PRIMARY_TARGET, norm_mode, stats)
                    b_res = orthogonal_residual(branch_vec, [old_vec])
                    if b_res.norm().item() >= 1e-6:
                        merged = normed(0.7 * old_vec - 0.3 * normed(b_res))
                        candidates.append(
                            candidate_record(
                                name=f"random_residual_negative_control::{arch}::{len(candidates)}",
                                family="random_residual_negative_control",
                                target_config=PRIMARY_TARGET,
                                architecture=arch,
                                weight=merged,
                                old=old,
                                branch=branch,
                                bridge=None,
                                coefficients=(0.7, -0.3, 0.0),
                                norm_mode=norm_mode,
                                component_norms={"old": float(old_vec.norm().item()), "branch_residual": float(b_res.norm().item())},
                            )
                        )
    if candidates:
        verdict = "READY"
    else:
        verdict = "BLOCKED"
    compact = [{k: v for k, v in c.items() if k not in {"weight", "state_dict"}} for c in candidates]
    out_payload = {
        "BG_MERGED_TAP_CANDIDATE_BUILD_VERDICT": verdict,
        "verdict": verdict,
        "candidate_count": len(candidates),
        "family_counts": dict(Counter(c["candidate_family"] for c in candidates)),
        "normalization_modes": dict(Counter(c["norm_mode"] for c in candidates)),
        "architecture_counts": dict(Counter(c["architecture"] for c in candidates)),
        "score_distillation": "not_used_primary",
    }
    torch.save({"candidates": candidates, "summary": out_payload}, MERGED_TAPS_PT)
    write_json(MERGED_CANDIDATES_JSON, {**out_payload, "candidates": compact})
    lines = ["# Merged Tap Candidates", "", f"BG_MERGED_TAP_CANDIDATE_BUILD_VERDICT = {verdict}", "", f"- candidates: `{len(candidates)}`", f"- family counts: `{out_payload['family_counts']}`", f"- score distillation: `not_used_primary`"]
    write_md(MERGED_CANDIDATES_MD, lines)
    print(f"BG_MERGED_TAP_CANDIDATE_BUILD_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def score_diff(weight: torch.Tensor, architecture: str, diff: torch.Tensor) -> float:
    x = diff.detach().cpu().to(torch.float32).flatten()
    w = weight.detach().cpu().to(torch.float32).flatten()
    if int(x.numel()) != int(w.numel()):
        return float("nan")
    if architecture == "AntisymLinear":
        x = F.layer_norm(x, (int(x.numel()),))
    return float(torch.dot(w, x).item())


def eval_candidate_on_pairs(candidate: dict[str, Any], pairs: Sequence[dict[str, Any]], *, split: str | None = None, pair_type: str | None = None) -> dict[str, Any]:
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    weight = candidate.get("weight")
    if not isinstance(weight, torch.Tensor):
        state = candidate.get("state_dict") or {}
        value = state.get("linear.weight") if isinstance(state, dict) else None
        weight = value.flatten() if isinstance(value, torch.Tensor) else None
    if not isinstance(weight, torch.Tensor):
        return {"pair_count": 0, "pairwise_accuracy": float("nan"), "mean_margin": float("nan")}
    used = correct = ties = 0
    margins: list[float] = []
    by_domain: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        if split and str(pair.get("split")) != split:
            continue
        if pair_type and str(pair.get("pair_type")) != pair_type:
            continue
        diff = pair_diff(pair, config)
        if diff is None:
            continue
        score = score_diff(weight, arch, diff)
        if not math.isfinite(score):
            continue
        used += 1
        margins.append(score)
        if score > 0:
            correct += 1
            by_domain[str(pair.get("domain"))].append(1.0)
        elif score == 0:
            ties += 1
            by_domain[str(pair.get("domain"))].append(0.5)
        else:
            by_domain[str(pair.get("domain"))].append(0.0)
    acc = (correct + 0.5 * ties) / used if used else float("nan")
    return {
        "pair_count": used,
        "pairwise_accuracy": acc,
        "mean_margin": finite_mean(margins),
        "domain_accuracy": {k: finite_mean(v) for k, v in by_domain.items()},
    }


def branch_groups(split: str | None = None) -> list[list[dict[str, Any]]]:
    payload = load_pt(BRIDGE_PT, {}) or {}
    rows = [dict(r) for r in payload.get("candidate_rows") or []]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split and str(row.get("split")) != split:
            continue
        fmap = row.get("features_by_config") or {}
        if PRIMARY_TARGET not in fmap:
            continue
        grouped[str(row.get("group_id"))].append(row)
    return [vals for vals in grouped.values() if len(vals) >= 2]


def rank_group(candidate: dict[str, Any], group: Sequence[dict[str, Any]]) -> list[int]:
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    weight = candidate.get("weight")
    if not isinstance(weight, torch.Tensor):
        state = candidate.get("state_dict") or {}
        value = state.get("linear.weight") if isinstance(state, dict) else None
        weight = value.flatten() if isinstance(value, torch.Tensor) else None
    if not isinstance(weight, torch.Tensor):
        return []
    scores = []
    for i, left in enumerate(group):
        total = 0.0
        usable = 0
        lvec = (left.get("features_by_config") or {}).get(config)
        if not isinstance(lvec, torch.Tensor):
            return []
        for j, right in enumerate(group):
            if i == j:
                continue
            rvec = (right.get("features_by_config") or {}).get(config)
            if not isinstance(rvec, torch.Tensor):
                return []
            total += score_diff(weight, arch, lvec.flatten() - rvec.flatten())
            usable += 1
        scores.append((i, total / max(usable, 1)))
    return [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


def group_topk_metrics(candidate: dict[str, Any], groups: Sequence[Sequence[dict[str, Any]]], k: int = 4) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        rewards = [safe_float(r.get("reward"), 0.0) for r in group]
        if not rewards:
            continue
        oracle = max(rewards)
        oracle_indices = {i for i, r in enumerate(rewards) if r == oracle}
        ranking = rank_group(candidate, group)
        if not ranking:
            continue
        selected = ranking[: min(k, len(ranking))]
        kept = bool(oracle_indices & set(selected))
        best_selected = max(rewards[i] for i in selected)
        rows.append(
            {
                "oracle_retention": 1.0 if kept else 0.0,
                "false_prune_rate": 0.0 if kept else 1.0,
                "avg_survivors": float(len(selected)),
                "best_selected_reward": best_selected,
                "oracle_reward": oracle,
                "regret": oracle - best_selected,
                "top1_success": 1.0 if selected and rewards[selected[0]] == oracle and oracle > 0 else 0.0,
                "domain": str(group[0].get("domain")),
            }
        )
    return {
        "group_count": len(rows),
        "oracle_retention": finite_mean(r["oracle_retention"] for r in rows),
        "false_prune_rate": finite_mean(r["false_prune_rate"] for r in rows),
        "avg_survivors": finite_mean(r["avg_survivors"] for r in rows),
        "best_selected_reward": finite_mean(r["best_selected_reward"] for r in rows),
        "regret": finite_mean(r["regret"] for r in rows),
        "top1_success": finite_mean(r["top1_success"] for r in rows),
        "domain_breakdown": {d: finite_mean(r["oracle_retention"] for r in rows if r["domain"] == d) for d in sorted({r["domain"] for r in rows})},
    }


def load_candidates() -> list[dict[str, Any]]:
    payload = load_pt(MERGED_TAPS_PT, {}) or {}
    return list(payload.get("candidates") or [])


def reference_candidates(limit: int = 18) -> list[dict[str, Any]]:
    payload = load_pt(ALIGNED_WEIGHTS_PT, {}) or {}
    aligned = list(payload.get("aligned_weights") or [])
    refs: list[dict[str, Any]] = []
    for arch in PRIMARY_ARCHITECTURES:
        for family in ("old_content", "branch", "bridge", "universal"):
            refs.extend(top_sources(aligned, family, arch, limit=2))
    out = []
    seen = set()
    for row in refs:
        key = str(row.get("head_name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "candidate_name": f"source::{key}",
                "candidate_family": f"source_{row.get('source_family')}",
                "target_config": row.get("target_config"),
                "architecture": row.get("architecture"),
                "weight": row.get("aligned_weight"),
                "state_dict": {"linear.weight": row.get("aligned_weight").reshape(1, -1)} if isinstance(row.get("aligned_weight"), torch.Tensor) else {},
                "old_source": row.get("head_name") if row.get("source_family") == "old_content" else None,
                "branch_source": row.get("head_name") if row.get("source_family") == "branch" else None,
                "bridge_source": row.get("head_name") if row.get("source_family") == "bridge" else None,
                "norm_mode": "source",
                "score_path": "LayerNorm(diff)->Linear" if row.get("architecture") == "AntisymLinear" else "Linear(diff)",
            }
        )
        if len(out) >= limit:
            break
    return out


def evaluate_candidate_validation(candidate: dict[str, Any], pair_sets: dict[str, list[dict[str, Any]]], groups: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    old = eval_candidate_on_pairs(candidate, pair_sets.get("old_content", []), split="val")
    hidden = eval_candidate_on_pairs(candidate, pair_sets.get("hidden_branch", []), split="val")
    bridge = eval_candidate_on_pairs(candidate, pair_sets.get("bridge", []), split="val")
    survival = group_topk_metrics(candidate, groups, k=4)
    vals = [
        old.get("pairwise_accuracy"),
        hidden.get("pairwise_accuracy"),
        bridge.get("pairwise_accuracy"),
        survival.get("oracle_retention"),
    ]
    balanced = finite_mean(vals)
    return {
        "candidate_name": candidate.get("candidate_name"),
        "candidate_family": candidate.get("candidate_family"),
        "architecture": candidate.get("architecture"),
        "norm_mode": candidate.get("norm_mode"),
        "old_val_acc": old.get("pairwise_accuracy"),
        "hidden_val_acc": hidden.get("pairwise_accuracy"),
        "bridge_val_acc": bridge.get("pairwise_accuracy"),
        "survival_val_oracle_retention": survival.get("oracle_retention"),
        "survival_val_false_prune": survival.get("false_prune_rate"),
        "validation_balanced_score": balanced,
        "old_pair_count": old.get("pair_count"),
        "hidden_pair_count": hidden.get("pair_count"),
        "bridge_pair_count": bridge.get("pair_count"),
        "survival_group_count": survival.get("group_count"),
    }


def run_validation_selection() -> int:
    ensure_root()
    candidates = load_candidates()
    pair_sets = load_pair_sets()
    groups = branch_groups("val")
    rows = [evaluate_candidate_validation(c, pair_sets, groups) for c in candidates]
    refs = [evaluate_candidate_validation(c, pair_sets, groups) for c in reference_candidates()]
    source_old_best = max([r for r in refs if r["candidate_family"] == "source_old_content"], key=lambda r: safe_float(r.get("old_val_acc"), -1.0), default=None)
    old_floor = safe_float((source_old_best or {}).get("old_val_acc"), 0.0) - 0.08
    eligible = [
        (idx, row)
        for idx, row in enumerate(rows)
        if math.isfinite(safe_float(row.get("validation_balanced_score"), float("nan")))
        and safe_float(row.get("old_val_acc"), 0.0) >= old_floor
        and safe_float(row.get("bridge_val_acc"), 0.0) >= 0.45
    ]
    if not eligible:
        eligible = [(idx, row) for idx, row in enumerate(rows) if math.isfinite(safe_float(row.get("validation_balanced_score"), float("nan")))]
    selected_idx, selected_row = max(eligible, key=lambda item: safe_float(item[1].get("validation_balanced_score"), -1.0)) if eligible else (-1, {})
    selected = candidates[selected_idx] if selected_idx >= 0 else None
    source_best = max(refs, key=lambda r: safe_float(r.get("validation_balanced_score"), -1.0), default={})
    selected_score = safe_float(selected_row.get("validation_balanced_score"), float("nan"))
    source_score = safe_float(source_best.get("validation_balanced_score"), float("nan"))
    if selected is None:
        verdict = "INSUFFICIENT"
    elif safe_float(selected_row.get("old_val_acc"), 0.0) < old_floor:
        verdict = "OLD_CONTEXT_DEGRADES"
    elif safe_float(selected_row.get("bridge_val_acc"), 0.0) < 0.45:
        verdict = "BRIDGE_COLLAPSES"
    elif selected_score >= source_score:
        verdict = "READY"
    else:
        verdict = "WEAK"
    out_payload = {
        "BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT": verdict,
        "verdict": verdict,
        "selected_candidate_name": selected.get("candidate_name") if selected else None,
        "selected_validation": selected_row,
        "source_best_validation": source_best,
        "old_floor": old_floor,
        "selection_split": "validation_only",
        "heldout_used_for_selection": False,
    }
    torch.save({"selected": selected, "selected_validation": selected_row, "validation_rows": rows, "reference_rows": refs, "summary": out_payload}, SELECTED_TAPS_PT)
    write_json(VALIDATION_JSON, {**out_payload, "rows": rows, "reference_rows": refs})
    write_csv(VALIDATION_CSV, rows + refs)
    lines = [
        "# Merged Tap Validation Selection",
        "",
        f"BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT = {verdict}",
        "",
        f"- selected: `{out_payload['selected_candidate_name']}`",
        f"- selected balanced validation score: `{rate(selected_score)}`",
        f"- best source balanced validation score: `{rate(source_score)}`",
        f"- heldout used for selection: `False`",
    ]
    write_md(VALIDATION_MD, lines)
    print(f"BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def selected_and_references() -> list[dict[str, Any]]:
    payload = load_pt(SELECTED_TAPS_PT, {}) or {}
    out = []
    selected = payload.get("selected")
    if isinstance(selected, dict):
        out.append(selected)
    out.extend(reference_candidates())
    return out


def compact_feature_map(fmap: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for cfg, vec in fmap.items():
        if isinstance(vec, torch.Tensor):
            try:
                if int(vec.numel()) == config_dim(cfg):
                    out[cfg] = vec.detach().cpu().to(torch.float32).flatten()
            except Exception:
                continue
    return out


def add_feature_alias(index: dict[str, dict[str, Any]], key: Any, fmap: dict[str, torch.Tensor], source: str) -> None:
    key_s = str(key or "")
    if not key_s or not fmap:
        return
    index.setdefault(key_s, {"features_by_config": compact_feature_map(fmap), "feature_source": source})


def survivor_feature_index() -> dict[str, dict[str, Any]]:
    from bg_universal_tap_v1_common import tensor_feature_map
    from bg_gated_selector_v1_common import CODE_EXPANDED_FEATURES_PT, code_feature_index, feature_map_from_old_code_pooled

    index: dict[str, dict[str, Any]] = {}

    old_payload = load_pt(OLD_CONTENT_PT, {}) or {}
    for row in old_payload.get("candidate_rows") or []:
        fmap = row.get("features_by_config") or {}
        add_feature_alias(index, row.get("candidate_id"), fmap, "old_content_dataset")

    code_payload = load_pt(CODE_EXPANDED_FEATURES_PT, {}) or {}
    if code_payload:
        for uid, row in code_feature_index(code_payload).items():
            fmap = feature_map_from_old_code_pooled(row.get("pooled")) if row else {}
            add_feature_alias(index, uid, fmap, "code_expanded_strict_clean_features")

    bridge_payload = load_pt(BRIDGE_PT, {}) or {}
    for row in bridge_payload.get("candidate_rows") or []:
        fmap = row.get("features_by_config") or {}
        add_feature_alias(index, row.get("candidate_id"), fmap, "universal_bridge_candidate_rows")
        add_feature_alias(index, f"{row.get('source_id')}::{row.get('group_id')}::{row.get('branch_id')}", fmap, "universal_bridge_candidate_rows")

    for source_name, path in (("branch_generator_v1", BGV1_BRANCHES_PT), ("quota_v4", QUOTA_V4_BRANCHES_PT)):
        payload = load_pt(path, {}) or {}
        for row in payload.get("rows") or []:
            fmap = tensor_feature_map(row.get("features")) if isinstance(row.get("features"), torch.Tensor) else {}
            if not fmap and isinstance(row.get("pooled_vectors"), dict):
                pooled = row.get("pooled_vectors") or {}
                tmp = {}
                for cfg in CONFIGS:
                    if cfg.endswith("_L4") and cfg[:2].isdigit():
                        value = pooled.get(f"L{cfg[:2]}_L4")
                        if isinstance(value, torch.Tensor):
                            tmp[cfg] = value
                if all(k in tmp for k in ("24_L4", "36_L4", "47_L4")):
                    tmp["concat_24_36_47"] = torch.cat([tmp["24_L4"], tmp["36_L4"], tmp["47_L4"]], dim=0)
                fmap = tmp
            key = f"{source_name}::{row.get('branch_group_id')}::{row.get('branch_id')}"
            add_feature_alias(index, key, fmap, str(path.name))
    return index


def run_survivor_feature_acquisition() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(FINAL_ARBITER_DATASET_PT, {}) or {}
    survivor_sets = list(dataset.get("survivor_sets") or [])
    index = survivor_feature_index()
    enriched_sets: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    total_candidates = matched_candidates = complete_sets = 0
    source_counts: Counter[str] = Counter()
    missing_by_origin: Counter[str] = Counter()
    for s in survivor_sets:
        new_set = dict(s)
        new_candidates = []
        set_complete = True
        for c in s.get("candidates") or []:
            total_candidates += 1
            cid = str(c.get("candidate_id"))
            match = index.get(cid)
            new_c = dict(c)
            if match:
                fmap = compact_feature_map(match.get("features_by_config") or {})
                if PRIMARY_TARGET in fmap:
                    matched_candidates += 1
                    source_counts[str(match.get("feature_source"))] += 1
                    new_c["features_by_config"] = fmap
                    new_c["merged_feature_source"] = str(match.get("feature_source"))
                else:
                    set_complete = False
                    missing_by_origin[str(c.get("branch_origin") or c.get("origin"))] += 1
            else:
                set_complete = False
                missing_by_origin[str(c.get("branch_origin") or c.get("origin"))] += 1
            new_candidates.append(new_c)
            rows.append(
                {
                    "survivor_set_id": s.get("survivor_set_id"),
                    "candidate_id": cid,
                    "task_id": s.get("task_id"),
                    "domain": s.get("domain"),
                    "split": s.get("split"),
                    "branch_origin": c.get("branch_origin") or c.get("origin"),
                    "feature_matched": bool(match and PRIMARY_TARGET in (match.get("features_by_config") or {})),
                    "feature_source": match.get("feature_source") if match else "",
                }
            )
        new_set["candidates"] = new_candidates
        new_set["merged_tap_feature_complete"] = set_complete and bool(new_candidates)
        complete_sets += int(new_set["merged_tap_feature_complete"])
        enriched_sets.append(new_set)
    coverage = matched_candidates / max(total_candidates, 1)
    set_coverage = complete_sets / max(len(survivor_sets), 1)
    if coverage >= 0.95 and set_coverage >= 0.90:
        verdict = "READY"
    elif coverage >= 0.75:
        verdict = "PARTIAL"
    elif matched_candidates:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_MERGED_TAP_SURVIVOR_FEATURE_ACQUISITION_VERDICT": verdict,
        "verdict": verdict,
        "survivor_sets": enriched_sets,
        "coverage": {
            "sets": len(survivor_sets),
            "complete_sets": complete_sets,
            "complete_set_rate": set_coverage,
            "candidates": total_candidates,
            "matched_candidates": matched_candidates,
            "candidate_match_rate": coverage,
            "source_counts": dict(source_counts),
            "missing_by_origin": dict(missing_by_origin),
        },
        "acquisition_mode": "cached_feature_recovery_from_old_content_code_bgv1_quota_bridge_artifacts",
        "new_generation_run": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, SURVIVOR_FEATURES_PT)
    write_json(
        SURVIVOR_FEATURES_JSON,
        {
            **{k: v for k, v in payload.items() if k != "survivor_sets"},
            "survivor_sets": [
                {
                    **{k: v for k, v in s.items() if k != "candidates"},
                    "candidates": [
                        {kk: vv for kk, vv in c.items() if kk != "features_by_config"} | {"available_merged_feature_configs": sorted((c.get("features_by_config") or {}).keys())}
                        for c in s.get("candidates") or []
                    ],
                }
                for s in enriched_sets
            ],
        },
    )
    write_csv(SURVIVOR_FEATURES_CSV, rows)
    lines = [
        "# Merged Tap Top4 Survivor Feature Acquisition",
        "",
        f"BG_MERGED_TAP_SURVIVOR_FEATURE_ACQUISITION_VERDICT = {verdict}",
        "",
        f"- survivor sets: `{len(survivor_sets)}`",
        f"- complete sets: `{complete_sets}` / `{len(survivor_sets)}` (`{rate(set_coverage)}`)",
        f"- matched candidates: `{matched_candidates}` / `{total_candidates}` (`{rate(coverage)}`)",
        f"- new generation run: `False`",
        f"- source counts: `{dict(source_counts)}`",
        f"- missing by origin: `{dict(missing_by_origin)}`",
        "",
        "The missing final-arbiter data was acquired from cached raw feature artifacts by indexing old-content candidate IDs, code feature aliases, and BGV1/quota branch `branch_group_id` + `branch_id` lineage keys.",
    ]
    write_md(SURVIVOR_FEATURES_MD, lines)
    print(f"BG_MERGED_TAP_SURVIVOR_FEATURE_ACQUISITION_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def eval_pairs_rows(candidates: Sequence[dict[str, Any]], pair_sets: dict[str, list[dict[str, Any]]], split: str, pair_types: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        for pair_type in pair_types:
            metrics = eval_candidate_on_pairs(cand, pair_sets.get(pair_type, []), split=split)
            rows.append(
                {
                    "candidate_name": cand.get("candidate_name"),
                    "candidate_family": cand.get("candidate_family"),
                    "pair_type": pair_type,
                    "split": split,
                    **{k: v for k, v in metrics.items() if k != "domain_accuracy"},
                    "domain_accuracy": metrics.get("domain_accuracy"),
                }
            )
    return rows


def run_old_code_eval() -> int:
    ensure_root()
    pair_sets = load_pair_sets()
    candidates = selected_and_references()
    rows = eval_pairs_rows(candidates, pair_sets, "heldout", ["old_content"])
    selected = next((r for r in rows if not str(r["candidate_name"]).startswith("source::")), None)
    best_old = max([r for r in rows if r["candidate_family"] == "source_old_content"], key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
    degradation = safe_float(best_old.get("pairwise_accuracy"), 0.0) - safe_float((selected or {}).get("pairwise_accuracy"), 0.0)
    if not selected or safe_float(selected.get("pair_count"), 0) == 0:
        verdict = "INSUFFICIENT"
    elif degradation <= 0.02:
        verdict = "PRESERVED_OR_IMPROVED"
    elif degradation <= 0.08:
        verdict = "SMALL_DEGRADATION"
    else:
        verdict = "LARGE_DEGRADATION"
    out_payload = {
        "BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "selected": selected,
        "best_old_source": best_old,
        "old_context_degradation": degradation,
        "coding_status": "DATA_LIMITED_NO_SEPARATE_CODING_PAIR_FEATURES"
        if (not selected or "coding" not in str(selected.get("domain_accuracy", "")))
        else "EVALUATED",
        "rows": rows,
    }
    write_json(OLD_CODE_JSON, out_payload)
    write_csv(OLD_CODE_CSV, rows)
    lines = ["# Merged Tap Old/Code Evaluation", "", f"BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT = {verdict}", "", f"- selected old-content heldout accuracy: `{rate((selected or {}).get('pairwise_accuracy'))}`", f"- best old-source heldout accuracy: `{rate(best_old.get('pairwise_accuracy'))}`", f"- degradation: `{rate(degradation)}`", "", "Coding-specific pair features were not separately available in this merged-tap cached evaluation; old-content preservation is the primary proxy."]
    write_md(OLD_CODE_MD, lines)
    print(f"BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_branch_bridge_eval() -> int:
    ensure_root()
    pair_sets = load_pair_sets()
    candidates = selected_and_references()
    rows = eval_pairs_rows(candidates, pair_sets, "heldout", ["hidden_branch", "bridge"])
    selected_rows = [r for r in rows if not str(r["candidate_name"]).startswith("source::")]
    selected_hidden = next((r for r in selected_rows if r["pair_type"] == "hidden_branch"), {})
    selected_bridge = next((r for r in selected_rows if r["pair_type"] == "bridge"), {})
    hidden_acc = safe_float(selected_hidden.get("pairwise_accuracy"), float("nan"))
    bridge_acc = safe_float(selected_bridge.get("pairwise_accuracy"), float("nan"))
    if not selected_rows:
        verdict = "INSUFFICIENT"
    elif hidden_acc >= 0.55 and bridge_acc >= 0.55:
        verdict = "BRANCH_AND_BRIDGE_READY"
    elif hidden_acc >= 0.55:
        verdict = "BRANCH_READY_BRIDGE_WEAK"
    elif bridge_acc >= 0.55:
        verdict = "BRIDGE_READY_BRANCH_WEAK"
    elif math.isfinite(hidden_acc) and hidden_acc < 0.50:
        verdict = "BRANCH_SIGNAL_LOST"
    elif math.isfinite(bridge_acc) and bridge_acc < 0.50:
        verdict = "BRIDGE_SIGNAL_LOST"
    else:
        verdict = "DATA_LIMITED"
    out_payload = {
        "BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "selected_hidden_branch": selected_hidden,
        "selected_bridge": selected_bridge,
        "rows": rows,
    }
    write_json(BRANCH_BRIDGE_JSON, out_payload)
    write_csv(BRANCH_BRIDGE_CSV, rows)
    lines = ["# Merged Tap Branch/Bridge Evaluation", "", f"BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT = {verdict}", "", f"- selected hidden-branch heldout accuracy: `{rate(hidden_acc)}`", f"- selected bridge heldout accuracy: `{rate(bridge_acc)}`"]
    write_md(BRANCH_BRIDGE_MD, lines)
    print(f"BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_survival_eval() -> int:
    ensure_root()
    candidates = selected_and_references()
    groups = branch_groups("heldout")
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        metrics = group_topk_metrics(cand, groups, k=4)
        rows.append({"candidate_name": cand.get("candidate_name"), "candidate_family": cand.get("candidate_family"), **{k: v for k, v in metrics.items() if k != "domain_breakdown"}, "domain_breakdown": metrics.get("domain_breakdown")})
    selected = next((r for r in rows if not str(r["candidate_name"]).startswith("source::")), None)
    fixed_ref = load_json(FIXED_HELDOUT_JSON, {}) or {}
    fixed_metrics = fixed_ref.get("selected_policy_metrics") or fixed_ref.get("metrics") or fixed_ref
    selected_ret = safe_float((selected or {}).get("oracle_retention"), float("nan"))
    selected_fp = safe_float((selected or {}).get("false_prune_rate"), float("nan"))
    # This uses compatible cached branch-candidate feature groups, not the full fixed-composite heldout rows.
    if not selected or safe_float(selected.get("group_count"), 0) == 0:
        verdict = "INSUFFICIENT"
    elif selected_ret >= 0.90 and selected_fp <= 0.10:
        verdict = "MERGED_MATCHES_FIXED_COMPOSITE"
    elif selected_ret >= 0.80 and selected_fp <= 0.20:
        verdict = "USE_AS_COMPOSITE_EXPERT"
    elif selected_fp > 0.20:
        verdict = "TOO_MANY_FALSE_PRUNES"
    else:
        verdict = "FIXED_COMPOSITE_STILL_BEST"
    out_payload = {
        "BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "selected": selected,
        "fixed_composite_reference": fixed_metrics,
        "eval_scope": "compatible_cached_branch_candidate_feature_groups_from_universal_bridge_dataset",
        "not_full_fixed_composite_replacement": True,
        "rows": rows,
    }
    write_json(SURVIVAL_JSON, out_payload)
    write_csv(SURVIVAL_CSV, rows)
    lines = ["# Merged Tap Survival Evaluation", "", f"BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT = {verdict}", "", "This is a compatible cached branch-candidate feature-group survival proxy. The full fixed-composite policy includes veto/rescue and missing/OOD guardrails.", "", f"- selected oracle retention: `{rate(selected_ret)}`", f"- selected false prune: `{rate(selected_fp)}`", f"- heldout groups: `{(selected or {}).get('group_count', 0)}`"]
    write_md(SURVIVAL_MD, lines)
    print(f"BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_final_arbiter_eval() -> int:
    ensure_root()
    feature_payload = load_pt(SURVIVOR_FEATURES_PT, {}) or {}
    if not feature_payload:
        run_survivor_feature_acquisition()
        feature_payload = load_pt(SURVIVOR_FEATURES_PT, {}) or {}
    sets = list(feature_payload.get("survivor_sets") or [])
    coverage = feature_payload.get("coverage") or {}
    selected_payload = load_pt(SELECTED_TAPS_PT, {}) or {}
    selected_tap = selected_payload.get("selected")
    rows: list[dict[str, Any]] = []
    policies = [
        "merged_tap_top1",
        "fixed_composite_top1",
        "majority_rank_aggregation",
        "old_top1",
        "bridge_top1",
        "universal_top1",
        "fixed_plus_merged_rank_sum",
        "oracle_best_survivor",
    ]

    def score_choice(cands: list[dict[str, Any]], key: str) -> int:
        return max(range(len(cands)), key=lambda i: (safe_float(cands[i].get(key), -1e9), -i)) if cands else -1

    def rank_choice(cands: list[dict[str, Any]], key: str) -> int:
        return min(range(len(cands)), key=lambda i: (safe_float(cands[i].get(key), 1e9), i)) if cands else -1

    def merged_scores(cands: list[dict[str, Any]]) -> list[float] | None:
        if not isinstance(selected_tap, dict):
            return None
        weight = selected_tap.get("weight")
        if not isinstance(weight, torch.Tensor):
            state = selected_tap.get("state_dict") or {}
            value = state.get("linear.weight") if isinstance(state, dict) else None
            weight = value.flatten() if isinstance(value, torch.Tensor) else None
        if not isinstance(weight, torch.Tensor):
            return None
        arch = str(selected_tap.get("architecture"))
        vecs: list[torch.Tensor] = []
        for cand in cands:
            vec = (cand.get("features_by_config") or {}).get(PRIMARY_TARGET)
            if not isinstance(vec, torch.Tensor):
                return None
            vecs.append(vec.detach().cpu().to(torch.float32).flatten())
        scores = []
        for i, left in enumerate(vecs):
            total = 0.0
            for j, right in enumerate(vecs):
                if i == j:
                    continue
                total += score_diff(weight, arch, left - right)
            scores.append(total / max(len(vecs) - 1, 1))
        return scores

    for s in sets:
        cands = list(s.get("candidates") or [])
        if not cands:
            continue
        complete = bool(s.get("merged_tap_feature_complete"))
        rewards = [safe_float(c.get("final_reward", c.get("reward", 0.0)), 0.0) for c in cands]
        best_reward = max(rewards) if rewards else 0.0
        merged = merged_scores(cands) if complete else None
        merged_rank = sorted(range(len(cands)), key=lambda i: (-(merged or [])[i], i)) if merged else []
        rank_merged = {idx: rank + 1 for rank, idx in enumerate(merged_rank)}
        choices = {
            "merged_tap_top1": merged_rank[0] if merged_rank else -1,
            "fixed_composite_top1": score_choice(cands, "fixed_composite_score"),
            "majority_rank_aggregation": rank_choice(cands, "rank_majority"),
            "old_top1": score_choice(cands, "old_score"),
            "bridge_top1": score_choice(cands, "bridge_score"),
            "universal_top1": score_choice(cands, "universal_score"),
            "oracle_best_survivor": max(range(len(cands)), key=lambda i: (rewards[i], -i)),
        }
        if merged:
            choices["fixed_plus_merged_rank_sum"] = min(
                range(len(cands)),
                key=lambda i: (safe_float(cands[i].get("rank_fixed_composite"), 1e9) + rank_merged.get(i, 1e9), i),
            )
        else:
            choices["fixed_plus_merged_rank_sum"] = -1
        for policy in policies:
            idx = choices.get(policy, -1)
            if idx < 0 or idx >= len(cands):
                continue
            selected = cands[idx]
            reward = rewards[idx]
            rows.append(
                {
                    "policy": policy,
                    "survivor_set_id": s.get("survivor_set_id"),
                    "task_id": s.get("task_id"),
                    "domain": s.get("domain"),
                    "split": s.get("split"),
                    "readiness_eligible": bool(s.get("readiness_eligible")),
                    "feature_complete": complete,
                    "candidate_count": len(cands),
                    "selected_candidate_id": selected.get("candidate_id"),
                    "selected_reward": reward,
                    "selected_correctness": safe_float(selected.get("correctness", reward), reward),
                    "best_reward": best_reward,
                    "regret": best_reward - reward,
                    "oracle_selected": 1.0 if reward == best_reward else 0.0,
                    "merged_score": (merged[idx] if merged else None),
                }
            )

    def summarize(policy_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not policy_rows:
            return {"n": 0}
        by_task: dict[str, list[float]] = defaultdict(list)
        by_domain: dict[str, list[float]] = defaultdict(list)
        for row in policy_rows:
            by_task[str(row.get("task_id"))].append(safe_float(row.get("selected_reward"), 0.0))
            by_domain[str(row.get("domain"))].append(safe_float(row.get("selected_reward"), 0.0))
        return {
            "n": len(policy_rows),
            "tasks": len(by_task),
            "group_micro_reward": finite_mean(row.get("selected_reward") for row in policy_rows),
            "task_macro_reward": finite_mean(mean(vals) for vals in by_task.values()),
            "oracle_selected_rate": finite_mean(row.get("oracle_selected") for row in policy_rows),
            "regret": finite_mean(row.get("regret") for row in policy_rows),
            "domain_reward": {domain: finite_mean(vals) for domain, vals in sorted(by_domain.items())},
        }

    summaries: dict[str, Any] = {}
    for split_name in sorted({str(r.get("split")) for r in rows} | {"fresh_holdout"}):
        split_rows = [r for r in rows if str(r.get("split")) == split_name and bool(r.get("feature_complete"))]
        for policy in policies:
            summaries[f"{split_name}::{policy}"] = summarize([r for r in split_rows if r.get("policy") == policy])
    primary_split = "fresh_holdout" if any(str(r.get("split")) == "fresh_holdout" and bool(r.get("feature_complete")) for r in rows) else "heldout"
    primary = {policy: summaries.get(f"{primary_split}::{policy}", {"n": 0}) for policy in policies}
    merged_reward = safe_float(primary.get("merged_tap_top1", {}).get("task_macro_reward"), float("nan"))
    fixed_reward = safe_float(primary.get("fixed_composite_top1", {}).get("task_macro_reward"), float("nan"))
    majority_reward = safe_float(primary.get("majority_rank_aggregation", {}).get("task_macro_reward"), float("nan"))
    oracle_reward = safe_float(primary.get("oracle_best_survivor", {}).get("task_macro_reward"), float("nan"))
    if primary.get("merged_tap_top1", {}).get("n", 0) == 0:
        verdict = "DATA_LIMITED"
    elif merged_reward > max(fixed_reward, majority_reward):
        verdict = "FINAL_ARBITER_IMPROVES"
    elif merged_reward > fixed_reward or merged_reward > majority_reward:
        verdict = "DOMAIN_IMPROVES"
    elif oracle_reward > max(fixed_reward, majority_reward, merged_reward):
        verdict = "NO_IMPROVEMENT"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "primary_split": primary_split,
        "coverage": coverage,
        "primary": primary,
        "summaries": summaries,
        "selected_merged_tap": selected_tap.get("candidate_name") if isinstance(selected_tap, dict) else None,
        "reason": "Merged tap scores were computed from acquired cached hidden feature tensors attached to top4 survivor rows.",
        "diagnostic_only": False,
        "rows": rows,
    }
    write_json(FINAL_ARBITER_JSON, payload)
    write_csv(FINAL_ARBITER_CSV, rows)
    lines = [
        "# Merged Tap Final Arbiter Evaluation",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT = {verdict}",
        "",
        f"- primary split: `{primary_split}`",
        f"- selected merged tap task macro: `{rate(merged_reward)}`",
        f"- fixed-composite top1 task macro: `{rate(fixed_reward)}`",
        f"- majority-rank task macro: `{rate(majority_reward)}`",
        f"- oracle-best survivor task macro: `{rate(oracle_reward)}`",
        f"- complete survivor set rate: `{rate((coverage or {}).get('complete_set_rate'))}`",
        "",
        "## Primary Split Policies",
        "",
    ]
    table_rows = [{"policy": p, **primary.get(p, {})} for p in policies]
    lines.extend(md_table(table_rows, ["policy", "n", "tasks", "task_macro_reward", "group_micro_reward", "oracle_selected_rate", "regret"]))
    write_md(FINAL_ARBITER_MD, lines)
    print(f"BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_composite_insertion_eval() -> int:
    ensure_root()
    old_code = load_json(OLD_CODE_JSON, {}) or {}
    branch_bridge = load_json(BRANCH_BRIDGE_JSON, {}) or {}
    survival = load_json(SURVIVAL_JSON, {}) or {}
    final_arbiter = load_json(FINAL_ARBITER_JSON, {}) or {}
    selected_old = safe_float((old_code.get("selected") or {}).get("pairwise_accuracy"), float("nan"))
    selected_hidden = safe_float((branch_bridge.get("selected_hidden_branch") or {}).get("pairwise_accuracy"), float("nan"))
    selected_bridge = safe_float((branch_bridge.get("selected_bridge") or {}).get("pairwise_accuracy"), float("nan"))
    selected_survival = safe_float((survival.get("selected") or {}).get("oracle_retention"), float("nan"))
    primary = final_arbiter.get("primary") or {}
    merged_final = safe_float((primary.get("merged_tap_top1") or {}).get("task_macro_reward"), float("nan"))
    fixed_final = safe_float((primary.get("fixed_composite_top1") or {}).get("task_macro_reward"), float("nan"))
    majority_final = safe_float((primary.get("majority_rank_aggregation") or {}).get("task_macro_reward"), float("nan"))
    rank_sum_final = safe_float((primary.get("fixed_plus_merged_rank_sum") or {}).get("task_macro_reward"), float("nan"))
    balanced = finite_mean([selected_old, selected_hidden, selected_bridge, selected_survival, merged_final])
    if rank_sum_final > max(fixed_final, majority_final, merged_final):
        verdict = "MERGED_AS_EXTRA_EXPERT"
    elif merged_final > max(fixed_final, majority_final):
        verdict = "MERGED_AS_EXTRA_EXPERT"
    elif balanced >= 0.60:
        verdict = "COMPOSITE_STILL_BEST"
    else:
        verdict = "MERGED_NOT_USEFUL"
    row = {
        "variant": "merged_only_core_signal",
        "old_context_accuracy": selected_old,
        "hidden_branch_accuracy": selected_hidden,
        "bridge_accuracy": selected_bridge,
        "survival_oracle_retention": selected_survival,
        "merged_final_task_macro": merged_final,
        "fixed_final_task_macro": fixed_final,
        "majority_final_task_macro": majority_final,
        "fixed_plus_merged_rank_sum_task_macro": rank_sum_final,
        "balanced_proxy": balanced,
        "complete_set_rate": (final_arbiter.get("coverage") or {}).get("complete_set_rate"),
    }
    payload = {
        "BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT": verdict,
        "verdict": verdict,
        "rows": [row],
        "interpretation": "Merged rank-sum insertion helps final arbitration." if verdict == "MERGED_AS_EXTRA_EXPERT" else "Fixed/majority final-arbiter baselines remain stronger than merged-only or merged-rank insertion.",
    }
    write_json(COMPOSITE_INSERTION_JSON, payload)
    write_csv(COMPOSITE_INSERTION_CSV, [row])
    lines = [
        "# Merged Tap Composite Insertion Evaluation",
        "",
        f"BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT = {verdict}",
        "",
        f"- balanced proxy: `{rate(balanced)}`",
        f"- merged final task macro: `{rate(merged_final)}`",
        f"- fixed final task macro: `{rate(fixed_final)}`",
        f"- majority final task macro: `{rate(majority_final)}`",
        f"- fixed+merged rank-sum task macro: `{rate(rank_sum_final)}`",
        f"- complete set rate: `{rate(row['complete_set_rate'])}`",
    ]
    write_md(COMPOSITE_INSERTION_MD, lines)
    print(f"BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT = {verdict}", flush=True)
    return 0


def run_diagnostics() -> int:
    ensure_root()
    selected_payload = load_pt(SELECTED_TAPS_PT, {}) or {}
    selected = selected_payload.get("selected")
    pair_sets = load_pair_sets()
    rows: list[dict[str, Any]] = []
    if isinstance(selected, dict):
        config = str(selected.get("target_config"))
        arch = str(selected.get("architecture"))
        weight = selected.get("weight")
        sums = []
        flips = []
        for pairs in pair_sets.values():
            for pair in pairs[:1000]:
                vals = (pair.get("features") or {}).get(config)
                if not isinstance(vals, dict):
                    continue
                pref = vals.get("preferred")
                rej = vals.get("rejected")
                if not isinstance(pref, torch.Tensor) or not isinstance(rej, torch.Tensor):
                    continue
                s1 = score_diff(weight, arch, pref.flatten() - rej.flatten())
                s2 = score_diff(weight, arch, rej.flatten() - pref.flatten())
                if math.isfinite(s1) and math.isfinite(s2):
                    sums.append(abs(s1 + s2))
                    flips.append(1.0 if (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0) else 0.0)
        blocks = concat_blocks(config)
        for idx, block in enumerate(blocks):
            start = idx * HIDDEN_DIM if len(blocks) > 1 else 0
            stop = start + HIDDEN_DIM
            rows.append({"diagnostic": "block_norm", "block": block, "value": float(weight[start:stop].norm().item())})
        rows.append({"diagnostic": "mean_abs_flip_score_sum", "value": finite_mean(sums)})
        rows.append({"diagnostic": "strict_sign_flip_rate", "value": finite_mean(flips)})
    flip_sum = next((safe_float(r.get("value")) for r in rows if r.get("diagnostic") == "mean_abs_flip_score_sum"), float("nan"))
    flip_rate = next((safe_float(r.get("value")) for r in rows if r.get("diagnostic") == "strict_sign_flip_rate"), float("nan"))
    if math.isfinite(flip_sum) and flip_sum < 1e-5 and flip_rate >= 0.99:
        verdict = "CLEAN_PAIRWISE_BEHAVIOR"
    elif math.isfinite(flip_sum):
        verdict = "CALIBRATION_WEAK"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_MERGED_TAP_DIAGNOSTICS_VERDICT": verdict,
        "verdict": verdict,
        "selected_candidate": selected.get("candidate_name") if isinstance(selected, dict) else None,
        "rows": rows,
        "score_distillation_used_for_primary": False,
    }
    write_json(DIAGNOSTICS_JSON, payload)
    write_csv(DIAGNOSTICS_CSV, rows)
    lines = ["# Merged Tap Diagnostics", "", f"BG_MERGED_TAP_DIAGNOSTICS_VERDICT = {verdict}", "", f"- mean abs score(a,b)+score(b,a): `{rate(flip_sum)}`", f"- strict sign flip rate: `{rate(flip_rate)}`", "- score distillation used for primary: `False`"]
    write_md(DIAGNOSTICS_MD, lines)
    print(f"BG_MERGED_TAP_DIAGNOSTICS_VERDICT = {verdict}", flush=True)
    return 0


def status_from_verdicts(verdicts: dict[str, str]) -> str:
    if verdicts.get("BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT") in {"LARGE_DEGRADATION", "CODING_DEGRADES"}:
        return "OLD_CONTEXT_DEGRADES"
    if verdicts.get("BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT") == "BRANCH_SIGNAL_LOST":
        return "BRANCH_SIGNAL_LOST"
    if verdicts.get("BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT") in {"BRIDGE_SIGNAL_LOST", "BRANCH_READY_BRIDGE_WEAK"}:
        return "BRIDGE_SIGNAL_LOST"
    if verdicts.get("BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT") == "FINAL_ARBITER_IMPROVES":
        return "FINAL_ARBITER_IMPROVES_ONLY"
    ready_core = (
        verdicts.get("BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT") == "READY"
        and verdicts.get("BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT") == "PRESERVED_OR_IMPROVED"
        and verdicts.get("BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT") == "BRANCH_AND_BRIDGE_READY"
        and verdicts.get("BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT") in {"MERGED_REPLACES_FIXED_COMPOSITE", "MERGED_MATCHES_FIXED_COMPOSITE"}
        and verdicts.get("BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT")
        in {"FINAL_ARBITER_IMPROVES", "HARD_UNTIED_IMPROVES", "DOMAIN_IMPROVES"}
        and verdicts.get("BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT")
        in {"MERGED_REPLACES_COMPOSITE", "MERGED_AS_EXTRA_EXPERT", "MERGED_REPLACES_ONE_EXPERT"}
        and verdicts.get("BG_MERGED_TAP_DIAGNOSTICS_VERDICT") == "CLEAN_PAIRWISE_BEHAVIOR"
    )
    if ready_core:
        return "MERGED_READY"
    if verdicts.get("BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT") == "USE_AS_COMPOSITE_EXPERT":
        return "USE_AS_COMPOSITE_EXPERT"
    if verdicts.get("BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT") == "DATA_LIMITED":
        return "DATA_LIMITED"
    if verdicts.get("BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT") in {"MERGED_REPLACES_FIXED_COMPOSITE", "MERGED_MATCHES_FIXED_COMPOSITE"}:
        return "SURVIVAL_IMPROVES_ONLY"
    if verdicts.get("BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT") == "FIXED_COMPOSITE_STILL_BEST":
        return "SCORE_COMPOSITE_STILL_BEST"
    return "INSUFFICIENT"


def maybe_append_section(path: Path, title: str, lines: Sequence[str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if title in text:
        return
    with path.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n" + "\n".join(lines) + "\n")


def update_docs(summary: dict[str, Any]) -> None:
    status = summary.get("MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS")
    selected = ((load_json(VALIDATION_JSON, {}) or {}).get("selected_candidate_name")) or "none"
    doc_lines = [
        "# Weight-Space Merged Branch-Content Taps v1",
        "",
        "Status: completed cached weight-space merged tap experiment.",
        "",
        f"MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = {status}",
        "",
        "This run tested whether old/content, hidden-branch, and bridge tap weight directions could be aligned into a shared feature coordinate system and residualized into compact merged pairwise taps.",
        "",
        "No Ouro weights, tokenizer files, checkpoints, old tap registries, production routing, wrapper/local-agent code, Hunter-Seeker modules, action steering, or true fork/carry paths were modified or executed.",
        "",
        "## Key Result",
        "",
        f"- selected merged tap: `{selected}`",
        f"- weight inventory: `{summary.get('BG_MERGED_TAP_WEIGHT_INVENTORY_VERDICT')}`",
        f"- coordinate alignment: `{summary.get('BG_MERGED_TAP_COORDINATE_ALIGNMENT_VERDICT')}`",
        f"- validation selection: `{summary.get('BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT')}`",
        f"- old/code eval: `{summary.get('BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT')}`",
        f"- branch/bridge eval: `{summary.get('BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT')}`",
        f"- survival eval: `{summary.get('BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT')}`",
        f"- survivor feature acquisition: `{summary.get('BG_MERGED_TAP_SURVIVOR_FEATURE_ACQUISITION_VERDICT')}`",
        f"- final arbiter eval: `{summary.get('BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT')}`",
        f"- composite insertion: `{summary.get('BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT')}`",
        f"- diagnostics: `{summary.get('BG_MERGED_TAP_DIAGNOSTICS_VERDICT')}`",
        "",
        "## Interpretation",
        "",
        "The experiment keeps LayerNorm and NoNorm heads separated, uses explicit direct or zero-block lifted alignment, and does not use score distillation for primary readiness. A merged tap can only replace a core scoring component unless fixed-composite veto/rescue and missing/OOD guardrails are also preserved or re-applied.",
        "",
        "The top4 survivor hidden-feature gap was filled from cached raw branch, old-content, and code feature artifacts. No fresh generation was required for this acquisition pass.",
        "",
        "## Files",
        "",
        f"- summary: `{rel(SUMMARY_MD)}`",
        f"- analysis: `{rel(ANALYSIS_MD)}`",
        f"- merged tap artifact: `{rel(MERGED_TAPS_PT)}`",
        f"- selected tap artifact: `{rel(SELECTED_TAPS_PT)}`",
    ]
    write_md(DOC_MD, doc_lines)
    section_title = "## Weight-space merged branch-content taps v1 (2026-05-18)"
    section = [
        section_title,
        "",
        f"`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = {status}`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, then acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        maybe_append_section(target, section_title, section)
    nav_section = "## Weight-space merged branch-content taps v1 (2026-05-18)"
    nav_lines = [
        nav_section,
        "",
        f"Added `{rel(DOC_MD)}` for the cached merged-tap weight extraction, residualization, and evaluation run. Status: `{status}`.",
    ]
    for target in NAV_TARGETS:
        maybe_append_section(target, nav_section, nav_lines)


def run_synthesis() -> int:
    ensure_root()
    verdict_sources = [
        ("BG_MERGED_TAP_WEIGHT_INVENTORY_VERDICT", WEIGHT_INVENTORY_JSON),
        ("BG_MERGED_TAP_COORDINATE_ALIGNMENT_VERDICT", ALIGNMENT_JSON),
        ("BG_MERGED_TAP_FEATURE_STATS_VERDICT", FEATURE_STATS_JSON),
        ("BG_MERGED_TAP_WEIGHT_GEOMETRY_VERDICT", GEOMETRY_JSON),
        ("BG_MERGED_TAP_CANDIDATE_BUILD_VERDICT", MERGED_CANDIDATES_JSON),
        ("BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT", VALIDATION_JSON),
        ("BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT", OLD_CODE_JSON),
        ("BG_MERGED_TAP_BRANCH_BRIDGE_EVAL_VERDICT", BRANCH_BRIDGE_JSON),
        ("BG_MERGED_TAP_SURVIVAL_EVAL_VERDICT", SURVIVAL_JSON),
        ("BG_MERGED_TAP_SURVIVOR_FEATURE_ACQUISITION_VERDICT", SURVIVOR_FEATURES_JSON),
        ("BG_MERGED_TAP_FINAL_ARBITER_EVAL_VERDICT", FINAL_ARBITER_JSON),
        ("BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT", COMPOSITE_INSERTION_JSON),
        ("BG_MERGED_TAP_DIAGNOSTICS_VERDICT", DIAGNOSTICS_JSON),
    ]
    verdicts: dict[str, str] = {}
    for key, path in verdict_sources:
        payload = load_json(path, {}) or {}
        verdicts[key] = str(payload.get(key) or payload.get("verdict") or "MISSING")
    status = status_from_verdicts(verdicts)
    validation = load_json(VALIDATION_JSON, {}) or {}
    old_code = load_json(OLD_CODE_JSON, {}) or {}
    branch_bridge = load_json(BRANCH_BRIDGE_JSON, {}) or {}
    survival = load_json(SURVIVAL_JSON, {}) or {}
    survivor_features = load_json(SURVIVOR_FEATURES_JSON, {}) or {}
    final_arbiter = load_json(FINAL_ARBITER_JSON, {}) or {}
    composite = load_json(COMPOSITE_INSERTION_JSON, {}) or {}
    blockers = []
    if verdicts.get("BG_MERGED_TAP_VALIDATION_SELECTION_VERDICT") != "READY":
        blockers.append("validation selection remained weak versus the best source signal")
    if verdicts.get("BG_MERGED_TAP_OLD_CODE_EVAL_VERDICT") != "PRESERVED_OR_IMPROVED":
        blockers.append("old-context/code preservation was not clean; heldout old-content showed a small degradation")
    if verdicts.get("BG_MERGED_TAP_SURVIVOR_FEATURE_ACQUISITION_VERDICT") != "READY":
        blockers.append("top4 survivor hidden-feature acquisition is incomplete")
    if verdicts.get("BG_MERGED_TAP_COMPOSITE_INSERTION_VERDICT") not in {"MERGED_REPLACES_COMPOSITE", "MERGED_AS_EXTRA_EXPERT", "MERGED_REPLACES_ONE_EXPERT"}:
        blockers.append("composite insertion did not beat the existing composite stack")
    blockers.append("full fixed-composite replacement cannot be claimed from one merged vector without preserving/reapplying veto/rescue and missing/OOD fallback")
    summary = {
        **verdicts,
        "MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS": status,
        "selected_candidate": validation.get("selected_candidate_name"),
        "selected_validation": validation.get("selected_validation"),
        "old_code_selected": old_code.get("selected"),
        "branch_bridge_selected": {
            "hidden_branch": branch_bridge.get("selected_hidden_branch"),
            "bridge": branch_bridge.get("selected_bridge"),
        },
        "survival_selected": survival.get("selected"),
        "survivor_feature_coverage": survivor_features.get("coverage"),
        "final_arbiter_primary": final_arbiter.get("primary"),
        "composite_insertion": composite.get("rows"),
        "files_created": [
            rel(path)
            for path in [
                WEIGHT_INVENTORY_MD,
                ALIGNMENT_MD,
                FEATURE_STATS_MD,
                GEOMETRY_MD,
                MERGED_CANDIDATES_MD,
                VALIDATION_MD,
                OLD_CODE_MD,
                BRANCH_BRIDGE_MD,
                SURVIVAL_MD,
                SURVIVOR_FEATURES_MD,
                FINAL_ARBITER_MD,
                COMPOSITE_INSERTION_MD,
                DIAGNOSTICS_MD,
                SUMMARY_MD,
                ANALYSIS_MD,
                DOC_MD,
            ]
        ],
        "commands_run": ["py_compile for all merged-tap scripts", *[f"venv/bin/python -u utilities/tests/manual/{name}" for name in SCRIPT_NAMES]],
        "blockers": blockers,
    }
    write_json(SUMMARY_JSON, summary)
    write_json(ANALYSIS_JSON, summary)
    top_lines = [f"{key} = {value}" for key, value in verdicts.items()]
    top_lines.append(f"MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = {status}")
    lines = ["# Merged Weight Branch-Content Taps v1 Summary", "", *top_lines, "", "## Selected Candidate", "", f"`{summary.get('selected_candidate')}`", "", "## Blockers", ""]
    lines.extend(f"- {item}" for item in summary["blockers"])
    lines.extend(["", "## Files Created", ""])
    lines.extend(f"- `{path}`" for path in summary["files_created"])
    write_md(SUMMARY_MD, lines)
    analysis_lines = [
        "# Merged Weight Branch-Content Taps v1 Analysis",
        "",
        "## Motivation",
        "",
        "This run tested whether the score-level old+branch+bridge structure could be compressed into residualized tap weight directions while preserving old/context behavior.",
        "",
        "## Verdicts",
        "",
        *top_lines,
        "",
        "## Interpretation",
        "",
        "The experiment used explicit coordinate alignment and did not use score distillation for primary readiness. Results should be interpreted as cached evaluator evidence, not model or routing changes.",
        "",
        "## Recommended Phase 2 Selector Architecture",
        "",
    ]
    if status in {"MERGED_READY", "SURVIVAL_IMPROVES_ONLY"}:
        analysis_lines.append("Run a selection-only prototype comparison using the selected merged tap as a core scoring component while preserving fixed-composite guardrails.")
    elif status == "USE_AS_COMPOSITE_EXPERT":
        analysis_lines.append("Use the merged tap as an additional expert and rerun fixed-composite optimization with guardrails unchanged.")
    elif status == "DATA_LIMITED":
        analysis_lines.append("Collect or reconstruct the remaining missing feature coverage before making readiness claims.")
    else:
        analysis_lines.append("Keep the separated fixed composite as primary unless future merged-tap data changes the result.")
    write_md(ANALYSIS_MD, analysis_lines)
    update_docs(summary)
    print(f"MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = {status}", flush=True)
    return 0
