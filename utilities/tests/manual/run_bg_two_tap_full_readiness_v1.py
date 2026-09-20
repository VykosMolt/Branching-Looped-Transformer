"""Two old-anchored taps full readiness probe v1.

This run tests whether the two old-anchored transplanted taps
(`coding_reasoning` and `mixed_objective_all`) can stand alone as the only
tap family for both old-domain selection and branch selection.

It only reads cached feature/tap artifacts and writes a new report. It does
not train Ouro, modify checkpoints/tokenizers, overwrite tap registries, run
wrapper/local-agent or Hunter-Seeker code, apply steering, or change routing.
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

from bg_gated_selector_v1_common import feature_map_from_old_code_pooled, load_old_code_pairs
from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, config_dim, md_table, rel
from bg_merged_tap_v1_common import (
    BRIDGE_PT,
    EXTRACTED_WEIGHTS_PT,
    GATED_HEADS_PT,
    GENERATOR_HEADS_PT,
    GENERATOR_TAP_HEADS_PT,
    HEAD_REGISTRY_PT,
    HIDDEN_BRANCH_PT,
    HIDDEN_TAPS_PT,
    MIXED_HEADS_PT,
    OLD_CONTENT_PT,
    PRIMARY_TARGET,
    QUOTA_V4_BRANCHES_PT,
    SALVAGE_HEADS_PT,
    UNIVERSAL_HEADS_PT,
    V4_HEADS_PT,
    compact_head_name,
    eval_candidate_on_pairs,
    extract_weight,
    finite_mean,
    head_rows_from_payload,
    infer_family,
    json_default,
    load_pt,
    metric_score,
    pair_diff,
    rate,
    safe_float,
    score_diff,
)


OUT_ROOT = PROBE_ROOT / "bg_two_tap_full_readiness_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "two_tap_full_readiness.json"
REPORT_MD = OUT_ROOT / "two_tap_full_readiness.md"
PAIR_ROWS_CSV = OUT_ROOT / "two_tap_full_readiness_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "two_tap_full_readiness_group_rows.csv"
DATASET_INVENTORY_CSV = OUT_ROOT / "dataset_inventory.csv"
ARTIFACT_PT = OUT_ROOT / "two_tap_full_readiness_v1.pt"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_two_tap_full_readiness_v1.md"

OLD_ANCHOR_ROOT = PROBE_ROOT / "bg_old_anchored_branch_valid_taps_v1_2026-05-30"
OLD_ANCHOR_PT = OLD_ANCHOR_ROOT / "old_anchored_branch_valid_taps_v1.pt"

CODE_TAP_NAME = "transplant::MIX_CODE_REASONING::full_residual::AntisymLinearNoNorm::102"
OBJECTIVE_TAP_NAME = "transplant::MIX_OBJECTIVE_ALL::full_residual::AntisymLinearNoNorm::135"

REASONING_NATURAL_PT = PROBE_ROOT / "reasoning_natural_distractor_features_2026-05-17.pt"
REASONING_TRACE_PT = PROBE_ROOT / "reasoning_trace_features_2026-05-17.pt"
SCIENCE_NATURAL_PT = PROBE_ROOT / "science_natural_distractor_features_2026-05-17.pt"
GSM8K_EXPANDED_PT = PROBE_ROOT / "clean_gsm8k_expanded_tap_features_2026-05-16.pt"
GSM8K_EXTREME_PT = PROBE_ROOT / "clean_gsm8k_extreme_tap_features_2026-05-16.pt"
HIDDEN_V1_DATASET_PT = PROBE_ROOT / "bg_hidden_origin_taps_2026-05-18/hidden_origin_tap_dataset.pt"
HIDDEN_V2_DATASET_PT = PROBE_ROOT / "bg_hidden_origin_diversity_v2_2026-05-18/hidden_origin_tap_dataset_v2.pt"
HIDDEN_V3_DATASET_PT = PROBE_ROOT / "bg_hidden_origin_diversity_v3_2026-05-18/hidden_origin_tap_dataset_v3.pt"
HIDDEN_V4_DATASET_PT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18/hidden_origin_quota_dataset_v4.pt"
GENERATOR_V1_DATASET_PT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18/selector_dataset.pt"
SALVAGE_DATASETS_PT = PROBE_ROOT / "bg_hidden_origin_split_salvage_2026-05-18/salvage_datasets.pt"
FIXED_SURVIVAL_DATASET_PT = PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18/survival_dataset.pt"
SURVIVOR_FEATURES_PT = PROBE_ROOT / "bg_merged_weight_branch_content_taps_v1_2026-05-18/top4_survivor_hidden_features.pt"

EXTRA_HEAD_SPECS = [
    ("hidden_origin_v2", PROBE_ROOT / "bg_hidden_origin_diversity_v2_2026-05-18/hidden_origin_tap_heads_v2.pt"),
    ("hidden_origin_v3", PROBE_ROOT / "bg_hidden_origin_diversity_v3_2026-05-18/hidden_origin_tap_heads_v3.pt"),
]

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_two_tap_branch_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_merged_weight_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
]

NAV_TARGETS = [
    PROJECT_ROOT / "shared/docs/evaluator/README.md",
    PROJECT_ROOT / "shared/docs/README.md",
    PROJECT_ROOT / "PROJECT_TREE_MAP.md",
    PROJECT_ROOT / "PROJECT_COMPONENTS.md",
]

SOURCE_HEAD_SPECS = [
    ("old_registry", HEAD_REGISTRY_PT),
    ("old_mixed_domain", MIXED_HEADS_PT),
    ("universal", UNIVERSAL_HEADS_PT),
    ("hidden_origin_v1", HIDDEN_TAPS_PT),
    ("hidden_origin_v2", EXTRA_HEAD_SPECS[0][1]),
    ("hidden_origin_v3", EXTRA_HEAD_SPECS[1][1]),
    ("hidden_origin_v4", V4_HEADS_PT),
    ("branch_generator_v1_selector", GENERATOR_HEADS_PT),
    ("branch_generator_v1_hidden_tap", GENERATOR_TAP_HEADS_PT),
    ("hidden_origin_salvage", SALVAGE_HEADS_PT),
    ("gated_selector", GATED_HEADS_PT),
]

SUPPORTED_HEAD_CONFIGS = {
    "24_L4",
    "24_mean",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "concat_24_36",
    "concat_36_47",
    "concat_24_36_47",
    "30_L4",
    "42_L4",
    "concat_24_30_36",
    "concat_36_42_47",
}


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


def append_section(path: Path, title: str, lines: Sequence[str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if title in text:
        return
    with path.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n" + "\n".join(lines) + "\n")


def tensor_weight(candidate: dict[str, Any]) -> torch.Tensor | None:
    weight = candidate.get("weight")
    if isinstance(weight, torch.Tensor):
        return weight.detach().cpu().to(torch.float32).flatten()
    state = candidate.get("state_dict") or {}
    if isinstance(state, dict):
        value = state.get("linear.weight") or state.get("weight") or state.get("score.weight")
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(torch.float32).flatten()
    return None


def normed(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = vec.detach().cpu().to(torch.float32).flatten()
    n = float(x.norm().item())
    return x / n if n >= eps else torch.zeros_like(x)


def load_two_taps() -> list[dict[str, Any]]:
    payload = torch.load(OLD_ANCHOR_PT, map_location="cpu", weights_only=False)
    candidates = list(payload.get("candidates") or [])
    code = next(c for c in candidates if c.get("candidate_name") == CODE_TAP_NAME)
    objective = next(c for c in candidates if c.get("candidate_name") == OBJECTIVE_TAP_NAME)
    code_w = tensor_weight(code)
    obj_w = tensor_weight(objective)
    if not isinstance(code_w, torch.Tensor) or not isinstance(obj_w, torch.Tensor):
        raise RuntimeError("missing two-tap weights")
    fused = normed(normed(code_w) + normed(obj_w))
    return [
        {
            "candidate_name": "two_tap::coding_reasoning",
            "candidate_family": "two_tap",
            "tap_role": "coding_reasoning",
            "target_config": PRIMARY_TARGET,
            "architecture": str(code.get("architecture") or "AntisymLinearNoNorm"),
            "weight": code_w,
            "state_dict": {"linear.weight": code_w.reshape(1, -1)},
        },
        {
            "candidate_name": "two_tap::mixed_objective_all",
            "candidate_family": "two_tap",
            "tap_role": "mixed_objective_all",
            "target_config": PRIMARY_TARGET,
            "architecture": str(objective.get("architecture") or "AntisymLinearNoNorm"),
            "weight": obj_w,
            "state_dict": {"linear.weight": obj_w.reshape(1, -1)},
        },
        {
            "candidate_name": "two_tap::equal_weight_fused_direction",
            "candidate_family": "two_tap",
            "tap_role": "equal_weight_fused_direction",
            "target_config": PRIMARY_TARGET,
            "architecture": "AntisymLinearNoNorm",
            "weight": fused,
            "state_dict": {"linear.weight": fused.reshape(1, -1)},
        },
    ]


def source_candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for source_label, path in SOURCE_HEAD_SPECS:
        payload = load_pt(path, None)
        for idx, raw in enumerate(head_rows_from_payload(payload)):
            row = dict(raw)
            arch = str(row.get("architecture") or row.get("model_family") or row.get("family") or "")
            config = str(row.get("config") or row.get("feature_config") or "")
            if arch not in {"AntisymLinear", "AntisymLinearNoNorm"} or config not in SUPPORTED_HEAD_CONFIGS:
                continue
            try:
                dim = config_dim(config)
            except Exception:
                continue
            weight, key = extract_weight(row)
            if not isinstance(weight, torch.Tensor) or int(weight.numel()) != dim:
                continue
            family, support = infer_family(source_label, row)
            if family not in {"old_content", "branch", "bridge", "universal"}:
                continue
            name = f"{compact_head_name(row, source_label, idx)}::row={idx}"
            dedup = (name, family, arch, config, int(weight.numel()))
            if dedup in seen:
                continue
            seen.add(dedup)
            out.append(
                {
                    "candidate_name": f"source::{name}",
                    "candidate_family": f"source_{family}",
                    "source_family": family,
                    "pair_type_support": support,
                    "source_run": source_label,
                    "target_config": config,
                    "architecture": arch,
                    "weight": weight.detach().cpu().to(torch.float32).flatten(),
                    "state_dict": {"linear.weight": weight.detach().cpu().to(torch.float32).flatten().reshape(1, -1)},
                    "metric_score": metric_score(row),
                    "extractable_weight_key": key,
                }
            )
    return out


def feature_pair(preferred: dict[str, torch.Tensor], rejected: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for config in sorted(set(preferred) & set(rejected)):
        try:
            dim = config_dim(config)
        except Exception:
            continue
        pv = preferred.get(config)
        rv = rejected.get(config)
        if isinstance(pv, torch.Tensor) and isinstance(rv, torch.Tensor) and int(pv.numel()) == int(rv.numel()) == dim:
            out[config] = {
                "preferred": pv.detach().cpu().to(torch.float32).flatten(),
                "rejected": rv.detach().cpu().to(torch.float32).flatten(),
            }
    return out


def pairs_from_mcq_feature_file(path: Path, dataset_name: str, domain: str) -> list[dict[str, Any]]:
    payload = load_pt(path, {}) or {}
    index: dict[str, dict[str, Any]] = {}
    fmap_index: dict[str, dict[str, torch.Tensor]] = {}
    for row in payload.get("candidate_features") or []:
        uid = str(row.get("candidate_uid"))
        if not uid:
            continue
        index[uid] = row
        fmap_index[uid] = feature_map_from_old_code_pooled(row.get("pooled"))
    pairs: list[dict[str, Any]] = []
    for set_name, eval_rows in (payload.get("eval_sets") or {}).items():
        for task in eval_rows or []:
            uids = [str(uid) for uid in task.get("candidate_uids") or []]
            labels = [str(label).lower() for label in task.get("labels") or []]
            if not labels or len(labels) != len(uids):
                labels = []
                for uid in uids:
                    meta = (index.get(uid) or {}).get("candidate_metadata") or {}
                    labels.append("correct" if bool(meta.get("is_correct")) else "incorrect")
            correct = [i for i, label in enumerate(labels) if label == "correct"]
            wrong = [i for i, label in enumerate(labels) if label != "correct"]
            for i in correct:
                for j in wrong:
                    pref = fmap_index.get(uids[i]) or {}
                    rej = fmap_index.get(uids[j]) or {}
                    features = feature_pair(pref, rej)
                    if not features:
                        continue
                    pairs.append(
                        {
                            "pair_id": f"{dataset_name}::{set_name}::{task.get('task_id')}::{i}>{j}",
                            "pair_type": "old_domain",
                            "source_id": dataset_name,
                            "source_dataset": dataset_name,
                            "variant": set_name,
                            "domain": domain,
                            "task_id": str(task.get("task_id")),
                            "group_id": f"{dataset_name}::{task.get('task_id')}",
                            "split": "domain_probe",
                            "label_source": "mcq_correct_answer",
                            "reward_preferred": 1.0,
                            "reward_rejected": 0.0,
                            "reward_gap": 1.0,
                            "features": features,
                            "available_configs": sorted(features.keys()),
                        }
                    )
    return pairs


def pairs_from_gsm_file(path: Path, dataset_name: str) -> list[dict[str, Any]]:
    payload = load_pt(path, {}) or {}
    pairs: list[dict[str, Any]] = []
    for record in payload.get("records") or []:
        pooled = record.get("pooled")
        labels = record.get("labels")
        if not isinstance(pooled, torch.Tensor):
            continue
        try:
            bools = [bool(x) for x in labels.tolist()]
        except Exception:
            bools = [bool(x) for x in labels or []]
        if pooled.ndim != 4 or len(bools) != int(pooled.shape[0]):
            continue
        maps = [feature_map_from_old_code_pooled(pooled[idx]) for idx in range(int(pooled.shape[0]))]
        correct = [i for i, value in enumerate(bools) if value]
        wrong = [i for i, value in enumerate(bools) if not value]
        for i in correct:
            for j in wrong:
                features = feature_pair(maps[i], maps[j])
                if not features:
                    continue
                task_id = f"{record.get('source')}::{record.get('problem_id')}::{record.get('tournament_id')}"
                pairs.append(
                    {
                        "pair_id": f"{dataset_name}::{task_id}::{i}>{j}",
                        "pair_type": "old_domain",
                        "source_id": dataset_name,
                        "source_dataset": dataset_name,
                        "variant": "gsm8k_candidate_tournament",
                        "domain": "math_simple_arithmetic",
                        "task_id": task_id,
                        "group_id": f"{dataset_name}::{task_id}",
                        "split": "domain_probe",
                        "label_source": "gsm8k_parser_correctness",
                        "reward_preferred": 1.0,
                        "reward_rejected": 0.0,
                        "reward_gap": 1.0,
                        "features": features,
                        "available_configs": sorted(features.keys()),
                    }
                )
    return pairs


def clone_pairs(pairs: Iterable[dict[str, Any]], dataset_name: str, default_pair_type: str, readiness_eligible: bool = True) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        row = dict(pair)
        row.setdefault("source_dataset", dataset_name)
        row.setdefault("source_id", dataset_name)
        row.setdefault("pair_type", default_pair_type)
        row.setdefault("split", row.get("split") or "all")
        row["readiness_eligible"] = readiness_eligible
        out.append(row)
    return out


def branch_pair_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []

    def add_from_pairs(name: str, path: Path, pair_type: str, readiness: bool = True) -> None:
        payload = load_pt(path, {}) or {}
        pairs = clone_pairs(payload.get("pairs") or [], name, pair_type, readiness)
        datasets.append({"dataset_name": name, "dataset_kind": "branch_pair", "path": path, "pairs": pairs, "readiness_eligible": readiness})

    add_from_pairs("universal_hidden_branch_pairs", HIDDEN_BRANCH_PT, "hidden_branch", True)
    add_from_pairs("universal_bridge_pairs", BRIDGE_PT, "bridge", True)
    add_from_pairs("hidden_origin_v1_pairs", HIDDEN_V1_DATASET_PT, "hidden_branch", True)

    for name, path in (
        ("hidden_origin_v2", HIDDEN_V2_DATASET_PT),
        ("hidden_origin_v3", HIDDEN_V3_DATASET_PT),
        ("hidden_origin_quota_v4", HIDDEN_V4_DATASET_PT),
        ("branch_generator_v1", GENERATOR_V1_DATASET_PT),
    ):
        payload = load_pt(path, {}) or {}
        by_variant = payload.get("pairs_by_variant")
        if not isinstance(by_variant, dict):
            datasets.append({"dataset_name": f"{name}::pairs", "dataset_kind": "branch_pair", "path": path, "pairs": clone_pairs(payload.get("pairs") or [], f"{name}::pairs", "hidden_branch", True), "readiness_eligible": True})
            continue
        for variant, pairs in by_variant.items():
            diagnostic = any(token in str(variant).lower() for token in ("alpha_0_02", "sampled", "diagnostic", "l47", "true_fork"))
            datasets.append(
                {
                    "dataset_name": f"{name}::{variant}",
                    "dataset_kind": "branch_pair",
                    "path": path,
                    "variant": variant,
                    "pairs": clone_pairs(pairs or [], f"{name}::{variant}", "hidden_branch", not diagnostic),
                    "readiness_eligible": not diagnostic,
                }
            )

    salvage = load_pt(SALVAGE_DATASETS_PT, {}) or {}
    for mode_name, mode in (salvage.get("modes") or {}).items():
        by_variant = mode.get("pairs_by_variant") if isinstance(mode, dict) else None
        if not isinstance(by_variant, dict):
            continue
        for variant, pairs in by_variant.items():
            diagnostic_variant = any(token in str(variant).lower() for token in ("alpha", "sampled", "diagnostic", "l47"))
            diagnostic_mode = not (
                str(mode_name).startswith("strict_cross_version_clean::main")
                or str(mode_name).startswith("v3_clean::main")
                or str(mode_name).startswith("old_frozen_tap_clean::all_signal_tasks")
            )
            readiness = not diagnostic_variant and not diagnostic_mode
            datasets.append(
                {
                    "dataset_name": f"salvage::{mode_name}::{variant}",
                    "dataset_kind": "branch_pair",
                    "path": SALVAGE_DATASETS_PT,
                    "variant": variant,
                    "mode_name": mode_name,
                    "pairs": clone_pairs(pairs or [], f"salvage::{mode_name}::{variant}", "hidden_branch", readiness),
                    "readiness_eligible": readiness,
                }
            )
    return datasets


def old_domain_pair_datasets() -> list[dict[str, Any]]:
    old_payload = load_pt(OLD_CONTENT_PT, {}) or {}
    code_pairs = load_old_code_pairs()
    for pair in code_pairs:
        pair["pair_type"] = "old_code"
    return [
        {
            "dataset_name": "universal_old_content_pairs",
            "dataset_kind": "old_domain_pair",
            "path": OLD_CONTENT_PT,
            "pairs": clone_pairs(old_payload.get("pairs") or [], "universal_old_content_pairs", "old_content", True),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "old_code_pairs",
            "dataset_kind": "old_domain_pair",
            "path": "load_old_code_pairs",
            "pairs": clone_pairs(code_pairs, "old_code_pairs", "old_code", True),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "reasoning_natural_distractors",
            "dataset_kind": "old_domain_pair",
            "path": REASONING_NATURAL_PT,
            "pairs": pairs_from_mcq_feature_file(REASONING_NATURAL_PT, "reasoning_natural_distractors", "reasoning"),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "reasoning_trace_primary",
            "dataset_kind": "old_domain_pair",
            "path": REASONING_TRACE_PT,
            "pairs": pairs_from_mcq_feature_file(REASONING_TRACE_PT, "reasoning_trace_primary", "reasoning"),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "science_natural_distractors",
            "dataset_kind": "old_domain_pair",
            "path": SCIENCE_NATURAL_PT,
            "pairs": pairs_from_mcq_feature_file(SCIENCE_NATURAL_PT, "science_natural_distractors", "science"),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "gsm8k_expanded_candidates",
            "dataset_kind": "old_domain_pair",
            "path": GSM8K_EXPANDED_PT,
            "pairs": pairs_from_gsm_file(GSM8K_EXPANDED_PT, "gsm8k_expanded_candidates"),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "gsm8k_extreme_candidates",
            "dataset_kind": "old_domain_pair",
            "path": GSM8K_EXTREME_PT,
            "pairs": pairs_from_gsm_file(GSM8K_EXTREME_PT, "gsm8k_extreme_candidates"),
            "readiness_eligible": True,
        },
    ]


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


def pair_eval_rows(dataset: dict[str, Any], candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs, eval_split = filter_pairs_for_primary(dataset.get("pairs") or [])
    rows_by_name: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        rows_by_name[str(candidate.get("candidate_name"))] = {
            "dataset_name": dataset["dataset_name"],
            "dataset_kind": dataset["dataset_kind"],
            "readiness_eligible": bool(dataset.get("readiness_eligible")),
            "eval_split": eval_split,
            "candidate_name": candidate.get("candidate_name"),
            "candidate_family": candidate.get("candidate_family"),
            "source_family": candidate.get("source_family"),
            "source_run": candidate.get("source_run"),
            "tap_role": candidate.get("tap_role"),
            "target_config": candidate.get("target_config"),
            "architecture": candidate.get("architecture"),
            "pair_count": 0,
            "pairwise_accuracy": float("nan"),
            "mean_margin": float("nan"),
            "domain_accuracy": {},
        }
    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        weight = tensor_weight(candidate)
        config = str(candidate.get("target_config"))
        arch = str(candidate.get("architecture"))
        if isinstance(weight, torch.Tensor):
            candidates_by_key[(config, arch)].append(candidate)
    for (config, arch), key_candidates in candidates_by_key.items():
        diffs = []
        domains = []
        for pair in pairs:
            diff = pair_diff(pair, config)
            if diff is None:
                continue
            diffs.append(diff)
            domains.append(str(pair.get("domain")))
        if not diffs:
            continue
        x = torch.stack(diffs, dim=0).to(torch.float32)
        if arch == "AntisymLinear":
            x = F.layer_norm(x, (x.shape[1],))
        valid_candidates = []
        weights = []
        for candidate in key_candidates:
            weight = tensor_weight(candidate)
            if isinstance(weight, torch.Tensor) and int(weight.numel()) == x.shape[1]:
                valid_candidates.append(candidate)
                weights.append(weight.to(torch.float32).flatten())
        if not weights:
            continue
        w = torch.stack(weights, dim=0)
        scores = x @ w.T
        domains_unique = sorted(set(domains))
        for col, candidate in enumerate(valid_candidates):
            sc = scores[:, col]
            correct = (sc > 0).to(torch.float32)
            ties = (sc == 0).to(torch.float32)
            acc = float((correct.sum() + 0.5 * ties.sum()).item() / max(int(sc.numel()), 1))
            domain_accuracy = {}
            for domain in domains_unique:
                idxs = [idx for idx, value in enumerate(domains) if value == domain]
                if not idxs:
                    continue
                subset = sc[idxs]
                domain_accuracy[domain] = float((((subset > 0).to(torch.float32).sum() + 0.5 * (subset == 0).to(torch.float32).sum()) / max(int(subset.numel()), 1)).item())
            row = rows_by_name[str(candidate.get("candidate_name"))]
            row["pair_count"] = int(sc.numel())
            row["pairwise_accuracy"] = acc
            row["mean_margin"] = float(sc.mean().item()) if sc.numel() else float("nan")
            row["domain_accuracy"] = domain_accuracy
    rows: list[dict[str, Any]] = list(rows_by_name.values())
    return rows


def summarize_pair_dataset(rows: Sequence[dict[str, Any]], dataset_kind: str) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_kind") == dataset_kind:
            by_dataset[str(row.get("dataset_name"))].append(row)
    summaries = []
    for dataset_name, vals in sorted(by_dataset.items()):
        eligible = bool(vals[0].get("readiness_eligible")) if vals else False
        two_rows = [r for r in vals if r.get("candidate_family") == "two_tap" and safe_float(r.get("pairwise_accuracy"), float("nan")) == safe_float(r.get("pairwise_accuracy"), float("nan"))]
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
                "best_two_tap_accuracy": two_acc,
                "best_reference": best_ref.get("candidate_name"),
                "best_reference_family": best_ref.get("candidate_family"),
                "best_reference_accuracy": ref_acc,
                "delta_two_minus_reference": two_acc - ref_acc if math.isfinite(two_acc) and math.isfinite(ref_acc) else float("nan"),
                "matches_or_exceeds_reference": bool(math.isfinite(two_acc) and math.isfinite(ref_acc) and two_acc + 1e-9 >= ref_acc),
            }
        )
    return {"datasets": summaries}


def candidate_feature(row: dict[str, Any], config: str) -> torch.Tensor | None:
    fmap = row.get("features_by_config") or row.get("features") or {}
    value = fmap.get(config)
    if isinstance(value, torch.Tensor):
        try:
            if int(value.numel()) == config_dim(config):
                return value.detach().cpu().to(torch.float32).flatten()
        except Exception:
            return None
    return None


def group_reward(row: dict[str, Any]) -> float:
    for key in ("reward", "final_reward", "correctness"):
        value = row.get(key)
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            return x
    return 0.0


def source_group_rank(candidate: dict[str, Any], group: Sequence[dict[str, Any]]) -> list[int]:
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    weight = tensor_weight(candidate)
    if not isinstance(weight, torch.Tensor):
        return []
    scores = []
    for i, left in enumerate(group):
        lvec = candidate_feature(left, config)
        if not isinstance(lvec, torch.Tensor):
            return []
        vals = []
        for j, right in enumerate(group):
            if i == j:
                continue
            rvec = candidate_feature(right, config)
            if not isinstance(rvec, torch.Tensor):
                return []
            vals.append(score_diff(weight, arch, lvec - rvec))
        scores.append((i, finite_mean(vals)))
    return [idx for idx, score in sorted(scores, key=lambda item: (-safe_float(item[1], -1e9), item[0]))]


def normalize_scores(scores: Sequence[float]) -> list[float]:
    vals = [safe_float(score, 0.0) for score in scores]
    if not vals:
        return []
    mu = mean(vals)
    var = mean((v - mu) ** 2 for v in vals)
    sd = math.sqrt(var)
    if sd < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / sd for v in vals]


def two_tap_group_scores(group: Sequence[dict[str, Any]], code_tap: dict[str, Any], objective_tap: dict[str, Any]) -> list[float]:
    def tap_scores(tap: dict[str, Any]) -> list[float]:
        config = str(tap.get("target_config"))
        arch = str(tap.get("architecture"))
        weight = tensor_weight(tap)
        if not isinstance(weight, torch.Tensor):
            return []
        out = []
        for i, left in enumerate(group):
            lvec = candidate_feature(left, config)
            if not isinstance(lvec, torch.Tensor):
                return []
            vals = []
            for j, right in enumerate(group):
                if i == j:
                    continue
                rvec = candidate_feature(right, config)
                if not isinstance(rvec, torch.Tensor):
                    return []
                vals.append(score_diff(weight, arch, lvec - rvec))
            out.append(finite_mean(vals))
        return out

    code_scores = normalize_scores(tap_scores(code_tap))
    obj_scores = normalize_scores(tap_scores(objective_tap))
    if not code_scores or not obj_scores or len(code_scores) != len(obj_scores):
        return []
    return [(c + o) * 0.5 for c, o in zip(code_scores, obj_scores)]


def group_metric_from_order(group: Sequence[dict[str, Any]], order: Sequence[int], k: int = 4) -> dict[str, Any]:
    rewards = [group_reward(row) for row in group]
    if not rewards or not order:
        return {}
    oracle = max(rewards)
    oracle_indices = {idx for idx, reward in enumerate(rewards) if reward == oracle}
    selected = list(order[: min(k, len(order))])
    kept = bool(set(selected) & oracle_indices)
    best_selected = max(rewards[idx] for idx in selected)
    return {
        "oracle_retention": 1.0 if kept else 0.0,
        "false_prune_rate": 0.0 if kept else 1.0,
        "avg_survivors": float(len(selected)),
        "best_selected_reward": best_selected,
        "oracle_reward": oracle,
        "regret": oracle - best_selected,
        "top1_success": 1.0 if selected and rewards[selected[0]] == oracle and oracle > 0 else 0.0,
        "top1_reward": rewards[selected[0]] if selected else float("nan"),
    }


def summarize_group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"group_count": 0}
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain"))].append(row)
    return {
        "group_count": len(rows),
        "oracle_retention": finite_mean(row.get("oracle_retention") for row in rows),
        "false_prune_rate": finite_mean(row.get("false_prune_rate") for row in rows),
        "avg_survivors": finite_mean(row.get("avg_survivors") for row in rows),
        "best_selected_reward": finite_mean(row.get("best_selected_reward") for row in rows),
        "top1_success": finite_mean(row.get("top1_success") for row in rows),
        "top1_reward": finite_mean(row.get("top1_reward") for row in rows),
        "regret": finite_mean(row.get("regret") for row in rows),
        "domain_retention": {domain: finite_mean(row.get("oracle_retention") for row in vals) for domain, vals in sorted(by_domain.items())},
    }


def groups_from_candidate_rows(path: Path, dataset_name: str, split: str | None = None) -> list[list[dict[str, Any]]]:
    payload = load_pt(path, {}) or {}
    rows = [dict(row) for row in payload.get("candidate_rows") or []]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split and str(row.get("split")) != split:
            continue
        if not isinstance((row.get("features_by_config") or {}).get(PRIMARY_TARGET), torch.Tensor):
            continue
        row.setdefault("source_dataset", dataset_name)
        grouped[str(row.get("group_id") or row.get("branch_group_id"))].append(row)
    return [vals for vals in grouped.values() if len(vals) >= 2]


def groups_from_survivor_features(split: str | None = None) -> list[list[dict[str, Any]]]:
    payload = load_pt(SURVIVOR_FEATURES_PT, {}) or {}
    groups: list[list[dict[str, Any]]] = []
    for survivor_set in payload.get("survivor_sets") or []:
        if split and str(survivor_set.get("split") or survivor_set.get("v1_split")) != split:
            continue
        candidates = []
        for cand in survivor_set.get("candidates") or []:
            row = dict(cand)
            if not isinstance((row.get("features_by_config") or {}).get(PRIMARY_TARGET), torch.Tensor):
                continue
            row["reward"] = group_reward(row)
            row["group_id"] = survivor_set.get("survivor_set_id")
            row["domain"] = survivor_set.get("domain")
            row["split"] = survivor_set.get("split") or survivor_set.get("v1_split") or "all"
            candidates.append(row)
        if len(candidates) >= 2:
            groups.append(candidates)
    return groups


def group_datasets() -> list[dict[str, Any]]:
    out = [
        {
            "dataset_name": "universal_bridge_candidate_groups",
            "dataset_kind": "branch_group",
            "path": BRIDGE_PT,
            "groups": groups_from_candidate_rows(BRIDGE_PT, "universal_bridge_candidate_groups", split="heldout"),
            "readiness_eligible": True,
        },
        {
            "dataset_name": "top4_survivor_hidden_feature_groups",
            "dataset_kind": "branch_group",
            "path": SURVIVOR_FEATURES_PT,
            "groups": groups_from_survivor_features(split="fresh_holdout"),
            "readiness_eligible": True,
        },
    ]
    # Inventory-only: this dataset has full fixed-composite score matrices but
    # no raw hidden features for unseen taps, so the two transplanted taps cannot
    # be rescored directly against every original candidate set.
    fixed = load_pt(FIXED_SURVIVAL_DATASET_PT, {}) or {}
    out.append(
        {
            "dataset_name": "fixed_composite_survival_dataset_score_matrix_only",
            "dataset_kind": "branch_group_incompatible",
            "path": FIXED_SURVIVAL_DATASET_PT,
            "groups": [],
            "record_count": len(fixed.get("records") or []),
            "readiness_eligible": False,
            "incompatibility_reason": "score_matrices_only_no_raw_features_for_new_two_tap_rescore",
        }
    )
    return out


def eval_group_dataset(dataset: dict[str, Any], two_taps: Sequence[dict[str, Any]], ref_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = list(dataset.get("groups") or [])
    if not groups:
        return []
    code_tap = next(t for t in two_taps if t.get("tap_role") == "coding_reasoning")
    objective_tap = next(t for t in two_taps if t.get("tap_role") == "mixed_objective_all")
    rows: list[dict[str, Any]] = []

    def add_policy(policy_name: str, family: str, candidate: dict[str, Any] | None, group_rows: list[dict[str, Any]]) -> None:
        summary = summarize_group_rows(group_rows)
        rows.append(
            {
                "dataset_name": dataset["dataset_name"],
                "dataset_kind": dataset["dataset_kind"],
                "readiness_eligible": bool(dataset.get("readiness_eligible")),
                "policy_name": policy_name,
                "candidate_name": (candidate or {}).get("candidate_name"),
                "candidate_family": family,
                "source_family": (candidate or {}).get("source_family"),
                **summary,
            }
        )

    two_rows = []
    for group in groups:
        scores = two_tap_group_scores(group, code_tap, objective_tap)
        if not scores:
            continue
        order = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]
        metric = group_metric_from_order(group, order, k=4)
        if metric:
            two_rows.append({"domain": group[0].get("domain"), **metric})
    add_policy("two_tap_equal_score_normalized_top4", "two_tap", None, two_rows)

    for candidate in ref_candidates:
        group_rows = []
        for group in groups:
            order = source_group_rank(candidate, group)
            metric = group_metric_from_order(group, order, k=4)
            if metric:
                group_rows.append({"domain": group[0].get("domain"), **metric})
        if group_rows:
            add_policy(str(candidate.get("candidate_name")), str(candidate.get("candidate_family")), candidate, group_rows)
    return rows


def group_summary(group_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        by_dataset[str(row.get("dataset_name"))].append(row)
    datasets = []
    for dataset_name, vals in sorted(by_dataset.items()):
        two = max([r for r in vals if r.get("candidate_family") == "two_tap"], key=lambda r: safe_float(r.get("oracle_retention"), -1.0), default={})
        refs = [r for r in vals if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
        ref = max(refs, key=lambda r: (safe_float(r.get("oracle_retention"), -1.0), -safe_float(r.get("false_prune_rate"), 1.0)), default={})
        two_ret = safe_float(two.get("oracle_retention"), float("nan"))
        ref_ret = safe_float(ref.get("oracle_retention"), float("nan"))
        datasets.append(
            {
                "dataset_name": dataset_name,
                "readiness_eligible": bool(vals[0].get("readiness_eligible")) if vals else False,
                "best_two_tap": two.get("policy_name"),
                "best_two_tap_retention": two_ret,
                "best_two_tap_false_prune": two.get("false_prune_rate"),
                "best_reference": ref.get("candidate_name") or ref.get("policy_name"),
                "best_reference_family": ref.get("candidate_family"),
                "best_reference_retention": ref_ret,
                "best_reference_false_prune": ref.get("false_prune_rate"),
                "delta_two_minus_reference": two_ret - ref_ret if math.isfinite(two_ret) and math.isfinite(ref_ret) else float("nan"),
                "matches_or_exceeds_reference": bool(math.isfinite(two_ret) and math.isfinite(ref_ret) and two_ret + 1e-9 >= ref_ret),
                "group_count": two.get("group_count") or ref.get("group_count") or 0,
            }
        )
    return {"datasets": datasets}


def readiness_from_summaries(domain_summary: dict[str, Any], branch_pair_summary: dict[str, Any], branch_group_summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    domain_sets = [d for d in domain_summary.get("datasets", []) if d.get("readiness_eligible") and int(d.get("pair_count") or 0) > 0]
    branch_pair_sets = [d for d in branch_pair_summary.get("datasets", []) if d.get("readiness_eligible") and int(d.get("pair_count") or 0) > 0]
    branch_group_sets = [d for d in branch_group_summary.get("datasets", []) if d.get("readiness_eligible") and int(d.get("group_count") or 0) > 0]
    domain_failures = [d for d in domain_sets if not d.get("matches_or_exceeds_reference")]
    branch_pair_failures = [d for d in branch_pair_sets if not d.get("matches_or_exceeds_reference")]
    branch_group_failures = [d for d in branch_group_sets if not d.get("matches_or_exceeds_reference")]
    data_ok = bool(domain_sets) and bool(branch_pair_sets or branch_group_sets)
    domain_ok = data_ok and not domain_failures
    branch_ok = data_ok and not branch_pair_failures and not branch_group_failures
    if domain_ok and branch_ok:
        status = "TWO_TAP_FULL_READY"
    elif branch_ok and not domain_ok:
        status = "BRANCH_READY_DOMAIN_GAP"
    elif domain_ok and not branch_ok:
        status = "DOMAIN_READY_BRANCH_GAP"
    elif data_ok:
        status = "TWO_TAP_PARTIAL_NOT_READY"
    else:
        status = "DATA_LIMITED"
    return status, {
        "domain_dataset_count": len(domain_sets),
        "branch_pair_dataset_count": len(branch_pair_sets),
        "branch_group_dataset_count": len(branch_group_sets),
        "domain_failures": domain_failures,
        "branch_pair_failures": branch_pair_failures,
        "branch_group_failures": branch_group_failures,
        "domain_ok": domain_ok,
        "branch_ok": branch_ok,
        "data_ok": data_ok,
    }


def main() -> int:
    ensure_root()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    two_taps = load_two_taps()
    refs = source_candidates()
    old_refs = [c for c in refs if c.get("candidate_family") == "source_old_content"]
    branch_refs = [c for c in refs if c.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
    pair_candidates = two_taps + old_refs + branch_refs

    old_datasets = old_domain_pair_datasets()
    branch_datasets = branch_pair_datasets()
    pair_rows: list[dict[str, Any]] = []
    for dataset in old_datasets + branch_datasets:
        pair_rows.extend(pair_eval_rows(dataset, pair_candidates))

    domain_summary = summarize_pair_dataset(pair_rows, "old_domain_pair")
    branch_pair_summary = summarize_pair_dataset(pair_rows, "branch_pair")

    group_rows: list[dict[str, Any]] = []
    group_data = group_datasets()
    # Keep group reference evaluation bounded to the native branch/universal heads
    # that can score the primary concat config. Pairwise comparisons above cover
    # all compatible native configs.
    primary_refs = [c for c in branch_refs if c.get("target_config") == PRIMARY_TARGET]
    group_refs = []
    for family in ("source_branch", "source_bridge", "source_universal"):
        family_refs = [c for c in primary_refs if c.get("candidate_family") == family]
        family_refs = sorted(family_refs, key=lambda c: safe_float(c.get("metric_score"), -1.0), reverse=True)
        group_refs.extend(family_refs[:12])
    for dataset in group_data:
        group_rows.extend(eval_group_dataset(dataset, two_taps, group_refs))
    branch_group_summary = group_summary(group_rows)

    status, readiness = readiness_from_summaries(domain_summary, branch_pair_summary, branch_group_summary)

    inventory_rows = []
    for dataset in old_datasets + branch_datasets + group_data:
        inventory_rows.append(
            {
                "dataset_name": dataset.get("dataset_name"),
                "dataset_kind": dataset.get("dataset_kind"),
                "path": rel(dataset.get("path")) if isinstance(dataset.get("path"), Path) else dataset.get("path"),
                "pairs": len(dataset.get("pairs") or []),
                "groups": len(dataset.get("groups") or []),
                "records": dataset.get("record_count", ""),
                "readiness_eligible": dataset.get("readiness_eligible"),
                "incompatibility_reason": dataset.get("incompatibility_reason", ""),
            }
        )

    payload = {
        "BG_TWO_TAP_FULL_READINESS_VERDICT": status,
        "status": status,
        "tap_names": {
            "coding_reasoning": CODE_TAP_NAME,
            "mixed_objective_all": OBJECTIVE_TAP_NAME,
        },
        "reference_counts": {
            "old_domain_reference_heads": len(old_refs),
            "branch_universal_reference_heads": len(branch_refs),
            "all_reference_heads": len(refs),
        },
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_pair_summary,
        "branch_group_summary": branch_group_summary,
        "readiness": readiness,
        "dataset_inventory": inventory_rows,
        "anti_leakage": {
            "no_ouro_training": True,
            "no_old_tap_registry_update": True,
            "no_action_steering": True,
            "no_production_routing_change": True,
            "tap_scores_not_used_as_labels": True,
            "diagnostic_alpha_0_02_sampled_l47_excluded_from_readiness": True,
        },
    }
    torch.save({"summary": payload, "two_taps": two_taps, "reference_candidates": refs}, ARTIFACT_PT)
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(PAIR_ROWS_CSV, pair_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(DATASET_INVENTORY_CSV, inventory_rows)

    domain_table = domain_summary.get("datasets", [])
    branch_pair_table = branch_pair_summary.get("datasets", [])
    branch_group_table = branch_group_summary.get("datasets", [])
    lines = [
        "# Two-Tap Full Readiness v1",
        "",
        f"BG_TWO_TAP_FULL_READINESS_VERDICT = {status}",
        "",
        f"- old-domain reference heads tested: `{len(old_refs)}`",
        f"- branch/universal reference heads tested: `{len(branch_refs)}`",
        f"- domain datasets: `{readiness['domain_dataset_count']}` readiness-eligible",
        f"- branch pair datasets: `{readiness['branch_pair_dataset_count']}` readiness-eligible",
        f"- branch group datasets: `{readiness['branch_group_dataset_count']}` readiness-eligible",
        f"- domain ok: `{readiness['domain_ok']}`",
        f"- branch ok: `{readiness['branch_ok']}`",
        "",
        "Readiness requires the two-tap set to match or exceed the best old-content reference on old-domain datasets and the best branch/universal reference on branch datasets. Diagnostic alpha 0.02, sampled, L47, and true-fork rows are evaluated but not used for readiness.",
        "",
        "## Old-Domain Datasets",
        "",
    ]
    lines.extend(md_table(domain_table, ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Branch Pair Datasets", ""])
    lines.extend(md_table(branch_pair_table, ["dataset_name", "readiness_eligible", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Branch Group Datasets", ""])
    lines.extend(md_table(branch_group_table, ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_reference_family"]))
    if readiness["domain_failures"]:
        lines.extend(["", "## Domain Gaps", ""])
        lines.extend(md_table(readiness["domain_failures"], ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "best_reference_family"]))
    if readiness["branch_pair_failures"] or readiness["branch_group_failures"]:
        lines.extend(["", "## Branch Gaps", ""])
        lines.extend(md_table(readiness["branch_pair_failures"], ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "best_reference_family"]))
        lines.extend(md_table(readiness["branch_group_failures"], ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "best_reference_family"]))
    lines.extend(
        [
            "",
            "## Incompatible Cached Data",
            "",
            "The fixed-composite survival dataset is inventory-only here because it stores expert score matrices but not raw hidden features for new tap rescoring. The universal bridge candidate rows and top4 survivor feature acquisition cover the available raw-feature branch-group evaluations.",
            "",
            "## Files",
            "",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- pair rows: `{rel(PAIR_ROWS_CSV)}`",
            f"- group rows: `{rel(GROUP_ROWS_CSV)}`",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# Two-Tap Full Readiness v1",
            "",
            f"BG_TWO_TAP_FULL_READINESS_VERDICT = {status}",
            "",
            "This probe tests whether the old-anchored `coding_reasoning` and `mixed_objective_all` transplanted taps can be the only taps for both old-domain scoring and branch scoring.",
            "",
            "## Result",
            "",
            f"- domain ok: `{readiness['domain_ok']}`",
            f"- branch ok: `{readiness['branch_ok']}`",
            f"- status: `{status}`",
            "",
            "## Caveat",
            "",
            "The fixed-composite survival score-matrix cache cannot be directly rescored by new taps because raw hidden features are not stored there. Raw-feature branch evaluation uses the universal bridge candidate rows and top4 survivor hidden-feature cache.",
            "",
            "## Files",
            "",
            f"- report: `{rel(REPORT_MD)}`",
            f"- rows: `{rel(PAIR_ROWS_CSV)}`, `{rel(GROUP_ROWS_CSV)}`",
        ],
    )

    section_title = "## Two-tap full readiness v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_TWO_TAP_FULL_READINESS_VERDICT = {status}`. Tested the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps against old-domain references and branch/universal references across cached old-domain and branch datasets. Domain ok `{readiness['domain_ok']}`; branch ok `{readiness['branch_ok']}`. No Ouro training, steering, registry update, routing change, or production change was performed.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [
        section_title,
        "",
        f"Added `{rel(DOC_MD)}`. Status: `{status}`.",
    ]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)

    print(f"BG_TWO_TAP_FULL_READINESS_VERDICT = {status}", flush=True)
    print(f"domain_ok = {readiness['domain_ok']} branch_ok = {readiness['branch_ok']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
