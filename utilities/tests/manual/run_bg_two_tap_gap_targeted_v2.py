"""Gap-targeted two-tap repair v2.

Starts from the two old-anchored transplanted taps and trains copied tap
vectors on cached labeled pair data targeted at the gaps found by full
readiness v1: old-content heldout, science, GSM8K expanded, code, and
branch/bridge pairs.

This does not train Ouro, modify old taps, overwrite tap registries, update
tokenizers/checkpoints, run generation, run wrapper/local-agent code, apply
steering, or change routing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import PRIMARY_TARGET, finite_mean, json_default, pair_diff, rate, safe_float
from run_bg_two_tap_full_readiness_v1 import (
    ARTIFACT_PT as V1_FULL_ARTIFACT_PT,
    DOC_MD as V1_FULL_DOC_MD,
    CODE_TAP_NAME,
    OBJECTIVE_TAP_NAME,
    append_section,
    branch_pair_datasets,
    eval_group_dataset,
    group_datasets,
    group_summary,
    load_two_taps,
    old_domain_pair_datasets,
    pair_eval_rows,
    readiness_from_summaries,
    source_candidates,
    summarize_pair_dataset,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)


OUT_ROOT = PROBE_ROOT / "bg_two_tap_gap_targeted_v2_2026-05-30"
TRAINING_JSON = OUT_ROOT / "training_log.json"
TRAINING_MD = OUT_ROOT / "training_report.md"
VAL_ROWS_CSV = OUT_ROOT / "validation_rows.csv"
CLEAN_HOLDOUT_JSON = OUT_ROOT / "clean_holdout_eval.json"
CLEAN_HOLDOUT_MD = OUT_ROOT / "clean_holdout_eval.md"
FULL_REPLAY_JSON = OUT_ROOT / "full_replay_eval.json"
FULL_REPLAY_MD = OUT_ROOT / "full_replay_eval.md"
PAIR_ROWS_CSV = OUT_ROOT / "full_replay_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "full_replay_group_rows.csv"
ARTIFACT_PT = OUT_ROOT / "two_tap_gap_targeted_v2.pt"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_two_tap_gap_targeted_v2.md"

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    V1_FULL_DOC_MD,
    PROJECT_ROOT / "docs/evaluator/bg_two_tap_branch_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_merged_weight_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
]

NAV_TARGETS = [
    PROJECT_ROOT / "shared/docs/evaluator/README.md",
    PROJECT_ROOT / "shared/docs/README.md",
    PROJECT_ROOT / "PROJECT_TREE_MAP.md",
    PROJECT_ROOT / "PROJECT_COMPONENTS.md",
]

RECIPE_GRID = [
    {
        "recipe_name": "balanced_gap_repair",
        "epochs": 80,
        "lr": 3e-4,
        "anchor_l2": 0.025,
        "norm_l2": 0.01,
        "domain_multiplier": 1.6,
        "science_multiplier": 2.0,
        "math_multiplier": 1.5,
        "code_multiplier": 1.4,
        "branch_multiplier": 1.5,
        "bridge_multiplier": 1.8,
    },
    {
        "recipe_name": "old_domain_preserve_repair",
        "epochs": 80,
        "lr": 2e-4,
        "anchor_l2": 0.05,
        "norm_l2": 0.02,
        "domain_multiplier": 2.4,
        "science_multiplier": 2.6,
        "math_multiplier": 2.0,
        "code_multiplier": 2.0,
        "branch_multiplier": 0.9,
        "bridge_multiplier": 1.0,
    },
    {
        "recipe_name": "branch_bridge_repair",
        "epochs": 80,
        "lr": 3e-4,
        "anchor_l2": 0.015,
        "norm_l2": 0.008,
        "domain_multiplier": 1.0,
        "science_multiplier": 1.2,
        "math_multiplier": 1.0,
        "code_multiplier": 1.0,
        "branch_multiplier": 2.4,
        "bridge_multiplier": 2.8,
    },
    {
        "recipe_name": "science_bridge_repair",
        "epochs": 80,
        "lr": 3e-4,
        "anchor_l2": 0.02,
        "norm_l2": 0.01,
        "domain_multiplier": 1.4,
        "science_multiplier": 3.2,
        "math_multiplier": 1.2,
        "code_multiplier": 1.2,
        "branch_multiplier": 1.6,
        "bridge_multiplier": 2.4,
    },
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def split_tasks_for_dataset(pairs: Sequence[dict[str, Any]]) -> dict[str, str]:
    tasks = sorted({str(pair.get("task_id")) for pair in pairs if pair.get("task_id") is not None})
    tasks = sorted(tasks, key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    n = len(tasks)
    if n <= 1:
        return {task: "train" for task in tasks}
    train_cut = max(1, int(round(n * 0.60)))
    val_cut = max(train_cut + 1, int(round(n * 0.80))) if n >= 3 else n
    out = {}
    for idx, task in enumerate(tasks):
        if idx < train_cut:
            out[task] = "train"
        elif idx < val_cut:
            out[task] = "val"
        else:
            out[task] = "heldout"
    if n >= 3 and "heldout" not in out.values():
        out[tasks[-1]] = "heldout"
    return out


def split_domain_probe_datasets(datasets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for dataset in datasets:
        pairs = [dict(pair) for pair in dataset.get("pairs") or []]
        splits = {str(pair.get("split") or "all") for pair in pairs}
        if splits <= {"domain_probe", "all", "none", ""}:
            task_split = split_tasks_for_dataset(pairs)
            for pair in pairs:
                pair["split"] = task_split.get(str(pair.get("task_id")), "train")
        row = dict(dataset)
        row["pairs"] = pairs
        out.append(row)
    return out


def select_split_dataset(datasets: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out = []
    split_aliases = {split}
    if split == "heldout":
        split_aliases.update({"test", "fresh_holdout"})
    for dataset in datasets:
        pairs = []
        for pair in dataset.get("pairs") or []:
            ps = str(pair.get("split") or "all")
            if ps in split_aliases:
                row = dict(pair)
                row["split"] = "all"
                pairs.append(row)
        out.append({**dataset, "pairs": pairs})
    return out


def example_weight(pair: dict[str, Any], dataset_name: str, recipe: dict[str, Any], role: str) -> float:
    kind = str(pair.get("pair_type") or "")
    domain = str(pair.get("domain") or "")
    name = dataset_name.lower()
    weight = 1.0
    if kind in {"old_content", "old_code", "old_domain"} or "old" in name or "distractor" in name or "gsm8k" in name:
        weight *= float(recipe["domain_multiplier"])
    if domain == "science" or "science" in name:
        weight *= float(recipe["science_multiplier"])
    if domain == "math_simple_arithmetic" or "gsm8k" in name:
        weight *= float(recipe["math_multiplier"])
    if domain == "coding" or "code" in name:
        weight *= float(recipe["code_multiplier"])
    if "hidden" in name or "branch_generator" in name or "salvage" in name or "quota" in name:
        weight *= float(recipe["branch_multiplier"])
    if "bridge" in name:
        weight *= float(recipe["bridge_multiplier"])
    if role == "coding_reasoning" and domain in {"coding", "reasoning", "math_simple_arithmetic"}:
        weight *= 1.15
    if role == "mixed_objective_all" and domain in {"science", "math_simple_arithmetic"}:
        weight *= 1.15
    return max(0.05, float(weight))


def collect_training_examples(datasets: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    examples = []
    aliases = {split}
    if split == "heldout":
        aliases.update({"test", "fresh_holdout"})
    for dataset in datasets:
        name = str(dataset.get("dataset_name"))
        readiness = bool(dataset.get("readiness_eligible", True))
        if not readiness:
            continue
        for pair in dataset.get("pairs") or []:
            if str(pair.get("split") or "all") not in aliases:
                continue
            diff = pair_diff(pair, PRIMARY_TARGET)
            if diff is None:
                continue
            examples.append({"dataset_name": name, "pair": pair, "diff": diff.detach().cpu().to(torch.float32)})
    return examples


def tensorize_examples(examples: Sequence[dict[str, Any]], recipe: dict[str, Any], role: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        return torch.empty((0, 6144), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
    x = torch.stack([ex["diff"] for ex in examples], dim=0).to(torch.float32)
    weights = torch.tensor([example_weight(ex["pair"], ex["dataset_name"], recipe, role) for ex in examples], dtype=torch.float32)
    weights = weights / max(float(weights.mean().item()), 1e-6)
    return x, weights


def train_one(anchor: torch.Tensor, train_examples: Sequence[dict[str, Any]], val_examples: Sequence[dict[str, Any]], recipe: dict[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
    x_train, w_train = tensorize_examples(train_examples, recipe, role)
    x_val, w_val = tensorize_examples(val_examples, recipe, role)
    anchor = anchor.detach().cpu().to(torch.float32).flatten()
    vec = torch.nn.Parameter(anchor.clone())
    opt = torch.optim.Adam([vec], lr=float(recipe["lr"]))
    with torch.no_grad():
        initial_scores = x_train @ anchor if x_train.numel() else torch.ones(1)
        score_scale = float(torch.quantile(initial_scores.abs().clamp_min(1e-6), 0.50).item())
        score_scale = max(score_scale, 1.0)
        anchor_norm = float(anchor.norm().item())
    best = anchor.clone()
    best_val = -float("inf")
    logs = []
    batch_size = 512
    for epoch in range(int(recipe["epochs"])):
        if not x_train.numel():
            break
        perm = torch.randperm(x_train.shape[0])
        total = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            wb = w_train[idx]
            logits = (xb @ vec) / score_scale
            margin_loss = (F.softplus(-logits) * wb).mean()
            anchor_loss = (vec - anchor).pow(2).mean()
            norm_loss = ((vec.norm() - anchor_norm) / max(anchor_norm, 1e-6)).pow(2)
            loss = margin_loss + float(recipe["anchor_l2"]) * anchor_loss + float(recipe["norm_l2"]) * norm_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([vec], 1.0)
            opt.step()
            total += float(loss.item())
        with torch.no_grad():
            if x_val.numel():
                val_logits = x_val @ vec
                val_score = float(((val_logits > 0).to(torch.float32) * w_val).sum().item() / max(float(w_val.sum().item()), 1e-6))
            else:
                val_score = 0.0
            if val_score > best_val:
                best_val = val_score
                best = vec.detach().clone()
        if epoch in {0, 4, 9, 19, 39, int(recipe["epochs"]) - 1}:
            logs.append({"epoch": epoch + 1, "train_loss_sum": total, "weighted_val_accuracy": best_val})
    return best.detach().cpu(), {
        "role": role,
        "recipe_name": recipe["recipe_name"],
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_weighted_val_accuracy": best_val,
        "score_scale": score_scale,
        "anchor_cosine": float(torch.dot(best / best.norm().clamp_min(1e-8), anchor / anchor.norm().clamp_min(1e-8)).item()),
        "logs": logs,
    }


def build_candidates_from_weights(code_weight: torch.Tensor, objective_weight: torch.Tensor, recipe_name: str) -> list[dict[str, Any]]:
    code_weight = code_weight.detach().cpu().to(torch.float32).flatten()
    objective_weight = objective_weight.detach().cpu().to(torch.float32).flatten()
    fused = (code_weight / code_weight.norm().clamp_min(1e-8) + objective_weight / objective_weight.norm().clamp_min(1e-8))
    fused = fused / fused.norm().clamp_min(1e-8)
    return [
        {
            "candidate_name": f"two_tap_v2::{recipe_name}::coding_reasoning",
            "candidate_family": "two_tap",
            "tap_role": "coding_reasoning",
            "target_config": PRIMARY_TARGET,
            "architecture": "AntisymLinearNoNorm",
            "weight": code_weight,
            "state_dict": {"linear.weight": code_weight.reshape(1, -1)},
        },
        {
            "candidate_name": f"two_tap_v2::{recipe_name}::mixed_objective_all",
            "candidate_family": "two_tap",
            "tap_role": "mixed_objective_all",
            "target_config": PRIMARY_TARGET,
            "architecture": "AntisymLinearNoNorm",
            "weight": objective_weight,
            "state_dict": {"linear.weight": objective_weight.reshape(1, -1)},
        },
        {
            "candidate_name": f"two_tap_v2::{recipe_name}::equal_weight_fused_direction",
            "candidate_family": "two_tap",
            "tap_role": "equal_weight_fused_direction",
            "target_config": PRIMARY_TARGET,
            "architecture": "AntisymLinearNoNorm",
            "weight": fused,
            "state_dict": {"linear.weight": fused.reshape(1, -1)},
        },
    ]


def eval_pair_suite(datasets: Sequence[dict[str, Any]], two_taps: Sequence[dict[str, Any]], refs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = list(two_taps) + list(refs)
    for dataset in datasets:
        rows.extend(pair_eval_rows(dataset, candidates))
    domain_summary = summarize_pair_dataset(rows, "old_domain_pair")
    branch_summary = summarize_pair_dataset(rows, "branch_pair")
    return rows, domain_summary, branch_summary


def selection_score(domain_summary: dict[str, Any], branch_summary: dict[str, Any]) -> float:
    deltas = []
    for summary in (domain_summary, branch_summary):
        for dataset in summary.get("datasets", []):
            if int(dataset.get("pair_count") or 0) <= 0:
                continue
            delta = safe_float(dataset.get("delta_two_minus_reference"), float("nan"))
            if math.isfinite(delta):
                deltas.append(delta)
    if not deltas:
        return -1e9
    # Mean delta rewards broad improvement; worst-case delta discourages a tap
    # that repairs one domain while collapsing another.
    return float(mean(deltas) + 0.5 * min(deltas))


def run_group_replay(two_taps: Sequence[dict[str, Any]], branch_refs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_refs = [c for c in branch_refs if c.get("target_config") == PRIMARY_TARGET]
    group_refs = []
    for family in ("source_branch", "source_bridge", "source_universal"):
        family_refs = [c for c in primary_refs if c.get("candidate_family") == family]
        family_refs = sorted(family_refs, key=lambda c: safe_float(c.get("metric_score"), -1.0), reverse=True)
        group_refs.extend(family_refs[:12])
    rows: list[dict[str, Any]] = []
    for dataset in group_datasets():
        rows.extend(eval_group_dataset(dataset, two_taps, group_refs))
    return rows, group_summary(rows)


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in candidate.items() if k not in {"weight", "state_dict"}}


def main() -> int:
    ensure_root()
    torch.manual_seed(12345)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    base_taps = load_two_taps()
    code_anchor = tensor_weight(next(t for t in base_taps if t.get("tap_role") == "coding_reasoning"))
    objective_anchor = tensor_weight(next(t for t in base_taps if t.get("tap_role") == "mixed_objective_all"))
    if not isinstance(code_anchor, torch.Tensor) or not isinstance(objective_anchor, torch.Tensor):
        raise RuntimeError("missing base tap weights")
    refs = source_candidates()
    old_refs = [c for c in refs if c.get("candidate_family") == "source_old_content"]
    branch_refs = [c for c in refs if c.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
    all_refs = old_refs + branch_refs

    old_datasets_clean = split_domain_probe_datasets(old_domain_pair_datasets())
    branch_datasets_clean = branch_pair_datasets()
    clean_datasets = old_datasets_clean + branch_datasets_clean

    train_examples = collect_training_examples(clean_datasets, "train")
    val_examples = collect_training_examples(clean_datasets, "val")
    heldout_datasets = select_split_dataset(clean_datasets, "heldout")
    val_datasets = select_split_dataset(clean_datasets, "val")

    recipe_records = []
    validation_rows: list[dict[str, Any]] = []
    trained_payloads = []
    for recipe in RECIPE_GRID:
        code_weight, code_log = train_one(code_anchor, train_examples, val_examples, recipe, "coding_reasoning")
        objective_weight, objective_log = train_one(objective_anchor, train_examples, val_examples, recipe, "mixed_objective_all")
        taps = build_candidates_from_weights(code_weight, objective_weight, str(recipe["recipe_name"]))
        rows, domain_summary, branch_summary = eval_pair_suite(val_datasets, taps, all_refs)
        score = selection_score(domain_summary, branch_summary)
        recipe_record = {
            "recipe_name": recipe["recipe_name"],
            "selection_score": score,
            "domain_summary": domain_summary,
            "branch_summary": branch_summary,
            "code_log": code_log,
            "objective_log": objective_log,
            "recipe": recipe,
        }
        recipe_records.append(recipe_record)
        validation_rows.extend({**row, "recipe_name": recipe["recipe_name"]} for row in rows if row.get("candidate_family") == "two_tap")
        trained_payloads.append({"recipe": recipe, "taps": taps, "record": recipe_record})

    selected_payload = max(trained_payloads, key=lambda p: safe_float(p["record"].get("selection_score"), -1e9))
    selected_taps = selected_payload["taps"]
    selected_recipe = str(selected_payload["recipe"]["recipe_name"])

    clean_rows, clean_domain_summary, clean_branch_pair_summary = eval_pair_suite(heldout_datasets, selected_taps, all_refs)
    clean_group_rows, clean_branch_group_summary = run_group_replay(selected_taps, branch_refs)
    clean_status, clean_readiness = readiness_from_summaries(clean_domain_summary, clean_branch_pair_summary, clean_branch_group_summary)

    # Full replay keeps the exact previous full-readiness suite. This is useful
    # for continuity, but it is diagnostic because v2 trained on splits drawn
    # from some of those cached domain-probe datasets.
    full_old = old_domain_pair_datasets()
    full_branch = branch_pair_datasets()
    full_rows, full_domain_summary, full_branch_pair_summary = eval_pair_suite(full_old + full_branch, selected_taps, all_refs)
    full_group_rows, full_branch_group_summary = run_group_replay(selected_taps, branch_refs)
    full_status, full_readiness = readiness_from_summaries(full_domain_summary, full_branch_pair_summary, full_branch_group_summary)

    if clean_status == "TWO_TAP_FULL_READY":
        final_status = "TWO_TAP_GAP_TARGETED_READY"
    elif clean_status in {"DOMAIN_READY_BRANCH_GAP", "BRANCH_READY_DOMAIN_GAP"}:
        final_status = "TWO_TAP_GAP_TARGETED_PARTIAL"
    elif clean_readiness.get("domain_ok") or clean_readiness.get("branch_ok"):
        final_status = "TWO_TAP_GAP_TARGETED_USEFUL_BUT_NOT_READY"
    else:
        final_status = "TWO_TAP_GAP_TARGETED_NOT_READY"

    training_payload = {
        "BG_TWO_TAP_GAP_TARGETED_V2_TRAINING_VERDICT": "READY" if recipe_records else "BLOCKED",
        "selected_recipe": selected_recipe,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "recipes": recipe_records,
        "heldout_used_for_selection": False,
    }
    clean_payload = {
        "BG_TWO_TAP_GAP_TARGETED_V2_CLEAN_HOLDOUT_VERDICT": clean_status,
        "status": clean_status,
        "domain_summary": clean_domain_summary,
        "branch_pair_summary": clean_branch_pair_summary,
        "branch_group_summary": clean_branch_group_summary,
        "readiness": clean_readiness,
        "readiness_bearing": True,
    }
    full_payload = {
        "BG_TWO_TAP_GAP_TARGETED_V2_FULL_REPLAY_VERDICT": full_status,
        "status": full_status,
        "domain_summary": full_domain_summary,
        "branch_pair_summary": full_branch_pair_summary,
        "branch_group_summary": full_branch_group_summary,
        "readiness": full_readiness,
        "readiness_bearing": False,
    }
    summary = {
        "BG_TWO_TAP_GAP_TARGETED_V2_STATUS": final_status,
        "status": final_status,
        "selected_recipe": selected_recipe,
        "base_taps": {"coding_reasoning": CODE_TAP_NAME, "mixed_objective_all": OBJECTIVE_TAP_NAME},
        "clean_holdout_status": clean_status,
        "full_replay_status": full_status,
        "training": training_payload,
        "clean_holdout": clean_payload,
        "full_replay": full_payload,
        "reference_counts": {
            "old_domain_reference_heads": len(old_refs),
            "branch_universal_reference_heads": len(branch_refs),
        },
        "anti_leakage": {
            "trained_only_copied_tap_vectors": True,
            "heldout_used_for_selection": False,
            "full_replay_is_diagnostic_after_training": True,
            "no_ouro_training": True,
            "no_registry_update": True,
            "no_action_steering": True,
            "no_routing_change": True,
        },
    }

    torch.save(
        {
            "summary": summary,
            "selected_taps": selected_taps,
            "base_taps": base_taps,
            "trained_payloads": trained_payloads,
        },
        ARTIFACT_PT,
    )
    write_json(TRAINING_JSON, training_payload)
    write_json(CLEAN_HOLDOUT_JSON, clean_payload)
    write_json(FULL_REPLAY_JSON, full_payload)
    write_json(SUMMARY_JSON, summary)
    write_csv(VAL_ROWS_CSV, validation_rows)
    write_csv(PAIR_ROWS_CSV, full_rows)
    write_csv(GROUP_ROWS_CSV, full_group_rows)

    def table(summary_obj: dict[str, Any], key: str) -> list[dict[str, Any]]:
        return list((summary_obj.get(key) or {}).get("datasets") or [])

    lines = [
        "# Two-Tap Gap-Targeted v2",
        "",
        f"BG_TWO_TAP_GAP_TARGETED_V2_STATUS = {final_status}",
        "",
        f"- selected recipe: `{selected_recipe}`",
        f"- train examples: `{len(train_examples)}`",
        f"- validation examples: `{len(val_examples)}`",
        f"- clean heldout status: `{clean_status}`",
        f"- full replay status: `{full_status}`",
        f"- heldout used for selection: `False`",
        "",
        "## Clean Heldout Old-Domain",
        "",
    ]
    lines.extend(md_table(table(clean_payload, "domain_summary"), ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference"]))
    lines.extend(["", "## Clean Heldout Branch Pairs", ""])
    lines.extend(md_table(table(clean_payload, "branch_pair_summary"), ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference"]))
    lines.extend(["", "## Clean Heldout Branch Groups", ""])
    lines.extend(md_table(table(clean_payload, "branch_group_summary"), ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "matches_or_exceeds_reference"]))
    lines.extend(["", "## Full Replay", ""])
    lines.append("Full replay uses the same broad suite as v1 and is diagnostic after v2 training.")
    lines.extend(["", "### Full Replay Old-Domain", ""])
    lines.extend(md_table(table(full_payload, "domain_summary"), ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference"]))
    lines.extend(["", "### Full Replay Branch Groups", ""])
    lines.extend(md_table(table(full_payload, "branch_group_summary"), ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "matches_or_exceeds_reference"]))
    write_md(TRAINING_MD, lines)
    write_md(CLEAN_HOLDOUT_MD, lines)
    write_md(FULL_REPLAY_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# Two-Tap Gap-Targeted v2",
            "",
            f"BG_TWO_TAP_GAP_TARGETED_V2_STATUS = {final_status}",
            "",
            "This run fine-tuned copied versions of the two old-anchored transplanted taps against cached labeled gap datasets. It did not modify old taps or registries.",
            "",
            "## Result",
            "",
            f"- selected recipe: `{selected_recipe}`",
            f"- clean heldout status: `{clean_status}`",
            f"- full replay status: `{full_status}`",
            f"- heldout used for selection: `False`",
            "",
            "## Files",
            "",
            f"- summary: `{rel(SUMMARY_MD)}`",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- full replay rows: `{rel(PAIR_ROWS_CSV)}`",
        ],
    )
    section_title = "## Two-tap gap-targeted v2 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_TWO_TAP_GAP_TARGETED_V2_STATUS = {final_status}`. Selected `{selected_recipe}` on validation only. Clean heldout status `{clean_status}`; full replay status `{full_status}`. Only copied tap vectors were trained; no old taps, registries, Ouro weights, steering, routing, or production behavior were modified.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [section_title, "", f"Added `{rel(DOC_MD)}`. Status: `{final_status}`."]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)

    print(f"BG_TWO_TAP_GAP_TARGETED_V2_STATUS = {final_status}", flush=True)
    print(f"clean_holdout_status = {clean_status}", flush=True)
    print(f"full_replay_status = {full_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
