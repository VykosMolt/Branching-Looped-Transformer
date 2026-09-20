"""DualAnchor branch-gap repair v1.

Targeted repair pass for the layer-native `MIX_CODE_REASONING` and
`MIX_OBJECTIVE_ALL` tap pair, now referred to as DualAnchor. This trains only
copied diagnostic tap vectors at 24_L4, 36_L4, and 47_L4. It does not train
Ouro, mutate checkpoints/tokenizers, overwrite old tap registries, run wrapper
or Hunter-Seeker code, apply steering, or change routing.
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
from bg_merged_tap_v1_common import finite_mean, json_default, normed, pair_diff, safe_float, score_diff
from run_bg_layer_native_two_tap_readiness_v1 import (
    ANCHOR_GROUPS,
    DOC_TARGETS,
    LAYER_CONFIGS,
    NAV_TARGETS,
    PRIMARY_ARCHES,
    append_section,
    branch_pair_datasets,
    build_layer_native_candidates,
    bundle_specs,
    candidate_feature,
    candidate_name,
    eval_group_dataset,
    group_datasets_any,
    group_metric_from_order,
    old_domain_pair_datasets,
    readiness,
    source_candidates,
    source_group_rank,
    summarize_group_rows,
    summarize_pair_dataset,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_branch_gap_repair_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "dualanchor_branch_gap_repair.json"
REPORT_MD = OUT_ROOT / "dualanchor_branch_gap_repair.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
TRAINING_LOG_JSON = OUT_ROOT / "training_log.json"
PAIR_ROWS_CSV = OUT_ROOT / "dualanchor_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "dualanchor_group_rows.csv"
SELECTED_ROWS_CSV = OUT_ROOT / "selected_bundle_rows.csv"
MODEL_ROWS_CSV = OUT_ROOT / "trained_candidate_rows.csv"
ARTIFACT_PT = OUT_ROOT / "dualanchor_branch_gap_repair_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_dualanchor_branch_gap_repair_v1.md"
LAYER_NATIVE_REFERENCE_JSON = PROBE_ROOT / "bg_layer_native_two_tap_readiness_v1_2026-05-30/layer_native_two_tap_readiness.json"

SEED = 20260530
EPOCHS = 50
MAX_TRAIN_EXAMPLES = 9000
FAST_ARCHES = ("AntisymLinearNoNorm",)
GAP_KEYWORDS = (
    "branch_generator_v1",
    "hidden_origin_v3",
    "salvage::strict_cross_version_clean",
    "salvage::v3_clean",
    "salvage::old_frozen_tap_clean",
    "universal_bridge",
)
READINESS_ONLY_NO_TRAIN = (
    "hidden_origin_v3::high_yield_recipe_subset",
    "salvage::old_frozen_tap_clean::all_signal_tasks::primary_safe_deterministic",
)

REPAIR_RECIPES = [
    {
        "recipe": "dualanchor_gap_repair_anchor_strong",
        "init_recipe": "old_plus_branch_bridge_70_15_15",
        "lr": 3e-4,
        "anchor_lambda": 0.12,
        "old_weight": 2.4,
        "branch_weight": 1.4,
        "bridge_weight": 1.6,
        "gap_weight": 2.6,
        "salvage_weight": 2.2,
        "bgv1_weight": 3.0,
        "selection_metric": "constrained_val_score",
    },
    {
        "recipe": "dualanchor_gap_repair_branch_heavy",
        "init_recipe": "old_plus_branch_bridge_60_20_20",
        "lr": 3e-4,
        "anchor_lambda": 0.06,
        "old_weight": 1.5,
        "branch_weight": 2.2,
        "bridge_weight": 1.8,
        "gap_weight": 3.6,
        "salvage_weight": 2.8,
        "bgv1_weight": 4.0,
        "selection_metric": "constrained_val_score",
    },
    {
        "recipe": "dualanchor_gap_repair_universal_bridge",
        "init_recipe": "old_plus_bridge_50_50",
        "lr": 2e-4,
        "anchor_lambda": 0.04,
        "old_weight": 1.3,
        "branch_weight": 1.8,
        "bridge_weight": 3.0,
        "gap_weight": 3.2,
        "salvage_weight": 2.2,
        "bgv1_weight": 3.2,
        "selection_metric": "constrained_val_score",
    },
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"weight", "state_dict"}}


def split_pairs(datasets: Sequence[dict[str, Any]], split: str, *, include_readiness_only: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.get("readiness_eligible", True):
            continue
        name = str(dataset.get("dataset_name") or "")
        if not include_readiness_only and any(token in name for token in READINESS_ONLY_NO_TRAIN):
            continue
        for pair in dataset.get("pairs") or []:
            if str(pair.get("split") or "all") != split:
                continue
            row = dict(pair)
            row.setdefault("source_dataset", name)
            row.setdefault("dataset_kind", dataset.get("dataset_kind"))
            rows.append(row)
    return rows


def pair_weight(pair: dict[str, Any], recipe: dict[str, Any]) -> float:
    name = str(pair.get("source_dataset") or "").lower()
    pair_type = str(pair.get("pair_type") or "").lower()
    domain = str(pair.get("domain") or "").lower()
    weight = 1.0
    if pair_type in {"old_content", "old_code", "old_domain"} or "old_content" in name or "old_code" in name:
        weight *= float(recipe["old_weight"])
    elif "bridge" in name or pair_type == "bridge":
        weight *= float(recipe["bridge_weight"])
    else:
        weight *= float(recipe["branch_weight"])
    if any(token in name for token in GAP_KEYWORDS):
        weight *= float(recipe["gap_weight"])
    if "branch_generator_v1" in name:
        weight *= float(recipe["bgv1_weight"])
    if "salvage" in name or "hidden_origin_v3" in name:
        weight *= float(recipe["salvage_weight"])
    if domain in {"reasoning", "science"} and ("hidden_origin" in name or "salvage" in name):
        weight *= 1.25
    if domain == "coding":
        weight *= 1.10
    return max(0.05, float(weight))


def prepare_matrix(pairs: Sequence[dict[str, Any]], config: str, arch: str, recipe: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    weights: list[float] = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != config_dim(config):
            continue
        x = diff.detach().cpu().to(torch.float32).flatten()
        if arch == "AntisymLinear":
            x = F.layer_norm(x, (x.shape[0],))
        rows.append(x)
        weights.append(pair_weight(pair, recipe))
    if not rows:
        return torch.empty(0, config_dim(config)), torch.empty(0)
    x = torch.stack(rows, dim=0)
    w = torch.tensor(weights, dtype=torch.float32)
    return x, w / max(float(w.mean().item()), 1e-8)


def acc_for_weight(weight: torch.Tensor, pairs: Sequence[dict[str, Any]], config: str, arch: str) -> float:
    scores = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != int(weight.numel()):
            continue
        scores.append(score_diff(weight, arch, diff))
    if not scores:
        return float("nan")
    return float(sum(1.0 if s > 0 else 0.5 if s == 0 else 0.0 for s in scores) / len(scores))


def acc_for_matrix(weight: torch.Tensor, x: torch.Tensor) -> float:
    if not x.numel():
        return float("nan")
    with torch.no_grad():
        scores = x @ weight.detach()
        return float(((scores > 0).to(torch.float32).sum() + 0.5 * (scores == 0).to(torch.float32).sum()).item() / max(int(scores.numel()), 1))


def validation_breakdown(
    candidate: dict[str, Any],
    old_val: Sequence[dict[str, Any]],
    branch_val: Sequence[dict[str, Any]],
    gap_val: Sequence[dict[str, Any]],
) -> dict[str, float]:
    weight = tensor_weight(candidate)
    if not isinstance(weight, torch.Tensor):
        return {}
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    old_acc = acc_for_weight(weight, old_val, config, arch)
    branch_acc = acc_for_weight(weight, branch_val, config, arch)
    gap_acc = acc_for_weight(weight, gap_val, config, arch)
    constrained = finite_mean([old_acc, branch_acc, gap_acc, gap_acc])
    if math.isfinite(old_acc) and old_acc < 0.80:
        constrained -= 0.25 * (0.80 - old_acc)
    return {
        "old_val_acc": old_acc,
        "branch_val_acc": branch_acc,
        "gap_val_acc": gap_acc,
        "constrained_val_score": constrained,
    }


def candidate_map(candidates: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out = {}
    for cand in candidates:
        key = (
            str(cand.get("tap_role")),
            str(cand.get("target_config")),
            str(cand.get("architecture")),
            str(cand.get("recipe")),
        )
        out[key] = cand
    return out


def train_one(
    base: dict[str, Any],
    init: dict[str, Any],
    recipe: dict[str, Any],
    train_pairs: Sequence[dict[str, Any]],
    old_val: Sequence[dict[str, Any]],
    branch_val: Sequence[dict[str, Any]],
    gap_val: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = str(base.get("target_config"))
    arch = str(base.get("architecture"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor = tensor_weight(base)
    init_weight = tensor_weight(init)
    if not isinstance(anchor, torch.Tensor) or not isinstance(init_weight, torch.Tensor):
        raise RuntimeError("missing train weights")
    x, sample_w = prepare_matrix(train_pairs, config, arch, recipe)
    if int(x.shape[0]) > MAX_TRAIN_EXAMPLES:
        g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(int(x.shape[0]), generator=g)[:MAX_TRAIN_EXAMPLES]
        x = x[idx]
        sample_w = sample_w[idx]
    x = x.to(device)
    sample_w = sample_w.to(device)
    old_val_x, _ = prepare_matrix(old_val, config, arch, recipe)
    branch_val_x, _ = prepare_matrix(branch_val, config, arch, recipe)
    gap_val_x, _ = prepare_matrix(gap_val, config, arch, recipe)
    old_val_x = old_val_x.to(device)
    branch_val_x = branch_val_x.to(device)
    gap_val_x = gap_val_x.to(device)
    param = torch.nn.Parameter(init_weight.detach().cpu().to(torch.float32).flatten().clone().to(device))
    opt = torch.optim.AdamW([param], lr=float(recipe["lr"]), weight_decay=0.01)
    anchor = anchor.detach().cpu().to(torch.float32).flatten().to(device)
    anchor_u = normed(anchor)
    anchor_norm = float(anchor.norm().item())
    best_payload: dict[str, Any] = {}
    best_score = -1e9
    selection_metric = str(recipe.get("selection_metric") or "constrained_val_score")
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad(set_to_none=True)
        logits = x @ param
        pair_loss = (F.softplus(-logits) * sample_w).mean()
        direction_loss = (1.0 - torch.dot(normed(param), anchor_u)).clamp(min=0.0)
        norm_loss = (param.norm() - anchor_norm).pow(2) / max(anchor_norm**2, 1e-8)
        loss = pair_loss + float(recipe["anchor_lambda"]) * (direction_loss + 0.1 * norm_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param], 1.0)
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            val = {
                "old_val_acc": acc_for_matrix(param, old_val_x),
                "branch_val_acc": acc_for_matrix(param, branch_val_x),
                "gap_val_acc": acc_for_matrix(param, gap_val_x),
            }
            val["constrained_val_score"] = finite_mean([val["old_val_acc"], val["branch_val_acc"], val["gap_val_acc"], val["gap_val_acc"]])
            if math.isfinite(val["old_val_acc"]) and val["old_val_acc"] < 0.80:
                val["constrained_val_score"] -= 0.25 * (0.80 - val["old_val_acc"])
            score = safe_float(val.get(selection_metric), -1.0)
            history.append({"epoch": epoch, "loss": float(loss.item()), **val})
            if score > best_score:
                best_score = score
                best_payload = {"weight": param.detach().cpu().clone(), "val": val, "epoch": epoch}
    init_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in candidate_name(init))[:96]
    trained = {
        "candidate_name": f"dualanchor_gap_repair::{base.get('tap_role')}::{recipe['recipe']}::{init_tag}::{config}::{arch}",
        "candidate_family": "layer_native_two_tap_trained",
        "tap_role": base.get("tap_role"),
        "recipe": str(recipe["recipe"]),
        "target_config": config,
        "architecture": arch,
        "weight": best_payload["weight"],
        "state_dict": {"linear.weight": best_payload["weight"].reshape(1, -1)},
        "old_source": base.get("old_source"),
        "init_source": init.get("candidate_name"),
        "best_epoch": best_payload["epoch"],
        **best_payload["val"],
    }
    log = {
        "candidate_name": trained["candidate_name"],
        "base": base.get("candidate_name"),
        "init": init.get("candidate_name"),
        "recipe": recipe,
        "train_examples": int(x.shape[0]),
        "selected_epoch": best_payload["epoch"],
        "selection_metric": selection_metric,
        "history": history,
        **best_payload["val"],
    }
    return trained, log


def select_initializers(base_candidates: Sequence[dict[str, Any]], old_val: Sequence[dict[str, Any]], branch_val: Sequence[dict[str, Any]], gap_val: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in ANCHOR_GROUPS:
        for config in LAYER_CONFIGS:
            for arch in FAST_ARCHES:
                local = [
                    c
                    for c in base_candidates
                    if c.get("tap_role") == group
                    and c.get("target_config") == config
                    and c.get("architecture") == arch
                    and c.get("candidate_family") in {"layer_native_two_tap", "layer_native_two_tap_sparse", "layer_native_two_tap_transplant"}
                ]
                scored = [(validation_breakdown(c, old_val, branch_val, gap_val), c) for c in local]
                for metric in ("gap_val_acc", "constrained_val_score"):
                    for _, cand in sorted(scored, key=lambda item: safe_float(item[0].get(metric), -1.0), reverse=True)[:1]:
                        name = candidate_name(cand)
                        if name not in seen:
                            seen.add(name)
                            selected.append(cand)
    return selected


def train_candidates(base_candidates: Sequence[dict[str, Any]], train_pairs: Sequence[dict[str, Any]], old_val: Sequence[dict[str, Any]], branch_val: Sequence[dict[str, Any]], gap_val: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = candidate_map(base_candidates)
    trained: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for group in ANCHOR_GROUPS:
        for config in LAYER_CONFIGS:
            for arch in FAST_ARCHES:
                old = by_key.get((group, config, arch, "old_only"))
                if old is None:
                    continue
                for recipe in REPAIR_RECIPES:
                    preferred = by_key.get((group, config, arch, str(recipe["init_recipe"])))
                    recipe_inits = [preferred or old]
                    seen: set[str] = set()
                    for init in recipe_inits:
                        name = candidate_name(init)
                        if name in seen:
                            continue
                        seen.add(name)
                        try:
                            cand, log = train_one(old, init, recipe, train_pairs, old_val, branch_val, gap_val)
                        except Exception as exc:
                            logs.append({"candidate": f"{group}/{config}/{arch}/{recipe['recipe']}/{name}", "error": str(exc)})
                            continue
                        trained.append(cand)
                        logs.append(log)
                        print(f"trained {cand['candidate_name']} old={log.get('old_val_acc'):.4f} branch={log.get('branch_val_acc'):.4f} gap={log.get('gap_val_acc'):.4f}", flush=True)
    return trained, logs


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


def pair_scores_for_candidate(candidate: dict[str, Any], pairs: Sequence[dict[str, Any]]) -> list[float]:
    weight = tensor_weight(candidate)
    if not isinstance(weight, torch.Tensor):
        return []
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    scores: list[float] = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != int(weight.numel()):
            continue
        scores.append(score_diff(weight, arch, diff))
    return scores


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


def bundle_pair_scores(bundle: dict[str, Any], by_name: dict[str, dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[float]:
    member_scores: list[list[float]] = []
    for name in bundle.get("members", []) or []:
        member = by_name.get(str(name))
        if member is None:
            continue
        scores = pair_scores_for_candidate(member, pairs)
        if scores:
            member_scores.append(normalized(scores))
    if not member_scores:
        return []
    n = min(len(scores) for scores in member_scores)
    return [finite_mean(scores[i] for scores in member_scores) for i in range(n)]


def bundle_group_scores(group: Sequence[dict[str, Any]], bundle: dict[str, Any], by_name: dict[str, dict[str, Any]]) -> list[float]:
    member_scores: list[list[float]] = []
    for name in bundle.get("members", []) or []:
        member = by_name.get(str(name))
        if member is None:
            continue
        weight = tensor_weight(member)
        if not isinstance(weight, torch.Tensor):
            continue
        config = str(member.get("target_config"))
        arch = str(member.get("architecture"))
        scores = []
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
            scores.append(finite_mean(vals))
        if ok and len(scores) == len(group):
            member_scores.append(normalized(scores))
    if not member_scores:
        return []
    return [finite_mean(scores[i] for scores in member_scores) for i in range(len(group))]


def reference_lookups() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not LAYER_NATIVE_REFERENCE_JSON.exists():
        return {}, {}
    payload = json.loads(LAYER_NATIVE_REFERENCE_JSON.read_text(encoding="utf-8"))
    pair_lookup = {}
    for section in ("domain_summary", "branch_pair_summary"):
        for row in (payload.get(section) or {}).get("datasets") or []:
            pair_lookup[str(row.get("dataset_name"))] = row
    group_lookup = {}
    for row in (payload.get("branch_group_summary") or {}).get("datasets") or []:
        group_lookup[str(row.get("dataset_name"))] = row
    return pair_lookup, group_lookup


def eval_bundle_on_pairs(bundle: dict[str, Any], by_name: dict[str, dict[str, Any]], dataset: dict[str, Any], pair_reference_lookup: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    pairs, eval_split = filter_pairs_for_primary(dataset.get("pairs") or [])
    scores = bundle_pair_scores(bundle, by_name, pairs)
    ref = (pair_reference_lookup or {}).get(str(dataset.get("dataset_name"))) or {}
    best_ref_acc = safe_float(ref.get("best_reference_accuracy"), float("nan"))
    acc = accuracy_from_scores(scores)
    return {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_kind": dataset.get("dataset_kind"),
        "readiness_eligible": bool(dataset.get("readiness_eligible")),
        "eval_split": eval_split,
        "pair_count": len(scores),
        "selected_bundle_accuracy": acc,
        "best_reference_accuracy": best_ref_acc,
        "delta_selected_minus_reference": acc - best_ref_acc if math.isfinite(acc) and math.isfinite(best_ref_acc) else float("nan"),
        "matches_or_exceeds_reference": bool(math.isfinite(acc) and math.isfinite(best_ref_acc) and acc + 1e-9 >= best_ref_acc),
        "selected_bundle": bundle.get("bundle_name"),
        "best_reference": ref.get("best_reference"),
        "best_reference_family": ref.get("best_reference_family"),
    }


def eval_bundle_on_groups(bundle: dict[str, Any], by_name: dict[str, dict[str, Any]], dataset: dict[str, Any], group_reference_lookup: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics = []
    for group in dataset.get("groups") or []:
        scores = bundle_group_scores(group, bundle, by_name)
        if not scores:
            continue
        order = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]
        metric = group_metric_from_order(group, order, k=4)
        if metric:
            metrics.append(metric)
    ref = (group_reference_lookup or {}).get(str(dataset.get("dataset_name"))) or {}
    best_ref_ret = safe_float(ref.get("best_reference_retention"), float("nan"))
    ret = finite_mean(m.get("oracle_retention") for m in metrics)
    false_prune = finite_mean(m.get("false_prune_rate") for m in metrics)
    return {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_kind": dataset.get("dataset_kind"),
        "readiness_eligible": bool(dataset.get("readiness_eligible")),
        "group_count": len(metrics),
        "selected_bundle_retention": ret,
        "selected_bundle_false_prune": false_prune,
        "selected_bundle_best_reward": finite_mean(m.get("best_selected_reward") for m in metrics),
        "best_reference_retention": best_ref_ret,
        "delta_selected_minus_reference": ret - best_ref_ret if math.isfinite(ret) and math.isfinite(best_ref_ret) else float("nan"),
        "matches_or_exceeds_reference": bool(math.isfinite(ret) and math.isfinite(best_ref_ret) and ret + 1e-9 >= best_ref_ret),
        "selected_bundle": bundle.get("bundle_name"),
        "best_reference": ref.get("best_reference"),
        "best_reference_family": ref.get("best_reference_family"),
    }


def build_val_dataset(datasets: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out = []
    for dataset in datasets:
        name = str(dataset.get("dataset_name") or "")
        if any(token in name for token in READINESS_ONLY_NO_TRAIN):
            continue
        pairs = []
        for pair in dataset.get("pairs") or []:
            if str(pair.get("split") or "all") != split:
                continue
            row = dict(pair)
            row["split"] = "all"
            row.setdefault("source_dataset", name)
            pairs.append(row)
        if pairs:
            out.append({**dataset, "pairs": pairs})
    return out


def select_bundle_on_validation(
    bundles: Sequence[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    old_val_datasets: Sequence[dict[str, Any]],
    branch_val_datasets: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    repair_recipes = {str(r["recipe"]) for r in REPAIR_RECIPES}
    eligible = [
        b
        for b in bundles
        if str(b.get("bundle_name", "")).startswith("bundle::two_tap_equal::")
        and str(b.get("recipe")) in repair_recipes
        and str(b.get("architecture")) in FAST_ARCHES
    ]
    for bundle in eligible:
        old_accs = [accuracy_from_scores(bundle_pair_scores(bundle, by_name, filter_pairs_for_primary(d.get("pairs") or [])[0])) for d in old_val_datasets]
        branch_accs = [accuracy_from_scores(bundle_pair_scores(bundle, by_name, filter_pairs_for_primary(d.get("pairs") or [])[0])) for d in branch_val_datasets]
        gap_accs = [
            acc
            for acc, d in zip(branch_accs, branch_val_datasets)
            if any(token in str(d.get("dataset_name", "")).lower() for token in GAP_KEYWORDS)
        ]
        old_accs = [a for a in old_accs if math.isfinite(a)]
        branch_accs = [a for a in branch_accs if math.isfinite(a)]
        gap_accs = [a for a in gap_accs if math.isfinite(a)]
        old_min = min(old_accs) if old_accs else float("nan")
        score = finite_mean(branch_accs + gap_accs + gap_accs + old_accs)
        if math.isfinite(old_min) and old_min < 0.60:
            score -= 0.5 * (0.60 - old_min)
        rows.append(
            {
                "bundle_name": bundle.get("bundle_name"),
                "validation_score": score,
                "old_val_mean_accuracy": finite_mean(old_accs),
                "old_val_min_accuracy": old_min,
                "branch_val_mean_accuracy": finite_mean(branch_accs),
                "gap_val_mean_accuracy": finite_mean(gap_accs),
                "old_val_pass": bool(not old_accs or old_min >= 0.60),
            }
        )
    selected = max(rows, key=lambda r: safe_float(r.get("validation_score"), -1e9), default={})
    by_bundle = {str(b.get("bundle_name")): b for b in bundles}
    return by_bundle.get(str(selected.get("bundle_name")), eligible[0] if eligible else {}), rows


def main() -> int:
    ensure_root()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    print(f"DualAnchor branch-gap repair device = {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    refs = source_candidates()
    layer_refs = [r for r in refs if r.get("target_config") in LAYER_CONFIGS and r.get("architecture") in FAST_ARCHES]
    base_candidates, source_summary = build_layer_native_candidates(layer_refs)
    old_datasets = old_domain_pair_datasets()
    branch_datasets = branch_pair_datasets()

    old_train = split_pairs(old_datasets, "train")
    old_val = split_pairs(old_datasets, "val")
    branch_train = split_pairs(branch_datasets, "train")
    branch_val = split_pairs(branch_datasets, "val")
    gap_train = [p for p in branch_train if any(token in str(p.get("source_dataset", "")).lower() for token in GAP_KEYWORDS)]
    gap_val = [p for p in branch_val if any(token in str(p.get("source_dataset", "")).lower() for token in GAP_KEYWORDS) and not any(token in str(p.get("source_dataset", "")) for token in READINESS_ONLY_NO_TRAIN)]
    train_pairs = old_train + branch_train + gap_train

    trained, logs = train_candidates(base_candidates, train_pairs, old_val, branch_val, gap_val)
    candidates = base_candidates + trained
    bundles = bundle_specs(candidates)
    by_name = {candidate_name(c): c for c in candidates}

    old_val_datasets = build_val_dataset(old_datasets, "val")
    branch_val_datasets = build_val_dataset(branch_datasets, "val")
    selected_bundle, selection_rows = select_bundle_on_validation(bundles, by_name, old_val_datasets, branch_val_datasets)
    pair_ref_lookup, group_ref_lookup = reference_lookups()

    selected_pair_rows = [eval_bundle_on_pairs(selected_bundle, by_name, dataset, pair_ref_lookup) for dataset in old_datasets + branch_datasets]
    selected_group_rows = [eval_bundle_on_groups(selected_bundle, by_name, dataset, group_ref_lookup) for dataset in group_datasets_any()]

    domain_summary = {
        "datasets": [
            {
                "dataset_name": row.get("dataset_name"),
                "dataset_kind": row.get("dataset_kind"),
                "readiness_eligible": row.get("readiness_eligible"),
                "eval_split": row.get("eval_split"),
                "pair_count": row.get("pair_count"),
                "best_two_tap": row.get("selected_bundle"),
                "best_two_tap_family": "layer_native_bundle",
                "best_two_tap_accuracy": row.get("selected_bundle_accuracy"),
                "best_reference": row.get("best_reference"),
                "best_reference_family": row.get("best_reference_family"),
                "best_reference_accuracy": row.get("best_reference_accuracy"),
                "delta_two_minus_reference": row.get("delta_selected_minus_reference"),
                "matches_or_exceeds_reference": row.get("matches_or_exceeds_reference"),
            }
            for row in selected_pair_rows
            if row.get("dataset_kind") == "old_domain_pair"
        ]
    }
    branch_summary = {
        "datasets": [
            {
                "dataset_name": row.get("dataset_name"),
                "dataset_kind": row.get("dataset_kind"),
                "readiness_eligible": row.get("readiness_eligible"),
                "eval_split": row.get("eval_split"),
                "pair_count": row.get("pair_count"),
                "best_two_tap": row.get("selected_bundle"),
                "best_two_tap_family": "layer_native_bundle",
                "best_two_tap_accuracy": row.get("selected_bundle_accuracy"),
                "best_reference": row.get("best_reference"),
                "best_reference_family": row.get("best_reference_family"),
                "best_reference_accuracy": row.get("best_reference_accuracy"),
                "delta_two_minus_reference": row.get("delta_selected_minus_reference"),
                "matches_or_exceeds_reference": row.get("matches_or_exceeds_reference"),
            }
            for row in selected_pair_rows
            if row.get("dataset_kind") == "branch_pair"
        ]
    }
    group_summary = {
        "datasets": [
            {
                "dataset_name": row.get("dataset_name"),
                "readiness_eligible": row.get("readiness_eligible"),
                "group_count": row.get("group_count"),
                "best_two_tap": row.get("selected_bundle"),
                "best_two_tap_family": "layer_native_bundle",
                "best_two_tap_retention": row.get("selected_bundle_retention"),
                "best_two_tap_false_prune": row.get("selected_bundle_false_prune"),
                "best_two_tap_top1_reward": None,
                "best_reference": row.get("best_reference"),
                "best_reference_family": row.get("best_reference_family"),
                "best_reference_retention": row.get("best_reference_retention"),
                "best_reference_false_prune": None,
                "best_reference_top1_reward": None,
                "delta_two_minus_reference": row.get("delta_selected_minus_reference"),
                "matches_or_exceeds_reference": row.get("matches_or_exceeds_reference"),
            }
            for row in selected_group_rows
        ]
    }
    verdict, ready = readiness(domain_summary, branch_summary, group_summary)

    selected_domain_failures = [
        r
        for r in selected_pair_rows
        if r.get("dataset_kind") == "old_domain_pair"
        and r.get("readiness_eligible")
        and int(r.get("pair_count") or 0) > 0
        and not r.get("matches_or_exceeds_reference")
    ]
    selected_branch_failures = [
        r
        for r in selected_pair_rows
        if r.get("dataset_kind") == "branch_pair"
        and r.get("readiness_eligible")
        and int(r.get("pair_count") or 0) > 0
        and not r.get("matches_or_exceeds_reference")
    ]
    selected_group_failures = [
        r
        for r in selected_group_rows
        if r.get("readiness_eligible") and int(r.get("group_count") or 0) > 0 and not r.get("matches_or_exceeds_reference")
    ]
    selected_domain_ok = not selected_domain_failures
    selected_branch_ok = not selected_branch_failures and not selected_group_failures

    if verdict == "LAYER_NATIVE_TWO_TAP_READY" and selected_domain_ok and selected_branch_ok:
        status = "DUALANCHOR_BRANCH_READY"
    elif ready["domain_ok"] and ready["branch_ok"]:
        status = "DUALANCHOR_BRANCH_READY_BEST_CANDIDATE_ONLY"
    elif ready["domain_ok"] and (not ready["branch_ok"]):
        status = "DUALANCHOR_BRANCH_GAP_REDUCED"
    elif ready["branch_ok"]:
        status = "DUALANCHOR_BRANCH_READY_DOMAIN_GAP"
    else:
        status = "DUALANCHOR_BRANCH_NOT_READY"

    no_train_notes = {
        "readiness_only_no_train_datasets": list(READINESS_ONLY_NO_TRAIN),
        "hidden_origin_v3_high_yield_has_no_train_split": True,
        "salvage_old_frozen_tap_clean_primary_has_no_train_split": True,
    }
    payload = {
        "BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT": status,
        "BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT": verdict,
        "status": status,
        "readiness_verdict": verdict,
        "selected_bundle": selected_bundle.get("bundle_name"),
        "selected_bundle_validation": next((r for r in selection_rows if r.get("bundle_name") == selected_bundle.get("bundle_name")), {}),
        "train_counts": {
            "old_train": len(old_train),
            "old_val": len(old_val),
            "branch_train": len(branch_train),
            "branch_val": len(branch_val),
            "gap_train": len(gap_train),
            "gap_val": len(gap_val),
            "train_pairs_after_gap_duplication": len(train_pairs),
        },
        "source_summary": source_summary,
        "base_candidate_count": len(base_candidates),
        "trained_candidate_count": len(trained),
        "bundle_count": len(bundles),
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_summary,
        "branch_group_summary": group_summary,
        "readiness": ready,
        "selected_bundle_summary": {
            "domain_ok": selected_domain_ok,
            "branch_ok": selected_branch_ok,
            "domain_failures": selected_domain_failures,
            "branch_pair_failures": selected_branch_failures,
            "branch_group_failures": selected_group_failures,
        },
        "no_train_notes": no_train_notes,
        "anti_leakage": {
            "train_splits_only_for_weight_updates": True,
            "validation_splits_only_for_epoch_and_bundle_selection": True,
            "heldout_test_domain_probe_not_used_for_selection": True,
            "readiness_only_val_sets_not_used_for_selection": True,
            "no_ouro_training": True,
            "no_old_tap_registry_update": True,
            "no_action_steering": True,
            "no_production_routing_change": True,
        },
    }
    torch.save(
        {
            "summary": payload,
            "candidates": candidates,
            "trained": trained,
            "bundles": bundles,
            "selected_bundle": selected_bundle,
            "references": layer_refs,
            "training_logs": logs,
        },
        ARTIFACT_PT,
    )
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_json(TRAINING_LOG_JSON, {"logs": logs, "selection_rows": selection_rows})
    write_csv(PAIR_ROWS_CSV, selected_pair_rows)
    write_csv(GROUP_ROWS_CSV, selected_group_rows)
    write_csv(SELECTED_ROWS_CSV, selected_pair_rows + selected_group_rows)
    write_csv(MODEL_ROWS_CSV, [clean_row(row) for row in trained])

    lines = [
        "# DualAnchor Branch-Gap Repair v1",
        "",
        f"BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = {status}",
        f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
        "",
        f"- selected bundle: `{selected_bundle.get('bundle_name')}`",
        f"- old train/val pairs: `{len(old_train)}` / `{len(old_val)}`",
        f"- branch train/val pairs: `{len(branch_train)}` / `{len(branch_val)}`",
        f"- gap train/val pairs: `{len(gap_train)}` / `{len(gap_val)}`",
        f"- trained candidate heads: `{len(trained)}`",
        f"- bundle count: `{len(bundles)}`",
        f"- best-candidate domain ok: `{ready['domain_ok']}`",
        f"- best-candidate branch ok: `{ready['branch_ok']}`",
        f"- selected-bundle domain ok: `{selected_domain_ok}`",
        f"- selected-bundle branch ok: `{selected_branch_ok}`",
        "",
        "This run trained copied DualAnchor layer-native heads only. It did not update old taps, registries, Ouro weights, routing, or steering.",
        "",
        "## Old-Domain Datasets",
        "",
    ]
    lines.extend(md_table(domain_summary.get("datasets", []), ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Branch Pair Datasets", ""])
    lines.extend(md_table(branch_summary.get("datasets", []), ["dataset_name", "readiness_eligible", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Branch Group Datasets", ""])
    lines.extend(md_table(group_summary.get("datasets", []), ["dataset_name", "group_count", "best_two_tap_retention", "best_reference_retention", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Selected Bundle Gaps", ""])
    if selected_domain_failures:
        lines.extend(["### Domain", ""])
        lines.extend(md_table(selected_domain_failures, ["dataset_name", "pair_count", "selected_bundle_accuracy", "best_reference_accuracy", "delta_selected_minus_reference", "best_reference_family"]))
    if selected_branch_failures or selected_group_failures:
        lines.extend(["### Branch", ""])
        lines.extend(md_table(selected_branch_failures, ["dataset_name", "pair_count", "selected_bundle_accuracy", "best_reference_accuracy", "delta_selected_minus_reference", "best_reference_family"]))
        lines.extend(md_table(selected_group_failures, ["dataset_name", "group_count", "selected_bundle_retention", "best_reference_retention", "delta_selected_minus_reference", "best_reference_family"]))
    lines.extend(["", "## No-Train Readiness Gaps", ""])
    lines.append("Some readiness failures cannot be directly trained without leakage because their readiness split is the only split available.")
    lines.extend(md_table([no_train_notes], ["hidden_origin_v3_high_yield_has_no_train_split", "salvage_old_frozen_tap_clean_primary_has_no_train_split", "readiness_only_no_train_datasets"]))
    lines.extend(["", "## Files", "", f"- artifact: `{rel(ARTIFACT_PT)}`", f"- report json: `{rel(REPORT_JSON)}`", f"- pair rows: `{rel(PAIR_ROWS_CSV)}`", f"- group rows: `{rel(GROUP_ROWS_CSV)}`", f"- selected rows: `{rel(SELECTED_ROWS_CSV)}`"])
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# DualAnchor Branch-Gap Repair v1",
            "",
            f"BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = {status}",
            f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
            "",
            "`DualAnchor` is the branch-valid `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` tap pair evaluated as layer-native heads at `24_L4`, `36_L4`, and `47_L4`.",
            "",
            "## Result",
            "",
            f"- selected bundle: `{selected_bundle.get('bundle_name')}`",
            f"- best-candidate domain ok: `{ready['domain_ok']}`",
            f"- best-candidate branch ok: `{ready['branch_ok']}`",
            f"- selected-bundle domain ok: `{selected_domain_ok}`",
            f"- selected-bundle branch ok: `{selected_branch_ok}`",
            f"- status: `{status}`",
            "",
            "## Interpretation",
            "",
            "This pass targeted the remaining BGV1, hidden-origin-v3/salvage, and universal-bridge branch gaps while preserving old-domain behavior. It trained only copied tap vectors under a new artifact path.",
            "",
            f"Report: `{rel(REPORT_MD)}`.",
        ],
    )
    section_title = "## DualAnchor branch-gap repair v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = {status}`; readiness verdict `{verdict}`. Targeted copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-native heads at `24_L4`, `36_L4`, and `47_L4` on train-split branch-gap examples with old-domain preservation. Best-candidate domain ok `{ready['domain_ok']}`; best-candidate branch ok `{ready['branch_ok']}`; selected-bundle branch ok `{selected_branch_ok}`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [section_title, "", f"Added `{rel(DOC_MD)}`. Status: `{status}`; readiness `{verdict}`."]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)

    print(f"BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = {status}", flush=True)
    print(f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}", flush=True)
    print(f"selected_bundle = {selected_bundle.get('bundle_name')}", flush=True)
    print(f"best_domain_ok = {ready['domain_ok']} best_branch_ok = {ready['branch_ok']}", flush=True)
    print(f"selected_domain_ok = {selected_domain_ok} selected_branch_ok = {selected_branch_ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
