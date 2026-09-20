"""Constrained layer-native two-tap training v1.

Train copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local taps at
24_L4, 36_L4, and 47_L4 on actual old-domain plus branch/bridge labels, while
penalizing movement away from the old anchors.

This creates new diagnostic tap copies only. It does not train Ouro, mutate
checkpoints/tokenizers, update old tap registries, run wrapper/local-agent or
Hunter-Seeker code, apply steering, or change routing.
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
from bg_merged_tap_v1_common import BRIDGE_PT, finite_mean, json_default, normed, pair_diff, safe_float, score_diff
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
    group_datasets_any,
    group_metric_from_order,
    group_reward,
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


OUT_ROOT = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "constrained_train_eval.json"
REPORT_MD = OUT_ROOT / "constrained_train_eval.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
TRAINING_LOG_JSON = OUT_ROOT / "training_log.json"
PAIR_ROWS_CSV = OUT_ROOT / "constrained_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "constrained_group_rows.csv"
MODEL_ROWS_CSV = OUT_ROOT / "trained_candidate_rows.csv"
ARTIFACT_PT = OUT_ROOT / "layer_native_two_tap_constrained_train_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_layer_native_two_tap_constrained_train_v1.md"

EPOCHS = 80
SEED = 20260530
MAX_TRAIN_EXAMPLES_PER_FAMILY = 12000

TRAIN_RECIPES = [
    {
        "recipe": "old_preserve_balanced",
        "init_recipe": "old_only",
        "lr": 3e-4,
        "anchor_lambda": 0.20,
        "old_weight": 2.0,
        "branch_weight": 1.0,
        "bridge_weight": 1.0,
    },
    {
        "recipe": "branch_bridge_balanced",
        "init_recipe": "old_plus_branch_bridge_70_15_15",
        "lr": 3e-4,
        "anchor_lambda": 0.08,
        "old_weight": 1.0,
        "branch_weight": 1.8,
        "bridge_weight": 1.8,
    },
    {
        "recipe": "bridge_heavy_conservative",
        "init_recipe": "old_plus_branch_bridge_75_15_10",
        "lr": 3e-4,
        "anchor_lambda": 0.12,
        "old_weight": 1.2,
        "branch_weight": 1.2,
        "bridge_weight": 2.4,
    },
    {
        "recipe": "branch_heavy_conservative",
        "init_recipe": "old_plus_branch_85_15",
        "lr": 3e-4,
        "anchor_lambda": 0.12,
        "old_weight": 1.2,
        "branch_weight": 2.4,
        "bridge_weight": 1.2,
    },
    {
        "recipe": "low_anchor_branch_bridge",
        "init_recipe": "old_plus_branch_bridge_70_15_15",
        "lr": 1e-4,
        "anchor_lambda": 0.02,
        "old_weight": 0.8,
        "branch_weight": 2.2,
        "bridge_weight": 2.2,
    },
]

ADAPTIVE_TRAIN_RECIPES = [
    {
        "recipe": "adaptive_branch_from_val",
        "lr": 1e-3,
        "anchor_lambda": 0.0,
        "old_weight": 0.20,
        "branch_weight": 3.0,
        "bridge_weight": 3.0,
        "selection_metric": "branch_val_acc",
    },
    {
        "recipe": "adaptive_branch_anchor_light",
        "lr": 3e-4,
        "anchor_lambda": 0.01,
        "old_weight": 0.50,
        "branch_weight": 3.0,
        "bridge_weight": 3.0,
        "selection_metric": "branch_val_acc",
    },
    {
        "recipe": "adaptive_balanced_rescue",
        "lr": 3e-4,
        "anchor_lambda": 0.02,
        "old_weight": 1.0,
        "branch_weight": 2.0,
        "bridge_weight": 2.0,
        "selection_metric": "balanced_val_acc",
    },
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def split_pairs(datasets: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.get("readiness_eligible", True):
            continue
        for pair in dataset.get("pairs") or []:
            if str(pair.get("split") or "all") != split:
                continue
            row = dict(pair)
            row.setdefault("source_dataset", dataset.get("dataset_name"))
            row.setdefault("dataset_kind", dataset.get("dataset_kind"))
            out.append(row)
    return out


def pair_type_weight(pair: dict[str, Any], recipe: dict[str, Any]) -> float:
    pair_type = str(pair.get("pair_type") or "")
    dataset = str(pair.get("source_dataset") or "")
    if pair_type in {"old_content", "old_code", "old_domain"} or "old" in dataset:
        return float(recipe["old_weight"])
    if pair_type == "bridge" or "bridge" in dataset:
        return float(recipe["bridge_weight"])
    return float(recipe["branch_weight"])


def prepare_matrix(pairs: Sequence[dict[str, Any]], config: str, arch: str, recipe: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rows = []
    weights = []
    domains = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != config_dim(config):
            continue
        rows.append(diff.detach().cpu().to(torch.float32).flatten())
        weights.append(pair_type_weight(pair, recipe))
        domains.append(str(pair.get("domain") or pair.get("source_dataset") or "unknown"))
    if not rows:
        return torch.empty(0, config_dim(config)), torch.empty(0), []
    x = torch.stack(rows, dim=0).to(torch.float32)
    if arch == "AntisymLinear":
        x = F.layer_norm(x, (x.shape[1],))
    w = torch.tensor(weights, dtype=torch.float32)
    return x, w / max(float(w.mean().item()), 1e-8), domains


def acc_for_weight(weight: torch.Tensor, x: torch.Tensor) -> float:
    if not x.numel():
        return float("nan")
    scores = x @ weight.detach().to(torch.float32)
    return float(((scores > 0).to(torch.float32).sum() + 0.5 * (scores == 0).to(torch.float32).sum()).item() / max(int(scores.numel()), 1))


def validation_breakdown(candidate: dict[str, Any], old_val: Sequence[dict[str, Any]], branch_val: Sequence[dict[str, Any]]) -> dict[str, float]:
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    weight = tensor_weight(candidate)
    if not isinstance(weight, torch.Tensor):
        return {}
    dummy_recipe = {"old_weight": 1.0, "branch_weight": 1.0, "bridge_weight": 1.0}
    old_x, _, _ = prepare_matrix(old_val, config, arch, dummy_recipe)
    branch_x, _, _ = prepare_matrix(branch_val, config, arch, dummy_recipe)
    return {
        "old_val_acc": acc_for_weight(weight, old_x),
        "branch_val_acc": acc_for_weight(weight, branch_x),
        "balanced_val_acc": finite_mean([acc_for_weight(weight, old_x), acc_for_weight(weight, branch_x)]),
    }


def train_one(
    base: dict[str, Any],
    init: dict[str, Any],
    recipe: dict[str, Any],
    train_pairs: Sequence[dict[str, Any]],
    old_val: Sequence[dict[str, Any]],
    branch_val: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = str(base.get("target_config"))
    arch = str(base.get("architecture"))
    anchor = tensor_weight(base)
    init_weight = tensor_weight(init)
    if not isinstance(anchor, torch.Tensor) or not isinstance(init_weight, torch.Tensor):
        raise RuntimeError("missing train weights")
    x, sample_w, _ = prepare_matrix(train_pairs, config, arch, recipe)
    if int(x.shape[0]) > MAX_TRAIN_EXAMPLES_PER_FAMILY:
        g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(int(x.shape[0]), generator=g)[:MAX_TRAIN_EXAMPLES_PER_FAMILY]
        x = x[idx]
        sample_w = sample_w[idx]
    param = torch.nn.Parameter(init_weight.detach().cpu().to(torch.float32).flatten().clone())
    opt = torch.optim.AdamW([param], lr=float(recipe["lr"]), weight_decay=0.01)
    anchor_u = normed(anchor)
    best_payload: dict[str, Any] = {}
    best_score = -1e9
    selection_metric = str(recipe.get("selection_metric") or "balanced_val_acc")
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad(set_to_none=True)
        scores = x @ param
        pair_loss = (F.softplus(-scores) * sample_w).mean()
        direction_loss = (1.0 - torch.dot(normed(param), anchor_u)).clamp(min=0.0)
        norm_loss = (param.norm() - anchor.norm()).pow(2) / max(float(anchor.norm().item()) ** 2, 1e-8)
        loss = pair_loss + float(recipe["anchor_lambda"]) * (direction_loss + 0.1 * norm_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param], 1.0)
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            cand = {
                **{k: v for k, v in base.items() if k not in {"weight", "state_dict"}},
                "weight": param.detach().cpu().clone(),
                "state_dict": {"linear.weight": param.detach().cpu().clone().reshape(1, -1)},
            }
            val = validation_breakdown(cand, old_val, branch_val)
            score = safe_float(val.get(selection_metric), -1.0)
            history.append({"epoch": epoch, "loss": float(loss.item()), **val})
            if score > best_score:
                best_score = score
                best_payload = {"weight": param.detach().cpu().clone(), "val": val, "epoch": epoch}
    init_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in candidate_name(init))[:96]
    trained = {
        "candidate_name": f"trained_layer_native::{base.get('tap_role')}::{recipe['recipe']}::{init_tag}::{config}::{arch}",
        "candidate_family": "layer_native_two_tap_trained",
        "tap_role": base.get("tap_role"),
        "recipe": str(recipe["recipe"]),
        "init_recipe": str(init.get("recipe")),
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
        "history": history,
        "selected_epoch": best_payload["epoch"],
        "selection_metric": selection_metric,
        **best_payload["val"],
    }
    return trained, log


def train_candidates(base_candidates: Sequence[dict[str, Any]], train_pairs: Sequence[dict[str, Any]], old_val: Sequence[dict[str, Any]], branch_val: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cand in base_candidates:
        key = (str(cand.get("tap_role")), str(cand.get("target_config")), str(cand.get("architecture")), str(cand.get("recipe")))
        by_key[key] = cand
    trained: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for group in ANCHOR_GROUPS:
        for config in LAYER_CONFIGS:
            for arch in PRIMARY_ARCHES:
                old = by_key.get((group, config, arch, "old_only"))
                if old is None:
                    continue
                for recipe in TRAIN_RECIPES:
                    init = by_key.get((group, config, arch, str(recipe["init_recipe"])), old)
                    try:
                        cand, log = train_one(old, init, recipe, train_pairs, old_val, branch_val)
                    except Exception as exc:
                        logs.append({"candidate": f"{group}/{config}/{arch}/{recipe['recipe']}", "error": str(exc)})
                        continue
                    trained.append(cand)
                    logs.append(log)
                scored_inits = []
                for init in base_candidates:
                    if init.get("tap_role") != group or init.get("target_config") != config or init.get("architecture") != arch:
                        continue
                    if init.get("candidate_family") not in {"layer_native_two_tap", "layer_native_two_tap_sparse", "layer_native_two_tap_transplant"}:
                        continue
                    val = validation_breakdown(init, old_val, branch_val)
                    scored_inits.append((val, init))
                selected_inits: list[dict[str, Any]] = []
                seen_init_names: set[str] = set()
                for key in ("branch_val_acc", "balanced_val_acc", "old_val_acc"):
                    for _, init in sorted(scored_inits, key=lambda item: safe_float(item[0].get(key), -1.0), reverse=True)[:3]:
                        name = candidate_name(init)
                        if name not in seen_init_names:
                            seen_init_names.add(name)
                            selected_inits.append(init)
                for init in selected_inits:
                    for recipe in ADAPTIVE_TRAIN_RECIPES:
                        try:
                            cand, log = train_one(old, init, recipe, train_pairs, old_val, branch_val)
                        except Exception as exc:
                            logs.append({"candidate": f"{group}/{config}/{arch}/{recipe['recipe']}/{candidate_name(init)}", "error": str(exc)})
                            continue
                        trained.append(cand)
                        logs.append(log)
    return trained, logs


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


def pair_eval_rows_for_dataset(dataset: dict[str, Any], candidates: Sequence[dict[str, Any]], refs: Sequence[dict[str, Any]], bundles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs, eval_split = filter_pairs_for_primary(dataset.get("pairs") or [])
    rows: list[dict[str, Any]] = []
    all_single = list(candidates) + list(refs)
    for candidate in all_single:
        scores, domains = pair_scores_for_candidate(candidate, pairs)
        if not scores:
            continue
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
            }
        )
    by_name = {str(c.get("candidate_name")): c for c in candidates}
    for bundle in bundles:
        member_rows = [by_name[name] for name in bundle.get("members", []) if name in by_name]
        member_scores: list[list[float]] = []
        domains: list[str] | None = None
        for member in member_rows:
            scores, ds = pair_scores_for_candidate(member, pairs)
            if not scores:
                continue
            domains = ds if domains is None else domains
            member_scores.append(normalized(scores))
        if not member_scores or domains is None:
            continue
        n = min(len(scores) for scores in member_scores)
        scores = [finite_mean(ms[i] for ms in member_scores) for i in range(n)]
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
            }
        )
    return rows


def group_by_domain(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("domain"))].append(row)
    return out


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


def eval_group_dataset(dataset: dict[str, Any], candidates: Sequence[dict[str, Any]], refs: Sequence[dict[str, Any]], bundles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = list(dataset.get("groups") or [])
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
                "top1_reward": finite_mean(m.get("top1_reward") for m in group_metrics),
                "regret": finite_mean(m.get("regret") for m in group_metrics),
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


def main() -> int:
    ensure_root()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    refs = source_candidates()
    layer_refs = [r for r in refs if r.get("target_config") in LAYER_CONFIGS and r.get("architecture") in PRIMARY_ARCHES]
    base_candidates, source_summary = build_layer_native_candidates(layer_refs)
    old_datasets = old_domain_pair_datasets()
    branch_datasets = branch_pair_datasets()
    old_train = split_pairs(old_datasets, "train")
    old_val = split_pairs(old_datasets, "val")
    branch_train = split_pairs(branch_datasets, "train")
    branch_val = split_pairs(branch_datasets, "val")
    train_pairs = old_train + branch_train
    trained, logs = train_candidates(base_candidates, train_pairs, old_val, branch_val)
    candidates = base_candidates + trained
    bundles = bundle_specs(candidates)

    pair_rows: list[dict[str, Any]] = []
    for dataset in old_datasets + branch_datasets:
        pair_rows.extend(pair_eval_rows_for_dataset(dataset, candidates, layer_refs, bundles))
    domain_summary = summarize_pair_dataset(pair_rows, "old_domain_pair")
    branch_summary = summarize_pair_dataset(pair_rows, "branch_pair")

    group_rows: list[dict[str, Any]] = []
    bounded_refs = []
    for fam in ("source_branch", "source_bridge", "source_universal"):
        rows = [
            r
            for r in layer_refs
            if r.get("candidate_family") == fam and r.get("target_config") in LAYER_CONFIGS and r.get("architecture") in PRIMARY_ARCHES
        ]
        bounded_refs.extend(sorted(rows, key=lambda r: safe_float(r.get("metric_score"), -1.0), reverse=True)[:24])
    for dataset in group_datasets_any():
        group_rows.extend(eval_group_dataset(dataset, candidates, bounded_refs, bundles))
    group_summary = summarize_group_rows(group_rows)
    verdict, ready = readiness(domain_summary, branch_summary, group_summary)
    if verdict == "LAYER_NATIVE_TWO_TAP_READY":
        status = "CONSTRAINED_TWO_TAP_READY"
    elif ready["branch_ok"] or ready["domain_ok"]:
        status = "CONSTRAINED_TWO_TAP_PARTIAL"
    else:
        status = "CONSTRAINED_TWO_TAP_NOT_READY"
    payload = {
        "BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT": status,
        "BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT": verdict,
        "status": status,
        "readiness_verdict": verdict,
        "train_counts": {
            "old_train": len(old_train),
            "old_val": len(old_val),
            "branch_train": len(branch_train),
            "branch_val": len(branch_val),
        },
        "source_summary": source_summary,
        "base_candidate_count": len(base_candidates),
        "trained_candidate_count": len(trained),
        "bundle_count": len(bundles),
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_summary,
        "branch_group_summary": group_summary,
        "readiness": ready,
        "anti_leakage": {
            "train_splits_only_for_weight_updates": True,
            "validation_splits_only_for_epoch_selection": True,
            "domain_probe_sets_not_used_for_training": True,
            "no_ouro_training": True,
            "no_old_tap_registry_update": True,
            "no_action_steering": True,
            "no_production_routing_change": True,
        },
    }
    torch.save({"summary": payload, "candidates": candidates, "trained": trained, "bundles": bundles, "references": layer_refs, "training_logs": logs}, ARTIFACT_PT)
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_json(TRAINING_LOG_JSON, {"logs": logs})
    write_csv(PAIR_ROWS_CSV, pair_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(MODEL_ROWS_CSV, [{k: v for k, v in row.items() if k not in {"weight", "state_dict"}} for row in trained])
    lines = [
        "# Layer-Native Two-Tap Constrained Training v1",
        "",
        f"BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = {status}",
        f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
        "",
        f"- old train/val pairs: `{len(old_train)}` / `{len(old_val)}`",
        f"- branch train/val pairs: `{len(branch_train)}` / `{len(branch_val)}`",
        f"- trained candidate heads: `{len(trained)}`",
        f"- bundle count: `{len(bundles)}`",
        f"- domain ok: `{ready['domain_ok']}`",
        f"- branch ok: `{ready['branch_ok']}`",
        "",
        "Training used train splits only for weight updates, validation splits for epoch/model diagnostics, and kept domain-probe reasoning/science/GSM8K sets as evaluation only.",
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
    lines.extend(["", "## Files", "", f"- artifact: `{rel(ARTIFACT_PT)}`", f"- report json: `{rel(REPORT_JSON)}`", f"- pair rows: `{rel(PAIR_ROWS_CSV)}`", f"- group rows: `{rel(GROUP_ROWS_CSV)}`"])
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# Layer-Native Two-Tap Constrained Training v1",
            "",
            f"BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = {status}",
            f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}",
            "",
            "Copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local heads were trained on actual train-split old-domain and branch/bridge labels with old-anchor penalties. No Ouro weights or old registries were changed.",
            "",
            "## Result",
            "",
            f"- domain ok: `{ready['domain_ok']}`",
            f"- branch ok: `{ready['branch_ok']}`",
            f"- status: `{status}`",
            "",
            f"Report: `{rel(REPORT_MD)}`.",
        ],
    )
    section_title = "## Layer-native two-tap constrained training v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = {status}`; readiness verdict `{verdict}`. Trained copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local taps at `24_L4`, `36_L4`, and `47_L4` on train-split old-domain plus branch/bridge labels with anchor preservation. Domain ok `{ready['domain_ok']}`; branch ok `{ready['branch_ok']}`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [section_title, "", f"Added `{rel(DOC_MD)}`. Status: `{status}`; readiness `{verdict}`."]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)
    print(f"BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = {status}", flush=True)
    print(f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = {verdict}", flush=True)
    print(f"domain_ok = {ready['domain_ok']} branch_ok = {ready['branch_ok']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
