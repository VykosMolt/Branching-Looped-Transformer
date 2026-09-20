"""Task-disjoint reruns for the load-bearing claims in paper draft v4.4.

This script is intentionally additive: it reads the historical frozen-feature and
branch-lineage artifacts, constructs deterministic task-level splits, refits only
small linear readouts, and writes a new audit bundle.  It does not mutate model
weights, tap registries, historical reports, or the paper draft.

Tier 1:
  * audits the actual DualAnchor training/evaluation task IDs;
  * re-aggregates the predeclared v3 clean heldout tasks captured by the v3 run;
  * re-aggregates the pre-DualAnchor scaffold after removing every train/val task;
  * evaluates a freshly refit CoreContent-style head on actual DualAnchor terminal
    branch pools (not reconstructed CoreContent candidate groups).

Tier 2:
  * repartitions all CoreContent-v2 groups by SHA-256(task_id), ignoring their
    historical row/group split labels;
  * trains code, HH/alignment, reasoning, all-domain, and random-20-HH linear heads;
  * reports a powered cross-domain transfer matrix and task-clustered intervals.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for item in (str(HERE), str(PROJECT_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import bg_core_tap_audit_v1_common as cc  # noqa: E402
import bg_corecontent_v2_features as features  # noqa: E402
import bg_selection_only_phase2_v1_common as scaffold  # noqa: E402
import run_bg_layer_native_two_tap_constrained_train_v1 as dual_train  # noqa: E402


PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
OUT_ROOT = PROBE_ROOT / "paper_v44_task_disjoint_rerun_2026-07-13"
DUAL_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31"
V3_GUARD = PROBE_ROOT / "bg_hidden_origin_diversity_v3_2026-05-18/split_guard_v3.json"
SCAFFOLD_ROOT = PROBE_ROOT / "bg_selection_only_phase2_prototype_v1_2026-05-18"
FIXED_ROOT = PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18"

DOMAINS = ("coding", "reasoning", "math", "logic", "alignment")
CONFIGS = ("24_L4", "36_L4", "47_L4")
LAYER_POS = {"24_L4": 0, "36_L4": 1, "47_L4": 2}
SPLIT_SALT = "ouro-paper-v4.4-task-disjoint-v1"
SEED = 20260713


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=("all", "tier1", "tier2"), default="all")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    tmp.replace(path)


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return str(value)


def finite_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def stable_int(*parts: Any) -> int:
    text = "::".join(str(x) for x in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def deterministic_split(task_id: str) -> str:
    # 70/10/20. SHA-256 is stable across processes/PYTHONHASHSEED values.
    bucket = stable_int(SPLIT_SALT, task_id) % 10_000
    if bucket < 7000:
        return "train"
    if bucket < 8000:
        return "val"
    return "heldout"


def crossing_report(by_split: dict[str, set[str]]) -> dict[str, Any]:
    train = set(by_split.get("train", set()))
    val = set(by_split.get("val", set()))
    heldout = set(by_split.get("heldout", set()))
    return {
        "train_task_ids": sorted(train),
        "val_task_ids": sorted(val),
        "heldout_task_ids": sorted(heldout),
        "train_val_crossing_task_ids": sorted(train & val),
        "train_heldout_crossing_task_ids": sorted(train & heldout),
        "val_heldout_crossing_task_ids": sorted(val & heldout),
        "crossing_task_id_count": len((train & val) | (train & heldout) | (val & heldout)),
        "counts": {"train": len(train), "val": len(val), "heldout": len(heldout)},
    }


def tap_training_task_ids() -> tuple[set[str], dict[str, Any]]:
    """Recover the IDs actually consumed by the layer-native DualAnchor trainer."""
    by_split: dict[str, set[str]] = defaultdict(set)
    datasets = dual_train.old_domain_pair_datasets() + dual_train.branch_pair_datasets()
    for dataset in datasets:
        for pair in dataset.get("pairs") or []:
            task_id = str(pair.get("task_id") or "")
            if task_id:
                by_split[str(pair.get("split") or "all")].add(task_id)
    report = crossing_report(by_split)
    report["all_split_counts"] = {key: len(value) for key, value in sorted(by_split.items())}
    report["dataset_count"] = len(datasets)
    # Validation affected model/epoch selection, so it is training-side for a clean eval.
    return set(by_split["train"]) | set(by_split["val"]), report


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tier1_audit() -> dict[str, Any]:
    started = time.time()
    train_side, source_split_audit = tap_training_task_ids()

    dual = torch.load(DUAL_ROOT / "architecture_looped_rows.pt", map_location="cpu", weights_only=False)
    all_task_rows = list(dual.get("task_rows") or [])
    all_stage_rows = list(dual.get("stage_rows") or [])
    all_terminal_rows = list(dual.get("terminal_rows") or [])
    all_branch_rows = list(dual.get("rows") or [])
    original_eval_ids = {str(row.get("task_id")) for row in all_task_rows}
    original_cross = sorted(original_eval_ids & train_side)

    guard = read_json(V3_GUARD)
    predeclared_heldout = set(guard["v3_heldout_candidate_task_ids"])
    captured_predeclared_ids = predeclared_heldout & original_eval_ids
    # The v3 guard covered hidden-origin training, but DualAnchor also includes
    # older content anchors.  Enforce disjointness against the union of every
    # recovered train/validation source used by the composed tap family.
    excluded_anchor_seen = sorted(captured_predeclared_ids & train_side)
    captured_clean_ids = sorted(captured_predeclared_ids - train_side)
    clean_cross = sorted(set(captured_clean_ids) & train_side)
    clean_task_rows = [row for row in all_task_rows if str(row.get("task_id")) in captured_clean_ids]
    clean_stage_rows = [row for row in all_stage_rows if str(row.get("task_id")) in captured_clean_ids]
    clean_terminal_rows = [row for row in all_terminal_rows if str(row.get("task_id")) in captured_clean_ids]

    stage_retention = finite_mean(row.get("stage_oracle_retained") for row in clean_stage_rows)
    terminal_retained = finite_mean(row.get("terminal_oracle_retained") for row in clean_task_rows)
    forced_top1 = finite_mean(row.get("terminal_forced_top1_oracle") for row in clean_task_rows)
    forced_reward = finite_mean(row.get("terminal_forced_top1_reward") for row in clean_task_rows)
    best_reward = finite_mean(row.get("terminal_best_reward") for row in clean_task_rows)
    diverse = [row for row in clean_task_rows if float(row.get("terminal_reward_diverse") or 0) > 0]

    terminal_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_terminal_rows:
        terminal_by_policy[str(row.get("policy"))].append(row)
    terminal_policy = {
        policy: {
            "n": len(rows),
            "oracle_retained": finite_mean(r.get("oracle_retained") for r in rows),
            "best_selected_reward": finite_mean(r.get("best_selected_reward") for r in rows),
            "first_selected_oracle": finite_mean(r.get("first_selected_oracle") for r in rows),
            "first_selected_reward": finite_mean(r.get("first_selected_reward") for r in rows),
        }
        for policy, rows in sorted(terminal_by_policy.items())
    }

    # Pre-DualAnchor scaffold: deterministically keep only IDs absent from every
    # train/validation artifact consumed by the later canonical tap family.
    scaffold_payload = read_json(SCAFFOLD_ROOT / "selection_only_policy_outputs.json")
    scaffold_rows = list(scaffold_payload.get("group_rows") or [])
    scaffold_original_ids = {str(row.get("task_id")) for row in scaffold_rows}
    scaffold_clean_ids = scaffold_original_ids - train_side
    scaffold_clean_rows = [row for row in scaffold_rows if str(row.get("task_id")) in scaffold_clean_ids]
    scaffold_metrics = scaffold.add_task_macro(scaffold_clean_rows)
    scaffold_selected = scaffold_metrics.get("fixed_composite_conservative_top4", {})

    fixed_rows = read_csv_rows(FIXED_ROOT / "heldout_survival_rows.csv")
    fixed_clean = [
        row for row in fixed_rows
        if row.get("policy") == "fixed_composite_conservative_top4"
        and str(row.get("task_id")) in scaffold_clean_ids
    ]

    # A true-survivor integrated evaluation is added after the fresh CoreContent
    # head is trained in tier 2. Preserve enough inputs here to identify the pool.
    result = {
        "TIER1_TASK_DISJOINT_AUDIT_VERDICT": "CONTAMINATED_REPLACED_WITH_CLEAN_SUBSET" if not clean_cross else "BLOCKED",
        "historical_dualanchor_source_split_audit": source_split_audit,
        "historical_dualanchor_eval": {
            "task_count": len(original_eval_ids),
            "training_side_crossing_task_ids": original_cross,
            "training_side_crossing_task_id_count": len(original_cross),
            "mixed_split_counts": (dual.get("summary") or {}).get("task_inventory", {}).get("split_counts"),
        },
        "clean_dualanchor_eval": {
            "split_definition": "predeclared v3 heldout IDs from split_guard_v3.json",
            "task_ids": captured_clean_ids,
            "task_count": len(captured_clean_ids),
            "predeclared_heldout_total": len(predeclared_heldout),
            "captured_predeclared_task_count": len(captured_predeclared_ids),
            "excluded_because_an_anchor_saw_task_ids": excluded_anchor_seen,
            "captured_fraction": len(captured_clean_ids) / max(1, len(predeclared_heldout)),
            "crossing_task_ids": clean_cross,
            "crossing_task_id_count": len(clean_cross),
            "stage_decisions": len(clean_stage_rows),
            "stage_oracle_retention": stage_retention,
            "terminal_oracle_retained": terminal_retained,
            "terminal_forced_top1_oracle": forced_top1,
            "terminal_forced_top1_reward": forced_reward,
            "terminal_best_reward": best_reward,
            "reward_diverse_task_count": len(diverse),
            "reward_diverse_forced_top1_oracle": finite_mean(r.get("terminal_forced_top1_oracle") for r in diverse),
            "reward_diverse_forced_top1_reward": finite_mean(r.get("terminal_forced_top1_reward") for r in diverse),
            "reward_diverse_best_reward": finite_mean(r.get("terminal_best_reward") for r in diverse),
            "terminal_policy_summary": terminal_policy,
        },
        "clean_scaffold_eval": {
            "split_definition": "historical heldout suite minus all recovered train/validation task IDs",
            "original_task_count": len(scaffold_original_ids),
            "original_crossing_task_ids": sorted(scaffold_original_ids & train_side),
            "original_crossing_task_id_count": len(scaffold_original_ids & train_side),
            "task_ids": sorted(scaffold_clean_ids),
            "task_count": len(scaffold_clean_ids),
            "crossing_task_id_count": len(scaffold_clean_ids & train_side),
            "selection_only_top4": scaffold_selected,
            "fixed_composite_top4": {
                "groups": len(fixed_clean),
                "oracle_retention": finite_mean(r.get("oracle_retention") for r in fixed_clean),
                "false_prune_rate": finite_mean(r.get("false_prune_rate") for r in fixed_clean),
                "average_survivors": finite_mean(r.get("average_survivors") for r in fixed_clean),
            },
        },
        "notes": [
            "The 0.9848 historical aggregate mixed train, validation, and heldout tasks.",
            "The canonical DualAnchor pair sources contain eight train-to-heldout task-ID crossings.",
            "The L47 tree ablation is not reinterpreted by this split audit.",
            "The clean DualAnchor subset is cached deterministic branch generation, re-aggregated on a split declared before the v3 run.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    # Non-serializable branch rows are retained only for the in-process integrated eval.
    result["_runtime"] = {"dual": dual, "branch_rows": all_branch_rows, "clean_task_rows": clean_task_rows}
    return result


def compact_feature_groups() -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Load one domain at a time and retain only L4 vectors needed by this rerun."""
    by_domain: dict[str, dict[str, list[dict[str, Any]]]] = {}
    split_ids: dict[str, set[str]] = defaultdict(set)
    historical: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Any] = {}
    for domain in DOMAINS:
        print(f"  loading compact {domain} features", flush=True)
        raw = features.load_feature_groups(domains=[domain], splits=None)
        part: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in raw:
            task_id = str(group.get("task_id") or group.get("group_id"))
            split = deterministic_split(task_id)
            split_ids[split].add(task_id)
            historical[str(group.get("split"))].add(task_id)
            candidates = []
            for cand in group.get("candidates") or []:
                tensor = cand.get("features")
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape[-3:]) != (3, 4, 2048):
                    continue
                candidates.append({
                    "x": tensor[:, 3, :].detach().cpu().to(torch.float16).contiguous(),
                    "reward": float(cand.get("reward", 0.0)),
                })
            if len(candidates) < 2 or len({c["reward"] for c in candidates}) < 2:
                continue
            part[split].append({
                "task_id": task_id,
                "group_id": str(group.get("group_id")),
                "domain": domain,
                "candidates": candidates,
            })
        by_domain[domain] = dict(part)
        counts[domain] = {key: len(part.get(key, [])) for key in ("train", "val", "heldout")}
        del raw
        gc.collect()
    return by_domain, {
        "split_salt": SPLIT_SALT,
        "algorithm": "sha256(salt::task_id) modulo 10000; 70/10/20",
        "task_split_audit": crossing_report(split_ids),
        "historical_task_split_audit": crossing_report(historical),
        "group_counts": counts,
    }


def pair_tensor(groups: Sequence[dict[str, Any]], cap: int, salt: str) -> tuple[torch.Tensor, list[str]]:
    rows: list[tuple[int, str, torch.Tensor]] = []
    for group in groups:
        cands = group["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if cands[i]["reward"] <= cands[j]["reward"]:
                    continue
                key = stable_int(salt, group["task_id"], group["group_id"], i, j)
                rows.append((key, group["task_id"], cands[i]["x"] - cands[j]["x"]))
    rows.sort(key=lambda item: item[0])
    rows = rows[:cap]
    if not rows:
        return torch.empty(0, 3, 2048, dtype=torch.float16), []
    return torch.stack([row[2] for row in rows]), [row[1] for row in rows]


def train_weight(x: torch.Tensor, config_idx: int, epochs: int, device: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    dev = torch.device(device)
    data = x[:, config_idx, :].to(dev, dtype=torch.float32)
    # Per-example scaling prevents a few large-norm domains from dominating while
    # preserving the antisymmetric direction of every pair.
    data = data / data.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    w = torch.zeros(data.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=1e-3, weight_decay=0.01)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.softplus(-(data @ w)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
    out = w.detach().cpu()
    del data, w, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def zscores(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    sd = values.std(unbiased=False)
    if not math.isfinite(float(sd)) or float(sd) < 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / sd


def policy_scores(group: dict[str, Any], weights: dict[str, torch.Tensor], policy: str) -> torch.Tensor:
    feats = torch.stack([c["x"] for c in group["candidates"]]).float()
    if policy == "blockwise":
        channels = []
        for config in CONFIGS:
            idx = LAYER_POS[config]
            channels.append(zscores(feats[:, idx, :] @ weights[config]))
        return torch.stack(channels).mean(0)
    idx = LAYER_POS[policy]
    return feats[:, idx, :] @ weights[policy]


def eval_groups(weights: dict[str, torch.Tensor], policy: str, groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        scores = policy_scores(group, weights, policy)
        rewards = [float(c["reward"]) for c in group["candidates"]]
        pick = int(torch.argmax(scores).item())
        oracle = max(rewards)
        pair_correct = 0.0
        pair_total = 0
        for i in range(len(rewards)):
            for j in range(i + 1, len(rewards)):
                if rewards[i] == rewards[j]:
                    continue
                hi, lo = (i, j) if rewards[i] > rewards[j] else (j, i)
                pair_total += 1
                pair_correct += 1.0 if scores[hi] > scores[lo] else (0.5 if scores[hi] == scores[lo] else 0.0)
        rows.append({
            "task_id": group["task_id"],
            "group_id": group["group_id"],
            "domain": group["domain"],
            "top1": 1.0 if rewards[pick] >= oracle else 0.0,
            "selected_reward": rewards[pick],
            "oracle_reward": oracle,
            "pair_correct": pair_correct,
            "pair_total": pair_total,
        })
    return rows


def summarize_eval(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "groups": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "top1": finite_mean(row["top1"] for row in rows),
        "pairwise": sum(float(row["pair_correct"]) for row in rows) / max(1, sum(int(row["pair_total"]) for row in rows)),
        "pairs": sum(int(row["pair_total"]) for row in rows),
        "selected_reward": finite_mean(row["selected_reward"] for row in rows),
        "oracle_reward": finite_mean(row["oracle_reward"] for row in rows),
    }


def clustered_ci(rows: Sequence[dict[str, Any]], key: str, draws: int, salt: str) -> list[float]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    if len(tasks) < 2:
        return [float("nan"), float("nan")]
    rng = random.Random(stable_int(SEED, salt))
    vals = []
    for _ in range(draws):
        sample = [rng.choice(tasks) for _ in tasks]
        selected = [row for task in sample for row in by_task[task]]
        if key == "pairwise":
            denom = sum(int(row["pair_total"]) for row in selected)
            value = sum(float(row["pair_correct"]) for row in selected) / max(1, denom)
        else:
            value = finite_mean(row[key] for row in selected)
        vals.append(value)
    vals.sort()
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def select_policy(weights: dict[str, torch.Tensor], val_by_domain: dict[str, list[dict[str, Any]]], target: Sequence[str]) -> tuple[str, dict[str, float]]:
    scores = {}
    for policy in (*CONFIGS, "blockwise"):
        per_domain = []
        for domain in target:
            rows = eval_groups(weights, policy, val_by_domain.get(domain, []))
            if rows:
                per_domain.append(summarize_eval(rows)["top1"])
        scores[policy] = finite_mean(per_domain)
    best = max(scores, key=lambda key: (scores[key], key == "blockwise", key))
    return best, scores


def train_models(data: dict[str, dict[str, list[dict[str, Any]]]], epochs: int, device: str) -> dict[str, Any]:
    train_pairs: dict[str, torch.Tensor] = {}
    pair_task_ids: dict[str, list[str]] = {}
    for domain in DOMAINS:
        tensor, tids = pair_tensor(data[domain].get("train", []), 20_000, f"pairs::{domain}")
        train_pairs[domain] = tensor
        pair_task_ids[domain] = tids
        print(f"  {domain}: {tensor.shape[0]} train pairs", flush=True)

    hh_tasks = sorted({t for t in pair_task_ids["alignment"]}, key=lambda t: stable_int("random20", t))[:20]
    hh20_mask = torch.tensor([task in set(hh_tasks) for task in pair_task_ids["alignment"]], dtype=torch.bool)
    sources = {
        "code_trained": (train_pairs["coding"], ("coding",)),
        "hh_trained": (train_pairs["alignment"], ("alignment",)),
        "reasoning_trained": (train_pairs["reasoning"], ("reasoning",)),
        "hh_random20": (train_pairs["alignment"][hh20_mask], ("alignment",)),
    }
    balanced_n = min(8_000, *(int(train_pairs[d].shape[0]) for d in DOMAINS))
    all_balanced = torch.cat([train_pairs[d][:balanced_n] for d in DOMAINS], dim=0)
    sources["all_core_balanced"] = (all_balanced, DOMAINS)

    val_by_domain = {domain: data[domain].get("val", []) for domain in DOMAINS}
    models: dict[str, Any] = {}
    for model_name, (tensor, target) in sources.items():
        print(f"  training {model_name}: {tensor.shape[0]} pairs", flush=True)
        weights = {}
        for config in CONFIGS:
            weights[config] = train_weight(tensor, LAYER_POS[config], epochs, device, SEED + stable_int(model_name, config) % 100_000)
        selected, val_scores = select_policy(weights, val_by_domain, target)
        models[model_name] = {
            "weights": weights,
            "selected_policy": selected,
            "val_policy_top1": val_scores,
            "training_pairs": int(tensor.shape[0]),
            "training_tasks": hh_tasks if model_name == "hh_random20" else None,
        }
        print(f"    selected={selected} val={val_scores[selected]:.4f}", flush=True)
    return models


def domain_transfer_eval(data: dict[str, dict[str, list[dict[str, Any]]]], models: dict[str, Any], draws: int) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    raw: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model_name, model in models.items():
        matrix[model_name] = {}
        for domain in DOMAINS:
            rows = eval_groups(model["weights"], model["selected_policy"], data[domain].get("heldout", []))
            raw[(model_name, domain)] = rows
            summary = summarize_eval(rows)
            summary["top1_ci95_task_clustered"] = clustered_ci(rows, "top1", draws, f"{model_name}:{domain}:top1")
            summary["pairwise_ci95_task_clustered"] = clustered_ci(rows, "pairwise", draws, f"{model_name}:{domain}:pairwise")
            matrix[model_name][domain] = summary

    comparisons = {}
    for name, a, b, domain in (
        ("code_specialist_vs_hh_on_code", "code_trained", "hh_trained", "coding"),
        ("hh_specialist_vs_code_on_hh", "hh_trained", "code_trained", "alignment"),
        ("reasoning_specialist_vs_all_on_reasoning", "reasoning_trained", "all_core_balanced", "reasoning"),
        ("hh_random20_vs_full_hh_on_hh", "hh_random20", "hh_trained", "alignment"),
    ):
        comparisons[name] = {
            "domain": domain,
            "a": a,
            "b": b,
            "top1_delta": matrix[a][domain]["top1"] - matrix[b][domain]["top1"],
            "pairwise_delta": matrix[a][domain]["pairwise"] - matrix[b][domain]["pairwise"],
        }
    return {"matrix": matrix, "comparisons": comparisons}


def integrated_true_survivors(tier1: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    runtime = tier1["_runtime"]
    dual = runtime["dual"]
    clean_task_rows = runtime["clean_task_rows"]
    by_branch = {str(row.get("branch_id")): row for row in runtime["branch_rows"]}
    terminal_lookup = {
        (str(row.get("task_id")), str(row.get("policy"))): row
        for row in (dual.get("terminal_rows") or [])
    }
    records = []
    for task in clean_task_rows:
        pool = [by_branch[str(branch_id)] for branch_id in task.get("final_branch_ids") or [] if str(branch_id) in by_branch]
        if len(pool) < 2:
            continue
        task_id = str(task.get("task_id"))
        top5_row = terminal_lookup.get((task_id, "dualanchor_terminal_top5"), {})
        top1_row = terminal_lookup.get((task_id, "dualanchor_forced_top1"), {})
        survivor_idx = [int(i) for i in top5_row.get("selected_indices") or [] if 0 <= int(i) < len(pool)]
        forced_indices = [int(i) for i in top1_row.get("selected_indices") or [] if 0 <= int(i) < len(pool)]
        if not survivor_idx or not forced_indices:
            continue
        rewards = [float(row.get("reward", row.get("deterministic_reward", 0.0))) for row in pool]
        oracle = max(rewards)
        compact = {
            "task_id": str(task.get("task_id")),
            "group_id": str(task.get("task_id")),
            "domain": str(task.get("domain")),
            "candidates": [
                {"x": pool[i]["features"][:, 3, :].detach().cpu().to(torch.float16), "reward": rewards[i]}
                for i in survivor_idx
            ],
        }
        core_scores = policy_scores(compact, model["weights"], model["selected_policy"])
        core_pick_local = int(torch.argmax(core_scores).item())
        core_pick = survivor_idx[core_pick_local]
        forced_pick = forced_indices[0]
        pair_correct = pair_total = 0
        local_rewards = [rewards[i] for i in survivor_idx]
        for i in range(len(local_rewards)):
            for j in range(i + 1, len(local_rewards)):
                if local_rewards[i] == local_rewards[j]:
                    continue
                hi, lo = (i, j) if local_rewards[i] > local_rewards[j] else (j, i)
                pair_total += 1
                pair_correct += 1 if core_scores[hi] > core_scores[lo] else (0.5 if core_scores[hi] == core_scores[lo] else 0)
        records.append({
            "task_id": task.get("task_id"),
            "domain": task.get("domain"),
            "survivor_has_oracle": 1.0 if any(rewards[i] >= oracle for i in survivor_idx) else 0.0,
            "corecontent_top1_oracle": 1.0 if rewards[core_pick] >= oracle else 0.0,
            "dualanchor_forced_top1_oracle": 1.0 if rewards[forced_pick] >= oracle else 0.0,
            "corecontent_selected_reward": rewards[core_pick],
            "dualanchor_selected_reward": rewards[forced_pick],
            "oracle_reward": oracle,
            "pair_correct": pair_correct,
            "pair_total": pair_total,
            "reward_diverse": len(set(local_rewards)) > 1,
        })
    diverse = [row for row in records if row["reward_diverse"]]
    return {
        "interpretation": "actual DualAnchor v3 terminal pools; top-5 selected by DualAnchor; fresh all-domain pairwise head ranks within top-5",
        "task_count": len(records),
        "task_ids": [str(row["task_id"]) for row in records],
        "crossing_task_id_count": tier1["clean_dualanchor_eval"]["crossing_task_id_count"],
        "survivor_oracle_retention": finite_mean(row["survivor_has_oracle"] for row in records),
        "corecontent_top1_oracle": finite_mean(row["corecontent_top1_oracle"] for row in records),
        "corecontent_pairwise": sum(row["pair_correct"] for row in records) / max(1, sum(row["pair_total"] for row in records)),
        "corecontent_pairwise_pairs": sum(row["pair_total"] for row in records),
        "dualanchor_forced_top1_oracle": finite_mean(row["dualanchor_forced_top1_oracle"] for row in records),
        "corecontent_selected_reward": finite_mean(row["corecontent_selected_reward"] for row in records),
        "dualanchor_selected_reward": finite_mean(row["dualanchor_selected_reward"] for row in records),
        "oracle_reward": finite_mean(row["oracle_reward"] for row in records),
        "reward_diverse_task_count": len(diverse),
        "reward_diverse_corecontent_top1": finite_mean(row["corecontent_top1_oracle"] for row in diverse),
        "reward_diverse_dualanchor_top1": finite_mean(row["dualanchor_forced_top1_oracle"] for row in diverse),
        "rows": records,
    }


def report_markdown(payload: dict[str, Any]) -> list[str]:
    lines = ["# Paper v4.4 Task-Disjoint Rerun", ""]
    if "tier1" in payload:
        t = payload["tier1"]
        d = t["clean_dualanchor_eval"]
        s = t["clean_scaffold_eval"]
        lines += [
            "## Tier 1 — branch survival and selection", "",
            f"Verdict: **{t['TIER1_TASK_DISJOINT_AUDIT_VERDICT']}**.", "",
            f"Canonical train-to-heldout split crossings: **{len(t['historical_split_audit']['train_heldout_crossing_task_ids'])}**. "
            f"The historical mixed-role 48-task aggregate contains **{t['historical_dualanchor_eval']['training_side_crossing_task_id_count']}** training-side IDs. "
            f"Clean rerun crossing IDs: **{d['crossing_task_id_count']}**.", "",
            "| Evaluation | N | Retention | Forced/final | Best available |",
            "|---|---:|---:|---:|---:|",
            f"| DualAnchor stages | {d['stage_decisions']} decisions / {d['task_count']} tasks | {d['stage_oracle_retention']:.4f} | — | — |",
            f"| DualAnchor terminal | {d['task_count']} | {d['terminal_oracle_retained']:.4f} | {d['terminal_forced_top1_reward']:.4f} reward | {d['terminal_best_reward']:.4f} reward |",
            f"| Scaffold top-4 | {s['selection_only_top4'].get('n', 0)} groups / {s['task_count']} tasks | {s['selection_only_top4'].get('oracle_retention', float('nan')):.4f} | {s['selection_only_top4'].get('task_macro_reward', float('nan')):.4f} reward | {s['selection_only_top4'].get('task_macro_best_selected_reward', float('nan')):.4f} reward |",
            "",
        ]
        if t.get("integrated_true_survivors"):
            i = t["integrated_true_survivors"]
            lines += [
                "### Integrated check on actual survivor pools", "",
                f"N={i['task_count']}, crossing IDs={i['crossing_task_id_count']}, top-5 retention={i['survivor_oracle_retention']:.4f}, "
                f"CoreContent top-1={i['corecontent_top1_oracle']:.4f}, CoreContent pairwise={i['corecontent_pairwise']:.4f} "
                f"({i['corecontent_pairwise_pairs']} unequal-reward pairs), DualAnchor forced top-1={i['dualanchor_forced_top1_oracle']:.4f}.", "",
            ]
    if "tier2" in payload:
        t = payload["tier2"]
        lines += [
            "## Tier 2 — powered domain transfer", "",
            f"Deterministic crossing task IDs: **{t['split_audit']['task_split_audit']['crossing_task_id_count']}**.", "",
            "| Train head | Eval domain | N groups | Top-1 [95% CI] | Pairwise [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
        for model_name, domains in t["evaluation"]["matrix"].items():
            for domain, row in domains.items():
                tci = row["top1_ci95_task_clustered"]
                pci = row["pairwise_ci95_task_clustered"]
                lines.append(
                    f"| {model_name} | {domain} | {row['groups']} | {row['top1']:.4f} [{tci[0]:.4f}, {tci[1]:.4f}] | "
                    f"{row['pairwise']:.4f} [{pci[0]:.4f}, {pci[1]:.4f}] |"
                )
        lines.append("")
    lines += ["## Reproducibility", "", f"Split salt: `{SPLIT_SALT}`. Seed: `{SEED}`."]
    return lines


def main() -> int:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    final_path = OUT_ROOT / "task_disjoint_rerun.json"
    payload: dict[str, Any] = {
        "run_date": "2026-07-13",
        "split_salt": SPLIT_SALT,
        "seed": SEED,
        "args": vars(args),
    }
    tier1 = None
    if args.tier in ("all", "tier1"):
        print("[tier1] auditing and re-aggregating branch artifacts", flush=True)
        tier1 = tier1_audit()
        payload["tier1"] = {key: value for key, value in tier1.items() if key != "_runtime"}
        write_json(OUT_ROOT / "tier1_branch_survival_selection.json", payload["tier1"])

    if args.tier in ("all", "tier2"):
        print("[tier2] constructing deterministic task-disjoint feature splits", flush=True)
        data, split_audit = compact_feature_groups()
        if split_audit["task_split_audit"]["crossing_task_id_count"] != 0:
            raise RuntimeError("deterministic split guard failed")
        models = train_models(data, args.epochs, args.device)
        evaluation = domain_transfer_eval(data, models, args.bootstrap)
        tier2 = {
            "TIER2_DOMAIN_TRANSFER_VERDICT": "POWERED_TASK_DISJOINT_COMPLETE",
            "split_audit": split_audit,
            "models": {
                name: {key: value for key, value in model.items() if key != "weights"}
                for name, model in models.items()
            },
            "evaluation": evaluation,
        }
        payload["tier2"] = tier2
        write_json(OUT_ROOT / "tier2_domain_transfer.json", tier2)
        torch.save({
            "split_salt": SPLIT_SALT,
            "seed": SEED,
            "models": models,
        }, OUT_ROOT / "domain_transfer_linear_heads.pt")

        if tier1 is None and (OUT_ROOT / "tier1_branch_survival_selection.json").exists():
            # Integrated evaluation requires runtime tensors, so reconstruct tier1.
            tier1 = tier1_audit()
            payload.setdefault("tier1", {key: value for key, value in tier1.items() if key != "_runtime"})
        if tier1 is not None:
            integrated = integrated_true_survivors(tier1, models["all_core_balanced"])
            payload["tier1"]["integrated_true_survivors"] = integrated
            write_json(OUT_ROOT / "tier1_branch_survival_selection.json", payload["tier1"])
            write_json(OUT_ROOT / "integrated_true_survivors.json", integrated)

    write_json(final_path, payload)
    write_md(OUT_ROOT / "task_disjoint_rerun.md", report_markdown(payload))
    print(f"PAPER_V44_TASK_DISJOINT_RERUN = COMPLETE\nreport = {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
