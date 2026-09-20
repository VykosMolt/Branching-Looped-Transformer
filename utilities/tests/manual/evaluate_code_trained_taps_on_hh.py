"""Evaluate code-trained tiny taps on HH-RLHF preference pairs.

This inverse transfer probe retrains code-specific heads from saved code
features when head weights are not reusable, then compares them to HH-trained
tiny heads on the HH RLTT capture. It does not generate candidates or capture
new features.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()
sys.path.insert(0, str(THIS_DIR))

try:
    from utilities.tests.manual.code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json
except ModuleNotFoundError:
    from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json

from evaluate_hh_transfer_on_clean_gsm8k_extreme import (  # noqa: E402
    build_hh_features,
    train_hh_head,
)
from math_bg_probe_lib import MATH_CONFIGS  # noqa: E402
from train_code_specific_tiny_heads_and_eval import (  # noqa: E402
    best_by_arch,
    feature_map,
    split_pairs_by_task,
    train_head,
)


HH_CAPTURE = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
EXPANDED_FEATURES_PT = REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.pt"
EXPANDED_EVAL_SET_JSON = REPORT_DIR / "code_expanded_strict_clean_eval_set_2026-05-17.json"
EXPANDED_HEADS_JSON = REPORT_DIR / "code_specific_heads_expanded_strict_clean_2026-05-17.json"
EXPANDED_COMPARISON_JSON = REPORT_DIR / "expanded_strict_clean_code_projection_comparison_2026-05-17.json"

OUTPUT_JSON = REPORT_DIR / "code_trained_taps_on_hh_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_trained_taps_on_hh_2026-05-17.md"

ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hh-capture", default=str(HH_CAPTURE))
    parser.add_argument("--code-features", default=str(EXPANDED_FEATURES_PT))
    parser.add_argument("--code-eval-set", default=str(EXPANDED_EVAL_SET_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hh-train-pairs", type=int, default=180)
    parser.add_argument("--random-splits", type=int, default=0)
    parser.add_argument("--random-split-size", type=int, default=20)
    parser.add_argument("--random-split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def architecture_label(architecture: str) -> str:
    return "NoNorm" if architecture == "AntisymLinearNoNorm" else "AntisymLinear"


def row_compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {
        "head_family": row.get("head_family", ""),
        "config": row["config"],
        "architecture": row["architecture"],
        "family_architecture": row.get("family_architecture", ""),
        "accuracy": m["canonical_accuracy"],
        "centered_accuracy": m["centered_accuracy"],
        "score_mean": m["score_mean"],
        "score_std": m["score_std"],
        "antisym_mean_abs": m["antisymmetry_mean_abs"],
    }


def best_overall(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["canonical_accuracy"],
            row["metrics"]["centered_accuracy"],
            -row["metrics"]["antisymmetry_mean_abs"],
        ),
    )


def best_by_family(rows: Sequence[dict[str, Any]], family: str) -> dict[str, Any] | None:
    return best_overall([row for row in rows if row.get("head_family") == family])


def best_by_family_arch(rows: Sequence[dict[str, Any]], family_arch: str) -> dict[str, Any] | None:
    return best_overall([row for row in rows if row.get("family_architecture") == family_arch])


def tensor_stats(values: torch.Tensor) -> tuple[float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan")
    vals = [float(x) for x in values.detach().cpu().flatten().tolist()]
    return float(mean(vals)), float(pstdev(vals)) if len(vals) > 1 else 0.0


@torch.no_grad()
def evaluate_pair_split(
    *,
    head: torch.nn.Module,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    idx = list(indices)
    left = chosen[idx].to(device=device, dtype=torch.float32)
    right = rejected[idx].to(device=device, dtype=torch.float32)
    head = head.to(device)
    canonical = head(left, right).detach().cpu()
    reverse = head(right, left).detach().cpu()
    head = head.to("cpu")
    centered = canonical - reverse
    antisym_sum = canonical + reverse
    score_mean, score_std = tensor_stats(canonical)
    reverse_mean, reverse_std = tensor_stats(reverse)
    centered_mean, centered_std = tensor_stats(centered)
    return {
        "n_pairs": len(idx),
        "canonical_accuracy": float((canonical > 0).to(torch.float32).mean().item()) if idx else float("nan"),
        "flipped_accuracy": float((reverse < 0).to(torch.float32).mean().item()) if idx else float("nan"),
        "centered_accuracy": float((centered > 0).to(torch.float32).mean().item()) if idx else float("nan"),
        "pairwise_accuracy": float((centered > 0).to(torch.float32).mean().item()) if idx else float("nan"),
        "score_mean": score_mean,
        "score_std": score_std,
        "reverse_score_mean": reverse_mean,
        "reverse_score_std": reverse_std,
        "centered_score_mean": centered_mean,
        "centered_score_std": centered_std,
        "antisymmetry_mean_abs": float(antisym_sum.abs().mean().item()) if idx else float("nan"),
        "antisymmetry_max_abs": float(antisym_sum.abs().max().item()) if idx else float("nan"),
        "antisymmetry_sum_std": float(antisym_sum.std(unbiased=False).item()) if idx else float("nan"),
    }


def evaluate_heads_on_hh(
    *,
    heads: Sequence[dict[str, Any]],
    hh_payload: dict[str, Any],
    split_name: str,
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for head_info in heads:
        config = str(head_info["config"])
        if config not in feature_cache:
            feature_cache[config] = build_hh_features(hh_payload, config)
        chosen, rejected = feature_cache[config]
        metrics = evaluate_pair_split(
            head=head_info["head"],
            chosen=chosen,
            rejected=rejected,
            indices=indices,
            device=device,
        )
        rows.append({
            "eval_split": split_name,
            "head_family": head_info["head_family"],
            "config": config,
            "architecture": head_info["architecture"],
            "family_architecture": f"{head_info['head_family']}_{architecture_label(head_info['architecture'])}",
            "train_metrics": head_info["train_metrics"],
            "metrics": metrics,
        })
    return {
        "eval_split": split_name,
        "n_pairs": len(indices),
        "table": rows,
        "best_hh_trained": best_by_family(rows, "HH"),
        "best_code_trained": best_by_family(rows, "CODE"),
        "best_antisymlinear": best_overall([row for row in rows if row["architecture"] == "AntisymLinear"]),
        "best_nonorm": best_overall([row for row in rows if row["architecture"] == "AntisymLinearNoNorm"]),
        "best_HH_AntisymLinear": best_by_family_arch(rows, "HH_AntisymLinear"),
        "best_HH_NoNorm": best_by_family_arch(rows, "HH_NoNorm"),
        "best_CODE_AntisymLinear": best_by_family_arch(rows, "CODE_AntisymLinear"),
        "best_CODE_NoNorm": best_by_family_arch(rows, "CODE_NoNorm"),
    }


def train_hh_heads(
    *,
    hh_payload: dict[str, Any],
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    heads: list[dict[str, Any]] = []
    for config in MATH_CONFIGS:
        print(f"training HH baseline heads {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        for architecture in ARCHITECTURES:
            head, train_metrics = train_hh_head(
                architecture,
                config,
                chosen,
                rejected,
                train_idx,
                eval_idx,
                args,
                device,
            )
            heads.append({
                "head_family": "HH",
                "config": config,
                "architecture": architecture,
                "head": head,
                "train_metrics": train_metrics,
            })
    return heads


def train_code_heads(
    *,
    feature_payload: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pooled_by_uid = feature_map(feature_payload)
    raw_pairs = [
        pair for pair in feature_payload.get("training_pairs_primary", []) or []
        if pair.get("preferred_uid") in pooled_by_uid and pair.get("rejected_uid") in pooled_by_uid
    ]
    all_pairs = list(feature_payload.get("training_pairs_primary", []) or [])
    missing_pair_count = len(all_pairs) - len(raw_pairs)
    task_ids = sorted({str(pair["task_id"]) for pair in raw_pairs})
    if len(task_ids) < 8 or len(raw_pairs) < 30:
        raise SystemExit(
            "CODE_TO_HH_TRANSFER_VERDICT=NOT_RUN: insufficient code train pairs "
            f"tasks={len(task_ids)} pairs={len(raw_pairs)}"
        )
    train_pairs, val_pairs, validation = split_pairs_by_task(raw_pairs, int(args.seed))
    heads: list[dict[str, Any]] = []
    for config in MATH_CONFIGS:
        print(f"training code-specific heads {config}", flush=True)
        for architecture in ARCHITECTURES:
            head, train_metrics = train_head(
                architecture=architecture,
                config=config,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                pooled_by_uid=pooled_by_uid,
                args=args,
                device=device,
            )
            heads.append({
                "head_family": "CODE",
                "config": config,
                "architecture": architecture,
                "head": head,
                "train_metrics": train_metrics,
            })
    metadata = {
        "source": "retrained_from_expanded_strict_clean_code_features",
        "training_task_count": len(task_ids),
        "primary_training_pair_count": len(raw_pairs),
        "source_pair_count": len(all_pairs),
        "missing_feature_pair_count": missing_pair_count,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "validation": validation,
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": min(int(args.batch_size), len(train_pairs)),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "swap_augmentation": False,
        "lambda_sym": 0.0,
    }
    return heads, metadata


def verdict_for_hh_baseline(best_hh: dict[str, Any] | None) -> str:
    if not best_hh:
        return "NOT_RUN"
    acc = float(best_hh["metrics"]["canonical_accuracy"])
    if acc >= 0.60:
        return "GOOD"
    if acc >= 0.55:
        return "WEAK"
    return "POOR"


def verdict_for_code_transfer(best_code: dict[str, Any] | None, best_hh: dict[str, Any] | None) -> str:
    if not best_code or not best_hh:
        return "NOT_RUN"
    code_acc = float(best_code["metrics"]["canonical_accuracy"])
    hh_acc = float(best_hh["metrics"]["canonical_accuracy"])
    if code_acc >= hh_acc - 0.05 and code_acc >= 0.60:
        return "GOOD"
    if code_acc >= 0.55 or code_acc >= hh_acc - 0.10:
        return "WEAK"
    return "POOR"


def recommended_next(verdict: str) -> str:
    if verdict == "POOR":
        return "keep_domain_specific_code_heads_as_specialists"
    if verdict == "WEAK":
        return "keep_HH_general_head_plus_code_specialist_head"
    if verdict == "GOOD":
        return "investigate_shared_objective_coherence_axis"
    return "fix_artifact_or_feature_blocker"


def interpretation(verdict: str) -> str:
    if verdict == "POOR":
        return "Code-specific projection appears domain-specialized; use it as a code specialist rather than a general HH head."
    if verdict == "WEAK":
        return "Code-specific projection retains some HH preference-pair signal, but it should not replace the HH/general head."
    if verdict == "GOOD":
        return "Code-specific projection may have learned a broader relational coherence axis; this needs a follow-up beyond this inverse transfer probe."
    return "The inverse transfer probe did not run."


def located_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    candidates = [
        EXPANDED_HEADS_JSON,
        REPORT_DIR / "code_specific_tiny_head_control_2026-05-17.json",
        EXPANDED_COMPARISON_JSON,
    ]
    return {
        "code_head_artifact_candidates": [
            {"path": repo_path(path), "exists": path.exists(), "contains_reusable_weights": False}
            for path in candidates
        ],
        "code_features": {"path": repo_path(output_path(args.code_features)), "exists": output_path(args.code_features).exists()},
        "code_eval_set": {"path": repo_path(output_path(args.code_eval_set)), "exists": output_path(args.code_eval_set).exists()},
        "hh_capture": {"path": repo_path(output_path(args.hh_capture)), "exists": output_path(args.hh_capture).exists()},
        "reuse_decision": "retrained_code_specific_heads_because_prior_artifacts_store_metadata/results_not_weights",
    }


def summary_for_split(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_pairs": result.get("n_pairs"),
        "best_hh_trained": row_compact(result.get("best_hh_trained")),
        "best_code_trained": row_compact(result.get("best_code_trained")),
        "best_antisymlinear": row_compact(result.get("best_antisymlinear")),
        "best_nonorm": row_compact(result.get("best_nonorm")),
        "best_HH_AntisymLinear": row_compact(result.get("best_HH_AntisymLinear")),
        "best_HH_NoNorm": row_compact(result.get("best_HH_NoNorm")),
        "best_CODE_AntisymLinear": row_compact(result.get("best_CODE_AntisymLinear")),
        "best_CODE_NoNorm": row_compact(result.get("best_CODE_NoNorm")),
    }


def acc(row: dict[str, Any] | None) -> float:
    if not row:
        return float("nan")
    return float(row["metrics"]["canonical_accuracy"])


def describe(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(clean),
        "mean": float(mean(clean)),
        "std": float(pstdev(clean)) if len(clean) > 1 else 0.0,
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def random_eval_splits(n: int, split_size: int, n_splits: int, seed: int) -> list[list[int]]:
    if split_size <= 0 or split_size >= n:
        raise ValueError(f"random split size must be in [1, {n - 1}], got {split_size}")
    rng = random.Random(seed)
    all_indices = list(range(n))
    return [sorted(rng.sample(all_indices, split_size)) for _ in range(n_splits)]


def family_accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "best_hh_trained",
        "best_code_trained",
        "best_HH_AntisymLinear",
        "best_HH_NoNorm",
        "best_CODE_AntisymLinear",
        "best_CODE_NoNorm",
    )
    return {key: describe([float(row[key]["accuracy"]) for row in rows if isinstance(row.get(key), dict)]) for key in keys}


def per_config_accuracy_summary(split_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for split in split_results:
        for row in split["table"]:
            key = (str(row["head_family"]), str(row["config"]), str(row["architecture"]))
            buckets.setdefault(key, []).append(float(row["metrics"]["canonical_accuracy"]))
    out = []
    for (family, config, architecture), values in sorted(buckets.items()):
        out.append({
            "head_family": family,
            "config": config,
            "architecture": architecture,
            "accuracy": describe(values),
        })
    return out


def write_random_report_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    stats = summary["accuracy_summary"]
    lines = [
        "# Code-Trained And HH-Trained Tiny Taps On Random HH Splits",
        "",
        f"RANDOM_HH_SPLIT_EVAL_VERDICT = {summary['RANDOM_HH_SPLIT_EVAL_VERDICT']}",
        "",
        "## Accuracy Summary",
        "",
        f"- random_splits: `{summary['random_splits']}`",
        f"- split_size: `{summary['random_split_size']}`",
        f"- random_split_seed: `{summary['random_split_seed']}`",
        f"- best HH-trained accuracy: `{stats['best_hh_trained']}`",
        f"- best code-trained accuracy: `{stats['best_code_trained']}`",
        f"- best HH AntisymLinear accuracy: `{stats['best_HH_AntisymLinear']}`",
        f"- best HH NoNorm accuracy: `{stats['best_HH_NoNorm']}`",
        f"- best CODE AntisymLinear accuracy: `{stats['best_CODE_AntisymLinear']}`",
        f"- best CODE NoNorm accuracy: `{stats['best_CODE_NoNorm']}`",
        "",
        "## Per-Split Best Rows",
        "",
        "| split | eval_indices | best_hh_acc | best_hh_config | best_code_acc | best_code_config |",
        "| ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in summary["per_split_best"]:
        hh = row["best_hh_trained"]
        code = row["best_code_trained"]
        lines.append(
            f"| {row['split_id']} | `{row['eval_indices']}` | {rate(hh['accuracy'])} | "
            f"`{hh['config']} / {hh['architecture']}` | {rate(code['accuracy'])} | "
            f"`{code['config']} / {code['architecture']}` |"
        )
    lines.extend([
        "",
        "## Per-Config Accuracy Summary",
        "",
        "| family | config | architecture | mean | std | min | max |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["per_config_accuracy_summary"]:
        s = row["accuracy"]
        lines.append(
            f"| `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | "
            f"{rate(s['mean'])} | {rate(s['std'])} | {rate(s['min'])} | {rate(s['max'])} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- hh_capture: `{summary['hh_capture']}`",
        f"- code_features: `{summary['code_features']}`",
        f"- code_training: `{summary['code_training']}`",
        "",
        "## Commands Run",
        "",
        "```bash",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_code_trained_taps_on_hh.py",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_trained_taps_on_hh.py --random-splits 10 --random-split-size 20 --output opi/taps/probes/code_trained_vs_hh_trained_random20_hh_splits_2026-05-17.json --output-md opi/taps/probes/code_trained_vs_hh_trained_random20_hh_splits_2026-05-17.md",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_random_splits(
    *,
    args: argparse.Namespace,
    hh_payload: dict[str, Any],
    code_heads: list[dict[str, Any]],
    code_training: dict[str, Any],
    device: torch.device,
    out_json: Path,
    out_md: Path,
) -> None:
    n_hh = len(hh_payload.get("packs", []))
    splits = random_eval_splits(
        n=n_hh,
        split_size=int(args.random_split_size),
        n_splits=int(args.random_splits),
        seed=int(args.random_split_seed),
    )
    split_results: list[dict[str, Any]] = []
    per_split_best: list[dict[str, Any]] = []
    all_indices = set(range(n_hh))
    for split_id, eval_idx in enumerate(splits):
        train_idx = sorted(all_indices - set(eval_idx))
        print(f"random HH split {split_id + 1}/{len(splits)} train={len(train_idx)} eval={len(eval_idx)}", flush=True)
        hh_heads = train_hh_heads(hh_payload=hh_payload, train_idx=train_idx, eval_idx=eval_idx, args=args, device=device)
        result = evaluate_heads_on_hh(
            heads=hh_heads + code_heads,
            hh_payload=hh_payload,
            split_name=f"random20_split_{split_id}",
            indices=eval_idx,
            device=device,
        )
        split_results.append(result)
        per_split_best.append({
            "split_id": split_id,
            "eval_indices": eval_idx,
            "best_hh_trained": row_compact(result.get("best_hh_trained")),
            "best_code_trained": row_compact(result.get("best_code_trained")),
            "best_HH_AntisymLinear": row_compact(result.get("best_HH_AntisymLinear")),
            "best_HH_NoNorm": row_compact(result.get("best_HH_NoNorm")),
            "best_CODE_AntisymLinear": row_compact(result.get("best_CODE_AntisymLinear")),
            "best_CODE_NoNorm": row_compact(result.get("best_CODE_NoNorm")),
        })
    summary_rows = []
    for row in per_split_best:
        summary_rows.append({
            key: row[key]
            for key in (
                "best_hh_trained",
                "best_code_trained",
                "best_HH_AntisymLinear",
                "best_HH_NoNorm",
                "best_CODE_AntisymLinear",
                "best_CODE_NoNorm",
            )
        })
    payload = {
        "random_hh_split_eval_verdict": "DONE",
        "eval_results": split_results,
        "per_config_accuracy_summary": per_config_accuracy_summary(split_results),
        "summary": {
            "RANDOM_HH_SPLIT_EVAL_VERDICT": "DONE",
            "random_splits": int(args.random_splits),
            "random_split_size": int(args.random_split_size),
            "random_split_seed": int(args.random_split_seed),
            "hh_capture": repo_path(output_path(args.hh_capture)),
            "code_features": repo_path(output_path(args.code_features)),
            "code_training": code_training,
            "accuracy_summary": family_accuracy_summary(summary_rows),
            "per_split_best": per_split_best,
            "files_modified_or_created": [
                "shared/utilities/tests/manual/evaluate_code_trained_taps_on_hh.py",
                repo_path(out_json),
                repo_path(out_md),
            ],
        },
    }
    write_json(out_json, payload)
    write_random_report_md(out_md, payload)
    stats = payload["summary"]["accuracy_summary"]
    print(f"RANDOM_HH_SPLIT_EVAL_VERDICT = DONE", flush=True)
    print(f"best_hh_trained_accuracy = {stats['best_hh_trained']}", flush=True)
    print(f"best_code_trained_accuracy = {stats['best_code_trained']}", flush=True)
    print(f"best_HH_AntisymLinear_accuracy = {stats['best_HH_AntisymLinear']}", flush=True)
    print(f"best_HH_NoNorm_accuracy = {stats['best_HH_NoNorm']}", flush=True)
    print(f"best_CODE_AntisymLinear_accuracy = {stats['best_CODE_AntisymLinear']}", flush=True)
    print(f"best_CODE_NoNorm_accuracy = {stats['best_CODE_NoNorm']}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    heldout = payload["eval_results"]["heldout_hh_eval"]
    diagnostic = payload["eval_results"]["all200_diagnostic"]
    lines = [
        "# Code-Trained Tiny Taps On HH-RLHF",
        "",
        f"CODE_TO_HH_TRANSFER_VERDICT = {summary['CODE_TO_HH_TRANSFER_VERDICT']}",
        f"HH_BASELINE_VERDICT = {summary['HH_BASELINE_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## 1. Artifacts Located / Retrained",
        "",
        f"- located_artifacts: `{summary['located_artifacts']}`",
        f"- code_training: `{summary['code_training']}`",
        f"- hh_training: `{summary['hh_training']}`",
        "",
        "## 2. HH Eval Split",
        "",
        f"- split_policy: `{summary['hh_split']['split_policy']}`",
        f"- train_pairs: `{summary['hh_split']['train_pairs']}`",
        f"- heldout_eval_pairs: `{summary['hh_split']['heldout_eval_pairs']}`",
        f"- all200_diagnostic_pairs: `{summary['hh_split']['all200_diagnostic_pairs']}`",
        "",
        "## 3. HH-Trained Baseline Table",
        "",
        table_md([row for row in heldout["table"] if row["head_family"] == "HH"]),
        "",
        "## 4. Code-Trained-On-HH Transfer Table",
        "",
        table_md([row for row in heldout["table"] if row["head_family"] == "CODE"]),
        "",
        "## 5. Best Config Comparison",
        "",
        f"- heldout best HH-trained: `{summary['heldout']['best_hh_trained']}`",
        f"- heldout best code-trained: `{summary['heldout']['best_code_trained']}`",
        f"- all200 diagnostic best HH-trained: `{summary['all200_diagnostic']['best_hh_trained']}`",
        f"- all200 diagnostic best code-trained: `{summary['all200_diagnostic']['best_code_trained']}`",
        "",
        "## 6. NoNorm vs AntisymLinear Behavior",
        "",
        f"- heldout best AntisymLinear: `{summary['heldout']['best_antisymlinear']}`",
        f"- heldout best NoNorm: `{summary['heldout']['best_nonorm']}`",
        f"- heldout best CODE AntisymLinear: `{summary['heldout']['best_CODE_AntisymLinear']}`",
        f"- heldout best CODE NoNorm: `{summary['heldout']['best_CODE_NoNorm']}`",
        "",
        "## 7. Interpretation",
        "",
        summary["one_sentence_interpretation"],
        "",
        "The all-200 table is diagnostic only because it includes the HH baseline training pairs.",
        "",
        "## All-200 Diagnostic Table",
        "",
        table_md(diagnostic["table"]),
        "",
        "## 8. Files Modified / Created",
        "",
    ]
    lines.extend(f"- `{item}`" for item in summary["files_modified_or_created"])
    lines.extend([
        "",
        "## 9. Commands Run",
        "",
        "```bash",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_code_trained_taps_on_hh.py",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_trained_taps_on_hh.py",
        "```",
        "",
        "## 10. Blockers",
        "",
        summary.get("blockers") or "None.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def table_md(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| family | config | architecture | canonical | flipped | centered | pairwise | score_mean | score_std | antisym_mean_abs | antisym_max_abs |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | "
            f"{rate(m['canonical_accuracy'])} | {rate(m['flipped_accuracy'])} | "
            f"{rate(m['centered_accuracy'])} | {rate(m['pairwise_accuracy'])} | "
            f"{rate(m['score_mean'])} | {rate(m['score_std'])} | "
            f"{rate(m['antisymmetry_mean_abs'])} | {rate(m['antisymmetry_max_abs'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    hh_path = output_path(args.hh_capture)
    code_features_path = output_path(args.code_features)
    code_eval_set_path = output_path(args.code_eval_set)
    out_json = output_path(args.output)
    out_md = output_path(args.output_md)
    located = located_artifacts(args)
    blockers: list[str] = []
    for key in ("hh_capture", "code_features", "code_eval_set"):
        if not located[key]["exists"]:
            blockers.append(f"missing {key}: {located[key]['path']}")
    if blockers:
        payload = {
            "code_to_hh_transfer_verdict": "NOT_RUN",
            "hh_baseline_verdict": "NOT_RUN",
            "summary": {
                "CODE_TO_HH_TRANSFER_VERDICT": "NOT_RUN",
                "HH_BASELINE_VERDICT": "NOT_RUN",
                "RECOMMENDED_NEXT": recommended_next("NOT_RUN"),
                "located_artifacts": located,
                "blockers": "; ".join(blockers),
                "files_modified_or_created": ["shared/utilities/tests/manual/evaluate_code_trained_taps_on_hh.py", repo_path(out_json), repo_path(out_md)],
            },
        }
        write_json(out_json, payload)
        write_report_md(out_md, {**payload, "eval_results": {"heldout_hh_eval": {"table": []}, "all200_diagnostic": {"table": []}}})
        raise SystemExit("CODE_TO_HH_TRANSFER_VERDICT=NOT_RUN")

    eval_set_payload = load_json(code_eval_set_path)
    feature_payload = torch.load(code_features_path, map_location="cpu", weights_only=False)
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    n_hh = len(hh_payload.get("packs", []))
    if n_hh < 2:
        raise SystemExit("HH capture has fewer than 2 pairs")
    if int(args.random_splits) > 0:
        code_heads, code_training = train_code_heads(feature_payload=feature_payload, args=args, device=device)
        run_random_splits(
            args=args,
            hh_payload=hh_payload,
            code_heads=code_heads,
            code_training={
                **code_training,
                "code_features": repo_path(code_features_path),
                "code_eval_set": repo_path(code_eval_set_path),
                "expanded_eval_set_verdict": eval_set_payload.get("expanded_strict_clean_set_verdict", ""),
                "expanded_feature_verdict": feature_payload.get("meta", {}).get("expanded_strict_clean_feature_verdict", ""),
            },
            device=device,
            out_json=out_json,
            out_md=out_md,
        )
        return
    train_end = min(int(args.hh_train_pairs), n_hh - 1)
    train_idx = list(range(train_end))
    heldout_idx = list(range(train_end, n_hh))
    all_idx = list(range(n_hh))

    hh_heads = train_hh_heads(hh_payload=hh_payload, train_idx=train_idx, eval_idx=heldout_idx, args=args, device=device)
    code_heads, code_training = train_code_heads(feature_payload=feature_payload, args=args, device=device)
    all_heads = hh_heads + code_heads

    eval_results = {
        "heldout_hh_eval": evaluate_heads_on_hh(
            heads=all_heads,
            hh_payload=hh_payload,
            split_name="heldout_hh_eval",
            indices=heldout_idx,
            device=device,
        ),
        "all200_diagnostic": evaluate_heads_on_hh(
            heads=all_heads,
            hh_payload=hh_payload,
            split_name="all200_diagnostic",
            indices=all_idx,
            device=device,
        ),
    }
    heldout = eval_results["heldout_hh_eval"]
    best_hh = heldout.get("best_hh_trained")
    best_code = heldout.get("best_code_trained")
    hh_verdict = verdict_for_hh_baseline(best_hh)
    code_verdict = verdict_for_code_transfer(best_code, best_hh)
    rec_next = recommended_next(code_verdict)
    summary = {
        "CODE_TO_HH_TRANSFER_VERDICT": code_verdict,
        "HH_BASELINE_VERDICT": hh_verdict,
        "RECOMMENDED_NEXT": rec_next,
        "located_artifacts": located,
        "hh_split": {
            "split_policy": "first_180_train_last_20_primary_eval_no_saved_split_metadata_found",
            "train_pairs": len(train_idx),
            "heldout_eval_pairs": len(heldout_idx),
            "all200_diagnostic_pairs": len(all_idx),
        },
        "hh_training": {
            "hh_capture": repo_path(hh_path),
            "canonical_pairs": n_hh,
            "train_pairs": len(train_idx),
            "heldout_pairs": len(heldout_idx),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "swap_augmentation": False,
            "lambda_sym": 0.0,
        },
        "code_training": {
            **code_training,
            "code_features": repo_path(code_features_path),
            "code_eval_set": repo_path(code_eval_set_path),
            "expanded_eval_set_verdict": eval_set_payload.get("expanded_strict_clean_set_verdict", ""),
            "expanded_feature_verdict": feature_payload.get("meta", {}).get("expanded_strict_clean_feature_verdict", ""),
        },
        "heldout": summary_for_split(heldout),
        "all200_diagnostic": summary_for_split(eval_results["all200_diagnostic"]),
        "one_sentence_interpretation": interpretation(code_verdict),
        "blockers": "",
        "files_modified_or_created": [
            "shared/utilities/tests/manual/evaluate_code_trained_taps_on_hh.py",
            repo_path(out_json),
            repo_path(out_md),
        ],
    }
    payload = {
        "code_to_hh_transfer_verdict": code_verdict,
        "hh_baseline_verdict": hh_verdict,
        "recommended_next": rec_next,
        "hh_capture": repo_path(hh_path),
        "code_features": repo_path(code_features_path),
        "code_eval_set": repo_path(code_eval_set_path),
        "eval_results": eval_results,
        "summary": summary,
        "notes": [
            "Primary HH eval uses the last 20 pairs because no reusable saved HH split metadata was found.",
            "The all-200 diagnostic includes HH baseline training pairs and is not the primary verdict source.",
            "The probe uses pairwise logits only; labels are canonical HH chosen/rejected pairs.",
        ],
    }
    write_json(out_json, payload)
    write_report_md(out_md, payload)
    print(f"CODE_TO_HH_TRANSFER_VERDICT = {code_verdict}", flush=True)
    print(f"HH_BASELINE_VERDICT = {hh_verdict}", flush=True)
    print(f"RECOMMENDED_NEXT = {rec_next}", flush=True)
    print(f"best_hh_heldout = {row_compact(best_hh)}", flush=True)
    print(f"best_code_heldout = {row_compact(best_code)}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
