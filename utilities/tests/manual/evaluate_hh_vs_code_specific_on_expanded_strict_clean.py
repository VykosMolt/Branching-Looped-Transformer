"""Evaluate HH-trained and code-specific tiny taps on expanded strict-clean code."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
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
    split_indices,
    train_hh_head,
)
from math_bg_probe_lib import MATH_CONFIGS  # noqa: E402
from train_code_specific_tiny_heads_and_eval import (  # noqa: E402
    best_by_arch,
    config_features,
    evaluate_matrices,
    feature_map,
    records_for_eval_set,
    score_matrix,
    split_pairs_by_task,
    train_head,
)


EVAL_SET_JSON = REPORT_DIR / "code_expanded_strict_clean_eval_set_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.pt"
HH_CAPTURE = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"

OUTPUT_JSON = REPORT_DIR / "expanded_strict_clean_code_projection_comparison_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "expanded_strict_clean_code_projection_comparison_2026-05-17.md"
HEADS_JSON = REPORT_DIR / "code_specific_heads_expanded_strict_clean_2026-05-17.json"
HEADS_MD = REPORT_DIR / "code_specific_heads_expanded_strict_clean_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "expanded_strict_clean_code_projection_comparison_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
]

ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=str(EVAL_SET_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--hh-capture", default=str(HH_CAPTURE))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default="")
    parser.add_argument("--heads-output", default=str(HEADS_JSON))
    parser.add_argument("--heads-md", default="")
    parser.add_argument("--summary-output", default=str(SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(SUMMARY_MD))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    if not args.heads_md:
        args.heads_md = str(Path(args.heads_output).with_suffix(".md"))
    return args


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


def label_arch(architecture: str) -> str:
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
        "top1": m["top1_tournament_acc"],
        "over_random": m["top1_over_random_baseline"],
        "pairwise": m["pairwise_acc"],
        "cycle": m["cycle_rate"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
    }


def best_overall(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    )


def best_by_family(rows: Sequence[dict[str, Any]], family: str) -> dict[str, Any] | None:
    return best_overall([row for row in rows if row.get("head_family") == family])


def best_by_family_arch(rows: Sequence[dict[str, Any]], family_architecture: str) -> dict[str, Any] | None:
    return best_overall([row for row in rows if row.get("family_architecture") == family_architecture])


def transfer_verdict(best: dict[str, Any] | None, baseline: float) -> str:
    if best is None:
        return "NOT_RUN"
    metrics = best["metrics"]
    top1 = float(metrics["top1_tournament_acc"])
    pairwise = float(metrics["pairwise_acc"])
    cycle = float(metrics["cycle_rate"])
    if top1 >= baseline + 0.15 and pairwise >= 0.60 and cycle <= 0.05:
        return "GOOD"
    if top1 >= baseline + 0.05 or pairwise >= 0.55:
        return "WEAK"
    return "POOR"


def random_top1_baseline(records: Sequence[dict[str, Any]]) -> float:
    if not records:
        return float("nan")
    return float(sum(float(row["labels"].to(torch.float32).mean()) for row in records) / len(records))


def per_task_breakdown(
    matrices: Sequence[torch.Tensor],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mat, record in zip(matrices, records):
        totals = mat.sum(dim=1)
        pred = int(torch.argmax(totals).item())
        top2_margin = 0.0
        if totals.numel() >= 2:
            vals = torch.topk(totals, k=2).values
            top2_margin = float(vals[0] - vals[1])
        label_names = list(record["label_names"])
        rows.append({
            "task_id": record["task_id"],
            "source": record.get("source", "unknown"),
            "n_candidates": len(record["candidate_uids"]),
            "label_counts": dict(Counter(label_names)),
            "predicted_index": pred,
            "predicted_candidate_uid": record["candidate_uids"][pred],
            "predicted_label": label_names[pred],
            "predicted_is_correct": bool(record["labels"][pred]),
            "top2_margin": top2_margin,
        })
    return rows


def evaluate_set(
    *,
    set_name: str,
    records: list[dict[str, Any]],
    heads: Sequence[dict[str, Any]],
    baseline: float,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matrices_by_row: list[list[torch.Tensor]] = []
    for head_info in heads:
        config = head_info["config"]
        architecture = head_info["architecture"]
        family = head_info["head_family"]
        head = head_info["head"]
        feats_by_record = config_features(records, config)
        head = head.to(device)
        matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
        head = head.to("cpu")
        metrics = evaluate_matrices(matrices, records, baseline)
        row = {
            "eval_set": set_name,
            "head_family": family,
            "config": config,
            "architecture": architecture,
            "family_architecture": f"{family}_{label_arch(architecture)}",
            "train_metrics": head_info["train_metrics"],
            "metrics": metrics,
        }
        rows.append(row)
        matrices_by_row.append(matrices)

    best = best_overall(rows)
    best_idx = rows.index(best) if best in rows else -1
    best_matrices = matrices_by_row[best_idx] if best_idx >= 0 else []
    return {
        "eval_set": set_name,
        "n_tournaments": len(records),
        "n_candidates": sum(len(row["candidate_uids"]) for row in records),
        "random_top1_baseline": baseline,
        "transfer_table": rows,
        "best_overall": best,
        "best_hh_trained": best_by_family(rows, "HH"),
        "best_code_trained": best_by_family(rows, "CODE"),
        "best_antisymlinear": best_overall([row for row in rows if row["architecture"] == "AntisymLinear"]),
        "best_nonorm": best_overall([row for row in rows if row["architecture"] == "AntisymLinearNoNorm"]),
        "best_HH_AntisymLinear": best_by_family_arch(rows, "HH_AntisymLinear"),
        "best_HH_NoNorm": best_by_family_arch(rows, "HH_NoNorm"),
        "best_CODE_AntisymLinear": best_by_family_arch(rows, "CODE_AntisymLinear"),
        "best_CODE_NoNorm": best_by_family_arch(rows, "CODE_NoNorm"),
        "per_task_breakdown_best": per_task_breakdown(best_matrices, records) if best_matrices else [],
    }


def train_hh_heads(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hh_path = output_path(args.hh_capture)
    if not hh_path.exists():
        raise SystemExit(f"missing HH capture: {hh_path}")
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), int(args.heldout), int(args.seed))
    heads: list[dict[str, Any]] = []
    for config in MATH_CONFIGS:
        print(f"training HH heads {config}", flush=True)
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
    meta = {
        "hh_capture": repo_path(hh_path),
        "hh_pairs_total": len(hh_payload["packs"]),
        "hh_train_pairs": len(train_idx),
        "hh_heldout_pairs": len(eval_idx),
        "heldout": int(args.heldout),
        "seed": int(args.seed),
        "retrained": True,
    }
    return heads, meta


def train_code_heads(
    *,
    eval_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    pooled_by_uid: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    eval_task_ids = set(str(task_id) for task_id in eval_payload.get("all16_task_ids", []))
    raw_pairs = list(feature_payload.get("training_pairs_primary", []) or [])
    leakage = sorted({str(pair.get("task_id")) for pair in raw_pairs if str(pair.get("task_id")) in eval_task_ids})
    if leakage:
        return [], "BLOCKED", {"blocked_reason": "expanded eval task leakage in code training pairs", "leakage_task_ids": leakage}
    missing_feature_pairs = [
        pair for pair in raw_pairs
        if pair.get("preferred_uid") not in pooled_by_uid or pair.get("rejected_uid") not in pooled_by_uid
    ]
    if missing_feature_pairs:
        return [], "BLOCKED", {
            "blocked_reason": "training pairs missing features",
            "missing_pair_count": len(missing_feature_pairs),
        }
    task_ids = sorted({str(pair["task_id"]) for pair in raw_pairs})
    if len(task_ids) < 8 or len(raw_pairs) < 30:
        return [], "INSUFFICIENT_TRAIN", {
            "training_task_count": len(task_ids),
            "primary_training_pair_count": len(raw_pairs),
        }

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
    meta = {
        "training_task_count": len(task_ids),
        "primary_training_pair_count": len(raw_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "validation": validation,
        "excluded_eval_task_ids": sorted(eval_task_ids),
        "reused_previous_heads": False,
        "retrained": True,
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": min(int(args.batch_size), len(train_pairs)),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "swap_augmentation": False,
        "lambda_sym": 0.0,
    }
    return heads, "RETRAINED", meta


def write_heads_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code-Specific Heads For Expanded Strict-Clean Eval",
        "",
        f"CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT = {payload['code_specific_expanded_train_verdict']}",
        "",
        f"- training_tasks: `{payload['metadata'].get('training_task_count')}`",
        f"- primary_training_pairs: `{payload['metadata'].get('primary_training_pair_count')}`",
        f"- retrained: `{payload['metadata'].get('retrained')}`",
        f"- reused_previous_heads: `{payload['metadata'].get('reused_previous_heads', False)}`",
        f"- excluded_eval_task_ids: `{payload['metadata'].get('excluded_eval_task_ids', [])}`",
        "",
        "No candidate from OLD6 or NEW10 is present in code-specific training pairs.",
        "",
    ]
    if payload.get("metadata", {}).get("validation"):
        lines.extend([
            "## Validation Split",
            "",
            f"- validation: `{payload['metadata']['validation']}`",
            "",
        ])
    if payload.get("metadata", {}).get("blocked_reason"):
        lines.extend(["## Blocker", "", payload["metadata"]["blocked_reason"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def source_split_summary(eval_results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("MBPP_primary", "HumanEval_primary"):
        result = eval_results.get(name)
        if not result:
            continue
        out[name] = {
            "n_tasks": result["n_tournaments"],
            "random_top1_baseline": result["random_top1_baseline"],
            "best_hh_trained": row_compact(result.get("best_hh_trained")),
            "best_code_trained": row_compact(result.get("best_code_trained")),
            "best_overall": row_compact(result.get("best_overall")),
        }
    return out


def comparison_verdict(
    *,
    hh_best: dict[str, Any] | None,
    code_best: dict[str, Any] | None,
    hh_verdict: str,
    code_verdict: str,
) -> str:
    if hh_best is None or code_best is None:
        return "NOT_RUN"
    hh_m = hh_best["metrics"]
    code_m = code_best["metrics"]
    hh_top1 = float(hh_m["top1_tournament_acc"])
    code_top1 = float(code_m["top1_tournament_acc"])
    hh_pair = float(hh_m["pairwise_acc"])
    code_pair = float(code_m["pairwise_acc"])
    code_advantage = (code_top1 >= hh_top1 + 0.10 or code_pair >= hh_pair + 0.10) and code_pair >= 0.60
    both_good = hh_verdict == "GOOD" and code_verdict == "GOOD"
    both_at_least_weak = hh_verdict in {"GOOD", "WEAK"} and code_verdict in {"GOOD", "WEAK"}
    hh_recovers = hh_top1 >= code_top1 - 0.05 and hh_pair >= code_pair - 0.05 and both_at_least_weak
    if code_advantage:
        return "CODE_SPECIFIC_ADVANTAGE"
    if both_good:
        return "BOTH_GOOD"
    if hh_recovers:
        return "HH_RECOVERS"
    return "BOTH_WEAK"


def recommended_next(verdict: str) -> str:
    if verdict == "CODE_SPECIFIC_ADVANTAGE":
        return "update_BG_phase1_plan_for_domain_specific_projection_training"
    if verdict == "HH_RECOVERS":
        return "keep_HH_taps_as_general_default_and_continue_task_curriculum"
    if verdict == "BOTH_GOOD":
        return "use_HH_as_general_default_code_specific_as_optional_specialist"
    if verdict == "BOTH_WEAK":
        return "investigate_pooling_or_richer_head_before_more_code_data"
    return "fix_artifact_or_feature_blocker"


def winning_layer(row: dict[str, Any] | None) -> str:
    if not row:
        return "NA"
    config = str(row.get("config", ""))
    if config.startswith("47_"):
        return "47"
    if config.startswith("36_"):
        return "36"
    if config.startswith("24_"):
        return "24"
    return "NA"


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Expanded strict-clean code projection comparison (2026-05-17)"
    all16 = summary["ALL16"]
    old6 = summary["OLD6"]
    new10 = summary["NEW10"]
    text = "\n".join([
        "",
        title,
        "",
        f"- EXPANDED_STRICT_CLEAN_SET_VERDICT: `{summary['EXPANDED_STRICT_CLEAN_SET_VERDICT']}`",
        f"- EXPANDED_STRICT_CLEAN_FEATURE_VERDICT: `{summary['EXPANDED_STRICT_CLEAN_FEATURE_VERDICT']}`",
        f"- CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT: `{summary['CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT']}`",
        f"- EXPANDED_HH_TRANSFER_VERDICT: `{summary['EXPANDED_HH_TRANSFER_VERDICT']}`",
        f"- EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT: `{summary['EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT']}`",
        f"- EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT: `{summary['EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- OLD6 best HH / CODE: `{old6['best_hh_trained']}` / `{old6['best_code_trained']}`",
        f"- NEW10 best HH / CODE: `{new10['best_hh_trained']}` / `{new10['best_code_trained']}`",
        f"- ALL16 best HH / CODE: `{all16['best_hh_trained']}` / `{all16['best_code_trained']}`",
        f"- best HH-trained row: `{summary['best_hh_trained_row']}`",
        f"- best code-trained row: `{summary['best_code_trained_row']}`",
        f"- code-specific advantage survived: `{summary['code_specific_advantage_survived']}`",
        f"- winning architecture: `{summary['winning_architecture']}`",
        f"- winning layer: `{summary['winning_layer']}`",
        f"- interpretation: {summary['one_sentence_interpretation']}",
        "- full report: `opi/taps/probes/expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md`",
        "",
    ])
    appended: list[str] = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def write_comparison_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Expanded Strict-Clean Code Projection Comparison",
        "",
        f"EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT = {payload['expanded_strict_clean_comparison_verdict']}",
        "",
    ]
    for set_name in ("ALL16_primary", "OLD6_primary", "NEW10_primary", "ALL16_plus_wrong_code"):
        result = payload["eval_results"].get(set_name)
        if not result:
            continue
        lines.extend([
            f"## {set_name}",
            "",
            f"- n_tournaments: `{result['n_tournaments']}`",
            f"- n_candidates: `{result['n_candidates']}`",
            f"- random_top1_baseline: `{result['random_top1_baseline']}`",
            f"- best HH-trained: `{row_compact(result.get('best_hh_trained'))}`",
            f"- best code-trained: `{row_compact(result.get('best_code_trained'))}`",
            f"- best AntisymLinear: `{row_compact(result.get('best_antisymlinear'))}`",
            f"- best NoNorm: `{row_compact(result.get('best_nonorm'))}`",
            "",
            "| family | config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in result["transfer_table"]:
            m = row["metrics"]
            lines.append(
                f"| `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | "
                f"{rate(m['top1_tournament_acc'])} | {rate(m['top1_over_random_baseline'])} | "
                f"{rate(m['pairwise_acc'])} | {rate(m['condorcet_winner_rate'])} | "
                f"{rate(m['cycle_rate'])} | {rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
            )
        lines.extend(["", "### Per-Task Breakdown For Best Row", ""])
        for row in result.get("per_task_breakdown_best", []):
            lines.append(
                f"- `{row['task_id']}` source=`{row.get('source', 'unknown')}` pred_label=`{row['predicted_label']}` "
                f"correct=`{row['predicted_is_correct']}` margin={row['top2_margin']:.3f}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_for_eval_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_tasks": result.get("n_tournaments"),
        "n_candidates": result.get("n_candidates"),
        "random_top1_baseline": result.get("random_top1_baseline"),
        "best_hh_trained": row_compact(result.get("best_hh_trained")),
        "best_code_trained": row_compact(result.get("best_code_trained")),
        "best_overall": row_compact(result.get("best_overall")),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Expanded Strict-Clean Code Projection Comparison Summary",
        "",
        f"EXPANDED_STRICT_CLEAN_SET_VERDICT = {summary['EXPANDED_STRICT_CLEAN_SET_VERDICT']}",
        f"EXPANDED_STRICT_CLEAN_FEATURE_VERDICT = {summary['EXPANDED_STRICT_CLEAN_FEATURE_VERDICT']}",
        f"CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT = {summary['CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT']}",
        f"EXPANDED_HH_TRANSFER_VERDICT = {summary['EXPANDED_HH_TRANSFER_VERDICT']}",
        f"EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT = {summary['EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT']}",
        f"EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT = {summary['EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## 1. Eval Set Construction",
        "",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- OLD6: `{summary['OLD6']}`",
        f"- NEW10: `{summary['NEW10']}`",
        f"- ALL16: `{summary['ALL16']}`",
        "",
        "## 2. Feature Coverage",
        "",
        f"- feature verdict: `{summary['EXPANDED_STRICT_CLEAN_FEATURE_VERDICT']}`",
        f"- feature file: `{summary['features_path']}`",
        f"- required candidates: `{summary['feature_required_candidates']}`",
        f"- recaptured candidates: `{summary['feature_recaptured_candidates']}`",
        "",
        "## 3. Training Splits",
        "",
        f"- HH training: `{summary['hh_training']}`",
        f"- code training: `{summary['code_training']}`",
        "",
        "## 4. HH-Trained Results",
        "",
        f"- ALL16 verdict: `{summary['EXPANDED_HH_TRANSFER_VERDICT']}`",
        f"- best HH-trained row: `{summary['best_hh_trained_row']}`",
        "",
        "## 5. Code-Trained Results",
        "",
        f"- ALL16 verdict: `{summary['EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT']}`",
        f"- best code-trained row: `{summary['best_code_trained_row']}`",
        "",
        "## 6. Side-By-Side Comparison",
        "",
        f"- comparison verdict: `{summary['EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT']}`",
        f"- code-specific advantage survived: `{summary['code_specific_advantage_survived']}`",
        f"- winning architecture: `{summary['winning_architecture']}`",
        f"- winning layer: `{summary['winning_layer']}`",
        "",
        "## 7. OLD6 vs NEW10 Comparison",
        "",
        f"- OLD6: `{summary['OLD6']}`",
        f"- NEW10: `{summary['NEW10']}`",
        "",
        "## 8. MBPP vs HumanEval",
        "",
        f"- source split: `{summary['source_split']}`",
        "",
        "## 9. Interpretation",
        "",
        summary["one_sentence_interpretation"],
        "",
        "## 10. Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    if not summary.get("docs_updated"):
        lines.append("- none appended; section already present or doc missing")
    lines.extend(["", "## 11. Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary["files_modified_or_created"])
    lines.extend([
        "",
        "## 12. Commands Run",
        "",
        "```bash",
        "venv/bin/python -m py_compile utilities/tests/manual/build_expanded_strict_clean_eval_set.py",
        "venv/bin/python -m py_compile utilities/tests/manual/ensure_expanded_strict_clean_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_hh_vs_code_specific_on_expanded_strict_clean.py",
        "venv/bin/python -u utilities/tests/manual/build_expanded_strict_clean_eval_set.py",
        "venv/bin/python -u utilities/tests/manual/ensure_expanded_strict_clean_features.py --eval-set opi/taps/probes/code_expanded_strict_clean_eval_set_2026-05-17.json --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_hh_vs_code_specific_on_expanded_strict_clean.py --eval-set opi/taps/probes/code_expanded_strict_clean_eval_set_2026-05-17.json --features opi/taps/probes/code_expanded_strict_clean_features_2026-05-17.pt",
        "```",
        "",
        "## 13. Blockers",
        "",
        summary.get("blockers") or "None.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    eval_set_path = output_path(args.eval_set)
    features_path = output_path(args.features)
    out_json = output_path(args.output)
    out_md = output_path(args.output_md)
    heads_json = output_path(args.heads_output)
    heads_md = output_path(args.heads_md)
    summary_json = output_path(args.summary_output)
    summary_md = output_path(args.summary_md)

    eval_payload = load_json(eval_set_path)
    set_verdict = str(eval_payload.get("expanded_strict_clean_set_verdict", "BLOCKED"))
    if set_verdict not in {"READY", "PARTIAL"}:
        raise SystemExit(f"EXPANDED_STRICT_CLEAN_SET_VERDICT={set_verdict}")
    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    feature_meta = feature_payload.get("meta", {})
    feature_verdict = str(feature_meta.get("expanded_strict_clean_feature_verdict", "BLOCKED"))
    if feature_verdict not in {"READY", "RECAPTURED"}:
        raise SystemExit(f"EXPANDED_STRICT_CLEAN_FEATURE_VERDICT={feature_verdict}")

    pooled_by_uid = feature_map(feature_payload)
    records_by_set: dict[str, list[dict[str, Any]]] = {}
    for set_name, rows in feature_payload.get("eval_sets", {}).items():
        if not rows:
            continue
        records_by_set[set_name] = records_for_eval_set(rows, pooled_by_uid)
    required_eval_sets = ("OLD6_primary", "NEW10_primary", "ALL16_primary")
    missing_sets = [name for name in required_eval_sets if name not in records_by_set]
    if missing_sets:
        raise SystemExit(f"missing required eval sets with features: {missing_sets}")

    hh_heads, hh_train_meta = train_hh_heads(args, device)
    code_heads, code_train_verdict, code_train_meta = train_code_heads(
        eval_payload=eval_payload,
        feature_payload=feature_payload,
        pooled_by_uid=pooled_by_uid,
        args=args,
        device=device,
    )
    if code_train_verdict == "BLOCKED":
        raise SystemExit(f"CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT=BLOCKED: {code_train_meta}")
    heads_payload = {
        "code_specific_expanded_train_verdict": code_train_verdict,
        "metadata": code_train_meta,
        "features": repo_path(features_path),
        "eval_set": repo_path(eval_set_path),
    }
    write_json(heads_json, heads_payload)
    write_heads_md(heads_md, heads_payload)

    all_heads = hh_heads + code_heads
    if not all_heads:
        raise SystemExit("no heads available for evaluation")
    eval_results: dict[str, Any] = {}
    for set_name, records in records_by_set.items():
        baseline = random_top1_baseline(records)
        eval_results[set_name] = evaluate_set(
            set_name=set_name,
            records=records,
            heads=all_heads,
            baseline=baseline,
            device=device,
        )

    all16 = eval_results["ALL16_primary"]
    all16_baseline = float(all16["random_top1_baseline"])
    best_hh = all16.get("best_hh_trained")
    best_code = all16.get("best_code_trained") if code_heads else None
    hh_verdict = transfer_verdict(best_hh, all16_baseline)
    code_verdict = transfer_verdict(best_code, all16_baseline) if code_heads else "NOT_RUN"
    comparison = comparison_verdict(
        hh_best=best_hh,
        code_best=best_code,
        hh_verdict=hh_verdict,
        code_verdict=code_verdict,
    )
    best_overall_all16 = all16.get("best_overall")
    code_specific_advantage_survived = "yes" if comparison == "CODE_SPECIFIC_ADVANTAGE" else "no"
    one_sentence = (
        "Code-specific projection training remains better than HH-trained projection on the expanded strict-clean code branch-selection set."
        if comparison == "CODE_SPECIFIC_ADVANTAGE"
        else (
            "HH-trained projection recovers to near code-specific performance on the expanded strict-clean code branch-selection set."
            if comparison == "HH_RECOVERS"
            else (
                "Both HH-trained and code-specific projections clear the expanded strict-clean code branch-selection threshold."
                if comparison == "BOTH_GOOD"
                else "Neither projection family gives a strong expanded strict-clean code branch-selection result under this tiny-head protocol."
            )
        )
    )

    summary = {
        "EXPANDED_STRICT_CLEAN_SET_VERDICT": set_verdict,
        "EXPANDED_STRICT_CLEAN_FEATURE_VERDICT": feature_verdict,
        "CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT": code_train_verdict,
        "EXPANDED_HH_TRANSFER_VERDICT": hh_verdict,
        "EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT": code_verdict,
        "EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT": comparison,
        "RECOMMENDED_NEXT": recommended_next(comparison),
        "eval_task_count": all16["n_tournaments"],
        "eval_primary_candidate_count": all16["n_candidates"],
        "features_path": repo_path(features_path),
        "feature_required_candidates": feature_meta.get("required_candidates_total"),
        "feature_recaptured_candidates": feature_meta.get("recaptured_candidates"),
        "hh_training": hh_train_meta,
        "code_training": code_train_meta,
        "OLD6": summary_for_eval_result(eval_results["OLD6_primary"]),
        "NEW10": summary_for_eval_result(eval_results["NEW10_primary"]),
        "ALL16": summary_for_eval_result(all16),
        "best_hh_trained_row": row_compact(best_hh),
        "best_code_trained_row": row_compact(best_code),
        "best_nonorm_row": row_compact(all16.get("best_nonorm")),
        "best_antisymlinear_row": row_compact(all16.get("best_antisymlinear")),
        "code_specific_advantage_survived": code_specific_advantage_survived,
        "winning_architecture": label_arch(str(best_overall_all16.get("architecture"))) if best_overall_all16 else "NA",
        "winning_layer": winning_layer(best_overall_all16),
        "winning_family": best_overall_all16.get("head_family", "NA") if best_overall_all16 else "NA",
        "source_split": source_split_summary(eval_results),
        "one_sentence_interpretation": one_sentence,
        "blockers": "",
    }

    result_payload = {
        "expanded_strict_clean_comparison_verdict": comparison,
        "expanded_hh_transfer_verdict": hh_verdict,
        "expanded_code_specific_transfer_verdict": code_verdict,
        "expanded_strict_clean_set_verdict": set_verdict,
        "expanded_strict_clean_feature_verdict": feature_verdict,
        "code_specific_expanded_train_verdict": code_train_verdict,
        "eval_set": repo_path(eval_set_path),
        "features": repo_path(features_path),
        "hh_capture": repo_path(output_path(args.hh_capture)),
        "hh_training": hh_train_meta,
        "code_training": code_train_meta,
        "eval_results": eval_results,
        "summary": summary,
    }
    write_json(out_json, result_payload)
    write_comparison_md(out_md, result_payload)

    docs_updated = append_docs(summary)
    summary["docs_updated"] = docs_updated
    summary["files_modified_or_created"] = [
        "shared/utilities/tests/manual/build_expanded_strict_clean_eval_set.py",
        "shared/utilities/tests/manual/ensure_expanded_strict_clean_features.py",
        "shared/utilities/tests/manual/evaluate_hh_vs_code_specific_on_expanded_strict_clean.py",
        repo_path(eval_set_path),
        repo_path(Path(str(eval_set_path)).with_suffix(".md")),
        repo_path(features_path),
        repo_path(Path(str(features_path)).with_suffix(".md")),
        repo_path(heads_json),
        repo_path(heads_md),
        repo_path(out_json),
        repo_path(out_md),
        repo_path(summary_json),
        repo_path(summary_md),
        *docs_updated,
    ]
    write_json(summary_json, summary)
    write_summary_md(summary_md, summary)

    print(f"EXPANDED_HH_TRANSFER_VERDICT = {hh_verdict}", flush=True)
    print(f"EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT = {code_verdict}", flush=True)
    print(f"EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT = {comparison}", flush=True)
    print(f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    print(f"Wrote {summary_json}", flush=True)
    print(f"Wrote {summary_md}", flush=True)


if __name__ == "__main__":
    main()
