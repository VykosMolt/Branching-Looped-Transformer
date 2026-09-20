"""Evaluate HH-trained linear taps on strict-clean screened code tournaments."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
import numpy as np
import torch
CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()
sys.path.insert(0, str(THIS_DIR))

try:
    from utilities.tests.manual.code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json
except ModuleNotFoundError:
    from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json

from evaluate_hh_transfer_on_clean_gsm8k_extreme import (  # noqa: E402
    build_hh_features,
    config_features,
    evaluate_matrices,
    random_top1_baseline,
    score_matrix,
    split_indices,
    train_hh_head,
)
from math_bg_probe_lib import MATH_CONFIGS  # noqa: E402


TRANSFER_SET_JSON = REPORT_DIR / "code_strict_clean_transfer_set_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "code_strict_clean_transfer_features_2026-05-17.pt"
HH_CAPTURE = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
OUTPUT_JSON = REPORT_DIR / "code_strict_clean_transfer_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_transfer_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "code_strict_clean_transfer_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "code_strict_clean_transfer_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-set", default=str(TRANSFER_SET_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--hh-capture", default=str(HH_CAPTURE))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default="")
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
    return args


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def row_compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {
        "config": row["config"],
        "architecture": row["architecture"],
        "top1": m["top1_tournament_acc"],
        "pairwise": m["pairwise_acc"],
        "cycle": m["cycle_rate"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
    }


def best_by_arch(rows: Sequence[dict[str, Any]], architecture: str) -> dict[str, Any] | None:
    arch_rows = [row for row in rows if row.get("architecture") == architecture]
    if not arch_rows:
        return None
    return max(
        arch_rows,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    )


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


def recommended_next(verdict: str, set_verdict: str) -> str:
    if set_verdict == "TOO_SMALL":
        return "screen_more_tasks_before_transfer"
    if verdict == "GOOD":
        return "code_strict_clean_signal_confirmed_small_n__stop_code_or_screen_more_tasks"
    if verdict == "WEAK":
        return "screen_more_strict_clean_tasks_or_consider_code_specific_training"
    if verdict == "POOR":
        return "code_transfer_weak_on_near_miss__investigate_code_specific_training_or_task_quality"
    return "screen_more_tasks_before_transfer"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_records(feature_payload: dict[str, Any], set_name: str) -> list[dict[str, Any]]:
    by_uid = {str(row["candidate_uid"]): row for row in feature_payload["candidate_features"]}
    records: list[dict[str, Any]] = []
    for tournament in feature_payload["eval_sets"][set_name]:
        candidate_features = [by_uid[str(uid)] for uid in tournament["candidate_uids"]]
        records.append({
            "tournament_id": int(tournament["tournament_id"]),
            "task_id": tournament["task_id"],
            "source": tournament.get("source", "unknown"),
            "difficulty": tournament.get("difficulty", "unknown"),
            "function_name": tournament.get("function_name", ""),
            "prompt": tournament.get("prompt", ""),
            "labels": torch.tensor(list(tournament["labels"]), dtype=torch.bool),
            "pooled": torch.stack([row["pooled"].to(torch.float32) for row in candidate_features], dim=0),
            "candidate_metadata": [row["candidate_metadata"] for row in candidate_features],
            "candidate_uids": list(tournament["candidate_uids"]),
        })
    return records


def per_task_breakdown(matrices: Sequence[torch.Tensor], records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mat, record in zip(matrices, records):
        totals = mat.sum(dim=1)
        pred = int(torch.argmax(totals).item())
        labels = record["labels"]
        meta = record["candidate_metadata"][pred]
        top2_margin = 0.0
        if totals.numel() >= 2:
            vals = torch.topk(totals, k=2).values
            top2_margin = float(vals[0] - vals[1])
        rows.append({
            "task_id": record["task_id"],
            "n_candidates": int(labels.numel()),
            "label_counts": dict(Counter("correct" if bool(x) else "incorrect" for x in labels.tolist())),
            "predicted_index": pred,
            "predicted_candidate_uid": meta.get("candidate_uid", ""),
            "predicted_label": meta.get("label", meta.get("unit_test_label", "")),
            "predicted_is_correct": bool(labels[pred]),
            "predicted_mode": meta.get("mode", ""),
            "predicted_screening_role": meta.get("screening_role", ""),
            "top2_margin": top2_margin,
        })
    return rows


def evaluate_set(
    *,
    set_name: str,
    records: list[dict[str, Any]],
    heads: list[tuple[str, str, Any, dict[str, Any]]],
    baseline: float,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_matrices: list[torch.Tensor] = []
    best: dict[str, Any] | None = None
    for config, architecture, head, train_metrics in heads:
        feats_by_record = config_features(records, config)
        head = head.to(device)
        matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
        head = head.to("cpu")
        metrics = evaluate_matrices(matrices, records, baseline)
        row = {
            "eval_set": set_name,
            "config": config,
            "architecture": architecture,
            "train_metrics": train_metrics,
            "metrics": metrics,
        }
        rows.append(row)
        if best is None or (
            metrics["top1_tournament_acc"],
            metrics["pairwise_acc"],
            -metrics["cycle_rate"],
        ) > (
            best["metrics"]["top1_tournament_acc"],
            best["metrics"]["pairwise_acc"],
            -best["metrics"]["cycle_rate"],
        ):
            best = row
            best_matrices = matrices
    best_antisym = best_by_arch(rows, "AntisymLinear")
    best_nonorm = best_by_arch(rows, "AntisymLinearNoNorm")
    return {
        "eval_set": set_name,
        "n_tournaments": len(records),
        "n_candidates": sum(int(row["labels"].numel()) for row in records),
        "random_top1_baseline": baseline,
        "transfer_table": rows,
        "best_hh_trained": best,
        "best_antisymlinear": best_antisym,
        "best_nonorm": best_nonorm,
        "per_task_breakdown_best": per_task_breakdown(best_matrices, records) if best_matrices else [],
    }


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Strict-clean code transfer micro-eval (2026-05-17)"
    best_a = summary.get("best_antisymlinear", "NA")
    best_n = summary.get("best_nonorm", "NA")
    winner = summary.get("best_hh_trained", "NA")
    text = "\n".join([
        "",
        title,
        "",
        f"- STRICT_CLEAN_TRANSFER_SET_VERDICT: `{summary['STRICT_CLEAN_TRANSFER_SET_VERDICT']}`",
        f"- STRICT_CLEAN_FEATURE_VERDICT: `{summary['STRICT_CLEAN_FEATURE_VERDICT']}`",
        f"- STRICT_CLEAN_TRANSFER_VERDICT: `{summary['STRICT_CLEAN_TRANSFER_VERDICT']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- candidates: `{summary['primary_candidates']}` primary / `{summary['secondary_candidates']}` secondary",
        f"- random_top1_baseline: `{summary['random_top1_baseline_primary']}`",
        f"- best AntisymLinear row: `{best_a}`",
        f"- best NoNorm row: `{best_n}`",
        f"- winner top1 / pairwise / cycle: `{summary['winner_top1_pairwise_cycle']}`",
        f"- transfer survived on strict-clean correct-vs-near_miss candidates: `{summary['strict_clean_transfer_survived']}`",
        "- full report: `opi/taps/probes/code_strict_clean_transfer_2026-05-17_summary.md`",
        f"- interpretation: {summary['one_sentence_interpretation']}",
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


def write_transfer_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strict-Clean Code HH Transfer Micro-Eval",
        "",
        f"STRICT_CLEAN_TRANSFER_VERDICT = {payload['strict_clean_transfer_verdict']}",
        "",
    ]
    for set_name in ("strict_clean_primary", "strict_clean_plus_wrong_code"):
        result = payload["eval_results"][set_name]
        lines.extend([
            f"## {set_name}",
            "",
            f"- n_tournaments: `{result['n_tournaments']}`",
            f"- n_candidates: `{result['n_candidates']}`",
            f"- random_top1_baseline: `{result['random_top1_baseline']}`",
            f"- best AntisymLinear: `{row_compact(result.get('best_antisymlinear'))}`",
            f"- best NoNorm: `{row_compact(result.get('best_nonorm'))}`",
            "",
            "| config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in result["transfer_table"]:
            m = row["metrics"]
            lines.append(
                f"| `{row['config']}` | `{row['architecture']}` | {rate(m['top1_tournament_acc'])} | "
                f"{rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | "
                f"{rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} | "
                f"{rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
            )
        lines.extend(["", "### Per-Task Breakdown For Best Row", ""])
        for row in result["per_task_breakdown_best"]:
            lines.append(
                f"- `{row['task_id']}` pred_label=`{row['predicted_label']}` "
                f"correct=`{row['predicted_is_correct']}` mode=`{row['predicted_mode']}` margin={row['top2_margin']:.3f}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Strict-Clean Code Transfer Micro-Eval Summary",
        "",
        f"STRICT_CLEAN_TRANSFER_SET_VERDICT = {summary['STRICT_CLEAN_TRANSFER_SET_VERDICT']}",
        f"STRICT_CLEAN_FEATURE_VERDICT = {summary['STRICT_CLEAN_FEATURE_VERDICT']}",
        f"STRICT_CLEAN_TRANSFER_VERDICT = {summary['STRICT_CLEAN_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Transfer Set Construction",
        "",
        f"- tasks: `{summary['tasks']}`",
        f"- primary candidates: `{summary['primary_candidates']}`",
        f"- secondary candidates: `{summary['secondary_candidates']}`",
        f"- primary label counts: `{summary['primary_label_counts']}`",
        f"- secondary wrong_code candidates: `{summary['secondary_wrong_code']}`",
        "",
        "## Feature Capture Summary",
        "",
        f"- feature verdict: `{summary['STRICT_CLEAN_FEATURE_VERDICT']}`",
        f"- feature file: `{summary['features_path']}`",
        f"- unique feature candidates: `{summary['feature_unique_candidates']}`",
        "",
        "## Strict-Clean Primary Transfer Table",
        "",
        f"- random_top1_baseline: `{summary['random_top1_baseline_primary']}`",
        f"- best AntisymLinear: `{summary['best_antisymlinear']}`",
        f"- best NoNorm: `{summary['best_nonorm']}`",
        f"- best overall: `{summary['best_hh_trained']}`",
        "",
        "See `opi/taps/probes/code_strict_clean_transfer_2026-05-17.md` for the full per-config table.",
        "",
        "## Secondary Diagnostic Transfer Table",
        "",
        f"- random_top1_baseline: `{summary['random_top1_baseline_secondary']}`",
        f"- best secondary overall: `{summary['best_secondary_hh_trained']}`",
        "",
        "## Per-Task Breakdown",
        "",
    ]
    for row in summary["primary_per_task_breakdown_best"]:
        lines.append(
            f"- `{row['task_id']}` pred_label=`{row['predicted_label']}` "
            f"correct=`{row['predicted_is_correct']}` mode=`{row['predicted_mode']}`"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        summary["one_sentence_interpretation"],
        "",
        "Because n is small, this is a strict-clean code transfer signal check, not a broad code-transfer proof.",
        "",
        "## Docs Updated",
        "",
    ])
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    if not summary.get("docs_updated"):
        lines.append("- none appended; section already present or doc missing")
    lines.extend(["", "## Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary["files_modified_or_created"])
    lines.extend([
        "",
        "## Commands Run",
        "",
        "```bash",
        "venv/bin/python -m py_compile utilities/tests/manual/build_code_strict_clean_transfer_set.py",
        "venv/bin/python -m py_compile utilities/tests/manual/capture_code_strict_clean_transfer_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_hh_transfer_on_code_strict_clean.py",
        "venv/bin/python -u utilities/tests/manual/build_code_strict_clean_transfer_set.py",
        "venv/bin/python -u utilities/tests/manual/capture_code_strict_clean_transfer_features.py --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_code_strict_clean.py",
        "```",
        "",
        "## Blockers",
        "",
        summary.get("blockers") or "None.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    transfer_set_path = output_path(args.transfer_set)
    features_path = output_path(args.features)
    hh_path = output_path(args.hh_capture)
    out_json = output_path(args.output)
    out_md = output_path(args.output_md)
    if not transfer_set_path.exists():
        raise SystemExit(f"missing transfer set: {transfer_set_path}")
    if not features_path.exists():
        raise SystemExit(f"missing features: {features_path}")
    if not hh_path.exists():
        raise SystemExit("HH_CAPTURE_MISSING")

    transfer_set = load_json(transfer_set_path)
    set_verdict = str(transfer_set.get("strict_clean_transfer_set_verdict", "BLOCKED"))
    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    feature_meta = feature_payload.get("meta", {})
    feature_verdict = str(feature_meta.get("strict_clean_feature_verdict", "BLOCKED"))
    if feature_verdict != "READY":
        raise SystemExit("STRICT_CLEAN_FEATURE_VERDICT=BLOCKED")
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), args.heldout, args.seed)

    records_by_set = {
        "strict_clean_primary": build_records(feature_payload, "strict_clean_primary"),
        "strict_clean_plus_wrong_code": build_records(feature_payload, "strict_clean_plus_wrong_code"),
    }
    baselines = {name: random_top1_baseline(records) for name, records in records_by_set.items()}

    heads: list[tuple[str, str, Any, dict[str, Any]]] = []
    for config in MATH_CONFIGS:
        print(f"training HH heads {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        for architecture in ("AntisymLinear", "AntisymLinearNoNorm"):
            head, train_metrics = train_hh_head(architecture, config, chosen, rejected, train_idx, eval_idx, args, device)
            heads.append((config, architecture, head, train_metrics))

    eval_results = {}
    for set_name, records in records_by_set.items():
        eval_results[set_name] = evaluate_set(
            set_name=set_name,
            records=records,
            heads=heads,
            baseline=baselines[set_name],
            device=device,
        )

    primary = eval_results["strict_clean_primary"]
    secondary = eval_results["strict_clean_plus_wrong_code"]
    primary_best = primary["best_hh_trained"]
    verdict = transfer_verdict(primary_best, primary["random_top1_baseline"])
    transfer_survived = "yes" if verdict == "GOOD" else ("weak" if verdict == "WEAK" else "no")
    one_sentence = (
        "HH-trained tiny taps show a strict-clean code transfer signal on the screened correct-vs-near_miss micro-set."
        if verdict == "GOOD"
        else (
            "HH-trained tiny taps show a weak strict-clean code transfer signal on this small correct-vs-near_miss micro-set."
            if verdict == "WEAK"
            else "HH-trained tiny taps did not show a reliable strict-clean code transfer signal on this small micro-set."
        )
    )
    set_summary = transfer_set.get("summary", {})
    primary_label_counts = dict(Counter(
        cand["label"]
        for tournament in transfer_set.get("tournaments", [])
        for cand in tournament.get("strict_clean_primary_candidates", [])
    ))
    winner_metrics = primary_best["metrics"] if primary_best else {}
    summary = {
        "STRICT_CLEAN_TRANSFER_SET_VERDICT": set_verdict,
        "STRICT_CLEAN_FEATURE_VERDICT": feature_verdict,
        "STRICT_CLEAN_TRANSFER_VERDICT": verdict,
        "RECOMMENDED_NEXT": recommended_next(verdict, set_verdict),
        "tasks": set_summary.get("n_tasks", len(records_by_set["strict_clean_primary"])),
        "primary_candidates": set_summary.get("n_candidates_primary"),
        "secondary_candidates": set_summary.get("n_candidates_secondary"),
        "primary_label_counts": primary_label_counts,
        "secondary_wrong_code": set_summary.get("n_wrong_code_secondary"),
        "features_path": repo_path(features_path),
        "feature_unique_candidates": feature_meta.get("n_candidates_unique"),
        "random_top1_baseline_primary": primary["random_top1_baseline"],
        "random_top1_baseline_secondary": secondary["random_top1_baseline"],
        "best_antisymlinear": row_compact(primary.get("best_antisymlinear")),
        "best_nonorm": row_compact(primary.get("best_nonorm")),
        "best_hh_trained": row_compact(primary_best),
        "best_secondary_hh_trained": row_compact(secondary.get("best_hh_trained")),
        "winner_top1_pairwise_cycle": {
            "top1": winner_metrics.get("top1_tournament_acc"),
            "pairwise": winner_metrics.get("pairwise_acc"),
            "cycle": winner_metrics.get("cycle_rate"),
        },
        "strict_clean_transfer_survived": transfer_survived,
        "primary_per_task_breakdown_best": primary.get("per_task_breakdown_best", []),
        "one_sentence_interpretation": one_sentence,
        "blockers": "",
    }
    result_payload = {
        "strict_clean_transfer_verdict": verdict,
        "strict_clean_transfer_set_verdict": set_verdict,
        "strict_clean_feature_verdict": feature_verdict,
        "transfer_set": repo_path(transfer_set_path),
        "features_path": repo_path(features_path),
        "hh_capture": repo_path(hh_path),
        "feature_summary": feature_meta,
        "eval_results": eval_results,
        "summary": summary,
    }
    write_json(out_json, result_payload)
    write_transfer_md(out_md, result_payload)
    docs_updated = append_docs(summary)
    summary["docs_updated"] = docs_updated
    summary["files_modified_or_created"] = [
        "shared/utilities/tests/manual/build_code_strict_clean_transfer_set.py",
        "shared/utilities/tests/manual/capture_code_strict_clean_transfer_features.py",
        "shared/utilities/tests/manual/evaluate_hh_transfer_on_code_strict_clean.py",
        "opi/taps/probes/code_strict_clean_transfer_set_2026-05-17.json",
        "opi/taps/probes/code_strict_clean_transfer_set_2026-05-17.md",
        "opi/taps/probes/code_strict_clean_transfer_features_2026-05-17.pt",
        "opi/taps/probes/code_strict_clean_transfer_features_2026-05-17.md",
        repo_path(out_json),
        repo_path(out_md),
        repo_path(output_path(args.summary_output)),
        repo_path(output_path(args.summary_md)),
        *docs_updated,
    ]
    write_json(output_path(args.summary_output), summary)
    write_summary_md(output_path(args.summary_md), summary)
    print(f"STRICT_CLEAN_TRANSFER_VERDICT = {verdict}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    print(f"Wrote {args.summary_output}", flush=True)
    print(f"Wrote {args.summary_md}", flush=True)


if __name__ == "__main__":
    main()
