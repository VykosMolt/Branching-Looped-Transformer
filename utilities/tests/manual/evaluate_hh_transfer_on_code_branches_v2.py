"""Evaluate HH-trained AntisymLinear / NoNorm taps on v2 code branch tournaments."""
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
    from utilities.tests.manual.code_branch_pilot_lib import REPORT_DIR, PROJECT_ROOT, load_json, output_path, repo_path, write_json
except ModuleNotFoundError:
    from code_branch_pilot_lib import REPORT_DIR, PROJECT_ROOT, load_json, output_path, repo_path, write_json

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


DEFAULT_FEATURES = REPORT_DIR / "code_branch_tap_features_v2_2026-05-16.pt"
DEFAULT_HH = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_transfer_v2_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_transfer_v2_2026-05-16.md"
SUMMARY_JSON = REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.json"
SUMMARY_MD = REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--hh-capture", default=str(DEFAULT_HH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default="")
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


def best_by_arch(rows: Sequence[dict[str, Any]], arch: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["architecture"] == arch]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    )


def transfer_verdict(best: dict[str, Any] | None, baseline: float, n_tournaments: int, primary: str) -> str:
    required = {"strict_clean": 5, "diagnostic_runnable": 8, "diagnostic_mixed": 10}.get(primary, 1)
    if best is None or n_tournaments < required:
        return "NOT_RUN"
    m = best["metrics"]
    top1 = float(m["top1_tournament_acc"])
    pairwise = float(m["pairwise_acc"])
    cycles = float(m["cycle_rate"])
    if top1 >= baseline + 0.15 and pairwise >= 0.60 and cycles <= 0.05:
        return "GOOD"
    if top1 >= baseline + 0.05 or pairwise >= 0.55:
        return "WEAK"
    return "POOR"


def breakdown_metrics(matrices: Sequence[torch.Tensor], records: Sequence[dict[str, Any]], baseline: float, key: str) -> dict[str, Any]:
    out = {}
    for value in sorted({str(row.get(key, "unknown")) for row in records}):
        idx = [i for i, row in enumerate(records) if str(row.get(key, "unknown")) == value]
        if not idx:
            continue
        out[value] = evaluate_matrices([matrices[i] for i in idx], [records[i] for i in idx], baseline)
    return out


def prediction_stage_breakdown(matrices: Sequence[torch.Tensor], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stages = []
    correct = []
    difficulties = []
    for mat, row in zip(matrices, records):
        pred = int(torch.argmax(mat.sum(dim=1)).item())
        meta = row["candidate_metadata"][pred]
        stages.append(str(meta.get("candidate_stage", "unknown")))
        difficulties.append(str(meta.get("difficulty", row.get("difficulty", "unknown"))))
        correct.append(bool(meta.get("is_correct")))
    return {
        "predicted_stage_counts": dict(Counter(stages)),
        "predicted_correct_by_stage": {
            stage: {
                "n": sum(1 for s in stages if s == stage),
                "correct": sum(1 for s, ok in zip(stages, correct) if s == stage and ok),
            }
            for stage in sorted(set(stages))
        },
        "predicted_difficulty_counts": dict(Counter(difficulties)),
    }


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


def recommended_next(verdicts: dict[str, str]) -> str:
    if verdicts["CODE_V2_INTERFACE_VERDICT"] == "BLOCKED":
        return "fix_local_agent_invocation_before_code_pilot"
    if verdicts["CODE_V2_TASKSET_VERDICT"] == "BLOCKED":
        return "restore_or_add_harder_code_taskset"
    if verdicts["CODE_V2_GENERATION_VERDICT"] == "WRAPPER_BLOCKED":
        return "inspect_local_agent_wrapper_or_use_direct_route_only"
    if verdicts["CODE_V2_GENERATION_VERDICT"] == "TOO_DUPLICATE":
        return "reduce_final_outputs_and_harvest_pre_repair_candidates"
    if verdicts["CODE_V2_TOURNAMENT_VERDICT"] == "TOO_EASY":
        return "add_harder_tasks_and_disable_final_repair_candidates"
    if verdicts["CODE_V2_TOURNAMENT_VERDICT"] == "TOO_HARD":
        return "add_medium_tasks_or_allow_repaired_final_candidates"
    if verdicts["CODE_V2_TOURNAMENT_VERDICT"] == "TOO_FEW_TOURNAMENTS":
        return "increase_tasks_and_candidate_modes"
    if verdicts["CODE_V2_TRANSFER_VERDICT"] == "GOOD":
        return "expand_code_branch_pilot_v2_to_50_tournaments"
    if verdicts["CODE_V2_TRANSFER_VERDICT"] == "WEAK":
        return "expand_code_pilot_or_add_code_specific_training_control"
    if verdicts["CODE_V2_TRANSFER_VERDICT"] == "POOR":
        return "investigate_code_domain_mismatch_or_train_code_specific_taps"
    return "increase_tasks_and_candidate_modes"


def append_docs(summary: dict[str, Any]) -> list[str]:
    docs = [
        PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
        PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ]
    title = "## Code branch pilot v2 (2026-05-16)"
    best_a = summary.get("best_antisymlinear", "NA")
    best_n = summary.get("best_nonorm", "NA")
    best = summary.get("best_hh_trained", "NA")
    text = "\n".join([
        "",
        title,
        "",
        f"- CODE_V2_INTERFACE_VERDICT: `{summary['CODE_V2_INTERFACE_VERDICT']}`",
        f"- CODE_V2_TASKSET_VERDICT: `{summary['CODE_V2_TASKSET_VERDICT']}`",
        f"- CODE_V2_GENERATION_VERDICT: `{summary['CODE_V2_GENERATION_VERDICT']}`",
        f"- CODE_V2_TOURNAMENT_VERDICT: `{summary['CODE_V2_TOURNAMENT_VERDICT']}`",
        f"- CODE_V2_TRANSFER_VERDICT: `{summary['CODE_V2_TRANSFER_VERDICT']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- candidates: `{summary['candidates']}`",
        f"- duplicate rate: `{summary['duplicate_rate']}`",
        f"- unit-test label counts: `{summary['label_counts']}`",
        f"- strict_clean tournaments: `{summary['strict_clean_tournaments']}`",
        f"- diagnostic_mixed tournaments: `{summary['diagnostic_mixed_tournaments']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best AntisymLinear row: `{best_a}`",
        f"- best NoNorm row: `{best_n}`",
        f"- winner: `{best}`",
        f"- candidate-stage harvesting fixed v1 too-successful-wrapper problem: `{summary['stage_harvesting_interpretation']}`",
        "- full report: `opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.md`",
        f"- interpretation: {summary['interpretation']}",
        "",
    ])
    appended = []
    for path in docs:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def write_transfer_md(path: Path, payload: dict[str, Any]) -> None:
    verdict_key = "CODE_V2_MINI_TRANSFER_VERDICT" if "mini_patched" in str(path) else "CODE_V2_TRANSFER_VERDICT"
    lines = [
        "# Code Branch v2 HH-Trained Linear Transfer",
        "",
        f"{verdict_key} = {payload['code_v2_transfer_verdict']}",
        "",
        f"- primary_eval_set: `{payload['primary_eval_set']}`",
        f"- n_tournaments: `{payload['feature_summary']['n_tournaments']}`",
        f"- n_candidates: `{payload['feature_summary']['n_candidates']}`",
        f"- random_top1_baseline: `{payload['random_top1_baseline']:.3f}`",
        "",
        "## Transfer Table",
        "",
        "| config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["transfer_table"]:
        m = row["metrics"]
        lines.append(
            f"| `{row['config']}` | {row['architecture']} | {rate(m['top1_tournament_acc'])} | "
            f"{rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | "
            f"{rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} | "
            f"{rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
        )
    lines.extend(["", "## Best Rows", ""])
    for label, row in (
        ("best AntisymLinear", payload.get("best_antisymlinear")),
        ("best NoNorm", payload.get("best_nonorm")),
        ("best overall", payload.get("best_hh_trained")),
    ):
        if not row:
            lines.append(f"- {label}: `NA`")
            continue
        m = row["metrics"]
        lines.append(
            f"- {label}: `{row['config']}` / `{row['architecture']}` "
            f"top1={m['top1_tournament_acc']:.3f} pairwise={m['pairwise_acc']:.3f} cycle={m['cycle_rate']:.3f}"
        )
    lines.extend([
        "",
        "## Best Breakdown",
        "",
        f"- source_breakdown: `{payload.get('best_source_breakdown')}`",
        f"- difficulty_breakdown: `{payload.get('best_difficulty_breakdown')}`",
        f"- prediction_stage_breakdown: `{payload.get('best_prediction_stage_breakdown')}`",
        "",
        "## Relation To Expanded Clean GSM8K",
        "",
        "The expanded clean GSM8K comparison remains the prior positive branch-selection transfer signal: "
        "`EXPANDED_LINEAR_TRANSFER_VERDICT = GOOD`, `GRU_CONTROL_VERDICT = GRU_WEAK`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(summary: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Pilot v2 Summary",
        "",
        f"CODE_V2_INTERFACE_VERDICT = {summary['CODE_V2_INTERFACE_VERDICT']}",
        f"CODE_V2_TASKSET_VERDICT = {summary['CODE_V2_TASKSET_VERDICT']}",
        f"CODE_V2_GENERATION_VERDICT = {summary['CODE_V2_GENERATION_VERDICT']}",
        f"CODE_V2_TOURNAMENT_VERDICT = {summary['CODE_V2_TOURNAMENT_VERDICT']}",
        f"CODE_V2_TRANSFER_VERDICT = {summary['CODE_V2_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Interface v2 Summary",
        "",
        f"- interface report: `{summary['interface_report']}`",
        "",
        "## Taskset v2 Summary",
        "",
        f"- tasks: `{summary['tasks']}`",
        f"- source_mix: `{summary['source_mix']}`",
        f"- difficulty_mix: `{summary['difficulty_mix']}`",
        "",
        "## Candidate Generation Diversity Summary",
        "",
        f"- candidates: `{summary['candidates']}`",
        f"- duplicate_rate: `{summary['duplicate_rate']}`",
        f"- stage_breakdown: `{summary['stage_breakdown']}`",
        "",
        "## Deduplication Summary",
        "",
        f"- duplicate_rate: `{summary['duplicate_rate']}`",
        "",
        "## Unit-Test Label Summary",
        "",
        f"- label_counts: `{summary['label_counts']}`",
        f"- label_by_stage: `{summary['label_by_stage']}`",
        "",
        "## Tournament Construction Summary",
        "",
        f"- strict_clean tournaments: `{summary['strict_clean_tournaments']}`",
        f"- diagnostic_mixed tournaments: `{summary['diagnostic_mixed_tournaments']}`",
        f"- primary_eval_set: `{summary['primary_eval_set']}`",
        "",
        "## Feature Capture Summary",
        "",
        f"- feature file: `{summary['features_path']}`",
        f"- feature candidates: `{summary['feature_candidates']}`",
        "",
        "## HH-Trained AntisymLinear / NoNorm Transfer Table",
        "",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best AntisymLinear: `{summary['best_antisymlinear']}`",
        f"- best NoNorm: `{summary['best_nonorm']}`",
        f"- best overall: `{summary['best_hh_trained']}`",
        "",
        "## Best-Head Comparison",
        "",
        f"- winner_family: `{summary['winner_family']}`",
        f"- winner_layer_family: `{summary['winner_layer_family']}`",
        "",
        "## Relation To v1 Code Pilot",
        "",
        f"- v1 relation: `{summary['v1_relation']}`",
        "",
        "## Relation To Expanded Clean GSM8K Result",
        "",
        "Prior clean GSM8K expanded result: `EXPANDED_LINEAR_TRANSFER_VERDICT = GOOD`; `GRU_CONTROL_VERDICT = GRU_WEAK`.",
        "",
        "## Markdown Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    lines.extend(["", "## Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary.get("files_created", []))
    lines.extend(["", "## Commands Run", "", "```bash"])
    lines.extend(summary.get("commands_run", []))
    lines.extend(["```", "", "## Blockers", "", summary.get("blockers") or "None.", ""])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    write_json(SUMMARY_JSON, summary)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    features_path = output_path(args.features)
    hh_path = output_path(args.hh_capture)
    out_json = output_path(args.output)
    out_md = output_path(args.output_md)
    if not features_path.exists():
        raise SystemExit(f"missing features: {features_path}")
    if not hh_path.exists():
        raise SystemExit("HH_CAPTURE_MISSING")

    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    records = feature_payload["records"]
    meta = feature_payload.get("meta", {})
    input_json = output_path(meta.get("input_json", ""))
    tournaments_payload = load_json(input_json)
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), args.heldout, args.seed)
    baseline = random_top1_baseline(records)
    transfer_rows: list[dict[str, Any]] = []
    best_matrices: list[torch.Tensor] = []
    best = None
    for config in MATH_CONFIGS:
        print(f"training/evaluating {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        feats_by_record = config_features(records, config)
        for architecture in ("AntisymLinear", "AntisymLinearNoNorm"):
            head, train_metrics = train_hh_head(architecture, config, chosen, rejected, train_idx, eval_idx, args, device)
            head = head.to(device)
            matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
            metrics = evaluate_matrices(matrices, records, baseline)
            row = {
                "config": config,
                "architecture": architecture,
                "train_metrics": train_metrics,
                "metrics": metrics,
                "source_breakdown": breakdown_metrics(matrices, records, baseline, "source"),
                "difficulty_breakdown": breakdown_metrics(matrices, records, baseline, "difficulty"),
            }
            transfer_rows.append(row)
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
    verdict = transfer_verdict(best, baseline, len(records), str(meta.get("primary_eval_set")))
    best_antisym = best_by_arch(transfer_rows, "AntisymLinear")
    best_nonorm = best_by_arch(transfer_rows, "AntisymLinearNoNorm")
    nonorm_cycle_bug = [
        {"config": row["config"], "cycle_rate": row["metrics"]["cycle_rate"]}
        for row in transfer_rows
        if row["architecture"] == "AntisymLinearNoNorm" and float(row["metrics"]["cycle_rate"]) > 1e-9
    ]
    if nonorm_cycle_bug:
        raise SystemExit(f"AntisymLinearNoNorm nonzero cycle rate: {nonorm_cycle_bug}")
    result = {
        "code_v2_transfer_verdict": verdict,
        "code_v2_mini_transfer_verdict": verdict,
        "features_path": repo_path(features_path),
        "hh_capture": repo_path(hh_path),
        "tournaments_json": repo_path(input_json),
        "primary_eval_set": meta.get("primary_eval_set"),
        "feature_summary": meta,
        "random_top1_baseline": baseline,
        "transfer_table": transfer_rows,
        "best_hh_trained": best,
        "best_antisymlinear": best_antisym,
        "best_nonorm": best_nonorm,
        "best_source_breakdown": best.get("source_breakdown", {}) if best else {},
        "best_difficulty_breakdown": best.get("difficulty_breakdown", {}) if best else {},
        "best_prediction_stage_breakdown": prediction_stage_breakdown(best_matrices, records) if best_matrices else {},
        "commands_run": [
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py --features opi/taps/probes/code_branch_tap_features_v2_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --output opi/taps/probes/code_branch_transfer_v2_2026-05-16.json",
        ],
    }
    write_json(out_json, result)
    write_transfer_md(out_md, result)

    if "mini_patched" in str(out_json):
        print(f"CODE_V2_MINI_TRANSFER_VERDICT = {verdict}", flush=True)
        print(f"Wrote {out_json}", flush=True)
        print(f"Wrote {out_md}", flush=True)
        return

    interface = load_json(REPORT_DIR / "code_branch_v2_interface_inspection_2026-05-16.json")
    taskset = load_json(tournaments_payload.get("taskset", REPORT_DIR / "code_branch_taskset_v2_2026-05-16.json"))
    gen = load_json(tournaments_payload.get("candidates_json", REPORT_DIR / "code_branch_candidates_v2_2026-05-16.json"))
    tourn = tournaments_payload
    verdicts = {
        "CODE_V2_INTERFACE_VERDICT": interface.get("code_v2_interface_verdict", "BLOCKED"),
        "CODE_V2_TASKSET_VERDICT": taskset.get("code_v2_taskset_verdict", "BLOCKED"),
        "CODE_V2_GENERATION_VERDICT": gen.get("code_v2_generation_verdict", "WRAPPER_BLOCKED"),
        "CODE_V2_TOURNAMENT_VERDICT": tourn.get("code_v2_tournament_verdict", "TOO_FEW_TOURNAMENTS"),
        "CODE_V2_TRANSFER_VERDICT": verdict,
    }
    best_compact = row_compact(best)
    winner_family = best_compact["architecture"] if isinstance(best_compact, dict) else "NOT_RUN"
    winner_layer = str(best_compact["config"]).split("_", 1)[0] if isinstance(best_compact, dict) else "NOT_RUN"
    summary = {
        **verdicts,
        "RECOMMENDED_NEXT": recommended_next(verdicts),
        "interface_report": "opi/taps/probes/code_branch_v2_interface_inspection_2026-05-16.md",
        "tasks": len(taskset.get("tasks", [])),
        "source_mix": taskset.get("source_mix", {}),
        "difficulty_mix": taskset.get("difficulty_mix", {}),
        "candidates": tourn.get("summary", {}).get("candidates_evaluated"),
        "duplicate_rate": tourn.get("summary", {}).get("duplicate_rate"),
        "stage_breakdown": tourn.get("summary", {}).get("stage_breakdown", {}),
        "label_counts": {
            "correct": tourn.get("summary", {}).get("correct_candidates"),
            "near_miss": tourn.get("summary", {}).get("near_miss_candidates"),
            "wrong_code": tourn.get("summary", {}).get("wrong_code_candidates"),
            "runtime_error": tourn.get("summary", {}).get("runtime_error_candidates"),
            "malformed": tourn.get("summary", {}).get("malformed_candidates"),
            "legacy_nonsense": tourn.get("summary", {}).get("legacy_nonsense_candidates", tourn.get("summary", {}).get("nonsense_candidates")),
        },
        "label_by_stage": tourn.get("summary", {}).get("label_by_stage", {}),
        "strict_clean_tournaments": tourn.get("summary", {}).get("strict_clean_tournaments"),
        "diagnostic_mixed_tournaments": tourn.get("summary", {}).get("diagnostic_mixed_tournaments"),
        "primary_eval_set": meta.get("primary_eval_set"),
        "features_path": repo_path(features_path),
        "feature_candidates": meta.get("n_candidates"),
        "random_top1_baseline": baseline,
        "best_antisymlinear": row_compact(best_antisym),
        "best_nonorm": row_compact(best_nonorm),
        "best_hh_trained": best_compact,
        "winner_family": winner_family,
        "winner_layer_family": winner_layer,
        "stage_harvesting_interpretation": "yes" if (tourn.get("summary", {}).get("diagnostic_mixed_tournaments", 0) or 0) >= 12 else "partial",
        "interpretation": "The v2 code pilot used objective unit-test labels and candidate-stage harvesting; transfer remains a preliminary code-branch signal.",
        "v1_relation": "v2 increased task count, added hard/devil tasks, deduplicated by AST, and harvested direct-short / first-tool / repair-stage branches.",
        "files_created": [
            "shared/utilities/tests/manual/inspect_code_branch_v2_requirements.py",
            "shared/utilities/tests/manual/build_code_branch_taskset_v2.py",
            "shared/utilities/tests/manual/generate_code_branch_candidates_v2.py",
            "shared/utilities/tests/manual/evaluate_code_branch_candidates_v2.py",
            "shared/utilities/tests/manual/capture_code_branch_tap_features_v2.py",
            "shared/utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py",
            "opi/taps/probes/code_branch_v2_interface_inspection_2026-05-16.json",
            "opi/taps/probes/code_branch_v2_interface_inspection_2026-05-16.md",
            "opi/taps/probes/code_branch_taskset_v2_2026-05-16.json",
            "opi/taps/probes/code_branch_taskset_v2_2026-05-16.md",
            "opi/taps/probes/code_branch_candidates_v2_2026-05-16.json",
            "opi/taps/probes/code_branch_candidates_v2_2026-05-16.md",
            "opi/taps/probes/code_branch_candidates_v2_2026-05-16.log",
            "opi/taps/probes/code_branch_tournaments_v2_2026-05-16.json",
            "opi/taps/probes/code_branch_tournaments_v2_2026-05-16.md",
            "opi/taps/probes/code_branch_tap_features_v2_2026-05-16.pt",
            "opi/taps/probes/code_branch_tap_features_v2_2026-05-16.md",
            "opi/taps/probes/code_branch_transfer_v2_2026-05-16.json",
            "opi/taps/probes/code_branch_transfer_v2_2026-05-16.md",
            "opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.json",
            "opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.md",
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/inspect_code_branch_v2_requirements.py utilities/tests/manual/build_code_branch_taskset_v2.py utilities/tests/manual/generate_code_branch_candidates_v2.py utilities/tests/manual/evaluate_code_branch_candidates_v2.py utilities/tests/manual/capture_code_branch_tap_features_v2.py utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py",
            "venv/bin/python -u utilities/tests/manual/inspect_code_branch_v2_requirements.py",
            "venv/bin/python -u utilities/tests/manual/build_code_branch_taskset_v2.py --target-tasks 40 --min-tasks 25 --include-devil --max-easy-fraction 0.40 --output opi/taps/probes/code_branch_taskset_v2_2026-05-16.json",
            "venv/bin/python -u utilities/tests/manual/generate_code_branch_candidates_v2.py --taskset opi/taps/probes/code_branch_taskset_v2_2026-05-16.json --max-tasks 40 --min-tasks 25 --max-candidates-per-task 6 --hard-cap-total-candidates 300 --target-usable-tournaments 25 --device cuda",
            "venv/bin/python -u utilities/tests/manual/evaluate_code_branch_candidates_v2.py --candidates opi/taps/probes/code_branch_candidates_v2_2026-05-16.json --taskset opi/taps/probes/code_branch_taskset_v2_2026-05-16.json --output opi/taps/probes/code_branch_tournaments_v2_2026-05-16.json",
            "venv/bin/python -u utilities/tests/manual/capture_code_branch_tap_features_v2.py --input opi/taps/probes/code_branch_tournaments_v2_2026-05-16.json --output opi/taps/probes/code_branch_tap_features_v2_2026-05-16.pt --device cuda",
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py --features opi/taps/probes/code_branch_tap_features_v2_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --output opi/taps/probes/code_branch_transfer_v2_2026-05-16.json",
        ],
        "blockers": "",
    }
    summary["docs_updated"] = append_docs(summary)
    write_summary(summary)
    print(f"CODE_V2_TRANSFER_VERDICT = {verdict}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    print(f"Wrote {SUMMARY_JSON}", flush=True)
    print(f"Wrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
