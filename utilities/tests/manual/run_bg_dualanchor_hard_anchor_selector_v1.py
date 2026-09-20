"""DualAnchor hard-anchor selector v1.

Select a fixed pre-repair DualAnchor bundle from existing layer-native
artifacts using validation data only, with old-domain preservation as a hard
gate and branch performance as the secondary objective. This does not train
Ouro, train taps, update registries, use the failed branch-gap repair weights,
run wrapper/local-agent code, apply steering, or change routing.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import finite_mean, json_default, pair_diff, safe_float, score_diff
from run_bg_layer_native_two_tap_readiness_v1 import (
    DOC_TARGETS,
    LAYER_CONFIGS,
    NAV_TARGETS,
    append_section,
    branch_pair_datasets,
    bundle_specs,
    old_domain_pair_datasets,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_hard_anchor_selector_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "hard_anchor_selector.json"
REPORT_MD = OUT_ROOT / "hard_anchor_selector.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
VAL_ROWS_CSV = OUT_ROOT / "validation_bundle_rows.csv"
FULL_ROWS_CSV = OUT_ROOT / "full_replay_selected_rows.csv"
DIAGNOSTIC_ROWS_CSV = OUT_ROOT / "diagnostic_bundle_readiness_rows.csv"
ARTIFACT_PT = OUT_ROOT / "dualanchor_hard_anchor_selector_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_dualanchor_hard_anchor_selector_v1.md"

CONSTRAINED_ARTIFACT_PT = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/layer_native_two_tap_constrained_train_v1.pt"
CONSTRAINED_PAIR_ROWS = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/constrained_pair_rows.csv"
CONSTRAINED_GROUP_ROWS = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/constrained_group_rows.csv"

OLD_VAL_MIN_ACC = 0.70
OLD_VAL_MEAN_ACC = 0.78
OLD_CODE_VAL_ACC = 0.80
NO_SELECT_BRANCH_DATASET_TOKENS = (
    "hidden_origin_v3::high_yield_recipe_subset",
    "salvage::old_frozen_tap_clean::all_signal_tasks::primary_safe_deterministic",
)


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_constrained() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(CONSTRAINED_ARTIFACT_PT, map_location="cpu", weights_only=False)
    candidates = list(payload.get("candidates") or [])
    refs = list(payload.get("references") or [])
    bundles = [b for b in bundle_specs(candidates) if str(b.get("bundle_name", "")).startswith("bundle::two_tap_equal::")]
    return candidates, refs, bundles


def selected_pairs(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    for pair in dataset.get("pairs") or []:
        if str(pair.get("split") or "all") != split:
            continue
        rows.append(pair)
    return rows


def normalize(vals: Sequence[float]) -> list[float]:
    xs = [safe_float(v, 0.0) for v in vals]
    if not xs:
        return []
    mu = mean(xs)
    sd = math.sqrt(mean((x - mu) ** 2 for x in xs))
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]


def scores_for_candidate(candidate: dict[str, Any], pairs: Sequence[dict[str, Any]]) -> list[float]:
    weight = tensor_weight(candidate)
    if not isinstance(weight, torch.Tensor):
        return []
    config = str(candidate.get("target_config"))
    arch = str(candidate.get("architecture"))
    out = []
    for pair in pairs:
        diff = pair_diff(pair, config)
        if diff is None or int(diff.numel()) != int(weight.numel()):
            continue
        out.append(score_diff(weight, arch, diff))
    return out


def scores_for_bundle(bundle: dict[str, Any], by_name: dict[str, dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[float]:
    member_scores = []
    for name in bundle.get("members") or []:
        member = by_name.get(str(name))
        if member is None:
            continue
        scores = scores_for_candidate(member, pairs)
        if scores:
            member_scores.append(normalize(scores))
    if not member_scores:
        return []
    n = min(len(scores) for scores in member_scores)
    return [finite_mean(scores[i] for scores in member_scores) for i in range(n)]


def accuracy(scores: Sequence[float]) -> float:
    if not scores:
        return float("nan")
    return float(sum(1.0 if s > 0 else 0.5 if s == 0 else 0.0 for s in scores) / len(scores))


def validation_datasets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old = []
    for dataset in old_domain_pair_datasets():
        pairs = selected_pairs(dataset, "val")
        if pairs:
            old.append({**dataset, "pairs": pairs})
    branch = []
    for dataset in branch_pair_datasets():
        name = str(dataset.get("dataset_name") or "")
        if not dataset.get("readiness_eligible", True):
            continue
        if any(tok in name for tok in NO_SELECT_BRANCH_DATASET_TOKENS):
            continue
        pairs = selected_pairs(dataset, "val")
        if pairs:
            branch.append({**dataset, "pairs": pairs})
    return old, branch


def evaluate_validation(bundles: Sequence[dict[str, Any]], by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    old_datasets, branch_datasets = validation_datasets()
    rows = []
    for bundle in bundles:
        old_accs = []
        old_code_acc = float("nan")
        for dataset in old_datasets:
            acc = accuracy(scores_for_bundle(bundle, by_name, dataset.get("pairs") or []))
            old_accs.append(acc)
            if str(dataset.get("dataset_name")) == "old_code_pairs":
                old_code_acc = acc
        branch_accs = []
        gap_accs = []
        for dataset in branch_datasets:
            acc = accuracy(scores_for_bundle(bundle, by_name, dataset.get("pairs") or []))
            branch_accs.append(acc)
            name = str(dataset.get("dataset_name") or "")
            if any(tok in name for tok in ("branch_generator_v1", "hidden_origin_v3", "salvage", "universal_bridge")):
                gap_accs.append(acc)
        old_accs = [x for x in old_accs if math.isfinite(x)]
        branch_accs = [x for x in branch_accs if math.isfinite(x)]
        gap_accs = [x for x in gap_accs if math.isfinite(x)]
        old_min = min(old_accs) if old_accs else float("nan")
        old_mean = finite_mean(old_accs)
        branch_mean = finite_mean(branch_accs)
        gap_mean = finite_mean(gap_accs)
        old_pass = bool(
            math.isfinite(old_min)
            and math.isfinite(old_mean)
            and old_min >= OLD_VAL_MIN_ACC
            and old_mean >= OLD_VAL_MEAN_ACC
            and (not math.isfinite(old_code_acc) or old_code_acc >= OLD_CODE_VAL_ACC)
        )
        score = branch_mean + 0.5 * gap_mean + (0.25 * old_mean)
        if not old_pass:
            score -= 10.0
        rows.append(
            {
                "bundle_name": bundle.get("bundle_name"),
                "recipe": bundle.get("recipe"),
                "architecture": bundle.get("architecture"),
                "old_val_min_acc": old_min,
                "old_val_mean_acc": old_mean,
                "old_code_val_acc": old_code_acc,
                "branch_val_mean_acc": branch_mean,
                "gap_val_mean_acc": gap_mean,
                "old_gate_pass": old_pass,
                "selection_score": score,
            }
        )
    return rows


def row_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return safe_float(row.get(key), default)


def summarize_full_for_bundle(bundle_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows = [r for r in read_csv(CONSTRAINED_PAIR_ROWS) if r.get("candidate_name") == bundle_name]
    group_rows = [r for r in read_csv(CONSTRAINED_GROUP_ROWS) if (r.get("policy_name") == bundle_name or r.get("candidate_name") == bundle_name)]
    selected_rows: list[dict[str, Any]] = []
    domain_failures = []
    branch_failures = []
    group_failures = []
    for row in pair_rows:
        kind = row.get("dataset_kind")
        if str(row.get("readiness_eligible")) != "True":
            continue
        if int(float(row.get("pair_count") or 0)) <= 0:
            continue
        acc = row_float(row, "pairwise_accuracy")
        # Get reference from all rows for same dataset.
        selected_rows.append(row)
        if kind == "old_domain_pair":
            pass
    all_pair_rows = read_csv(CONSTRAINED_PAIR_ROWS)
    by_dataset = defaultdict(list)
    for row in all_pair_rows:
        by_dataset[row.get("dataset_name")].append(row)
    for row in pair_rows:
        if str(row.get("readiness_eligible")) != "True" or int(float(row.get("pair_count") or 0)) <= 0:
            continue
        dataset = row.get("dataset_name")
        kind = row.get("dataset_kind")
        acc = row_float(row, "pairwise_accuracy")
        if kind == "old_domain_pair":
            refs = [r for r in by_dataset[dataset] if r.get("candidate_family") == "source_old_content"]
        else:
            refs = [r for r in by_dataset[dataset] if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
        best_ref = max((row_float(r, "pairwise_accuracy") for r in refs), default=float("nan"))
        out = {
            "dataset_name": dataset,
            "dataset_kind": kind,
            "pair_count": int(float(row.get("pair_count") or 0)),
            "bundle_accuracy": acc,
            "best_reference_accuracy": best_ref,
            "delta_bundle_minus_reference": acc - best_ref if math.isfinite(best_ref) else float("nan"),
            "matches_or_exceeds_reference": bool(math.isfinite(best_ref) and acc + 1e-9 >= best_ref),
        }
        selected_rows.append(out)
        if not out["matches_or_exceeds_reference"]:
            if kind == "old_domain_pair":
                domain_failures.append(out)
            else:
                branch_failures.append(out)
    all_group_rows = read_csv(CONSTRAINED_GROUP_ROWS)
    group_by_dataset = defaultdict(list)
    for row in all_group_rows:
        group_by_dataset[row.get("dataset_name")].append(row)
    for row in group_rows:
        if str(row.get("readiness_eligible")) != "True" or int(float(row.get("group_count") or 0)) <= 0:
            continue
        dataset = row.get("dataset_name")
        ret = row_float(row, "oracle_retention")
        refs = [r for r in group_by_dataset[dataset] if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
        best_ref = max((row_float(r, "oracle_retention") for r in refs), default=float("nan"))
        out = {
            "dataset_name": dataset,
            "dataset_kind": "branch_group",
            "group_count": int(float(row.get("group_count") or 0)),
            "bundle_retention": ret,
            "bundle_false_prune": row_float(row, "false_prune_rate"),
            "best_reference_retention": best_ref,
            "delta_bundle_minus_reference": ret - best_ref if math.isfinite(best_ref) else float("nan"),
            "matches_or_exceeds_reference": bool(math.isfinite(best_ref) and ret + 1e-9 >= best_ref),
        }
        selected_rows.append(out)
        if not out["matches_or_exceeds_reference"]:
            group_failures.append(out)
    summary = {
        "domain_ok": not domain_failures,
        "branch_pair_ok": not branch_failures,
        "branch_group_ok": not group_failures,
        "branch_ok": not branch_failures and not group_failures,
        "domain_failures": domain_failures,
        "branch_pair_failures": branch_failures,
        "branch_group_failures": group_failures,
    }
    return selected_rows, summary


def diagnostic_bundle_readiness() -> list[dict[str, Any]]:
    rows = read_csv(CONSTRAINED_PAIR_ROWS)
    group_rows = read_csv(CONSTRAINED_GROUP_ROWS)
    bundle_names = sorted({r.get("candidate_name") for r in rows if str(r.get("candidate_name", "")).startswith("bundle::two_tap_equal::")})
    out = []
    for name in bundle_names:
        _, summary = summarize_full_for_bundle(str(name))
        out.append(
            {
                "bundle_name": name,
                "domain_ok": summary["domain_ok"],
                "branch_ok": summary["branch_ok"],
                "domain_failure_count": len(summary["domain_failures"]),
                "branch_pair_failure_count": len(summary["branch_pair_failures"]),
                "branch_group_failure_count": len(summary["branch_group_failures"]),
            }
        )
    return out


def main() -> int:
    ensure_root()
    candidates, refs, bundles = load_constrained()
    by_name = {str(c.get("candidate_name")): c for c in candidates}
    val_rows = evaluate_validation(bundles, by_name)
    passing = [r for r in val_rows if r.get("old_gate_pass")]
    selected_row = max(passing or val_rows, key=lambda r: row_float(r, "selection_score", -1e9), default={})
    selected_bundle = next((b for b in bundles if b.get("bundle_name") == selected_row.get("bundle_name")), {})
    full_rows, full_summary = summarize_full_for_bundle(str(selected_row.get("bundle_name")))
    diagnostic_rows = diagnostic_bundle_readiness()
    diagnostic_ready = [r for r in diagnostic_rows if r.get("domain_ok") and r.get("branch_ok")]
    if selected_row.get("old_gate_pass") and full_summary["domain_ok"] and full_summary["branch_ok"]:
        verdict = "DUALANCHOR_HARD_ANCHOR_READY"
    elif selected_row.get("old_gate_pass") and full_summary["domain_ok"]:
        verdict = "DUALANCHOR_DOMAIN_PRESERVED_BRANCH_GAP"
    elif selected_row.get("old_gate_pass"):
        verdict = "DUALANCHOR_VAL_ANCHOR_PASS_FULL_DOMAIN_GAP"
    elif diagnostic_ready:
        verdict = "DUALANCHOR_DIAGNOSTIC_READY_ONLY"
    else:
        verdict = "DUALANCHOR_HARD_ANCHOR_NOT_READY"
    payload = {
        "BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT": verdict,
        "verdict": verdict,
        "selected_bundle": selected_row.get("bundle_name"),
        "selected_validation": selected_row,
        "full_replay_summary": full_summary,
        "diagnostic_ready_bundle_count": len(diagnostic_ready),
        "diagnostic_ready_bundles": diagnostic_ready[:20],
        "candidate_counts": {
            "candidates": len(candidates),
            "references": len(refs),
            "two_tap_equal_bundles": len(bundles),
            "validation_old_gate_pass": len(passing),
        },
        "anti_leakage": {
            "uses_failed_branch_gap_repair_weights": False,
            "uses_pre_repair_constrained_artifact": True,
            "validation_selected_only": True,
            "full_replay_diagnostic_for_selected": True,
            "diagnostic_ready_bundle_search_uses_full_replay": True,
            "no_training": True,
            "no_ouro_training": True,
            "no_registry_update": True,
            "no_action_steering": True,
        },
    }
    torch.save({"summary": payload, "selected_bundle": selected_bundle, "validation_rows": val_rows}, ARTIFACT_PT)
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(VAL_ROWS_CSV, val_rows)
    write_csv(FULL_ROWS_CSV, full_rows)
    write_csv(DIAGNOSTIC_ROWS_CSV, diagnostic_rows)
    lines = [
        "# DualAnchor Hard-Anchor Selector v1",
        "",
        f"BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = {verdict}",
        "",
        f"- selected bundle: `{selected_row.get('bundle_name')}`",
        f"- validation old gate pass: `{selected_row.get('old_gate_pass')}`",
        f"- old val min/mean/code: `{selected_row.get('old_val_min_acc')}` / `{selected_row.get('old_val_mean_acc')}` / `{selected_row.get('old_code_val_acc')}`",
        f"- branch val mean / gap mean: `{selected_row.get('branch_val_mean_acc')}` / `{selected_row.get('gap_val_mean_acc')}`",
        f"- full domain ok: `{full_summary['domain_ok']}`",
        f"- full branch ok: `{full_summary['branch_ok']}`",
        f"- diagnostic full-suite ready bundles: `{len(diagnostic_ready)}`",
        "",
        "This selector uses only pre-repair DualAnchor candidates from the constrained layer-native artifact. It does not use the failed branch-gap repair weights.",
        "",
        "## Full Replay Domain Failures",
        "",
    ]
    lines.extend(md_table(full_summary["domain_failures"], ["dataset_name", "pair_count", "bundle_accuracy", "best_reference_accuracy", "delta_bundle_minus_reference"]))
    lines.extend(["", "## Full Replay Branch Pair Failures", ""])
    lines.extend(md_table(full_summary["branch_pair_failures"], ["dataset_name", "pair_count", "bundle_accuracy", "best_reference_accuracy", "delta_bundle_minus_reference"]))
    lines.extend(["", "## Full Replay Branch Group Failures", ""])
    lines.extend(md_table(full_summary["branch_group_failures"], ["dataset_name", "group_count", "bundle_retention", "best_reference_retention", "delta_bundle_minus_reference"]))
    lines.extend(["", "## Diagnostic Ready Bundles", ""])
    lines.extend(md_table(diagnostic_ready[:20], ["bundle_name", "domain_ok", "branch_ok", "domain_failure_count", "branch_pair_failure_count", "branch_group_failure_count"]))
    lines.extend(["", "## Files", "", f"- artifact: `{rel(ARTIFACT_PT)}`", f"- validation rows: `{rel(VAL_ROWS_CSV)}`", f"- full replay rows: `{rel(FULL_ROWS_CSV)}`"])
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# DualAnchor Hard-Anchor Selector v1",
            "",
            f"BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = {verdict}",
            "",
            "This run selected a fixed DualAnchor bundle from the pre-repair constrained layer-native artifact using validation data only, with old-domain preservation as a hard gate. It did not train new taps or use the failed branch-gap repair weights.",
            "",
            f"Report: `{rel(REPORT_MD)}`.",
        ],
    )
    section_title = "## DualAnchor hard-anchor selector v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = {verdict}`. Selected fixed pre-repair DualAnchor bundle `{selected_row.get('bundle_name')}` with validation old-anchor gate `{selected_row.get('old_gate_pass')}`. Full replay domain ok `{full_summary['domain_ok']}`; branch ok `{full_summary['branch_ok']}`. No training, failed-repair weights, old registry update, steering, or routing change.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [section_title, "", f"Added `{rel(DOC_MD)}`. Status: `{verdict}`."]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)
    print(f"BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = {verdict}", flush=True)
    print(f"selected_bundle = {selected_row.get('bundle_name')}", flush=True)
    print(f"full_domain_ok = {full_summary['domain_ok']} full_branch_ok = {full_summary['branch_ok']}", flush=True)
    print(f"diagnostic_ready_bundle_count = {len(diagnostic_ready)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
