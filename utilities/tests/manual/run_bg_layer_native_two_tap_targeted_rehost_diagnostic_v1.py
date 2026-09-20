"""Targeted layer-native two-tap rehost diagnostic v1.

This is a diagnostic, not a clean readiness-bearing run. It takes the best
specialist reference directions from the remaining branch failures and rehosts
those exact layer-native weight vectors under the two allowed tap identities:
`MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL`.

The purpose is to separate mechanical coordinate compatibility from genuine
generalization. If this passes, it means the two-identity/layer-native carrier
can represent the missing branch signals. It does not prove the selected
directions are free of evaluation-artifact coupling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import json_default, normed, safe_float
from run_bg_layer_native_two_tap_readiness_v1 import (
    ANCHOR_GROUPS,
    DOC_TARGETS,
    LAYER_CONFIGS,
    NAV_TARGETS,
    PRIMARY_ARCHES,
    append_section,
    branch_pair_datasets,
    bundle_specs,
    eval_group_dataset,
    group_datasets_any,
    old_domain_pair_datasets,
    pair_eval_rows_for_dataset,
    readiness,
    source_slug,
    summarize_group_rows,
    summarize_pair_dataset,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)
from run_bg_two_tap_full_readiness_v1 import source_candidates


OUT_ROOT = PROBE_ROOT / "bg_layer_native_two_tap_targeted_rehost_diagnostic_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "targeted_rehost_diagnostic.json"
REPORT_MD = OUT_ROOT / "targeted_rehost_diagnostic.md"
PAIR_ROWS_CSV = OUT_ROOT / "targeted_rehost_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "targeted_rehost_group_rows.csv"
ARTIFACT_PT = OUT_ROOT / "targeted_rehost_diagnostic_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md"
CONSTRAINED_SUMMARY_JSON = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/summary.json"
CONSTRAINED_ARTIFACT_PT = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/layer_native_two_tap_constrained_train_v1.pt"


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_constrained_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not CONSTRAINED_ARTIFACT_PT.exists():
        return [], []
    payload = torch.load(CONSTRAINED_ARTIFACT_PT, map_location="cpu", weights_only=False)
    return list(payload.get("candidates") or []), list(payload.get("references") or [])


def target_reference_names(summary: dict[str, Any]) -> set[str]:
    failures = summary.get("readiness", {}).get("branch_pair_failures", [])
    return {str(row.get("best_reference")) for row in failures if row.get("best_reference")}


def build_targeted_rehosts(refs: Sequence[dict[str, Any]], target_names: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ref in refs:
        name = str(ref.get("candidate_name"))
        if name not in target_names:
            continue
        if ref.get("target_config") not in LAYER_CONFIGS or ref.get("architecture") not in PRIMARY_ARCHES:
            continue
        weight = tensor_weight(ref)
        if not isinstance(weight, torch.Tensor):
            continue
        for group in ANCHOR_GROUPS:
            w = normed(weight)
            out.append(
                {
                    "candidate_name": f"targeted_rehost::{group}::{source_slug(ref)}::{ref.get('target_config')}::{ref.get('architecture')}",
                    "candidate_family": "layer_native_two_tap_targeted_rehost",
                    "tap_role": group,
                    "recipe": "targeted_rehost_eval_reference",
                    "target_config": ref.get("target_config"),
                    "architecture": ref.get("architecture"),
                    "weight": w,
                    "state_dict": {"linear.weight": w.reshape(1, -1)},
                    "target_reference": name,
                    "target_reference_family": ref.get("candidate_family"),
                    "diagnostic_only": True,
                }
            )
    return out


def main() -> int:
    ensure_root()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    constrained_summary = load_json(CONSTRAINED_SUMMARY_JSON, {})
    target_names = target_reference_names(constrained_summary)
    base_candidates, refs_from_artifact = load_constrained_candidates()
    refs = refs_from_artifact or [
        r
        for r in source_candidates()
        if r.get("target_config") in LAYER_CONFIGS and r.get("architecture") in PRIMARY_ARCHES
    ]
    targeted = build_targeted_rehosts(refs, target_names)
    candidates = base_candidates + targeted
    bundles = bundle_specs(candidates)

    pair_rows: list[dict[str, Any]] = []
    for dataset in old_domain_pair_datasets() + branch_pair_datasets():
        pair_rows.extend(pair_eval_rows_for_dataset(dataset, candidates, refs, bundles))
    domain_summary = summarize_pair_dataset(pair_rows, "old_domain_pair")
    branch_summary = summarize_pair_dataset(pair_rows, "branch_pair")

    bounded_refs = []
    for fam in ("source_branch", "source_bridge", "source_universal"):
        rows = [
            r
            for r in refs
            if r.get("candidate_family") == fam
            and r.get("target_config") in LAYER_CONFIGS
            and r.get("architecture") in PRIMARY_ARCHES
        ]
        bounded_refs.extend(sorted(rows, key=lambda r: safe_float(r.get("metric_score"), -1.0), reverse=True)[:24])
    group_rows: list[dict[str, Any]] = []
    for dataset in group_datasets_any():
        group_rows.extend(eval_group_dataset(dataset, candidates, bounded_refs, bundles))
    group_summary = summarize_group_rows(group_rows)
    readiness_verdict, ready = readiness(domain_summary, branch_summary, group_summary)
    if readiness_verdict == "LAYER_NATIVE_TWO_TAP_READY":
        verdict = "TARGETED_REHOST_DIAGNOSTIC_PASSES"
    else:
        verdict = "TARGETED_REHOST_DIAGNOSTIC_STILL_GAPS"
    payload = {
        "BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT": verdict,
        "BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT_WITH_TARGETED_REHOST": readiness_verdict,
        "verdict": verdict,
        "readiness_verdict": readiness_verdict,
        "diagnostic_only": True,
        "target_reference_names": sorted(target_names),
        "targeted_rehost_count": len(targeted),
        "base_candidate_count": len(base_candidates),
        "domain_summary": domain_summary,
        "branch_pair_summary": branch_summary,
        "branch_group_summary": group_summary,
        "readiness": ready,
        "anti_leakage_caveat": {
            "uses_best_references_from_prior_failure_report": True,
            "readiness_bearing": False,
            "purpose": "mechanical upper bound for two-identity layer-native rehosting",
        },
    }
    torch.save({"summary": payload, "targeted_rehosts": targeted, "candidates": candidates, "references": refs}, ARTIFACT_PT)
    write_json(REPORT_JSON, payload)
    write_csv(PAIR_ROWS_CSV, pair_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)

    lines = [
        "# Layer-Native Two-Tap Targeted Rehost Diagnostic v1",
        "",
        f"BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = {verdict}",
        f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT_WITH_TARGETED_REHOST = {readiness_verdict}",
        "",
        "This is diagnostic only. It rehosts the exact specialist reference directions from the remaining branch failures under the two allowed tap identities. It answers mechanical representability, not clean generalization.",
        "",
        f"- targeted rehost candidates: `{len(targeted)}`",
        f"- domain ok: `{ready['domain_ok']}`",
        f"- branch ok: `{ready['branch_ok']}`",
        "",
        "## Branch Pair Summary",
        "",
    ]
    lines.extend(md_table(branch_summary.get("datasets", []), ["dataset_name", "readiness_eligible", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap", "best_reference_family"]))
    lines.extend(["", "## Domain Summary", ""])
    lines.extend(md_table(domain_summary.get("datasets", []), ["dataset_name", "pair_count", "best_two_tap_accuracy", "best_reference_accuracy", "delta_two_minus_reference", "matches_or_exceeds_reference", "best_two_tap"]))
    lines.extend(["", "## Files", "", f"- report: `{rel(REPORT_JSON)}`", f"- artifact: `{rel(ARTIFACT_PT)}`"])
    write_md(REPORT_MD, lines)
    write_md(
        DOC_MD,
        [
            "# Layer-Native Two-Tap Targeted Rehost Diagnostic v1",
            "",
            f"BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = {verdict}",
            f"BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT_WITH_TARGETED_REHOST = {readiness_verdict}",
            "",
            "Diagnostic only: exact specialist reference directions from the remaining branch failures were rehosted under `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` to test mechanical coordinate compatibility. This should not be treated as artifact-robust readiness.",
            "",
            f"Report: `{rel(REPORT_MD)}`.",
        ],
    )
    section_title = "## Layer-native two-tap targeted rehost diagnostic v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = {verdict}`. Diagnostic-only exact source rehost under the two tap identities produced readiness verdict `{readiness_verdict}`. This is not clean readiness because target references came from prior branch failure reports.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [section_title, "", f"Added `{rel(DOC_MD)}`. Status: `{verdict}`."]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)
    print(f"BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = {verdict}", flush=True)
    print(f"readiness_with_targeted_rehost = {readiness_verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
