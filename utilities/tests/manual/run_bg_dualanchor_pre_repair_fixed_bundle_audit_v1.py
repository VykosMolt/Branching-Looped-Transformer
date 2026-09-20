"""DualAnchor pre-repair fixed-bundle audit v1.

Fast CSV-only audit over the existing pre-repair constrained replay rows. It
does not train, does not recompute feature scores, does not use the failed
branch-gap repair weights, and does not update any tap registries.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import json_default, safe_float
from run_bg_layer_native_two_tap_readiness_v1 import DOC_TARGETS, NAV_TARGETS, append_section, write_csv, write_json, write_md


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_pre_repair_fixed_bundle_audit_v1_2026-05-30"
REPORT_JSON = OUT_ROOT / "pre_repair_fixed_bundle_audit.json"
REPORT_MD = OUT_ROOT / "pre_repair_fixed_bundle_audit.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
BUNDLE_ROWS_CSV = OUT_ROOT / "bundle_audit_rows.csv"
SELECTED_ROWS_CSV = OUT_ROOT / "selected_bundle_rows.csv"
ARTIFACT_PT = OUT_ROOT / "dualanchor_pre_repair_fixed_bundle_audit_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md"

CONSTRAINED_PAIR_ROWS = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/constrained_pair_rows.csv"
CONSTRAINED_GROUP_ROWS = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30/constrained_group_rows.csv"


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return safe_float(row.get(key), default)


def eligible(row: dict[str, Any], count_key: str) -> bool:
    return str(row.get("readiness_eligible")) == "True" and int(float(row.get(count_key) or 0)) > 0


def main() -> int:
    ensure_root()
    pair_rows = read_csv(CONSTRAINED_PAIR_ROWS)
    group_rows = read_csv(CONSTRAINED_GROUP_ROWS)
    by_dataset = defaultdict(list)
    for row in pair_rows:
        by_dataset[row.get("dataset_name")].append(row)
    group_by_dataset = defaultdict(list)
    for row in group_rows:
        group_by_dataset[row.get("dataset_name")].append(row)

    bundle_names = sorted({r.get("candidate_name") for r in pair_rows if str(r.get("candidate_name", "")).startswith("bundle::two_tap_equal::")})
    audit_rows = []
    selected_detail_by_bundle: dict[str, list[dict[str, Any]]] = {}
    for bundle in bundle_names:
        details = []
        domain_failures = []
        branch_pair_failures = []
        branch_group_failures = []
        domain_deltas = []
        branch_deltas = []
        group_deltas = []
        for row in pair_rows:
            if row.get("candidate_name") != bundle or not eligible(row, "pair_count"):
                continue
            dataset = row.get("dataset_name")
            kind = row.get("dataset_kind")
            acc = f(row, "pairwise_accuracy")
            if kind == "old_domain_pair":
                refs = [r for r in by_dataset[dataset] if r.get("candidate_family") == "source_old_content"]
            else:
                refs = [r for r in by_dataset[dataset] if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
            best_ref = max((f(r, "pairwise_accuracy") for r in refs), default=float("nan"))
            delta = acc - best_ref if math.isfinite(best_ref) else float("nan")
            ok = bool(math.isfinite(delta) and delta + 1e-9 >= 0.0)
            detail = {
                "bundle_name": bundle,
                "dataset_name": dataset,
                "dataset_kind": kind,
                "count": int(float(row.get("pair_count") or 0)),
                "metric": "pairwise_accuracy",
                "bundle_value": acc,
                "best_reference_value": best_ref,
                "delta": delta,
                "ok": ok,
            }
            details.append(detail)
            if kind == "old_domain_pair":
                domain_deltas.append(delta)
                if not ok:
                    domain_failures.append(detail)
            else:
                branch_deltas.append(delta)
                if not ok:
                    branch_pair_failures.append(detail)
        for row in group_rows:
            policy = row.get("policy_name") or row.get("candidate_name")
            if policy != bundle or not eligible(row, "group_count"):
                continue
            dataset = row.get("dataset_name")
            ret = f(row, "oracle_retention")
            refs = [r for r in group_by_dataset[dataset] if r.get("candidate_family") in {"source_branch", "source_bridge", "source_universal"}]
            best_ref = max((f(r, "oracle_retention") for r in refs), default=float("nan"))
            delta = ret - best_ref if math.isfinite(best_ref) else float("nan")
            ok = bool(math.isfinite(delta) and delta + 1e-9 >= 0.0)
            detail = {
                "bundle_name": bundle,
                "dataset_name": dataset,
                "dataset_kind": "branch_group",
                "count": int(float(row.get("group_count") or 0)),
                "metric": "oracle_retention",
                "bundle_value": ret,
                "best_reference_value": best_ref,
                "delta": delta,
                "ok": ok,
            }
            details.append(detail)
            group_deltas.append(delta)
            if not ok:
                branch_group_failures.append(detail)
        domain_ok = not domain_failures and bool(domain_deltas)
        branch_ok = not branch_pair_failures and not branch_group_failures and bool(branch_deltas or group_deltas)
        audit = {
            "bundle_name": bundle,
            "domain_ok": domain_ok,
            "branch_ok": branch_ok,
            "ready": domain_ok and branch_ok,
            "domain_failure_count": len(domain_failures),
            "branch_pair_failure_count": len(branch_pair_failures),
            "branch_group_failure_count": len(branch_group_failures),
            "domain_min_delta": min(domain_deltas) if domain_deltas else float("nan"),
            "branch_pair_min_delta": min(branch_deltas) if branch_deltas else float("nan"),
            "branch_group_min_delta": min(group_deltas) if group_deltas else float("nan"),
            "domain_mean_delta": sum(domain_deltas) / len(domain_deltas) if domain_deltas else float("nan"),
            "branch_pair_mean_delta": sum(branch_deltas) / len(branch_deltas) if branch_deltas else float("nan"),
            "branch_group_mean_delta": sum(group_deltas) / len(group_deltas) if group_deltas else float("nan"),
        }
        audit_rows.append(audit)
        selected_detail_by_bundle[bundle] = details

    domain_ok = [r for r in audit_rows if r["domain_ok"]]
    ready = [r for r in audit_rows if r["ready"]]
    if ready:
        selected = max(ready, key=lambda r: (f(r, "branch_pair_mean_delta"), f(r, "branch_group_mean_delta"), f(r, "domain_mean_delta")))
        verdict = "DUALANCHOR_FIXED_BUNDLE_READY_DIAGNOSTIC"
    elif domain_ok:
        selected = min(
            domain_ok,
            key=lambda r: (
                r["branch_pair_failure_count"] + r["branch_group_failure_count"],
                -f(r, "branch_pair_mean_delta", -999.0),
                -f(r, "branch_group_mean_delta", -999.0),
            ),
        )
        verdict = "DUALANCHOR_FIXED_BUNDLE_DOMAIN_READY_BRANCH_GAP"
    else:
        selected = min(audit_rows, key=lambda r: (r["domain_failure_count"], r["branch_pair_failure_count"] + r["branch_group_failure_count"]))
        verdict = "DUALANCHOR_FIXED_BUNDLE_NOT_READY"

    selected_details = selected_detail_by_bundle.get(str(selected.get("bundle_name")), [])
    payload = {
        "BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT": verdict,
        "verdict": verdict,
        "selected_bundle": selected,
        "ready_bundle_count": len(ready),
        "domain_ok_bundle_count": len(domain_ok),
        "bundle_count": len(audit_rows),
        "anti_leakage": {
            "csv_only_existing_replay": True,
            "uses_failed_branch_gap_repair_weights": False,
            "uses_pre_repair_constrained_rows": True,
            "diagnostic_full_replay_selection": True,
            "no_training": True,
            "no_ouro_training": True,
            "no_registry_update": True,
            "no_action_steering": True,
        },
    }
    torch.save({"summary": payload, "selected_details": selected_details}, ARTIFACT_PT)
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(BUNDLE_ROWS_CSV, audit_rows)
    write_csv(SELECTED_ROWS_CSV, selected_details)

    selected_failures = [r for r in selected_details if not r.get("ok")]
    lines = [
        "# DualAnchor Pre-Repair Fixed-Bundle Audit v1",
        "",
        f"BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = {verdict}",
        "",
        f"- selected bundle: `{selected.get('bundle_name')}`",
        f"- ready fixed bundles: `{len(ready)}` / `{len(audit_rows)}`",
        f"- domain-ok fixed bundles: `{len(domain_ok)}` / `{len(audit_rows)}`",
        f"- selected domain ok: `{selected.get('domain_ok')}`",
        f"- selected branch ok: `{selected.get('branch_ok')}`",
        f"- selected domain failures: `{selected.get('domain_failure_count')}`",
        f"- selected branch pair failures: `{selected.get('branch_pair_failure_count')}`",
        f"- selected branch group failures: `{selected.get('branch_group_failure_count')}`",
        "",
        "This is a CSV-only diagnostic over existing pre-repair constrained replay rows. It does not use the failed repair weights.",
        "",
        "## Selected Bundle Failures",
        "",
    ]
    lines.extend(md_table(selected_failures, ["dataset_name", "dataset_kind", "count", "metric", "bundle_value", "best_reference_value", "delta"]))
    lines.extend(["", "## Ready Bundles", ""])
    lines.extend(md_table(ready[:20], ["bundle_name", "domain_ok", "branch_ok", "domain_min_delta", "branch_pair_min_delta", "branch_group_min_delta"]))
    lines.extend(["", "## Files", "", f"- artifact: `{rel(ARTIFACT_PT)}`", f"- bundle rows: `{rel(BUNDLE_ROWS_CSV)}`", f"- selected rows: `{rel(SELECTED_ROWS_CSV)}`"])
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    write_md(
        DOC_MD,
        [
            "# DualAnchor Pre-Repair Fixed-Bundle Audit v1",
            "",
            f"BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = {verdict}",
            "",
            "CSV-only audit of fixed `bundle::two_tap_equal::*` DualAnchor bundles from the pre-repair constrained layer-native replay. No training or failed-repair weights were used.",
            "",
            f"Report: `{rel(REPORT_MD)}`.",
        ],
    )
    section_title = "## DualAnchor pre-repair fixed-bundle audit v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = {verdict}`. Selected diagnostic fixed bundle `{selected.get('bundle_name')}`. Ready fixed bundles `{len(ready)}` / `{len(audit_rows)}`; domain-ok fixed bundles `{len(domain_ok)}` / `{len(audit_rows)}`. No training, failed-repair weights, old registry update, steering, or routing change.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [section_title, "", f"Added `{rel(DOC_MD)}`. Status: `{verdict}`."]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)
    print(f"BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = {verdict}", flush=True)
    print(f"selected_bundle = {selected.get('bundle_name')}", flush=True)
    print(f"ready_bundle_count = {len(ready)} domain_ok_bundle_count = {len(domain_ok)} bundle_count = {len(audit_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
