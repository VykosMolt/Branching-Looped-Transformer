"""Build split-specific pairwise datasets for hidden-origin split salvage."""
from __future__ import annotations

import time
from collections import Counter

import torch

from bg_hidden_origin_split_salvage_common import (
    EVAL_MODES_JSON,
    SALVAGE_DATASETS_PT,
    SALVAGE_ROOT,
    build_pairs_for_record,
    compact_pair,
    diagnostic_alpha_rows,
    ensure_salvage_root,
    load_json,
    md_table,
    primary_rows,
    rel,
    sampled_expected_rows,
    write_json,
    write_md,
)
from bg_hidden_origin_diversity_v3_common import diagnostic_alpha_v3_row, primary_safe_v3_row, sampled_reward


OUT_JSON = SALVAGE_ROOT / "salvage_datasets.json"
OUT_MD = SALVAGE_ROOT / "salvage_datasets.md"


def verdict_for(records: list[dict[str, object]], total_pairs: int) -> str:
    if total_pairs <= 0:
        return "BLOCKED"
    if any((r.get("heldout_support") or {}).get("readiness_support") for r in records):
        return "READY"
    if any((r.get("heldout_support") or {}).get("weak_support") for r in records):
        return "WEAK"
    return "DATA_LIMITED"


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    modes_payload = load_json(EVAL_MODES_JSON, {}) or {}
    if modes_payload.get("verdict") == "BLOCKED":
        payload = {"BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "eval modes blocked"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Salvage Datasets", "", "BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = BLOCKED", flush=True)
        return 1
    records = list(modes_payload.get("records") or [])
    rows = primary_rows()
    alpha_rows = diagnostic_alpha_rows()
    sampled_rows = sampled_expected_rows()
    if not rows or not records:
        payload = {"BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing rows or modes"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Salvage Datasets", "", "BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = BLOCKED", flush=True)
        return 1

    modes: dict[str, dict[str, object]] = {}
    total_primary_pairs = 0
    for record in records:
        mode_key = f"{record['mode_name']}::{record['fold_id']}"
        primary_pairs, primary_meta = build_pairs_for_record(
            rows,
            record,
            variant="primary_safe_deterministic",
            label_source="deterministic",
            row_filter=primary_safe_v3_row,
        )
        alpha_pairs, alpha_meta = build_pairs_for_record(
            alpha_rows,
            record,
            variant="diagnostic_alpha_0_02",
            label_source="deterministic",
            row_filter=diagnostic_alpha_v3_row,
        )
        sampled_pairs, sampled_meta = build_pairs_for_record(
            sampled_rows,
            record,
            variant="diagnostic_sampled_expected",
            label_source="sampled_expected",
            row_filter=lambda row: sampled_reward(row) is not None,
        )
        total_primary_pairs += len(primary_pairs)
        modes[mode_key] = {
            "record": record,
            "pairs": primary_pairs,
            "pairs_by_variant": {
                "primary_safe_deterministic": primary_pairs,
                "diagnostic_alpha_0_02": alpha_pairs,
                "diagnostic_sampled_expected": sampled_pairs,
            },
            "variant_meta": {
                "primary_safe_deterministic": primary_meta,
                "diagnostic_alpha_0_02": alpha_meta,
                "diagnostic_sampled_expected": sampled_meta,
            },
        }

    verdict = verdict_for(records, total_primary_pairs)
    payload = {
        "BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "eval_mode_verdict": modes_payload.get("verdict"),
        "modes": modes,
        "readiness_minimums": modes_payload.get("readiness_minimums"),
        "weak_minimums": modes_payload.get("weak_minimums"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, SALVAGE_DATASETS_PT)

    json_modes = {}
    for key, item in modes.items():
        json_modes[key] = {
            "record": item["record"],
            "variant_meta": item["variant_meta"],
            "pairs": [compact_pair(pair) for pair in item["pairs"]],
            "pairs_by_variant": {
                variant: [compact_pair(pair) for pair in pairs]
                for variant, pairs in (item["pairs_by_variant"] or {}).items()
            },
        }
    json_payload = {k: v for k, v in payload.items() if k != "modes"} | {"modes": json_modes}
    write_json(OUT_JSON, json_payload)

    rows_md = []
    for key, item in modes.items():
        record = item["record"]
        primary_meta = item["variant_meta"]["primary_safe_deterministic"]
        split_counts = Counter(pair["split"] for pair in item["pairs"])
        support = record["heldout_support"]
        rows_md.append(
            {
                "mode": record["mode_name"],
                "fold": record["fold_id"],
                "pairs_train": split_counts.get("train", 0),
                "pairs_val": split_counts.get("val", 0),
                "pairs_test": split_counts.get("test", 0),
                "support_tasks": support["support_task_count"],
                "behavior_groups": support["behaviorally_diverse_groups"],
                "non_tie_pairs": support["non_tie_pairs"],
                "tie_rate": f"{primary_meta['tie_rate']:.3f}",
                "readiness": support["readiness_support"],
            }
        )
    lines = [
        "# Hidden-Origin Salvage Datasets",
        "",
        f"BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = {verdict}",
        "",
        "All primary datasets omit reward ties and remain task-disjoint by mode/fold. Diagnostic alpha-0.02 and sampled-expected variants are stored separately.",
        "",
        "## Mode/Fold Pair Counts",
        "",
    ]
    lines.extend(md_table(rows_md[:120], ["mode", "fold", "pairs_train", "pairs_val", "pairs_test", "support_tasks", "behavior_groups", "non_tie_pairs", "tie_rate", "readiness"]))
    lines.extend(["", f"Wrote `{rel(SALVAGE_DATASETS_PT)}`."])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(SALVAGE_DATASETS_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

