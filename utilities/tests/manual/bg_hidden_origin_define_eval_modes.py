"""Define leakage-aware hidden-origin split-salvage evaluation modes."""
from __future__ import annotations

import time

from bg_hidden_origin_split_salvage_common import (
    AUDIT_JSON,
    EVAL_MODES_JSON,
    READINESS_MINIMUMS,
    SALVAGE_ROOT,
    WEAK_MINIMUMS,
    define_eval_modes_from_rows,
    ensure_salvage_root,
    load_json,
    md_table,
    primary_rows,
    rel,
    write_json,
    write_md,
)


OUT_MD = SALVAGE_ROOT / "eval_modes.md"


def verdict_for(records: list[dict[str, object]]) -> str:
    fixed = [r for r in records if r.get("mode_name") in {"strict_cross_version_clean", "v3_clean"}]
    strict = next((r for r in fixed if r.get("mode_name") == "strict_cross_version_clean"), None)
    v3 = next((r for r in fixed if r.get("mode_name") == "v3_clean"), None)
    grouped = [r for r in records if r.get("mode_name") == "grouped_kfold_v3"]
    if strict and (strict.get("heldout_support") or {}).get("readiness_support") and (strict.get("contamination_flags") or {}).get("v3_clean"):
        return "STRICT_READY"
    if v3 and (v3.get("heldout_support") or {}).get("readiness_support") and (v3.get("contamination_flags") or {}).get("v3_clean"):
        return "V3_CLEAN_READY"
    if grouped and sum(1 for r in grouped if (r.get("heldout_support") or {}).get("weak_support")) >= max(3, len(grouped) // 2):
        return "CV_READY"
    if any((r.get("heldout_support") or {}).get("weak_support") for r in records):
        return "WEAK_ONLY"
    return "NO_VALID_LARGER_SPLIT"


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    audit = load_json(AUDIT_JSON, {}) or {}
    if audit.get("verdict") == "BLOCKED":
        payload = {"BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "audit blocked"}
        write_json(EVAL_MODES_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Eval Modes", "", "BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = BLOCKED", flush=True)
        return 1
    rows = primary_rows()
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "no primary rows"}
        write_json(EVAL_MODES_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Eval Modes", "", "BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = BLOCKED", flush=True)
        return 1
    modes = define_eval_modes_from_rows(rows)
    records = list(modes["records"])
    verdict = verdict_for(records)
    payload = {
        "BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT": verdict,
        "verdict": verdict,
        "readiness_minimums": READINESS_MINIMUMS,
        "weak_minimums": WEAK_MINIMUMS,
        **modes,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(EVAL_MODES_JSON, payload)

    fixed_rows = []
    for record in records:
        if record["mode_type"] not in {"fixed", "eval_only"}:
            continue
        support = record["heldout_support"]
        flags = record["contamination_flags"]
        fixed_rows.append(
            {
                "mode": record["mode_name"],
                "fold": record["fold_id"],
                "support_tasks": support["support_task_count"],
                "behavior_groups": support["behaviorally_diverse_groups"],
                "non_tie_pairs": support["non_tie_pairs"],
                "readiness": support["readiness_support"],
                "weak": support["weak_support"],
                "v1_clean": flags["v1_clean"],
                "v2_clean": flags["v2_clean"],
                "v3_clean": flags["v3_clean"],
                "empirical_clean": flags["v3_empirical_direction_clean"],
            }
        )
    cv_rows = []
    for record in records:
        if record["mode_type"] not in {"grouped_kfold", "leave_one_task_out"}:
            continue
        support = record["heldout_support"]
        flags = record["contamination_flags"]
        cv_rows.append(
            {
                "mode": record["mode_name"],
                "fold": record["fold_id"],
                "support_tasks": support["support_task_count"],
                "behavior_groups": support["behaviorally_diverse_groups"],
                "non_tie_pairs": support["non_tie_pairs"],
                "weak": support["weak_support"],
                "v3_clean": flags["v3_clean"],
                "empirical_clean": flags["v3_empirical_direction_clean"],
            }
        )
    lines = [
        "# Hidden-Origin Split-Salvage Eval Modes",
        "",
        f"BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = {verdict}",
        "",
        "Readiness support requires heldout task IDs with non-tie pairs, behaviorally diverse groups, and non-tie pair counts. Grouped CV is task-disjoint for salvage retraining but is not automatically empirical-direction clean.",
        "",
        "## Fixed Modes",
        "",
    ]
    lines.extend(md_table(fixed_rows, ["mode", "fold", "support_tasks", "behavior_groups", "non_tie_pairs", "readiness", "weak", "v1_clean", "v2_clean", "v3_clean", "empirical_clean"]))
    lines.extend(["", "## CV/LOTO Modes", ""])
    lines.extend(md_table(cv_rows[:80], ["mode", "fold", "support_tasks", "behavior_groups", "non_tie_pairs", "weak", "v3_clean", "empirical_clean"]))
    lines.extend(
        [
            "",
            "## Task Pools",
            "",
            f"- all_task_ids: `{len(modes['all_task_ids'])}`",
            f"- behaviorally_diverse_task_ids: `{len(modes['behaviorally_diverse_task_ids'])}`",
            f"- reward_signal_task_ids: `{len(modes['reward_signal_task_ids'])}`",
            f"- grouped_kfold_folds: `{modes['fold_count']['grouped_kfold_v3']}`",
            f"- leave_one_task_out_folds: `{modes['fold_count']['leave_one_task_out_v3']}`",
            "",
            f"Wrote `{rel(EVAL_MODES_JSON)}`.",
        ]
    )
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(EVAL_MODES_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

