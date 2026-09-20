"""Optional cached-candidate selection sanity check.

This deliberately does not count as hidden-origin branch evidence.
"""
from __future__ import annotations

import time
from collections import defaultdict

from bg_hidden_branch_suite_common import PROBE_ROOT, REPORT_ROOT, ensure_report_root, load_json, md_table, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "cached_branch_selection_sanity.json"
OUT_MD = REPORT_ROOT / "cached_branch_selection_sanity.md"
PREFIX_SCORES = PROBE_ROOT / "bg_trajectory_prediction_2026-05-18/prefix_scores.json"
PREDICTIVE_POWER = PROBE_ROOT / "bg_trajectory_prediction_2026-05-18/predictive_power.json"


def main() -> int:
    ensure_report_root()
    started = time.time()
    predictive = load_json(PREDICTIVE_POWER, {})
    scores = load_json(PREFIX_SCORES, {})
    if not predictive and not scores:
        verdict = "SKIPPED"
        payload = {
            "BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT": verdict,
            "reason": "cached trajectory selection artifacts not present",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    else:
        best = predictive.get("best_cells") or predictive.get("summary", {}).get("best_cells") or []
        rows = predictive.get("rows") or predictive.get("predictive_rows") or []
        if best or rows:
            verdict = "CACHED_SELECTION_SIGNAL_GOOD"
        else:
            verdict = "INSUFFICIENT"
        payload = {
            "BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT": verdict,
            "interpretation": "Useful offline sanity only; does not prove hidden-origin branch viability.",
            "predictive_power_file": rel(PREDICTIVE_POWER),
            "prefix_scores_file": rel(PREFIX_SCORES),
            "best_cells_preview": best[:5] if isinstance(best, list) else best,
            "score_row_count": len(scores.get("prefix_scores") or []),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    write_json(OUT_JSON, payload)
    lines = [
        "# Cached Branch Selection Sanity",
        "",
        f"BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT = {payload['BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT']}",
        "",
        "This is an offline cached candidate/prefix sanity check only. It is explicitly not evidence of same-prefix hidden-state branch persistence.",
    ]
    if "score_row_count" in payload:
        lines.append(f"- score_row_count: `{payload['score_row_count']}`")
    write_md(OUT_MD, lines)
    print(f"BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT = {payload['BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
