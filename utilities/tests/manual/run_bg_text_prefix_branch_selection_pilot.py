"""Text-prefix BG branch selection pilot.

This reuses ordinary text prefixes from the shared branch pool. It does not
fork hidden states, splice KV caches, or implement latent beam search.
"""
from __future__ import annotations

import json
import time

from bg_steering_suite_lib import REPORT_ROOT, load_json, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "text_prefix_branch_selection_results.json"
OUT_MD = REPORT_ROOT / "text_prefix_branch_selection_results.md"


def main() -> int:
    started = time.time()
    partial = load_json(REPORT_ROOT / "partial_routing_results.json", {})
    if not partial or partial.get("BG_PARTIAL_ROUTING_VERDICT") == "INSUFFICIENT":
        verdict = "INSUFFICIENT"
        payload = {
            "BG_LATENT_BRANCH_SELECTION_VERDICT": verdict,
            "verdict": verdict,
            "branching_type": "TEXT_PREFIX_BRANCHING_ONLY",
            "skipped_reason": "partial routing results unavailable or insufficient",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    else:
        rows = [r for r in partial.get("task_results", []) if not r.get("is_devil")][:8]
        n = len(rows)
        rand1 = sum(float(r["policy_success"]["random_top1_expected"]) for r in rows) / max(n, 1)
        bg1 = sum(bool(r["policy_success"]["bg_top1_conservative"]) for r in rows) / max(n, 1)
        rand2 = sum(float(r["policy_success"]["random_top2_expected"]) for r in rows) / max(n, 1)
        bg2 = sum(bool(r["policy_success"]["bg_top2_conservative"]) for r in rows) / max(n, 1)
        oracle = sum(bool(r["policy_success"]["oracle_continue_all"]) for r in rows) / max(n, 1)
        delta = max(bg1 - rand1, bg2 - rand2)
        if n < 6 or oracle < 0.10:
            verdict = "INSUFFICIENT"
        elif delta >= 0.05:
            verdict = "HELPS"
        elif delta < -0.05:
            verdict = "HURTS"
        else:
            verdict = "NEUTRAL"
        payload = {
            "BG_LATENT_BRANCH_SELECTION_VERDICT": verdict,
            "verdict": verdict,
            "branching_type": "TEXT_PREFIX_BRANCHING_ONLY",
            "metrics": {
                "evaluable_tasks": n,
                "random_top1_success": rand1,
                "bg_top1_success": bg1,
                "random_top2_success": rand2,
                "bg_top2_success": bg2,
                "oracle_top1_success": oracle,
                "bg_lift": delta,
                "oracle_gap": oracle - max(bg1, bg2),
            },
            "task_ids": [r["task_id"] for r in rows],
            "elapsed_seconds": round(time.time() - started, 3),
        }
    write_json(OUT_JSON, payload)
    write_md(
        OUT_MD,
        [
            "# BG Text-Prefix Branch Selection Pilot (2026-05-18)",
            "",
            f"BG_LATENT_BRANCH_SELECTION_VERDICT = {payload['BG_LATENT_BRANCH_SELECTION_VERDICT']}",
            "",
            "- branching type: `TEXT_PREFIX_BRANCHING_ONLY`",
            "- no hidden-state mutation, KV-cache splicing, or latent forking was used.",
            f"- metrics: `{payload.get('metrics')}`",
        ],
    )
    print(f"BG_LATENT_BRANCH_SELECTION_VERDICT = {payload['BG_LATENT_BRANCH_SELECTION_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
