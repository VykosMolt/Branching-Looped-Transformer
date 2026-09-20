"""Optional old-context replay for Branch Generator v1 selector."""
from __future__ import annotations

import time
from pathlib import Path

from bg_branch_generator_v1_common import (
    OLD_CONTEXT_CSV,
    OLD_CONTEXT_JSON,
    OLD_CONTEXT_MD,
    SELECTOR_HEADS_PT,
    ensure_bgv1_root,
    load_json,
    load_pt,
    md_table,
    rel,
    write_csv,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    trained = load_pt(SELECTOR_HEADS_PT, {}) or {}
    heads = list(trained.get("heads") or [])
    compatible_new = [row for row in heads if row.get("variant") == "generator_v1_only_primary_safe" and row.get("flip_diagnostics", {}).get("passes")]
    if not compatible_new:
        verdict = "SKIPPED"
        rows = []
        reason = "no trained generator-v1 selector; v4 old-context replay remains the relevant diagnostic"
    else:
        # This stage is deliberately bounded and uses cached reports only. The
        # dedicated v4 replay already covered compatible old pools; this records
        # whether a new generator-v1 selector exists to justify another replay.
        v4_replay = Path("opi/taps/probes/bg_hidden_origin_quota_v4_2026-05-18/old_context_replay_v4.json")
        cached = load_json(v4_replay, {}) or {}
        verdict = "PARTIAL_MATCH" if cached.get("verdict") in {"PARTIAL_MATCH", "MATCHES_OLD_TAPS", "DIVERGES_BUT_USEFUL"} else "INSUFFICIENT"
        rows = list(cached.get("rows") or [])
        reason = "reused cached v4 old-context replay coverage; no new old candidates generated"
    payload = {
        "BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT": verdict,
        "verdict": verdict,
        "training_verdict": trained.get("verdict"),
        "compatible_generator_v1_heads": len(compatible_new),
        "rows": rows,
        "diagnostic_only": True,
        "production_routing_changed": False,
        "reason": reason,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OLD_CONTEXT_JSON, payload)
    write_csv(OLD_CONTEXT_CSV, rows[:1000] if isinstance(rows, list) else [])
    lines = [
        "# Branch Generator V1 Old-Context Replay",
        "",
        f"BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT = {verdict}",
        "",
        f"- compatible_generator_v1_heads: `{len(compatible_new)}`",
        f"- diagnostic_only: `True`",
        f"- production_routing_changed: `False`",
        f"- reason: `{reason}`",
        f"- json: `{rel(OLD_CONTEXT_JSON)}`",
    ]
    if rows:
        lines.extend(["", "## Cached Rows", ""])
        lines.extend(md_table(rows[:40], sorted({k for row in rows[:40] if isinstance(row, dict) for k in row.keys()})[:10]))
    write_md(OLD_CONTEXT_MD, lines)
    print(f"BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
