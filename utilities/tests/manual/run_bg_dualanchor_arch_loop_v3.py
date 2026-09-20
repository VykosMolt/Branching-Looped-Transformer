from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

import run_bg_dualanchor_architecture_looped_stratified_probe_v2 as v2
from bg_dualanchor_arch_loop_v3_common import (
    L47_ABLATION_CSV,
    LINEAGE_JSONL,
    OUT_ROOT,
    RECOVERY_CSV,
    ROWS_CSV,
    ROWS_JSON,
    ROWS_PT,
    RUN_JSON,
    RUN_MD,
    STAGE_JSONL,
    STAGE_ROWS_CSV,
    TASK_ROWS_CSV,
    TERMINAL_POLICY_CSV,
    ensure_root,
    write_json,
)


def _add_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def _patch_v2_paths() -> None:
    ensure_root()
    v2.OUT_ROOT = OUT_ROOT
    v2.REPORT_JSON = RUN_JSON
    v2.REPORT_MD = RUN_MD
    v2.SUMMARY_JSON = OUT_ROOT / "run_summary.json"
    v2.SUMMARY_MD = OUT_ROOT / "run_summary.md"
    v2.ARTIFACT_PT = ROWS_PT
    v2.PARTIAL_PT = OUT_ROOT / "architecture_looped_rows.partial.pt"
    v2.STATE_JSON = OUT_ROOT / "state.json"
    v2.ROWS_CSV = ROWS_CSV
    v2.STAGE_ROWS_CSV = STAGE_ROWS_CSV
    v2.TASK_ROWS_CSV = TASK_ROWS_CSV
    v2.TERMINAL_POLICY_CSV = TERMINAL_POLICY_CSV
    v2.L47_ABLATION_CSV = L47_ABLATION_CSV
    v2.RECOVERY_CSV = RECOVERY_CSV


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _postprocess_aliases() -> None:
    if ROWS_CSV.exists():
        with ROWS_CSV.open(newline="") as f:
            compact_rows = list(csv.DictReader(f))
        write_json(ROWS_JSON, compact_rows)
        _write_jsonl(LINEAGE_JSONL, compact_rows)
    if STAGE_ROWS_CSV.exists():
        with STAGE_ROWS_CSV.open(newline="") as f:
            stage_rows = list(csv.DictReader(f))
        _write_jsonl(STAGE_JSONL, stage_rows)
    if ROWS_PT.exists():
        data = torch.load(ROWS_PT, map_location="cpu", weights_only=False)
        summary = dict(data.get("summary") or {})
        summary["BG_DUALANCHOR_ARCH_LOOP_V3_RUN_VERDICT"] = (
            "READY" if str(summary.get("status", "")).startswith("ARCHITECTURE_LOOPED_SURVIVAL_READY") else "PARTIAL"
        )
        summary["v3_short_name"] = "dualanchor_architecture_looped_stratified_probe_v3"
        summary["primary_interpretation"] = "confidence-gated terminal top1 remains primary unless hard slices prove unconditional collapse."
        write_json(RUN_JSON, summary)
        if RUN_MD.exists():
            text = RUN_MD.read_text()
            text = text.replace("DualAnchor Architecture-Looped Stratified Probe v2", "DualAnchor Architecture-Looped Stratified Probe v3")
            text = text.replace("stratified probe v2", "stratified probe v3")
            text = text.replace("This scales the architecture-shaped cumulative-hook probe", "This scales and stress-tests the architecture-shaped cumulative-hook probe")
            RUN_MD.write_text(text)


def main() -> int:
    _add_default("--max-tasks", "48")
    _add_default("--children-per-parent", "1")
    _add_default("--max-new-tokens", "32")
    _add_default("--split-mode", "all")
    _patch_v2_paths()
    code = v2.main()
    _postprocess_aliases()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

