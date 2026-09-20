"""Capture BG pooled features for partial branch pools."""
from __future__ import annotations

import time
import traceback
from collections import Counter

import torch

from bg_steering_suite_lib import (
    REPORT_ROOT,
    branch_rows_for_task,
    domain_hint_for_task,
    load_branch_pools,
    load_task_suite,
    rel,
    task_by_id,
    write_json,
    write_md,
)


OUT_PT = REPORT_ROOT / "partial_branch_features.pt"
OUT_INDEX = REPORT_ROOT / "partial_branch_features_index.json"
OUT_MD = REPORT_ROOT / "partial_branch_features.md"


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    by_task = task_by_id(tasks)
    branch_payload = load_branch_pools()
    records = []
    errors = []
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
        for branch in branch_payload.get("branches") or []:
            task = by_task.get(str(branch.get("task_id")))
            if task is None or not branch.get("initial_partial_text"):
                continue
            try:
                features = extractor.encode_prompt_candidate(
                    task_generation_text(task),
                    str(branch["initial_partial_text"]),
                    domain_hint=domain_hint_for_task(task),
                    max_length=1536,
                )
                records.append(
                    {
                        "task_id": str(branch["task_id"]),
                        "domain": task["domain"],
                        "branch_id": int(branch["branch_id"]),
                        "shape": list(features.shape),
                        "features": features,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "task_id": branch.get("task_id"),
                        "branch_id": branch.get("branch_id"),
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                )
    finally:
        if extractor is not None:
            extractor.cleanup()

    usable = [row for row in branch_payload.get("branches") or [] if row.get("initial_partial_text") and not row.get("generation_error")]
    rate = len(records) / max(len(usable), 1)
    if rate >= 0.999 and records:
        verdict = "READY"
    elif rate >= 0.80:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    records_sorted = sorted(records, key=lambda r: (r["task_id"], r["branch_id"]))
    torch_payload = {
        "BG_PARTIAL_FEATURE_VERDICT": verdict,
        "records": records_sorted,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_payload, OUT_PT)
    index = {
        "BG_PARTIAL_FEATURE_VERDICT": verdict,
        "verdict": verdict,
        "feature_file": rel(OUT_PT),
        "captured_count": len(records),
        "usable_branch_count": len(usable),
        "capture_rate": rate,
        "errors": errors,
        "counts_by_domain": dict(Counter(row["domain"] for row in records)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_INDEX, index)
    lines = [
        "# BG Partial Branch Features (2026-05-18)",
        "",
        f"BG_PARTIAL_FEATURE_VERDICT = {verdict}",
        "",
        f"- feature file: `{rel(OUT_PT)}`",
        f"- captured_count: `{len(records)}`",
        f"- usable_branch_count: `{len(usable)}`",
        f"- capture_rate: `{rate:.3f}`",
        f"- counts_by_domain: `{index['counts_by_domain']}`",
    ]
    if errors:
        lines.extend(["", "## Errors", *[f"- `{e['task_id']}` branch `{e['branch_id']}`: {e['error'][:200]}" for e in errors[:10]]])
    write_md(OUT_MD, lines)
    print(f"BG_PARTIAL_FEATURE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_PT)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


def task_generation_text(task: dict) -> str:
    return str(task.get("prompt") or task.get("question") or "")


if __name__ == "__main__":
    raise SystemExit(main())
