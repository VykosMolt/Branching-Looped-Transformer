"""Capture read-only BG features for every trajectory prefix checkpoint."""
from __future__ import annotations

import time
import traceback
from collections import Counter

import torch

from bg_trajectory_prediction_lib import (
    REPORT_ROOT,
    domain_hint_for_task,
    iter_prefix_rows,
    load_continued,
    load_partials,
    load_task_suite,
    prefix_key,
    rel,
    task_by_id,
    write_json,
    write_md,
)


OUT_PT = REPORT_ROOT / "prefix_features.pt"
OUT_INDEX = REPORT_ROOT / "prefix_features_index.json"
OUT_MD = REPORT_ROOT / "prefix_features.md"


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    by_task = task_by_id(tasks)
    partials = load_partials()
    continued = load_continued()
    evaluable_keys = {
        (str(row["task_id"]), int(row["branch_id"]), int(row["prefix_length"]))
        for row in continued.get("continued_prefixes") or []
        if row.get("evaluable", True)
    }
    target_prefixes = [
        row
        for row in iter_prefix_rows(partials)
        if (str(row["task_id"]), int(row["branch_id"]), int(row["prefix_length"])) in evaluable_keys
    ]
    records = []
    errors = []
    if OUT_PT.exists():
        try:
            old = torch.load(OUT_PT, map_location="cpu", weights_only=False)
            records = list(old.get("records") or [])
        except Exception:
            records = []
    done = {
        (str(row.get("task_id")), int(row.get("branch_id", -1)), int(row.get("prefix_length", -1)))
        for row in records
    }

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
        for idx, prefix in enumerate(target_prefixes):
            key = (str(prefix["task_id"]), int(prefix["branch_id"]), int(prefix["prefix_length"]))
            if key in done:
                continue
            task = by_task.get(str(prefix["task_id"]))
            if task is None:
                continue
            try:
                features = extractor.encode_prompt_candidate(
                    str(task.get("prompt") or task.get("question") or ""),
                    str(prefix["prefix_text"]),
                    domain_hint=domain_hint_for_task(task),
                    max_length=1536,
                )
                records.append(
                    {
                        "feature_key": prefix_key(str(prefix["task_id"]), int(prefix["branch_id"]), int(prefix["prefix_length"])),
                        "task_id": str(prefix["task_id"]),
                        "domain": task["domain"],
                        "branch_id": int(prefix["branch_id"]),
                        "prefix_length": int(prefix["prefix_length"]),
                        "shape": list(features.shape),
                        "features": features,
                    }
                )
                done.add(key)
            except Exception as exc:
                errors.append(
                    {
                        "task_id": prefix.get("task_id"),
                        "branch_id": prefix.get("branch_id"),
                        "prefix_length": prefix.get("prefix_length"),
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                )
            if idx % 50 == 0 and records:
                torch.save({"complete": False, "records": records, "errors": errors}, OUT_PT)
    finally:
        if extractor is not None:
            extractor.cleanup()

    captured = len(records)
    target_count = len(target_prefixes)
    rate = captured / max(target_count, 1)
    if rate >= 0.999 and captured:
        verdict = "READY"
    elif rate >= 0.80:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_TRAJECTORY_PREFIX_FEATURE_VERDICT": verdict,
        "verdict": verdict,
        "complete": True,
        "target_prefix_count": target_count,
        "captured_count": captured,
        "capture_rate": rate,
        "feature_file": rel(OUT_PT),
        "errors": errors,
        "counts_by_domain": dict(Counter(row["domain"] for row in records)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save({**payload, "records": records}, OUT_PT)
    write_json(OUT_INDEX, payload)
    lines = [
        "# BG Trajectory Prefix Features (2026-05-18)",
        "",
        f"BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = {verdict}",
        "",
        f"- feature_file: `{rel(OUT_PT)}`",
        f"- target_prefix_count: `{target_count}`",
        f"- captured_count: `{captured}`",
        f"- capture_rate: `{rate:.3f}`",
        f"- counts_by_domain: `{payload['counts_by_domain']}`",
    ]
    if errors:
        lines.extend(["", "## Errors", "", *[f"- `{e['task_id']}` b{e['branch_id']} p{e['prefix_length']}: {e['error'][:200]}" for e in errors[:10]]])
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_PT)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
