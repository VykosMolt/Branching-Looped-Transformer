"""Capture BG features for code-like wrapper candidates."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import torch

from wrapper_bg_matched_lib import (
    OUT_DIR,
    candidate_code,
    feature_input_text,
    load_json,
    load_task_suite,
    sha,
    trace_candidates,
    write_text,
)


TRACES_JSON = OUT_DIR / "candidate_traces.json"
EVAL_JSON = OUT_DIR / "candidate_eval.json"
OUT_PT = OUT_DIR / "wrapper_candidate_features.pt"
OUT_MD = OUT_DIR / "wrapper_candidate_features.md"
MAX_TOTAL_SECONDS = 60 * 60


def main() -> None:
    tasks = {task["task_id"]: task for task in load_task_suite()}
    traces = load_json(TRACES_JSON)
    eval_payload = load_json(EVAL_JSON)
    eval_by_task = eval_payload.get("by_task", {})

    work: list[dict[str, Any]] = []
    for trace_row in traces.get("candidate_traces", []):
        task_id = trace_row["task_id"]
        task = tasks.get(task_id)
        if task is None:
            continue
        trace_data = load_json(trace_row["trace_path"])
        for artifact in trace_candidates(trace_data):
            code, code_source = candidate_code(artifact)
            if not code:
                work.append({
                    "task_id": task_id,
                    "candidate_uid": artifact.get("candidate_uid"),
                    "status": "BG_FEATURE_SKIPPED_NOT_CODE",
                    "code_source": "",
                })
                continue
            text = feature_input_text(task["prompt"], code)
            work.append({
                "task_id": task_id,
                "candidate_uid": artifact.get("candidate_uid"),
                "stage": artifact.get("stage"),
                "code_source": code_source,
                "feature_text_hash": sha(text, 24),
                "feature_text": text,
                "status": "pending",
                "oracle_priority": bool(eval_by_task.get(task_id, {}).get("oracle_candidate_exists")),
            })

    pending = [row for row in work if row["status"] == "pending"]
    pending.sort(key=lambda row: (not row.get("oracle_priority"), row["task_id"], row["candidate_uid"]))
    started = time.perf_counter()
    feature_cache: dict[str, torch.Tensor] = {}
    features: dict[str, dict[str, torch.Tensor]] = {}
    metadata: list[dict[str, Any]] = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = BGTransformerFeatureExtractor(device=device, dtype="auto")
    try:
        for row in pending:
            if time.perf_counter() - started > MAX_TOTAL_SECONDS:
                row["status"] = "BG_FEATURE_SKIPPED_TIME_CAP"
                metadata.append({k: v for k, v in row.items() if k != "feature_text"})
                continue
            h = row["feature_text_hash"]
            try:
                if h not in feature_cache:
                    feature_cache[h] = extractor.encode_text_to_pooled_features(row["feature_text"], max_length=1536)
                features.setdefault(row["task_id"], {})[row["candidate_uid"]] = feature_cache[h]
                row["status"] = "READY"
                row["shape"] = list(feature_cache[h].shape)
            except Exception as exc:  # noqa: BLE001 - report
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"
            metadata.append({k: v for k, v in row.items() if k != "feature_text"})
    finally:
        if hasattr(extractor, "cleanup"):
            extractor.cleanup()

    for row in work:
        if row["status"] == "BG_FEATURE_SKIPPED_NOT_CODE":
            metadata.append(row)
    ready_count = sum(1 for row in metadata if row.get("status") == "READY")
    code_like_total = len(pending)
    ready_tasks = {row["task_id"] for row in metadata if row.get("status") == "READY"}
    useful_candidates = sum(1 for row in pending if row.get("status") == "READY")
    if code_like_total and ready_count == code_like_total:
        verdict = "READY"
    elif code_like_total and ready_count / max(code_like_total, 1) >= 0.80:
        verdict = "PARTIAL"
    elif ready_tasks:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    torch.save(
        {
            "WRAPPER_BG_FEATURE_VERDICT": verdict,
            "features": features,
            "metadata": metadata,
            "summary": {
                "device": device,
                "code_like_candidates": code_like_total,
                "features_ready": ready_count,
                "feature_tasks": len(ready_tasks),
                "cache_entries": len(feature_cache),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        },
        OUT_PT,
    )
    lines = [
        "# Wrapper Candidate BG Features",
        "",
        f"WRAPPER_BG_FEATURE_VERDICT = {verdict}",
        "",
        f"- device: `{device}`",
        f"- code-like candidates: `{code_like_total}`",
        f"- features ready: `{ready_count}`",
        f"- feature tasks: `{len(ready_tasks)}`",
        f"- status counts: `{dict(Counter(row.get('status') for row in metadata))}`",
        f"- output: `{OUT_PT}`",
    ]
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_BG_FEATURE_VERDICT = {verdict}")
    print(f"wrote {OUT_PT}")
    print(f"wrote {OUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
