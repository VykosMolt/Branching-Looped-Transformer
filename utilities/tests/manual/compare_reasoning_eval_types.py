"""Compare generated-answer, natural-distractor, and generated-trace reasoning evaluations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, repo_path, write_json


GENERATED_TRANSFER = REPORT_DIR / "reasoning_branch_transfer_2026-05-17.json"
DISTRACTOR_TRANSFER = REPORT_DIR / "reasoning_natural_distractor_transfer_2026-05-17.json"
TRACE_TRANSFER = REPORT_DIR / "reasoning_trace_transfer_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "reasoning_eval_type_comparison_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_eval_type_comparison_2026-05-17.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def row_metric(row: Any, name: str) -> float:
    if not isinstance(row, dict):
        return float("nan")
    try:
        return float(row.get(name, float("nan")))
    except Exception:
        return float("nan")


def best_metrics(payload: dict[str, Any]) -> dict[str, float]:
    best = payload.get("summary", {}).get("best_overall", {})
    return {"top1": row_metric(best, "top1"), "pairwise": row_metric(best, "pairwise"), "cycle": row_metric(best, "cycle")}


def near_ceiling(m: dict[str, float]) -> bool:
    return m["top1"] >= 0.95 and m["pairwise"] >= 0.95


def verdict(generated: dict[str, Any], distractor: dict[str, Any], trace: dict[str, Any]) -> str:
    gm = best_metrics(generated)
    dm = best_metrics(distractor)
    tm = best_metrics(trace)
    if any(math.isnan(m["pairwise"]) or math.isnan(m["top1"]) for m in (gm, dm, tm)):
        return "INSUFFICIENT"
    if near_ceiling(gm) and near_ceiling(dm) and near_ceiling(tm):
        return "ALL_EASY"
    if dm["top1"] - tm["top1"] >= 0.10 or dm["pairwise"] - tm["pairwise"] >= 0.10:
        return "TRACE_HARDER"
    if tm["top1"] - dm["top1"] >= 0.10 or tm["pairwise"] - dm["pairwise"] >= 0.10:
        return "DISTRACTORS_HARDER"
    if tm["top1"] - gm["top1"] >= 0.10 or tm["pairwise"] - gm["pairwise"] >= 0.10 or dm["top1"] - gm["top1"] >= 0.10 or dm["pairwise"] - gm["pairwise"] >= 0.10:
        return "GENERATED_HARDER"
    return "INSUFFICIENT"


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    s = payload.get("summary", {})
    return {
        "n_tournaments": s.get("n_tournaments"),
        "random_top1_baseline": s.get("random_top1_baseline"),
        "best_hh": s.get("best_hh"),
        "best_code": s.get("best_code"),
        "best_nonorm": s.get("best_nonorm"),
        "best_antisymlinear": s.get("best_antisymlinear"),
        "best_overall": s.get("best_overall"),
        "verdict": s.get("REASONING_TRANSFER_VERDICT")
        or s.get("REASONING_DISTRACTOR_TRANSFER_VERDICT")
        or s.get("REASONING_TRACE_TRANSFER_VERDICT"),
        "specialist_verdict": s.get("REASONING_SPECIALIST_VERDICT") or s.get("REASONING_TRACE_SPECIALIST_VERDICT"),
    }


def interpretation(v: str) -> str:
    return {
        "TRACE_HARDER": "Generated reasoning traces were harder than natural answer distractors by the best-row comparison.",
        "DISTRACTORS_HARDER": "Natural answer distractors remain harder than generated reasoning traces.",
        "GENERATED_HARDER": "The first generated reasoning branch pilot remains the hardest setting by best-row comparison.",
        "ALL_EASY": "All three reasoning settings are near ceiling and do not yet separate good from great readouts.",
        "INSUFFICIENT": "At least one reasoning setting is missing or the differences are not decisive.",
    }[v]


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Evaluation Type Comparison",
        "",
        f"REASONING_TRACE_DIFFICULTY_VERDICT = {payload['reasoning_trace_difficulty_verdict']}",
        "",
        f"- generated_answer_branches: `{s['generated_answer_branches']}`",
        f"- natural_distractors: `{s['natural_distractors']}`",
        f"- generated_reasoning_traces: `{s['generated_reasoning_traces']}`",
        f"- interpretation: {s['interpretation']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    generated = load_json(GENERATED_TRANSFER)
    distractor = load_json(DISTRACTOR_TRANSFER)
    trace = load_json(TRACE_TRANSFER)
    v = verdict(generated, distractor, trace)
    payload = {
        "reasoning_trace_difficulty_verdict": v,
        "summary": {
            "REASONING_TRACE_DIFFICULTY_VERDICT": v,
            "generated_answer_branches": compact(generated),
            "natural_distractors": compact(distractor),
            "generated_reasoning_traces": compact(trace),
            "interpretation": interpretation(v),
        },
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    write_md(OUTPUT_MD, payload)
    print(f"REASONING_TRACE_DIFFICULTY_VERDICT = {v}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
