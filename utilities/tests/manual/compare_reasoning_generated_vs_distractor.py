"""Compare generated reasoning branch pilot with natural MCQ distractor validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, repo_path, write_json


GENERATED_TRANSFER = REPORT_DIR / "reasoning_branch_transfer_2026-05-17.json"
GENERATED_DATA = REPORT_DIR / "reasoning_branch_pilot_2026-05-17.json"
DISTRACTOR_TRANSFER = REPORT_DIR / "reasoning_natural_distractor_transfer_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "reasoning_generated_vs_distractor_comparison_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_generated_vs_distractor_comparison_2026-05-17.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict[str, Any] | str, name: str) -> float:
    if not isinstance(row, dict):
        return float("nan")
    return float(row.get(name, float("nan")))


def verdict(generated: dict[str, Any], distractor: dict[str, Any]) -> str:
    g = generated.get("summary", {}).get("best_overall", {})
    d = distractor.get("summary", {}).get("best_overall", {})
    if not g or not d:
        return "INSUFFICIENT"
    g_top, d_top = metric(g, "top1"), metric(d, "top1")
    g_pair, d_pair = metric(g, "pairwise"), metric(d, "pairwise")
    if g_top - d_top >= 0.10 or g_pair - d_pair >= 0.10:
        return "DISTRACTORS_HARDER"
    if d_top - g_top >= 0.10 or d_pair - g_pair >= 0.10:
        return "GENERATED_HARDER"
    if min(g_top, d_top) >= 0.95 and min(g_pair, d_pair) >= 0.95:
        return "BOTH_EASY"
    return "INSUFFICIENT"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Generated Vs Natural Distractor Comparison",
        "",
        f"REASONING_DIFFICULTY_VERDICT = {payload['reasoning_difficulty_verdict']}",
        "",
        f"- generated n_tournaments: `{s['generated_n_tournaments']}`",
        f"- distractor n_tournaments: `{s['distractor_n_tournaments']}`",
        f"- generated random_top1_baseline: `{s['generated_random_top1_baseline']}`",
        f"- distractor random_top1_baseline: `{s['distractor_random_top1_baseline']}`",
        f"- generated best HH: `{s['generated_best_hh']}`",
        f"- distractor best HH: `{s['distractor_best_hh']}`",
        f"- generated best CODE: `{s['generated_best_code']}`",
        f"- distractor best CODE: `{s['distractor_best_code']}`",
        f"- generated best overall: `{s['generated_best_overall']}`",
        f"- distractor best overall: `{s['distractor_best_overall']}`",
        f"- interpretation: `{s['interpretation']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    generated = load_json(GENERATED_TRANSFER)
    generated_data = load_json(GENERATED_DATA)
    distractor = load_json(DISTRACTOR_TRANSFER)
    v = verdict(generated, distractor)
    interpretation = {
        "DISTRACTORS_HARDER": "Natural distractors reduced transfer performance relative to generated answer branches.",
        "BOTH_EASY": "Both generated reasoning and natural distractors were near ceiling; reasoning remains promising but not yet stress-tested.",
        "GENERATED_HARDER": "Generated branch pilot was harder than the natural distractor set.",
        "INSUFFICIENT": "Comparison data is missing or not decisive.",
    }[v]
    gs = generated.get("summary", {})
    ds = distractor.get("summary", {})
    payload = {
        "reasoning_difficulty_verdict": v,
        "summary": {
            "REASONING_DIFFICULTY_VERDICT": v,
            "generated_n_tournaments": gs.get("n_tournaments"),
            "distractor_n_tournaments": ds.get("n_tournaments"),
            "generated_random_top1_baseline": gs.get("random_top1_baseline"),
            "distractor_random_top1_baseline": ds.get("random_top1_baseline"),
            "generated_best_hh": gs.get("best_hh"),
            "distractor_best_hh": ds.get("best_hh"),
            "generated_best_code": gs.get("best_code"),
            "distractor_best_code": ds.get("best_code"),
            "generated_best_nonorm": gs.get("best_nonorm", "not_reported"),
            "distractor_best_nonorm": ds.get("best_nonorm"),
            "generated_best_antisymlinear": gs.get("best_antisymlinear", "not_reported"),
            "distractor_best_antisymlinear": ds.get("best_antisymlinear"),
            "generated_best_overall": gs.get("best_overall"),
            "distractor_best_overall": ds.get("best_overall"),
            "generated_data_counts": generated_data.get("summary", {}),
            "interpretation": interpretation,
        },
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    write_md(OUTPUT_MD, payload)
    print(f"REASONING_DIFFICULTY_VERDICT = {v}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
