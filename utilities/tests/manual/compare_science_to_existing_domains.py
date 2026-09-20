"""Compare science-domain tap behavior to existing HH/reasoning/math/code domains."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, repo_path, write_json


SCIENCE_TRANSFER = REPORT_DIR / "science_natural_distractor_transfer_2026-05-17.json"
BG_FIXED = REPORT_DIR / "bg_fixed_config_cross_domain_audit_2026-05-17.json"
REASONING_NATURAL = REPORT_DIR / "reasoning_natural_distractor_transfer_2026-05-17.json"
REASONING_TRACE = REPORT_DIR / "reasoning_trace_transfer_2026-05-17.json"
CLEAN_GSM8K = REPORT_DIR / "clean_gsm8k_expanded_transfer_2026-05-16.json"
STRICT_CODE = REPORT_DIR / "expanded_strict_clean_code_projection_comparison_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "science_domain_comparison_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "science_domain_comparison_2026-05-17.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: Any, key: str) -> float:
    if not isinstance(row, dict):
        return float("nan")
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def compact_best(payload: dict[str, Any]) -> dict[str, Any]:
    s = payload.get("summary", {})
    return {
        "best_hh": s.get("best_hh"),
        "best_code": s.get("best_code"),
        "best_nonorm": s.get("best_nonorm"),
        "best_antisymlinear": s.get("best_antisymlinear"),
        "best_overall": s.get("best_overall"),
        "verdict": s.get("SCIENCE_TRANSFER_VERDICT")
        or s.get("REASONING_DISTRACTOR_TRANSFER_VERDICT")
        or s.get("REASONING_TRACE_TRANSFER_VERDICT")
        or payload.get("expanded_clean_gsm8k_verdict")
        or payload.get("EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT"),
    }


def science_analogy(science: dict[str, Any]) -> str:
    s = science.get("summary", {})
    if not s:
        return "INSUFFICIENT"
    specialist = s.get("SCIENCE_SPECIALIST_VERDICT")
    sub = s.get("per_subdomain", {})
    if not sub:
        return "INSUFFICIENT"
    code_adv = []
    hh_adv = []
    for bucket, row in sub.items():
        if int(row.get("n_tournaments", 0)) < 10:
            continue
        pair_adv = row.get("code_advantage_pairwise")
        top_adv = row.get("code_advantage_top1")
        if pair_adv is not None and float(pair_adv) >= 0.10 or top_adv is not None and float(top_adv) >= 0.10:
            code_adv.append(bucket)
        if pair_adv is not None and float(pair_adv) <= -0.10 or top_adv is not None and float(top_adv) <= -0.10:
            hh_adv.append(bucket)
    if code_adv and hh_adv:
        return "HETEROGENEOUS"
    if specialist == "SPECIALIST_NEEDED" or code_adv:
        return "RESEMBLES_CODE_OBJECTIVE"
    if specialist == "GENERAL_SUFFICIENT":
        return "RESEMBLES_REASONING"
    best_nonorm = s.get("best_nonorm", {})
    best_overall = s.get("best_overall", {})
    if isinstance(best_nonorm, dict) and isinstance(best_overall, dict):
        if abs(metric(best_nonorm, "pairwise") - metric(best_overall, "pairwise")) <= 0.02:
            return "RESEMBLES_MATH_OBJECTIVE"
    return "INSUFFICIENT"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Science Domain Comparison",
        "",
        f"SCIENCE_DOMAIN_ANALOGY_VERDICT = {payload['science_domain_analogy_verdict']}",
        "",
        f"- science: `{s['science']}`",
        f"- reasoning_natural: `{s['reasoning_natural']}`",
        f"- reasoning_trace: `{s['reasoning_trace']}`",
        f"- clean_gsm8k: `{s['clean_gsm8k']}`",
        f"- strict_clean_code: `{s['strict_clean_code']}`",
        f"- fixed_config_context: `{s['fixed_config_context']}`",
        f"- interpretation: {s['interpretation']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    science = load_json(SCIENCE_TRANSFER)
    reasoning_natural = load_json(REASONING_NATURAL)
    reasoning_trace = load_json(REASONING_TRACE)
    clean_gsm8k = load_json(CLEAN_GSM8K)
    strict_code = load_json(STRICT_CODE)
    bg_fixed = load_json(BG_FIXED)
    v = science_analogy(science)
    interpretation = {
        "RESEMBLES_REASONING": "Science natural distractors resemble reasoning MCQ: existing general HH/code readouts are sufficient without a clear science specialist requirement.",
        "RESEMBLES_CODE_OBJECTIVE": "Science shows a code-like objective/readout advantage in at least one subdomain.",
        "RESEMBLES_MATH_OBJECTIVE": "Science resembles objective scalar-readable domains where NoNorm is competitive with the best row.",
        "HETEROGENEOUS": "Science subdomains split between HH-like and code-like behavior, so routing should be subdomain-aware.",
        "INSUFFICIENT": "Science comparison is not decisive from the available artifacts.",
    }[v]
    payload = {
        "science_domain_analogy_verdict": v,
        "summary": {
            "SCIENCE_DOMAIN_ANALOGY_VERDICT": v,
            "science": compact_best(science),
            "science_subdomains": science.get("summary", {}).get("per_subdomain", {}),
            "reasoning_natural": compact_best(reasoning_natural),
            "reasoning_trace": compact_best(reasoning_trace),
            "clean_gsm8k": {
                "best_hh": clean_gsm8k.get("best_hh_trained"),
                "best_antisymlinear": clean_gsm8k.get("best_antisymlinear"),
                "best_nonorm": clean_gsm8k.get("best_nonorm"),
                "verdict": clean_gsm8k.get("expanded_linear_transfer_verdict"),
            },
            "strict_clean_code": strict_code.get("summary", strict_code),
            "fixed_config_context": bg_fixed.get("summary", {}),
            "interpretation": interpretation,
        },
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    write_md(OUTPUT_MD, payload)
    print(f"SCIENCE_DOMAIN_ANALOGY_VERDICT = {v}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
