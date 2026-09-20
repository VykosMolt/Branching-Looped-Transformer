"""Inventory current BG/tap transfer artifacts from saved reports only."""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "probes"
OUT_JSON = REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.json"
OUT_MD = REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.md"


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def clean_float(value: Any) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    if math.isnan(f):
        return "nan"
    return f


def metric(row: Any, key: str) -> Any:
    if not isinstance(row, dict):
        return None
    if key in row:
        return clean_float(row[key])
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        mapping = {
            "top1": "top1_tournament_acc",
            "pairwise": "pairwise_acc",
            "cycle": "cycle_rate",
            "condorcet": "condorcet_winner_rate",
        }
        return clean_float(metrics.get(mapping.get(key, key)))
    return None


def compact_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row or "NA"
    return {
        "config": row.get("config", "NA"),
        "architecture": row.get("architecture", "GRU" if str(row.get("config", "")).startswith("gru") else "NA"),
        "top1": metric(row, "top1") if metric(row, "top1") is not None else metric(row, "centered_top1_tournament_acc"),
        "pairwise": metric(row, "pairwise") if metric(row, "pairwise") is not None else metric(row, "centered_pairwise_acc"),
        "cycle": metric(row, "cycle") if metric(row, "cycle") is not None else metric(row, "centered_cycle_rate"),
    }


def best_by_arch(table: list[Any], arch: str) -> Any:
    rows = [row for row in table if isinstance(row, dict) and row.get("architecture") == arch]
    if not rows:
        return "NA"
    return compact_row(max(rows, key=lambda row: (
        metric(row, "top1") if metric(row, "top1") is not None else -1,
        metric(row, "pairwise") if metric(row, "pairwise") is not None else -1,
        -(metric(row, "cycle") if metric(row, "cycle") is not None else 999),
    )))


def best_gru_from_table(table: list[Any]) -> Any:
    if not table:
        return "NA"
    def _score(row: dict[str, Any]) -> tuple[float, float, float]:
        m = row.get("metrics", {}) if isinstance(row, dict) else {}
        return (
            float(m.get("centered_top1_tournament_acc", m.get("top1_tournament_acc", -1))),
            float(m.get("centered_pairwise_acc", m.get("pairwise_acc", -1))),
            -float(m.get("centered_cycle_rate", m.get("cycle_rate", 999))),
        )
    return compact_row(max([row for row in table if isinstance(row, dict)], key=_score))


def n_tournaments(data: dict[str, Any]) -> Any:
    primary = data.get("primary_eval_set")
    if primary == "diagnostic_runnable" and "diagnostic_runnable_tournaments" in data:
        return data["diagnostic_runnable_tournaments"]
    if primary == "diagnostic_mixed" and "diagnostic_mixed_tournaments" in data:
        return data["diagnostic_mixed_tournaments"]
    if primary == "strict_clean" and "strict_clean_tournaments" in data:
        return data["strict_clean_tournaments"]
    if "primary_tournament_ids" in data and isinstance(data.get("primary_tournament_ids"), list):
        return len(data["primary_tournament_ids"])
    for path in (
        ("feature_summary", "n_tournaments"),
        ("generation_summary", "clean_tournaments_kept"),
        ("summary", "strict_clean_tournaments"),
        ("summary", "after", "strict_clean"),
        ("input_summary", "n_tournaments"),
    ):
        cur: Any = data
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    for key in ("clean_tournaments", "strict_clean_tournaments", "new_strict_clean"):
        if key in data:
            return data[key]
    return "NA"


def eval_set(data: dict[str, Any], status: str) -> str:
    if data.get("primary_eval_set"):
        return str(data["primary_eval_set"])
    if data.get("clean_transfer_verdict"):
        return "clean_gsm8k_micro"
    if data.get("expanded_linear_transfer_verdict"):
        return "clean_gsm8k_expanded"
    if data.get("CODE_TRANSFER_VERDICT") == "NOT_RUN":
        return "not_run"
    if data.get("CODE_V2_TRANSFER_VERDICT"):
        return str(data.get("primary_eval_set", "diagnostic"))
    if data.get("CODE_V2_MINI_TRANSFER_VERDICT"):
        return str(data.get("primary_eval_set", "diagnostic_runnable"))
    if status in {"confounded", "dirty", "dataset_quality"}:
        return status
    return "unknown"


FAMILIES: list[dict[str, Any]] = [
    {
        "artifact_family": "math_pilot_validity",
        "domain": "math_mixed",
        "result_family": "validity_probe",
        "json": REPORT_DIR / "math_data_validity_2026-05-16.json",
        "md": REPORT_DIR / "math_data_validity_2026-05-16.md",
        "patterns": ["*math_data_validity*2026-05-16*.json", "*math_data_validity*2026-05-16*.md"],
        "status": "confounded",
    },
    {
        "artifact_family": "clean_gsm8k_micro",
        "domain": "gsm8k",
        "result_family": "transfer_micro",
        "json": REPORT_DIR / "clean_gsm8k_extreme_transfer_2026-05-16.json",
        "md": REPORT_DIR / "clean_gsm8k_extreme_transfer_2026-05-16.md",
        "patterns": ["*clean_gsm8k_extreme*2026-05-16*.json", "*clean_gsm8k_extreme*2026-05-16*.md"],
        "status": "clean_small",
    },
    {
        "artifact_family": "clean_gsm8k_expanded_gru",
        "domain": "gsm8k",
        "result_family": "transfer_and_gru_control",
        "json": REPORT_DIR / "clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.json",
        "md": REPORT_DIR / "clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md",
        "patterns": ["*clean_gsm8k_expanded*2026-05-16*.json", "*clean_gsm8k_expanded*2026-05-16*.md"],
        "status": "clean_small",
    },
    {
        "artifact_family": "code_branch_v1",
        "domain": "code",
        "result_family": "pilot_no_transfer",
        "json": REPORT_DIR / "code_branch_pilot_2026-05-16_summary.json",
        "md": REPORT_DIR / "code_branch_pilot_2026-05-16_summary.md",
        "patterns": ["*code_branch_pilot_2026-05-16*.json", "*code_branch_pilot_2026-05-16*.md"],
        "status": "diagnostic_no_transfer",
    },
    {
        "artifact_family": "code_branch_v2_prefix",
        "domain": "code",
        "result_family": "historical_dirty_transfer",
        "json": REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.json",
        "md": REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.md",
        "patterns": ["*code_branch_pilot_v2_2026-05-16*.json", "*code_branch_pilot_v2_2026-05-16*.md", "*code_branch_v2_nonsense*2026-05-16*.json", "*code_branch_v2_nonsense*2026-05-16*.md"],
        "status": "dirty_historical",
    },
    {
        "artifact_family": "code_branch_v2_harness_fixes",
        "domain": "code",
        "result_family": "harness_fix",
        "json": REPORT_DIR / "code_branch_v2_harness_agent_fixes_2026-05-16.json",
        "md": REPORT_DIR / "code_branch_v2_harness_agent_fixes_2026-05-16.md",
        "patterns": ["*code_branch_v2_harness_agent_fixes*2026-05-16*.json", "*code_branch_v2_harness_agent_fixes*2026-05-16*.md", "*code_branch_v2_patch_status*2026-05-16*.json", "*code_branch_v2_patch_status*2026-05-16*.md"],
        "status": "clean_fix",
    },
    {
        "artifact_family": "patched_code_v2_mini",
        "domain": "code",
        "result_family": "patched_transfer",
        "json": REPORT_DIR / "code_branch_pilot_v2_mini_patched_2026-05-16_summary.json",
        "md": REPORT_DIR / "code_branch_pilot_v2_mini_patched_2026-05-16_summary.md",
        "patterns": ["*code_branch_pilot_v2_mini_patched*2026-05-16*.json", "*code_branch_pilot_v2_mini_patched*2026-05-16*.md"],
        "status": "diagnostic_current",
    },
    {
        "artifact_family": "near_miss_enrichment10",
        "domain": "code",
        "result_family": "dataset_quality",
        "json": REPORT_DIR / "code_branch_near_miss_enrichment10_2026-05-17_summary.json",
        "md": REPORT_DIR / "code_branch_near_miss_enrichment10_2026-05-17_summary.md",
        "patterns": ["*near_miss*2026-05-17*.json", "*near_miss*2026-05-17*.md"],
        "status": "dataset_quality",
    },
    {
        "artifact_family": "near_miss_balancing",
        "domain": "code",
        "result_family": "dataset_quality_balancing",
        "json": REPORT_DIR / "code_branch_near_miss_balancing_2026-05-17_summary.json",
        "md": REPORT_DIR / "code_branch_near_miss_balancing_2026-05-17_summary.md",
        "patterns": ["*balancing*2026-05-17*.json", "*balancing*2026-05-17*.md"],
        "status": "dataset_quality",
    },
    {
        "artifact_family": "future_risks",
        "domain": "code",
        "result_family": "risk_register",
        "json": REPORT_DIR / "code_branch_future_risks_2026-05-16.json",
        "md": REPORT_DIR / "code_branch_future_risks_2026-05-16.md",
        "patterns": ["*future_risks*2026-05-16*.json", "*future_risks*2026-05-16*.md"],
        "status": "risk_register",
    },
]


def matching_paths(patterns: list[str]) -> list[str]:
    found: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(str(REPORT_DIR / pattern)):
            found.add(repo_path(Path(path)))
    return sorted(found)


def summarize_family(spec: dict[str, Any]) -> dict[str, Any]:
    data = load_json(spec["json"])
    table = data.get("transfer_table") or data.get("linear_transfer_table") or []
    row_a = data.get("best_antisymlinear") or best_by_arch(table, "AntisymLinear")
    row_n = data.get("best_nonorm") or best_by_arch(table, "AntisymLinearNoNorm")
    row_g = data.get("best_gru") or best_gru_from_table(data.get("gru_table", []))
    if isinstance(row_a, dict) and "metrics" in row_a:
        row_a = compact_row(row_a)
    if isinstance(row_n, dict) and "metrics" in row_n:
        row_n = compact_row(row_n)
    if isinstance(row_g, dict) and "metrics" in row_g:
        row_g = compact_row(row_g)
    return {
        "artifact_family": spec["artifact_family"],
        "domain": spec["domain"],
        "result_family": spec["result_family"],
        "artifact_status": spec["status"],
        "artifact_paths": matching_paths(spec["patterns"]),
        "json_path": repo_path(spec["json"]),
        "md_path": repo_path(spec["md"]),
        "json_exists": spec["json"].exists(),
        "markdown_exists": spec["md"].exists(),
        "n_tournaments": n_tournaments(data),
        "eval_set_type": eval_set(data, spec["status"]),
        "best_antisymlinear": compact_row(row_a),
        "best_nonorm": compact_row(row_n),
        "best_gru": compact_row(row_g),
        "key_verdicts": {k: v for k, v in data.items() if k.isupper() or k.endswith("_verdict") or k in {"data_validity", "clean_gsm8k_verdict", "clean_transfer_verdict", "expanded_clean_gsm8k_verdict", "expanded_linear_transfer_verdict", "gru_control_verdict"}},
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Current BG Transfer Artifact Inventory",
        "",
        f"CURRENT_ARTIFACT_INVENTORY_VERDICT = {payload['CURRENT_ARTIFACT_INVENTORY_VERDICT']}",
        "",
        f"- families_found: `{payload['summary']['families_found']}`",
        f"- families_missing_json: `{payload['summary']['families_missing_json']}`",
        f"- required_families_present: `{payload['summary']['required_families_present']}`",
        "",
        "| family | domain | status | json | md | n_tournaments | eval_set | best AntisymLinear | best NoNorm | best GRU |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["families"]:
        lines.append(
            f"| `{row['artifact_family']}` | `{row['domain']}` | `{row['artifact_status']}` | "
            f"{row['json_exists']} | {row['markdown_exists']} | `{row['n_tournaments']}` | "
            f"`{row['eval_set_type']}` | `{row['best_antisymlinear']}` | `{row['best_nonorm']}` | `{row['best_gru']}` |"
        )
    lines.extend(["", "## Artifact Paths", ""])
    for row in payload["families"]:
        lines.append(f"### {row['artifact_family']}")
        for artifact_path in row["artifact_paths"]:
            lines.append(f"- `{artifact_path}`")
        if not row["artifact_paths"]:
            lines.append("- `MISSING`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    families = [summarize_family(spec) for spec in FAMILIES]
    required = {"clean_gsm8k_expanded_gru", "patched_code_v2_mini", "near_miss_enrichment10", "near_miss_balancing"}
    present_required = {row["artifact_family"] for row in families if row["artifact_family"] in required and row["json_exists"] and row["markdown_exists"]}
    if present_required == required:
        verdict = "READY"
    elif present_required:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "CURRENT_ARTIFACT_INVENTORY_VERDICT": verdict,
        "summary": {
            "families_found": sum(1 for row in families if row["json_exists"] or row["markdown_exists"]),
            "families_missing_json": [row["artifact_family"] for row in families if not row["json_exists"]],
            "required_families_present": sorted(present_required),
            "required_families_missing": sorted(required - present_required),
        },
        "families": families,
        "outputs": {"json": repo_path(OUT_JSON), "md": repo_path(OUT_MD)},
    }
    write_json(OUT_JSON, payload)
    write_md(OUT_MD, payload)
    print(f"CURRENT_ARTIFACT_INVENTORY_VERDICT = {verdict}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
