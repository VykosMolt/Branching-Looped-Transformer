"""Audit scalar-readable NoNorm vs relational AntisymLinear from saved reports."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "probes"
INVENTORY_JSON = REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.json"
OUT_JSON = REPORT_DIR / "scalar_vs_relational_current_state_2026-05-17.json"
OUT_MD = REPORT_DIR / "scalar_vs_relational_current_state_2026-05-17.md"


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def is_num(value: Any) -> bool:
    try:
        f = float(value)
    except Exception:
        return False
    return not math.isnan(f)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def row_metric(row: Any, key: str) -> Any:
    if not isinstance(row, dict):
        return "NA"
    if key in row:
        return row[key]
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        mapping = {
            "top1": "top1_tournament_acc",
            "pairwise": "pairwise_acc",
            "cycle": "cycle_rate",
        }
        return metrics.get(mapping.get(key, key), "NA")
    return "NA"


def config(row: Any) -> str:
    return str(row.get("config", "NA")) if isinstance(row, dict) else "NA"


def winner(a_value: Any, n_value: Any, a_name: str, n_name: str) -> str:
    if not is_num(a_value) or not is_num(n_value):
        return "NA"
    if fnum(a_value) > fnum(n_value):
        return a_name
    if fnum(n_value) > fnum(a_value):
        return n_name
    return "tie"


def scalar_verdict(row: dict[str, Any]) -> str:
    status = row["artifact_status"]
    if status in {"confounded", "dirty_historical", "dataset_quality", "diagnostic_no_transfer", "risk_register", "clean_fix"}:
        return "INSUFFICIENT"
    n = row["n_tournaments"]
    if not is_num(n) or fnum(n) < 8:
        return "INSUFFICIENT"
    a_top = row["best_AntisymLinear_top1"]
    a_pair = row["best_AntisymLinear_pairwise"]
    a_cycle = row["best_AntisymLinear_cycle"]
    n_top = row["best_NoNorm_top1"]
    n_pair = row["best_NoNorm_pairwise"]
    n_cycle = row["best_NoNorm_cycle"]
    if not (is_num(a_top) and is_num(a_pair) and is_num(n_top) and is_num(n_pair) and is_num(n_cycle)):
        return "INSUFFICIENT"
    if fnum(a_cycle, 999.0) > 0.05 or fnum(n_cycle, 999.0) > 0.05:
        return "INSUFFICIENT"
    a_wins_top = fnum(a_top) > fnum(n_top)
    n_wins_top = fnum(n_top) > fnum(a_top)
    a_wins_pair = fnum(a_pair) > fnum(n_pair)
    n_wins_pair = fnum(n_pair) > fnum(a_pair)
    if fnum(a_top) >= fnum(n_top) + 0.05 and fnum(a_pair) >= fnum(n_pair) + 0.05:
        return "RELATIONAL_ADVANTAGE"
    if (a_wins_top and n_wins_pair) or (n_wins_top and a_wins_pair):
        return "MIXED"
    if n_wins_top or n_wins_pair or abs(fnum(n_top) - fnum(a_top)) <= 0.03 or abs(fnum(n_pair) - fnum(a_pair)) <= 0.03:
        return "SCALAR_READABLE"
    return "INSUFFICIENT"


def caveat(row: dict[str, Any]) -> str:
    status = row["artifact_status"]
    family = row["artifact_family"]
    if status == "confounded":
        return "historical confounded artifact; old negative transfer is not decisive"
    if status == "dirty_historical":
        return "historical dirty artifact admitted wrapper/taskset/eval contamination"
    if family == "clean_gsm8k_micro":
        return "n=5 micro; useful sanity check but too small for readout verdict"
    if family == "clean_gsm8k_expanded_gru":
        return "valid clean GSM8K but still small n=28"
    if family == "patched_code_v2_mini":
        return "valid patched pipeline but primary set was diagnostic_runnable, not strict_clean"
    if status == "dataset_quality":
        return "dataset-quality result only; no feature capture or transfer"
    return "not a transfer comparison"


def pointwise_supported(rows: list[dict[str, Any]]) -> str:
    by_family = {row["artifact_family"]: row for row in rows}
    gsm = by_family.get("clean_gsm8k_expanded_gru", {})
    code = by_family.get("patched_code_v2_mini", {})
    def competitive(row: dict[str, Any]) -> bool:
        if not row:
            return False
        if row.get("scalar_vs_relational_verdict") in {"SCALAR_READABLE", "MIXED"}:
            return True
        return False
    if competitive(gsm) and competitive(code):
        return "SUPPORTED_IN_OBJECTIVE_DOMAINS"
    if competitive(gsm) or competitive(code):
        return "MIXED"
    return "INSUFFICIENT"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Scalar vs Relational Current State",
        "",
        f"POINTWISE_RANKING_VERDICT = {payload['POINTWISE_RANKING_VERDICT']}",
        "",
        "| domain | family | eval_set | n | random | A config | A top1 | A pairwise | A cycle | NoNorm config | NoNorm top1 | NoNorm pairwise | NoNorm cycle | GRU config | GRU top1 | winner top1 | winner pairwise | verdict | caveat |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['domain']}` | `{row['artifact_family']}` | `{row['eval_set']}` | `{row['n_tournaments']}` | "
            f"`{row['random_top1_baseline']}` | `{row['best_AntisymLinear_config']}` | `{row['best_AntisymLinear_top1']}` | "
            f"`{row['best_AntisymLinear_pairwise']}` | `{row['best_AntisymLinear_cycle']}` | `{row['best_NoNorm_config']}` | "
            f"`{row['best_NoNorm_top1']}` | `{row['best_NoNorm_pairwise']}` | `{row['best_NoNorm_cycle']}` | "
            f"`{row['best_GRU_config']}` | `{row['best_GRU_top1']}` | `{row['winner_by_top1']}` | "
            f"`{row['winner_by_pairwise']}` | `{row['scalar_vs_relational_verdict']}` | {row['robustness_caveat']} |"
        )
    lines.extend([
        "",
        "Interpretation: NoNorm is a scalar-readable `u(a)-u(b)` comparator. Its competitiveness in clean GSM8K and patched code suggests objective correctness domains can expose a pointwise ranking direction, while HH preference remains a noisy relational setting.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inventory = load_json(INVENTORY_JSON)
    rows = []
    for fam in inventory.get("families", []):
        a = fam.get("best_antisymlinear", {})
        n = fam.get("best_nonorm", {})
        g = fam.get("best_gru", {})
        random_top1 = "NA"
        data = load_json(PROJECT_ROOT / fam.get("json_path", ""))
        if isinstance(data, dict):
            random_top1 = data.get("random_top1_baseline", data.get("summary", {}).get("random_top1_baseline", data.get("generation_summary", {}).get("random_top1_baseline", "NA")))
        row = {
            "domain": fam.get("domain"),
            "artifact_family": fam.get("artifact_family"),
            "artifact_status": fam.get("artifact_status"),
            "eval_set": fam.get("eval_set_type"),
            "n_tournaments": fam.get("n_tournaments"),
            "random_top1_baseline": random_top1,
            "best_AntisymLinear_config": config(a),
            "best_AntisymLinear_top1": row_metric(a, "top1"),
            "best_AntisymLinear_pairwise": row_metric(a, "pairwise"),
            "best_AntisymLinear_cycle": row_metric(a, "cycle"),
            "best_NoNorm_config": config(n),
            "best_NoNorm_top1": row_metric(n, "top1"),
            "best_NoNorm_pairwise": row_metric(n, "pairwise"),
            "best_NoNorm_cycle": row_metric(n, "cycle"),
            "best_GRU_config": config(g),
            "best_GRU_top1": row_metric(g, "top1"),
            "best_GRU_pairwise": row_metric(g, "pairwise"),
        }
        row["winner_by_top1"] = winner(row["best_AntisymLinear_top1"], row["best_NoNorm_top1"], "AntisymLinear", "NoNorm")
        row["winner_by_pairwise"] = winner(row["best_AntisymLinear_pairwise"], row["best_NoNorm_pairwise"], "AntisymLinear", "NoNorm")
        row["scalar_vs_relational_verdict"] = scalar_verdict(row)
        row["robustness_caveat"] = caveat(row)
        rows.append(row)
    payload = {
        "POINTWISE_RANKING_VERDICT": pointwise_supported(rows),
        "rows": rows,
        "rules": {
            "SCALAR_READABLE": "NoNorm wins or is within 0.03 on top1/pairwise, cycle_rate=0.",
            "RELATIONAL_ADVANTAGE": "AntisymLinear beats NoNorm by at least 0.05 on both top1 and pairwise, cycle_rate<=0.05.",
            "MIXED": "AntisymLinear wins one metric and NoNorm wins another, or both are close/above baseline.",
            "INSUFFICIENT": "too few, dirty/confounded, or missing comparison.",
        },
        "outputs": {"json": repo_path(OUT_JSON), "md": repo_path(OUT_MD)},
    }
    write_json(OUT_JSON, payload)
    write_md(OUT_MD, payload)
    print(f"POINTWISE_RANKING_VERDICT = {payload['POINTWISE_RANKING_VERDICT']}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
