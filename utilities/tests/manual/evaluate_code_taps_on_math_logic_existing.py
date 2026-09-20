"""Evaluate code-trained registry taps on existing math and logic-style branch sets."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json
from evaluate_bg_fixed_configs_cross_domain import rate, records_from_pt, reconstruct_heads
from math_bg_probe_lib import config_vector
from train_code_specific_tiny_heads_and_eval import evaluate_matrices, score_matrix


HEADS_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
GSM8K_FEATURES = REPORT_DIR / "clean_gsm8k_expanded_tap_features_2026-05-16.pt"
NATURAL_SET = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.json"
NATURAL_FEATURES = REPORT_DIR / "reasoning_natural_distractor_features_2026-05-17.pt"
TRACE_JSON = REPORT_DIR / "reasoning_option_traces_2026-05-17.json"
TRACE_FEATURES = REPORT_DIR / "reasoning_trace_features_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "code_taps_on_math_logic_existing_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_taps_on_math_logic_existing_2026-05-17.md"

FIXED_ROWS = {
    ("CODE", "24_L4", "AntisymLinear"),
    ("CODE", "36_L4", "AntisymLinear"),
    ("CODE", "36_L4", "AntisymLinearNoNorm"),
    ("CODE", "47_L4", "AntisymLinearNoNorm"),
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pooled_map(feature_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32) for row in feature_payload.get("candidate_features", []) or []}


def records_from_candidate_eval(feature_path: Path, eval_set_name: str) -> list[dict[str, Any]]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    by_uid = pooled_map(payload)
    records = []
    for idx, row in enumerate(payload.get("eval_sets", {}).get(eval_set_name, []) or []):
        uids = [str(uid) for uid in row.get("candidate_uids", [])]
        if not uids or any(uid not in by_uid for uid in uids):
            continue
        labels = [label == "correct" for label in row.get("labels", [])]
        records.append({
            "tournament_id": idx,
            "task_id": row.get("task_id", str(idx)),
            "source": row.get("dataset", row.get("source", "unknown")),
            "option_count": int(row.get("n_options", len(uids))),
            "candidate_uids": uids,
            "label_names": list(row.get("labels", [])),
            "labels": torch.tensor(labels, dtype=torch.bool),
            "pooled": torch.stack([by_uid[uid] for uid in uids], dim=0),
        })
    return records


def random_top1(records: Sequence[dict[str, Any]]) -> float:
    return float(mean(float(row["labels"].to(torch.float32).mean()) for row in records)) if records else float("nan")


def config_features(records: Sequence[dict[str, Any]], config: str) -> list[torch.Tensor]:
    return [torch.stack([config_vector(pooled, config) for pooled in row["pooled"]], dim=0).to(torch.float32) for row in records]


@torch.no_grad()
def evaluate_domain(domain: str, records: list[dict[str, Any]], heads: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = random_top1(records)
    for info in heads:
        feats = config_features(records, info["config"])
        head = info["head"].to(device)
        matrices = [score_matrix(head, feat, device) for feat in feats]
        head = head.to("cpu")
        rows.append({
            "domain": domain,
            "head_family": info["head_family"],
            "architecture": info["architecture"],
            "family_architecture": info["family_architecture"],
            "config": info["config"],
            "metrics": evaluate_matrices(matrices, records, baseline),
        })
    return rows


def best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (
        float(row["metrics"].get("pairwise_acc", float("nan"))),
        float(row["metrics"].get("top1_tournament_acc", float("nan"))),
        -float(row["metrics"].get("cycle_rate", 1.0) or 0.0),
    ))


def compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {
        "domain": row["domain"],
        "family": row["head_family"],
        "config": row["config"],
        "architecture": row["architecture"],
        "top1": m["top1_tournament_acc"],
        "over_random": m["top1_over_random_baseline"],
        "pairwise": m["pairwise_acc"],
        "cycle": m["cycle_rate"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
    }


def verdict(row: dict[str, Any] | None, baseline: float) -> str:
    if not row:
        return "NOT_RUN"
    m = row["metrics"]
    if float(m["pairwise_acc"]) >= 0.60 and float(m["top1_tournament_acc"]) >= baseline + 0.15 and float(m["cycle_rate"]) <= 0.05:
        return "GOOD"
    if float(m["pairwise_acc"]) >= 0.55 or float(m["top1_tournament_acc"]) >= baseline + 0.05:
        return "WEAK"
    return "POOR"


def domain_summary(name: str, records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    code_rows = [row for row in rows if row["head_family"] == "CODE"]
    hh_rows = [row for row in rows if row["head_family"] == "HH"]
    baseline = random_top1(records)
    return {
        "domain": name,
        "n_tournaments": len(records),
        "n_candidates": sum(len(row["candidate_uids"]) for row in records),
        "random_top1_baseline": baseline,
        "best_code": compact(best(code_rows)),
        "best_hh": compact(best(hh_rows)),
        "best_overall": compact(best(rows)),
        "best_code_verdict": verdict(best(code_rows), baseline),
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Code Taps On Existing Math / Logic Sets",
        "",
        f"CODE_TAPS_ON_MATH_VERDICT = {payload['code_taps_on_math_verdict']}",
        f"CODE_TAPS_ON_LOGIC_VERDICT = {payload['code_taps_on_logic_verdict']}",
        "",
        "## Domain Summary",
        "",
    ]
    for row in s["domains"]:
        lines.append(f"- `{row['domain']}`: `{row}`")
    lines.extend([
        "",
        "## Fixed CODE Configs",
        "",
        "| domain | family | config | architecture | top1 | over_random | pairwise | cycle | margin_mean | margin_std |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["fixed_code_rows"]:
        m = row["metrics"]
        lines.append(
            f"| `{row['domain']}` | `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | "
            f"{rate(m['top1_tournament_acc'])} | {rate(m['top1_over_random_baseline'])} | "
            f"{rate(m['pairwise_acc'])} | {rate(m['cycle_rate'])} | {rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- CLEAN_GSM8K_EXPANDED uses existing exact-answer-verifier labels and cached pooled features.",
        "- REASONING_NATURAL_DISTRACTOR and REASONING_TRACE use official answer-key labels and cached pooled features.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = torch.load(HEADS_PT, map_location="cpu", weights_only=False)
    heads = reconstruct_heads(registry)
    domains: dict[str, list[dict[str, Any]]] = {}
    blockers: list[str] = []
    if GSM8K_FEATURES.exists():
        domains["CLEAN_GSM8K_EXPANDED"] = records_from_pt(GSM8K_FEATURES)
    else:
        blockers.append(f"missing {repo_path(GSM8K_FEATURES)}")
    if NATURAL_FEATURES.exists():
        domains["REASONING_NATURAL_DISTRACTOR"] = records_from_candidate_eval(NATURAL_FEATURES, "reasoning_natural_distractors")
    else:
        blockers.append(f"missing {repo_path(NATURAL_FEATURES)}")
    if TRACE_FEATURES.exists():
        domains["REASONING_TRACE"] = records_from_candidate_eval(TRACE_FEATURES, "reasoning_trace_primary")
    else:
        blockers.append(f"missing optional {repo_path(TRACE_FEATURES)}")
    if "REASONING_NATURAL_DISTRACTOR" in domains or "REASONING_TRACE" in domains:
        combined = []
        for name in ("REASONING_NATURAL_DISTRACTOR", "REASONING_TRACE"):
            for row in domains.get(name, []):
                combined.append({**row, "source": f"{name}:{row.get('source', 'unknown')}"})
        domains["LOGIC_COMBINED"] = combined

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, records in domains.items():
        if not records:
            continue
        rows = evaluate_domain(name, records, heads, device)
        all_rows.extend(rows)
        summaries.append(domain_summary(name, records, rows))
    math_summary = next((row for row in summaries if row["domain"] == "CLEAN_GSM8K_EXPANDED"), None)
    logic_summary = next((row for row in summaries if row["domain"] == "LOGIC_COMBINED"), None)
    math_v = math_summary["best_code_verdict"] if math_summary else "NOT_RUN"
    logic_v = logic_summary["best_code_verdict"] if logic_summary else "NOT_RUN"
    fixed = [row for row in all_rows if (row["head_family"], row["config"], row["architecture"]) in FIXED_ROWS]
    payload = {
        "code_taps_on_math_verdict": math_v,
        "code_taps_on_logic_verdict": logic_v,
        "summary": {
            "CODE_TAPS_ON_MATH_VERDICT": math_v,
            "CODE_TAPS_ON_LOGIC_VERDICT": logic_v,
            "domains": summaries,
            "best_code_rows": {row["domain"]: row["best_code"] for row in summaries},
            "best_hh_rows": {row["domain"]: row["best_hh"] for row in summaries},
            "blockers": blockers,
        },
        "all_rows": all_rows,
        "fixed_code_rows": fixed,
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    write_md(OUTPUT_MD, payload)
    print(f"CODE_TAPS_ON_MATH_VERDICT = {math_v}")
    print(f"CODE_TAPS_ON_LOGIC_VERDICT = {logic_v}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
