"""Evaluate fixed tiny-head configs across cached HH/GSM8K/code domains."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json
from evaluate_hh_transfer_on_clean_gsm8k_extreme import build_hh_features
from math_bg_probe_lib import MATH_CONFIGS, config_vector
from train_code_specific_tiny_heads_and_eval import (
    HEAD_CLASSES,
    evaluate_matrices,
    records_for_eval_set,
    score_matrix,
)


HEADS_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
MATRIX_JSON = REPORT_DIR / "bg_cross_domain_eval_matrix_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "bg_fixed_config_cross_domain_audit_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_fixed_config_cross_domain_audit_2026-05-17.md"

FIXED_CONFIGS = (
    "24_L4",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "47_concat_all_loops",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default=str(HEADS_PT))
    parser.add_argument("--matrix", default=str(MATRIX_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def reconstruct_heads(registry: dict[str, Any]) -> list[dict[str, Any]]:
    heads = []
    for row in registry.get("heads", []) or []:
        head = HEAD_CLASSES[row["architecture"]](int(row["dim"]))
        head.load_state_dict(row["state_dict"])
        head.eval()
        heads.append({**row, "head": head})
    return heads


def random_top1(records: Sequence[dict[str, Any]]) -> float:
    vals = [float(row["labels"].to(torch.float32).mean()) for row in records]
    return float(mean(vals)) if vals else float("nan")


def records_from_pt(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for idx, row in enumerate(payload.get("records", []) or []):
        labels_raw = row.get("labels")
        labels = labels_raw.to(torch.bool) if hasattr(labels_raw, "to") else torch.tensor(labels_raw, dtype=torch.bool)
        n = int(row["pooled"].shape[0])
        label_names = ["correct" if bool(x) else "incorrect" for x in labels.tolist()]
        out.append({
            "tournament_id": int(row.get("tournament_id", idx)),
            "task_id": str(row.get("task_id") or row.get("problem_id") or idx),
            "source": row.get("source", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "function_name": row.get("function_name", ""),
            "candidate_uids": [f"{row.get('task_id') or row.get('problem_id') or idx}::{i}" for i in range(n)],
            "label_names": label_names,
            "labels": labels,
            "pooled": row["pooled"].detach().cpu().to(torch.float32),
        })
    return out


def records_from_candidate_features(path: Path, eval_set: str) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    by_uid = {str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32) for row in payload.get("candidate_features", []) or []}
    return records_for_eval_set(payload.get("eval_sets", {}).get(eval_set, []) or [], by_uid)


@torch.no_grad()
def evaluate_hh(heads: list[dict[str, Any]], feature_path: Path, device: torch.device) -> list[dict[str, Any]]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    indices = list(range(len(payload.get("packs", []) or [])))
    rows = []
    feature_cache = {}
    for info in heads:
        config = info["config"]
        if config not in feature_cache:
            feature_cache[config] = build_hh_features(payload, config)
        chosen, rejected = feature_cache[config]
        head = info["head"].to(device)
        left = chosen[indices].to(device)
        right = rejected[indices].to(device)
        scores = head(left, right).detach().cpu()
        reverse = head(right, left).detach().cpu()
        head = head.to("cpu")
        acc = float((scores > 0).to(torch.float32).mean())
        anti = scores + reverse
        rows.append({
            "domain": "HH_200",
            "head_family": info["head_family"],
            "architecture": info["architecture"],
            "family_architecture": info["family_architecture"],
            "config": config,
            "fixed_config": config in FIXED_CONFIGS,
            "metrics": {
                "n_pairs": len(indices),
                "n_tournaments": len(indices),
                "random_top1_baseline": 0.5,
                "top1_tournament_acc": acc,
                "top1_over_random_baseline": acc - 0.5,
                "pairwise_acc": acc,
                "condorcet_winner_rate": float("nan"),
                "cycle_rate": float("nan"),
                "margin_mean": float(scores.mean()),
                "margin_std": float(scores.std(unbiased=False)),
                "canonical_accuracy": acc,
                "flipped_accuracy": float((reverse < 0).to(torch.float32).mean()),
                "antisymmetry_mean_abs": float(anti.abs().mean()),
            },
        })
    return rows


def config_features(records: Sequence[dict[str, Any]], config: str) -> list[torch.Tensor]:
    return [torch.stack([config_vector(pooled, config) for pooled in row["pooled"]], dim=0).to(torch.float32) for row in records]


@torch.no_grad()
def evaluate_records(domain: str, records: list[dict[str, Any]], heads: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    baseline = random_top1(records)
    for info in heads:
        feats = config_features(records, info["config"])
        head = info["head"].to(device)
        matrices = [score_matrix(head, feat, device) for feat in feats]
        head = head.to("cpu")
        m = evaluate_matrices(matrices, records, baseline)
        source_breakdown = {}
        for source in sorted({str(row.get("source", "unknown")) for row in records}):
            idx = [i for i, row in enumerate(records) if str(row.get("source", "unknown")) == source]
            if idx:
                source_records = [records[i] for i in idx]
                source_mats = [matrices[i] for i in idx]
                source_breakdown[source] = evaluate_matrices(source_mats, source_records, random_top1(source_records))
        rows.append({
            "domain": domain,
            "head_family": info["head_family"],
            "architecture": info["architecture"],
            "family_architecture": info["family_architecture"],
            "config": info["config"],
            "fixed_config": info["config"] in FIXED_CONFIGS,
            "metrics": m,
            "source_breakdown": source_breakdown,
        })
    return rows


def best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (row["metrics"]["top1_tournament_acc"], row["metrics"]["pairwise_acc"], -float(row["metrics"].get("cycle_rate", 0) or 0)))


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
        "cycle": m.get("cycle_rate"),
        "margin_mean": m.get("margin_mean"),
        "margin_std": m.get("margin_std"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains = sorted({row["domain"] for row in rows})
    best_per_domain = {domain: compact(best([row for row in rows if row["domain"] == domain])) for domain in domains}
    best_per_family = {}
    for family in ("HH", "CODE"):
        family_rows = [row for row in rows if row["head_family"] == family]
        best_per_family[family] = compact(best(family_rows))
    fixed_rows = [row for row in rows if row["fixed_config"]]
    stability = []
    domain_best_pairwise = {domain: max(row["metrics"]["pairwise_acc"] for row in rows if row["domain"] == domain) for domain in domains}
    for key, group in group_rows(fixed_rows).items():
        near = sum(1 for row in group if row["metrics"]["pairwise_acc"] >= domain_best_pairwise[row["domain"]] - 0.05)
        avg_pair = mean(float(row["metrics"]["pairwise_acc"]) for row in group)
        avg_over = mean(float(row["metrics"]["top1_over_random_baseline"]) for row in group if not math.isnan(float(row["metrics"]["top1_over_random_baseline"])))
        stability.append({"head_key": key, "domains_within_0p05_of_best": near, "domains_seen": len(group), "avg_pairwise": avg_pair, "avg_top1_over_random": avg_over})
    stability.sort(key=lambda row: (row["domains_within_0p05_of_best"], row["avg_pairwise"]), reverse=True)
    return {"domains": domains, "best_per_domain": best_per_domain, "best_per_family": best_per_family, "stability": stability[:20]}


def group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[f"{row['head_family']}_{row['architecture']}::{row['config']}"].append(row)
    return out


def verdicts(rows: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    all16 = [row for row in rows if row["domain"] == "CODE_STRICT_CLEAN_ALL16"]
    hh = [row for row in rows if row["domain"] == "HH_200"]
    if not all16 or not hh:
        return "PARTIAL", "INSUFFICIENT", {}
    best_code_all16 = best([row for row in all16 if row["head_family"] == "CODE"])
    best_hh_all16 = best([row for row in all16 if row["head_family"] == "HH"])
    best_code_hh = best([row for row in hh if row["head_family"] == "CODE"])
    best_hh_hh = best([row for row in hh if row["head_family"] == "HH"])
    code_adv = bool(best_code_all16 and best_hh_all16 and (
        best_code_all16["metrics"]["top1_tournament_acc"] >= best_hh_all16["metrics"]["top1_tournament_acc"] + 0.10
        or best_code_all16["metrics"]["pairwise_acc"] >= best_hh_all16["metrics"]["pairwise_acc"] + 0.10
    ))
    hh_adv = bool(best_code_hh and best_hh_hh and best_hh_hh["metrics"]["pairwise_acc"] >= best_code_hh["metrics"]["pairwise_acc"] + 0.05)
    if code_adv and hh_adv:
        gs = "DOMAIN_SPECIALISTS_NEEDED"
    elif best_code_all16 and best_hh_all16 and best_hh_all16["metrics"]["pairwise_acc"] >= best_code_all16["metrics"]["pairwise_acc"] - 0.05 and not hh_adv:
        gs = "GENERAL_HEAD_SUFFICIENT"
    elif code_adv:
        gs = "MIXED_SHARED_AXIS"
    else:
        gs = "INSUFFICIENT"
    details = {
        "CODE_SPECIFIC_ADVANTAGE_ON_STRICT_CLEAN": code_adv,
        "HH_GENERAL_ADVANTAGE_ON_HH": hh_adv,
        "SHARED_COHERENCE_AXIS": "supported" if not hh_adv and code_adv else ("weak" if code_adv else "unsupported"),
        "best_code_all16": compact(best_code_all16),
        "best_hh_all16": compact(best_hh_all16),
        "best_code_hh": compact(best_code_hh),
        "best_hh_hh": compact(best_hh_hh),
    }
    return "READY", gs, details


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# BG Fixed-Config Cross-Domain Audit",
        "",
        f"FIXED_CONFIG_AUDIT_VERDICT = {payload['fixed_config_audit_verdict']}",
        f"GENERALIST_SPECIALIST_VERDICT = {payload['generalist_specialist_verdict']}",
        "",
        "## Fixed-Config Table",
        "",
        "| domain | family | config | architecture | top1 | over_random | pairwise | cycle | margin_mean | margin_std |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["fixed_config_rows"]:
        m = row["metrics"]
        lines.append(f"| `{row['domain']}` | `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | {rate(m['top1_tournament_acc'])} | {rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | {rate(m.get('cycle_rate'))} | {rate(m.get('margin_mean'))} | {rate(m.get('margin_std'))} |")
    lines.extend(["", "## Best Per Domain", "", f"`{s['best_per_domain']}`", "", "## Stability", ""])
    for row in s["stability"][:12]:
        lines.append(f"- `{row['head_key']}` within_0.05=`{row['domains_within_0p05_of_best']}/{row['domains_seen']}` avg_pairwise=`{row['avg_pairwise']:.3f}`")
    lines.extend(["", "## Specialist / Generalist", "", f"`{payload['specialist_details']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    registry = torch.load(output_path(args.heads), map_location="cpu", weights_only=False)
    matrix = load_json(output_path(args.matrix))
    heads = reconstruct_heads(registry)
    rows = []
    for spec in matrix.get("eval_sets", []) or []:
        if spec.get("feature_status") != "READY":
            continue
        domain = spec["domain"]
        feature_path = PROJECT_ROOT / spec["feature_path"]
        if spec["domain_type"] == "hh_pairs":
            rows.extend(evaluate_hh(heads, feature_path, device))
        elif spec["domain_type"] == "records_pt":
            rows.extend(evaluate_records(domain, records_from_pt(feature_path), heads, device))
        elif spec["domain_type"] == "candidate_features_eval_set":
            rows.extend(evaluate_records(domain, records_from_candidate_features(feature_path, spec["eval_set_name"]), heads, device))
    verdict, gs_verdict, details = verdicts(rows)
    summary = summarize(rows)
    summary.update({
        "FIXED_CONFIG_AUDIT_VERDICT": verdict,
        "GENERALIST_SPECIALIST_VERDICT": gs_verdict,
        "all_domain_average_pairwise": mean(float(row["metrics"]["pairwise_acc"]) for row in rows) if rows else float("nan"),
    })
    payload = {
        "fixed_config_audit_verdict": verdict,
        "generalist_specialist_verdict": gs_verdict,
        "summary": summary,
        "specialist_details": details,
        "fixed_config_rows": [row for row in rows if row["fixed_config"]],
        "all_rows": rows,
        "outputs": {"json": repo_path(output_path(args.output)), "md": repo_path(output_path(args.output_md))},
    }
    write_json(output_path(args.output), payload)
    write_md(output_path(args.output_md), payload)
    print(f"FIXED_CONFIG_AUDIT_VERDICT = {verdict}")
    print(f"GENERALIST_SPECIALIST_VERDICT = {gs_verdict}")
    print(f"Wrote {output_path(args.output)}")
    print(f"Wrote {output_path(args.output_md)}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
