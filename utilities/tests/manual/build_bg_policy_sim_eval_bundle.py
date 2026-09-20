"""Build a unified read-only eval bundle for BG controller-policy simulation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json  # noqa: E402
from evaluate_hh_transfer_on_clean_gsm8k_extreme import build_hh_features  # noqa: E402
from evaluate_mixed_domain_heads import (  # noqa: E402
    build_eval_sets,
    config_features,
    load_json,
    load_pt,
    reconstruct_mixed,
    reconstruct_registry,
    score_matrix,
)
from math_bg_probe_lib import config_vector  # noqa: E402


SPLITS_JSON = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "mixed_tap_features_2026-05-17.pt"
REGISTRY_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
MIXED_HEADS_PT = REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.pt"
MIXED_EVAL_JSON = REPORT_DIR / "mixed_domain_head_evaluation_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.md"

OBJECTIVE_DOMAINS = {
    "CLEAN_GSM8K_EXPANDED",
    "CODE_RUNNABLE_DIAGNOSTIC",
    "CODE_STRICT_CLEAN_ALL16",
    "REASONING_NATURAL_DISTRACTOR",
    "REASONING_TRACE",
    "SCIENCE_OVERALL",
    "SCIENCE_BIOLOGY",
    "SCIENCE_CHEMISTRY",
    "SCIENCE_MEDICINE",
    "SCIENCE_GENERAL",
    "SCIENCE_OTHER",
}
SCIENCE_DOMAINS = {
    "SCIENCE_OVERALL",
    "SCIENCE_BIOLOGY",
    "SCIENCE_CHEMISTRY",
    "SCIENCE_MEDICINE",
    "SCIENCE_GENERAL",
    "SCIENCE_OTHER",
}

ROLE_REQUESTS = {
    "HH_GENERAL": [
        ("HH", "47_concat_L1_L4", "AntisymLinearNoNorm"),
        ("HH", "47_mean", "AntisymLinear"),
        ("HH", "36_mean", "AntisymLinear"),
        ("HH", "36_mean", "AntisymLinearNoNorm"),
    ],
    "CODE_SPECIALIST": [
        ("CODE", "36_L4", "AntisymLinear"),
        ("CODE", "24_L4", "AntisymLinear"),
        ("CODE", "36_mean", "AntisymLinear"),
        ("CODE", "36_L4", "AntisymLinearNoNorm"),
    ],
    "OBJECTIVE_MIXED_PRIMARY": [
        ("MIX_CODE_REASONING", "36_L4", "AntisymLinearNoNorm"),
        ("MIX_CODE_REASONING", "36_L4", "AntisymLinear"),
        ("MIX_CODE_REASONING", "24_L4", "AntisymLinear"),
    ],
    "OBJECTIVE_MIXED_BROAD": [
        ("MIX_OBJECTIVE_ALL", "36_L4", "AntisymLinearNoNorm"),
        ("MIX_OBJECTIVE_ALL", "36_mean", "AntisymLinearNoNorm"),
        ("MIX_OBJECTIVE_ALL", "36_L4", "AntisymLinear"),
    ],
    "SCIENCE_AWARE_MIXED": [
        ("MIX_CODE_SCIENCE", "36_L4", "AntisymLinearNoNorm"),
        ("MIX_CODE_SCIENCE", "36_mean", "AntisymLinearNoNorm"),
        ("MIX_CODE_SCIENCE_MED", "36_L4", "AntisymLinearNoNorm"),
        ("MIX_CODE_SCIENCE_MED", "36_mean", "AntisymLinearNoNorm"),
    ],
    "RISKY_HH_OBJECTIVE": [
        ("MIX_HH_OBJECTIVE", "36_L4", "AntisymLinearNoNorm"),
        ("MIX_HH_OBJECTIVE", "24_L1", "AntisymLinear"),
    ],
}
FIXED_ROLE_PRIMARY = {
    "HH_GENERAL": ("HH", "47_concat_L1_L4", "AntisymLinearNoNorm"),
    "CODE_SPECIALIST": ("CODE", "36_L4", "AntisymLinear"),
    "OBJECTIVE_MIXED_PRIMARY": ("MIX_CODE_REASONING", "36_L4", "AntisymLinearNoNorm"),
    "OBJECTIVE_MIXED_BROAD": ("MIX_OBJECTIVE_ALL", "36_L4", "AntisymLinearNoNorm"),
    "SCIENCE_AWARE_MIXED": ("MIX_CODE_SCIENCE_MED", "36_L4", "AntisymLinearNoNorm"),
    "RISKY_HH_OBJECTIVE": ("MIX_HH_OBJECTIVE", "36_L4", "AntisymLinearNoNorm"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--registry", default=str(REGISTRY_PT))
    parser.add_argument("--mixed-heads", default=str(MIXED_HEADS_PT))
    parser.add_argument("--mixed-eval", default=str(MIXED_EVAL_JSON))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def head_key(info: dict[str, Any]) -> str:
    return f"{info['head_group']}::{info['config']}::{info['architecture']}"


def role_tuple_key(row: tuple[str, str, str]) -> str:
    family, config, architecture = row
    return f"{family}::{config}::{architecture}"


def domain_kind(name: str) -> str:
    if name.startswith("HH_"):
        return "pair"
    if name == "CODE_STRICT_CLEAN_ALL16":
        return "code_strict_clean"
    if name == "CODE_RUNNABLE_DIAGNOSTIC":
        return "code_diagnostic"
    if name.startswith("SCIENCE_") or name.startswith("REASONING_"):
        return "MCQ"
    return "tournament"


def safe_mean(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(float(v))]
    return float(mean(vals)) if vals else float("nan")


def margin_stats(vals: list[float]) -> dict[str, float]:
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return {
        "mean": float(mean(vals)) if vals else float("nan"),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
    }


def record_pairs(labels: torch.Tensor) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, ok in enumerate(labels.tolist()):
        if not bool(ok):
            continue
        for j, other in enumerate(labels.tolist()):
            if i != j and not bool(other):
                pairs.append((i, j))
    return pairs


def random_top1(records: list[dict[str, Any]]) -> float:
    vals = [float(row["labels"].to(torch.float32).mean()) for row in records if int(row["labels"].numel()) > 0]
    return float(mean(vals)) if vals else float("nan")


def compact_records(domain: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        labels = row["labels"].detach().cpu().to(torch.bool)
        pairs = record_pairs(labels)
        rec = {
            "record_id": f"{domain}::{idx}",
            "record_index": idx,
            "task_id": str(row.get("task_id", idx)),
            "source": str(row.get("source", "")),
            "subdomain_bucket": str(row.get("subdomain_bucket", "")),
            "n_options": int(labels.numel()),
            "candidate_uids": [str(uid) for uid in row.get("candidate_uids", [])],
            "label_names": list(row.get("label_names", [])),
            "labels": [bool(x) for x in labels.tolist()],
            "pair_indices": pairs,
            "pair_total": len(pairs),
            "eval_type": domain_kind(domain),
        }
        out.append(rec)
    return out


def score_record_from_matrix(mat: torch.Tensor, labels: torch.Tensor, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    labels = labels.detach().cpu().to(torch.bool)
    pair_scores = [float(mat[i, j]) for i, j in pairs]
    pair_correct = sum(1 for score in pair_scores if score > 0)
    totals = mat.sum(dim=1)
    pred = int(torch.argmax(totals).item()) if totals.numel() else -1
    top1_correct = bool(labels[pred].item()) if pred >= 0 and labels.numel() else False
    if totals.numel() >= 2:
        top2 = torch.topk(totals, k=2).values
        margin = float(top2[0] - top2[1])
    elif pair_scores:
        margin = abs(pair_scores[0])
    else:
        margin = 0.0
    return {
        "pair_scores": pair_scores,
        "pair_correct": int(pair_correct),
        "pair_total": int(len(pair_scores)),
        "pair_decisions": [score > 0 for score in pair_scores],
        "pred_index": pred,
        "top1_correct": top1_correct,
        "margin": float(abs(margin)),
    }


@torch.no_grad()
def score_tournament_domain(
    records: list[dict[str, Any]],
    compact: list[dict[str, Any]],
    head_info: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    feats_by_record = config_features(records, head_info["config"])
    head = head_info["head"].to(device)
    matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
    head = head.to("cpu")
    scored_records = [
        score_record_from_matrix(mat, row["labels"], compact_row["pair_indices"])
        for mat, row, compact_row in zip(matrices, records, compact)
    ]
    return summarize_head_domain(scored_records, random_top1(records))


@torch.no_grad()
def score_hh_domain(
    payload: dict[str, Any],
    indices: list[int],
    head_info: dict[str, Any],
    hh_feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    config = head_info["config"]
    if config not in hh_feature_cache:
        hh_feature_cache[config] = build_hh_features(payload, config)
    chosen, rejected = hh_feature_cache[config]
    head = head_info["head"].to(device)
    left = chosen[indices].to(device)
    right = rejected[indices].to(device)
    scores = head(left, right).detach().cpu()
    head = head.to("cpu")
    scored_records = []
    for score in scores.tolist():
        scored_records.append(
            {
                "pair_scores": [float(score)],
                "pair_correct": int(float(score) > 0),
                "pair_total": 1,
                "pair_decisions": [float(score) > 0],
                "pred_index": 0 if float(score) > 0 else 1,
                "top1_correct": bool(float(score) > 0),
                "margin": abs(float(score)),
            }
        )
    return summarize_head_domain(scored_records, 0.5)


def summarize_head_domain(scored_records: list[dict[str, Any]], baseline: float) -> dict[str, Any]:
    pair_total = sum(int(row["pair_total"]) for row in scored_records)
    pair_correct = sum(int(row["pair_correct"]) for row in scored_records)
    top_total = len(scored_records)
    top_correct = sum(1 for row in scored_records if row.get("top1_correct"))
    margins = [float(row["margin"]) for row in scored_records]
    stats = margin_stats(margins)
    return {
        "records": scored_records,
        "metrics": {
            "pairwise_acc": pair_correct / pair_total if pair_total else float("nan"),
            "top1": top_correct / top_total if top_total else float("nan"),
            "top1_over_random": top_correct / top_total - baseline if top_total else float("nan"),
            "random_top1_baseline": baseline,
            "pair_correct": pair_correct,
            "pair_total": pair_total,
            "record_total": top_total,
            "margin_mean": stats["mean"],
            "margin_std": stats["std"],
        },
    }


def add_candidate_roles(heads_meta: dict[str, dict[str, Any]], mixed_eval: dict[str, Any]) -> dict[str, str]:
    for key, info in heads_meta.items():
        info["role_tags"] = []
        info["candidate_head"] = False
    for role, requests in ROLE_REQUESTS.items():
        for request in requests:
            key = role_tuple_key(request)
            if key in heads_meta:
                heads_meta[key]["role_tags"].append(role)
                heads_meta[key]["candidate_head"] = True

    best_by_group: dict[str, tuple[str, float]] = {}
    objective_domains = {
        "CLEAN_GSM8K_EXPANDED",
        "CODE_RUNNABLE_DIAGNOSTIC",
        "CODE_STRICT_CLEAN_ALL16",
        "REASONING_NATURAL_DISTRACTOR",
        "REASONING_TRACE",
        "SCIENCE_OVERALL",
    }
    rows = mixed_eval.get("rows", []) or []
    for group in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL", "MIX_CODE_SCIENCE", "MIX_CODE_SCIENCE_MED", "MIX_HH_OBJECTIVE"):
        by_key: dict[str, list[float]] = {}
        for row in rows:
            if row.get("head_group") != group or row.get("eval_set") not in objective_domains:
                continue
            key = f"{row['head_group']}::{row['config']}::{row['architecture']}"
            by_key.setdefault(key, []).append(float(row["metrics"]["pairwise_acc"]))
        if by_key:
            best_key, best_score = max(by_key.items(), key=lambda item: safe_mean(item[1]))
            best_by_group[group] = (best_key, safe_mean(best_score))

    group_to_role = {
        "MIX_CODE_REASONING": "OBJECTIVE_MIXED_PRIMARY",
        "MIX_OBJECTIVE_ALL": "OBJECTIVE_MIXED_BROAD",
        "MIX_CODE_SCIENCE": "SCIENCE_AWARE_MIXED",
        "MIX_CODE_SCIENCE_MED": "SCIENCE_AWARE_MIXED",
        "MIX_HH_OBJECTIVE": "RISKY_HH_OBJECTIVE",
    }
    for group, (key, _score) in best_by_group.items():
        if key in heads_meta:
            role = group_to_role[group]
            tag = f"{role}_BEST_SAVED"
            if role not in heads_meta[key]["role_tags"]:
                heads_meta[key]["role_tags"].append(role)
            heads_meta[key]["role_tags"].append(tag)
            heads_meta[key]["candidate_head"] = True

    role_primary: dict[str, str] = {}
    for role, request in FIXED_ROLE_PRIMARY.items():
        key = role_tuple_key(request)
        if key in heads_meta:
            role_primary[role] = key
    return role_primary


def compute_oracles(
    heads_meta: dict[str, dict[str, Any]],
    head_scores: dict[str, dict[str, Any]],
    domains: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle: dict[str, Any] = {}
    existing: dict[str, Any] = {}
    for domain in domains:
        best_key = None
        best_metric = -1.0
        existing_key = None
        existing_metric = -1.0
        for key, by_domain in head_scores.items():
            if domain not in by_domain:
                continue
            metric = float(by_domain[domain]["metrics"]["pairwise_acc"])
            if metric > best_metric:
                best_key, best_metric = key, metric
            if heads_meta[key]["head_group"] in {"HH", "CODE"} and metric > existing_metric:
                existing_key, existing_metric = key, metric
        oracle[domain] = {"head_key": best_key, "pairwise_acc": best_metric}
        existing[domain] = {"head_key": existing_key, "pairwise_acc": existing_metric}
    return oracle, existing


def code_similarity_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in records:
        pooled = row["pooled"].detach().cpu().to(torch.float32)
        if int(pooled.shape[0]) < 2:
            max_cos = float("nan")
            mean_cos = float("nan")
        else:
            vecs = torch.stack([config_vector(item, "36_L4").to(torch.float32) for item in pooled], dim=0)
            vecs = F.normalize(vecs, dim=1)
            sims = []
            for i in range(vecs.shape[0]):
                for j in range(i + 1, vecs.shape[0]):
                    sims.append(float((vecs[i] * vecs[j]).sum()))
            max_cos = max(sims) if sims else float("nan")
            mean_cos = float(mean(sims)) if sims else float("nan")
        out[str(row.get("task_id", len(out)))] = {"max_feature_cosine_36_L4": max_cos, "mean_feature_cosine_36_L4": mean_cos}
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BG Policy Simulator Eval Bundle (2026-05-17)",
        "",
        f"BG_POLICY_EVAL_BUNDLE_VERDICT = {payload['meta']['bg_policy_eval_bundle_verdict']}",
        "",
        "## Domains",
        "| domain | type | records | candidates | pairs | random top1 | feature coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["domain_summary"].items():
        lines.append(
            f"| {name} | {row['eval_type']} | {row['records']} | {row['candidates']} | "
            f"{row['pairs']} | {row['random_top1_baseline']:.3f} | {row['feature_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Heads",
            f"- total heads scored: {payload['meta']['heads_scored']}",
            f"- candidate heads: {payload['meta']['candidate_heads']}",
            f"- head groups: `{json.dumps(payload['meta']['head_groups'])}`",
            f"- score coverage: `{json.dumps(payload['score_coverage'])}`",
        ]
    )
    if payload.get("skipped_domains"):
        lines.extend(["", "## Skipped Domains"])
        for item in payload["skipped_domains"]:
            lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    splits = load_json(args.splits)
    features = load_pt(args.features)
    mixed_eval = load_json(args.mixed_eval)
    registry_heads = reconstruct_registry(args.registry)
    mixed_heads, _ = reconstruct_mixed(args.mixed_heads)
    heads = registry_heads + mixed_heads
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    eval_records = build_eval_sets(splits, features)
    domains: dict[str, Any] = {}
    skipped_domains: list[str] = []
    for domain, records in eval_records.items():
        compact = compact_records(domain, records)
        domains[domain] = {
            "eval_type": domain_kind(domain),
            "objective": domain in OBJECTIVE_DOMAINS,
            "science": domain in SCIENCE_DOMAINS,
            "records": compact,
            "random_top1_baseline": random_top1(records),
            "feature_coverage": 1.0,
        }
        if domain in {"CODE_STRICT_CLEAN_ALL16", "CODE_RUNNABLE_DIAGNOSTIC"}:
            sims = code_similarity_features(records)
            for rec in domains[domain]["records"]:
                rec["contrast_feature"] = sims.get(rec["task_id"], {})

    hh_domain = splits.get("domains", {}).get("HH")
    hh_payload = None
    if hh_domain:
        hh_payload = load_pt(hh_domain["feature_path"])
        for domain, indices_key in (("HH_HELDOUT20", "eval_indices"), ("HH_200_DIAGNOSTIC", "diagnostic_eval_indices")):
            indices = list(hh_domain.get(indices_key, []))
            domains[domain] = {
                "eval_type": "pair",
                "objective": False,
                "science": False,
                "records": [
                    {
                        "record_id": f"{domain}::{i}",
                        "record_index": pos,
                        "pair_index": int(i),
                        "task_id": f"hh/{i}",
                        "source": "hh",
                        "subdomain_bucket": "",
                        "n_options": 2,
                        "candidate_uids": [f"hh/{i}/chosen", f"hh/{i}/rejected"],
                        "label_names": ["chosen", "rejected"],
                        "labels": [True, False],
                        "pair_indices": [(0, 1)],
                        "pair_total": 1,
                        "eval_type": "pair",
                    }
                    for pos, i in enumerate(indices)
                ],
                "random_top1_baseline": 0.5,
                "feature_coverage": 1.0,
                "hh_indices": indices,
            }

    heads_meta = {
        head_key(info): {
            "head_key": head_key(info),
            "head_group": info["head_group"],
            "architecture": info["architecture"],
            "config": info["config"],
            "family_architecture": info["family_architecture"],
            "dim": int(info["dim"]),
            "deployable": not str(info["head_group"]).startswith("MIX_HH_OBJECTIVE"),
        }
        for info in heads
    }
    role_primary = add_candidate_roles(heads_meta, mixed_eval)
    head_by_key = {head_key(info): info for info in heads}

    head_scores: dict[str, dict[str, Any]] = {}
    hh_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, info in head_by_key.items():
        head_scores[key] = {}
        for domain, spec in domains.items():
            if domain.startswith("HH_"):
                if hh_payload is None:
                    continue
                scored = score_hh_domain(hh_payload, spec["hh_indices"], info, hh_cache, device)
            else:
                scored = score_tournament_domain(eval_records[domain], spec["records"], info, device)
            head_scores[key][domain] = scored

    oracle, existing_oracle = compute_oracles(heads_meta, head_scores, domains)
    domain_summary = {}
    for name, spec in domains.items():
        domain_summary[name] = {
            "eval_type": spec["eval_type"],
            "records": len(spec["records"]),
            "candidates": sum(int(row["n_options"]) for row in spec["records"]),
            "pairs": sum(int(row["pair_total"]) for row in spec["records"]),
            "random_top1_baseline": float(spec["random_top1_baseline"]),
            "feature_coverage": float(spec["feature_coverage"]),
        }

    included = set(domains)
    ready = (
        "HH_HELDOUT20" in included
        and "CODE_STRICT_CLEAN_ALL16" in included
        and ({"REASONING_NATURAL_DISTRACTOR", "REASONING_TRACE"} & included)
        and "SCIENCE_OVERALL" in included
        and "CLEAN_GSM8K_EXPANDED" in included
    )
    partial = "HH_HELDOUT20" in included and "CODE_STRICT_CLEAN_ALL16" in included and len(included & OBJECTIVE_DOMAINS) >= 2
    verdict = "READY" if ready else ("PARTIAL" if partial and head_scores else "BLOCKED")
    score_coverage = {
        domain: sum(1 for key in heads_meta if domain in head_scores.get(key, {}))
        for domain in domains
    }
    meta = {
        "bg_policy_eval_bundle_verdict": verdict,
        "splits_json": repo_path(output_path(args.splits)),
        "features_pt": repo_path(output_path(args.features)),
        "registry_pt": repo_path(output_path(args.registry)),
        "mixed_heads_pt": repo_path(output_path(args.mixed_heads)),
        "heads_scored": len(heads_meta),
        "candidate_heads": sum(1 for row in heads_meta.values() if row.get("candidate_head")),
        "head_groups": dict(Counter(row["head_group"] for row in heads_meta.values())),
        "role_primary": role_primary,
        "objective_domains": sorted(included & OBJECTIVE_DOMAINS),
    }
    pt_payload = {
        "meta": meta,
        "domains": domains,
        "domain_summary": domain_summary,
        "heads": heads_meta,
        "role_primary": role_primary,
        "head_scores": head_scores,
        "oracle_domain_best": oracle,
        "existing_specialist_best": existing_oracle,
        "score_coverage": score_coverage,
        "skipped_domains": skipped_domains,
    }
    out = output_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_payload, out)
    json_payload = {
        "meta": meta,
        "domain_summary": domain_summary,
        "heads": heads_meta,
        "role_primary": role_primary,
        "oracle_domain_best": oracle,
        "existing_specialist_best": existing_oracle,
        "score_coverage": score_coverage,
        "skipped_domains": skipped_domains,
    }
    write_json(output_path(args.output_json), json_payload)
    write_markdown(output_path(args.output_md), json_payload)
    print(f"BG_POLICY_EVAL_BUNDLE_VERDICT = {verdict}")
    print(f"domains = {', '.join(sorted(domains))}")
    print(f"heads_scored = {len(heads_meta)}")
    print(f"wrote {repo_path(out)}")


if __name__ == "__main__":
    main()
