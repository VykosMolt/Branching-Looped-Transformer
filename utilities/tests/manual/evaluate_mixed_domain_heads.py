"""Evaluate mixed-domain heads against cached HH/code/science/reasoning baselines."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402
from evaluate_hh_transfer_on_clean_gsm8k_extreme import build_hh_features  # noqa: E402
from math_bg_probe_lib import config_vector  # noqa: E402
from train_code_specific_tiny_heads_and_eval import HEAD_CLASSES, evaluate_matrices, score_matrix  # noqa: E402


SPLITS_JSON = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "mixed_tap_features_2026-05-17.pt"
MIXED_HEADS_PT = REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.pt"
REGISTRY_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "mixed_domain_head_evaluation_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "mixed_domain_head_evaluation_2026-05-17.md"

FIXED_CONFIGS = {
    "24_L4",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_concat_all_loops",
}
FIXED_ROWS = {
    ("24_L4", "AntisymLinear"),
    ("24_L4", "AntisymLinearNoNorm"),
    ("36_L4", "AntisymLinear"),
    ("36_L4", "AntisymLinearNoNorm"),
    ("36_mean", "AntisymLinear"),
    ("36_mean", "AntisymLinearNoNorm"),
    ("47_L4", "AntisymLinearNoNorm"),
    ("47_concat_all_loops", "AntisymLinear"),
    ("47_concat_all_loops", "AntisymLinearNoNorm"),
}
OBJECTIVE_REGRET_EVALS = {
    "CODE_STRICT_CLEAN_ALL16",
    "CODE_RUNNABLE_DIAGNOSTIC",
    "REASONING_NATURAL_DISTRACTOR",
    "REASONING_TRACE",
    "SCIENCE_OVERALL",
    "CLEAN_GSM8K_EXPANDED",
}
SCIENCE_EVAL_NAMES = {
    "biology": "SCIENCE_BIOLOGY",
    "chemistry": "SCIENCE_CHEMISTRY",
    "medicine": "SCIENCE_MEDICINE",
    "general_science": "SCIENCE_GENERAL",
    "other_science": "SCIENCE_OTHER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--mixed-heads", default=str(MIXED_HEADS_PT))
    parser.add_argument("--registry", default=str(REGISTRY_PT))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(output_path(path), map_location="cpu", weights_only=False)


def feature_map(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32)
        for row in payload.get("candidate_features", []) or []
    }


def labels_tensor(labels: Any) -> torch.Tensor:
    if hasattr(labels, "to"):
        return labels.to(torch.bool).detach().cpu()
    vals = []
    for label in labels:
        if isinstance(label, bool):
            vals.append(label)
        else:
            vals.append(str(label) == "correct")
    return torch.tensor(vals, dtype=torch.bool)


def records_from_rows(rows: Sequence[dict[str, Any]], pooled_by_uid: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        uids = [str(uid) for uid in row.get("candidate_uids", [])]
        labels = [str(label) for label in row.get("labels", [])]
        records.append(
            {
                "tournament_id": int(row.get("tournament_id", idx)),
                "task_id": str(row.get("task_id", idx)),
                "source": str(row.get("source") or row.get("dataset") or row.get("source_dataset") or "unknown"),
                "subdomain_bucket": str(row.get("subdomain_bucket", "")),
                "n_options": int(row.get("n_options", len(uids))),
                "candidate_uids": uids,
                "label_names": labels,
                "labels": torch.tensor([label == "correct" for label in labels], dtype=torch.bool),
                "pooled": torch.stack([pooled_by_uid[uid] for uid in uids], dim=0),
            }
        )
    return records


def records_from_pt_records(records_raw: Sequence[dict[str, Any]], indices: Sequence[int]) -> list[dict[str, Any]]:
    records = []
    for pos in indices:
        row = records_raw[pos]
        labels = labels_tensor(row.get("labels", []))
        n = int(row["pooled"].shape[0])
        records.append(
            {
                "tournament_id": int(row.get("tournament_id", pos)),
                "task_id": str(row.get("task_id") or row.get("problem_id") or pos),
                "source": str(row.get("source", "unknown")),
                "subdomain_bucket": str(row.get("subdomain_bucket", "")),
                "n_options": n,
                "candidate_uids": [f"{row.get('task_id') or row.get('problem_id') or pos}::{i}" for i in range(n)],
                "label_names": ["correct" if bool(x) else "incorrect" for x in labels.tolist()],
                "labels": labels,
                "pooled": row["pooled"].detach().cpu().to(torch.float32),
            }
        )
    return records


def random_top1(records: Sequence[dict[str, Any]]) -> float:
    vals = [float(row["labels"].to(torch.float32).mean()) for row in records]
    return float(mean(vals)) if vals else float("nan")


def config_features(records: Sequence[dict[str, Any]], config: str) -> list[torch.Tensor]:
    return [torch.stack([config_vector(pooled, config) for pooled in row["pooled"]], dim=0).to(torch.float32) for row in records]


def pairwise_comparison_count(records: Sequence[dict[str, Any]]) -> int:
    total = 0
    for row in records:
        labels = row["labels"]
        n_correct = int(labels.sum().item())
        total += n_correct * (int(labels.numel()) - n_correct)
    return total


def filter_records(records: Sequence[dict[str, Any]], pred) -> list[dict[str, Any]]:
    return [row for row in records if pred(row)]


def breakdowns(matrices: list[torch.Tensor], records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("source", "subdomain_bucket", "n_options"):
        vals = sorted({str(row.get(key, "")) for row in records if str(row.get(key, ""))})
        sub: dict[str, Any] = {}
        for val in vals:
            idx = [i for i, row in enumerate(records) if str(row.get(key, "")) == val]
            if not idx:
                continue
            rows = [records[i] for i in idx]
            mats = [matrices[i] for i in idx]
            metric = evaluate_matrices(mats, rows, random_top1(rows))
            metric["n_pairs"] = pairwise_comparison_count(rows)
            sub[val] = metric
        out[key] = sub
    return out


@torch.no_grad()
def evaluate_records(eval_name: str, records: list[dict[str, Any]], heads: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    baseline = random_top1(records)
    for info in heads:
        feats_by_record = config_features(records, info["config"])
        head = info["head"].to(device)
        matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
        head = head.to("cpu")
        metrics = evaluate_matrices(matrices, records, baseline)
        metrics["random_top1_baseline"] = baseline
        metrics["n_pairs"] = pairwise_comparison_count(records)
        row = {
            "eval_set": eval_name,
            "head_group": info["head_group"],
            "architecture": info["architecture"],
            "family_architecture": info["family_architecture"],
            "config": info["config"],
            "fixed_config": info["config"] in FIXED_CONFIGS,
            "fixed_row_requested": (info["config"], info["architecture"]) in FIXED_ROWS,
            "metrics": metrics,
            "breakdowns": breakdowns(matrices, records),
        }
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_hh(eval_name: str, indices: list[int], heads: list[dict[str, Any]], hh_path: str, device: torch.device) -> list[dict[str, Any]]:
    payload = load_pt(hh_path)
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rows = []
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
        acc = float((scores > 0).to(torch.float32).mean()) if scores.numel() else float("nan")
        rows.append(
            {
                "eval_set": eval_name,
                "head_group": info["head_group"],
                "architecture": info["architecture"],
                "family_architecture": info["family_architecture"],
                "config": config,
                "fixed_config": config in FIXED_CONFIGS,
                "fixed_row_requested": (config, info["architecture"]) in FIXED_ROWS,
                "metrics": {
                    "n_tournaments": len(indices),
                    "n_pairs": len(indices),
                    "random_top1_baseline": 0.5,
                    "top1_tournament_acc": acc,
                    "top1_over_random_baseline": acc - 0.5,
                    "pairwise_acc": acc,
                    "condorcet_winner_rate": float("nan"),
                    "cycle_rate": float("nan"),
                    "margin_mean": float(scores.mean()) if scores.numel() else float("nan"),
                    "margin_std": float(scores.std(unbiased=False)) if scores.numel() else float("nan"),
                    "canonical_accuracy": acc,
                    "flipped_accuracy": float((reverse < 0).to(torch.float32).mean()) if reverse.numel() else float("nan"),
                    "antisymmetry_mean_abs": float((scores + reverse).abs().mean()) if scores.numel() else float("nan"),
                },
                "breakdowns": {},
            }
        )
    return rows


def reconstruct_registry(path: str | Path) -> list[dict[str, Any]]:
    payload = load_pt(path)
    heads = []
    for row in payload.get("heads", []) or []:
        head = HEAD_CLASSES[row["architecture"]](int(row["dim"]))
        head.load_state_dict(row["state_dict"])
        head.eval()
        group = "HH" if row["head_family"] == "HH" else "CODE"
        heads.append(
            {
                "head_group": group,
                "architecture": row["architecture"],
                "family_architecture": f"{group}_{row['architecture']}",
                "config": row["config"],
                "dim": int(row["dim"]),
                "head": head,
            }
        )
    return heads


def reconstruct_mixed(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_pt(path)
    heads = []
    for row in payload.get("heads", []) or []:
        head = HEAD_CLASSES[row["architecture"]](int(row["dim"]))
        head.load_state_dict(row["state_dict"])
        head.eval()
        heads.append(
            {
                "head_group": row["head_group"],
                "architecture": row["architecture"],
                "family_architecture": row["family_architecture"],
                "config": row["config"],
                "dim": int(row["dim"]),
                "head": head,
            }
        )
    return heads, payload.get("meta", {})


def best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row["metrics"].get("pairwise_acc", float("-inf"))),
            float(row["metrics"].get("top1_tournament_acc", float("-inf"))),
            -float(row["metrics"].get("cycle_rate", 0.0) or 0.0),
        ),
    )


def compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {
        "eval_set": row["eval_set"],
        "head_group": row["head_group"],
        "config": row["config"],
        "architecture": row["architecture"],
        "top1": m.get("top1_tournament_acc"),
        "pairwise": m.get("pairwise_acc"),
        "cycle": m.get("cycle_rate"),
        "random_top1_baseline": m.get("random_top1_baseline"),
        "n_pairs": m.get("n_pairs"),
        "margin_mean": m.get("margin_mean"),
        "margin_std": m.get("margin_std"),
    }


def build_eval_sets(splits: dict[str, Any], features: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pooled_by_uid = feature_map(features)
    eval_sets: dict[str, list[dict[str, Any]]] = {}
    domains = splits["domains"]
    if "CODE" in domains:
        eval_sets["CODE_STRICT_CLEAN_ALL16"] = records_from_rows(domains["CODE"]["eval_sets"]["CODE_STRICT_CLEAN_ALL16"]["rows"], pooled_by_uid)
    if "REASONING_NATURAL" in domains:
        eval_sets["REASONING_NATURAL_DISTRACTOR"] = records_from_rows(domains["REASONING_NATURAL"]["eval_sets"]["REASONING_NATURAL"]["rows"], pooled_by_uid)
    if "REASONING_TRACE" in domains:
        eval_sets["REASONING_TRACE"] = records_from_rows(domains["REASONING_TRACE"]["eval_sets"]["REASONING_TRACE"]["rows"], pooled_by_uid)
    if "SCIENCE" in domains:
        science_records = records_from_rows(domains["SCIENCE"]["eval_sets"]["SCIENCE"]["rows"], pooled_by_uid)
        eval_sets["SCIENCE_OVERALL"] = science_records
        for bucket, eval_name in SCIENCE_EVAL_NAMES.items():
            rows = filter_records(science_records, lambda row, b=bucket: row.get("subdomain_bucket") == b)
            if rows:
                eval_sets[eval_name] = rows
    record_domains = features.get("record_domains", {})
    if "CODE_RUNNABLE" in record_domains and "CODE_RUNNABLE" in domains:
        idx = domains["CODE_RUNNABLE"]["eval_sets"]["CODE_RUNNABLE_DIAGNOSTIC"]["record_indices"]
        eval_sets["CODE_RUNNABLE_DIAGNOSTIC"] = records_from_pt_records(record_domains["CODE_RUNNABLE"], idx)
    if "GSM8K" in record_domains and "GSM8K" in domains and splits.get("gsm8k_eval_status") in {"READY", "RECAPTURED"}:
        idx = domains["GSM8K"]["eval_sets"]["CLEAN_GSM8K_EXPANDED"]["record_indices"]
        eval_sets["CLEAN_GSM8K_EXPANDED"] = records_from_pt_records(record_domains["GSM8K"], idx)
    return eval_sets


def regret_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_names = sorted({row["eval_set"] for row in rows})
    regrets: dict[str, Any] = {}
    mixed_family_names = sorted({row["head_group"] for row in rows if str(row["head_group"]).startswith("MIX_")})
    for eval_name in eval_names:
        eval_rows = [row for row in rows if row["eval_set"] == eval_name]
        existing = [row for row in eval_rows if row["head_group"] in {"HH", "CODE", "SCIENCE"}]
        mixed = [row for row in eval_rows if str(row["head_group"]).startswith("MIX_")]
        best_existing = best(existing)
        best_mixed = best(mixed)
        if not best_existing or not best_mixed:
            continue
        family_regrets: dict[str, Any] = {}
        for family in mixed_family_names:
            family_best = best([row for row in mixed if row["head_group"] == family])
            if not family_best:
                continue
            family_regrets[family] = {
                "best_family_row": compact(family_best),
                "regret_pairwise": family_best["metrics"]["pairwise_acc"] - best_existing["metrics"]["pairwise_acc"],
                "regret_top1": family_best["metrics"]["top1_tournament_acc"] - best_existing["metrics"]["top1_tournament_acc"],
            }
        regrets[eval_name] = {
            "best_existing_specialist": compact(best_existing),
            "best_mixed": compact(best_mixed),
            "regret_pairwise": best_mixed["metrics"]["pairwise_acc"] - best_existing["metrics"]["pairwise_acc"],
            "regret_top1": best_mixed["metrics"]["top1_tournament_acc"] - best_existing["metrics"]["top1_tournament_acc"],
            "family_regrets": family_regrets,
            "domain_regret_noisy": abs(best_mixed["metrics"]["pairwise_acc"] - best_existing["metrics"]["pairwise_acc"]) <= 0.05
            and int(best_existing["metrics"].get("n_pairs", 9999)) < 75,
        }
    objective_vals = [regrets[name]["regret_pairwise"] for name in OBJECTIVE_REGRET_EVALS if name in regrets]
    strict = regrets.get("CODE_STRICT_CLEAN_ALL16", {}).get("regret_pairwise")
    if strict is None:
        strict_status = "NOT_APPLICABLE"
    elif -0.05 <= strict <= 0.05:
        strict_status = "BORDERLINE_AT_N16"
    elif strict > 0.05:
        strict_status = "CLEAN_WIN"
    else:
        strict_status = "CLEAN_LOSS"
    return {
        "per_eval": regrets,
        "average_objective_regret_pairwise": float(mean(objective_vals)) if objective_vals else float("nan"),
        "worst_objective_regret_pairwise": min(objective_vals) if objective_vals else float("nan"),
        "strict_clean_code_regret_pairwise": strict,
        "strict_clean_code_regret_status": strict_status,
        "hh_regret_pairwise": regrets.get("HH_HELDOUT20", {}).get("regret_pairwise"),
        "science_medicine_regret_pairwise": regrets.get("SCIENCE_MEDICINE", {}).get("regret_pairwise"),
        "reasoning_trace_regret_pairwise": regrets.get("REASONING_TRACE", {}).get("regret_pairwise"),
        "domain_regret_noisy": any(item.get("domain_regret_noisy") for item in regrets.values()),
    }


def verdicts(regret: dict[str, Any]) -> tuple[str, dict[str, str], bool]:
    per_eval = regret.get("per_eval", {})
    if len(per_eval) < 4:
        return "INSUFFICIENT", {}, False
    avg_obj = regret.get("average_objective_regret_pairwise", float("nan"))
    worst_obj = regret.get("worst_objective_regret_pairwise", float("nan"))
    strict = regret.get("strict_clean_code_regret_pairwise")
    strict_status = regret.get("strict_clean_code_regret_status")
    hh_reg = regret.get("hh_regret_pairwise")
    non_code_gain = any(
        item.get("regret_pairwise", -999) >= 0.05
        for name, item in per_eval.items()
        if not name.startswith("CODE") and not name.startswith("HH")
    )
    provisional = False
    if not math.isnan(avg_obj) and avg_obj >= -0.03 and worst_obj >= -0.08 and strict is not None and strict >= -0.05 and non_code_gain:
        main = "OBJECTIVE_MIXED_USEFUL"
        provisional = strict_status == "BORDERLINE_AT_N16"
    elif (not math.isnan(avg_obj) and avg_obj > 0 and strict is not None and strict < -0.05) or (hh_reg is not None and hh_reg < -0.10):
        main = "MIXED_DILUTES_SPECIALISTS"
    elif hh_reg is not None and hh_reg >= -0.05 and not math.isnan(avg_obj) and avg_obj >= -0.05 and strict is not None and strict >= -0.05:
        main = "MIXED_HH_OBJECTIVE_USEFUL"
    else:
        main = "SPECIALISTS_WIN"

    family_verdicts: dict[str, str] = {}
    families = sorted(
        {
            family
            for item in per_eval.values()
            for family in (item.get("family_regrets") or {}).keys()
        }
    )
    for family in families:
        fam_items = [
            item["family_regrets"][family]["regret_pairwise"]
            for item in per_eval.values()
            if family in (item.get("family_regrets") or {})
        ]
        fam_objective = [
            item["family_regrets"][family]["regret_pairwise"]
            for name, item in per_eval.items()
            if name in OBJECTIVE_REGRET_EVALS and family in (item.get("family_regrets") or {})
        ]
        fam_strict = per_eval.get("CODE_STRICT_CLEAN_ALL16", {}).get("family_regrets", {}).get(family, {}).get("regret_pairwise")
        fam_hh = per_eval.get("HH_HELDOUT20", {}).get("family_regrets", {}).get(family, {}).get("regret_pairwise")
        fam_non_code_gain = any(
            item["family_regrets"][family]["regret_pairwise"] >= 0.05
            for name, item in per_eval.items()
            if not name.startswith("CODE") and not name.startswith("HH") and family in (item.get("family_regrets") or {})
        )
        if len(fam_items) < 4:
            family_verdicts[family] = "INSUFFICIENT"
        elif (fam_strict is not None and fam_strict < -0.05) or (family == "MIX_HH_OBJECTIVE" and fam_hh is not None and fam_hh < -0.10):
            family_verdicts[family] = "DILUTED"
        elif fam_objective and mean(fam_objective) >= -0.03 and min(fam_objective) >= -0.08 and fam_non_code_gain:
            family_verdicts[family] = "USEFUL"
        elif any(abs(x) <= 0.05 for x in fam_items):
            family_verdicts[family] = "BORDERLINE"
        elif min(fam_items) < -0.08:
            family_verdicts[family] = "SPECIALIST_BEATS"
        else:
            family_verdicts[family] = "SPECIALIST_BEATS"
    return main, family_verdicts, provisional


def summarize(rows: list[dict[str, Any]], regret: dict[str, Any]) -> dict[str, Any]:
    eval_sets = sorted({row["eval_set"] for row in rows})
    best_per_eval = {name: compact(best([row for row in rows if row["eval_set"] == name])) for name in eval_sets}
    best_per_group: dict[str, Any] = {}
    for group in sorted({row["head_group"] for row in rows}):
        group_rows = [row for row in rows if row["head_group"] == group]
        best_per_group[group] = compact(best(group_rows))
    fixed_rows = [row for row in rows if row["fixed_row_requested"]]
    return {
        "eval_sets": eval_sets,
        "best_per_eval": best_per_eval,
        "best_per_group": best_per_group,
        "fixed_config_rows": [compact(row) for row in fixed_rows],
        "regret_summary": regret,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines = [
        "# Mixed-Domain Head Evaluation (2026-05-17)",
        "",
        f"MIXED_HEAD_UTILITY_VERDICT = {meta['mixed_head_utility_verdict']}",
        f"MIXED_HEAD_UTILITY_PROVISIONAL = {meta['mixed_head_utility_provisional']}",
        f"STRICT_CLEAN_CODE_REGRET_STATUS = {meta['strict_clean_code_regret_status']}",
        f"DOMAIN_REGRET_NOISY = {meta['domain_regret_noisy']}",
        "",
        "Borderline regret at current n should be treated as directionally informative, not decisive.",
        "",
        "## Best Per Eval",
    ]
    for name, row in payload["summary"]["best_per_eval"].items():
        lines.append(f"- {name}: `{json.dumps(row, default=str)}`")
    lines.extend(["", "## Regret Table"])
    for name, item in payload["summary"]["regret_summary"]["per_eval"].items():
        lines.append(
            f"- {name}: pairwise_regret={item['regret_pairwise']:.3f}, top1_regret={item['regret_top1']:.3f}, "
            f"best_mixed={item['best_mixed']['head_group']} {item['best_mixed']['config']} {item['best_mixed']['architecture']}, "
            f"best_existing={item['best_existing_specialist']['head_group']} {item['best_existing_specialist']['config']} {item['best_existing_specialist']['architecture']}"
        )
    lines.extend(["", "## Per-Family Verdicts"])
    for family, verdict in payload["per_family_verdicts"].items():
        lines.append(f"- {family}: {verdict}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    splits = load_json(args.splits)
    features = load_pt(args.features)
    mixed_heads, mixed_meta = reconstruct_mixed(args.mixed_heads)
    registry_heads = reconstruct_registry(args.registry)
    heads = registry_heads + mixed_heads
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    eval_sets = build_eval_sets(splits, features)
    rows: list[dict[str, Any]] = []
    for name, records in eval_sets.items():
        rows.extend(evaluate_records(name, records, heads, device))
    hh_domain = splits.get("domains", {}).get("HH")
    if hh_domain:
        hh_path = hh_domain["feature_path"]
        rows.extend(evaluate_hh("HH_HELDOUT20", list(hh_domain.get("eval_indices", [])), heads, hh_path, device))
        rows.extend(evaluate_hh("HH_200_DIAGNOSTIC", list(hh_domain.get("diagnostic_eval_indices", [])), heads, hh_path, device))

    regret = regret_tables(rows)
    main_verdict, family_verdicts, provisional = verdicts(regret)
    meta = {
        "mixed_head_utility_verdict": main_verdict,
        "mixed_head_utility_provisional": provisional,
        "strict_clean_code_regret_status": regret["strict_clean_code_regret_status"],
        "domain_regret_noisy": regret["domain_regret_noisy"],
        "average_objective_regret_pairwise": regret["average_objective_regret_pairwise"],
        "worst_objective_regret_pairwise": regret["worst_objective_regret_pairwise"],
        "strict_clean_code_regret_pairwise": regret["strict_clean_code_regret_pairwise"],
        "hh_regret_pairwise": regret["hh_regret_pairwise"],
        "mixed_training_meta": mixed_meta,
        "n_rows": len(rows),
    }
    payload = {
        "meta": meta,
        "summary": summarize(rows, regret),
        "per_family_verdicts": family_verdicts,
        "rows": rows,
    }
    write_json(output_path(args.output), payload)
    write_markdown(output_path(args.output_md), payload)
    print(f"MIXED_HEAD_UTILITY_VERDICT = {main_verdict}")
    print(f"STRICT_CLEAN_CODE_REGRET_STATUS = {regret['strict_clean_code_regret_status']}")
    print(f"average_objective_regret_pairwise = {regret['average_objective_regret_pairwise']:.3f}")
    print(f"wrote {repo_path(output_path(args.output))}")


if __name__ == "__main__":
    main()
