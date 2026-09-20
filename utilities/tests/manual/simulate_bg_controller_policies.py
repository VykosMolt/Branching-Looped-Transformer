"""Simulate read-only BG controller policies over cached head scores."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402


BUNDLE_PT = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.pt"
COMPARISON_JSON = REPORT_DIR / "bg_candidate_head_comparison_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "bg_controller_policy_simulation_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_controller_policy_simulation_2026-05-17.md"

OBJECTIVE_DOMAINS = {
    "CLEAN_GSM8K_EXPANDED",
    "CODE_RUNNABLE_DIAGNOSTIC",
    "CODE_STRICT_CLEAN_ALL16",
    "REASONING_NATURAL_DISTRACTOR",
    "REASONING_TRACE",
    "SCIENCE_OVERALL",
}
SCIENCE_DOMAINS = {
    "SCIENCE_OVERALL",
    "SCIENCE_BIOLOGY",
    "SCIENCE_CHEMISTRY",
    "SCIENCE_MEDICINE",
    "SCIENCE_GENERAL",
    "SCIENCE_OTHER",
}
DEPLOYABLE_NON_DEFER_POLICIES = {
    "HH_ONLY",
    "CODE_ONLY",
    "OBJECTIVE_MIXED_ONLY",
    "OBJECTIVE_MIXED_BROAD_ONLY",
    "DOMAIN_ROUTED_SIMPLE",
    "DOMAIN_ROUTED_OBJECTIVE_BROAD",
    "CONTRAST_ROUTED_feature_only_0.70",
    "CONTRAST_ROUTED_feature_only_0.80",
    "CONTRAST_ROUTED_feature_only_0.90",
    "CONTRAST_ROUTED_feature_only_0.95",
    "OBJECTIVE_THEN_CODE_BACKUP_0.00",
    "OBJECTIVE_THEN_CODE_BACKUP_0.10",
    "OBJECTIVE_THEN_CODE_BACKUP_0.20",
    "OBJECTIVE_THEN_CODE_BACKUP_0.30",
    "OBJECTIVE_THEN_CODE_BACKUP_0.50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(BUNDLE_PT))
    parser.add_argument("--comparison", default=str(COMPARISON_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    return parser.parse_args()


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(output_path(path), map_location="cpu", weights_only=False)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def safe_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return float(mean(vals)) if vals else float("nan")


def role_key(bundle: dict[str, Any], role: str) -> str:
    return bundle["role_primary"][role]


def z_margin(bundle: dict[str, Any], head_key: str, domain: str, rec_idx: int) -> float:
    rec = bundle["head_scores"][head_key][domain]["records"][rec_idx]
    std = float(bundle["head_scores"][head_key][domain]["metrics"].get("margin_std") or 0.0)
    denom = std if std > 1e-8 else 1.0
    return float(rec["margin"]) / denom


def head_outcome(bundle: dict[str, Any], head_key: str, domain: str, rec_idx: int) -> dict[str, Any]:
    rec = bundle["head_scores"][head_key][domain]["records"][rec_idx]
    return {
        "deferred": False,
        "head_key": head_key,
        "pair_correct": int(rec["pair_correct"]),
        "pair_total": int(rec["pair_total"]),
        "top1_correct": bool(rec.get("top1_correct", False)),
        "confidence": z_margin(bundle, head_key, domain, rec_idx),
        "pair_decisions": list(rec.get("pair_decisions", [])),
        "pred_index": int(rec.get("pred_index", -1)),
    }


def defer_outcome(pair_total: int) -> dict[str, Any]:
    return {
        "deferred": True,
        "head_key": "DEFER",
        "pair_correct": 0,
        "pair_total": int(pair_total),
        "top1_correct": False,
        "confidence": 0.0,
        "pair_decisions": [],
        "pred_index": -1,
    }


def vote_outcome(
    bundle: dict[str, Any],
    head_keys: list[str],
    domain: str,
    rec_idx: int,
    *,
    defer_on_disagreement: bool = False,
    require_all_agree: bool = False,
) -> dict[str, Any]:
    outs = [head_outcome(bundle, key, domain, rec_idx) for key in head_keys]
    pair_total = outs[0]["pair_total"] if outs else 0
    if not outs:
        return defer_outcome(pair_total)
    disagreements = 0
    pair_correct = 0
    pair_decisions: list[bool] = []
    for pair_idx in range(pair_total):
        votes = [bool(out["pair_decisions"][pair_idx]) for out in outs if pair_idx < len(out["pair_decisions"])]
        if len(set(votes)) > 1:
            disagreements += 1
        if require_all_agree and len(set(votes)) > 1:
            pair_decisions.append(False)
            continue
        decision = sum(votes) >= (len(votes) / 2.0)
        pair_decisions.append(bool(decision))
        pair_correct += int(decision)
    if defer_on_disagreement and disagreements:
        return defer_outcome(pair_total)
    best = max(outs, key=lambda row: row["confidence"])
    pred_counts: dict[int, int] = {}
    for out in outs:
        pred_counts[int(out["pred_index"])] = pred_counts.get(int(out["pred_index"]), 0) + 1
    pred_index, pred_votes = max(pred_counts.items(), key=lambda item: (item[1], item[0]))
    top1_correct = pred_votes > len(outs) / 2.0 and any(out["pred_index"] == pred_index and out["top1_correct"] for out in outs)
    if pred_votes <= len(outs) / 2.0:
        top1_correct = bool(best["top1_correct"])
    return {
        "deferred": False,
        "head_key": "+".join(head_keys),
        "pair_correct": pair_correct,
        "pair_total": pair_total,
        "top1_correct": bool(top1_correct),
        "confidence": max(out["confidence"] for out in outs),
        "pair_decisions": pair_decisions,
        "pred_index": pred_index if pred_votes > len(outs) / 2.0 else best["pred_index"],
    }


def domain_routed_simple_head(bundle: dict[str, Any], domain: str) -> str:
    if domain.startswith("HH_"):
        return role_key(bundle, "HH_GENERAL")
    if domain == "CODE_STRICT_CLEAN_ALL16":
        return role_key(bundle, "CODE_SPECIALIST")
    if domain in OBJECTIVE_DOMAINS or domain in SCIENCE_DOMAINS:
        return role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")
    return role_key(bundle, "HH_GENERAL")


def domain_routed_broad_head(bundle: dict[str, Any], domain: str) -> str:
    if domain.startswith("HH_"):
        return role_key(bundle, "HH_GENERAL")
    if domain == "CODE_STRICT_CLEAN_ALL16":
        return role_key(bundle, "CODE_SPECIALIST")
    if domain in SCIENCE_DOMAINS:
        return role_key(bundle, "SCIENCE_AWARE_MIXED")
    if domain in OBJECTIVE_DOMAINS:
        return role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")
    return role_key(bundle, "HH_GENERAL")


def contrast_detect(record: dict[str, Any], threshold: float) -> bool:
    val = record.get("contrast_feature", {}).get("max_feature_cosine_36_L4")
    try:
        return float(val) >= threshold
    except Exception:
        return False


def policy_outcome(bundle: dict[str, Any], policy: dict[str, Any], domain: str, rec_idx: int) -> dict[str, Any]:
    record = bundle["domains"][domain]["records"][rec_idx]
    kind = policy["kind"]
    if kind == "oracle":
        return head_outcome(bundle, bundle["oracle_domain_best"][domain]["head_key"], domain, rec_idx)
    if kind == "role_only":
        return head_outcome(bundle, role_key(bundle, policy["role"]), domain, rec_idx)
    if kind == "domain_simple":
        return head_outcome(bundle, domain_routed_simple_head(bundle, domain), domain, rec_idx)
    if kind == "domain_broad":
        return head_outcome(bundle, domain_routed_broad_head(bundle, domain), domain, rec_idx)
    if kind == "contrast":
        if domain.startswith("HH_"):
            key = role_key(bundle, "HH_GENERAL")
        elif domain in {"CODE_STRICT_CLEAN_ALL16", "CODE_RUNNABLE_DIAGNOSTIC"} and contrast_detect(record, float(policy["threshold"])):
            key = role_key(bundle, "CODE_SPECIALIST")
        elif domain in OBJECTIVE_DOMAINS or domain in SCIENCE_DOMAINS:
            key = role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")
        else:
            key = role_key(bundle, "HH_GENERAL")
        return head_outcome(bundle, key, domain, rec_idx)
    if kind == "objective_code_backup":
        if domain.startswith("HH_"):
            key = role_key(bundle, "HH_GENERAL")
        elif domain == "CODE_STRICT_CLEAN_ALL16":
            obj = role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")
            key = role_key(bundle, "CODE_SPECIALIST") if z_margin(bundle, obj, domain, rec_idx) < float(policy["threshold"]) else obj
        elif domain in OBJECTIVE_DOMAINS or domain in SCIENCE_DOMAINS:
            key = role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")
        else:
            key = role_key(bundle, "HH_GENERAL")
        return head_outcome(bundle, key, domain, rec_idx)
    if kind == "margin_defer":
        key = domain_routed_simple_head(bundle, domain)
        out = head_outcome(bundle, key, domain, rec_idx)
        return defer_outcome(out["pair_total"]) if out["confidence"] < float(policy["threshold"]) else out
    if kind == "disagreement_defer":
        routed = domain_routed_simple_head(bundle, domain)
        routed_out = head_outcome(bundle, routed, domain, rec_idx)
        threshold = float(policy["threshold"])
        for key in [role_key(bundle, "HH_GENERAL"), role_key(bundle, "CODE_SPECIALIST"), role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")]:
            if key == routed:
                continue
            other = head_outcome(bundle, key, domain, rec_idx)
            if other["confidence"] < threshold:
                continue
            if other["pair_decisions"] != routed_out["pair_decisions"]:
                return defer_outcome(routed_out["pair_total"])
        return routed_out
    if kind == "two_vote":
        return vote_outcome(
            bundle,
            [role_key(bundle, "HH_GENERAL"), role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")],
            domain,
            rec_idx,
            defer_on_disagreement=bool(policy.get("defer_on_disagreement", False)),
        )
    if kind == "three_vote":
        return vote_outcome(
            bundle,
            [role_key(bundle, "HH_GENERAL"), role_key(bundle, "CODE_SPECIALIST"), role_key(bundle, "OBJECTIVE_MIXED_PRIMARY")],
            domain,
            rec_idx,
            defer_on_disagreement=bool(policy.get("defer_on_disagreement", False)),
            require_all_agree=bool(policy.get("require_all_agree", False)),
        )
    if kind == "risky":
        return head_outcome(bundle, role_key(bundle, "RISKY_HH_OBJECTIVE"), domain, rec_idx)
    raise ValueError(f"unknown policy kind {kind}")


def policy_definitions() -> list[dict[str, Any]]:
    policies = [
        {"name": "ORACLE_DOMAIN_BEST", "kind": "oracle", "deployable": False},
        {"name": "HH_ONLY", "kind": "role_only", "role": "HH_GENERAL", "deployable": True},
        {"name": "CODE_ONLY", "kind": "role_only", "role": "CODE_SPECIALIST", "deployable": True},
        {"name": "OBJECTIVE_MIXED_ONLY", "kind": "role_only", "role": "OBJECTIVE_MIXED_PRIMARY", "deployable": True},
        {"name": "OBJECTIVE_MIXED_BROAD_ONLY", "kind": "role_only", "role": "OBJECTIVE_MIXED_BROAD", "deployable": True},
        {"name": "DOMAIN_ROUTED_SIMPLE", "kind": "domain_simple", "deployable": True},
        {"name": "DOMAIN_ROUTED_OBJECTIVE_BROAD", "kind": "domain_broad", "deployable": True},
    ]
    for threshold in (0.70, 0.80, 0.90, 0.95):
        policies.append({"name": f"CONTRAST_ROUTED_feature_only_{threshold:.2f}", "kind": "contrast", "threshold": threshold, "deployable": True})
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.5):
        policies.append({"name": f"OBJECTIVE_THEN_CODE_BACKUP_{threshold:.2f}", "kind": "objective_code_backup", "threshold": threshold, "deployable": True})
    policies.extend(
        [
            {"name": "GENERAL_AND_OBJECTIVE_VOTE_margin", "kind": "two_vote", "deployable": True},
            {"name": "GENERAL_AND_OBJECTIVE_VOTE_defer", "kind": "two_vote", "defer_on_disagreement": True, "deployable": True, "defer_policy": True},
            {"name": "THREE_HEAD_VOTE_margin", "kind": "three_vote", "deployable": True},
            {"name": "THREE_HEAD_VOTE_defer", "kind": "three_vote", "defer_on_disagreement": True, "deployable": True, "defer_policy": True},
            {"name": "CONSENSUS_SELECT_HH_OBJECTIVE", "kind": "two_vote", "defer_on_disagreement": True, "deployable": True, "defer_policy": True},
            {"name": "CONSENSUS_SELECT_ALL_THREE", "kind": "three_vote", "defer_on_disagreement": True, "require_all_agree": True, "deployable": True, "defer_policy": True},
            {"name": "RISKY_HH_OBJECTIVE_ABLATION", "kind": "risky", "deployable": False},
            {"name": "ORACLE_POLICY_WITH_DEFER", "kind": "oracle", "deployable": False, "defer_policy": True},
        ]
    )
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.5):
        policies.append({"name": f"MARGIN_DEFER_{threshold:.2f}", "kind": "margin_defer", "threshold": threshold, "deployable": True, "defer_policy": True})
    for threshold in (0.1, 0.2, 0.3):
        policies.append({"name": f"DISAGREEMENT_DEFER_{threshold:.2f}", "kind": "disagreement_defer", "threshold": threshold, "deployable": True, "defer_policy": True})
    return policies


def evaluate_policy(bundle: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    outcomes_by_domain: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for domain, spec in bundle["domains"].items():
        outcomes = []
        for idx, rec in enumerate(spec["records"]):
            out = policy_outcome(bundle, policy, domain, idx)
            out.update({"domain": domain, "record_id": rec["record_id"], "task_id": rec["task_id"]})
            outcomes.append(out)
            all_records.append(out)
        outcomes_by_domain[domain] = outcomes
        by_domain[domain] = summarize_outcomes(outcomes, spec["random_top1_baseline"])
    overall = summarize_outcomes(all_records, 0.5)
    objective_pairwise = safe_mean([by_domain[d]["pairwise_acc"] for d in OBJECTIVE_DOMAINS if d in by_domain])
    oracle_gaps = {}
    for domain, metrics in by_domain.items():
        oracle_pair = float(bundle["oracle_domain_best"][domain]["pairwise_acc"])
        policy_pair = float(metrics["pairwise_acc"])
        oracle_gaps[domain] = oracle_pair - policy_pair if not math.isnan(policy_pair) else float("nan")
    selective = selective_accuracy(all_records)
    return {
        "policy": policy,
        "overall": overall,
        "objective_average_pairwise": objective_pairwise,
        "hh_heldout_pairwise": by_domain.get("HH_HELDOUT20", {}).get("pairwise_acc"),
        "strict_clean_pairwise": by_domain.get("CODE_STRICT_CLEAN_ALL16", {}).get("pairwise_acc"),
        "domain_breakdown": by_domain,
        "oracle_gap_pairwise": oracle_gaps,
        "average_oracle_gap_pairwise": safe_mean(list(oracle_gaps.values())),
        "objective_average_oracle_gap_pairwise": safe_mean([oracle_gaps[d] for d in OBJECTIVE_DOMAINS if d in oracle_gaps]),
        "selective_accuracy": selective,
        "outcomes": outcomes_by_domain,
    }


def summarize_outcomes(outcomes: list[dict[str, Any]], random_top1_baseline: float) -> dict[str, Any]:
    total = sum(int(row["pair_total"]) for row in outcomes)
    accepted = [row for row in outcomes if not row.get("deferred")]
    accepted_total = sum(int(row["pair_total"]) for row in accepted)
    correct = sum(int(row["pair_correct"]) for row in accepted)
    top_total = len(accepted)
    top_correct = sum(1 for row in accepted if row.get("top1_correct"))
    return {
        "pairwise_acc": correct / accepted_total if accepted_total else float("nan"),
        "top1": top_correct / top_total if top_total else float("nan"),
        "top1_over_random": top_correct / top_total - random_top1_baseline if top_total else float("nan"),
        "coverage": accepted_total / total if total else 0.0,
        "defer_rate": 1.0 - (accepted_total / total if total else 0.0),
        "accepted_pairs": accepted_total,
        "total_pairs": total,
        "defers": len(outcomes) - len(accepted),
    }


def selective_accuracy(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["pair_total"]) for row in outcomes)
    rows = sorted(outcomes, key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
    out: dict[str, Any] = {}
    for target in (0.50, 0.70, 0.80, 0.90, 1.00):
        need = total * target
        acc_total = 0
        acc_correct = 0
        for row in rows:
            if acc_total >= need:
                break
            acc_total += int(row["pair_total"])
            if not row.get("deferred"):
                acc_correct += int(row["pair_correct"])
        out[f"{int(target * 100)}pct"] = acc_correct / acc_total if acc_total else float("nan")
    return out


def detector_quality(bundle: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for domain in ("CODE_STRICT_CLEAN_ALL16", "CODE_RUNNABLE_DIAGNOSTIC"):
        if domain not in bundle["domains"]:
            continue
        positive = domain == "CODE_STRICT_CLEAN_ALL16"
        for rec in bundle["domains"][domain]["records"]:
            val = rec.get("contrast_feature", {}).get("max_feature_cosine_36_L4", float("nan"))
            rows.append({"domain": domain, "positive": positive, "score": float(val)})
    by_threshold = {}
    for threshold in (0.70, 0.80, 0.90, 0.95):
        tp = sum(row["positive"] and row["score"] >= threshold for row in rows)
        fp = sum((not row["positive"]) and row["score"] >= threshold for row in rows)
        fn = sum(row["positive"] and row["score"] < threshold for row in rows)
        tn = sum((not row["positive"]) and row["score"] < threshold for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (tp + fn) if tp + fn else 0.0
        by_threshold[f"{threshold:.2f}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    best = max(by_threshold.items(), key=lambda item: (item[1]["f1"], item[1]["precision"])) if by_threshold else ("NA", {})
    verdict = "READY" if best[1] and best[1]["precision"] >= 0.70 and best[1]["recall"] >= 0.50 else ("DEPLOYABILITY_WEAK" if by_threshold else "NOT_RUN")
    return {"verdict": verdict, "best_threshold": best[0], "by_threshold": by_threshold}


def fallback_adjust(policy_result: dict[str, Any], fallback_result: dict[str, Any], random_pairwise: float = 0.5) -> dict[str, Any]:
    random_scores = {}
    routed_scores = {}
    for domain, metrics in policy_result["domain_breakdown"].items():
        coverage = float(metrics["coverage"])
        selective = float(metrics["pairwise_acc"]) if not math.isnan(float(metrics["pairwise_acc"])) else 0.0
        routed = float(fallback_result["domain_breakdown"].get(domain, {}).get("pairwise_acc", random_pairwise))
        random_scores[domain] = coverage * selective + (1.0 - coverage) * random_pairwise
        routed_scores[domain] = coverage * selective + (1.0 - coverage) * routed
    return {
        "random_fallback_average": safe_mean(list(random_scores.values())),
        "domain_routed_fallback_average": safe_mean(list(routed_scores.values())),
        "by_domain_random": random_scores,
        "by_domain_domain_routed": routed_scores,
    }


def choose_verdicts(bundle: dict[str, Any], results: dict[str, Any], detector: dict[str, Any]) -> dict[str, Any]:
    deployable = {
        name: row
        for name, row in results.items()
        if row["policy"].get("deployable") and not row["policy"].get("defer_policy")
    }
    non_defer_baseline = max(deployable.values(), key=lambda row: (row["objective_average_pairwise"], row["overall"]["pairwise_acc"]))
    domain_simple = results["DOMAIN_ROUTED_SIMPLE"]
    best_single = max(
        [results[name] for name in ("HH_ONLY", "CODE_ONLY", "OBJECTIVE_MIXED_ONLY", "OBJECTIVE_MIXED_BROAD_ONLY")],
        key=lambda row: row["objective_average_pairwise"],
    )
    strict_existing = bundle["existing_specialist_best"]["CODE_STRICT_CLEAN_ALL16"]["pairwise_acc"]
    hh_ref = results["HH_ONLY"]["domain_breakdown"]["HH_HELDOUT20"]["pairwise_acc"]
    domain_strict_regret = domain_simple["strict_clean_pairwise"] - strict_existing
    domain_hh_loss = domain_simple["hh_heldout_pairwise"] - hh_ref
    defer_summary = defer_verdict(results, non_defer_baseline["overall"]["pairwise_acc"], domain_simple)
    oracle_gap = domain_simple["average_oracle_gap_pairwise"]
    if math.isnan(oracle_gap):
        oracle_gap_verdict = "INSUFFICIENT"
    elif oracle_gap <= 0.03:
        oracle_gap_verdict = "SMALL"
    elif oracle_gap <= 0.08:
        oracle_gap_verdict = "MODERATE"
    else:
        oracle_gap_verdict = "LARGE"
    objective_only = results["OBJECTIVE_MIXED_ONLY"]
    best_policy_verdict = "INSUFFICIENT"
    if defer_summary["defer_policy_verdict"] == "DEFER_STRONG":
        best_policy_verdict = "DEFER_POLICY_WINS"
    elif (
        domain_simple["objective_average_pairwise"] - best_single["objective_average_pairwise"] >= 0.03
        and domain_hh_loss >= -0.05
        and domain_strict_regret >= -0.05
    ):
        best_policy_verdict = "DOMAIN_ROUTED_WINS"
    elif (
        abs(objective_only["objective_average_pairwise"] - domain_simple["objective_average_pairwise"]) <= 0.02
        and objective_only["strict_clean_pairwise"] - strict_existing >= -0.05
    ):
        best_policy_verdict = "OBJECTIVE_MIXED_DEFAULT_WINS"
    elif domain_simple["average_oracle_gap_pairwise"] <= 0.02:
        best_policy_verdict = "SINGLE_HEAD_SUFFICIENT"
    else:
        best_policy_verdict = "DOMAIN_ROUTED_WINS"
    recommended = {
        "DOMAIN_ROUTED_WINS": "HH_GENERAL_PLUS_OBJECTIVE_MIXED_PLUS_CODE_BACKUP",
        "OBJECTIVE_MIXED_DEFAULT_WINS": "HH_GENERAL_PLUS_OBJECTIVE_MIXED_PLUS_CODE_BACKUP",
        "DEFER_POLICY_WINS": "MARGIN_DEFER_ROUTING",
        "SINGLE_HEAD_SUFFICIENT": "HH_GENERAL_PLUS_OBJECTIVE_MIXED_ROUTE",
        "INSUFFICIENT": "INSUFFICIENT",
    }[best_policy_verdict]
    return {
        "bg_policy_sim_verdict": "READY" if results else "BLOCKED",
        "best_policy_verdict": best_policy_verdict,
        "recommended_bg_policy": recommended,
        "contrast_detector_verdict": detector["verdict"],
        "defer_policy_verdict": defer_summary["defer_policy_verdict"],
        "oracle_gap_verdict": oracle_gap_verdict,
        "best_non_defer_policy": non_defer_baseline["policy"]["name"],
        "best_single_policy": best_single["policy"]["name"],
        "defer_summary": defer_summary,
    }


def defer_verdict(results: dict[str, Any], baseline_pairwise: float, fallback: dict[str, Any]) -> dict[str, Any]:
    best = {"policy": "NA", "improvement70": float("-inf"), "improvement80": float("-inf"), "improvement90": float("-inf"), "coverage": 0.0}
    fallback_rows = {}
    for name, row in results.items():
        if not row["policy"].get("defer_policy"):
            continue
        sel = row["selective_accuracy"]
        improvement70 = float(sel.get("70pct", float("nan"))) - baseline_pairwise
        improvement80 = float(sel.get("80pct", float("nan"))) - baseline_pairwise
        improvement90 = float(sel.get("90pct", float("nan"))) - baseline_pairwise
        fallback_rows[name] = fallback_adjust(row, fallback)
        if improvement70 > best["improvement70"]:
            best = {
                "policy": name,
                "improvement70": improvement70,
                "improvement80": improvement80,
                "improvement90": improvement90,
                "coverage": float(row["overall"]["coverage"]),
            }
    fallback_gain = max(
        (
            val["domain_routed_fallback_average"] - baseline_pairwise
            for val in fallback_rows.values()
            if not math.isnan(float(val["domain_routed_fallback_average"]))
        ),
        default=float("-inf"),
    )
    if best["improvement70"] >= 0.05 and best["coverage"] >= 0.70:
        verdict = "DEFER_STRONG"
    elif (0.02 <= best["improvement90"] <= 0.05) or fallback_gain >= 0.02:
        verdict = "DEFER_MARGINAL"
    else:
        verdict = "DEFER_NOT_USEFUL"
    return {"defer_policy_verdict": verdict, "best_defer_policy": best, "fallback_adjusted": fallback_rows}


def wrong_examples(bundle: dict[str, Any], results: dict[str, Any], policy_name: str) -> list[dict[str, Any]]:
    row = results[policy_name]
    examples = []
    role_heads = set(bundle["role_primary"].values())
    for domain, outcomes in row["outcomes"].items():
        for idx, out in enumerate(outcomes):
            if out.get("deferred") or out["pair_correct"] == out["pair_total"]:
                continue
            alternatives = []
            for key in role_heads:
                alt = head_outcome(bundle, key, domain, idx)
                if alt["pair_correct"] > out["pair_correct"]:
                    alternatives.append(key)
            if alternatives:
                examples.append(
                    {
                        "domain": domain,
                        "record_id": out["record_id"],
                        "policy_head": out["head_key"],
                        "better_heads": alternatives[:5],
                    }
                )
            if len(examples) >= 30:
                return examples
    return examples


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines = [
        "# BG Controller Policy Simulation (2026-05-17)",
        "",
        f"BG_POLICY_SIM_VERDICT = {meta['bg_policy_sim_verdict']}",
        f"BEST_POLICY_VERDICT = {meta['best_policy_verdict']}",
        f"RECOMMENDED_BG_POLICY = {meta['recommended_bg_policy']}",
        f"CONTRAST_DETECTOR_VERDICT = {meta['contrast_detector_verdict']}",
        f"DEFER_POLICY_VERDICT = {meta['defer_policy_verdict']}",
        f"ORACLE_GAP_VERDICT = {meta['oracle_gap_verdict']}",
        "",
        "## Policy Results",
        "| policy | coverage | pairwise | objective avg | HH heldout | strict clean | oracle gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(payload["policy_results"].items(), key=lambda item: (item[1]["policy"].get("deployable") is False, item[0])):
        lines.append(
            f"| {name} | {row['overall']['coverage']:.3f} | {row['overall']['pairwise_acc']:.3f} | "
            f"{row['objective_average_pairwise']:.3f} | {row['hh_heldout_pairwise']:.3f} | "
            f"{row['strict_clean_pairwise']:.3f} | {row['average_oracle_gap_pairwise']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Contrast Detector",
            f"- best threshold: {payload['contrast_detector']['best_threshold']}",
            f"- threshold metrics: `{json.dumps(payload['contrast_detector']['by_threshold'], default=str)}`",
            "",
            "## Defer Summary",
            f"- `{json.dumps(meta['defer_summary'], default=str)[:1600]}`",
            "",
            "## Wrong But Another Head Was Right",
        ]
    )
    for item in payload["wrong_but_other_head_right"][:20]:
        lines.append(f"- {item['domain']} / {item['record_id']}: policy={item['policy_head']}, alternatives={item['better_heads']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle = load_pt(args.bundle)
    comparison = load_json(args.comparison)
    if bundle.get("meta", {}).get("bg_policy_eval_bundle_verdict") == "BLOCKED" or comparison.get("meta", {}).get("bg_head_comparison_verdict") == "BLOCKED":
        payload = {
            "meta": {
                "bg_policy_sim_verdict": "BLOCKED",
                "best_policy_verdict": "INSUFFICIENT",
                "recommended_bg_policy": "INSUFFICIENT",
                "contrast_detector_verdict": "NOT_RUN",
                "defer_policy_verdict": "NOT_RUN",
                "oracle_gap_verdict": "INSUFFICIENT",
            },
            "policy_results": {},
            "blockers": ["bundle or comparison blocked"],
        }
        write_json(output_path(args.output), payload)
        write_markdown(output_path(args.output_md), payload)
        print("BG_POLICY_SIM_VERDICT = BLOCKED")
        return
    policies = policy_definitions()
    results = {policy["name"]: evaluate_policy(bundle, policy) for policy in policies}
    detector = detector_quality(bundle)
    meta = choose_verdicts(bundle, results, detector)
    wrong = wrong_examples(bundle, results, "DOMAIN_ROUTED_SIMPLE")
    # Keep JSON compact enough for inspection; outcomes remain available for uncertainty bootstrap.
    payload = {
        "meta": meta,
        "policy_results": results,
        "contrast_detector": detector,
        "wrong_but_other_head_right": wrong,
        "blockers": [],
    }
    write_json(output_path(args.output), payload)
    write_markdown(output_path(args.output_md), payload)
    print(f"BG_POLICY_SIM_VERDICT = {meta['bg_policy_sim_verdict']}")
    print(f"BEST_POLICY_VERDICT = {meta['best_policy_verdict']}")
    print(f"RECOMMENDED_BG_POLICY = {meta['recommended_bg_policy']}")
    print(f"CONTRAST_DETECTOR_VERDICT = {meta['contrast_detector_verdict']}")
    print(f"DEFER_POLICY_VERDICT = {meta['defer_policy_verdict']}")
    print(f"ORACLE_GAP_VERDICT = {meta['oracle_gap_verdict']}")
    print(f"wrote {repo_path(output_path(args.output))}")


if __name__ == "__main__":
    main()
