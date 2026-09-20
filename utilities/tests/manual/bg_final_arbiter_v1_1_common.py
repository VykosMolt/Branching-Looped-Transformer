"""Final arbiter among fixed-composite top4 survivors v1.1.

This experiment only trains small standalone final-arbiter heads over cached
top4 survivor sets. It does not train Ouro, mutate checkpoints/tokenizers/tap
registries, import or execute Hunter-Seeker modules, run wrapper/local-agent
code, run ARC/MATH generation loops, apply action steering, or change routing.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import bg_final_arbiter_top4_v1_common as v1
from bg_fixed_composite_survival_v1_common import safe_float
from bg_hidden_origin_quota_v4_common import PROBE_ROOT, PROJECT_ROOT, md_table, rel, write_csv, write_md
from bg_universal_tap_v1_common import load_json


OUT_ROOT = PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_1_2026-05-18"
V1_ROOT = PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_2026-05-18"

SPLIT_GUARD_JSON = OUT_ROOT / "inventory_split_guard.json"
SPLIT_GUARD_MD = OUT_ROOT / "inventory_split_guard.md"
TASK_SPLIT_CSV = OUT_ROOT / "task_split_plan.csv"
DATASET_PT = OUT_ROOT / "final_arbiter_v1_1_dataset.pt"
DATASET_JSON = OUT_ROOT / "final_arbiter_v1_1_dataset.json"
DATASET_MD = OUT_ROOT / "final_arbiter_v1_1_dataset.md"
FEATURES_PT = OUT_ROOT / "final_arbiter_v1_1_features.pt"
FEATURES_JSON = OUT_ROOT / "final_arbiter_v1_1_features.json"
FEATURES_MD = OUT_ROOT / "final_arbiter_v1_1_features.md"
BASELINES_JSON = OUT_ROOT / "baselines.json"
BASELINES_MD = OUT_ROOT / "baselines.md"
BASELINES_CSV = OUT_ROOT / "baseline_rows.csv"
MODEL_PT = OUT_ROOT / "final_arbiter_top4_v1_1.pt"
TRAINING_JSON = OUT_ROOT / "training_log.json"
TRAINING_MD = OUT_ROOT / "training_report.md"
HELDOUT_JSON = OUT_ROOT / "heldout_eval.json"
HELDOUT_MD = OUT_ROOT / "heldout_eval.md"
HELDOUT_CSV = OUT_ROOT / "heldout_eval_rows.csv"
DOMAIN_JSON = OUT_ROOT / "domain_analysis.json"
DOMAIN_MD = OUT_ROOT / "domain_analysis.md"
TIE_JSON = OUT_ROOT / "tie_analysis.json"
TIE_MD = OUT_ROOT / "tie_analysis.md"
TIE_CSV = OUT_ROOT / "tie_rows.csv"
ABLATION_JSON = OUT_ROOT / "ablation.json"
ABLATION_MD = OUT_ROOT / "ablation.md"
ABLATION_CSV = OUT_ROOT / "ablation_rows.csv"
CALIBRATION_JSON = OUT_ROOT / "calibration_ood.json"
CALIBRATION_MD = OUT_ROOT / "calibration_ood.md"
FAILURE_JSON = OUT_ROOT / "failure_analysis.json"
FAILURE_MD = OUT_ROOT / "failure_analysis.md"
FAILURE_CSV = OUT_ROOT / "failure_cases.csv"
READINESS_JSON = OUT_ROOT / "selection_readiness.json"
READINESS_MD = OUT_ROOT / "selection_readiness.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_final_arbiter_top4_survivors_v1_1.md"

SEED = 20260518
DOMAINS = list(v1.DOMAINS)
SCORE_KEYS = list(v1.SCORE_KEYS)
RANK_EXPERTS = [
    "fixed_composite",
    "old",
    "old_objective_mixed",
    "old_code_reasoning",
    "v4_hidden_origin",
    "bridge",
    "bridge_only",
    "hidden_branch",
    "branch",
    "universal",
    "gated",
    "generator_v1_selector",
]

V1_REFERENCE = {
    "trained_task_macro": 0.6679947560419517,
    "ranks_only_task_macro": 0.7041752165982672,
    "majority_task_macro": 0.6251168490652075,
    "fixed_task_macro": 0.6159256725946193,
    "oracle_task_macro": 0.897828317373461,
    "reasoning_task_macro": 0.4883720930232558,
    "science_task_macro": 0.5234848484848484,
    "math_task_macro": 0.6666666666666666,
    "coding_task_macro": 0.8125,
}

SCRIPT_NAMES = [
    "bg_final_arbiter_v1_1_inventory_split_guard.py",
    "build_bg_final_arbiter_v1_1_dataset.py",
    "build_bg_final_arbiter_v1_1_features.py",
    "run_bg_final_arbiter_v1_1_baselines.py",
    "train_bg_final_arbiter_v1_1.py",
    "evaluate_bg_final_arbiter_v1_1.py",
    "analyze_bg_final_arbiter_v1_1_domains.py",
    "analyze_bg_final_arbiter_v1_1_ties.py",
    "analyze_bg_final_arbiter_v1_1_ablation.py",
    "analyze_bg_final_arbiter_v1_1_calibration_ood.py",
    "analyze_bg_final_arbiter_v1_1_failures.py",
    "analyze_bg_final_arbiter_v1_1_selection_readiness.py",
    "analyze_bg_final_arbiter_v1_1.py",
]

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_final_arbiter_top4_survivors_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_selection_only_phase2_prototype_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
    PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
]

_FEATURE_CACHE: dict[str, Any] | None = None


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Path):
        return rel(value)
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def finite_mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(mean(vals)) if vals else default


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def rank_name(score_key: str) -> str:
    return f"rank_{score_key.removesuffix('_score')}"


def source_sets() -> list[dict[str, Any]]:
    payload = v1.load_features_payload()
    raw_sets = payload.get("raw_sets") or []
    feature_sets = payload.get("sets") or []
    out = []
    for idx, raw in enumerate(raw_sets):
        s = {k: val for k, val in raw.items() if k != "candidates"}
        s["set_idx"] = idx
        s["candidates"] = [dict(c) for c in raw.get("candidates") or []]
        if idx < len(feature_sets):
            s["v1_candidate_feature_indices"] = list(feature_sets[idx].get("candidate_feature_indices") or [])
            s["v1_candidate_rows"] = list(feature_sets[idx].get("candidate_rows") or [])
        else:
            s["v1_candidate_feature_indices"] = []
            s["v1_candidate_rows"] = []
        out.append(s)
    return out


def build_task_split_plan() -> list[dict[str, Any]]:
    sets = source_sets()
    by_task: dict[str, dict[str, Any]] = {}
    for s in sets:
        tid = str(s.get("task_id"))
        meta = by_task.setdefault(
            tid,
            {
                "task_id": tid,
                "domain": str(s.get("domain")),
                "v1_split": str(s.get("split")),
                "survivor_sets": 0,
                "candidate_count": 0,
            },
        )
        meta["survivor_sets"] += 1
        meta["candidate_count"] += len(s.get("candidates") or [])
    non_v1_heldout: dict[str, list[str]] = defaultdict(list)
    for tid, meta in by_task.items():
        if meta["v1_split"] != "heldout":
            non_v1_heldout[str(meta["domain"])].append(tid)

    fresh_holdout: set[str] = set()
    for domain, task_ids in non_v1_heldout.items():
        ordered = sorted(task_ids, key=lambda t: (stable_hash(f"fresh:{domain}:{t}"), t))
        holdout_n = max(1, round(0.30 * len(ordered))) if len(ordered) >= 5 else 0
        fresh_holdout.update(ordered[:holdout_n])

    remaining_by_domain: dict[str, list[str]] = defaultdict(list)
    for tid, meta in by_task.items():
        if meta["v1_split"] != "heldout" and tid not in fresh_holdout:
            remaining_by_domain[str(meta["domain"])].append(tid)
    validation: set[str] = set()
    for domain, task_ids in remaining_by_domain.items():
        ordered = sorted(task_ids, key=lambda t: (stable_hash(f"val:{domain}:{t}"), t))
        val_n = max(1, round(0.20 * len(ordered))) if len(ordered) >= 5 else 0
        validation.update(ordered[:val_n])

    rows = []
    for tid, meta in sorted(by_task.items()):
        old_split = str(meta["v1_split"])
        if old_split == "heldout":
            split = "v1_heldout_replay_diagnostic"
            eligible = False
        elif tid in fresh_holdout:
            split = "fresh_holdout"
            eligible = True
        elif tid in validation:
            split = "val"
            eligible = False
        else:
            split = "train"
            eligible = False
        rows.append(
            {
                **meta,
                "v1_1_split": split,
                "readiness_eligible": eligible,
                "outer_fold": stable_hash(f"outer:{tid}") % 4,
                "old_v1_split_overlap": old_split,
            }
        )
    return rows


def split_lookup() -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in build_task_split_plan()}


def run_split_guard() -> int:
    ensure_root()
    started = time.time()
    task_rows = build_task_split_plan()
    sets = source_sets()
    lookup = {str(row["task_id"]): row for row in task_rows}
    mode_rows = []
    for mode in ["fresh_holdout", "grouped_nested_cv", "v1_heldout_replay_diagnostic", "domain_holdout_diagnostics"]:
        if mode == "grouped_nested_cv":
            mode_sets = [s for s in sets if lookup[str(s.get("task_id"))]["v1_1_split"] != "v1_heldout_replay_diagnostic"]
            eligible = bool(mode_sets)
        elif mode == "domain_holdout_diagnostics":
            mode_sets = sets
            eligible = False
        else:
            mode_sets = [s for s in sets if lookup[str(s.get("task_id"))]["v1_1_split"] == mode]
            eligible = mode == "fresh_holdout" and bool(mode_sets)
        mode_rows.append(
            {
                "mode": mode,
                "task_count": len({str(s.get("task_id")) for s in mode_sets}),
                "survivor_set_count": len(mode_sets),
                "candidate_count": sum(len(s.get("candidates") or []) for s in mode_sets),
                "domain_distribution": dict(Counter(str(s.get("domain")) for s in mode_sets)),
                "coding_support": sum(1 for s in mode_sets if s.get("domain") == "coding"),
                "science_support": sum(1 for s in mode_sets if s.get("domain") == "science"),
                "v1_heldout_overlap_sets": sum(1 for s in mode_sets if str(s.get("split")) == "heldout"),
                "readiness_eligible": eligible,
            }
        )
    fresh = next(row for row in mode_rows if row["mode"] == "fresh_holdout")
    domains = fresh["domain_distribution"]
    if fresh["survivor_set_count"] >= 40 and len(domains) >= 3 and domains.get("science", 0) > 0:
        verdict = "FRESH_HELDOUT_READY"
    elif next(row for row in mode_rows if row["mode"] == "grouped_nested_cv")["survivor_set_count"] >= 80:
        verdict = "NESTED_CV_READY"
    elif fresh["survivor_set_count"]:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT": verdict,
        "verdict": verdict,
        "evaluation_modes": mode_rows,
        "task_split_plan": task_rows,
        "anti_leakage_policy": [
            "The v1 heldout rank-only result is treated as a hypothesis.",
            "The readiness-bearing fresh v1.1 holdout is selected from tasks that were not v1 heldout.",
            "Previous v1 heldout replay is diagnostic only.",
            "No model or threshold is selected on the v1.1 fresh holdout.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SPLIT_GUARD_JSON, payload)
    write_csv(TASK_SPLIT_CSV, task_rows)
    lines = ["# Final Arbiter V1.1 Inventory Split Guard", "", f"BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT = {verdict}", "", "## Evaluation Modes", ""]
    lines.extend(md_table(mode_rows, ["mode", "task_count", "survivor_set_count", "candidate_count", "domain_distribution", "v1_heldout_overlap_sets", "readiness_eligible"]))
    lines.extend(["", "## Anti-Leakage Policy", ""])
    lines.extend(f"- {item}" for item in payload["anti_leakage_policy"])
    write_md(SPLIT_GUARD_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def sets_with_v1_1_splits() -> list[dict[str, Any]]:
    lookup = split_lookup()
    out = []
    for s in source_sets():
        tid = str(s.get("task_id"))
        row = lookup[tid]
        ss = dict(s)
        ss["v1_split"] = str(s.get("split"))
        ss["split"] = row["v1_1_split"]
        ss["readiness_eligible"] = bool(row["readiness_eligible"])
        ss["outer_fold"] = int(row["outer_fold"])
        ss["source_status"] = "reused_heldout" if row["v1_1_split"] == "v1_heldout_replay_diagnostic" else "reused_diagnostic"
        out.append(ss)
    return out


def build_pair_rows(sets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for s in sets:
        candidates = s.get("candidates") or []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                ri = safe_float(candidates[i].get("final_reward"), 0.0)
                rj = safe_float(candidates[j].get("final_reward"), 0.0)
                if ri == rj:
                    continue
                pref, rej = (candidates[i], candidates[j]) if ri > rj else (candidates[j], candidates[i])
                pairs.append(
                    {
                        "survivor_set_id": s.get("survivor_set_id"),
                        "task_id": s.get("task_id"),
                        "domain": s.get("domain"),
                        "split": s.get("split"),
                        "preferred_candidate_id": pref.get("candidate_id"),
                        "rejected_candidate_id": rej.get("candidate_id"),
                        "reward_gap": abs(ri - rj),
                    }
                )
    return pairs


def dataset_counts(sets: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "survivor_sets": len(sets),
        "tasks": len({str(s.get("task_id")) for s in sets}),
        "domains": dict(Counter(str(s.get("domain")) for s in sets)),
        "splits": dict(Counter(str(s.get("split")) for s in sets)),
        "sets_by_domain_split": dict(Counter(f"{s.get('domain')}::{s.get('split')}" for s in sets)),
        "candidates_by_split": dict(Counter(str(s.get("split")) for s in sets for _ in (s.get("candidates") or []))),
        "pairwise_pairs": len(pairs),
        "pairwise_by_split": dict(Counter(str(p.get("split")) for p in pairs)),
        "tie_distribution": dict(Counter(int(s.get("tie_count") or 0) for s in sets)),
        "fresh_holdout_sets": sum(1 for s in sets if s.get("split") == "fresh_holdout"),
        "v1_replay_sets": sum(1 for s in sets if s.get("split") == "v1_heldout_replay_diagnostic"),
    }


def run_dataset() -> int:
    ensure_root()
    started = time.time()
    if not SPLIT_GUARD_JSON.exists():
        run_split_guard()
    sets = sets_with_v1_1_splits()
    pairs = build_pair_rows(sets)
    counts = dataset_counts(sets, pairs)
    if counts["fresh_holdout_sets"] >= 40 and counts["splits"].get("train", 0) and counts["splits"].get("val", 0):
        verdict = "READY"
    elif counts["survivor_sets"] >= 80:
        verdict = "SMALL_BUT_USABLE"
    elif counts["survivor_sets"]:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_FINAL_ARBITER_V1_1_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "survivor_sets": sets,
        "pairwise_pairs": pairs,
        "counts": counts,
        "label_policy": "final reward/correctness/verifier labels only; tap scores and ranks are input features only; tied-best candidates are preserved",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, DATASET_PT)
    write_json(DATASET_JSON, {k: val for k, val in payload.items() if k not in {"survivor_sets"}})
    lines = ["# Final Arbiter V1.1 Dataset", "", f"BG_FINAL_ARBITER_V1_1_DATASET_VERDICT = {verdict}", "", f"- counts: `{counts}`", "", "## Sets By Domain/Split", ""]
    lines.extend(md_table([{"key": k, "sets": v} for k, v in counts["sets_by_domain_split"].items()], ["key", "sets"]))
    write_md(DATASET_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def load_dataset_payload() -> dict[str, Any]:
    if not DATASET_PT.exists():
        run_dataset()
    return torch.load(DATASET_PT, map_location="cpu", weights_only=False)


def ranks_for_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    ranks = {}
    for expert in RANK_EXPERTS:
        key = f"rank_{expert}"
        if key in candidate:
            ranks[expert] = safe_float(candidate.get(key), 99.0)
    return ranks


def rank_order(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda cid: (safe_float(values[cid], 99.0), cid))
    return {cid: idx + 1 for idx, cid in enumerate(ordered)}


def add_consensus_ranks(survivor_set: dict[str, Any]) -> None:
    candidates = survivor_set.get("candidates") or []
    avg_by_id = {}
    median_by_id = {}
    for candidate in candidates:
        cid = str(candidate.get("candidate_id"))
        rank_vals = list(ranks_for_candidate(candidate).values())
        if rank_vals:
            avg_by_id[cid] = finite_mean(rank_vals, 99.0)
            median_by_id[cid] = float(median(rank_vals))
        else:
            avg_by_id[cid] = 99.0
            median_by_id[cid] = 99.0
    avg_order = rank_order(avg_by_id)
    med_order = rank_order(median_by_id)
    for candidate in candidates:
        cid = str(candidate.get("candidate_id"))
        candidate["rank_borda"] = avg_order.get(cid, 99)
        candidate["rank_median"] = med_order.get(cid, 99)
        candidate["average_rank"] = avg_by_id.get(cid, 99.0)
        candidate["median_rank"] = median_by_id.get(cid, 99.0)


def feature_map(candidate: dict[str, Any], survivor_set: dict[str, Any]) -> dict[str, float]:
    k = max(int(survivor_set.get("survivor_count") or len(survivor_set.get("candidates") or []) or 1), 1)
    ranks = ranks_for_candidate(candidate)
    out: dict[str, float] = {}
    for score_key in SCORE_KEYS:
        out[score_key] = safe_float(candidate.get(score_key), 0.0)
    for expert in RANK_EXPERTS:
        r = safe_float(candidate.get(f"rank_{expert}"), float(k))
        out[f"rank_{expert}"] = r / k
        out[f"{expert}_top1_flag"] = 1.0 if int(r) == 1 else 0.0
        out[f"{expert}_top2_flag"] = 1.0 if int(r) <= 2 else 0.0
    out["rank_borda"] = safe_float(candidate.get("rank_borda"), float(k)) / k
    out["rank_median"] = safe_float(candidate.get("rank_median"), float(k)) / k
    rank_vals = [safe_float(v, float(k)) for v in ranks.values()]
    avg_rank = finite_mean(rank_vals, float(k))
    out["average_rank"] = avg_rank / k
    out["median_rank"] = (float(median(rank_vals)) if rank_vals else float(k)) / k
    out["rank_variance"] = finite_mean([(r - avg_rank) ** 2 for r in rank_vals], 0.0) / max(k * k, 1)
    rank_counts = Counter(int(r) for r in rank_vals)
    total = max(sum(rank_counts.values()), 1)
    out["rank_entropy"] = -sum((n / total) * math.log(max(n / total, 1e-9)) for n in rank_counts.values())
    out["top1_vote_count"] = sum(1.0 for r in rank_vals if int(r) == 1)
    out["top2_vote_count"] = sum(1.0 for r in rank_vals if int(r) <= 2)
    bridge_rank = safe_float(candidate.get("rank_bridge"), float(k))
    old_rank = safe_float(candidate.get("rank_old"), float(k))
    hidden_rank = safe_float(candidate.get("rank_hidden_branch"), float(k))
    fixed_rank = safe_float(candidate.get("rank_fixed_composite"), float(k))
    out["bridge_rank"] = bridge_rank / k
    out["bridge_score"] = safe_float(candidate.get("bridge_score"), 0.0)
    bridge_scores = [safe_float(c.get("bridge_score"), 0.0) for c in survivor_set.get("candidates") or []]
    out["bridge_score_margin"] = safe_float(candidate.get("bridge_score"), 0.0) - max(bridge_scores or [0.0])
    out["bridge_rank_advantage"] = (avg_rank - bridge_rank) / k
    out["bridge_top1_flag"] = 1.0 if int(bridge_rank) == 1 else 0.0
    out["bridge_top2_flag"] = 1.0 if int(bridge_rank) <= 2 else 0.0
    out["bridge_vs_old_rank_gap"] = (bridge_rank - old_rank) / k
    out["bridge_vs_hidden_rank_gap"] = (bridge_rank - hidden_rank) / k
    out["bridge_vs_fixed_rank_gap"] = (bridge_rank - fixed_rank) / k
    out["bridge_missing_flag"] = 0.0 if math.isfinite(safe_float(candidate.get("bridge_score"), float("nan"))) else 1.0
    domain = str(survivor_set.get("domain"))
    for d in DOMAINS:
        out[f"domain_{d}"] = 1.0 if domain == d else 0.0
    out["science_flag"] = 1.0 if domain == "science" else 0.0
    out["reasoning_flag"] = 1.0 if domain == "reasoning" else 0.0
    out["coding_flag"] = 1.0 if domain == "coding" else 0.0
    out["math_flag"] = 1.0 if domain == "math_simple_arithmetic" else 0.0
    out["science_x_bridge_rank"] = out["science_flag"] * out["bridge_rank"]
    out["science_x_old_rank"] = out["science_flag"] * (old_rank / k)
    out["science_x_hidden_rank"] = out["science_flag"] * (hidden_rank / k)
    out["science_x_rank_disagreement"] = out["science_flag"] * out["rank_variance"]
    out["reasoning_x_bridge_rank"] = out["reasoning_flag"] * out["bridge_rank"]
    out["reasoning_x_rank_disagreement"] = out["reasoning_flag"] * out["rank_variance"]
    out["coding_x_old_code_rank"] = out["coding_flag"] * (safe_float(candidate.get("rank_old_code_reasoning"), float(k)) / k)
    out["coding_x_verifier_available"] = out["coding_flag"] * (1.0 if candidate.get("coding_label") is not None else 0.0)
    out["expert_consensus_tie"] = 1.0 if out["top1_vote_count"] == 0.0 or out["rank_variance"] < 1e-9 else 0.0
    out["parse_success"] = 1.0 if candidate.get("parse_success", True) else 0.0
    out["repetition_rate"] = safe_float(candidate.get("repetition_rate"), 0.0)
    out["empty_output"] = 1.0 if candidate.get("empty_output") else 0.0
    out["hit_max_tokens"] = 1.0 if candidate.get("hit_max_tokens") else 0.0
    out["output_length_log"] = math.log1p(max(safe_float(candidate.get("output_length"), 0.0), 0.0))
    out["verifier_available"] = 1.0 if candidate.get("coding_label") is not None else 0.0
    out["syntax_valid"] = 1.0 if candidate.get("parse_success", True) else 0.0
    out["high_disagreement_flag"] = 1.0 if out["rank_variance"] > 0.20 else 0.0
    out["clean_branch_flag"] = 1.0 if candidate.get("branch_origin") == "clean_branch" else 0.0
    out["hidden_branch_origin_flag"] = 1.0 if candidate.get("branch_origin") == "hidden_branch" else 0.0
    return out


def feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    names = list(feature_names)

    def idxs(pred: Any) -> list[int]:
        return [i for i, name in enumerate(names) if pred(name)]

    groups = {
        "rank_only": idxs(lambda n: n.startswith("rank_") or n.endswith("_top1_flag") or n.endswith("_top2_flag") or n in {"average_rank", "median_rank", "rank_variance", "rank_entropy", "top1_vote_count", "top2_vote_count"}),
        "bridge_preserving": idxs(lambda n: "bridge" in n or n.startswith("domain_") or n.endswith("_flag") or n in {"average_rank", "median_rank", "rank_variance", "rank_entropy", "top1_vote_count", "top2_vote_count"}),
        "tie_aware_rank": idxs(lambda n: n.startswith("rank_") or n.endswith("_top1_flag") or n.endswith("_top2_flag") or n in {"average_rank", "median_rank", "rank_variance", "rank_entropy", "top1_vote_count", "top2_vote_count", "expert_consensus_tie"}),
        "domain_gated": idxs(lambda n: n.startswith("rank_") or n.startswith("domain_") or "_x_" in n or n in {"science_flag", "reasoning_flag", "coding_flag", "math_flag", "rank_variance", "top1_vote_count", "top2_vote_count"}),
        "science_residual": idxs(lambda n: n.startswith("rank_") or "bridge" in n or "science" in n or "reasoning" in n or n.startswith("domain_") or n in {"rank_variance", "rank_entropy", "top1_vote_count", "top2_vote_count"}),
        "full_features": list(range(len(names))),
        "score_features": idxs(lambda n: n.endswith("_score") or n.endswith("_margin")),
        "domain_features": idxs(lambda n: n.startswith("domain_") or n in {"science_flag", "reasoning_flag", "coding_flag", "math_flag"} or "_x_" in n),
        "bridge_rank": idxs(lambda n: "bridge" in n and "score" not in n),
        "bridge_score": idxs(lambda n: "bridge" in n and ("score" in n or "margin" in n)),
        "old_rank": idxs(lambda n: "old" in n and n.startswith("rank_")),
        "hidden_rank": idxs(lambda n: ("hidden" in n or "branch" in n) and n.startswith("rank_")),
        "tie_features": idxs(lambda n: "tie" in n),
        "metadata": idxs(lambda n: n.endswith("_flag") or n.startswith("domain_")),
    }
    groups["rank_heavy_bridge"] = sorted(set(groups["rank_only"] + groups["bridge_preserving"] + groups["domain_features"]))
    return {key: value for key, value in groups.items() if value}


def build_feature_payload() -> dict[str, Any]:
    dataset = load_dataset_payload()
    sets = dataset.get("survivor_sets") or []
    feature_dicts = []
    candidate_rows = []
    feature_sets = []
    offset = 0
    for set_idx, s in enumerate(sets):
        add_consensus_ranks(s)
        best_ids = set(str(cid) for cid in (s.get("best_candidate_ids") or []))
        for cand_idx, c in enumerate(s.get("candidates") or []):
            feature_dicts.append(feature_map(c, s))
            candidate_rows.append(
                {
                    "global_candidate_idx": offset + cand_idx,
                    "candidate_local_idx": cand_idx,
                    "set_idx": set_idx,
                    "survivor_set_id": s.get("survivor_set_id"),
                    "task_id": s.get("task_id"),
                    "domain": s.get("domain"),
                    "split": s.get("split"),
                    "candidate_id": c.get("candidate_id"),
                    "reward": safe_float(c.get("final_reward"), 0.0),
                    "correctness": safe_float(c.get("correctness"), 0.0),
                    "is_best": 1.0 if str(c.get("candidate_id")) in best_ids else 0.0,
                    "parse_success": 1.0 if c.get("parse_success", True) else 0.0,
                    "stable": 1.0 if c.get("parse_success", True) and not c.get("empty_output") and safe_float(c.get("repetition_rate"), 0.0) < 0.75 else 0.0,
                }
            )
        n = len(s.get("candidates") or [])
        feature_sets.append(
            {
                **{k: val for k, val in s.items() if k != "candidates"},
                "candidate_feature_indices": list(range(offset, offset + n)),
                "candidate_rows": candidate_rows[offset : offset + n],
                "rewards": [candidate_rows[offset + j]["reward"] for j in range(n)],
                "best_mask": [candidate_rows[offset + j]["is_best"] for j in range(n)],
                "best_candidate_ids": list(best_ids),
            }
        )
        offset += n
    feature_names = sorted({key for fmap in feature_dicts for key in fmap})
    x_raw = torch.tensor([[safe_float(fmap.get(name), 0.0) for name in feature_names] for fmap in feature_dicts], dtype=torch.float32)
    groups = feature_groups(feature_names)
    return {
        "feature_names": feature_names,
        "feature_groups": groups,
        "x_raw": x_raw,
        "candidate_rows": candidate_rows,
        "sets": feature_sets,
        "raw_sets": sets,
    }


def run_features() -> int:
    ensure_root()
    started = time.time()
    if not DATASET_PT.exists():
        run_dataset()
    payload = build_feature_payload()
    names = payload["feature_names"]
    groups = payload["feature_groups"]
    if not payload["candidate_rows"]:
        verdict = "BLOCKED"
    elif not groups.get("rank_only"):
        verdict = "RANK_FEATURES_MISSING"
    elif not groups.get("bridge_preserving"):
        verdict = "BRIDGE_FEATURES_MISSING"
    else:
        verdict = "READY"
    out = {
        "BG_FINAL_ARBITER_V1_1_FEATURES_VERDICT": verdict,
        "verdict": verdict,
        **payload,
        "counts": {
            "feature_count": len(names),
            "candidate_rows": len(payload["candidate_rows"]),
            "sets": len(payload["sets"]),
            "by_split": dict(Counter(str(row.get("split")) for row in payload["candidate_rows"])),
            "by_domain": dict(Counter(str(row.get("domain")) for row in payload["candidate_rows"])),
            "groups": {key: len(val) for key, val in groups.items()},
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(out, FEATURES_PT)
    write_json(FEATURES_JSON, {k: val for k, val in out.items() if k not in {"x_raw", "raw_sets", "sets"}} | {"feature_names": names, "feature_groups": {k: len(v) for k, v in groups.items()}})
    lines = ["# Final Arbiter V1.1 Features", "", f"BG_FINAL_ARBITER_V1_1_FEATURES_VERDICT = {verdict}", "", f"- feature count: `{len(names)}`", f"- candidates: `{len(payload['candidate_rows'])}`", "", "## Feature Groups", ""]
    lines.extend(md_table([{"group": k, "features": len(v)} for k, v in groups.items()], ["group", "features"]))
    lines.extend(["", "## Feature Names", ""])
    lines.extend(f"- `{name}`" for name in names)
    write_md(FEATURES_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_FEATURES_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def load_features_payload() -> dict[str, Any]:
    global _FEATURE_CACHE
    if _FEATURE_CACHE is not None:
        return _FEATURE_CACHE
    if not FEATURES_PT.exists():
        run_features()
    _FEATURE_CACHE = torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
    return _FEATURE_CACHE


def candidate_rows_for_set(s: dict[str, Any]) -> list[dict[str, Any]]:
    return list(s.get("candidate_rows") or [])


def raw_candidates_for_set(s: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sets = load_features_payload().get("raw_sets") or []
    idx = int(s.get("set_idx", 0))
    return list((raw_sets[idx] if idx < len(raw_sets) else {}).get("candidates") or [])


def aggregate_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str] = ("policy",)) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["::".join(str(row.get(k)) for k in keys)].append(row)
    out = {}
    for key, vals in grouped.items():
        nums: dict[str, list[float]] = defaultdict(list)
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in vals:
            by_task[str(row.get("task_id"))].append(row)
            for name, value in row.items():
                x = safe_float(value)
                if math.isfinite(x):
                    nums[name].append(x)
        task_macro = [finite_mean([r.get("final_selected_reward") for r in task_rows], 0.0) for task_rows in by_task.values()]
        out[key] = {
            "n": len(vals),
            **{name: finite_mean(values, 0.0) for name, values in nums.items()},
            "task_macro_final_reward": finite_mean(task_macro, 0.0),
        }
    return out


def metrics_for_choices(sets: Sequence[dict[str, Any]], choices: dict[str, int], policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for s in sets:
        cand_rows = candidate_rows_for_set(s)
        if not cand_rows:
            continue
        idx = max(0, min(int(choices.get(str(s.get("survivor_set_id")), 0)), len(cand_rows) - 1))
        rewards = [safe_float(row.get("reward"), 0.0) for row in cand_rows]
        best = max(rewards)
        selected = cand_rows[idx]
        reward = safe_float(selected.get("reward"), 0.0)
        rows.append(
            {
                "policy": policy,
                "survivor_set_id": s.get("survivor_set_id"),
                "task_id": s.get("task_id"),
                "domain": s.get("domain"),
                "split": s.get("split"),
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_index": idx,
                "final_selected_reward": reward,
                "final_selected_correctness": safe_float(selected.get("correctness"), 0.0),
                "top1_success": 1.0 if reward == best else 0.0,
                "oracle_selected_rate": 1.0 if reward == best else 0.0,
                "strict_accuracy": 1.0 if reward == best else 0.0,
                "tie_relaxed_accuracy": 1.0 if reward == best else 0.0,
                "regret": best - reward,
                "tie_adjusted_regret": best - reward,
                "best_survivor_reward": best,
                "parse_success_rate": safe_float(selected.get("parse_success"), 1.0),
                "stable_rate": safe_float(selected.get("stable"), 1.0),
            }
        )
    metrics = aggregate_rows(rows)
    return metrics.get(policy, {"n": 0}), rows


def rank_average_choice(s: dict[str, Any], experts: Sequence[str] = RANK_EXPERTS, weights: dict[str, float] | None = None) -> int:
    raw = raw_candidates_for_set(s)
    if not raw:
        return 0
    weights = weights or {}
    scores = []
    for idx, c in enumerate(raw):
        total = 0.0
        denom = 0.0
        for expert in experts:
            r = safe_float(c.get(f"rank_{expert}"), float(len(raw)))
            w = safe_float(weights.get(expert), 1.0)
            total += w * r
            denom += abs(w)
        scores.append((total / max(denom, 1e-9), idx))
    return sorted(scores)[0][1]


def choose_by_score(s: dict[str, Any], score_key: str) -> int:
    raw = raw_candidates_for_set(s)
    return sorted(range(len(raw)), key=lambda i: (-safe_float(raw[i].get(score_key), 0.0), i))[0] if raw else 0


def deterministic_random_choice(s: dict[str, Any]) -> int:
    n = len(candidate_rows_for_set(s))
    return stable_hash(str(s.get("survivor_set_id"))) % max(n, 1)


def oracle_choice(s: dict[str, Any]) -> int:
    rewards = [safe_float(row.get("reward"), 0.0) for row in candidate_rows_for_set(s)]
    return sorted(range(len(rewards)), key=lambda i: (-rewards[i], i))[0] if rewards else 0


def v1_selected_model_choice(s: dict[str, Any]) -> int:
    artifact = v1.load_model_artifact()
    model = v1.instantiate_selected_model(artifact)
    payload = v1.load_features_payload()
    if model is None:
        return 0
    indices = s.get("v1_candidate_feature_indices") or []
    if not indices:
        return 0
    with torch.no_grad():
        scores = model(payload["x"][indices])
    return int(torch.argmax(scores).item()) if scores.numel() else 0


def baseline_choice(s: dict[str, Any], policy: str) -> int:
    raw = raw_candidates_for_set(s)
    if not raw:
        return 0
    if policy == "random_survivor":
        return deterministic_random_choice(s)
    if policy in {"first_survivor", "fixed_composite_top1"}:
        return rank_average_choice(s, ["fixed_composite"])
    if policy == "clean_branch_if_present":
        for idx, c in enumerate(raw):
            if c.get("branch_origin") == "clean_branch":
                return idx
        return 0
    score_policies = {
        "old_frozen_bg_top1": "old_frozen_bg_score",
        "old_objective_mixed_top1": "old_objective_mixed_score",
        "old_code_reasoning_top1": "old_code_reasoning_score",
        "v4_hidden_origin_top1": "v4_hidden_origin_score",
        "bridge_top1": "bridge_score",
        "bridge_only_top1": "bridge_only_score",
        "hidden_branch_top1": "hidden_branch_score",
        "universal_top1": "universal_score",
        "learned_gated_top1": "gated_score",
    }
    if policy in score_policies:
        return choose_by_score(s, score_policies[policy])
    if policy in {"majority_rank_aggregation", "borda_rank_sum_aggregation", "ranks_only_simple_aggregation"}:
        return rank_average_choice(s)
    if policy == "median_rank_aggregation":
        vals = []
        for idx, c in enumerate(raw):
            rank_vals = [safe_float(c.get(f"rank_{expert}"), float(len(raw))) for expert in RANK_EXPERTS]
            vals.append((float(median(rank_vals)) if rank_vals else 99.0, idx))
        return sorted(vals)[0][1]
    if policy == "bridge_preserving_rank_rule":
        return rank_average_choice(s, weights={"bridge": 2.4, "bridge_only": 1.5, "hidden_branch": 1.2, "fixed_composite": 1.0, "old": 0.8})
    if policy == "domain_rule_baseline":
        domain = str(s.get("domain"))
        if domain == "coding":
            return rank_average_choice(s, ["old_code_reasoning", "old_objective_mixed", "fixed_composite", "old"], {"old_code_reasoning": 2.0})
        if domain == "science":
            return rank_average_choice(s, weights={"bridge": 2.5, "hidden_branch": 1.3, "v4_hidden_origin": 1.0, "fixed_composite": 1.0})
        if domain == "reasoning":
            return rank_average_choice(s, weights={"bridge": 1.7, "fixed_composite": 1.3, "old": 1.0, "hidden_branch": 1.0})
        if domain == "math_simple_arithmetic":
            return rank_average_choice(s, ["fixed_composite", "old", "old_objective_mixed"])
        return rank_average_choice(s)
    if policy == "final_arbiter_v1_selected_model_replay":
        return v1_selected_model_choice(s)
    if policy == "oracle_best_survivor":
        return oracle_choice(s)
    return 0


def baseline_policies() -> list[str]:
    return [
        "random_survivor",
        "first_survivor",
        "clean_branch_if_present",
        "fixed_composite_top1",
        "old_frozen_bg_top1",
        "old_objective_mixed_top1",
        "old_code_reasoning_top1",
        "v4_hidden_origin_top1",
        "bridge_top1",
        "bridge_only_top1",
        "hidden_branch_top1",
        "universal_top1",
        "learned_gated_top1",
        "majority_rank_aggregation",
        "borda_rank_sum_aggregation",
        "median_rank_aggregation",
        "ranks_only_simple_aggregation",
        "bridge_preserving_rank_rule",
        "domain_rule_baseline",
        "final_arbiter_v1_selected_model_replay",
        "oracle_best_survivor",
    ]


def evaluate_baselines(split: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_features_payload()
    sets = [s for s in payload["sets"] if split is None or s.get("split") == split]
    rows = []
    for policy in baseline_policies():
        choices = {str(s.get("survivor_set_id")): baseline_choice(s, policy) for s in sets}
        _, policy_rows = metrics_for_choices(sets, choices, policy)
        rows.extend(policy_rows)
    return rows, aggregate_rows(rows)


def run_baselines() -> int:
    ensure_root()
    started = time.time()
    if not FEATURES_PT.exists():
        run_features()
    rows, metrics = evaluate_baselines()
    fresh_rows, fresh_metrics = evaluate_baselines("fresh_holdout")
    replay_rows, replay_metrics = evaluate_baselines("v1_heldout_replay_diagnostic")
    verdict = "READY" if rows else "BLOCKED"
    payload = {
        "BG_FINAL_ARBITER_V1_1_BASELINES_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "metrics_by_policy": metrics,
        "fresh_holdout_metrics_by_policy": fresh_metrics,
        "v1_heldout_replay_metrics_by_policy": replay_metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(BASELINES_JSON, payload)
    write_csv(BASELINES_CSV, rows)
    table = [{"policy": key, **value} for key, value in sorted(fresh_metrics.items())]
    lines = ["# Final Arbiter V1.1 Baselines", "", f"BG_FINAL_ARBITER_V1_1_BASELINES_VERDICT = {verdict}", "", "## Fresh Holdout Metrics", ""]
    lines.extend(md_table(table, ["policy", "n", "final_selected_reward", "task_macro_final_reward", "top1_success", "regret"]))
    write_md(BASELINES_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_BASELINES_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


class LinearScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class TinyMLPScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = min(32, max(8, dim // 2))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_loss(scores: torch.Tensor, rewards: Sequence[float]) -> torch.Tensor:
    r = torch.tensor([safe_float(v, 0.0) for v in rewards], dtype=torch.float32)
    if not r.numel():
        return scores.sum() * 0.0
    best = float(r.max().item())
    target = (r == best).to(torch.float32)
    target = target / target.sum().clamp_min(1.0)
    return -(target * F.log_softmax(scores, dim=0)).sum()


def model_family_spec(family: str, groups: dict[str, list[int]]) -> tuple[list[int], str]:
    if family == "rank_only_listwise":
        return groups["rank_only"], "linear"
    if family == "rank_heavy_bridge_preserving_listwise":
        return groups["rank_heavy_bridge"], "linear"
    if family == "tie_aware_rank_listwise":
        return groups["tie_aware_rank"], "linear"
    if family == "domain_gated_rank_arbiter":
        return groups["domain_gated"], "linear"
    if family == "science_residual_arbiter":
        return groups["science_residual"], "linear"
    if family == "robust_rank_aggregation":
        return groups["rank_only"], "linear"
    if family == "full_feature_listwise_v1_retrain":
        return groups["full_features"], "linear"
    if family == "tiny_mlp_rank_bridge":
        return groups["rank_heavy_bridge"], "mlp"
    return groups["rank_only"], "linear"


def make_model(kind: str, dim: int) -> nn.Module:
    return TinyMLPScorer(dim) if kind == "mlp" else LinearScorer(dim)


def split_sets(split: str) -> list[dict[str, Any]]:
    return [s for s in load_features_payload()["sets"] if s.get("split") == split]


def train_candidate_indices(sets: Sequence[dict[str, Any]]) -> list[int]:
    return [idx for s in sets for idx in (s.get("candidate_feature_indices") or [])]


def normalized_x(x_raw: torch.Tensor, feature_indices: Sequence[int], train_indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x_raw[:, list(feature_indices)]
    if train_indices:
        train_x = x[list(train_indices)]
        mean_vec = train_x.mean(dim=0)
        std_vec = train_x.std(dim=0).clamp_min(1e-6)
    else:
        mean_vec = torch.zeros(x.shape[1], dtype=torch.float32)
        std_vec = torch.ones(x.shape[1], dtype=torch.float32)
    return (x - mean_vec) / std_vec, mean_vec, std_vec


def evaluate_model(model: nn.Module, sets: Sequence[dict[str, Any]], x: torch.Tensor, policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    choices = {}
    with torch.no_grad():
        for s in sets:
            indices = s.get("candidate_feature_indices") or []
            scores = model(x[indices]) if indices else torch.empty(0)
            choices[str(s.get("survivor_set_id"))] = int(torch.argmax(scores).item()) if scores.numel() else 0
    return metrics_for_choices(sets, choices, policy)


def train_one_family(family: str, seed: int, lr: float, epochs: int = 100) -> dict[str, Any]:
    payload = load_features_payload()
    groups = payload["feature_groups"]
    feature_indices, kind = model_family_spec(family, groups)
    train_sets = split_sets("train")
    val_sets = split_sets("val")
    x, mean_vec, std_vec = normalized_x(payload["x_raw"], feature_indices, train_candidate_indices(train_sets))
    torch.manual_seed(seed)
    model = make_model(kind, x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_state = None
    best_val = -999.0
    best_epoch = 0
    stale = 0
    train_log = []
    for epoch in range(epochs):
        model.train()
        order = sorted(range(len(train_sets)), key=lambda i: stable_hash(f"{family}:{seed}:{lr}:{epoch}:{train_sets[i].get('survivor_set_id')}"))
        losses = []
        for set_idx in order:
            s = train_sets[set_idx]
            indices = s.get("candidate_feature_indices") or []
            if len(indices) < 2:
                continue
            scores = model(x[indices])
            loss = set_loss(scores, s.get("rewards") or [])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().item()))
        train_metrics, _ = evaluate_model(model, train_sets, x, f"{family}_train")
        val_metrics, _ = evaluate_model(model, val_sets, x, f"{family}_val")
        val_score = safe_float(val_metrics.get("task_macro_final_reward"), -999.0)
        train_log.append({"epoch": epoch, "loss": finite_mean(losses, 0.0), "train": train_metrics, "val": val_metrics})
        if val_score > best_val + 1e-9:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 18:
            break
    if best_state:
        model.load_state_dict(best_state)
    train_metrics, _ = evaluate_model(model, train_sets, x, family)
    val_metrics, val_rows = evaluate_model(model, val_sets, x, family)
    return {
        "family": family,
        "seed": seed,
        "lr": lr,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_rows": val_rows,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_kind": kind,
        "feature_indices": list(feature_indices),
        "feature_names": [payload["feature_names"][i] for i in feature_indices],
        "normalization": {"mean": mean_vec.detach().cpu(), "std": std_vec.detach().cpu(), "fit_split": "train"},
        "train_log": train_log,
    }


def run_training() -> int:
    ensure_root()
    started = time.time()
    if not BASELINES_JSON.exists():
        run_baselines()
    families = [
        "rank_only_listwise",
        "rank_heavy_bridge_preserving_listwise",
        "tie_aware_rank_listwise",
        "domain_gated_rank_arbiter",
        "science_residual_arbiter",
        "robust_rank_aggregation",
        "full_feature_listwise_v1_retrain",
        "tiny_mlp_rank_bridge",
    ]
    runs = []
    rows = []
    science_support = len([s for s in split_sets("train") + split_sets("val") if s.get("domain") == "science"])
    for family in families:
        if family == "science_residual_arbiter" and science_support < 20:
            continue
        for lr in (1e-4, 3e-4, 1e-3):
            for seed in (SEED, SEED + 1, SEED + 2):
                run = train_one_family(family, seed, lr)
                runs.append(run)
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "lr": lr,
                        "best_epoch": run["best_epoch"],
                        **{f"val_{k}": val for k, val in run["val_metrics"].items() if isinstance(val, (int, float))},
                        **{f"train_{k}": val for k, val in run["train_metrics"].items() if isinstance(val, (int, float))},
                    }
                )
    best = max(runs, key=lambda r: safe_float(r["val_metrics"].get("task_macro_final_reward"), -999.0)) if runs else None
    baseline_payload = load_json(BASELINES_JSON, {}) or {}
    val_baselines = aggregate_rows([r for r in baseline_payload.get("rows") or [] if r.get("split") == "val"])
    majority_val = safe_float(val_baselines.get("majority_rank_aggregation", {}).get("task_macro_final_reward"), -999.0)
    fixed_val = safe_float(val_baselines.get("fixed_composite_top1", {}).get("task_macro_final_reward"), -999.0)
    v1_val = safe_float(val_baselines.get("final_arbiter_v1_selected_model_replay", {}).get("task_macro_final_reward"), -999.0)
    best_val = safe_float((best or {}).get("val_metrics", {}).get("task_macro_final_reward"), -999.0)
    best_family = str((best or {}).get("family", ""))
    if not best:
        verdict = "INSUFFICIENT"
    elif best_val > max(majority_val, fixed_val, v1_val) + 1e-9 and "rank_heavy_bridge" in best_family:
        verdict = "BRIDGE_PRESERVING_BEST"
    elif best_val > max(majority_val, fixed_val, v1_val) + 1e-9 and "rank_only" in best_family:
        verdict = "RANK_ONLY_BEST"
    elif best_val > max(majority_val, fixed_val, v1_val) + 1e-9:
        verdict = "READY"
    elif best_val > max(majority_val, fixed_val) + 1e-9:
        verdict = "WEAK"
    elif best and best_val < safe_float(best.get("train_metrics", {}).get("task_macro_final_reward"), 0.0) - 0.20:
        verdict = "OVERFIT"
    else:
        verdict = "NO_LEARNING"
    artifact = {
        "BG_FINAL_ARBITER_V1_1_TRAINING_VERDICT": verdict,
        "verdict": verdict,
        "selected_model": {k: val for k, val in (best or {}).items() if k not in {"train_log", "val_rows"}},
        "all_runs": [{k: val for k, val in r.items() if k not in {"state_dict", "train_log", "val_rows", "normalization"}} for r in runs],
        "training_rows": rows,
        "baseline_validation": {
            "majority_rank_aggregation": majority_val,
            "fixed_composite_top1": fixed_val,
            "final_arbiter_v1_selected_model_replay": v1_val,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(artifact, MODEL_PT)
    write_json(TRAINING_JSON, {k: val for k, val in artifact.items() if k not in {"selected_model"} or isinstance(val, (str, int, float, list, dict))})
    lines = ["# Final Arbiter V1.1 Training", "", f"BG_FINAL_ARBITER_V1_1_TRAINING_VERDICT = {verdict}", "", f"- selected family: `{best_family}`", f"- selected val task macro reward: `{best_val:.4f}`", f"- majority val task macro reward: `{majority_val:.4f}`", f"- fixed val task macro reward: `{fixed_val:.4f}`", f"- v1 replay val task macro reward: `{v1_val:.4f}`", "", "## Runs", ""]
    lines.extend(md_table(rows, ["family", "seed", "lr", "best_epoch", "val_task_macro_final_reward", "train_task_macro_final_reward", "val_regret"]))
    write_md(TRAINING_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_TRAINING_VERDICT = {verdict}", flush=True)
    return 0


def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PT.exists():
        run_training()
    return torch.load(MODEL_PT, map_location="cpu", weights_only=False)


def instantiate_selected(artifact: dict[str, Any]) -> tuple[nn.Module | None, torch.Tensor | None]:
    selected = artifact.get("selected_model") or {}
    payload = load_features_payload()
    feature_indices = list(selected.get("feature_indices") or [])
    if not feature_indices:
        return None, None
    x = payload["x_raw"][:, feature_indices]
    norm = selected.get("normalization") or {}
    mean_vec = norm.get("mean")
    std_vec = norm.get("std")
    if not isinstance(mean_vec, torch.Tensor) or not isinstance(std_vec, torch.Tensor):
        train_sets = split_sets("train")
        x, mean_vec, std_vec = normalized_x(payload["x_raw"], feature_indices, train_candidate_indices(train_sets))
    else:
        x = (x - mean_vec) / std_vec.clamp_min(1e-6)
    model = make_model(str(selected.get("model_kind", "linear")), x.shape[1])
    state = selected.get("state_dict")
    if state:
        model.load_state_dict(state)
    model.eval()
    return model, x


def evaluate_selected_on_split(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not MODEL_PT.exists():
        run_training()
    payload = load_features_payload()
    artifact = load_model_artifact()
    sets = [s for s in payload["sets"] if s.get("split") == split]
    rows, metrics = evaluate_baselines(split)
    model, x = instantiate_selected(artifact)
    if model is not None and x is not None:
        policy = f"v1_1_{(artifact.get('selected_model') or {}).get('family', 'arbiter')}"
        m, r = evaluate_model(model, sets, x, policy)
        rows.extend(r)
        metrics[policy] = m
    return rows, metrics


def run_heldout_eval() -> int:
    ensure_root()
    started = time.time()
    rows, metrics = evaluate_selected_on_split("fresh_holdout")
    replay_rows, replay_metrics = evaluate_selected_on_split("v1_heldout_replay_diagnostic")
    artifact = load_model_artifact()
    trained_policy = f"v1_1_{(artifact.get('selected_model') or {}).get('family', 'arbiter')}"
    trained = metrics.get(trained_policy, {})
    majority = metrics.get("majority_rank_aggregation", {})
    fixed = metrics.get("fixed_composite_top1", {})
    v1_replay = metrics.get("final_arbiter_v1_selected_model_replay", {})
    oracle = metrics.get("oracle_best_survivor", {})
    trained_task = safe_float(trained.get("task_macro_final_reward"), -999.0)
    majority_task = safe_float(majority.get("task_macro_final_reward"), -999.0)
    fixed_task = safe_float(fixed.get("task_macro_final_reward"), -999.0)
    v1_task = safe_float(v1_replay.get("task_macro_final_reward"), -999.0)
    oracle_task = safe_float(oracle.get("task_macro_final_reward"), -999.0)
    gap = max(oracle_task - majority_task, 1e-9)
    closure = (trained_task - majority_task) / gap
    by_domain = aggregate_rows(rows, ("policy", "domain"))
    coding = by_domain.get(f"{trained_policy}::coding", {})
    math_row = by_domain.get(f"{trained_policy}::math_simple_arithmetic", {})
    science = by_domain.get(f"{trained_policy}::science", {})
    fixed_coding = by_domain.get("fixed_composite_top1::coding", {})
    fixed_math = by_domain.get("fixed_composite_top1::math_simple_arithmetic", {})
    coding_ok = safe_float(coding.get("task_macro_final_reward"), 0.0) >= safe_float(fixed_coding.get("task_macro_final_reward"), 0.0) - 1e-9
    math_ok = safe_float(math_row.get("task_macro_final_reward"), 0.0) >= safe_float(fixed_math.get("task_macro_final_reward"), 0.0) - 1e-9
    science_not_worse = safe_float(science.get("task_macro_final_reward"), 0.0) >= V1_REFERENCE["science_task_macro"] - 1e-9
    checks = {
        "task_macro_final_reward_ge_0_75": trained_task >= 0.75,
        "improves_majority": trained_task > majority_task + 1e-9,
        "improves_fixed": trained_task > fixed_task + 1e-9,
        "improves_v1_selected_replay": trained_task > v1_task + 1e-9,
        "coding_not_degraded": coding_ok,
        "math_not_degraded": math_ok,
        "science_not_worse_than_v1_reference": science_not_worse,
        "gap_closure_ge_35pct": closure >= 0.35,
    }
    family = str((artifact.get("selected_model") or {}).get("family", ""))
    if all(checks[k] for k in ["task_macro_final_reward_ge_0_75", "improves_majority", "improves_fixed", "improves_v1_selected_replay", "coding_not_degraded", "math_not_degraded", "science_not_worse_than_v1_reference"]):
        verdict = "FINAL_ARBITER_READY"
    elif trained_task > V1_REFERENCE["trained_task_macro"] + 1e-9 and "rank" in family:
        verdict = "RANK_HEAVY_CONFIRMED"
    elif trained_task > V1_REFERENCE["trained_task_macro"] + 1e-9:
        verdict = "FINAL_ARBITER_WEAK_BUT_IMPROVED"
    elif not science_not_worse:
        verdict = "SCIENCE_WEAK"
    elif rows:
        verdict = "NO_IMPROVEMENT"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "readiness_eval_mode": "fresh_v1_1_task_holdout",
        "trained_policy": trained_policy,
        "rows": rows,
        "metrics_by_policy": metrics,
        "metrics_by_policy_domain": by_domain,
        "v1_heldout_replay_diagnostic": {"rows": replay_rows, "metrics_by_policy": replay_metrics},
        "gap_closure_from_majority_to_oracle": closure,
        "success_checks": checks,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(HELDOUT_JSON, payload)
    write_csv(HELDOUT_CSV, rows)
    table = [{"policy": key, **value} for key, value in sorted(metrics.items())]
    lines = ["# Final Arbiter V1.1 Heldout Evaluation", "", f"BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT = {verdict}", "", f"- readiness eval mode: `fresh_v1_1_task_holdout`", f"- trained policy: `{trained_policy}`", f"- gap closure: `{closure:.3f}`", "", "## Fresh Holdout Metrics", ""]
    lines.extend(md_table(table, ["policy", "n", "final_selected_reward", "task_macro_final_reward", "top1_success", "regret"]))
    write_md(HELDOUT_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_domain_analysis() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON.exists():
        run_heldout_eval()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    trained_policy = heldout.get("trained_policy")
    by_domain = heldout.get("metrics_by_policy_domain") or {}
    rows = []
    for domain in DOMAINS:
        trained = by_domain.get(f"{trained_policy}::{domain}", {})
        fixed = by_domain.get(f"fixed_composite_top1::{domain}", {})
        majority = by_domain.get(f"majority_rank_aggregation::{domain}", {})
        oracle = by_domain.get(f"oracle_best_survivor::{domain}", {})
        v1_base = V1_REFERENCE.get(f"{domain.split('_')[0]}_task_macro", float("nan"))
        rows.append(
            {
                "domain": domain,
                "survivor_sets": trained.get("n"),
                "v1_reference_task_macro": v1_base,
                "v1_1_task_macro_reward": trained.get("task_macro_final_reward"),
                "fixed_task_macro_reward": fixed.get("task_macro_final_reward"),
                "majority_task_macro_reward": majority.get("task_macro_final_reward"),
                "oracle_task_macro_reward": oracle.get("task_macro_final_reward"),
                "improvement_vs_v1_reference": safe_float(trained.get("task_macro_final_reward"), 0.0) - safe_float(v1_base, 0.0),
                "regret": trained.get("regret"),
                "oracle_selected_rate": trained.get("oracle_selected_rate"),
            }
        )
    science = next((r for r in rows if r["domain"] == "science"), {})
    reasoning = next((r for r in rows if r["domain"] == "reasoning"), {})
    coding = next((r for r in rows if r["domain"] == "coding"), {})
    if safe_float(coding.get("v1_1_task_macro_reward"), 0.0) < min(V1_REFERENCE["coding_task_macro"], safe_float(coding.get("fixed_task_macro_reward"), 0.0)) - 1e-9:
        verdict = "CODING_DEGRADES"
    elif safe_float(science.get("improvement_vs_v1_reference"), 0.0) > 0.02:
        verdict = "SCIENCE_IMPROVED"
    elif safe_float(reasoning.get("improvement_vs_v1_reference"), 0.0) > 0.02:
        verdict = "REASONING_IMPROVED"
    elif safe_float(science.get("v1_1_task_macro_reward"), 0.0) < V1_REFERENCE["science_task_macro"] - 1e-9:
        verdict = "SCIENCE_REMAINS_WEAK"
    elif safe_float(reasoning.get("v1_1_task_macro_reward"), 0.0) < V1_REFERENCE["reasoning_task_macro"] - 1e-9:
        verdict = "REASONING_REMAINS_WEAK"
    elif heldout.get("verdict") == "FINAL_ARBITER_READY":
        verdict = "MULTIDOMAIN_READY"
    else:
        verdict = "CODING_PRESERVED"
    payload = {
        "BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT": verdict,
        "verdict": verdict,
        "domain_rows": rows,
        "trained_policy": trained_policy,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(DOMAIN_JSON, payload)
    lines = ["# Final Arbiter V1.1 Domain Analysis", "", f"BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT = {verdict}", "", "## Domains", ""]
    lines.extend(md_table(rows, ["domain", "survivor_sets", "v1_reference_task_macro", "v1_1_task_macro_reward", "fixed_task_macro_reward", "majority_task_macro_reward", "oracle_task_macro_reward", "improvement_vs_v1_reference"]))
    write_md(DOMAIN_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT = {verdict}", flush=True)
    return 0


def run_tie_analysis() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON.exists():
        run_heldout_eval()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    trained_policy = heldout.get("trained_policy")
    rows = [r for r in heldout.get("rows") or [] if r.get("policy") == trained_policy]
    raw_by_id = {s.get("survivor_set_id"): s for s in load_features_payload().get("raw_sets") or []}
    tie_rows = []
    for row in rows:
        s = raw_by_id.get(row.get("survivor_set_id"), {})
        tie_count = int(s.get("tie_count") or 0)
        tie_rows.append(
            {
                **row,
                "tie_count": tie_count,
                "is_tied_best_set": 1.0 if tie_count > 1 else 0.0,
                "tie_rate_domain": row.get("domain"),
            }
        )
    tied = [r for r in tie_rows if r["is_tied_best_set"]]
    untied = [r for r in tie_rows if not r["is_tied_best_set"]]
    summary = {
        "tie_set_count": len(tied),
        "tie_rate": len(tied) / max(len(tie_rows), 1),
        "strict_accuracy": finite_mean([r.get("strict_accuracy") for r in tie_rows], 0.0),
        "tied_best_relaxed_accuracy": finite_mean([r.get("tie_relaxed_accuracy") for r in tie_rows], 0.0),
        "tied_set_accuracy": finite_mean([r.get("tie_relaxed_accuracy") for r in tied], 0.0),
        "untied_set_accuracy": finite_mean([r.get("strict_accuracy") for r in untied], 0.0),
        "tie_rate_by_domain": dict(Counter(str(r.get("domain")) for r in tied)),
    }
    if summary["tie_rate"] >= 0.30 and summary["tied_set_accuracy"] >= summary["untied_set_accuracy"] - 0.05:
        verdict = "TIES_ARE_LABEL_NOISE"
    elif "tie_aware" in str(trained_policy) and summary["tied_set_accuracy"] >= summary["untied_set_accuracy"] - 0.10:
        verdict = "TIE_AWARENESS_HELPS"
    elif summary["tie_rate"] < 0.10:
        verdict = "TIES_NOT_MAIN_BLOCKER"
    elif summary["tied_set_accuracy"] < summary["untied_set_accuracy"] - 0.20:
        verdict = "TIE_HANDLING_WEAK"
    else:
        verdict = "TIES_NOT_MAIN_BLOCKER"
    payload = {
        "BG_FINAL_ARBITER_V1_1_TIE_VERDICT": verdict,
        "verdict": verdict,
        "summary": summary,
        "rows": tie_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(TIE_JSON, payload)
    write_csv(TIE_CSV, tie_rows)
    lines = ["# Final Arbiter V1.1 Tie Analysis", "", f"BG_FINAL_ARBITER_V1_1_TIE_VERDICT = {verdict}", "", f"- summary: `{summary}`"]
    write_md(TIE_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_TIE_VERDICT = {verdict}", flush=True)
    return 0


def evaluate_selected_with_mask(mask_name: str, remove_group: str | None = None, keep_group: str | None = None, split: str = "fresh_holdout") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = load_model_artifact()
    selected = artifact.get("selected_model") or {}
    payload = load_features_payload()
    feature_names = selected.get("feature_names") or []
    feature_indices = selected.get("feature_indices") or []
    model, x = instantiate_selected(artifact)
    if model is None or x is None:
        return {}, []
    x = x.clone()
    selected_name_to_col = {name: i for i, name in enumerate(feature_names)}
    global_groups = feature_groups(payload["feature_names"])
    if remove_group:
        remove_names = {payload["feature_names"][i] for i in global_groups.get(remove_group, [])}
        for name in remove_names:
            if name in selected_name_to_col:
                x[:, selected_name_to_col[name]] = 0.0
    if keep_group:
        keep_names = {payload["feature_names"][i] for i in global_groups.get(keep_group, [])}
        for name in feature_names:
            if name not in keep_names and name in selected_name_to_col:
                x[:, selected_name_to_col[name]] = 0.0
    sets = [s for s in payload["sets"] if s.get("split") == split]
    return evaluate_model(model, sets, x, f"ablation_{mask_name}")


def run_ablation() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON.exists():
        run_heldout_eval()
    summary_rows = []
    detail_rows = []
    base, rows = evaluate_selected_with_mask("all_rank_heavy_features")
    summary_rows.append({"ablation": "all_rank_heavy_features", **base})
    detail_rows.extend({**r, "ablation": "all_rank_heavy_features"} for r in rows)
    for group, label in [
        ("bridge_rank", "no_bridge_rank"),
        ("bridge_score", "no_bridge_score"),
        ("old_rank", "no_old_rank"),
        ("hidden_rank", "no_hidden_rank"),
        ("domain_features", "no_domain_features"),
        ("tie_features", "no_tie_features"),
        ("score_features", "no_score_features"),
        ("metadata", "no_metadata"),
    ]:
        metrics, rows = evaluate_selected_with_mask(label, remove_group=group)
        summary_rows.append({"ablation": label, **metrics})
        detail_rows.extend({**r, "ablation": label} for r in rows)
    for group, label in [("rank_only", "ranks_only"), ("bridge_preserving", "bridge_required")]:
        metrics, rows = evaluate_selected_with_mask(label, keep_group=group)
        summary_rows.append({"ablation": label, **metrics})
        detail_rows.extend({**r, "ablation": label} for r in rows)
    base_reward = safe_float(base.get("task_macro_final_reward"), 0.0)
    drops = {r["ablation"]: base_reward - safe_float(r.get("task_macro_final_reward"), 0.0) for r in summary_rows if r["ablation"] != "all_rank_heavy_features"}
    largest = max(drops, key=drops.get) if drops else ""
    artifact = load_model_artifact()
    family = str((artifact.get("selected_model") or {}).get("family", ""))
    if "rank" in family and safe_float(next((r for r in summary_rows if r["ablation"] == "ranks_only"), {}).get("task_macro_final_reward"), 0.0) >= base_reward - 0.02:
        verdict = "RANKS_DOMINATE"
    elif largest in {"no_bridge_rank", "no_bridge_score"} and drops.get(largest, 0.0) > 0.04:
        verdict = "BRIDGE_MANDATORY"
    elif largest == "no_domain_features":
        verdict = "DOMAIN_GATE_MATTERS"
    elif drops.get("no_score_features", 0.0) < -0.02:
        verdict = "RAW_SCORES_HURT"
    elif drops:
        verdict = "FULL_FEATURES_NEEDED"
    else:
        verdict = "INCONCLUSIVE"
    payload = {
        "BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT": verdict,
        "verdict": verdict,
        "ablation_summary": summary_rows,
        "largest_drop": largest,
        "rows": detail_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(ABLATION_JSON, payload)
    write_csv(ABLATION_CSV, detail_rows)
    lines = ["# Final Arbiter V1.1 Ablation", "", f"BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT = {verdict}", "", f"- largest drop: `{largest}`", "", "## Ablations", ""]
    lines.extend(md_table(summary_rows, ["ablation", "n", "task_macro_final_reward", "final_selected_reward", "top1_success", "regret"]))
    write_md(ABLATION_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT = {verdict}", flush=True)
    return 0


def run_calibration_ood() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON.exists():
        run_heldout_eval()
    stress_rows = []
    nominal, _ = evaluate_selected_with_mask("nominal")
    stress_rows.append({"stress": "nominal", **nominal})
    for group, label in [
        ("bridge_rank", "remove_bridge_score_rank"),
        ("old_rank", "remove_old_content_rank"),
        ("hidden_rank", "remove_hidden_branch_rank"),
        ("score_features", "remove_raw_scores_entirely"),
        ("domain_features", "remove_domain_features"),
    ]:
        metrics, _ = evaluate_selected_with_mask(label, remove_group=group)
        stress_rows.append({"stress": label, **metrics})
    heldout = load_json(HELDOUT_JSON, {}) or {}
    trained_policy = heldout.get("trained_policy")
    for domain in DOMAINS:
        vals = [r for r in heldout.get("rows") or [] if r.get("policy") == trained_policy and r.get("domain") == domain]
        stress_rows.append({"stress": f"{domain}_only", **(aggregate_rows(vals).get(trained_policy, {}))})
    nominal_reward = safe_float(nominal.get("task_macro_final_reward"), 0.0)
    worst = max([nominal_reward - safe_float(row.get("task_macro_final_reward"), nominal_reward) for row in stress_rows[1:]] or [0.0])
    bridge_drop = max([nominal_reward - safe_float(row.get("task_macro_final_reward"), nominal_reward) for row in stress_rows if "bridge" in str(row.get("stress"))] or [0.0])
    if worst <= 0.05:
        verdict = "ROBUST"
    elif bridge_drop > 0.12:
        verdict = "BRIDGE_MISSING_FRAGILE"
    elif worst <= 0.12:
        verdict = "CONSERVATIVE_BUT_SAFE"
    elif any(str(row.get("stress")).endswith("_only") and safe_float(row.get("task_macro_final_reward"), 0.0) < 0.50 for row in stress_rows):
        verdict = "DOMAIN_OOD_FRAGILE"
    else:
        verdict = "CALIBRATION_WEAK"
    payload = {
        "BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT": verdict,
        "verdict": verdict,
        "stress_rows": stress_rows,
        "worst_degradation": worst,
        "fallback_policy": "Bridge missing falls back to conservative rank aggregation; coding missing falls back to old/objective rank; science high disagreement falls back to tie-aware conservative rank aggregation.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(CALIBRATION_JSON, payload)
    lines = ["# Final Arbiter V1.1 Calibration/OOD", "", f"BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT = {verdict}", "", f"- worst degradation: `{worst:.4f}`", "", "## Stress", ""]
    lines.extend(md_table(stress_rows, ["stress", "n", "task_macro_final_reward", "final_selected_reward", "top1_success", "regret"]))
    write_md(CALIBRATION_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT = {verdict}", flush=True)
    return 0


def run_failure_analysis() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON.exists():
        run_heldout_eval()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    trained_policy = heldout.get("trained_policy")
    rows = [r for r in heldout.get("rows") or [] if r.get("policy") == trained_policy]
    raw_by_id = {s.get("survivor_set_id"): s for s in load_features_payload().get("raw_sets") or []}
    failure_rows = []
    summary = Counter()
    for row in rows:
        if safe_float(row.get("top1_success"), 0.0) >= 1.0:
            continue
        s = raw_by_id.get(row.get("survivor_set_id"), {})
        candidates = s.get("candidates") or []
        selected_id = str(row.get("selected_candidate_id"))
        selected = next((c for c in candidates if str(c.get("candidate_id")) == selected_id), {})
        domain = str(row.get("domain"))
        root = "rank_consensus_wrong"
        if int(s.get("tie_count") or 0) > 1:
            root = "tie_ambiguity"
        elif domain == "science":
            root = "science_weak"
        elif domain == "reasoning":
            root = "reasoning_weak"
        elif safe_float(selected.get("rank_bridge"), 99.0) <= 2:
            root = "bridge_misrank"
        elif any((s.get("missing_expert_mask") or {}).values()):
            root = "missing_expert"
        summary[root] += 1
        failure_rows.append(
            {
                "task_id": row.get("task_id"),
                "domain": domain,
                "survivor_set_id": row.get("survivor_set_id"),
                "selected_candidate_id": selected_id,
                "oracle_candidate_ids": s.get("best_candidate_ids"),
                "selected_reward": row.get("final_selected_reward"),
                "oracle_reward": s.get("best_reward"),
                "expert_ranks_selected": {f"rank_{expert}": selected.get(f"rank_{expert}") for expert in RANK_EXPERTS},
                "expert_scores_selected": {key: selected.get(key) for key in SCORE_KEYS},
                "root_cause_tag": root,
            }
        )
    if summary.get("science_weak", 0) >= max(1, sum(summary.values()) // 3):
        verdict = "SCIENCE_BLOCKER_REMAINS"
    elif summary.get("reasoning_weak", 0) >= max(1, sum(summary.values()) // 3):
        verdict = "REASONING_BLOCKER_REMAINS"
    elif summary.get("tie_ambiguity", 0) >= max(1, sum(summary.values()) // 3):
        verdict = "TIE_AMBIGUITY_REMAINS"
    elif failure_rows:
        verdict = "FAILURES_REDUCED"
    else:
        verdict = "FAILURES_REDUCED"
    payload = {
        "BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT": verdict,
        "verdict": verdict,
        "prior_v1_failures": {"expert_misrank": 28, "tie_ambiguity": 33, "science_weak": 24},
        "failure_summary": dict(summary),
        "failure_cases": failure_rows,
        "top_20_failure_cases": failure_rows[:20],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(FAILURE_JSON, payload)
    write_csv(FAILURE_CSV, failure_rows)
    lines = ["# Final Arbiter V1.1 Failure Analysis", "", f"BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT = {verdict}", "", f"- summary: `{dict(summary)}`", "", "## Top Failures", ""]
    lines.extend(md_table(failure_rows[:20], ["task_id", "domain", "root_cause_tag", "selected_reward", "oracle_reward", "selected_candidate_id", "oracle_candidate_ids"]))
    write_md(FAILURE_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT = {verdict}", flush=True)
    return 0


def run_selection_readiness() -> int:
    ensure_root()
    started = time.time()
    if not FAILURE_JSON.exists():
        run_failure_analysis()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    domain = load_json(DOMAIN_JSON, {}) or {}
    ood = load_json(CALIBRATION_JSON, {}) or {}
    failure = load_json(FAILURE_JSON, {}) or {}
    checks = heldout.get("success_checks") or {}
    if heldout.get("verdict") == "FINAL_ARBITER_READY" and ood.get("verdict") in {"ROBUST", "CONSERVATIVE_BUT_SAFE"}:
        verdict = "READY_FOR_STEERING_COMPARISON"
        blocker = ""
    elif domain.get("verdict") == "SCIENCE_REMAINS_WEAK" or failure.get("verdict") == "SCIENCE_BLOCKER_REMAINS":
        verdict = "NEEDS_SCIENCE_ARBITER"
        blocker = "science final arbitration remains weak"
    elif domain.get("verdict") == "REASONING_REMAINS_WEAK" or failure.get("verdict") == "REASONING_BLOCKER_REMAINS":
        verdict = "NEEDS_REASONING_ARBITER"
        blocker = "reasoning final arbitration remains weak"
    elif checks.get("improves_majority") or heldout.get("verdict") in {"RANK_HEAVY_CONFIRMED", "FINAL_ARBITER_WEAK_BUT_IMPROVED"}:
        verdict = "FINAL_ARBITER_WEAK_BUT_IMPROVED"
        blocker = "v1.1 improved but did not meet readiness threshold"
    else:
        verdict = "NOT_READY"
        blocker = "final arbiter did not improve enough"
    payload = {
        "BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT": verdict,
        "verdict": verdict,
        "checks": checks,
        "blocker": blocker,
        "no_steering_tested": True,
        "next_prompt": (
            "Phase 2b selection-only locked baseline vs selection + trained steering corridor"
            if verdict == "READY_FOR_STEERING_COMPARISON"
            else ""
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(READINESS_JSON, payload)
    lines = ["# Final Arbiter V1.1 Selection Readiness", "", f"BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT = {verdict}", "", f"- blocker: `{blocker or 'none'}`", "- no steering was tested."]
    write_md(READINESS_MD, lines)
    print(f"BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT = {verdict}", flush=True)
    return 0


def stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "split_guard": load_json(SPLIT_GUARD_JSON, {}) or {},
        "dataset": load_json(DATASET_JSON, {}) or {},
        "features": load_json(FEATURES_JSON, {}) or {},
        "baselines": load_json(BASELINES_JSON, {}) or {},
        "training": load_json(TRAINING_JSON, {}) or {},
        "heldout": load_json(HELDOUT_JSON, {}) or {},
        "domain": load_json(DOMAIN_JSON, {}) or {},
        "ties": load_json(TIE_JSON, {}) or {},
        "ablation": load_json(ABLATION_JSON, {}) or {},
        "ood": load_json(CALIBRATION_JSON, {}) or {},
        "failures": load_json(FAILURE_JSON, {}) or {},
        "readiness": load_json(READINESS_JSON, {}) or {},
    }


def payload_verdict(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or payload.get("verdict") or "INSUFFICIENT")


def final_status(data: dict[str, dict[str, Any]]) -> tuple[str, str]:
    heldout = payload_verdict(data["heldout"], "BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT")
    readiness = payload_verdict(data["readiness"], "BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT")
    domain = payload_verdict(data["domain"], "BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT")
    training_family = str(((data.get("training") or {}).get("selected_model") or {}).get("family", ""))
    if heldout == "FINAL_ARBITER_READY" and "rank" in training_family:
        arbiter_status = "RANK_HEAVY_READY"
    elif heldout == "FINAL_ARBITER_READY":
        arbiter_status = "FINAL_ARBITER_READY"
    elif heldout in {"RANK_HEAVY_CONFIRMED", "FINAL_ARBITER_WEAK_BUT_IMPROVED"}:
        arbiter_status = "FINAL_ARBITER_WEAK_BUT_IMPROVED"
    elif domain == "SCIENCE_REMAINS_WEAK":
        arbiter_status = "SCIENCE_LIMITED"
    elif domain == "REASONING_REMAINS_WEAK":
        arbiter_status = "REASONING_LIMITED"
    elif heldout == "NO_IMPROVEMENT":
        arbiter_status = "NO_IMPROVEMENT"
    else:
        arbiter_status = "NOT_READY"
    if readiness == "READY_FOR_STEERING_COMPARISON":
        phase_status = "READY_FOR_PHASE2B_STEERING_COMPARISON"
    elif readiness == "FINAL_ARBITER_WEAK_BUT_IMPROVED":
        phase_status = "NEEDS_MORE_FINAL_ARBITER_WORK"
    elif readiness in {"NEEDS_SCIENCE_ARBITER", "NEEDS_REASONING_ARBITER", "NEEDS_DOMAIN_GATE"}:
        phase_status = "NEEDS_DOMAIN_SPECIALIZATION"
    else:
        phase_status = "NOT_READY"
    return arbiter_status, phase_status


def top_lines(data: dict[str, dict[str, Any]], arbiter_status: str, phase_status: str) -> list[str]:
    return [
        f"BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT = {payload_verdict(data['split_guard'], 'BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_DATASET_VERDICT = {payload_verdict(data['dataset'], 'BG_FINAL_ARBITER_V1_1_DATASET_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_FEATURES_VERDICT = {payload_verdict(data['features'], 'BG_FINAL_ARBITER_V1_1_FEATURES_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_BASELINES_VERDICT = {payload_verdict(data['baselines'], 'BG_FINAL_ARBITER_V1_1_BASELINES_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_TRAINING_VERDICT = {payload_verdict(data['training'], 'BG_FINAL_ARBITER_V1_1_TRAINING_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT = {payload_verdict(data['heldout'], 'BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT = {payload_verdict(data['domain'], 'BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_TIE_VERDICT = {payload_verdict(data['ties'], 'BG_FINAL_ARBITER_V1_1_TIE_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT = {payload_verdict(data['ablation'], 'BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT = {payload_verdict(data['ood'], 'BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT = {payload_verdict(data['failures'], 'BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT')}",
        f"BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT = {payload_verdict(data['readiness'], 'BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT')}",
        f"FINAL_ARBITER_TOP4_V1_1_STATUS = {arbiter_status}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = {phase_status}",
    ]


def recommendation(phase_status: str, arbiter_status: str) -> str:
    if phase_status == "READY_FOR_PHASE2B_STEERING_COMPARISON":
        return "Proceed to Phase 2b with the v1.1 final arbiter as the locked selection-only baseline."
    if arbiter_status == "FINAL_ARBITER_WEAK_BUT_IMPROVED":
        return "Run v1.2 or collect more final-arbiter data before Phase 2b; v1.1 is useful but not ready."
    if arbiter_status == "SCIENCE_LIMITED":
        return "Build a science-specific arbiter or collect science survivor data."
    if arbiter_status == "REASONING_LIMITED":
        return "Build a reasoning-specific arbiter or collect reasoning survivor data."
    if arbiter_status == "NO_IMPROVEMENT":
        return "Return to expert/bridge signal quality; v1.1 did not improve final selection."
    return "Continue final-arbiter work before Phase 2b steering comparison."


def docs_lines(data: dict[str, dict[str, Any]], arbiter_status: str, phase_status: str) -> list[str]:
    heldout = data.get("heldout", {})
    training = data.get("training", {})
    return [
        "# Final Arbiter Among Top4 Survivors V1.1",
        "",
        "V1.1 tests the v1 ablation lead that rank-heavy arbitration generalizes better than the full-feature listwise model. It keeps branch generation and fixed-composite top4 survival unchanged.",
        "",
        "## Verdicts",
        "",
        *top_lines(data, arbiter_status, phase_status),
        "",
        "## Anti-Leakage Split Strategy",
        "",
        "The v1 heldout rank-only result is treated as a hypothesis. V1.1 uses a fresh task-disjoint holdout selected from tasks that were not v1 heldout; previous v1 heldout replay is diagnostic only.",
        "",
        "## Training and Evaluation",
        "",
        f"- selected model: `{((training.get('selected_model') or {}).get('family'))}`",
        f"- heldout verdict: `{payload_verdict(heldout, 'BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT')}`",
        f"- success checks: `{heldout.get('success_checks')}`",
        "",
        "## Domain, Ties, Ablation, OOD, Failures",
        "",
        f"- domain verdict: `{payload_verdict(data['domain'], 'BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT')}`",
        f"- tie verdict: `{payload_verdict(data['ties'], 'BG_FINAL_ARBITER_V1_1_TIE_VERDICT')}`",
        f"- ablation verdict: `{payload_verdict(data['ablation'], 'BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT')}`",
        f"- calibration/OOD verdict: `{payload_verdict(data['ood'], 'BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT')}`",
        f"- failure verdict: `{payload_verdict(data['failures'], 'BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT')}`",
        "",
        "## Recommendation",
        "",
        recommendation(phase_status, arbiter_status),
        "",
        "No action steering was tested. No production routing changed. No true fork/carry or compute-saving claim is made.",
        "",
    ]


def append_once(path: Path, section: Sequence[str], marker: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + "\n".join(section).strip() + "\n", encoding="utf-8")


def update_docs(data: dict[str, dict[str, Any]], arbiter_status: str, phase_status: str) -> None:
    marker = "## Final arbiter among top4 survivors v1.1 (2026-05-18)"
    section = [
        marker,
        "",
        f"FINAL_ARBITER_TOP4_V1_1_STATUS = {arbiter_status}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = {phase_status}",
        "",
        f"- split guard: `{payload_verdict(data['split_guard'], 'BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT')}`",
        f"- selected model: `{((data.get('training') or {}).get('selected_model') or {}).get('family')}`",
        f"- heldout eval: `{payload_verdict(data['heldout'], 'BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT')}`",
        f"- readiness: `{payload_verdict(data['readiness'], 'BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT')}`",
        f"- recommendation: {recommendation(phase_status, arbiter_status)}",
        "- no action steering was tested.",
        "",
    ]
    write_md(DOC_MD, docs_lines(data, arbiter_status, phase_status))
    for target in DOC_TARGETS:
        append_once(target, section, marker)


def run_synthesis() -> int:
    ensure_root()
    started = time.time()
    required = [
        (SPLIT_GUARD_JSON, run_split_guard),
        (DATASET_PT, run_dataset),
        (FEATURES_PT, run_features),
        (BASELINES_JSON, run_baselines),
        (MODEL_PT, run_training),
        (HELDOUT_JSON, run_heldout_eval),
        (DOMAIN_JSON, run_domain_analysis),
        (TIE_JSON, run_tie_analysis),
        (ABLATION_JSON, run_ablation),
        (CALIBRATION_JSON, run_calibration_ood),
        (FAILURE_JSON, run_failure_analysis),
        (READINESS_JSON, run_selection_readiness),
    ]
    for path, fn in required:
        if not path.exists():
            fn()
    data = stage_payloads()
    arbiter_status, phase_status = final_status(data)
    rec = recommendation(phase_status, arbiter_status)
    payload = {
        "top_lines": top_lines(data, arbiter_status, phase_status),
        "stage_payloads": data,
        "FINAL_ARBITER_TOP4_V1_1_STATUS": arbiter_status,
        "SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1": phase_status,
        "recommended_next": rec,
        "files_created": {
            "output_root": rel(OUT_ROOT),
            "model": rel(MODEL_PT),
            "summary": rel(SUMMARY_JSON),
            "analysis": rel(ANALYSIS_JSON),
            "doc": rel(DOC_MD),
        },
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{name}" for name in SCRIPT_NAMES],
        "blockers": [] if phase_status == "READY_FOR_PHASE2B_STEERING_COMPARISON" else [rec],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    lines = ["# Final Arbiter Top4 Survivors V1.1 Summary", "", *payload["top_lines"], "", "## Recommendation", "", rec, "", "## Files Created", ""]
    lines.extend(f"- `{path}`" for path in payload["files_created"].values())
    lines.extend(["", "## Commands Run", ""])
    lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {b}" for b in payload["blockers"])
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, docs_lines(data, arbiter_status, phase_status))
    update_docs(data, arbiter_status, phase_status)
    print(f"FINAL_ARBITER_TOP4_V1_1_STATUS = {arbiter_status}", flush=True)
    print(f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = {phase_status}", flush=True)
    return 0
