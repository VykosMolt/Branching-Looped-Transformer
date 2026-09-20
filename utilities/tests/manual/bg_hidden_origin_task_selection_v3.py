"""Select reasoning/science MCQ tasks for hidden-origin diversity v3."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

from bg_hidden_origin_diversity_v3_common import (
    TASK_SELECTION_JSON,
    V2_ROOT,
    V3_ROOT,
    ensure_v3_root,
    load_json,
    load_more_candidate_tasks,
    md_table,
    rel,
    write_csv,
    write_json,
    write_md,
)


OUT_JSON = TASK_SELECTION_JSON
OUT_MD = V3_ROOT / "task_selection_v3.md"
OUT_CSV = V3_ROOT / "task_selection_v3_rows.csv"
MAX_SCREENED = 192
DESIRED_SELECTED = 128
MIN_READY = 48
MIN_PARTIAL = 24


CLASS_PRIORITY = {
    "perturbation_sensitive": 95.0,
    "baseline_wrong_parseable": 88.0,
    "baseline_parse_fragile": 84.0,
    "baseline_correct_low_confidence": 76.0,
    "baseline_correct_confident": 38.0,
    "baseline_empty_or_unstable": -40.0,
    "unscreened_candidate": 42.0,
    "unknown": 30.0,
}


def screening_rows_by_task() -> dict[str, dict[str, Any]]:
    payload = load_json(V2_ROOT / "task_screening.json", {}) or {}
    out = {}
    for row in list(payload.get("rows") or []) + list(payload.get("selected_rows") or []):
        out[str(row.get("task_id"))] = row
    return out


def prior_task_signals() -> dict[str, dict[str, Any]]:
    audit = load_json(V3_ROOT / "v3_audit.json", {}) or {}
    out = {}
    for row in list(audit.get("task_rows") or []):
        out[str(row.get("task_id"))] = row
    disagreement = defaultdict(int)
    for row in list(audit.get("selector_disagreement_rows") or []):
        if row.get("old_v1_v2_disagree"):
            disagreement[str(row.get("task_id"))] += 1
    for task_id, count in disagreement.items():
        out.setdefault(task_id, {})["old_v1_v2_disagreement_groups"] = count
    return out


def preferred_recipe_for(row: dict[str, Any], audit: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    cls = str(row.get("screening_class") or "unknown")
    branch_points = ["L24", "L36"]
    alpha_buckets = ["alpha_0_01", "alpha_0_005"]
    delta_families = ["random_orthogonal", "paired_plus_minus", "v1_tap_aligned", "v2_tap_aligned"]
    if cls in {"baseline_parse_fragile", "perturbation_sensitive"}:
        branch_points = ["L24", "L36"]
        alpha_buckets = ["alpha_0_01", "alpha_0_005", "alpha_0_02"]
        delta_families = ["paired_plus_minus", "hidden_origin_empirical", "empirical_plus_noise", "v2_tap_aligned", "random_orthogonal"]
    elif cls == "baseline_wrong_parseable":
        delta_families = ["v1_tap_aligned", "v2_tap_aligned", "hidden_origin_whitened", "old_tap_aligned", "paired_plus_minus"]
    elif cls == "baseline_correct_low_confidence":
        delta_families = ["old_tap_aligned", "v2_tap_aligned", "random_orthogonal", "paired_plus_minus"]
    top_branch = [
        row["condition"]
        for row in list(audit.get("top_diversity_yielding_branch_points") or [])
        if row.get("condition") in {"L24", "L36"}
    ]
    top_delta = [
        str(row["condition"])
        for row in list(audit.get("top_diversity_yielding_delta_families") or [])
        if str(row.get("condition")) not in {"clean", "None", "unknown"}
    ]
    if top_branch:
        branch_points = list(dict.fromkeys(top_branch + branch_points))[:2]
    if top_delta:
        translated = [translate_family_name(name) for name in top_delta]
        delta_families = list(dict.fromkeys(translated + delta_families))[:6]
    return branch_points, delta_families, alpha_buckets


def translate_family_name(name: str) -> str:
    mapping = {
        "random": "random_orthogonal",
        "branch_specific_noise": "empirical_plus_noise",
        "v1_hidden_origin_tap": "v1_tap_aligned",
        "v2_hidden_origin_tap": "v2_tap_aligned",
        "old_tap_aligned": "old_tap_aligned",
        "hidden_origin_empirical": "hidden_origin_empirical",
        "hidden_origin_whitened": "hidden_origin_whitened",
        "adapter_proxy": "adapter_proxy",
        "sequence_adapter_proxy": "sequence_adapter_proxy",
    }
    return mapping.get(name, name)


def score_task(task: dict[str, Any], screen: dict[str, Any], prior: dict[str, Any]) -> tuple[float, str]:
    cls = str(screen.get("screening_class") or prior.get("screening_class") or "unscreened_candidate")
    score = CLASS_PRIORITY.get(cls, CLASS_PRIORITY["unknown"])
    score += 8.0 * float(prior.get("behaviorally_diverse_groups") or 0)
    score += 0.25 * float(prior.get("non_tie_pairs") or 0)
    score += 4.0 * float(prior.get("old_v1_v2_disagreement_groups") or 0)
    if bool(screen.get("perturbation_sensitive")):
        score += 10.0
    margin = screen.get("answer_margin")
    try:
        margin_f = float(margin)
        if margin_f < 1.0:
            score += 8.0
        elif margin_f > 4.0 and cls == "baseline_correct_confident":
            score -= 10.0
    except Exception:
        pass
    if prior.get("groups") and not prior.get("non_tie_pairs") and cls == "baseline_correct_confident":
        score -= 20.0
    if prior.get("parse_rate") is not None and float(prior.get("parse_rate") or 0.0) <= 0.25:
        score -= 15.0
    if score >= 70:
        tier = "high"
    elif score >= 45:
        tier = "medium"
    else:
        tier = "low"
    return score, tier


def build_candidate_rows() -> list[dict[str, Any]]:
    screens = screening_rows_by_task()
    prior = prior_task_signals()
    tasks = load_more_candidate_tasks()
    by_id = {task["task_id"]: dict(task) for task in tasks}
    for task_id, row in screens.items():
        if task_id not in by_id:
            item = {
                "task_id": task_id,
                "domain": row.get("domain"),
                "source_dataset": row.get("source_dataset"),
                "question": row.get("question"),
                "options": row.get("options"),
                "correct_option": row.get("correct_option"),
                "prompt": row.get("prompt"),
            }
            by_id[task_id] = item
    audit = load_json(V3_ROOT / "v3_audit.json", {}) or {}
    rows = []
    for task_id, task in sorted(by_id.items()):
        domain = str(task.get("domain") or "").lower()
        if domain not in {"reasoning", "science"}:
            continue
        screen = screens.get(task_id, {})
        prior_row = prior.get(task_id, {})
        score, tier = score_task(task, screen, prior_row)
        branch_points, delta_families, alpha_buckets = preferred_recipe_for(screen or prior_row, audit)
        row = {
            "task_id": task_id,
            "domain": domain,
            "source_dataset": task.get("source_dataset"),
            "prompt": task.get("prompt"),
            "question": task.get("question"),
            "options": task.get("options"),
            "answer": task.get("correct_option") or task.get("answer") or task.get("answer_key"),
            "correct_option": task.get("correct_option") or task.get("answer") or task.get("answer_key"),
            "screening_class": screen.get("screening_class") or prior_row.get("screening_class") or "unscreened_candidate",
            "prior_diversity_status": {
                "groups": prior_row.get("groups", 0),
                "behaviorally_diverse_groups": prior_row.get("behaviorally_diverse_groups", 0),
                "reward_diverse_groups": prior_row.get("reward_diverse_groups", 0),
                "non_tie_pairs": prior_row.get("non_tie_pairs", 0),
                "tie_rate": prior_row.get("tie_rate"),
            },
            "preferred_branch_points": branch_points,
            "preferred_delta_family_candidates": delta_families,
            "preferred_alpha_bucket_candidates": alpha_buckets,
            "priority_score": round(float(score), 3),
            "priority_tier": tier,
            "clean_correct": screen.get("clean_correct"),
            "clean_parse_success": screen.get("clean_parse_success"),
            "answer_margin": screen.get("answer_margin"),
            "perturbation_sensitive": screen.get("perturbation_sensitive"),
            "old_v1_v2_disagreement_groups": prior_row.get("old_v1_v2_disagreement_groups", 0),
        }
        rows.append(row)
    rows.sort(key=lambda item: (-float(item["priority_score"]), str(item["domain"]), str(item["task_id"])))
    return rows[:MAX_SCREENED]


def balanced_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["priority_tier"] in {"high", "medium"}]
    fallback = [row for row in rows if row["priority_tier"] == "low"]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible + fallback:
        by_domain[str(row["domain"])].append(row)
    selected = []
    used = set()
    while len(selected) < limit and any(by_domain.values()):
        progressed = False
        domain_order = sorted(by_domain, key=lambda d: (-len(by_domain[d]), d))
        for domain in domain_order:
            bucket = by_domain[domain]
            while bucket:
                row = bucket.pop(0)
                if row["task_id"] in used:
                    continue
                selected.append(row)
                used.add(row["task_id"])
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected[:limit]


def main() -> int:
    started = time.time()
    ensure_v3_root()
    candidates = build_candidate_rows()
    selected = balanced_select(candidates, DESIRED_SELECTED)
    high_medium = [row for row in selected if row["priority_tier"] in {"high", "medium"}]
    if len(high_medium) >= MIN_READY:
        verdict = "READY"
    elif len(selected) >= MIN_PARTIAL:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_HIDDEN_ORIGIN_TASK_SELECTION_V3_VERDICT": verdict,
        "verdict": verdict,
        "screened_candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_high_medium_count": len(high_medium),
        "selected_task_ids": [row["task_id"] for row in selected],
        "selected_tasks": selected,
        "candidate_rows": candidates,
        "domain_counts": dict(Counter(row["domain"] for row in candidates)),
        "selected_domain_counts": dict(Counter(row["domain"] for row in selected)),
        "screening_class_counts": dict(Counter(row["screening_class"] for row in candidates)),
        "selected_screening_class_counts": dict(Counter(row["screening_class"] for row in selected)),
        "priority_tier_counts": dict(Counter(row["priority_tier"] for row in selected)),
        "domain_skew": dict(Counter(row["domain"] for row in selected)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, selected)
    display = [
        {
            "task_id": row["task_id"],
            "domain": row["domain"],
            "class": row["screening_class"],
            "tier": row["priority_tier"],
            "score": row["priority_score"],
            "prior_non_ties": row["prior_diversity_status"]["non_tie_pairs"],
            "branch": ",".join(row["preferred_branch_points"]),
            "delta": ",".join(row["preferred_delta_family_candidates"][:3]),
        }
        for row in selected[:120]
    ]
    lines = [
        "# Hidden-Origin Task Selection V3",
        "",
        f"BG_HIDDEN_ORIGIN_TASK_SELECTION_V3_VERDICT = {verdict}",
        "",
        f"- screened_candidate_count: `{len(candidates)}`",
        f"- selected_count: `{len(selected)}`",
        f"- selected_high_medium_count: `{len(high_medium)}`",
        f"- selected_domain_counts: `{payload['selected_domain_counts']}`",
        f"- selected_screening_class_counts: `{payload['selected_screening_class_counts']}`",
        "",
        "Task selection favors parse-fragile, wrong-parseable, low-confidence, perturbation-sensitive, prior-diverse, and evaluator-disagreement tasks. Confident clean-correct tasks are only retained as fallback.",
        "",
        "## Selected Tasks",
        "",
    ]
    lines.extend(md_table(display, ["task_id", "domain", "class", "tier", "score", "prior_non_ties", "branch", "delta"]))
    lines.extend(["", "## Outputs", "", f"- JSON: `{rel(OUT_JSON)}`", f"- CSV: `{rel(OUT_CSV)}`"])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TASK_SELECTION_V3_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

