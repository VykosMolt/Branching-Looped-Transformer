"""Expand BG text-prefix branch selection on cached non-code branch pools.

This is a deployable control-path analysis over ordinary generated text
prefixes. It does not mutate hidden states, fork caches, or use the wrapper.
"""
from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from typing import Any

from bg_preconsolidation_common import OUT_ROOT, STEERING_SUITE_ROOT, finite, rel, write_json, write_md


OUT_RESULTS_JSON = OUT_ROOT / "text_prefix_expansion_results.json"
OUT_ANALYSIS_JSON = OUT_ROOT / "text_prefix_expansion_analysis.json"
OUT_ANALYSIS_MD = OUT_ROOT / "text_prefix_expansion_analysis.md"
SOURCE_JSON = STEERING_SUITE_ROOT / "partial_routing_results.json"
ALLOWED_DOMAINS = {"reasoning", "science", "gsm8k"}
PREFIX_BY_DOMAIN = {"reasoning": 64, "science": 32, "gsm8k": 256}
TARGET_TASKS = 40
MIN_TASKS = 30
HARD_CAP = 50


def load_json(path, default=None):
    import json

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def expected_random_topk(successes: list[bool], k: int) -> float:
    if not successes:
        return 0.0
    n = len(successes)
    k = min(k, n)
    combos = list(itertools.combinations(range(n), k))
    return sum(any(successes[i] for i in combo) for combo in combos) / max(len(combos), 1)


def compact_task_record(row: dict[str, Any]) -> dict[str, Any]:
    branch_ids = [int(x) for x in row.get("branch_ids") or []]
    eval_by_branch = {
        int(c.get("branch_id")): c.get("evaluation") or {}
        for c in row.get("continuations") or []
        if c.get("branch_id") is not None
    }
    success_by_branch = {bid: bool(eval_by_branch.get(bid, {}).get("success")) for bid in branch_ids}
    ordered_successes = [success_by_branch.get(bid, False) for bid in branch_ids]
    ranking = [int(x) for x in row.get("rankings", {}).get("conservative", {}).get("ranking_branch_ids") or []]
    bg_top1 = ranking[0] if ranking else None
    bg_top2 = ranking[:2]
    policy = row.get("policy_success") or {}
    random_top1 = finite(policy.get("random_top1_expected"), expected_random_topk(ordered_successes, 1))
    random_top2 = finite(policy.get("random_top2_expected"), expected_random_topk(ordered_successes, 2))
    return {
        "task_id": row.get("task_id"),
        "domain": row.get("domain"),
        "prefix_length": PREFIX_BY_DOMAIN.get(str(row.get("domain")), None),
        "branch_count": len(branch_ids),
        "branch_ids": branch_ids,
        "branch_success": success_by_branch,
        "random_top1_success_expected": random_top1,
        "random_top2_success_expected": random_top2,
        "bg_top1_branch": bg_top1,
        "bg_top2_branches": bg_top2,
        "bg_top1_success": bool(success_by_branch.get(bg_top1, False)) if bg_top1 is not None else False,
        "bg_top2_success": any(success_by_branch.get(bid, False) for bid in bg_top2),
        "oracle_success": any(ordered_successes),
        "margin_sum": row.get("rankings", {}).get("conservative", {}).get("margin_sum") or [],
        "score_matrix": row.get("rankings", {}).get("conservative", {}).get("score_matrix") or [],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rand1 = avg([finite(row.get("random_top1_success_expected")) for row in rows]) or 0.0
    rand2 = avg([finite(row.get("random_top2_success_expected")) for row in rows]) or 0.0
    bg1 = avg([1.0 if row.get("bg_top1_success") else 0.0 for row in rows]) or 0.0
    bg2 = avg([1.0 if row.get("bg_top2_success") else 0.0 for row in rows]) or 0.0
    oracle = avg([1.0 if row.get("oracle_success") else 0.0 for row in rows]) or 0.0
    pair_correct = 0
    pair_total = 0
    margins: list[float] = []
    labels: list[float] = []
    selected_margins: list[float] = []
    selected_labels: list[float] = []
    for row in rows:
        branch_ids = list(row.get("branch_ids") or [])
        success = {int(k): bool(v) for k, v in (row.get("branch_success") or {}).items()}
        score_matrix = row.get("score_matrix") or []
        for i, left in enumerate(branch_ids):
            for j, right in enumerate(branch_ids):
                if i >= j:
                    continue
                left_success = success.get(int(left), False)
                right_success = success.get(int(right), False)
                if left_success == right_success:
                    continue
                pair_total += 1
                score = finite(score_matrix[i][j]) if i < len(score_matrix) and j < len(score_matrix[i]) else 0.0
                preferred_left = score > 0.0
                if (left_success and preferred_left) or (right_success and not preferred_left):
                    pair_correct += 1
        for idx, bid in enumerate(branch_ids):
            margin = finite((row.get("margin_sum") or [0.0] * len(branch_ids))[idx]) if idx < len(row.get("margin_sum") or []) else 0.0
            margins.append(margin)
            labels.append(1.0 if success.get(int(bid), False) else 0.0)
            if row.get("bg_top1_branch") == bid:
                selected_margins.append(margin)
                selected_labels.append(1.0 if row.get("bg_top1_success") else 0.0)
    return {
        "evaluable_tasks": n,
        "random_top1_success": rand1,
        "bg_top1_success": bg1,
        "bg_top1_lift": bg1 - rand1,
        "random_top2_success": rand2,
        "bg_top2_success": bg2,
        "bg_top2_lift": bg2 - rand2,
        "oracle_success": oracle,
        "oracle_gap": oracle - max(bg1, bg2),
        "pairwise_branch_ranking_accuracy": pair_correct / pair_total if pair_total else None,
        "pairwise_branch_pairs": pair_total,
        "margin_success_pearson": pearson(margins, labels),
        "margin_success_spearman": spearman(margins, labels),
        "selected_margin_success_pearson": pearson(selected_margins, selected_labels),
    }


def verdict_from_metrics(metrics: dict[str, Any]) -> str:
    n = int(metrics.get("evaluable_tasks") or 0)
    if n < MIN_TASKS:
        return "INSUFFICIENT"
    best_lift = max(finite(metrics.get("bg_top1_lift")), finite(metrics.get("bg_top2_lift")))
    worst_lift = min(finite(metrics.get("bg_top1_lift")), finite(metrics.get("bg_top2_lift")))
    if best_lift >= 0.05:
        return "HELPS"
    if best_lift > 0.0:
        return "WEAK_POSITIVE"
    if worst_lift < -0.05:
        return "HURTS"
    return "NEUTRAL"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source = load_json(SOURCE_JSON, {})
    source_rows = list(source.get("task_results") or [])
    selected = [
        compact_task_record(row)
        for row in source_rows
        if str(row.get("domain")) in ALLOWED_DOMAINS and not row.get("is_devil")
    ][:HARD_CAP]
    # Preserve the exact cached 40-task non-code expansion when available.
    selected = selected[:TARGET_TASKS] if len(selected) >= TARGET_TASKS else selected
    metrics = aggregate(selected)
    by_domain = {}
    for domain in sorted({str(row.get("domain")) for row in selected}):
        by_domain[domain] = aggregate([row for row in selected if row.get("domain") == domain])
    by_prefix = {}
    for prefix_length in sorted({int(row.get("prefix_length") or -1) for row in selected}):
        by_prefix[str(prefix_length)] = aggregate([row for row in selected if int(row.get("prefix_length") or -1) == prefix_length])
    verdict = verdict_from_metrics(metrics)
    payload = {
        "BG_TEXT_PREFIX_EXPANSION_VERDICT": verdict,
        "verdict": verdict,
        "branching_type": "TEXT_PREFIX_BRANCHING_ONLY",
        "source_artifact": rel(SOURCE_JSON),
        "source_bg_partial_routing_verdict": source.get("BG_PARTIAL_ROUTING_VERDICT"),
        "generated_new_prefixes": False,
        "task_count_target": TARGET_TASKS,
        "minimum_task_count": MIN_TASKS,
        "hard_cap": HARD_CAP,
        "metrics": metrics,
        "by_domain": by_domain,
        "by_prefix_length": by_prefix,
        "task_results": selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_RESULTS_JSON, payload)
    write_json(OUT_ANALYSIS_JSON, {k: v for k, v in payload.items() if k != "task_results"})
    lines = [
        "# BG Text-Prefix Branch Selection Expansion",
        "",
        f"BG_TEXT_PREFIX_EXPANSION_VERDICT = {verdict}",
        "",
        "- branching_type: `TEXT_PREFIX_BRANCHING_ONLY`",
        f"- source_artifact: `{rel(SOURCE_JSON)}`",
        "- generated_new_prefixes: `False`",
        f"- evaluable_tasks: `{metrics['evaluable_tasks']}`",
        f"- bg_top1_lift: `{metrics['bg_top1_lift']}`",
        f"- bg_top2_lift: `{metrics['bg_top2_lift']}`",
        f"- oracle_gap: `{metrics['oracle_gap']}`",
        f"- pairwise_branch_ranking_accuracy: `{metrics['pairwise_branch_ranking_accuracy']}`",
        f"- margin_success_spearman: `{metrics['margin_success_spearman']}`",
        "",
        "| domain | n | random top1 | BG top1 | lift | random top2 | BG top2 | lift | oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, row in sorted(by_domain.items()):
        lines.append(
            f"| `{domain}` | {int(row['evaluable_tasks'])} | {finite(row['random_top1_success']):.3f} | "
            f"{finite(row['bg_top1_success']):.3f} | {finite(row['bg_top1_lift']):+.3f} | "
            f"{finite(row['random_top2_success']):.3f} | {finite(row['bg_top2_success']):.3f} | "
            f"{finite(row['bg_top2_lift']):+.3f} | {finite(row['oracle_success']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Task Records",
            "",
            "| task | domain | prefix | BG top1 | BG top2 | oracle |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for row in selected:
        lines.append(
            f"| `{row['task_id']}` | `{row['domain']}` | {row['prefix_length']} | "
            f"`{row['bg_top1_success']}` | `{row['bg_top2_success']}` | `{row['oracle_success']}` |"
        )
    write_md(OUT_ANALYSIS_MD, lines)
    print(f"BG_TEXT_PREFIX_EXPANSION_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_RESULTS_JSON)}")
    print(f"Wrote {rel(OUT_ANALYSIS_JSON)}")
    print(f"Wrote {rel(OUT_ANALYSIS_MD)}")
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
