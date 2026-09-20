"""Merged-tap final arbiter integration v1.1.

This cached probe runs the full follow-up suite requested after merged-tap
integration v1:

* grouped task-disjoint CV for final-arbiter policy stability,
* domain-oracle diagnostics,
* simple domain-gated/fallback policies,
* reasoning failure analysis,
* math fallback analysis.

It does not train Ouro, mutate checkpoints/tokenizers/tap registries, run
wrapper/local-agent or Hunter-Seeker code, apply steering, run generation, or
change production routing.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import (
    SURVIVOR_FEATURES_PT,
    finite_mean,
    rate,
    safe_float,
)
from run_bg_merged_tap_final_arbiter_integration_v1 import (
    BASE_POLICIES as V1_BASE_POLICIES,
    base_choices,
    load_data,
    summarize,
)


OUT_ROOT = PROBE_ROOT / "bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18"
GROUPED_JSON = OUT_ROOT / "grouped_cv_eval.json"
GROUPED_MD = OUT_ROOT / "grouped_cv_eval.md"
GROUPED_CSV = OUT_ROOT / "grouped_cv_rows.csv"
DOMAIN_JSON = OUT_ROOT / "domain_oracle_diagnostic.json"
DOMAIN_MD = OUT_ROOT / "domain_oracle_diagnostic.md"
DOMAIN_CSV = OUT_ROOT / "domain_oracle_rows.csv"
REASONING_JSON = OUT_ROOT / "reasoning_failure_probe.json"
REASONING_MD = OUT_ROOT / "reasoning_failure_probe.md"
REASONING_CSV = OUT_ROOT / "reasoning_failure_rows.csv"
MATH_JSON = OUT_ROOT / "math_fallback_probe.json"
MATH_MD = OUT_ROOT / "math_fallback_probe.md"
MATH_CSV = OUT_ROOT / "math_fallback_rows.csv"
READINESS_JSON = OUT_ROOT / "selection_readiness.json"
READINESS_MD = OUT_ROOT / "selection_readiness.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_merged_tap_final_arbiter_integration_v1_1.md"

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_merged_tap_final_arbiter_integration_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_merged_weight_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_final_arbiter_top4_survivors_v1_1.md",
    PROJECT_ROOT / "docs/evaluator/bg_selection_only_phase2_prototype_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
]
NAV_TARGETS = [
    PROJECT_ROOT / "shared/docs/evaluator/README.md",
    PROJECT_ROOT / "shared/docs/README.md",
    PROJECT_ROOT / "PROJECT_TREE_MAP.md",
    PROJECT_ROOT / "PROJECT_COMPONENTS.md",
]

BASE_POLICIES = [
    "merged_tap_top1",
    "fixed_composite_top1",
    "majority_rank_aggregation",
    "universal_top1",
    "old_top1",
    "bridge_top1",
    "code_top1",
    "old_code_reasoning_top1",
    "merged_fixed_rank_sum",
    "merged_universal_rank_sum",
    "merged_fixed_universal_rank_sum",
    "merged_bridge_universal_rank_sum",
    "merged_old_fixed_rank_sum",
]

DOMAIN_RULE_POLICIES = [
    "math_universal_else_merged",
    "math_fixed_else_merged",
    "reasoning_majority_else_merged",
    "reasoning_universal_else_merged",
    "math_universal_reasoning_majority_else_merged",
    "math_universal_reasoning_universal_else_merged",
    "coding_oldcode_math_universal_science_merged_reasoning_majority",
    "coding_merged_math_universal_science_merged_reasoning_majority",
    "coding_merged_math_universal_science_merged_reasoning_universal",
    "science_merged_math_universal_else_majority",
    "domain_prespecified_v1",
    "domain_old_code_guard_v1",
]

POSTHOC_DIAGNOSTIC_POLICIES = [
    # Added after the first v1.1 diagnostic pass showed that reasoning prefers
    # fixed-composite while math prefers universal. These rows are useful for
    # next-step planning but are not readiness-bearing in this run.
    "math_universal_reasoning_fixed_else_merged_posthoc",
    "coding_merged_math_universal_science_merged_reasoning_fixed_posthoc",
]

META_POLICIES = [
    "fold_global_val_best",
    "fold_domain_val_best",
    "outer_domain_oracle_diagnostic",
    "oracle_best_survivor",
]

EVAL_POLICIES = BASE_POLICIES + DOMAIN_RULE_POLICIES + POSTHOC_DIAGNOSTIC_POLICIES + META_POLICIES
SELECTABLE_POLICIES = BASE_POLICIES + DOMAIN_RULE_POLICIES
READINESS_EXCLUDED_POLICIES = {"oracle_best_survivor", "outer_domain_oracle_diagnostic", *POSTHOC_DIAGNOSTIC_POLICIES}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
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


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def append_section(path: Path, title: str, lines: Sequence[str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if title in text:
        return
    with path.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n" + "\n".join(lines) + "\n")


def task_domain_map(sets: Sequence[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for s in sets:
        counts[str(s.get("task_id"))][str(s.get("domain"))] += 1
    for task, counter in counts.items():
        out[task] = counter.most_common(1)[0][0]
    return out


def make_grouped_folds(sets: Sequence[dict[str, Any]], k: int = 5) -> list[set[str]]:
    domain_by_task = task_domain_map(sets)
    set_counts = Counter(str(s.get("task_id")) for s in sets)
    folds: list[set[str]] = [set() for _ in range(k)]
    fold_load = [0 for _ in range(k)]
    by_domain: dict[str, list[str]] = defaultdict(list)
    for task, domain in domain_by_task.items():
        by_domain[domain].append(task)
    for domain in sorted(by_domain):
        tasks = sorted(by_domain[domain], key=lambda task: (-set_counts[task], task))
        for task in tasks:
            idx = min(range(k), key=lambda i: (fold_load[i], len(folds[i]), i))
            folds[idx].add(task)
            fold_load[idx] += set_counts[task]
    return folds


def split_sets(sets: Sequence[dict[str, Any]], tasks: set[str]) -> list[dict[str, Any]]:
    return [s for s in sets if str(s.get("task_id")) in tasks]


def all_tasks(sets: Sequence[dict[str, Any]]) -> set[str]:
    return {str(s.get("task_id")) for s in sets}


def choose_from_policy(
    policy: str,
    cands: list[dict[str, Any]],
    selected_tap: dict[str, Any],
    domain: str,
    domain_map: dict[str, str] | None = None,
    global_policy: str | None = None,
) -> tuple[int, str, dict[str, Any]]:
    choices, aux = base_choices(cands, selected_tap)
    rewards = [safe_float(c.get("final_reward", c.get("reward", 0.0)), 0.0) for c in cands]
    if rewards:
        choices["oracle_best_survivor"] = max(range(len(cands)), key=lambda i: (rewards[i], -i))
    mapped = policy
    if policy == "fold_global_val_best":
        mapped = global_policy or "merged_tap_top1"
    elif policy == "fold_domain_val_best":
        mapped = (domain_map or {}).get(domain, global_policy or "merged_tap_top1")
    elif policy == "outer_domain_oracle_diagnostic":
        mapped = (domain_map or {}).get(domain, "oracle_best_survivor")
    elif policy == "domain_prespecified_v1":
        if domain == "coding":
            mapped = "merged_old_fixed_rank_sum"
        elif domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        else:
            mapped = "merged_tap_top1"
    elif policy == "domain_old_code_guard_v1":
        if domain in {"coding", "math_simple_arithmetic"}:
            mapped = "merged_fixed_universal_rank_sum"
        else:
            mapped = "merged_tap_top1"
    elif policy == "math_universal_else_merged":
        mapped = "universal_top1" if domain == "math_simple_arithmetic" else "merged_tap_top1"
    elif policy == "math_fixed_else_merged":
        mapped = "fixed_composite_top1" if domain == "math_simple_arithmetic" else "merged_tap_top1"
    elif policy == "reasoning_majority_else_merged":
        mapped = "majority_rank_aggregation" if domain == "reasoning" else "merged_tap_top1"
    elif policy == "reasoning_universal_else_merged":
        mapped = "universal_top1" if domain == "reasoning" else "merged_tap_top1"
    elif policy == "math_universal_reasoning_majority_else_merged":
        if domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        elif domain == "reasoning":
            mapped = "majority_rank_aggregation"
        else:
            mapped = "merged_tap_top1"
    elif policy == "math_universal_reasoning_universal_else_merged":
        if domain in {"math_simple_arithmetic", "reasoning"}:
            mapped = "universal_top1"
        else:
            mapped = "merged_tap_top1"
    elif policy == "coding_oldcode_math_universal_science_merged_reasoning_majority":
        if domain == "coding":
            mapped = "old_code_reasoning_top1"
        elif domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        elif domain == "reasoning":
            mapped = "majority_rank_aggregation"
        else:
            mapped = "merged_tap_top1"
    elif policy == "coding_merged_math_universal_science_merged_reasoning_majority":
        if domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        elif domain == "reasoning":
            mapped = "majority_rank_aggregation"
        else:
            mapped = "merged_tap_top1"
    elif policy == "coding_merged_math_universal_science_merged_reasoning_universal":
        if domain in {"math_simple_arithmetic", "reasoning"}:
            mapped = "universal_top1"
        else:
            mapped = "merged_tap_top1"
    elif policy == "science_merged_math_universal_else_majority":
        if domain in {"science", "coding"}:
            mapped = "merged_tap_top1"
        elif domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        else:
            mapped = "majority_rank_aggregation"
    elif policy == "math_universal_reasoning_fixed_else_merged_posthoc":
        if domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        elif domain == "reasoning":
            mapped = "fixed_composite_top1"
        else:
            mapped = "merged_tap_top1"
    elif policy == "coding_merged_math_universal_science_merged_reasoning_fixed_posthoc":
        if domain == "math_simple_arithmetic":
            mapped = "universal_top1"
        elif domain == "reasoning":
            mapped = "fixed_composite_top1"
        else:
            mapped = "merged_tap_top1"
    idx = choices.get(mapped, -1)
    return int(idx) if isinstance(idx, int) else -1, mapped, aux


def evaluate_policy(
    sets: Sequence[dict[str, Any]],
    selected_tap: dict[str, Any],
    policy: str,
    *,
    fold_id: int | str = "",
    mode: str = "",
    domain_map: dict[str, str] | None = None,
    global_policy: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in sets:
        cands = list(s.get("candidates") or [])
        if not cands:
            continue
        domain = str(s.get("domain"))
        rewards = [safe_float(c.get("final_reward", c.get("reward", 0.0)), 0.0) for c in cands]
        best_idx = max(range(len(cands)), key=lambda i: (rewards[i], -i))
        choice, mapped, aux = choose_from_policy(policy, cands, selected_tap, domain, domain_map=domain_map, global_policy=global_policy)
        if choice < 0 or choice >= len(cands):
            continue
        selected = cands[choice]
        reward = rewards[choice]
        rows.append(
            {
                "mode": mode,
                "fold_id": fold_id,
                "policy": policy,
                "mapped_policy": mapped,
                "survivor_set_id": s.get("survivor_set_id"),
                "task_id": s.get("task_id"),
                "domain": domain,
                "source_split": s.get("split"),
                "candidate_count": len(cands),
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_reward": reward,
                "best_reward": rewards[best_idx],
                "oracle_selected": 1.0 if reward == rewards[best_idx] else 0.0,
                "regret": rewards[best_idx] - reward,
                "selected_branch_origin": selected.get("branch_origin") or selected.get("origin"),
                "merged_rank": (aux.get("merged_ranks") or {}).get(choice),
                "fixed_rank": (aux.get("fixed_ranks") or {}).get(choice),
                "universal_rank": (aux.get("universal_ranks") or {}).get(choice),
                "bridge_rank": (aux.get("bridge_ranks") or {}).get(choice),
                "old_rank": (aux.get("old_ranks") or {}).get(choice),
            }
        )
    return rows


def summarize_with_domains(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    if not rows:
        return summary
    domain_group_micro: dict[str, list[float]] = defaultdict(list)
    mapped = Counter(str(row.get("mapped_policy")) for row in rows if row.get("mapped_policy"))
    for row in rows:
        domain_group_micro[str(row.get("domain"))].append(safe_float(row.get("selected_reward"), 0.0))
    summary["domain_group_micro"] = {k: finite_mean(v) for k, v in sorted(domain_group_micro.items())}
    summary["mapped_policy_counts"] = dict(sorted(mapped.items()))
    return summary


def select_global_policy(inner_sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], policies: Sequence[str]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for policy in policies:
        ev = evaluate_policy(inner_sets, selected_tap, policy, mode="inner_val_selection")
        summary = summarize_with_domains(ev)
        summaries[policy] = summary
        rows.append({"policy": policy, **summary})
    selected_policy = max(policies, key=lambda p: (safe_float(summaries[p].get("task_macro_reward"), -1e9), safe_float(summaries[p].get("oracle_selected_rate"), -1e9), -policies.index(p)))
    return selected_policy, summaries[selected_policy], rows


def select_domain_map(inner_sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], policies: Sequence[str], fallback: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    domains = sorted({str(s.get("domain")) for s in inner_sets})
    out: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for domain in domains:
        dsets = [s for s in inner_sets if str(s.get("domain")) == domain]
        if len({str(s.get("task_id")) for s in dsets}) < 2:
            out[domain] = fallback
            rows.append({"domain": domain, "selected_policy": fallback, "reason": "insufficient_inner_domain_tasks", "sets": len(dsets)})
            continue
        scored = []
        for policy in policies:
            summary = summarize_with_domains(evaluate_policy(dsets, selected_tap, policy, mode="inner_domain_selection"))
            scored.append((policy, safe_float(summary.get("task_macro_reward"), -1e9), safe_float(summary.get("oracle_selected_rate"), -1e9), summary))
        selected = max(scored, key=lambda x: (x[1], x[2], -policies.index(x[0])))
        out[domain] = selected[0]
        rows.append({"domain": domain, "selected_policy": selected[0], "task_macro_reward": selected[1], "oracle_selected_rate": selected[2], "sets": len(dsets), "tasks": len({str(s.get("task_id")) for s in dsets})})
    return out, rows


def select_outer_domain_oracle(outer_sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], policies: Sequence[str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    domains = sorted({str(s.get("domain")) for s in outer_sets})
    out: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for domain in domains:
        dsets = [s for s in outer_sets if str(s.get("domain")) == domain]
        scored = []
        for policy in policies:
            summary = summarize_with_domains(evaluate_policy(dsets, selected_tap, policy, mode="outer_domain_oracle_selection"))
            scored.append((policy, safe_float(summary.get("task_macro_reward"), -1e9), safe_float(summary.get("oracle_selected_rate"), -1e9), summary))
        selected = max(scored, key=lambda x: (x[1], x[2], -policies.index(x[0])))
        out[domain] = selected[0]
        rows.append({"domain": domain, "selected_policy": selected[0], "task_macro_reward": selected[1], "oracle_selected_rate": selected[2], "sets": len(dsets), "tasks": len({str(s.get("task_id")) for s in dsets})})
    return out, rows


def grouped_cv(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any]) -> dict[str, Any]:
    folds = make_grouped_folds(sets, 5)
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_task_set = all_tasks(sets)
    for fold_id, outer_tasks in enumerate(folds):
        inner_val_tasks = folds[(fold_id + 1) % len(folds)]
        train_tasks = all_task_set - outer_tasks - inner_val_tasks
        outer_sets = split_sets(sets, outer_tasks)
        inner_val_sets = split_sets(sets, inner_val_tasks)
        global_policy, global_summary, global_rows = select_global_policy(inner_val_sets, selected_tap, SELECTABLE_POLICIES)
        domain_map, domain_rows = select_domain_map(inner_val_sets, selected_tap, SELECTABLE_POLICIES, global_policy)
        outer_oracle_map, outer_oracle_rows = select_outer_domain_oracle(outer_sets, selected_tap, SELECTABLE_POLICIES)
        for row in global_rows:
            selection_rows.append({"fold_id": fold_id, "selection_type": "global", **row})
        for row in domain_rows:
            selection_rows.append({"fold_id": fold_id, "selection_type": "domain", **row})
        for row in outer_oracle_rows:
            selection_rows.append({"fold_id": fold_id, "selection_type": "outer_domain_oracle_diagnostic", **row})
        fold_rows.append(
            {
                "fold_id": fold_id,
                "outer_tasks": len(outer_tasks),
                "inner_val_tasks": len(inner_val_tasks),
                "train_tasks": len(train_tasks),
                "outer_sets": len(outer_sets),
                "inner_val_sets": len(inner_val_sets),
                "global_selected_policy": global_policy,
                "global_selected_inner_val_task_macro": global_summary.get("task_macro_reward"),
                "domain_map": domain_map,
                "outer_domain_oracle_map": outer_oracle_map,
                "outer_domain_distribution": dict(Counter(str(s.get("domain")) for s in outer_sets)),
            }
        )
        for policy in EVAL_POLICIES:
            dmap = domain_map if policy == "fold_domain_val_best" else None
            if policy == "outer_domain_oracle_diagnostic":
                dmap = outer_oracle_map
            rows.extend(evaluate_policy(outer_sets, selected_tap, policy, fold_id=fold_id, mode="grouped_outer_cv", domain_map=dmap, global_policy=global_policy))
    by_policy = []
    for policy in EVAL_POLICIES:
        pr = [r for r in rows if r.get("policy") == policy]
        by_policy.append({"policy": policy, **summarize_with_domains(pr)})
    policy_summary = {row["policy"]: row for row in by_policy}
    best_probe_policy = max(
        [row for row in by_policy if row["policy"] != "oracle_best_survivor" and row.get("n", 0)],
        key=lambda row: (safe_float(row.get("task_macro_reward"), -1e9), safe_float(row.get("oracle_selected_rate"), -1e9)),
    )
    best_readiness_policy = max(
        [row for row in by_policy if row["policy"] not in READINESS_EXCLUDED_POLICIES and row.get("n", 0)],
        key=lambda row: (safe_float(row.get("task_macro_reward"), -1e9), safe_float(row.get("oracle_selected_rate"), -1e9)),
    )
    merged = policy_summary.get("merged_tap_top1", {})
    fixed = policy_summary.get("fixed_composite_top1", {})
    majority = policy_summary.get("majority_rank_aggregation", {})
    selected = policy_summary.get("fold_global_val_best", {})
    domain_selected = policy_summary.get("fold_domain_val_best", {})
    oracle = policy_summary.get("oracle_best_survivor", {})
    selected_reward = safe_float(selected.get("task_macro_reward"), float("nan"))
    domain_selected_reward = safe_float(domain_selected.get("task_macro_reward"), float("nan"))
    merged_reward = safe_float(merged.get("task_macro_reward"), float("nan"))
    fixed_reward = safe_float(fixed.get("task_macro_reward"), float("nan"))
    majority_reward = safe_float(majority.get("task_macro_reward"), float("nan"))
    oracle_reward = safe_float(oracle.get("task_macro_reward"), float("nan"))
    best_selected_reward = max(selected_reward, domain_selected_reward, merged_reward)
    best_readiness_reward = safe_float(best_readiness_policy.get("task_macro_reward"), float("nan"))
    best_readiness_domains = best_readiness_policy.get("domain_task_macro") or {}
    readiness_domains_ok = all(
        safe_float(best_readiness_domains.get(domain), 0.0) >= 0.70
        for domain in ["coding", "math_simple_arithmetic", "reasoning", "science"]
    )
    gap_base = max(fixed_reward, majority_reward)
    gap_closure = (max(best_selected_reward, best_readiness_reward) - gap_base) / max(oracle_reward - gap_base, 1e-9)
    if best_readiness_reward >= 0.75 and best_readiness_reward > max(fixed_reward, majority_reward) and gap_closure >= 0.20 and readiness_domains_ok:
        verdict = "FINAL_ARBITER_CV_READY"
    elif best_readiness_reward >= 0.75 and best_readiness_reward > max(fixed_reward, majority_reward):
        verdict = "DOMAIN_FALLBACK_USEFUL_REASONING_WEAK"
    elif merged_reward > max(fixed_reward, majority_reward) and merged_reward >= 0.70:
        verdict = "MERGED_TOP1_STABLE_BUT_BELOW_TARGET"
    elif best_selected_reward > max(fixed_reward, majority_reward):
        verdict = "DOMAIN_RULE_WEAK_BUT_USEFUL"
    else:
        verdict = "NO_STABLE_IMPROVEMENT"
    return {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_GROUPED_CV_VERDICT": verdict,
        "verdict": verdict,
        "folds": fold_rows,
        "policy_summaries": by_policy,
        "policy_summary_map": policy_summary,
        "best_probe_policy": best_probe_policy,
        "best_readiness_policy": best_readiness_policy,
        "selection_rows": selection_rows,
        "rows": rows,
        "gap_closure_vs_best_fixed_majority": gap_closure,
        "anti_leakage": {
            "outer_group": "task_id",
            "inner_selection_split": "next grouped fold",
            "fresh_holdout_not_used_for_selection": True,
            "v1_heldout_replay_not_used_for_selection": True,
            "no_new_generation": True,
        },
    }


def replay_splits(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], policies: Sequence[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in sorted({str(s.get("split")) for s in sets}):
        split_sets = [s for s in sets if str(s.get("split")) == split]
        for policy in policies:
            pr = evaluate_policy(split_sets, selected_tap, policy, mode="source_split_replay")
            rows.extend(pr)
            summaries[f"{split}::{policy}"] = summarize_with_domains(pr)
    return {"rows": rows, "summaries": summaries}


def domain_oracle_diagnostic(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], grouped_payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    domain_table: list[dict[str, Any]] = []
    for domain in sorted({str(s.get("domain")) for s in sets}):
        dsets = [s for s in sets if str(s.get("domain")) == domain]
        domain_policy_rows = []
        for policy in SELECTABLE_POLICIES + ["oracle_best_survivor"]:
            ev = evaluate_policy(dsets, selected_tap, policy, mode="full_domain_diagnostic")
            summary = summarize_with_domains(ev)
            row = {"domain": domain, "policy": policy, **summary}
            domain_policy_rows.append(row)
            rows.append(row)
        best_non_oracle = max([r for r in domain_policy_rows if r["policy"] != "oracle_best_survivor"], key=lambda r: (safe_float(r.get("task_macro_reward"), -1e9), safe_float(r.get("oracle_selected_rate"), -1e9)))
        oracle = next(r for r in domain_policy_rows if r["policy"] == "oracle_best_survivor")
        merged = next(r for r in domain_policy_rows if r["policy"] == "merged_tap_top1")
        universal = next(r for r in domain_policy_rows if r["policy"] == "universal_top1")
        fixed = next(r for r in domain_policy_rows if r["policy"] == "fixed_composite_top1")
        majority = next(r for r in domain_policy_rows if r["policy"] == "majority_rank_aggregation")
        domain_table.append(
            {
                "domain": domain,
                "best_policy": best_non_oracle["policy"],
                "best_task_macro": best_non_oracle.get("task_macro_reward"),
                "merged_task_macro": merged.get("task_macro_reward"),
                "universal_task_macro": universal.get("task_macro_reward"),
                "fixed_task_macro": fixed.get("task_macro_reward"),
                "majority_task_macro": majority.get("task_macro_reward"),
                "oracle_task_macro": oracle.get("task_macro_reward"),
                "sets": len(dsets),
                "tasks": len({str(s.get("task_id")) for s in dsets}),
            }
        )
    best_gaps = [
        safe_float(row.get("best_task_macro"), 0.0) - safe_float(row.get("merged_task_macro"), 0.0)
        for row in domain_table
    ]
    if any(gap > 0.10 for gap in best_gaps):
        verdict = "DOMAIN_SPECIALIZATION_HAS_HEADROOM"
    elif any(str(row.get("best_policy")) != "merged_tap_top1" for row in domain_table):
        verdict = "DOMAIN_SPECIALIZATION_WEAK_SIGNAL"
    else:
        verdict = "MERGED_TOP1_DOMINATES_DOMAIN_DIAGNOSTIC"
    return {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_DOMAIN_ORACLE_VERDICT": verdict,
        "verdict": verdict,
        "domain_table": domain_table,
        "rows": rows,
        "diagnostic_only": True,
        "reason": "This uses observed domain outcomes to estimate domain-specialization headroom and is not readiness-bearing.",
        "grouped_cv_best_probe_policy": (grouped_payload.get("best_probe_policy") or {}).get("policy"),
        "grouped_cv_best_readiness_policy": (grouped_payload.get("best_readiness_policy") or {}).get("policy"),
    }


def outcome_for_policy(s: dict[str, Any], selected_tap: dict[str, Any], policy: str) -> dict[str, Any] | None:
    rows = evaluate_policy([s], selected_tap, policy)
    return rows[0] if rows else None


def reasoning_failure_probe(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any]) -> dict[str, Any]:
    reasoning_sets = [s for s in sets if str(s.get("domain")) == "reasoning"]
    rows: list[dict[str, Any]] = []
    categories = Counter()
    policies = ["merged_tap_top1", "universal_top1", "majority_rank_aggregation", "fixed_composite_top1", "bridge_top1", "old_top1"]
    for s in reasoning_sets:
        outs = {policy: outcome_for_policy(s, selected_tap, policy) for policy in policies}
        merged = outs["merged_tap_top1"]
        if not merged:
            continue
        best_reward = safe_float(merged.get("best_reward"), 0.0)
        merged_reward = safe_float(merged.get("selected_reward"), 0.0)
        right = {policy: bool(out and safe_float(out.get("selected_reward"), 0.0) == best_reward) for policy, out in outs.items()}
        if right["merged_tap_top1"]:
            category = "merged_correct"
        elif right["majority_rank_aggregation"]:
            category = "merged_wrong_majority_right"
        elif right["universal_top1"]:
            category = "merged_wrong_universal_right"
        elif right["fixed_composite_top1"]:
            category = "merged_wrong_fixed_right"
        elif right["bridge_top1"]:
            category = "merged_wrong_bridge_right"
        elif right["old_top1"]:
            category = "merged_wrong_old_right"
        else:
            category = "all_tracked_wrong"
        categories[category] += 1
        rows.append(
            {
                "category": category,
                "survivor_set_id": s.get("survivor_set_id"),
                "task_id": s.get("task_id"),
                "source_split": s.get("split"),
                "best_reward": best_reward,
                "merged_reward": merged_reward,
                "merged_regret": best_reward - merged_reward,
                "universal_reward": safe_float((outs["universal_top1"] or {}).get("selected_reward"), float("nan")),
                "majority_reward": safe_float((outs["majority_rank_aggregation"] or {}).get("selected_reward"), float("nan")),
                "fixed_reward": safe_float((outs["fixed_composite_top1"] or {}).get("selected_reward"), float("nan")),
                "bridge_reward": safe_float((outs["bridge_top1"] or {}).get("selected_reward"), float("nan")),
                "old_reward": safe_float((outs["old_top1"] or {}).get("selected_reward"), float("nan")),
                "merged_rank": merged.get("merged_rank"),
                "fixed_rank": merged.get("fixed_rank"),
                "universal_rank": merged.get("universal_rank"),
                "bridge_rank": merged.get("bridge_rank"),
                "old_rank": merged.get("old_rank"),
            }
        )
    by_category = []
    for category in sorted(categories):
        vals = [row for row in rows if row.get("category") == category]
        by_category.append(
            {
                "category": category,
                "count": len(vals),
                "mean_merged_regret": finite_mean(row.get("merged_regret") for row in vals),
                "source_splits": dict(Counter(str(row.get("source_split")) for row in vals)),
            }
        )
    total = len(rows)
    rescue = categories["merged_wrong_majority_right"] + categories["merged_wrong_universal_right"] + categories["merged_wrong_fixed_right"]
    if total and rescue / total >= 0.20:
        verdict = "REASONING_FALLBACK_HEADROOM"
    elif categories["all_tracked_wrong"] > categories["merged_wrong_majority_right"] + categories["merged_wrong_universal_right"]:
        verdict = "REASONING_SIGNAL_BLOCKER"
    elif categories["merged_correct"] / max(total, 1) >= 0.70:
        verdict = "REASONING_MERGED_MOSTLY_OK"
    else:
        verdict = "REASONING_BLOCKER_REMAINS"
    return {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_REASONING_PROBE_VERDICT": verdict,
        "verdict": verdict,
        "reasoning_set_count": total,
        "category_counts": dict(sorted(categories.items())),
        "category_table": by_category,
        "rows": rows,
    }


def math_fallback_probe(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any]) -> dict[str, Any]:
    policies = [
        "merged_tap_top1",
        "universal_top1",
        "fixed_composite_top1",
        "majority_rank_aggregation",
        "math_universal_else_merged",
        "math_fixed_else_merged",
        "math_universal_reasoning_majority_else_merged",
        "math_universal_reasoning_universal_else_merged",
        "oracle_best_survivor",
    ]
    all_rows: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    math_sets = [s for s in sets if str(s.get("domain")) == "math_simple_arithmetic"]
    for policy in policies:
        rows = evaluate_policy(sets, selected_tap, policy, mode="math_fallback_full")
        math_rows = evaluate_policy(math_sets, selected_tap, policy, mode="math_fallback_math_only")
        all_rows.extend(rows)
        all_rows.extend(math_rows)
        full = summarize_with_domains(rows)
        math_summary = summarize_with_domains(math_rows)
        table.append(
            {
                "policy": policy,
                "overall_task_macro": full.get("task_macro_reward"),
                "overall_group_micro": full.get("group_micro_reward"),
                "math_task_macro": math_summary.get("task_macro_reward"),
                "math_group_micro": math_summary.get("group_micro_reward"),
                "math_oracle_selected_rate": math_summary.get("oracle_selected_rate"),
                "math_regret": math_summary.get("regret"),
            }
        )
    merged_math = next(row for row in table if row["policy"] == "merged_tap_top1")
    universal_math = next(row for row in table if row["policy"] == "universal_top1")
    fallback = next(row for row in table if row["policy"] == "math_universal_else_merged")
    if safe_float(universal_math.get("math_task_macro"), 0.0) > safe_float(merged_math.get("math_task_macro"), 0.0) + 0.05:
        verdict = "MATH_UNIVERSAL_FALLBACK_USEFUL"
    elif safe_float(fallback.get("overall_task_macro"), 0.0) > safe_float(merged_math.get("overall_task_macro"), 0.0) + 0.02:
        verdict = "MATH_FALLBACK_IMPROVES_OVERALL"
    else:
        verdict = "MATH_FALLBACK_NOT_ENOUGH"
    return {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_MATH_FALLBACK_VERDICT": verdict,
        "verdict": verdict,
        "policy_table": table,
        "rows": all_rows,
    }


def readiness(grouped: dict[str, Any], domain: dict[str, Any], reasoning: dict[str, Any], math_probe: dict[str, Any]) -> dict[str, Any]:
    policy_map = grouped.get("policy_summary_map") or {}
    merged = policy_map.get("merged_tap_top1", {})
    fixed = policy_map.get("fixed_composite_top1", {})
    majority = policy_map.get("majority_rank_aggregation", {})
    domain_cv = policy_map.get("fold_domain_val_best", {})
    global_cv = policy_map.get("fold_global_val_best", {})
    oracle = policy_map.get("oracle_best_survivor", {})
    best_probe = grouped.get("best_probe_policy") or {}
    best_readiness = grouped.get("best_readiness_policy") or {}
    best_probe_reward = safe_float(best_probe.get("task_macro_reward"), float("nan"))
    best_reward = safe_float(best_readiness.get("task_macro_reward"), float("nan"))
    merged_reward = safe_float(merged.get("task_macro_reward"), float("nan"))
    fixed_reward = safe_float(fixed.get("task_macro_reward"), float("nan"))
    majority_reward = safe_float(majority.get("task_macro_reward"), float("nan"))
    oracle_reward = safe_float(oracle.get("task_macro_reward"), float("nan"))
    domain_cv_reward = safe_float(domain_cv.get("task_macro_reward"), float("nan"))
    global_cv_reward = safe_float(global_cv.get("task_macro_reward"), float("nan"))
    base = max(fixed_reward, majority_reward)
    gap_closure = (max(best_reward, merged_reward, domain_cv_reward, global_cv_reward) - base) / max(oracle_reward - base, 1e-9)
    best_domains = best_readiness.get("domain_task_macro") or {}
    merged_domains = merged.get("domain_task_macro") or {}
    coding_ok = safe_float(best_domains.get("coding"), 0.0) >= 0.70
    math_ok = safe_float(best_domains.get("math_simple_arithmetic"), 0.0) >= 0.70
    science_ok = safe_float(best_domains.get("science"), 0.0) >= 0.70
    reasoning_ok = safe_float(best_domains.get("reasoning"), 0.0) >= 0.70
    if best_reward >= 0.75 and best_reward > base and coding_ok and math_ok and science_ok and reasoning_ok:
        verdict = "READY_FOR_PHASE2B_STEERING_COMPARISON"
        status = "FINAL_ARBITER_READY"
    elif best_reward >= 0.75 and best_reward > base and not reasoning_ok:
        verdict = "NEEDS_REASONING_ARBITER"
        status = "DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED"
    elif merged_reward > base and merged_reward >= 0.70:
        verdict = "USE_MERGED_TAP_TOP1_BUT_NEEDS_DOMAIN_FALLBACK"
        status = "MERGED_TOP1_STABLE_BUT_NOT_READY"
    elif safe_float(domain_cv_reward, -1e9) > base:
        verdict = "NEEDS_DOMAIN_GATE"
        status = "DOMAIN_GATE_USEFUL_BUT_UNSTABLE"
    elif math_probe.get("verdict") == "MATH_UNIVERSAL_FALLBACK_USEFUL":
        verdict = "NEEDS_MATH_FALLBACK"
        status = "MATH_LIMITED"
    elif reasoning.get("verdict") in {"REASONING_FALLBACK_HEADROOM", "REASONING_BLOCKER_REMAINS", "REASONING_SIGNAL_BLOCKER"}:
        verdict = "NEEDS_REASONING_ARBITER"
        status = "REASONING_LIMITED"
    else:
        verdict = "NOT_READY"
        status = "NO_STABLE_IMPROVEMENT"
    return {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_SELECTION_READINESS_VERDICT": verdict,
        "MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS": status,
        "SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1": verdict,
        "best_probe_policy": best_probe.get("policy"),
        "best_probe_task_macro": best_probe_reward,
        "best_readiness_policy": best_readiness.get("policy"),
        "best_readiness_task_macro": best_reward,
        "merged_tap_top1_task_macro": merged_reward,
        "fixed_composite_top1_task_macro": fixed_reward,
        "majority_rank_aggregation_task_macro": majority_reward,
        "domain_cv_task_macro": domain_cv_reward,
        "global_cv_task_macro": global_cv_reward,
        "oracle_best_survivor_task_macro": oracle_reward,
        "gap_closure_vs_best_fixed_majority": gap_closure,
        "coding_ok_for_best_readiness_policy": coding_ok,
        "math_ok_for_best_readiness_policy": math_ok,
        "science_ok_for_best_readiness_policy": science_ok,
        "reasoning_ok_for_best_readiness_policy": reasoning_ok,
        "merged_domain_task_macro": merged_domains,
        "best_readiness_domain_task_macro": best_domains,
        "no_action_steering_tested": True,
        "no_new_generation": True,
        "caveats": [
            "Grouped CV reuses cached top4 survivor sets; it is stronger than the inspected fresh replay for selection stability but is not new branch generation.",
            "The previous fresh holdout has already been inspected and is diagnostic only for future rule selection.",
            "Merged tap top1 is useful, but a validation-selected domain gate must be stable before Phase 2b readiness.",
        ],
    }


def write_grouped(grouped: dict[str, Any]) -> None:
    write_json(GROUPED_JSON, {k: v for k, v in grouped.items() if k != "rows"})
    write_csv(GROUPED_CSV, grouped.get("rows") or [])
    rows = grouped.get("policy_summaries") or []
    lines = [
        "# Merged Tap Final Arbiter Integration v1.1 Grouped CV",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_GROUPED_CV_VERDICT = {grouped.get('verdict')}",
        "",
        "This is task-disjoint grouped CV over cached top4 survivor sets. Inner folds select global/domain policies; outer folds estimate generalization. No action steering or new generation was run.",
        "",
        "## Policy Summary",
        "",
    ]
    lines.extend(md_table(rows, ["policy", "n", "tasks", "task_macro_reward", "group_micro_reward", "oracle_selected_rate", "regret", "domain_task_macro", "mapped_policy_counts"]))
    lines.extend(["", "## Fold Selection", ""])
    lines.extend(md_table(grouped.get("folds") or [], ["fold_id", "outer_tasks", "outer_sets", "inner_val_tasks", "inner_val_sets", "global_selected_policy", "global_selected_inner_val_task_macro", "domain_map", "outer_domain_distribution"]))
    write_md(GROUPED_MD, lines)


def write_domain(domain: dict[str, Any]) -> None:
    write_json(DOMAIN_JSON, {k: v for k, v in domain.items() if k != "rows"})
    write_csv(DOMAIN_CSV, domain.get("rows") or [])
    lines = [
        "# Domain Oracle Diagnostic",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_DOMAIN_ORACLE_VERDICT = {domain.get('verdict')}",
        "",
        "This diagnostic uses observed outcomes to estimate domain-specialization headroom. It is not readiness-bearing.",
        "",
    ]
    lines.extend(md_table(domain.get("domain_table") or [], ["domain", "sets", "tasks", "best_policy", "best_task_macro", "merged_task_macro", "universal_task_macro", "fixed_task_macro", "majority_task_macro", "oracle_task_macro"]))
    write_md(DOMAIN_MD, lines)


def write_reasoning(reasoning: dict[str, Any]) -> None:
    write_json(REASONING_JSON, {k: v for k, v in reasoning.items() if k != "rows"})
    write_csv(REASONING_CSV, reasoning.get("rows") or [])
    lines = [
        "# Reasoning Failure Probe",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_REASONING_PROBE_VERDICT = {reasoning.get('verdict')}",
        "",
    ]
    lines.extend(md_table(reasoning.get("category_table") or [], ["category", "count", "mean_merged_regret", "source_splits"]))
    write_md(REASONING_MD, lines)


def write_math(math_probe: dict[str, Any]) -> None:
    write_json(MATH_JSON, {k: v for k, v in math_probe.items() if k != "rows"})
    write_csv(MATH_CSV, math_probe.get("rows") or [])
    lines = [
        "# Math Fallback Probe",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_MATH_FALLBACK_VERDICT = {math_probe.get('verdict')}",
        "",
    ]
    lines.extend(md_table(math_probe.get("policy_table") or [], ["policy", "overall_task_macro", "overall_group_micro", "math_task_macro", "math_group_micro", "math_oracle_selected_rate", "math_regret"]))
    write_md(MATH_MD, lines)


def write_readiness(readiness_payload: dict[str, Any]) -> None:
    write_json(READINESS_JSON, readiness_payload)
    lines = [
        "# Merged Tap Integration v1.1 Selection Readiness",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_SELECTION_READINESS_VERDICT = {readiness_payload.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_SELECTION_READINESS_VERDICT')}",
        f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = {readiness_payload.get('MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS')}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = {readiness_payload.get('SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1')}",
        "",
        f"- best readiness-eligible policy: `{readiness_payload.get('best_readiness_policy')}`",
        f"- best readiness-eligible grouped-CV task macro: `{rate(readiness_payload.get('best_readiness_task_macro'))}`",
        f"- best diagnostic probe policy: `{readiness_payload.get('best_probe_policy')}`",
        f"- best diagnostic probe grouped-CV task macro: `{rate(readiness_payload.get('best_probe_task_macro'))}`",
        f"- merged top1 grouped-CV task macro: `{rate(readiness_payload.get('merged_tap_top1_task_macro'))}`",
        f"- fixed top1 grouped-CV task macro: `{rate(readiness_payload.get('fixed_composite_top1_task_macro'))}`",
        f"- majority grouped-CV task macro: `{rate(readiness_payload.get('majority_rank_aggregation_task_macro'))}`",
        f"- oracle best-survivor grouped-CV task macro: `{rate(readiness_payload.get('oracle_best_survivor_task_macro'))}`",
        f"- no action steering tested: `{readiness_payload.get('no_action_steering_tested')}`",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {item}" for item in readiness_payload.get("caveats") or [])
    write_md(READINESS_MD, lines)


def write_summary(summary: dict[str, Any]) -> None:
    write_json(SUMMARY_JSON, summary)
    lines = [
        "# Merged Tap Final Arbiter Integration v1.1 Summary",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_GROUPED_CV_VERDICT = {summary.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_GROUPED_CV_VERDICT')}",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_DOMAIN_ORACLE_VERDICT = {summary.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_DOMAIN_ORACLE_VERDICT')}",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_REASONING_PROBE_VERDICT = {summary.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_REASONING_PROBE_VERDICT')}",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_MATH_FALLBACK_VERDICT = {summary.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_MATH_FALLBACK_VERDICT')}",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_SELECTION_READINESS_VERDICT = {summary.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_SELECTION_READINESS_VERDICT')}",
        f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = {summary.get('MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS')}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = {summary.get('SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1')}",
        "",
        f"- best readiness-eligible policy: `{summary.get('best_readiness_policy')}`",
        f"- best readiness-eligible grouped-CV task macro: `{rate(summary.get('best_readiness_task_macro'))}`",
        f"- best diagnostic probe policy: `{summary.get('best_probe_policy')}`",
        f"- best diagnostic probe grouped-CV task macro: `{rate(summary.get('best_probe_task_macro'))}`",
        f"- merged top1 grouped-CV task macro: `{rate(summary.get('merged_tap_top1_task_macro'))}`",
        f"- output root: `{rel(OUT_ROOT)}`",
    ]
    write_md(SUMMARY_MD, lines)


def write_doc(summary: dict[str, Any], grouped: dict[str, Any], domain: dict[str, Any], reasoning: dict[str, Any], math_probe: dict[str, Any]) -> None:
    lines = [
        "# Merged Tap Final Arbiter Integration v1.1",
        "",
        f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = {summary.get('MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS')}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = {summary.get('SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1')}",
        "",
        "This run tested the complete follow-up probe set after merged-tap integration v1: grouped task-disjoint CV, domain oracle diagnostics, domain-gated/fallback rules, reasoning failures, and math fallback behavior. It used cached top4 survivor features only. No action steering, routing change, wrapper/local-agent code, Hunter-Seeker execution, or new generation was run.",
        "",
        "## Main Result",
        "",
        f"- grouped-CV verdict: `{grouped.get('verdict')}`",
        f"- domain diagnostic verdict: `{domain.get('verdict')}`",
        f"- reasoning probe verdict: `{reasoning.get('verdict')}`",
        f"- math fallback verdict: `{math_probe.get('verdict')}`",
        f"- readiness: `{summary.get('BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_SELECTION_READINESS_VERDICT')}`",
        f"- best readiness-eligible policy: `{summary.get('best_readiness_policy')}`",
        f"- best readiness-eligible grouped-CV task macro: `{rate(summary.get('best_readiness_task_macro'))}`",
        f"- best diagnostic probe policy: `{summary.get('best_probe_policy')}`",
        f"- best diagnostic probe grouped-CV task macro: `{rate(summary.get('best_probe_task_macro'))}`",
        f"- merged top1 grouped-CV task macro: `{rate(summary.get('merged_tap_top1_task_macro'))}`",
        "",
        "## Files",
        "",
        f"- grouped CV: `{rel(GROUPED_MD)}`",
        f"- domain oracle diagnostic: `{rel(DOMAIN_MD)}`",
        f"- reasoning failure probe: `{rel(REASONING_MD)}`",
        f"- math fallback probe: `{rel(MATH_MD)}`",
        f"- readiness: `{rel(READINESS_MD)}`",
        f"- summary: `{rel(SUMMARY_MD)}`",
    ]
    write_md(DOC_MD, lines)
    section_title = "## Merged tap final arbiter integration v1.1 (2026-05-18)"
    section = [
        section_title,
        "",
        f"`MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = {summary.get('MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS')}`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = {summary.get('SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1')}`. Grouped task-disjoint CV best readiness-eligible policy was `{summary.get('best_readiness_policy')}` with task macro `{rate(summary.get('best_readiness_task_macro'))}`; merged tap top1 was `{rate(summary.get('merged_tap_top1_task_macro'))}`. This is final-arbiter integration only; no action steering or routing change was tested.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav_title = "## Merged tap final arbiter integration v1.1 (2026-05-18)"
    nav = [
        nav_title,
        "",
        f"Added `{rel(DOC_MD)}`. Status: `{summary.get('MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS')}`; readiness: `{summary.get('SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1')}`.",
    ]
    for target in NAV_TARGETS:
        append_section(target, nav_title, nav)


def main() -> int:
    ensure_root()
    if not SURVIVOR_FEATURES_PT.exists():
        raise FileNotFoundError(f"missing survivor features: {SURVIVOR_FEATURES_PT}")
    sets, selected_tap = load_data()
    if not sets:
        raise RuntimeError("no survivor sets loaded")
    grouped = grouped_cv(sets, selected_tap)
    domain = domain_oracle_diagnostic(sets, selected_tap, grouped)
    reasoning = reasoning_failure_probe(sets, selected_tap)
    math_probe = math_fallback_probe(sets, selected_tap)
    readiness_payload = readiness(grouped, domain, reasoning, math_probe)
    replay = replay_splits(
        sets,
        selected_tap,
        [
            "merged_tap_top1",
            "fixed_composite_top1",
            "majority_rank_aggregation",
            "universal_top1",
            "math_universal_else_merged",
            "math_universal_reasoning_majority_else_merged",
            "oracle_best_survivor",
        ],
    )
    summary = {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_GROUPED_CV_VERDICT": grouped.get("verdict"),
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_DOMAIN_ORACLE_VERDICT": domain.get("verdict"),
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_REASONING_PROBE_VERDICT": reasoning.get("verdict"),
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_MATH_FALLBACK_VERDICT": math_probe.get("verdict"),
        **readiness_payload,
        "dataset": {
            "survivor_sets": len(sets),
            "candidates": sum(len(s.get("candidates") or []) for s in sets),
            "tasks": len(all_tasks(sets)),
            "source_split_counts": dict(Counter(str(s.get("split")) for s in sets)),
            "domain_counts": dict(Counter(str(s.get("domain")) for s in sets)),
        },
        "diagnostic_replay_summaries": replay.get("summaries"),
        "files_created": [
            rel(GROUPED_JSON),
            rel(GROUPED_MD),
            rel(GROUPED_CSV),
            rel(DOMAIN_JSON),
            rel(DOMAIN_MD),
            rel(DOMAIN_CSV),
            rel(REASONING_JSON),
            rel(REASONING_MD),
            rel(REASONING_CSV),
            rel(MATH_JSON),
            rel(MATH_MD),
            rel(MATH_CSV),
            rel(READINESS_JSON),
            rel(READINESS_MD),
            rel(SUMMARY_JSON),
            rel(SUMMARY_MD),
            rel(DOC_MD),
        ],
    }
    write_grouped(grouped)
    write_domain(domain)
    write_reasoning(reasoning)
    write_math(math_probe)
    write_readiness(readiness_payload)
    write_summary(summary)
    write_doc(summary, grouped, domain, reasoning, math_probe)
    print(f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = {summary.get('MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS')}", flush=True)
    print(f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = {summary.get('SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
