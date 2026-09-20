"""Merged-tap final arbiter integration v1.

This cached integration run uses the selected merged weight tap as a top4
final-arbiter expert. It does not train Ouro, mutate checkpoints/tokenizers/tap
registries, run wrapper/local-agent or Hunter-Seeker code, apply steering, run
new generation, or change production routing.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import (
    PRIMARY_TARGET,
    SELECTED_TAPS_PT,
    SURVIVOR_FEATURES_PT,
    finite_mean,
    rate,
    safe_float,
    score_diff,
)


OUT_ROOT = PROBE_ROOT / "bg_merged_tap_final_arbiter_integration_v1_2026-05-18"
EVAL_JSON = OUT_ROOT / "integration_eval.json"
EVAL_MD = OUT_ROOT / "integration_eval.md"
EVAL_CSV = OUT_ROOT / "integration_rows.csv"
READINESS_JSON = OUT_ROOT / "selection_readiness.json"
READINESS_MD = OUT_ROOT / "selection_readiness.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_merged_tap_final_arbiter_integration_v1.md"

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
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
    "domain_prespecified_v1",
    "domain_old_code_guard_v1",
    "domain_val_best",
    "oracle_best_survivor",
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


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


def load_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    survivor_payload = torch.load(SURVIVOR_FEATURES_PT, map_location="cpu", weights_only=False)
    selected_payload = torch.load(SELECTED_TAPS_PT, map_location="cpu", weights_only=False)
    return list(survivor_payload.get("survivor_sets") or []), dict(selected_payload.get("selected") or {})


def rank_from_scores(scores: Sequence[float]) -> list[int]:
    return [idx for idx, _ in sorted(enumerate(scores), key=lambda x: (-safe_float(x[1], -1e9), x[0]))]


def rank_map(order: Sequence[int]) -> dict[int, int]:
    return {idx: rank + 1 for rank, idx in enumerate(order)}


def score_choice(cands: list[dict[str, Any]], key: str) -> int:
    return max(range(len(cands)), key=lambda i: (safe_float(cands[i].get(key), -1e9), -i)) if cands else -1


def rank_choice(cands: list[dict[str, Any]], key: str) -> int:
    return min(range(len(cands)), key=lambda i: (safe_float(cands[i].get(key), 1e9), i)) if cands else -1


def rank_sum_choice(rank_maps: Sequence[dict[int, int]], count: int) -> int:
    return min(range(count), key=lambda i: (sum(r.get(i, 99) for r in rank_maps), i)) if count else -1


def merged_scores(cands: list[dict[str, Any]], selected_tap: dict[str, Any]) -> list[float] | None:
    weight = selected_tap.get("weight")
    if not isinstance(weight, torch.Tensor):
        state = selected_tap.get("state_dict") or {}
        value = state.get("linear.weight") if isinstance(state, dict) else None
        weight = value.flatten() if isinstance(value, torch.Tensor) else None
    if not isinstance(weight, torch.Tensor):
        return None
    arch = str(selected_tap.get("architecture"))
    vecs: list[torch.Tensor] = []
    for cand in cands:
        vec = (cand.get("features_by_config") or {}).get(PRIMARY_TARGET)
        if not isinstance(vec, torch.Tensor):
            return None
        vecs.append(vec.detach().cpu().to(torch.float32).flatten())
    scores = []
    for i, left in enumerate(vecs):
        total = 0.0
        for j, right in enumerate(vecs):
            if i == j:
                continue
            total += score_diff(weight, arch, left - right)
        scores.append(total / max(len(vecs) - 1, 1))
    return scores


def base_choices(cands: list[dict[str, Any]], selected_tap: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    merged = merged_scores(cands, selected_tap)
    merged_order = rank_from_scores(merged or []) if merged else []
    merged_ranks = rank_map(merged_order)
    fixed_order = rank_from_scores([safe_float(c.get("fixed_composite_score"), -1e9) for c in cands])
    universal_order = rank_from_scores([safe_float(c.get("universal_score"), -1e9) for c in cands])
    old_order = rank_from_scores([safe_float(c.get("old_score"), -1e9) for c in cands])
    bridge_order = rank_from_scores([safe_float(c.get("bridge_score"), -1e9) for c in cands])
    code_order = rank_from_scores([safe_float(c.get("code_score"), -1e9) for c in cands])
    choices = {
        "merged_tap_top1": merged_order[0] if merged_order else -1,
        "fixed_composite_top1": fixed_order[0] if fixed_order else -1,
        "majority_rank_aggregation": rank_choice(cands, "rank_majority"),
        "universal_top1": universal_order[0] if universal_order else -1,
        "old_top1": old_order[0] if old_order else -1,
        "bridge_top1": bridge_order[0] if bridge_order else -1,
        "code_top1": code_order[0] if code_order else -1,
        "old_code_reasoning_top1": score_choice(cands, "old_code_reasoning_score"),
        "merged_fixed_rank_sum": rank_sum_choice([merged_ranks, rank_map(fixed_order)], len(cands)),
        "merged_universal_rank_sum": rank_sum_choice([merged_ranks, rank_map(universal_order)], len(cands)),
        "merged_fixed_universal_rank_sum": rank_sum_choice([merged_ranks, rank_map(fixed_order), rank_map(universal_order)], len(cands)),
        "merged_bridge_universal_rank_sum": rank_sum_choice([merged_ranks, rank_map(bridge_order), rank_map(universal_order)], len(cands)),
        "merged_old_fixed_rank_sum": rank_sum_choice([merged_ranks, rank_map(old_order), rank_map(fixed_order)], len(cands)),
    }
    aux = {
        "merged_scores": merged,
        "merged_ranks": merged_ranks,
        "fixed_ranks": rank_map(fixed_order),
        "universal_ranks": rank_map(universal_order),
        "old_ranks": rank_map(old_order),
        "bridge_ranks": rank_map(bridge_order),
        "code_ranks": rank_map(code_order),
    }
    return choices, aux


def evaluate_rows_for_policy(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], policy: str, domain_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in sets:
        cands = list(s.get("candidates") or [])
        if not cands:
            continue
        choices, aux = base_choices(cands, selected_tap)
        rewards = [safe_float(c.get("final_reward", c.get("reward", 0.0)), 0.0) for c in cands]
        best_idx = max(range(len(cands)), key=lambda i: (rewards[i], -i))
        choices["oracle_best_survivor"] = best_idx
        domain = str(s.get("domain"))
        if policy == "domain_prespecified_v1":
            if domain == "coding":
                # Preserves old/code sensitivity by mixing code, fixed, and merged.
                choice = choices.get("merged_old_fixed_rank_sum", -1)
            elif domain == "math_simple_arithmetic":
                choice = choices.get("universal_top1", -1)
            else:
                choice = choices.get("merged_tap_top1", -1)
        elif policy == "domain_old_code_guard_v1":
            if domain in {"coding", "math_simple_arithmetic"}:
                choice = choices.get("merged_fixed_universal_rank_sum", -1)
            else:
                choice = choices.get("merged_tap_top1", -1)
        elif policy == "domain_val_best":
            mapped = (domain_map or {}).get(domain, "merged_tap_top1")
            choice = choices.get(mapped, -1)
        else:
            choice = choices.get(policy, -1)
        if choice < 0 or choice >= len(cands):
            continue
        selected = cands[choice]
        reward = rewards[choice]
        rows.append(
            {
                "policy": policy,
                "survivor_set_id": s.get("survivor_set_id"),
                "task_id": s.get("task_id"),
                "domain": domain,
                "split": s.get("split"),
                "candidate_count": len(cands),
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_reward": reward,
                "best_reward": rewards[best_idx],
                "oracle_selected": 1.0 if reward == rewards[best_idx] else 0.0,
                "regret": rewards[best_idx] - reward,
                "selected_branch_origin": selected.get("branch_origin") or selected.get("origin"),
                "mapped_policy": (domain_map or {}).get(domain) if policy == "domain_val_best" else "",
                "merged_rank": (aux.get("merged_ranks") or {}).get(choice),
                "fixed_rank": (aux.get("fixed_ranks") or {}).get(choice),
                "universal_rank": (aux.get("universal_ranks") or {}).get(choice),
            }
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    by_task: dict[str, list[float]] = defaultdict(list)
    by_domain_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        task = str(row.get("task_id"))
        domain = str(row.get("domain"))
        reward = safe_float(row.get("selected_reward"), 0.0)
        by_task[task].append(reward)
        by_domain_task[domain][task].append(reward)
    return {
        "n": len(rows),
        "tasks": len(by_task),
        "task_macro_reward": finite_mean(mean(vals) for vals in by_task.values()),
        "group_micro_reward": finite_mean(row.get("selected_reward") for row in rows),
        "oracle_selected_rate": finite_mean(row.get("oracle_selected") for row in rows),
        "regret": finite_mean(row.get("regret") for row in rows),
        "domain_task_macro": {domain: finite_mean(mean(vals) for vals in tasks.values()) for domain, tasks in sorted(by_domain_task.items())},
    }


def domain_val_best_map(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], candidate_policies: Sequence[str]) -> dict[str, str]:
    val_sets = [s for s in sets if str(s.get("split")) == "val"]
    domains = sorted({str(s.get("domain")) for s in val_sets})
    out: dict[str, str] = {}
    for domain in domains:
        domain_sets = [s for s in val_sets if str(s.get("domain")) == domain]
        scored = []
        for policy in candidate_policies:
            rows = evaluate_rows_for_policy(domain_sets, selected_tap, policy)
            scored.append((policy, safe_float(summarize(rows).get("task_macro_reward"), -1e9), len(rows)))
        out[domain] = max(scored, key=lambda x: (x[1], x[2], -candidate_policies.index(x[0])))[0]
    return out


def select_policy_on_validation(sets: Sequence[dict[str, Any]], selected_tap: dict[str, Any], domain_map: dict[str, str]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    val_sets = [s for s in sets if str(s.get("split")) == "val"]
    rows = []
    policy_summaries = {}
    for policy in BASE_POLICIES:
        if policy == "oracle_best_survivor":
            continue
        eval_rows = evaluate_rows_for_policy(val_sets, selected_tap, policy, domain_map=domain_map)
        summary = summarize(eval_rows)
        policy_summaries[policy] = summary
        rows.append({"policy": policy, **summary})
    # Guardrails are intentionally simple and validation-only.
    fixed = policy_summaries.get("fixed_composite_top1", {})
    fixed_coding = safe_float((fixed.get("domain_task_macro") or {}).get("coding"), 0.0)
    fixed_math = safe_float((fixed.get("domain_task_macro") or {}).get("math_simple_arithmetic"), 0.0)
    eligible = []
    for policy, summary in policy_summaries.items():
        domains = summary.get("domain_task_macro") or {}
        coding_ok = safe_float(domains.get("coding"), 0.0) >= max(0.0, fixed_coding - 0.05)
        math_ok = safe_float(domains.get("math_simple_arithmetic"), 0.0) >= max(0.0, fixed_math - 0.05)
        if coding_ok and math_ok:
            eligible.append((policy, summary))
    if not eligible:
        eligible = list(policy_summaries.items())
    selected_policy, selected_summary = max(eligible, key=lambda x: (safe_float(x[1].get("task_macro_reward"), -1e9), safe_float(x[1].get("oracle_selected_rate"), -1e9)))
    return selected_policy, selected_summary, rows


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


def main() -> int:
    ensure_root()
    sets, selected_tap = load_data()
    candidate_for_domain_map = [
        "merged_tap_top1",
        "fixed_composite_top1",
        "majority_rank_aggregation",
        "universal_top1",
        "old_top1",
        "code_top1",
        "merged_fixed_rank_sum",
        "merged_universal_rank_sum",
        "merged_fixed_universal_rank_sum",
    ]
    domain_map = domain_val_best_map(sets, selected_tap, candidate_for_domain_map)
    selected_policy, selected_val_summary, val_rows = select_policy_on_validation(sets, selected_tap, domain_map)
    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    for split in sorted({str(s.get("split")) for s in sets}):
        split_sets = [s for s in sets if str(s.get("split")) == split]
        for policy in BASE_POLICIES:
            rows = evaluate_rows_for_policy(split_sets, selected_tap, policy, domain_map=domain_map)
            all_rows.extend(rows)
            split_summaries[f"{split}::{policy}"] = summarize(rows)
    fresh_selected = split_summaries.get(f"fresh_holdout::{selected_policy}", {"n": 0})
    fresh_merged = split_summaries.get("fresh_holdout::merged_tap_top1", {"n": 0})
    fresh_fixed = split_summaries.get("fresh_holdout::fixed_composite_top1", {"n": 0})
    fresh_majority = split_summaries.get("fresh_holdout::majority_rank_aggregation", {"n": 0})
    fresh_oracle = split_summaries.get("fresh_holdout::oracle_best_survivor", {"n": 0})
    selected_reward = safe_float(fresh_selected.get("task_macro_reward"), float("nan"))
    fixed_reward = safe_float(fresh_fixed.get("task_macro_reward"), float("nan"))
    majority_reward = safe_float(fresh_majority.get("task_macro_reward"), float("nan"))
    merged_reward = safe_float(fresh_merged.get("task_macro_reward"), float("nan"))
    oracle_reward = safe_float(fresh_oracle.get("task_macro_reward"), float("nan"))
    gap_closure = (selected_reward - max(fixed_reward, majority_reward)) / max(oracle_reward - max(fixed_reward, majority_reward), 1e-9)
    domains = fresh_selected.get("domain_task_macro") or {}
    coding_ok = safe_float(domains.get("coding"), 0.0) >= 0.70
    math_ok = safe_float(domains.get("math_simple_arithmetic"), 0.0) >= 0.70
    if selected_reward >= 0.75 and selected_reward > max(fixed_reward, majority_reward) and coding_ok and math_ok:
        readiness = "READY_FOR_PHASE2B_STEERING_COMPARISON"
        status = "FINAL_ARBITER_INTEGRATED_READY"
    elif selected_reward > max(fixed_reward, majority_reward, merged_reward - 1e-9):
        readiness = "FINAL_ARBITER_WEAK_BUT_IMPROVED"
        status = "FINAL_ARBITER_INTEGRATION_USEFUL"
    elif merged_reward > max(fixed_reward, majority_reward):
        readiness = "USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY"
        status = "MERGED_TOP1_USEFUL"
    else:
        readiness = "NOT_READY"
        status = "NO_IMPROVEMENT"
    eval_payload = {
        "BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_EVAL_VERDICT": status,
        "verdict": status,
        "selected_policy": selected_policy,
        "selected_validation_summary": selected_val_summary,
        "domain_val_best_map": domain_map,
        "fresh_holdout_selected": fresh_selected,
        "fresh_holdout_merged_top1": fresh_merged,
        "fresh_holdout_fixed_top1": fresh_fixed,
        "fresh_holdout_majority": fresh_majority,
        "fresh_holdout_oracle": fresh_oracle,
        "gap_closure_vs_best_fixed_majority": gap_closure,
        "split_summaries": split_summaries,
        "validation_selection_rows": val_rows,
        "rows": all_rows,
        "anti_leakage": {
            "policy_selected_on_validation_only": True,
            "fresh_holdout_used_for_evaluation_only": True,
            "no_new_generation": True,
        },
    }
    readiness_payload = {
        "BG_MERGED_TAP_FINAL_ARBITER_SELECTION_READINESS_VERDICT": readiness,
        "verdict": readiness,
        "status": status,
        "selected_policy": selected_policy,
        "fresh_holdout_task_macro_reward": selected_reward,
        "fresh_holdout_fixed_task_macro_reward": fixed_reward,
        "fresh_holdout_majority_task_macro_reward": majority_reward,
        "fresh_holdout_merged_top1_task_macro_reward": merged_reward,
        "fresh_holdout_oracle_task_macro_reward": oracle_reward,
        "gap_closure_vs_best_fixed_majority": gap_closure,
        "coding_ok": coding_ok,
        "math_ok": math_ok,
        "remaining_limits": [
            "validation split is small, especially science/coding/math",
            "old-context pairwise preservation from merged-tap v1 remains SMALL_DEGRADATION",
            "this is final-arbiter integration only, not action steering",
        ],
    }
    summary = {
        **{k: v for k, v in eval_payload.items() if k != "rows"},
        **readiness_payload,
        "MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS": status,
        "SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION": readiness,
        "files_created": [rel(p) for p in [EVAL_JSON, EVAL_MD, EVAL_CSV, READINESS_JSON, READINESS_MD, SUMMARY_JSON, SUMMARY_MD, DOC_MD]],
    }
    write_json(EVAL_JSON, eval_payload)
    write_csv(EVAL_CSV, all_rows)
    write_json(READINESS_JSON, readiness_payload)
    write_json(SUMMARY_JSON, summary)
    eval_lines = [
        "# Merged Tap Final Arbiter Integration v1",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_EVAL_VERDICT = {status}",
        "",
        f"- selected policy: `{selected_policy}`",
        f"- validation-selected task macro on fresh holdout: `{rate(selected_reward)}`",
        f"- merged tap top1 task macro on fresh holdout: `{rate(merged_reward)}`",
        f"- fixed-composite top1 task macro on fresh holdout: `{rate(fixed_reward)}`",
        f"- majority-rank task macro on fresh holdout: `{rate(majority_reward)}`",
        f"- oracle-best survivor task macro on fresh holdout: `{rate(oracle_reward)}`",
        f"- gap closure vs best fixed/majority: `{rate(gap_closure)}`",
        "",
        "## Validation Selection",
        "",
    ]
    eval_lines.extend(md_table(val_rows, ["policy", "n", "tasks", "task_macro_reward", "group_micro_reward", "oracle_selected_rate", "regret", "domain_task_macro"]))
    eval_lines.extend(["", "## Fresh Holdout Key Policies", ""])
    key_table = []
    for policy in ["merged_tap_top1", selected_policy, "fixed_composite_top1", "majority_rank_aggregation", "universal_top1", "domain_prespecified_v1", "domain_val_best", "oracle_best_survivor"]:
        key = f"fresh_holdout::{policy}"
        if key in split_summaries:
            key_table.append({"policy": policy, **split_summaries[key]})
    eval_lines.extend(md_table(key_table, ["policy", "n", "tasks", "task_macro_reward", "group_micro_reward", "oracle_selected_rate", "regret", "domain_task_macro"]))
    write_md(EVAL_MD, eval_lines)
    readiness_lines = [
        "# Merged Tap Final Arbiter Selection Readiness",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_SELECTION_READINESS_VERDICT = {readiness}",
        "",
        f"- selected policy: `{selected_policy}`",
        f"- fresh holdout task macro reward: `{rate(selected_reward)}`",
        f"- minimum target: `0.7500`",
        f"- coding ok: `{coding_ok}`",
        f"- math ok: `{math_ok}`",
        f"- no steering tested: `True`",
        "",
        "## Remaining Limits",
        "",
    ]
    readiness_lines.extend(f"- {item}" for item in readiness_payload["remaining_limits"])
    write_md(READINESS_MD, readiness_lines)
    summary_lines = [
        "# Merged Tap Final Arbiter Integration v1 Summary",
        "",
        f"BG_MERGED_TAP_FINAL_ARBITER_INTEGRATION_EVAL_VERDICT = {status}",
        f"BG_MERGED_TAP_FINAL_ARBITER_SELECTION_READINESS_VERDICT = {readiness}",
        f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = {status}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = {readiness}",
        "",
        f"- selected policy: `{selected_policy}`",
        f"- fresh task macro: `{rate(selected_reward)}`",
        f"- merged top1 fresh task macro: `{rate(merged_reward)}`",
        f"- fixed top1 fresh task macro: `{rate(fixed_reward)}`",
        f"- majority fresh task macro: `{rate(majority_reward)}`",
        f"- oracle fresh task macro: `{rate(oracle_reward)}`",
    ]
    write_md(SUMMARY_MD, summary_lines)
    doc_lines = [
        "# Merged Tap Final Arbiter Integration v1",
        "",
        f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = {status}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = {readiness}",
        "",
        "This cached run integrated the selected merged weight tap as a final-arbiter expert among fixed-composite top4 survivors. It did not train Ouro, run action steering, change routing, or run new generation.",
        "",
        "## Result",
        "",
        f"- selected policy: `{selected_policy}`",
        f"- fresh holdout task macro: `{rate(selected_reward)}`",
        f"- merged tap top1: `{rate(merged_reward)}`",
        f"- fixed-composite top1: `{rate(fixed_reward)}`",
        f"- majority-rank: `{rate(majority_reward)}`",
        f"- oracle best survivor: `{rate(oracle_reward)}`",
        f"- readiness: `{readiness}`",
        "",
        "The merged tap remains useful as a final-arbiter signal, but this run does not clear Phase 2a readiness if the validation-selected integrated policy misses the 0.75 task-macro target or domain guardrails.",
        "",
        "## Files",
        "",
        f"- integration eval: `{rel(EVAL_MD)}`",
        f"- readiness: `{rel(READINESS_MD)}`",
        f"- summary: `{rel(SUMMARY_MD)}`",
    ]
    write_md(DOC_MD, doc_lines)
    section_title = "## Merged tap final arbiter integration v1 (2026-05-18)"
    section = [
        section_title,
        "",
        f"`MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = {status}`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = {readiness}`. The selected validation policy was `{selected_policy}` with fresh-holdout task macro `{rate(selected_reward)}`. This is final-arbiter integration only; no action steering or routing change was tested.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav_section = "## Merged tap final arbiter integration v1 (2026-05-18)"
    nav_lines = [
        nav_section,
        "",
        f"Added `{rel(DOC_MD)}`. Status: `{status}`; readiness: `{readiness}`.",
    ]
    for target in NAV_TARGETS:
        append_section(target, nav_section, nav_lines)
    print(f"MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = {status}", flush=True)
    print(f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = {readiness}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
