"""Two-tap old-anchored branch selector v1.

Tests whether the old-anchored transplanted coding/reasoning and
mixed-objective taps can be the only scoring taps for top4 branch selection.

This is selection/survival only. It uses cached candidate feature groups and
old/code pair datasets. It does not train Ouro, update old taps, overwrite
registries, run generation, apply steering, or change production routing.
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

from bg_gated_selector_v1_common import load_old_code_pairs
from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import (
    FINAL_ARBITER_JSON,
    PRIMARY_TARGET,
    branch_groups,
    eval_candidate_on_pairs,
    finite_mean,
    group_topk_metrics,
    load_pair_sets,
    rate,
    safe_float,
    score_diff,
)


OLD_ANCHOR_ROOT = PROBE_ROOT / "bg_old_anchored_branch_valid_taps_v1_2026-05-30"
OLD_ANCHOR_PT = OLD_ANCHOR_ROOT / "old_anchored_branch_valid_taps_v1.pt"
OUT_ROOT = PROBE_ROOT / "bg_two_tap_branch_selector_v1_2026-05-30"
EVAL_JSON = OUT_ROOT / "two_tap_branch_selector_eval.json"
EVAL_MD = OUT_ROOT / "two_tap_branch_selector_eval.md"
ROWS_CSV = OUT_ROOT / "two_tap_branch_selector_rows.csv"
POLICY_PT = OUT_ROOT / "two_tap_branch_selector_v1.pt"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_two_tap_branch_selector_v1.md"

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_merged_weight_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_selection_only_phase2_prototype_v1.md",
]
NAV_TARGETS = [
    PROJECT_ROOT / "shared/docs/evaluator/README.md",
    PROJECT_ROOT / "shared/docs/README.md",
    PROJECT_ROOT / "PROJECT_TREE_MAP.md",
    PROJECT_ROOT / "PROJECT_COMPONENTS.md",
]

CODE_TAP_NAME = "transplant::MIX_CODE_REASONING::full_residual::AntisymLinearNoNorm::102"
OBJECTIVE_TAP_NAME = "transplant::MIX_OBJECTIVE_ALL::full_residual::AntisymLinearNoNorm::135"

POLICY_WEIGHTS = {
    "two_tap_equal": (0.5, 0.5),
    "two_tap_code_heavy": (0.65, 0.35),
    "two_tap_objective_heavy": (0.35, 0.65),
    "two_tap_code_primary_objective_rescue": (0.8, 0.2),
    "two_tap_objective_primary_code_rescue": (0.2, 0.8),
}


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().to(torch.float32)
        return {"shape": list(x.shape), "mean": float(x.mean().item()) if x.numel() else 0.0}
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


def load_taps() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = torch.load(OLD_ANCHOR_PT, map_location="cpu", weights_only=False)
    candidates = list(payload.get("candidates") or [])
    code = next(c for c in candidates if c.get("candidate_name") == CODE_TAP_NAME)
    objective = next(c for c in candidates if c.get("candidate_name") == OBJECTIVE_TAP_NAME)
    return code, objective, candidates


def tap_weight(tap: dict[str, Any]) -> torch.Tensor:
    weight = tap.get("weight")
    if isinstance(weight, torch.Tensor):
        return weight.detach().cpu().to(torch.float32).flatten()
    state = tap.get("state_dict") or {}
    value = state.get("linear.weight") if isinstance(state, dict) else None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).flatten()
    raise ValueError(f"missing tap weight for {tap.get('candidate_name')}")


def candidate_scores(group: Sequence[dict[str, Any]], tap: dict[str, Any]) -> list[float]:
    weight = tap_weight(tap)
    arch = str(tap.get("architecture"))
    vecs = []
    for row in group:
        vec = (row.get("features_by_config") or {}).get(PRIMARY_TARGET)
        if not isinstance(vec, torch.Tensor):
            return []
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


def normalize_scores(scores: Sequence[float]) -> list[float]:
    vals = [safe_float(s, 0.0) for s in scores]
    if not vals:
        return []
    mu = mean(vals)
    var = mean((v - mu) ** 2 for v in vals)
    sd = math.sqrt(var)
    if sd < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / sd for v in vals]


def rank_order_for_policy(group: Sequence[dict[str, Any]], policy: str, code_tap: dict[str, Any], objective_tap: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    if policy == "code_tap_top4":
        scores = candidate_scores(group, code_tap)
    elif policy == "objective_tap_top4":
        scores = candidate_scores(group, objective_tap)
    elif policy in POLICY_WEIGHTS:
        cw, ow = POLICY_WEIGHTS[policy]
        code_scores = normalize_scores(candidate_scores(group, code_tap))
        obj_scores = normalize_scores(candidate_scores(group, objective_tap))
        if not code_scores or not obj_scores:
            return [], {}
        scores = [cw * c + ow * o for c, o in zip(code_scores, obj_scores)]
    elif policy == "old_code_anchor_top4":
        # This is approximated by removing branch/bridge residuals: use the
        # matching old source embedded in the old-anchored artifact if present.
        scores = candidate_scores(group, code_tap)
    elif policy == "oracle_top4":
        rewards = [safe_float(row.get("reward"), 0.0) for row in group]
        order = [i for i, _ in sorted(enumerate(rewards), key=lambda x: (-x[1], x[0]))]
        return order, {"oracle_scores": rewards}
    else:
        return [], {}
    order = [i for i, _ in sorted(enumerate(scores), key=lambda x: (-safe_float(x[1], -1e9), x[0]))]
    return order, {"scores": scores}


def eval_policy_on_groups(groups: Sequence[Sequence[dict[str, Any]]], policy: str, code_tap: dict[str, Any], objective_tap: dict[str, Any], split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        rewards = [safe_float(row.get("reward"), 0.0) for row in group]
        if not rewards:
            continue
        order, aux = rank_order_for_policy(group, policy, code_tap, objective_tap)
        if not order:
            continue
        selected = order[: min(4, len(order))]
        oracle = max(rewards)
        oracle_indices = {i for i, r in enumerate(rewards) if r == oracle}
        kept = bool(set(selected) & oracle_indices)
        best_selected = max(rewards[i] for i in selected)
        top1_reward = rewards[selected[0]]
        rows.append(
            {
                "split": split,
                "policy": policy,
                "group_id": group[0].get("group_id"),
                "domain": group[0].get("domain"),
                "group_size": len(group),
                "selected_count": len(selected),
                "oracle_retention": 1.0 if kept else 0.0,
                "false_prune_rate": 0.0 if kept else 1.0,
                "best_selected_reward": best_selected,
                "oracle_reward": oracle,
                "regret": oracle - best_selected,
                "top1_success": 1.0 if top1_reward == oracle and oracle > 0 else 0.0,
                "top1_reward": top1_reward,
                "selected_indices": selected,
            }
        )
    return summarize_survival_rows(rows), rows


def summarize_survival_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"group_count": 0}
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain"))].append(row)
    return {
        "group_count": len(rows),
        "oracle_retention": finite_mean(row.get("oracle_retention") for row in rows),
        "false_prune_rate": finite_mean(row.get("false_prune_rate") for row in rows),
        "avg_survivors": finite_mean(row.get("selected_count") for row in rows),
        "best_selected_reward": finite_mean(row.get("best_selected_reward") for row in rows),
        "top1_success": finite_mean(row.get("top1_success") for row in rows),
        "top1_reward": finite_mean(row.get("top1_reward") for row in rows),
        "regret": finite_mean(row.get("regret") for row in rows),
        "domain_retention": {d: finite_mean(r.get("oracle_retention") for r in vals) for d, vals in sorted(by_domain.items())},
        "domain_false_prune": {d: finite_mean(r.get("false_prune_rate") for r in vals) for d, vals in sorted(by_domain.items())},
        "domain_best_selected_reward": {d: finite_mean(r.get("best_selected_reward") for r in vals) for d, vals in sorted(by_domain.items())},
    }


def build_pair_sets() -> dict[str, list[dict[str, Any]]]:
    pair_sets = load_pair_sets()
    code_pairs = load_old_code_pairs()
    for pair in code_pairs:
        pair["pair_type"] = "old_code"
    pair_sets["old_code"] = code_pairs
    return pair_sets


def pair_eval(tap: dict[str, Any], pair_sets: dict[str, list[dict[str, Any]]], split: str) -> dict[str, Any]:
    old = eval_candidate_on_pairs(tap, pair_sets["old_content"], split=split)
    code = eval_candidate_on_pairs(tap, pair_sets["old_code"], split=split)
    hidden = eval_candidate_on_pairs(tap, pair_sets["hidden_branch"], split=split)
    bridge = eval_candidate_on_pairs(tap, pair_sets["bridge"], split=split)
    return {
        "old_acc": old.get("pairwise_accuracy"),
        "old_pairs": old.get("pair_count"),
        "code_acc": code.get("pairwise_accuracy"),
        "code_pairs": code.get("pair_count"),
        "hidden_branch_acc": hidden.get("pairwise_accuracy"),
        "hidden_branch_pairs": hidden.get("pair_count"),
        "bridge_acc": bridge.get("pairwise_accuracy"),
        "bridge_pairs": bridge.get("pair_count"),
    }


def fixed_composite_reference() -> dict[str, Any]:
    path = PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18/heldout_survival_eval.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def choose_on_validation(policy_summaries: dict[str, dict[str, Any]]) -> str:
    candidates = []
    for policy, summary in policy_summaries.items():
        if policy == "oracle_top4":
            continue
        retention = safe_float(summary.get("oracle_retention"), 0.0)
        false_prune = safe_float(summary.get("false_prune_rate"), 1.0)
        best_reward = safe_float(summary.get("best_selected_reward"), -1e9)
        top1 = safe_float(summary.get("top1_success"), -1e9)
        if retention >= 0.90 and false_prune <= 0.10:
            candidates.append((policy, retention, false_prune, best_reward, top1))
    if not candidates:
        candidates = [
            (
                policy,
                safe_float(summary.get("oracle_retention"), 0.0),
                safe_float(summary.get("false_prune_rate"), 1.0),
                safe_float(summary.get("best_selected_reward"), -1e9),
                safe_float(summary.get("top1_success"), -1e9),
            )
            for policy, summary in policy_summaries.items()
            if policy != "oracle_top4"
        ]
    return max(candidates, key=lambda x: (x[1], -x[2], x[3], x[4], -list(POLICY_WEIGHTS.keys()).index(x[0]) if x[0] in POLICY_WEIGHTS else -99))[0]


def main() -> int:
    ensure_root()
    code_tap, objective_tap, candidates = load_taps()
    pair_sets = build_pair_sets()
    groups_by_split = {split: branch_groups(split) for split in ("val", "heldout")}
    policies = list(POLICY_WEIGHTS) + ["code_tap_top4", "objective_tap_top4", "oracle_top4"]
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for split, groups in groups_by_split.items():
        for policy in policies:
            summary, rows = eval_policy_on_groups(groups, policy, code_tap, objective_tap, split)
            summaries[f"{split}::{policy}"] = summary
            all_rows.extend(rows)
    val_summaries = {policy: summaries.get(f"val::{policy}", {"group_count": 0}) for policy in policies}
    selected_policy = choose_on_validation(val_summaries)
    heldout_selected = summaries.get(f"heldout::{selected_policy}", {"group_count": 0})
    heldout_code = summaries.get("heldout::code_tap_top4", {"group_count": 0})
    heldout_objective = summaries.get("heldout::objective_tap_top4", {"group_count": 0})
    heldout_oracle = summaries.get("heldout::oracle_top4", {"group_count": 0})
    pair_eval_code = pair_eval(code_tap, pair_sets, "heldout")
    pair_eval_objective = pair_eval(objective_tap, pair_sets, "heldout")
    ref = fixed_composite_reference()
    fixed_top4_reference = {
        "oracle_retention": 0.931,
        "false_prune_rate": 0.069,
        "avg_survivors": 3.873,
        "source": "fixed_composite_conservative_top4_reference_from_prior_report",
    }
    retention = safe_float(heldout_selected.get("oracle_retention"), float("nan"))
    false_prune = safe_float(heldout_selected.get("false_prune_rate"), float("nan"))
    code_ok = safe_float(pair_eval_code.get("code_acc"), 0.0) >= 0.75 and safe_float(pair_eval_objective.get("code_acc"), 0.0) >= 0.80
    old_ok = safe_float(pair_eval_code.get("old_acc"), 0.0) >= 0.50 and safe_float(pair_eval_objective.get("old_acc"), 0.0) >= 0.50
    if retention >= 0.93 and false_prune <= 0.07 and code_ok and old_ok:
        status = "TWO_TAP_BRANCH_SELECTOR_READY"
    elif retention >= 0.90 and false_prune <= 0.10 and code_ok and old_ok:
        status = "TWO_TAP_BRANCH_SELECTOR_USABLE"
    elif retention >= 0.85 and false_prune <= 0.15:
        status = "TWO_TAP_BRANCH_SELECTOR_WEAK"
    else:
        status = "TWO_TAP_BRANCH_SELECTOR_NOT_READY"
    payload = {
        "BG_TWO_TAP_BRANCH_SELECTOR_STATUS": status,
        "status": status,
        "selected_policy": selected_policy,
        "selected_policy_validation": val_summaries.get(selected_policy),
        "selected_policy_heldout": heldout_selected,
        "heldout_code_tap": heldout_code,
        "heldout_objective_tap": heldout_objective,
        "heldout_oracle": heldout_oracle,
        "pair_eval_code_tap": pair_eval_code,
        "pair_eval_objective_tap": pair_eval_objective,
        "fixed_composite_reference": fixed_top4_reference,
        "fixed_composite_artifact_summary": ref,
        "policy_summaries": summaries,
        "tap_names": {"coding_reasoning": CODE_TAP_NAME, "mixed_objective_all": OBJECTIVE_TAP_NAME},
        "anti_leakage": {
            "selected_on_validation_only": True,
            "heldout_used_for_selection": False,
            "only_two_scoring_taps": True,
            "veto_rescue_ood_guardrails_not_reimplemented": True,
            "no_ouro_training": True,
            "no_action_steering": True,
            "no_routing_change": True,
        },
    }
    torch.save({"policy": payload, "code_tap": code_tap, "objective_tap": objective_tap}, POLICY_PT)
    write_json(EVAL_JSON, {**payload, "rows": all_rows})
    write_csv(ROWS_CSV, all_rows)
    summary = {
        **payload,
        "files_created": [rel(p) for p in [EVAL_JSON, EVAL_MD, ROWS_CSV, POLICY_PT, SUMMARY_JSON, SUMMARY_MD, DOC_MD]],
    }
    write_json(SUMMARY_JSON, summary)
    table = []
    for policy in policies:
        row = {"policy": policy, **summaries.get(f"heldout::{policy}", {})}
        table.append(row)
    lines = [
        "# Two-Tap Branch Selector v1",
        "",
        f"BG_TWO_TAP_BRANCH_SELECTOR_STATUS = {status}",
        "",
        f"- selected policy: `{selected_policy}`",
        f"- selected heldout oracle retention: `{rate(heldout_selected.get('oracle_retention'))}`",
        f"- selected heldout false prune: `{rate(heldout_selected.get('false_prune_rate'))}`",
        f"- selected heldout avg survivors: `{rate(heldout_selected.get('avg_survivors'))}`",
        f"- fixed-composite reference retention / false prune: `0.9310` / `0.0690`",
        f"- only two scoring taps: `True`",
        f"- guardrails reimplemented: `False`",
        "",
        "## Heldout Policies",
        "",
    ]
    lines.extend(md_table(table, ["policy", "group_count", "oracle_retention", "false_prune_rate", "avg_survivors", "best_selected_reward", "top1_success", "regret", "domain_retention"]))
    lines.extend(
        [
            "",
            "## Tap Preservation",
            "",
            f"- coding/reasoning tap old/code/hidden/bridge heldout: `{rate(pair_eval_code.get('old_acc'))}` / `{rate(pair_eval_code.get('code_acc'))}` / `{rate(pair_eval_code.get('hidden_branch_acc'))}` / `{rate(pair_eval_code.get('bridge_acc'))}`",
            f"- mixed-objective tap old/code/hidden/bridge heldout: `{rate(pair_eval_objective.get('old_acc'))}` / `{rate(pair_eval_objective.get('code_acc'))}` / `{rate(pair_eval_objective.get('hidden_branch_acc'))}` / `{rate(pair_eval_objective.get('bridge_acc'))}`",
        ]
    )
    write_md(EVAL_MD, lines)
    write_md(SUMMARY_MD, lines)
    doc_lines = [
        "# Two-Tap Branch Selector v1",
        "",
        f"BG_TWO_TAP_BRANCH_SELECTOR_STATUS = {status}",
        "",
        "This run tested whether the old-anchored `coding_reasoning` and `mixed_objective_all` transplanted taps can be the only scoring taps for top4 branch selection. It did not use the old+branch+bridge+universal fixed composite as the scoring policy, did not train Ouro, did not run steering, and did not change routing.",
        "",
        "## Result",
        "",
        f"- selected policy: `{selected_policy}`",
        f"- heldout oracle retention: `{rate(heldout_selected.get('oracle_retention'))}`",
        f"- heldout false prune: `{rate(heldout_selected.get('false_prune_rate'))}`",
        f"- heldout avg survivors: `{rate(heldout_selected.get('avg_survivors'))}`",
        f"- fixed-composite reference retention / false prune: `0.9310` / `0.0690`",
        f"- status: `{status}`",
        "",
        "## Interpretation",
        "",
        "The two transplanted old-anchored taps are sufficient for the cached branch-group survival proxy if they meet the selected retention/false-prune thresholds. This is not a production replacement because veto/rescue and missing/OOD guardrails were not reimplemented in the two-tap-only policy.",
        "",
        "## Files",
        "",
        f"- report: `{rel(EVAL_MD)}`",
        f"- artifact: `{rel(POLICY_PT)}`",
        f"- rows: `{rel(ROWS_CSV)}`",
    ]
    write_md(DOC_MD, doc_lines)
    section_title = "## Two-tap branch selector v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_TWO_TAP_BRANCH_SELECTOR_STATUS = {status}`. Selected `{selected_policy}` using only the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps. Heldout oracle retention `{rate(heldout_selected.get('oracle_retention'))}`, false prune `{rate(heldout_selected.get('false_prune_rate'))}`, avg survivors `{rate(heldout_selected.get('avg_survivors'))}`. No action steering, routing change, Ouro training, or tap-registry update was performed.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [
        section_title,
        "",
        f"Added `{rel(DOC_MD)}`. Status: `{status}`; selected policy: `{selected_policy}`.",
    ]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)
    print(f"BG_TWO_TAP_BRANCH_SELECTOR_STATUS = {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
