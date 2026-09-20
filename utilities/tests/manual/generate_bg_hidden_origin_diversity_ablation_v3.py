"""Run targeted hidden-origin branch-generation diversity ablation v3."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from collections import Counter
from typing import Any

import torch

from bg_hidden_origin_diversity_v3_common import (
    DIAGNOSTIC_ALPHA,
    DIRECTION_BANK_V3_PT,
    DIVERSITY_ABLATION_PARTIAL_PT,
    DIVERSITY_ABLATION_PT,
    MAX_NEW_TOKENS,
    PRIMARY_ALPHA_CAP,
    SEED,
    V3_ROOT,
    alpha_bucket,
    append_jsonl,
    branch_group_metrics,
    candidate_pair_stats,
    compact_branch_row_v3,
    deterministic_reward,
    ensure_v3_root,
    evaluate_mcq,
    generate_with_hook_v2,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v3_branch_rows,
    load_best_v1_head,
    load_best_v2_head,
    load_json,
    md_table,
    normalize_branch_point,
    primary_safe_v3_row,
    read_jsonl,
    rel,
    rms_normalize,
    row_reward,
    score_group_with_head,
    selected_v3_tasks,
    split_role_for_task,
    stable_v2_row,
    verdict_for_data_targets,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import HIDDEN_DIM, capture_prefix_features
from expand_bg_hidden_origin_branch_dataset import score_old_taps
from src.evaluator.bg_hidden_branching import delta_rms


OUT_PT = DIVERSITY_ABLATION_PT
OUT_JSON = V3_ROOT / "diversity_ablation_v3.json"
OUT_CSV = V3_ROOT / "diversity_ablation_v3.csv"
PROGRESS_JSONL = V3_ROOT / "diversity_ablation_progress.jsonl"
STATE_JSON = V3_ROOT / "diversity_ablation_state_v3.json"
OUT_MD = V3_ROOT / "diversity_ablation_report_v3.md"
REPORT_JSON = V3_ROOT / "diversity_ablation_report_v3.json"
PARTIAL_PT = DIVERSITY_ABLATION_PARTIAL_PT


FAMILY_ORDER = [
    "random_orthogonal",
    "paired_plus_minus",
    "old_tap_aligned",
    "v1_tap_aligned",
    "v2_tap_aligned",
    "hidden_origin_empirical",
    "hidden_origin_whitened",
    "adapter_proxy",
    "sequence_adapter_proxy",
    "empirical_plus_noise",
]
ALPHA_VALUE = {"alpha_0_005": 0.005, "alpha_0_01": 0.010, "alpha_0_02": 0.020}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-branch-rows", type=int, default=2500)
    parser.add_argument("--max-selected-tasks-used", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--sample-budget", type=int, default=32)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_bank() -> dict[str, Any]:
    if not DIRECTION_BANK_V3_PT.exists():
        return {"directions_by_layer": {}, "directions": []}
    return torch.load(DIRECTION_BANK_V3_PT, map_location="cpu", weights_only=False)


def bank_entries(bank: dict[str, Any], layer: int, family: str) -> list[dict[str, Any]]:
    by_layer = bank.get("directions_by_layer") or {}
    vals = list(by_layer.get(str(layer)) or by_layer.get(layer) or [])
    aliases = {
        "empirical_plus_noise": {"hidden_origin_empirical", "v2_tap_aligned", "v1_tap_aligned"},
        "paired_plus_minus": {"random_orthogonal"},
    }
    allowed = aliases.get(family, {family})
    out = []
    for row in vals:
        if row.get("family") in allowed and row.get("perturbation_usable") and isinstance(row.get("tensor"), torch.Tensor):
            if int(row["tensor"].numel()) == HIDDEN_DIM:
                out.append(row)
    return out


def orthogonalize(raw: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = raw.detach().cpu().flatten().to(torch.float32).clone()
    for base in basis:
        unit = rms_normalize(base)
        out = out - torch.dot(out, unit) / torch.dot(unit, unit).clamp(min=1e-8) * unit
    return rms_normalize(out)


def add_delta(entries: list[dict[str, Any]], basis: list[torch.Tensor], direction: torch.Tensor, alpha: float, family: str, dtype: str, name: str) -> None:
    vec = orthogonalize(direction, basis)
    basis.append(vec)
    entries.append(
        {
            "branch_id": len(entries),
            "delta": vec * float(alpha),
            "delta_family": family,
            "delta_type": dtype,
            "direction_name": name,
        }
    )


def make_family_deltas(layer: int, alpha: float, k: int, family: str, seed: int, bank: dict[str, Any]) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    entries = [
        {
            "branch_id": 0,
            "delta": torch.zeros(HIDDEN_DIM, dtype=torch.float32),
            "delta_family": "clean",
            "delta_type": "clean_zero",
            "direction_name": "clean_zero",
        }
    ]
    basis: list[torch.Tensor] = []
    if family == "paired_plus_minus":
        while len(entries) < k:
            base = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen), basis)
            basis.append(base)
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                if len(entries) >= k:
                    break
                entries.append(
                    {
                        "branch_id": len(entries),
                        "delta": base * float(alpha) * sign,
                        "delta_family": family,
                        "delta_type": f"paired_{label}",
                        "direction_name": f"paired_base_{len(basis)-1}",
                    }
                )
        return entries[:k]
    if family == "random_orthogonal":
        for idx in range(k - 1):
            add_delta(entries, basis, torch.randn(HIDDEN_DIM, generator=gen), alpha, family, f"random_orthogonal_{idx}", f"random_{idx}")
        return entries[:k]

    candidates = bank_entries(bank, layer, family)
    if not candidates:
        for idx in range(k - 1):
            add_delta(entries, basis, torch.randn(HIDDEN_DIM, generator=gen), alpha, "random_fallback", f"{family}_fallback_random_{idx}", f"{family}_fallback")
        return entries[:k]
    idx = 0
    while len(entries) < k:
        item = candidates[idx % len(candidates)]
        base = item["tensor"].detach().cpu().flatten().to(torch.float32)
        if family == "empirical_plus_noise":
            raw = base + 0.20 * torch.randn(HIDDEN_DIM, generator=gen)
            add_delta(entries, basis, raw, alpha, family, f"noise_around_{item.get('family')}", str(item.get("name")))
        else:
            vec = orthogonalize(base, basis)
            basis.append(vec)
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                if len(entries) >= k:
                    break
                entries.append(
                    {
                        "branch_id": len(entries),
                        "delta": vec * float(alpha) * sign,
                        "delta_family": family,
                        "delta_type": f"{family}_{label}",
                        "direction_name": str(item.get("name") or family),
                    }
                )
        idx += 1
        if idx > len(candidates) * 4 and len(entries) < k:
            add_delta(entries, basis, torch.randn(HIDDEN_DIM, generator=gen), alpha, f"{family}_plus_random", "random_fill", f"{family}_random_fill")
    return entries[:k]


def task_k(task: dict[str, Any]) -> int:
    tier = str(task.get("priority_tier") or "")
    cls = str(task.get("screening_class") or "")
    if tier == "high" and cls in {"perturbation_sensitive", "baseline_wrong_parseable", "baseline_parse_fragile"}:
        return 8
    if tier in {"high", "medium"}:
        return 6
    return 4


def planned_groups(tasks: list[dict[str, Any]], bank: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    def interleaved_tasks() -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = {role: [] for role in ("heldout_candidate", "train_candidate", "val_candidate", "unassigned")}
        for task in tasks:
            role = split_role_for_task(str(task["task_id"]))
            if os.environ.get("V3_HELDOUT_ONLY") == "1" and role != "heldout_candidate":
                continue
            buckets.setdefault(role, []).append(task)
        for vals in buckets.values():
            vals.sort(key=lambda row: (-float(row.get("priority_score") or 0.0), str(row.get("task_id"))))
        ordered: list[dict[str, Any]] = []
        role_sequence = ("heldout_candidate",) if os.environ.get("V3_HELDOUT_ONLY") == "1" else ("heldout_candidate", "heldout_candidate", "heldout_candidate", "train_candidate", "val_candidate", "unassigned")
        while any(buckets.values()):
            progressed = False
            for role in role_sequence:
                vals = buckets.get(role) or []
                if vals:
                    ordered.append(vals.pop(0))
                    progressed = True
            if not progressed:
                break
        return ordered

    groups = []
    rows_budget = 0
    diagnostic_l47_count = 0
    for task_index, task in enumerate(interleaved_tasks()):
        if rows_budget >= max_rows:
            break
        k = task_k(task)
        preferred_families = list(task.get("preferred_delta_family_candidates") or [])
        families = list(dict.fromkeys(preferred_families + FAMILY_ORDER))
        tier = str(task.get("priority_tier") or "")
        cls = str(task.get("screening_class") or "")
        if tier != "high":
            families = families[:5]
        branch_points = [bp for bp in list(task.get("preferred_branch_points") or ["L24", "L36"]) if bp in {"L24", "L36"}]
        if not branch_points:
            branch_points = ["L24", "L36"]
        if len(branch_points) == 1:
            branch_points = [branch_points[0], "L36" if branch_points[0] == "L24" else "L24"]
        usable_by_layer: dict[str, list[str]] = {}
        for branch_point in branch_points:
            layer = int(branch_point[1:])
            usable = []
            for family in families:
                if family in {"random_orthogonal", "paired_plus_minus"} or bank_entries(bank, layer, family):
                    usable.append(family)
            usable_by_layer[branch_point] = usable or ["random_orthogonal", "paired_plus_minus"]
        fam_cycle = list(dict.fromkeys(usable_by_layer[branch_points[0]] + usable_by_layer[branch_points[1]] + FAMILY_ORDER))
        offset = task_index % max(len(fam_cycle), 1)
        fam_cycle = fam_cycle[offset:] + fam_cycle[:offset]
        primary_group_count = 4 if tier == "high" else 3
        recipes: list[tuple[str, str, str]] = []
        base_recipes = [
            (branch_points[0], "alpha_0_005"),
            (branch_points[1], "alpha_0_01"),
            (branch_points[0], "alpha_0_01"),
            (branch_points[1], "alpha_0_005"),
        ]
        for idx, (branch_point, abucket) in enumerate(base_recipes[:primary_group_count]):
            usable = usable_by_layer.get(branch_point) or fam_cycle
            family = next((fam for fam in fam_cycle[idx:] + fam_cycle[:idx] if fam in usable), usable[0])
            recipes.append((branch_point, abucket, family))
        if tier == "high" or cls in {"baseline_parse_fragile", "perturbation_sensitive"}:
            diag_branch = branch_points[task_index % len(branch_points)]
            diag_usable = usable_by_layer.get(diag_branch) or fam_cycle
            recipes.append((diag_branch, "alpha_0_02", diag_usable[0]))
        for branch_point, abucket, family in recipes:
            if rows_budget + k > max_rows:
                break
            layer = int(branch_point[1:])
            groups.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "branch_point": branch_point,
                    "target_layer": layer,
                    "target_loop": 1,
                    "alpha_bucket": abucket,
                    "alpha": ALPHA_VALUE[abucket],
                    "safety_envelope": abucket != "alpha_0_02",
                    "K": k,
                    "primary_delta_family": family,
                    "diagnostic": abucket == "alpha_0_02",
                }
            )
            rows_budget += k
            if rows_budget >= max_rows:
                break
        if tier == "high" and diagnostic_l47_count < 24 and rows_budget + k <= max_rows:
            groups.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "branch_point": "L47",
                    "target_layer": 47,
                    "target_loop": 1,
                    "alpha_bucket": "alpha_0_02",
                    "alpha": 0.020,
                    "safety_envelope": False,
                    "K": k,
                    "primary_delta_family": "random_orthogonal",
                    "diagnostic": True,
                }
            )
            rows_budget += k
            diagnostic_l47_count += 1
    return groups


def stable_group_complete(rows: list[dict[str, Any]], group_id: str) -> bool:
    vals = [row for row in rows if row.get("branch_group_id") == group_id]
    return len([row for row in vals if stable_v2_row(row)]) >= 2


def load_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = load_json(STATE_JSON, {}) or {}
    completed = set(str(x) for x in state.get("completed_branch_group_ids", []))
    for row in read_jsonl(PROGRESS_JSONL):
        if row.get("branch_group_id") and row.get("status") == "completed":
            completed.add(str(row["branch_group_id"]))
    for gid, vals in group_rows(rows).items():
        if stable_group_complete(rows, gid):
            completed.add(str(gid))
    return {
        "completed_branch_group_ids": sorted(completed),
        "completed_task_ids": sorted({str(row.get("task_id")) for row in rows if str(row.get("branch_group_id")) in completed}),
    }


def save_partial(rows: list[dict[str, Any]], errors: list[dict[str, Any]], state: dict[str, Any], complete: bool = False) -> None:
    payload = {
        "complete": bool(complete),
        "rows": rows,
        "errors": errors,
        "completed_branch_group_ids": sorted(set(state.get("completed_branch_group_ids", []))),
        "completed_task_ids": sorted(set(state.get("completed_task_ids", []))),
        "saved_at": time.time(),
    }
    torch.save(payload, PARTIAL_PT)
    write_json(STATE_JSON, {k: v for k, v in payload.items() if k not in {"rows"}})
    compact = [compact_branch_row_v3(row) for row in rows]
    write_json(OUT_JSON, {"complete": bool(complete), "rows": compact, "errors": errors, "completed_branch_group_ids": payload["completed_branch_group_ids"]})
    write_csv(OUT_CSV, compact)


def add_sampled_rewards(model: Any, tokenizer: Any, task: dict[str, Any], group: list[dict[str, Any]], spec: dict[str, Any], device: torch.device, max_new_tokens: int, remaining_budget: int) -> int:
    if remaining_budget <= 0:
        return 0
    used = 0
    # Sampled expected reward is diagnostic only; keep it bounded so primary
    # deterministic branch generation and checkpointing remain the load-bearing
    # path.
    sample_targets = sorted(group, key=lambda row: (int(row.get("branch_id", 999)) != 0, int(row.get("branch_id", 999))))[:2]
    if len(sample_targets) < 2 and len(group) > 1:
        sample_targets = [group[0], group[1]]
    for row in sample_targets:
        if used + 2 > remaining_budget:
            break
        samples = []
        rewards = []
        for sample_id in range(2):
            gen = generate_with_hook_v2(
                model,
                tokenizer,
                task["prompt"],
                row["delta"],
                spec,
                device,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
            score = evaluate_mcq(task, gen["output_text"])
            rewards.append(float(score["reward"]))
            samples.append(
                {
                    "sample_id": sample_id,
                    "output_text": gen["output_text"],
                    "parsed_answer": score["parsed_answer"],
                    "correct": bool(score["correct"]),
                    "reward": float(score["reward"]),
                    "parse_success": bool(score["parse_success"]),
                    "hit_max_tokens": bool(gen["hit_max_tokens"]),
                    "output_length": int(gen["token_count"]),
                }
            )
            used += 1
        if rewards:
            row["sampled_outputs"] = samples
            row["sampled_expected_reward"] = float(sum(rewards) / len(rewards))
    return used


def generation_stats(rows: list[dict[str, Any]], errors: list[dict[str, Any]], split_payload: dict[str, Any]) -> dict[str, Any]:
    v3_rows = [row for row in rows if str(row.get("branch_group_id", "")).startswith("v3::")]
    v3_primary = [row for row in v3_rows if primary_safe_v3_row(row)]
    combined_primary = [row for row in load_all_v3_branch_rows(include_prior=True) + v3_primary if primary_safe_v3_row(row)]
    # De-duplicate combined rows; load_all_v3_branch_rows already includes saved v3 rows
    by_key = {(str(row.get("branch_group_id")), int(row.get("branch_id", -1))): row for row in combined_primary}
    combined_primary = list(by_key.values())
    new_groups = {gid: vals for gid, vals in group_rows(v3_primary).items() if len(vals) >= 2}
    combined_groups = {gid: vals for gid, vals in group_rows(combined_primary).items() if len(vals) >= 2}
    heldout_ids = {str(x) for x in split_payload.get("v3_heldout_candidate_task_ids", [])}
    heldout_groups = {gid: vals for gid, vals in combined_groups.items() if str(vals[0].get("task_id")) in heldout_ids}
    heldout_pair_stats = candidate_pair_stats(heldout_groups)
    new_diverse = [gid for gid, vals in new_groups.items() if group_is_behaviorally_diverse_v2(vals)]
    combined_diverse = [gid for gid, vals in combined_groups.items() if group_is_behaviorally_diverse_v2(vals)]
    heldout_diverse = [gid for gid, vals in heldout_groups.items() if group_is_behaviorally_diverse_v2(vals)]
    pair_stats = candidate_pair_stats(combined_groups)
    stable_count = sum(1 for row in v3_rows if stable_v2_row(row))
    return {
        "new_rows": len(v3_rows),
        "new_stable_rows": stable_count,
        "new_primary_stable_rows": len(v3_primary),
        "new_primary_stable_groups": len(new_groups),
        "new_behaviorally_diverse_groups": len(new_diverse),
        "new_reward_diverse_groups": len([gid for gid, vals in new_groups.items() if group_is_reward_diverse_v2(vals)]),
        "combined_primary_stable_groups": len(combined_groups),
        "combined_behaviorally_diverse_groups": len(combined_diverse),
        "combined_reward_diverse_groups": len([gid for gid, vals in combined_groups.items() if group_is_reward_diverse_v2(vals)]),
        "combined_tasks": len({str(row.get("task_id")) for row in combined_primary}),
        "heldout_candidate_task_ids": len(heldout_ids),
        "heldout_behaviorally_diverse_groups": len(heldout_diverse),
        "heldout_non_tie_pairs": heldout_pair_stats["non_tie_pairs"],
        "candidate_pair_stats": pair_stats,
        "heldout_pair_stats": heldout_pair_stats,
        "stable_rate_new": stable_count / max(len(v3_rows), 1),
        "parse_rate_new": sum(1 for row in v3_rows if row.get("parse_success")) / max(len(v3_rows), 1),
        "errors": len(errors),
    }


def should_sample(task: dict[str, Any], group: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    if spec.get("branch_point") == "L47":
        return False
    if str(task.get("priority_tier")) == "high":
        return True
    if group_is_behaviorally_diverse_v2(group):
        return True
    if str(task.get("screening_class")) in {"baseline_parse_fragile", "perturbation_sensitive"}:
        return True
    return False


def write_report(verdict: str, rows: list[dict[str, Any]], errors: list[dict[str, Any]], stats: dict[str, Any], selected_task_count: int, sampled_used: int, elapsed: float) -> None:
    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": verdict,
        "verdict": verdict,
        "stats": stats,
        "selected_task_count": selected_task_count,
        "row_count": len(rows),
        "errors": errors,
        "counts_by_domain": dict(Counter(str(row.get("domain")) for row in rows)),
        "counts_by_screening_class": dict(Counter(str(row.get("task_screening_class")) for row in rows)),
        "counts_by_split_guard_role": dict(Counter(str(row.get("split_guard_role")) for row in rows)),
        "counts_by_branch_point": dict(Counter(normalize_branch_point(row) for row in rows)),
        "counts_by_alpha_bucket": dict(Counter(str(row.get("alpha_bucket")) for row in rows)),
        "counts_by_primary_delta_family": dict(Counter(str(row.get("primary_delta_family")) for row in rows)),
        "sampled_outputs": sampled_used,
        "checkpointing": {
            "progress_jsonl": rel(PROGRESS_JSONL),
            "state_json": rel(STATE_JSON),
            "partial_pt": rel(PARTIAL_PT),
            "resumable": True,
            "skip_completed_branch_groups_unless_FORCE_RERUN": True,
        },
        "elapsed_seconds": round(elapsed, 3),
    }
    write_json(REPORT_JSON, payload)
    display_groups = []
    primary_groups = {gid: vals for gid, vals in group_rows([row for row in rows if primary_safe_v3_row(row)]).items() if len(vals) >= 2}
    for gid, vals in sorted(primary_groups.items())[:220]:
        display_groups.append(
            {
                "branch_group_id": gid,
                "task_id": vals[0].get("task_id"),
                "domain": vals[0].get("domain"),
                "class": vals[0].get("task_screening_class"),
                "split": vals[0].get("split_guard_role"),
                "branch_point": normalize_branch_point(vals[0]),
                "alpha": vals[0].get("alpha_bucket"),
                "family": vals[0].get("primary_delta_family"),
                "K": vals[0].get("K"),
                "rewards": sorted({deterministic_reward(row) for row in vals}),
                "answers": sorted({str(row.get("parsed_answer")) for row in vals}),
                "diverse": group_is_behaviorally_diverse_v2(vals),
            }
        )
    lines = [
        "# Hidden-Origin Diversity Ablation V3",
        "",
        f"BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = {verdict}",
        "",
        f"- new_rows: `{stats.get('new_rows')}`",
        f"- new_behaviorally_diverse_groups: `{stats.get('new_behaviorally_diverse_groups')}`",
        f"- combined_behaviorally_diverse_groups: `{stats.get('combined_behaviorally_diverse_groups')}`",
        f"- heldout_candidate_task_ids: `{stats.get('heldout_candidate_task_ids')}`",
        f"- heldout_behaviorally_diverse_groups: `{stats.get('heldout_behaviorally_diverse_groups')}`",
        f"- heldout_non_tie_pairs: `{stats.get('heldout_non_tie_pairs')}`",
        f"- tie_rate: `{stats.get('candidate_pair_stats', {}).get('tie_rate')}`",
        f"- sampled_outputs: `{sampled_used}`",
        f"- errors: `{len(errors)}`",
        "",
        "Alpha `0.02`, sampled expected reward, and L47 diagnostic branches are reported separately and are not used for primary selector-readiness targets.",
        "",
        "## Primary-Safe Groups",
        "",
    ]
    lines.extend(md_table(display_groups, ["branch_group_id", "task_id", "domain", "class", "split", "branch_point", "alpha", "family", "K", "rewards", "answers", "diverse"]))
    if errors:
        lines.extend(["", "## Errors", "", *[f"- `{e.get('task_id')}` `{e.get('branch_group_id')}` b{e.get('branch_id')}: {str(e.get('error'))[:240]}" for e in errors[:50]]])
    write_md(OUT_MD, lines)


def main() -> int:
    args = parse_args()
    ensure_v3_root()
    started = time.time()
    split_payload = load_json(V3_ROOT / "split_guard_v3.json", {}) or {}
    if bool(args.finalize_only):
        if not PARTIAL_PT.exists():
            payload = {
                "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": "BLOCKED",
                "verdict": "BLOCKED",
                "blocker": "missing partial generation artifact",
            }
            write_json(REPORT_JSON, payload)
            write_md(OUT_MD, ["# Hidden-Origin Diversity Ablation V3", "", "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED"])
            print("BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED", flush=True)
            return 1
        partial = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False)
        rows = list(partial.get("rows") or [])
        errors = list(partial.get("errors") or [])
        stats = generation_stats(rows, errors, split_payload)
        verdict = verdict_for_data_targets(
            new_v3_behaviorally_diverse_groups=int(stats["new_behaviorally_diverse_groups"]),
            combined_behaviorally_diverse_groups=int(stats["combined_behaviorally_diverse_groups"]),
            heldout_task_ids=int(stats["heldout_candidate_task_ids"]),
            heldout_behaviorally_diverse_groups=int(stats["heldout_behaviorally_diverse_groups"]),
            heldout_non_tie_pairs=int(stats["heldout_non_tie_pairs"]),
            stable_rate=float(stats["stable_rate_new"]),
            tie_rate=float(stats["candidate_pair_stats"]["tie_rate"]),
            errors=len(errors),
            rows=len(rows),
        )
        sampled_used = sum(len(row.get("sampled_outputs") or []) for row in rows)
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": verdict,
            "verdict": verdict,
            "rows": rows,
            "errors": errors,
            "stats": stats,
            "selected_task_count": len({str(row.get("task_id")) for row in rows}),
            "selected_task_ids": sorted({str(row.get("task_id")) for row in rows}),
            "completed_branch_group_ids": sorted({str(row.get("branch_group_id")) for row in rows}),
            "completed_task_ids": sorted({str(row.get("task_id")) for row in rows}),
            "sampled_outputs": sampled_used,
            "max_branch_rows": int(args.max_branch_rows),
            "max_new_tokens": int(args.max_new_tokens),
            "finalized_from_partial": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        torch.save(payload, OUT_PT)
        write_report(verdict, rows, errors, stats, payload["selected_task_count"], sampled_used, time.time() - started)
        save_partial(rows, errors, {"completed_branch_group_ids": payload["completed_branch_group_ids"], "completed_task_ids": payload["completed_task_ids"]}, complete=True)
        print(f"BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = {verdict}", flush=True)
        print(f"Wrote {rel(OUT_PT)}", flush=True)
        return 0 if verdict != "BLOCKED" else 1
    if split_payload.get("verdict") == "BLOCKED":
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "split guard blocked",
        }
        write_json(REPORT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diversity Ablation V3", "", "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED", flush=True)
        return 1
    bank = load_bank()
    if not bank.get("directions_by_layer"):
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "missing usable direction bank",
        }
        write_json(REPORT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diversity Ablation V3", "", "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED", flush=True)
        return 1
    tasks = selected_v3_tasks()[: int(args.max_selected_tasks_used)]
    if not tasks:
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "missing selected tasks",
        }
        write_json(REPORT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diversity Ablation V3", "", "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = BLOCKED", flush=True)
        return 1

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if PARTIAL_PT.exists():
        payload = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False)
        rows = list(payload.get("rows") or [])
        errors = list(payload.get("errors") or [])
    state = load_state(rows)
    completed = set(state.get("completed_branch_group_ids", []))
    force = os.environ.get("FORCE_RERUN") == "1"
    sampled_used = sum(len(row.get("sampled_outputs") or []) for row in rows)
    groups = planned_groups(tasks, bank, int(args.max_branch_rows))
    v1_head = load_best_v1_head()
    v2_head = load_best_v2_head()

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for spec_index, spec in enumerate(groups):
            task = spec["task"]
            group_id = (
                f"v3::{task['task_id']}::{spec['branch_point']}::{spec['alpha_bucket']}::"
                f"family={spec['primary_delta_family']}::k={spec['K']}::class={task.get('screening_class', 'unknown')}"
            )
            if group_id in completed and not force:
                continue
            if len(rows) >= int(args.max_branch_rows):
                break
            seed = (
                SEED
                + 7919 * int(spec["task_index"])
                + 101 * int(spec["target_layer"])
                + int(float(spec["alpha"]) * 100000)
                + 17 * spec_index
            )
            delta_entries = make_family_deltas(
                int(spec["target_layer"]),
                float(spec["alpha"]),
                int(spec["K"]),
                str(spec["primary_delta_family"]),
                seed,
                bank,
            )
            group_new: list[dict[str, Any]] = []
            group_errors: list[dict[str, Any]] = []
            branch_spec = {
                "target_layer": int(spec["target_layer"]),
                "target_loop": int(spec["target_loop"]),
                "branch_point": spec["branch_point"],
                "alpha": float(spec["alpha"]),
                "safety_envelope": bool(spec["safety_envelope"]),
            }
            for delta_entry in delta_entries:
                if len(rows) >= int(args.max_branch_rows):
                    break
                branch_id = int(delta_entry["branch_id"])
                try:
                    prefix = capture_prefix_features(model, tokenizer, task["prompt"], delta_entry["delta"], branch_spec, device)
                    gen = generate_with_hook_v2(
                        model,
                        tokenizer,
                        task["prompt"],
                        delta_entry["delta"],
                        branch_spec,
                        device,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                    )
                    score = evaluate_mcq(task, gen["output_text"])
                    row = {
                        "task_id": task["task_id"],
                        "domain": task["domain"],
                        "source_dataset": task.get("source_dataset"),
                        "task_screening_class": task.get("screening_class"),
                        "priority_score": task.get("priority_score"),
                        "priority_tier": task.get("priority_tier"),
                        "split_guard_role": split_role_for_task(task["task_id"], split_payload),
                        "branch_group_id": group_id,
                        "branch_id": branch_id,
                        "branch_method": "hook_intervention_per_branch",
                        "branch_point": spec["branch_point"],
                        "target_layer": int(spec["target_layer"]),
                        "target_loop": int(spec["target_loop"]),
                        "alpha": float(spec["alpha"]),
                        "alpha_bucket": spec["alpha_bucket"],
                        "safety_envelope": bool(spec["safety_envelope"]),
                        "diagnostic_alpha": bool(spec["alpha_bucket"] == "alpha_0_02"),
                        "diagnostic_l47": bool(spec["branch_point"] == "L47"),
                        "K": int(spec["K"]),
                        "primary_delta_family": spec["primary_delta_family"],
                        "delta_family": delta_entry["delta_family"],
                        "delta_type": delta_entry["delta_type"],
                        "direction_name": delta_entry["direction_name"],
                        "effective_delta_rms": float(delta_rms(delta_entry["delta"])),
                        "delta": delta_entry["delta"].detach().cpu().to(torch.float32),
                        "features": prefix["features"],
                        "pooled_vectors": prefix["pooled_vectors"],
                        "last_token_vectors": prefix["last_token_vectors"],
                        "output_text": gen["output_text"],
                        "parsed_answer": score["parsed_answer"],
                        "correct": bool(score["correct"]),
                        "deterministic_correct": bool(score["correct"]),
                        "reward": float(score["reward"]),
                        "deterministic_reward": float(score["reward"]),
                        "sampled_expected_reward": None,
                        "label_source": "deterministic",
                        "parse_success": bool(score["parse_success"]),
                        "parse_failure_reason": score["parse_failure_reason"],
                        "repetition_rate": float(score["repetition_rate"]),
                        "empty_output": bool(score["empty_output"]),
                        "hit_max_tokens": bool(gen["hit_max_tokens"]),
                        "output_length": int(gen["token_count"]),
                        "generation_seconds": float(gen["generation_seconds"]),
                        "hook_modifications": int(gen["hook_diagnostics"].get("modifications", 0)),
                        "prefix_hook_modifications": int(prefix["hook_diagnostics"].get("modifications", 0)),
                        "cuda_error": "",
                        "nan_inf": bool(prefix["nan_inf"]),
                    }
                    group_new.append(row)
                except Exception as exc:
                    err = {
                        "task_id": task.get("task_id"),
                        "branch_group_id": group_id,
                        "branch_id": branch_id,
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                    errors.append(err)
                    group_errors.append(err)
            if group_new:
                score_old_taps(group_new)
                score_group_with_head(group_new, v1_head, score_key="v1_tap_score")
                score_group_with_head(group_new, v2_head, score_key="v2_tap_score")
                rows.extend(group_new)
                completed.add(group_id)
                state = {
                    "completed_branch_group_ids": sorted(completed),
                    "completed_task_ids": sorted({str(row.get("task_id")) for row in rows if str(row.get("branch_group_id")) in completed}),
                }
                save_partial(rows, errors, state, complete=False)
                if should_sample(task, group_new, spec) and sampled_used < int(args.sample_budget):
                    try:
                        sampled_used += add_sampled_rewards(
                            model,
                            tokenizer,
                            task,
                            group_new,
                            branch_spec,
                            device,
                            int(args.max_new_tokens),
                            max(0, int(args.sample_budget) - sampled_used),
                        )
                    except Exception as exc:
                        err = {
                            "task_id": task.get("task_id"),
                            "branch_group_id": group_id,
                            "branch_id": "sampled_group",
                            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                        }
                        errors.append(err)
                        group_errors.append(err)
                save_partial(rows, errors, state, complete=False)
            else:
                completed.add(group_id)
            state = {
                "completed_branch_group_ids": sorted(completed),
                "completed_task_ids": sorted({str(row.get("task_id")) for row in rows if str(row.get("branch_group_id")) in completed}),
            }
            append_jsonl(
                PROGRESS_JSONL,
                {
                    "status": "completed",
                    "branch_group_id": group_id,
                    "task_id": task.get("task_id"),
                    "split_guard_role": split_role_for_task(task["task_id"], split_payload),
                    "branch_point": spec["branch_point"],
                    "alpha_bucket": spec["alpha_bucket"],
                    "primary_delta_family": spec["primary_delta_family"],
                    "K": spec["K"],
                    "rows_added": len(group_new),
                    "errors_added": len(group_errors),
                    "behaviorally_diverse": group_is_behaviorally_diverse_v2(group_new) if len(group_new) >= 2 else False,
                    "reward_values": sorted({deterministic_reward(row) for row in group_new}) if group_new else [],
                    "saved_at": time.time(),
                },
            )
            save_partial(rows, errors, state, complete=False)
            stats = generation_stats(rows, errors, split_payload)
            provisional = verdict_for_data_targets(
                new_v3_behaviorally_diverse_groups=int(stats["new_behaviorally_diverse_groups"]),
                combined_behaviorally_diverse_groups=int(stats["combined_behaviorally_diverse_groups"]),
                heldout_task_ids=int(stats["heldout_candidate_task_ids"]),
                heldout_behaviorally_diverse_groups=int(stats["heldout_behaviorally_diverse_groups"]),
                heldout_non_tie_pairs=int(stats["heldout_non_tie_pairs"]),
                stable_rate=float(stats["stable_rate_new"]),
                tie_rate=float(stats["candidate_pair_stats"]["tie_rate"]),
                errors=len(errors),
                rows=len(rows),
            )
            if provisional == "DIVERSITY_TARGET_MET":
                break
    except KeyboardInterrupt:
        stats = generation_stats(rows, errors, split_payload)
        verdict = verdict_for_data_targets(
            new_v3_behaviorally_diverse_groups=int(stats["new_behaviorally_diverse_groups"]),
            combined_behaviorally_diverse_groups=int(stats["combined_behaviorally_diverse_groups"]),
            heldout_task_ids=int(stats["heldout_candidate_task_ids"]),
            heldout_behaviorally_diverse_groups=int(stats["heldout_behaviorally_diverse_groups"]),
            heldout_non_tie_pairs=int(stats["heldout_non_tie_pairs"]),
            stable_rate=float(stats["stable_rate_new"]),
            tie_rate=float(stats["candidate_pair_stats"]["tie_rate"]),
            errors=len(errors),
            rows=len(rows),
        )
        write_report(verdict, rows, errors, stats, len(tasks), sampled_used, time.time() - started)
        raise
    finally:
        if extractor is not None:
            extractor.cleanup()

    stats = generation_stats(rows, errors, split_payload)
    verdict = verdict_for_data_targets(
        new_v3_behaviorally_diverse_groups=int(stats["new_behaviorally_diverse_groups"]),
        combined_behaviorally_diverse_groups=int(stats["combined_behaviorally_diverse_groups"]),
        heldout_task_ids=int(stats["heldout_candidate_task_ids"]),
        heldout_behaviorally_diverse_groups=int(stats["heldout_behaviorally_diverse_groups"]),
        heldout_non_tie_pairs=int(stats["heldout_non_tie_pairs"]),
        stable_rate=float(stats["stable_rate_new"]),
        tie_rate=float(stats["candidate_pair_stats"]["tie_rate"]),
        errors=len(errors),
        rows=len(rows),
    )
    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "errors": errors,
        "stats": stats,
        "selected_task_count": len(tasks),
        "selected_task_ids": [task["task_id"] for task in tasks],
        "completed_branch_group_ids": sorted(completed),
        "completed_task_ids": sorted({str(row.get("task_id")) for row in rows}),
        "sampled_outputs": sampled_used,
        "max_branch_rows": int(args.max_branch_rows),
        "max_new_tokens": int(args.max_new_tokens),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    save_partial(rows, errors, {"completed_branch_group_ids": sorted(completed), "completed_task_ids": payload["completed_task_ids"]}, complete=True)
    write_report(verdict, rows, errors, stats, len(tasks), sampled_used, time.time() - started)
    print(f"BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
