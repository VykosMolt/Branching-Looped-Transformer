"""Generate behaviorally targeted hidden-origin branch outcome groups v2."""
from __future__ import annotations

import argparse
import time
import traceback
from collections import Counter
from typing import Any

import torch

from bg_hidden_origin_diversity_v2_common import (
    DIVERSE_BRANCHES_PARTIAL_PT,
    DIVERSE_BRANCHES_PT,
    MAX_NEW_TOKENS,
    PRIMARY_ALPHA_CAP,
    SEED,
    TASK_SCREENING_JSON,
    V2_ROOT,
    alpha_bucket,
    compact_branch_row,
    deterministic_reward,
    ensure_v2_root,
    evaluate_mcq,
    generate_with_hook_v2,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v2_branch_rows,
    load_direction_bank_payload,
    load_json,
    load_more_candidate_tasks,
    make_diverse_branch_deltas,
    md_table,
    rel,
    safe_primary_row,
    selected_screening_classes,
    stable_v2_row,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import capture_prefix_features
from expand_bg_hidden_origin_branch_dataset import score_old_taps
from src.evaluator.bg_hidden_branching import delta_rms


OUT_PT = DIVERSE_BRANCHES_PT
OUT_JSON = V2_ROOT / "diverse_hidden_origin_branches.json"
OUT_CSV = V2_ROOT / "diverse_hidden_origin_branches.csv"
OUT_MD = V2_ROOT / "diverse_generation_report.md"
REPORT_JSON = V2_ROOT / "diverse_generation_report.json"
PARTIAL_PT = DIVERSE_BRANCHES_PARTIAL_PT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-branch-rows", type=int, default=1500)
    parser.add_argument("--max-tasks", type=int, default=128)
    parser.add_argument("--sample-borderline", action="store_true", default=True)
    parser.add_argument("--no-sample-borderline", dest="sample_borderline", action="store_false")
    parser.add_argument("--sample-budget", type=int, default=160)
    parser.add_argument("--include-diagnostic-alpha", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def selected_tasks(max_tasks: int) -> list[dict[str, Any]]:
    screening = load_json(TASK_SCREENING_JSON, {}) or {}
    selected = list(screening.get("selected_rows") or [])
    if selected:
        return selected[:max_tasks]
    by_id = {task["task_id"]: task for task in load_more_candidate_tasks()}
    ids = list(screening.get("selected_task_ids") or [])
    tasks = [by_id[task_id] for task_id in ids if task_id in by_id]
    return tasks[:max_tasks]


def task_k(row: dict[str, Any]) -> int:
    cls = str(row.get("screening_class") or "")
    if cls == "perturbation_sensitive":
        return 8
    if cls in {"baseline_wrong_parseable", "baseline_parse_fragile", "baseline_correct_low_confidence"}:
        return 6
    return 4


def generation_specs(include_diagnostic_alpha: bool = False) -> list[dict[str, Any]]:
    specs = []
    base = [
        (24, 0.010, True),
        (36, 0.010, True),
        (24, 0.005, True),
        (36, 0.005, True),
    ]
    if include_diagnostic_alpha:
        base.extend(
            [
                (24, 0.020, False),
                (36, 0.020, False),
                (47, 0.020, False),
            ]
        )
    for layer, alpha, safety in base:
        specs.append(
            {
                "target_layer": layer,
                "target_loop": 1,
                "branch_point": f"L{layer}_L1",
                "alpha": float(alpha),
                "alpha_bucket": alpha_bucket(alpha),
                "safety_envelope": bool(safety),
                "diagnostic": not bool(safety),
            }
        )
    return specs


def save_partial(rows: list[dict[str, Any]], errors: list[dict[str, Any]], complete: bool = False) -> None:
    payload = {"complete": bool(complete), "rows": rows, "errors": errors, "saved_at": time.time()}
    torch.save(payload, PARTIAL_PT)
    write_json(OUT_JSON, {"complete": bool(complete), "rows": [compact_branch_row(row) for row in rows], "errors": errors})
    write_csv(OUT_CSV, [compact_branch_row(row) for row in rows])


def candidate_pair_stats(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidate_pairs = 0
    tie_pairs = 0
    for vals in groups.values():
        ordered = list(vals)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                candidate_pairs += 1
                if deterministic_reward(ordered[i]) == deterministic_reward(ordered[j]):
                    tie_pairs += 1
    return {
        "candidate_pairs": candidate_pairs,
        "tie_pairs": tie_pairs,
        "non_tie_pairs": candidate_pairs - tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
    }


def generation_stats(new_rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    combined = load_all_v2_branch_rows(include_prior=True)
    new_stable = [row for row in new_rows if stable_v2_row(row)]
    new_primary = [row for row in new_rows if safe_primary_row(row) and stable_v2_row(row)]
    combined_primary = [row for row in combined if safe_primary_row(row) and stable_v2_row(row)]
    new_groups = {gid: vals for gid, vals in group_rows(new_primary).items() if len(vals) >= 2}
    combined_groups = {gid: vals for gid, vals in group_rows(combined_primary).items() if len(vals) >= 2}
    new_diverse = [gid for gid, vals in new_groups.items() if group_is_behaviorally_diverse_v2(vals)]
    combined_diverse = [gid for gid, vals in combined_groups.items() if group_is_behaviorally_diverse_v2(vals)]
    reward_diverse = [gid for gid, vals in combined_groups.items() if group_is_reward_diverse_v2(vals)]
    pair_stats = candidate_pair_stats(combined_groups)
    tasks_with_pairs = {
        str(vals[0].get("task_id"))
        for vals in combined_groups.values()
        if group_is_reward_diverse_v2(vals)
    }
    return {
        "new_rows": len(new_rows),
        "new_stable_rows": len(new_stable),
        "new_primary_stable_rows": len(new_primary),
        "new_primary_stable_groups": len(new_groups),
        "new_behaviorally_diverse_groups": len(new_diverse),
        "combined_primary_stable_groups": len(combined_groups),
        "combined_behaviorally_diverse_groups": len(combined_diverse),
        "combined_reward_diverse_groups": len(reward_diverse),
        "combined_tasks": len({str(row.get("task_id")) for row in combined_primary}),
        "candidate_heldout_task_ids": len(tasks_with_pairs),
        "candidate_pair_stats": pair_stats,
        "errors": len(errors),
        "stable_rate_new": len(new_stable) / max(len(new_rows), 1),
    }


def verdict_for(stats: dict[str, Any]) -> str:
    if stats["new_rows"] == 0 and stats["errors"] > 0:
        return "BLOCKED"
    if stats["new_rows"] and stats["stable_rate_new"] < 0.50:
        return "UNSTABLE"
    pairs = stats["candidate_pair_stats"]["non_tie_pairs"]
    if (
        stats["combined_primary_stable_groups"] >= 50
        and stats["combined_behaviorally_diverse_groups"] >= 20
        and stats["candidate_heldout_task_ids"] >= 4
        and pairs >= 30
    ):
        return "READY"
    if stats["new_rows"] and stats["combined_behaviorally_diverse_groups"] < 20:
        return "LOW_DIVERSITY"
    if stats["new_rows"]:
        return "PARTIAL"
    return "BLOCKED"


def needs_sampling(group: list[dict[str, Any]]) -> bool:
    if not group:
        return False
    cls = str(group[0].get("task_screening_class") or "")
    if cls in selected_screening_classes():
        return True
    if group_is_behaviorally_diverse_v2(group):
        return True
    if any(not row.get("parse_success", False) for row in group):
        return True
    return False


def add_sampled_rewards(
    *,
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    group: list[dict[str, Any]],
    spec: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
    remaining_budget: int,
) -> int:
    used = 0
    for row in group:
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
            rewards.append(float(score["reward"]))
            used += 1
        if rewards:
            row["sampled_outputs"] = samples
            row["sampled_expected_reward"] = float(sum(rewards) / len(rewards))
    return used


def main() -> int:
    args = parse_args()
    ensure_v2_root()
    started = time.time()
    tasks = selected_tasks(int(args.max_tasks))
    if not tasks:
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "no selected tasks from task screening",
        }
        write_json(REPORT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diverse Branch Generation V2", "", "BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT = BLOCKED", flush=True)
        return 1

    bank = load_direction_bank_payload()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if PARTIAL_PT.exists():
        payload = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False)
        rows = list(payload.get("rows") or [])
        errors = list(payload.get("errors") or [])
    done = {(str(row.get("branch_group_id")), int(row.get("branch_id", -1))) for row in rows}
    sampled_used = sum(len(row.get("sampled_outputs") or []) for row in rows)

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        stop_generation = False
        specs = generation_specs(include_diagnostic_alpha=bool(args.include_diagnostic_alpha))
        for task_index, task in enumerate(tasks):
            for spec in specs:
                if len(rows) >= int(args.max_branch_rows):
                    stop_generation = True
                    break
                k = task_k(task)
                seed = SEED + 7919 * task_index + 101 * int(spec["target_layer"]) + int(float(spec["alpha"]) * 100000)
                deltas = make_diverse_branch_deltas(
                    layer=int(spec["target_layer"]),
                    alpha=float(spec["alpha"]),
                    k=k,
                    seed=seed,
                    direction_bank=bank,
                )
                group_id = (
                    f"v2::{task['task_id']}::{spec['branch_point']}::"
                    f"{spec['alpha_bucket']}::k={k}::class={task.get('screening_class', 'unknown')}"
                )
                group_new: list[dict[str, Any]] = []
                for delta_entry in deltas:
                    if len(rows) >= int(args.max_branch_rows):
                        break
                    branch_id = int(delta_entry["branch_id"])
                    if (group_id, branch_id) in done:
                        continue
                    try:
                        prefix = capture_prefix_features(model, tokenizer, task["prompt"], delta_entry["delta"], spec, device)
                        gen = generate_with_hook_v2(
                            model,
                            tokenizer,
                            task["prompt"],
                            delta_entry["delta"],
                            spec,
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
                            "branch_group_id": group_id,
                            "branch_id": branch_id,
                            "branch_method": "hook_intervention_per_branch",
                            "branch_point": spec["branch_point"],
                            "target_layer": int(spec["target_layer"]),
                            "target_loop": int(spec["target_loop"]),
                            "alpha": float(spec["alpha"]),
                            "alpha_bucket": spec["alpha_bucket"],
                            "safety_envelope": bool(spec["safety_envelope"]),
                            "diagnostic_alpha": bool(spec["diagnostic"]),
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
                            "label_source": "deterministic",
                        }
                        rows.append(row)
                        group_new.append(row)
                        done.add((group_id, branch_id))
                    except Exception as exc:
                        errors.append(
                            {
                                "task_id": task["task_id"],
                                "branch_group_id": group_id,
                                "branch_id": branch_id,
                                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                            }
                        )
                    save_partial(rows, errors, complete=False)
                if group_new:
                    score_old_taps(group_new)
                    if bool(args.sample_borderline) and sampled_used < int(args.sample_budget) and needs_sampling(group_new):
                        try:
                            sampled_used += add_sampled_rewards(
                                model=model,
                                tokenizer=tokenizer,
                                task=task,
                                group=group_new,
                                spec=spec,
                                device=device,
                                max_new_tokens=int(args.max_new_tokens),
                                remaining_budget=max(0, int(args.sample_budget) - sampled_used),
                            )
                        except Exception as exc:
                            errors.append(
                                {
                                    "task_id": task["task_id"],
                                    "branch_group_id": group_id,
                                    "branch_id": "sampled_group",
                                    "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                                }
                            )
                    save_partial(rows, errors, complete=False)
                stats = generation_stats(rows, errors)
                if (
                    stats["combined_primary_stable_groups"] >= 100
                    and stats["combined_behaviorally_diverse_groups"] >= 50
                    and stats["candidate_heldout_task_ids"] >= 8
                    and stats["candidate_pair_stats"]["non_tie_pairs"] >= 60
                ):
                    stop_generation = True
                    break
            if stop_generation:
                break
    finally:
        if extractor is not None:
            extractor.cleanup()

    score_old_taps(rows)
    stats = generation_stats(rows, errors)
    verdict = verdict_for(stats)
    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "errors": errors,
        "stats": stats,
        "selected_task_count": len(tasks),
        "selected_task_ids": [task["task_id"] for task in tasks],
        "counts_by_domain": dict(Counter(str(row.get("domain")) for row in rows)),
        "counts_by_screening_class": dict(Counter(str(row.get("task_screening_class")) for row in rows)),
        "counts_by_branch_point": dict(Counter(str(row.get("branch_point")) for row in rows)),
        "counts_by_alpha_bucket": dict(Counter(str(row.get("alpha_bucket")) for row in rows)),
        "counts_by_delta_family": dict(Counter(str(row.get("delta_family")) for row in rows)),
        "sampled_outputs": sampled_used,
        "include_diagnostic_alpha": bool(args.include_diagnostic_alpha),
        "max_branch_rows": int(args.max_branch_rows),
        "max_new_tokens": int(args.max_new_tokens),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    save_partial(rows, errors, complete=True)
    compact_rows = [compact_branch_row(row) for row in rows]
    compact_payload = {k: v for k, v in payload.items() if k != "rows"} | {"rows": compact_rows}
    write_json(OUT_JSON, compact_payload)
    write_json(REPORT_JSON, compact_payload)

    primary_groups = {
        gid: vals
        for gid, vals in group_rows([row for row in rows if safe_primary_row(row) and stable_v2_row(row)]).items()
        if len(vals) >= 2
    }
    display_rows = [
        {
            "branch_group_id": gid,
            "task_id": vals[0].get("task_id"),
            "domain": vals[0].get("domain"),
            "class": vals[0].get("task_screening_class"),
            "branch_point": vals[0].get("branch_point"),
            "alpha": vals[0].get("alpha_bucket"),
            "branches": len(vals),
            "rewards": sorted({deterministic_reward(row) for row in vals}),
            "answers": sorted({str(row.get("parsed_answer")) for row in vals}),
            "diverse": group_is_behaviorally_diverse_v2(vals),
        }
        for gid, vals in sorted(primary_groups.items())
    ]
    lines = [
        "# Hidden-Origin Diverse Branch Generation V2",
        "",
        f"BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT = {verdict}",
        "",
        f"- new_rows: `{len(rows)}`",
        f"- new_primary_stable_groups: `{stats['new_primary_stable_groups']}`",
        f"- new_behaviorally_diverse_groups: `{stats['new_behaviorally_diverse_groups']}`",
        f"- combined_primary_stable_groups: `{stats['combined_primary_stable_groups']}`",
        f"- combined_behaviorally_diverse_groups: `{stats['combined_behaviorally_diverse_groups']}`",
        f"- non_tie_pairs: `{stats['candidate_pair_stats']['non_tie_pairs']}`",
        f"- tie_rate: `{stats['candidate_pair_stats']['tie_rate']:.3f}`",
        f"- sampled_outputs: `{sampled_used}`",
        f"- errors: `{len(errors)}`",
        "",
        "Alpha `0.02` rows are generated as a separate diagnostic bucket and are not mixed into the primary safe-alpha headline.",
        "",
        "## New Primary Safe Groups",
        "",
    ]
    lines.extend(md_table(display_rows[:160], ["branch_group_id", "task_id", "domain", "class", "branch_point", "alpha", "branches", "rewards", "answers", "diverse"]))
    if errors:
        lines.extend(["", "## Errors", "", *[f"- `{e.get('task_id')}` `{e.get('branch_group_id')}` b{e.get('branch_id')}: {str(e.get('error'))[:240]}" for e in errors[:30]]])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    print(f"Wrote {rel(REPORT_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
