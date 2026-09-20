"""Bounded expansion of same-prefix hidden-origin branch outcome data."""
from __future__ import annotations

import argparse
import time
import traceback
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_hidden_origin_tap_common import (
    HIDDEN_DIM,
    OUT_ROOT,
    SEED,
    branch_key,
    capture_prefix_features,
    ensure_out_root,
    evaluate_mcq,
    generate_with_hook,
    group_is_behaviorally_diverse,
    group_is_reward_diverse,
    group_rows,
    is_safe_alpha,
    load_all_branch_rows,
    load_candidate_tasks,
    md_table,
    rel,
    stable_row,
    write_csv,
    write_json,
    write_md,
)
from src.evaluator.bg_hidden_branching import delta_rms, make_branch_deltas


OUT_PT = OUT_ROOT / "expanded_hidden_origin_branches.pt"
OUT_JSON = OUT_ROOT / "expanded_hidden_origin_branches.json"
OUT_CSV = OUT_ROOT / "expanded_hidden_origin_branches.csv"
OUT_MD = OUT_ROOT / "expansion_report.md"
REPORT_JSON = OUT_ROOT / "expansion_report.json"
PARTIAL_PT = OUT_ROOT / "expanded_hidden_origin_branches.partial.pt"

K = 4
MAX_NEW_TOKENS = 128
SAFE_SPECS = (
    {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.005, "safety_envelope": True},
    {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.010, "safety_envelope": True},
    {"branch_point": "L36_L1", "target_layer": 36, "target_loop": 1, "alpha": 0.010, "safety_envelope": True},
)
DIAG_SPEC = {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.020, "safety_envelope": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tasks", type=int, default=24)
    parser.add_argument("--max-branch-rows", type=int, default=512)
    parser.add_argument("--include-diagnostic-alpha", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"features", "pooled_vectors", "last_token_vectors", "delta"}
    out = {k: v for k, v in row.items() if k not in skip}
    feats = row.get("features")
    if isinstance(feats, torch.Tensor):
        out["features_shape"] = list(feats.shape)
    pooled = row.get("pooled_vectors") or {}
    out["pooled_vector_keys"] = sorted(pooled.keys())
    return out


def save_partial(rows: list[dict[str, Any]], errors: list[dict[str, Any]], complete: bool = False) -> None:
    payload = {"complete": complete, "rows": rows, "errors": errors, "saved_at": time.time()}
    torch.save(payload, PARTIAL_PT)
    write_json(OUT_JSON, {"complete": complete, "rows": [compact_row(r) for r in rows], "errors": errors})
    write_csv(OUT_CSV, [compact_row(r) for r in rows])


def select_tasks(existing_rows: list[dict[str, Any]], max_new_tasks: int) -> list[dict[str, Any]]:
    existing_tasks = {str(row.get("task_id")) for row in existing_rows}
    candidates = [task for task in load_candidate_tasks() if task["task_id"] not in existing_tasks]
    by_domain = defaultdict(list)
    for task in candidates:
        by_domain[task["domain"]].append(task)
    selected: list[dict[str, Any]] = []
    target_each = max(1, int(max_new_tasks) // 2)
    for domain in ("reasoning", "science"):
        selected.extend(by_domain[domain][:target_each])
    if len(selected) < max_new_tasks:
        used = {task["task_id"] for task in selected}
        for task in candidates:
            if task["task_id"] not in used:
                selected.append(task)
                used.add(task["task_id"])
            if len(selected) >= max_new_tasks:
                break
    return selected[:max_new_tasks]


def score_old_taps(rows: list[dict[str, Any]]) -> None:
    try:
        from src.evaluator.bg_controller import BGController

        controller = BGController.from_artifacts(device="cpu")
    except Exception:
        return
    for vals in group_rows(rows).values():
        try:
            ordered = sorted(vals, key=lambda row: int(row["branch_id"]))
            details = controller.rank_candidates(
                [row["features"] for row in ordered],
                domain_hint=str(ordered[0].get("domain") or "reasoning"),
                mode="conservative",
                return_details=True,
            )
            margin = details["margin_sum"]
            wins = details["wins"]
            ranking = list(details["ranking"])
            for idx, row in enumerate(ordered):
                row["tap_margin_sum"] = float(margin[idx].item())
                row["tap_wins"] = int(wins[idx].item())
                row["tap_rank"] = int(ranking.index(idx) + 1)
                row["old_frozen_selected_head"] = details.get("selected_head")
        except Exception:
            for row in vals:
                row.setdefault("tap_margin_sum", 0.0)
                row.setdefault("tap_rank", 0)


def expansion_verdict(existing_rows: list[dict[str, Any]], expanded_rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    by_key = {}
    for row in existing_rows + expanded_rows:
        by_key[branch_key(row)] = row
    all_rows = list(by_key.values())
    safe_stable = [row for row in all_rows if is_safe_alpha(row) and stable_row(row)]
    groups = {gid: vals for gid, vals in group_rows(safe_stable).items() if len(vals) >= 2}
    diverse = [gid for gid, vals in groups.items() if group_is_behaviorally_diverse(vals)]
    reward_diverse = [gid for gid, vals in groups.items() if group_is_reward_diverse(vals)]
    stats = {
        "existing_rows": len(existing_rows),
        "expanded_rows": len(expanded_rows),
        "errors": len(errors),
        "total_safe_stable_groups": len(groups),
        "total_behaviorally_diverse_groups": len(diverse),
        "total_reward_diverse_groups": len(reward_diverse),
        "expanded_tasks": len({row.get("task_id") for row in expanded_rows}),
        "total_tasks": len({row.get("task_id") for row in all_rows}),
    }
    if not expanded_rows and not errors:
        return "SKIPPED", stats
    if len(groups) >= 20 and len(diverse) >= 8 and stats["total_tasks"] >= 12:
        return "READY", stats
    if expanded_rows and len(diverse) == 0:
        return "NO_BEHAVIORAL_DIVERSITY", stats
    if expanded_rows:
        return "PARTIAL", stats
    return "BLOCKED", stats


def main() -> int:
    args = parse_args()
    ensure_out_root()
    started = time.time()
    existing_rows = load_all_branch_rows()
    tasks = select_tasks(existing_rows, int(args.max_new_tasks))
    specs = list(SAFE_SPECS)
    if args.include_diagnostic_alpha:
        specs.append(DIAG_SPEC)
    planned_rows = min(len(tasks) * len(specs) * K, int(args.max_branch_rows))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if PARTIAL_PT.exists():
        payload = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False)
        rows = list(payload.get("rows") or [])
        errors = list(payload.get("errors") or [])

    done = {(str(row.get("branch_group_id")), int(row.get("branch_id", -1))) for row in rows}
    if not tasks or planned_rows <= 0:
        verdict, stats = expansion_verdict(existing_rows, rows, errors)
        report = {
            "BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT": "SKIPPED" if verdict != "BLOCKED" else verdict,
            "verdict": "SKIPPED" if verdict != "BLOCKED" else verdict,
            "reason": "no new reasoning/science MCQ tasks selected",
            "stats": stats,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(REPORT_JSON, report)
        write_md(OUT_MD, ["# Hidden-Origin Branch Dataset Expansion", "", f"BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT = {report['verdict']}"])
        print(f"BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT = {report['verdict']}", flush=True)
        return 0 if report["verdict"] != "BLOCKED" else 1

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = None
    branch_rows_written = len(rows)
    try:
        extractor = BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for task_index, task in enumerate(tasks):
            for spec in specs:
                if branch_rows_written >= int(args.max_branch_rows):
                    break
                seed = SEED + 1009 * task_index + 17 * int(spec["target_layer"]) + int(float(spec["alpha"]) * 100000)
                deltas = make_branch_deltas(
                    HIDDEN_DIM,
                    K,
                    float(spec["alpha"]),
                    seed,
                    allow_diagnostic_alpha=not bool(spec["safety_envelope"]),
                )
                group_id = f"{task['task_id']}::{spec['branch_point']}::alpha={float(spec['alpha']):.3f}::safe={int(bool(spec['safety_envelope']))}"
                group_rows_new: list[dict[str, Any]] = []
                for branch_id, delta in enumerate(deltas):
                    if branch_rows_written >= int(args.max_branch_rows):
                        break
                    key = (group_id, int(branch_id))
                    if key in done:
                        continue
                    try:
                        prefix = capture_prefix_features(model, tokenizer, task["prompt"], delta, spec, device)
                        gen = generate_with_hook(model, tokenizer, task["prompt"], delta, spec, device, MAX_NEW_TOKENS)
                        score = evaluate_mcq(task, gen["output_text"])
                        row = {
                            "task_id": task["task_id"],
                            "domain": task["domain"],
                            "source_dataset": task.get("source_dataset"),
                            "branch_group_id": group_id,
                            "branch_id": int(branch_id),
                            "branch_method": "hook_intervention_per_branch",
                            "branch_point": spec["branch_point"],
                            "target_layer": int(spec["target_layer"]),
                            "target_loop": int(spec["target_loop"]),
                            "alpha": float(spec["alpha"]),
                            "safety_envelope": bool(spec["safety_envelope"]),
                            "delta_type": ["clean_zero", "random_plus", "random_minus", "random_orthogonal"][branch_id]
                            if branch_id < 4
                            else "random_extra",
                            "effective_delta_rms": float(delta_rms(delta)),
                            "delta": delta.detach().cpu().to(torch.float32),
                            "features": prefix["features"],
                            "pooled_vectors": prefix["pooled_vectors"],
                            "last_token_vectors": prefix["last_token_vectors"],
                            "output_text": gen["output_text"],
                            "parsed_answer": score["parsed_answer"],
                            "correct": bool(score["correct"]),
                            "reward": float(score["reward"]),
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
                        rows.append(row)
                        group_rows_new.append(row)
                        done.add(key)
                        branch_rows_written += 1
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
                if group_rows_new:
                    score_old_taps(group_rows_new)
                    save_partial(rows, errors, complete=False)
    finally:
        if extractor is not None:
            extractor.cleanup()

    score_old_taps(rows)
    verdict, stats = expansion_verdict(existing_rows, rows, errors)
    payload = {
        "BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "errors": errors,
        "stats": stats,
        "selected_task_count": len(tasks),
        "selected_task_ids": [task["task_id"] for task in tasks],
        "counts_by_domain": dict(Counter(row.get("domain") for row in rows)),
        "safe_specs": SAFE_SPECS,
        "diagnostic_spec_enabled": bool(args.include_diagnostic_alpha),
        "k": K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    save_partial(rows, errors, complete=True)
    compact_rows = [compact_row(r) for r in rows]
    write_json(OUT_JSON, {k: v for k, v in payload.items() if k != "rows"} | {"rows": compact_rows})
    write_json(REPORT_JSON, {k: v for k, v in payload.items() if k != "rows"} | {"rows": compact_rows})

    groups = group_rows([row for row in rows if is_safe_alpha(row) and stable_row(row)])
    md_rows = [
        {
            "branch_group_id": gid,
            "task_id": vals[0].get("task_id"),
            "domain": vals[0].get("domain"),
            "alpha": vals[0].get("alpha"),
            "branches": len(vals),
            "rewards": sorted({row.get("reward") for row in vals}),
            "behaviorally_diverse": group_is_behaviorally_diverse(vals),
            "reward_diverse": group_is_reward_diverse(vals),
        }
        for gid, vals in sorted(groups.items())
    ]
    lines = [
        "# Hidden-Origin Branch Dataset Expansion",
        "",
        f"BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT = {verdict}",
        "",
        f"- expanded_rows: `{len(rows)}`",
        f"- expanded_tasks: `{stats['expanded_tasks']}`",
        f"- total_safe_stable_groups: `{stats['total_safe_stable_groups']}`",
        f"- total_behaviorally_diverse_groups: `{stats['total_behaviorally_diverse_groups']}`",
        f"- errors: `{len(errors)}`",
        "",
        "## Expanded Groups",
        "",
    ]
    lines.extend(md_table(md_rows[:120], ["branch_group_id", "task_id", "domain", "alpha", "branches", "rewards", "behaviorally_diverse", "reward_diverse"]))
    if errors:
        lines.extend(["", "## Errors", "", *[f"- `{e['task_id']}` b{e['branch_id']}: {str(e['error'])[:240]}" for e in errors[:20]]])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    print(f"Wrote {rel(REPORT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    return 0 if verdict in {"READY", "PARTIAL", "NO_BEHAVIORAL_DIVERSITY", "SKIPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
