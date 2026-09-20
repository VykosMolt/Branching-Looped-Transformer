"""Screen MCQ tasks for hidden-origin branch diversity potential."""
from __future__ import annotations

import argparse
import time
import traceback
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_hidden_origin_diversity_v2_common import (
    MAX_NEW_TOKENS,
    SEED,
    TASK_SCREENING_JSON,
    V2_ROOT,
    answer_logit_margin,
    balanced_tasks,
    classify_screening_row,
    clean_generation,
    deterministic_reward,
    ensure_v2_root,
    evaluate_mcq,
    generate_with_hook_v2,
    load_all_branch_rows,
    load_json,
    load_more_candidate_tasks,
    make_diverse_branch_deltas,
    md_table,
    rel,
    selected_screening_classes,
    write_csv,
    write_json,
    write_md,
)


OUT_JSON = TASK_SCREENING_JSON
OUT_MD = V2_ROOT / "task_screening.md"
OUT_CSV = V2_ROOT / "task_screening_rows.csv"
PARTIAL_JSON = V2_ROOT / "task_screening.partial.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--max-selected", type=int, default=96)
    parser.add_argument("--smoke-limit", type=int, default=64)
    parser.add_argument("--smoke-alpha", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_partial(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    write_json(PARTIAL_JSON, {"rows": rows, "errors": errors, "saved_at": time.time()})
    write_csv(OUT_CSV, rows)


def choose_candidates(limit: int) -> list[dict[str, Any]]:
    prior_tasks = {str(row.get("task_id")) for row in load_all_branch_rows()}
    all_tasks = load_more_candidate_tasks()
    fresh = [task for task in all_tasks if task["task_id"] not in prior_tasks]
    reused = [task for task in all_tasks if task["task_id"] in prior_tasks]
    selected = balanced_tasks(fresh, int(limit), seed=SEED)
    if len(selected) < limit:
        selected.extend(balanced_tasks(reused, int(limit) - len(selected), seed=SEED + 17))
    return selected[:limit]


def select_task_rows(rows: list[dict[str, Any]], max_selected: int) -> list[dict[str, Any]]:
    preferred = selected_screening_classes()
    priority = {
        "perturbation_sensitive": 0,
        "baseline_wrong_parseable": 1,
        "baseline_parse_fragile": 2,
        "baseline_correct_low_confidence": 3,
        "baseline_correct_confident": 4,
        "baseline_empty_or_unstable": 5,
    }
    eligible = [row for row in rows if row.get("screening_class") in preferred]
    fallback = [row for row in rows if row.get("screening_class") not in preferred and row.get("screening_class") != "baseline_empty_or_unstable"]
    eligible.sort(key=lambda row: (priority.get(str(row.get("screening_class")), 99), str(row.get("domain")), str(row.get("task_id"))))
    fallback.sort(key=lambda row: (priority.get(str(row.get("screening_class")), 99), str(row.get("domain")), str(row.get("task_id"))))
    combined = eligible + fallback
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        by_domain[str(row.get("domain"))].append(row)
    out: list[dict[str, Any]] = []
    used = set()
    while len(out) < max_selected and any(by_domain.values()):
        progressed = False
        for domain in ("reasoning", "science"):
            vals = by_domain.get(domain) or []
            while vals:
                row = vals.pop(0)
                if row["task_id"] in used:
                    continue
                out.append(row)
                used.add(row["task_id"])
                progressed = True
                break
            if len(out) >= max_selected:
                break
        if not progressed:
            break
    return out[:max_selected]


def main() -> int:
    args = parse_args()
    ensure_v2_root()
    started = time.time()
    candidates = choose_candidates(int(args.max_candidates))
    partial = (load_json(PARTIAL_JSON, {}) or {}) if bool(args.resume) else {}
    rows: list[dict[str, Any]] = list(partial.get("rows") or [])
    errors: list[dict[str, Any]] = list(partial.get("errors") or [])
    done = {str(row.get("task_id")) for row in rows}

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        smoke_spec = {"target_layer": 24, "target_loop": 1, "branch_point": "L24_L1", "alpha": float(args.smoke_alpha), "safety_envelope": False}
        for idx, task in enumerate(candidates):
            if task["task_id"] in done:
                continue
            try:
                clean = clean_generation(model, tokenizer, task["prompt"], device, int(args.max_new_tokens))
                clean_eval = evaluate_mcq(task, clean["output_text"])
                margin = answer_logit_margin(model, tokenizer, task, device)
                smoke_eval: dict[str, Any] | None = None
                smoke_text = ""
                if len(rows) < int(args.smoke_limit):
                    deltas = make_diverse_branch_deltas(
                        layer=24,
                        alpha=float(args.smoke_alpha),
                        k=2,
                        seed=SEED + idx * 997,
                        direction_bank={"directions_by_layer": {}},
                    )
                    smoke = generate_with_hook_v2(
                        model,
                        tokenizer,
                        task["prompt"],
                        deltas[1]["delta"],
                        smoke_spec,
                        device,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                    )
                    smoke_text = smoke["output_text"]
                    smoke_eval = evaluate_mcq(task, smoke_text)
                perturbation_sensitive = False
                if smoke_eval is not None:
                    perturbation_sensitive = (
                        smoke_eval.get("parsed_answer") != clean_eval.get("parsed_answer")
                        or float(smoke_eval.get("reward", 0.0)) != float(clean_eval.get("reward", 0.0))
                    )
                row = {
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "source_dataset": task.get("source_dataset"),
                    "question": task["question"],
                    "options": task["options"],
                    "correct_option": task["correct_option"],
                    "prompt": task["prompt"],
                    "clean_output_text": clean["output_text"],
                    "clean_parsed_answer": clean_eval["parsed_answer"],
                    "clean_correct": bool(clean_eval["correct"]),
                    "clean_reward": float(clean_eval["reward"]),
                    "clean_parse_success": bool(clean_eval["parse_success"]),
                    "clean_output_length": int(clean["token_count"]),
                    "clean_hit_max_tokens": bool(clean["hit_max_tokens"]),
                    "answer_margin": margin.get("answer_margin"),
                    "answer_logits": margin.get("answer_logits"),
                    "smoke_output_text": smoke_text,
                    "smoke_parsed_answer": smoke_eval.get("parsed_answer") if smoke_eval else None,
                    "smoke_correct": bool(smoke_eval.get("correct")) if smoke_eval else None,
                    "smoke_reward": float(smoke_eval.get("reward")) if smoke_eval else None,
                    "perturbation_sensitive": perturbation_sensitive,
                    "unstable": bool(clean_eval.get("empty_output")),
                }
                row["screening_class"] = classify_screening_row(row)
                rows.append(row)
                done.add(task["task_id"])
            except Exception as exc:
                errors.append(
                    {
                        "task_id": task["task_id"],
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                )
            save_partial(rows, errors)
    finally:
        if extractor is not None:
            extractor.cleanup()

    selected_rows = select_task_rows(rows, int(args.max_selected))
    preferred_count = sum(1 for row in selected_rows if row.get("screening_class") in selected_screening_classes())
    if len(selected_rows) >= 32 and preferred_count >= 16:
        verdict = "READY"
    elif len(selected_rows) >= 16:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT": verdict,
        "verdict": verdict,
        "candidate_count": len(candidates),
        "screened_count": len(rows),
        "selected_count": len(selected_rows),
        "selected_preferred_count": preferred_count,
        "selected_task_ids": [row["task_id"] for row in selected_rows],
        "class_counts": dict(Counter(str(row.get("screening_class")) for row in rows)),
        "selected_class_counts": dict(Counter(str(row.get("screening_class")) for row in selected_rows)),
        "domain_counts": dict(Counter(str(row.get("domain")) for row in rows)),
        "selected_domain_counts": dict(Counter(str(row.get("domain")) for row in selected_rows)),
        "rows": rows,
        "selected_rows": selected_rows,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, rows)
    display = [
        {
            "task_id": row["task_id"],
            "domain": row["domain"],
            "class": row["screening_class"],
            "clean": row["clean_parsed_answer"],
            "gold": row["correct_option"],
            "reward": row["clean_reward"],
            "smoke": row.get("smoke_parsed_answer"),
            "sensitive": row["perturbation_sensitive"],
            "margin": row.get("answer_margin"),
        }
        for row in selected_rows[:120]
    ]
    lines = [
        "# Hidden-Origin Task Screening V2",
        "",
        f"BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT = {verdict}",
        "",
        f"- screened_count: `{len(rows)}`",
        f"- selected_count: `{len(selected_rows)}`",
        f"- selected_preferred_count: `{preferred_count}`",
        f"- class_counts: `{payload['class_counts']}`",
        f"- selected_class_counts: `{payload['selected_class_counts']}`",
        f"- errors: `{len(errors)}`",
        "",
        "## Selected Tasks",
        "",
    ]
    lines.extend(md_table(display, ["task_id", "domain", "class", "clean", "gold", "reward", "smoke", "sensitive", "margin"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
