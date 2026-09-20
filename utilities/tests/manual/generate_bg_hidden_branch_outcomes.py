"""Continue hidden-origin branches to final MCQ outcomes."""
from __future__ import annotations

import time
import traceback
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_hidden_branch_suite_common import (
    REPORT_ROOT,
    ensure_report_root,
    evaluate_mcq,
    load_json,
    load_task_subset,
    md_table,
    rel,
    write_csv,
    write_json,
    write_md,
)
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, delta_rms


PERSISTENCE_PT = REPORT_ROOT / "hidden_branch_persistence.pt"
OUT_JSON = REPORT_ROOT / "hidden_branch_outcomes.json"
OUT_MD = REPORT_ROOT / "hidden_branch_outcomes.md"
OUT_CSV = REPORT_ROOT / "hidden_branch_outcomes_rows.csv"
PARTIAL_JSON = REPORT_ROOT / "hidden_branch_outcomes.partial.json"
MAX_NEW_TOKENS = 128


def _generate_with_hook(model: Any, tokenizer: Any, prompt: str, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    hook = HiddenDeltaLayerHook(
        model,
        target_layer=int(row["target_layer"]),
        target_loops=[int(row["target_loop"])],
        delta=row["delta"],
        position=-1,
        max_rms_fraction=max(delta_rms(row["delta"]), 0.02),
    )
    started = time.time()
    try:
        hook.apply()
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        hook.remove()
    new_ids = generated[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "output_text": text,
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= MAX_NEW_TOKENS,
        "generation_seconds": round(time.time() - started, 3),
        "hook_diagnostics": hook.diagnostics(),
    }


def outcome_verdict(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("safety_envelope"):
            grouped[row["branch_group_id"]].append(row)
    eligible = [vals for vals in grouped.values() if len(vals) >= 4]
    task_count = len({vals[0]["task_id"] for vals in eligible}) if eligible else 0
    diverse = 0
    reward_diverse = 0
    for vals in eligible:
        outputs = {str(v.get("parsed_answer")) for v in vals}
        rewards = {float(v.get("reward", 0.0)) for v in vals}
        if len(outputs) > 1:
            diverse += 1
        if len(rewards) > 1:
            reward_diverse += 1
    stats = {
        "eligible_groups": len(eligible),
        "eligible_task_count": task_count,
        "behaviorally_diverse_groups": diverse,
        "reward_diverse_groups": reward_diverse,
        "behavioral_diversity_rate": diverse / max(len(eligible), 1),
        "reward_diversity_rate": reward_diverse / max(len(eligible), 1),
    }
    if task_count >= 8 and reward_diverse > 0:
        return "READY", stats
    if eligible and (diverse > 0 or reward_diverse > 0):
        return "PARTIAL", stats
    if eligible:
        return "NO_BEHAVIORAL_DIVERSITY", stats
    return "BLOCKED", stats


def main() -> int:
    ensure_report_root()
    started = time.time()
    if not PERSISTENCE_PT.exists():
        payload = {"BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT": "BLOCKED", "blocker": "missing persistence pt"}
        write_json(OUT_JSON, payload)
        print("BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = BLOCKED")
        return 1
    payload = torch.load(PERSISTENCE_PT, map_location="cpu", weights_only=False)
    records = list(payload.get("records") or [])
    tasks = {row["task_id"]: row for row in load_task_subset()}
    if not records or not tasks:
        out = {"BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT": "BLOCKED", "blocker": "missing records or tasks"}
        write_json(OUT_JSON, out)
        print("BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = BLOCKED")
        return 1

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    previous = {}
    if PARTIAL_JSON.exists():
        previous = load_json(PARTIAL_JSON, {}) or {}
        rows = [row for row in list(previous.get("rows") or []) if row.get("branch_group_id")]
        errors = list(previous.get("errors") or [])
    done = {
        (
            str(row.get("branch_group_id")),
            int(row.get("branch_id", -1)),
        )
        for row in rows
    }
    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for idx, rec in enumerate(records):
            if not rec.get("branch_group_id"):
                continue
            done_key = (str(rec.get("branch_group_id")), int(rec.get("branch_id", -1)))
            if done_key in done:
                continue
            task = tasks.get(rec["task_id"])
            if task is None:
                continue
            try:
                gen = _generate_with_hook(model, tokenizer, task["prompt"], rec, device)
                score = evaluate_mcq(task, gen["output_text"])
                row = {
                    "task_id": rec["task_id"],
                    "domain": rec["domain"],
                    "branch_group_id": rec["branch_group_id"],
                    "branch_id": int(rec["branch_id"]),
                    "branch_method": rec["branch_method"],
                    "branch_point": rec["branch_point"],
                    "target_layer": int(rec["target_layer"]),
                    "target_loop": int(rec["target_loop"]),
                    "alpha": float(rec["alpha"]),
                    "safety_envelope": bool(rec["safety_envelope"]),
                    "delta_type": rec["delta_type"],
                    "effective_delta_rms": float(rec["effective_delta_rms"]),
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
                    "tap_margin_sum": float(rec.get("tap_margin_sum", 0.0)),
                    "tap_rank": int(rec.get("tap_rank", 0)),
                    "generation_seconds": float(gen["generation_seconds"]),
                    "cuda_error": "",
                    "hook_modifications": int(gen["hook_diagnostics"].get("modifications", 0)),
                }
                rows.append(row)
                done.add(done_key)
            except Exception as exc:
                errors.append(
                    {
                        "task_id": rec.get("task_id"),
                        "branch_group_id": rec.get("branch_group_id"),
                        "branch_id": rec.get("branch_id"),
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                )
            write_json(PARTIAL_JSON, {"complete": False, "rows": rows, "errors": errors, "done_count": len(done)})
    finally:
        if extractor is not None:
            extractor.cleanup()

    verdict, stats = outcome_verdict(rows)
    out_payload = {
        "BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "row_count": len(rows),
        "task_count": len({r["task_id"] for r in rows}),
        "branch_group_count": len({r["branch_group_id"] for r in rows}),
        "max_new_tokens": MAX_NEW_TOKENS,
        "deterministic_decode": True,
        "stats": stats,
        "counts_by_domain": dict(Counter(r["domain"] for r in rows)),
        "reward_counts": dict(Counter(str(r["reward"]) for r in rows)),
        "correct_count": sum(1 for r in rows if r["correct"]),
        "parse_success_count": sum(1 for r in rows if r["parse_success"]),
        "rows": rows,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, out_payload)
    write_csv(OUT_CSV, rows)
    group_summaries = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["branch_group_id"]].append(row)
    for gid, vals in grouped.items():
        group_summaries.append(
            {
                "branch_group_id": gid,
                "task_id": vals[0]["task_id"],
                "domain": vals[0]["domain"],
                "alpha": vals[0]["alpha"],
                "safe": vals[0]["safety_envelope"],
                "correct": sum(1 for v in vals if v["correct"]),
                "parsed_answers": sorted({str(v["parsed_answer"]) for v in vals}),
                "rewards": sorted({float(v["reward"]) for v in vals}),
            }
        )
    lines = [
        "# BG Hidden-Origin Branch Outcomes",
        "",
        f"BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- task_count: `{out_payload['task_count']}`",
        f"- branch_group_count: `{out_payload['branch_group_count']}`",
        f"- stats: `{stats}`",
        "",
        "## Group Outcomes",
        "",
    ]
    lines.extend(md_table(group_summaries[:30], ["branch_group_id", "domain", "alpha", "safe", "correct", "parsed_answers", "rewards"]))
    if errors:
        lines.extend(["", "## Errors", "", *[f"- `{e['task_id']}` b{e['branch_id']}: {e['error'][:200]}" for e in errors[:10]]])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL", "NO_BEHAVIORAL_DIVERSITY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
