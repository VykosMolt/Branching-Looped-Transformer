"""Generate concise option-defending traces for official MCQ options."""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from code_branch_pilot_lib import REPORT_DIR, RLTT_MODEL_PATH, repo_path, write_json


INPUT_JSON = REPORT_DIR / "reasoning_trace_task_set_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "reasoning_option_traces_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_option_traces_2026-05-17.md"
FINAL_RE = re.compile(r"FINAL ANSWER\s*:\s*([A-E])", re.IGNORECASE)


class FinalAnswerStop(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_len: int, letter: str) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.letter = letter.upper()
        self.pattern = re.compile(rf"FINAL ANSWER\s*:\s*{self.letter}\b", re.IGNORECASE)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        return bool(self.pattern.search(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--max-questions", type=int, default=40)
    parser.add_argument("--max-traces", type=int, default=200)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def prompt_for(task: dict[str, Any], letter: str, option_text: str) -> str:
    return (
        "You are given a multiple-choice question and a selected answer option.\n"
        "Write a concise reasoning trace that supports the selected option.\n"
        "Keep it under 5 short sentences.\n"
        f"End with exactly:\nFINAL ANSWER: {letter}\n\n"
        f"Question:\n{task['question']}\n\n"
        f"Options:\n{options_text(task['options'])}\n\n"
        f"Selected option to justify:\n{letter}. {option_text}\n\n"
        "Remember:\n"
        "- Defend the selected option.\n"
        "- Do not mention whether it is correct.\n"
        "- Do not discuss other options at length.\n"
        f"- End with exactly FINAL ANSWER: {letter}"
    )


def truncate_after_final_answer(text: str) -> str:
    match = re.search(r"(.*?FINAL ANSWER\s*:\s*[A-E])\b", text or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else (text or "").strip()


def parse_final_answer(text: str) -> str:
    match = FINAL_RE.search(text or "")
    return match.group(1).upper() if match else ""


@torch.no_grad()
def generate_one(model: Any, tokenizer: Any, prompt: str, letter: str, args: argparse.Namespace) -> tuple[str, int, bool, bool]:
    device = torch.device(args.device)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length)).to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "stopping_criteria": StoppingCriteriaList([FinalAnswerStop(tokenizer, prompt_len, letter)]),
    }
    out = model.generate(**inputs, **gen_kwargs)
    generated_ids = out[0, prompt_len:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    truncated = truncate_after_final_answer(raw)
    stopped_on_final = bool(FINAL_RE.search(truncated))
    hit_max = int(generated_ids.numel()) >= int(args.max_new_tokens) and not stopped_on_final
    return truncated, int(generated_ids.numel()), hit_max, stopped_on_final


def candidate_uid(task_id: str, dataset: str, letter: str) -> str:
    return f"reasoning_trace::{dataset}::{task_id}::{letter}"


def classify_and_build(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    tournaments: list[dict[str, Any]] = []
    for task_id, rows in by_task.items():
        usable = [row for row in rows if row["usable"]]
        correct = [row for row in usable if row["is_correct"]]
        incorrect = [row for row in usable if not row["is_correct"]]
        if correct and incorrect:
            task = rows[0]
            ordered = sorted(correct + incorrect, key=lambda row: row["option_letter"])
            tournaments.append({
                "task_id": task_id,
                "dataset": task["dataset"],
                "question": task["question"],
                "answer_key": task["answer_key"],
                "n_options": task["n_options"],
                "candidate_uids": [row["candidate_uid"] for row in ordered],
                "labels": ["correct" if row["is_correct"] else "incorrect" for row in ordered],
                "label_counts": dict(Counter("correct" if row["is_correct"] else "incorrect" for row in ordered)),
            })
    total = len(candidates)
    parse_failures = sum(1 for row in candidates if not row["final_answer_present"] or not row["parse_matches_selected_option"])
    hit_max = sum(1 for row in candidates if row["hit_max_new_tokens"])
    usable = sum(1 for row in candidates if row["usable"])
    diagnostics = {
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / max(total, 1),
        "hit_max_new_tokens_count": hit_max,
        "hit_max_new_tokens_rate": hit_max / max(total, 1),
        "usable_candidate_count": usable,
        "usable_candidate_rate": usable / max(total, 1),
        "kept_tournament_count": len(tournaments),
        "per_dataset_kept_count": dict(Counter(row["dataset"] for row in tournaments)),
        "random_top1_baseline": float(mean(1.0 / row["n_options"] for row in tournaments)) if tournaments else 0.0,
    }
    return tournaments, diagnostics


def verdict(diagnostics: dict[str, Any], timed_out: bool, blocked: bool) -> str:
    if blocked:
        return "BLOCKED"
    if timed_out:
        return "TIMEOUT"
    kept = int(diagnostics.get("kept_tournament_count", 0))
    parse_rate = float(diagnostics.get("parse_failure_rate", 1.0))
    hit_rate = float(diagnostics.get("hit_max_new_tokens_rate", 1.0))
    if kept >= 20 and parse_rate <= 0.10 and hit_rate <= 0.10:
        return "READY"
    if parse_rate > 0.10 and kept < 20:
        return "PARSER_CONFUSED"
    if kept >= 10:
        return "PARTIAL"
    if parse_rate > 0.10:
        return "PARSER_CONFUSED"
    return "BLOCKED"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Option Traces",
        "",
        f"REASONING_TRACE_DATA_VERDICT = {payload['reasoning_trace_data_verdict']}",
        "",
        f"- tasks_seen: `{s['tasks_seen']}`",
        f"- generated_traces: `{s['generated_traces']}`",
        f"- kept_tournaments: `{s['kept_tournaments']}`",
        f"- parse_failure_rate: `{s['parse_failure_rate']}`",
        f"- hit_max_new_tokens_rate: `{s['hit_max_new_tokens_rate']}`",
        f"- usable_candidate_rate: `{s['usable_candidate_rate']}`",
        f"- random_top1_baseline_kept: `{s['random_top1_baseline_kept']}`",
        f"- per_dataset_kept_count: `{s['per_dataset_kept_count']}`",
        f"- elapsed_seconds: `{s['elapsed_seconds']}`",
        "",
        "## Kept Tournaments",
        "",
    ]
    for row in payload.get("tournaments", [])[:80]:
        lines.append(f"- `{row['task_id']}` dataset=`{row['dataset']}` labels=`{row['label_counts']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    data = load_json(args.input)
    set_verdict = data.get("reasoning_trace_task_set_verdict", "BLOCKED")
    if set_verdict not in {"READY", "PARTIAL"}:
        payload = {
            "reasoning_trace_data_verdict": "BLOCKED",
            "summary": {
                "REASONING_TRACE_DATA_VERDICT": "BLOCKED",
                "tasks_seen": 0,
                "generated_traces": 0,
                "kept_tournaments": 0,
                "parse_failure_rate": 1.0,
                "hit_max_new_tokens_rate": 1.0,
                "usable_candidate_rate": 0.0,
                "random_top1_baseline_kept": 0.0,
                "per_dataset_kept_count": {},
                "elapsed_seconds": 0.0,
                "blocker": f"task_set_verdict={set_verdict}",
            },
            "tasks": data.get("tasks", []),
            "candidates": [],
            "tournaments": [],
        }
        write_json(out_json, payload)
        write_md(out_md, payload)
        raise SystemExit("REASONING_TRACE_DATA_VERDICT=BLOCKED")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    tasks = list(data.get("tasks", []) or [])[: int(args.max_questions)]
    planned = sum(len(task.get("options", {})) for task in tasks)
    if planned > int(args.max_traces):
        trimmed: list[dict[str, Any]] = []
        total = 0
        for task in tasks:
            n = len(task.get("options", {}))
            if total + n > int(args.max_traces):
                break
            trimmed.append(task)
            total += n
        tasks = trimmed

    device = torch.device(args.device)
    blocked = False
    timed_out = False
    candidates: list[dict[str, Any]] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        for task in tasks:
            for letter, option_text in task["options"].items():
                if time.time() - start > int(args.max_seconds):
                    timed_out = True
                    break
                prompt = prompt_for(task, letter, option_text)
                error = ""
                raw = ""
                token_len = 0
                hit_max = False
                stopped_on_final = False
                try:
                    raw, token_len, hit_max, stopped_on_final = generate_one(model, tokenizer, prompt, letter, args)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                parsed = parse_final_answer(raw)
                final_present = bool(parsed)
                matches = parsed == letter
                generated_nonempty = bool(raw.strip())
                is_correct = letter == task["answer_key"]
                usable = bool(final_present and matches and not hit_max and generated_nonempty and not error)
                candidates.append({
                    "candidate_uid": candidate_uid(task["task_id"], task["dataset"], letter),
                    "task_id": task["task_id"],
                    "dataset": task["dataset"],
                    "question": task["question"],
                    "options": task["options"],
                    "option_letter": letter,
                    "option_text": option_text,
                    "answer_key": task["answer_key"],
                    "is_correct": is_correct,
                    "n_options": int(task.get("n_options", len(task["options"]))),
                    "generated_trace": raw,
                    "final_answer_present": final_present,
                    "parsed_final_answer": parsed,
                    "parse_matches_selected_option": matches,
                    "output_token_length": token_len,
                    "hit_max_new_tokens": hit_max,
                    "stopped_on_final_answer": stopped_on_final,
                    "generation_error": error,
                    "usable": usable,
                })
            if timed_out:
                break
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        blocked = True
        candidates.append({
            "candidate_uid": "model_load_or_generation_error",
            "task_id": "",
            "dataset": "",
            "question": "",
            "option_letter": "",
            "option_text": "",
            "answer_key": "",
            "is_correct": False,
            "n_options": 0,
            "generated_trace": "",
            "final_answer_present": False,
            "parsed_final_answer": "",
            "parse_matches_selected_option": False,
            "output_token_length": 0,
            "hit_max_new_tokens": False,
            "stopped_on_final_answer": False,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "usable": False,
        })
    tournaments, diagnostics = classify_and_build([row for row in candidates if row["candidate_uid"] != "model_load_or_generation_error"])
    v = verdict(diagnostics, timed_out, blocked)
    elapsed = time.time() - start
    summary = {
        "REASONING_TRACE_DATA_VERDICT": v,
        "tasks_seen": len({row["task_id"] for row in candidates if row.get("task_id")}),
        "generated_traces": len([row for row in candidates if row["candidate_uid"] != "model_load_or_generation_error"]),
        "kept_tournaments": len(tournaments),
        "parse_failure_rate": diagnostics.get("parse_failure_rate", 1.0),
        "hit_max_new_tokens_rate": diagnostics.get("hit_max_new_tokens_rate", 1.0),
        "usable_candidate_rate": diagnostics.get("usable_candidate_rate", 0.0),
        "random_top1_baseline_kept": diagnostics.get("random_top1_baseline", 0.0),
        "per_dataset_kept_count": diagnostics.get("per_dataset_kept_count", {}),
        "elapsed_seconds": elapsed,
    }
    payload = {
        "reasoning_trace_data_verdict": v,
        "summary": summary,
        "generation_settings": {
            "model_path": repo_path(args.model_path),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_seconds": int(args.max_seconds),
        },
        "task_set_summary": data.get("summary", {}),
        "diagnostics": diagnostics,
        "tasks": tasks,
        "candidates": candidates,
        "tournaments": tournaments,
        "outputs": {"json": repo_path(out_json), "md": repo_path(out_md)},
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"REASONING_TRACE_DATA_VERDICT = {v}")
    print(f"kept_tournaments = {len(tournaments)}")
    print(f"generated_traces = {summary['generated_traces']}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if v in {"BLOCKED", "TIMEOUT"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
