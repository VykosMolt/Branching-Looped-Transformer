"""Generate a small answer-key-labeled reasoning branch pilot."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[4]
os.environ["HF_HOME"] = str(PROJECT_ROOT_FALLBACK / "artifacts" / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(PROJECT_ROOT_FALLBACK / "artifacts" / "hf_cache" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT_FALLBACK / "artifacts" / "hf_cache" / "datasets")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, RLTT_MODEL_PATH, repo_path, write_json

from datasets import load_dataset  # noqa: E402


OUTPUT_JSON = REPORT_DIR / "reasoning_branch_pilot_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_branch_pilot_2026-05-17.md"
FINAL_RE = re.compile(r"FINAL ANSWER\s*:\s*([A-E])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-tasks", type=int, default=40)
    p.add_argument("--min-tasks", type=int, default=20)
    p.add_argument("--candidates-per-task", type=int, default=4)
    p.add_argument("--target-mixed-tournaments", type=int, default=25)
    p.add_argument("--output", default=str(OUTPUT_JSON))
    p.add_argument("--output-md", default=str(OUTPUT_MD))
    p.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=768)
    return p.parse_args()


def normalize_options(labels: list[str], texts: list[str]) -> tuple[list[str], dict[str, str]]:
    out_labels = []
    out = {}
    for i, text in enumerate(texts):
        label = str(labels[i] if i < len(labels) else chr(ord("A") + i)).strip().upper()
        if label not in {"A", "B", "C", "D", "E"}:
            label = chr(ord("A") + len(out_labels))
        out_labels.append(label)
        out[label] = str(text)
    return out_labels, out


def load_arc(max_n: int) -> list[dict[str, Any]]:
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
    rows = []
    for row in ds:
        choices = row["choices"]
        labels, options = normalize_options(list(choices["label"]), list(choices["text"]))
        answer = str(row.get("answerKey", "")).strip().upper()
        if answer in labels:
            rows.append({"dataset": "ai2_arc_challenge", "task_id": f"ARC-Challenge/{len(rows)}", "question": row["question"], "options": options, "answer": answer})
        if len(rows) >= max_n:
            break
    return rows


def load_openbook(max_n: int) -> list[dict[str, Any]]:
    ds = load_dataset("openbookqa", "main", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
    rows = []
    for row in ds:
        choices = row["choices"]
        labels, options = normalize_options(list(choices["label"]), list(choices["text"]))
        answer = str(row.get("answerKey", "")).strip().upper()
        if answer in labels:
            rows.append({"dataset": "openbookqa", "task_id": f"OpenBookQA/{len(rows)}", "question": row["question_stem"], "options": options, "answer": answer})
        if len(rows) >= max_n:
            break
    return rows


def load_commonsense(max_n: int) -> list[dict[str, Any]]:
    ds = load_dataset("commonsense_qa", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
    rows = []
    for row in ds:
        choices = row["choices"]
        labels, options = normalize_options(list(choices["label"]), list(choices["text"]))
        answer = str(row.get("answerKey", "")).strip().upper()
        if answer in labels:
            rows.append({"dataset": "commonsense_qa", "task_id": f"CommonsenseQA/{len(rows)}", "question": row["question"], "options": options, "answer": answer})
        if len(rows) >= max_n:
            break
    return rows


def load_tasks(max_tasks: int) -> tuple[list[dict[str, Any]], list[str]]:
    errors = []
    tasks = []
    loaders = (load_arc, load_openbook, load_commonsense)
    per_source = max(10, max_tasks // 2)
    for loader in loaders:
        try:
            tasks.extend(loader(per_source))
        except Exception as exc:
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
        if len(tasks) >= max_tasks:
            break
    return tasks[:max_tasks], errors


def prompt_for(task: dict[str, Any]) -> str:
    opts = "\n".join(f"{label}. {text}" for label, text in task["options"].items())
    return (
        "You must answer by writing exactly one final option letter.\n"
        "Use this format:\n"
        "FINAL ANSWER: <A/B/C/D/E>\n\n"
        "Do not write anything after the final answer line.\n\n"
        f"Question:\n{task['question']}\n\n"
        f"Options:\n{opts}\n"
    )


def parse_answer(text: str, valid: set[str]) -> str:
    m = FINAL_RE.search(text or "")
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    found = [x.upper() for x in re.findall(r"\b([A-E])\b", text or "", flags=re.IGNORECASE) if x.upper() in valid]
    return found[-1] if found else ""


@torch.no_grad()
def generate_one(model: Any, tokenizer: Any, prompt: str, mode: dict[str, Any], args: argparse.Namespace) -> str:
    device = torch.device(args.device)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_length)).to(device)
    gen_kwargs = {
        "max_new_tokens": int(mode["max_new_tokens"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if float(mode["temperature"]) <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": float(mode["temperature"]), "top_p": 0.95})
    out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def classify_task(candidates: list[dict[str, Any]]) -> str:
    labels = Counter(row["label"] for row in candidates)
    parseable = labels["correct"] + labels["incorrect"]
    if labels["correct"] and labels["incorrect"]:
        return "mixed"
    if parseable and labels["correct"] == parseable:
        return "all_correct"
    if parseable and labels["incorrect"] == parseable:
        return "all_incorrect"
    return "unparseable_only"


def verdict(total_candidates: int, mixed: int, correct: int, unparseable: int) -> str:
    if total_candidates == 0:
        return "BLOCKED"
    if unparseable / total_candidates > 0.20:
        return "PARSER_CONFUSED"
    if mixed >= 12:
        return "READY"
    if correct / total_candidates > 0.75:
        return "TOO_EASY"
    if correct / total_candidates < 0.10:
        return "TOO_HARD"
    return "BLOCKED"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Branch Pilot",
        "",
        f"REASONING_BRANCH_DATA_VERDICT = {payload['reasoning_branch_data_verdict']}",
        "",
        f"- tasks_seen: `{s['tasks_seen']}`",
        f"- mixed_tournaments: `{s['mixed_tournaments']}`",
        f"- candidates: `{s['candidates_total']}`",
        f"- label_counts: `{s['label_counts']}`",
        f"- parser_unparseable_rate: `{s['unparseable_rate']}`",
        f"- dataset_counts: `{s['dataset_counts']}`",
        f"- dataset_load_errors: `{payload.get('dataset_load_errors', [])}`",
        "",
        "## Mixed Tournaments",
        "",
    ]
    for row in payload["tournaments"][:40]:
        lines.append(f"- `{row['task_id']}` dataset=`{row['dataset']}` labels=`{row['label_counts']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    tasks, load_errors = load_tasks(int(args.max_tasks))
    if len(tasks) < int(args.min_tasks):
        payload = {
            "reasoning_branch_data_verdict": "BLOCKED",
            "summary": {"tasks_seen": len(tasks), "candidates_total": 0, "mixed_tournaments": 0, "label_counts": {}, "unparseable_rate": 0, "dataset_counts": {}},
            "dataset_load_errors": load_errors,
            "tasks": tasks,
            "candidates": [],
            "tournaments": [],
        }
        write_json(out_json, payload)
        write_md(out_md, payload)
        raise SystemExit("REASONING_BRANCH_DATA_VERDICT=BLOCKED")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True)
    model.to(device)
    model.eval()
    modes = [
        {"mode": "deterministic", "temperature": 0.0, "max_new_tokens": 64},
        {"mode": "sample_0p3", "temperature": 0.3, "max_new_tokens": 64},
        {"mode": "sample_0p7", "temperature": 0.7, "max_new_tokens": 96},
        {"mode": "sample_1p0", "temperature": 1.0, "max_new_tokens": 96},
    ][: int(args.candidates_per_task)]
    candidates = []
    tournaments = []
    for task in tasks:
        valid = set(task["options"].keys())
        prompt = prompt_for(task)
        task_candidates = []
        for idx, mode in enumerate(modes):
            raw = generate_one(model, tokenizer, prompt, mode, args)
            parsed = parse_answer(raw, valid)
            label = "unparseable" if not parsed else ("correct" if parsed == task["answer"] else "incorrect")
            row = {
                "candidate_uid": f"reasoning::{task['task_id']}::{idx}",
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "question": task["question"],
                "options": task["options"],
                "answer_key": task["answer"],
                "mode": mode["mode"],
                "raw_text": raw,
                "parsed_answer": parsed,
                "label": label,
            }
            candidates.append(row)
            task_candidates.append(row)
        cls = classify_task(task_candidates)
        if cls == "mixed":
            tournaments.append({
                **task,
                "prompt": prompt,
                "candidate_uids": [row["candidate_uid"] for row in task_candidates if row["label"] in {"correct", "incorrect"}],
                "labels": [row["label"] for row in task_candidates if row["label"] in {"correct", "incorrect"}],
                "label_counts": dict(Counter(row["label"] for row in task_candidates)),
            })
        if len(tournaments) >= int(args.target_mixed_tournaments) and len(candidates) >= int(args.min_tasks) * len(modes):
            break
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    labels = Counter(row["label"] for row in candidates)
    v = verdict(len(candidates), len(tournaments), labels["correct"], labels["unparseable"])
    summary = {
        "REASONING_BRANCH_DATA_VERDICT": v,
        "tasks_seen": len({row["task_id"] for row in candidates}),
        "mixed_tournaments": len(tournaments),
        "candidates_total": len(candidates),
        "label_counts": dict(labels),
        "unparseable_rate": labels["unparseable"] / max(len(candidates), 1),
        "dataset_counts": dict(Counter(row["dataset"] for row in candidates)),
    }
    payload = {
        "reasoning_branch_data_verdict": v,
        "summary": summary,
        "dataset_load_errors": load_errors,
        "tasks": tasks,
        "candidates": candidates,
        "tournaments": tournaments,
        "outputs": {"json": repo_path(out_json), "md": repo_path(out_md)},
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"REASONING_BRANCH_DATA_VERDICT = {v}")
    print(f"mixed_tournaments = {len(tournaments)}")
    print(f"candidates_total = {len(candidates)}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if v == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
