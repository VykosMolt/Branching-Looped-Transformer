"""Generate a tiny clean GSM8K-only branch tournament benchmark.

This is intentionally not a gate-scale generator. It stops after a small
number of prompts, uses short answer-constrained prompts, and keeps only clean
parseable mixed correct/incorrect branch tournaments.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    DEFAULT_RLTT_PATH,
    PROJECT_ROOT,
    answers_equal,
    candidate_text,
    classify_wrong_math_branch,
    extract_answer,
    extract_gold_answer,
    output_path,
    parse_number,
    resolve_local,
)


OUT_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_micro_2026-05-16.json"
OUT_MD = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_micro_2026-05-16.md"
OUT_LOG = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_micro_2026-05-16.log"
FINAL_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.json"
FINAL_MD = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.md"
MAX_ATTEMPTS_TOTAL = 180
FINAL_RE = re.compile(r"FINAL\s+ANSWER\s*:\s*([^\r\n]+)", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class PromptMode:
    name: str
    max_new_tokens: int
    temperatures: Tuple[float, ...]
    template: str


PROMPT_MODES = (
    PromptMode(
        name="answer_only",
        max_new_tokens=48,
        temperatures=(0.2, 0.5, 0.8, 1.0),
        template=(
            "Solve privately. Do not show reasoning.\n"
            "Output exactly one line:\n"
            "FINAL ANSWER: <number>\n\n"
            "Problem:\n{problem}"
        ),
    ),
    PromptMode(
        name="final_first",
        max_new_tokens=96,
        temperatures=(0.2, 0.5, 0.8, 1.0),
        template=(
            "Give the final answer first.\n"
            "First line must be:\n"
            "FINAL ANSWER: <number>\n\n"
            "You may add at most two short check lines after that, but no long reasoning.\n\n"
            "Problem:\n{problem}"
        ),
    ),
    PromptMode(
        name="compact_reasoning",
        max_new_tokens=192,
        temperatures=(0.3, 0.6, 0.9, 1.1),
        template=(
            "Solve briefly using at most four short equations.\n"
            "End with exactly:\n"
            "FINAL ANSWER: <number>\n\n"
            "Do not write anything after the FINAL ANSWER line.\n\n"
            "Problem:\n{problem}"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_RLTT_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_RLTT_PATH)
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--target-clean-tournaments", type=int, default=8)
    parser.add_argument("--attempts-per-mode", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--skip-prompt-token-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--log-file", default=str(OUT_LOG))
    return parser.parse_args()


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_logging(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    return handle


def git_commit() -> Optional[str]:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def snippet(text: object, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class FinalAnswerStopping(StoppingCriteria):
    """Stop after a numeric FINAL ANSWER line has been terminated."""

    def __init__(self, tokenizer, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **_kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        if generated.numel() <= 0:
            return False
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        match = FINAL_RE.search(text)
        if not match:
            return False
        answer = match.group(1)
        if NUMBER_RE.search(answer) is None:
            return False
        answer_end = match.end()
        if len(text) <= answer_end:
            return False
        return bool(re.search(r"\s", text[answer_end - 1 : answer_end + 2]))


def final_answer_parse(completion: str) -> Tuple[bool, bool, Optional[str], str]:
    match = FINAL_RE.search(completion)
    if match:
        line = match.group(1).strip()
        number_match = NUMBER_RE.findall(line)
        if number_match:
            return True, True, number_match[-1], "final_answer_marker"
        return True, False, None, "failed"
    fallback = extract_answer(completion)
    return False, fallback is not None, fallback, "gsm8k_fallback" if fallback is not None else "failed"


def numeric_near_miss(extracted: Optional[str], gold: str) -> bool:
    if extracted is None:
        return False
    pred = parse_number(extracted)
    target = parse_number(gold)
    if pred is None or target is None:
        return False
    diff = abs(pred - target)
    if target == 0:
        return diff <= 1
    if abs(target) > 1:
        return diff / abs(target) <= 0.10
    return diff <= 1


def classify_wrong_branch(question: str, gold_solution: str, gold_answer: str, attempt: Dict[str, object]) -> Dict[str, object]:
    try:
        return classify_wrong_math_branch(
            prompt=question,
            reference_solution=gold_solution,
            gold_answer=gold_answer,
            candidate_text=str(attempt["completion_text"]),
            extracted_answer=attempt.get("extracted_answer"),
            truncated=bool(attempt["hit_max_new_tokens"]),
        )
    except Exception as exc:
        near = numeric_near_miss(attempt.get("extracted_answer"), gold_answer)
        return {
            "classification": "near_miss" if near else "nonsense",
            "reason": "numeric_fallback" if near else f"classifier_failed:{type(exc).__name__}",
            "classifier_fallback": True,
            "structurally_math_like": True,
        }


def temperatures_for_mode(mode: PromptMode, attempts_per_mode: int) -> List[float]:
    temps = list(mode.temperatures)
    if attempts_per_mode <= len(temps):
        return temps[:attempts_per_mode]
    out = []
    for idx in range(attempts_per_mode):
        out.append(temps[idx % len(temps)])
    return out


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    mode: PromptMode,
    temperature: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[str, int, bool]:
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    ).to(device)
    prompt_len = int(enc["input_ids"].shape[-1])
    stopping = StoppingCriteriaList([FinalAnswerStopping(tokenizer, prompt_len)])
    outputs = model.generate(
        **enc,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(mode.max_new_tokens),
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping,
        use_cache=True,
    )
    gen = outputs[0, prompt_len:].detach().cpu().tolist()
    eos_id = tokenizer.eos_token_id
    eos_pos = gen.index(eos_id) if eos_id is not None and eos_id in gen else None
    token_count = (eos_pos + 1) if eos_pos is not None else len(gen)
    hit_max = eos_pos is None and len(gen) >= int(mode.max_new_tokens)
    completion = tokenizer.decode(gen, skip_special_tokens=True).strip()
    del outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return completion, int(token_count), bool(hit_max)


def attempt_row(
    *,
    problem_id: int,
    question: str,
    gold_answer: str,
    prompt_text: str,
    mode: PromptMode,
    attempt_index: int,
    temperature: float,
    completion: str,
    token_count: int,
    hit_max: bool,
) -> Dict[str, object]:
    final_present, parseable, extracted, parser_method = final_answer_parse(completion)
    is_correct = bool(parseable and extracted is not None and answers_equal(extracted, gold_answer))
    return {
        "source": "gsm8k",
        "problem_id": int(problem_id),
        "prompt_text": prompt_text,
        "prompt_mode": mode.name,
        "attempt_index": int(attempt_index),
        "temperature": float(temperature),
        "max_new_tokens": int(mode.max_new_tokens),
        "output_token_length": int(token_count),
        "hit_max_new_tokens": bool(hit_max),
        "final_answer_present": bool(final_present),
        "parseable": bool(parseable),
        "parser_method": parser_method,
        "extracted_answer": extracted,
        "gold_answer": gold_answer,
        "is_correct": bool(is_correct),
        "completion_text": completion,
        "candidate_text": f"Problem: {question}\n\nSolution:{completion}",
    }


def branch_is_clean(row: Dict[str, object]) -> bool:
    return (
        bool(row["parseable"])
        and bool(row["final_answer_present"])
        and not bool(row["hit_max_new_tokens"])
    )


def clean_verdict(tournaments: Sequence[Dict[str, object]]) -> str:
    kept = [a for t in tournaments for a in t["attempts"]]
    incorrect = [a for a in kept if not bool(a["is_correct"])]
    if any(bool(a["hit_max_new_tokens"]) for a in incorrect):
        return "STILL_TRUNCATED"
    if any(not bool(a["parseable"]) for a in kept):
        return "PARSER_CONFUSED"
    if len(tournaments) < 5:
        return "TOO_FEW_TOURNAMENTS"
    if not kept or any(not bool(a["final_answer_present"]) for a in kept):
        return "PARSER_CONFUSED"
    return "CLEAN"


def summarize_attempts(all_attempts: Sequence[Dict[str, object]], tournaments: Sequence[Dict[str, object]]) -> Dict[str, object]:
    clean_attempts = [a for a in all_attempts if branch_is_clean(a)]
    kept_attempts = [a for t in tournaments for a in t["attempts"]]
    wrong_kept = [a for a in kept_attempts if not bool(a["is_correct"])]
    near = sum(1 for a in wrong_kept if a.get("wrong_classification") == "near_miss")
    nonsense = sum(1 for a in wrong_kept if a.get("wrong_classification") == "nonsense")
    return {
        "attempts_generated": len(all_attempts),
        "clean_attempts": len(clean_attempts),
        "correct_clean_attempts": sum(1 for a in clean_attempts if a["is_correct"]),
        "incorrect_clean_attempts": sum(1 for a in clean_attempts if not a["is_correct"]),
        "clean_tournaments_kept": len(tournaments),
        "kept_branches": len(kept_attempts),
        "kept_correct_branches": sum(1 for a in kept_attempts if a["is_correct"]),
        "kept_incorrect_branches": len(wrong_kept),
        "kept_incorrect_hit_max_new_tokens": sum(1 for a in wrong_kept if a["hit_max_new_tokens"]),
        "kept_parseable_rate": (
            sum(1 for a in kept_attempts if a["parseable"]) / max(len(kept_attempts), 1)
        ),
        "kept_final_answer_present_rate": (
            sum(1 for a in kept_attempts if a["final_answer_present"]) / max(len(kept_attempts), 1)
        ),
        "near_miss_incorrect_branches": near,
        "nonsense_incorrect_branches": nonsense,
        "near_miss_fraction": near / max(len(wrong_kept), 1),
        "temperature_distribution_kept": dict(Counter(str(a["temperature"]) for a in kept_attempts)),
        "prompt_mode_distribution_kept": dict(Counter(str(t["prompt_mode"]) for t in tournaments)),
    }


def write_micro_md(path: Path, result: Dict[str, object]) -> None:
    summary = result["summary"]
    lines = [
        "# Clean GSM8K Extreme Micro Generation",
        "",
        f"CLEAN_GSM8K_VERDICT = {result['clean_gsm8k_verdict']}",
        "",
        "## Summary",
        "",
        f"- prompts_processed: `{summary['prompts_processed']}`",
        f"- prompts_skipped_long: `{summary['prompts_skipped_long']}`",
        f"- attempts_generated: `{summary['attempts_generated']}`",
        f"- clean_attempts: `{summary['clean_attempts']}`",
        f"- correct_clean_attempts: `{summary['correct_clean_attempts']}`",
        f"- incorrect_clean_attempts: `{summary['incorrect_clean_attempts']}`",
        f"- clean_tournaments_kept: `{summary['clean_tournaments_kept']}`",
        "- tournament source: `GSM8K only`",
        f"- prompt_mode_distribution_kept: `{summary['prompt_mode_distribution_kept']}`",
        f"- temperature_distribution_kept: `{summary['temperature_distribution_kept']}`",
        f"- near_miss_incorrect_branches: `{summary['near_miss_incorrect_branches']}`",
        f"- nonsense_incorrect_branches: `{summary['nonsense_incorrect_branches']}`",
        f"- near_miss_fraction: `{summary['near_miss_fraction']:.3f}`",
        "",
        "## Kept Tournaments",
        "",
    ]
    for t in result["tournaments"][:8]:
        lines.extend([
            f"### Tournament {t['tournament_id']} problem={t['problem_id']} mode={t['prompt_mode']}",
            "",
            f"Prompt: {snippet(t['question'], 300)}",
            "",
            f"Gold answer: `{t['gold_answer']}`",
            "",
        ])
        for a in t["attempts"]:
            cls = a.get("wrong_classification", "correct") if not a["is_correct"] else "correct"
            lines.extend([
                (
                    f"- branch={a['attempt_index']} temp={a['temperature']} correct={a['is_correct']} "
                    f"answer=`{a['extracted_answer']}` class=`{cls}` tokens={a['output_token_length']}"
                ),
                "```text",
                snippet(a["completion_text"], 500),
                "```",
                "",
            ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_provisional_final(result: Dict[str, object]) -> None:
    verdict = result["clean_gsm8k_verdict"]
    recommended = "generation_tuning_or_code_pivot" if verdict != "CLEAN" else "generation_clean_feature_capture_pending"
    payload = {
        "clean_gsm8k_verdict": verdict,
        "clean_transfer_verdict": "NOT_RUN",
        "recommended_next": recommended,
        "generation_summary": result["summary"],
        "feature_capture": None,
        "transfer": None,
    }
    FINAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    FINAL_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Clean GSM8K Extreme Transfer",
        "",
        f"CLEAN_GSM8K_VERDICT = {verdict}",
        "CLEAN_TRANSFER_VERDICT = NOT_RUN",
        f"RECOMMENDED_NEXT = {recommended}",
        "",
        "## Generation Summary",
        "",
        f"- clean_tournaments_kept: `{result['summary']['clean_tournaments_kept']}`",
        f"- attempts_generated: `{result['summary']['attempts_generated']}`",
        "",
        "Feature capture and transfer evaluation are pending or blocked by generation verdict.",
        "",
    ]
    FINAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_handle = setup_logging(output_path(args.log_file))
    if args.generation_batch_size != 1:
        raise SystemExit("This micro probe requires --generation-batch-size 1")
    if args.n_prompts > 15:
        raise SystemExit("--n-prompts hard cap is 15")
    if args.n_prompts * len(PROMPT_MODES) * args.attempts_per_mode > MAX_ATTEMPTS_TOTAL:
        raise SystemExit("requested attempts exceed hard cap of 180")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_path = resolve_local(args.model_path)
    tokenizer_path = resolve_local(args.tokenizer_path)

    print(f"loading tokenizer: {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading Ouro-RLTT model: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0

    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)

    prompts_processed = 0
    prompts_skipped_long = 0
    attempts_generated = 0
    tournaments: List[Dict[str, object]] = []
    all_attempts: List[Dict[str, object]] = []
    problem_diagnostics: List[Dict[str, object]] = []

    for dataset_index in indices:
        if prompts_processed >= args.n_prompts:
            break
        ex = ds[dataset_index]
        question = str(ex["question"])
        gold_solution = str(ex["answer"])
        gold_answer = extract_gold_answer("gsm8k", gold_solution)
        if gold_answer is None:
            continue
        prompt_lengths = [
            len(tokenizer(mode.template.format(problem=question), add_special_tokens=False)["input_ids"])
            for mode in PROMPT_MODES
        ]
        if max(prompt_lengths) > args.skip_prompt_token_length:
            prompts_skipped_long += 1
            continue

        prompts_processed += 1
        problem_attempts: List[Dict[str, object]] = []
        kept = None
        print(
            f"prompt {prompts_processed}/{args.n_prompts} dataset_index={dataset_index} gold={gold_answer}",
            flush=True,
        )
        for mode in PROMPT_MODES:
            if kept is not None:
                break
            mode_attempts: List[Dict[str, object]] = []
            prompt_text = mode.template.format(problem=question)
            for local_idx, temp in enumerate(temperatures_for_mode(mode, args.attempts_per_mode)):
                if attempts_generated >= MAX_ATTEMPTS_TOTAL:
                    break
                completion, token_count, hit_max = generate_one(
                    model, tokenizer, prompt_text, mode, temp, args, device)
                row = attempt_row(
                    problem_id=dataset_index,
                    question=question,
                    gold_answer=gold_answer,
                    prompt_text=prompt_text,
                    mode=mode,
                    attempt_index=len(problem_attempts),
                    temperature=temp,
                    completion=completion,
                    token_count=token_count,
                    hit_max=hit_max,
                )
                attempts_generated += 1
                mode_attempts.append(row)
                problem_attempts.append(row)
                all_attempts.append(row)
                print(
                    f"  {mode.name} temp={temp} answer={row['extracted_answer']} "
                    f"correct={row['is_correct']} clean={branch_is_clean(row)} tokens={token_count}",
                    flush=True,
                )
            clean = [a for a in mode_attempts if branch_is_clean(a)]
            correct = [a for a in clean if bool(a["is_correct"])]
            incorrect = [a for a in clean if not bool(a["is_correct"])]
            if len(clean) >= 2 and correct and incorrect:
                for a in incorrect:
                    cls = classify_wrong_branch(question, gold_solution, gold_answer, a)
                    a["wrong_classification"] = cls["classification"]
                    a["classifier_reason"] = cls["reason"]
                    a["classifier_fallback"] = bool(cls.get("classifier_fallback", False))
                for a in correct:
                    a["wrong_classification"] = None
                    a["classifier_reason"] = None
                    a["classifier_fallback"] = False
                kept = {
                    "tournament_id": len(tournaments),
                    "source": "gsm8k",
                    "problem_id": int(dataset_index),
                    "dataset_index": int(dataset_index),
                    "question": question,
                    "gold_solution": gold_solution,
                    "gold_answer": gold_answer,
                    "prompt_mode": mode.name,
                    "attempts": clean,
                    "correct_indices": [i for i, a in enumerate(clean) if a["is_correct"]],
                    "incorrect_indices": [i for i, a in enumerate(clean) if not a["is_correct"]],
                    "unclean_attempts": [a for a in problem_attempts if not branch_is_clean(a)],
                    "all_problem_attempts": problem_attempts,
                }
                tournaments.append(kept)
                print(f"  kept clean tournament {kept['tournament_id']} via {mode.name}", flush=True)
                break
        problem_diagnostics.append({
            "dataset_index": int(dataset_index),
            "question": question,
            "gold_answer": gold_answer,
            "attempt_count": len(problem_attempts),
            "formed_tournament": kept is not None,
        })
        if len(tournaments) >= args.target_clean_tournaments:
            break
        if attempts_generated >= MAX_ATTEMPTS_TOTAL:
            break

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    verdict = clean_verdict(tournaments)
    summary = summarize_attempts(all_attempts, tournaments)
    summary.update({
        "prompts_processed": prompts_processed,
        "prompts_skipped_long": prompts_skipped_long,
        "target_clean_tournaments": int(args.target_clean_tournaments),
        "max_prompts": int(args.n_prompts),
        "hard_attempt_cap": MAX_ATTEMPTS_TOTAL,
    })
    result = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "model_path_resolved": model_path,
            "tokenizer_path_resolved": tokenizer_path,
            "args": vars(args),
            "source": "gsm8k",
        },
        "clean_gsm8k_verdict": verdict,
        "summary": summary,
        "tournaments": tournaments,
        "all_attempts": all_attempts,
        "problem_diagnostics": problem_diagnostics,
    }

    out_json = output_path(args.output_json)
    out_md = output_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    write_micro_md(out_md, result)
    write_provisional_final(result)
    print(f"CLEAN_GSM8K_VERDICT = {verdict}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
