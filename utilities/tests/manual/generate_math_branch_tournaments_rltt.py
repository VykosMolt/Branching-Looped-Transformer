"""Generate verifier-clean math branch tournaments with Ouro-RLTT.

Keeps only prompts with at least one generated correct branch and one generated
incorrect branch by an exact-answer verifier. A smoke-only flag can inject the
reference solution as a correct branch to exercise downstream code; that flag is
off by default and should not be used for gate data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shlex
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RLTT_PATH,
    DEFAULT_TOKENIZER_PATH,
    PROJECT_ROOT,
    answers_equal,
    candidate_text,
    classify_wrong_math_branch,
    extract_answer,
    gold_solution_text,
    load_math_problems,
    output_path,
    problem_prompt,
    resolve_local,
    tournament_difficulty_classification,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate-prep", action="store_true")
    p.add_argument("--model-path", default=DEFAULT_RLTT_PATH)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--source", choices=("gsm8k", "math", "mixed"), default="gsm8k")
    p.add_argument("--sources", nargs="+", choices=("gsm8k", "math"), default=["gsm8k", "math"])
    p.add_argument("--max-problems", type=int, default=100)
    p.add_argument("--prompts-per-source-budget", type=int, default=50)
    p.add_argument("--budget-strata", nargs="+", type=int, default=[256, 512, 1024])
    p.add_argument("--attempts-per-problem", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--generation-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-math-level", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report-every", type=int, default=10)
    p.add_argument("--allow-reference-correct-for-smoke", action="store_true",
                   help="Smoke-test fallback only. Do not use for gate data.")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    args = p.parse_args()
    if args.output_json is None:
        args.output_json = (
            f"{DEFAULT_OUTPUT_DIR}/math_gate_prep_generation.json"
            if args.gate_prep else f"{DEFAULT_OUTPUT_DIR}/math_branch_tournaments_rltt.json"
        )
    if args.output_md is None and args.gate_prep:
        args.output_md = f"{DEFAULT_OUTPUT_DIR}/math_gate_prep_generation.md"
    return args


@torch.no_grad()
def generate_attempts(model, tokenizer, prompt: str, args: argparse.Namespace,
                      device: torch.device, max_new_tokens: Optional[int] = None) -> List[Dict[str, object]]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    ).to(device)
    prompt_len = enc["input_ids"].shape[-1]
    completions = []
    batch_size = max(1, int(args.generation_batch_size))
    remaining = int(args.attempts_per_problem)
    budget = int(max_new_tokens if max_new_tokens is not None else args.max_new_tokens)

    def _decode_outputs(outputs: torch.Tensor, retry_from_oom: bool) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for row in outputs:
            gen = row[prompt_len:]
            gen_ids = gen.detach().cpu().tolist()
            eos_id = tokenizer.eos_token_id
            eos_pos = gen_ids.index(eos_id) if eos_id is not None and eos_id in gen_ids else None
            token_count = (eos_pos + 1) if eos_pos is not None else len(gen_ids)
            rows.append({
                "completion": tokenizer.decode(gen, skip_special_tokens=True).strip(),
                "output_tokens": int(token_count),
                "truncated": bool(eos_pos is None and len(gen_ids) >= budget),
                "oom_retry": bool(retry_from_oom),
            })
        return rows

    def _generate_batch(n: int, retry_from_oom: bool = False) -> List[Dict[str, object]]:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        outputs = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=budget,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        rows = _decode_outputs(outputs, retry_from_oom)
        del outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    while remaining > 0:
        n = min(batch_size, remaining)
        try:
            completions.extend(_generate_batch(n))
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            if n <= 1:
                raise
            print(
                f"CUDA OOM at max_new_tokens={budget} sub_batch={n}; "
                "retrying the same sub-batch one sequence at a time.",
                flush=True,
            )
            for _ in range(n):
                completions.extend(_generate_batch(1, retry_from_oom=True))
        remaining -= n
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return completions


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[int(rank)])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def gate_prep_run_meta(args: argparse.Namespace, model_path: str, tokenizer_path: str) -> Dict[str, object]:
    return {
        "args": vars(args),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "model_path_resolved": model_path,
        "tokenizer_path_resolved": tokenizer_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script_shas": {
            "generate_math_branch_tournaments_rltt.py": sha256_file(Path(__file__).resolve()),
            "math_bg_probe_lib.py": sha256_file(THIS_DIR / "math_bg_probe_lib.py"),
        },
    }


def checkpoint_protocol(args: argparse.Namespace, source: str, budget: int, cell_seed: int) -> Dict[str, object]:
    return {
        "source": source,
        "budget": int(budget),
        "cell_seed": int(cell_seed),
        "prompts_per_source_budget": int(args.prompts_per_source_budget),
        "attempts_per_problem": int(args.attempts_per_problem),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "generation_batch_size": int(args.generation_batch_size),
        "max_prompt_length": int(args.max_prompt_length),
        "min_math_level": int(args.min_math_level),
        "model_path": str(args.model_path),
        "tokenizer_path": str(args.tokenizer_path),
    }


def partial_checkpoint_path(args: argparse.Namespace) -> Path:
    return output_path(args.output_json).with_name("math_gate_prep_generation.partial.json")


def cell_checkpoint_path(args: argparse.Namespace, source: str, budget: int) -> Path:
    return output_path(args.output_json).with_name(f"cell_{source}_{int(budget)}.json")


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def checkpoint_matches(payload: Dict[str, object], expected: Dict[str, object]) -> bool:
    if not bool(payload.get("completed")):
        return False
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        return False
    return all(protocol.get(key) == value for key, value in expected.items())


def load_cell_checkpoint(args: argparse.Namespace, source: str, budget: int, cell_seed: int) -> Optional[Dict[str, object]]:
    path = cell_checkpoint_path(args, source, budget)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ignoring unreadable cell checkpoint {path}: {exc}", flush=True)
        return None
    expected = checkpoint_protocol(args, source, int(budget), int(cell_seed))
    if not checkpoint_matches(payload, expected):
        print(f"ignoring mismatched cell checkpoint {path}", flush=True)
        return None
    return payload


def write_cell_checkpoint(
    args: argparse.Namespace,
    source: str,
    budget: int,
    cell_seed: int,
    source_meta: Dict[str, object],
    summary: Dict[str, object],
    tournaments: Sequence[Dict[str, object]],
) -> Path:
    path = cell_checkpoint_path(args, source, budget)
    payload = {
        "completed": True,
        "protocol": checkpoint_protocol(args, source, int(budget), int(cell_seed)),
        "source_meta": source_meta,
        "summary": summary,
        "tournaments": list(tournaments),
    }
    write_json_atomic(path, payload)
    return path


def build_attempt(problem, attempt_idx: int, completion_info: Dict[str, object], budget: int) -> Dict[str, object]:
    completion = str(completion_info["completion"])
    extracted = extract_answer(completion)
    is_correct = extracted is not None and answers_equal(extracted, problem.gold_answer)
    attempt = {
        "attempt_index": int(attempt_idx),
        "source": "generated",
        "completion": completion,
        "candidate_text": candidate_text(problem, completion),
        "extracted_answer": extracted,
        "is_correct": bool(is_correct),
        "is_parseable": extracted is not None,
        "output_tokens": int(completion_info["output_tokens"]),
        "truncated": bool(completion_info["truncated"]),
        "oom_retry": bool(completion_info.get("oom_retry", False)),
        "budget": int(budget),
    }
    if not is_correct:
        difficulty = classify_wrong_math_branch(
            prompt=problem.question,
            reference_solution=problem.gold_solution,
            gold_answer=problem.gold_answer,
            candidate_text=completion,
            extracted_answer=extracted,
            truncated=bool(completion_info["truncated"]),
        )
        attempt.update({
            "wrong_classification": difficulty["classification"],
            "classifier_reason": difficulty["reason"],
            "classifier_fallback": bool(difficulty["classifier_fallback"]),
            "structurally_math_like": bool(difficulty["structurally_math_like"]),
            "shared_intermediate_count": int(difficulty.get("shared_intermediate_count", 0)),
        })
    return attempt


def source_budget_summary(source: str, budget: int, prompts_requested: int) -> Dict[str, object]:
    return {
        "source": source,
        "budget": int(budget),
        "prompts_requested": int(prompts_requested),
        "attempts_generated": 0,
        "correct_attempts": 0,
        "incorrect_attempts": 0,
        "unparseable_attempts": 0,
        "truncated_attempts": 0,
        "all_correct_prompts": 0,
        "all_wrong_prompts": 0,
        "mixed_prompts": 0,
        "kept_tournaments": 0,
        "near_miss_dominant": 0,
        "nonsense_dominant": 0,
        "trivial": 0,
        "kept_near_miss_branches": 0,
        "kept_nonsense_branches": 0,
        "kept_incorrect_branches": 0,
        "_output_tokens": [],
    }


def finalize_summary(summary: Dict[str, object]) -> Dict[str, object]:
    tokens = list(summary.pop("_output_tokens", []))
    attempts = int(summary["attempts_generated"])
    kept = int(summary["kept_tournaments"])
    wrong_kept = int(summary["kept_incorrect_branches"])
    summary["kept_rate"] = kept / max(int(summary["prompts_requested"]), 1)
    summary["correct_rate"] = int(summary["correct_attempts"]) / max(attempts, 1)
    summary["unparseable_rate"] = int(summary["unparseable_attempts"]) / max(attempts, 1)
    summary["truncation_rate"] = int(summary["truncated_attempts"]) / max(attempts, 1)
    summary["output_token_mean"] = float(sum(tokens) / len(tokens)) if tokens else float("nan")
    summary["output_token_median"] = float(percentile(tokens, 50.0)) if tokens else float("nan")
    summary["output_token_p95"] = float(percentile(tokens, 95.0)) if tokens else float("nan")
    summary["near_miss_fraction"] = int(summary["near_miss_dominant"]) / max(kept, 1) if kept else float("nan")
    summary["nonsense_fraction"] = int(summary["nonsense_dominant"]) / max(kept, 1) if kept else float("nan")
    summary["wrong_branch_near_miss_fraction"] = (
        int(summary["kept_near_miss_branches"]) / max(wrong_kept, 1) if wrong_kept else float("nan")
    )
    return summary


def collect_example(examples: Dict[str, Dict[str, List[Dict[str, object]]]], tournament: Dict[str, object]) -> None:
    source = str(tournament["source"])
    examples.setdefault(source, {"near_miss": [], "nonsense": []})
    for attempt in tournament["attempts"]:
        if attempt.get("is_correct"):
            continue
        cls = attempt.get("wrong_classification")
        bucket = "near_miss" if cls == "near_miss" else "nonsense"
        if len(examples[source][bucket]) >= 2:
            continue
        examples[source][bucket].append({
            "source": source,
            "budget": tournament["budget"],
            "prompt_snippet": str(tournament["question"])[:240],
            "gold_answer": tournament["gold_answer"],
            "candidate_answer": attempt.get("extracted_answer"),
            "classifier_reasoning": attempt.get("classifier_reason"),
            "completion_snippet": str(attempt.get("completion", ""))[:300],
        })


def cell_verdict(summary: Dict[str, object]) -> str:
    kept = int(summary["kept_tournaments"])
    frac = float(summary["near_miss_fraction"]) if math.isfinite(float(summary["near_miss_fraction"])) else 0.0
    if frac >= 0.30 and kept >= 10:
        return "GREEN"
    if frac >= 0.30 and kept < 10:
        return "YELLOW_INSUFFICIENT_N"
    if frac >= 0.15:
        return "YELLOW"
    return "RED"


def source_verdict(source: str, summaries: Sequence[Dict[str, object]]) -> str:
    source_rows = [row for row in summaries if row["source"] == source]
    total_kept = sum(int(row["kept_tournaments"]) for row in source_rows)
    if total_kept < 10:
        return "RED_INSUFFICIENT_YIELD"
    verdicts = [str(row["cell_verdict"]) for row in source_rows]
    for verdict in ("GREEN", "YELLOW", "YELLOW_INSUFFICIENT_N", "RED"):
        if verdict in verdicts:
            return verdict
    return "RED"


def worst_verdict(verdicts: Sequence[str]) -> str:
    order = {
        "GREEN": 0,
        "YELLOW": 1,
        "YELLOW_INSUFFICIENT_N": 2,
        "RED": 3,
        "RED_INSUFFICIENT_YIELD": 4,
    }
    return max(verdicts, key=lambda v: order[v])


def fmt_rate(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if not math.isfinite(f) else f"{f:.3f}"


def write_gate_prep_md(path: Path, result: Dict[str, object]) -> None:
    meta = result["meta"]
    verdicts = result["verdicts"]
    lines = [
        "# Math Gate-Prep Generation",
        "",
        f"GATE_PREP_VERDICT_GSM8K = {verdicts.get('gsm8k', 'RED_INSUFFICIENT_YIELD')}",
        f"GATE_PREP_VERDICT_MATH  = {verdicts.get('math', 'RED_INSUFFICIENT_YIELD')}",
        f"GATE_PREP_VERDICT       = {verdicts['overall']}",
        "",
        "## Provenance",
        "",
        f"- command: `{meta['command']}`",
        f"- model path: `{meta['model_path_resolved']}`",
        f"- tokenizer path: `{meta['tokenizer_path_resolved']}`",
        f"- timestamp: `{meta['timestamp']}`",
        f"- git commit: `{meta.get('git_commit') or 'unavailable'}`",
        f"- script SHA256: `{meta['script_shas']['generate_math_branch_tournaments_rltt.py']}`",
        f"- lib SHA256: `{meta['script_shas']['math_bg_probe_lib.py']}`",
        "",
        "## Overall",
        "",
        "| source | budget | prompts | attempts | kept | kept_rate | correct_rate | unparseable_rate | truncation_rate | near_miss_fraction | nonsense_fraction | token_median | token_p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["source_budget"]:
        lines.append(
            f"| {row['source']} | {row['budget']} | {row['prompts_requested']} | "
            f"{row['attempts_generated']} | {row['kept_tournaments']} | {fmt_rate(row['kept_rate'])} | "
            f"{fmt_rate(row['correct_rate'])} | {fmt_rate(row['unparseable_rate'])} | "
            f"{fmt_rate(row['truncation_rate'])} | {fmt_rate(row['near_miss_fraction'])} | "
            f"{fmt_rate(row['nonsense_fraction'])} | {fmt_rate(row['output_token_median'])} | "
            f"{fmt_rate(row['output_token_p95'])} |"
        )
    composition = result["kept_set_composition"]
    lines.extend([
        "",
        "## Kept-Set Composition",
        "",
        f"- total kept tournaments: {composition['total_kept_tournaments']}",
        f"- kept by source: GSM8K={composition['by_source'].get('gsm8k', 0)}, MATH={composition['by_source'].get('math', 0)}",
        f"- kept by budget: {composition['by_budget']}",
        (
            "- dominant split: "
            f"near-miss={composition['near_miss_dominant']} ({fmt_rate(composition['near_miss_dominant_fraction'])}), "
            f"nonsense={composition['nonsense_dominant']} ({fmt_rate(composition['nonsense_dominant_fraction'])})"
        ),
        "",
        "## Example Branches",
        "",
    ])
    examples = result.get("examples", {})
    for source in ("gsm8k", "math"):
        lines.append(f"### {source.upper()}")
        for bucket in ("near_miss", "nonsense"):
            rows = examples.get(source, {}).get(bucket, [])
            lines.append("")
            lines.append(f"**{bucket.replace('_', '-')}**")
            if not rows:
                lines.append("")
                lines.append("- none available")
                continue
            for ex in rows[:2]:
                lines.extend([
                    "",
                    f"- budget: {ex['budget']}",
                    f"  prompt: {ex['prompt_snippet']}",
                    f"  gold answer: `{ex['gold_answer']}`",
                    f"  candidate answer: `{ex['candidate_answer']}`",
                    f"  classifier reasoning: `{ex['classifier_reasoning']}`",
                ])
        lines.append("")
    lines.extend([
        "## Proceeding Criterion",
        "",
        "GREEN: near-miss-dominant fraction >= 30% in at least one budget stratum with at least 10 kept tournaments. Action: scale to gate-scale.",
        "",
        "YELLOW: near-miss-dominant fraction 15-30% in the best budget stratum. Action: tune generation parameters once more, then accept if the second pass remains 15-30%.",
        "",
        "YELLOW_INSUFFICIENT_N: near-miss-dominant fraction is high, but the source-budget cell has fewer than 10 kept tournaments. Action: increase yield before treating it as GREEN.",
        "",
        "RED: near-miss-dominant fraction < 15% in all budget strata. Action: stop and investigate generator/verifier or pivot.",
        "",
        "RED_INSUFFICIENT_YIELD: fewer than 10 kept tournaments across all budgets for a source. Action: stop and fix tournament yield.",
        "",
    ])
    if verdicts["overall"] in {"RED", "RED_INSUFFICIENT_YIELD"}:
        lines.extend([
            "## Stop Fixes",
            "",
            "Do not proceed to tap training or a full sweep.",
            "",
            "- lower temperature (0.5, 0.3)",
            "- stricter `Final answer:` format constraint",
            "- more attempts per prompt (8 instead of 4)",
            "- different budget allocation",
            "- source-specific generation settings",
            "- pivot to code branch dataset",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_cell_payload(
    cell_payload: Dict[str, object],
    tournaments: List[Dict[str, object]],
    source_budget_rows: List[Dict[str, object]],
    source_meta: Dict[str, object],
    examples: Dict[str, Dict[str, List[Dict[str, object]]]],
) -> None:
    summary = dict(cell_payload["summary"])
    source = str(summary["source"])
    budget = int(summary["budget"])
    source_budget_rows.append(summary)
    source_meta[f"{source}_{budget}"] = cell_payload.get("source_meta", {})
    for local_tournament in cell_payload.get("tournaments", []):
        tournament = dict(local_tournament)
        tournament["cell_tournament_id"] = int(tournament.get("tournament_id", len(tournaments)))
        tournament["tournament_id"] = len(tournaments)
        tournaments.append(tournament)
        collect_example(examples, tournament)


def build_gate_prep_result(
    args: argparse.Namespace,
    model_path: str,
    tokenizer_path: str,
    source_budget_rows: Sequence[Dict[str, object]],
    source_meta: Dict[str, object],
    examples: Dict[str, Dict[str, List[Dict[str, object]]]],
    tournaments: Sequence[Dict[str, object]],
    completed_cells: Sequence[str],
    partial: bool = False,
) -> Dict[str, object]:
    verdicts = {source: source_verdict(source, source_budget_rows) for source in args.sources}
    verdicts.setdefault("gsm8k", "RED_INSUFFICIENT_YIELD")
    verdicts.setdefault("math", "RED_INSUFFICIENT_YIELD")
    verdicts["overall"] = worst_verdict([verdicts["gsm8k"], verdicts["math"]])

    by_source = {source: sum(1 for t in tournaments if t["source"] == source) for source in args.sources}
    by_budget = {str(b): sum(1 for t in tournaments if int(t["budget"]) == int(b)) for b in args.budget_strata}
    near_dom = sum(1 for t in tournaments if t["difficulty_classification"] == "near_miss_dominant")
    nonsense_dom = sum(1 for t in tournaments if t["difficulty_classification"] == "nonsense_dominant")
    composition = {
        "total_kept_tournaments": len(tournaments),
        "by_source": by_source,
        "by_budget": by_budget,
        "near_miss_dominant": int(near_dom),
        "nonsense_dominant": int(nonsense_dom),
        "near_miss_dominant_fraction": near_dom / max(len(tournaments), 1) if tournaments else float("nan"),
        "nonsense_dominant_fraction": nonsense_dom / max(len(tournaments), 1) if tournaments else float("nan"),
    }
    meta = gate_prep_run_meta(args, model_path, tokenizer_path)
    meta.update({
        "source_meta": source_meta,
        "completed_cells": list(completed_cells),
        "partial": bool(partial),
        "cell_checkpoint_files": [
            str(cell_checkpoint_path(args, row["source"], int(row["budget"])))
            for row in source_budget_rows
        ],
    })
    return {
        "meta": meta,
        "verdicts": verdicts,
        "source_budget": list(source_budget_rows),
        "kept_set_composition": composition,
        "examples": examples,
        "tournaments": list(tournaments),
    }


def write_partial_checkpoint(
    args: argparse.Namespace,
    model_path: str,
    tokenizer_path: str,
    source_budget_rows: Sequence[Dict[str, object]],
    source_meta: Dict[str, object],
    examples: Dict[str, Dict[str, List[Dict[str, object]]]],
    tournaments: Sequence[Dict[str, object]],
    completed_cells: Sequence[str],
) -> Path:
    result = build_gate_prep_result(
        args=args,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        source_budget_rows=source_budget_rows,
        source_meta=source_meta,
        examples=examples,
        tournaments=tournaments,
        completed_cells=completed_cells,
        partial=True,
    )
    path = partial_checkpoint_path(args)
    write_json_atomic(path, result)
    return path


def run_gate_prep(args: argparse.Namespace, model, tokenizer, device: torch.device,
                  model_path: str, tokenizer_path: str) -> None:
    tournaments: List[Dict[str, object]] = []
    source_budget_rows: List[Dict[str, object]] = []
    source_meta: Dict[str, object] = {}
    examples: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    completed_cells: List[str] = []
    cell_index = 0
    for source in args.sources:
        for budget in args.budget_strata:
            cell_seed = int(args.seed) + cell_index * 9973
            cell_key = f"{source}_{int(budget)}"
            existing = load_cell_checkpoint(args, source, int(budget), cell_seed)
            if existing is not None:
                append_cell_payload(existing, tournaments, source_budget_rows, source_meta, examples)
                completed_cells.append(cell_key)
                print(
                    f"resume_skip source={source} budget={budget} "
                    f"checkpoint={cell_checkpoint_path(args, source, int(budget))}",
                    flush=True,
                )
                write_partial_checkpoint(
                    args, model_path, tokenizer_path, source_budget_rows,
                    source_meta, examples, tournaments, completed_cells)
                cell_index += 1
                continue

            problems, cell_meta = load_math_problems(
                source, int(args.prompts_per_source_budget), cell_seed, args.min_math_level)
            summary = source_budget_summary(source, int(budget), int(args.prompts_per_source_budget))
            cell_tournaments: List[Dict[str, object]] = []
            print(f"\n=== Gate-prep cell source={source} budget={budget} seed={cell_seed} ===")
            for idx, problem in enumerate(problems):
                prompt = problem_prompt(problem)
                completions = generate_attempts(model, tokenizer, prompt, args, device, max_new_tokens=int(budget))
                attempts = [build_attempt(problem, attempt_idx, info, int(budget))
                            for attempt_idx, info in enumerate(completions)]

                summary["attempts_generated"] += len(attempts)
                summary["correct_attempts"] += sum(1 for a in attempts if a["is_correct"])
                summary["incorrect_attempts"] += sum(1 for a in attempts if not a["is_correct"])
                summary["unparseable_attempts"] += sum(1 for a in attempts if not a["is_parseable"])
                summary["truncated_attempts"] += sum(1 for a in attempts if a["truncated"])
                summary["_output_tokens"].extend(int(a["output_tokens"]) for a in attempts)

                has_correct = any(a["is_correct"] for a in attempts)
                has_incorrect = any(not a["is_correct"] for a in attempts)
                if has_correct and has_incorrect:
                    summary["mixed_prompts"] += 1
                    difficulty = tournament_difficulty_classification(attempts)
                    tournament = {
                        "tournament_id": len(cell_tournaments),
                        "source": problem.source,
                        "budget": int(budget),
                        "dataset_index": problem.dataset_index,
                        "level": problem.level,
                        "question": problem.question,
                        "gold_answer": problem.gold_answer,
                        "gold_solution": problem.gold_solution,
                        "attempts": attempts,
                        "correct_indices": [i for i, a in enumerate(attempts) if a["is_correct"]],
                        "incorrect_indices": [i for i, a in enumerate(attempts) if not a["is_correct"]],
                        "attempts_to_first_correct": next(
                            (i + 1 for i, a in enumerate(attempts) if a["is_correct"]), None),
                        "correct_fraction": sum(1 for a in attempts if a["is_correct"]) / max(len(attempts), 1),
                        "tournament_formation_budget": int(budget),
                        "difficulty_classification": difficulty["classification"],
                        "near_miss_fraction": difficulty["near_miss_fraction"],
                        "classifier_fallback": difficulty["classifier_fallback"],
                        "near_miss_branches": difficulty["near_miss_branches"],
                        "nonsense_branches": difficulty["nonsense_branches"],
                        "incorrect_branches": difficulty["incorrect_branches"],
                    }
                    cell_tournaments.append(tournament)
                    summary["kept_tournaments"] += 1
                    if difficulty["classification"] == "near_miss_dominant":
                        summary["near_miss_dominant"] += 1
                    elif difficulty["classification"] == "nonsense_dominant":
                        summary["nonsense_dominant"] += 1
                    else:
                        summary["trivial"] += 1
                    summary["kept_near_miss_branches"] += int(difficulty["near_miss_branches"])
                    summary["kept_nonsense_branches"] += int(difficulty["nonsense_branches"])
                    summary["kept_incorrect_branches"] += int(difficulty["incorrect_branches"])
                elif has_correct:
                    summary["all_correct_prompts"] += 1
                else:
                    summary["all_wrong_prompts"] += 1

                if args.report_every > 0 and ((idx + 1) % args.report_every == 0 or idx + 1 == len(problems)):
                    print(
                        f"{source} budget={budget} {idx+1}/{len(problems)} prompts "
                        f"kept={summary['kept_tournaments']} correct={summary['correct_attempts']} "
                        f"incorrect={summary['incorrect_attempts']} unparseable={summary['unparseable_attempts']} "
                        f"truncated={summary['truncated_attempts']}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            finalized = finalize_summary(summary)
            finalized["cell_verdict"] = cell_verdict(finalized)
            cell_path = write_cell_checkpoint(
                args=args,
                source=source,
                budget=int(budget),
                cell_seed=cell_seed,
                source_meta=cell_meta,
                summary=finalized,
                tournaments=cell_tournaments,
            )
            append_cell_payload(
                {
                    "source_meta": cell_meta,
                    "summary": finalized,
                    "tournaments": cell_tournaments,
                },
                tournaments,
                source_budget_rows,
                source_meta,
                examples,
            )
            completed_cells.append(cell_key)
            partial_path = write_partial_checkpoint(
                args, model_path, tokenizer_path, source_budget_rows,
                source_meta, examples, tournaments, completed_cells)
            print(
                f"cell_done source={source} budget={budget} kept={finalized['kept_tournaments']} "
                f"near_miss_fraction={fmt_rate(finalized['near_miss_fraction'])} "
                f"cell_verdict={finalized['cell_verdict']}"
            )
            print(f"wrote_cell_checkpoint={cell_path}")
            print(f"wrote_partial_checkpoint={partial_path}")
            cell_index += 1

    result = build_gate_prep_result(
        args=args,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        source_budget_rows=source_budget_rows,
        source_meta=source_meta,
        examples=examples,
        tournaments=tournaments,
        completed_cells=completed_cells,
        partial=False,
    )
    out = output_path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, result)
    if args.output_md:
        write_gate_prep_md(output_path(args.output_md), result)

    print("\n=== Gate-prep verdict ===")
    print(f"GATE_PREP_VERDICT_GSM8K = {result['verdicts']['gsm8k']}")
    print(f"GATE_PREP_VERDICT_MATH  = {result['verdicts']['math']}")
    print(f"GATE_PREP_VERDICT       = {result['verdicts']['overall']}")
    print(f"Wrote {out}")
    if args.output_md:
        print(f"Wrote {output_path(args.output_md)}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model_path = resolve_local(args.model_path)
    tokenizer_path = resolve_local(args.tokenizer_path)

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading RLTT model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model_device = next(model.parameters()).device
    print(
        f"Runtime device requested={args.device} model_device={model_device} "
        f"cuda_available={torch.cuda.is_available()}"
    )
    if model_device.type == "cuda":
        print(
            "CUDA memory after model load: "
            f"allocated={torch.cuda.memory_allocated(model_device) / (1024 ** 3):.2f} GiB "
            f"reserved={torch.cuda.memory_reserved(model_device) / (1024 ** 3):.2f} GiB"
        )
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0

    if args.gate_prep:
        run_gate_prep(args, model, tokenizer, device, model_path, tokenizer_path)
        return

    problems, source_meta = load_math_problems(
        args.source, args.max_problems, args.seed, args.min_math_level)
    if not problems:
        raise SystemExit("No math problems loaded.")

    tournaments: List[Dict[str, object]] = []
    total_attempts = 0
    generated_correct = 0
    generated_incorrect = 0
    unparseable = 0

    for idx, problem in enumerate(problems):
        prompt = problem_prompt(problem)
        completions = generate_attempts(model, tokenizer, prompt, args, device)
        attempts = []
        seen_text = set()
        for attempt_idx, completion_info in enumerate(completions):
            completion = str(completion_info["completion"])
            if not completion or completion in seen_text:
                continue
            seen_text.add(completion)
            extracted = extract_answer(completion)
            if extracted is None:
                unparseable += 1
                is_correct = False
            else:
                is_correct = answers_equal(extracted, problem.gold_answer)
            total_attempts += 1
            generated_correct += int(is_correct)
            generated_incorrect += int(not is_correct)
            attempts.append({
                "attempt_index": attempt_idx,
                "source": "generated",
                "completion": completion,
                "candidate_text": candidate_text(problem, completion),
                "extracted_answer": extracted,
                "is_correct": bool(is_correct),
                "output_tokens": int(completion_info["output_tokens"]),
                "truncated": bool(completion_info["truncated"]),
            })

        if args.allow_reference_correct_for_smoke and not any(a["is_correct"] for a in attempts):
            attempts.insert(0, {
                "attempt_index": -1,
                "source": "reference_smoke",
                "completion": problem.gold_solution,
                "candidate_text": gold_solution_text(problem),
                "extracted_answer": problem.gold_answer,
                "is_correct": True,
            })

        if any(a["is_correct"] for a in attempts) and any(not a["is_correct"] for a in attempts):
            tournaments.append({
                "tournament_id": len(tournaments),
                "source": problem.source,
                "dataset_index": problem.dataset_index,
                "level": problem.level,
                "question": problem.question,
                "gold_answer": problem.gold_answer,
                "attempts": attempts,
                "correct_indices": [i for i, a in enumerate(attempts) if a["is_correct"]],
                "incorrect_indices": [i for i, a in enumerate(attempts) if not a["is_correct"]],
            })

        if args.report_every > 0 and ((idx + 1) % args.report_every == 0 or idx + 1 == len(problems)):
            keep = len(tournaments)
            print(
                f"{idx+1}/{len(problems)} prompts  kept={keep} "
                f"gen_correct={generated_correct} gen_incorrect={generated_incorrect} "
                f"unparseable={unparseable}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    meta = {
        "args": vars(args),
        "model_path_resolved": model_path,
        "tokenizer_path_resolved": tokenizer_path,
        "source_meta": source_meta,
        "problems_seen": len(problems),
        "tournaments_kept": len(tournaments),
        "total_generated_attempts": total_attempts,
        "generated_correct": generated_correct,
        "generated_incorrect": generated_incorrect,
        "unparseable_attempts": unparseable,
        "kept_prompt_rate": len(tournaments) / max(len(problems), 1),
        "reference_smoke_enabled": bool(args.allow_reference_correct_for_smoke),
    }
    result = {"meta": meta, "tournaments": tournaments}

    out = output_path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    print("\n=== Generated math tournaments ===")
    print(f"problems_seen={len(problems)}")
    print(f"tournaments_kept={len(tournaments)}")
    print(f"kept_prompt_rate={meta['kept_prompt_rate']:.3f}")
    print(f"generated_correct={generated_correct} generated_incorrect={generated_incorrect}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
