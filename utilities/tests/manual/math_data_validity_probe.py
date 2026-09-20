"""Validate the cached math pilot tournaments post hoc.

No branch generation and no model loading. The only optional transformers load
is the local Ouro-RLTT tokenizer, used to recover generated-output token counts
for old pilot artifacts that did not store `output_tokens`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    DEFAULT_RLTT_PATH,
    PROJECT_ROOT,
    classify_wrong_math_branch,
    extract_answer,
    extract_gold_answer,
    output_path,
    resolve_local,
)


DEFAULT_INPUT = PROJECT_ROOT / "opi/taps/probes/math_branch_tournaments_rltt.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "opi/taps/probes/math_data_validity_2026-05-16.md"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "opi/taps/probes/math_data_validity_2026-05-16.json"
MAX_NEW_TOKENS_DEFAULT = 160

FINAL_MARKER_RE = re.compile(
    r"(final\s+answer\s*(?:is|:)|answer\s*(?:is|:)|####|\\boxed\s*\{)",
    flags=re.IGNORECASE,
)

RECOMMENDED_NEXT = {
    "CLEAN": (
        "TRANSFER_POOR is real. Pivot to code as Track B training domain per "
        "the v4 plan. Optionally queue full-split HH capture to test "
        "capacity-vs-transfer separation cleanly."
    ),
    "TRUNCATION_CONFOUNDED": (
        "TRANSFER_POOR is suspect. Regenerate the math tournaments at "
        "max_new_tokens=1024 or higher with strict final-answer wrapper, on "
        "5-10 GSM8K prompts only (don't fight MATH yet). Re-run the transfer "
        "probe on the clean tournaments before any architectural pivot."
    ),
    "PARSER_CONFOUNDED": (
        "Fix the parser to flag parse failures separately from incorrect "
        "answers. Re-classify the existing tournaments. Re-run the transfer "
        "probe. The pilot's '0 unparseable' claim was wrong and the "
        "kept-tournament filter let through parse failures as incorrect."
    ),
    "BOTH": "Regenerate at higher budget with fixed parser. Re-run probe.",
    "UNCLEAR": (
        "Inventory shows missing diagnostic fields. Manually inspect the 10 "
        "random branches printed in Part D and make a judgment call."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT))
    parser.add_argument("--tokenizer-path", default=DEFAULT_RLTT_PATH)
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pct(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{100.0 * value:.1f}%"


def num(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.1f}" if abs(value) >= 10 else f"{value:.3f}"


def snippet(text: object, limit: int) -> str:
    value = str(text)
    value = value.replace("\r\n", "\n")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def compact(text: object, limit: int) -> str:
    return snippet(" ".join(str(text).split()), limit)


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


def load_tournament_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        found = sorted(PROJECT_ROOT.glob("**/math_branch_tournaments*.json"))
        raise SystemExit(
            f"Missing {path}. Search found: {', '.join(str(p) for p in found) or 'nothing'}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokenizer(tokenizer_path: str):
    resolved = resolve_local(tokenizer_path)
    return AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=True,
        local_files_only=True,
    ), resolved


def token_length(tokenizer, completion: str, attempt: Dict[str, object]) -> Tuple[int, str]:
    if "output_tokens" in attempt:
        return int(attempt["output_tokens"]), "stored_output_tokens"
    encoded = tokenizer(str(completion), add_special_tokens=False)
    return int(len(encoded["input_ids"])), "tokenizer_recovered_completion"


def has_final_marker_at_end(completion: str) -> bool:
    text = str(completion).strip()
    if not text:
        return False
    tail = text[-220:]
    match = list(FINAL_MARKER_RE.finditer(tail))
    if not match:
        return False
    last = match[-1]
    after = tail[last.end() :].strip()
    if "\\boxed" in last.group(0).lower():
        return "}" in after or tail.rstrip().endswith("}")
    # Treat it as an end marker only when the marker's answer payload is near
    # the end rather than followed by another paragraph of reasoning.
    return len(after) <= 80 and "\n\n" not in after


def load_reference_solutions(tournaments: Sequence[Dict[str, object]]) -> Tuple[Dict[Tuple[str, int], str], List[str]]:
    refs: Dict[Tuple[str, int], str] = {}
    notes: List[str] = []

    need_gsm = any(t.get("source") == "gsm8k" for t in tournaments)
    need_math = any(t.get("source") == "math" for t in tournaments)

    if need_gsm:
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main", split="test")
            for t in tournaments:
                if t.get("source") == "gsm8k":
                    idx = int(t.get("dataset_index"))
                    refs[("gsm8k", idx)] = str(ds[idx]["answer"])
        except Exception as exc:
            notes.append(f"GSM8K reference recovery failed: {type(exc).__name__}: {exc}")

    if need_math:
        try:
            from math_bg_probe_lib import _load_math_dataset

            ds, ds_name = _load_math_dataset()
            for t in tournaments:
                if t.get("source") == "math":
                    idx = int(t.get("dataset_index"))
                    ex = ds[idx]
                    refs[("math", idx)] = str(ex.get("solution", ex.get("Solution", "")))
            notes.append(f"MATH references recovered from {ds_name}")
        except Exception as exc:
            notes.append(f"MATH reference recovery failed: {type(exc).__name__}: {exc}")

    missing = []
    for t in tournaments:
        key = (str(t.get("source")), int(t.get("dataset_index")))
        if key not in refs:
            missing.append(key)
    if missing:
        notes.append(f"Missing reference solutions for {len(missing)} tournaments; classifier used numeric fallback.")
    return refs, notes


def analyze_attempts(
    payload: Dict[str, object],
    tokenizer,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, int], str], List[str], Dict[str, object]]:
    tournaments = payload.get("tournaments", [])
    meta = payload.get("meta", {})
    max_new_tokens = int(meta.get("args", {}).get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT))
    references, notes = load_reference_solutions(tournaments)
    rows: List[Dict[str, object]] = []
    length_sources = Counter()

    for tournament in tournaments:
        source = str(tournament.get("source", "unknown"))
        dataset_index = int(tournament.get("dataset_index", -1))
        reference_solution = references.get((source, dataset_index), "")
        if not reference_solution:
            reference_solution = str(tournament.get("gold_solution", ""))
        for attempt in tournament.get("attempts", []):
            completion = str(attempt.get("completion", ""))
            length, length_source = token_length(tokenizer, completion, attempt)
            length_sources[length_source] += 1
            stored_extracted = attempt.get("extracted_answer")
            recomputed_extracted = extract_answer(completion)
            has_extractable = recomputed_extracted is not None
            was_truncated = bool(attempt.get("truncated", False)) or length >= max_new_tokens - 2
            is_correct = bool(attempt.get("is_correct"))
            classifier: Optional[Dict[str, object]] = None
            if not is_correct:
                classifier = classify_wrong_math_branch(
                    prompt=str(tournament.get("question", "")),
                    reference_solution=reference_solution,
                    gold_answer=str(tournament.get("gold_answer", "")),
                    candidate_text=completion,
                    extracted_answer=recomputed_extracted,
                    truncated=was_truncated,
                )
            rows.append({
                "tournament_id": int(tournament.get("tournament_id", -1)),
                "source": source,
                "dataset_index": dataset_index,
                "question": str(tournament.get("question", "")),
                "gold_answer": str(tournament.get("gold_answer", "")),
                "attempt_index": int(attempt.get("attempt_index", -1)),
                "is_correct": is_correct,
                "completion": completion,
                "candidate_text": str(attempt.get("candidate_text", "")),
                "stored_extracted_answer": stored_extracted,
                "recomputed_extracted_answer": recomputed_extracted,
                "has_extractable_answer": has_extractable,
                "stored_recomputed_match": stored_extracted == recomputed_extracted,
                "output_token_length": length,
                "length_source": length_source,
                "was_truncated": was_truncated,
                "has_final_marker": has_final_marker_at_end(completion),
                "classifier": classifier,
            })

    analysis_meta = {
        "max_new_tokens": max_new_tokens,
        "length_sources": dict(length_sources),
        "reference_notes": notes,
    }
    return rows, references, notes, analysis_meta


def aggregate_by_source_correctness(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        correctness = "correct" if row["is_correct"] else "incorrect"
        groups[(row["source"], correctness)].append(row)
    out: List[Dict[str, object]] = []
    for (source, correctness), items in sorted(groups.items()):
        lengths = [int(r["output_token_length"]) for r in items]
        out.append({
            "source": source,
            "correctness": correctness,
            "n": len(items),
            "mean_len": statistics.mean(lengths) if lengths else float("nan"),
            "median_len": statistics.median(lengths) if lengths else float("nan"),
            "p95_len": percentile(lengths, 95),
            "trunc_rate": sum(bool(r["was_truncated"]) for r in items) / max(len(items), 1),
            "has_final_marker": sum(bool(r["has_final_marker"]) for r in items) / max(len(items), 1),
            "has_extractable_answer": sum(bool(r["has_extractable_answer"]) for r in items) / max(len(items), 1),
        })
    return out


def classify_tournaments(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    by_tournament: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_tournament[int(row["tournament_id"])].append(row)

    tournament_rows: List[Dict[str, object]] = []
    for tournament_id, items in sorted(by_tournament.items()):
        source = items[0]["source"]
        incorrect = [r for r in items if not r["is_correct"]]
        near = sum(1 for r in incorrect if r["classifier"] and r["classifier"]["classification"] == "near_miss")
        nonsense = sum(1 for r in incorrect if r["classifier"] and r["classifier"]["classification"] == "nonsense")
        if not incorrect:
            label = "trivial"
        elif near == nonsense:
            label = "mixed"
        elif near > nonsense:
            label = "near_miss_dominant"
        else:
            label = "nonsense_dominant"
        tournament_rows.append({
            "tournament_id": tournament_id,
            "source": source,
            "incorrect_branches": len(incorrect),
            "near_miss_branches": near,
            "nonsense_branches": nonsense,
            "classification": label,
        })

    agg: Dict[str, Counter] = defaultdict(Counter)
    branch_agg: Dict[str, Counter] = defaultdict(Counter)
    for row in tournament_rows:
        agg[row["source"]]["n_tournaments"] += 1
        agg[row["source"]][row["classification"]] += 1
    for row in rows:
        if row["is_correct"]:
            continue
        cls = row["classifier"]["classification"] if row["classifier"] else "unknown"
        branch_agg[row["source"]]["incorrect_branches"] += 1
        branch_agg[row["source"]][cls] += 1

    source_rows: List[Dict[str, object]] = []
    for source in sorted(agg.keys()):
        total_wrong = branch_agg[source]["incorrect_branches"]
        source_rows.append({
            "source": source,
            "n_tournaments": agg[source]["n_tournaments"],
            "near_miss_dominant": agg[source]["near_miss_dominant"],
            "nonsense_dominant": agg[source]["nonsense_dominant"],
            "mixed": agg[source]["mixed"],
            "trivial": agg[source]["trivial"],
            "incorrect_branches": total_wrong,
            "near_miss_branches": branch_agg[source]["near_miss"],
            "nonsense_branches": branch_agg[source]["nonsense"],
            "near_miss_fraction": branch_agg[source]["near_miss"] / max(total_wrong, 1),
            "nonsense_fraction": branch_agg[source]["nonsense"] / max(total_wrong, 1),
        })
    return tournament_rows, source_rows, [
        {
            "source": source,
            **dict(counts),
        }
        for source, counts in sorted(branch_agg.items())
    ]


def select_examples(tournaments: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    by_source = {str(t.get("source")): t for t in tournaments}
    for source in ("gsm8k", "math"):
        if source in by_source:
            examples.append(by_source[source])
    clean = None
    for t in tournaments:
        if len(t.get("correct_indices", [])) == 1 and len(t.get("incorrect_indices", [])) == 1:
            clean = t
            break
    if clean is None:
        for t in tournaments:
            if len(t.get("correct_indices", [])) == 1 or len(t.get("incorrect_indices", [])) == 1:
                clean = t
                break
    if clean is not None and all(clean.get("tournament_id") != e.get("tournament_id") for e in examples):
        examples.append(clean)
    elif clean is not None:
        # Keep exactly three examples when possible, even if no 1-correct/1-wrong
        # tournament exists in a four-attempt artifact.
        for t in tournaments:
            if all(t.get("tournament_id") != e.get("tournament_id") for e in examples):
                examples.append(t)
                break
    return examples[:3]


def sample_incorrect_branches(rows: Sequence[Dict[str, object]], seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    samples: List[Dict[str, object]] = []
    for source in ("gsm8k", "math"):
        candidates = [r for r in rows if r["source"] == source and not r["is_correct"]]
        rng.shuffle(candidates)
        samples.extend(candidates[:5])
    return samples


def determine_validity(
    trunc_table: Sequence[Dict[str, object]],
    classifier_source_rows: Sequence[Dict[str, object]],
    meta_unparseable: int,
) -> Tuple[str, str, Dict[str, object]]:
    table = {(r["source"], r["correctness"]): r for r in trunc_table}
    math_incorrect = table.get(("math", "incorrect"))
    parser_cells = [
        r for r in trunc_table
        if r["correctness"] == "incorrect" and r["has_extractable_answer"] < 1.0
    ]
    math_cls = next((r for r in classifier_source_rows if r["source"] == "math"), None)
    if math_incorrect is None or math_cls is None:
        return "UNCLEAR", RECOMMENDED_NEXT["UNCLEAR"], {
            "reason": "missing MATH incorrect cell or classifier aggregate"
        }

    overall_wrong = sum(int(r["incorrect_branches"]) for r in classifier_source_rows)
    overall_near = sum(int(r["near_miss_branches"]) for r in classifier_source_rows)
    overall_near_fraction = overall_near / max(overall_wrong, 1)
    incorrect_rows = [r for r in trunc_table if r["correctness"] == "incorrect"]
    overall_incorrect_n = sum(int(r["n"]) for r in incorrect_rows)
    overall_truncated = sum(float(r["trunc_rate"]) * int(r["n"]) for r in incorrect_rows)
    overall_trunc_rate = overall_truncated / max(overall_incorrect_n, 1)

    trunc_confounded = (
        math_incorrect["trunc_rate"] > 0.50
        or overall_near_fraction < 0.30
    )
    parser_confounded = bool(parser_cells) and meta_unparseable == 0

    if trunc_confounded and parser_confounded:
        validity = "BOTH"
    elif trunc_confounded:
        validity = "TRUNCATION_CONFOUNDED"
    elif parser_confounded:
        validity = "PARSER_CONFOUNDED"
    elif (
        math_incorrect["trunc_rate"] < 0.30
        and math_cls["near_miss_fraction"] > 0.50
        and overall_near_fraction >= 0.30
        and math_incorrect["has_extractable_answer"] == 1.0
    ):
        validity = "CLEAN"
    else:
        validity = "UNCLEAR"
    return validity, RECOMMENDED_NEXT[validity], {
        "math_incorrect_trunc_rate": math_incorrect["trunc_rate"],
        "math_incorrect_near_miss_fraction": math_cls["near_miss_fraction"],
        "overall_incorrect_trunc_rate": overall_trunc_rate,
        "overall_incorrect_near_miss_fraction": overall_near_fraction,
        "parser_confounded_cells": [
            {
                "source": r["source"],
                "correctness": r["correctness"],
                "has_extractable_answer": r["has_extractable_answer"],
            }
            for r in parser_cells
        ],
        "pilot_meta_unparseable_attempts": meta_unparseable,
    }


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def write_report(result: Dict[str, object], path: Path) -> None:
    payload = result["input_summary"]
    lines: List[str] = [
        "# Math Pilot Data Validity (2026-05-16)",
        "",
        f"DATA_VALIDITY = {result['data_validity']}",
        f"RECOMMENDED_NEXT = {result['recommended_next']}",
        "",
        "## Part A - Tournament JSON Structure",
        "",
        f"- File: `{result['input_json']}`",
        f"- Top-level keys: `{payload['top_level_keys']}`",
        f"- Meta keys: `{payload['meta_keys']}`",
        f"- n_tournaments: `{payload['n_tournaments']}`",
        f"- Tournament keys: `{payload['tournament_keys']}`",
        f"- Candidate keys: `{payload['attempt_keys']}`",
        f"- Stored output token field: `{payload['stores_output_tokens']}`",
        f"- Stored truncated flag: `{payload['stores_truncated']}`",
        f"- max_new_tokens: `{result['analysis_meta']['max_new_tokens']}`",
        f"- Token length source counts: `{result['analysis_meta']['length_sources']}`",
        "",
        "### Example Tournaments",
        "",
    ]
    for ex in result["example_tournaments"]:
        lines.extend([
            f"#### Tournament {ex['tournament_id']} ({ex['source']})",
            "",
            f"Prompt: {ex['question']}",
            "",
            f"Gold answer: `{ex['gold_answer']}`",
            "",
        ])
        for attempt in ex["attempts"]:
            lines.extend([
                f"- Branch {attempt['attempt_index']} correct={attempt['is_correct']} extracted=`{attempt['extracted_answer']}`",
                "",
                "```text",
                attempt["candidate_text"],
                "```",
                "",
            ])

    lines.extend(["## Part B - Truncation Diagnosis", ""])
    trunc_rows = []
    for row in result["truncation_table"]:
        trunc_rows.append([
            row["source"],
            row["correctness"],
            row["n"],
            num(row["mean_len"]),
            num(row["median_len"]),
            num(row["p95_len"]),
            pct(row["trunc_rate"]),
            pct(row["has_final_marker"]),
            pct(row["has_extractable_answer"]),
        ])
    lines.extend(md_table(
        [
            "source",
            "correctness",
            "n",
            "mean_len",
            "median_len",
            "p95_len",
            "trunc_rate",
            "has_final_marker",
            "has_extractable_answer",
        ],
        trunc_rows,
    ))

    lines.extend(["", "## Part C - Near-Miss / Nonsense Classifier", ""])
    cls_rows = []
    for row in result["classifier_source_table"]:
        cls_rows.append([
            row["source"],
            row["n_tournaments"],
            row["near_miss_dominant"],
            row["nonsense_dominant"],
            row["mixed"],
            row["trivial"],
            pct(row["near_miss_fraction"]),
            pct(row["nonsense_fraction"]),
        ])
    lines.extend(md_table(
        [
            "source",
            "n_tournaments",
            "near_miss_dominant",
            "nonsense_dominant",
            "mixed",
            "trivial",
            "near_miss_branch_frac",
            "nonsense_branch_frac",
        ],
        cls_rows,
    ))

    lines.extend(["", "## Part D - Sample Incorrect Branches", ""])
    for row in result["sample_incorrect_branches"]:
        cls = row["classifier"]
        lines.extend([
            f"### {row['source']} tournament={row['tournament_id']} branch={row['attempt_index']}",
            "",
            f"- Prompt: {row['prompt_snippet']}",
            f"- Gold answer: `{row['gold_answer']}`",
            f"- output_token_length: `{row['output_token_length']}`",
            f"- was_truncated: `{row['was_truncated']}`",
            f"- extracted_answer: `{row['recomputed_extracted_answer']}`",
            f"- classifier verdict: `{cls['classification']}` ({cls['reason']})",
            "",
            "```text",
            row["completion"],
            "```",
            "",
        ])

    lines.extend(["## Edge Cases", ""])
    for note in result["analysis_meta"].get("reference_notes", []):
        lines.append(f"- {note}")
    if not result["analysis_meta"].get("reference_notes"):
        lines.append("- None.")
    lines.extend([
        f"- Stored-vs-recomputed parser mismatches: `{result['parser_mismatches']}`",
        f"- Validity details: `{result['validity_details']}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    input_path = output_path(args.input_json)
    payload = load_tournament_json(input_path)
    tournaments = payload.get("tournaments", [])
    if not tournaments:
        raise SystemExit(f"No tournaments in {input_path}")

    tokenizer, tokenizer_path = load_tokenizer(args.tokenizer_path)
    rows, _refs, notes, analysis_meta = analyze_attempts(payload, tokenizer)
    analysis_meta["tokenizer_path"] = tokenizer_path
    analysis_meta["reference_notes"] = notes

    truncation_table = aggregate_by_source_correctness(rows)
    tournament_classes, classifier_source_table, classifier_branch_table = classify_tournaments(rows)
    meta_unparseable = int(payload.get("meta", {}).get("unparseable_attempts", -1))
    data_validity, recommended_next, validity_details = determine_validity(
        truncation_table,
        classifier_source_table,
        meta_unparseable,
    )

    example_tournaments = []
    for tournament in select_examples(tournaments):
        example_tournaments.append({
            "tournament_id": int(tournament.get("tournament_id", -1)),
            "source": str(tournament.get("source", "unknown")),
            "question": snippet(tournament.get("question", ""), 800),
            "gold_answer": str(tournament.get("gold_answer", "")),
            "attempts": [
                {
                    "attempt_index": int(a.get("attempt_index", -1)),
                    "is_correct": bool(a.get("is_correct")),
                    "extracted_answer": a.get("extracted_answer"),
                    "candidate_text": snippet(a.get("candidate_text", ""), 500),
                }
                for a in tournament.get("attempts", [])
            ],
        })

    sample_rows = []
    for row in sample_incorrect_branches(rows, args.seed):
        sample_rows.append({
            "source": row["source"],
            "tournament_id": row["tournament_id"],
            "attempt_index": row["attempt_index"],
            "prompt_snippet": compact(row["question"], 200),
            "gold_answer": row["gold_answer"],
            "completion": snippet(row["completion"], 1000),
            "output_token_length": row["output_token_length"],
            "was_truncated": row["was_truncated"],
            "recomputed_extracted_answer": row["recomputed_extracted_answer"],
            "classifier": row["classifier"],
        })

    first_tournament = tournaments[0]
    first_attempt = first_tournament.get("attempts", [{}])[0]
    result = {
        "data_validity": data_validity,
        "recommended_next": recommended_next,
        "input_json": repo_path(input_path),
        "input_summary": {
            "top_level_keys": list(payload.keys()),
            "meta_keys": list(payload.get("meta", {}).keys()),
            "n_tournaments": len(tournaments),
            "tournament_keys": list(first_tournament.keys()),
            "attempt_keys": list(first_attempt.keys()),
            "stores_output_tokens": "output_tokens" in first_attempt,
            "stores_truncated": "truncated" in first_attempt,
        },
        "analysis_meta": analysis_meta,
        "truncation_table": truncation_table,
        "classifier_tournament_table": tournament_classes,
        "classifier_source_table": classifier_source_table,
        "classifier_branch_table": classifier_branch_table,
        "example_tournaments": example_tournaments,
        "sample_incorrect_branches": sample_rows,
        "parser_mismatches": sum(not bool(r["stored_recomputed_match"]) for r in rows),
        "validity_details": validity_details,
    }

    out_md = output_path(args.output_md)
    out_json = output_path(args.output_json)
    write_report(result, out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(f"DATA_VALIDITY = {data_validity}")
    print(f"RECOMMENDED_NEXT = {recommended_next}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
