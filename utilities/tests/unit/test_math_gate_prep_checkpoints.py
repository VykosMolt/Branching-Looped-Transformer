from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manual"))

from generate_math_branch_tournaments_rltt import (  # noqa: E402
    append_cell_payload,
    cell_checkpoint_path,
    load_cell_checkpoint,
    partial_checkpoint_path,
    write_cell_checkpoint,
    write_partial_checkpoint,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_json=str(tmp_path / "math_gate_prep_generation.json"),
        output_md=str(tmp_path / "math_gate_prep_generation.md"),
        sources=["gsm8k", "math"],
        budget_strata=[256, 512, 1024],
        prompts_per_source_budget=50,
        attempts_per_problem=4,
        temperature=0.7,
        top_p=0.95,
        generation_batch_size=2,
        max_prompt_length=384,
        min_math_level=4,
        model_path="shared/models/ouro_rltt_local",
        tokenizer_path="shared/models/ouro_rltt_local",
    )


def _summary() -> dict[str, object]:
    return {
        "source": "gsm8k",
        "budget": 256,
        "prompts_requested": 50,
        "attempts_generated": 200,
        "correct_attempts": 130,
        "incorrect_attempts": 70,
        "unparseable_attempts": 0,
        "truncated_attempts": 200,
        "all_correct_prompts": 20,
        "all_wrong_prompts": 5,
        "mixed_prompts": 25,
        "kept_tournaments": 1,
        "kept_rate": 0.5,
        "correct_rate": 0.65,
        "unparseable_rate": 0.0,
        "truncation_rate": 1.0,
        "output_token_mean": 256.0,
        "output_token_median": 256.0,
        "output_token_p95": 256.0,
        "near_miss_dominant": 1,
        "nonsense_dominant": 0,
        "trivial": 0,
        "kept_near_miss_branches": 1,
        "kept_nonsense_branches": 0,
        "kept_incorrect_branches": 1,
        "near_miss_fraction": 1.0,
        "nonsense_fraction": 0.0,
        "wrong_branch_near_miss_fraction": 1.0,
        "cell_verdict": "GREEN",
    }


def _tournament() -> dict[str, object]:
    return {
        "tournament_id": 0,
        "source": "gsm8k",
        "budget": 256,
        "question": "What is 2 + 2?",
        "gold_answer": "4",
        "difficulty_classification": "near_miss_dominant",
        "attempts": [
            {"is_correct": True, "completion": "Final answer: 4"},
            {
                "is_correct": False,
                "wrong_classification": "near_miss",
                "classifier_reason": "numeric_tolerance",
                "extracted_answer": "5",
                "completion": "2 + 2 = 5. Final answer: 5",
            },
        ],
    }


def test_cell_checkpoint_round_trip_and_protocol_match(tmp_path: Path) -> None:
    args = _args(tmp_path)

    path = write_cell_checkpoint(
        args=args,
        source="gsm8k",
        budget=256,
        cell_seed=42,
        source_meta={"source_requested": "gsm8k"},
        summary=_summary(),
        tournaments=[_tournament()],
    )

    assert path == cell_checkpoint_path(args, "gsm8k", 256)
    payload = load_cell_checkpoint(args, "gsm8k", 256, 42)
    assert payload is not None
    assert payload["completed"] is True
    assert payload["summary"]["kept_tournaments"] == 1

    changed = _args(tmp_path)
    changed.temperature = 0.3
    assert load_cell_checkpoint(changed, "gsm8k", 256, 42) is None


def test_partial_checkpoint_aggregates_completed_cells(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cell_payload = {
        "source_meta": {"source_requested": "gsm8k"},
        "summary": _summary(),
        "tournaments": [_tournament()],
    }
    tournaments: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    source_meta: dict[str, object] = {}
    examples: dict[str, dict[str, list[dict[str, object]]]] = {}

    append_cell_payload(cell_payload, tournaments, rows, source_meta, examples)
    path = write_partial_checkpoint(
        args=args,
        model_path="/abs/model",
        tokenizer_path="/abs/tokenizer",
        source_budget_rows=rows,
        source_meta=source_meta,
        examples=examples,
        tournaments=tournaments,
        completed_cells=["gsm8k_256"],
    )

    assert path == partial_checkpoint_path(args)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["meta"]["partial"] is True
    assert payload["meta"]["completed_cells"] == ["gsm8k_256"]
    assert payload["source_budget"][0]["cell_verdict"] == "GREEN"
    assert payload["tournaments"][0]["tournament_id"] == 0
    assert payload["tournaments"][0]["cell_tournament_id"] == 0
