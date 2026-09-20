from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manual"))

from math_bg_probe_lib import (  # noqa: E402
    classify_wrong_math_branch,
    cycle_rate,
    margin_calibration,
)


def test_cycle_rate_transitive_matrix_is_zero() -> None:
    utility = torch.tensor([3.0, 2.0, 1.0, 0.0])
    mat = utility[:, None] - utility[None, :]

    assert cycle_rate(mat, triplets_per_matrix=100, seed=7) == 0.0


def test_cycle_rate_detects_three_way_cycle() -> None:
    mat = torch.zeros(3, 3)
    mat[0, 1] = 1.0
    mat[1, 0] = -1.0
    mat[1, 2] = 1.0
    mat[2, 1] = -1.0
    mat[2, 0] = 1.0
    mat[0, 2] = -1.0

    assert cycle_rate(mat, triplets_per_matrix=25, seed=3) == 1.0


def test_margin_calibration_uses_equal_population_bins() -> None:
    mat = torch.zeros(3, 3)
    mat[0, 1] = 2.0
    mat[1, 0] = -2.0
    mat[0, 2] = -2.0
    mat[2, 0] = 2.0
    labels = torch.tensor([True, False, False])

    cal = margin_calibration(mat, labels, n_bins=1)

    expected_conf = float(torch.sigmoid(torch.tensor(2.0)))
    assert cal["n_pairs"] == 2
    assert cal["bins"][0]["count"] == 2
    assert cal["bins"][0]["accuracy"] == pytest.approx(0.5)
    assert cal["ece"] == pytest.approx(abs(0.5 - expected_conf))


def test_near_miss_classifier_accepts_numeric_tolerance() -> None:
    result = classify_wrong_math_branch(
        prompt="What is 50 plus 50?",
        reference_solution="50 + 50 = 100.",
        gold_answer="100",
        candidate_text="50 + 50 = 108. Final answer: 108",
        extracted_answer="108",
    )

    assert result["classification"] == "near_miss"
    assert "numeric_tolerance" in result["reason"]


def test_near_miss_classifier_uses_shared_intermediate() -> None:
    result = classify_wrong_math_branch(
        prompt="A pattern starts with six groups. Find the final count.",
        reference_solution="Each group has 7 items, so 6 * 7 = 42.",
        gold_answer="42",
        candidate_text="The useful intermediate is 7. I then choose badly. Final answer: 80",
        extracted_answer="80",
    )

    assert result["classification"] == "near_miss"
    assert "shared_intermediate" in result["reason"]


def test_near_miss_classifier_marks_unparseable_as_nonsense() -> None:
    result = classify_wrong_math_branch(
        prompt="What is 2 + 2?",
        reference_solution="2 + 2 = 4.",
        gold_answer="4",
        candidate_text="I cannot finish this.",
        extracted_answer=None,
    )

    assert result["classification"] == "nonsense"
    assert result["reason"] == "unparseable_final_answer"
