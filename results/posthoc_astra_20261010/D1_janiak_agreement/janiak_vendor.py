"""Functions vendored verbatim from Denis Janiak's reasoning-geometry repository, commit 26c48be.

MIT License, Copyright (c) 2026 Denis Janiak (full text in LICENSE_reasoning-geometry next to this file).

Sources (function bodies copied unchanged; only imports are gathered here):
  applications/incremental_abstention.py  _finite, is_parseable_answer, _row_logprob, _winning_answer,
      _plurality_outcome, _mean_field, _impute_and_scale, crossfit_logistic_predictions, _auacc,
      prompt_metrics, paired_bootstrap_delta, METRIC_NAMES
  baselines/deepconf_asymmetry.py          _average_ranks, auroc, bootstrap_auroc, _pearson
  controls/orgad_agreement_control.py      MIN_STRATUM_CLASS, spearman, crossfit_residuals, agreement_strata,
      stratum_readout, agreement_level_counts

Not vendored: aggregate_prompt_features / prompt_accounting (they need data.trace_caps for cap accounting that
B0 does not use). `aggregate_question` below restates their feature logic for one question; test_vendor.py
checks it, and every vendored function, against Denis's code by import.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Mapping

import numpy as np

METRIC_NAMES = ("auacc", "aurc", "brier", "log_loss")
MIN_STRATUM_CLASS = 10


# ---- applications/incremental_abstention.py -------------------------------------------------------------------

def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def is_parseable_answer(value) -> bool:
    return value is not None and bool(str(value).strip())


def _row_logprob(row: Mapping) -> float:
    for key in ("logprob_score", "mean_logprob"):
        value = _finite(row.get(key))
        if value is not None:
            return value
    return float("-inf")


def _winning_answer(rows: list[Mapping]) -> str | None:
    parsed = [row for row in rows if is_parseable_answer(row.get("predicted_answer"))]
    if not parsed:
        return None
    counts = Counter(str(row["predicted_answer"]) for row in parsed)
    best = max(counts.values())
    candidates = sorted(answer for answer, count in counts.items() if count == best)
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda answer: max(
            _row_logprob(row)
            for row in parsed
            if str(row["predicted_answer"]) == answer
        ),
    )


def _plurality_outcome(rows: list[Mapping]) -> float:
    winner = _winning_answer(rows)
    if winner is None:
        return 0.0
    gold = next(
        (str(row.get("gold_answer")) for row in rows if row.get("gold_answer") is not None),
        "",
    )
    return float(winner == gold)


def _mean_field(group: list[Mapping], key: str) -> float:
    values = [_finite(row.get(key)) for row in group]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else float("nan")


def _impute_and_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    means = np.nanmean(np.where(np.isfinite(train), train, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train = np.where(np.isfinite(train), train, means)
    test = np.where(np.isfinite(test), test, means)
    scale = np.std(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (train - means) / scale, (test - means) / scale


def crossfit_logistic_predictions(
    features: np.ndarray,
    outcomes: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Fit a logistic readout inside each supplied prompt fold."""
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(outcomes, dtype=float)
    fold_values = np.asarray(folds)
    if len(x) != len(y) or len(y) != len(fold_values):
        raise ValueError("features, outcomes, and folds must have the same length")
    probabilities = np.full(len(y), np.nan, dtype=float)
    usable = np.isfinite(y) & np.isfinite(fold_values.astype(float))
    if not usable.any():
        return probabilities
    for fold in sorted(set(fold_values[usable].tolist())):
        test_mask = usable & (fold_values == fold)
        train_mask = usable & (fold_values != fold)
        if not test_mask.any() or not train_mask.any():
            continue
        train_x, test_x = _impute_and_scale(x[train_mask], x[test_mask])
        train_y = y[train_mask].astype(int)
        if len(np.unique(train_y)) < 2:
            probabilities[test_mask] = float(np.mean(train_y))
            continue
        model = LogisticRegression(max_iter=2000, random_state=seed)
        try:
            model.fit(train_x, train_y)
            probabilities[test_mask] = model.predict_proba(test_x)[:, 1]
        except ValueError:
            probabilities[test_mask] = float(np.mean(train_y))
    return probabilities


def _auacc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    usable = np.isfinite(scores) & np.isfinite(outcomes)
    if not usable.any():
        return float("nan")
    scores = np.asarray(scores[usable], dtype=float)
    outcomes = np.asarray(outcomes[usable], dtype=float)
    order = np.argsort(-scores, kind="stable")
    values = outcomes[order]
    if len(values) == 1:
        return float(values[0])
    coverages = np.arange(1, len(values) + 1, dtype=float) / len(values)
    accuracies = np.cumsum(values) / np.arange(1, len(values) + 1, dtype=float)
    return float(np.trapezoid(accuracies, coverages))


def prompt_metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, float | int | None]:
    """Return ranking and strictly proper scores for calibrated OOF probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(probabilities) & np.isfinite(outcomes)
    n = int(usable.sum())
    if n == 0:
        return {"auacc": None, "aurc": None, "brier": None, "log_loss": None, "n": 0}
    p = np.clip(probabilities[usable], 1e-12, 1 - 1e-12)
    y = outcomes[usable]
    auacc = _auacc(p, y)
    aurc = float((1.0 - 1.0 / n) - auacc)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {
        "auacc": float(auacc),
        "aurc": aurc,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": log_loss,
        "n": n,
    }


def paired_bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    outcomes: np.ndarray,
    *,
    metric: str = "auacc",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired prompt-bootstrap delta ``left - right`` for one metric."""
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRIC_NAMES}")
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(left) & np.isfinite(right) & np.isfinite(outcomes)
    left, right, outcomes = left[usable], right[usable], outcomes[usable]
    if not len(outcomes):
        return {
            "metric": metric,
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_two_sided": None,
            "n_valid": 0,
        }

    def value(scores: np.ndarray, labels: np.ndarray) -> float:
        return float(prompt_metrics(scores, labels)[metric])

    point = value(left, outcomes) - value(right, outcomes)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(outcomes), size=len(outcomes))
        draws[index] = value(left[sampled], outcomes[sampled]) - value(
            right[sampled], outcomes[sampled]
        )
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": float(
            min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
        ),
        "n_valid": int(len(draws)),
    }


# ---- baselines/deepconf_asymmetry.py ---------------------------------------------------------------------------

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so tied scores cannot bias AUROC either way."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def auroc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    """Probability a correct prompt outranks an incorrect one; 0.5 is chance.

    Unlike AUACC this does not inherit the base accuracy, which is the whole
    reason the module exists.
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(scores) & np.isfinite(outcomes)
    scores, outcomes = scores[usable], outcomes[usable]
    positives = outcomes > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auroc(
    scores: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Percentile interval over prompt resamples, matching the frozen convention."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    point = auroc(scores, outcomes)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, len(outcomes), len(outcomes))
        value = auroc(scores[index], outcomes[index])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return {"point_estimate": point, "ci_low": None, "ci_high": None, "n_valid": 0}
    return {
        "point_estimate": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n_valid": len(draws),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    usable = np.isfinite(left) & np.isfinite(right)
    left, right = left[usable], right[usable]
    if len(left) < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


# ---- controls/orgad_agreement_control.py -----------------------------------------------------------------------

def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson on tie-averaged ranks, over the pairs where both are finite."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3:
        return float("nan")
    return _pearson(_average_ranks(left[usable]), _average_ranks(right[usable]))


def crossfit_residuals(
    target: np.ndarray,
    covariate: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """Out-of-fold residual of ``target`` after a linear fit on ``covariate``.

    Fitting in-sample would let the residual keep whatever the covariate explains on
    these very prompts, which is the leak the frozen readouts avoid everywhere else.
    """
    target = np.asarray(target, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    folds = np.asarray(folds)
    residuals = np.full(len(target), np.nan, dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test & np.isfinite(target) & np.isfinite(covariate)
        if train.sum() < 2 or np.std(covariate[train]) < 1e-12:
            continue
        design = np.column_stack([np.ones(int(train.sum())), covariate[train]])
        coefficients, *_ = np.linalg.lstsq(design, target[train], rcond=None)
        predicted = coefficients[0] + coefficients[1] * covariate[test]
        residuals[test] = target[test] - predicted
    return residuals


def agreement_strata(agreement: np.ndarray) -> dict[str, np.ndarray]:
    """Unanimous versus split, as boolean masks over the prompt list.

    Agreement is a share of *parseable* siblings, so the denominator varies and the
    levels are not a clean eighths grid.  Unanimity is the one cut that means the
    same thing at every denominator, and it is where the objection actually bites:
    if geometry were an agreement proxy it would be flat across a stratum where
    agreement does not vary at all.
    """
    agreement = np.asarray(agreement, dtype=float)
    return {
        "unanimous": agreement >= 1.0,
        "split": np.isfinite(agreement) & (agreement < 1.0),
    }


def stratum_readout(
    scores: np.ndarray,
    outcomes: np.ndarray,
    mask: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """AUROC inside one stratum, or an explicit refusal when it is too small."""
    scores, outcomes = np.asarray(scores, dtype=float)[mask], np.asarray(outcomes, dtype=float)[mask]
    n_correct = int((outcomes > 0.5).sum())
    n_wrong = int((outcomes <= 0.5).sum())
    body = {
        "n_prompts": int(mask.sum()),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "base_accuracy": float(np.mean(outcomes)) if len(outcomes) else float("nan"),
    }
    if min(n_correct, n_wrong) < MIN_STRATUM_CLASS:
        return {**body, "auroc": None, "reported": False}
    return {
        **body,
        "auroc": bootstrap_auroc(scores, outcomes, n_bootstrap=n_bootstrap, seed=seed),
        "reported": True,
    }


def agreement_level_counts(agreement: np.ndarray, outcomes: np.ndarray) -> list[dict]:
    """Prompts and accuracy at each distinct agreement level, for the scope note."""
    agreement = np.asarray(agreement, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    table = []
    for level in sorted(set(agreement[np.isfinite(agreement)].tolist())):
        mask = agreement == level
        table.append(
            {
                "agreement": float(level),
                "n_prompts": int(mask.sum()),
                "accuracy": float(np.mean(outcomes[mask])),
            }
        )
    return table


# ---- restated: the per-question feature logic of aggregate_prompt_features -------------------------------------

def aggregate_question(group: list[Mapping]) -> dict:
    """One question's B0 features and outcome, as aggregate_prompt_features computes them: length, entropy and
    log-probability averaged over all traces (unparsed included); vote agreement over parseable traces (0 if none);
    outcome by _plurality_outcome (all-unparsed = automatic failure, 0)."""
    parsed = [row for row in group if is_parseable_answer(row.get("predicted_answer"))]
    winner = _winning_answer(group)
    vote = (
        sum(str(row.get("predicted_answer")) == winner for row in parsed) / len(parsed)
        if parsed and winner is not None
        else 0.0
    )
    return {
        "length": _mean_field(group, "length_score"),
        "entropy": _mean_field(group, "entropy_score"),
        "logprob": _mean_field(group, "logprob_score"),
        "vote_agreement": float(vote),
        "outcome": _plurality_outcome(group),
        "automatic_failure": not parsed,
        "valid_plurality": bool(parsed),
        "winner": winner,
    }
