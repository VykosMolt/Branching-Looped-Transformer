"""Lightweight tests for hidden-origin branch utilities."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_hidden_branching import (  # noqa: E402
    HIDDEN_DIM,
    HookHiddenOriginBrancher,
    TrueLatentForkCarry,
    cosine,
    delta_rms,
    make_branch_deltas,
    rms_distance,
    rms_normalize,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    v = torch.arange(1, 17, dtype=torch.float32)
    vn = rms_normalize(v)
    _assert(abs(float(vn.pow(2).mean().sqrt().item()) - 1.0) < 1e-6, "rms_normalize failed")

    deltas = make_branch_deltas(hidden_dim=128, k=4, alpha=0.01, seed=123)
    _assert(len(deltas) == 4, "wrong delta count")
    _assert(all(tuple(d.shape) == (128,) for d in deltas), "wrong delta shape")
    _assert(all(torch.isfinite(d).all().item() for d in deltas), "non-finite delta")
    _assert(delta_rms(deltas[0]) == 0.0, "zero branch missing")
    _assert(abs(delta_rms(deltas[1]) - 0.01) < 1e-6, "delta RMS not alpha")
    _assert(abs(delta_rms(deltas[2]) - 0.01) < 1e-6, "negative delta RMS not alpha")
    _assert(abs(cosine(deltas[1], deltas[2]) + 1.0) < 1e-5, "+/- deltas not antipodal")
    _assert(abs(cosine(deltas[1], deltas[3])) < 1e-4, "orthogonal delta cosine too high")
    _assert(rms_distance(deltas[1], deltas[2]) > 0.019, "rms distance too low")

    try:
        make_branch_deltas(hidden_dim=128, k=4, alpha=0.05, seed=123)
        raise AssertionError("safe alpha cap was not enforced")
    except ValueError:
        pass
    diag = make_branch_deltas(hidden_dim=128, k=4, alpha=0.05, seed=123, allow_diagnostic_alpha=True)
    _assert(abs(delta_rms(diag[1]) - 0.05) < 1e-6, "diagnostic alpha not allowed")

    mock = torch.nn.Linear(4, 4)
    before = [p.requires_grad for p in mock.parameters()]
    brancher = HookHiddenOriginBrancher(mock)
    true_fork = TrueLatentForkCarry(mock)
    after = [p.requires_grad for p in mock.parameters()]
    _assert(before == after, "brancher changed requires_grad")
    _assert(brancher.method_label == "hook_intervention_per_branch", "hook method label missing")
    _assert(true_fork.method_label == "true_fork_carry", "true fork label missing")

    print("BG_HIDDEN_BRANCH_UTILITY_TESTS = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
