"""V3 Horizon Logic power extension -- generation shards.

This is a thin wrapper around bg_v2_overnight_horizon_generate: the SAME locked
protocol (task pool, hash-sorted ordering, per-task seeds, prompts, sampling
params, strict pre-answer cut, feature capture) with exactly one change: output
lands in a fresh v3 run directory instead of the sealed 20260724 run.

Shard disjointness: the v2 main run consumed offsets [0, 170) of the hash-sorted
4000-task pool. V3 shards MUST use --task-offset >= 170. The pool cap (4000) and
sort key are inherited unmodified, so offsets index the identical ordering.

Usage (one shard):
  venv/bin/python -u utilities/tests/manual/bg_v3_horizon_power_generate.py \
      --n-tasks 170 --k 4 --max-new 448 --seed 20260724 --tag v3 \
      --task-offset 170 --budget-seconds 9000
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))

import bg_v2_overnight_horizon_generate as v2gen  # noqa: E402

V3_ROOT = PROJECT_ROOT / "artifacts/reports/horizon_power_v3_20260726/horizon_logic"

MIN_V3_OFFSET = 170  # offsets below this belong to the sealed v2 main run


def main() -> None:
    # Parse a copy of argv only to enforce the disjointness guard; the real
    # parsing happens inside the v2 main() on the same argv.
    import argparse

    guard = argparse.ArgumentParser(add_help=False)
    guard.add_argument("--task-offset", type=int, default=0)
    known, _ = guard.parse_known_args()
    if known.task_offset < MIN_V3_OFFSET:
        raise SystemExit(
            f"refusing to run: --task-offset {known.task_offset} overlaps the "
            f"sealed v2 main run (offsets [0, {MIN_V3_OFFSET}))"
        )

    v2gen.OUT_ROOT = V3_ROOT
    v2gen.main()


if __name__ == "__main__":
    main()
