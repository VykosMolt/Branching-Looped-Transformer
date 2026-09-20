"""Acquire hidden feature tensors for top4 survivor rows from cached artifacts."""
from __future__ import annotations

from bg_merged_tap_v1_common import run_survivor_feature_acquisition


if __name__ == "__main__":
    raise SystemExit(run_survivor_feature_acquisition())
