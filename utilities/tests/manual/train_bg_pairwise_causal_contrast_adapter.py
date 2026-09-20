#!/usr/bin/env python3
"""Optional pairwise causal contrast adapter diagnostic.

This file exists so the adapter experiment has a reproducible optional entry
point.  The main prompt makes this part optional; by default the script records
SKIPPED unless explicitly enabled with BG_RUN_PAIRWISE_ADAPTER=1.
"""
from __future__ import annotations

import os
import time

from bg_causal_adapter_common import OUT_ROOT, load_adapter_dataset, rel, write_json, write_md


OUT_JSON = OUT_ROOT / "pairwise_contrast_adapter.json"
OUT_MD = OUT_ROOT / "pairwise_contrast_adapter.md"


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    enabled = os.environ.get("BG_RUN_PAIRWISE_ADAPTER", "0") == "1"
    dataset = load_adapter_dataset()
    if not enabled:
        verdict = "SKIPPED"
        reason = "optional pairwise diagnostic was not requested; free-generation eval has priority"
    elif not dataset.get("examples"):
        verdict = "INSUFFICIENT"
        reason = "adapter dataset missing"
    else:
        # The logit-margin adapter is the load-bearing causal adapter probe in
        # this bundle.  A full pairwise hidden-state objective would add another
        # training run; leave it explicit instead of silently changing scope.
        verdict = "SKIPPED"
        reason = "not run in this bounded adapter pass"
    payload = {
        "BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT": verdict,
        "reason": reason,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_md(
        OUT_MD,
        [
            "# BG Pairwise Causal Contrast Adapter",
            "",
            f"BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = {verdict}",
            "",
            f"- reason: `{reason}`",
        ],
    )
    print(f"BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
