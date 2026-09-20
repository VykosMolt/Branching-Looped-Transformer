from __future__ import annotations

import time
from collections import Counter

import torch

from bg_convergence_hairs_rs_v1_common import (
    DATASET_JSON,
    DATASET_MD,
    DATASET_PT,
    HAIR_CANDIDATES_CSV,
    HAIR_PAIRS_CSV,
    build_hair_dataset,
    md_table,
    status_line,
    write_csv,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    dataset = build_hair_dataset()
    candidate_rows = dataset.get("candidate_rows") or []
    pair_rows = dataset.get("pair_rows") or []
    summary = dict(dataset.get("summary") or {})
    if not candidate_rows or not pair_rows:
        verdict = "BLOCKED"
    elif summary.get("l30_candidate_count", 0) and summary.get("l42_candidate_count", 0):
        verdict = "READY"
    elif summary.get("l30_candidate_count", 0):
        verdict = "L30_ONLY"
    elif summary.get("l42_candidate_count", 0):
        verdict = "L42_ONLY"
    else:
        verdict = "REGEN_REQUIRED"
    summary["BG_CONVERGENCE_HAIR_DATASET_VERDICT"] = verdict
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    dataset["summary"] = summary
    torch.save(dataset, DATASET_PT)
    write_csv(HAIR_CANDIDATES_CSV, candidate_rows)
    write_csv(HAIR_PAIRS_CSV, pair_rows)
    json_payload = {
        **summary,
        "sample_candidates": candidate_rows[:10],
        "sample_pairs": pair_rows[:10],
    }
    write_json(DATASET_JSON, json_payload)
    lines = [
        "# DualAnchor Convergence Hair Dataset v1",
        "",
        status_line("BG_CONVERGENCE_HAIR_DATASET_VERDICT", verdict),
        "",
        "## Summary",
        "",
        f"- candidate rows: `{len(candidate_rows)}`",
        f"- pair rows: `{len(pair_rows)}`",
        f"- tasks: `{summary.get('task_count')}`",
        f"- domains: `{summary.get('domain_counts')}`",
        f"- L30 candidates: `{summary.get('l30_candidate_count')}`",
        f"- L42 candidates: `{summary.get('l42_candidate_count')}`",
        f"- hidden availability rate: `{summary.get('hidden_available_rate')}`",
        f"- logits available: `{summary.get('logits_available')}`",
        "",
        "DualAnchor pair margins are adjacent next-stage readout margins. No L30/L42 tap head is introduced.",
        "",
        "## Candidate Hair Counts",
        "",
    ]
    hair_counts = [{"hair_stage": key, "count": value} for key, value in sorted(Counter(row.get("hair_stage") for row in candidate_rows).items())]
    lines.extend(md_table(hair_counts, ["hair_stage", "count"]))
    lines.extend(["", "## Pair Sample", ""])
    lines.extend(md_table(pair_rows[:20], ["task_id", "domain", "hair_stage", "hidden_cosine_distance", "hidden_rms_normalized", "dualanchor_abs_avg_margin", "anchor_disagreement_pair", "reward_tied_eval_only"]))
    lines.extend([
        "",
        "## Boundary",
        "",
        "- Final reward, parsed answer, and output text fields are evaluation/diagnostic-only.",
        "- Branch classification/taxonomy is not used as a runtime architecture input.",
    ])
    write_md(DATASET_MD, lines)
    print(status_line("BG_CONVERGENCE_HAIR_DATASET_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

