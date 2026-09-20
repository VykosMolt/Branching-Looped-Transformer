"""Write Branch Generator v1 rich branch-outcome diagnostics schema."""
from __future__ import annotations

import time

from bg_branch_generator_v1_common import (
    RICH_SCHEMA_JSON,
    RICH_SCHEMA_MD,
    ensure_bgv1_root,
    rel,
    write_json,
    write_md,
)


PRIMARY_REWARD = {
    "correct_parsed_answer": 1.0,
    "parseable_wrong_answer": 0.0,
    "parse_failure": -0.2,
    "empty_output": -0.5,
    "severe_repetition": -0.3,
    "cuda_or_runtime_error": -1.0,
}


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    schema = {
        "primary_label": "deterministic downstream reward",
        "primary_reward": PRIMARY_REWARD,
        "selector_readiness_label_policy": "primary deterministic reward only; ties omitted in pairwise labels",
        "auxiliary_fields": [
            "parsed_answer",
            "answer_correctness",
            "parse_success",
            "parse_failure_reason",
            "output_length",
            "hit_max_tokens",
            "repetition_rate",
            "empty_output",
            "option_logits_if_available",
            "correct_option_logit_if_available",
            "predicted_option_margin_if_available",
            "entropy_over_options_if_available",
            "answer_stability_n2_sampled_decode_for_borderline_tasks",
            "deterministic_vs_sampled_agreement",
            "old_v1_v2_v3_v4_salvage_selector_disagreement",
            "branch_hidden_distance_from_clean",
            "branch_logit_kl_from_clean_if_available",
            "tap_score_spread",
            "branch_feature_norm_rms",
            "off_manifold_warnings",
        ],
        "allowed_uses": [
            "branch-generator reward-shaping diagnostics",
            "recipe selection on train/val yield",
            "label-noise analysis",
            "future richer outcome modeling",
        ],
        "forbidden_uses": [
            "satisfying selector readiness",
            "collapsing reward ties into arbitrary labels",
            "using tap score as training label",
            "using heldout selector success as generator reward",
        ],
    }
    verdict = "READY"
    payload = {
        "BG_RICH_OUTCOME_SCHEMA_V1_VERDICT": verdict,
        "verdict": verdict,
        "schema": schema,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(RICH_SCHEMA_JSON, payload)
    lines = [
        "# Rich Outcome Schema V1",
        "",
        f"BG_RICH_OUTCOME_SCHEMA_V1_VERDICT = {verdict}",
        "",
        "Primary selector labels remain deterministic final reward only. Auxiliary fields are diagnostics for branch-generator design and noise analysis.",
        "",
        f"- schema_json: `{rel(RICH_SCHEMA_JSON)}`",
        f"- primary_reward: `{PRIMARY_REWARD}`",
        f"- auxiliary_fields: `{schema['auxiliary_fields']}`",
    ]
    write_md(RICH_SCHEMA_MD, lines)
    print(f"BG_RICH_OUTCOME_SCHEMA_V1_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
