"""Decide whether quota-directed v4 generation is needed after split salvage."""
from __future__ import annotations

import time

from bg_hidden_origin_split_salvage_common import (
    AUDIT_JSON,
    CV_STABILITY_JSON,
    EVAL_MODES_JSON,
    SALVAGE_EVAL_JSON,
    SALVAGE_ROOT,
    V4_QUOTA_JSON,
    ensure_salvage_root,
    load_json,
    md_table,
    rel,
    write_json,
    write_md,
)


OUT_MD = SALVAGE_ROOT / "v4_quota_need.md"


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    audit = load_json(AUDIT_JSON, {}) or {}
    modes = load_json(EVAL_MODES_JSON, {}) or {}
    eval_payload = load_json(SALVAGE_EVAL_JSON, {}) or {}
    cv = load_json(CV_STABILITY_JSON, {}) or {}
    if not audit or not modes:
        verdict = "INSUFFICIENT"
    else:
        eval_verdict = str(eval_payload.get("verdict") or "")
        mode_verdict = str(modes.get("verdict") or "")
        cv_verdict = str(cv.get("verdict") or "")
        combined = (audit.get("version_summary") or {}).get("combined", {})
        all_signal = audit.get("all_reward_signal_support") or {}
        strict = (audit.get("strict_split_bottleneck") or {}).get("support") or {}
        if eval_verdict in {"SELECTOR_READY", "OLD_TAPS_BEST", "ENSEMBLE_BEST"}:
            verdict = "NO_V4_NEEDED_SELECTOR_READY"
        elif mode_verdict in {"NO_VALID_LARGER_SPLIT", "WEAK_ONLY", "CV_READY"} and int(all_signal.get("non_tie_pairs", 0)) >= 80 and int(combined.get("behaviorally_diverse_groups", 0)) >= 60:
            verdict = "V4_REQUIRED_HELDOUT_BALANCE"
        elif cv_verdict in {"STABLE_POSITIVE", "WEAK_POSITIVE"}:
            verdict = "V4_OPTIONAL_WEAK_SIGNAL"
        elif int(combined.get("behaviorally_diverse_groups", 0)) < 60 or float(combined.get("tie_rate", 1.0)) > 0.95:
            verdict = "V4_REQUIRED_BRANCH_GENERATOR"
        elif int(strict.get("non_tie_pairs", 0)) < 80:
            verdict = "V4_REQUIRED_HELDOUT_BALANCE"
        else:
            verdict = "STOP_NO_SIGNAL"

    quotas = {
        "heldout_task_ids": 8,
        "heldout_behaviorally_diverse_groups": 20,
        "heldout_non_tie_pairs": 120,
        "train_behaviorally_diverse_groups": 60,
        "val_behaviorally_diverse_groups": 15,
        "generation_policy": "generate per split, not generate-then-split",
        "heldout_policy": "reserve heldout tasks before generation and exclude from empirical direction construction",
        "stop_rule": "stop only when split quotas are met",
        "recipe": "use v3 high-yield task screening and non-random direction recipe; prioritize L36, K=8, primary alpha<=0.01, perturbation-sensitive/wrong-parseable tasks",
    }
    payload = {
        "BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT": verdict,
        "verdict": verdict,
        "selector_eval_verdict": eval_payload.get("verdict"),
        "eval_mode_verdict": modes.get("verdict"),
        "cv_stability_verdict": cv.get("verdict"),
        "best_selector_available": eval_payload.get("HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE"),
        "strict_support": (audit.get("strict_split_bottleneck") or {}).get("support"),
        "all_signal_support": audit.get("all_reward_signal_support"),
        "combined_summary": (audit.get("version_summary") or {}).get("combined"),
        "proposed_v4_quotas": quotas,
        "decision": (
            "Existing data has useful signal but the original clean heldout pool is not reward-pair balanced."
            if verdict == "V4_REQUIRED_HELDOUT_BALANCE"
            else "See verdict and supporting fields."
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(V4_QUOTA_JSON, payload)
    lines = [
        "# Hidden-Origin V4 Quota Need",
        "",
        f"BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = {verdict}",
        "",
        f"- selector_eval_verdict: `{payload['selector_eval_verdict']}`",
        f"- eval_mode_verdict: `{payload['eval_mode_verdict']}`",
        f"- cv_stability_verdict: `{payload['cv_stability_verdict']}`",
        f"- best_selector_available: `{payload['best_selector_available']}`",
        "",
        "## Proposed V4 Quotas",
        "",
    ]
    lines.extend(md_table([quotas], list(quotas.keys())))
    lines.extend(
        [
            "",
            "V4 should generate by pre-reserved split quotas rather than generating a large pool and hoping the split is balanced afterward.",
            "",
            f"Wrote `{rel(V4_QUOTA_JSON)}`.",
        ]
    )
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(V4_QUOTA_JSON)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

