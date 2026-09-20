from __future__ import annotations

import time
from collections import Counter

from bg_convergence_hairs_rs_v1_common import (
    OUT_ROOT,
    TIE_JSON,
    finite_mean,
    load_dataset,
    md_table,
    safe_float,
    status_line,
    write_csv,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    dataset = load_dataset()
    pair_rows = list(dataset.get("pair_rows") or [])
    if not pair_rows:
        verdict = "INCONCLUSIVE"
    converged_cos = 0.02
    converged_rms = 0.20
    da_indiff = 0.10
    rows = []
    for row in pair_rows:
        reward_tied = safe_float(row.get("reward_tied_eval_only"), 0.0) > 0
        hidden_conv = safe_float(row.get("hidden_cosine_distance"), 1.0) <= converged_cos and safe_float(row.get("hidden_rms_normalized"), 1.0) <= converged_rms
        hidden_div = not hidden_conv
        da_near = safe_float(row.get("dualanchor_abs_avg_margin"), 1.0) <= da_indiff and safe_float(row.get("anchor_abs_margin_max"), 1.0) <= da_indiff
        if reward_tied and hidden_div:
            label = "reward_tied_hidden_divergent"
        elif reward_tied and hidden_conv:
            label = "reward_tied_hidden_converged"
        elif not reward_tied and hidden_conv:
            label = "reward_diverse_hidden_converged"
        else:
            label = "reward_diverse_hidden_divergent"
        rows.append({**row, "diagnostic_label": label, "hidden_converged": hidden_conv, "dualanchor_indifferent": da_near})
    reward_ties = [row for row in rows if safe_float(row.get("reward_tied_eval_only"), 0.0) > 0]
    hidden_div_ties = [row for row in reward_ties if row.get("diagnostic_label") == "reward_tied_hidden_divergent"]
    hidden_conv_ties = [row for row in reward_ties if row.get("diagnostic_label") == "reward_tied_hidden_converged"]
    science_rows = [row for row in rows if row.get("domain") == "science"]
    reasoning_rows = [row for row in rows if row.get("domain") == "reasoning"]
    science_reward_ties = [row for row in science_rows if safe_float(row.get("reward_tied_eval_only"), 0.0) > 0]
    tie_hidden_div_rate = len(hidden_div_ties) / max(len(reward_ties), 1)
    tie_hidden_conv_rate = len(hidden_conv_ties) / max(len(reward_ties), 1)
    science_tie_rate = len(science_reward_ties) / max(len(science_rows), 1)
    if not rows:
        verdict = "INCONCLUSIVE"
    elif tie_hidden_div_rate >= 0.50:
        verdict = "TIES_HIDE_LATENT_DIVERSITY"
    elif science_tie_rate >= 0.80 and finite_mean(safe_float(row.get("left_final_reward_eval_only"), 0.0) for row in science_reward_ties) <= 0.05:
        verdict = "SCIENCE_TIES_ARE_NO_GOOD_BRANCH"
    elif tie_hidden_conv_rate >= 0.70:
        verdict = "TIES_MOSTLY_TRUE_CONVERGENCE"
    else:
        verdict = "REASONING_TIES_MIXED"
    summary = {
        "BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT": verdict,
        "pair_count": len(rows),
        "reward_tie_count": len(reward_ties),
        "reward_tied_hidden_divergent_rate": tie_hidden_div_rate,
        "reward_tied_hidden_converged_rate": tie_hidden_conv_rate,
        "reward_tied_dualanchor_indifferent_rate": finite_mean(row.get("dualanchor_indifferent") for row in reward_ties),
        "domain_counts": dict(Counter(row.get("domain") for row in rows)),
        "label_counts": dict(Counter(row.get("diagnostic_label") for row in rows)),
        "science_reward_tie_rate": science_tie_rate,
        "reasoning_reward_tie_rate": sum(1 for row in reasoning_rows if safe_float(row.get("reward_tied_eval_only"), 0.0) > 0) / max(len(reasoning_rows), 1),
        "classification_runtime_use": "forbidden; diagnostic-only labels are not runtime architecture inputs",
        "elapsed_seconds": round(time.time() - started, 3),
        "examples": rows[:20],
    }
    write_json(TIE_JSON, summary)
    write_csv(OUT_ROOT / "tie_decomposition_rows.csv", rows)
    lines = [
        "# Tie Decomposition Diagnostic v1",
        "",
        status_line("BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT", verdict),
        "",
        "This is diagnostic-only. It does not introduce branch classification into the architecture.",
        "",
        "## Metrics",
        "",
        f"- reward-tied pairs: `{len(reward_ties)}` / `{len(rows)}`",
        f"- reward-tied but hidden-divergent rate: `{tie_hidden_div_rate:.4f}`",
        f"- reward-tied and hidden-converged rate: `{tie_hidden_conv_rate:.4f}`",
        f"- reward-tied DualAnchor-indifferent rate: `{summary['reward_tied_dualanchor_indifferent_rate']}`",
        f"- science reward tie rate: `{science_tie_rate:.4f}`",
        f"- reasoning reward tie rate: `{summary['reasoning_reward_tie_rate']:.4f}`",
        "",
        "## Diagnostic Label Counts",
        "",
        *md_table([{"label": k, "count": v} for k, v in sorted(summary["label_counts"].items())], ["label", "count"]),
        "",
        "## Examples",
        "",
        *md_table(rows[:20], ["task_id", "domain", "hair_stage", "diagnostic_label", "hidden_cosine_distance", "hidden_rms_normalized", "dualanchor_abs_avg_margin", "reward_tied_eval_only"]),
    ]
    write_md(OUT_ROOT / "tie_decomposition_diagnostic.md", lines)
    print(status_line("BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

