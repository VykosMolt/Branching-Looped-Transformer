from __future__ import annotations

from collections import defaultdict
from itertools import product

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, finite_mean, grouped_mean, load_task_rows, load_terminal_rows, md_table, read_csv, report_lines, safe_float, write_csv, write_json, write_md


def _task_map() -> dict[str, dict[str, str]]:
    return {row["task_id"]: row for row in load_task_rows()}


def main() -> int:
    task_by_id = _task_map()
    forced_rows = [row for row in load_terminal_rows() if row.get("policy") == "dualanchor_forced_top1"]
    enriched = []
    for row in forced_rows:
        task = task_by_id.get(row.get("task_id"), {})
        enriched.append({**row, **{f"task_{k}": v for k, v in task.items()}})
    grid_rows = []
    for margin, corr_min, agreement_required in product([0.10, 0.15, 0.20, 0.25, 0.30, 0.80, 1.20], [0.3, 0.5, 0.7], [True, False]):
        for split_name, pred in (
            ("calibration", lambda r: r.get("task_split") in {"val", "calibration"}),
            ("heldout", lambda r: r.get("task_split") == "heldout"),
            ("all", lambda r: True),
        ):
            rows = [r for r in enriched if pred(r)]
            accepted = [
                r
                for r in rows
                if safe_float(r.get("terminal_margin"), -999) >= margin
                and safe_float(r.get("terminal_corr"), -999) >= corr_min
                and (not agreement_required or safe_float(r.get("terminal_top1_disagreement"), 1.0) <= 0)
            ]
            hard = [r for r in rows if safe_float(r.get("task_positive_oracle"), 0.0) > 0 and safe_float(r.get("task_terminal_reward_diverse"), 0.0) > 0]
            hard_accepted = [r for r in accepted if safe_float(r.get("task_positive_oracle"), 0.0) > 0 and safe_float(r.get("task_terminal_reward_diverse"), 0.0) > 0]
            grid_rows.append(
                {
                    "split": split_name,
                    "margin": margin,
                    "corr_min": corr_min,
                    "agreement_required": agreement_required,
                    "task_count": len(rows),
                    "accepted_count": len(accepted),
                    "confident_rate": len(accepted) / len(rows) if rows else 0.0,
                    "confident_oracle": finite_mean(r.get("first_selected_oracle") for r in accepted),
                    "confident_reward": finite_mean(r.get("first_selected_reward") for r in accepted),
                    "hard_count": len(hard),
                    "hard_accepted_count": len(hard_accepted),
                    "hard_confident_oracle": finite_mean(r.get("first_selected_oracle") for r in hard_accepted),
                    "hard_confident_reward": finite_mean(r.get("first_selected_reward") for r in hard_accepted),
                }
            )
    # Pick on calibration only; conservative tie-break prefers high confident oracle, then coverage.
    cal = [r for r in grid_rows if r["split"] == "calibration" and r["accepted_count"] > 0]
    cal.sort(key=lambda r: (safe_float(r.get("confident_oracle"), 0.0), safe_float(r.get("hard_confident_oracle"), 0.0), safe_float(r.get("confident_rate"), 0.0)), reverse=True)
    selected = cal[0] if cal else {}
    heldout_match = [
        r
        for r in grid_rows
        if r["split"] == "heldout"
        and r["margin"] == selected.get("margin")
        and r["corr_min"] == selected.get("corr_min")
        and r["agreement_required"] == selected.get("agreement_required")
    ]
    heldout = heldout_match[0] if heldout_match else {}
    forced_by_slice = grouped_mean(enriched, "task_domain", ["first_selected_oracle", "first_selected_reward"])
    if safe_float(heldout.get("confident_oracle"), 0.0) >= 0.95 and safe_float(heldout.get("confident_rate"), 0.0) >= 0.25:
        verdict = "TERMINAL_CONFIDENCE_GATED_READY"
    elif safe_float(heldout.get("hard_confident_oracle"), 0.0) < 0.85:
        verdict = "TERMINAL_WEAK_ON_HARD_SLICE"
    else:
        verdict = "TERMINAL_DEFER_REQUIRED"
    payload = {
        "BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT": verdict,
        "selected_on_calibration": selected,
        "heldout_evaluation": heldout,
        "forced_by_domain": forced_by_slice,
        "threshold_rows": grid_rows,
        "note": "Thresholds selected on calibration/val only; heldout rows are evaluation only.",
    }
    write_json(OUT_ROOT / "terminal_confidence.json", payload)
    write_csv(OUT_ROOT / "terminal_confidence_rows.csv", grid_rows)
    lines = report_lines(
        "DualAnchor Terminal Confidence v3",
        "BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT",
        verdict,
        [
            ("Selected Calibration Threshold", [f"- {k}: `{v}`" for k, v in selected.items()]),
            ("Heldout Evaluation", [f"- {k}: `{v}`" for k, v in heldout.items()]),
            ("Forced Top1 By Domain", md_table(forced_by_slice, ["task_domain", "count", "first_selected_oracle", "first_selected_reward"])),
        ],
    )
    write_md(OUT_ROOT / "terminal_confidence.md", lines)
    print(f"BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

