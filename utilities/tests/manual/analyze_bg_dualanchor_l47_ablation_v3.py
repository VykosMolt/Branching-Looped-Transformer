from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, grouped_mean, load_l47_rows, md_table, report_lines, safe_float, write_csv, write_json, write_md


def main() -> int:
    rows = load_l47_rows()
    summary = grouped_mean(rows, "ablation", ["oracle_retained_vs_full", "best_reward", "forced_top1_reward", "forced_top1_oracle", "candidate_count"])
    by_name = {row["ablation"]: row for row in summary}
    full = by_name.get("full_l47_enabled", {})
    nonterminal_disabled = by_name.get("nonterminal_l47_disabled_tree_replay", {})
    l2_only = by_name.get("l2_47_only_tree_replay", {})
    if safe_float(full.get("oracle_retained_vs_full"), 0.0) >= 0.99 and safe_float(nonterminal_disabled.get("oracle_retained_vs_full"), 1.0) < 0.50:
        verdict = "FULL_L47_REQUIRED"
    elif safe_float(l2_only.get("oracle_retained_vs_full"), 0.0) > safe_float(nonterminal_disabled.get("oracle_retained_vs_full"), 0.0):
        verdict = "NONTERMINAL_L47_USEFUL"
    else:
        verdict = "REPLAY_ONLY_CAVEAT"
    payload = {
        "BG_DUALANCHOR_L47_ABLATION_V3_VERDICT": verdict,
        "summary": summary,
        "caveat": "Candidate-tree replay over generated lineages; regenerated L47 ablation not run in this stage.",
    }
    write_json(OUT_ROOT / "l47_ablation.json", payload)
    write_csv(OUT_ROOT / "l47_ablation_rows.csv", rows)
    lines = report_lines("DualAnchor L47 Ablation v3", "BG_DUALANCHOR_L47_ABLATION_V3_VERDICT", verdict, [("Summary", md_table(summary, ["ablation", "count", "oracle_retained_vs_full", "best_reward", "forced_top1_reward", "forced_top1_oracle", "candidate_count"])), ("Caveat", [payload["caveat"]])])
    write_md(OUT_ROOT / "l47_ablation.md", lines)
    print(f"BG_DUALANCHOR_L47_ABLATION_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

