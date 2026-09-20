"""Write Stage 2 recommendation, docs, and final summary for the trajectory sweep."""
from __future__ import annotations

import math
import time

from bg_trajectory_prediction_lib import REPORT_ROOT, append_once, load_json, md_table, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "stage2_recommendation.json"
OUT_MD = REPORT_ROOT / "stage2_recommendation.md"
SUMMARY_JSON = REPORT_ROOT / "summary.json"
SUMMARY_MD = REPORT_ROOT / "summary.md"
DOC_PATH = "docs/evaluator/bg_trajectory_prediction_sweep.md"


def _verdict(path: str, key: str) -> str:
    data = load_json(REPORT_ROOT / path, {})
    return str(data.get(key) or data.get("verdict") or "NOT_RUN")


def _metric(value: object) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return ""
    return f"{f:.3f}"


def _strength(row: dict) -> float:
    pair = row.get("pairwise_accuracy", row.get("pairwise_predictive_accuracy"))
    pair_component = -1.0
    if pair is not None:
        try:
            pair_component = float(pair) - 0.5
        except Exception:
            pair_component = -1.0
    return max(float(row.get("top1_lift") or 0.0), float(row.get("top2_lift") or 0.0), pair_component)


def _compact_cell(cell: object) -> dict[str, object] | str:
    if not isinstance(cell, dict):
        return str(cell)
    return {
        "domain": cell.get("domain"),
        "prefix_length": cell.get("prefix_length"),
        "head_id": cell.get("head_id"),
        "config": cell.get("config"),
        "architecture": cell.get("architecture"),
        "top1_lift": cell.get("top1_lift"),
        "top2_lift": cell.get("top2_lift"),
        "pairwise_accuracy": cell.get("pairwise_predictive_accuracy"),
        "oracle_success": cell.get("oracle_success"),
        "n_tasks": cell.get("n_tasks"),
        "n_pairwise_comparisons": cell.get("n_pairwise_comparisons"),
    }


def _compact_target(target: object) -> dict[str, object] | str:
    if not isinstance(target, dict):
        return str(target)
    return {
        "domain": target.get("domain"),
        "prefix_length": target.get("prefix_length"),
        "head_id": target.get("head_id"),
        "head_config": target.get("head_config"),
        "architecture": target.get("architecture"),
        "top1_lift": target.get("top1_lift"),
        "top2_lift": target.get("top2_lift"),
        "pairwise_accuracy": target.get("pairwise_accuracy"),
        "oracle_success": target.get("oracle_success"),
    }


def _heatmap_rows(rows: list[dict]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        out.append(
            {
                "domain": row.get("domain"),
                "prefix": row.get("prefix_length"),
                "best_head": row.get("best_head_id"),
                "config": row.get("best_config"),
                "top1_lift": _metric(row.get("top1_lift")),
                "top2_lift": _metric(row.get("top2_lift")),
                "pair_acc": _metric(row.get("pairwise_accuracy")),
                "oracle": _metric(row.get("oracle_success")),
                "n": row.get("n_tasks"),
            }
        )
    return out


def _top_cell_rows(rows: list[dict], limit: int = 12) -> list[dict[str, object]]:
    out = []
    for row in rows[:limit]:
        out.append(
            {
                "domain": row.get("domain"),
                "prefix": row.get("prefix_length"),
                "head": row.get("head_id"),
                "config": row.get("config"),
                "top1_lift": _metric(row.get("top1_lift")),
                "top2_lift": _metric(row.get("top2_lift")),
                "pair_acc": _metric(row.get("pairwise_predictive_accuracy")),
                "oracle": _metric(row.get("oracle_success")),
                "n": row.get("n_tasks"),
            }
        )
    return out


def _best_by(rows: list[dict], key: str) -> list[dict[str, object]]:
    grouped: dict[object, dict] = {}
    for row in rows:
        value = row.get(key)
        if value not in grouped or _strength(row) > _strength(grouped[value]):
            grouped[value] = row
    out = []
    for value, row in sorted(grouped.items(), key=lambda item: str(item[0])):
        out.append(
            {
                key: value,
                "domain": row.get("domain"),
                "prefix": row.get("prefix_length"),
                "head": row.get("head_id"),
                "config": row.get("config", row.get("best_config")),
                "top1_lift": _metric(row.get("top1_lift")),
                "top2_lift": _metric(row.get("top2_lift")),
                "pair_acc": _metric(row.get("pairwise_predictive_accuracy", row.get("pairwise_accuracy"))),
                "oracle": _metric(row.get("oracle_success")),
            }
        )
    return out


def main() -> int:
    started = time.time()
    preflight = _verdict("preflight.json", "BG_TRAJECTORY_PREFLIGHT_VERDICT")
    suite = _verdict("task_suite.json", "BG_TRAJECTORY_TASK_SUITE_VERDICT")
    partials = _verdict("partials.json", "BG_TRAJECTORY_PARTIALS_VERDICT")
    continuation = _verdict("continued_prefixes.json", "BG_TRAJECTORY_CONTINUATION_VERDICT")
    features = _verdict("prefix_features_index.json", "BG_TRAJECTORY_PREFIX_FEATURE_VERDICT")
    scores = _verdict("prefix_scores.json", "BG_TRAJECTORY_PREFIX_SCORE_VERDICT")
    predictive = load_json(REPORT_ROOT / "predictive_power.json", {})
    prediction_verdict = str(predictive.get("BG_TRAJECTORY_PREDICTION_VERDICT") or predictive.get("verdict") or "NOT_RUN")
    best = predictive.get("BEST_PREDICTIVE_CELL", "NONE")
    target = predictive.get("RECOMMENDED_STEERING_TARGET", "NONE")
    generator_limited = bool(predictive.get("GENERATOR_REACHABILITY_LIMITED"))
    heatmap = list(predictive.get("heatmap_by_domain_prefix") or [])
    top_cells = list(predictive.get("top_cells") or [])
    compact_best = _compact_cell(best)
    compact_target = _compact_target(target)
    heat_rows = _heatmap_rows(heatmap)
    top_rows = _top_cell_rows(top_cells)
    prefix_rows = _best_by(top_cells, "prefix_length")
    config_rows = _best_by(top_cells, "config")
    domain_rows = _best_by(top_cells, "domain")

    if prediction_verdict == "STRONG":
        recommended_next = "run_targeted_BG_steering_sensitivity_probe_at_best_cell"
        recommendation = (
            "Run a targeted Stage 2 steering-sensitivity probe at the best predictive cell. "
            "Measure state movement in the BG-readable direction, output stability, final correctness, "
            "and positive-vs-negative-vs-random controls."
        )
    elif prediction_verdict == "WEAK_POSITIVE":
        recommended_next = "either_expand_prediction_sweep_or_run_small_targeted_steering_probe"
        recommendation = (
            "Either expand the prediction sweep for more power or run a small targeted steering probe at the weak-positive cell. "
            "Treat success as causal-sensitivity evidence only if controls separate cleanly."
        )
    elif prediction_verdict == "NEUTRAL":
        recommended_next = "improve_candidate_generation_or_routing_calibration_before_steering"
        recommendation = "Do not run steering yet as an architectural claim; improve branch diversity or routing calibration first."
    else:
        recommended_next = "fix_reachability_or_eval_data_before_steering"
        recommendation = "Do not run steering yet; fix reachability/evaluation coverage or gather more usable trajectories first."

    payload = {
        "BG_TRAJECTORY_PREDICTION_VERDICT": prediction_verdict,
        "BEST_PREDICTIVE_CELL": compact_best,
        "RECOMMENDED_STEERING_TARGET": compact_target,
        "GENERATOR_REACHABILITY_LIMITED": generator_limited,
        "RECOMMENDED_NEXT": recommended_next,
        "recommendation": recommendation,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_md(
        OUT_MD,
        [
            "# BG Stage 2 Steering Recommendation (2026-05-18)",
            "",
            f"BG_TRAJECTORY_PREDICTION_VERDICT = {prediction_verdict}",
            f"BEST_PREDICTIVE_CELL = `{compact_best}`",
            f"RECOMMENDED_STEERING_TARGET = `{compact_target}`",
            f"GENERATOR_REACHABILITY_LIMITED = {str(generator_limited).lower()}",
            f"RECOMMENDED_NEXT = {recommended_next}",
            "",
            recommendation,
            "",
            "No steering was run in this script.",
        ],
    )

    summary = {
        "BG_TRAJECTORY_PREFLIGHT_VERDICT": preflight,
        "BG_TRAJECTORY_TASK_SUITE_VERDICT": suite,
        "BG_TRAJECTORY_PARTIALS_VERDICT": partials,
        "BG_TRAJECTORY_CONTINUATION_VERDICT": continuation,
        "BG_TRAJECTORY_PREFIX_FEATURE_VERDICT": features,
        "BG_TRAJECTORY_PREFIX_SCORE_VERDICT": scores,
        "BG_TRAJECTORY_PREDICTION_VERDICT": prediction_verdict,
        "BEST_PREDICTIVE_CELL": compact_best,
        "RECOMMENDED_STEERING_TARGET": compact_target,
        "GENERATOR_REACHABILITY_LIMITED": generator_limited,
        "RECOMMENDED_NEXT": recommended_next,
        "report_paths": {
            "preflight": rel(REPORT_ROOT / "preflight.md"),
            "task_suite": rel(REPORT_ROOT / "task_suite.md"),
            "partials": rel(REPORT_ROOT / "partials.md"),
            "continued_prefixes": rel(REPORT_ROOT / "continued_prefixes.md"),
            "prefix_features": rel(REPORT_ROOT / "prefix_features.md"),
            "prefix_scores": rel(REPORT_ROOT / "prefix_scores.md"),
            "predictive_power": rel(REPORT_ROOT / "predictive_power.md"),
            "stage2_recommendation": rel(OUT_MD),
        },
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/bg_trajectory_prediction_preflight.py",
            "venv/bin/python -m py_compile utilities/tests/manual/build_bg_trajectory_task_suite.py",
            "venv/bin/python -m py_compile utilities/tests/manual/generate_bg_trajectory_partials.py",
            "venv/bin/python -m py_compile utilities/tests/manual/continue_bg_trajectory_prefixes.py",
            "venv/bin/python -m py_compile utilities/tests/manual/capture_bg_trajectory_prefix_features.py",
            "venv/bin/python -m py_compile utilities/tests/manual/score_bg_trajectory_prefixes.py",
            "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_trajectory_predictive_power.py",
            "venv/bin/python -m py_compile utilities/tests/manual/write_bg_stage2_steering_recommendation.py",
            "venv/bin/python -m compileall -q src/evaluator utilities/tests/manual",
            "venv/bin/python utilities/tests/manual/test_bg_controller_unit.py",
            "env BG_SKIP_LIVE_MODEL=1 venv/bin/python utilities/tests/manual/test_bg_transformer_features_unit.py",
            "venv/bin/python -u utilities/tests/manual/bg_trajectory_prediction_preflight.py",
            "venv/bin/python -u utilities/tests/manual/build_bg_trajectory_task_suite.py",
            "venv/bin/python -u utilities/tests/manual/generate_bg_trajectory_partials.py",
            "venv/bin/python -u utilities/tests/manual/continue_bg_trajectory_prefixes.py",
            "venv/bin/python -u utilities/tests/manual/capture_bg_trajectory_prefix_features.py",
            "venv/bin/python -u utilities/tests/manual/score_bg_trajectory_prefixes.py",
            "venv/bin/python -u utilities/tests/manual/analyze_bg_trajectory_predictive_power.py",
            "venv/bin/python -u utilities/tests/manual/write_bg_stage2_steering_recommendation.py",
        ],
        "files_modified_or_created": [
            "shared/utilities/tests/manual/bg_trajectory_prediction_lib.py",
            "shared/utilities/tests/manual/bg_trajectory_prediction_preflight.py",
            "shared/utilities/tests/manual/build_bg_trajectory_task_suite.py",
            "shared/utilities/tests/manual/generate_bg_trajectory_partials.py",
            "shared/utilities/tests/manual/continue_bg_trajectory_prefixes.py",
            "shared/utilities/tests/manual/capture_bg_trajectory_prefix_features.py",
            "shared/utilities/tests/manual/score_bg_trajectory_prefixes.py",
            "shared/utilities/tests/manual/analyze_bg_trajectory_predictive_power.py",
            "shared/utilities/tests/manual/write_bg_stage2_steering_recommendation.py",
            DOC_PATH,
            "docs/evaluator/current_state.md",
            "docs/evaluator/domain_transfer_ledger.md",
            "docs/evaluator/bg_steering_suite.md",
            "docs/evaluator/bg_transformer_integration.md",
            "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
            rel(REPORT_ROOT),
        ],
        "blockers": [],
    }
    if generator_limited:
        summary["blockers"].append("generator reachability was limited for the predictive sweep")
    write_json(SUMMARY_JSON, summary)

    lines = [
        "# BG Trajectory Prediction Sweep Summary (2026-05-18)",
        "",
        f"BG_TRAJECTORY_PREFLIGHT_VERDICT = {preflight}",
        f"BG_TRAJECTORY_TASK_SUITE_VERDICT = {suite}",
        f"BG_TRAJECTORY_PARTIALS_VERDICT = {partials}",
        f"BG_TRAJECTORY_CONTINUATION_VERDICT = {continuation}",
        f"BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = {features}",
        f"BG_TRAJECTORY_PREFIX_SCORE_VERDICT = {scores}",
        f"BG_TRAJECTORY_PREDICTION_VERDICT = {prediction_verdict}",
        f"BEST_PREDICTIVE_CELL = `{compact_best}`",
        f"RECOMMENDED_STEERING_TARGET = `{compact_target}`",
        f"GENERATOR_REACHABILITY_LIMITED = {str(generator_limited).lower()}",
        f"RECOMMENDED_NEXT = {recommended_next}",
        "",
        "## 1. Experimental design",
        "",
        "The sweep generated shared direct Ouro-RLTT trajectories, sliced token-prefix checkpoints, continued every prefix under matched budgets, and scored prefix features with existing BG heads. Labels were used only after continuation for analysis.",
        "",
        "## 2. Task suite",
        "",
        f"Task suite verdict: `{suite}`. Report: `{summary['report_paths']['task_suite']}`.",
        "",
        "## 3. Partial generation",
        "",
        f"Partial generation verdict: `{partials}`. Report: `{summary['report_paths']['partials']}`.",
        "",
        "## 4. Continuation/evaluation",
        "",
        f"Continuation verdict: `{continuation}`. Report: `{summary['report_paths']['continued_prefixes']}`.",
        "",
        "## 5. Feature capture",
        "",
        f"Prefix feature verdict: `{features}`. Report: `{summary['report_paths']['prefix_features']}`.",
        "",
        "## 6. Prefix scoring",
        "",
        f"Prefix score verdict: `{scores}`. Report: `{summary['report_paths']['prefix_scores']}`.",
        "",
        "## 7. Predictive heatmaps",
        "",
        f"Predictive power report: `{summary['report_paths']['predictive_power']}`.",
        "",
        *md_table(heat_rows, ["domain", "prefix", "best_head", "config", "top1_lift", "top2_lift", "pair_acc", "oracle", "n"]),
        "",
        "## 8. Prefix-length trend",
        "",
        *md_table(prefix_rows, ["prefix_length", "domain", "head", "config", "top1_lift", "top2_lift", "pair_acc", "oracle"]),
        "",
        "## 9. Layer/config trend",
        "",
        *md_table(config_rows, ["config", "domain", "prefix", "head", "top1_lift", "top2_lift", "pair_acc", "oracle"]),
        "",
        "## 10. Domain trend",
        "",
        *md_table(domain_rows, ["domain", "prefix", "head", "config", "top1_lift", "top2_lift", "pair_acc", "oracle"]),
        "",
        "## 11. Best predictive cell",
        "",
        f"`{compact_best}`",
        "",
        "## 12. Stage-2 recommendation",
        "",
        recommendation,
        "",
        "## 13. Docs updated",
        "",
        f"- `{DOC_PATH}`",
        "- `docs/evaluator/current_state.md`",
        "- `docs/evaluator/domain_transfer_ledger.md`",
        "- `docs/evaluator/bg_steering_suite.md`",
        "- `docs/evaluator/bg_transformer_integration.md`",
        "- `docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md`",
        "",
        "## 14. Files modified / created",
        "",
        *[f"- `{path}`" for path in summary["files_modified_or_created"]],
        "",
        "## 15. Commands run",
        "",
        *[f"- `{cmd}`" for cmd in summary["commands_run"]],
        "",
        "## 16. Blockers",
        "",
        *([f"- {item}" for item in summary["blockers"]] if summary["blockers"] else ["- none"]),
    ]
    write_md(SUMMARY_MD, lines)

    doc_lines = [
        "# BG Trajectory Prediction Sweep",
        "",
        "## Purpose",
        "",
        "This read-only Stage 1 sweep asks where BG trajectory signal is predictive before any steering is attempted.",
        "",
        "## Context",
        "",
        "It follows the controller, transformer feature capture, best-of-N smoke, steering/routing suite, and wrapper-matched diagnostic. The wrapper is not used here.",
        "",
        "## Domains",
        "",
        "Headline domains are reasoning MCQ, science MCQ, and GSM8K/simple arithmetic. Code/devil tasks are intentionally excluded from the headline because direct Ouro code reachability was generator-limited.",
        "",
        "## Prefix Lengths",
        "",
        "Each generated trajectory is sliced at 32, 64, 128, and 256 generated-token checkpoints.",
        "",
        "## Configs",
        "",
        "The sweep scores locked heads plus compatible existing config-level heads, including 24_L4, 36_L4, 36_mean, 47_L4, 47_concat_L1_L4, and 47_concat_all_loops where artifacts exist.",
        "",
        "## Fairness Constraints",
        "",
        "All heads score the same generated prefixes. Prefix continuations are generated once and evaluated post-hoc. Answer keys and labels are never included in BG feature text or used during scoring.",
        "",
        "## Reachability Logic",
        "",
        "Oracle success at a prefix means at least one branch from the same task and prefix length continued to a correct final answer.",
        "",
        "## Predictive Summary",
        "",
        f"`BG_TRAJECTORY_PREDICTION_VERDICT = {prediction_verdict}`",
        "",
        f"`BEST_PREDICTIVE_CELL = {compact_best}`",
        "",
        f"`RECOMMENDED_STEERING_TARGET = {compact_target}`",
        "",
        "## Predictive Heatmap Summary",
        "",
        *md_table(heat_rows, ["domain", "prefix", "best_head", "config", "top1_lift", "top2_lift", "pair_acc", "oracle", "n"]),
        "",
        "## Caveats",
        "",
        "This is read-only evidence. It does not establish that steering along a BG readout direction will improve generation. Stage 2 must use positive, negative, random, and zero controls.",
    ]
    write_md(DOC_PATH, doc_lines)

    section_title = "BG trajectory prediction sweep (2026-05-18)"
    section_lines = [
        f"BG_TRAJECTORY_PREFLIGHT_VERDICT = `{preflight}`.",
        f"BG_TRAJECTORY_TASK_SUITE_VERDICT = `{suite}`.",
        f"BG_TRAJECTORY_PARTIALS_VERDICT = `{partials}`.",
        f"BG_TRAJECTORY_CONTINUATION_VERDICT = `{continuation}`.",
        f"BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = `{features}`.",
        f"BG_TRAJECTORY_PREFIX_SCORE_VERDICT = `{scores}`.",
        f"BG_TRAJECTORY_PREDICTION_VERDICT = `{prediction_verdict}`.",
        f"BEST_PREDICTIVE_CELL = `{compact_best}`.",
        f"RECOMMENDED_STEERING_TARGET = `{compact_target}`.",
        f"GENERATOR_REACHABILITY_LIMITED = `{str(generator_limited).lower()}`.",
        f"Interpretation: {recommendation}",
        f"Full reports: `{rel(SUMMARY_MD)}`, `{rel(REPORT_ROOT / 'predictive_power.md')}`, `{rel(OUT_MD)}`.",
    ]
    for doc in [
        "docs/evaluator/current_state.md",
        "docs/evaluator/domain_transfer_ledger.md",
        "docs/evaluator/bg_steering_suite.md",
        "docs/evaluator/bg_transformer_integration.md",
        "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]:
        append_once(doc, section_title, section_lines)

    print(f"BG_TRAJECTORY_PREDICTION_VERDICT = {prediction_verdict}")
    print(f"RECOMMENDED_NEXT = {recommended_next}")
    print(f"Wrote {rel(SUMMARY_JSON)}")
    print(f"Wrote {rel(SUMMARY_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
