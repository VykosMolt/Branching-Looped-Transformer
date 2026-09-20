"""Analyze reasoning trace loop geometry and write the final trace/math-logic summary."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


TRACE_FEATURES = REPORT_DIR / "reasoning_trace_features_2026-05-17.pt"
TRACE_TASK_SET = REPORT_DIR / "reasoning_trace_task_set_2026-05-17.json"
TRACE_DATA = REPORT_DIR / "reasoning_option_traces_2026-05-17.json"
TRACE_TRANSFER = REPORT_DIR / "reasoning_trace_transfer_2026-05-17.json"
EVAL_COMPARISON = REPORT_DIR / "reasoning_eval_type_comparison_2026-05-17.json"
CODE_MATH_LOGIC = REPORT_DIR / "code_taps_on_math_logic_existing_2026-05-17.json"
GEOM_JSON = REPORT_DIR / "reasoning_trace_geometry_2026-05-17.json"
GEOM_MD = REPORT_DIR / "reasoning_trace_geometry_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "reasoning_trace_and_code_taps_math_logic_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "reasoning_trace_and_code_taps_math_logic_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/history/current_state_snapshot_2026-05-17.md",
    PROJECT_ROOT / "docs/evaluator/history/clean_gsm8k_and_code_next_2026-05-16.md",
    PROJECT_ROOT / "docs/evaluator/history/antisymlinear_pivot_2026-05-15.md",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def loop_metrics(tensors: torch.Tensor) -> dict[str, Any]:
    out = {}
    for li, layer in enumerate((24, 36, 47)):
        x = tensors[:, li].to(torch.float32)
        cos = torch.zeros((4, 4), dtype=torch.float32)
        for i in range(4):
            for j in range(4):
                cos[i, j] = F.cosine_similarity(x[:, i], x[:, j], dim=-1).mean()
        off = [float(cos[i, j]) for i in range(4) for j in range(i + 1, 4)]
        out[str(layer)] = {
            "L1_L4_cos": float(cos[0, 3]),
            "L2_L4_cos": float(cos[1, 3]),
            "mean_offdiag_cos": float(mean(off)) if off else float("nan"),
            "min_offdiag_cos": min(off) if off else float("nan"),
        }
    return out


def correct_distractor_distances(payload: dict[str, Any]) -> dict[str, Any]:
    by_uid = {str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32) for row in payload.get("candidate_features", []) or []}
    distances: dict[str, list[float]] = {str(layer): [] for layer in (24, 36, 47)}
    for tour in payload.get("eval_sets", {}).get("reasoning_trace_primary", []) or []:
        uids = [str(uid) for uid in tour.get("candidate_uids", [])]
        labels = list(tour.get("labels", []) or [])
        correct = [uid for uid, label in zip(uids, labels) if label == "correct" and uid in by_uid]
        incorrect = [uid for uid, label in zip(uids, labels) if label != "correct" and uid in by_uid]
        if len(correct) != 1:
            continue
        c = by_uid[correct[0]]
        for uid in incorrect:
            d = by_uid[uid]
            for li, layer in enumerate((24, 36, 47)):
                c_vec = c[li].mean(dim=0)
                d_vec = d[li].mean(dim=0)
                sim = float(F.cosine_similarity(c_vec, d_vec, dim=0))
                distances[str(layer)].append(1.0 - sim)
    return {
        layer: {"mean_cosine_distance": float(mean(vals)) if vals else float("nan"), "n_pairs": len(vals)}
        for layer, vals in distances.items()
    }


def analyze_geometry() -> dict[str, Any]:
    if not TRACE_FEATURES.exists():
        return {
            "reasoning_trace_geometry_verdict": "NOT_RUN",
            "summary": {"REASONING_TRACE_GEOMETRY_VERDICT": "NOT_RUN", "blocker": f"missing {repo_path(TRACE_FEATURES)}"},
        }
    try:
        payload = torch.load(TRACE_FEATURES, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "reasoning_trace_geometry_verdict": "BLOCKED",
            "summary": {"REASONING_TRACE_GEOMETRY_VERDICT": "BLOCKED", "blocker": f"{type(exc).__name__}: {exc}"},
        }
    rows = [row["pooled"].detach().cpu() for row in payload.get("candidate_features", []) or []]
    if not rows:
        return {"reasoning_trace_geometry_verdict": "BLOCKED", "summary": {"REASONING_TRACE_GEOMETRY_VERDICT": "BLOCKED", "blocker": "no feature rows"}}
    tensors = torch.stack(rows, dim=0)
    generated_geom = load_json(REPORT_DIR / "loop_layer_diagnostics_current_domains_2026-05-17.json")
    natural_geom = load_json(REPORT_DIR / "reasoning_natural_distractor_geometry_2026-05-17.json")
    return {
        "reasoning_trace_geometry_verdict": "READY",
        "summary": {
            "REASONING_TRACE_GEOMETRY_VERDICT": "READY",
            "n_vectors": int(tensors.shape[0]),
            "loop_geometry": loop_metrics(tensors),
            "correct_vs_distractor_distances": correct_distractor_distances(payload),
            "generated_reasoning_geometry": generated_geom.get("diagnostics", {}).get("reasoning", generated_geom.get("summary", {}).get("reasoning", {})),
            "natural_distractor_geometry": natural_geom.get("summary", {}),
            "margin_vs_loop_divergence": "not_computed",
        },
        "outputs": {"json": repo_path(GEOM_JSON), "md": repo_path(GEOM_MD)},
    }


def recommended_next(summary: dict[str, Any]) -> str:
    trace_transfer = summary["REASONING_TRACE_TRANSFER_VERDICT"]
    trace_specialist = summary["REASONING_TRACE_SPECIALIST_VERDICT"]
    math_v = summary["CODE_TAPS_ON_MATH_VERDICT"]
    logic_v = summary["CODE_TAPS_ON_LOGIC_VERDICT"]
    data_v = summary["REASONING_TRACE_DATA_VERDICT"]
    if data_v in {"BLOCKED", "TIMEOUT"} or trace_transfer == "NOT_RUN":
        return "improve_reasoning_trace_generation_or_use_BBH_style_tasks"
    if trace_transfer == "GOOD" and trace_specialist == "GENERAL_SUFFICIENT":
        return "keep_reasoning_routed_to_general_head_and_focus_controller_policy"
    if trace_specialist == "SPECIALIST_NEEDED":
        return "train_reasoning_specific_tiny_head_or_route_reasoning_to_code_like_objective_head"
    if math_v == "GOOD" and logic_v == "GOOD":
        return "treat_code_head_as_objective_coherence_specialist_candidate"
    return "improve_reasoning_trace_generation_or_use_BBH_style_tasks"


def one_sentence(summary: dict[str, Any]) -> str:
    if summary["REASONING_TRACE_SPECIALIST_VERDICT"] == "GENERAL_SUFFICIENT":
        return "Generated reasoning traces still favor the general HH readout enough that a reasoning-specific specialist is not justified by this probe."
    if summary["REASONING_TRACE_SPECIALIST_VERDICT"] == "SPECIALIST_NEEDED":
        return "Generated reasoning traces expose a specialist gap, so a reasoning-specific or objective-coherence head should be tested next."
    return "The reasoning trace probe is not decisive enough to change routing; keep it as an open branch-selection diagnostic."


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Reasoning trace near-miss + code taps on math/logic (2026-05-17)"
    text = "\n".join([
        "",
        title,
        "",
        f"- REASONING_TRACE_TASK_SET_VERDICT: `{summary['REASONING_TRACE_TASK_SET_VERDICT']}`",
        f"- REASONING_TRACE_DATA_VERDICT: `{summary['REASONING_TRACE_DATA_VERDICT']}`",
        f"- REASONING_TRACE_FEATURE_VERDICT: `{summary['REASONING_TRACE_FEATURE_VERDICT']}`",
        f"- REASONING_TRACE_TRANSFER_VERDICT: `{summary['REASONING_TRACE_TRANSFER_VERDICT']}`",
        f"- REASONING_TRACE_SPECIALIST_VERDICT: `{summary['REASONING_TRACE_SPECIALIST_VERDICT']}`",
        f"- REASONING_TRACE_DIFFICULTY_VERDICT: `{summary['REASONING_TRACE_DIFFICULTY_VERDICT']}`",
        f"- CODE_TAPS_ON_MATH_VERDICT: `{summary['CODE_TAPS_ON_MATH_VERDICT']}`",
        f"- CODE_TAPS_ON_LOGIC_VERDICT: `{summary['CODE_TAPS_ON_LOGIC_VERDICT']}`",
        f"- dataset/task counts: `{summary['dataset_task_counts']}`",
        f"- best HH row: `{summary.get('best_hh')}`",
        f"- best code row: `{summary.get('best_code')}`",
        f"- best NoNorm row: `{summary.get('best_nonorm')}`",
        f"- best AntisymLinear row: `{summary.get('best_antisymlinear')}`",
        f"- comparison to natural distractor reasoning: `{summary.get('reasoning_eval_type_comparison')}`",
        f"- comparison to clean GSM8K/code taps: `{summary.get('code_taps_summary')}`",
        "- full reports: `opi/taps/probes/reasoning_trace_and_code_taps_math_logic_2026-05-17_summary.md`, `opi/taps/probes/reasoning_trace_transfer_2026-05-17.md`, `opi/taps/probes/code_taps_on_math_logic_existing_2026-05-17.md`",
        f"- interpretation: {summary['one_sentence_interpretation']}",
        "",
    ])
    updated = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            updated.append(repo_path(path))
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        updated.append(repo_path(path))
    return updated


def write_geometry_md(payload: dict[str, Any]) -> None:
    s = payload.get("summary", {})
    lines = [
        "# Reasoning Trace Geometry",
        "",
        f"REASONING_TRACE_GEOMETRY_VERDICT = {s.get('REASONING_TRACE_GEOMETRY_VERDICT', payload.get('reasoning_trace_geometry_verdict'))}",
        "",
        f"- n_vectors: `{s.get('n_vectors')}`",
        f"- loop_geometry: `{s.get('loop_geometry')}`",
        f"- correct_vs_distractor_distances: `{s.get('correct_vs_distractor_distances')}`",
        f"- natural_distractor_geometry: `{s.get('natural_distractor_geometry')}`",
        f"- margin_vs_loop_divergence: `{s.get('margin_vs_loop_divergence')}`",
        "",
    ]
    GEOM_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Trace And Code Taps Math/Logic Summary",
        "",
        f"REASONING_TRACE_TASK_SET_VERDICT = {s['REASONING_TRACE_TASK_SET_VERDICT']}",
        f"REASONING_TRACE_DATA_VERDICT = {s['REASONING_TRACE_DATA_VERDICT']}",
        f"REASONING_TRACE_FEATURE_VERDICT = {s['REASONING_TRACE_FEATURE_VERDICT']}",
        f"REASONING_TRACE_TRANSFER_VERDICT = {s['REASONING_TRACE_TRANSFER_VERDICT']}",
        f"REASONING_TRACE_SPECIALIST_VERDICT = {s['REASONING_TRACE_SPECIALIST_VERDICT']}",
        f"REASONING_TRACE_DIFFICULTY_VERDICT = {s['REASONING_TRACE_DIFFICULTY_VERDICT']}",
        f"CODE_TAPS_ON_MATH_VERDICT = {s['CODE_TAPS_ON_MATH_VERDICT']}",
        f"CODE_TAPS_ON_LOGIC_VERDICT = {s['CODE_TAPS_ON_LOGIC_VERDICT']}",
        f"RECOMMENDED_NEXT = {s['RECOMMENDED_NEXT']}",
        "",
        "## 1. Reasoning Trace Task Set",
        "",
        f"- dataset/task counts: `{s['dataset_task_counts']}`",
        f"- random_top1_baseline: `{s.get('random_top1_baseline')}`",
        "",
        "## 2. Trace Generation",
        "",
        f"`{s.get('trace_generation_summary')}`",
        "",
        "## 3. Feature Capture",
        "",
        f"- feature verdict: `{s['REASONING_TRACE_FEATURE_VERDICT']}`",
        f"- feature geometry: `{s.get('geometry_summary')}`",
        "",
        "## 4. Trace Transfer Results",
        "",
        f"- best HH row: `{s.get('best_hh')}`",
        f"- best code row: `{s.get('best_code')}`",
        f"- best NoNorm row: `{s.get('best_nonorm')}`",
        f"- best AntisymLinear row: `{s.get('best_antisymlinear')}`",
        "",
        "## 5. Comparison Across Reasoning Eval Types",
        "",
        f"`{s.get('reasoning_eval_type_comparison')}`",
        "",
        "## 6. Code Taps On Math/Logic",
        "",
        f"`{s.get('code_taps_summary')}`",
        "",
        "## 7. Optional Geometry",
        "",
        f"`{s.get('geometry_summary')}`",
        "",
        "## 8. Interpretation",
        "",
        s["one_sentence_interpretation"],
        "",
        "## 9. Docs Updated",
        "",
    ]
    for path in payload.get("docs_updated", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## 10. Files Modified / Created", ""])
    for path in payload.get("files_modified_or_created", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## 11. Commands Run", "", "```bash"])
    lines.extend(payload.get("commands_run", []))
    lines.extend(["```", "", "## 12. Blockers", "", payload.get("blockers", "None."), ""])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    geom = analyze_geometry()
    write_json(GEOM_JSON, geom)
    write_geometry_md(geom)

    task_set = load_json(TRACE_TASK_SET)
    trace_data = load_json(TRACE_DATA)
    trace_transfer = load_json(TRACE_TRANSFER)
    eval_comparison = load_json(EVAL_COMPARISON)
    code_math_logic = load_json(CODE_MATH_LOGIC)
    feature_meta = {}
    if TRACE_FEATURES.exists():
        try:
            feature_meta = torch.load(TRACE_FEATURES, map_location="cpu", weights_only=False).get("meta", {})
        except Exception as exc:
            feature_meta = {"load_error": f"{type(exc).__name__}: {exc}"}

    task_summary = task_set.get("summary", {})
    data_summary = trace_data.get("summary", {})
    transfer_summary = trace_transfer.get("summary", {})
    eval_summary = eval_comparison.get("summary", {})
    code_summary = code_math_logic.get("summary", {})
    summary = {
        "REASONING_TRACE_TASK_SET_VERDICT": task_summary.get("REASONING_TRACE_TASK_SET_VERDICT", task_set.get("reasoning_trace_task_set_verdict", "BLOCKED")),
        "REASONING_TRACE_DATA_VERDICT": data_summary.get("REASONING_TRACE_DATA_VERDICT", trace_data.get("reasoning_trace_data_verdict", "BLOCKED")),
        "REASONING_TRACE_FEATURE_VERDICT": feature_meta.get("reasoning_trace_feature_verdict", "READY" if TRACE_FEATURES.exists() else "BLOCKED").upper(),
        "REASONING_TRACE_TRANSFER_VERDICT": transfer_summary.get("REASONING_TRACE_TRANSFER_VERDICT", trace_transfer.get("reasoning_trace_transfer_verdict", "NOT_RUN")),
        "REASONING_TRACE_SPECIALIST_VERDICT": transfer_summary.get("REASONING_TRACE_SPECIALIST_VERDICT", trace_transfer.get("reasoning_trace_specialist_verdict", "INSUFFICIENT")),
        "REASONING_TRACE_DIFFICULTY_VERDICT": eval_summary.get("REASONING_TRACE_DIFFICULTY_VERDICT", eval_comparison.get("reasoning_trace_difficulty_verdict", "INSUFFICIENT")),
        "CODE_TAPS_ON_MATH_VERDICT": code_summary.get("CODE_TAPS_ON_MATH_VERDICT", code_math_logic.get("code_taps_on_math_verdict", "NOT_RUN")),
        "CODE_TAPS_ON_LOGIC_VERDICT": code_summary.get("CODE_TAPS_ON_LOGIC_VERDICT", code_math_logic.get("code_taps_on_logic_verdict", "NOT_RUN")),
        "dataset_task_counts": task_summary.get("dataset_breakdown", {}),
        "random_top1_baseline": transfer_summary.get("random_top1_baseline", data_summary.get("random_top1_baseline_kept")),
        "trace_generation_summary": data_summary,
        "best_hh": transfer_summary.get("best_hh"),
        "best_code": transfer_summary.get("best_code"),
        "best_nonorm": transfer_summary.get("best_nonorm"),
        "best_antisymlinear": transfer_summary.get("best_antisymlinear"),
        "reasoning_eval_type_comparison": eval_summary,
        "code_taps_summary": code_summary,
        "geometry_summary": geom.get("summary", {}),
    }
    summary["RECOMMENDED_NEXT"] = recommended_next(summary)
    summary["one_sentence_interpretation"] = one_sentence(summary)
    docs_updated = append_docs(summary)
    files = [
        "shared/utilities/tests/manual/build_reasoning_trace_task_set.py",
        "shared/utilities/tests/manual/generate_reasoning_option_traces.py",
        "shared/utilities/tests/manual/capture_reasoning_trace_features.py",
        "shared/utilities/tests/manual/evaluate_heads_on_reasoning_traces.py",
        "shared/utilities/tests/manual/compare_reasoning_eval_types.py",
        "shared/utilities/tests/manual/evaluate_code_taps_on_math_logic_existing.py",
        "shared/utilities/tests/manual/analyze_reasoning_trace_geometry.py",
        repo_path(TRACE_TASK_SET),
        repo_path(TRACE_DATA),
        repo_path(TRACE_FEATURES),
        repo_path(TRACE_TRANSFER),
        repo_path(EVAL_COMPARISON),
        repo_path(CODE_MATH_LOGIC),
        repo_path(GEOM_JSON),
        repo_path(GEOM_MD),
        repo_path(SUMMARY_JSON),
        repo_path(SUMMARY_MD),
    ] + docs_updated
    commands = [
        "venv/bin/python -m py_compile utilities/tests/manual/build_reasoning_trace_task_set.py",
        "venv/bin/python -m py_compile utilities/tests/manual/generate_reasoning_option_traces.py",
        "venv/bin/python -m py_compile utilities/tests/manual/capture_reasoning_trace_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_heads_on_reasoning_traces.py",
        "venv/bin/python -m py_compile utilities/tests/manual/compare_reasoning_eval_types.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_code_taps_on_math_logic_existing.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_reasoning_trace_geometry.py",
        "venv/bin/python -u utilities/tests/manual/build_reasoning_trace_task_set.py --target-total 30 --min-total 20 --max-total 40",
        "venv/bin/python -u utilities/tests/manual/generate_reasoning_option_traces.py --input opi/taps/probes/reasoning_trace_task_set_2026-05-17.json --output opi/taps/probes/reasoning_option_traces_2026-05-17.json --device cuda > opi/taps/probes/reasoning_option_traces_2026-05-17.log 2>&1",
        "venv/bin/python -u utilities/tests/manual/capture_reasoning_trace_features.py --input opi/taps/probes/reasoning_option_traces_2026-05-17.json --output opi/taps/probes/reasoning_trace_features_2026-05-17.pt --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_heads_on_reasoning_traces.py --features opi/taps/probes/reasoning_trace_features_2026-05-17.pt --heads opi/taps/probes/bg_head_registry_2026-05-17.pt --input opi/taps/probes/reasoning_option_traces_2026-05-17.json --output opi/taps/probes/reasoning_trace_transfer_2026-05-17.json",
        "venv/bin/python -u utilities/tests/manual/compare_reasoning_eval_types.py",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_taps_on_math_logic_existing.py",
        "venv/bin/python -u utilities/tests/manual/analyze_reasoning_trace_geometry.py",
    ]
    blockers = []
    if summary["REASONING_TRACE_FEATURE_VERDICT"] in {"BLOCKED", "TIMEOUT"}:
        blockers.append("Reasoning trace feature capture did not complete.")
    if code_summary.get("blockers"):
        blockers.extend(code_summary.get("blockers", []))
    payload = {
        "summary": summary,
        "docs_updated": docs_updated,
        "files_modified_or_created": files,
        "commands_run": commands,
        "blockers": "None." if not blockers else "; ".join(blockers),
        "outputs": {"json": repo_path(SUMMARY_JSON), "md": repo_path(SUMMARY_MD)},
    }
    write_json(SUMMARY_JSON, payload)
    write_summary_md(payload)
    print(f"REASONING_TRACE_GEOMETRY_VERDICT = {geom.get('summary', {}).get('REASONING_TRACE_GEOMETRY_VERDICT')}")
    print(f"REASONING_TRACE_TRANSFER_VERDICT = {summary['REASONING_TRACE_TRANSFER_VERDICT']}")
    print(f"CODE_TAPS_ON_MATH_VERDICT = {summary['CODE_TAPS_ON_MATH_VERDICT']}")
    print(f"CODE_TAPS_ON_LOGIC_VERDICT = {summary['CODE_TAPS_ON_LOGIC_VERDICT']}")
    print(f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}")
    print(f"Wrote {GEOM_JSON}")
    print(f"Wrote {GEOM_MD}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
