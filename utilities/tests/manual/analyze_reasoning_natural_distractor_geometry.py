"""Analyze natural distractor reasoning geometry and write final audit summary."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


FEATURES_PT = REPORT_DIR / "reasoning_natural_distractor_features_2026-05-17.pt"
SET_JSON = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.json"
TRANSFER_JSON = REPORT_DIR / "reasoning_natural_distractor_transfer_2026-05-17.json"
COMPARISON_JSON = REPORT_DIR / "reasoning_generated_vs_distractor_comparison_2026-05-17.json"
GEOM_JSON = REPORT_DIR / "reasoning_natural_distractor_geometry_2026-05-17.json"
GEOM_MD = REPORT_DIR / "reasoning_natural_distractor_geometry_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "reasoning_natural_distractor_audit_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "reasoning_natural_distractor_audit_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/math_and_gsm8k_status.md",
    PROJECT_ROOT / "docs/evaluator/tap_interface.md",
    PROJECT_ROOT / "docs/evaluator/history/antisymlinear_pivot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/history/clean_gsm8k_and_code_next_2026-05-16.md",
    PROJECT_ROOT / "docs/evaluator/history/current_state_snapshot_2026-05-17.md",
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


def correct_distractor_distances(feature_payload: dict[str, Any]) -> dict[str, Any]:
    by_uid = {
        str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32)
        for row in feature_payload.get("candidate_features", []) or []
    }
    distances: dict[str, list[float]] = {str(layer): [] for layer in (24, 36, 47)}
    for tournament in feature_payload.get("eval_sets", {}).get("reasoning_natural_distractors", []) or []:
        uids = [str(uid) for uid in tournament["candidate_uids"]]
        labels = list(tournament["labels"])
        correct = [uid for uid, label in zip(uids, labels) if label == "correct"]
        incorrect = [uid for uid, label in zip(uids, labels) if label != "correct"]
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
        layer: {
            "mean_cosine_distance": float(mean(vals)) if vals else float("nan"),
            "n_pairs": len(vals),
        }
        for layer, vals in distances.items()
    }


def analyze_geometry() -> dict[str, Any]:
    if not FEATURES_PT.exists():
        return {"reasoning_natural_distractor_geometry_verdict": "NOT_RUN", "summary": {"REASONING_DISTRACTOR_GEOMETRY_VERDICT": "NOT_RUN"}}
    payload = torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
    rows = [row["pooled"].detach().cpu() for row in payload.get("candidate_features", []) or []]
    if not rows:
        return {"reasoning_natural_distractor_geometry_verdict": "BLOCKED", "summary": {"REASONING_DISTRACTOR_GEOMETRY_VERDICT": "BLOCKED"}}
    tensors = torch.stack(rows, dim=0)
    previous = load_json(REPORT_DIR / "loop_layer_diagnostics_current_domains_2026-05-17.json")
    result = {
        "reasoning_natural_distractor_geometry_verdict": "READY",
        "summary": {
            "REASONING_DISTRACTOR_GEOMETRY_VERDICT": "READY",
            "n_vectors": int(tensors.shape[0]),
            "loop_geometry": loop_metrics(tensors),
            "correct_vs_distractor_distances": correct_distractor_distances(payload),
            "previous_generated_reasoning_geometry": previous.get("diagnostics", {}).get("reasoning", {}),
            "margin_geometry_correlation": "not_computed",
        },
        "outputs": {"json": repo_path(GEOM_JSON), "md": repo_path(GEOM_MD)},
    }
    return result


def recommended_next(transfer: str, difficulty: str, specialist: str) -> str:
    if transfer == "POOR":
        return "rethink_reasoning_feature_format_or_dataset"
    if transfer == "GOOD" and difficulty == "DISTRACTORS_HARDER":
        return "add_reasoning_as_third_objective_eval_domain_and_design_specialist_check"
    if transfer == "GOOD" and difficulty == "BOTH_EASY":
        return "reasoning_promising_but_need_harder_dataset_or_generated_near_misses"
    if specialist == "GENERAL_SUFFICIENT":
        return "use_HH_general_head_for_reasoning_until_harder_data_says_otherwise"
    if specialist == "SPECIALIST_NEEDED":
        return "train_reasoning_specific_tiny_head_on_distractor_or_generated_reasoning_pairs"
    return "rethink_reasoning_feature_format_or_dataset"


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Hard reasoning natural-distractor validation (2026-05-17)"
    best_hh = summary.get("best_hh", {})
    best_code = summary.get("best_code", {})
    text = "\n".join([
        "",
        title,
        "",
        f"- REASONING_DISTRACTOR_SET_VERDICT: `{summary['REASONING_DISTRACTOR_SET_VERDICT']}`",
        f"- REASONING_DISTRACTOR_FEATURE_VERDICT: `{summary['REASONING_DISTRACTOR_FEATURE_VERDICT']}`",
        f"- REASONING_DISTRACTOR_TRANSFER_VERDICT: `{summary['REASONING_DISTRACTOR_TRANSFER_VERDICT']}`",
        f"- REASONING_SPECIALIST_VERDICT: `{summary['REASONING_SPECIALIST_VERDICT']}`",
        f"- REASONING_DIFFICULTY_VERDICT: `{summary['REASONING_DIFFICULTY_VERDICT']}`",
        f"- dataset/task counts: `{summary['dataset_task_counts']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best HH row: `{best_hh}`",
        f"- best code row: `{best_code}`",
        f"- best NoNorm row: `{summary.get('best_nonorm', {})}`",
        f"- best AntisymLinear row: `{summary.get('best_antisymlinear', {})}`",
        f"- generated-vs-distractor comparison: `{summary.get('generated_vs_distractor', {})}`",
        f"- reasoning third objective eval domain: `{summary['reasoning_third_domain_status']}`",
        f"- reasoning specialist justified yet: `{summary['reasoning_specialist_status']}`",
        "- full reports: `opi/taps/probes/reasoning_natural_distractor_audit_2026-05-17_summary.md`, `opi/taps/probes/reasoning_natural_distractor_transfer_2026-05-17.md`, `opi/taps/probes/reasoning_generated_vs_distractor_comparison_2026-05-17.md`",
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
        "# Reasoning Natural Distractor Geometry",
        "",
        f"REASONING_DISTRACTOR_GEOMETRY_VERDICT = {s.get('REASONING_DISTRACTOR_GEOMETRY_VERDICT', payload.get('reasoning_natural_distractor_geometry_verdict'))}",
        "",
        f"- n_vectors: `{s.get('n_vectors')}`",
        f"- loop_geometry: `{s.get('loop_geometry')}`",
        f"- correct_vs_distractor_distances: `{s.get('correct_vs_distractor_distances')}`",
        f"- previous_generated_reasoning_geometry: `{s.get('previous_generated_reasoning_geometry')}`",
        f"- margin_geometry_correlation: `{s.get('margin_geometry_correlation')}`",
        "",
    ]
    GEOM_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Natural Distractor Audit Summary",
        "",
        f"REASONING_DISTRACTOR_SET_VERDICT = {s['REASONING_DISTRACTOR_SET_VERDICT']}",
        f"REASONING_DISTRACTOR_FEATURE_VERDICT = {s['REASONING_DISTRACTOR_FEATURE_VERDICT']}",
        f"REASONING_DISTRACTOR_TRANSFER_VERDICT = {s['REASONING_DISTRACTOR_TRANSFER_VERDICT']}",
        f"REASONING_SPECIALIST_VERDICT = {s['REASONING_SPECIALIST_VERDICT']}",
        f"REASONING_DIFFICULTY_VERDICT = {s['REASONING_DIFFICULTY_VERDICT']}",
        f"RECOMMENDED_NEXT = {s['RECOMMENDED_NEXT']}",
        "",
        "## 1. Distractor Set Construction",
        "",
        f"- dataset/task counts: `{s['dataset_task_counts']}`",
        f"- n_candidates: `{s['n_candidates']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        "",
        "## 2. Feature Capture Summary",
        "",
        f"- feature verdict: `{s['REASONING_DISTRACTOR_FEATURE_VERDICT']}`",
        f"- feature file: `opi/taps/probes/reasoning_natural_distractor_features_2026-05-17.pt`",
        "",
        "## 3. Natural Distractor Transfer Results",
        "",
        f"- best HH row: `{s['best_hh']}`",
        f"- best code row: `{s['best_code']}`",
        f"- best NoNorm row: `{s.get('best_nonorm')}`",
        f"- best AntisymLinear row: `{s.get('best_antisymlinear')}`",
        "",
        "## 4. Generated-vs-Distractor Comparison",
        "",
        f"`{s.get('generated_vs_distractor')}`",
        "",
        "## 5. Reasoning Geometry Diagnostic",
        "",
        f"`{s.get('geometry_summary')}`",
        "",
        "## 6. Interpretation",
        "",
        s["one_sentence_interpretation"],
        "",
        "## 7. Docs Updated",
        "",
    ]
    for path in payload.get("docs_updated", []):
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## 8. Files Modified / Created",
        "",
    ])
    for path in payload.get("files_modified_or_created", []):
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## 9. Commands Run",
        "",
        "```bash",
    ])
    lines.extend(payload.get("commands_run", []))
    lines.extend([
        "```",
        "",
        "## 10. Blockers",
        "",
        payload.get("blockers", "None."),
        "",
    ])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    geom = analyze_geometry()
    write_json(GEOM_JSON, geom)
    write_geometry_md(geom)

    set_payload = load_json(SET_JSON)
    transfer = load_json(TRANSFER_JSON)
    comparison = load_json(COMPARISON_JSON)
    set_summary = set_payload.get("summary", {})
    transfer_summary = transfer.get("summary", {})
    comparison_summary = comparison.get("summary", {})
    feature_meta = {}
    if FEATURES_PT.exists():
        try:
            feature_meta = torch.load(FEATURES_PT, map_location="cpu", weights_only=False).get("meta", {})
        except Exception as exc:
            feature_meta = {"load_error": f"{type(exc).__name__}: {exc}"}
    transfer_v = transfer_summary.get("REASONING_DISTRACTOR_TRANSFER_VERDICT", transfer.get("reasoning_distractor_transfer_verdict", "NOT_RUN"))
    specialist_v = transfer_summary.get("REASONING_SPECIALIST_VERDICT", transfer.get("reasoning_specialist_verdict", "INSUFFICIENT"))
    difficulty_v = comparison_summary.get("REASONING_DIFFICULTY_VERDICT", comparison.get("reasoning_difficulty_verdict", "INSUFFICIENT"))
    recommended = recommended_next(transfer_v, difficulty_v, specialist_v)
    if difficulty_v == "BOTH_EASY":
        interpretation = "Natural distractors did not sufficiently stress-test reasoning; reasoning remains promising, but a specialist is not justified from this set alone."
    elif difficulty_v == "DISTRACTORS_HARDER":
        interpretation = "Natural distractors reduced performance relative to generated branches and are a better stress test for reasoning readouts."
    else:
        interpretation = "Natural distractor validation was not decisive enough to change the reasoning plan."
    if specialist_v == "SPECIALIST_NEEDED":
        specialist_status = "yes_proxy_gap_observed"
    elif specialist_v == "GENERAL_SUFFICIENT":
        specialist_status = "not_yet_general_head_sufficient"
    else:
        specialist_status = "not_yet"
    summary = {
        "REASONING_DISTRACTOR_SET_VERDICT": set_summary.get("REASONING_DISTRACTOR_SET_VERDICT", set_payload.get("reasoning_distractor_set_verdict", "BLOCKED")),
        "REASONING_DISTRACTOR_FEATURE_VERDICT": feature_meta.get("reasoning_distractor_feature_verdict", "READY" if FEATURES_PT.exists() else "BLOCKED").upper(),
        "REASONING_DISTRACTOR_TRANSFER_VERDICT": transfer_v,
        "REASONING_SPECIALIST_VERDICT": specialist_v,
        "REASONING_DIFFICULTY_VERDICT": difficulty_v,
        "RECOMMENDED_NEXT": recommended,
        "dataset_task_counts": set_summary.get("dataset_breakdown", {}),
        "n_candidates": set_summary.get("n_candidates"),
        "random_top1_baseline": transfer_summary.get("random_top1_baseline", set_summary.get("random_top1_baseline")),
        "best_hh": transfer_summary.get("best_hh"),
        "best_code": transfer_summary.get("best_code"),
        "best_nonorm": transfer_summary.get("best_nonorm"),
        "best_antisymlinear": transfer_summary.get("best_antisymlinear"),
        "generated_vs_distractor": comparison_summary,
        "geometry_summary": geom.get("summary", {}),
        "reasoning_third_domain_status": "still_supported" if transfer_v in {"GOOD", "WEAK"} else "not_supported_by_this_probe",
        "reasoning_specialist_status": specialist_status,
        "one_sentence_interpretation": interpretation,
    }
    docs_updated = append_docs(summary)
    files = [
        "shared/utilities/tests/manual/build_reasoning_natural_distractor_set.py",
        "shared/utilities/tests/manual/capture_reasoning_natural_distractor_features.py",
        "shared/utilities/tests/manual/evaluate_heads_on_reasoning_natural_distractors.py",
        "shared/utilities/tests/manual/compare_reasoning_generated_vs_distractor.py",
        "shared/utilities/tests/manual/analyze_reasoning_natural_distractor_geometry.py",
        repo_path(SET_JSON),
        repo_path(FEATURES_PT),
        repo_path(TRANSFER_JSON),
        repo_path(COMPARISON_JSON),
        repo_path(GEOM_JSON),
        repo_path(GEOM_MD),
        repo_path(SUMMARY_JSON),
        repo_path(SUMMARY_MD),
    ] + docs_updated
    commands = [
        "venv/bin/python -m py_compile utilities/tests/manual/build_reasoning_natural_distractor_set.py",
        "venv/bin/python -m py_compile utilities/tests/manual/capture_reasoning_natural_distractor_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_heads_on_reasoning_natural_distractors.py",
        "venv/bin/python -m py_compile utilities/tests/manual/compare_reasoning_generated_vs_distractor.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_reasoning_natural_distractor_geometry.py",
        "venv/bin/python -u utilities/tests/manual/build_reasoning_natural_distractor_set.py --target-arc 30 --target-secondary 30 --max-total 80",
        "venv/bin/python -u utilities/tests/manual/capture_reasoning_natural_distractor_features.py --input opi/taps/probes/reasoning_natural_distractor_set_2026-05-17.json --output opi/taps/probes/reasoning_natural_distractor_features_2026-05-17.pt --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_heads_on_reasoning_natural_distractors.py --features opi/taps/probes/reasoning_natural_distractor_features_2026-05-17.pt --heads opi/taps/probes/bg_head_registry_2026-05-17.pt --input opi/taps/probes/reasoning_natural_distractor_set_2026-05-17.json --output opi/taps/probes/reasoning_natural_distractor_transfer_2026-05-17.json",
        "venv/bin/python -u utilities/tests/manual/compare_reasoning_generated_vs_distractor.py",
        "venv/bin/python -u utilities/tests/manual/analyze_reasoning_natural_distractor_geometry.py",
    ]
    payload = {
        "summary": summary,
        "docs_updated": docs_updated,
        "files_modified_or_created": files,
        "commands_run": commands,
        "blockers": "None.",
        "outputs": {"json": repo_path(SUMMARY_JSON), "md": repo_path(SUMMARY_MD)},
    }
    write_json(SUMMARY_JSON, payload)
    write_summary_md(payload)
    print(f"REASONING_DISTRACTOR_GEOMETRY_VERDICT = {geom.get('summary', {}).get('REASONING_DISTRACTOR_GEOMETRY_VERDICT')}")
    print(f"REASONING_NATURAL_DISTRACTOR_SUMMARY = {recommended}")
    print(f"Wrote {GEOM_JSON}")
    print(f"Wrote {GEOM_MD}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
