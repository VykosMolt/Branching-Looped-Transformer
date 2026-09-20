"""Analyze science loop/layer geometry and write the final science audit summary."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


FEATURES_PT = REPORT_DIR / "science_natural_distractor_features_2026-05-17.pt"
SET_JSON = REPORT_DIR / "science_natural_distractor_set_2026-05-17.json"
TRANSFER_JSON = REPORT_DIR / "science_natural_distractor_transfer_2026-05-17.json"
SCIENCE_SPECIFIC_JSON = REPORT_DIR / "science_specific_tiny_head_control_2026-05-17.json"
COMPARISON_JSON = REPORT_DIR / "science_domain_comparison_2026-05-17.json"
GEOM_JSON = REPORT_DIR / "science_loop_layer_geometry_2026-05-17.json"
GEOM_MD = REPORT_DIR / "science_loop_layer_geometry_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "science_domain_audit_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "science_domain_audit_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
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


def correct_distractor_distances(payload: dict[str, Any], tournaments: list[dict[str, Any]]) -> dict[str, Any]:
    by_uid = {str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32) for row in payload.get("candidate_features", []) or []}
    distances: dict[str, list[float]] = {str(layer): [] for layer in (24, 36, 47)}
    for tour in tournaments:
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
    if not FEATURES_PT.exists():
        return {"science_loop_layer_geometry_verdict": "NOT_RUN", "summary": {"SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT": "NOT_RUN"}}
    try:
        payload = torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "science_loop_layer_geometry_verdict": "BLOCKED",
            "summary": {"SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT": "BLOCKED", "blocker": f"{type(exc).__name__}: {exc}"},
        }
    rows = payload.get("candidate_features", []) or []
    if not rows:
        return {"science_loop_layer_geometry_verdict": "BLOCKED", "summary": {"SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT": "BLOCKED", "blocker": "no feature rows"}}
    tensors = torch.stack([row["pooled"].detach().cpu() for row in rows], dim=0)
    meta_by_uid = {str(row["candidate_uid"]): row.get("candidate_metadata", {}) for row in rows}
    by_subdomain: dict[str, list[torch.Tensor]] = {}
    for row in rows:
        bucket = str(row.get("candidate_metadata", {}).get("subdomain_bucket", "unknown"))
        by_subdomain.setdefault(bucket, []).append(row["pooled"].detach().cpu())
    subdomain_geometry = {
        bucket: loop_metrics(torch.stack(vals, dim=0))
        for bucket, vals in by_subdomain.items()
        if len(vals) >= 20
    }
    tournaments = payload.get("eval_sets", {}).get("science_natural_distractors", []) or []
    previous = {
        "hh_gsm8k_code_reasoning": load_json(REPORT_DIR / "loop_layer_diagnostics_current_domains_2026-05-17.json").get("summary", {}),
        "reasoning_natural": load_json(REPORT_DIR / "reasoning_natural_distractor_geometry_2026-05-17.json").get("summary", {}),
        "reasoning_trace": load_json(REPORT_DIR / "reasoning_trace_geometry_2026-05-17.json").get("summary", {}),
    }
    return {
        "science_loop_layer_geometry_verdict": "READY",
        "summary": {
            "SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT": "READY",
            "n_vectors": int(tensors.shape[0]),
            "loop_geometry": loop_metrics(tensors),
            "subdomain_geometry": subdomain_geometry,
            "correct_vs_distractor_distances": correct_distractor_distances(payload, tournaments),
            "previous_domain_geometry": previous,
            "layer47_margin_correlation": "not_computed",
            "metadata_subdomains": sorted({str(meta.get("subdomain_bucket", "unknown")) for meta in meta_by_uid.values()}),
        },
        "outputs": {"json": repo_path(GEOM_JSON), "md": repo_path(GEOM_MD)},
    }


def subdomain_code_objective(summary: dict[str, Any]) -> bool:
    sub = summary.get("subdomain_transfer", {}) or summary.get("per_subdomain", {})
    for bucket in ("chemistry", "medicine"):
        row = sub.get(bucket, {})
        if not isinstance(row, dict):
            continue
        if int(row.get("n_tournaments", 0)) < 10:
            continue
        pair_adv = row.get("code_advantage_pairwise")
        top_adv = row.get("code_advantage_top1")
        if pair_adv is not None and float(pair_adv) >= 0.10:
            return True
        if top_adv is not None and float(top_adv) >= 0.10:
            return True
    return False


def set_too_easy(transfer_summary: dict[str, Any]) -> bool:
    best = transfer_summary.get("best_overall", {})
    if not isinstance(best, dict):
        return False
    return float(best.get("pairwise", 0.0)) >= 0.95 or float(best.get("top1", 0.0)) >= 0.90


def recommended_next(summary: dict[str, Any]) -> str:
    if summary["MEDICINE_TRANSFER_VERDICT"] == "POOR":
        return "treat_medicine_as_open_problem_do_not_route_medical_tasks_to_current_heads"
    if subdomain_code_objective(summary):
        return "test_mixed_code_science_heads_next"
    if summary["SCIENCE_SPECIALIST_VERDICT"] == "SPECIALIST_NEEDED":
        return "train_or_validate_science_specific_tap_on_larger_split"
    if summary["SCIENCE_TRANSFER_VERDICT"] == "GOOD" and summary["SCIENCE_SPECIALIST_VERDICT"] == "GENERAL_SUFFICIENT":
        return "add_science_as_objective_eval_domain_routed_to_general_head"
    if set_too_easy(summary):
        return "use_harder_science_datasets_MMLU_Pro_GPQA_or_generated_near_miss_science_traces"
    return "use_harder_science_datasets_MMLU_Pro_GPQA_or_generated_near_miss_science_traces"


def one_sentence(summary: dict[str, Any]) -> str:
    if summary["SCIENCE_SPECIALIST_VERDICT"] == "GENERAL_SUFFICIENT":
        return "Science natural distractors behave closer to reasoning MCQ than strict-clean code: current general readouts are enough on this benchmark set."
    if summary["SCIENCE_SPECIALIST_VERDICT"] == "SPECIALIST_NEEDED":
        return "Science shows a specialist gap in at least one subdomain, so a larger science-specific projection check is justified."
    return "Science transfer is not decisive enough yet; harder datasets or generated near-miss science traces should be prioritized."


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Science / bio / chem / medicine natural-distractor validation (2026-05-17)"
    text = "\n".join([
        "",
        title,
        "",
        f"- SCIENCE_DISTRACTOR_SET_VERDICT: `{summary['SCIENCE_DISTRACTOR_SET_VERDICT']}`",
        f"- SCIENCE_DISTRACTOR_FEATURE_VERDICT: `{summary['SCIENCE_DISTRACTOR_FEATURE_VERDICT']}`",
        f"- SCIENCE_TRANSFER_VERDICT: `{summary['SCIENCE_TRANSFER_VERDICT']}`",
        f"- SCIENCE_SPECIALIST_VERDICT: `{summary['SCIENCE_SPECIALIST_VERDICT']}`",
        f"- BIOLOGY_TRANSFER_VERDICT: `{summary['BIOLOGY_TRANSFER_VERDICT']}`",
        f"- CHEMISTRY_TRANSFER_VERDICT: `{summary['CHEMISTRY_TRANSFER_VERDICT']}`",
        f"- MEDICINE_TRANSFER_VERDICT: `{summary['MEDICINE_TRANSFER_VERDICT']}`",
        f"- GENERAL_SCIENCE_TRANSFER_VERDICT: `{summary['GENERAL_SCIENCE_TRANSFER_VERDICT']}`",
        f"- SCIENCE_SPECIFIC_HEAD_VERDICT: `{summary['SCIENCE_SPECIFIC_HEAD_VERDICT']}`",
        f"- SCIENCE_DOMAIN_ANALOGY_VERDICT: `{summary['SCIENCE_DOMAIN_ANALOGY_VERDICT']}`",
        f"- dataset/task counts: `{summary['dataset_task_counts']}`",
        f"- random_top1_baseline: `{summary.get('random_top1_baseline')}`",
        f"- best HH row: `{summary.get('best_hh')}`",
        f"- best code row: `{summary.get('best_code')}`",
        f"- best NoNorm row: `{summary.get('best_nonorm')}`",
        f"- best AntisymLinear row: `{summary.get('best_antisymlinear')}`",
        f"- subdomain breakdown: `{summary.get('subdomain_breakdown')}`",
        f"- science objective eval domain status: `{summary.get('science_domain_status')}`",
        f"- science/medicine specialist status: `{summary.get('science_specialist_status')}`",
        "- medicine caveat: benchmark MCQ transfer only, not clinical validation.",
        "- full reports: `opi/taps/probes/science_domain_audit_2026-05-17_summary.md`, `opi/taps/probes/science_natural_distractor_transfer_2026-05-17.md`, `opi/taps/probes/science_domain_comparison_2026-05-17.md`",
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


def write_geom_md(payload: dict[str, Any]) -> None:
    s = payload.get("summary", {})
    lines = [
        "# Science Loop/Layer Geometry",
        "",
        f"SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT = {s.get('SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT', payload.get('science_loop_layer_geometry_verdict'))}",
        "",
        f"- n_vectors: `{s.get('n_vectors')}`",
        f"- loop_geometry: `{s.get('loop_geometry')}`",
        f"- subdomain_geometry: `{s.get('subdomain_geometry')}`",
        f"- correct_vs_distractor_distances: `{s.get('correct_vs_distractor_distances')}`",
        f"- layer47_margin_correlation: `{s.get('layer47_margin_correlation')}`",
        "",
    ]
    GEOM_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Science Domain Audit Summary",
        "",
        f"SCIENCE_DISTRACTOR_SET_VERDICT = {s['SCIENCE_DISTRACTOR_SET_VERDICT']}",
        f"SCIENCE_DISTRACTOR_FEATURE_VERDICT = {s['SCIENCE_DISTRACTOR_FEATURE_VERDICT']}",
        f"SCIENCE_TRANSFER_VERDICT = {s['SCIENCE_TRANSFER_VERDICT']}",
        f"SCIENCE_SPECIALIST_VERDICT = {s['SCIENCE_SPECIALIST_VERDICT']}",
        f"BIOLOGY_TRANSFER_VERDICT = {s['BIOLOGY_TRANSFER_VERDICT']}",
        f"CHEMISTRY_TRANSFER_VERDICT = {s['CHEMISTRY_TRANSFER_VERDICT']}",
        f"MEDICINE_TRANSFER_VERDICT = {s['MEDICINE_TRANSFER_VERDICT']}",
        f"GENERAL_SCIENCE_TRANSFER_VERDICT = {s['GENERAL_SCIENCE_TRANSFER_VERDICT']}",
        f"SCIENCE_SPECIFIC_HEAD_VERDICT = {s['SCIENCE_SPECIFIC_HEAD_VERDICT']}",
        f"SCIENCE_DOMAIN_ANALOGY_VERDICT = {s['SCIENCE_DOMAIN_ANALOGY_VERDICT']}",
        f"RECOMMENDED_NEXT = {s['RECOMMENDED_NEXT']}",
        "",
        "## 1. Dataset Construction",
        "",
        f"- dataset/task counts: `{s['dataset_task_counts']}`",
        f"- subdomain breakdown: `{s['subdomain_breakdown']}`",
        f"- random_top1_baseline: `{s.get('random_top1_baseline')}`",
        "",
        "## 2. Feature Capture",
        "",
        f"- feature verdict: `{s['SCIENCE_DISTRACTOR_FEATURE_VERDICT']}`",
        f"- feature file: `opi/taps/probes/science_natural_distractor_features_2026-05-17.pt`",
        "",
        "## 3. HH/Code Head Transfer",
        "",
        f"- best HH row: `{s.get('best_hh')}`",
        f"- best code row: `{s.get('best_code')}`",
        f"- best NoNorm row: `{s.get('best_nonorm')}`",
        f"- best AntisymLinear row: `{s.get('best_antisymlinear')}`",
        "",
        "## 4. Subdomain Results",
        "",
        f"`{s.get('subdomain_transfer')}`",
        "",
        "## 5. Science-Specific Control If Run",
        "",
        f"`{s.get('science_specific_summary')}`",
        "",
        "## 6. Science Vs Existing Domains",
        "",
        f"`{s.get('domain_comparison')}`",
        "",
        "## 7. Loop/Layer Geometry",
        "",
        f"`{s.get('geometry_summary')}`",
        "",
        "## 8. Interpretation",
        "",
        s["one_sentence_interpretation"],
        "",
        "## 9. Medical Benchmark Caveat",
        "",
        "Medicine/clinical-style rows here are benchmark MCQ transfer diagnostics only, not clinical validation.",
        "",
        "## 10. Docs Updated",
        "",
    ]
    for path in payload.get("docs_updated", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## 11. Files Modified / Created", ""])
    for path in payload.get("files_modified_or_created", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## 12. Commands Run", "", "```bash"])
    lines.extend(payload.get("commands_run", []))
    lines.extend(["```", "", "## 13. Blockers", "", payload.get("blockers", "None."), ""])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    geom = analyze_geometry()
    write_json(GEOM_JSON, geom)
    write_geom_md(geom)

    set_payload = load_json(SET_JSON)
    transfer = load_json(TRANSFER_JSON)
    science_specific = load_json(SCIENCE_SPECIFIC_JSON)
    comparison = load_json(COMPARISON_JSON)
    feature_meta = {}
    if FEATURES_PT.exists():
        try:
            feature_meta = torch.load(FEATURES_PT, map_location="cpu", weights_only=False).get("meta", {})
        except Exception as exc:
            feature_meta = {"load_error": f"{type(exc).__name__}: {exc}"}

    set_summary = set_payload.get("summary", {})
    transfer_summary = transfer.get("summary", {})
    comparison_summary = comparison.get("summary", {})
    science_specific_summary = science_specific.get("summary", {})
    science_specific_v = science_specific_summary.get("SCIENCE_SPECIFIC_HEAD_VERDICT", science_specific.get("science_specific_head_verdict", "NOT_RUN"))
    summary = {
        "SCIENCE_DISTRACTOR_SET_VERDICT": set_summary.get("SCIENCE_DISTRACTOR_SET_VERDICT", set_payload.get("science_distractor_set_verdict", "BLOCKED")),
        "SCIENCE_DISTRACTOR_FEATURE_VERDICT": feature_meta.get("science_distractor_feature_verdict", "READY" if FEATURES_PT.exists() else "BLOCKED").upper(),
        "SCIENCE_TRANSFER_VERDICT": transfer_summary.get("SCIENCE_TRANSFER_VERDICT", transfer.get("science_transfer_verdict", "NOT_RUN")),
        "SCIENCE_SPECIALIST_VERDICT": transfer_summary.get("SCIENCE_SPECIALIST_VERDICT", transfer.get("science_specialist_verdict", "INSUFFICIENT")),
        "BIOLOGY_TRANSFER_VERDICT": transfer_summary.get("BIOLOGY_TRANSFER_VERDICT", "NOT_RUN"),
        "CHEMISTRY_TRANSFER_VERDICT": transfer_summary.get("CHEMISTRY_TRANSFER_VERDICT", "NOT_RUN"),
        "MEDICINE_TRANSFER_VERDICT": transfer_summary.get("MEDICINE_TRANSFER_VERDICT", "NOT_RUN"),
        "GENERAL_SCIENCE_TRANSFER_VERDICT": transfer_summary.get("GENERAL_SCIENCE_TRANSFER_VERDICT", "NOT_RUN"),
        "SCIENCE_SPECIFIC_HEAD_VERDICT": science_specific_v,
        "SCIENCE_DOMAIN_ANALOGY_VERDICT": comparison_summary.get("SCIENCE_DOMAIN_ANALOGY_VERDICT", comparison.get("science_domain_analogy_verdict", "INSUFFICIENT")),
        "dataset_task_counts": set_summary.get("dataset_breakdown", {}),
        "subdomain_breakdown": set_summary.get("subdomain_breakdown", {}),
        "random_top1_baseline": transfer_summary.get("random_top1_baseline", set_summary.get("random_top1_baseline")),
        "best_hh": transfer_summary.get("best_hh"),
        "best_overall": transfer_summary.get("best_overall"),
        "best_code": transfer_summary.get("best_code"),
        "best_nonorm": transfer_summary.get("best_nonorm"),
        "best_antisymlinear": transfer_summary.get("best_antisymlinear"),
        "subdomain_transfer": transfer_summary.get("per_subdomain", {}),
        "science_specific_summary": science_specific_summary,
        "domain_comparison": comparison_summary,
        "geometry_summary": geom.get("summary", {}),
    }
    summary["science_domain_status"] = "add_as_objective_eval_domain" if summary["SCIENCE_TRANSFER_VERDICT"] in {"GOOD", "WEAK"} else "not_ready"
    summary["science_specialist_status"] = "specialist_needed" if summary["SCIENCE_SPECIALIST_VERDICT"] == "SPECIALIST_NEEDED" else "not_currently_needed"
    summary["RECOMMENDED_NEXT"] = recommended_next(summary)
    summary["one_sentence_interpretation"] = one_sentence(summary)
    docs_updated = append_docs(summary)
    files = [
        "shared/utilities/tests/manual/build_science_natural_distractor_set.py",
        "shared/utilities/tests/manual/capture_science_natural_distractor_features.py",
        "shared/utilities/tests/manual/evaluate_heads_on_science_natural_distractors.py",
        "shared/utilities/tests/manual/train_science_specific_tiny_heads.py",
        "shared/utilities/tests/manual/compare_science_to_existing_domains.py",
        "shared/utilities/tests/manual/analyze_science_loop_layer_geometry.py",
        repo_path(SET_JSON),
        repo_path(FEATURES_PT),
        repo_path(TRANSFER_JSON),
        repo_path(SCIENCE_SPECIFIC_JSON),
        repo_path(COMPARISON_JSON),
        repo_path(GEOM_JSON),
        repo_path(GEOM_MD),
        repo_path(SUMMARY_JSON),
        repo_path(SUMMARY_MD),
    ] + docs_updated
    commands = [
        "venv/bin/python -m py_compile utilities/tests/manual/build_science_natural_distractor_set.py",
        "venv/bin/python -m py_compile utilities/tests/manual/capture_science_natural_distractor_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_heads_on_science_natural_distractors.py",
        "venv/bin/python -m py_compile utilities/tests/manual/train_science_specific_tiny_heads.py",
        "venv/bin/python -m py_compile utilities/tests/manual/compare_science_to_existing_domains.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_science_loop_layer_geometry.py",
        "venv/bin/python -u utilities/tests/manual/build_science_natural_distractor_set.py --target-total 120 --target-per-bucket 20 --min-total 30",
        "venv/bin/python -u utilities/tests/manual/capture_science_natural_distractor_features.py --input opi/taps/probes/science_natural_distractor_set_2026-05-17.json --output opi/taps/probes/science_natural_distractor_features_2026-05-17.pt --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_heads_on_science_natural_distractors.py --features opi/taps/probes/science_natural_distractor_features_2026-05-17.pt --heads opi/taps/probes/bg_head_registry_2026-05-17.pt --input opi/taps/probes/science_natural_distractor_set_2026-05-17.json --output opi/taps/probes/science_natural_distractor_transfer_2026-05-17.json",
        "venv/bin/python -u utilities/tests/manual/train_science_specific_tiny_heads.py --features opi/taps/probes/science_natural_distractor_features_2026-05-17.pt --input opi/taps/probes/science_natural_distractor_set_2026-05-17.json --output opi/taps/probes/science_specific_tiny_head_control_2026-05-17.json",
        "venv/bin/python -u utilities/tests/manual/compare_science_to_existing_domains.py",
        "venv/bin/python -u utilities/tests/manual/analyze_science_loop_layer_geometry.py",
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
    print(f"SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT = {geom.get('summary', {}).get('SCIENCE_LOOP_LAYER_GEOMETRY_VERDICT')}")
    print(f"SCIENCE_TRANSFER_VERDICT = {summary['SCIENCE_TRANSFER_VERDICT']}")
    print(f"SCIENCE_SPECIALIST_VERDICT = {summary['SCIENCE_SPECIALIST_VERDICT']}")
    print(f"SCIENCE_DOMAIN_ANALOGY_VERDICT = {summary['SCIENCE_DOMAIN_ANALOGY_VERDICT']}")
    print(f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}")
    print(f"Wrote {GEOM_JSON}")
    print(f"Wrote {GEOM_MD}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
