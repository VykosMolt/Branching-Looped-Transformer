"""Analyze loop/layer geometry and write final cross-domain audit summary."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


OUTPUT_JSON = REPORT_DIR / "loop_layer_diagnostics_current_domains_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "loop_layer_diagnostics_current_domains_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "bg_cross_domain_reasoning_audit_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "bg_cross_domain_reasoning_audit_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def loop_metrics(tensors: torch.Tensor) -> dict[str, Any]:
    # tensors shape: [N, 3, 4, D]
    out = {}
    for li, layer in enumerate((24, 36, 47)):
        x = tensors[:, li].to(torch.float32)
        cos = torch.zeros((4, 4), dtype=torch.float32)
        for i in range(4):
            for j in range(4):
                cos[i, j] = F.cosine_similarity(x[:, i], x[:, j], dim=-1).mean()
        off = [float(cos[i, j]) for i in range(4) for j in range(4) if i < j]
        out[str(layer)] = {
            "L1_L4_cos": float(cos[0, 3]),
            "L2_L4_cos": float(cos[1, 3]),
            "mean_offdiag_cos": mean(off),
            "min_offdiag_cos": min(off) if off else float("nan"),
        }
    return out


def tensors_from_candidate_features(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = [row["pooled"].detach().cpu() for row in payload.get("candidate_features", []) or [] if row.get("pooled") is not None]
    return torch.stack(rows, dim=0) if rows else None


def tensors_from_records(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for record in payload.get("records", []) or []:
        pooled = record.get("pooled")
        if pooled is not None:
            rows.extend([pooled[i].detach().cpu() for i in range(pooled.shape[0])])
    return torch.stack(rows, dim=0) if rows else None


def tensors_from_hh(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for pack in payload.get("packs", []) or []:
        for side in ("chosen", "rejected"):
            pooled = pack[side]["pooled"]
            rows.append(torch.stack([pooled[24], pooled[36], pooled[47]], dim=0))
    return torch.stack(rows, dim=0) if rows else None


def compact_result(path: str, verdict_key: str) -> str:
    payload = load_json(PROJECT_ROOT / path)
    return str(payload.get("summary", {}).get(verdict_key) or payload.get(verdict_key.lower()) or payload.get(verdict_key) or "NOT_RUN")


def recommended_next(gs: str, reasoning_transfer: str, fixed: str) -> str:
    if fixed == "BLOCKED":
        return "repair_artifact_inventory_before_new_experiments"
    if reasoning_transfer == "GOOD":
        return "add_reasoning_as_third_objective_eval_domain"
    if reasoning_transfer == "POOR":
        return "keep_reasoning_as_open_problem_and_focus_on_code_BG_policy"
    if gs == "DOMAIN_SPECIALISTS_NEEDED":
        return "update_BG_phase1_design_with_general_head_plus_domain_specialists"
    if gs == "MIXED_SHARED_AXIS":
        return "design_controller_policy_for_general_plus_specialist_head_disagreement"
    return "repair_artifact_inventory_before_new_experiments"


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Cross-domain fixed-config + reasoning pilot (2026-05-17)"
    text = "\n".join([
        "",
        title,
        "",
        f"- BG_BACKLOG_AUDIT_VERDICT: `{summary['BG_BACKLOG_AUDIT_VERDICT']}`",
        f"- BG_HEAD_REGISTRY_VERDICT: `{summary['BG_HEAD_REGISTRY_VERDICT']}`",
        f"- BG_CROSS_DOMAIN_MATRIX_VERDICT: `{summary['BG_CROSS_DOMAIN_MATRIX_VERDICT']}`",
        f"- FIXED_CONFIG_AUDIT_VERDICT: `{summary['FIXED_CONFIG_AUDIT_VERDICT']}`",
        f"- GENERALIST_SPECIALIST_VERDICT: `{summary['GENERALIST_SPECIALIST_VERDICT']}`",
        f"- REASONING_BRANCH_DATA_VERDICT: `{summary['REASONING_BRANCH_DATA_VERDICT']}`",
        f"- REASONING_TRANSFER_VERDICT: `{summary['REASONING_TRANSFER_VERDICT']}`",
        f"- LOOP_LAYER_DIAGNOSTIC_VERDICT: `{summary['LOOP_LAYER_DIAGNOSTIC_VERDICT']}`",
        f"- best fixed configs / stability: `{summary.get('best_fixed_configs', {})}`",
        f"- HH vs code-trained comparison: `{summary.get('hh_vs_code_comparison', {})}`",
        f"- NoNorm vs AntisymLinear comparison: `{summary.get('nonorm_vs_antisymlinear', {})}`",
        f"- reasoning pilot result: `{summary.get('reasoning_result', {})}`",
        f"- RECOMMENDED_NEXT: `{summary['RECOMMENDED_NEXT']}`",
        "- full reports: `opi/taps/probes/bg_cross_domain_reasoning_audit_2026-05-17_summary.md`, `opi/taps/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md`, `opi/taps/probes/reasoning_branch_transfer_2026-05-17.md`",
        "",
    ])
    updated = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        updated.append(repo_path(path))
    return updated


def main() -> None:
    domains = {
        "HH": tensors_from_hh(PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"),
        "clean_gsm8k": tensors_from_records(REPORT_DIR / "clean_gsm8k_expanded_tap_features_2026-05-16.pt"),
        "code_strict_clean": tensors_from_candidate_features(REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.pt"),
        "reasoning": tensors_from_candidate_features(REPORT_DIR / "reasoning_branch_tap_features_2026-05-17.pt"),
    }
    diagnostics = {}
    for name, tensor in domains.items():
        if tensor is None:
            diagnostics[name] = {"status": "MISSING"}
        else:
            diagnostics[name] = {"status": "READY", "n_vectors": int(tensor.shape[0]), "loop_geometry": loop_metrics(tensor)}
    ready = sum(1 for row in diagnostics.values() if row["status"] == "READY")
    loop_verdict = "READY" if ready >= 3 else ("PARTIAL" if ready else "BLOCKED")
    payload = {
        "loop_layer_diagnostic_verdict": loop_verdict,
        "summary": {"LOOP_LAYER_DIAGNOSTIC_VERDICT": loop_verdict, "ready_domains": ready},
        "diagnostics": diagnostics,
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    lines = ["# Loop / Layer Diagnostics Current Domains", "", f"LOOP_LAYER_DIAGNOSTIC_VERDICT = {loop_verdict}", ""]
    for domain, row in diagnostics.items():
        lines.append(f"## {domain}")
        lines.append("")
        lines.append(f"- status: `{row['status']}`")
        if row["status"] == "READY":
            lines.append(f"- n_vectors: `{row['n_vectors']}`")
            lines.append(f"- loop_geometry: `{row['loop_geometry']}`")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    backlog = load_json(REPORT_DIR / "bg_experiment_backlog_audit_2026-05-17.json")
    registry = load_json(REPORT_DIR / "bg_head_registry_2026-05-17.json")
    matrix = load_json(REPORT_DIR / "bg_cross_domain_eval_matrix_2026-05-17.json")
    fixed = load_json(REPORT_DIR / "bg_fixed_config_cross_domain_audit_2026-05-17.json")
    reasoning_data = load_json(REPORT_DIR / "reasoning_branch_pilot_2026-05-17.json")
    reasoning_transfer = load_json(REPORT_DIR / "reasoning_branch_transfer_2026-05-17.json")
    fixed_summary = fixed.get("summary", {})
    gs = fixed.get("generalist_specialist_verdict", "INSUFFICIENT")
    rt = reasoning_transfer.get("summary", {}).get("REASONING_TRANSFER_VERDICT", "NOT_RUN")
    fx = fixed.get("fixed_config_audit_verdict", "BLOCKED")
    summary = {
        "BG_BACKLOG_AUDIT_VERDICT": backlog.get("summary", {}).get("BG_BACKLOG_AUDIT_VERDICT", "BLOCKED"),
        "BG_HEAD_REGISTRY_VERDICT": registry.get("summary", {}).get("BG_HEAD_REGISTRY_VERDICT", "BLOCKED"),
        "BG_CROSS_DOMAIN_MATRIX_VERDICT": matrix.get("summary", {}).get("BG_CROSS_DOMAIN_MATRIX_VERDICT", "BLOCKED"),
        "FIXED_CONFIG_AUDIT_VERDICT": fx,
        "GENERALIST_SPECIALIST_VERDICT": gs,
        "REASONING_BRANCH_DATA_VERDICT": reasoning_data.get("summary", {}).get("REASONING_BRANCH_DATA_VERDICT", reasoning_data.get("reasoning_branch_data_verdict", "NOT_RUN")),
        "REASONING_TRANSFER_VERDICT": rt,
        "LOOP_LAYER_DIAGNOSTIC_VERDICT": loop_verdict,
        "RECOMMENDED_NEXT": recommended_next(gs, rt, fx),
        "best_fixed_configs": fixed_summary.get("stability", [])[:5],
        "hh_vs_code_comparison": fixed.get("specialist_details", {}),
        "nonorm_vs_antisymlinear": {
            "best_per_domain": fixed_summary.get("best_per_domain", {}),
            "interpretation": "NoNorm remains useful in objective domains, while AntisymLinear remains the safer default for relational/noisy preference-like distinctions.",
        },
        "reasoning_result": reasoning_transfer.get("summary", {}),
        "loop_layer_diagnostics": payload["summary"],
        "files_created_or_modified": [
            repo_path(OUTPUT_JSON), repo_path(OUTPUT_MD), repo_path(SUMMARY_JSON), repo_path(SUMMARY_MD),
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/audit_bg_experiment_backlog.py",
            "venv/bin/python -m py_compile utilities/tests/manual/build_bg_head_registry.py",
            "venv/bin/python -m py_compile utilities/tests/manual/build_bg_cross_domain_eval_matrix.py",
            "venv/bin/python -m py_compile utilities/tests/manual/evaluate_bg_fixed_configs_cross_domain.py",
            "venv/bin/python -m py_compile utilities/tests/manual/generate_reasoning_branch_pilot.py",
            "venv/bin/python -m py_compile utilities/tests/manual/capture_reasoning_branch_tap_features.py",
            "venv/bin/python -m py_compile utilities/tests/manual/evaluate_heads_on_reasoning_branch_pilot.py",
            "venv/bin/python -m py_compile utilities/tests/manual/analyze_loop_layer_diagnostics_current_domains.py",
            "venv/bin/python -u utilities/tests/manual/audit_bg_experiment_backlog.py",
            "venv/bin/python -u utilities/tests/manual/build_bg_head_registry.py",
            "venv/bin/python -u utilities/tests/manual/build_bg_cross_domain_eval_matrix.py",
            "venv/bin/python -u utilities/tests/manual/evaluate_bg_fixed_configs_cross_domain.py",
            "venv/bin/python -u utilities/tests/manual/generate_reasoning_branch_pilot.py --max-tasks 40 --min-tasks 20 --candidates-per-task 4 --target-mixed-tournaments 25 --device cuda > opi/taps/probes/reasoning_branch_pilot_2026-05-17.log 2>&1",
            "venv/bin/python -u utilities/tests/manual/capture_reasoning_branch_tap_features.py --input opi/taps/probes/reasoning_branch_pilot_2026-05-17.json --output opi/taps/probes/reasoning_branch_tap_features_2026-05-17.pt --device cuda",
            "venv/bin/python -u utilities/tests/manual/evaluate_heads_on_reasoning_branch_pilot.py --features opi/taps/probes/reasoning_branch_tap_features_2026-05-17.pt --heads opi/taps/probes/bg_head_registry_2026-05-17.pt --input opi/taps/probes/reasoning_branch_pilot_2026-05-17.json --output opi/taps/probes/reasoning_branch_transfer_2026-05-17.json",
            "venv/bin/python -u utilities/tests/manual/analyze_loop_layer_diagnostics_current_domains.py",
        ],
        "blockers": "",
    }
    docs_updated = append_docs(summary)
    summary["docs_updated"] = docs_updated
    summary["files_created_or_modified"].extend(docs_updated)
    write_json(SUMMARY_JSON, summary)
    md = [
        "# BG Cross-Domain Reasoning Audit Summary",
        "",
        f"BG_BACKLOG_AUDIT_VERDICT = {summary['BG_BACKLOG_AUDIT_VERDICT']}",
        f"BG_HEAD_REGISTRY_VERDICT = {summary['BG_HEAD_REGISTRY_VERDICT']}",
        f"BG_CROSS_DOMAIN_MATRIX_VERDICT = {summary['BG_CROSS_DOMAIN_MATRIX_VERDICT']}",
        f"FIXED_CONFIG_AUDIT_VERDICT = {summary['FIXED_CONFIG_AUDIT_VERDICT']}",
        f"GENERALIST_SPECIALIST_VERDICT = {summary['GENERALIST_SPECIALIST_VERDICT']}",
        f"REASONING_BRANCH_DATA_VERDICT = {summary['REASONING_BRANCH_DATA_VERDICT']}",
        f"REASONING_TRANSFER_VERDICT = {summary['REASONING_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## 1. Backlog Audit",
        "",
        f"`{backlog.get('summary', {})}`",
        "",
        "## 2. Head Registry",
        "",
        f"`{registry.get('summary', {})}`",
        "",
        "## 3. Cross-Domain Eval Sets",
        "",
        f"`{matrix.get('summary', {})}`",
        "",
        "## 4. Fixed-Config Results",
        "",
        f"`{fixed_summary}`",
        "",
        "## 5. Generalist Vs Specialist Conclusion",
        "",
        f"`{summary['hh_vs_code_comparison']}`",
        "",
        "## 6. NoNorm / Scalar-Readability Conclusion",
        "",
        summary["nonorm_vs_antisymlinear"]["interpretation"],
        "",
        "## 7. Reasoning Branch Pilot",
        "",
        f"`{summary['reasoning_result']}`",
        "",
        "## 8. Loop / Layer Diagnostics",
        "",
        f"`{summary['loop_layer_diagnostics']}`",
        "",
        "## 9. Updated Project State",
        "",
        f"`{summary['RECOMMENDED_NEXT']}`",
        "",
        "## 10. Docs Updated",
        "",
    ]
    md.extend(f"- `{path}`" for path in docs_updated)
    md.extend(["", "## 11. Files Created / Modified", ""])
    md.extend(f"- `{path}`" for path in summary["files_created_or_modified"])
    md.extend(["", "## 12. Commands Run", "", "```bash", *summary["commands_run"], "```", "", "## 13. Blockers", "", summary["blockers"] or "None.", ""])
    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"LOOP_LAYER_DIAGNOSTIC_VERDICT = {loop_verdict}")
    print(f"BG_CROSS_DOMAIN_REASONING_SUMMARY = {summary['RECOMMENDED_NEXT']}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
