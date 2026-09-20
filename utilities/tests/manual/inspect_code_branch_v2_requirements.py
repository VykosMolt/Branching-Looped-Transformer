"""Inspect v1 code-pilot failure modes and v2 local-agent capture surface."""
from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, import_agent_modules, load_json, repo_path, write_json


DEFAULT_SUMMARY = REPORT_DIR / "code_branch_pilot_2026-05-16_summary.json"
DEFAULT_CANDIDATES = REPORT_DIR / "code_branch_candidates_2026-05-16.json"
DEFAULT_TOURNAMENTS = REPORT_DIR / "code_branch_tournaments_2026-05-16.json"
DEFAULT_JSON = REPORT_DIR / "code_branch_v2_interface_inspection_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_v2_interface_inspection_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--tournaments", default=str(DEFAULT_TOURNAMENTS))
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def signature(obj: Any, attr: str) -> str:
    fn = getattr(obj, attr, None)
    if not callable(fn):
        return ""
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "signature_unavailable"


def duplicate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_task[str(cand.get("task_id", ""))].append(cand)
    rows = {}
    total = 0
    unique = 0
    for task_id, items in sorted(by_task.items()):
        sigs = {
            str(item.get("ast_hash") or item.get("normalized_code_hash") or item.get("code_signature") or "")
            for item in items
            if item.get("final_code")
        }
        rows[task_id] = {
            "candidates": len(items),
            "unique_signatures": len(sigs),
            "duplicate_rate": 1.0 - (len(sigs) / max(len(items), 1)),
        }
        total += len(items)
        unique += len(sigs)
    return {
        "overall_duplicate_rate": 1.0 - (unique / max(total, 1)),
        "by_task": rows,
    }


def candidate_stage_capabilities(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    stages = Counter(str(c.get("candidate_stage", "unknown")) for c in candidates)
    return {
        "stage_counts": dict(stages),
        "has_raw_direct_code": any(c.get("raw_model_text") and c.get("candidate_stage") == "direct_final" for c in candidates),
        "has_sanitized_code": any(c.get("sanitized_code") for c in candidates),
        "has_first_tool_code": any(c.get("candidate_stage") == "first_tool_code" for c in candidates),
        "has_failed_tool_code": any(c.get("candidate_stage") == "first_failed_or_first_repair_code" for c in candidates),
        "has_repaired_final_code": any(c.get("candidate_stage") == "repaired_final" for c in candidates),
    }


def inspect_wrapper_surface() -> tuple[str, list[dict[str, Any]], dict[str, Any], str]:
    imports: list[dict[str, Any]] = []
    settings: dict[str, Any] = {}
    verdict = "READY"
    decision = ""
    try:
        modules = import_agent_modules()
    except Exception as exc:
        return "BLOCKED", [{"module": "local_agent_bundle", "ok": False, "error": f"{type(exc).__name__}: {exc}"}], {}, (
            "Local-agent imports failed; v2 generation should not run."
        )
    for name in ("ouro_direct", "ouro_agent_improved", "ouro_policies", "ouro_config", "ouro_backend", "ouro_prompts"):
        imports.append({"module": f"src.local_agent.{name}", "ok": True, "error": ""})
    agent = modules["agent"]
    direct = modules["direct"]
    policies = modules["policies"]
    backend = modules["backend"]
    config = modules["config"]
    surface = [
        {"name": "ouro_direct.direct_answer", "available": callable(getattr(direct, "direct_answer", None)), "signature": signature(direct, "direct_answer")},
        {"name": "ouro_agent_improved.run_task_mode", "available": callable(getattr(agent, "run_task_mode", None)), "signature": signature(agent, "run_task_mode")},
        {"name": "ouro_agent_improved.run_agent_task", "available": callable(getattr(agent, "run_agent_task", None)), "signature": signature(agent, "run_agent_task")},
        {"name": "ouro_agent_improved.parse_action", "available": callable(getattr(agent, "parse_action", None)), "signature": signature(agent, "parse_action")},
        {"name": "ouro_agent_improved.code_action_prefill_for_state", "available": callable(getattr(agent, "code_action_prefill_for_state", None)), "signature": signature(agent, "code_action_prefill_for_state")},
        {"name": "ouro_agent_improved.final_python_code_for_answer", "available": callable(getattr(agent, "final_python_code_for_answer", None)), "signature": signature(agent, "final_python_code_for_answer")},
        {"name": "ouro_policies.sanitize_tool_input", "available": callable(getattr(policies, "sanitize_tool_input", None)), "signature": signature(policies, "sanitize_tool_input")},
        {"name": "ouro_backend.DeepThinkModelManager", "available": callable(getattr(backend, "DeepThinkModelManager", None)), "signature": signature(backend, "DeepThinkModelManager")},
    ]
    for key in (
        "DIRECT_MAX_TOKENS",
        "DIRECT_TEMPERATURE",
        "CODE_FAST_PREFILL",
        "CODE_FIRST_PASS_TOKENS",
        "CODE_REPAIR_TOKENS",
        "CODE_FIRST_PASS_UT_STEPS",
        "CODE_REPAIR_UT_STEPS",
        "HARD_CODE_TOOL_MODE",
        "REQUIRE_HARD_CODE_VERIFICATION",
        "HARD_CODE_AGENT_MAX_TOKENS",
        "HARD_CODE_FIRST_ACTION_TOKENS",
        "PYTHON_TOOL_TIMEOUT_SEC",
        "HARD_CODE_TASK_WALLCLOCK_SEC",
    ):
        settings[key] = getattr(config, key, None)
    required = [
        "ouro_agent_improved.parse_action",
        "ouro_agent_improved.code_action_prefill_for_state",
        "ouro_agent_improved.final_python_code_for_answer",
        "ouro_policies.sanitize_tool_input",
        "ouro_backend.DeepThinkModelManager",
    ]
    available = {row["name"]: bool(row["available"]) for row in surface}
    if not all(available.get(name) for name in required):
        verdict = "BLOCKED"
        decision = "Required direct/action parsing/sanitization functions are missing."
    elif available.get("ouro_agent_improved.run_task_mode"):
        verdict = "READY"
        decision = (
            "Non-invasive v2 path is available: direct raw generations, action-prefilled first-tool inputs, "
            "public-test repair prompts, and optional final repaired answers. The inspection script did not load the model."
        )
    else:
        verdict = "PARTIAL"
        decision = "Direct/final generation appears possible, but tool-loop repaired-final capture is unavailable."
    return verdict, imports + surface, settings, decision


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Pilot v2 Interface Inspection",
        "",
        f"CODE_V2_INTERFACE_VERDICT = {payload['code_v2_interface_verdict']}",
        "",
        "## Why v1 Failed",
        "",
    ]
    for key, value in payload["v1_failure_summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## v1 Candidate Stage Coverage",
        "",
    ])
    for key, value in payload["v1_stage_capabilities"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Wrapper Surface",
        "",
        "| item | ok/available | signature/error |",
        "| --- | ---: | --- |",
    ])
    for row in payload["wrapper_surface"]:
        label = row.get("module") or row.get("name")
        ok = row.get("ok", row.get("available", ""))
        detail = row.get("error", row.get("signature", ""))
        lines.append(f"| `{label}` | {ok} | `{detail}` |")
    lines.extend([
        "",
        "## Relevant Settings",
        "",
        "| setting | value |",
        "| --- | ---: |",
    ])
    for key, value in payload["settings"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Decision", "", payload["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    candidate_payload = load_json(args.candidates)
    tournament_payload = load_json(args.tournaments)
    candidates = list(candidate_payload.get("candidates", []))
    candidate_evals = list(tournament_payload.get("candidate_evaluations", []))
    labels = Counter(row.get("unit_test_label", "unknown") for row in candidate_evals)
    dup = duplicate_summary(candidates)
    v1_summary = {
        "tasks": summary.get("tasks"),
        "candidates": summary.get("candidates"),
        "correct_candidates": labels.get("correct", 0),
        "near_miss_candidates": labels.get("near_miss", 0),
        "nonsense_candidates": labels.get("nonsense", 0),
        "strict_clean_tournaments": tournament_payload.get("summary", {}).get("strict_clean_tournaments", 0),
        "diagnostic_mixed_tournaments": tournament_payload.get("summary", {}).get("diagnostic_mixed_tournaments", 0),
        "duplicate_rate": round(float(dup["overall_duplicate_rate"]), 3),
        "stage_distribution": candidate_payload.get("summary", {}).get("stage_breakdown", {}),
        "failure_reason": "wrapper outputs were too often polished correct branches for the v1 task mix",
    }
    verdict, surface, settings, decision = inspect_wrapper_surface()
    payload = {
        "code_v2_interface_verdict": verdict,
        "v1_failure_summary": v1_summary,
        "v1_duplicate_summary": dup,
        "v1_stage_capabilities": candidate_stage_capabilities(candidates),
        "wrapper_surface": surface,
        "settings": settings,
        "decision": decision,
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_V2_INTERFACE_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
