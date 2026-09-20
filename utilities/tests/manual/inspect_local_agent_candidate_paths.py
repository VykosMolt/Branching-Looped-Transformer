"""Inspect local-agent candidate branch paths for BG wrapper export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = ROOT / "artifacts" / "reports" / "probes"
OUT_JSON = REPORT_DIR / "local_agent_candidate_path_inventory_2026-05-18.json"
OUT_MD = REPORT_DIR / "local_agent_candidate_path_inventory_2026-05-18.md"


FILES = {
    "direct": ROOT / "shared/src" / "local_agent" / "ouro_direct.py",
    "agent": ROOT / "shared/src" / "local_agent" / "ouro_agent_improved.py",
    "policies": ROOT / "shared/src" / "local_agent" / "ouro_policies.py",
    "config": ROOT / "shared/src" / "local_agent" / "ouro_config.py",
    "manual_v1": ROOT / "shared/utilities" / "tests" / "manual" / "generate_code_branch_candidates_local_agent.py",
    "manual_v2": ROOT / "shared/utilities" / "tests" / "manual" / "generate_code_branch_candidates_v2.py",
}


NEEDLES = {
    "direct_final_answer": ("direct", r"def direct_answer\("),
    "direct_finalize": ("direct", r"def _finalize_direct_output\("),
    "direct_acceptance": ("direct", r"def _direct_output_acceptance\("),
    "direct_code_extract": ("direct", r"_extract_python_code_block"),
    "agent_entry": ("agent", r"def run_agent_task\("),
    "agent_mode_entry": ("agent", r"def run_task_mode\("),
    "parse_action": ("agent", r"def parse_action\("),
    "tool_execute": ("agent", r"def execute_tool\("),
    "tool_sanitize": ("agent", r"sanitize_tool_input"),
    "tool_transition": ("agent", r"record_tool_transition\("),
    "grounded_final": ("agent", r"build_grounded_final_answer"),
    "unverified_final": ("agent", r"build_unverified_attempt_answer"),
    "manual_direct_candidates": ("manual_v2", r"def direct_code_candidate\("),
    "manual_first_tool_candidates": ("manual_v2", r"def first_tool_candidate\("),
}


def line_matches(path: Path, pattern: str) -> list[int]:
    if not path.exists():
        return []
    rx = re.compile(pattern)
    out: list[int] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if rx.search(line):
            out.append(idx)
    return out


def inspect() -> dict[str, Any]:
    files = {name: {"path": str(path), "exists": path.exists()} for name, path in FILES.items()}
    markers: dict[str, Any] = {}
    for marker, (file_key, pattern) in NEEDLES.items():
        path = FILES[file_key]
        lines = line_matches(path, pattern)
        markers[marker] = {
            "file": str(path),
            "pattern": pattern,
            "found": bool(lines),
            "lines": lines[:20],
        }

    direct_ready = all(markers[key]["found"] for key in ("direct_final_answer", "direct_finalize", "direct_acceptance"))
    agent_ready = all(
        markers[key]["found"]
        for key in ("agent_entry", "parse_action", "tool_execute", "tool_sanitize", "grounded_final")
    )
    if direct_ready and agent_ready:
        verdict = "READY"
    elif direct_ready or agent_ready:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    insertion_points = [
        {
            "route": "direct",
            "file": str(FILES["direct"]),
            "function": "direct_answer",
            "line": markers["direct_final_answer"]["lines"][:1],
            "safe_hook": "append CandidateArtifact after each direct attempt is finalized and accepted/rejected",
        },
        {
            "route": "agent",
            "file": str(FILES["agent"]),
            "function": "run_agent_task",
            "line": markers["agent_entry"]["lines"][:1],
            "safe_hook": "append CandidateArtifact after parse_action, sanitize_tool_input, and tool execution result",
        },
        {
            "route": "agent",
            "file": str(FILES["agent"]),
            "function": "finish",
            "line": line_matches(FILES["agent"], r"def finish\(outcome"),
            "safe_hook": "set selected_candidate_uid/final_answer only when explicit candidate tracing is enabled",
        },
    ]

    artifacts = {
        "direct final answer": "direct_answer accepted cleaned output",
        "first python/cpp tool input": "parse_action output before sanitize_tool_input",
        "failed tool input": "ExecutionResult after execute_tool/apply_external_evaluator with success=False",
        "repair tool input": "later debug-phase python/cpp attempts in run_agent_task",
        "final answer code block": "TaskEvaluator.build_grounded_final_answer final_code",
        "sanitized code": "sanitize_tool_input(action, tool_input)",
        "verifier/tool output": "ExecutionResult.observation",
        "status/prose outputs": "force_final_only candidate and build_unverified_attempt_answer fallback",
    }
    return {
        "WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT": verdict,
        "files": files,
        "markers": markers,
        "artifacts": artifacts,
        "existing_candidate_trace": {
            "found": False,
            "note": "Existing manual branch scripts harvest candidates externally; no shared wrapper CandidateTrace object existed before this interface.",
        },
        "minimal_safe_insertion_points": insertion_points,
        "normal_call_sites": {
            "run_task_mode": markers["agent_mode_entry"],
            "manual_v2_direct": markers["manual_direct_candidates"],
            "manual_v2_first_tool": markers["manual_first_tool_candidates"],
        },
    }


def write_md(data: dict[str, Any]) -> str:
    lines = [
        "# Local-Agent Candidate Path Inventory",
        "",
        f"WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT = {data['WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT']}",
        "",
        "## Routes",
        "",
        f"- Direct route ready: {data['markers']['direct_final_answer']['found']}",
        f"- Tool/agent route ready: {data['markers']['agent_entry']['found']}",
        f"- Existing shared trace object: {data['existing_candidate_trace']['found']}",
        "",
        "## Candidate Artifacts",
    ]
    for name, location in data["artifacts"].items():
        lines.append(f"- {name}: {location}")
    lines.extend(["", "## Minimal Safe Insertion Points"])
    for item in data["minimal_safe_insertion_points"]:
        line = item.get("line") or []
        lines.append(f"- {item['route']} `{item['function']}` {line}: {item['safe_hook']}")
    lines.extend(["", "## Interpretation", "Direct and ReAct/tool candidate artifacts have narrow opt-in insertion points; default wrapper calls can remain unchanged."])
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = inspect()
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(write_md(data), encoding="utf-8")
    print(f"WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT = {data['WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT']}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
