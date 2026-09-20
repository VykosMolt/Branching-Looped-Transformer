"""Shared helpers for the wrapper-matched BG candidate-selection experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "probes" / "wrapper_bg_matched_2026-05-18"
TRACE_DIR = OUT_DIR / "traces"
TASKSET_SOURCE = PROJECT_ROOT / "artifacts" / "reports" / "probes" / "code_branch_taskset_v2_mini_patched_2026-05-16.json"
DEVIL_INVENTORY = PROJECT_ROOT / "artifacts" / "reports" / "probes" / "bg_devil_task_inventory_2026-05-18.json"
LOCAL_AGENT_DIR = PROJECT_ROOT / "shared/src" / "local_agent"
MANUAL_DIR = PROJECT_ROOT / "shared/utilities" / "tests" / "manual"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LOCAL_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_AGENT_DIR))
if str(MANUAL_DIR) not in sys.path:
    sys.path.insert(0, str(MANUAL_DIR))


PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ANY_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)
CODE_SHAPE_RE = re.compile(r"(?m)^\s*(def|class|from|import)\b")


def repo_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()[:n]


def task_slug(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")


def snippet(text: Any, limit: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def all_tests(task: dict[str, Any]) -> list[str]:
    tests = list(task.get("tests") or [])
    if tests:
        return [str(t) for t in tests if str(t).strip()]
    return [str(t) for t in list(task.get("public_tests") or []) + list(task.get("hidden_tests") or []) if str(t).strip()]


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    out["tests"] = all_tests(out)
    out.setdefault("public_tests", [])
    out.setdefault("hidden_tests", [])
    out.setdefault("timeout_seconds", out.get("expected_timeout_sec", 5.0))
    out.setdefault("expected_timeout_sec", out.get("timeout_seconds", 5.0))
    out.setdefault("difficulty", "unknown")
    out.setdefault("source", "unknown")
    out.setdefault("is_devil", out.get("difficulty") == "devil" or "offline_dynamic_connectivity" in out.get("task_id", "") or "minimum_xor_paths" in out.get("task_id", ""))
    out.setdefault("evaluator_info", {"kind": "python_unit_tests", "tests": len(out["tests"])})
    out.setdefault("source_artifact", repo_path(TASKSET_SOURCE))
    return out


def load_task_suite(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else OUT_DIR / "task_suite.json"
    payload = load_json(p)
    return [normalize_task(task) for task in payload.get("tasks", [])]


def strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    match = PY_FENCE_RE.search(raw) or ANY_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:python|py)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    return raw.strip()


def extract_code_from_text(text: str) -> str:
    raw = text or ""
    match = PY_FENCE_RE.search(raw) or ANY_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    if "FINAL ANSWER:" in raw.upper():
        raw = re.split(r"FINAL ANSWER\s*:", raw, flags=re.IGNORECASE)[-1]
    raw = strip_code_fences(raw)
    for marker in ("[Observation]", "[Verifier]", "[System]", "<END_FINAL>"):
        if marker in raw:
            raw = raw.split(marker, 1)[0]
    return raw.strip()


def looks_code_like(text: str) -> bool:
    raw = text or ""
    if not raw.strip():
        return False
    if PY_FENCE_RE.search(raw) or ANY_FENCE_RE.search(raw):
        return True
    return bool(CODE_SHAPE_RE.search(raw))


def candidate_code(artifact: dict[str, Any]) -> tuple[str, str]:
    for field in ("final_code", "sanitized_code"):
        value = str(artifact.get(field) or "").strip()
        if value:
            return value, field
    raw_tool_input = str(artifact.get("raw_tool_input") or "")
    if looks_code_like(raw_tool_input):
        return extract_code_from_text(raw_tool_input), "raw_tool_input"
    raw_text = str(artifact.get("raw_text") or "")
    if looks_code_like(raw_text):
        return extract_code_from_text(raw_text), "raw_text"
    return "", ""


def code_parse_status(code: str, function_name: str = "") -> dict[str, Any]:
    if not code.strip():
        return {"code_like": False, "syntax_ok": False, "has_expected_function": False, "error": "empty"}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"code_like": looks_code_like(code), "syntax_ok": False, "has_expected_function": False, "error": str(exc)}
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    return {
        "code_like": True,
        "syntax_ok": True,
        "has_expected_function": bool(function_name and function_name in functions) if function_name else bool(functions),
        "functions": functions,
        "error": "",
    }


def trace_candidates(trace: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, artifact in enumerate(trace.get("candidates") or []):
        item = dict(artifact)
        item.setdefault("candidate_uid", f"missing_uid_{idx}")
        item["_trace_index"] = idx
        out.append(item)
    return out


def stage_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(c.get("stage") or "unknown") for c in candidates))


def feature_input_text(prompt: str, code: str) -> str:
    return f"Problem:\n{prompt}\n\nCandidate solution:\n```python\n{code}\n```"


def label_is_success(label: str) -> bool:
    return label == "correct"


def label_is_near_or_success(label: str) -> bool:
    return label in {"correct", "near_miss"}


def label_rank(label: str) -> int:
    order = {
        "correct": 0,
        "near_miss": 1,
        "wrong_code": 2,
        "runtime_error": 3,
        "malformed": 4,
        "safety_rejected": 5,
        "not_code": 6,
        "missing": 7,
    }
    return order.get(label, 8)


def verdict_from_delta(delta: float, n: int, *, helps_threshold: float = 0.05, min_n: int = 8) -> str:
    if n < min_n:
        return "INSUFFICIENT"
    if delta >= helps_threshold:
        return "HELPS"
    if delta < -helps_threshold:
        return "HURTS"
    return "NEUTRAL"


def markdown_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines
