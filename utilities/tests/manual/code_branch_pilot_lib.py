"""Shared helpers for the small code branch-selection pilot."""
from __future__ import annotations

import hashlib
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOCAL_AGENT_DIR = PROJECT_ROOT / "shared/src" / "local_agent"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "probes"
RLTT_MODEL_PATH = PROJECT_ROOT / "shared/models" / "ouro_rltt_local"


def configure_local_agent_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("LOCAL_AGENT_OURO_MODEL_ID", str(RLTT_MODEL_PATH))
    os.environ.setdefault("LOCAL_AGENT_OURO_LOAD_IN_4BIT", "0")
    os.environ.setdefault("LOCAL_AGENT_OURO_USE_CACHE", "1")
    os.environ.setdefault("LOCAL_AGENT_OURO_VERBOSE", "0")
    os.environ.setdefault("LOCAL_AGENT_ORACLE_LOOKUPS_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_CODE_REFERENCE_FASTPATH", "0")
    os.environ.setdefault("LOCAL_AGENT_FAST_ASSISTED_SOLVER_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_SEARCH_PROVIDER", "disabled")
    os.environ.setdefault("LOCAL_AGENT_EXTERNAL_EXPERTS_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_EXTERNAL_EXPERT_AUTO_ENABLED", "0")


def ensure_local_agent_path() -> None:
    if str(LOCAL_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_AGENT_DIR))


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def output_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def snippet(text: Any, limit: int = 240) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def import_agent_modules() -> dict[str, Any]:
    configure_local_agent_env()
    ensure_local_agent_path()
    import ouro_agent_improved as agent  # type: ignore
    import ouro_backend as backend  # type: ignore
    import ouro_config as config  # type: ignore
    import ouro_direct as direct  # type: ignore
    import ouro_policies as policies  # type: ignore
    import ouro_prompts as prompts  # type: ignore
    import ouro_types as types  # type: ignore

    return {
        "agent": agent,
        "backend": backend,
        "config": config,
        "direct": direct,
        "policies": policies,
        "prompts": prompts,
        "types": types,
    }


_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    match = _PY_FENCE_RE.search(cleaned) or _ANY_FENCE_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:python|py)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def extract_python_code(text: str, agent: Optional[Any] = None) -> str:
    raw = text or ""
    if agent is not None:
        for name in ("_extract_python_code_block", "_extract_any_code_block"):
            fn = getattr(agent, name, None)
            if callable(fn):
                try:
                    code = fn(raw)
                    if code:
                        return str(code).strip()
                except Exception:
                    pass
    code = strip_code_fences(raw)
    if "FINAL ANSWER:" in code.upper():
        code = re.split(r"FINAL ANSWER\s*:", code, flags=re.IGNORECASE)[-1]
        code = strip_code_fences(code)
    stop_markers = ("[Observation]", "[Verifier]", "[System]", "<END_FINAL>")
    for marker in stop_markers:
        if marker in code:
            code = code.split(marker, 1)[0]
    return code.strip()


def sanitize_python_candidate(text: str, policies: Optional[Any] = None, agent: Optional[Any] = None) -> str:
    candidate = text or ""
    if policies is not None and hasattr(policies, "sanitize_tool_input"):
        try:
            cleaned = policies.sanitize_tool_input("python", candidate)
            if cleaned and ("def " in cleaned or "class " in cleaned or "import " in cleaned):
                return cleaned.strip()
        except Exception:
            pass
    return extract_python_code(candidate, agent=agent)


def code_signature(code: str) -> str:
    normalized = "\n".join(line.rstrip() for line in (code or "").strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


_IGNORED_TEST_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "check",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _candidate_call_names(expr: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name and name not in _IGNORED_TEST_CALLS:
            names.append(name)
    return names


def function_name_from_tests(tests: Iterable[str]) -> str:
    for test in tests:
        text = str(test)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    names = _candidate_call_names(node.test)
                    if names:
                        return names[0]
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
            name = match.group(1)
            if name not in _IGNORED_TEST_CALLS and name != "assert":
                return name
    return ""


def function_arity_from_tests(tests: Iterable[str], function_name: str) -> int | None:
    if not function_name:
        return None
    for test in tests:
        try:
            tree = ast.parse(str(test))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) == function_name:
                if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                    continue
                return len(node.args)
    return None


def function_signature_hint(function_name: str, arity: int | None) -> str:
    if arity is None:
        return f"def {function_name}(*args):"
    args = ", ".join(f"arg{i + 1}" for i in range(max(arity, 0)))
    return f"def {function_name}({args}):"


def final_answer_code_block(code: str) -> str:
    return "FINAL ANSWER:\n```python\n" + (code or "").strip() + "\n```"
