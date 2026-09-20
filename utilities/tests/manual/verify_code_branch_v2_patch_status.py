"""Verify patched code-branch v2 wrapper, taskset, and safety behavior."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import (
    REPORT_DIR,
    function_arity_from_tests,
    function_name_from_tests,
    function_signature_hint,
    import_agent_modules,
    repo_path,
    write_json,
)
from evaluate_code_branch_candidates_v2 import safety_check


DEFAULT_JSON = REPORT_DIR / "code_branch_v2_patch_status_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_v2_patch_status_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    return args


def check_wrapper() -> dict[str, Any]:
    result: dict[str, Any] = {"checks": {}, "errors": []}
    try:
        modules = import_agent_modules()
    except Exception as exc:
        result["errors"].append(f"local-agent import failed: {type(exc).__name__}: {exc}")
        return result
    agent = modules["agent"]
    policies = modules["policies"]
    wrapper_status = "[Max steps reached \u2014 last output was a tool action]"
    prose_status = "The function should add one. It can return x + 1."
    valid_block = "Here is the implementation:\n```python\ndef add_one(x):\n    return x + 1\n```"
    sanitized = policies.sanitize_tool_input("python", wrapper_status)
    final_wrapper = agent.final_python_code_for_answer("Implement add_one(x).", wrapper_status)
    final_prose = agent.final_python_code_for_answer("Implement add_one(x).", prose_status)
    final_block = agent.final_python_code_for_answer("Implement add_one(x).", valid_block)
    result["checks"] = {
        "sanitize_wrapper_status_empty": sanitized == "",
        "final_wrapper_status_empty": final_wrapper == "",
        "final_prose_status_empty": final_prose == "",
        "valid_python_block_extracted": "def add_one" in final_block and "return x + 1" in final_block,
        "sanitized_wrapper_value": sanitized,
        "final_wrapper_value": final_wrapper,
        "final_prose_value": final_prose,
        "valid_block_value": final_block,
    }
    for key in (
        "sanitize_wrapper_status_empty",
        "final_wrapper_status_empty",
        "final_prose_status_empty",
        "valid_python_block_extracted",
    ):
        if not result["checks"][key]:
            result["errors"].append(f"wrapper check failed: {key}")
    return result


def load_mbpp_task(task_id: int) -> dict[str, Any] | None:
    try:
        from datasets import load_dataset

        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception as exc:
        return {"dataset_error": f"{type(exc).__name__}: {exc}"}
    for row in ds:
        if int(row.get("task_id", -1)) == task_id:
            return dict(row)
    return None


def check_taskset() -> dict[str, Any]:
    result: dict[str, Any] = {"checks": {}, "errors": [], "skipped": []}
    for numeric_id in (232, 306, 237):
        row = load_mbpp_task(numeric_id)
        key = f"mbpp/{numeric_id}"
        if row is None:
            result["skipped"].append(f"{key} not present")
            continue
        if "dataset_error" in row:
            result["skipped"].append(f"MBPP unavailable: {row['dataset_error']}")
            return result
        tests = [str(x).strip() for x in row.get("test_list", []) if str(x).strip()]
        fn = function_name_from_tests(tests)
        arity = function_arity_from_tests(tests, fn)
        signature = function_signature_hint(fn, arity)
        if numeric_id == 232:
            ok = fn == "larg_nnum"
            result["checks"]["mbpp_232_function_name_larg_nnum"] = ok
            result["checks"]["mbpp_232_function_name"] = fn
            if not ok:
                result["errors"].append(f"mbpp/232 resolved to {fn!r}, expected 'larg_nnum'")
        elif numeric_id == 306:
            ok = arity == 4 and signature.count("arg") == 4
            result["checks"]["mbpp_306_signature_arity_4"] = ok
            result["checks"]["mbpp_306_signature"] = signature
            if not ok:
                result["errors"].append(f"mbpp/306 signature hint was {signature!r}, expected arity 4")
        elif numeric_id == 237:
            from build_code_branch_taskset_v2 import mbpp_task_note

            note = mbpp_task_note("mbpp/237")
            ok = "canonicalize" in note.lower() and "tuple" in note.lower()
            result["checks"]["mbpp_237_tuple_canonicalization_hint"] = ok
            result["checks"]["mbpp_237_note"] = note
            if not ok:
                result["errors"].append("mbpp/237 tuple canonicalization hint missing")
    return result


def check_safety() -> dict[str, Any]:
    samples = {
        "sys_setrecursionlimit_allowed": "import sys\nsys.setrecursionlimit(10000)\ndef f():\n    return 1\n",
        "from_sys_setrecursionlimit_allowed": "from sys import setrecursionlimit\nsetrecursionlimit(10000)\ndef f():\n    return 1\n",
        "os_rejected": "import os\ndef f():\n    return os.getcwd()\n",
        "subprocess_rejected": "import subprocess\ndef f():\n    return subprocess.check_output(['true'])\n",
        "socket_rejected": "import socket\ndef f():\n    return socket.socket()\n",
        "file_open_rejected": "def f():\n    return open('/tmp/x').read()\n",
        "sys_exit_rejected": "import sys\ndef f():\n    return sys.exit(0)\n",
    }
    checks: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for name, code in samples.items():
        ok, reason = safety_check(code)
        reasons[name] = reason
        if name.endswith("_allowed"):
            checks[name] = ok
        else:
            checks[name] = not ok
    errors = [f"safety check failed: {name}" for name, ok in checks.items() if not ok]
    return {"checks": checks, "reasons": reasons, "errors": errors}


def verdict_for(wrapper: dict[str, Any], taskset: dict[str, Any], safety: dict[str, Any]) -> str:
    if wrapper.get("errors") or safety.get("errors"):
        return "BLOCKED"
    if taskset.get("errors") or taskset.get("skipped"):
        return "PARTIAL"
    return "READY"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch v2 Patch Status",
        "",
        f"CODE_V2_PATCH_STATUS = {payload['code_v2_patch_status']}",
        "",
        "## Wrapper",
        "",
    ]
    for key, value in payload["wrapper"]["checks"].items():
        if isinstance(value, bool):
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Taskset", ""])
    for key, value in payload["taskset"]["checks"].items():
        lines.append(f"- {key}: `{value}`")
    if payload["taskset"].get("skipped"):
        lines.append(f"- skipped: `{payload['taskset']['skipped']}`")
    lines.extend(["", "## Safety", ""])
    for key, value in payload["safety"]["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Errors", ""])
    errors = payload["wrapper"].get("errors", []) + payload["taskset"].get("errors", []) + payload["safety"].get("errors", [])
    lines.extend(f"- {error}" for error in errors)
    if not errors:
        lines.append("None.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    wrapper = check_wrapper()
    taskset = check_taskset()
    safety = check_safety()
    verdict = verdict_for(wrapper, taskset, safety)
    payload = {
        "code_v2_patch_status": verdict,
        "wrapper": wrapper,
        "taskset": taskset,
        "safety": safety,
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_V2_PATCH_STATUS = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
