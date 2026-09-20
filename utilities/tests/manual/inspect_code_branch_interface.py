"""Inspect safe local-agent invocation points for the code branch pilot."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, import_agent_modules, repo_path, write_json


DEFAULT_JSON = REPORT_DIR / "code_branch_interface_inspection_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_interface_inspection_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def has_callable(module: Any, name: str) -> bool:
    return callable(getattr(module, name, None))


def signature(module: Any, name: str) -> str:
    fn = getattr(module, name, None)
    if not callable(fn):
        return ""
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "signature_unavailable"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Interface Inspection",
        "",
        f"CODE_INTERFACE_VERDICT = {payload['code_interface_verdict']}",
        "",
        "## Importability",
        "",
        "| module | ok | error |",
        "| --- | ---: | --- |",
    ]
    for row in payload["imports"]:
        lines.append(f"| `{row['module']}` | {row['ok']} | `{row.get('error', '')}` |")
    lines.extend([
        "",
        "## Callable Surface",
        "",
        "| callable | available | signature |",
        "| --- | ---: | --- |",
    ])
    for row in payload["callables"]:
        lines.append(f"| `{row['name']}` | {row['available']} | `{row['signature']}` |")
    lines.extend([
        "",
        "## Code Settings",
        "",
        "| setting | value |",
        "| --- | ---: |",
    ])
    for key, value in payload["settings"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Invocation Decision",
        "",
        payload["decision"],
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    modules = {}
    imports = []
    verdict = "READY"
    try:
        modules = import_agent_modules()
    except Exception as exc:
        verdict = "BLOCKED"
        imports.append({"module": "local_agent_bundle", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    if modules:
        for name in ("direct", "agent", "policies", "config", "backend", "prompts"):
            imports.append({"module": f"src.local_agent.{name}", "ok": True, "error": ""})

    callables = []
    if modules:
        surface = [
            ("ouro_direct.direct_answer", modules["direct"], "direct_answer"),
            ("ouro_agent_improved.run_task_mode", modules["agent"], "run_task_mode"),
            ("ouro_agent_improved.run_agent_task", modules["agent"], "run_agent_task"),
            ("ouro_agent_improved.parse_action", modules["agent"], "parse_action"),
            ("ouro_agent_improved.code_action_prefill_for_state", modules["agent"], "code_action_prefill_for_state"),
            ("ouro_policies.sanitize_tool_input", modules["policies"], "sanitize_tool_input"),
            ("ouro_backend.DeepThinkModelManager", modules["backend"], "DeepThinkModelManager"),
        ]
        for display, module, attr in surface:
            callables.append({
                "name": display,
                "available": has_callable(module, attr),
                "signature": signature(module, attr),
            })
        required = [
            ("direct", "direct_answer"),
            ("agent", "run_task_mode"),
            ("agent", "parse_action"),
            ("policies", "sanitize_tool_input"),
            ("backend", "DeepThinkModelManager"),
        ]
        if not all(has_callable(modules[module], attr) for module, attr in required):
            verdict = "BLOCKED"

    settings = {}
    if modules:
        config = modules["config"]
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

    decision = (
        "Safe path found: use the local Ouro-RLTT backend through the local-agent prompt wrappers, "
        "`sanitize_tool_input`, and the direct/action code interfaces. The inspection script did not load the model."
        if verdict == "READY"
        else "No safe local-agent invocation path was found; later stages should not run."
    )
    payload = {
        "code_interface_verdict": verdict,
        "imports": imports,
        "callables": callables,
        "settings": settings,
        "decision": decision,
        "outputs": {
            "json": repo_path(Path(args.output)),
            "md": repo_path(Path(args.output_md)),
        },
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_INTERFACE_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
