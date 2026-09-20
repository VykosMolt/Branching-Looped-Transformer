"""Locate local devil code task definitions and existing tests for BG smoke."""
from __future__ import annotations

import ast
import json
import shlex
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "opi/taps/probes"
OUT_JSON = REPORT_DIR / "bg_devil_task_inventory_2026-05-18.json"
OUT_MD = REPORT_DIR / "bg_devil_task_inventory_2026-05-18.md"
TARGETS = {
    "offline_dynamic_connectivity": "local_dsa/offline_dynamic_connectivity",
    "minimum_xor_paths": "local_dsa/minimum_xor_paths",
}
SEARCH_ROOTS = [
    "opi/taps/probes",
    "shared/utilities/tests/manual",
    "datasets",
    "data",
    "src",
    "tests",
]


def _safe_read(path: Path, limit: int = 5_000_000) -> str:
    if not path.exists() or not path.is_file() or path.stat().st_size > limit:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _literalish_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literalish_eval(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literalish_eval(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_literalish_eval(key): _literalish_eval(value) for key, value in zip(node.keys, node.values)}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strip"
        and not node.args
        and not node.keywords
    ):
        value = _literalish_eval(node.func.value)
        return value.strip() if isinstance(value, str) else value
    raise ValueError(f"unsupported task literal node: {ast.dump(node)[:200]}")


def _load_devil_tasks_from_v2() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "shared/utilities/tests/manual/build_code_branch_taskset_v2.py"
    text = _safe_read(path)
    if not text:
        return []
    tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DEVIL_TASKS":
            tasks = _literalish_eval(node.value)
            for task in tasks:
                task["source_artifact_path"] = str(path.relative_to(PROJECT_ROOT))
            return list(tasks)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEVIL_TASKS":
                    tasks = _literalish_eval(node.value)
                    for task in tasks:
                        task["source_artifact_path"] = str(path.relative_to(PROJECT_ROOT))
                    return list(tasks)
    return []


def _search_mentions() -> dict[str, list[str]]:
    mentions: dict[str, list[str]] = {name: [] for name in TARGETS}
    for root_name in SEARCH_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in {".pt", ".bin", ".safetensors", ".png", ".jpg", ".jpeg"}:
                continue
            text = _safe_read(path)
            if not text:
                continue
            rel = str(path.relative_to(PROJECT_ROOT))
            for name in TARGETS:
                if name in text:
                    mentions[name].append(rel)
    return mentions


def _write_task_files(task: dict[str, Any], short_name: str) -> dict[str, str]:
    prompt_path = REPORT_DIR / f"bg_devil_task_{short_name}_prompt_2026-05-18.txt"
    test_path = REPORT_DIR / f"bg_devil_task_{short_name}_2026-05-18.json"
    prompt_path.write_text(str(task.get("prompt", "")).strip() + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "prompt_path": str(prompt_path.relative_to(PROJECT_ROOT)),
        "optional_test_json": str(test_path.relative_to(PROJECT_ROOT)),
    }


def _command_for(short_name: str, paths: dict[str, str]) -> str:
    output = f"opi/taps/probes/bg_transformer_best_of_n_smoke_{short_name}_2026-05-18.json"
    parts = [
        "shared/venv/bin/python",
        "-u",
        "shared/utilities/tests/manual/run_bg_transformer_best_of_n_smoke.py",
        "--prompt-file",
        paths["prompt_path"],
        "--domain-hint",
        "code",
        "--n-candidates",
        "4",
        "--max-new-tokens",
        "512",
        "--temperature",
        "0.7",
        "--top-p",
        "0.95",
        "--mode",
        "conservative",
        "--device",
        "cuda",
        "--output",
        output,
        "--optional-test-json",
        paths["optional_test_json"],
    ]
    return " ".join(shlex.quote(part) for part in parts)


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# BG Devil Code Task Inventory (2026-05-18)",
        "",
        f"BG_DEVIL_TASK_INVENTORY_VERDICT = {payload['verdict']}",
        "",
        "## Tasks",
        "",
    ]
    for name, row in payload["tasks"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Status: `{row['status']}`",
                f"- Task id: `{row.get('task_id')}`",
                f"- Function: `{row.get('function_name')}`",
                f"- Source artifact: `{row.get('source_artifact_path')}`",
                f"- Prompt path: `{row.get('prompt_path')}`",
                f"- Optional test JSON: `{row.get('optional_test_json')}`",
                f"- Test status: `{row.get('test_status')}`",
                "",
            ]
        )
    if payload.get("devil_smoke_commands"):
        lines.extend(["## Runnable Devil Smoke Commands", ""])
        lines.extend(f"```bash\n{cmd}\n```" for cmd in payload["devil_smoke_commands"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = _load_devil_tasks_from_v2()
    mentions = _search_mentions()
    by_name = {str(task.get("function_name")): task for task in tasks}
    out_tasks: dict[str, dict[str, Any]] = {}
    commands: list[str] = []

    for short_name, task_id in TARGETS.items():
        task = by_name.get(short_name)
        if task is None:
            out_tasks[short_name] = {
                "status": "TASK_NOT_FOUND",
                "task_id": task_id,
                "mentions": mentions.get(short_name, []),
                "test_status": "TESTS_MISSING",
            }
            continue
        public_tests = list(task.get("public_tests") or [])
        hidden_tests = list(task.get("hidden_tests") or [])
        test_status = "READY" if public_tests or hidden_tests else "TESTS_MISSING"
        paths = _write_task_files(task, short_name)
        command = _command_for(short_name, paths)
        commands.append(command)
        out_tasks[short_name] = {
            "status": "FOUND",
            "task_id": task.get("task_id"),
            "function_name": task.get("function_name"),
            "prompt": task.get("prompt"),
            "prompt_chars": len(str(task.get("prompt", ""))),
            "public_test_count": len(public_tests),
            "hidden_test_count": len(hidden_tests),
            "test_status": test_status,
            "tests_visibility": task.get("tests_visibility"),
            "timeout_seconds": task.get("timeout_seconds"),
            "source_artifact_path": task.get("source_artifact_path"),
            "mentions": mentions.get(short_name, []),
            **paths,
            "smoke_command": command,
        }

    found = [row for row in out_tasks.values() if row["status"] == "FOUND"]
    ready = [row for row in found if row.get("test_status") == "READY"]
    if len(ready) == len(TARGETS):
        verdict = "READY"
    elif found:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    payload = {
        "verdict": verdict,
        "BG_DEVIL_TASK_INVENTORY_VERDICT": verdict,
        "tasks": out_tasks,
        "devil_smoke_commands": commands,
        "search_roots": SEARCH_ROOTS,
    }
    write_reports(payload)
    print(f"BG_DEVIL_TASK_INVENTORY_VERDICT = {verdict}")
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")
    for cmd in commands:
        print(cmd)
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
