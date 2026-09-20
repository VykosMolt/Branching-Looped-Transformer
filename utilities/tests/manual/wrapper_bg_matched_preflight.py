"""Preflight checks for the wrapper-matched BG candidate-selection experiment."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from wrapper_bg_matched_lib import (
    DEVIL_INVENTORY,
    OUT_DIR,
    PROJECT_ROOT,
    TASKSET_SOURCE,
    repo_path,
    write_json,
    write_text,
)


OUT_JSON = OUT_DIR / "preflight.json"
OUT_MD = OUT_DIR / "preflight.md"


def module_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"name": name, "ok": True, "path": getattr(module, "__file__", ""), "error": ""}
    except Exception as exc:  # noqa: BLE001 - report
        return {"name": name, "ok": False, "path": "", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, Any] = {}
    for name in (
        "candidate_export",
        "candidate_capture",
        "src.evaluator.bg_controller",
        "src.evaluator.bg_transformer_features",
    ):
        statuses[name] = module_status(name)

    dataclass_lightweight = False
    capture_functions_callable = False
    controller_loaded = False
    feature_extractor_imported = statuses["src.evaluator.bg_transformer_features"]["ok"]
    controller_error = ""
    try:
        from candidate_export import CandidateArtifact, CandidateTrace

        dataclass_lightweight = inspect.isclass(CandidateArtifact) and inspect.isclass(CandidateTrace)
    except Exception as exc:  # noqa: BLE001
        statuses["candidate_export"]["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from candidate_capture import run_agent_with_candidates, run_direct_with_candidates

        capture_functions_callable = callable(run_agent_with_candidates) and callable(run_direct_with_candidates)
    except Exception as exc:  # noqa: BLE001
        statuses["candidate_capture"]["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from src.evaluator.bg_controller import BGController

        controller = BGController.from_artifacts(device="cpu")
        controller_loaded = all(name in controller.heads for name in ("hh_general", "objective_mixed", "code_specialist_backup"))
    except Exception as exc:  # noqa: BLE001
        controller_error = f"{type(exc).__name__}: {exc}"

    helper_paths = {
        "code_branch_pilot_lib": PROJECT_ROOT / "shared/utilities/tests/manual/code_branch_pilot_lib.py",
        "evaluate_code_branch_candidates_v2": PROJECT_ROOT / "shared/utilities/tests/manual/evaluate_code_branch_candidates_v2.py",
        "build_code_branch_taskset_v2": PROJECT_ROOT / "shared/utilities/tests/manual/build_code_branch_taskset_v2.py",
        "taskset_source": TASKSET_SOURCE,
        "devil_inventory": DEVIL_INVENTORY,
    }
    helper_status = {name: {"path": repo_path(path), "exists": path.exists()} for name, path in helper_paths.items()}
    model_path = PROJECT_ROOT / "shared/models/ouro_rltt_local"
    code_task_source_available = TASKSET_SOURCE.exists()
    devil_available = DEVIL_INVENTORY.exists()

    if dataclass_lightweight and capture_functions_callable and controller_loaded and feature_extractor_imported and code_task_source_available:
        verdict = "READY"
    elif dataclass_lightweight and capture_functions_callable and controller_loaded:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    payload = {
        "WRAPPER_BG_PREFLIGHT_VERDICT": verdict,
        "module_status": statuses,
        "candidate_export_dataclasses_lightweight": dataclass_lightweight,
        "candidate_capture_functions_callable": capture_functions_callable,
        "bg_controller_locked_heads_loaded": controller_loaded,
        "bg_controller_error": controller_error,
        "bg_transformer_feature_extractor_imported": feature_extractor_imported,
        "feature_extractor_instantiation": "deferred_to_feature_capture",
        "local_ouro_model": {"path": repo_path(model_path), "exists": model_path.exists()},
        "helper_status": helper_status,
        "code_task_source_available": code_task_source_available,
        "devil_task_inventory_available": devil_available,
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Wrapper-Matched BG Preflight",
        "",
        f"WRAPPER_BG_PREFLIGHT_VERDICT = {verdict}",
        "",
        f"- candidate export dataclasses lightweight: `{dataclass_lightweight}`",
        f"- candidate capture functions callable: `{capture_functions_callable}`",
        f"- BG controller locked heads loaded: `{controller_loaded}`",
        f"- BG feature extractor imported: `{feature_extractor_imported}`",
        f"- feature extractor instantiation: `deferred_to_feature_capture`",
        f"- local model exists: `{model_path.exists()}`",
        f"- task source exists: `{code_task_source_available}`",
        f"- devil inventory exists: `{devil_available}`",
    ]
    if controller_error:
        lines.extend(["", f"Controller error: `{controller_error}`"])
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_BG_PREFLIGHT_VERDICT = {verdict}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
