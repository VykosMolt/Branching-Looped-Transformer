"""Inspect read-only BG controller artifacts for the v8.1 controller layer."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_controller import (  # noqa: E402
    BGHeadSpec,
    HEAD_CLASSES,
    config_dim,
)


REPORT_DIR = PROJECT_ROOT / "opi/taps/probes"
OUTPUT_JSON = REPORT_DIR / "bg_controller_artifact_inventory_2026-05-18.json"
OUTPUT_MD = REPORT_DIR / "bg_controller_artifact_inventory_2026-05-18.md"

ARTIFACTS = {
    "bg_head_registry_pt": REPORT_DIR / "bg_head_registry_2026-05-17.pt",
    "bg_head_registry_json": REPORT_DIR / "bg_head_registry_2026-05-17.json",
    "mixed_domain_tiny_heads_pt": REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.pt",
    "mixed_domain_tiny_heads_json": REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.json",
    "mixed_head_controller_implications_json": REPORT_DIR / "mixed_head_controller_implications_2026-05-17.json",
    "bg_controller_policy_simulator_summary_json": REPORT_DIR / "bg_controller_policy_simulator_2026-05-17_summary.json",
    "bg_policy_sim_eval_bundle_pt": REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.pt",
    "bg_policy_sim_eval_bundle_json": REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.json",
    "mixed_tap_features_pt": REPORT_DIR / "mixed_tap_features_2026-05-17.pt",
    "bg_cross_domain_eval_matrix_json": REPORT_DIR / "bg_cross_domain_eval_matrix_2026-05-17.json",
}

LOCKED = {
    "hh_general": {
        "source": "bg_head_registry_pt",
        "group": "HH",
        "config": "47_concat_L1_L4",
        "architecture": "AntisymLinearNoNorm",
        "family": "HH",
        "domain_role": "HH/preference/unknown/default semantic preference head",
    },
    "objective_mixed": {
        "source": "mixed_domain_tiny_heads_pt",
        "group": "MIX_CODE_REASONING",
        "config": "36_L4",
        "architecture": "AntisymLinearNoNorm",
        "family": "MIX_CODE_REASONING",
        "domain_role": "default objective selector for objective branch selection",
    },
    "code_specialist_backup": {
        "source": "bg_head_registry_pt",
        "group": "CODE",
        "config": "36_L4",
        "architecture": "AntisymLinear",
        "family": "CODE",
        "domain_role": "strict-clean code backup/diagnostic/escalation",
    },
}


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_group(row: dict[str, Any]) -> str:
    return str(row.get("head_group") or row.get("head_family") or "")


def find_head(rows: list[dict[str, Any]], group: str, config: str, architecture: str) -> dict[str, Any] | None:
    for row in rows:
        if row_group(row) == group and row.get("config") == config and row.get("architecture") == architecture:
            return row
    return None


def check_architecture(row: dict[str, Any]) -> dict[str, Any]:
    architecture = str(row["architecture"])
    dim = int(row.get("dim", config_dim(str(row["config"]))))
    state_dict = row.get("state_dict") or {}
    result = {
        "architecture": architecture,
        "dim": dim,
        "state_dict_keys": sorted(state_dict.keys()),
        "compatible": False,
        "error": "",
    }
    try:
        head = HEAD_CLASSES[architecture](dim)
        head.load_state_dict(state_dict)
        left = torch.randn(2, dim)
        right = torch.randn(2, dim)
        with torch.no_grad():
            score = head(left, right)
            reverse = head(right, left)
        result["output_shape"] = list(score.shape)
        result["antisymmetry_max_abs"] = float((score + reverse).abs().max())
        result["compatible"] = bool(result["antisymmetry_max_abs"] < 1e-5)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def calibration_from_bundle(bundle: dict[str, Any], head_keys: dict[str, str]) -> dict[str, Any]:
    head_scores = bundle.get("head_scores") or {}
    out: dict[str, Any] = {"available": False, "heads": {}}
    for name, key in head_keys.items():
        by_domain = {}
        if key in head_scores:
            for domain, row in head_scores[key].items():
                metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
                value = metrics.get("margin_std")
                try:
                    std = float(value)
                except Exception:
                    continue
                if not math.isnan(std):
                    by_domain[str(domain)] = std
        vals = [v for v in by_domain.values() if v > 1e-8]
        out["heads"][name] = {
            "head_key": key,
            "domain_margin_std": by_domain,
            "fallback_std": float(mean(vals)) if vals else None,
            "domains_with_stats": len(vals),
        }
    out["available"] = any(row["domains_with_stats"] > 0 for row in out["heads"].values())
    return out


def inspect_feature_shape(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    try:
        payload = load_pt(path)
        row = (payload.get("candidate_features") or [None])[0]
        pooled = row.get("pooled") if isinstance(row, dict) else None
        return {
            "available": pooled is not None,
            "sample_shape": list(pooled.shape) if isinstance(pooled, torch.Tensor) else None,
            "candidate_feature_count": len(payload.get("candidate_features") or []),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BG Controller Artifact Inventory (2026-05-18)",
        "",
        f"BG_CONTROLLER_ARTIFACT_VERDICT = {payload['top_lines']['BG_CONTROLLER_ARTIFACT_VERDICT']}",
        "",
        "## Locked Heads",
        "| head | status | artifact | family | config | architecture | dim | calibration std |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for name, row in payload["locked_heads"].items():
        spec = row.get("spec") or {}
        cal = spec.get("calibration_std")
        cal_text = "NA" if cal is None else f"{float(cal):.6f}"
        lines.append(
            f"| {name} | {row['status']} | `{row.get('artifact_path')}` | {spec.get('family', 'NA')} | "
            f"{spec.get('config', 'NA')} | {spec.get('architecture', 'NA')} | {spec.get('dim', 'NA')} | {cal_text} |"
        )
    lines.extend(
        [
            "",
            "## Required Artifacts",
            "| artifact | exists | path |",
            "| --- | --- | --- |",
        ]
    )
    for name, row in payload["artifacts"].items():
        lines.append(f"| {name} | {row['exists']} | `{row['path']}` |")
    lines.extend(
        [
            "",
            "## Feature Shapes",
            f"- expected `36_L4`: `{payload['expected_input_dims']['36_L4']}`",
            f"- expected `47_concat_L1_L4`: `{payload['expected_input_dims']['47_concat_L1_L4']}`",
            f"- mixed feature sample shape: `{payload['feature_sample'].get('sample_shape')}`",
            "",
            "## Calibration And Replay",
            f"- margin calibration stats available: `{payload['margin_calibration']['available']}`",
            f"- replay eval bundle exists: `{payload['replay_bundle']['exists']}`",
            "",
            "## Blockers",
        ]
    )
    blockers = payload.get("blockers") or []
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    artifacts = {
        name: {"path": repo_path(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        for name, path in ARTIFACTS.items()
    }
    registry = load_pt(ARTIFACTS["bg_head_registry_pt"]) if ARTIFACTS["bg_head_registry_pt"].exists() else {}
    mixed = load_pt(ARTIFACTS["mixed_domain_tiny_heads_pt"]) if ARTIFACTS["mixed_domain_tiny_heads_pt"].exists() else {}
    bundle = load_pt(ARTIFACTS["bg_policy_sim_eval_bundle_pt"]) if ARTIFACTS["bg_policy_sim_eval_bundle_pt"].exists() else {}

    rows_by_source = {
        "bg_head_registry_pt": list(registry.get("heads", []) or []),
        "mixed_domain_tiny_heads_pt": list(mixed.get("heads", []) or []),
    }
    role_primary = dict(bundle.get("role_primary") or bundle.get("meta", {}).get("role_primary") or {})
    head_keys = {
        "hh_general": role_primary.get("HH_GENERAL", "HH::47_concat_L1_L4::AntisymLinearNoNorm"),
        "objective_mixed": role_primary.get("OBJECTIVE_MIXED_PRIMARY", "MIX_CODE_REASONING::36_L4::AntisymLinearNoNorm"),
        "code_specialist_backup": role_primary.get("CODE_SPECIALIST", "CODE::36_L4::AntisymLinear"),
    }
    calibration = calibration_from_bundle(bundle, head_keys)

    locked_heads: dict[str, Any] = {}
    blockers: list[str] = []
    for name, request in LOCKED.items():
        artifact_path = ARTIFACTS[request["source"]]
        row = find_head(rows_by_source.get(request["source"], []), request["group"], request["config"], request["architecture"])
        if row is None:
            locked_heads[name] = {
                "status": "missing",
                "artifact_path": repo_path(artifact_path),
                "request": request,
            }
            if name in {"hh_general", "objective_mixed"}:
                blockers.append(f"missing required production head: {name}")
            continue
        arch = check_architecture(row)
        fallback_std = calibration["heads"].get(name, {}).get("fallback_std")
        spec = BGHeadSpec(
            name=name,
            family=str(request["family"]),
            config=str(row["config"]),
            architecture=str(row["architecture"]),
            artifact_path=repo_path(artifact_path),
            domain_role=str(request["domain_role"]),
            dim=int(row["dim"]),
            calibration_std=fallback_std,
        )
        status = "ready" if arch.get("compatible") else "incompatible"
        if status != "ready":
            blockers.append(f"incompatible head architecture for {name}: {arch.get('error') or arch.get('antisymmetry_max_abs')}")
        locked_heads[name] = {
            "status": status,
            "artifact_path": repo_path(artifact_path),
            "head_key": head_keys[name],
            "spec": spec.__dict__,
            "architecture_check": arch,
        }

    replay_exists = ARTIFACTS["bg_policy_sim_eval_bundle_pt"].exists() and ARTIFACTS["bg_policy_sim_eval_bundle_json"].exists()
    have_hh = locked_heads.get("hh_general", {}).get("status") == "ready"
    have_obj = locked_heads.get("objective_mixed", {}).get("status") == "ready"
    have_code = locked_heads.get("code_specialist_backup", {}).get("status") == "ready"
    if have_hh and have_obj and have_code and replay_exists:
        verdict = "READY"
    elif have_hh and have_obj:
        verdict = "PARTIAL"
        if not have_code:
            blockers.append("code specialist backup missing")
        if not replay_exists:
            blockers.append("replay eval bundle missing")
    else:
        verdict = "BLOCKED"

    payload = {
        "top_lines": {"BG_CONTROLLER_ARTIFACT_VERDICT": verdict},
        "artifacts": artifacts,
        "locked_heads": locked_heads,
        "expected_input_dims": {
            "36_L4": config_dim("36_L4"),
            "47_concat_L1_L4": config_dim("47_concat_L1_L4"),
            "47_concat_all_loops": config_dim("47_concat_all_loops"),
        },
        "architecture_classes": sorted(HEAD_CLASSES.keys()),
        "margin_calibration": calibration,
        "replay_bundle": {
            "exists": replay_exists,
            "pt": repo_path(ARTIFACTS["bg_policy_sim_eval_bundle_pt"]),
            "json": repo_path(ARTIFACTS["bg_policy_sim_eval_bundle_json"]),
            "domains": sorted((bundle.get("domains") or {}).keys()),
        },
        "feature_sample": inspect_feature_shape(ARTIFACTS["mixed_tap_features_pt"]),
        "blockers": blockers,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_md(OUTPUT_MD, payload)
    print(f"BG_CONTROLLER_ARTIFACT_VERDICT = {verdict}")
    print(f"wrote {repo_path(OUTPUT_JSON)}")
    print(f"wrote {repo_path(OUTPUT_MD)}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
