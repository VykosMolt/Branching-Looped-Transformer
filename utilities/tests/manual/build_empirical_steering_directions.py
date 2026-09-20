"""Build empirical BG steering directions from frozen Stage 1 prefix features."""
from __future__ import annotations

import time
import traceback
from typing import Any

import torch

from bg_empirical_steering_common import (
    OUT_ROOT,
    PROJECT_ROOT,
    auc_score,
    cosine,
    examples_for_target,
    heldout_split,
    pairwise_accuracy,
    rel,
    stack_xy,
    train_logistic_direction,
    unit,
    write_json,
    write_md,
)
from bg_stage2_steering_preflight import head_weight_vector, load_head_row
from src.evaluator.bg_controller import BGController


OUT_PT = OUT_ROOT / "directions.pt"
OUT_JSON = OUT_ROOT / "directions.json"
OUT_MD = OUT_ROOT / "directions.md"
PREFLIGHT_JSON = OUT_ROOT / "preflight.json"
SEED = 20260518


def load_json(path, default=None):
    import json

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def direction_metrics(direction: torch.Tensor, x_train: torch.Tensor, y_train: torch.Tensor, x_heldout: torch.Tensor, y_heldout: torch.Tensor) -> dict[str, Any]:
    scores_train = x_train @ direction
    scores_heldout = x_heldout @ direction
    return {
        "train_auc": auc_score(y_train, scores_train),
        "heldout_auc": auc_score(y_heldout, scores_heldout),
        "heldout_pairwise_accuracy": pairwise_accuracy(y_heldout, scores_heldout),
    }


def build_for_target(target: dict[str, Any], controller: BGController) -> dict[str, Any]:
    domain = str(target["domain"])
    prefix_length = int(target["prefix_length"])
    config = str(target["config"])
    examples = examples_for_target(domain, prefix_length, config)
    train_tasks, heldout_tasks = heldout_split(examples, heldout_task_count=8 if target.get("target_id") == "T1" else 4)
    x_train, y_train, train_rows = stack_xy(examples, train_tasks)
    x_heldout, y_heldout, heldout_rows = stack_xy(examples, heldout_tasks)
    success_train = x_train[y_train > 0.5]
    fail_train = x_train[y_train <= 0.5]
    if int(success_train.shape[0]) == 0 or int(fail_train.shape[0]) == 0:
        raise RuntimeError(f"target {target['target_id']} has no train contrast after split")

    directions = []
    raw_head = load_head_row(str(target["head_id"]), controller)
    raw_vec = unit(head_weight_vector(raw_head))
    raw_metrics = direction_metrics(raw_vec, x_train, y_train, x_heldout, y_heldout)
    directions.append(
        {
            "direction_name": "RAW_NONORM_READOUT",
            "direction_source": str(target["head_id"]),
            "vector": raw_vec,
            "norm_before": float(torch.linalg.vector_norm(head_weight_vector(raw_head)).item()),
            **raw_metrics,
        }
    )

    mean_raw = success_train.mean(dim=0) - fail_train.mean(dim=0)
    mean_vec = unit(mean_raw)
    directions.append(
        {
            "direction_name": "EMPIRICAL_MEAN_DIFF",
            "direction_source": "mean(success_prefix_features)-mean(fail_prefix_features)",
            "vector": mean_vec,
            "norm_before": float(torch.linalg.vector_norm(mean_raw).item()),
            **direction_metrics(mean_vec, x_train, y_train, x_heldout, y_heldout),
        }
    )

    diag_std = x_train.std(dim=0, unbiased=False).clamp(min=1e-4)
    white_raw = mean_raw / diag_std
    white_vec = unit(white_raw)
    directions.append(
        {
            "direction_name": "EMPIRICAL_WHITENED_DIFF",
            "direction_source": "diag_std^-1 * mean_diff",
            "vector": white_vec,
            "norm_before": float(torch.linalg.vector_norm(white_raw).item()),
            **direction_metrics(white_vec, x_train, y_train, x_heldout, y_heldout),
        }
    )

    logistic = train_logistic_direction(x_train, y_train, x_heldout, y_heldout, seed=SEED + int(prefix_length))
    directions.append(
        {
            "direction_name": "LOGISTIC_SUCCESS_PROBE",
            "direction_source": "torch_logistic_regression_on_frozen_prefix_features",
            "vector": logistic["direction"],
            "norm_before": logistic["norm_before"],
            "train_auc": logistic["train_auc"],
            "heldout_auc": logistic["heldout_auc"],
            "heldout_pairwise_accuracy": logistic["heldout_pairwise_accuracy"],
            "train_loss": logistic["train_loss"],
        }
    )

    # Auxiliary 24_L4 empirical direction is built for geometry diagnostics only.
    auxiliary: list[dict[str, Any]] = []
    try:
        aux_examples = examples_for_target(domain, prefix_length, "24_L4")
        aux_train, aux_heldout = heldout_split(aux_examples, heldout_task_count=len(heldout_tasks))
        ax_train, ay_train, _ = stack_xy(aux_examples, aux_train)
        ax_heldout, ay_heldout, _ = stack_xy(aux_examples, aux_heldout)
        aux_success = ax_train[ay_train > 0.5]
        aux_fail = ax_train[ay_train <= 0.5]
        if int(aux_success.shape[0]) >= 10 and int(aux_fail.shape[0]) >= 10:
            aux_raw = aux_success.mean(dim=0) - aux_fail.mean(dim=0)
            aux_vec = unit(aux_raw)
            auxiliary.append(
                {
                    "direction_name": "AUX_24_L4_EMPIRICAL_MEAN_DIFF",
                    "config": "24_L4",
                    "layer": 24,
                    "norm_before": float(torch.linalg.vector_norm(aux_raw).item()),
                    **direction_metrics(aux_vec, ax_train, ay_train, ax_heldout, ay_heldout),
                }
            )
    except Exception:
        auxiliary = []

    cosines: dict[str, float] = {}
    for left in directions:
        for right in directions:
            if left["direction_name"] >= right["direction_name"]:
                continue
            cosines[f"{left['direction_name']}__{right['direction_name']}"] = cosine(left["vector"], right["vector"])
    for row in directions:
        row["cosine_to_raw_nonorm"] = cosine(raw_vec, row["vector"])
        row["direction_norm"] = float(torch.linalg.vector_norm(row["vector"]).item())

    return {
        "target_id": target["target_id"],
        "domain": domain,
        "prefix_length": prefix_length,
        "config": config,
        "layer": int(target["layer"]),
        "head_id": target["head_id"],
        "train_task_ids": sorted(train_tasks),
        "heldout_task_ids": sorted(heldout_tasks),
        "training_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "training_successes": int(y_train.sum().item()),
        "training_failures": int((1.0 - y_train).sum().item()),
        "heldout_successes": int(y_heldout.sum().item()),
        "heldout_failures": int((1.0 - y_heldout).sum().item()),
        "directions": directions,
        "direction_cosines": cosines,
        "auxiliary_directions": auxiliary,
    }


def json_safe_target(target_payload: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in target_payload.items() if k != "directions"}
    out["directions"] = []
    for row in target_payload["directions"]:
        out["directions"].append({k: v for k, v in row.items() if k != "vector"})
    return out


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(PREFLIGHT_JSON, {})
    if preflight.get("BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT") == "BLOCKED":
        payload = {"BG_EMPIRICAL_DIRECTION_BUILD_VERDICT": "BLOCKED", "blocker": "preflight blocked", "targets": []}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Empirical Steering Directions", "", "BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = BLOCKED"])
        print("BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = BLOCKED")
        return 1
    try:
        controller = BGController.from_artifacts(device="cpu")
        selected = list(preflight.get("selected_targets") or [])
        # Keep the sweep focused on the requested primary target unless the
        # caller later chooses to use secondary directions explicitly.
        targets = [row for row in selected if row.get("target_id") == "T1"] or selected[:1]
        built = [build_for_target(target, controller) for target in targets]
        direction_count_primary = len(built[0]["directions"]) if built else 0
        if direction_count_primary >= 3 and any(row["direction_name"] == "EMPIRICAL_MEAN_DIFF" for row in built[0]["directions"]):
            verdict = "READY"
        elif direction_count_primary >= 2:
            verdict = "PARTIAL"
        else:
            verdict = "BLOCKED"

        pt_payload = {
            "BG_EMPIRICAL_DIRECTION_BUILD_VERDICT": verdict,
            "targets": built,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        torch.save(pt_payload, OUT_PT)
        json_payload = {
            "BG_EMPIRICAL_DIRECTION_BUILD_VERDICT": verdict,
            "targets": [json_safe_target(row) for row in built],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, json_payload)

        lines = [
            "# BG Empirical Steering Directions",
            "",
            f"BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = {verdict}",
            "",
            "| target | direction | config | train | heldout | heldout AUC | cosine(raw) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for target in built:
            for row in target["directions"]:
                auc = row.get("heldout_auc")
                cos = row.get("cosine_to_raw_nonorm")
                lines.append(
                    f"| `{target['target_id']}` | `{row['direction_name']}` | `{target['config']}` | "
                    f"{target['training_examples']} | {target['heldout_examples']} | "
                    f"{auc if auc is not None else ''} | {cos if cos is not None else ''} |"
                )
        write_md(OUT_MD, lines)
        print(f"BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = {verdict}")
        print(f"Wrote {rel(OUT_PT)}")
        print(f"Wrote {rel(OUT_JSON)}")
        print(f"Wrote {rel(OUT_MD)}")
        return 0 if verdict in {"READY", "PARTIAL"} else 1
    except Exception as exc:
        err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        payload = {"BG_EMPIRICAL_DIRECTION_BUILD_VERDICT": "BLOCKED", "error": err, "elapsed_seconds": round(time.time() - started, 3)}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Empirical Steering Directions", "", "BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = BLOCKED", "", "```", err, "```"])
        print("BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = BLOCKED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
