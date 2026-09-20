"""Preflight inventory for BG Stage 2 steering sensitivity v3."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.evaluator.bg_controller import BGController, HEAD_CLASSES, config_dim  # noqa: E402
from src.evaluator.bg_steering_hook import (  # noqa: E402
    BGLatentLoopBoundaryFork,
    BGLayerHookSteering,
    build_intervention_mode,
)
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor  # noqa: E402


REPORT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
REGISTRY_PT = PROJECT_ROOT / "opi/taps/probes/bg_head_registry_2026-05-17.pt"
MIXED_PT = PROJECT_ROOT / "opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.pt"
OUT_JSON = REPORT_ROOT / "preflight.json"
OUT_MD = REPORT_ROOT / "preflight.md"
SEED = 20260518

TARGET_REQUESTS = [
    {"target_id": "T1", "domain": "reasoning", "prefix_length": 64},
    {"target_id": "T2", "domain": "reasoning", "prefix_length": 256},
    {"target_id": "T3", "domain": "science", "prefix_length": 32},
    {"target_id": "T4", "domain": "gsm8k", "prefix_length": 256},
]

REQUIRED_STAGE1 = [
    "predictive_power.json",
    "predictive_power.md",
    "partials.json",
    "continued_prefixes.json",
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def strength(row: dict[str, Any]) -> float:
    pair = row.get("pairwise_predictive_accuracy")
    pair_component = float(pair) - 0.5 if pair is not None else -1.0
    return max(float(row.get("top1_lift", 0.0)), float(row.get("top2_lift", 0.0)), pair_component)


def layer_from_config(config: str) -> int:
    return int(str(config).split("_", 1)[0])


def select_nonorm_target(cells: list[dict[str, Any]], domain: str, prefix_length: int) -> dict[str, Any] | None:
    rows = [
        row
        for row in cells
        if str(row.get("domain")) == domain
        and int(row.get("prefix_length", -1)) == int(prefix_length)
        and str(row.get("architecture")) == "AntisymLinearNoNorm"
        and "concat" not in str(row.get("config", ""))
        and config_dim(str(row.get("config"))) == 2048
    ]
    rows.sort(key=lambda row: (-strength(row), -float(row.get("oracle_success", 0.0)), str(row.get("head_id"))))
    return dict(rows[0]) if rows else None


def select_latent_direction(cells: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in cells
        if str(row.get("domain")) == str(target["domain"])
        and int(row.get("prefix_length", -1)) == int(target["prefix_length"])
        and str(row.get("architecture")) == "AntisymLinearNoNorm"
        and str(row.get("config")) in {"47_L4", "47_mean"}
        and config_dim(str(row.get("config"))) == 2048
    ]
    rows.sort(key=lambda row: (-strength(row), -float(row.get("oracle_success", 0.0)), str(row.get("head_id"))))
    if rows:
        selected = dict(rows[0])
        selected["latent_direction_layer_mismatch"] = False
        return selected
    selected = dict(target)
    selected["latent_direction_layer_mismatch"] = layer_from_config(str(target["config"])) != 47
    return selected


def find_diagnostic_d1(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in cells:
        if (
            str(row.get("domain")) == "reasoning"
            and int(row.get("prefix_length", -1)) == 256
            and str(row.get("family")) == "MIX_CODE_REASONING"
            and str(row.get("config")) == "36_mean"
            and str(row.get("architecture")) == "AntisymLinear"
        ):
            return dict(row)
    return None


def parse_head_id(head_id: str) -> dict[str, str]:
    parts = str(head_id).split("::")
    if parts[0] == "locked":
        return {"source": "locked", "locked_name": parts[1]}
    if len(parts) != 4:
        raise RuntimeError(f"cannot parse head_id: {head_id}")
    return {"source": parts[0], "family": parts[1], "config": parts[2], "architecture": parts[3]}


def load_head_row(head_id: str, controller: BGController | None = None) -> dict[str, Any]:
    parsed = parse_head_id(head_id)
    if parsed["source"] == "locked":
        if controller is None:
            controller = BGController.from_artifacts(device="cpu")
        name = parsed["locked_name"]
        spec = controller.specs[name]
        return {
            "head_id": head_id,
            "source": "locked",
            "family": spec.family,
            "config": spec.config,
            "architecture": spec.architecture,
            "dim": spec.dim,
            "state_dict": controller.heads[name].state_dict(),
            "artifact": spec.artifact_path,
        }
    path = REGISTRY_PT if parsed["source"] == "registry" else MIXED_PT
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for idx, row in enumerate(payload.get("heads") or []):
        family = str(row.get("head_group") or row.get("head_family") or row.get("family") or "")
        if (
            family == parsed["family"]
            and str(row.get("config")) == parsed["config"]
            and str(row.get("architecture")) == parsed["architecture"]
        ):
            return {
                "head_id": head_id,
                "source": parsed["source"],
                "family": family,
                "config": str(row.get("config")),
                "architecture": str(row.get("architecture")),
                "dim": int(row.get("dim", config_dim(str(row.get("config"))))),
                "state_dict": row["state_dict"],
                "artifact": rel(path),
                "row_index": idx,
            }
    raise RuntimeError(f"head not found in {rel(path)}: {head_id}")


def head_weight_vector(head_row: dict[str, Any]) -> torch.Tensor:
    architecture = str(head_row["architecture"])
    dim = int(head_row["dim"])
    head = HEAD_CLASSES[architecture](dim)
    head.load_state_dict(head_row["state_dict"])
    weight = head.linear.weight.detach().flatten().to(dtype=torch.float32)
    if int(weight.numel()) != dim:
        raise RuntimeError(f"head weight shape mismatch for {head_row['head_id']}: {tuple(weight.shape)} vs dim {dim}")
    norm = torch.linalg.vector_norm(weight).clamp(min=1e-12)
    return weight / norm


def deterministic_generate_ids(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, hook: BGLayerHookSteering | None = None) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    if hook is not None:
        hook.apply(position=-1)
    try:
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        if hook is not None:
            hook.remove()
    return out.detach().cpu()


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {
        "BG_STAGE2_PREFLIGHT_VERDICT": "BLOCKED",
        "LAYER_HOOK_INJECTION_VERDICT": "BLOCKED",
        "LATENT_LOOP_BOUNDARY_FORK_VERDICT": "SKIPPED",
        "errors": [],
        "warnings": [],
    }
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        artifacts = []
        for name in REQUIRED_STAGE1:
            path = STAGE1_ROOT / name
            artifacts.append({"path": rel(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None})
        missing = [row["path"] for row in artifacts if not row["exists"]]
        predictive = load_json(STAGE1_ROOT / "predictive_power.json", {})
        cells = list(predictive.get("all_cells") or [])
        if missing or not cells:
            payload.update(
                {
                    "BG_STAGE2_PREFLIGHT_VERDICT": "BLOCKED",
                    "blocker": f"missing Stage 1 artifacts or predictive cells: {missing}",
                    "stage1_artifacts": artifacts,
                }
            )
            write_json(OUT_JSON, payload)
            write_md(OUT_MD, ["# BG Stage 2 Steering Preflight", "", "BG_STAGE2_PREFLIGHT_VERDICT = BLOCKED", "", f"- blocker: `{payload['blocker']}`"])
            print("BG_STAGE2_PREFLIGHT_VERDICT = BLOCKED")
            return 1

        controller = BGController.from_artifacts(device="cpu")
        controller_heads = sorted(controller.heads)
        targets = []
        for request in TARGET_REQUESTS:
            selected = select_nonorm_target(cells, request["domain"], int(request["prefix_length"]))
            if selected is None:
                targets.append({**request, "status": "missing_nonorm_cell"})
                continue
            head_row = load_head_row(str(selected["head_id"]), controller)
            direction = head_weight_vector(head_row)
            latent = select_latent_direction(cells, selected)
            latent_head_row = load_head_row(str(latent["head_id"]), controller)
            latent_direction = head_weight_vector(latent_head_row)
            targets.append(
                {
                    **request,
                    "status": "ready",
                    "head_id": selected["head_id"],
                    "family": selected.get("family"),
                    "config": selected.get("config"),
                    "architecture": selected.get("architecture"),
                    "layer": layer_from_config(str(selected.get("config"))),
                    "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                    "top1_lift": selected.get("top1_lift"),
                    "pairwise_accuracy": selected.get("pairwise_predictive_accuracy"),
                    "oracle_success": selected.get("oracle_success"),
                    "latent_direction_head_id": latent["head_id"],
                    "latent_direction_config": latent["config"],
                    "latent_direction_norm": float(torch.linalg.vector_norm(latent_direction).item()),
                    "latent_direction_layer_mismatch": bool(latent.get("latent_direction_layer_mismatch", False)),
                }
            )

        d1 = find_diagnostic_d1(cells)
        d1_status = "missing"
        if d1 is not None:
            d1_head_row = load_head_row(str(d1["head_id"]), controller)
            d1_direction = head_weight_vector(d1_head_row)
            d1_status = "ready" if int(d1_direction.numel()) == 2048 else "bad_shape"

        layer_hook_smoke: dict[str, Any] = {"status": "not_run"}
        latent_smoke: dict[str, Any] = {"status": "not_run"}
        ready_targets = [row for row in targets if row.get("status") == "ready"]
        if ready_targets:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            extractor = BGTransformerFeatureExtractor(device=device, dtype="auto", force_all_loops=True)
            model = extractor.model
            tokenizer = extractor.tokenizer
            smoke_target = ready_targets[0]
            smoke_row = load_head_row(str(smoke_target["head_id"]), controller)
            smoke_direction = head_weight_vector(smoke_row)
            spec = build_intervention_mode("single_loop_L1", 0.0)
            hook = BGLayerHookSteering(
                model,
                target_layer=int(smoke_target["layer"]),
                target_loops=spec["target_loops"],
                loop_alpha_scales=spec["loop_alpha_scales"],
                direction=smoke_direction,
                alpha=0.0,
            )
            prompt = "Answer with exactly one token if possible. What is 2 + 2?"
            no_hook_ids = deterministic_generate_ids(model, tokenizer, prompt, max_new_tokens=8, hook=None)
            hook_ids = deterministic_generate_ids(model, tokenizer, prompt, max_new_tokens=8, hook=hook)
            layer_hook_equal = bool(torch.equal(no_hook_ids, hook_ids))
            layer_hook_smoke = {
                "status": "ready" if layer_hook_equal else "blocked",
                "zero_alpha_equivalence": layer_hook_equal,
                "loop_index_source": hook.loop_index_source,
                "hook_forward_calls": hook.forward_call_count,
                "hook_modifications": hook.modifications,
                "no_hook_ids": no_hook_ids[0].tolist(),
                "hook_ids": hook_ids[0].tolist(),
            }
            payload["LAYER_HOOK_INJECTION_VERDICT"] = "READY" if layer_hook_equal else "BLOCKED"

            latent_target = ready_targets[0]
            latent_row = load_head_row(str(latent_target["latent_direction_head_id"]), controller)
            latent_direction = head_weight_vector(latent_row)
            fork = BGLatentLoopBoundaryFork(
                model,
                direction=latent_direction,
                alpha=0.0,
                intervention_mode="single_loop_L1",
                use_cache=False,
            )
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=False)
            enc = {key: value.to(extractor.device) for key, value in enc.items()}
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=extractor.device)
            try:
                fork_result = fork.run(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position=-1)
                logit_delta = float(
                    (fork_result["clean_logits"].float() - fork_result["full_logits"].float())
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                    .item()
                )
                latent_smoke = {
                    "status": "blocked_for_generation_continuation",
                    "zero_alpha_loop_forward_equivalence": logit_delta < 1e-4
                    and float(fork_result["zero_alpha_full_clean_max_abs_delta"]) < 1e-4,
                    "zero_alpha_logit_max_abs_delta": logit_delta,
                    "zero_alpha_hidden_max_abs_delta": fork_result["zero_alpha_full_clean_max_abs_delta"],
                    "total_ut_steps_restored": fork_result["total_ut_steps_restored"],
                    "reason": "loop-boundary forward equivalence can be tested, but HF generation cannot cleanly resume from post-loop hidden boundary without cache/state forking",
                }
            except Exception as exc:
                latent_smoke = {
                    "status": "blocked",
                    "reason": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                }
            payload["LATENT_LOOP_BOUNDARY_FORK_VERDICT"] = "BLOCKED"
        else:
            payload["LAYER_HOOK_INJECTION_VERDICT"] = "BLOCKED"
            payload["LATENT_LOOP_BOUNDARY_FORK_VERDICT"] = "SKIPPED"

        target_ready_count = sum(1 for row in targets if row.get("status") == "ready")
        if payload["LAYER_HOOK_INJECTION_VERDICT"] != "READY":
            verdict = "BLOCKED"
        elif target_ready_count >= 4 and payload["LATENT_LOOP_BOUNDARY_FORK_VERDICT"] == "READY":
            verdict = "READY"
        elif target_ready_count >= 3:
            verdict = "PARTIAL"
        else:
            verdict = "BLOCKED"

        payload.update(
            {
                "BG_STAGE2_PREFLIGHT_VERDICT": verdict,
                "controller_load": "OK",
                "transformer_feature_extractor_load": "OK",
                "controller_heads_loaded": controller_heads,
                "model_path": rel(MODEL_PATH),
                "model_path_exists": MODEL_PATH.exists(),
                "stage1_artifacts": artifacts,
                "targets": targets,
                "diagnostic_d1": d1 or "MISSING",
                "diagnostic_d1_status": d1_status,
                "layer_hook_smoke": layer_hook_smoke,
                "latent_loop_boundary_smoke": latent_smoke,
                "alpha_cap": 0.02,
                "steering_direction_rule": "AntisymLinearNoNorm only; AntisymLinear D1 is diagnostic readout only",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "BG_STAGE2_PREFLIGHT_VERDICT": "BLOCKED",
                "LAYER_HOOK_INJECTION_VERDICT": "BLOCKED",
                "LATENT_LOOP_BOUNDARY_FORK_VERDICT": "SKIPPED",
                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    finally:
        if extractor is not None:
            extractor.cleanup()

    write_json(OUT_JSON, payload)
    lines = [
        "# BG Stage 2 Steering Preflight v3 (2026-05-18)",
        "",
        f"BG_STAGE2_PREFLIGHT_VERDICT = {payload['BG_STAGE2_PREFLIGHT_VERDICT']}",
        f"LAYER_HOOK_INJECTION_VERDICT = {payload['LAYER_HOOK_INJECTION_VERDICT']}",
        f"LATENT_LOOP_BOUNDARY_FORK_VERDICT = {payload['LATENT_LOOP_BOUNDARY_FORK_VERDICT']}",
        "",
        f"- model_path_exists: `{payload.get('model_path_exists')}`",
        f"- controller_heads_loaded: `{payload.get('controller_heads_loaded')}`",
        f"- diagnostic_d1_status: `{payload.get('diagnostic_d1_status')}`",
        f"- zero_alpha_hook_equivalence: `{payload.get('layer_hook_smoke', {}).get('zero_alpha_equivalence')}`",
        f"- latent_loop_forward_equivalence: `{payload.get('latent_loop_boundary_smoke', {}).get('zero_alpha_loop_forward_equivalence')}`",
        f"- latent_reason: `{payload.get('latent_loop_boundary_smoke', {}).get('reason', '')}`",
        "",
        "## Targets",
        "",
        "| Target | Domain | Prefix | Head | Config | Layer | Direction norm | Latent direction | Latent mismatch |",
        "|---|---|---:|---|---|---:|---:|---|---|",
    ]
    for row in payload.get("targets", []):
        lines.append(
            "| {target_id} | {domain} | {prefix_length} | `{head}` | `{config}` | {layer} | {norm:.6f} | `{latent}` | {mismatch} |".format(
                target_id=row.get("target_id"),
                domain=row.get("domain"),
                prefix_length=row.get("prefix_length"),
                head=row.get("head_id", row.get("status")),
                config=row.get("config", ""),
                layer=row.get("layer", ""),
                norm=float(row.get("direction_norm", 0.0)),
                latent=row.get("latent_direction_head_id", ""),
                mismatch=row.get("latent_direction_layer_mismatch", ""),
            )
        )
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", payload["error"], "```"])
    write_md(OUT_MD, lines)
    print(f"BG_STAGE2_PREFLIGHT_VERDICT = {payload['BG_STAGE2_PREFLIGHT_VERDICT']}")
    print(f"LAYER_HOOK_INJECTION_VERDICT = {payload['LAYER_HOOK_INJECTION_VERDICT']}")
    print(f"LATENT_LOOP_BOUNDARY_FORK_VERDICT = {payload['LATENT_LOOP_BOUNDARY_FORK_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if payload["BG_STAGE2_PREFLIGHT_VERDICT"] in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
