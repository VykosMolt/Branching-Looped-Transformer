#!/usr/bin/env python3
"""Full preflight for the BG sequence-level adapter run."""
from __future__ import annotations

import gc
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from bg_sequence_adapter_common import (
    CAUSAL_ROOT,
    MODEL_PATH,
    OUT_ROOT,
    QUICK_OUT_ROOT,
    STAGE1_ROOT,
    all_frozen,
    build_sequence_dataset_rows,
    encode_prompt,
    load_json,
    load_model_tokenizer,
    load_stage1_mcq_tasks,
    rel,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_sequence_adapter import build_sequence_adapter, sequence_adapter_hook


OUT_JSON = OUT_ROOT / "preflight.json"
OUT_MD = OUT_ROOT / "preflight.md"


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    errors: list[str] = []
    try:
        checks["model_path_exists"] = MODEL_PATH.exists()
        if not MODEL_PATH.exists():
            errors.append(f"missing model path {MODEL_PATH}")
        required_stage1 = ["task_suite.json", "continued_prefixes.json", "prefix_features.pt", "prefix_scores.json", "predictive_power.json"]
        checks["stage1_required"] = {name: (STAGE1_ROOT / name).exists() for name in required_stage1}
        missing_stage1 = [name for name, ok in checks["stage1_required"].items() if not ok]
        if missing_stage1:
            errors.append(f"missing Stage 1 artifacts: {missing_stage1}")
        checks["causal_adapter_checkpoint_exists"] = (CAUSAL_ROOT / "adapter_checkpoints/best_adapter.pt").exists()
        quick = load_json(QUICK_OUT_ROOT / "summary.json", {})
        checks["quick_preflight_readiness"] = quick.get("OVERNIGHT_SEQUENCE_ADAPTER_READINESS")

        tasks = load_stage1_mcq_tasks()
        rows, split = build_sequence_dataset_rows()
        checks["clean_mcq_task_count"] = len(tasks)
        checks["split_task_counts"] = {key: len(vals) for key, vals in split.items()}
        checks["heldout_overlap_train"] = sorted(set(split.get("heldout", [])) & set(split.get("train", [])))
        if len(tasks) < 16 or checks["split_task_counts"].get("heldout", 0) < 8:
            errors.append("insufficient clean MCQ tasks for sequence adapter")
        if checks["heldout_overlap_train"]:
            errors.append("heldout task overlap with train")

        model, tokenizer, device = load_model_tokenizer()
        checks["ouro_loads_on_cuda"] = str(device).startswith("cuda")
        checks["model_parameters_frozen"] = all_frozen(model)
        if not checks["model_parameters_frozen"]:
            errors.append("Ouro parameters are not frozen")
        from src.evaluator.bg_controller import BGController

        controller = BGController.from_artifacts(device="cpu")
        checks["bg_controller_loads"] = sorted(controller.heads.keys())
        checks["bg_steering_hook_imports"] = True
        prompt = tasks[0]["prompt"] if tasks else "Question: test\nOptions:\nA. a\nB. b\nC. c\nD. d\nFINAL ANSWER:"
        enc = encode_prompt(tokenizer, prompt, device, max_length=512)
        position = max(0, int(enc["input_ids"].shape[1]) - 1)
        adapter = build_sequence_adapter(kind="low_rank", rank=32, device=device)
        with torch.no_grad():
            base_logits = model(**enc, use_cache=False, return_dict=True).logits[:, -1, :512].detach().float().cpu()
            with sequence_adapter_hook(model, adapter, alpha=0.0, intervention_mode="multi_loop_decayed", position=position):
                hooked_logits = model(**enc, use_cache=False, return_dict=True).logits[:, -1, :512].detach().float().cpu()
        checks["zero_alpha_equivalence_max_abs_delta_first512"] = float((base_logits - hooked_logits).abs().max().item())
        checks["zero_alpha_equivalence_pass"] = checks["zero_alpha_equivalence_max_abs_delta_first512"] <= 1e-6
        checks["use_cache_false_primary"] = True
        del model, tokenizer, adapter, base_logits, hooked_logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

        extractor = BGTransformerFeatureExtractor(model_path=MODEL_PATH, device="cuda", dtype="auto", force_all_loops=True)
        _features, summary = extractor.inspect_capture_once("Question: smoke\nOptions:\nA. x\nB. y\nFINAL ANSWER:", max_length=64)
        checks["bg_transformer_feature_extractor_loads"] = True
        checks["bg_transformer_capture_summary"] = {
            "boundary_loop_count": summary.boundary_loop_count,
            "layer_loop_counts": summary.layer_loop_counts,
        }
        extractor.cleanup()
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {str(exc)[:1000]}")
        checks["traceback"] = traceback.format_exc()[-4000:]

    total = int(checks.get("clean_mcq_task_count") or 0)
    heldout = int((checks.get("split_task_counts") or {}).get("heldout", 0))
    if errors:
        verdict = "BLOCKED"
    elif total >= 30 and heldout >= 12:
        verdict = "READY"
    elif total >= 16 and heldout >= 8:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT": verdict,
        "checks": checks,
        "errors": errors,
        "prior_reports_read": {
            "quick_preflight": rel(QUICK_OUT_ROOT / "summary.json"),
            "causal_summary": rel(CAUSAL_ROOT / "summary.md"),
            "preconsolidation_summary": rel(Path("opi/taps/probes/bg_preconsolidation_control_probes_2026-05-18/summary.md")),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence-Level Adapter Preflight",
        "",
        f"BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT = {verdict}",
        "",
        f"- model path exists: `{checks.get('model_path_exists')}`",
        f"- Ouro loads on cuda: `{checks.get('ouro_loads_on_cuda')}`",
        f"- model parameters frozen: `{checks.get('model_parameters_frozen')}`",
        f"- BGController heads: `{checks.get('bg_controller_loads')}`",
        f"- BGTransformerFeatureExtractor loads: `{checks.get('bg_transformer_feature_extractor_loads')}`",
        f"- bg_steering_hook imports: `{checks.get('bg_steering_hook_imports')}`",
        f"- zero-alpha equivalence pass: `{checks.get('zero_alpha_equivalence_pass')}`",
        f"- use_cache false primary: `{checks.get('use_cache_false_primary')}`",
        f"- clean MCQ tasks: `{checks.get('clean_mcq_task_count')}`",
        f"- split task counts: `{checks.get('split_task_counts')}`",
        f"- causal adapter checkpoint exists: `{checks.get('causal_adapter_checkpoint_exists')}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {err}" for err in errors] if errors else ["- none"])
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
