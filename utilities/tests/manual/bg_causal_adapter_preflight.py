#!/usr/bin/env python3
"""Preflight for tiny causal BG intervention adapter training."""
from __future__ import annotations

import gc
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from bg_causal_adapter_common import (
    MODEL_PATH,
    OUT_ROOT,
    PROJECT_ROOT,
    STAGE1_ROOT,
    all_frozen,
    build_dataset_rows,
    load_model_tokenizer,
    option_token_ids,
    rel,
    write_json,
    write_md,
)
from src.evaluator.bg_controller import BGController
from src.evaluator.bg_steering_hook import BGLayerHookSteering
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor


OUT_JSON = OUT_ROOT / "preflight.json"
OUT_MD = OUT_ROOT / "preflight.md"


def zero_alpha_equivalence() -> dict[str, Any]:
    extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
    model = extractor.model
    tokenizer = extractor.tokenizer
    task = next(row for row in build_dataset_rows()[0] if row["split"] == "heldout")
    prompt = f"{task['prompt']}\n\nPartial answer attempt:\n{task['prefix_text']}\nFINAL ANSWER:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False)
    enc = {key: value.to(extractor.device) for key, value in enc.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(123)
    with torch.inference_mode():
        clean = model.generate(
            **enc,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    direction = torch.zeros(2048, dtype=torch.float32)
    direction[0] = 1.0
    hook = BGLayerHookSteering(
        model,
        target_layer=36,
        target_loops=[1, 2, 3, 4],
        direction=direction,
        alpha=0.0,
        loop_alpha_scales={1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0},
        direction_normalization="l2",
    )
    torch.manual_seed(123)
    hook.apply(position=-1)
    try:
        with torch.inference_mode():
            hooked = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        hook.remove()
    same = bool(torch.equal(clean, hooked))
    result = {
        "zero_alpha_equivalence_passed": same,
        "clean_new_ids": clean[0, enc["input_ids"].shape[1] :].detach().cpu().tolist(),
        "hooked_new_ids": hooked[0, enc["input_ids"].shape[1] :].detach().cpu().tolist(),
        "hook_forward_call_count": hook.diagnostics()["forward_call_count"],
        "hook_modifications": hook.diagnostics()["modifications"],
        "hook_loop_index_source": hook.diagnostics()["loop_index_source"],
    }
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    artifacts = {
        "task_suite": STAGE1_ROOT / "task_suite.json",
        "continued_prefixes": STAGE1_ROOT / "continued_prefixes.json",
        "prefix_features": STAGE1_ROOT / "prefix_features.pt",
        "predictive_power": STAGE1_ROOT / "predictive_power.json",
    }
    artifact_status = {key: path.exists() for key, path in artifacts.items()}
    rows, split, tokenization = build_dataset_rows()
    clean_tasks = sorted({row["task_id"] for row in rows})
    split_counts = {name: len(ids) for name, ids in split.items()}
    example_counts = Counter(row["split"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    clean_mq = True
    token_details = {}
    model_load_ok = False
    all_params_frozen = False
    zero_alpha = {"zero_alpha_equivalence_passed": False, "error": "not_run"}

    try:
        model, tokenizer, _device = load_model_tokenizer()
        model_load_ok = True
        all_params_frozen = all_frozen(model)
        token_details = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in ["A", "B", "C", "D", "E"]}
        option_token_ids(tokenizer)
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as exc:
        token_details = {"error": repr(exc)}
        clean_mq = False

    controller_ok = False
    extractor_import_ok = False
    hook_import_ok = True
    try:
        BGController.from_artifacts(device="cpu")
        controller_ok = True
    except Exception:
        controller_ok = False
    try:
        from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor as _Extractor

        extractor_import_ok = _Extractor is not None
    except Exception:
        extractor_import_ok = False

    if model_load_ok and all_params_frozen and clean_mq and len(rows) >= 1:
        try:
            zero_alpha = zero_alpha_equivalence()
        except Exception as exc:
            zero_alpha = {"zero_alpha_equivalence_passed": False, "error": repr(exc)}

    train_tasks = split_counts.get("train", 0)
    heldout_tasks = split_counts.get("heldout", 0)
    if (
        MODEL_PATH.exists()
        and model_load_ok
        and all_params_frozen
        and controller_ok
        and extractor_import_ok
        and hook_import_ok
        and all(artifact_status.values())
        and clean_mq
        and zero_alpha.get("zero_alpha_equivalence_passed")
        and train_tasks >= 20
        and heldout_tasks >= 8
    ):
        verdict = "READY"
    elif train_tasks >= 10 and heldout_tasks >= 4 and model_load_ok and clean_mq:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    payload = {
        "BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT": verdict,
        "local_model_path": rel(MODEL_PATH),
        "model_path_exists": MODEL_PATH.exists(),
        "model_load_ok": model_load_ok,
        "all_ouro_params_frozen": all_params_frozen,
        "BGController_loads": controller_ok,
        "BGTransformerFeatureExtractor_imports": extractor_import_ok,
        "bg_steering_hook_imports": hook_import_ok,
        "zero_alpha_equivalence": zero_alpha,
        "use_cache_for_intervention_runs": False,
        "stage1_artifacts": {key: rel(path) for key, path in artifacts.items()},
        "stage1_artifact_status": artifact_status,
        "mcq_clean_token_targets": clean_mq,
        "option_tokenization": token_details,
        "task_split_counts": split_counts,
        "example_counts": dict(example_counts),
        "domain_example_counts": dict(domain_counts),
        "clean_mq_task_count": len(clean_tasks),
        "split_task_ids": split,
        "tokenization": tokenization,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Causal Adapter Preflight",
        "",
        f"BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = {verdict}",
        "",
        f"- model_load_ok: `{model_load_ok}`",
        f"- all_ouro_params_frozen: `{all_params_frozen}`",
        f"- BGController_loads: `{controller_ok}`",
        f"- zero_alpha_equivalence_passed: `{zero_alpha.get('zero_alpha_equivalence_passed')}`",
        f"- train_tasks: `{train_tasks}`",
        f"- heldout_tasks: `{heldout_tasks}`",
        f"- example_counts: `{dict(example_counts)}`",
        f"- option_tokenization: `{token_details}`",
    ]
    write_md(OUT_MD, lines)
    print(f"BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
