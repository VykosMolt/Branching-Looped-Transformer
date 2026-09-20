#!/usr/bin/env python3
"""Diagnostics for the trained BG sequence-level adapter."""
from __future__ import annotations

import gc
import math
import time
import traceback
from typing import Any

import torch
import torch.nn.functional as F

from bg_sequence_adapter_common import (
    OUT_ROOT,
    avg,
    encode_prompt,
    generation_prompt,
    load_empirical_direction,
    load_json,
    load_model_tokenizer,
    load_sequence_adapter,
    load_sequence_dataset,
    load_teacher_adapter,
    rel,
    rows_for_split,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_sequence_adapter import sequence_adapter_hook


OUT_JSON = OUT_ROOT / "diagnostics.json"
OUT_MD = OUT_ROOT / "diagnostics.md"


def option_metrics(logits: torch.Tensor, tokenizer: Any, task: dict[str, Any]) -> dict[str, Any]:
    letters = sorted(task["options"])
    ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in letters]
    values = logits[0, -1, ids].float()
    correct_letter = str(task["correct_option"])
    ci = letters.index(correct_letter)
    wrong = [i for i, letter in enumerate(letters) if letter != correct_letter]
    margin = values[ci] - torch.logsumexp(values[wrong], dim=0)
    pred = letters[int(torch.argmax(values).detach().cpu().item())]
    ce = F.cross_entropy(values.unsqueeze(0), torch.tensor([ci], device=values.device))
    return {"margin": float(margin.detach().cpu().item()), "ce": float(ce.detach().cpu().item()), "accuracy": pred == correct_letter, "predicted": pred}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().flatten().float().cpu()
    bb = b.detach().flatten().float().cpu()
    aa = aa / torch.linalg.vector_norm(aa).clamp(min=1e-8)
    bb = bb / torch.linalg.vector_norm(bb).clamp(min=1e-8)
    return float(torch.dot(aa, bb).item())


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    teacher_rows: list[dict[str, Any]] = []
    bg_rows: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {}
    try:
        dataset = load_sequence_dataset()
        heldout = rows_for_split(dataset, "heldout")
        training = load_json(OUT_ROOT / "sequence_training_log.json", {})
        alpha = float(training.get("best_val_alpha") or 0.01)
        model, tokenizer, device = load_model_tokenizer()
        adapter = load_sequence_adapter(device)
        if adapter is None:
            raise RuntimeError("sequence adapter checkpoint missing")
        for task in heldout:
            prompt = generation_prompt(task)
            enc = encode_prompt(tokenizer, prompt, device)
            position = max(0, int(enc["input_ids"].shape[1]) - 1)
            with torch.no_grad():
                base_logits = model(**enc, use_cache=False, return_dict=True).logits
                base_m = option_metrics(base_logits, tokenizer, task)
                with sequence_adapter_hook(model, adapter, alpha=alpha, intervention_mode="multi_loop_decayed", position=position):
                    adapter_logits = model(**enc, use_cache=False, return_dict=True).logits
                adapter_m = option_metrics(adapter_logits, tokenizer, task)
            teacher_rows.append(
                {
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "alpha": alpha,
                    "baseline_margin": base_m["margin"],
                    "adapter_margin": adapter_m["margin"],
                    "margin_lift": adapter_m["margin"] - base_m["margin"],
                    "baseline_ce": base_m["ce"],
                    "adapter_ce": adapter_m["ce"],
                    "baseline_accuracy": base_m["accuracy"],
                    "adapter_accuracy": adapter_m["accuracy"],
                }
            )
        raw = load_empirical_direction("RAW_NONORM_READOUT")
        teacher_adapter = load_teacher_adapter(device)
        probe = torch.randn(64, 2048, device=device)
        with torch.no_grad():
            seq_delta = adapter(probe).detach().float().mean(dim=0).cpu()
            geometry["sequence_average_delta_rms"] = float(seq_delta.pow(2).mean().sqrt().item())
            if raw is not None:
                geometry["cosine_to_raw_nonorm"] = cosine(seq_delta, raw)
            if teacher_adapter is not None:
                teacher_delta = teacher_adapter(probe).detach().float().mean(dim=0).cpu()
                geometry["cosine_to_teacher_forced_adapter_proxy"] = cosine(seq_delta, teacher_delta)
        del model, tokenizer, adapter, teacher_adapter, probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        heldout_eval = load_json(OUT_ROOT / "heldout_free_generation_eval.json", {})
        eval_rows = heldout_eval.get("rows", [])
        by_task: dict[str, dict[str, Any]] = {}
        for row in eval_rows:
            if row.get("decode") != "deterministic":
                continue
            by_task.setdefault(str(row["task_id"]), {})[str(row["method"])] = row
        if by_task:
            from src.evaluator.bg_controller import BGController
            from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

            controller = BGController.from_artifacts(device="cpu")
            extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
            tasks_by_id = {str(task["task_id"]): task for task in heldout}
            for idx, (task_id, methods) in enumerate(sorted(by_task.items())[:4]):
                base = methods.get("no_intervention_baseline")
                trained = methods.get("trained_sequence_adapter")
                task = tasks_by_id.get(task_id)
                if not base or not trained or task is None:
                    continue
                base_feat = extractor.encode_prompt_candidate(task["prompt"], base.get("generated_output", ""), domain_hint=task["domain"])
                trained_feat = extractor.encode_prompt_candidate(task["prompt"], trained.get("generated_output", ""), domain_hint=task["domain"])
                score = controller.score_pair(trained_feat, base_feat, domain_hint=task["domain"], mode="conservative", return_details=True)
                bg_rows.append({"task_id": task_id, "domain": task["domain"], "adapter_vs_baseline_bg_score": float(score["score"]), "selected_head": score.get("selected_head")})
            extractor.cleanup()
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT": "INSUFFICIENT",
            "BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT": "INSUFFICIENT",
            "BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT": "INCONCLUSIVE",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "teacher_forced_rows": teacher_rows,
            "bg_score_rows": bg_rows,
            "geometry": geometry,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Sequence Adapter Diagnostics", "", "BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT = INSUFFICIENT", "BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT = INSUFFICIENT", "BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT = INCONCLUSIVE", "", f"- error: `{type(exc).__name__}: {str(exc)[:500]}`"])
        print("BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT = INSUFFICIENT")
        return 1

    margin_lift = avg(row["margin_lift"] for row in teacher_rows)
    if margin_lift is None:
        tf_verdict = "INSUFFICIENT"
    elif margin_lift > 0.01:
        tf_verdict = "LOGIT_MARGIN_IMPROVES"
    else:
        tf_verdict = "NO_LOGIT_MARGIN_EFFECT"
    bg_shift = avg(abs(row["adapter_vs_baseline_bg_score"]) for row in bg_rows)
    if bg_shift is None:
        bg_verdict = "INSUFFICIENT"
    elif bg_shift > 1e-4:
        bg_verdict = "BG_SCORE_MOVES"
    else:
        bg_verdict = "BG_SCORE_DOES_NOT_MOVE"
    cos_vals = [abs(float(v)) for k, v in geometry.items() if k.startswith("cosine_to_") and v is not None and math.isfinite(float(v))]
    if not cos_vals:
        geometry_verdict = "INCONCLUSIVE"
    elif max(cos_vals) < 0.20:
        geometry_verdict = "NEW_WRITE_GEOMETRY"
    else:
        geometry_verdict = "MATCHES_PRIOR_DIRECTIONS"
    payload = {
        "BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT": tf_verdict,
        "BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT": bg_verdict,
        "BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT": geometry_verdict,
        "teacher_forced_margin_lift_mean": margin_lift,
        "teacher_forced_rows": teacher_rows,
        "bg_score_shift_abs_mean": bg_shift,
        "bg_score_rows": bg_rows,
        "geometry": geometry,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Diagnostics",
        "",
        f"BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT = {tf_verdict}",
        f"BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT = {bg_verdict}",
        f"BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT = {geometry_verdict}",
        "",
        f"- teacher-forced margin lift mean: `{margin_lift}`",
        f"- BG score abs shift mean: `{bg_shift}`",
        f"- geometry: `{geometry}`",
    ]
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT = {tf_verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
