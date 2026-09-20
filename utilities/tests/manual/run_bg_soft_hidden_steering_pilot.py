"""Guarded soft hidden-state steering stability pilot for BG."""
from __future__ import annotations

import math
import re
import time
import traceback

import torch

from bg_steering_suite_lib import (
    REPORT_ROOT,
    SEED,
    OuroTextGenerator,
    continuation_budget,
    evaluate_output,
    load_task_suite,
    domains_allowed_by_reachability,
    rel,
    task_generation_prompt,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "soft_steering_results.json"
OUT_MD = REPORT_ROOT / "soft_steering_results.md"


def _degeneracy(text: str) -> dict:
    tokens = str(text or "").split()
    repeated = 0
    for a, b in zip(tokens, tokens[1:]):
        if a == b:
            repeated += 1
    return {
        "empty": not bool(str(text or "").strip()),
        "token_count_text": len(tokens),
        "adjacent_repeat_rate": repeated / max(len(tokens) - 1, 1),
        "severe_repetition": repeated / max(len(tokens) - 1, 1) > 0.20 if tokens else False,
    }


class _SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float, target_loop: int = 4) -> None:
        self.direction = direction
        self.alpha = float(alpha)
        self.target_loop = int(target_loop)
        self.calls = 0
        self.interventions = 0
        self.rms_before: list[float] = []
        self.rms_residual: list[float] = []
        self.nan_or_inf = False

    def __call__(self, _module, _inp, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        self.calls += 1
        loop = ((self.calls - 1) % 4) + 1
        if self.alpha == 0.0 or loop != self.target_loop:
            return output
        h = tensor
        if not torch.isfinite(h).all():
            self.nan_or_inf = True
            return output
        direction = self.direction.to(device=h.device, dtype=h.dtype)
        rms = h.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        residual = self.alpha * direction.view(1, 1, -1) * rms.to(dtype=h.dtype)
        residual_rms = residual.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        max_rms = self.alpha * rms
        scale = torch.clamp(max_rms / residual_rms.clamp(min=1e-8), max=1.0).to(dtype=h.dtype)
        residual = residual * scale
        new_h = h + residual
        if not torch.isfinite(new_h).all():
            self.nan_or_inf = True
            return output
        self.interventions += 1
        if len(self.rms_before) < 64:
            self.rms_before.append(float(rms.mean().detach().cpu()))
            self.rms_residual.append(float(residual.float().pow(2).mean(dim=-1).sqrt().mean().detach().cpu()))
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        if isinstance(output, list):
            return [new_h] + list(output[1:])
        return new_h


def _generate(generator: OuroTextGenerator, prompt: str, max_new_tokens: int, seed: int, hook: _SteeringHook | None = None) -> tuple[dict, dict]:
    handle = None
    if hook is not None:
        layer = generator.model.model.layers[35]
        handle = layer.register_forward_hook(hook)
    try:
        gen = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.95, seed=seed)
    finally:
        if handle is not None:
            handle.remove()
    diagnostics = {}
    if hook is not None:
        diagnostics = {
            "hook_calls": hook.calls,
            "interventions": hook.interventions,
            "nan_or_inf": hook.nan_or_inf,
            "rms_before_mean": sum(hook.rms_before) / max(len(hook.rms_before), 1),
            "rms_residual_mean": sum(hook.rms_residual) / max(len(hook.rms_residual), 1),
        }
    return gen, diagnostics


def main() -> int:
    started = time.time()
    allowed = domains_allowed_by_reachability()
    if not allowed:
        payload = {
            "BG_SOFT_STEERING_VERDICT": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
            "skipped_reason": "reachability gate did not pass non-devil domains",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Soft Hidden Steering Pilot", "", "BG_SOFT_STEERING_VERDICT = INSUFFICIENT", "", "- skipped: reachability gate limited"])
        print("BG_SOFT_STEERING_VERDICT = INSUFFICIENT")
        return 0

    tasks = []
    suite = [t for t in load_task_suite() if t["domain"] in allowed and not t.get("is_devil")]
    for domain in ("code", "reasoning", "science", "gsm8k"):
        found = next((t for t in suite if t["domain"] == domain), None)
        if found:
            tasks.append(found)
    tasks = tasks[:4]
    from src.evaluator.bg_controller import BGController

    controller = BGController.from_artifacts(device="cpu")
    direction = controller.heads["objective_mixed"].linear.weight.detach().flatten().to(torch.float32)
    direction = direction / direction.pow(2).mean().sqrt().clamp(min=1e-8)
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    random_dir = torch.randn(direction.shape, generator=rng)
    random_dir = random_dir / random_dir.pow(2).mean().sqrt().clamp(min=1e-8)
    directions = {
        "positive_objective": direction,
        "negative_objective": -direction,
        "random_control": random_dir,
    }
    alphas = [0.005, 0.01, 0.02]
    rows = []
    destabilizing = False
    generator: OuroTextGenerator | None = None
    try:
        generator = OuroTextGenerator(device="cuda")
        for task in tasks:
            prompt = task_generation_prompt(task)
            budget = 256 if task["domain"] == "code" else 128
            try:
                base_gen, base_diag = _generate(generator, prompt, budget, SEED + int(task.get("suite_index", 0)) * 409, None)
                base_eval = evaluate_output(task, base_gen["text"])
            except Exception as exc:
                rows.append({"task_id": task["task_id"], "domain": task["domain"], "condition": "baseline", "error": traceback.format_exc()[-4000:]})
                continue
            base_deg = _degeneracy(base_gen["text"])
            rows.append({"task_id": task["task_id"], "domain": task["domain"], "condition": "baseline", "alpha": 0.0, "generation": base_gen, "evaluation": base_eval, "stability": base_deg, "hook": base_diag})
            for alpha in alphas:
                if destabilizing:
                    break
                for name, vec in directions.items():
                    hook = _SteeringHook(vec, alpha)
                    try:
                        gen, hook_diag = _generate(generator, prompt, budget, SEED + int(task.get("suite_index", 0)) * 409 + int(alpha * 10000), hook)
                        ev = evaluate_output(task, gen["text"])
                        deg = _degeneracy(gen["text"])
                        severe = bool(deg["empty"] or deg["severe_repetition"] or hook_diag.get("nan_or_inf"))
                        if base_deg["adjacent_repeat_rate"] > 0:
                            severe = severe or deg["adjacent_repeat_rate"] > base_deg["adjacent_repeat_rate"] * 1.5 + 0.05
                        destabilizing = destabilizing or severe
                        rows.append({"task_id": task["task_id"], "domain": task["domain"], "condition": name, "alpha": alpha, "generation": gen, "evaluation": ev, "stability": deg, "hook": hook_diag})
                    except Exception as exc:
                        destabilizing = True
                        rows.append({"task_id": task["task_id"], "domain": task["domain"], "condition": name, "alpha": alpha, "error": traceback.format_exc()[-4000:]})
                        break
    except Exception as exc:
        payload = {"BG_SOFT_STEERING_VERDICT": "SKIPPED_INTEGRATION_RISK", "verdict": "SKIPPED_INTEGRATION_RISK", "error": traceback.format_exc()}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Soft Hidden Steering Pilot", "", "BG_SOFT_STEERING_VERDICT = SKIPPED_INTEGRATION_RISK", "", "```text", payload["error"], "```"])
        print("BG_SOFT_STEERING_VERDICT = SKIPPED_INTEGRATION_RISK")
        return 0
    finally:
        if generator is not None:
            generator.cleanup()

    eval_rows = [r for r in rows if "evaluation" in r]
    baseline = [r for r in eval_rows if r.get("condition") == "baseline"]
    positive = [r for r in eval_rows if r.get("condition") == "positive_objective"]
    negative = [r for r in eval_rows if r.get("condition") == "negative_objective"]
    random_rows = [r for r in eval_rows if r.get("condition") == "random_control"]
    base_rate = sum(r["evaluation"].get("success", False) for r in baseline) / max(len(baseline), 1)
    pos_rate = sum(r["evaluation"].get("success", False) for r in positive) / max(len(positive), 1)
    neg_rate = sum(r["evaluation"].get("success", False) for r in negative) / max(len(negative), 1)
    rand_rate = sum(r["evaluation"].get("success", False) for r in random_rows) / max(len(random_rows), 1)
    if destabilizing:
        verdict = "DESTABILIZING"
    elif len(eval_rows) < 4:
        verdict = "INSUFFICIENT"
    elif pos_rate - base_rate >= 0.05 and pos_rate > neg_rate and pos_rate > rand_rate:
        verdict = "PROMISING"
    else:
        verdict = "STABLE_NO_EFFECT"
    payload = {
        "BG_SOFT_STEERING_VERDICT": verdict,
        "verdict": verdict,
        "metrics": {
            "baseline_success": base_rate,
            "positive_success": pos_rate,
            "negative_success": neg_rate,
            "random_success": rand_rate,
            "destabilizing": destabilizing,
            "rows": len(rows),
        },
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_md(OUT_MD, ["# BG Soft Hidden Steering Pilot (2026-05-18)", "", f"BG_SOFT_STEERING_VERDICT = {verdict}", "", f"- metrics: `{payload['metrics']}`"])
    print(f"BG_SOFT_STEERING_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
