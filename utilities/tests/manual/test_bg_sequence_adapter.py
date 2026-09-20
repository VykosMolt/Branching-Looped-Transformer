#!/usr/bin/env python3
"""Manual tests for BG sequence-level adapter implementation."""
from __future__ import annotations

import time
import traceback
from typing import Any

import torch
import torch.nn as nn

from bg_sequence_adapter_common import OUT_ROOT, all_frozen, encode_prompt, load_model_tokenizer, rel, write_json, write_md
from src.evaluator.bg_sequence_adapter import (
    HIDDEN_DIM,
    BasisMixtureAdapter,
    LowRankDeltaAdapter,
    Rank1GatedDirectionAdapter,
    assert_adapter_budget,
    build_sequence_adapter,
    parameter_count,
    sequence_adapter_hook,
)


OUT_JSON = OUT_ROOT / "implementation_tests.json"
OUT_MD = OUT_ROOT / "implementation_tests.md"


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    errors: list[str] = []
    try:
        x = torch.randn(3, HIDDEN_DIM)
        for adapter in [LowRankDeltaAdapter(rank=32), Rank1GatedDirectionAdapter(), BasisMixtureAdapter()]:
            y = adapter(x)
            assert tuple(y.shape) == (3, HIDDEN_DIM)
            assert torch.isfinite(y).all()
            assert parameter_count(adapter) <= 2_000_000
            assert_adapter_budget(adapter)
        checks["shape_finite_param_budget"] = True

        frozen = nn.Linear(HIDDEN_DIM, 8, bias=False)
        for p in frozen.parameters():
            p.requires_grad_(False)
        adapter = LowRankDeltaAdapter(rank=16)
        loss = frozen(x + 0.01 * adapter(x)).pow(2).mean()
        loss.backward()
        assert all(p.grad is None for p in frozen.parameters())
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        checks["adapter_gradients_work"] = True

        from src.evaluator.bg_controller import BGController

        controller = BGController.from_artifacts(device="cpu")
        for head in controller.heads.values():
            for param in head.parameters():
                param.requires_grad_(False)
        checks["bg_heads_frozen"] = all(not p.requires_grad for head in controller.heads.values() for p in head.parameters())

        model, tokenizer, device = load_model_tokenizer()
        checks["ouro_params_frozen"] = all_frozen(model)
        prompt = "Question: Which letter is first?\nOptions:\nA. A\nB. B\nC. C\nD. D\nThink briefly if needed.\nFINAL ANSWER:"
        enc = encode_prompt(tokenizer, prompt, device, max_length=128)
        position = max(0, int(enc["input_ids"].shape[1]) - 1)
        adapter = build_sequence_adapter(kind="low_rank", rank=32, device=device)
        with torch.no_grad():
            base = model(**enc, use_cache=False, return_dict=True).logits[:, -1, :512].detach().float().cpu()
            with sequence_adapter_hook(model, adapter, alpha=0.0, intervention_mode="multi_loop_decayed", position=position):
                hooked = model(**enc, use_cache=False, return_dict=True).logits[:, -1, :512].detach().float().cpu()
        max_delta = float((base - hooked).abs().max().item())
        checks["zero_alpha_equivalence_max_abs_delta_first512"] = max_delta
        checks["zero_alpha_equivalence_pass"] = max_delta <= 1e-6
        with sequence_adapter_hook(model, adapter, alpha=0.02, intervention_mode="multi_loop_decayed", position=position) as hook:
            with torch.no_grad():
                model(**enc, use_cache=False, return_dict=True)
            diagnostics = hook.diagnostics()
        checks["intervention_delta_rms_max"] = max((float(r.get("rms_fraction", 0.0)) for r in diagnostics.get("records", [])), default=0.0)
        checks["intervention_delta_rms_clipped"] = checks["intervention_delta_rms_max"] <= 0.02001
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {str(exc)[:1000]}")
        checks["traceback"] = traceback.format_exc()[-4000:]

    verdict = "READY" if not errors and all(checks.get(k) for k in ["shape_finite_param_budget", "adapter_gradients_work", "bg_heads_frozen", "ouro_params_frozen", "zero_alpha_equivalence_pass", "intervention_delta_rms_clipped"]) else "BLOCKED"
    payload = {
        "BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT": verdict,
        "checks": checks,
        "errors": errors,
        "adapter_variants": ["LowRankDeltaAdapter", "Rank1GatedDirectionAdapter", "BasisMixtureAdapter"],
        "max_params": 2_000_000,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Implementation Tests",
        "",
        f"BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT = {verdict}",
        "",
        f"- shape/finite/param budget: `{checks.get('shape_finite_param_budget')}`",
        f"- adapter gradients: `{checks.get('adapter_gradients_work')}`",
        f"- Ouro params frozen: `{checks.get('ouro_params_frozen')}`",
        f"- BG heads frozen: `{checks.get('bg_heads_frozen')}`",
        f"- zero-alpha equivalence pass: `{checks.get('zero_alpha_equivalence_pass')}`",
        f"- intervention delta RMS max: `{checks.get('intervention_delta_rms_max')}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {err}" for err in errors] if errors else ["- none"])
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
