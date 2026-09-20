"""Manual unit tests for BG Stage 2 steering helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_steering_hook import (  # noqa: E402
    BGLatentLoopBoundaryFork,
    BGLayerHookSteering,
    build_intervention_mode,
)


class TinyDecoderLayer(nn.Module):
    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), scale), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, current_ut: int = 0, **_kwargs):
        del current_ut
        return hidden_states + self.bias.view(1, 1, -1)


class TinyInner(nn.Module):
    def __init__(self, dim: int, layers: int, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.total_ut_steps = int(config.total_ut_steps)
        self.layers = nn.ModuleList([TinyDecoderLayer(dim, 0.01 * (idx + 1)) for idx in range(layers)])


class TinyModel(nn.Module):
    def __init__(self, dim: int = 4, layers: int = 4, vocab: int = 9) -> None:
        super().__init__()
        self.config = SimpleNamespace(total_ut_steps=4, use_cache=False)
        self.model = TinyInner(dim, layers, self.config)
        self.embed_tokens = nn.Embedding(vocab, dim)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(vocab * dim, dtype=torch.float32).view(vocab, dim) / 100.0)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(vocab * dim, dtype=torch.float32).view(vocab, dim) / 50.0)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        return_per_loop_hidden_states=False,
        logits_to_keep=1,
        **_kwargs,
    ):
        del attention_mask, position_ids, use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        states = []
        for current_ut in range(int(self.config.total_ut_steps)):
            for layer in self.model.layers:
                hidden = layer(hidden, current_ut=current_ut)
            states.append(hidden)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits_hidden = hidden[:, -logits_to_keep:, :]
        else:
            logits_hidden = hidden
        return SimpleNamespace(
            logits=self.lm_head(logits_hidden),
            per_loop_hidden_states=states if return_per_loop_hidden_states else None,
        )

    def generate(self, input_ids, max_new_tokens: int, **kwargs):
        del kwargs
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self(input_ids=ids, return_per_loop_hidden_states=False, logits_to_keep=1)
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def assert_raises(expected, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def unit_direction(dim: int = 4) -> torch.Tensor:
    vec = torch.zeros(dim)
    vec[0] = 1.0
    return vec


def test_validation() -> None:
    model = TinyModel()
    assert_raises(ValueError, BGLayerHookSteering, model, 2, [1], unit_direction(), 0.025)
    assert_raises(ValueError, BGLayerHookSteering, model, 2, [1], torch.ones(4), 0.01)


def test_modes() -> None:
    assert build_intervention_mode("single_loop_L1", 0.01)["target_loops"] == [1]
    assert build_intervention_mode("single_loop_L4", 0.01)["target_loops"] == [4]
    assert build_intervention_mode("multi_loop_uniform", 0.01)["loop_alpha_scales"] == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    assert build_intervention_mode("multi_loop_decayed", 0.01)["loop_alpha_scales"] == {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}


def test_layer_hook_once_and_remove() -> None:
    model = TinyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    hook = BGLayerHookSteering(model, target_layer=2, target_loops=[2], direction=unit_direction(), alpha=0.01)
    hook.apply(position=-1)
    with torch.no_grad():
        model(input_ids=input_ids)
    hook.remove()
    assert hook.modifications == 1, hook.diagnostics()
    before = hook.modifications
    with torch.no_grad():
        model(input_ids=input_ids)
    assert hook.modifications == before


def test_zero_alpha_generation_equivalence() -> None:
    model = TinyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        baseline = model.generate(input_ids=input_ids, max_new_tokens=3)
    hook = BGLayerHookSteering(model, target_layer=2, target_loops=[2], direction=unit_direction(), alpha=0.0)
    hook.apply(position=-1)
    try:
        with torch.no_grad():
            hooked = model.generate(input_ids=input_ids, max_new_tokens=3)
    finally:
        hook.remove()
    assert torch.equal(baseline, hooked)


def test_tiny_alpha_state_change_and_multiloop_records() -> None:
    model = TinyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits
    spec = build_intervention_mode("multi_loop_uniform", 0.01)
    hook = BGLayerHookSteering(
        model,
        target_layer=2,
        target_loops=spec["target_loops"],
        loop_alpha_scales=spec["loop_alpha_scales"],
        direction=unit_direction(),
        alpha=0.01,
    )
    hook.apply(position=-1)
    try:
        with torch.no_grad():
            changed = model(input_ids=input_ids).logits
    finally:
        hook.remove()
    assert not torch.equal(baseline, changed)
    diag = hook.diagnostics()
    assert set(diag["per_loop_activation_rms_change"]) == {"1", "2", "3", "4"}
    assert all(value > 0 for value in diag["per_loop_activation_rms_change"].values())


def test_decayed_loop_scales_order() -> None:
    model = TinyModel()
    spec = build_intervention_mode("multi_loop_decayed", 0.02)
    hook = BGLayerHookSteering(
        model,
        target_layer=2,
        target_loops=spec["target_loops"],
        loop_alpha_scales=spec["loop_alpha_scales"],
        direction=unit_direction(),
        alpha=0.02,
    )
    hook.apply(position=-1)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2, 3]]))
    finally:
        hook.remove()
    assert [round(r["alpha_eff"], 4) for r in hook.records] == [0.005, 0.01, 0.015, 0.02]


def test_latent_loop_boundary_fork_zero_alpha_and_restore() -> None:
    model = TinyModel()
    old_steps = model.config.total_ut_steps
    fork = BGLatentLoopBoundaryFork(
        model,
        direction=unit_direction(),
        alpha=0.0,
        intervention_mode="single_loop_L1",
    )
    with torch.no_grad():
        result = fork.run(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones((1, 3), dtype=torch.long))
    assert result["zero_alpha_full_clean_max_abs_delta"] < 1e-6
    assert result["hidden_delta_rms"] < 1e-6
    assert model.config.total_ut_steps == old_steps
    assert fork.total_ut_steps_restored
    assert fork.use_cache is False


def main() -> int:
    tests = [
        test_validation,
        test_modes,
        test_layer_hook_once_and_remove,
        test_zero_alpha_generation_equivalence,
        test_tiny_alpha_state_change_and_multiloop_records,
        test_decayed_loop_scales_order,
        test_latent_loop_boundary_fork_zero_alpha_and_restore,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("BG_STAGE2_HOOK_IMPLEMENTATION_VERDICT = PARTIAL")
    print("LAYER_HOOK_INJECTION_VERDICT = READY")
    print("LATENT_LOOP_BOUNDARY_FORK_VERDICT = BLOCKED_FOR_GENERATION_CONTINUATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
