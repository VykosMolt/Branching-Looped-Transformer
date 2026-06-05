"""Sequence-level BG intervention adapters.

This module is intentionally lightweight.  It defines small adapter modules and
the differentiable layer-hook used by the manual sequence-reward experiments;
it does not load Ouro, tokenizer files, checkpoints, or BG heads at import time.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluator.bg_causal_adapter import (
    HIDDEN_DIM,
    LowRankDeltaAdapter,
    Rank1GatedDirectionAdapter,
    assert_adapter_budget,
    parameter_count,
    rms,
    rms_normalize,
)


NUM_LOOPS = 4
TARGET_LAYER = 36
PRIMARY_MODE = "multi_loop_decayed"


def loop_scales(mode: str) -> tuple[list[int], dict[int, float]]:
    if mode == "single_loop_L1":
        return [1], {1: 1.0}
    if mode == "single_loop_L4":
        return [4], {4: 1.0}
    if mode == "multi_loop_uniform":
        return [1, 2, 3, 4], {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    if mode == "multi_loop_decayed":
        return [1, 2, 3, 4], {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
    raise ValueError(f"unsupported intervention mode {mode!r}")


class BasisMixtureAdapter(nn.Module):
    """Compact adapter useful for black-box or bandit optimization."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_basis: int = 16, router_hidden: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.basis = nn.Parameter(torch.randn(self.num_basis, self.hidden_dim) / (self.hidden_dim**0.5))
        self.router = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(router_hidden)),
            nn.GELU(),
            nn.Linear(int(router_hidden), self.num_basis),
        )

    def forward(self, hidden_at_position: torch.Tensor) -> torch.Tensor:
        h = hidden_at_position.float()
        weights = torch.softmax(self.router(h), dim=-1)
        basis = rms_normalize(self.basis.to(device=h.device, dtype=h.dtype))
        return rms_normalize(weights @ basis)


class SequenceAdapterInterventionHook:
    """Differentiable layer hook for sequence-level adapter optimization."""

    def __init__(
        self,
        model: Any,
        adapter: nn.Module,
        *,
        alpha: float,
        intervention_mode: str = PRIMARY_MODE,
        target_layer: int = TARGET_LAYER,
        max_rms_fraction: float = 0.02,
        max_alpha: float = 0.02,
    ) -> None:
        if float(alpha) < 0:
            raise ValueError("alpha must be non-negative")
        if float(alpha) > float(max_alpha) + 1e-12:
            raise ValueError(f"alpha must be <= {max_alpha}, got {alpha}")
        self.model = model
        self.adapter = adapter
        self.alpha = float(alpha)
        self.target_layer = int(target_layer)
        self.target_loops, self.scales = loop_scales(intervention_mode)
        self.intervention_mode = str(intervention_mode)
        self.max_rms_fraction = float(max_rms_fraction)
        if not (0 < self.max_rms_fraction <= 1.0):
            raise ValueError(f"max_rms_fraction must be in (0, 1], got {max_rms_fraction}")
        self.position = -1
        self.handle: Any | None = None
        self.forward_call_count = 0
        self.records: list[dict[str, Any]] = []
        self.delta_fraction_tensors: list[torch.Tensor] = []
        self.loop_index_source = "uninitialized"
        self.nan_or_inf = False

    def _target_module(self) -> Any:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("model does not expose model.layers")
        index = max(0, self.target_layer - 1)
        if index >= len(layers):
            raise RuntimeError(f"target layer {self.target_layer} maps to missing index {index}")
        return layers[index]

    def _loop(self, kwargs: dict[str, Any] | None) -> int:
        if kwargs and "current_ut" in kwargs:
            value = kwargs["current_ut"]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().item())
            self.loop_index_source = "current_ut"
            return int(value) + 1
        self.loop_index_source = "validated_call_counter"
        return ((self.forward_call_count - 1) % NUM_LOOPS) + 1

    def _hook(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any] | None, output: Any) -> Any:
        self.forward_call_count += 1
        loop = self._loop(kwargs)
        if loop not in self.target_loops or self.alpha == 0.0:
            return output
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"layer output is not tensor-like: {type(tensor).__name__}")
        if not torch.isfinite(tensor).all():
            self.nan_or_inf = True
            return output
        seq_len = int(tensor.shape[1]) if tensor.ndim >= 3 else 1
        pos = self.position if self.position >= 0 else seq_len + self.position
        pos = max(0, min(int(pos), seq_len - 1))
        changed = tensor.clone()
        target = changed[:, pos, :] if tensor.ndim == 3 else changed
        adapter_input = target.detach().float()
        direction = rms_normalize(self.adapter(adapter_input)).to(device=target.device, dtype=target.dtype)
        alpha_eff = self.alpha * float(self.scales.get(loop, 1.0))
        hidden_rms = rms(target, dim=-1, keepdim=True).to(device=target.device, dtype=target.dtype)
        delta = float(alpha_eff) * hidden_rms * direction
        delta_rms = rms(delta, dim=-1, keepdim=True).to(dtype=torch.float32)
        hidden_rms_f = hidden_rms.to(dtype=torch.float32).clamp(min=1e-8)
        max_rms = (self.max_rms_fraction * 0.999) * hidden_rms_f
        scale = torch.minimum(torch.ones_like(delta_rms), max_rms / delta_rms.clamp(min=1e-8))
        delta = delta * scale.to(device=delta.device, dtype=delta.dtype)
        if tensor.ndim == 3:
            changed[:, pos, :] = target + delta
        else:
            changed = target + delta
        if not torch.isfinite(changed).all():
            self.nan_or_inf = True
            return output
        frac_tensor = rms(delta, dim=-1, keepdim=False).to(dtype=torch.float32) / hidden_rms_f.squeeze(-1)
        frac = frac_tensor.mean()
        self.delta_fraction_tensors.append(frac)
        self.records.append(
            {
                "layer": self.target_layer,
                "loop": int(loop),
                "position": int(pos),
                "alpha_eff": float(alpha_eff),
                "rms_fraction": float(frac.detach().cpu().item()),
                "forward_call_index": int(self.forward_call_count),
            }
        )
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        if isinstance(output, list):
            return [changed] + list(output[1:])
        return changed

    def apply(self, position: int) -> "SequenceAdapterInterventionHook":
        self.remove()
        self.position = int(position)
        module = self._target_module()
        try:
            self.handle = module.register_forward_hook(
                lambda mod, args, kwargs, out: self._hook(mod, args, kwargs, out),
                with_kwargs=True,
            )
        except TypeError:
            self.handle = module.register_forward_hook(lambda mod, args, out: self._hook(mod, args, None, out))
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        per_loop: dict[str, list[float]] = defaultdict(list)
        for row in self.records:
            per_loop[str(row["loop"])].append(float(row["rms_fraction"]))
        return {
            "hook_forward_call_count": self.forward_call_count,
            "hook_modifications": len(self.records),
            "hook_loop_index_source": self.loop_index_source,
            "nan_or_inf_activations": self.nan_or_inf,
            "activation_rms_change": (
                sum(float(row["rms_fraction"]) for row in self.records) / max(len(self.records), 1)
            ),
            "per_loop_activation_rms_change": {
                key: sum(vals) / max(len(vals), 1) for key, vals in sorted(per_loop.items())
            },
            "records": self.records,
        }


@contextmanager
def sequence_adapter_hook(
    model: Any,
    adapter: nn.Module,
    *,
    alpha: float,
    intervention_mode: str,
    position: int,
    max_rms_fraction: float = 0.02,
    max_alpha: float = 0.02,
) -> Iterator[SequenceAdapterInterventionHook]:
    hook = SequenceAdapterInterventionHook(
        model,
        adapter,
        alpha=alpha,
        intervention_mode=intervention_mode,
        max_rms_fraction=max_rms_fraction,
        max_alpha=max_alpha,
    ).apply(position)
    try:
        yield hook
    finally:
        hook.remove()


def build_sequence_adapter(kind: str = "low_rank", rank: int = 32, device: torch.device | str | None = None) -> nn.Module:
    if kind == "low_rank":
        adapter: nn.Module = LowRankDeltaAdapter(rank=int(rank))
    elif kind == "rank1":
        adapter = Rank1GatedDirectionAdapter()
    elif kind == "basis_mixture":
        adapter = BasisMixtureAdapter()
    else:
        raise ValueError(f"unknown sequence adapter kind {kind!r}")
    assert_adapter_budget(adapter)
    if device is not None:
        adapter.to(device)
    return adapter


__all__ = [
    "BasisMixtureAdapter",
    "HIDDEN_DIM",
    "LowRankDeltaAdapter",
    "PRIMARY_MODE",
    "Rank1GatedDirectionAdapter",
    "SequenceAdapterInterventionHook",
    "assert_adapter_budget",
    "build_sequence_adapter",
    "loop_scales",
    "parameter_count",
    "rms_normalize",
    "sequence_adapter_hook",
]
