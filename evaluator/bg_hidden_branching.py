"""Hidden-origin branch helpers for BG Phase 2 probes.

The module is intentionally lightweight: it does not load Ouro, checkpoints, or
tokenizers at import time.  It provides branch-delta construction plus guarded
forward-hook machinery used by the manual hidden-branch suite.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch


NUM_LOOPS = 4
HIDDEN_DIM = 2048
SAFE_ALPHA_CAP = 0.02
DIAGNOSTIC_ALPHA_CAP = 0.10


def rms_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return ``v`` with RMS one along its flattened dimension."""
    if not isinstance(v, torch.Tensor):
        raise TypeError(f"v must be a torch.Tensor, got {type(v).__name__}")
    x = v.detach().to(dtype=torch.float32).clone()
    rms = x.pow(2).mean().sqrt().clamp(min=float(eps))
    return x / rms


def _validate_alpha(alpha: float, *, allow_diagnostic_alpha: bool = False) -> float:
    value = float(alpha)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"alpha must be finite and non-negative, got {alpha!r}")
    cap = DIAGNOSTIC_ALPHA_CAP if allow_diagnostic_alpha else SAFE_ALPHA_CAP
    if value > cap + 1e-6:
        label = "diagnostic" if allow_diagnostic_alpha else "safe"
        raise ValueError(f"{label} alpha must be <= {cap}, got {value}")
    return value


def _orthogonalize(candidate: torch.Tensor, basis: Iterable[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    out = candidate.to(dtype=torch.float32).clone()
    for b in basis:
        unit = rms_normalize(b, eps=eps)
        denom = torch.dot(unit.flatten(), unit.flatten()).clamp(min=eps)
        out = out - torch.dot(out.flatten(), unit.flatten()) / denom * unit
    return rms_normalize(out, eps=eps)


def make_branch_deltas(
    hidden_dim: int,
    k: int,
    alpha: float,
    seed: int,
    include_clean: bool = True,
    orthogonalize: bool = True,
    direction_bank: dict | None = None,
    allow_diagnostic_alpha: bool = False,
) -> list[torch.Tensor]:
    """Create deterministic branch deltas with RMS equal to ``alpha``.

    Default branch semantics:
    - branch 0: zero/clean if ``include_clean`` is true
    - branch 1: +random RMS-normalized delta
    - branch 2: -same random delta
    - branch 3: second orthogonal random delta
    - optional later branches: supplied bank directions, then more random dirs
    """
    dim = int(hidden_dim)
    if dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    count = int(k)
    if count <= 0:
        raise ValueError(f"k must be positive, got {k}")
    a = _validate_alpha(alpha, allow_diagnostic_alpha=allow_diagnostic_alpha)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    deltas: list[torch.Tensor] = []
    basis: list[torch.Tensor] = []
    if include_clean:
        deltas.append(torch.zeros(dim, dtype=torch.float32))

    base = rms_normalize(torch.randn(dim, generator=gen, dtype=torch.float32))
    basis.append(base)
    if len(deltas) < count:
        deltas.append(base * a)
    if len(deltas) < count:
        deltas.append(-base * a)

    if len(deltas) < count:
        raw = torch.randn(dim, generator=gen, dtype=torch.float32)
        second = _orthogonalize(raw, basis) if orthogonalize else rms_normalize(raw)
        basis.append(second)
        deltas.append(second * a)

    for name in sorted((direction_bank or {}).keys()):
        if len(deltas) >= count:
            break
        raw = direction_bank[name]
        if not isinstance(raw, torch.Tensor) or int(raw.numel()) != dim:
            continue
        direction = _orthogonalize(raw.flatten(), basis) if orthogonalize else rms_normalize(raw.flatten())
        basis.append(direction)
        deltas.append(direction * a)

    while len(deltas) < count:
        raw = torch.randn(dim, generator=gen, dtype=torch.float32)
        direction = _orthogonalize(raw, basis) if orthogonalize else rms_normalize(raw)
        basis.append(direction)
        deltas.append(direction * a)

    return [d.to(dtype=torch.float32) for d in deltas[:count]]


def delta_rms(delta: torch.Tensor) -> float:
    return float(delta.detach().to(dtype=torch.float32).pow(2).mean().sqrt().cpu().item())


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    af = a.detach().to(dtype=torch.float32).flatten()
    bf = b.detach().to(dtype=torch.float32).flatten()
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    if float(denom.item()) <= eps:
        return 1.0 if float(torch.linalg.vector_norm(af - bf).item()) <= eps else 0.0
    return float(torch.dot(af, bf).div(denom.clamp(min=eps)).cpu().item())


def rms_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().to(dtype=torch.float32) - b.detach().to(dtype=torch.float32)).pow(2).mean().sqrt().cpu().item())


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_output_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    if isinstance(output, list):
        return [tensor] + list(output[1:])
    return tensor


@dataclass
class HiddenBranchRecord:
    mechanism: str
    layer: int
    loop: int
    position: int
    delta_rms: float
    hidden_rms: float
    residual_rms: float
    rms_fraction: float
    forward_call_index: int

    def to_json(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "layer": int(self.layer),
            "loop": int(self.loop),
            "position": int(self.position),
            "delta_rms": float(self.delta_rms),
            "hidden_rms": float(self.hidden_rms),
            "residual_rms": float(self.residual_rms),
            "rms_fraction": float(self.rms_fraction),
            "forward_call_index": int(self.forward_call_index),
        }


class HiddenDeltaLayerHook:
    """Temporarily add an RMS-scaled delta at one decoder layer and token."""

    method_label = "hook_hidden_origin_branch"

    def __init__(
        self,
        model: Any,
        *,
        target_layer: int,
        target_loops: list[int],
        delta: torch.Tensor,
        position: int = -1,
        max_rms_fraction: float = DIAGNOSTIC_ALPHA_CAP,
    ) -> None:
        self.model = model
        self.target_layer = int(target_layer)
        self.target_loops = [int(x) for x in target_loops]
        self.delta = delta.detach().flatten().to(device="cpu", dtype=torch.float32)
        if int(self.delta.numel()) <= 0:
            raise ValueError("delta must be non-empty")
        if any(loop < 1 or loop > NUM_LOOPS for loop in self.target_loops):
            raise ValueError(f"target_loops must be in 1..{NUM_LOOPS}, got {self.target_loops}")
        self.position = int(position)
        self.max_rms_fraction = float(max_rms_fraction)
        if not (0.0 <= self.max_rms_fraction <= DIAGNOSTIC_ALPHA_CAP + 1e-6):
            raise ValueError(f"max_rms_fraction must be in [0, {DIAGNOSTIC_ALPHA_CAP}], got {max_rms_fraction}")
        self.handle: Any | None = None
        self.forward_call_count = 0
        self.records: list[dict[str, Any]] = []
        self.nan_or_inf = False

    def _target_module(self) -> Any:
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise RuntimeError("model does not expose model.layers")
        index = self.target_layer - 1
        if index < 0 or index >= len(layers):
            raise RuntimeError(f"target layer {self.target_layer} maps to index {index}, model has {len(layers)} layers")
        return layers[index]

    def _loop_from_kwargs(self, kwargs: dict[str, Any] | None) -> int:
        if kwargs and "current_ut" in kwargs:
            value = kwargs["current_ut"]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().item())
            return int(value) + 1
        return ((self.forward_call_count - 1) % NUM_LOOPS) + 1

    def _hook_impl(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any] | None, output: Any) -> Any:
        self.forward_call_count += 1
        loop = self._loop_from_kwargs(kwargs)
        if loop not in self.target_loops or delta_rms(self.delta) == 0.0:
            return output
        tensor = _output_tensor(output)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"layer output is not tensor-like: {type(tensor).__name__}")
        if tensor.shape[-1] != int(self.delta.numel()):
            raise RuntimeError(f"delta dim {self.delta.numel()} != hidden dim {tensor.shape[-1]}")
        if not torch.isfinite(tensor).all():
            self.nan_or_inf = True
            return output
        seq_len = int(tensor.shape[1]) if tensor.ndim >= 3 else 1
        pos = self.position if self.position >= 0 else seq_len + self.position
        if pos < 0 or pos >= seq_len:
            raise IndexError(f"position {self.position} resolves to {pos}, outside seq_len={seq_len}")

        changed = tensor.clone()
        target = changed[:, pos : pos + 1, :] if tensor.ndim == 3 else changed
        base_rms = target.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        delta_dev = self.delta.to(device=target.device, dtype=target.dtype).view(*([1] * (target.ndim - 1)), -1)
        residual = delta_dev * base_rms.to(dtype=target.dtype)
        residual_rms = residual.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        max_allowed = self.max_rms_fraction * base_rms
        residual = residual * torch.clamp(max_allowed / residual_rms, max=1.0).to(dtype=target.dtype)
        changed_slice = target + residual
        if tensor.ndim == 3:
            changed[:, pos : pos + 1, :] = changed_slice
        else:
            changed = changed_slice
        if not torch.isfinite(changed).all():
            self.nan_or_inf = True
            return output
        hidden_rms = float(base_rms.mean().detach().cpu().item())
        residual_rms_mean = float(residual.float().pow(2).mean(dim=-1, keepdim=True).sqrt().mean().detach().cpu().item())
        self.records.append(
            HiddenBranchRecord(
                mechanism=self.method_label,
                layer=self.target_layer,
                loop=loop,
                position=self.position,
                delta_rms=delta_rms(self.delta),
                hidden_rms=hidden_rms,
                residual_rms=residual_rms_mean,
                rms_fraction=residual_rms_mean / max(hidden_rms, 1e-8),
                forward_call_index=self.forward_call_count,
            ).to_json()
        )
        return _replace_output_tensor(output, changed)

    def apply(self) -> "HiddenDeltaLayerHook":
        self.remove()
        module = self._target_module()
        self.handle = module.register_forward_hook(
            lambda mod, args, kwargs, out: self._hook_impl(mod, args, kwargs, out),
            with_kwargs=True,
        )
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mechanism": self.method_label,
            "target_layer": self.target_layer,
            "target_loops": self.target_loops,
            "delta_rms": delta_rms(self.delta),
            "position": self.position,
            "forward_call_count": self.forward_call_count,
            "modifications": len(self.records),
            "nan_or_inf": bool(self.nan_or_inf),
            "records": self.records,
        }


class HookHiddenOriginBrancher:
    """Hidden-origin fallback brancher using same-prefix layer hooks."""

    method_label = "hook_intervention_per_branch"

    def __init__(self, model: Any) -> None:
        self.model = model

    def run_branch_with_hook(self, fn, *, target_layer: int, target_loops: list[int], delta: torch.Tensor, position: int = -1) -> dict[str, Any]:
        hook = HiddenDeltaLayerHook(
            self.model,
            target_layer=target_layer,
            target_loops=target_loops,
            delta=delta,
            position=position,
            max_rms_fraction=max(delta_rms(delta), SAFE_ALPHA_CAP),
        )
        started = time.time()
        try:
            hook.apply()
            result = fn()
        finally:
            hook.remove()
        return {
            "method": self.method_label,
            "result": result,
            "diagnostics": hook.diagnostics(),
            "elapsed_seconds": round(time.time() - started, 3),
        }


class TrueLatentForkCarry:
    """Placeholder interface for true branch-batch carry.

    The current local Ouro model exposes useful internal loop methods, but the
    standard generation path does not expose a validated resume-from-copied-layer
    hidden-state and matching cache API.  Feasibility scripts may mark loop
    boundary prompt-only carry as partial, but generation carry must not be faked.
    """

    method_label = "true_fork_carry"

    def __init__(self, model: Any) -> None:
        self.model = model

    def run_to_branch_point(self, *args, **kwargs):
        raise NotImplementedError("true layer-level fork/carry is not generation-ready for local Ouro")

    def fork_hidden(self, *args, **kwargs):
        raise NotImplementedError("true layer-level fork/carry is not generation-ready for local Ouro")

    def continue_branches_to_layers(self, *args, **kwargs):
        raise NotImplementedError("true layer-level fork/carry is not generation-ready for local Ouro")

    def continue_branches_to_output(self, *args, **kwargs):
        raise NotImplementedError("true autoregressive hidden-state carry requires branch-aware cache/state handling")
