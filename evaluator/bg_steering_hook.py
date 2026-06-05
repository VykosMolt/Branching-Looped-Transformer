"""Guarded BG steering interventions for local Ouro-RLTT.

This module is intentionally inert at import time.  It only registers hooks or
temporarily changes loop counts inside explicit helper calls.
"""
from __future__ import annotations

import inspect
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAP_LAYERS = (24, 36, 47)
LAYER_TO_CAPTURE_INDEX = {24: 23, 36: 35}
NUM_LOOPS = 4
HIDDEN_DIM = 2048
MAX_ALPHA = 0.02


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if value < 0:
        raise ValueError(f"alpha must be non-negative, got {value}")
    if value > MAX_ALPHA + 1e-12:
        raise ValueError(f"alpha must be <= {MAX_ALPHA}, got {value}")
    return value


def _validate_direction(direction: torch.Tensor, normalization: str = "l2") -> torch.Tensor:
    if not isinstance(direction, torch.Tensor):
        raise ValueError(f"direction must be a torch.Tensor, got {type(direction).__name__}")
    vec = direction.detach().flatten().to(device="cpu", dtype=torch.float32)
    if vec.ndim != 1 or int(vec.numel()) <= 0:
        raise ValueError(f"direction must be one-dimensional and non-empty, got shape {tuple(direction.shape)}")
    mode = str(normalization)
    if mode == "l2":
        norm = float(torch.linalg.vector_norm(vec).item())
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise ValueError(f"direction must be L2-unit within 1e-3, got norm={norm:.6f}")
    elif mode == "rms":
        rms = float(vec.pow(2).mean().sqrt().item())
        if not math.isfinite(rms) or abs(rms - 1.0) > 1e-3:
            raise ValueError(f"direction must be RMS-unit within 1e-3, got rms={rms:.6f}")
    else:
        raise ValueError(f"unknown direction normalization {normalization!r}; expected 'l2' or 'rms'")
    return vec


def _rms(tensor: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    return tensor.float().pow(2).mean(dim=dim, keepdim=keepdim).sqrt().clamp(min=1e-8)


def _replace_output_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    if isinstance(output, list):
        return [tensor] + list(output[1:])
    return tensor


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def build_intervention_mode(mode: str, alpha: float) -> dict[str, Any]:
    """Return loop targets and per-loop alpha scales for a Stage 2 mode."""
    _validate_alpha(alpha)
    if mode == "single_loop_L1":
        return {"mode": mode, "target_loops": [1], "loop_alpha_scales": {1: 1.0}}
    if mode == "single_loop_L4":
        return {"mode": mode, "target_loops": [4], "loop_alpha_scales": {4: 1.0}}
    if mode == "multi_loop_uniform":
        return {
            "mode": mode,
            "target_loops": [1, 2, 3, 4],
            "loop_alpha_scales": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        }
    if mode == "multi_loop_decayed":
        return {
            "mode": mode,
            "target_loops": [1, 2, 3, 4],
            "loop_alpha_scales": {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00},
        }
    raise ValueError(
        "unknown intervention mode "
        f"{mode!r}; expected single_loop_L1, single_loop_L4, multi_loop_uniform, or multi_loop_decayed"
    )


@dataclass
class SteeringRecord:
    mechanism: str
    layer: int | None
    loop: int
    position: int
    alpha_eff: float
    hidden_rms: float
    residual_rms: float
    rms_fraction: float
    clipped: bool
    forward_call_index: int

    def to_json(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "layer": self.layer,
            "loop": self.loop,
            "position": self.position,
            "alpha_eff": self.alpha_eff,
            "hidden_rms": self.hidden_rms,
            "residual_rms": self.residual_rms,
            "rms_fraction": self.rms_fraction,
            "clipped": self.clipped,
            "forward_call_index": self.forward_call_index,
        }


def _apply_direction_delta(
    hidden_slice: torch.Tensor,
    direction: torch.Tensor,
    alpha_eff: float,
    max_rms_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if alpha_eff == 0.0:
        return hidden_slice, {
            "hidden_rms": float(_rms(hidden_slice).mean().detach().cpu().item()),
            "residual_rms": 0.0,
            "rms_fraction": 0.0,
            "clipped": False,
        }
    direction_dev = direction.to(device=hidden_slice.device, dtype=hidden_slice.dtype)
    view_shape = [1] * hidden_slice.ndim
    view_shape[-1] = int(direction_dev.numel())
    base_rms = _rms(hidden_slice, dim=-1, keepdim=True)
    residual = float(alpha_eff) * direction_dev.view(*view_shape) * base_rms.to(dtype=hidden_slice.dtype)
    residual_rms = _rms(residual, dim=-1, keepdim=True)
    max_allowed = float(max_rms_fraction) * base_rms
    scale = torch.clamp(max_allowed / residual_rms.clamp(min=1e-8), max=1.0).to(dtype=hidden_slice.dtype)
    clipped = bool(torch.any(scale < 0.999999).detach().cpu().item())
    residual = residual * scale
    changed = hidden_slice + residual
    residual_rms_mean = float(_rms(residual, dim=-1, keepdim=True).mean().detach().cpu().item())
    hidden_rms_mean = float(base_rms.mean().detach().cpu().item())
    return changed, {
        "hidden_rms": hidden_rms_mean,
        "residual_rms": residual_rms_mean,
        "rms_fraction": residual_rms_mean / max(hidden_rms_mean, 1e-8),
        "clipped": clipped,
    }


class BGLayerHookSteering:
    """Temporary forward-hook steering at one Ouro decoder layer."""

    def __init__(
        self,
        model: Any,
        target_layer: int,
        target_loops: list[int],
        direction: torch.Tensor,
        alpha: float,
        loop_alpha_scales: dict[int, float] | None = None,
        max_rms_fraction: float = 0.05,
        direction_normalization: str = "l2",
    ) -> None:
        self.model = model
        self.target_layer = int(target_layer)
        self.target_loops = [int(loop) for loop in target_loops]
        bad_loops = [loop for loop in self.target_loops if loop < 1 or loop > NUM_LOOPS]
        if bad_loops:
            raise ValueError(f"target_loops must be 1-based loop ids in 1..{NUM_LOOPS}, got {bad_loops}")
        self.direction_normalization = str(direction_normalization)
        self.direction = _validate_direction(direction, normalization=self.direction_normalization)
        self.alpha = _validate_alpha(alpha)
        self.loop_alpha_scales = {int(k): float(v) for k, v in (loop_alpha_scales or {}).items()}
        self.max_rms_fraction = float(max_rms_fraction)
        if not (0 < self.max_rms_fraction <= 1.0):
            raise ValueError(f"max_rms_fraction must be in (0, 1], got {self.max_rms_fraction}")
        self.handle: Any | None = None
        self.records: list[dict[str, Any]] = []
        self.forward_call_count = 0
        self.nan_or_inf = False
        self.loop_index_source = "uninitialized"
        self.position = -1

    @property
    def modifications(self) -> int:
        return len(self.records)

    def _target_module(self) -> Any:
        inner_model = getattr(self.model, "model", None)
        layers = getattr(inner_model, "layers", None)
        if layers is None:
            raise RuntimeError("model does not expose model.layers for layer-hook steering")
        # Stage 1 uses human layer ids 24 and 36 as post-block captures at
        # zero-based layer indices 23 and 35.  Layer 47 is represented as a
        # boundary tap in BG features; if requested for layer-hook steering,
        # use the decoder block immediately before that boundary.
        index = max(0, self.target_layer - 1)
        if index >= len(layers):
            raise RuntimeError(f"target layer {self.target_layer} maps to index {index}, but model has {len(layers)} layers")
        return layers[index]

    def _loop_from_kwargs(self, kwargs: dict[str, Any] | None) -> int:
        if kwargs and "current_ut" in kwargs:
            value = kwargs.get("current_ut")
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().item())
            loop = int(value) + 1
            self.loop_index_source = "current_ut"
            return loop
        self.loop_index_source = "validated_call_counter"
        return ((self.forward_call_count - 1) % NUM_LOOPS) + 1

    def _hook_impl(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any] | None, output: Any) -> Any:
        self.forward_call_count += 1
        loop = self._loop_from_kwargs(kwargs)
        if loop not in self.target_loops:
            return output
        if self.alpha == 0.0:
            return output
        tensor = _output_tensor(output)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"layer output is not a tensor-like object: {type(tensor).__name__}")
        if tensor.shape[-1] != self.direction.numel():
            raise RuntimeError(
                f"direction dim {self.direction.numel()} does not match layer hidden dim {tensor.shape[-1]}"
            )
        if not torch.isfinite(tensor).all():
            self.nan_or_inf = True
            return output

        pos = self.position
        seq_len = int(tensor.shape[1]) if tensor.ndim >= 3 else 1
        pos_index = pos if pos >= 0 else seq_len + pos
        if pos_index < 0 or pos_index >= seq_len:
            raise IndexError(f"position {self.position} resolves to {pos_index}, outside sequence length {seq_len}")

        changed_tensor = tensor.clone()
        target_slice = changed_tensor[:, pos_index : pos_index + 1, :] if tensor.ndim == 3 else changed_tensor
        alpha_eff = self.alpha * float(self.loop_alpha_scales.get(loop, 1.0))
        changed_slice, stats = _apply_direction_delta(
            target_slice,
            self.direction,
            alpha_eff=alpha_eff,
            max_rms_fraction=self.max_rms_fraction,
        )
        if tensor.ndim == 3:
            changed_tensor[:, pos_index : pos_index + 1, :] = changed_slice
        else:
            changed_tensor = changed_slice
        if not torch.isfinite(changed_tensor).all():
            self.nan_or_inf = True
            return output

        self.records.append(
            SteeringRecord(
                mechanism="layer_hook_injection",
                layer=self.target_layer,
                loop=loop,
                position=self.position,
                alpha_eff=float(alpha_eff),
                hidden_rms=float(stats["hidden_rms"]),
                residual_rms=float(stats["residual_rms"]),
                rms_fraction=float(stats["rms_fraction"]),
                clipped=bool(stats["clipped"]),
                forward_call_index=self.forward_call_count,
            ).to_json()
        )
        return _replace_output_tensor(output, changed_tensor)

    def apply(self, position: int = -1) -> "BGLayerHookSteering":
        """Register a temporary forward hook at the requested token position."""
        self.remove()
        self.position = int(position)
        module = self._target_module()
        try:
            signature = inspect.signature(module.register_forward_hook)
            supports_kwargs = "with_kwargs" in signature.parameters
        except Exception:
            supports_kwargs = False
        if supports_kwargs:
            self.handle = module.register_forward_hook(
                lambda mod, args, kwargs, out: self._hook_impl(mod, args, kwargs, out),
                with_kwargs=True,
            )
        else:
            self.handle = module.register_forward_hook(
                lambda mod, args, out: self._hook_impl(mod, args, None, out)
            )
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        per_loop: dict[str, list[float]] = {}
        for record in self.records:
            per_loop.setdefault(str(record["loop"]), []).append(float(record["rms_fraction"]))
        return {
            "mechanism": "layer_hook_injection",
            "target_layer": self.target_layer,
            "target_loops": self.target_loops,
            "alpha": self.alpha,
            "loop_alpha_scales": self.loop_alpha_scales,
            "direction_normalization": self.direction_normalization,
            "forward_call_count": self.forward_call_count,
            "modifications": self.modifications,
            "loop_index_source": self.loop_index_source,
            "nan_or_inf": self.nan_or_inf,
            "records": self.records,
            "activation_rms_change": (
                sum(float(r["rms_fraction"]) for r in self.records) / max(len(self.records), 1)
            ),
            "per_loop_activation_rms_change": {
                loop: sum(vals) / max(len(vals), 1) for loop, vals in sorted(per_loop.items())
            },
        }


@contextmanager
def _temporary_total_ut_steps(model: Any, steps: int) -> Iterator[None]:
    inner = getattr(model, "model", None)
    old_config = getattr(getattr(model, "config", None), "total_ut_steps", None)
    old_inner_config = getattr(getattr(inner, "config", None), "total_ut_steps", None)
    old_inner_attr = getattr(inner, "total_ut_steps", None)
    try:
        if hasattr(model, "config") and hasattr(model.config, "total_ut_steps"):
            model.config.total_ut_steps = int(steps)
        if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "total_ut_steps"):
            inner.config.total_ut_steps = int(steps)
        if inner is not None and hasattr(inner, "total_ut_steps"):
            inner.total_ut_steps = int(steps)
        yield
    finally:
        if old_config is not None and hasattr(model, "config") and hasattr(model.config, "total_ut_steps"):
            model.config.total_ut_steps = old_config
        if old_inner_config is not None and inner is not None and hasattr(inner, "config") and hasattr(inner.config, "total_ut_steps"):
            inner.config.total_ut_steps = old_inner_config
        if old_inner_attr is not None and inner is not None and hasattr(inner, "total_ut_steps"):
            inner.total_ut_steps = old_inner_attr


class BGLatentLoopBoundaryFork:
    """Loop-boundary intervention helper.

    The helper can prove zero-alpha equivalence for loop-boundary forward
    computation.  It deliberately uses ``use_cache=False`` and does not attempt
    to persist a hidden boundary through HuggingFace text generation.
    """

    def __init__(
        self,
        model: Any,
        direction: torch.Tensor,
        alpha: float,
        intervention_mode: str,
        max_rms_fraction: float = 0.05,
        use_cache: bool = False,
    ) -> None:
        self.model = model
        self.direction = _validate_direction(direction)
        self.alpha = _validate_alpha(alpha)
        self.mode_spec = build_intervention_mode(intervention_mode, alpha)
        self.intervention_mode = intervention_mode
        self.max_rms_fraction = float(max_rms_fraction)
        self.use_cache = bool(use_cache)
        if self.use_cache:
            raise ValueError("BGLatentLoopBoundaryFork requires use_cache=False unless cache forking is implemented")
        self.records: list[dict[str, Any]] = []
        self.nan_or_inf = False
        self.total_ut_steps_restored = False

    def _forward_loops(
        self,
        *,
        loops: int,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _temporary_total_ut_steps(self.model, int(loops)):
            output = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_per_loop_hidden_states=True,
                logits_to_keep=1,
            )
        hidden_states = list(getattr(output, "per_loop_hidden_states", None) or [])
        if not hidden_states:
            raise RuntimeError("model did not return per_loop_hidden_states for latent boundary fork")
        return hidden_states[-1], output.logits

    def _perturb(self, hidden: torch.Tensor, loop: int, position: int, call_index: int) -> torch.Tensor:
        if loop not in self.mode_spec["target_loops"] or self.alpha == 0.0:
            return hidden
        pos_index = position if position >= 0 else int(hidden.shape[1]) + int(position)
        if pos_index < 0 or pos_index >= int(hidden.shape[1]):
            raise IndexError(f"position {position} is outside sequence length {hidden.shape[1]}")
        changed = hidden.clone()
        target_slice = changed[:, pos_index : pos_index + 1, :]
        alpha_eff = self.alpha * float(self.mode_spec["loop_alpha_scales"].get(loop, 1.0))
        changed_slice, stats = _apply_direction_delta(
            target_slice,
            self.direction,
            alpha_eff=alpha_eff,
            max_rms_fraction=self.max_rms_fraction,
        )
        changed[:, pos_index : pos_index + 1, :] = changed_slice
        if not torch.isfinite(changed).all():
            self.nan_or_inf = True
            return hidden
        self.records.append(
            SteeringRecord(
                mechanism="latent_loop_boundary_fork",
                layer=None,
                loop=int(loop),
                position=int(position),
                alpha_eff=float(alpha_eff),
                hidden_rms=float(stats["hidden_rms"]),
                residual_rms=float(stats["residual_rms"]),
                rms_fraction=float(stats["rms_fraction"]),
                clipped=bool(stats["clipped"]),
                forward_call_index=int(call_index),
            ).to_json()
        )
        return changed

    def run(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position: int = -1,
    ) -> dict[str, Any]:
        """Run loop-boundary fork diagnostics for one prompt/prefix tensor."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        started = time.time()
        old_steps = getattr(getattr(self.model, "config", None), "total_ut_steps", None)
        old_inner_steps = getattr(getattr(getattr(self.model, "model", None), "config", None), "total_ut_steps", None)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        with torch.inference_mode():
            full_hidden, full_logits = self._forward_loops(
                loops=NUM_LOOPS,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            if self.intervention_mode == "single_loop_L1":
                h1, _ = self._forward_loops(
                    loops=1,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                h1_perturbed = self._perturb(h1, loop=1, position=position, call_index=1)
                clean_hidden, clean_logits = self._forward_loops(
                    loops=3,
                    inputs_embeds=h1,
                    attention_mask=attention_mask,
                )
                intervened_hidden, intervened_logits = self._forward_loops(
                    loops=3,
                    inputs_embeds=h1_perturbed,
                    attention_mask=attention_mask,
                )
            elif self.intervention_mode == "single_loop_L4":
                clean_hidden, clean_logits = full_hidden, full_logits
                intervened_hidden = self._perturb(full_hidden, loop=4, position=position, call_index=4)
                intervened_logits = self.model.lm_head(intervened_hidden[:, -1:, :])
            else:
                current_clean: torch.Tensor | None = None
                current_intervened: torch.Tensor | None = None
                clean_logits = full_logits
                intervened_logits = full_logits
                for loop in range(1, NUM_LOOPS + 1):
                    if loop == 1:
                        current_clean, _ = self._forward_loops(
                            loops=1,
                            input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                        )
                        current_intervened = current_clean
                    else:
                        assert current_clean is not None
                        assert current_intervened is not None
                        current_clean, _ = self._forward_loops(
                            loops=1,
                            inputs_embeds=current_clean,
                            attention_mask=attention_mask,
                        )
                        current_intervened, intervened_logits = self._forward_loops(
                            loops=1,
                            inputs_embeds=current_intervened,
                            attention_mask=attention_mask,
                        )
                    current_intervened = self._perturb(
                        current_intervened,
                        loop=loop,
                        position=position,
                        call_index=loop,
                    )
                assert current_clean is not None
                assert current_intervened is not None
                clean_hidden = current_clean
                intervened_hidden = current_intervened
                intervened_logits = self.model.lm_head(intervened_hidden[:, -1:, :])

        new_steps = getattr(getattr(self.model, "config", None), "total_ut_steps", None)
        new_inner_steps = getattr(getattr(getattr(self.model, "model", None), "config", None), "total_ut_steps", None)
        self.total_ut_steps_restored = (old_steps == new_steps) and (old_inner_steps == new_inner_steps)
        hidden_delta = float((intervened_hidden.float() - clean_hidden.float()).pow(2).mean().sqrt().detach().cpu().item())
        full_clean_delta = float((clean_hidden.float() - full_hidden.float()).abs().max().detach().cpu().item())
        return {
            "mechanism": "latent_loop_boundary_fork",
            "intervention_mode": self.intervention_mode,
            "use_cache": False,
            "device": str(device),
            "clean_hidden": clean_hidden,
            "intervened_hidden": intervened_hidden,
            "clean_logits": clean_logits,
            "intervened_logits": intervened_logits,
            "full_logits": full_logits,
            "records": self.records,
            "nan_or_inf": self.nan_or_inf,
            "hidden_delta_rms": hidden_delta,
            "zero_alpha_full_clean_max_abs_delta": full_clean_delta,
            "total_ut_steps_restored": self.total_ut_steps_restored,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    def diagnostics(self) -> dict[str, Any]:
        per_loop: dict[str, list[float]] = {}
        for record in self.records:
            per_loop.setdefault(str(record["loop"]), []).append(float(record["rms_fraction"]))
        return {
            "mechanism": "latent_loop_boundary_fork",
            "intervention_mode": self.intervention_mode,
            "alpha": self.alpha,
            "use_cache": self.use_cache,
            "records": self.records,
            "nan_or_inf": self.nan_or_inf,
            "total_ut_steps_restored": self.total_ut_steps_restored,
            "activation_rms_change": (
                sum(float(r["rms_fraction"]) for r in self.records) / max(len(self.records), 1)
            ),
            "per_loop_activation_rms_change": {
                loop: sum(vals) / max(len(vals), 1) for loop, vals in sorted(per_loop.items())
            },
        }


class _LoopLayerCapture:
    def __init__(self) -> None:
        self.inter: dict[int, list[torch.Tensor]] = {idx: [] for idx in LAYER_TO_CAPTURE_INDEX.values()}
        self.boundary: list[torch.Tensor] = []

    def make_inter_hook(self, idx: int):
        def _hook(_module: Any, _inp: Any, output: Any) -> None:
            self.inter[idx].append(_output_tensor(output).detach())

        return _hook

    def boundary_hook(self, _module: Any, _inp: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            self.boundary = []
            return
        states = output[1]
        self.boundary = [h.detach() for h in states] if states is not None else []


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).to(device="cpu", dtype=torch.float32)
    m = attention_mask.squeeze(0).to(device="cpu", dtype=torch.float32).unsqueeze(-1)
    denom = m.sum(dim=0).clamp(min=1.0)
    return (h * m).sum(dim=0) / denom


@contextmanager
def _capture_hooks(model: Any, capture: _LoopLayerCapture) -> Iterator[None]:
    inner_model = getattr(model, "model", None)
    layers = getattr(inner_model, "layers", None)
    if inner_model is None or layers is None:
        raise RuntimeError("model does not expose model.layers for BG feature capture")
    handles = [inner_model.register_forward_hook(capture.boundary_hook)]
    for idx in LAYER_TO_CAPTURE_INDEX.values():
        handles.append(layers[idx].register_forward_hook(capture.make_inter_hook(idx)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_bg_features_with_model(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    device: torch.device | str | None = None,
    max_length: int = 1536,
) -> torch.Tensor:
    """Capture BG pooled features with an already loaded Ouro model."""
    if not str(text).strip():
        raise ValueError("text must be non-empty")
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=dev)
    capture = _LoopLayerCapture()
    with torch.inference_mode():
        with _capture_hooks(model, capture):
            model(**enc, use_cache=False)
    if len(capture.boundary) != NUM_LOOPS:
        raise RuntimeError(f"expected {NUM_LOOPS} boundary states, got {len(capture.boundary)}")
    per_layer = []
    for layer in TAP_LAYERS:
        states = capture.boundary if layer == 47 else capture.inter[LAYER_TO_CAPTURE_INDEX[layer]]
        if len(states) != NUM_LOOPS:
            raise RuntimeError(f"expected {NUM_LOOPS} states for layer {layer}, got {len(states)}")
        per_layer.append(torch.stack([_masked_mean_pool(states[idx], enc["attention_mask"]) for idx in range(NUM_LOOPS)]))
    return torch.stack(per_layer, dim=0).to(dtype=torch.float32)


def _decode_first_tokens(tokenizer: Any, text: str, tokens: int) -> str:
    ids = tokenizer(str(text or ""), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.numel() <= tokens:
        return str(text or "")
    return tokenizer.decode(ids[:tokens], skip_special_tokens=True)


def _basic_generation_metadata(tokenizer: Any, text: str, token_count: int, hit_max_tokens: bool, seconds: float) -> dict[str, Any]:
    words = str(text or "").split()
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return {
        "token_count": int(token_count),
        "hit_max_tokens": bool(hit_max_tokens),
        "empty_output": not bool(str(text or "").strip()),
        "repetition_rate": repeats / max(len(words) - 1, 1),
        "seconds": round(seconds, 3),
    }


def _generate_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_cache: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    new_ids = out[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    meta = _basic_generation_metadata(
        tokenizer,
        text,
        int(new_ids.numel()),
        int(new_ids.numel()) >= int(max_new_tokens),
        time.time() - started,
    )
    meta["text"] = text
    return out[0], meta


def measure_intervention_effect(
    model: Any,
    prompt: str,
    prefix_text: str,
    mechanism: str,
    steering_object: Any,
    post_intervention_tokens: int = 32,
    max_completion_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Generate baseline/intervened continuations and capture post-prefix BG features."""
    tokenizer = getattr(model, "tokenizer", None) or getattr(steering_object, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("measure_intervention_effect requires model.tokenizer or steering_object.tokenizer")
    continuation_prompt = f"{prompt.rstrip()}\n\nPartial attempt so far:\n{prefix_text.strip()}\n\nContinue from the partial attempt."
    use_cache = False
    baseline_ids, baseline_meta = _generate_ids(
        model,
        tokenizer,
        continuation_prompt,
        max_new_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        use_cache=use_cache,
    )
    if mechanism != "layer_hook_injection":
        raise RuntimeError("measure_intervention_effect currently supports generation only for layer_hook_injection")
    try:
        steering_object.apply(position=-1)
        intervened_ids, intervened_meta = _generate_ids(
            model,
            tokenizer,
            continuation_prompt,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
        )
    finally:
        steering_object.remove()

    baseline_post = str(prefix_text) + _decode_first_tokens(tokenizer, baseline_meta["text"], post_intervention_tokens)
    intervened_post = str(prefix_text) + _decode_first_tokens(tokenizer, intervened_meta["text"], post_intervention_tokens)
    feature_prompt = f"Prompt:\n{prompt}\n\nCandidate:\n"
    baseline_features = capture_bg_features_with_model(model, tokenizer, feature_prompt + baseline_post)
    intervened_features = capture_bg_features_with_model(model, tokenizer, feature_prompt + intervened_post)
    diag = steering_object.diagnostics()
    return {
        "mechanism": mechanism,
        "baseline_post_features": baseline_features,
        "intervened_post_features": intervened_features,
        "baseline_completion": baseline_meta["text"],
        "intervened_completion": intervened_meta["text"],
        "baseline_completion_metadata": baseline_meta,
        "intervened_completion_metadata": intervened_meta,
        "activation_rms_change": diag.get("activation_rms_change", 0.0),
        "per_loop_activation_rms_change": diag.get("per_loop_activation_rms_change", {}),
        "cache_intervention_mode": "disabled",
        "baseline_ids": baseline_ids.detach().cpu(),
        "intervened_ids": intervened_ids.detach().cpu(),
    }
