"""Read-only transformer feature capture for BG candidate selection.

This module loads the local Ouro-RLTT model, captures hidden states with
temporary forward hooks, and pools them into the BG controller feature format:
[layers=3, loops=4, hidden=2048]. It does not train, steer, or modify model
weights/checkpoints.
"""
from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/ouro_rltt_local"
DEFAULT_MAX_LENGTH = 1536

TAP_LAYERS = (24, 36, 47)
LAYER_TO_CAPTURE_INDEX = {24: 23, 36: 35}
NUM_LOOPS = 4
HIDDEN_DIM = 2048

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))


def format_prompt_candidate(prompt: str, candidate: str) -> str:
    """Build the deterministic text used only for read-only BG feature capture."""
    return f"Prompt:\n{prompt}\n\nCandidate:\n{candidate}"


@dataclass
class CaptureSummary:
    boundary_loop_count: int
    layer_loop_counts: dict[int, int]
    shapes_by_layer: dict[str, list[list[int]]]
    dtype_by_layer: dict[str, list[str]]
    device_by_layer: dict[str, list[str]]


class _LoopLayerCapture:
    """Temporary hook sink for post-block 24/36 states and loop-boundary states."""

    def __init__(self) -> None:
        self.inter: dict[int, list[torch.Tensor]] = {idx: [] for idx in LAYER_TO_CAPTURE_INDEX.values()}
        self.boundary: list[torch.Tensor] = []

    def make_inter_hook(self, idx: int):
        def _hook(_module, _inp, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.inter[idx].append(tensor.detach())

        return _hook

    def boundary_hook(self, _module, _inp, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            self.boundary = []
            return
        states = output[1]
        if states is None:
            self.boundary = []
            return
        self.boundary = [h.detach() for h in states]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _dtype_from_arg(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    value = str(dtype).lower()
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if value in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if value in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype {dtype!r}; use auto, bfloat16, float16, or float32")


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).to(device="cpu", dtype=torch.float32)
    m = attention_mask.squeeze(0).to(device="cpu", dtype=torch.float32).unsqueeze(-1)
    denom = m.sum(dim=0).clamp(min=1.0)
    return (h * m).sum(dim=0) / denom


class BGTransformerFeatureExtractor:
    """Read-only feature extractor for live Ouro-RLTT BG controller inputs."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        force_all_loops: bool = True,
        allow_partial: bool = False,
    ) -> None:
        self.model_path = _resolve_path(model_path)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is not available")
        if not self.model_path.exists():
            raise RuntimeError(f"Ouro-RLTT model path does not exist: {self.model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dtype = _dtype_from_arg(dtype, self.device)
        self.force_all_loops = bool(force_all_loops)
        self.allow_partial = bool(allow_partial)
        self.loaded_at = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._original_total_ut_steps = getattr(self.model.config, "total_ut_steps", None)
        self._original_early_exit_threshold = getattr(self.model.config, "early_exit_threshold", None)
        self._original_inner_total_ut_steps = getattr(getattr(self.model, "model", None), "total_ut_steps", None)
        if self.force_all_loops:
            if hasattr(self.model.config, "total_ut_steps"):
                self.model.config.total_ut_steps = NUM_LOOPS
            if hasattr(self.model.config, "early_exit_threshold"):
                self.model.config.early_exit_threshold = 1.0
            inner_model = getattr(self.model, "model", None)
            if inner_model is not None and hasattr(inner_model, "total_ut_steps"):
                inner_model.total_ut_steps = NUM_LOOPS

    @contextmanager
    def _capture_hooks(self, capture: _LoopLayerCapture) -> Iterator[None]:
        inner_model = getattr(self.model, "model", None)
        layers = getattr(inner_model, "layers", None)
        if inner_model is None or layers is None:
            raise RuntimeError("loaded model does not expose model.model.layers for BG hook capture")
        handles = [inner_model.register_forward_hook(capture.boundary_hook)]
        for idx in LAYER_TO_CAPTURE_INDEX.values():
            if idx >= len(layers):
                raise RuntimeError(f"model has {len(layers)} layers; cannot capture layer index {idx}")
            handles.append(layers[idx].register_forward_hook(capture.make_inter_hook(idx)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _validate_capture(self, capture: _LoopLayerCapture) -> None:
        if len(capture.boundary) != NUM_LOOPS:
            if self.allow_partial:
                return
            raise RuntimeError(
                f"expected {NUM_LOOPS} loop-boundary states for layer 47, got {len(capture.boundary)}; "
                "BGController compatibility requires four loops"
            )
        for layer, idx in LAYER_TO_CAPTURE_INDEX.items():
            count = len(capture.inter[idx])
            if count != NUM_LOOPS:
                if self.allow_partial:
                    return
                raise RuntimeError(
                    f"expected {NUM_LOOPS} captured states for layer {layer} hook index {idx}, got {count}; "
                    "BGController compatibility requires four loops"
                )

    def encode_text_to_pooled_features(self, text: str, max_length: int = DEFAULT_MAX_LENGTH) -> torch.Tensor:
        """Return pooled BG features shaped [layers=3, loops=4, hidden=2048]."""
        if not str(text).strip():
            raise ValueError("text must be non-empty")
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=self.device)

        capture = _LoopLayerCapture()
        with torch.inference_mode():
            with self._capture_hooks(capture):
                self.model(**enc, use_cache=False)

        self._validate_capture(capture)
        per_layer: list[torch.Tensor] = []
        for layer in TAP_LAYERS:
            if layer == 47:
                loop_states = capture.boundary
            else:
                loop_states = capture.inter[LAYER_TO_CAPTURE_INDEX[layer]]
            if len(loop_states) < NUM_LOOPS:
                raise RuntimeError(f"layer {layer} has only {len(loop_states)} loops; cannot build BG features")
            pooled = [_masked_mean_pool(loop_states[i], enc["attention_mask"]) for i in range(NUM_LOOPS)]
            per_layer.append(torch.stack(pooled, dim=0))

        features = torch.stack(per_layer, dim=0).to(device="cpu", dtype=torch.float32)
        expected = (len(TAP_LAYERS), NUM_LOOPS, HIDDEN_DIM)
        actual = tuple(int(x) for x in features.shape)
        if actual != expected:
            raise ValueError(f"captured BG feature shape {actual}; expected {expected}")

        del capture
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return features

    def encode_prompt_candidate(
        self,
        prompt: str,
        candidate: str,
        domain_hint: str = "objective",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> torch.Tensor:
        """Build deterministic prompt/candidate text and return pooled BG features.

        The domain hint is accepted for caller symmetry with BGController, but it
        is deliberately not included in the model input text.
        """
        del domain_hint
        return self.encode_text_to_pooled_features(format_prompt_candidate(prompt, candidate), max_length=max_length)

    def inspect_capture_once(self, text: str, max_length: int = 128) -> tuple[torch.Tensor, CaptureSummary]:
        """Capture one text and return features plus raw capture metadata."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=self.device)

        capture = _LoopLayerCapture()
        with torch.inference_mode():
            with self._capture_hooks(capture):
                self.model(**enc, use_cache=False)
        self._validate_capture(capture)

        shapes_by_layer: dict[str, list[list[int]]] = {}
        dtype_by_layer: dict[str, list[str]] = {}
        device_by_layer: dict[str, list[str]] = {}
        for layer in TAP_LAYERS:
            states = capture.boundary if layer == 47 else capture.inter[LAYER_TO_CAPTURE_INDEX[layer]]
            shapes_by_layer[str(layer)] = [list(t.shape) for t in states]
            dtype_by_layer[str(layer)] = [str(t.dtype) for t in states]
            device_by_layer[str(layer)] = [str(t.device) for t in states]

        per_layer = []
        for layer in TAP_LAYERS:
            states = capture.boundary if layer == 47 else capture.inter[LAYER_TO_CAPTURE_INDEX[layer]]
            per_layer.append(torch.stack([_masked_mean_pool(states[i], enc["attention_mask"]) for i in range(NUM_LOOPS)]))
        features = torch.stack(per_layer, dim=0).to(device="cpu", dtype=torch.float32)
        summary = CaptureSummary(
            boundary_loop_count=len(capture.boundary),
            layer_loop_counts={layer: len(capture.inter[idx]) for layer, idx in LAYER_TO_CAPTURE_INDEX.items()},
            shapes_by_layer=shapes_by_layer,
            dtype_by_layer=dtype_by_layer,
            device_by_layer=device_by_layer,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return features, summary

    def cleanup(self) -> None:
        """Release model resources and restore in-memory config fields before deletion."""
        if hasattr(self, "model"):
            if self._original_total_ut_steps is not None and hasattr(self.model.config, "total_ut_steps"):
                self.model.config.total_ut_steps = self._original_total_ut_steps
            if self._original_early_exit_threshold is not None and hasattr(self.model.config, "early_exit_threshold"):
                self.model.config.early_exit_threshold = self._original_early_exit_threshold
            inner_model = getattr(self.model, "model", None)
            if self._original_inner_total_ut_steps is not None and inner_model is not None:
                inner_model.total_ut_steps = self._original_inner_total_ut_steps
            self.model.to("cpu")
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "BGTransformerFeatureExtractor":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.cleanup()
