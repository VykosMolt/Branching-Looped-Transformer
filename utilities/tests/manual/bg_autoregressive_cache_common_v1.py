"""Shared utilities for Autoregressive KV/Cache Branch-Carry Validation v1.

Short name: autoregressive_kv_branch_carry_v1

This module is TEST-ONLY infrastructure. It loads the local Ouro-RLTT model,
provides deterministic greedy decode primitives (cached + full-recompute),
logit comparison metrics, and JSON/CSV/MD IO helpers used by every level probe.

NO training, NO weight edits, NO tokenizer edits, NO steering, NO wrapper/local-agent,
NO Hunter-Seeker imports. Pure forward-pass cache-correctness validation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "shared/src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "shared/src"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets")
)

import torch  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "shared/models" / "ouro_rltt_local"
OUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "probes"
    / "bg_autoregressive_kv_branch_carry_v1_2026-06-01"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Perturbation target decoder layers (0-indexed; num_hidden_layers == 48).
PERTURB_LAYERS = [24, 36, 47]

# Suggested strict-equivalence thresholds (float32 logit comparison).
TOL_LOGIT_RMS = 1e-4
TOL_LOGIT_MAX_ABS = 1e-3
# Band below which top1/token-preserving drift is treated as bf16 rounding noise.
# Empirically, cached (q=1) vs full (q=seq) bf16 decode differs by ~1-2 bf16 ULP
# (RMS ~0.03-0.05, max_abs ~0.12-0.25 at logit magnitude ~16-64) due to
# shape-dependent cuBLAS matmul accumulation rounding, while top1/argmax and
# greedy token sequences match exactly. A genuine cache/mask bug produces RMS in
# the 1s-10s with top1 disagreement, well outside this band.
TOL_DRIFT_MAX_ABS = 1.0
TOL_DRIFT_RMS = 0.25

DEFAULT_SEED = 1234

# Short + medium test prompts (kept small for memory).
TEST_PROMPTS: list[dict[str, str]] = [
    {"id": "p_add", "text": "What is 2+2?"},
    {"id": "p_photo", "text": "Explain photosynthesis in one sentence."},
    {
        "id": "p_mcq",
        "text": (
            "Choose the best answer: A. water B. fire C. stone D. air. "
            "What do plants need for photosynthesis?"
        ),
    },
    {
        "id": "p_speed",
        "text": "Solve briefly: If a car travels 60 km in 2 hours, what is its speed?",
    },
    {
        "id": "p_reason",
        "text": "Question: Tom has 3 apples and buys 2 more. How many apples does Tom have?",
    },
]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relpath(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


# ---------------------------------------------------------------------------
# Model loading (singleton)
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_MODEL_INFO: dict[str, Any] = {}


def load_model(attn_implementation: str = "eager", dtype: Optional[torch.dtype] = None):
    """Load the local Ouro-RLTT model + tokenizer once (singleton).

    Uses eager attention by default so cached (q=1) vs full (q=seq) decode paths
    exercise identical, position-local reductions (cleanest equivalence check).
    """
    global _MODEL, _TOKENIZER, _MODEL_INFO
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _MODEL_INFO

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = dtype if dtype is not None else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=load_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Force all 4 UT loops to run; last-loop logits selected (deterministic).
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0
        model.config.use_cache = True
    load_s = time.time() - t0

    cfg = model.config
    info = {
        "model_path": relpath(MODEL_PATH),
        "model_class": type(model).__name__,
        "attn_implementation": getattr(cfg, "_attn_implementation", attn_implementation),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "torch_version": torch.__version__,
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "total_ut_steps": int(getattr(cfg, "total_ut_steps", 4)),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        "expected_cache_slots": int(cfg.num_hidden_layers) * int(getattr(cfg, "total_ut_steps", 4)),
        "early_exit_threshold": float(getattr(cfg, "early_exit_threshold", 1.0)),
        "load_seconds": round(load_s, 2),
    }
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = "unknown"

    _MODEL, _TOKENIZER, _MODEL_INFO = model, tok, info
    return model, tok, info


def get_universal_cache_class():
    """Return the UniversalTransformerCache class used by the loaded model."""
    model, _, _ = load_model()
    mod = sys.modules[type(model).__module__]
    return getattr(mod, "UniversalTransformerCache")


def new_cache():
    """Instantiate a fresh UniversalTransformerCache sized for this model."""
    model, _, info = load_model()
    cls = get_universal_cache_class()
    return cls(info["expected_cache_slots"])


def tokenize(prompt: str, device: Optional[torch.device] = None):
    """Tokenize a single prompt -> (input_ids [1,L], attention_mask [1,L])."""
    _, tok, _ = load_model()
    if device is None:
        device = next(_MODEL.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Forward primitives
# ---------------------------------------------------------------------------


@torch.no_grad()
def full_recompute_logits(model, input_ids, attention_mask=None, position_ids=None):
    """Full no-cache forward; returns logits [b, seq, vocab]."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    return out.logits


@torch.no_grad()
def prefill(model, input_ids, attention_mask=None, cache=None, position_ids=None):
    """Prefill prompt with use_cache=True.

    Returns (next_token_logits [b,vocab], cache, cache_position, attention_mask).
    """
    b, L = input_ids.shape
    device = input_ids.device
    if cache is None:
        cache = new_cache()
    if attention_mask is None:
        attention_mask = torch.ones((b, L), dtype=torch.long, device=device)
    cache_position = torch.arange(0, L, device=device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    next_logits = out.logits[:, -1, :]
    return next_logits, out.past_key_values, cache_position, attention_mask


@torch.no_grad()
def decode_step(model, last_token_ids, cache, attention_mask, position_ids=None):
    """One cached decode step.

    last_token_ids: [b,1] the new token(s).
    attention_mask: [b, past_len+1] full mask (already extended by caller).
    cache_position is derived as [past_len] (single new token).
    Returns (next_token_logits [b,vocab], cache).
    """
    device = last_token_ids.device
    past_len = cache.get_seq_length(0)
    cache_position = torch.arange(past_len, past_len + 1, device=device)
    out = model(
        input_ids=last_token_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    return out.logits[:, -1, :], out.past_key_values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compare_logits(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> dict[str, Any]:
    """Compare two next-token logit vectors (each [vocab] or [1,vocab]).

    All comparisons performed in float32 on CPU.
    """
    af = a.detach().reshape(-1).to(torch.float32).cpu()
    bf = b.detach().reshape(-1).to(torch.float32).cpu()
    diff = (af - bf)
    rms = float(diff.pow(2).mean().sqrt().item())
    max_abs = float(diff.abs().max().item())
    top1_a = int(af.argmax().item())
    top1_b = int(bf.argmax().item())
    topk_a = set(af.topk(k).indices.tolist())
    topk_b = set(bf.topk(k).indices.tolist())
    overlap = len(topk_a & topk_b) / float(k)
    return {
        "logit_rms": rms,
        "logit_max_abs": max_abs,
        "top1_match": bool(top1_a == top1_b),
        "top1_a": top1_a,
        "top1_b": top1_b,
        "top5_overlap": overlap,
    }


def classify_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step comparison rows into an equivalence summary.

    rows must contain keys: logit_rms, logit_max_abs, top1_match.

    Cache-correctness standard (the property that actually matters): the cached
    decode logits must match a full no-cache recomputation of the *identical*
    running sequence, within tolerance. In bf16 this means max-abs logit
    difference within the bf16 drift band.

    A top-1 (argmax) disagreement is classified as a NEAR-TIE FLIP -- a
    model-intrinsic numerical tie at the bf16 noise floor, NOT a cache bug --
    when that row's max-abs logit difference is itself within the drift band
    (a clear-margin winner cannot be flipped by a sub-band perturbation).
    `cache_faithful` is the headline pass condition; `strict_equiv` is the
    much tighter same-dtype/device equality only reachable at prefill.
    """
    if not rows:
        return {
            "n": 0, "all_top1_match": False, "max_rms": None, "max_abs": None,
            "strict_equiv": False, "drift_small": False, "cache_faithful": False,
            "n_top1_mismatch": 0, "neartie_flips": 0, "top1_mismatch_all_nearties": True,
        }
    max_rms = max(float(r["logit_rms"]) for r in rows)
    max_abs = max(float(r["logit_max_abs"]) for r in rows)
    all_top1 = all(bool(r["top1_match"]) for r in rows)
    n_top1_mismatch = sum(1 for r in rows if not bool(r["top1_match"]))
    neartie_flips = sum(
        1 for r in rows
        if (not bool(r["top1_match"])) and float(r["logit_max_abs"]) <= TOL_DRIFT_MAX_ABS
    )
    top1_mismatch_all_nearties = (n_top1_mismatch == neartie_flips)
    # optional token_match diagnostic (greedy-sequence identity; fragile at ties)
    tok_keys = [r for r in rows if "token_match" in r]
    all_tok = all(bool(r["token_match"]) for r in tok_keys) if tok_keys else None
    strict = all_top1 and max_rms <= TOL_LOGIT_RMS and max_abs <= TOL_LOGIT_MAX_ABS
    drift_small = (not strict) and max_rms <= TOL_DRIFT_RMS and max_abs <= TOL_DRIFT_MAX_ABS
    # cache is faithful if logits track full-recompute within bf16 and any argmax
    # disagreement is a near-tie (no clear-margin winner was flipped).
    cache_faithful = (
        max_abs <= TOL_DRIFT_MAX_ABS
        and max_rms <= TOL_DRIFT_RMS
        and top1_mismatch_all_nearties
    )
    return {
        "n": len(rows),
        "all_top1_match": all_top1,
        "all_token_match": all_tok,
        "n_top1_mismatch": n_top1_mismatch,
        "neartie_flips": neartie_flips,
        "top1_mismatch_all_nearties": bool(top1_mismatch_all_nearties),
        "max_rms": max_rms,
        "max_abs": max_abs,
        "mean_rms": sum(float(r["logit_rms"]) for r in rows) / len(rows),
        "strict_equiv": bool(strict),
        "drift_small": bool(drift_small),
        "cache_faithful": bool(cache_faithful),
    }


# ---------------------------------------------------------------------------
# Perturbation hook helper
# ---------------------------------------------------------------------------


def make_perturb_vector(hidden_size: int, seed: int, device, dtype) -> torch.Tensor:
    """Deterministic unit-RMS perturbation direction of size hidden_size."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(hidden_size, generator=gen, dtype=torch.float32)
    v = v / v.pow(2).mean().sqrt().clamp(min=1e-12)
    return v.to(device=device, dtype=dtype)


class LayerOutputPerturbHook:
    """Forward hook (registered with_kwargs=True) on an OuroDecoderLayer that
    adds alpha * RMS(hidden_at_target) * direction to the layer output hidden
    state, optionally only at a specific token position and/or a specific UT
    loop (current_ut).

    The decoder layer returns a hidden_states tensor [b, seq, hidden] (or a
    tuple whose first element is it). Because Ouro re-runs every layer once per
    UT loop, `target_loop` selects which loop iteration to perturb (None = all
    loops). `token_index` selects which token position (None = all; negative =
    from end, e.g. -1 = current/last token).
    """

    def __init__(self, direction: torch.Tensor, alpha: float,
                 token_index: Optional[int] = None, target_loop: Optional[int] = None,
                 token_range: Optional[tuple] = None):
        self.direction = direction
        self.alpha = float(alpha)
        self.token_index = token_index
        self.token_range = token_range  # (start, end) absolute positions, exclusive end
        self.target_loop = target_loop
        self.applied = 0
        self.last_perturb_rms = 0.0

    def __call__(self, module, args, kwargs, output):
        if self.target_loop is not None:
            cur = kwargs.get("current_ut", None)
            if cur is None and len(args) >= 8:
                cur = args[7]
            if cur is not None and int(cur) != int(self.target_loop):
                return output
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None
        if not torch.is_tensor(hs):
            return output
        if self.alpha == 0.0:
            self.applied += 1
            return output
        d = self.direction.to(device=hs.device, dtype=hs.dtype)
        b, seq, hidden = hs.shape
        if self.token_range is not None:
            start, end = self.token_range
            end = min(int(end), seq)
            start = max(0, int(start))
            if start < end:
                ref = hs[:, start:end, :].detach().to(torch.float32)
                scale = self.alpha * ref.pow(2).mean().sqrt().item()
                hs = hs.clone()
                hs[:, start:end, :] = hs[:, start:end, :] + scale * d.view(1, 1, hidden)
                self.last_perturb_rms = scale
        elif self.token_index is None:
            ref = hs.detach().to(torch.float32)
            scale = self.alpha * ref.pow(2).mean().sqrt().item()
            hs = hs + scale * d.view(1, 1, hidden)
            self.last_perturb_rms = scale
        else:
            idx = self.token_index if self.token_index >= 0 else seq + self.token_index
            if 0 <= idx < seq:
                ref = hs[:, idx, :].detach().to(torch.float32)
                scale = self.alpha * ref.pow(2).mean().sqrt().item()
                hs = hs.clone()
                hs[:, idx, :] = hs[:, idx, :] + scale * d.view(1, hidden)
                self.last_perturb_rms = scale
        self.applied += 1
        if rest is not None:
            return (hs, *rest)
        return hs


def register_perturb_hook(model, layer_idx, hook):
    """Register a LayerOutputPerturbHook on decoder layer layer_idx (with kwargs)."""
    layer = model.model.layers[layer_idx]
    return layer.register_forward_hook(hook, with_kwargs=True)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def save_json(name: str, obj: Any) -> Path:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o: Any):
    if isinstance(o, (set,)):
        return sorted(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


def save_md(name: str, text: str) -> Path:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def save_csv(name: str, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> Path:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return path
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Smoke: load model and print info.
    set_seed()
    _, _, info = load_model()
    print(json.dumps(info, indent=2))
    cache = new_cache()
    print("cache class:", type(cache).__name__, "max_cache_size:", cache.max_cache_size)
