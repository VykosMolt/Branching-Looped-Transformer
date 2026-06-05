"""Loop-state geometry probe on HH-RLHF: Thinking vs RLTT.

Implements Experiments 1A (intrinsic geometry) and 1B (cross-backbone
geometry: Thinking vs RLTT) per handoff_claude_code_2026-05-14.md sections
6 and 7.

Run shape:
  Pass 1 - load Thinking backbone in bf16, capture per-token loop boundary
           states for 200 HH-RLHF (chosen, rejected) test pairs, save fp32
           states + masks + final-norm gamma to disk, free the model.
  Pass 2 - same for the RLTT backbone.
  Pass 3 - load both saved .pt files plus the published pairwise_epoch2 head;
           compute intrinsic geometry per backbone, cross-backbone geometry,
           score agreement, and the conditional readout-direction
           decomposition. Emit JSON + four verdicts.

Numerical contract:
  - Backbones are loaded in bf16 (12 GB VRAM cannot hold a fp32 2.6B Ouro,
    confirmed with the user). The captured boundary states are cast to fp32
    immediately and every downstream computation - geometry, head scoring,
    autograd readout - runs in fp32. This matches the existing probe stack
    (e.g. probe_hh_norm_mechanism_centered.py) and the GRU head's training
    dtype.
  - Sampling, sub-selection, and any stochastic step use the same seed
    (default 42).

Outputs land under artifacts/reports/evaluator/ (project convention; the
handoff's runs/ wording is honored as a JSON note).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator, validate_hook_output

NUM_UT_LOOPS = 4
DEFAULT_THINKING_PATH = "ByteDance/Ouro-2.6B-Thinking"
DEFAULT_RLTT_PATH = "models/ouro_rltt_local"
DEFAULT_TOKENIZER_PATH = "models/ouro_rltt_local"
DEFAULT_HEAD_PATH = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
DEFAULT_OUTPUT_DIR = "artifacts/reports/evaluator"

VERDICT_CONVERGED = "loops are converged on text"
VERDICT_HETEROGENEOUS = "loops are mildly heterogeneous"
VERDICT_ORTHOGONAL = "loops are nearly orthogonal"

GAMMA_CONFIRMED = "gamma attractor confirmed"
GAMMA_NOT_CONFIRMED = "gamma attractor not confirmed"

W1 = "W1: RLTT did not move loop-boundary geometry"
W2 = "W2: RLTT reshaped head-invisible subspace"
W3 = "W3: RLTT moved both states and scores - re-baseline"
W_INCONCLUSIVE = "intermediate / inconclusive"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--thinking-path", default=DEFAULT_THINKING_PATH)
    p.add_argument("--rltt-path", default=DEFAULT_RLTT_PATH)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH,
                   help="Single tokenizer used for both backbones (Ouro tokenizers are identical)")
    p.add_argument("--head-checkpoint", default=DEFAULT_HEAD_PATH)
    p.add_argument("--dataset", default="Anthropic/hh-rlhf")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--readout-grad-pairs", type=int, default=50,
                   help="Number of example pairs to average gradient over for readout direction")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--skip-capture", action="store_true",
                   help="Skip both capture passes; reuse existing .pt files in --output-dir")
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Capture machinery
# -----------------------------------------------------------------------------

class BoundaryCapture:
    """Hook on model.model that grabs the per-loop boundary list at output[1]."""

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.validated = False

    def hook(self, module, inputs, output) -> None:
        hidden_states = output[1]
        if not self.validated:
            validate_hook_output(hidden_states)
            self.validated = True
        self.states = [h.detach() for h in hidden_states]

    def clear(self) -> None:
        self.states = []


@dataclass
class ExamplePack:
    """Per-example captured tensors for one backbone."""
    chosen_states: List[torch.Tensor]   # NUM_UT_LOOPS tensors of [seq_c, hidden]
    chosen_mask: torch.Tensor           # [seq_c]
    rejected_states: List[torch.Tensor]  # NUM_UT_LOOPS tensors of [seq_r, hidden]
    rejected_mask: torch.Tensor          # [seq_r]


def capture_backbone(
    model_path: str,
    tokenizer_path: str,
    examples: List[Dict[str, str]],
    max_length: int,
    device: torch.device,
    early_exit_threshold: float,
    report_every: int,
    label: str,
) -> Tuple[List[ExamplePack], torch.Tensor, float]:
    """Load backbone, capture states for all examples, return packs + final-norm gamma + eps."""
    print(f"\n[{label}] loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{label}] loading model from {model_path} (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = early_exit_threshold
    print(f"[{label}] model ready in {time.time() - t0:.1f}s")

    eps = float(getattr(model.model.norm, "variance_epsilon", 1e-6))
    gamma = model.model.norm.weight.detach().to(device="cpu", dtype=torch.float32).clone()

    capture = BoundaryCapture()
    handle = model.model.register_forward_hook(capture.hook)

    packs: List[ExamplePack] = []
    n = len(examples)
    t_start = time.time()
    for idx, ex in enumerate(examples):
        tc = tokenizer(ex["chosen"], return_tensors="pt", truncation=True, max_length=max_length).to(device)
        tr = tokenizer(ex["rejected"], return_tensors="pt", truncation=True, max_length=max_length).to(device)

        capture.clear()
        with torch.no_grad():
            model(**tc, use_cache=False)
        c_states = [h.squeeze(0).to(device="cpu", dtype=torch.float32).clone() for h in capture.states]
        c_mask = tc["attention_mask"].squeeze(0).to(device="cpu", dtype=torch.float32).clone()

        capture.clear()
        with torch.no_grad():
            model(**tr, use_cache=False)
        r_states = [h.squeeze(0).to(device="cpu", dtype=torch.float32).clone() for h in capture.states]
        r_mask = tr["attention_mask"].squeeze(0).to(device="cpu", dtype=torch.float32).clone()

        packs.append(ExamplePack(
            chosen_states=c_states,
            chosen_mask=c_mask,
            rejected_states=r_states,
            rejected_mask=r_mask,
        ))

        if (idx + 1) % report_every == 0 or idx == n - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (n - (idx + 1)) / rate if rate > 0 else float("nan")
            print(f"[{label}] {idx + 1:>3}/{n}  elapsed={elapsed:5.1f}s  rate={rate:4.2f}/s  eta={eta:5.1f}s")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return packs, gamma, eps


def save_packs(path: Path, packs: List[ExamplePack], gamma: torch.Tensor, eps: float, meta: Dict[str, object]) -> None:
    payload = {
        "meta": meta,
        "gamma": gamma,
        "eps": eps,
        "packs": [
            {
                "chosen_states": p.chosen_states,
                "chosen_mask": p.chosen_mask,
                "rejected_states": p.rejected_states,
                "rejected_mask": p.rejected_mask,
            }
            for p in packs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[save] wrote {path}  ({size_mb:.1f} MB, {len(packs)} examples)")


def load_packs(path: Path) -> Tuple[List[ExamplePack], torch.Tensor, float, Dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    packs = [
        ExamplePack(
            chosen_states=d["chosen_states"],
            chosen_mask=d["chosen_mask"],
            rejected_states=d["rejected_states"],
            rejected_mask=d["rejected_mask"],
        )
        for d in payload["packs"]
    ]
    return packs, payload["gamma"], float(payload["eps"]), payload.get("meta", {})


# -----------------------------------------------------------------------------
# Pooling and geometry helpers
# -----------------------------------------------------------------------------

def mean_pool(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over mask-valid tokens. states: [seq, hidden], mask: [seq]."""
    m = mask.to(dtype=states.dtype).unsqueeze(-1)
    summed = (states * m).sum(dim=0)
    denom = m.sum(dim=0).clamp(min=1.0)
    return summed / denom


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(dtype=torch.float32).flatten()
    b = b.to(dtype=torch.float32).flatten()
    na = a.norm()
    nb = b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def effective_dim(matrix: torch.Tensor, top_k: int = 32) -> Dict[str, float]:
    """Top-k cumulative explained variance ratio of a [N, hidden] matrix."""
    x = matrix.to(dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return {"top_k": top_k, "cum_ratio": float("nan")}
    try:
        _, s, _ = torch.linalg.svd(x, full_matrices=False)
    except RuntimeError:
        return {"top_k": top_k, "cum_ratio": float("nan")}
    eig = (s ** 2).cpu().numpy()
    total = eig.sum()
    if total <= 0:
        return {"top_k": top_k, "cum_ratio": float("nan")}
    k = min(top_k, eig.size)
    cum = float(eig[:k].sum() / total)
    return {"top_k": k, "cum_ratio": cum, "total_components": int(eig.size)}


def apply_norm_k(state: torch.Tensor, gamma: torch.Tensor, eps: float, k: int) -> torch.Tensor:
    """Apply OuroRMSNorm K times (learned norm). state: [seq, hidden] fp32."""
    y = state
    g = gamma.to(device=y.device, dtype=torch.float32)
    for _ in range(k):
        # OuroRMSNorm: y * gamma / rms(y) where rms = sqrt(mean(y^2) + eps)
        var = y.pow(2).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(var + eps) * g
    return y


# -----------------------------------------------------------------------------
# Intrinsic geometry (1A)
# -----------------------------------------------------------------------------

def intrinsic_geometry(packs: List[ExamplePack], gamma: torch.Tensor, eps: float, label: str) -> Dict:
    print(f"\n[1A] intrinsic geometry: {label}")
    n = len(packs)

    # Mean-pool every loop, both chosen and rejected.
    pooled_c = [[mean_pool(p.chosen_states[i], p.chosen_mask) for i in range(NUM_UT_LOOPS)] for p in packs]
    pooled_r = [[mean_pool(p.rejected_states[i], p.rejected_mask) for i in range(NUM_UT_LOOPS)] for p in packs]

    # (a) pairwise cos(h_i, h_j) for both chosen and rejected, pooled.
    pair_cos: Dict[str, Dict[str, float]] = {}
    for i in range(NUM_UT_LOOPS):
        for j in range(i, NUM_UT_LOOPS):
            vals = []
            for k in range(n):
                vals.append(cosine(pooled_c[k][i], pooled_c[k][j]))
                vals.append(cosine(pooled_r[k][i], pooled_r[k][j]))
            pair_cos[f"L{i+1}_L{j+1}"] = stats(vals)

    # (b) L2 norm per loop (chosen and rejected pooled together).
    norm_per_loop: Dict[str, Dict[str, float]] = {}
    for i in range(NUM_UT_LOOPS):
        vals = []
        for k in range(n):
            vals.append(float(pooled_c[k][i].norm()))
            vals.append(float(pooled_r[k][i].norm()))
        norm_per_loop[f"L{i+1}"] = stats(vals)

    # (c) effective dim per loop.
    eff_dim_per_loop: Dict[str, Dict] = {}
    for i in range(NUM_UT_LOOPS):
        mat = torch.stack([pooled_c[k][i] for k in range(n)] + [pooled_r[k][i] for k in range(n)], dim=0)
        eff_dim_per_loop[f"L{i+1}"] = effective_dim(mat, top_k=32)

    # (d) diff-vector cross-loop cos(d_i, d_j).
    diffs = [[pooled_c[k][i] - pooled_r[k][i] for i in range(NUM_UT_LOOPS)] for k in range(n)]
    diff_cos: Dict[str, Dict[str, float]] = {}
    for i in range(NUM_UT_LOOPS):
        for j in range(i, NUM_UT_LOOPS):
            vals = [cosine(diffs[k][i], diffs[k][j]) for k in range(n)]
            diff_cos[f"d{i+1}_d{j+1}"] = stats(vals)

    # Within-loop chosen-vs-rejected baseline (vs the diff cosines context).
    within_loop_cr_cos: Dict[str, Dict[str, float]] = {}
    for i in range(NUM_UT_LOOPS):
        vals = [cosine(pooled_c[k][i], pooled_r[k][i]) for k in range(n)]
        within_loop_cr_cos[f"L{i+1}"] = stats(vals)

    # (e) repeat (a) and (d) after applying model.model.norm K times (per-token first, then pool).
    norm_k_sweep: Dict[str, Dict] = {}
    for kk in (0, 1, 2, 4, 8):
        # Apply norm K times at the per-token level, then mean-pool.
        c_kk = [[mean_pool(apply_norm_k(p.chosen_states[i], gamma, eps, kk), p.chosen_mask) for i in range(NUM_UT_LOOPS)] for p in packs]
        r_kk = [[mean_pool(apply_norm_k(p.rejected_states[i], gamma, eps, kk), p.rejected_mask) for i in range(NUM_UT_LOOPS)] for p in packs]

        pair_kk: Dict[str, Dict[str, float]] = {}
        for i in range(NUM_UT_LOOPS):
            for j in range(i, NUM_UT_LOOPS):
                vals = []
                for k in range(n):
                    vals.append(cosine(c_kk[k][i], c_kk[k][j]))
                    vals.append(cosine(r_kk[k][i], r_kk[k][j]))
                pair_kk[f"L{i+1}_L{j+1}"] = stats(vals)

        diffs_kk = [[c_kk[k][i] - r_kk[k][i] for i in range(NUM_UT_LOOPS)] for k in range(n)]
        diff_kk: Dict[str, Dict[str, float]] = {}
        for i in range(NUM_UT_LOOPS):
            for j in range(i, NUM_UT_LOOPS):
                vals = [cosine(diffs_kk[k][i], diffs_kk[k][j]) for k in range(n)]
                diff_kk[f"d{i+1}_d{j+1}"] = stats(vals)

        norm_k_sweep[f"k{kk}"] = {
            "pair_cos": pair_kk,
            "diff_cos": diff_kk,
        }

    # Aggregate cross-loop (off-diagonal) mean for the verdict.
    off_diag = []
    for key, st in pair_cos.items():
        a, b = key.split("_")
        if a != b:
            off_diag.append(st["mean"])
    mean_cross_loop = float(np.mean(off_diag)) if off_diag else float("nan")

    if mean_cross_loop > 0.85:
        verdict = VERDICT_CONVERGED
    elif mean_cross_loop >= 0.4:
        verdict = VERDICT_HETEROGENEOUS
    else:
        verdict = VERDICT_ORTHOGONAL

    # Gamma-attractor verdict at k=8.
    k8 = norm_k_sweep["k8"]["pair_cos"]
    k8_off = []
    for key, st in k8.items():
        a, b = key.split("_")
        if a != b:
            k8_off.append(st["mean"])
    k8_min_off = float(np.min(k8_off)) if k8_off else float("nan")
    gamma_verdict = GAMMA_CONFIRMED if (math.isfinite(k8_min_off) and k8_min_off > 0.95) else GAMMA_NOT_CONFIRMED

    return {
        "label": label,
        "n_examples": n,
        "pair_cos_pooled": pair_cos,
        "within_loop_chosen_vs_rejected_cos": within_loop_cr_cos,
        "norm_per_loop": norm_per_loop,
        "effective_dim_per_loop": eff_dim_per_loop,
        "diff_vector_cross_loop_cos": diff_cos,
        "norm_k_sweep": norm_k_sweep,
        "summary": {
            "mean_cross_loop_cosine_pooled": mean_cross_loop,
            "k8_min_off_diag_cosine": k8_min_off,
            "intrinsic_verdict": verdict,
            "gamma_attractor_verdict": gamma_verdict,
        },
    }


# -----------------------------------------------------------------------------
# Cross-backbone geometry + score agreement (1B)
# -----------------------------------------------------------------------------

def head_score_per_pack(head: PairwiseEvaluator, packs: List[ExamplePack], device: torch.device, swap: bool = False) -> List[float]:
    """Run the published head per-example. swap=False scores (chosen, rejected); swap=True flips."""
    head.eval()
    scores: List[float] = []
    with torch.no_grad():
        for p in packs:
            cs = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in p.chosen_states]
            rs = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in p.rejected_states]
            cm = p.chosen_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
            rm = p.rejected_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
            if swap:
                out = head(rs, rm, cs, cm)
            else:
                out = head(cs, cm, rs, rm)
            v = float(out.view(-1).item())
            scores.append(v if math.isfinite(v) else float("nan"))
    return scores


def cross_backbone_geometry(packs_t: List[ExamplePack], packs_r: List[ExamplePack]) -> Dict:
    print("\n[1B] cross-backbone geometry: per-loop cos(h_T, h_R)")
    assert len(packs_t) == len(packs_r), "pack count mismatch between backbones"
    n = len(packs_t)

    per_loop_cos_chosen: Dict[str, Dict[str, float]] = {}
    per_loop_cos_rejected: Dict[str, Dict[str, float]] = {}
    per_loop_cos_combined: Dict[str, Dict[str, float]] = {}
    for i in range(NUM_UT_LOOPS):
        c_vals = []
        r_vals = []
        for k in range(n):
            ct = mean_pool(packs_t[k].chosen_states[i], packs_t[k].chosen_mask)
            cr = mean_pool(packs_r[k].chosen_states[i], packs_r[k].chosen_mask)
            rt = mean_pool(packs_t[k].rejected_states[i], packs_t[k].rejected_mask)
            rr = mean_pool(packs_r[k].rejected_states[i], packs_r[k].rejected_mask)
            c_vals.append(cosine(ct, cr))
            r_vals.append(cosine(rt, rr))
        per_loop_cos_chosen[f"L{i+1}"] = stats(c_vals)
        per_loop_cos_rejected[f"L{i+1}"] = stats(r_vals)
        per_loop_cos_combined[f"L{i+1}"] = stats(c_vals + r_vals)

    means = [per_loop_cos_combined[f"L{i+1}"]["mean"] for i in range(NUM_UT_LOOPS)]
    valid_means = [m for m in means if math.isfinite(m)]
    overall_mean = float(np.mean(valid_means)) if valid_means else float("nan")
    overall_min = float(np.min(valid_means)) if valid_means else float("nan")

    return {
        "n_examples": n,
        "per_loop_cos_chosen": per_loop_cos_chosen,
        "per_loop_cos_rejected": per_loop_cos_rejected,
        "per_loop_cos_combined": per_loop_cos_combined,
        "summary": {
            "mean_cos_T_vs_R": overall_mean,
            "min_cos_T_vs_R": overall_min,
            "mean_cos_per_loop": {f"L{i+1}": means[i] for i in range(NUM_UT_LOOPS)},
        },
    }


def score_agreement(head: PairwiseEvaluator, packs_t: List[ExamplePack], packs_r: List[ExamplePack], device: torch.device) -> Dict:
    print("\n[1B] head score agreement Thinking vs RLTT")
    s_T = head_score_per_pack(head, packs_t, device)
    s_R = head_score_per_pack(head, packs_r, device)
    arr_T = np.asarray(s_T, dtype=np.float64)
    arr_R = np.asarray(s_R, dtype=np.float64)
    valid = np.isfinite(arr_T) & np.isfinite(arr_R)
    arr_T = arr_T[valid]
    arr_R = arr_R[valid]
    if arr_T.size < 2:
        pearson = float("nan")
    else:
        if np.std(arr_T) < 1e-12 or np.std(arr_R) < 1e-12:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(arr_T, arr_R)[0, 1])
    abs_diff = np.abs(arr_T - arr_R)
    decision_match = (np.sign(arr_T) == np.sign(arr_R))
    return {
        "n_valid": int(arr_T.size),
        "pearson_score_T_vs_R": pearson,
        "mean_abs_delta_score": float(abs_diff.mean()) if abs_diff.size else float("nan"),
        "median_abs_delta_score": float(np.median(abs_diff)) if abs_diff.size else float("nan"),
        "decision_agreement_rate": float(decision_match.mean()) if decision_match.size else float("nan"),
        "canonical_acc_thinking": float(np.mean(arr_T > 0)) if arr_T.size else float("nan"),
        "canonical_acc_rltt": float(np.mean(arr_R > 0)) if arr_R.size else float("nan"),
        "score_T_mean": float(arr_T.mean()) if arr_T.size else float("nan"),
        "score_T_std": float(arr_T.std()) if arr_T.size else float("nan"),
        "score_R_mean": float(arr_R.mean()) if arr_R.size else float("nan"),
        "score_R_std": float(arr_R.std()) if arr_R.size else float("nan"),
        "raw_scores_thinking": s_T,
        "raw_scores_rltt": s_R,
    }


def readout_direction_decomposition(
    head: PairwiseEvaluator,
    packs_t: List[ExamplePack],
    packs_r: List[ExamplePack],
    device: torch.device,
    n_grad_pairs: int,
) -> Dict:
    """Compute readout direction per loop via autograd, then decompose RLTT - Thinking deltas."""
    print(f"\n[1B] readout-direction decomposition (n_grad_pairs={n_grad_pairs})")
    # Use Thinking states as the basis for the readout direction (they are the
    # head's training distribution, by construction of pairwise_epoch2.pt).
    n_grad_pairs = min(n_grad_pairs, len(packs_t))

    # Accumulate per-loop direction in pooled (hidden,) space by mean-pooling
    # the per-token gradient over valid tokens, then averaging across pairs.
    hidden_dim = packs_t[0].chosen_states[0].shape[-1]
    accum = [torch.zeros(hidden_dim, dtype=torch.float64) for _ in range(NUM_UT_LOOPS)]
    n_used = 0

    for k in range(n_grad_pairs):
        p = packs_t[k]
        cs = [s.to(device=device, dtype=torch.float32).unsqueeze(0).clone().requires_grad_(True) for s in p.chosen_states]
        rs = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in p.rejected_states]
        cm = p.chosen_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
        rm = p.rejected_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
        head.zero_grad(set_to_none=True)
        out = head(cs, cm, rs, rm)
        score = out.sum()
        score.backward()
        for i in range(NUM_UT_LOOPS):
            g = cs[i].grad
            if g is None:
                continue
            g = g.squeeze(0).to(dtype=torch.float64)  # [seq, hidden]
            mask = cm.squeeze(0).to(dtype=torch.float64).unsqueeze(-1)
            denom = mask.sum().clamp(min=1.0)
            pooled_grad = (g * mask).sum(dim=0) / denom
            accum[i] += pooled_grad.cpu()
        n_used += 1

    head.zero_grad(set_to_none=True)
    direction_per_loop = [accum[i] / max(n_used, 1) for i in range(NUM_UT_LOOPS)]
    norms = [float(d.norm()) for d in direction_per_loop]

    per_loop_decomp: Dict[str, Dict[str, float]] = {}
    n = len(packs_t)
    for i in range(NUM_UT_LOOPS):
        d = direction_per_loop[i]
        nrm = d.norm().clamp(min=1e-12)
        u = d / nrm
        aligned_norms = []
        orth_norms = []
        ratios = []
        for k in range(n):
            ct = mean_pool(packs_t[k].chosen_states[i], packs_t[k].chosen_mask).to(dtype=torch.float64)
            cr = mean_pool(packs_r[k].chosen_states[i], packs_r[k].chosen_mask).to(dtype=torch.float64)
            delta = cr - ct  # RLTT - Thinking
            proj = float(delta @ u)
            aligned = u * proj
            orth = delta - aligned
            an = float(aligned.norm())
            on = float(orth.norm())
            aligned_norms.append(an)
            orth_norms.append(on)
            if an > 1e-12:
                ratios.append(on / an)
        per_loop_decomp[f"L{i+1}"] = {
            "readout_dir_norm": norms[i],
            "mean_aligned_norm": float(np.mean(aligned_norms)) if aligned_norms else float("nan"),
            "mean_orth_norm": float(np.mean(orth_norms)) if orth_norms else float("nan"),
            "mean_orth_over_aligned": float(np.mean(ratios)) if ratios else float("nan"),
            "median_orth_over_aligned": float(np.median(ratios)) if ratios else float("nan"),
        }

    ratios_overall = [v["mean_orth_over_aligned"] for v in per_loop_decomp.values() if math.isfinite(v["mean_orth_over_aligned"])]
    return {
        "n_grad_pairs": n_used,
        "per_loop": per_loop_decomp,
        "summary": {
            "min_orth_over_aligned": float(np.min(ratios_overall)) if ratios_overall else float("nan"),
            "max_orth_over_aligned": float(np.max(ratios_overall)) if ratios_overall else float("nan"),
            "mean_orth_over_aligned": float(np.mean(ratios_overall)) if ratios_overall else float("nan"),
        },
    }


# -----------------------------------------------------------------------------
# Verdict logic
# -----------------------------------------------------------------------------

def cross_backbone_verdict(geom: Dict, scores: Dict, decomp: Optional[Dict]) -> Dict:
    mean_cos = geom["summary"]["mean_cos_T_vs_R"]
    pearson = scores["pearson_score_T_vs_R"]

    if math.isfinite(mean_cos) and mean_cos > 0.99:
        v = W1
        rec = "train new head on Thinking activations"
    elif (
        math.isfinite(mean_cos) and mean_cos < 0.99
        and math.isfinite(pearson) and pearson > 0.99
        and decomp is not None
        and math.isfinite(decomp["summary"]["min_orth_over_aligned"])
        and decomp["summary"]["min_orth_over_aligned"] > 5.0
    ):
        v = W2
        rec = "train new head on RLTT activations"
    elif math.isfinite(mean_cos) and mean_cos < 0.99 and math.isfinite(pearson) and pearson < 0.99:
        v = W3
        rec = "STOP: do not start Experiment 2 yet; re-baseline first"
    else:
        v = W_INCONCLUSIVE
        rec = "verdict inconclusive; inspect numbers manually before Experiment 2"

    return {
        "cross_backbone_verdict": v,
        "experiment2_direction": rec,
        "evidence": {
            "mean_cos_T_vs_R": mean_cos,
            "pearson_score_T_vs_R": pearson,
            "min_orth_over_aligned": decomp["summary"]["min_orth_over_aligned"] if decomp else None,
        },
    }


# -----------------------------------------------------------------------------
# Pretty-print
# -----------------------------------------------------------------------------

def print_intrinsic(geom: Dict) -> None:
    label = geom["label"]
    s = geom["summary"]
    print(f"\n=== Intrinsic geometry ({label}) ===")
    print(f"  mean cross-loop cosine (pooled): {s['mean_cross_loop_cosine_pooled']:.4f}")
    print(f"  k8 min off-diag cosine:          {s['k8_min_off_diag_cosine']:.4f}")
    print(f"  intrinsic verdict:               {s['intrinsic_verdict']}")
    print(f"  gamma attractor verdict:         {s['gamma_attractor_verdict']}")
    print("  per-loop pair cosines (k0):")
    for key, st in geom["pair_cos_pooled"].items():
        print(f"    {key:>6}  mean={st['mean']:+.4f}  std={st['std']:.4f}  n={st['n']}")
    print("  per-loop L2 norm:")
    for key, st in geom["norm_per_loop"].items():
        print(f"    {key:>4}    mean={st['mean']:8.3f}  std={st['std']:7.3f}")
    print("  per-loop top-32 cumulative variance ratio:")
    for key, st in geom["effective_dim_per_loop"].items():
        print(f"    {key:>4}    cum_ratio={st['cum_ratio']:.4f}  total={st.get('total_components', '?')}")
    print("  diff-vector cross-loop cosines (k0):")
    for key, st in geom["diff_vector_cross_loop_cos"].items():
        print(f"    {key:>6}  mean={st['mean']:+.4f}  std={st['std']:.4f}")
    print("  K-norm pair-cos sweep (mean off-diag):")
    for kkey in ("k0", "k1", "k2", "k4", "k8"):
        pair = geom["norm_k_sweep"][kkey]["pair_cos"]
        offs = [v["mean"] for k, v in pair.items() if k.split("_")[0] != k.split("_")[1]]
        print(f"    {kkey}: mean_off_diag={float(np.mean(offs)):+.4f}  min_off_diag={float(np.min(offs)):+.4f}")


def print_cross(geom: Dict, scores: Dict, decomp: Optional[Dict], verdict: Dict) -> None:
    print("\n=== Cross-backbone (1B) ===")
    s = geom["summary"]
    print(f"  mean cos(h_T, h_R) overall: {s['mean_cos_T_vs_R']:.4f}")
    print(f"  min  cos(h_T, h_R) per loop: {s['min_cos_T_vs_R']:.4f}")
    print("  per-loop mean cos:")
    for key, val in s["mean_cos_per_loop"].items():
        print(f"    {key}: {val:+.4f}")
    print("  head score agreement:")
    print(f"    Pearson(score_T, score_R)    : {scores['pearson_score_T_vs_R']:.4f}")
    print(f"    mean |delta score|           : {scores['mean_abs_delta_score']:.4f}")
    print(f"    median |delta score|         : {scores['median_abs_delta_score']:.4f}")
    print(f"    decision agreement rate      : {scores['decision_agreement_rate']:.4f}")
    print(f"    canonical_acc Thinking       : {scores['canonical_acc_thinking']:.4f}")
    print(f"    canonical_acc RLTT           : {scores['canonical_acc_rltt']:.4f}")
    if decomp is not None:
        print("  readout-direction decomposition (per loop):")
        for key, val in decomp["per_loop"].items():
            print(f"    {key}: ||aligned||={val['mean_aligned_norm']:7.4f}  ||orth||={val['mean_orth_norm']:7.4f}  orth/aligned={val['mean_orth_over_aligned']:.3f}")
        print(f"    summary min orth/aligned: {decomp['summary']['min_orth_over_aligned']:.3f}")
    else:
        print("  readout-direction decomposition skipped (mean cos >= 0.99)")
    print(f"\n  CROSS-BACKBONE VERDICT: {verdict['cross_backbone_verdict']}")
    print(f"  EXPERIMENT 2 DIRECTION : {verdict['experiment2_direction']}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def select_examples(args: argparse.Namespace) -> List[Dict[str, str]]:
    print(f"Loading dataset {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)[: args.max_examples].tolist()
    indices.sort()
    print(f"Total {n_total} examples; sampled {len(indices)} with seed {args.seed}")
    return [{"chosen": ds[int(i)]["chosen"], "rejected": ds[int(i)]["rejected"]} for i in indices]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thinking_pt = out_dir / "hh_loop_states_200_thinking.pt"
    rltt_pt = out_dir / "hh_loop_states_200_rltt.pt"
    json_out = out_dir / "probe_loop_geometry_hh.json"

    examples = select_examples(args)

    # Pass 1: Thinking
    if args.skip_capture and thinking_pt.exists():
        print(f"\n[Thinking] reusing existing {thinking_pt}")
        packs_t, gamma_t, eps_t, _meta_t = load_packs(thinking_pt)
    else:
        packs_t, gamma_t, eps_t = capture_backbone(
            args.thinking_path, args.tokenizer_path, examples,
            args.max_length, device, args.early_exit_threshold,
            args.report_every, label="Thinking",
        )
        save_packs(thinking_pt, packs_t, gamma_t, eps_t, meta={
            "model_path": args.thinking_path,
            "tokenizer_path": args.tokenizer_path,
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "dtype": "fp32 states (model fwd in bf16)",
        })

    # Pass 2: RLTT
    if args.skip_capture and rltt_pt.exists():
        print(f"\n[RLTT] reusing existing {rltt_pt}")
        packs_r, gamma_r, eps_r, _meta_r = load_packs(rltt_pt)
    else:
        packs_r, gamma_r, eps_r = capture_backbone(
            args.rltt_path, args.tokenizer_path, examples,
            args.max_length, device, args.early_exit_threshold,
            args.report_every, label="RLTT",
        )
        save_packs(rltt_pt, packs_r, gamma_r, eps_r, meta={
            "model_path": args.rltt_path,
            "tokenizer_path": args.tokenizer_path,
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "dtype": "fp32 states (model fwd in bf16)",
        })

    # Pass 3: analysis
    geom_t = intrinsic_geometry(packs_t, gamma_t, eps_t, label="Thinking")
    geom_r = intrinsic_geometry(packs_r, gamma_r, eps_r, label="RLTT")
    cross = cross_backbone_geometry(packs_t, packs_r)

    print(f"\nLoading head from {args.head_checkpoint}")
    head = PairwiseEvaluator().to(device)
    ckpt = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(state_dict)
    head.eval()

    scores = score_agreement(head, packs_t, packs_r, device)

    decomp = None
    mean_cos = cross["summary"]["mean_cos_T_vs_R"]
    if math.isfinite(mean_cos) and mean_cos < 0.99:
        decomp = readout_direction_decomposition(head, packs_t, packs_r, device, args.readout_grad_pairs)
    else:
        print(f"\n[1B] mean cos={mean_cos:.4f} >= 0.99 -> skipping readout-direction decomposition (W1 path)")

    verdict = cross_backbone_verdict(cross, scores, decomp)

    print_intrinsic(geom_t)
    print_intrinsic(geom_r)
    print_cross(cross, scores, decomp, verdict)

    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "examples_evaluated": len(packs_t),
        "outputs": {
            "thinking_states_pt": str(thinking_pt),
            "rltt_states_pt": str(rltt_pt),
        },
        "intrinsic_thinking": geom_t,
        "intrinsic_rltt": geom_r,
        "cross_backbone_geometry": cross,
        "score_agreement": scores,
        "readout_direction_decomposition": decomp,
        "verdicts": {
            "intrinsic_thinking": geom_t["summary"]["intrinsic_verdict"],
            "intrinsic_rltt": geom_r["summary"]["intrinsic_verdict"],
            "gamma_attractor_thinking": geom_t["summary"]["gamma_attractor_verdict"],
            "gamma_attractor_rltt": geom_r["summary"]["gamma_attractor_verdict"],
            "cross_backbone": verdict["cross_backbone_verdict"],
            "experiment2_direction": verdict["experiment2_direction"],
            "evidence": verdict["evidence"],
        },
    }

    json_out.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nWrote {json_out}")
    print("\nFOUR VERDICTS:")
    print(f"  intrinsic-geometry (Thinking):    {geom_t['summary']['intrinsic_verdict']}")
    print(f"  intrinsic-geometry (RLTT):        {geom_r['summary']['intrinsic_verdict']}")
    print(f"  gamma-attractor (Thinking):       {geom_t['summary']['gamma_attractor_verdict']}")
    print(f"  gamma-attractor (RLTT):           {geom_r['summary']['gamma_attractor_verdict']}")
    print(f"  cross-backbone:                   {verdict['cross_backbone_verdict']}")
    print(f"  -> Experiment 2 direction:        {verdict['experiment2_direction']}")


if __name__ == "__main__":
    main()
