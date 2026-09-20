"""Core for Partial Cache Splice v2 — compute-saving branch-carry implementation.

Builds on autoregressive_kv_branch_carry_v1 (imports its common + helpers). The
central object is a TEST-ONLY suffix-recompute path:

  Key fact: the UniversalTransformerCache stores K/V but NOT the inter-layer
  residual stream. So to build a perturbed branch cache cheaply we ALSO capture
  the residual hidden state at the perturbation boundary (loop u, layer L output)
  during the *unperturbed* root prefill. For an additive boundary perturbation,
  H_boundary_perturbed = H_boundary + delta (no forward), and we then run ONLY the
  suffix (loop u layers L+1..47, then loops u+1..3), reusing the root's shared
  prefix slots. This avoids a full perturbed prompt prefill.

NO training, NO weight/tokenizer edits, NO steering, NO wrapper/local-agent,
NO Hunter-Seeker. Pure forward-pass cache construction with test-only orchestration.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H

# v2 output root
OUT_DIR = (
    C.PROJECT_ROOT / "artifacts" / "reports" / "probes"
    / "bg_partial_cache_splice_v2_2026-06-01"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [24, 36, 47]
ALPHAS = [0.0, 0.1, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SplicePlan:
    prompt_id: str
    boundary_loop: int
    boundary_layer: int
    boundary_slot: int
    shared_slots: list[int]
    recompute_slots: list[int]
    invalid_shared_slots: list[int]
    slot_policy_name: str
    expected_compute_saved: float
    implementation_mode: str
    notes: str = ""


@dataclass
class SpliceResult:
    splice_plan: SplicePlan
    equivalence_metrics: dict
    cache_slot_metrics: dict
    compute_metrics: dict
    pass_fail: str
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Slot policies
# ---------------------------------------------------------------------------


def slot_policy(boundary_loop: int, boundary_layer: int, num_ut: int, num_layers: int,
                policy: str = "downstream_only") -> dict:
    """Classify slots into shared / recompute / risky for a boundary at the
    OUTPUT of (boundary_loop, boundary_layer).

    downstream_only (empirically-correct policy): the perturbation is injected at
    the layer OUTPUT, so slot (u, boundary_layer) is computed at the layer INPUT
    and is UNAFFECTED. Shared = loops<u, plus loop u layers <= boundary_layer.
    Recompute = loop u layers > boundary_layer, plus all later loops.
    """
    shared, recompute = [], []
    for ut in range(num_ut):
        for layer in range(num_layers):
            slot = ut * num_layers + layer
            if ut < boundary_loop:
                shared.append(slot)
            elif ut == boundary_loop:
                if policy == "conservative":
                    # share strictly before boundary_slot; recompute boundary_slot+
                    (shared if layer < boundary_layer else recompute).append(slot)
                elif policy == "aggressive":
                    # share boundary_layer too AND one beyond (expected to fail)
                    (shared if layer <= boundary_layer + 1 else recompute).append(slot)
                elif policy == "loop_boundary":
                    # share prior loops only; recompute the entire boundary loop
                    recompute.append(slot)
                else:  # downstream_only (default, correct)
                    (shared if layer <= boundary_layer else recompute).append(slot)
            else:
                recompute.append(slot)
    return {"shared_slots": shared, "recompute_slots": recompute,
            "n_shared": len(shared), "n_recompute": len(recompute),
            "policy": policy, "boundary_slot": boundary_loop * num_layers + boundary_layer}


def layer_passes(boundary_loop: int, boundary_layer: int, num_ut: int, num_layers: int) -> dict:
    """Layer-pass accounting for the downstream_only policy."""
    prefix = boundary_loop * num_layers + (boundary_layer + 1)   # to capture H_boundary
    suffix = (num_layers - boundary_layer - 1) + (num_ut - boundary_loop - 1) * num_layers
    full = num_ut * num_layers
    return {"prefix_passes": prefix, "suffix_passes": suffix, "full_passes": full}


# ---------------------------------------------------------------------------
# Boundary-hidden capture + suffix recompute
# ---------------------------------------------------------------------------


class _BoundaryCaptureHook:
    """Forward hook on decoder layer L that records its OUTPUT at loop == target_loop."""

    def __init__(self, target_loop: int):
        self.target_loop = target_loop
        self.captured = None

    def __call__(self, module, args, kwargs, output):
        cur = kwargs.get("current_ut", None)
        if cur is None and len(args) >= 1:
            cur = kwargs.get("current_ut")
        if int(cur) == int(self.target_loop):
            hs = output[0] if isinstance(output, tuple) else output
            self.captured = hs.detach().clone()
        return output


@torch.no_grad()
def root_prefill_with_boundary(model, ids, am, boundary_loop, boundary_layer):
    """Full unperturbed root prefill that ALSO captures the residual hidden at the
    OUTPUT of (boundary_loop, boundary_layer). Returns
    (root_next_logits, root_cache, am_full, H_boundary)."""
    hook = _BoundaryCaptureHook(boundary_loop)
    handle = model.model.layers[boundary_layer].register_forward_hook(hook, with_kwargs=True)
    nl, cache, _, am_full = C.prefill(model, ids, am)
    handle.remove()
    return nl, cache, am_full, hook.captured


@torch.no_grad()
def prefix_prefill(model, ids, boundary_loop, boundary_layer, device,
                   position_ids=None, mask4d=None):
    """MINIMAL shared-prefix prefill: run loops 0..boundary_loop-1 fully plus
    loop boundary_loop layers 0..boundary_layer, writing ONLY the shared slots and
    capturing H_boundary (= output of (boundary_loop, boundary_layer)). This is the
    amortized cost shared across all branches; it avoids running the rest of the
    prompt. Returns (shared_cache, H_boundary, n_layer_passes, seconds).

    position_ids / mask4d default to an unpadded full causal prefill; pass explicit
    ones for left-padded prompts."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    P = ids.shape[1]
    inputs_embeds = model.model.embed_tokens(ids)
    dtype = inputs_embeds.dtype
    shared_cache = C.new_cache()
    pos_ids = position_ids if position_ids is not None else torch.arange(P, device=device).unsqueeze(0)
    cache_pos = torch.arange(P, device=device)
    pos_emb = model.model.rotary_emb(inputs_embeds, pos_ids)
    if mask4d is None:
        mask4d = _causal_mask_4d(P, dtype, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    passes = 0
    h = inputs_embeds
    for uu in range(boundary_loop + 1):
        last_layer = boundary_layer if uu == boundary_loop else (num_layers - 1)
        for l in range(last_layer + 1):
            h = model.model.layers[l](
                h, attention_mask=mask4d, position_ids=pos_ids, past_key_value=shared_cache,
                use_cache=True, cache_position=cache_pos, position_embeddings=pos_emb,
                current_ut=uu,
            )
            passes += 1
            if uu == boundary_loop and l == boundary_layer:
                H_boundary = h.detach().clone()
        if uu < boundary_loop:
            h = model.model.norm(h)  # post-norm feeds next loop
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    return shared_cache, H_boundary, passes, secs


def merge_prefix_suffix(shared_cache, suffix_cache, num_slots):
    """Build full branch cache from a minimal shared-prefix cache + suffix cache."""
    cls = type(shared_cache)
    out = cls(getattr(shared_cache, "max_cache_size", None))
    klist = [None] * num_slots
    vlist = [None] * num_slots
    for slot in range(num_slots):
        for src in (shared_cache, suffix_cache):
            if slot < len(src.key_cache) and src.key_cache[slot] is not None:
                klist[slot] = src.key_cache[slot].detach().clone()
                vlist[slot] = src.value_cache[slot].detach().clone()
                break
    out._key_cache = klist
    out._value_cache = vlist
    # seen tokens = prompt length (any populated slot's seq len)
    for k in klist:
        if k is not None:
            out._seen_tokens = k.shape[2]
            break
    return out


def _causal_mask_4d(P, dtype, device):
    min_dtype = torch.finfo(dtype).min
    m = torch.full((P, P), min_dtype, dtype=dtype, device=device)
    m = torch.triu(m, diagonal=1)
    return m[None, None, :, :]


def apply_boundary_perturbation(H_boundary, alpha, direction, token_range):
    """Reproduce LayerOutputPerturbHook(token_range) on a captured boundary hidden.
    Returns (H_perturbed, perturb_rms)."""
    if alpha == 0.0:
        return H_boundary.clone(), 0.0
    b, P, hidden = H_boundary.shape
    start, end = token_range
    start = max(0, int(start)); end = min(int(end), P)
    Hp = H_boundary.clone()
    if start < end:
        ref = Hp[:, start:end, :].detach().to(torch.float32)
        scale = float(alpha) * ref.pow(2).mean().sqrt().item()
        d = direction.to(device=Hp.device, dtype=Hp.dtype)
        Hp[:, start:end, :] = Hp[:, start:end, :] + scale * d.view(1, 1, hidden)
        return Hp, scale
    return Hp, 0.0


@torch.no_grad()
def suffix_recompute(model, H_boundary_perturbed, boundary_loop, boundary_layer, device,
                     position_ids=None, mask4d=None):
    """Run ONLY the suffix from the boundary hidden: finish loop `boundary_loop`
    (layers boundary_layer+1 .. end), then loops boundary_loop+1 .. last.

    Returns (next_logits [1,vocab], suffix_cache, n_layer_passes, seconds).
    suffix_cache holds ONLY the recomputed affected slots (others are None).

    position_ids / mask4d default to an unpadded full causal prefill."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_ut = int(getattr(cfg, "total_ut_steps", 4))
    P = H_boundary_perturbed.shape[1]
    dtype = H_boundary_perturbed.dtype

    suffix_cache = C.new_cache()
    pos_ids = position_ids if position_ids is not None else torch.arange(P, device=device).unsqueeze(0)
    cache_pos = torch.arange(P, device=device)
    pos_emb = model.model.rotary_emb(H_boundary_perturbed, pos_ids)
    if mask4d is None:
        mask4d = _causal_mask_4d(P, dtype, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    passes = 0
    h = H_boundary_perturbed
    # finish the boundary loop
    for l in range(boundary_layer + 1, num_layers):
        h = model.model.layers[l](
            h, attention_mask=mask4d, position_ids=pos_ids, past_key_value=suffix_cache,
            use_cache=True, cache_position=cache_pos, position_embeddings=pos_emb,
            current_ut=boundary_loop,
        )
        passes += 1
    h = model.model.norm(h)
    # remaining loops
    for uu in range(boundary_loop + 1, num_ut):
        for l in range(num_layers):
            h = model.model.layers[l](
                h, attention_mask=mask4d, position_ids=pos_ids, past_key_value=suffix_cache,
                use_cache=True, cache_position=cache_pos, position_embeddings=pos_emb,
                current_ut=uu,
            )
            passes += 1
        h = model.model.norm(h)
    logits = model.lm_head(h)
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    return logits[:, -1, :], suffix_cache, passes, secs


def merge_spliced_cache(root_cache, suffix_cache, recompute_slots):
    """Build the branch cache: clone root, overwrite recompute_slots with the
    suffix-recomputed K/V. Shared slots remain from root."""
    spliced = H.clone_universal_cache(root_cache)
    rset = set(recompute_slots)
    for slot in rset:
        if slot < len(suffix_cache.key_cache) and suffix_cache.key_cache[slot] is not None:
            spliced._key_cache[slot] = suffix_cache.key_cache[slot].detach().clone()
            spliced._value_cache[slot] = suffix_cache.value_cache[slot].detach().clone()
    return spliced


@torch.no_grad()
def full_perturbed_prefill(model, ids, am, boundary_loop, boundary_layer, alpha, direction,
                           token_range):
    """Oracle/reference: full prompt prefill with the perturbation hook at
    (boundary_loop, boundary_layer). Returns (next_logits, ref_cache, am_full,
    n_layer_passes, seconds, perturb_rms)."""
    cfg = model.config
    full_passes = int(getattr(cfg, "total_ut_steps", 4)) * cfg.num_hidden_layers
    hook = C.LayerOutputPerturbHook(direction, alpha, token_range=token_range,
                                    target_loop=boundary_loop)
    handle = C.register_perturb_hook(model, boundary_layer, hook)
    device = ids.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    nl, cache, _, am_full = C.prefill(model, ids, am)
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    handle.remove()
    return nl, cache, am_full, full_passes, secs, hook.last_perturb_rms


@torch.no_grad()
def build_spliced_branch(model, ids, am, boundary_loop, boundary_layer, alpha, direction,
                         token_range, policy="downstream_only", root_pack=None):
    """Full v2 splice path for one branch. If root_pack (nl, root_cache, am_full,
    H_boundary) is provided, reuse it (amortized prefix across branches).

    Returns dict with next_logits, spliced_cache, am_full, plan, accounting."""
    device = ids.device
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_ut = int(getattr(cfg, "total_ut_steps", 4))
    if root_pack is None:
        root_nl, root_cache, am_full, H_boundary = root_prefill_with_boundary(
            model, ids, am, boundary_loop, boundary_layer)
    else:
        root_nl, root_cache, am_full, H_boundary = root_pack

    pol = slot_policy(boundary_loop, boundary_layer, num_ut, num_layers, policy)
    Hp, prms = apply_boundary_perturbation(H_boundary, alpha, direction, token_range)
    next_logits, suffix_cache, passes, secs = suffix_recompute(
        model, Hp, boundary_loop, boundary_layer, device)
    spliced = merge_spliced_cache(root_cache, suffix_cache, pol["recompute_slots"])

    acc = layer_passes(boundary_loop, boundary_layer, num_ut, num_layers)
    return {
        "next_logits": next_logits, "spliced_cache": spliced, "am_full": am_full,
        "root_cache": root_cache, "perturb_rms": prms, "policy": pol,
        "suffix_passes_measured": passes, "suffix_seconds": secs,
        "accounting": acc,
    }


# ---------------------------------------------------------------------------
# Continuation + comparison helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_continue(model, cache, am_full, next_logits, n_steps, device):
    """Greedy continuation from a cache; returns (tokens, per_step_logits)."""
    tokens, step_logits = [], [next_logits[0].detach().float().cpu()]
    cur_am, nl = am_full, next_logits
    for _ in range(n_steps):
        t = int(nl[0].argmax().item())
        tokens.append(t)
        cur_am = H.update_attention_mask(cur_am, 1)
        nl, cache = C.decode_step(model, torch.tensor([[t]], device=device), cache, cur_am)
        step_logits.append(nl[0].detach().float().cpu())
    return tokens, step_logits


@torch.no_grad()
def replay_logits(model, cache, am_full, tokens, device):
    """Force-replay tokens through a cache; per-step next-token logits."""
    out, cur_am = [], am_full
    for t in tokens:
        cur_am = H.update_attention_mask(cur_am, 1)
        nl, cache = C.decode_step(model, torch.tensor([[t]], device=device), cache, cur_am)
        out.append(nl[0].detach().float().cpu())
    return out


def default_token_range(prompt_len: int):
    return (max(1, prompt_len // 2), prompt_len)


# IO passthrough to v2 OUT_DIR
def save_json(name, obj):
    return _save(name, obj, "json")


def save_md(name, text):
    p = OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(text)
    return p


def save_csv(name, rows, fieldnames=None):
    import csv
    p = OUT_DIR / name
    if not rows:
        with open(p, "w") as f:
            f.write("")
        return p
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _save(name, obj, kind):
    import json
    p = OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=C._json_default)
    return p
