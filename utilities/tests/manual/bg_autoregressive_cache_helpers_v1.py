"""PART B - Cache/branch helper implementation + unit tests.

Reusable TEST-ONLY helpers for UniversalTransformerCache cloning, expansion,
comparison, summarisation, and BranchState lineage bookkeeping. Includes a
self-contained unit-test main that exercises clone / reorder / compare against
a synthetic cache.

NO model weights touched. NO steering. NO training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import bg_autoregressive_cache_common_v1 as C


# ---------------------------------------------------------------------------
# BranchState
# ---------------------------------------------------------------------------


@dataclass
class BranchState:
    branch_id: str
    parent_branch_id: Optional[str]
    root_branch_id: str
    input_ids: torch.Tensor            # [b, total_len] running ids (prompt + generated)
    attention_mask: torch.Tensor       # [b, total_len]
    past_key_values: Any               # UniversalTransformerCache (own or shared/batched)
    cache_position: torch.Tensor       # positions used by the *next* decode step
    generated_ids: list[int] = field(default_factory=list)
    birth_loop: Optional[int] = None
    birth_layer: Optional[int] = None
    perturb_count: int = 0
    lineage_path: list[str] = field(default_factory=list)
    score: Optional[float] = None


def assert_branch_state_consistent(bs: BranchState, batched_index: Optional[int] = None) -> dict:
    """Verify input_ids / attention_mask / cache / cache_position / generated_ids align.

    batched_index: if the branch lives in a batched cache at row r, the per-branch
    bookkeeping tensors are batch-1 views; cache seq length is shared across the batch.
    Returns a dict of checks; raises AssertionError on hard inconsistency.
    """
    checks = {}
    L = int(bs.input_ids.shape[1])
    am_len = int(bs.attention_mask.shape[1])
    checks["input_ids_len"] = L
    checks["attention_mask_len"] = am_len
    checks["attn_eq_input_len"] = (am_len == L)
    assert am_len == L, f"attention_mask len {am_len} != input_ids len {L}"

    cache_seq = bs.past_key_values.get_seq_length(0)
    checks["cache_seq_length"] = cache_seq
    # number of valid (unmasked) tokens for this branch
    valid = int(bs.attention_mask[0].sum().item())
    checks["valid_tokens"] = valid
    # cache holds keys for all positions fed so far; for an unpadded branch this
    # equals input_ids length. (For left-padded batches valid < L.)
    checks["cache_seq_matches_len_or_valid"] = cache_seq in (L, valid)

    # generated_ids must be the tail of input_ids
    if bs.generated_ids:
        tail = bs.input_ids[0, -len(bs.generated_ids):].tolist()
        checks["generated_tail_match"] = (tail == bs.generated_ids)
        assert tail == bs.generated_ids, f"generated_ids {bs.generated_ids} != tail {tail}"
    else:
        checks["generated_tail_match"] = True

    checks["lineage_nonempty"] = bool(bs.lineage_path)
    checks["branch_in_lineage"] = bs.branch_id in bs.lineage_path
    return checks


# ---------------------------------------------------------------------------
# Cache clone / expand / compare / summarize
# ---------------------------------------------------------------------------


def clone_universal_cache(cache):
    """Deep-clone every non-None key/value tensor into a NEW cache.

    Preserves _seen_tokens and max_cache_size. Storage is not shared.
    """
    cls = type(cache)
    new = cls(getattr(cache, "max_cache_size", None))
    new._key_cache = [None if k is None else k.detach().clone() for k in cache.key_cache]
    new._value_cache = [None if v is None else v.detach().clone() for v in cache.value_cache]
    new._seen_tokens = getattr(cache, "_seen_tokens", 0)
    return new


def expand_universal_cache_for_branches(cache, num_branches: int):
    """Expand batch dim from 1 -> num_branches with independent storage per branch."""
    cls = type(cache)
    new = cls(getattr(cache, "max_cache_size", None))

    def _expand(t):
        if t is None:
            return None
        assert t.shape[0] == 1, (
            f"expand expects batch dim 1, got {t.shape[0]}"
        )
        # repeat copies storage (independent per branch)
        return t.repeat(num_branches, 1, 1, 1).contiguous()

    new._key_cache = [_expand(k) for k in cache.key_cache]
    new._value_cache = [_expand(v) for v in cache.value_cache]
    new._seen_tokens = getattr(cache, "_seen_tokens", 0)
    return new


def summarize_universal_cache(cache) -> dict:
    """Summary: populated slots, indices, shapes, seq lengths, seen tokens, dtype/device."""
    populated = [i for i, k in enumerate(cache.key_cache) if k is not None]
    shapes = {}
    seqlens = {}
    dtype = None
    device = None
    for i in populated:
        k = cache.key_cache[i]
        shapes[i] = list(k.shape)
        seqlens[i] = int(k.shape[2])
        if dtype is None:
            dtype = str(k.dtype)
            device = str(k.device)
    uniq_shapes = sorted({tuple(s) for s in shapes.values()})
    uniq_seqlens = sorted(set(seqlens.values()))
    return {
        "num_populated_slots": len(populated),
        "slot_indices": populated if len(populated) <= 16 else populated[:8] + ["..."] + populated[-4:],
        "unique_shapes": [list(s) for s in uniq_shapes],
        "unique_seq_lengths": uniq_seqlens,
        "seen_tokens": getattr(cache, "_seen_tokens", None),
        "max_cache_size": getattr(cache, "max_cache_size", None),
        "dtype": dtype,
        "device": device,
        "batch_size": (cache.key_cache[populated[0]].shape[0] if populated else None),
    }


def compare_universal_caches(cache_a, cache_b, per_slot_limit: int = 8) -> dict:
    """Compare populated slots: shapes, seq lengths, RMS/max-abs per slot, missing slots."""
    na = len(cache_a.key_cache)
    nb = len(cache_b.key_cache)
    n = max(na, nb)
    per_slot = []
    max_rms = 0.0
    max_abs = 0.0
    n_mismatch_shape = 0
    n_missing = 0
    n_compared = 0
    batch_a = batch_b = None
    for i in range(n):
        ka = cache_a.key_cache[i] if i < na else None
        kb = cache_b.key_cache[i] if i < nb else None
        if ka is None and kb is None:
            continue
        if (ka is None) != (kb is None):
            n_missing += 1
            per_slot.append({"slot": i, "status": "missing_one", "a_none": ka is None})
            continue
        if ka.shape != kb.shape:
            n_mismatch_shape += 1
            per_slot.append(
                {"slot": i, "status": "shape_mismatch", "a": list(ka.shape), "b": list(kb.shape)}
            )
            continue
        if batch_a is None:
            batch_a, batch_b = int(ka.shape[0]), int(kb.shape[0])
        kdiff = (ka.to(torch.float32) - kb.to(torch.float32))
        va = cache_a.value_cache[i].to(torch.float32)
        vb = cache_b.value_cache[i].to(torch.float32)
        vdiff = va - vb
        krms = float(kdiff.pow(2).mean().sqrt().item())
        vrms = float(vdiff.pow(2).mean().sqrt().item())
        kmax = float(kdiff.abs().max().item())
        vmax = float(vdiff.abs().max().item())
        slot_rms = max(krms, vrms)
        slot_max = max(kmax, vmax)
        max_rms = max(max_rms, slot_rms)
        max_abs = max(max_abs, slot_max)
        n_compared += 1
        if len(per_slot) < per_slot_limit:
            per_slot.append(
                {"slot": i, "status": "compared", "key_rms": krms, "val_rms": vrms,
                 "key_max_abs": kmax, "val_max_abs": vmax, "seq_len": int(ka.shape[2])}
            )
    return {
        "n_slots_compared": n_compared,
        "n_shape_mismatch": n_mismatch_shape,
        "n_missing_one": n_missing,
        "max_rms": max_rms,
        "max_abs": max_abs,
        "batch_a": batch_a,
        "batch_b": batch_b,
        "per_slot_sample": per_slot,
        "seq_len_a": cache_a.get_seq_length(0),
        "seq_len_b": cache_b.get_seq_length(0),
    }


# ---------------------------------------------------------------------------
# Branch construction / mutation
# ---------------------------------------------------------------------------


def make_branch_states_from_prefill(
    root_cache, input_ids, attention_mask, branch_first_tokens: list[int],
    parent_id: str = "root", root_id: str = "root",
) -> list[BranchState]:
    """Clone the (batch-1) prefill cache once per branch and seed each branch's
    first generated token. Returns independent BranchState objects."""
    device = input_ids.device
    states = []
    for i, t in enumerate(branch_first_tokens):
        bid = f"{root_id}.b{i}"
        cache_i = clone_universal_cache(root_cache)
        states.append(
            BranchState(
                branch_id=bid,
                parent_branch_id=parent_id,
                root_branch_id=root_id,
                input_ids=input_ids.clone(),
                attention_mask=attention_mask.clone(),
                past_key_values=cache_i,
                cache_position=torch.tensor([input_ids.shape[1]], device=device),
                generated_ids=[],
                lineage_path=[root_id, bid],
                score=None,
            )
        )
    return states


def update_attention_mask(attention_mask, n_new: int = 1):
    """Append n_new ones (valid tokens) to a [b, L] mask -> [b, L+n_new]."""
    b = attention_mask.shape[0]
    ones = torch.ones((b, n_new), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([attention_mask, ones], dim=1)


def concat_generated_token(bs: BranchState, token: int) -> None:
    """In-place: append a generated token id to a BranchState's bookkeeping."""
    device = bs.input_ids.device
    bs.input_ids = torch.cat(
        [bs.input_ids, torch.tensor([[token]], device=device)], dim=1
    )
    bs.attention_mask = update_attention_mask(bs.attention_mask, 1)
    bs.generated_ids.append(int(token))
    bs.cache_position = torch.tensor([bs.input_ids.shape[1] - 1], device=device)


def compute_cache_slot_order(num_ut: int, num_layers: int) -> list[dict]:
    """Map slot -> (loop, layer) in population order (loop outer, layer inner)."""
    out = []
    for ut in range(num_ut):
        for layer in range(num_layers):
            out.append({"slot": ut * num_layers + layer, "loop": ut, "layer": layer})
    return out


def affected_cache_slots_for_boundary(
    perturb_loop: int, perturb_layer: int, num_ut: int, num_layers: int
) -> dict:
    """Given a perturbation at (loop, layer), classify cache slots into
    definitely-shared (before the boundary in the loop/layer DAG) vs
    branch-specific (at/after boundary).

    Dependency structure: hidden flows layer 0..L within a loop, loops 0..U.
    A perturbation injected at the *output* of (perturb_loop, perturb_layer)
    affects: same loop layers > perturb_layer; all layers of later loops.
    (Slots store the K/V computed at the *input* of each (loop, layer).)
    """
    affected = set()
    shared = set()
    for ut in range(num_ut):
        for layer in range(num_layers):
            slot = ut * num_layers + layer
            if ut < perturb_loop:
                shared.add(slot)
            elif ut == perturb_loop:
                # K/V at layer L uses the hidden coming *into* layer L; the
                # perturbation is injected at the output of perturb_layer, so
                # layers <= perturb_layer are unaffected, layers > perturb_layer
                # are affected.
                if layer <= perturb_layer:
                    shared.add(slot)
                else:
                    affected.add(slot)
            else:
                affected.add(slot)
    return {
        "perturb_loop": perturb_loop,
        "perturb_layer": perturb_layer,
        "n_shared": len(shared),
        "n_affected": len(affected),
        "shared_slots": sorted(shared),
        "affected_slots": sorted(affected),
    }


# ---------------------------------------------------------------------------
# Prune / reorder (batched survivors)
# ---------------------------------------------------------------------------


def reorder_cache_and_branch_state(
    cache, branch_states: list[BranchState], beam_idx, batched_ids=None, batched_mask=None
):
    """Reorder a batched cache + parallel BranchState list (and optional batched
    id/mask tensors) by beam_idx. beam_idx may repeat or drop rows (prune+reorder)."""
    if not torch.is_tensor(beam_idx):
        beam_idx = torch.tensor(list(beam_idx), dtype=torch.long)
    cache.reorder_cache(beam_idx)
    idx_list = beam_idx.tolist()
    new_states = [branch_states[i] for i in idx_list]
    new_ids = batched_ids.index_select(0, beam_idx.to(batched_ids.device)) if batched_ids is not None else None
    new_mask = batched_mask.index_select(0, beam_idx.to(batched_mask.device)) if batched_mask is not None else None
    return cache, new_states, new_ids, new_mask


def prune_branch_states(cache, branch_states, survivor_indices, batched_ids=None, batched_mask=None):
    """Keep only survivor_indices (in the given order) from a batched cache."""
    return reorder_cache_and_branch_state(
        cache, branch_states, survivor_indices, batched_ids, batched_mask
    )


# Aliases matching the "Cache helper requirements" section names.
prune_branch_batch = prune_branch_states


def maybe_splice_cache_prefix(root_cache, branch_cache, perturb_loop, perturb_layer,
                              num_ut, num_layers, mode: str = "boundary"):
    """Build a spliced cache: copy 'shared' slots from root_cache and
    'affected' slots from branch_cache, per affected_cache_slots_for_boundary.

    mode='boundary' (Option A): copy root for shared slots, branch for affected.
    Returns (spliced_cache, slot_classification).
    """
    cls = type(branch_cache)
    spliced = cls(getattr(branch_cache, "max_cache_size", None))
    classification = affected_cache_slots_for_boundary(
        perturb_loop, perturb_layer, num_ut, num_layers
    )
    shared = set(classification["shared_slots"])
    n = max(len(root_cache.key_cache), len(branch_cache.key_cache))
    key_list = [None] * n
    val_list = [None] * n
    for slot in range(n):
        src = root_cache if slot in shared else branch_cache
        k = src.key_cache[slot] if slot < len(src.key_cache) else None
        v = src.value_cache[slot] if slot < len(src.value_cache) else None
        key_list[slot] = None if k is None else k.detach().clone()
        val_list[slot] = None if v is None else v.detach().clone()
    spliced._key_cache = key_list
    spliced._value_cache = val_list
    spliced._seen_tokens = getattr(branch_cache, "_seen_tokens", 0)
    return spliced, classification


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def _make_synthetic_cache(batch=2, heads=2, seq=3, dim=4, n_slots=8, seed=0):
    cls = C.get_universal_cache_class()
    cache = cls(n_slots)
    g = torch.Generator().manual_seed(seed)
    keys, vals = [], []
    for s in range(n_slots):
        if s % 3 == 2:  # leave some slots empty
            keys.append(None)
            vals.append(None)
        else:
            keys.append(torch.randn(batch, heads, seq, dim, generator=g))
            vals.append(torch.randn(batch, heads, seq, dim, generator=g))
    cache._key_cache = keys
    cache._value_cache = vals
    cache._seen_tokens = seq
    return cache


def run_unit_tests() -> dict:
    results = {}

    # --- clone ---
    base = _make_synthetic_cache()
    clone = clone_universal_cache(base)
    cmp = compare_universal_caches(base, clone)
    results["clone_shapes_match"] = (cmp["n_shape_mismatch"] == 0 and cmp["n_missing_one"] == 0)
    results["clone_values_match_rms0"] = (cmp["max_rms"] == 0.0 and cmp["max_abs"] == 0.0)
    # storage pointers differ
    ptr_diff = True
    for i, k in enumerate(base.key_cache):
        if k is not None:
            if k.data_ptr() == clone.key_cache[i].data_ptr():
                ptr_diff = False
                break
    results["clone_storage_differs"] = ptr_diff
    results["clone_seen_tokens_preserved"] = (clone._seen_tokens == base._seen_tokens)
    results["clone_max_cache_size_preserved"] = (clone.max_cache_size == base.max_cache_size)

    # modifying clone does not modify original
    first_slot = next(i for i, k in enumerate(base.key_cache) if k is not None)
    orig_val = base.key_cache[first_slot].clone()
    clone.key_cache[first_slot] += 100.0
    results["clone_isolation"] = bool(torch.equal(base.key_cache[first_slot], orig_val))
    cmp_after = compare_universal_caches(base, clone)
    results["compare_nonzero_after_modify"] = (cmp_after["max_abs"] > 0.0)

    # --- expand ---
    cls = C.get_universal_cache_class()
    single = cls(8)
    g = torch.Generator().manual_seed(7)
    single._key_cache = [torch.randn(1, 2, 3, 4, generator=g), None, torch.randn(1, 2, 3, 4, generator=g)]
    single._value_cache = [torch.randn(1, 2, 3, 4, generator=g), None, torch.randn(1, 2, 3, 4, generator=g)]
    single._seen_tokens = 3
    expanded = expand_universal_cache_for_branches(single, 4)
    exp_ok = all(
        (k is None) or (k.shape[0] == 4) for k in expanded.key_cache
    )
    # each branch row equals the original
    row_match = torch.equal(
        expanded.key_cache[0][0], single.key_cache[0][0]
    ) and torch.equal(expanded.key_cache[0][3], single.key_cache[0][0])
    # independence: modifying one row does not affect others
    expanded.key_cache[0][1] += 50.0
    indep = not torch.equal(expanded.key_cache[0][1], expanded.key_cache[0][2])
    results["expand_batch_dim"] = bool(exp_ok)
    results["expand_rows_equal_original"] = bool(row_match)
    results["expand_row_independence"] = bool(indep)

    # --- reorder ---
    rc = _make_synthetic_cache(batch=3, seed=11)
    # snapshot row 2 of first populated slot
    fs = next(i for i, k in enumerate(rc.key_cache) if k is not None)
    row0 = rc.key_cache[fs][0].clone()
    row2 = rc.key_cache[fs][2].clone()
    rc.reorder_cache(torch.tensor([2, 0, 1]))
    reorder_ok = torch.equal(rc.key_cache[fs][0], row2) and torch.equal(rc.key_cache[fs][1], row0)
    results["reorder_batch_dim"] = bool(reorder_ok)
    # prune to subset [2,0]
    rc2 = _make_synthetic_cache(batch=3, seed=12)
    r0 = rc2.key_cache[fs][0].clone()
    r2 = rc2.key_cache[fs][2].clone()
    rc2.reorder_cache(torch.tensor([2, 0]))
    prune_ok = (rc2.key_cache[fs].shape[0] == 2 and torch.equal(rc2.key_cache[fs][0], r2)
                and torch.equal(rc2.key_cache[fs][1], r0))
    results["reorder_prune_subset"] = bool(prune_ok)

    # --- slot helpers ---
    order = compute_cache_slot_order(4, 48)
    results["slot_order_count"] = (len(order) == 192)
    results["slot_order_formula"] = (order[48] == {"slot": 48, "loop": 1, "layer": 0})
    aff = affected_cache_slots_for_boundary(0, 24, 4, 48)
    # shared: loop0 layers 0..24 (25) ; affected: loop0 layers 25..47 (23) + loops1-3 (144) = 167
    results["affected_slots_count"] = (aff["n_affected"] == 167 and aff["n_shared"] == 25)

    # --- BranchState consistency ---
    cls2 = C.get_universal_cache_class()
    bs_cache = cls2(8)
    bs_cache._key_cache = [torch.randn(1, 2, 5, 4)]
    bs_cache._value_cache = [torch.randn(1, 2, 5, 4)]
    bs_cache._seen_tokens = 5
    bs = BranchState(
        branch_id="root.b0", parent_branch_id="root", root_branch_id="root",
        input_ids=torch.zeros(1, 5, dtype=torch.long),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        past_key_values=bs_cache, cache_position=torch.tensor([5]),
        generated_ids=[], lineage_path=["root", "root.b0"],
    )
    try:
        chk = assert_branch_state_consistent(bs)
        results["branch_state_consistency"] = chk["attn_eq_input_len"] and chk["cache_seq_matches_len_or_valid"]
    except AssertionError:
        results["branch_state_consistency"] = False

    return results


def main() -> int:
    C.set_seed()
    # ensure cache class available
    C.load_model()
    results = run_unit_tests()

    all_pass = all(bool(v) for v in results.values())
    clone_ok = results.get("clone_isolation") and results.get("clone_values_match_rms0") and results.get("clone_storage_differs")
    reorder_ok = results.get("reorder_batch_dim") and results.get("reorder_prune_subset")
    if not clone_ok:
        verdict = "CLONE_FAILED"
    elif not reorder_ok:
        verdict = "REORDER_FAILED"
    elif all_pass:
        verdict = "READY"
    else:
        verdict = "PARTIAL"

    report = {"verdict": verdict, "unit_tests": results}
    C.save_json("cache_helpers_report.json", report)

    md = ["# PART B - Cache helper unit tests\n",
          f"**BG_AUTOREGRESSIVE_CACHE_HELPERS_VERDICT = {verdict}**\n",
          "## Unit test results\n"]
    for k, v in results.items():
        md.append(f"- {'PASS' if v else 'FAIL'} `{k}`: {v}")
    md.append("\n## Helpers implemented\n")
    md.append("BranchState, clone_universal_cache, expand_universal_cache_for_branches, "
              "compare_universal_caches, summarize_universal_cache, assert_branch_state_consistent, "
              "make_branch_states_from_prefill, prune_branch_states/prune_branch_batch, "
              "reorder_cache_and_branch_state, concat_generated_token, update_attention_mask, "
              "compute_cache_slot_order, affected_cache_slots_for_boundary, maybe_splice_cache_prefix.")
    C.save_md("cache_helpers_report.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_HELPERS_VERDICT = {verdict}")
    print(json.dumps(results, indent=2))
    print("=" * 70)
    return 0 if verdict in ("READY", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
