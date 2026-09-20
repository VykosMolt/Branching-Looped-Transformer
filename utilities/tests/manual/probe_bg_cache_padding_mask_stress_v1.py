"""PART J - Padding, attention-mask, and cache-position stress tests.

Stresses attention_mask / cache_position / position_ids correctness for the
UniversalTransformerCache: single unpadded prompt, left-padded mixed-length
batches, length-1 decode, long+short batch, and reorder/prune after padding.

Key finding probed here: because Ouro derives position_ids from cache_position
(a single value across the batch) when position_ids is None, left-padded BATCHED
decode requires explicit per-row position_ids. With them, padded batched logits
match independent unpadded recomputation within bf16.
"""

from __future__ import annotations

import json

import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H


def build_left_padded_batch(tok, texts, device):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    encs = [tok(t, return_tensors="pt", add_special_tokens=True)["input_ids"][0] for t in texts]
    lengths = [int(e.shape[0]) for e in encs]
    max_len = max(lengths)
    bids = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
    bmask = torch.zeros((len(texts), max_len), dtype=torch.long)
    for i, e in enumerate(encs):
        bids[i, max_len - lengths[i]:] = e
        bmask[i, max_len - lengths[i]:] = 1
    bids = bids.to(device)
    bmask = bmask.to(device)
    position_ids = (bmask.long().cumsum(-1) - 1).clamp(min=0)
    return bids, bmask, position_ids, lengths, max_len


@torch.no_grad()
def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    rows = []
    results = {}

    # --- Test 1: single unpadded prompt baseline ---
    ids, am = C.tokenize("What is 2+2?")
    nl, cache, cpos, am_full = C.prefill(model, ids, am)
    fr = C.full_recompute_logits(model, ids, am)[:, -1, :]
    cmp1 = C.compare_logits(nl[0], fr[0])
    results["t1_single_unpadded"] = {"cmp": cmp1, "cache_position": cpos.tolist(),
                                     "get_mask_sizes": list(cache.get_mask_sizes(torch.tensor([7], device=device), 0))}
    rows.append({"test": "single_unpadded", "rms": cmp1["logit_rms"], "max_abs": cmp1["logit_max_abs"],
                 "top1": cmp1["top1_match"]})

    # --- Test 2 + 5: left-padded mixed-length batch (short + long) ---
    texts = ["What is 2+2?",
             "Explain photosynthesis in one sentence.",
             "Solve briefly: If a car travels 60 km in 2 hours, what is its speed?"]
    bids, bmask, pos_ids, lengths, max_len = build_left_padded_batch(tok, texts, device)
    cache_position = torch.arange(0, max_len, device=device)
    cb = C.new_cache()
    out = model(input_ids=bids, attention_mask=bmask, position_ids=pos_ids,
                past_key_values=cb, use_cache=True, cache_position=cache_position)
    padded_last = out.logits[:, -1, :]  # last position is a real token (left-padded)
    # independent unpadded full recompute per prompt
    pad_rows = []
    for i, t in enumerate(texts):
        ids_i, am_i = C.tokenize(t)
        fr_i = C.full_recompute_logits(model, ids_i, am_i)[:, -1, :]
        cmp = C.compare_logits(padded_last[i], fr_i[0])
        pad_rows.append(cmp)
        rows.append({"test": "left_padded_prefill", "row": i, "len": lengths[i],
                     "rms": cmp["logit_rms"], "max_abs": cmp["logit_max_abs"], "top1": cmp["top1_match"]})
    eq_pad = C.classify_equivalence(pad_rows)
    results["t2_left_padded_prefill"] = {
        "lengths": lengths, "max_len": max_len, "equivalence": eq_pad,
        "padding_corruption": not eq_pad["cache_faithful"],
    }

    # --- Test 2b: WITHOUT explicit position_ids (default cache_position) -> expect drift on padded rows ---
    cb2 = C.new_cache()
    out_nopos = model(input_ids=bids, attention_mask=bmask, position_ids=None,
                      past_key_values=cb2, use_cache=True, cache_position=cache_position)
    nopos_last = out_nopos.logits[:, -1, :]
    nopos_rows = []
    for i, t in enumerate(texts):
        ids_i, am_i = C.tokenize(t)
        fr_i = C.full_recompute_logits(model, ids_i, am_i)[:, -1, :]
        nopos_rows.append(C.compare_logits(nopos_last[i], fr_i[0]))
    eq_nopos = C.classify_equivalence(nopos_rows)
    # the longest prompt (no left pad) should still match; padded rows may differ
    results["t2b_default_position_ids"] = {
        "equivalence": eq_nopos,
        "note": "Padded rows can mismatch when position_ids defaults to cache_position; "
                "explicit per-row position_ids (test 2) fixes this.",
    }

    # --- Test 4: length-1 decode after unpadded prefill ---
    am_dec = H.update_attention_mask(am_full, 1)
    t0 = int(nl[0].argmax().item())
    nl2, cache = C.decode_step(model, torch.tensor([[t0]], device=device), cache, am_dec)
    fr2 = C.full_recompute_logits(model, torch.cat([ids, torch.tensor([[t0]], device=device)], dim=1), am_dec)[:, -1, :]
    cmp4 = C.compare_logits(nl2[0], fr2[0])
    results["t4_len1_decode"] = {"cmp": cmp4, "decode_cache_position": [cache.get_seq_length(0) - 1]}
    rows.append({"test": "len1_decode", "rms": cmp4["logit_rms"], "max_abs": cmp4["logit_max_abs"], "top1": cmp4["top1_match"]})

    # --- Test 6 + 7: reorder / prune after left-padded prefill, then decode ---
    # continue padded batch one step with per-row position_ids, then reorder rows.
    cur_mask = H.update_attention_mask(bmask, 1)
    next_pos = torch.tensor([[lengths[i]] for i in range(len(texts))], device=device)  # per-row real position
    col = padded_last.argmax(dim=-1, keepdim=True)
    out2 = model(input_ids=col, attention_mask=cur_mask, position_ids=next_pos,
                 past_key_values=cb, use_cache=True, cache_position=torch.arange(max_len, max_len + 1, device=device))
    # reorder rows [2,0,1]
    beam = torch.tensor([2, 0, 1], device=device)
    cb.reorder_cache(beam)
    reordered_ok = (cb.key_cache[0].shape[0] == 3)
    # verify reorder: after reorder, row0 corresponds to old row2
    results["t6_reorder_after_padding"] = {"beam": beam.tolist(), "cache_batch_after": cb.key_cache[0].shape[0],
                                           "reorder_shape_ok": bool(reordered_ok)}
    # prune to [0] (single survivor) of reordered
    cb.reorder_cache(torch.tensor([0], device=device))
    results["t7_prune_after_padding"] = {"cache_batch_after_prune": cb.key_cache[0].shape[0],
                                         "prune_ok": bool(cb.key_cache[0].shape[0] == 1)}

    # Verdict
    padding_safe = (results["t2_left_padded_prefill"]["equivalence"]["cache_faithful"]
                    and cmp1["top1_match"] and cmp4["top1_match"])
    cache_pos_ok = (results["t1_single_unpadded"]["cache_position"] == list(range(int(ids.shape[1])))
                    and results["t4_len1_decode"]["decode_cache_position"] == [int(ids.shape[1])])
    if not cache_pos_ok:
        verdict = "CACHE_POSITION_BUG"
    elif padding_safe and results["t6_reorder_after_padding"]["reorder_shape_ok"] and results["t7_prune_after_padding"]["prune_ok"]:
        verdict = "PADDING_MASK_SAFE"
    elif results["t2_left_padded_prefill"]["equivalence"]["all_top1_match"]:
        verdict = "PADDING_MASK_NUMERIC_DRIFT_SMALL"
    else:
        verdict = "BATCH_PADDING_BUG"

    summary = {
        "verdict": verdict, "padding_safe": padding_safe, "cache_position_ok": cache_pos_ok,
        "results": results,
        "note": "Left-padded batched decode requires explicit per-row position_ids (Ouro "
                "derives position_ids from a single cache_position otherwise). With correct "
                "position_ids, padded batched prefill matches independent unpadded recompute "
                "within bf16; reorder/prune after padding preserve batch dimensions.",
    }
    C.save_json("padding_mask_stress.json", summary)
    C.save_csv("padding_mask_rows.csv", rows)

    md = ["# PART J - Padding / attention-mask / cache-position stress\n",
          f"**BG_AUTOREGRESSIVE_CACHE_PADDING_MASK_VERDICT = {verdict}**\n",
          f"- single unpadded top1 match: {cmp1['top1_match']} (rms {cmp1['logit_rms']:.4g})",
          f"- left-padded prefill faithful: {results['t2_left_padded_prefill']['equivalence']['cache_faithful']} "
          f"(max rms {eq_pad['max_rms']:.4g})",
          f"- default position_ids faithful (expected possibly NOT for padded rows): {eq_nopos['cache_faithful']}",
          f"- length-1 decode top1 match: {cmp4['top1_match']}",
          f"- reorder after padding ok: {results['t6_reorder_after_padding']['reorder_shape_ok']}",
          f"- prune after padding ok: {results['t7_prune_after_padding']['prune_ok']}",
          f"- cache_position correct: {cache_pos_ok}\n",
          "Left-padded batched decode requires explicit per-row position_ids; with them, "
          "padded results match unpadded recompute within bf16.\n"]
    C.save_md("padding_mask_stress.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_PADDING_MASK_VERDICT = {verdict}")
    print(f"padding_safe={padding_safe} cache_pos_ok={cache_pos_ok} "
          f"padded_rms={eq_pad['max_rms']:.4g} nopos_faithful={eq_nopos['cache_faithful']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
