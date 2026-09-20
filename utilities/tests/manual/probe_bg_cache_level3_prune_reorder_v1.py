"""PART F - Level 3: prune/reorder survivor cache.

Create K batched branches, decode some steps, then prune+reorder survivors with
cache.reorder_cache(beam_idx) in lockstep with BranchState / input_ids /
attention_mask / generated_ids / lineage. Continue survivors and validate each
survivor's cached logits against full recomputation of that survivor's exact
sequence. Tests subset selection, order changes, and multi-round pruning.
"""

from __future__ import annotations

import json

import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H


@torch.no_grad()
def build_batched_branches(model, p, K):
    """Prefill prompt, expand to K branches seeded by top-K first tokens.
    Returns (bcache, bids, bmask, states, branch_first_tokens, prompt_len)."""
    ids, am = C.tokenize(p["text"])
    device = ids.device
    root_next_logits, root_cache, _, am_full = C.prefill(model, ids, am)
    first_tokens = root_next_logits[0].topk(K).indices.tolist()
    bcache = H.expand_universal_cache_for_branches(root_cache, K)
    bmask = am_full.repeat(K, 1)
    bids = ids.repeat(K, 1)
    states = []
    for i, ft in enumerate(first_tokens):
        bid = f"{p['id']}.b{i}"
        states.append(H.BranchState(
            branch_id=bid, parent_branch_id="root", root_branch_id=p["id"],
            input_ids=bids[i:i + 1].clone(), attention_mask=bmask[i:i + 1].clone(),
            past_key_values=bcache, cache_position=torch.tensor([ids.shape[1]], device=device),
            generated_ids=[], lineage_path=[p["id"], bid],
        ))
    # feed first tokens
    col = torch.tensor([[ft] for ft in first_tokens], device=device)
    bmask = H.update_attention_mask(bmask, 1)
    logits, bcache = C.decode_step(model, col, bcache, bmask)
    bids = torch.cat([bids, col], dim=1)
    for i in range(K):
        states[i].generated_ids.append(first_tokens[i])
        states[i].input_ids = bids[i:i + 1].clone()
        states[i].attention_mask = bmask[i:i + 1].clone()
    return bcache, bids, bmask, states, logits, int(ids.shape[1])


@torch.no_grad()
def batched_decode_steps(model, bcache, bids, bmask, states, next_logits, n_steps):
    """Greedy per-row decode for n_steps; updates bids/bmask/states in place-ish.
    Returns (bcache, bids, bmask, states, next_logits)."""
    device = bids.device
    K = bids.shape[0]
    for _ in range(n_steps):
        col = next_logits.argmax(dim=-1, keepdim=True)  # [K,1]
        bmask = H.update_attention_mask(bmask, 1)
        next_logits, bcache = C.decode_step(model, col, bcache, bmask)
        bids = torch.cat([bids, col], dim=1)
        toks = col.squeeze(1).tolist()
        for i in range(K):
            states[i].generated_ids.append(int(toks[i]))
            states[i].input_ids = bids[i:i + 1].clone()
            states[i].attention_mask = bmask[i:i + 1].clone()
    return bcache, bids, bmask, states, next_logits


@torch.no_grad()
def prune(bcache, bids, bmask, states, beam_idx):
    beam = torch.tensor(beam_idx, dtype=torch.long)
    bcache, states, bids, bmask = H.reorder_cache_and_branch_state(
        bcache, states, beam, bids, bmask
    )
    return bcache, bids, bmask, states


@torch.no_grad()
def validate_survivors(model, bids, bmask, states, live_next_logits):
    """For each survivor row, compare the LIVE pruned/reordered cache's
    continuation logits (live_next_logits[r], produced by the in-place reordered
    survivor cache) against a full no-cache recompute of that survivor's exact
    sequence. This is the real survivor-cache correctness test."""
    rows = []
    for r in range(bids.shape[0]):
        seq = bids[r:r + 1]
        msk = bmask[r:r + 1]
        full_logits = C.full_recompute_logits(model, seq, msk)[:, -1, :]
        cmp = C.compare_logits(live_next_logits[r], full_logits[0])
        # lineage alignment: generated tail must match input_ids tail
        chk = H.assert_branch_state_consistent(states[r])
        rows.append({
            "row": r, "branch_id": states[r].branch_id,
            "generated_ids": states[r].generated_ids,
            "lineage": states[r].lineage_path,
            "lineage_aligned": chk["generated_tail_match"] and chk["attn_eq_input_len"],
            **cmp,
        })
    return rows


# (prompt-agnostic) prune plans: list of rounds; each round = (beam_idx, n_continue)
def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()

    plans = [
        {"name": "K8_reorder_subset", "K": 8, "init_steps": 2,
         "rounds": [([3, 1, 6], 3)]},
        {"name": "K8_multiround_8to4to2", "K": 8, "init_steps": 1,
         "rounds": [([0, 2, 4, 6], 2), ([2, 0], 2)]},
        {"name": "K8_to3", "K": 8, "init_steps": 2,
         "rounds": [([5, 2, 7], 2)]},
        {"name": "K4_to1", "K": 4, "init_steps": 2,
         "rounds": [([2], 3)]},
    ]

    all_rows = []
    per_run = []
    for p in C.TEST_PROMPTS[:3]:  # 3 prompts keeps runtime modest; all plans each
        for plan in plans:
            bcache, bids, bmask, states, next_logits, plen = build_batched_branches(model, p, plan["K"])
            bcache, bids, bmask, states, next_logits = batched_decode_steps(
                model, bcache, bids, bmask, states, next_logits, plan["init_steps"]
            )
            order_log = []
            for (beam_idx, n_cont) in plan["rounds"]:
                pre_branch_ids = [s.branch_id for s in states]
                pre_gen = [list(s.generated_ids) for s in states]
                bcache, bids, bmask, states = prune(bcache, bids, bmask, states, beam_idx)
                # next_logits must be reindexed to survivors too, else the next
                # decode feeds K rows into a survivor-sized cache.
                next_logits = next_logits.index_select(
                    0, torch.tensor(beam_idx, dtype=torch.long, device=next_logits.device)
                )
                # verify survivor order: states[r] == pre[beam_idx[r]]
                order_ok = all(states[r].branch_id == pre_branch_ids[beam_idx[r]]
                               for r in range(len(beam_idx)))
                gen_ok = all(states[r].generated_ids == pre_gen[beam_idx[r]]
                             for r in range(len(beam_idx)))
                # cache batch dim equals survivor count
                fs = next(i for i, k in enumerate(bcache.key_cache) if k is not None)
                cache_batch_ok = (bcache.key_cache[fs].shape[0] == len(beam_idx)
                                  and bids.shape[0] == len(beam_idx))
                order_log.append({
                    "beam_idx": beam_idx, "survivor_order_ok": order_ok,
                    "generated_alignment_ok": gen_ok, "cache_batch_ok": cache_batch_ok,
                    "survivor_branch_ids": [s.branch_id for s in states],
                })
                # continue survivors
                bcache, bids, bmask, states, next_logits = batched_decode_steps(
                    model, bcache, bids, bmask, states, next_logits, n_cont
                )
            survivor_rows = validate_survivors(model, bids, bmask, states, next_logits)
            eq = C.classify_equivalence(survivor_rows)
            lineage_ok = all(r["lineage_aligned"] for r in survivor_rows)
            order_all_ok = all(o["survivor_order_ok"] and o["generated_alignment_ok"]
                               and o["cache_batch_ok"] for o in order_log)
            per_run.append({
                "prompt_id": p["id"], "plan": plan["name"], "K": plan["K"],
                "rounds": order_log, "survivor_eq": eq, "lineage_ok": lineage_ok,
                "order_all_ok": order_all_ok,
                "survivors": survivor_rows,
            })
            for r in survivor_rows:
                all_rows.append({"prompt_id": p["id"], "plan": plan["name"], **{
                    k: r[k] for k in ("row", "branch_id", "logit_rms", "logit_max_abs",
                                      "top1_match", "top5_overlap", "lineage_aligned")
                }})
            C.clear_cuda()

    survivors_faithful = all(pr["survivor_eq"]["cache_faithful"] for pr in per_run)
    all_lineage_ok = all(pr["lineage_ok"] for pr in per_run)
    all_order_ok = all(pr["order_all_ok"] for pr in per_run)
    strict = all(pr["survivor_eq"]["strict_equiv"] for pr in per_run)
    max_rms = max(pr["survivor_eq"]["max_rms"] for pr in per_run)
    max_abs = max(pr["survivor_eq"]["max_abs"] for pr in per_run)

    if not all_order_ok or not all_lineage_ok:
        verdict = "LINEAGE_CACHE_MISALIGNMENT"
    elif not survivors_faithful:
        verdict = "SURVIVOR_CACHE_MISMATCH"
    elif strict:
        verdict = "PRUNE_REORDER_VALID"
    else:
        verdict = "PRUNE_REORDER_NUMERIC_DRIFT_SMALL"

    summary = {
        "verdict": verdict, "survivors_faithful": survivors_faithful,
        "lineage_ok": all_lineage_ok, "order_ok": all_order_ok,
        "max_rms": max_rms, "max_abs": max_abs, "per_run": per_run,
        "note": "reorder_cache reorders the batch dim of every populated slot; BranchState "
                "list, input_ids, attention_mask, generated_ids and lineage are reordered in "
                "lockstep. Survivor caches validated against full recompute of each exact "
                "survivor sequence. (Note: validation recompute uses the survivor's full "
                "sequence; the cached path under test is the in-place reordered survivor cache.)",
    }
    C.save_json("level3_prune_reorder.json", summary)
    C.save_csv("level3_rows.csv", all_rows)

    md = ["# PART F - Level 3: prune/reorder survivor cache\n",
          f"**BG_AUTOREGRESSIVE_CACHE_LEVEL3_VERDICT = {verdict}**\n",
          f"- survivors cache-faithful vs full recompute: {survivors_faithful}",
          f"- lineage/order alignment ok: {all_lineage_ok}/{all_order_ok}",
          f"- max survivor RMS {max_rms:.4g}, max-abs {max_abs:.4g}\n",
          "## Runs\n"]
    for pr in per_run:
        md.append(f"### {pr['prompt_id']} / {pr['plan']}")
        for o in pr["rounds"]:
            md.append(f"- beam_idx={o['beam_idx']} order_ok={o['survivor_order_ok']} "
                      f"gen_ok={o['generated_alignment_ok']} cache_batch_ok={o['cache_batch_ok']} "
                      f"-> survivors {o['survivor_branch_ids']}")
        md.append(f"- survivor equivalence: {json.dumps(pr['survivor_eq'])}\n")
    C.save_md("level3_prune_reorder.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_LEVEL3_VERDICT = {verdict}")
    print(f"faithful={survivors_faithful} lineage_ok={all_lineage_ok} order_ok={all_order_ok} "
          f"max_rms={max_rms:.4g} max_abs={max_abs:.4g}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
