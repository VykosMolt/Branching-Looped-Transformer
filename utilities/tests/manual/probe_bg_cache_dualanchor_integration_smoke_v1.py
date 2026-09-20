"""PART L - DualAnchor branch-carry integration smoke.

Small end-to-end cache-integration test: fork K token-boundary branches, decode
a couple tokens, score with a deterministic proxy (cumulative token logprob),
select survivors, prune/reorder the survivor caches, continue, and compare the
survivors' live (reordered) cache logits to full recomputation of each survivor
sequence. Records BranchState lineage.

This is NOT the architecture-looped Phase 2 policy and NOT a DualAnchor scoring
run: DualAnchor scoring is deliberately SKIPPED (a logprob proxy is used) so no
tap registry is touched. The goal is cache correctness across fork->score->prune->carry.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as Fnn

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H

K = 4
INIT_STEPS = 2
SURVIVORS = 2
CONT_STEPS = 2
PROMPTS = C.TEST_PROMPTS[:3]


@torch.no_grad()
def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device

    all_rows = []
    per_prompt = []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        root_nl, root_cache, _, am_full = C.prefill(model, ids, am)
        first_tokens = root_nl[0].topk(K).indices.tolist()
        first_lp = Fnn.log_softmax(root_nl[0].float(), dim=-1)

        bcache = H.expand_universal_cache_for_branches(root_cache, K)
        bmask = am_full.repeat(K, 1)
        bids = ids.repeat(K, 1)
        states = []
        cum_logprob = []
        for i, ft in enumerate(first_tokens):
            bid = f"{p['id']}.b{i}"
            states.append(H.BranchState(
                branch_id=bid, parent_branch_id="root", root_branch_id=p["id"],
                input_ids=bids[i:i + 1].clone(), attention_mask=bmask[i:i + 1].clone(),
                past_key_values=bcache, cache_position=torch.tensor([ids.shape[1]], device=device),
                generated_ids=[ft], lineage_path=[p["id"], bid], score=None,
            ))
            cum_logprob.append(float(first_lp[ft].item()))
        # feed first tokens
        col = torch.tensor([[ft] for ft in first_tokens], device=device)
        bmask = H.update_attention_mask(bmask, 1)
        nl, bcache = C.decode_step(model, col, bcache, bmask)
        bids = torch.cat([bids, col], dim=1)

        # decode INIT_STEPS-1 more, accumulating logprob
        for _ in range(INIT_STEPS - 1):
            lp = Fnn.log_softmax(nl.float(), dim=-1)
            col = nl.argmax(dim=-1, keepdim=True)
            for i in range(K):
                cum_logprob[i] += float(lp[i, int(col[i, 0])].item())
            bmask = H.update_attention_mask(bmask, 1)
            nl, bcache = C.decode_step(model, col, bcache, bmask)
            bids = torch.cat([bids, col], dim=1)
            for i in range(K):
                states[i].generated_ids.append(int(col[i, 0]))

        # --- proxy selection: top-SURVIVORS by mean logprob ---
        scores = [cum_logprob[i] / max(1, len(states[i].generated_ids)) for i in range(K)]
        for i in range(K):
            states[i].score = scores[i]
        survivor_idx = sorted(range(K), key=lambda i: scores[i], reverse=True)[:SURVIVORS]

        pre_ids = [s.branch_id for s in states]
        beam = torch.tensor(survivor_idx, dtype=torch.long, device=device)
        bcache, states, bids, bmask = H.reorder_cache_and_branch_state(bcache, states, beam, bids, bmask)
        nl = nl.index_select(0, beam)
        lineage_ok = all(states[r].branch_id == pre_ids[survivor_idx[r]] for r in range(SURVIVORS))

        # continue survivors
        for _ in range(CONT_STEPS):
            col = nl.argmax(dim=-1, keepdim=True)
            bmask = H.update_attention_mask(bmask, 1)
            nl, bcache = C.decode_step(model, col, bcache, bmask)
            bids = torch.cat([bids, col], dim=1)
            for r in range(SURVIVORS):
                states[r].generated_ids.append(int(col[r, 0]))
                states[r].input_ids = bids[r:r + 1].clone()
                states[r].attention_mask = bmask[r:r + 1].clone()

        # validate survivors: live reordered-cache next logits vs full recompute
        rows = []
        for r in range(SURVIVORS):
            seq = bids[r:r + 1]
            full_logits = C.full_recompute_logits(model, seq, bmask[r:r + 1])[:, -1, :]
            cmp = C.compare_logits(nl[r], full_logits[0])
            chk = H.assert_branch_state_consistent(states[r])
            rows.append({"branch_id": states[r].branch_id, "lineage": states[r].lineage_path,
                         "generated": states[r].generated_ids, "score": states[r].score,
                         "lineage_aligned": chk["generated_tail_match"], **cmp})
            all_rows.append({"prompt_id": p["id"], "branch_id": states[r].branch_id,
                             "logit_rms": cmp["logit_rms"], "logit_max_abs": cmp["logit_max_abs"],
                             "top1_match": cmp["top1_match"], "score": states[r].score})
        eq = C.classify_equivalence(rows)
        per_prompt.append({
            "prompt_id": p["id"], "first_tokens": first_tokens, "scores": scores,
            "survivor_idx": survivor_idx, "lineage_ok": lineage_ok,
            "survivor_equivalence": eq, "survivors": rows,
            "scorer": "proxy_mean_token_logprob", "dualanchor_scoring": "skipped",
        })
        C.clear_cuda()

    survivors_faithful = all(pp["survivor_equivalence"]["cache_faithful"] for pp in per_prompt)
    lineage_ok = all(pp["lineage_ok"] for pp in per_prompt)
    max_rms = max(pp["survivor_equivalence"]["max_rms"] for pp in per_prompt)

    if survivors_faithful and lineage_ok:
        verdict = "BRANCH_PRUNE_CARRY_SMOKE_VALID"
    elif survivors_faithful:
        verdict = "CACHE_VALID_SELECTOR_SKIPPED"
    else:
        verdict = "SMOKE_MISMATCH"

    summary = {
        "verdict": verdict, "K": K, "survivors": SURVIVORS,
        "init_steps": INIT_STEPS, "cont_steps": CONT_STEPS,
        "survivors_faithful": survivors_faithful, "lineage_ok": lineage_ok, "max_rms": max_rms,
        "scorer": "proxy_mean_token_logprob", "dualanchor_scoring": "skipped (no tap registry touched)",
        "per_prompt": per_prompt,
        "note": "End-to-end fork->proxy-score->prune/reorder->carry survivors. Survivor "
                "caches validated against full recompute. DualAnchor scoring skipped by design.",
    }
    C.save_json("dualanchor_integration_smoke.json", summary)
    C.save_csv("dualanchor_integration_rows.csv", all_rows)

    md = ["# PART L - DualAnchor branch-carry integration smoke\n",
          f"**BG_AUTOREGRESSIVE_CACHE_DUALANCHOR_SMOKE_VERDICT = {verdict}**\n",
          f"- survivors faithful to full recompute: {survivors_faithful} (max rms {max_rms:.4g})",
          f"- lineage alignment ok: {lineage_ok}",
          "- scorer: proxy mean token logprob (DualAnchor scoring SKIPPED by design)\n",
          "## Per prompt\n"]
    for pp in per_prompt:
        md.append(f"### {pp['prompt_id']} survivors={pp['survivor_idx']} (of K={K})")
        for s in pp["survivors"]:
            md.append(f"- {s['branch_id']} score={s['score']:.4g} gen={s['generated']} "
                      f"rms={s['logit_rms']:.4g} top1={s['top1_match']} lineage_aligned={s['lineage_aligned']}")
        md.append("")
    C.save_md("dualanchor_integration_smoke.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_DUALANCHOR_SMOKE_VERDICT = {verdict}")
    print(f"survivors_faithful={survivors_faithful} lineage_ok={lineage_ok} max_rms={max_rms:.4g}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
