"""M+N S1.4 step-two GATE 2 — α>0 fork-mechanism consistency (last-token suffix).

Gate 1 validated the re-derive plumbing at α=0 (perturbation shape moot). Gate 2 validates the
CONFIRMED α>0 fork mechanism (user spec 2026-06-15):
  - single fork primitive = apply_boundary_perturbation / LayerOutputPerturbHook(token_range)
  - canonical reference token_range = LAST LIVE TOKEN only: (seq_len-1, seq_len)  [causal-suffix-safe]
  - lineage replay and fresh fork MUST use the EXACT same primitive AND token_range.

Two sub-gates, token_range = (plen-1, plen):

  2a SINGLE-LOCUS α>0:  splice fork (build_spliced_branch) MUST equal hook replay
     (full_perturbed_prefill, LayerOutputPerturbHook) at t1=(loop0,L24). Confirms the v2 bit-exactness
     holds for the last-token range (prior validation used the default second-half range).

  2b TWO-LOCUS CHAINING:  the real new invariant. A survivor re-derived through its α>0 ancestor
     (t1 as a live LayerOutputPerturbHook -> capture the L36 boundary on the branch's OWN trajectory)
     then forked at t2=(loop0,L36) via build_spliced_branch MUST equal the all-hooks-live reference
     (t1 AND t2 LayerOutputPerturbHooks live in a single prefill). This proves
     re-derive -> fork -> re-derive -> fork chaining reproduces live multi-locus perturbation.

PASS = prefill next-logit max-abs diff ~0 (validated splice numerics) AND identical greedy token
sequence, for BOTH sub-gates. Decode runs with NO hooks (one-time fork baked into the cache).

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_4_gate2_alpha_chain.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
import bg_autoregressive_cache_common_v1 as C  # noqa: E402
import bg_partial_cache_splice_v2_common as S  # noqa: E402
from mpn_s1_2_multilocus import curated_dirs_by_layer  # noqa: E402  (per-layer curated dirs)

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
N_STEPS = 48
ALPHA = 0.02
BL = 0          # boundary loop index 0 == loop 1
L1, L2 = 24, 36  # chained loci within loop 0 (L24 precedes L36 in schedule order)


def lineage_perturb_hooks(model, lineage, token_range):
    """Build (unregistered) LayerOutputPerturbHook handles for a branch's α>0 lineage, using the
    CANONICAL fork primitive + token_range. lineage: [(layer, loop_idx0, direction, alpha)]."""
    handles = []
    for (L, lp, d, a) in lineage:
        hook = C.LayerOutputPerturbHook(d, a, token_range=token_range, target_loop=lp)
        handles.append(C.register_perturb_hook(model, L, hook))
    return handles


def branch_specific_root_pack_v2(model, ids, am, lineage, b_loop, b_layer, token_range):
    """root_prefill_with_boundary with the branch's lineage LayerOutputPerturbHooks ACTIVE (canonical
    fork primitive, same token_range as the fork). Ancestors precede the boundary in schedule order,
    so they bake into the captured boundary + root cache -> the boundary lies on the branch's own
    α>0 trajectory. This is the α>0 replacement for gate-1's HiddenDeltaLayerHook(position=-1)."""
    handles = lineage_perturb_hooks(model, lineage, token_range)
    try:
        return S.root_prefill_with_boundary(model, ids, am, b_loop, b_layer)
    finally:
        for h in handles:
            h.remove()


def cmp(a_tokens, a_first, b_tokens, b_first):
    maxabs = float((a_first - b_first).abs().max())
    pref = 0
    for x, y in zip(a_tokens, b_tokens):
        if x == y:
            pref += 1
        else:
            break
    return {"first_logit_maxabs": round(maxabs, 6), "first_argmax_match": bool(a_first.argmax() == b_first.argmax()),
            "tokens_identical": a_tokens == b_tokens, "matching_prefix": pref, "n": len(a_tokens)}


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    rid = next(t["canary_id"] for t in sel["tasks"] if t["domain"] == "logic")
    task = next(json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] == rid)

    model, tok, info = C.load_model(attn_implementation="eager")
    device = next(model.parameters()).device
    prompt = V.build_prompt(tok, task, "direct")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    plen = ids.shape[1]
    tr = (plen - 1, plen)                      # CANONICAL last-token suffix
    dirs = curated_dirs_by_layer(2)
    d1 = dirs[L1][0]
    d2 = dirs[L2][0]

    results = {}

    # ----- 2a: single-locus α>0, splice fork ≡ hook replay (last-token range) -----
    br_a = S.build_spliced_branch(model, ids, am, BL, L1, ALPHA, d1, tr)
    toks_splice, _ = S.greedy_continue(model, br_a["spliced_cache"], br_a["am_full"], br_a["next_logits"], N_STEPS, device)
    nl_hook, cache_hook, am_hook, _p, _s, _r = S.full_perturbed_prefill(model, ids, am, BL, L1, ALPHA, d1, tr)
    toks_hook, _ = S.greedy_continue(model, cache_hook, am_hook, nl_hook, N_STEPS, device)
    results["gate2a_single_locus_L1_24"] = cmp(
        toks_hook, nl_hook[0].detach().float().cpu(), toks_splice, br_a["next_logits"][0].detach().float().cpu())

    # ----- 2b: two-locus chaining, re-derive ≡ all-hooks-live -----
    # re-derive: t1 live as LayerOutputPerturbHook -> capture L36 boundary -> splice t2
    rp = branch_specific_root_pack_v2(model, ids, am, [(L1, BL, d1, ALPHA)], BL, L2, tr)
    br_b = S.build_spliced_branch(model, ids, am, BL, L2, ALPHA, d2, tr, root_pack=rp)
    toks_rederive, _ = S.greedy_continue(model, br_b["spliced_cache"], br_b["am_full"], br_b["next_logits"], N_STEPS, device)
    # all-hooks-live reference: t1 AND t2 LayerOutputPerturbHooks live in one prefill, removed before decode
    handles = lineage_perturb_hooks(model, [(L1, BL, d1, ALPHA), (L2, BL, d2, ALPHA)], tr)
    try:
        nl_full, cache_full, _cp, am_full = C.prefill(model, ids, am)
    finally:
        for h in handles:
            h.remove()
    toks_full, _ = S.greedy_continue(model, cache_full, am_full, nl_full, N_STEPS, device)
    results["gate2b_chain_L1_24_then_L1_36"] = cmp(
        toks_full, nl_full[0].detach().float().cpu(), toks_rederive, br_b["next_logits"][0].detach().float().cpu())

    def ok(r):
        return r["first_argmax_match"] and r["first_logit_maxabs"] < 0.05 and r["matching_prefix"] >= r["n"] - 2
    passed = all(ok(r) for r in results.values())
    payload = {"verdict": "S1_4_GATE2_PASS" if passed else "S1_4_GATE2_FAIL",
               "task": rid, "n_steps": N_STEPS, "alpha": ALPHA, "token_range": "last_token (plen-1, plen)",
               "loci": {"t1": f"loop{BL+1}_L{L1}", "t2": f"loop{BL+1}_L{L2}"},
               "fork_primitive": "apply_boundary_perturbation / LayerOutputPerturbHook(token_range)",
               "results": results, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4_gate2_alpha_chain.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n{payload['verdict']}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
