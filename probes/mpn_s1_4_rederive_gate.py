"""M+N S1.4 step-two GATE 1 — branch-specific re-derive zero-perturb equivalence.

Per the chaining spec: a survivor becomes the new root for its descendants; the next splice
boundary must be captured from the branch's OWN execution trajectory (re-derive with all lineage
perturbations active), never the clean root. Before building the α>0 chained loop, validate the
re-derive + fork + continue PLUMBING is identity-correct at α=0 (where perturbation shape is moot):

  - CLEAN            : C.prefill -> greedy_continue
  - REDERIVE_EMPTY   : branch-specific root_pack with EMPTY lineage @(loop1,L24) -> build_spliced_branch(alpha=0) -> continue
  - REDERIVE_A0_LIN  : lineage=[(L24,loop1,dir,alpha=0)] (no-op hook) @(loop1,L36) -> build_spliced_branch(alpha=0) -> continue

All three must match CLEAN (prefill bit-exact; decode within bf16 band; token sequences agree).
This proves the re-derive machinery is identity-correct so the α>0 chained loop builds on solid ground.

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_4_rederive_gate.py
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
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, rms_normalize  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
N_STEPS = 48


def branch_specific_root_pack(model, ids, am, lineage, b_loop, b_layer):
    """root_prefill_with_boundary with the branch's lineage perturbation hooks ACTIVE, so the
    captured boundary + root cache lie on the branch's own trajectory (ancestors precede the
    boundary in schedule order, baked into the prefix). lineage: [(layer, loop, direction, alpha)]."""
    hooks = [HiddenDeltaLayerHook(model, target_layer=L, target_loops=[lp], delta=d * a,
                                  position=-1, max_rms_fraction=min(max(a, 1e-9), 0.10))
             for (L, lp, d, a) in lineage]
    for h in hooks:
        h.apply()
    try:
        return S.root_prefill_with_boundary(model, ids, am, b_loop, b_layer)
    finally:
        for h in hooks:
            h.remove()


def cmp(a_tokens, a_first, b_tokens, b_first):
    maxabs = float((a_first - b_first).abs().max())
    same_first_argmax = bool(a_first.argmax() == b_first.argmax())
    tok_match = a_tokens == b_tokens
    pref = 0
    for x, y in zip(a_tokens, b_tokens):
        if x == y:
            pref += 1
        else:
            break
    return {"first_logit_maxabs": round(maxabs, 5), "first_argmax_match": same_first_argmax,
            "tokens_identical": tok_match, "matching_prefix": pref, "n": len(a_tokens)}


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
    tr = S.default_token_range(plen)
    hidden = model.config.hidden_size
    zero = torch.zeros(hidden, dtype=torch.float32)
    rdir = rms_normalize(torch.randn(hidden))

    # CLEAN reference
    nl_c, cache_c, _cp, am_c = C.prefill(model, ids, am)
    clean_tokens, _ = S.greedy_continue(model, cache_c, am_c, nl_c, N_STEPS, device)
    clean_first = nl_c[0].detach().float().cpu()

    results = {}
    # REDERIVE_EMPTY: empty lineage, boundary (loop1=idx0, L24), alpha=0 fork
    rp = branch_specific_root_pack(model, ids, am, [], 0, 24)
    br = S.build_spliced_branch(model, ids, am, 0, 24, 0.0, zero, tr, root_pack=rp)
    toks, _ = S.greedy_continue(model, br["spliced_cache"], br["am_full"], br["next_logits"], N_STEPS, device)
    results["rederive_empty_L1_24"] = cmp(clean_tokens, clean_first, toks, br["next_logits"][0].detach().float().cpu())

    # REDERIVE_A0_LIN: lineage with one alpha=0 ancestor @L24/loop1, boundary (loop1, L36), alpha=0 fork
    rp2 = branch_specific_root_pack(model, ids, am, [(24, 1, rdir, 0.0)], 0, 36)
    br2 = S.build_spliced_branch(model, ids, am, 0, 36, 0.0, zero, tr, root_pack=rp2)
    toks2, _ = S.greedy_continue(model, br2["spliced_cache"], br2["am_full"], br2["next_logits"], N_STEPS, device)
    results["rederive_alpha0_lineage_L1_36"] = cmp(clean_tokens, clean_first, toks2, br2["next_logits"][0].detach().float().cpu())

    # gate: bit-exact-ish first logit (splice prefill is exact) + identical token prefix (allow bf16 tail drift)
    def ok(r):
        return r["first_argmax_match"] and r["first_logit_maxabs"] < 0.05 and r["matching_prefix"] >= r["n"] - 2
    passed = all(ok(r) for r in results.values())
    payload = {"verdict": "S1_4_GATE1_PASS" if passed else "S1_4_GATE1_FAIL",
               "task": rid, "n_steps": N_STEPS, "results": results,
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4_gate1_rederive.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n{payload['verdict']}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
