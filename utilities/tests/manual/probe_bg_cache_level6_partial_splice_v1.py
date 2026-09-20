"""PART I - Level 6: partial-cache splice diagnostic (stretch goal).

For a perturbation at (loop u, layer L), cache slots before the boundary are
provably shareable from the unperturbed root cache (the perturbation has not
happened yet at those points, so root and branch slots are bit-identical),
while slots at/after the boundary are branch-specific.

This probe builds an Option-A spliced cache (root for shared slots, branch for
affected slots), verifies it reproduces the full perturbed branch cache, and
continues generation. It is DIAGNOSTIC: Option A does not itself save compute
(the affected slots are copied from a full perturbed prefill). Real compute
saving (Option B: recompute only the affected suffix) would require model
surgery and is NOT implemented here, so NO compute-saving claim is made.

Conservative / boundary / aggressive splices are compared to map which slot
sharing assumptions hold.
"""

from __future__ import annotations

import json

import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H

LAYERS = [24, 36, 47]
PERTURB_LOOP = 0
# Large enough that the perturbation clearly propagates into the continuation,
# so the aggressive over-sharing negative control decisively diverges.
ALPHA = 2.0
N_STEPS = 4
PROMPTS = C.TEST_PROMPTS[:3]


@torch.no_grad()
def replay_from_cache(model, cache, am_full, tokens, device):
    """Force-replay tokens from a cache; return per-step next-token logits."""
    step_logits = []
    cur_am = am_full
    for t in tokens:
        cur_am = H.update_attention_mask(cur_am, 1)
        nl, cache = C.decode_step(model, torch.tensor([[t]], device=device), cache, cur_am)
        step_logits.append(nl[0].detach().float().cpu())
    return step_logits


def conservative_splice(root, branch, perturb_loop, num_ut, num_layers):
    """Share ONLY slots in loops strictly before perturb_loop (definitely-before);
    take branch for everything from perturb_loop onward."""
    cls = type(branch)
    spliced = cls(getattr(branch, "max_cache_size", None))
    n = max(len(root.key_cache), len(branch.key_cache))
    klist = [None] * n
    vlist = [None] * n
    n_shared = 0
    for slot in range(n):
        loop = slot // num_layers
        src = root if loop < perturb_loop else branch
        if loop < perturb_loop:
            n_shared += 1
        k = src.key_cache[slot] if slot < len(src.key_cache) else None
        v = src.value_cache[slot] if slot < len(src.value_cache) else None
        klist[slot] = None if k is None else k.detach().clone()
        vlist[slot] = None if v is None else v.detach().clone()
    spliced._key_cache = klist
    spliced._value_cache = vlist
    spliced._seen_tokens = getattr(branch, "_seen_tokens", 0)
    return spliced, n_shared


def aggressive_splice(root, branch, perturb_loop, perturb_layer, num_ut, num_layers):
    """Deliberately share TOO MANY slots: claim only the FINAL loop is affected
    (wrong, since perturb_loop=0 propagates through all loops). Used as a
    negative control -- continuation should diverge from the true branch."""
    spliced, classification = H.maybe_splice_cache_prefix(
        root, branch, num_ut - 1, perturb_layer, num_ut, num_layers
    )
    return spliced, len(classification["shared_slots"])


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    hidden = info["hidden_size"]
    num_ut = info["total_ut_steps"]
    num_layers = info["num_hidden_layers"]
    total_slots = info["expected_cache_slots"]
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(hidden, seed=6262, device=device, dtype=torch.bfloat16)

    all_rows = []
    per_config = []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        prompt_len = int(ids.shape[1])
        perturb_range = (max(1, prompt_len // 2), prompt_len)
        # unperturbed root cache (shared-slot source)
        root_nl, root_cache, _, am_full = C.prefill(model, ids, am)
        for layer in LAYERS:
            # perturbed branch prefill
            hook = C.LayerOutputPerturbHook(direction, ALPHA, token_range=perturb_range, target_loop=PERTURB_LOOP)
            handle = C.register_perturb_hook(model, layer, hook)
            branch_nl, branch_cache, _, am_full_b = C.prefill(model, ids, am)
            handle.remove()

            # boundary splice (Option A): root for shared, branch for affected
            spliced, classification = H.maybe_splice_cache_prefix(
                root_cache, branch_cache, PERTURB_LOOP, layer, num_ut, num_layers
            )
            cmp_cache = H.compare_universal_caches(spliced, branch_cache)
            slot_logic_exact = (cmp_cache["max_abs"] == 0.0 and cmp_cache["n_shape_mismatch"] == 0
                                and cmp_cache["n_missing_one"] == 0)

            # continuation: force the SAME tokens (branch greedy) through spliced,
            # branch, and full perturbed recompute.
            t0 = int(branch_nl[0].argmax().item())
            # greedy tokens from branch cache
            tokens = [t0]
            cur_am = H.update_attention_mask(am_full_b, 1)
            nl, bc_tmp = C.decode_step(model, torch.tensor([[t0]], device=device),
                                       H.clone_universal_cache(branch_cache), cur_am)
            for _ in range(N_STEPS - 1):
                t = int(nl[0].argmax().item())
                tokens.append(t)
                cur_am = H.update_attention_mask(cur_am, 1)
                nl, bc_tmp = C.decode_step(model, torch.tensor([[t]], device=device), bc_tmp, cur_am)

            spliced_logits = replay_from_cache(model, H.clone_universal_cache(spliced),
                                               am_full_b, tokens, device)
            branch_logits = replay_from_cache(model, H.clone_universal_cache(branch_cache),
                                              am_full_b, tokens, device)
            # full perturbed recompute per step
            full_logits = []
            for k in range(N_STEPS):
                seq = torch.cat([ids] + [torch.tensor([[t]], device=device) for t in tokens[:k + 1]], dim=1)
                msk = torch.ones((1, seq.shape[1]), dtype=torch.long, device=device)
                h2 = C.LayerOutputPerturbHook(direction, ALPHA, token_range=perturb_range, target_loop=PERTURB_LOOP)
                hd2 = C.register_perturb_hook(model, layer, h2)
                fl = C.full_recompute_logits(model, seq, msk)[:, -1, :]
                hd2.remove()
                full_logits.append(fl[0].detach().float().cpu())

            sp_vs_branch = C.classify_equivalence(
                [C.compare_logits(spliced_logits[k], branch_logits[k]) for k in range(N_STEPS)]
            )
            sp_vs_full = C.classify_equivalence(
                [C.compare_logits(spliced_logits[k], full_logits[k]) for k in range(N_STEPS)]
            )

            # conservative splice
            cons, cons_shared = conservative_splice(root_cache, branch_cache, PERTURB_LOOP, num_ut, num_layers)
            cons_cmp = H.compare_universal_caches(cons, branch_cache)
            cons_exact = (cons_cmp["max_abs"] == 0.0)

            # aggressive (negative) splice
            aggr, aggr_shared = aggressive_splice(root_cache, branch_cache, PERTURB_LOOP, layer, num_ut, num_layers)
            aggr_cmp = H.compare_universal_caches(aggr, branch_cache)
            aggr_logits = replay_from_cache(model, H.clone_universal_cache(aggr), am_full_b, tokens, device)
            aggr_vs_full = C.classify_equivalence(
                [C.compare_logits(aggr_logits[k], full_logits[k]) for k in range(N_STEPS)]
            )

            n_shared = classification["n_shared"]
            n_affected = classification["n_affected"]
            per_config.append({
                "prompt_id": p["id"], "layer": layer, "perturb_loop": PERTURB_LOOP, "alpha": ALPHA,
                "n_shared_slots": n_shared, "n_affected_slots": n_affected,
                "hypothetical_compute_saving_fraction": round(n_shared / total_slots, 4),
                "boundary_splice_reproduces_branch_cache_exactly": slot_logic_exact,
                "boundary_cache_compare_max_abs": cmp_cache["max_abs"],
                "spliced_vs_branch_continuation": sp_vs_branch,
                "spliced_vs_full_perturbed": sp_vs_full,
                "conservative_splice_exact": cons_exact, "conservative_shared": cons_shared,
                "aggressive_splice_cache_max_abs": aggr_cmp["max_abs"],
                "aggressive_shared": aggr_shared,
                "aggressive_vs_full_diverges": (not aggr_vs_full["cache_faithful"]),
                "aggressive_vs_full_max_abs": aggr_vs_full["max_abs"],
            })
            all_rows.append({
                "prompt_id": p["id"], "layer": layer, "n_shared": n_shared, "n_affected": n_affected,
                "boundary_exact": slot_logic_exact,
                "spliced_vs_full_rms": sp_vs_full["max_rms"],
                "aggressive_vs_full_max_abs": aggr_vs_full["max_abs"],
            })
            C.clear_cuda()

    boundary_logic_ok = all(c["boundary_splice_reproduces_branch_cache_exactly"] for c in per_config)
    conservative_ok = all(c["conservative_splice_exact"] for c in per_config)
    continuation_ok = all(c["spliced_vs_branch_continuation"]["cache_faithful"] for c in per_config)
    spliced_vs_full_ok = all(c["spliced_vs_full_perturbed"]["cache_faithful"] for c in per_config)
    aggressive_diverges = all(c["aggressive_vs_full_diverges"] for c in per_config)

    # Option A validates slot logic but does NOT save compute (Option B not implemented).
    if boundary_logic_ok and conservative_ok and continuation_ok and spliced_vs_full_ok:
        verdict = "SPLICE_SLOT_LOGIC_VALID_BUT_NO_COMPUTE_SAVING"
    elif boundary_logic_ok:
        verdict = "CONSERVATIVE_SPLICE_VALID"
    else:
        verdict = "SPLICE_INVALID"

    summary = {
        "verdict": verdict,
        "boundary_splice_reproduces_branch_cache": boundary_logic_ok,
        "conservative_splice_exact": conservative_ok,
        "spliced_continuation_matches_branch": continuation_ok,
        "spliced_continuation_matches_full_perturbed": spliced_vs_full_ok,
        "aggressive_oversharing_diverges_as_expected": aggressive_diverges,
        "compute_saving_implemented": False,
        "compute_saving_claimed": False,
        "layers": LAYERS, "perturb_loop": PERTURB_LOOP, "alpha": ALPHA, "n_steps": N_STEPS,
        "per_config": per_config,
        "note": "Boundary splice (Option A) reproduces the full perturbed branch cache "
                "bit-exactly because shared slots are identical in root and branch (the "
                "perturbation has not occurred yet there). This validates the shared/affected "
                "slot boundary logic. It copies affected slots from a full perturbed prefill, "
                "so it does NOT save compute. Real compute saving (recomputing only the "
                "affected suffix, Option B) would require model surgery and is NOT implemented. "
                "Aggressive over-sharing diverges, confirming affected slots are branch-specific.",
    }
    C.save_json("level6_partial_splice.json", summary)
    C.save_csv("level6_rows.csv", all_rows)

    md = ["# PART I - Level 6: partial-cache splice diagnostic\n",
          f"**BG_AUTOREGRESSIVE_CACHE_LEVEL6_VERDICT = {verdict}**\n",
          f"- boundary splice reproduces full branch cache exactly: {boundary_logic_ok}",
          f"- conservative splice exact: {conservative_ok}",
          f"- spliced continuation == branch continuation: {continuation_ok}",
          f"- spliced continuation == full perturbed recompute (within bf16): {spliced_vs_full_ok}",
          f"- aggressive over-sharing diverges (negative control): {aggressive_diverges}",
          "- **compute saving implemented: False; compute saving claimed: False**\n",
          "## Per config\n"]
    for c in per_config:
        md.append(f"- {c['prompt_id']} L{c['layer']} loop{c['perturb_loop']}: shared={c['n_shared_slots']} "
                  f"affected={c['n_affected_slots']} (hypothetical saving "
                  f"{c['hypothetical_compute_saving_fraction']*100:.1f}%) boundary_exact="
                  f"{c['boundary_splice_reproduces_branch_cache_exactly']} "
                  f"aggressive_vs_full_maxabs={c['aggressive_vs_full_max_abs']:.4g}")
    C.save_md("level6_partial_splice.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_LEVEL6_VERDICT = {verdict}")
    print(f"boundary_ok={boundary_logic_ok} conservative_ok={conservative_ok} "
          f"continuation_ok={continuation_ok} aggressive_diverges={aggressive_diverges}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
