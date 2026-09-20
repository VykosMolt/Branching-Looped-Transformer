"""PART G - Level 4: current-token layer perturbation during autoregressive decode.

After an unperturbed prompt prefill, clone a branch cache and apply a deterministic
hidden perturbation at decoder layer 24/36/47 (UT loop targeted via current_ut)
to the CURRENT generated token's hidden state, then continue generation with that
branch-specific cache. Validate the perturbed cached branch against a full
recomputation that re-applies the SAME perturbation at the same absolute token
position and (loop, layer) each step.

Deterministic. float32 logit comparison. Includes zero-perturbation control.
"""

from __future__ import annotations

import json

import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H

LAYERS = [24, 36, 47]
# Perturb at the FIRST UT loop so the injected change propagates through the
# remaining loops and is stored in many downstream cache slots -- i.e. it
# genuinely produces a branch-specific cache that must be carried forward.
# (Perturbing the final loop's last layer would change only the immediate
# next-token logits and store no downstream K/V -- nothing to carry.)
TARGET_LOOP = 0
# This model's residual-stream per-element RMS at the target layers is ~0.1-0.5,
# so the originally-suggested alphas (0.001-0.01) inject perturbations far below
# the bf16 logit noise floor (invisible). We sweep larger fractions; alpha>=1.0
# produces a clearly visible logit effect (maxabs ~1-17) that full recompute must
# reproduce. cached-vs-full drift is pure q=1/q=seq rounding and stays bf16-scale
# regardless of perturbation magnitude (the perturbation is identical in both paths).
ALPHAS = [0.0, 0.5, 1.0, 2.0]
N_STEPS = 4
PROMPTS = C.TEST_PROMPTS[:3]


@torch.no_grad()
def cached_perturbed_branch(model, root_cache, root_next_logits, ids, am_full,
                            layer, alpha, n_steps, direction):
    """Decode a branch where the FIRST generated token gets a current-token
    perturbation at (layer, TARGET_LOOP). Subsequent steps are unperturbed.
    Returns (tokens, per_step_logits, perturb_rms, had_effect)."""
    device = ids.device
    branch = H.clone_universal_cache(root_cache)
    t0 = int(root_next_logits[0].argmax().item())

    cur_am = H.update_attention_mask(am_full, 1)
    hook = C.LayerOutputPerturbHook(direction, alpha, token_index=-1, target_loop=TARGET_LOOP)
    handle = C.register_perturb_hook(model, layer, hook)
    nl, branch = C.decode_step(model, torch.tensor([[t0]], device=device), branch, cur_am)
    handle.remove()
    perturb_rms = hook.last_perturb_rms

    # had-effect: same step on a fresh clone WITHOUT the hook
    unp = H.clone_universal_cache(root_cache)
    nl_unp, _ = C.decode_step(model, torch.tensor([[t0]], device=device), unp,
                              H.update_attention_mask(am_full, 1))
    effect = C.compare_logits(nl[0], nl_unp[0])
    had_effect = effect["logit_max_abs"] > C.TOL_DRIFT_MAX_ABS if alpha > 0 else None

    tokens = [t0]
    step_logits = [nl[0].detach().float().cpu()]
    next_logits = nl
    for _ in range(n_steps - 1):
        t = int(next_logits[0].argmax().item())
        tokens.append(t)
        cur_am = H.update_attention_mask(cur_am, 1)
        next_logits, branch = C.decode_step(model, torch.tensor([[t]], device=device), branch, cur_am)
        step_logits.append(next_logits[0].detach().float().cpu())
    return tokens, step_logits, perturb_rms, had_effect, effect["logit_max_abs"]


@torch.no_grad()
def full_recompute_perturbed(model, ids, am, tokens, perturb_abs_pos, layer, alpha,
                             direction, n_steps):
    """Full no-cache recompute for each step, re-applying the perturbation at the
    fixed absolute token position `perturb_abs_pos` at (layer, TARGET_LOOP)."""
    device = ids.device
    step_logits = []
    for k in range(n_steps):
        seq = torch.cat(
            [ids] + [torch.tensor([[t]], device=device) for t in tokens[:k + 1]], dim=1
        )
        msk = torch.ones((1, seq.shape[1]), dtype=torch.long, device=device)
        hook = C.LayerOutputPerturbHook(direction, alpha, token_index=perturb_abs_pos,
                                        target_loop=TARGET_LOOP)
        handle = C.register_perturb_hook(model, layer, hook)
        logits = C.full_recompute_logits(model, seq, msk)[:, -1, :]
        handle.remove()
        step_logits.append(logits[0].detach().float().cpu())
    return step_logits


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    hidden = info["hidden_size"]
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(hidden, seed=4242, device=device, dtype=torch.bfloat16)

    all_rows = []
    per_config = []
    hook_targeting_ok = True
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        prompt_len = int(ids.shape[1])
        root_next_logits, root_cache, _, am_full = C.prefill(model, ids, am)
        perturb_abs_pos = prompt_len  # position of the first generated token t0
        for layer in LAYERS:
            for alpha in ALPHAS:
                tokens, cached_logits, prms, had_effect, effect_abs = cached_perturbed_branch(
                    model, root_cache, root_next_logits, ids, am_full, layer, alpha, N_STEPS, direction
                )
                full_logits = full_recompute_perturbed(
                    model, ids, am, tokens, perturb_abs_pos, layer, alpha, direction, N_STEPS
                )
                rows = []
                for k in range(N_STEPS):
                    cmp = C.compare_logits(cached_logits[k], full_logits[k])
                    rows.append(cmp)
                    all_rows.append({
                        "prompt_id": p["id"], "layer": layer, "loop": TARGET_LOOP,
                        "alpha": alpha, "step": k + 1,
                        "logit_rms": cmp["logit_rms"], "logit_max_abs": cmp["logit_max_abs"],
                        "top1_match": cmp["top1_match"], "top5_overlap": cmp["top5_overlap"],
                    })
                eq = C.classify_equivalence(rows)
                aff = H.affected_cache_slots_for_boundary(TARGET_LOOP, layer, info["total_ut_steps"], info["num_hidden_layers"])
                if alpha > 0 and had_effect is False:
                    hook_targeting_ok = False
                per_config.append({
                    "prompt_id": p["id"], "layer": layer, "loop": TARGET_LOOP, "alpha": alpha,
                    "tokens": tokens, "perturb_rms": prms,
                    "had_effect": had_effect, "perturb_vs_unperturbed_max_abs": effect_abs,
                    "equivalence": eq,
                    "affected_slots": aff["n_affected"], "shared_slots": aff["n_shared"],
                })
                C.clear_cuda()

    # control (alpha=0) and perturbed (alpha>0) breakdown
    perturbed_cfgs = [c for c in per_config if c["alpha"] > 0]
    control_cfgs = [c for c in per_config if c["alpha"] == 0]
    perturbed_faithful = all(c["equivalence"]["cache_faithful"] for c in perturbed_cfgs)
    control_faithful = all(c["equivalence"]["cache_faithful"] for c in control_cfgs)
    # hook works if, for each layer, the largest alpha produced a clearly visible
    # logit effect (> bf16 noise floor). Small alphas may be sub-noise.
    effective = [c for c in perturbed_cfgs if c["had_effect"]]
    layers_with_effect = {c["layer"] for c in effective}
    hook_works = layers_with_effect == set(LAYERS)
    strict = all(c["equivalence"]["strict_equiv"] for c in perturbed_cfgs)
    max_rms = max(c["equivalence"]["max_rms"] for c in perturbed_cfgs)
    max_abs = max(c["equivalence"]["max_abs"] for c in perturbed_cfgs)

    if not hook_works:
        verdict = "HOOK_TARGETING_LIMITED"
    elif not (perturbed_faithful and control_faithful):
        verdict = "PERTURB_CACHE_MISMATCH"
    elif strict:
        verdict = "CURRENT_TOKEN_LAYER_PERTURB_CARRY_VALID"
    else:
        verdict = "CURRENT_TOKEN_PERTURB_NUMERIC_DRIFT_SMALL"

    summary = {
        "verdict": verdict, "layers": LAYERS, "target_loop": TARGET_LOOP, "alphas": ALPHAS,
        "n_steps": N_STEPS, "perturbed_faithful": perturbed_faithful,
        "control_faithful": control_faithful, "hook_works": hook_works, "layers_with_effect": sorted(layers_with_effect),
        "max_rms": max_rms, "max_abs": max_abs, "per_config": per_config,
        "note": "Perturbation injected at the OUTPUT of (loop=TARGET_LOOP, layer) on the "
                "current token; full recompute re-applies it at the same fixed absolute "
                "position. Affected cache slots = loop TARGET_LOOP layers > perturb_layer.",
    }
    C.save_json("level4_current_token_perturb.json", summary)
    C.save_csv("level4_rows.csv", all_rows)

    md = ["# PART G - Level 4: current-token layer perturbation during decode\n",
          f"**BG_AUTOREGRESSIVE_CACHE_LEVEL4_VERDICT = {verdict}**\n",
          f"- perturbed branches faithful to full recompute: {perturbed_faithful}",
          f"- zero-perturbation control faithful: {control_faithful}",
          f"- hook works (each layer shows visible effect at max alpha): {hook_works}",
          f"- max perturbed RMS {max_rms:.4g}, max-abs {max_abs:.4g}",
          f"- target loop (current_ut) = {TARGET_LOOP}\n",
          "## Per config\n"]
    for c in per_config:
        md.append(f"- {c['prompt_id']} L{c['layer']} loop{c['loop']} a={c['alpha']}: "
                  f"effect_maxabs={c['perturb_vs_unperturbed_max_abs']:.4g} had_effect={c['had_effect']} "
                  f"faithful={c['equivalence']['cache_faithful']} rms={c['equivalence']['max_rms']:.4g} "
                  f"affected_slots={c['affected_slots']}")
    C.save_md("level4_current_token_perturb.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_LEVEL4_VERDICT = {verdict}")
    print(f"perturbed_faithful={perturbed_faithful} control_faithful={control_faithful} "
          f"hook_works={hook_works} max_rms={max_rms:.4g} max_abs={max_abs:.4g}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
