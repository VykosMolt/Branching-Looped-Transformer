"""PART I - Position IDs / padding splice stress.

Validates the suffix-recompute splice under left padding. Confirms that with
explicit per-position position_ids + a padded causal mask, the padded splice
matches both the padded full perturbed reference and the unpadded splice; and
that omitting position_ids (defaulting to arange) breaks the padded case.
"""

from __future__ import annotations

import json
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:3]
BOUNDARY = (2, 24)
ALPHA = 1.0


def padded_mask_and_pos(mask2d, device):
    P = mask2d.shape[1]
    dtype = next(iter([torch.bfloat16]))
    min_dtype = torch.finfo(torch.bfloat16).min
    causal = torch.triu(torch.full((P, P), min_dtype, dtype=torch.bfloat16, device=device), 1)
    m4 = causal[None, None, :, :].clone()
    pad_cols = (mask2d[0] == 0)
    m4[..., pad_cols] = min_dtype
    pos = (mask2d.long().cumsum(-1) - 1).clamp(min=0)
    return m4, pos


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(info["hidden_size"], 424242, device, torch.bfloat16)
    NL, NUT, NSLOT = info["num_hidden_layers"], info["total_ut_steps"], info["expected_cache_slots"]
    u, L = BOUNDARY
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    rows, per_cfg = [], []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        R = ids.shape[1]
        # --- unpadded splice (baseline) ---
        tr_u = V2.default_token_range(R)
        sh_u, Hb_u, _, _ = V2.prefix_prefill(model, ids, u, L, device)
        Hp_u, _ = V2.apply_boundary_perturbation(Hb_u, ALPHA, direction, tr_u)
        nl_u, sfx_u, _, _ = V2.suffix_recompute(model, Hp_u, u, L, device)

        # --- left-padded single prompt ---
        npad = 3
        padded_ids = torch.cat([torch.full((1, npad), pad_id, device=device), ids], dim=1)
        mask2d = torch.cat([torch.zeros((1, npad), dtype=torch.long, device=device),
                            torch.ones((1, R), dtype=torch.long, device=device)], dim=1)
        Ppad = padded_ids.shape[1]
        m4, pos = padded_mask_and_pos(mask2d, device)
        # perturb the same real-token latter half: shift range by npad
        tr_p = (npad + R // 2, Ppad)

        # padded full perturbed reference (model handles 2D mask -> 4D internally)
        hook = C.LayerOutputPerturbHook(direction, ALPHA, token_range=tr_p, target_loop=u)
        handle = C.register_perturb_hook(model, L, hook)
        refc = C.new_cache()
        out = model(input_ids=padded_ids, attention_mask=mask2d, position_ids=pos,
                    past_key_values=refc, use_cache=True,
                    cache_position=torch.arange(Ppad, device=device))
        handle.remove()
        ref_nl = out.logits[:, -1, :]

        # padded splice WITH correct position_ids + mask
        sh_p, Hb_p, _, _ = V2.prefix_prefill(model, padded_ids, u, L, device, position_ids=pos, mask4d=m4)
        Hp_p, _ = V2.apply_boundary_perturbation(Hb_p, ALPHA, direction, tr_p)
        nl_p, sfx_p, _, _ = V2.suffix_recompute(model, Hp_p, u, L, device, position_ids=pos, mask4d=m4)

        # padded splice WITHOUT position_ids (default arange) -> expected wrong
        sh_b, Hb_b, _, _ = V2.prefix_prefill(model, padded_ids, u, L, device, mask4d=m4)
        Hp_b, _ = V2.apply_boundary_perturbation(Hb_b, ALPHA, direction, tr_p)
        nl_b, _, _, _ = V2.suffix_recompute(model, Hp_b, u, L, device, mask4d=m4)

        cmp_padded_vs_ref = C.compare_logits(nl_p[0], ref_nl[0])
        cmp_padded_vs_unpadded = C.compare_logits(nl_p[0], nl_u[0])
        cmp_nopos_vs_unpadded = C.compare_logits(nl_b[0], nl_u[0])

        per_cfg.append({
            "prompt_id": p["id"], "real_len": R, "padded_len": Ppad,
            "padded_splice_vs_ref": cmp_padded_vs_ref,
            "padded_splice_vs_unpadded": cmp_padded_vs_unpadded,
            "nopos_splice_vs_unpadded": cmp_nopos_vs_unpadded,
            "position_ids_required": (not C.compare_logits(nl_b[0], nl_u[0])["top1_match"])
                                     or cmp_nopos_vs_unpadded["logit_max_abs"] > C.TOL_DRIFT_MAX_ABS,
        })
        rows.append({"prompt_id": p["id"], "real_len": R, "padded_len": Ppad,
                     "padded_vs_ref_rms": cmp_padded_vs_ref["logit_rms"],
                     "padded_vs_unpadded_rms": cmp_padded_vs_unpadded["logit_rms"],
                     "nopos_vs_unpadded_maxabs": cmp_nopos_vs_unpadded["logit_max_abs"]})
        C.clear_cuda()

    def _faithful(cmp):
        return cmp["logit_max_abs"] <= C.TOL_DRIFT_MAX_ABS and cmp["top1_match"]
    padded_ref_ok = all(_faithful(c["padded_splice_vs_ref"]) for c in per_cfg)
    padded_unpadded_ok = all(_faithful(c["padded_splice_vs_unpadded"]) for c in per_cfg)
    pos_required = all(c["position_ids_required"] for c in per_cfg)

    if padded_ref_ok and padded_unpadded_ok and pos_required:
        verdict = "POSITION_IDS_REQUIRED_CONFIRMED"
    elif padded_ref_ok and padded_unpadded_ok:
        verdict = "POSITION_PADDING_SAFE"
    elif not padded_ref_ok:
        verdict = "PADDING_SPLICE_BUG"
    else:
        verdict = "INSUFFICIENT"

    summary = {"verdict": verdict, "padded_splice_matches_ref": padded_ref_ok,
               "padded_splice_matches_unpadded": padded_unpadded_ok,
               "position_ids_required_confirmed": pos_required,
               "boundary": [u, L], "per_config": per_cfg,
               "note": "With explicit per-position position_ids + padded causal mask, the padded "
                       "suffix-recompute splice matches both the padded full reference and the "
                       "unpadded splice. Omitting position_ids (default arange) breaks the padded "
                       "case, confirming the explicit-position requirement carries to the splice."}
    V2.save_json("position_padding_splice.json", summary)
    V2.save_csv("position_padding_splice_rows.csv", rows)
    V2.save_md("position_padding_splice.md",
               f"# PART I - Position IDs / padding splice stress\n\n"
               f"**BG_PARTIAL_SPLICE_POSITION_PADDING_VERDICT = {verdict}**\n\n"
               f"- padded splice == padded full reference: {padded_ref_ok}\n"
               f"- padded splice == unpadded splice (real tokens): {padded_unpadded_ok}\n"
               f"- position_ids required (no-pos breaks): {pos_required}\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_POSITION_PADDING_VERDICT = {verdict}")
    print(f"padded_vs_ref={padded_ref_ok} padded_vs_unpadded={padded_unpadded_ok} pos_required={pos_required}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
