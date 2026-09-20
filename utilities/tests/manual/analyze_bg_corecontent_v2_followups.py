"""CoreContent v2 follow-ups (all on cached features unless noted):
(a) real-negatives-only heldout macro (reasoning/logic/alignment vs constructed coding/math)
(c) prune layer 47_L4 from the blockwise tap and re-confirm
(b1) coding heldout under all / compile-valid-only / syntax-only negatives (deflate the 0.93)
Read-only; no training, no extraction here. Writes followups.{json,md}.
"""
from __future__ import annotations
import json, time
from collections import defaultdict
import torch
import bg_corecontent_v2_common as v2
import bg_corecontent_v2_features as feat
import bg_corecontent_v2_models as M
import bg_core_tap_audit_v1_common as cc

OUT = v2.OUT_ROOT
CORE = v2.CORE_DOMAINS
SEMANTIC = {"wrong_operator", "wrong_loop_bound", "wrong_return_constant", "off_by_one", "return_none"}
SYNTAX = {"syntax_error"}


def _blockwise():
    return M.crafted_v2_policies().get("CoreContent_v2_blockwise", [])


def _hh():
    return cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", [])


def _top1_by_dom(channels, domains, split="heldout"):
    out = {}
    for d in domains:
        vals = []
        for g in M.groups(d, split):
            m = cc.group_selection_metrics(g, cc.policy_candidate_scores(g, channels))
            if m:
                vals.append(m["top1_oracle"])
        out[d] = vals
    return out


def _macro(d):  # d: {dom:[top1]}
    return cc.finite_mean([cc.finite_mean(v) for v in d.values() if v])


def main() -> int:
    t0 = time.time(); v2.ensure_root()
    blk = _blockwise(); hh = _hh()
    res = {}

    # (a) real-negatives-only macro
    real_doms = ("reasoning", "logic", "alignment")     # dataset-provided negatives
    constr_doms = ("coding", "math")                    # synthesized negatives
    blk_real = _top1_by_dom(blk, real_doms); hh_real = _top1_by_dom(hh, real_doms)
    blk_constr = _top1_by_dom(blk, constr_doms); hh_constr = _top1_by_dom(hh, constr_doms)
    a = {
        "real_neg_domains": list(real_doms),
        "blockwise_real_macro": round(_macro(blk_real), 4),
        "hh_real_macro": round(_macro(hh_real), 4),
        "blockwise_real_ci": [round(x, 4) for x in M.boot_macro_ci(blk_real)],
        "hh_real_ci": [round(x, 4) for x in M.boot_macro_ci(hh_real)],
        "blockwise_constructed_macro": round(_macro(blk_constr), 4),
        "hh_constructed_macro": round(_macro(hh_constr), 4),
        "per_domain": {d: {"blk": round(cc.finite_mean(blk_real.get(d, blk_constr.get(d, []))), 4)} for d in real_doms + constr_doms},
        "alignment_reasoning_only": {
            "blk": round(_macro({k: blk_real[k] for k in ("alignment", "reasoning")}), 4),
            "hh": round(_macro({k: hh_real[k] for k in ("alignment", "reasoning")}), 4)},
    }
    res["a_real_negatives"] = a

    # (c) prune layer 47
    blk_no47 = [c for c in blk if c[2] != "47_L4"]
    full = _top1_by_dom(blk, CORE); no47 = _top1_by_dom(blk_no47, CORE)
    c_ = {"full_blockwise_macro": round(_macro(full), 4), "no47_macro": round(_macro(no47), 4),
          "full_ci": [round(x, 4) for x in M.boot_macro_ci(full)],
          "no47_ci": [round(x, 4) for x in M.boot_macro_ci(no47)],
          "per_domain_full": {d: round(cc.finite_mean(full[d]), 4) for d in CORE},
          "per_domain_no47": {d: round(cc.finite_mean(no47[d]), 4) for d in CORE},
          "channels_full": len(blk), "channels_no47": len(blk_no47)}
    res["c_prune_l47"] = c_

    # (b1) coding under all / compile-valid-only / syntax-only negatives
    def coding_top1(channels, neg_filter):
        vals = []
        for g in M.groups("coding", "heldout"):
            pos = [cd for cd in g["candidates"] if cd["reward"] > 0]
            neg = [cd for cd in g["candidates"] if cd["reward"] <= 0 and neg_filter(cd.get("candidate_kind"))]
            if not pos or not neg:
                continue
            sub = {**g, "candidates": pos + neg}
            m = cc.group_selection_metrics(sub, cc.policy_candidate_scores(sub, channels))
            if m:
                vals.append(m["top1_oracle"])
        return round(cc.finite_mean(vals), 4), len(vals)
    filters = {"all": (lambda k: True),
               "compile_valid_only": (lambda k: k in SEMANTIC),
               "syntax_only": (lambda k: k in SYNTAX)}
    b1 = {}
    for fname, fn in filters.items():
        bt, bn = coding_top1(blk, fn); ht, hn = coding_top1(hh, fn)
        b1[fname] = {"blockwise_top1": bt, "hh_top1": ht, "groups": bn}
    res["b1_coding_negative_hardness"] = b1

    v2.write_json(OUT / "followups.json", {**res, "elapsed_seconds": round(time.time() - t0, 3)})
    lines = ["# CoreContent v2 follow-ups", "",
             "## (a) Real-negative vs constructed-negative heldout macro", "",
             f"- Real-negative domains (reasoning/logic/alignment): blockwise **{a['blockwise_real_macro']}** "
             f"{a['blockwise_real_ci']} vs HH **{a['hh_real_macro']}** {a['hh_real_ci']}.",
             f"- Constructed-negative domains (coding/math): blockwise {a['blockwise_constructed_macro']} vs HH {a['hh_constructed_macro']}.",
             f"- alignment+reasoning only: blockwise {a['alignment_reasoning_only']['blk']} vs HH {a['alignment_reasoning_only']['hh']}.", "",
             "## (c) Prune layer 47_L4", "",
             f"- full blockwise core macro {c_['full_blockwise_macro']} {c_['full_ci']} vs 24+36 only {c_['no47_macro']} {c_['no47_ci']}.",
             f"- per-domain full: {c_['per_domain_full']}",
             f"- per-domain no47: {c_['per_domain_no47']}", "",
             "## (b1) Coding heldout by negative hardness", "",
             *[f"- {k}: blockwise {v['blockwise_top1']} | HH {v['hh_top1']} ({v['groups']} groups)" for k, v in b1.items()]]
    v2.write_md(OUT / "followups.md", lines)
    print("=== (a) real-negative macro ===")
    print(f"  blockwise real(reason/logic/align)={a['blockwise_real_macro']} {a['blockwise_real_ci']} | HH={a['hh_real_macro']} {a['hh_real_ci']}")
    print(f"  blockwise constructed(coding/math)={a['blockwise_constructed_macro']} | HH={a['hh_constructed_macro']}")
    print(f"  alignment+reasoning only: blk={a['alignment_reasoning_only']['blk']} HH={a['alignment_reasoning_only']['hh']}")
    print("=== (c) prune L47 ===")
    print(f"  full={c_['full_blockwise_macro']} {c_['full_ci']} | no47={c_['no47_macro']} {c_['no47_ci']}")
    print(f"  no47 per-domain: {c_['per_domain_no47']}")
    print("=== (b1) coding negative hardness ===")
    for k, v in b1.items():
        print(f"  {k:20} blockwise={v['blockwise_top1']} HH={v['hh_top1']} (n={v['groups']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
