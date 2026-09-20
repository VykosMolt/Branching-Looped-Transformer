"""Premise probe: is DualAnchor's branch-action signal OPPOSING content selection,
and does removing it strengthen content selection — examined at EVERY layer 24/36/47.

Read-only, CPU, on existing artifacts:
  - features: shared/data/corecontent_v2/features/*_heldout_*.pt  (clean reward-diverse groups,
    candidates carry features[3 layers(24,36,47) x 4 loops x 2048] + reward)
  - w_CC (content): corecontent_v2_policy.pt channels (24_L4, 36_L4; L47 pruned for content)
  - w_DA (branch+action): MIX_CODE_REASONING + MIX_OBJECTIVE_ALL AntisymLinear heads at 24/36/47

Tests, per layer and combined:
  1. cos(w_DA, w_CC) per layer.
  2. content top1-oracle of w_CC vs w_DA vs the action-residual (w_DA ⟂ w_CC).
  3. does dropping the 47 branch-action component raise content selection (the ceiling).
No training, no generation, no steering.
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROJECT = _HERE.parents[2]
LAYIDX = {24: 0, 36: 1, 47: 2}
LOOP4 = 3  # L4 converged checkpoint
REAL_NEG = ("reasoning", "logic", "alignment")  # cleanest (real negatives) per docs
POLICY = PROJECT / "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/corecontent_v2_policy.pt"
HEADS = PROJECT / "opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.pt"
FEATDIR = PROJECT / "shared/data/corecontent_v2/features"
OUT = PROJECT / "opi/taps/probes/branch_training_offline_verifier_generator_v2_2026-06-07"


def _unit(v):
    n = v.norm()
    return v / n if n > 1e-12 else v


def load_w_cc():
    pol = torch.load(POLICY, map_location="cpu", weights_only=False)
    out = {}
    for w, arch, cfg in pol["channels"]["channels"]:
        layer = int(cfg.split("_")[0])
        out[layer] = w.float().reshape(-1)  # [2048]
    return out  # {24:.., 36:..}


def load_w_da():
    mh = torch.load(HEADS, map_location="cpu", weights_only=False)
    heads = mh["heads"]
    out = {}
    for layer in (24, 36, 47):
        ws = [h["state_dict"]["linear.weight"].float().reshape(-1)
              for h in heads if h.get("head_group") in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL")
              and h.get("config") == f"{layer}_L4"]
        if ws:
            out[layer] = torch.stack(ws).mean(0)  # average the two anchors (+ their seeds)
    return out  # {24,36,47}


def load_groups():
    groups = []
    for dom in REAL_NEG:
        for f in sorted(glob.glob(str(FEATDIR / f"{dom}_heldout_*.pt"))):
            d = torch.load(f, map_location="cpu", weights_only=False)
            for g in d["groups"]:
                cands = g["candidates"]
                rew = torch.tensor([float(c["reward"]) for c in cands])
                if len(cands) < 2 or rew.max() <= rew.min():
                    continue  # need reward-diverse
                feats = torch.stack([c["features"].float() for c in cands])  # [C,3,4,2048]
                pos = (rew == rew.max())
                groups.append({"domain": dom, "feats": feats, "pos": pos, "n": len(cands)})
    return groups


def _layer_feat(feats, layer):
    return feats[:, LAYIDX[layer], LOOP4, :]  # [C, 2048]


def top1_oracle(score_fn, groups):
    """fraction of groups whose top-scored candidate is a positive; + random baseline."""
    hit = 0
    rand = 0.0
    for g in groups:
        s = score_fn(g)  # [C]
        top = int(torch.argmax(s))
        hit += int(g["pos"][top])
        rand += float(g["pos"].sum()) / g["n"]
    n = max(1, len(groups))
    return round(hit / n, 4), round(rand / n, 4)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    w_cc = load_w_cc()
    w_da = load_w_da()
    groups = load_groups()
    print(f"groups (real-neg heldout): {len(groups)} | w_cc layers {sorted(w_cc)} | w_da layers {sorted(w_da)}", flush=True)

    # per-layer: cos(DA,CC) + content top1-oracle of CC, DA, and the action-residual (DA ⟂ CC)
    per_layer = []
    for L in (24, 36, 47):
        da = w_da.get(L)
        cc = w_cc.get(L)
        row = {"layer": L}
        if cc is not None and da is not None:
            row["cos_DA_CC"] = round(float(torch.dot(_unit(da), _unit(cc))), 4)
            ccu = _unit(cc)
            resid = da - torch.dot(da, ccu) * ccu  # branch-action residual ⟂ content
            row["DA_content_top1"], rb = top1_oracle(lambda g, d=L, w=da: _layer_feat(g["feats"], d) @ w, groups)
            row["CC_content_top1"], _ = top1_oracle(lambda g, d=L, w=cc: _layer_feat(g["feats"], d) @ w, groups)
            row["actionResid_top1"], _ = top1_oracle(lambda g, d=L, w=resid: _layer_feat(g["feats"], d) @ w, groups)
            row["random"] = rb
            row["resid_opposes"] = row["actionResid_top1"] < rb  # below chance = opposing content
        else:
            row["note"] = "no CoreContent channel at this layer (L47 pruned for content)"
            if da is not None:
                row["DA_content_top1"], row["random"] = top1_oracle(
                    lambda g, d=L, w=da: _layer_feat(g["feats"], d) @ w, groups)
        per_layer.append(row)

    # combined content selectors: CoreContent (24+36) vs DualAnchor (24+36+47) vs DA(24+36 only)
    def cc_block(g):  # CoreContent: 24+36
        return sum(_layer_feat(g["feats"], L) @ w_cc[L] for L in (24, 36))

    def da_full(g):   # DualAnchor across all 3 layers
        return sum(_layer_feat(g["feats"], L) @ w_da[L] for L in (24, 36, 47))

    def da_no47(g):   # DualAnchor with the 47 branch-action layer removed
        return sum(_layer_feat(g["feats"], L) @ w_da[L] for L in (24, 36))

    def da_47only(g):  # the 47 branch-action signal alone
        return _layer_feat(g["feats"], 47) @ w_da[47]

    combo = {}
    combo["CoreContent_24_36"], combo["random"] = top1_oracle(cc_block, groups)
    combo["DualAnchor_24_36_47"], _ = top1_oracle(da_full, groups)
    combo["DualAnchor_24_36_drop47"], _ = top1_oracle(da_no47, groups)
    combo["DualAnchor_47only_action"], _ = top1_oracle(da_47only, groups)
    combo["lift_dropping_47_from_DA"] = round(combo["DualAnchor_24_36_drop47"] - combo["DualAnchor_24_36_47"], 4)
    combo["47action_opposes_content"] = combo["DualAnchor_47only_action"] < combo["random"]

    # verdict
    opp = (combo["47action_opposes_content"] or any(r.get("resid_opposes") for r in per_layer)) \
        and combo["lift_dropping_47_from_DA"] >= 0
    verdict = "OPPOSING_SIGNAL_CONFIRMED" if opp else "NO_OPPOSING_SIGNAL"

    import json
    payload = {"BRANCH_ACTION_VS_CONTENT_VERDICT": verdict, "n_groups": len(groups),
               "domains": list(REAL_NEG), "per_layer": per_layer, "combined": combo}
    (OUT / "branch_action_vs_content_opposition.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== per-layer (content top1-oracle) ===")
    for r in per_layer:
        print(" ", r)
    print("\n=== combined ===")
    for k, v in combo.items():
        print(f"  {k}: {v}")
    print(f"\nBRANCH_ACTION_VS_CONTENT_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
