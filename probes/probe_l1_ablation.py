"""L1 ablation: does loop-1's distinct geometry carry head-visible signal?

Context. v10 (loop-geometry probe) showed Ouro's loop boundary states are
bipartite on HH text: L1 is geometrically distinct (L1<->L4 cos = 0.72)
while L2-L4 cluster tightly (L3<->L4 cos = 0.996). The published 5M-param
GRU head is boundary-style (it reads a single loop state replicated into
its 4-slot input). Before the debiased Experiment-2 retrain we want to
know whether L1 carries readable relational signal the published
architecture leaves on the table.

This is pure post-hoc analysis on the saved v10 Thinking states. No model
forward. The head weights are NOT modified - zero-shot evaluation only.

Variants (each = the named chosen/rejected loop state replicated into all
four of the head's input slots, matching the `only_loop_N` convention used
throughout the locus memo):

  (a) L4-only       - boundary-style, the published-architecture baseline
  (b) L1-only       - does L1 alone carry signal?
  (c) L2-only       - sanity: should track L4 if L2-L4 are one state
  (d) mean(L1, L4)  - does fusing the bipartite endpoints help?

A fifth row, natural 4-loop sequence [L1,L2,L3,L4], is printed as a
reference only (it is how the v10 score-agreement step fed the head, and
ties to the published 0.95 on this 200-pair subsample). It is NOT part of
the verdict logic.

Swap-balanced: every variant is scored in both (chosen, rejected) and
(rejected, chosen) orderings; centered_acc and bias_to_signal need both.

Outputs:
  artifacts/reports/evaluator/probe_l1_ablation.json
  + an "L1 ablation (2026-05-14)" subsection appended to the v10 chapter of
    docs/evaluator/pairwise_evaluator_locus_memo_v2_2026-05-11.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator

STATES_PT = "artifacts/reports/evaluator/hh_loop_states_200_thinking.pt"
HEAD_CKPT = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
OUT_JSON = "artifacts/reports/evaluator/probe_l1_ablation.json"
NUM_SLOTS = 4

# Verdict thresholds (exact, per the request).
CHANCE = 0.50
NOISE_WITHIN_PP = 0.02      # L1-only centered within 2 pp of chance -> noise
SIGNAL_CENTERED_MIN = 0.55  # L1-only centered >= 0.55 -> signal
SIGNAL_FUSION_GAIN_PP = 0.01  # mean(L1,L4) centered beats L4 by >= 1 pp -> signal
DEGEN_SIGN_REV_MAX = 0.05   # strict_sign_reversal < 0.05 ...
DEGEN_POS_RATE_MIN = 0.95   # ... and pos_rate > 0.95 -> degenerate (constant head)


def build_variant(states: List[torch.Tensor], kind: str) -> List[torch.Tensor]:
    """states = [L1, L2, L3, L4] per-token tensors [seq, hidden] for one prompt."""
    if kind == "L4_only":
        base = states[3]
    elif kind == "L1_only":
        base = states[0]
    elif kind == "L2_only":
        base = states[1]
    elif kind == "mean_L1_L4":
        base = 0.5 * (states[0] + states[3])
    elif kind == "natural_seq":
        return [s for s in states]  # [L1, L2, L3, L4]
    else:
        raise ValueError(kind)
    return [base, base, base, base]


@torch.no_grad()
def score(head: PairwiseEvaluator, a_states, a_mask, b_states, b_mask, device) -> float:
    a = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in a_states]
    b = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in b_states]
    am = a_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
    bm = b_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
    out = head(a, am, b, bm)
    v = float(out.view(-1).item())
    return v if math.isfinite(v) else float("nan")


def metrics(s_cr: np.ndarray, s_rc: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(s_cr) & np.isfinite(s_rc)
    s_cr = s_cr[valid]
    s_rc = s_rc[valid]
    n = int(s_cr.size)
    centered = s_cr - s_rc
    summed = s_cr + s_rc
    std_centered = float(np.std(centered))
    strict = np.sign(s_cr) != np.sign(s_rc)
    return {
        "n_valid": n,
        "canonical_acc": float(np.mean(s_cr > 0)) if n else float("nan"),
        "centered_acc": float(np.mean(centered > 0)) if n else float("nan"),
        "bias_to_signal": (abs(float(np.mean(summed))) / std_centered) if std_centered > 1e-9 else float("inf"),
        "s_cr_mean": float(np.mean(s_cr)) if n else float("nan"),
        "s_cr_std": float(np.std(s_cr)) if n else float("nan"),
        "s_rc_mean": float(np.mean(s_rc)) if n else float("nan"),
        "s_rc_std": float(np.std(s_rc)) if n else float("nan"),
        "pos_rate_cr": float(np.mean(s_cr > 0)) if n else float("nan"),
        "strict_sign_reversal_rate": float(np.mean(strict)) if n else float("nan"),
        "centered_mean": float(np.mean(centered)) if n else float("nan"),
        "centered_std": std_centered,
        "summed_mean": float(np.mean(summed)) if n else float("nan"),
    }


def is_degenerate(m: Dict[str, float]) -> bool:
    return (m["strict_sign_reversal_rate"] < DEGEN_SIGN_REV_MAX
            and m["pos_rate_cr"] > DEGEN_POS_RATE_MIN)


def decide_verdict(res: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    l1 = res["L1_only"]
    l4 = res["L4_only"]
    fused = res["mean_L1_L4"]

    l1_c = l1["centered_acc"]
    l4_c = l4["centered_acc"]
    fused_c = fused["centered_acc"]
    fusion_gain = fused_c - l4_c

    flags = []
    if is_degenerate(l1):
        flags.append("L1_only is degenerate on flip test (near-constant head output)")
    if is_degenerate(fused):
        flags.append("mean_L1_L4 is degenerate on flip test")

    near_chance = abs(l1_c - CHANCE) <= NOISE_WITHIN_PP
    signal_direct = l1_c >= SIGNAL_CENTERED_MIN
    signal_fusion = fusion_gain >= SIGNAL_FUSION_GAIN_PP

    if flags and not (signal_direct or signal_fusion):
        verdict = "Inconclusive — investigate before proceeding."
        rationale = "; ".join(flags)
        action = "Do NOT lock the Experiment 2 spec; inspect degeneracy first."
    elif signal_direct or signal_fusion:
        verdict = ("L1 carries readable relational signal — Experiment 2 should add an "
                   "early_tap=True option that ingests L1 alongside L4.")
        bits = []
        if signal_direct:
            bits.append(f"L1-only centered_acc {l1_c:.4f} >= {SIGNAL_CENTERED_MIN}")
        if signal_fusion:
            bits.append(f"mean(L1,L4) centered {fused_c:.4f} beats L4 {l4_c:.4f} by {fusion_gain*100:.2f} pp >= 1 pp")
        rationale = "; ".join(bits)
        action = "Experiment 2 spec: add early_tap=True ingesting L1 alongside L4."
    elif near_chance:
        verdict = ("L1 is head-invisible noise — Experiment 2 retrains the published "
                   "L4-only architecture as specified.")
        rationale = (f"L1-only centered_acc {l1_c:.4f} within {NOISE_WITHIN_PP*100:.0f} pp of chance "
                     f"({CHANCE}); fusion gain {fusion_gain*100:+.2f} pp < 1 pp")
        action = "Experiment 2 spec unchanged: retrain published L4-only architecture."
    else:
        verdict = "Inconclusive — investigate before proceeding."
        rationale = (f"L1-only centered_acc {l1_c:.4f} is between chance+2pp ({CHANCE+NOISE_WITHIN_PP:.2f}) "
                     f"and signal threshold ({SIGNAL_CENTERED_MIN}); fusion gain {fusion_gain*100:+.2f} pp < 1 pp")
        action = "Borderline — report numbers, decide Experiment 2 spec manually."

    borderline = (not signal_direct) and (0.52 < l1_c < 0.58)
    return {
        "verdict": verdict,
        "rationale": rationale,
        "recommended_action": action,
        "borderline": bool(borderline),
        "key_numbers": {
            "L1_only_centered_acc": l1_c,
            "L4_only_centered_acc": l4_c,
            "L2_only_centered_acc": res["L2_only"]["centered_acc"],
            "mean_L1_L4_centered_acc": fused_c,
            "fusion_gain_pp_over_L4": fusion_gain * 100.0,
            "degeneracy_flags": flags,
        },
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states_path = PROJECT_ROOT / STATES_PT
    if not states_path.exists():
        print(f"STOP: {STATES_PT} not found. Re-run the v10 probe to capture states.")
        sys.exit(2)

    payload = torch.load(states_path, map_location="cpu", weights_only=False)
    packs = payload["packs"]
    n_packs = len(packs)
    n_loops = len(packs[0]["chosen_states"])
    if n_loops < 4:
        print(f"STOP: saved states have only {n_loops} loop states per example "
              f"(need per-token L1..L4). Re-capture required.")
        sys.exit(2)
    print(f"Loaded {n_packs} packs, {n_loops} per-token loop states each, "
          f"dtype={packs[0]['chosen_states'][0].dtype}")
    print(f"Source meta: {payload.get('meta')}")

    head = PairwiseEvaluator().to(device)
    ckpt = torch.load(PROJECT_ROOT / HEAD_CKPT, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(sd)
    head.eval()
    print(f"Head loaded from {HEAD_CKPT}; device={device}")

    variant_kinds = ["L4_only", "L1_only", "L2_only", "mean_L1_L4", "natural_seq"]
    raw = {k: {"s_cr": [], "s_rc": []} for k in variant_kinds}

    for pk in packs:
        c_states = pk["chosen_states"]
        r_states = pk["rejected_states"]
        c_mask = pk["chosen_mask"]
        r_mask = pk["rejected_mask"]
        for kind in variant_kinds:
            cv = build_variant(c_states, kind)
            rv = build_variant(r_states, kind)
            s_cr = score(head, cv, c_mask, rv, r_mask, device)
            s_rc = score(head, rv, r_mask, cv, c_mask, device)
            raw[kind]["s_cr"].append(s_cr)
            raw[kind]["s_rc"].append(s_rc)

    res = {
        k: metrics(np.asarray(raw[k]["s_cr"], dtype=np.float64),
                   np.asarray(raw[k]["s_rc"], dtype=np.float64))
        for k in variant_kinds
    }

    verdict_block = decide_verdict(res)

    # ---- table ----
    print("\n=== L1 ablation (zero-shot, published head, 200 HH pairs) ===")
    hdr = (f"{'variant':>14} | {'canon':>6} | {'center':>6} | {'bias/sig':>8} | "
           f"{'s_cr μ±σ':>14} | {'s_rc μ±σ':>14} | {'pos':>5} | {'flip':>5}")
    print(hdr)
    print("-" * len(hdr))
    label = {
        "L4_only": "L4-only*",
        "L1_only": "L1-only*",
        "L2_only": "L2-only*",
        "mean_L1_L4": "mean(L1,L4)*",
        "natural_seq": "natural[L1-4]",
    }
    for k in variant_kinds:
        m = res[k]
        print(f"{label[k]:>14} | {m['canonical_acc']:.4f} | {m['centered_acc']:.4f} | "
              f"{m['bias_to_signal']:8.3f} | "
              f"{m['s_cr_mean']:+.3f}±{m['s_cr_std']:.2f} | "
              f"{m['s_rc_mean']:+.3f}±{m['s_rc_std']:.2f} | "
              f"{m['pos_rate_cr']:.3f} | {m['strict_sign_reversal_rate']:.3f}")
    print("* = verdict-relevant variant (named loop replicated 4x). "
          "natural[L1-4] is reference only.")

    print(f"\nVERDICT: {verdict_block['verdict']}")
    print(f"  rationale: {verdict_block['rationale']}")
    print(f"  action:    {verdict_block['recommended_action']}")
    if verdict_block["borderline"]:
        print(f"  ** BORDERLINE: L1-only centered_acc = "
              f"{res['L1_only']['centered_acc']:.4f} — treat verdict with caution. **")

    out = {
        "source_states": STATES_PT,
        "head_checkpoint": HEAD_CKPT,
        "n_pairs": n_packs,
        "protocol": "swap-balanced, zero-shot, fp32, no model forward",
        "variant_definition": "named loop state replicated into all 4 head input slots; "
                              "natural_seq = [L1,L2,L3,L4] reference only",
        "bias_to_signal_formula": "abs(mean(s_cr + s_rc)) / std(s_cr - s_rc)",
        "thresholds": {
            "noise_within_pp_of_chance": NOISE_WITHIN_PP,
            "signal_centered_min": SIGNAL_CENTERED_MIN,
            "signal_fusion_gain_pp": SIGNAL_FUSION_GAIN_PP,
            "degeneracy": {"strict_sign_reversal_max": DEGEN_SIGN_REV_MAX,
                           "pos_rate_min": DEGEN_POS_RATE_MIN},
        },
        "metrics": res,
        "verdict": verdict_block,
        "wall_seconds": round(time.time() - t0, 2),
    }
    out_path = PROJECT_ROOT / OUT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}  ({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
