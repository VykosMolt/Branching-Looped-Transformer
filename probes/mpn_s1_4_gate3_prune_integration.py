"""M+N S1.4 GATE 3 — live prune-scorer integration (DualAnchor validity + CoreContent_v2 content).

Gates 1+2 validated the α>0 chaining substrate (bit-exact). The full loop also needs the PRUNE:
DualAnchor (branch validity/survival ALONE, first prune) then CoreContent_v2 (content selection,
content-threshold, never judges validity). Both are the locked antisymmetric pairwise taps; their
NATIVE input is a candidate's TEXT re-encoded to pooled BG features [3 layers(24/36/47) x 4 loops
x 2048] via BGTransformerFeatureExtractor — exactly the training/eval pipeline. So the faithful
live prune = encode each branch's text -> assemble a group -> cc.policy_candidate_scores(group, policy).

This gate confirms that live path end-to-end on a hand-built group of branch-like texts:
  - extractor yields [3,4,2048] per text,
  - DualAnchor and CoreContent_v2 both return FINITE, per-branch-VARYING (relative tournament) scores,
  - scores are reused exactly from the locked pipeline (M.score_group -> cc.policy_candidate_scores).
Ranking quality is informational (taps are ~0.67 macro, not oracle); the gate is integration fidelity.

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_4_gate3_prune_integration.py
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402  (wraps cc scoring + v2 paths)

OUT = M.v2.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"

# Branch-like candidate texts (a coherent correct solve, a wrong solve, a degenerate/empty-ish one).
BRANCHES = [
    ("solve_ok", "Let me work it out. 17 + 28 = 45, then 45 * 2 = 90.\nFINAL ANSWER: 90"),
    ("solve_wrong", "I think the total is around fifty something.\nFINAL ANSWER: 53"),
    ("degenerate", "the the the and and so so therefore therefore um um um"),
    ("solve_ok2", "Step 1: 17+28=45. Step 2: double it -> 90. Checks out.\nFINAL ANSWER: 90"),
]


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = BGTransformerFeatureExtractor(device=dev, dtype="auto", force_all_loops=True)

    cands = []
    shapes = {}
    for name, text in BRANCHES:
        f = ext.encode_text_to_pooled_features(text, max_length=512)
        shapes[name] = tuple(int(x) for x in f.shape)
        cands.append({"features": f, "reward": 0.0, "branch_id": name})
    group = {"group_id": "gate3_smoke", "domain": "math", "task_id": "gate3",
             "candidates": cands, "source": "mpn_s1_4_gate3", "kind": "tournament"}

    da_policy = M.baseline_policies(include_diag=True).get("DualAnchor")
    cc_policy = M.crafted_v2_policies().get("CoreContent_v2_blockwise")
    da_scores = M.score_group(group, da_policy)
    cc_scores = M.score_group(group, cc_policy)

    def finite_vary(xs):
        fin = [x for x in xs if isinstance(x, float) and math.isfinite(x)]
        return len(fin) == len(xs) and len(set(round(x, 6) for x in fin)) > 1

    shape_ok = all(s == (3, 4, 2048) for s in shapes.values())
    da_ok = da_policy is not None and finite_vary(da_scores)
    cc_ok = cc_policy is not None and finite_vary(cc_scores)
    passed = shape_ok and da_ok and cc_ok

    by_branch = [{"branch_id": cands[i]["branch_id"],
                  "dualanchor": round(da_scores[i], 4) if math.isfinite(da_scores[i]) else None,
                  "corecontent": round(cc_scores[i], 4) if math.isfinite(cc_scores[i]) else None}
                 for i in range(len(cands))]
    payload = {"verdict": "S1_4_GATE3_PASS" if passed else "S1_4_GATE3_FAIL",
               "feature_shapes": shapes, "shape_ok": shape_ok,
               "dualanchor_channels": (len(da_policy) if da_policy else 0),
               "corecontent_channels": (len(cc_policy) if cc_policy else 0),
               "dualanchor_finite_varying": da_ok, "corecontent_finite_varying": cc_ok,
               "by_branch": by_branch,
               "note": "ranking informational; gate = integration fidelity (live encode->group->score)",
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4_gate3_prune_integration.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n{payload['verdict']}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
