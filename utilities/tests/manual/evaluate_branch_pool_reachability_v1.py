"""PART I — branch-pool reachability (Experiment 3).

Measures whether generated branch pools contain a verifier-correct branch BEFORE selection,
per domain and logic family. positive_oracle@k = chance a size-k subset contains a positive.
Key rule: no selector can rescue a pool with no good branch -> if @k is low, prioritize the
generator (branch training); if @k is high but selected accuracy low, prioritize selection.
Compares model-generated pools vs cheap dataset/option pools (control).
"""
from __future__ import annotations
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402


def _oracle_at_k(n, pos, k):
    k = min(k, n)
    if pos <= 0:
        return 0.0
    if n - pos < k:
        return 1.0
    return 1.0 - math.comb(n - pos, k) / math.comb(n, k)


def _slice(groups):
    n = len(groups)
    if not n:
        return {}
    o1 = o2 = o4 = rdiv = allw = allc = pk = 0.0
    nb = 0
    for g in groups:
        br = g["branch_attempts"]; m = len(br); nb += m
        pos = sum(1 for b in br if b["objective_reward"] > 0)
        o1 += _oracle_at_k(m, pos, 1); o2 += _oracle_at_k(m, pos, 2); o4 += _oracle_at_k(m, pos, 4)
        rdiv += 1 if g["external_oracle"]["reward_diverse"] else 0
        allw += 1 if g["external_oracle"]["all_wrong"] else 0
        allc += 1 if g["external_oracle"]["all_correct"] else 0
        pk += sum(1 for b in br if b["parse_ok"])
    return {"groups": n, "positive_oracle@1": round(o1 / n, 4), "positive_oracle@2": round(o2 / n, 4),
            "positive_oracle@4": round(o4 / n, 4), "reward_diverse_rate": round(rdiv / n, 4),
            "all_wrong_rate": round(allw / n, 4), "all_correct_rate": round(allc / n, 4),
            "parse_ok_rate": round(pk / nb, 4) if nb else 0,
            "avg_branches": round(nb / n, 2),
            "strategy_diversity": round(B.finite_mean([len({b["strategy_label"] for b in g["branch_attempts"]}) for g in groups]), 2)}


def main() -> int:
    started = time.time(); B.ensure_dirs()
    gen = B.load_generated_groups()
    cheap = [g for g in B.read_jsonl(B.PROC / "branch_pools_raw.jsonl")]
    gen_by_dom = {d: _slice([g for g in gen if g["domain"] == d]) for d in sorted({g["domain"] for g in gen})}
    gen_by_fam = {c: _slice([g for g in gen if g.get("provenance", {}).get("category") == c and g["domain"] == "logic"])
                  for c in sorted({g.get("provenance", {}).get("category") for g in gen if g["domain"] == "logic"})}
    gen_overall = _slice(gen)
    cheap_overall = _slice(cheap)
    rows = [{"generator": "model_generated", "domain": d, **m} for d, m in gen_by_dom.items()]
    rows += [{"generator": "cheap_dataset", "domain": "all", **cheap_overall}]
    # verdict
    o4 = gen_overall.get("positive_oracle@4", 0)
    logic_o4 = gen_by_dom.get("logic", {}).get("positive_oracle@4", 0)
    allw = gen_overall.get("all_wrong_rate", 1)
    if not gen:
        verdict = "DATA_LIMITED"
    elif o4 >= 0.6 and logic_o4 >= 0.5:
        verdict = "LOGIC_REACHABILITY_READY" if logic_o4 >= 0.5 else "BRANCH_REACHABILITY_READY"
    elif o4 >= 0.5:
        verdict = "BRANCH_REACHABILITY_READY" if logic_o4 >= 0.4 else "LOGIC_REACHABILITY_WEAK"
    elif allw > 0.6:
        verdict = "NO_ORACLE_TOO_HIGH"
    else:
        verdict = "GENERATOR_LIMITED"
    payload = {"BRANCH_POOL_REACHABILITY_VERDICT": verdict, "generated_overall": gen_overall,
               "generated_by_domain": gen_by_dom, "generated_by_logic_family": gen_by_fam,
               "cheap_overall": cheap_overall, "n_generated_groups": len(gen),
               "interpretation": ("generator-limited: prioritize branch training" if o4 < 0.5 else
                                  "pools reach oracle often: selection/self-selection is the lever"),
               "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(B.OUT_ROOT / "branch_pool_reachability.json", payload)
    B.write_csv(B.OUT_ROOT / "branch_pool_reachability_rows.csv", rows)
    B.write_md(B.OUT_ROOT / "branch_pool_reachability.md", [
        "# Branch Pool Reachability (Part I / Experiment 3)", "",
        B.status_line("BRANCH_POOL_REACHABILITY_VERDICT", verdict), "",
        f"Model-generated pools ({len(gen)} groups): {gen_overall}. No selector can rescue a pool with no good branch; "
        f"{payload['interpretation']}.", "",
        "## Model-generated reachability by domain", "",
        *B.md_table([{"domain": d, **m} for d, m in gen_by_dom.items()],
                    ["domain", "groups", "positive_oracle@1", "positive_oracle@4", "all_wrong_rate", "parse_ok_rate", "avg_branches"]),
        "", "## By logic family (generated)", "",
        *B.md_table([{"family": c, **m} for c, m in gen_by_fam.items()],
                    ["family", "groups", "positive_oracle@1", "positive_oracle@4", "all_wrong_rate"]),
        "", f"Cheap dataset/option pools (control, gold present): positive_oracle@4 {cheap_overall.get('positive_oracle@4')}.",
    ])
    B.prog("I_reachability", {"verdict": verdict, "generated_overall": gen_overall, "n": len(gen)})
    print(B.status_line("BRANCH_POOL_REACHABILITY_VERDICT", verdict))
    print(f"  generated={len(gen)} overall={gen_overall}")
    print(f"  by_domain @4: " + ", ".join(f"{d}={m.get('positive_oracle@4')}" for d, m in gen_by_dom.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
