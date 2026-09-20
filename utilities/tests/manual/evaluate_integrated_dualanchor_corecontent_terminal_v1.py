"""PART H — integrated DualAnchor + CoreContent terminal validation (Experiment 1).

Control/eval harness: on real DualAnchor terminal survivor sets, does CoreContent_v2_blockwise
choose better final outputs than DualAnchor forced top1 / mixedhead_MIX_HH_OBJECTIVE /
MIX_OBJECTIVE_ALL / MIX_CODE_REASONING / random / oracle? Reuses the corecontent_v2 feature
groups (coding/reasoning/math/logic/alignment, heldout) + locked scoring layer.

Survivor-set handoff = top-K by DualAnchor (locked terminal default); selectors rank WITHIN it.
External labels only. This validates the external baseline; it is NOT the desired final arch.
"""
from __future__ import annotations
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402
import bg_core_tap_audit_v1_common as cc  # noqa: E402

OUT = B.OUT_ROOT
CORE = ("coding", "reasoning", "math", "logic", "alignment")
SURVIVOR_K = 5


def _pruned_corecontent():
    blk = M.crafted_v2_policies().get("CoreContent_v2_blockwise", [])
    return [c for c in blk if c[2] != "47_L4"] or blk  # locked pruned 24+36


def _selectors():
    base = cc.build_tap_policies()
    sel = {
        "CoreContent_v2_blockwise": _pruned_corecontent(),
        "mixedhead_MIX_HH_OBJECTIVE": base.get("mixedhead_MIX_HH_OBJECTIVE", []),
        "MIX_OBJECTIVE_ALL": base.get("MIX_OBJECTIVE_ALL_only", []),
        "MIX_CODE_REASONING": base.get("MIX_CODE_REASONING_only", []),
        "DualAnchor_terminal": base.get("DualAnchor", []),
    }
    return {k: v for k, v in sel.items() if v}


def _scores(group, channels):
    return cc.policy_candidate_scores(group, channels)


def main() -> int:
    started = time.time(); B.ensure_dirs()
    da = cc.build_tap_policies().get("DualAnchor", [])
    selectors = _selectors()
    rng = random.Random(0)
    rows = []
    per_dom = defaultdict(lambda: defaultdict(list))  # domain -> selector -> [top1_oracle]
    retention = defaultdict(list)
    survivor_rows = []
    for dom in CORE:
        groups = M.groups(dom, "heldout")
        for g in groups:
            cands = g["candidates"]
            n = len(cands)
            if n < 2:
                continue
            rewards = [float(c["reward"]) for c in cands]
            gmax = max(rewards)
            if gmax <= 0:
                continue  # no positive oracle -> skip (diagnostic)
            # DualAnchor survivor set = top-K by DualAnchor score
            ds = _scores(g, da)
            order = sorted(range(n), key=lambda i: (ds[i] if ds[i] == ds[i] else -1e9), reverse=True)
            surv = order[:min(SURVIVOR_K, n)]
            surv_has_oracle = any(rewards[i] >= gmax for i in surv)
            retention[dom].append(1.0 if surv_has_oracle else 0.0)
            # each selector ranks within survivors
            for name, ch in selectors.items():
                sc = _scores(g, ch)
                pick = max(surv, key=lambda i: (sc[i] if sc[i] == sc[i] else -1e9))
                top1 = 1.0 if rewards[pick] >= gmax else 0.0
                per_dom[dom][name].append(top1)
            # baselines: random survivor, oracle, DualAnchor forced top1 (rank-1)
            per_dom[dom]["random_survivor"].append(1.0 if rewards[rng.choice(surv)] >= gmax else 0.0)
            per_dom[dom]["oracle"].append(1.0)
            per_dom[dom]["DualAnchor_forced_top1"].append(1.0 if rewards[order[0]] >= gmax else 0.0)
            survivor_rows.append({"group_id": g.get("group_id"), "domain": dom, "n": n,
                                  "survivor_has_oracle": surv_has_oracle})
    # aggregate
    sel_names = sorted({s for d in per_dom for s in per_dom[d]})
    macro = {}
    for s in sel_names:
        vals = [B.finite_mean(per_dom[d][s]) for d in CORE if per_dom[d].get(s)]
        macro[s] = round(B.finite_mean(vals), 4)
        for d in CORE:
            if per_dom[d].get(s):
                rows.append({"selector": s, "domain": d, "top1_oracle": round(B.finite_mean(per_dom[d][s]), 4),
                             "groups": len(per_dom[d][s])})
    ret_macro = round(B.finite_mean([B.finite_mean(retention[d]) for d in CORE if retention[d]]), 4)
    cc_v2 = macro.get("CoreContent_v2_blockwise", 0)
    hh = macro.get("mixedhead_MIX_HH_OBJECTIVE", 0)
    da_top1 = macro.get("DualAnchor_forced_top1", 0)
    best_alt = max((macro.get(k, 0) for k in ("mixedhead_MIX_HH_OBJECTIVE", "MIX_OBJECTIVE_ALL",
                                              "MIX_CODE_REASONING", "DualAnchor_terminal")), default=0)
    logic_cc = round(B.finite_mean(per_dom["logic"].get("CoreContent_v2_blockwise", [])), 4) if per_dom["logic"] else None
    logic_hh = round(B.finite_mean(per_dom["logic"].get("mixedhead_MIX_HH_OBJECTIVE", [])), 4) if per_dom["logic"] else None
    if cc_v2 > best_alt + 0.02 and cc_v2 > da_top1 + 0.02:
        verdict = "CORECONTENT_IMPROVES_TERMINAL"
    elif cc_v2 >= best_alt - 0.005:
        verdict = "CORECONTENT_IMPROVES_TERMINAL" if cc_v2 > best_alt else "CORECONTENT_FLAT_ON_SURVIVORS"
    elif logic_cc is not None and logic_cc < 0.4:
        verdict = "LOGIC_TERMINAL_STILL_WEAK"
    else:
        verdict = "CORECONTENT_FAILS_ON_SURVIVORS"
    payload = {"INTEGRATED_DUALANCHOR_CORECONTENT_TERMINAL_VERDICT": verdict,
               "selector_macro_top1": macro, "survivor_oracle_retention_macro": ret_macro,
               "corecontent_v2": cc_v2, "best_alternative_selector": best_alt, "dualanchor_forced_top1": da_top1,
               "logic_corecontent_v2": logic_cc, "logic_hh": logic_hh, "survivor_k": SURVIVOR_K,
               "rows": rows, "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(OUT / "integrated_terminal_eval.json", payload)
    B.write_csv(OUT / "integrated_terminal_rows.csv", rows)
    # persist survivor sets snapshot
    B.write_jsonl(B.DATA_ROOT / "terminal_survivor_sets.jsonl", survivor_rows)
    try:
        import pandas as pd
        pd.DataFrame(survivor_rows).to_parquet(B.DATA_ROOT / "terminal_survivor_sets.parquet", index=False)
    except Exception:
        pass
    B.write_md(OUT / "integrated_terminal_eval.md", [
        "# Integrated DualAnchor + CoreContent Terminal Eval (Part H / Experiment 1)", "",
        B.status_line("INTEGRATED_DUALANCHOR_CORECONTENT_TERMINAL_VERDICT", verdict), "",
        f"Survivor-set handoff = top-{SURVIVOR_K} by DualAnchor; selectors rank within survivors; external labels only. "
        f"Survivor oracle retention (macro): {ret_macro}.", "",
        "## Selector macro top1 (within DualAnchor survivors, heldout)", "",
        *[f"- {k}: {B.fmt(v)}" for k, v in sorted(macro.items(), key=lambda x: -x[1])], "",
        f"CoreContent_v2 {B.fmt(cc_v2)} vs best alt selector {B.fmt(best_alt)} vs DualAnchor forced top1 {B.fmt(da_top1)}. "
        f"Logic: CoreContent_v2 {B.fmt(logic_cc)} vs HH {B.fmt(logic_hh)}.", "",
        *B.md_table(sorted(rows, key=lambda r: (r["selector"], r["domain"])), ["selector", "domain", "top1_oracle", "groups"]),
    ])
    B.prog("H_integrated_terminal", {"verdict": verdict, "corecontent_v2": cc_v2, "best_alt": best_alt})
    print(B.status_line("INTEGRATED_DUALANCHOR_CORECONTENT_TERMINAL_VERDICT", verdict))
    print(f"  CoreContent_v2={cc_v2} best_alt={best_alt} DA_forced_top1={da_top1} retention={ret_macro} | logic cc={logic_cc} hh={logic_hh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
