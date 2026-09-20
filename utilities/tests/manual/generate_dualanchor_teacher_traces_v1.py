"""PART J — DualAnchor teacher trace generation.

DualAnchor is used as a branch-POLICY teacher (keep/prune/rescue/expand/defer/rank), NOT a
correctness source. Traces are computed over candidate pools (v2 feature groups) by running
DualAnchor scoring + a survivor-handoff narrowing with diversity rescue. teacher_is_ground_truth
is always false; agreement with the EXTERNAL reward is measured but never used as a label.
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

import torch  # noqa: E402
import branch_training_v1_common as B  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402
import bg_core_tap_audit_v1_common as cc  # noqa: E402

OUT = B.OUT_ROOT
CORE = ("coding", "reasoning", "math", "logic", "alignment")
BUDGET = 8
SURVIVOR_K = 5
DEFER_MARGIN = 0.15
MAX_PER_DOM_SPLIT = 2500


def _diversity_rescue(group, da_scores, survivors, pruned):
    """Pick the pruned candidate most distant (24_L4 feature) from survivors -> diversity rescue."""
    cv = cc._sc()["config_vector"]
    sv = [cv(group["candidates"][i], "24_L4") for i in survivors]
    sv = [v for v in sv if isinstance(v, torch.Tensor)]
    if not sv or not pruned:
        return None
    best, bestd = None, -1.0
    for i in pruned:
        vi = cv(group["candidates"][i], "24_L4")
        if not isinstance(vi, torch.Tensor):
            continue
        d = min(float((vi - s).norm()) for s in sv)
        if d > bestd:
            bestd, best = d, i
    return best


def trace_group(group, da) -> dict:
    cands = group["candidates"]
    n = len(cands)
    rewards = [float(c["reward"]) for c in cands]
    gmax = max(rewards) if rewards else 0.0
    s = cc.policy_candidate_scores(group, da)
    order = sorted(range(n), key=lambda i: (s[i] if s[i] == s[i] else -1e9), reverse=True)
    # survivors = top-half (rounded up), >=2, capped at SURVIVOR_K -> pruning is actually exercised
    k = max(2, min(SURVIVOR_K, -(-n // 2)))
    survivors = order[:k]
    pruned = order[k:]
    rescued = []
    r = _diversity_rescue(group, s, survivors, pruned)
    if r is not None and r not in survivors:
        rescued = [r]
    # defer if top1/top2 margin small (terminal confidence weak)
    defer = False
    if len(order) >= 2 and all(s[order[k]] == s[order[k]] for k in (0, 1)):
        defer = (s[order[0]] - s[order[1]]) < DEFER_MARGIN
    keep_ids = set(survivors) | set(rescued)
    labels = []
    for i in range(n):
        labels.append({"candidate_id": cands[i].get("branch_id", f"b{i}"),
                       "keep_label": int(i in keep_ids), "expand_label": int(i == order[0]),
                       "prune_label": int(i not in keep_ids), "rescue_label": int(i in rescued),
                       "defer_label": int(defer),
                       "teacher_confidence": round(float(s[i]), 4) if s[i] == s[i] else None})
    stage = {"stage": "terminal_L4_47", "candidate_ids": [c.get("branch_id", f"b{k}") for k, c in enumerate(cands)],
             "scores_or_margins": [round(float(x), 4) if x == x else None for x in s],
             "survivor_ids": [cands[i].get("branch_id", f"b{i}") for i in survivors],
             "pruned_ids": [cands[i].get("branch_id", f"b{i}") for i in pruned if i not in rescued],
             "rescued_ids": [cands[i].get("branch_id", f"b{i}") for i in rescued],
             "rescue_reasons": ["diversity"] * len(rescued), "defer": defer}
    trace = {"teacher": "DualAnchor", "teacher_role": "branch_policy", "teacher_is_ground_truth": False,
             "group_id": group.get("group_id"), "domain": group.get("domain"),
             "input_pool_ids": stage["candidate_ids"], "stage_traces": [stage],
             "terminal_rank": [cands[i].get("branch_id", f"b{i}") for i in order],
             "teacher_policy_labels": labels}
    # teacher quality = does the kept set retain a verifier-correct branch (vs random pruning baseline)
    oracle_kept = any(rewards[i] >= gmax and gmax > 0 for i in keep_ids)
    oracle_pruned = (gmax > 0) and not oracle_kept
    npos = sum(1 for r in rewards if r >= gmax) if gmax > 0 else 0
    kk = len(keep_ids)
    rand_ret = (1.0 - math.comb(n - npos, kk) / math.comb(n, kk)) if (gmax > 0 and n - npos >= kk) else (1.0 if gmax > 0 else float("nan"))
    return trace, {"keep": len(keep_ids) / n, "prune": (n - len(keep_ids)) / n, "rescue": len(rescued) / n,
                   "defer": 1.0 if defer else 0.0,
                   "oracle_retention": 1.0 if oracle_kept else (0.0 if gmax > 0 else float("nan")),
                   "random_keep_retention": rand_ret,
                   "false_prune": 1.0 if oracle_pruned else 0.0,
                   "keep_vs_correctness_divergence": 1.0 - _agreement(keep_ids, rewards)}


def _agreement(keep_ids, rewards):
    # fraction of candidates whose keep decision matches reward>0 (diagnostic)
    if not rewards:
        return float("nan")
    return sum(1 for i, r in enumerate(rewards) if (i in keep_ids) == (r > 0)) / len(rewards)


def main() -> int:
    started = time.time(); B.ensure_dirs()
    da = cc.build_tap_policies().get("DualAnchor", [])
    traces = []
    agg = defaultdict(list)
    by_dom = defaultdict(lambda: defaultdict(list))
    by_logic_fam = defaultdict(lambda: defaultdict(list))
    for dom in CORE:
        for split in ("train", "heldout"):
            groups = M.groups(dom, split)[:MAX_PER_DOM_SPLIT]
            for g in groups:
                if len(g["candidates"]) < 2:
                    continue
                tr, m = trace_group(g, da)
                tr["split"] = split
                traces.append(tr)
                for k, v in m.items():
                    agg[k].append(v); by_dom[dom][k].append(v)
    B.write_jsonl(B.TEACHER / "dualanchor_teacher_traces.jsonl", traces)
    flat = [{"group_id": t["group_id"], "domain": t["domain"], "split": t.get("split"),
             "n_keep": sum(l["keep_label"] for l in t["teacher_policy_labels"]),
             "defer": t["stage_traces"][0]["defer"], "n_rescued": len(t["stage_traces"][0]["rescued_ids"])}
            for t in traces]
    try:
        import pandas as pd
        pd.DataFrame(flat).to_parquet(B.TEACHER / "dualanchor_teacher_traces.parquet", index=False)
    except Exception:
        pass
    overall = {k: round(B.finite_mean(v), 4) for k, v in agg.items()}
    dom_metrics = {d: {k: round(B.finite_mean(v), 4) for k, v in by_dom[d].items()} for d in by_dom}
    ret = overall.get("oracle_retention", 0); rand = overall.get("random_keep_retention", 0)
    lift = round(ret - rand, 4)  # teacher pruning vs random pruning (the real quality signal)
    overall["oracle_retention_lift_over_random"] = lift
    logic_ret = dom_metrics.get("logic", {}).get("oracle_retention", 1)
    logic_lift = round(dom_metrics.get("logic", {}).get("oracle_retention", 0) - dom_metrics.get("logic", {}).get("random_keep_retention", 0), 4)
    # gate on retention + lift over random (NOT on keep-vs-correctness agreement, which must stay low by design)
    if ret >= 0.88 and lift >= 0.1:
        verdict = "TEACHER_USEFUL_FOR_BRANCH_POLICY"
    elif logic_ret < 0.7 or logic_lift < 0.03:
        verdict = "TEACHER_MISMATCH_ON_LOGIC"
    elif ret >= 0.78 and lift >= 0.05:
        verdict = "TEACHER_NOISY_BUT_USEFUL"
    elif traces:
        verdict = "TEACHER_TRACES_READY"
    else:
        verdict = "TEACHER_NOT_USABLE"
    payload = {"DUALANCHOR_TEACHER_TRACE_VERDICT": verdict, "teacher_is_ground_truth": False,
               "n_traces": len(traces), "overall": overall, "by_domain": dom_metrics,
               "note": "DualAnchor keep/prune/rescue/defer are POLICY labels; external-reward agreement is diagnostic only.",
               "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(OUT / "dualanchor_teacher_traces.json", payload)
    B.write_md(OUT / "dualanchor_teacher_traces.md", [
        "# DualAnchor Teacher Traces (Part J)", "", B.status_line("DUALANCHOR_TEACHER_TRACE_VERDICT", verdict), "",
        "DualAnchor is a branch-POLICY teacher (keep/prune/rescue/defer/rank). These are NOT correctness labels; "
        "external verifiers remain the only ground truth. Teacher quality = oracle_retention vs RANDOM pruning "
        "(lift). keep_vs_correctness_divergence is expected to be high (keep != correctness, by design).", "",
        f"Traces: {len(traces)}. Overall: oracle_retention {overall.get('oracle_retention')} vs random "
        f"{overall.get('random_keep_retention')} (lift {overall.get('oracle_retention_lift_over_random')}); "
        f"prune_rate {overall.get('prune')}, rescue_rate {overall.get('rescue')}, defer_rate {overall.get('defer')}, "
        f"false_prune_rate {overall.get('false_prune')}.", "",
        "## By domain (oracle_retention vs random)", "",
        *B.md_table([{"domain": d, **dom_metrics[d]} for d in dom_metrics],
                    ["domain", "prune", "rescue", "defer", "oracle_retention", "random_keep_retention", "false_prune"]),
    ])
    B.prog("J_teacher_traces", {"verdict": verdict, "n_traces": len(traces), "overall": overall})
    print(B.status_line("DUALANCHOR_TEACHER_TRACE_VERDICT", verdict))
    print(f"  retention={overall.get('oracle_retention')} random={overall.get('random_keep_retention')} "
          f"lift={overall.get('oracle_retention_lift_over_random')} false_prune={overall.get('false_prune')} defer={overall.get('defer')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
