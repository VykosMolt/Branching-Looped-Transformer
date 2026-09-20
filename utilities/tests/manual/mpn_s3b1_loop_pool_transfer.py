"""M+N S3B-1 — loop-distribution selector transfer test (the REAL Wall-B test).

S3B-0 confirmed the selector training path is sane IN-DISTRIBUTION (corecontent_v2 features). Wall B is
about TRANSFER: does any selector pick correct branches on the LOOP's OWN generated-branch distribution?

This generates heldout branch pools from the S1 harness and evaluates selectors as a pure TRANSFER test
(NO training here — selectors were trained on corecontent_v2 tasks, eval pools are S1/canary tasks, so the
guardrail "heldout split by TASK not candidate" holds by construction; pools are also saved for S3B-2):

  per task -> pool of generated branches (verifier-labeled, prompt+answer features [3,4,2048]):
    fork K   : one-time fork from CLEAN root @loop1 locus (alpha, second-half), SAMPLED decode (greedy collapses)
    sample K : plain temperature samples
  selectors evaluated (no training):
    S3B0_pairwise_blockwise, S3B0_listwise_blockwise (from s3b0_trained_selector.pt),
    CoreContent_v2_blockwise, DualAnchor, mixedhead_MIX_HH_OBJECTIVE, MIX_OBJECTIVE_ALL_only, RANDOM, ORACLE

CORE METRIC: selected_correct_when_oracle_present (only pools with >=1 correct branch count).
Panel: oracle_over_pool, selected_acc, oracle_conversion_rate, top2/top4_oracle_retention, selector_regret,
       pairwise_accuracy (= SEPARABILITY diagnostic: correct-vs-incorrect within-pool score ordering),
       domain breakdown. Verifier LABELS only.

SEPARABILITY READ: if a selector's pairwise_accuracy is HIGH but selected_correct_when_oracle_present is
LOW -> calibration/tie problem (locally fixable, S3B-2). If pairwise_accuracy ~0.5 even after refit -> the
hidden-state signal does not separate correct from incorrect generated branches -> selector wall needs
backbone/training-time integration, not just a refit.

Run on GPU (backgrounded):
  S1_N_PER=4 S1_KFORK=4 S1_KSAMP=6 S1_MNT=192 venv/bin/python utilities/tests/manual/mpn_s3b1_loop_pool_transfer.py
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
import bg_autoregressive_cache_common_v1 as C  # noqa: E402
import bg_partial_cache_splice_v2_common as S  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402
import bg_core_tap_audit_v1_common as cc  # noqa: E402
from mpn_s1_2_multilocus import curated_dirs_by_layer  # noqa: E402
from mpn_s1_4_reference_loop import make_extractor_reusing  # noqa: E402
from mpn_s1_5_divergence_ablation import parse, verify, score_text, sample_continue  # noqa: E402

OUT = M.v2.PROJECT_ROOT / "opi/taps/probes/mpn_s3b_2026-06-17"
S1OUT = V.PROJECT_ROOT / "opi/taps/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")

N_PER = int(os.environ.get("S1_N_PER", "4"))
K_FORK = int(os.environ.get("S1_KFORK", "4"))
K_SAMP = int(os.environ.get("S1_KSAMP", "6"))
MNT = int(os.environ.get("S1_MNT", "160"))
SAMP_BATCH = int(os.environ.get("S1_SAMP_BATCH", "2"))   # ~12GB GPU: keep parallel KV small
ALPHA = float(os.environ.get("S1_ALPHA", "0.05"))
TEMP = float(os.environ.get("S1_TEMP", "0.7"))
TOP_P = float(os.environ.get("S1_TOP_P", "0.95"))
BL, BLAYER = 0, 24   # fork from clean root at loop1_L24


def gen_pool(model, tok, ext, task, dirs, device):
    """Generate the branch pool for one task: K_FORK sampled fork branches + K_SAMP plain samples.
    Returns list of {text, correct, provenance, features[3,4,2048]}."""
    prompt = V.build_prompt(tok, task, "direct")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    plen = ids.shape[1]
    tr = (max(1, plen // 2), plen)            # second-half suffix (a diverging regime)
    texts = []
    # fork branches (one-time fork @loop1_L24 from clean root, sampled decode)
    rp = S.root_prefill_with_boundary(model, ids, am, BL, BLAYER)
    for k in range(K_FORK):
        d = dirs[BLAYER][k % len(dirs[BLAYER])]
        br = S.build_spliced_branch(model, ids, am, BL, BLAYER, ALPHA, d, tr, root_pack=rp)
        seed = abs(hash((task["canary_id"], "fork", k))) % (2**31)
        toks = sample_continue(model, br["spliced_cache"], br["am_full"], br["next_logits"], MNT, device, TEMP, TOP_P, seed)
        texts.append(("fork", tok.decode(toks, skip_special_tokens=True)))
        del br                                # free this branch's spliced+decode cache before the next
        C.clear_cuda()
    del rp
    C.clear_cuda()
    # plain samples (small batches to respect the ~12GB GPU)
    with torch.inference_mode():
        remaining = K_SAMP
        while remaining > 0:
            b = min(SAMP_BATCH, remaining)
            gen = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=MNT, do_sample=True,
                                 temperature=TEMP, top_p=TOP_P, num_return_sequences=b, pad_token_id=tok.pad_token_id)
            for row in gen:
                texts.append(("sample", tok.decode(row[plen:], skip_special_tokens=True)))
            del gen
            remaining -= b
            C.clear_cuda()
    # features (all generation caches freed; encode one text at a time)
    pool = []
    for prov, txt in texts:
        correct, _fin = verify(task, txt)
        feats = ext.encode_text_to_pooled_features(score_text(task, txt), max_length=512)
        pool.append({"text": txt, "correct": bool(correct), "provenance": prov, "features": feats})
        C.clear_cuda()
    return pool


def pool_panel(group, scores):
    cands = group["candidates"]
    rewards = [float(c["reward"]) for c in cands]
    correct = [r > 0 for r in rewards]
    finite = [(i, s) for i, s in enumerate(scores) if isinstance(s, float) and math.isfinite(s)]
    if not finite:
        return None
    order = [i for i, _ in sorted(finite, key=lambda x: x[1], reverse=True)]
    top = order[0]
    pa = cc.group_selection_metrics(group, scores).get("pairwise_accuracy", float("nan"))
    return {"oracle_present": any(correct), "reward_diverse": (0 < sum(correct) < len(correct)),
            "selected_correct": bool(correct[top]), "regret": max(rewards) - rewards[top],
            "top2_ret": any(correct[i] for i in order[:2]), "top4_ret": any(correct[i] for i in order[:4]),
            "pairwise_accuracy": pa, "n": len(cands)}


def eval_selector(policy, pools_by_dom):
    per_dom = {}
    for d, pools in pools_by_dom.items():
        rows = []
        for g in pools:
            r = pool_panel(g, M.score_group(g, policy))
            if r:
                rows.append(r)
        if not rows:
            continue
        op = [r for r in rows if r["oracle_present"]]
        rd = [r for r in rows if r["reward_diverse"] and math.isfinite(r["pairwise_accuracy"])]
        nop = max(1, len(op))
        per_dom[d] = {
            "n_pools": len(rows), "n_oracle_present": len(op),
            "oracle_over_pool": round(sum(r["oracle_present"] for r in rows) / len(rows), 4),
            "selected_acc": round(sum(r["selected_correct"] for r in rows) / len(rows), 4),
            "selected_correct_when_oracle_present": round(sum(r["selected_correct"] for r in op) / nop, 4),
            "oracle_conversion_rate": round(sum(r["selected_correct"] for r in op) / nop, 4),
            "top2_oracle_retention": round(sum(r["top2_ret"] for r in op) / nop, 4),
            "top4_oracle_retention": round(sum(r["top4_ret"] for r in op) / nop, 4),
            "selector_regret": round(sum(r["regret"] for r in rows) / len(rows), 4),
            "pairwise_accuracy_separability": (round(sum(r["pairwise_accuracy"] for r in rd) / len(rd), 4) if rd else None),
        }
    # oracle-conditioned keys macro ONLY over domains with >=1 oracle-present pool (else a 0-oracle domain
    # like coding wrongly contributes 0.0 and deflates every selector incl ORACLE). Pool-level keys macro over all.
    ALL_DOM = ["oracle_over_pool", "selected_acc", "selector_regret"]
    ORACLE_COND = ["selected_correct_when_oracle_present", "oracle_conversion_rate",
                   "top2_oracle_retention", "top4_oracle_retention"]
    op_doms = [d for d in per_dom if per_dom[d].get("n_oracle_present", 0) > 0]
    macro = {}
    for k in ALL_DOM:
        macro[k] = round(sum(per_dom[d][k] for d in per_dom) / max(1, len(per_dom)), 4)
    for k in ORACLE_COND:
        macro[k] = (round(sum(per_dom[d][k] for d in op_doms) / len(op_doms), 4) if op_doms else None)
    seps = [per_dom[d]["pairwise_accuracy_separability"] for d in per_dom
            if per_dom[d]["pairwise_accuracy_separability"] is not None]
    macro["pairwise_accuracy_separability"] = round(sum(seps) / len(seps), 4) if seps else None
    return {"macro": macro,
            "by_domain_core_metric": {d: per_dom[d]["selected_correct_when_oracle_present"] for d in per_dom},
            "per_domain": per_dom}


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((S1OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]
    dirs = curated_dirs_by_layer(max(K_FORK, 2))

    model, tok, info = C.load_model(attn_implementation="eager")
    device = next(model.parameters()).device
    ext = make_extractor_reusing(model, tok, device)

    # build pools
    pools_by_dom = {d: [] for d in DOMAINS}
    saved = {}
    for task in tasks:
        pool = gen_pool(model, tok, ext, task, dirs, device)
        cands = [{"features": b["features"], "reward": 1.0 if b["correct"] else 0.0,
                  "branch_id": f"{b['provenance']}{i}"} for i, b in enumerate(pool)]
        group = {"group_id": f"loop_{task['canary_id']}", "domain": task["domain"],
                 "task_id": task["canary_id"], "candidates": cands, "kind": "tournament"}
        pools_by_dom[task["domain"]].append(group)
        nc = sum(b["correct"] for b in pool)
        saved[task["canary_id"]] = {"domain": task["domain"],
                                    "branches": [{"text": b["text"][:400], "correct": b["correct"],
                                                  "provenance": b["provenance"]} for b in pool]}
        print(f"  {task['domain']:9} {task['canary_id']}: pool={len(pool)} correct={nc} "
              f"oracle_present={nc>0} reward_diverse={0<nc<len(pool)}", flush=True)

    # persist pools (features + labels) for S3B-2 (by-task split available)
    serial = {}
    for d, gs in pools_by_dom.items():
        serial[d] = []
        for g in gs:
            cands = [{"features": c["features"], "reward": c["reward"], "branch_id": c["branch_id"]}
                     for c in g["candidates"]]
            serial[d].append({"task_id": g["task_id"], "domain": g["domain"], "candidates": cands})
    torch.save({"pools_by_dom": serial}, OUT / "s3b1_loop_pools.pt")
    (OUT / "s3b1_pool_texts.json").write_text(json.dumps(saved, indent=2))

    # selectors (no training here)
    trained = torch.load(OUT / "s3b0_trained_selector.pt", map_location="cpu", weights_only=False) \
        if (OUT / "s3b0_trained_selector.pt").exists() else {}
    base = M.baseline_policies(include_diag=True)
    crafted = M.crafted_v2_policies()
    selectors = {
        "S3B0_pairwise_blockwise": trained.get("pairwise_blockwise_channels"),
        "S3B0_listwise_blockwise": trained.get("listwise_blockwise_channels"),
        "CoreContent_v2_blockwise": crafted.get("CoreContent_v2_blockwise"),
        "DualAnchor": base.get("DualAnchor"),
        "mixedhead_MIX_HH_OBJECTIVE": base.get("mixedhead_MIX_HH_OBJECTIVE"),
        "MIX_OBJECTIVE_ALL_only": base.get("MIX_OBJECTIVE_ALL_only"),
        "RANDOM": "RANDOM", "ORACLE": "ORACLE",
    }
    results = {}
    for name, pol in selectors.items():
        if pol is None:
            continue
        results[name] = eval_selector(pol, pools_by_dom)
        m = results[name]["macro"]
        print(f"  {name:28} sel@oracle={m['selected_correct_when_oracle_present']} "
              f"sel_acc={m['selected_acc']} top4_ret={m['top4_oracle_retention']} "
              f"sep(pairwise)={m['pairwise_accuracy_separability']}", flush=True)

    core = "selected_correct_when_oracle_present"
    def cm(n): return results.get(n, {}).get("macro", {}).get(core)
    best_trained = max([x for x in (cm("S3B0_pairwise_blockwise"), cm("S3B0_listwise_blockwise")) if x is not None], default=None)
    frozen = cm("CoreContent_v2_blockwise"); rand = cm("RANDOM")
    best_sep = max([results[n]["macro"]["pairwise_accuracy_separability"] for n in results
                    if results[n]["macro"]["pairwise_accuracy_separability"] is not None], default=None)
    # diagnosis: separable (some selector orders correct>incorrect well) vs not
    separable = best_sep is not None and best_sep >= 0.6
    transfers = best_trained is not None and rand is not None and best_trained >= rand + 0.1
    decision = {
        "core_metric": core, "best_trained_selector": best_trained, "frozen_corecontent": frozen,
        "random_baseline": rand, "best_pairwise_separability": best_sep,
        "transfer_beats_random_by_0.1": transfers, "signal_separable_ge_0.6": separable,
        "verdict": ("WALL_B_LOCALLY_SOLVABLE__CALIBRATION" if (separable and not transfers)
                    else "WALL_B_SELECTOR_TRANSFERS" if transfers
                    else "WALL_B_SIGNAL_PROBLEM__NEEDS_BACKBONE" if (not separable)
                    else "WALL_B_INCONCLUSIVE"),
        "note": ("separability HIGH but top1 transfer weak -> calibration/refit problem, train selector on "
                 "loop pools (S3B-2)" if (separable and not transfers) else
                 "a selector already converts oracle on generated pools -> Wall B transfers without refit" if transfers
                 else "hidden-state signal does NOT separate correct vs incorrect generated branches (pairwise~0.5) "
                 "-> selector wall needs backbone/training-time integration, not just a refit" if not separable
                 else "mixed/inconclusive; inspect per-domain"),
    }
    payload = {"verdict": "S3B1_LOOP_POOL_TRANSFER", "scope": "pure transfer eval (no training); the real Wall-B test",
               "config": {"N_PER": N_PER, "K_FORK": K_FORK, "K_SAMP": K_SAMP, "MNT": MNT, "alpha": ALPHA,
                          "fork_locus": f"loop{BL+1}_L{BLAYER}", "fork_token_range": "second-half", "decode": "sampled",
                          "temp": TEMP, "top_p": TOP_P, "score_input_mode": "prompt_plus_answer",
                          "heldout": "by TASK (eval tasks disjoint from selector training tasks)"},
               "n_tasks": len(tasks), "results": results, "decision": decision,
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s3b1_loop_pool_transfer.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"decision": decision}, indent=2))
    print(f"\n{decision['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
