"""M+N S1.5 — BOUNDED divergence-regime ablation (NOT a reachability study).

The reference loop (α=0.02, last-token, greedy, 12-locus chain) preserved oracle reachability with ZERO
gain: frozen branches diverge but never reach a NEW correct answer base missed. This ablation asks the
single bounded question:

  IS THERE ANY FROZEN-MODEL REGIME WHERE BRANCHES BECOME OUTCOME-DISTINCT WITHOUT DESTROYING CORRECTNESS?

Bounded design (cheap): a SINGLE-LOCUS fork probe. For each cell (alpha x token_range x decode), fork each
of loop-1's three loci (L24/L36/L47) INDEPENDENTLY from the CLEAN root (no re-derive chain), K branches per
locus (curated per-layer directions), decode full answers, tap-score (PROMPT+ANSWER format = interpretable),
verify (LABEL ONLY). The 12-locus chain was already characterized; here we sweep the FORK parameters cheaply.

  alpha       in {0.02, 0.05, 0.10}
  token_range in {last token, last-8 suffix, second-half suffix}   (S1_TR_SET=reduced drops second-half)
  decode      in {greedy, low-temp sampling (temp 0.7, top_p 0.95)}

Per-cell metrics (aggregated over 4 tasks x 3 loci x K branches):
  base_acc, oracle_over_branches, new_correct_rate_on_base_missed (DECISIVE), selected_acc,
  base_correct_preserved_rate, mean_terminal_spread, mean_unique_answers, mean_unique_outcomes,
  loci_diverged_rate, reconvergence_rate, parse_invalid_rate, mean_distinct_branches.

MAIN DECISION:
  * If NO cell finds new correct answers on base-missed tasks -> frozen branching is CLOSED under tested
    regimes; S3/backbone training (M+N) is the only serious lever.
  * If a cell DOES -> flag it; a K-matched sampling baseline must run before claiming any CARRY gain
    (sampling alone may explain it).

Verifier LABELS only; no control use. No backbone training; no model mutation; clean-root forks only
(never feed a spliced cache to root_prefill_with_boundary). N is NOT scaled (4 tasks, shakedown scope).

Run on GPU (backgrounded):
  S1_TR_SET=full S1_K=4 S1_MNT=96 venv/bin/python utilities/tests/manual/mpn_s1_5_divergence_ablation.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v1_common as B1  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
import bg_autoregressive_cache_common_v1 as C  # noqa: E402
import bg_autoregressive_cache_helpers_v1 as H  # noqa: E402
import bg_partial_cache_splice_v2_common as S  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402
from mpn_s1_2_multilocus import curated_dirs_by_layer  # noqa: E402
from mpn_s1_4_reference_loop import make_extractor_reusing  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")

N_PER = 1                                              # NOT scaled: 1/domain -> 4 tasks (shakedown)
K = int(os.environ.get("S1_K", "4"))
MNT = int(os.environ.get("S1_MNT", "96"))
BL = 0                                                 # boundary loop index 0 == loop 1
LOOP1_LAYERS = (24, 36, 47)
ALPHAS = [float(x) for x in os.environ.get("S1_ALPHAS", "0.02,0.05,0.10").split(",")]
TR_SET = os.environ.get("S1_TR_SET", "full")           # "full" | "reduced" (drops second-half)
TEMP = float(os.environ.get("S1_TEMP", "0.7"))
TOP_P = float(os.environ.get("S1_TOP_P", "0.95"))


def token_ranges(plen):
    tr = {"last": (plen - 1, plen), "last8": (max(1, plen - 8), plen)}
    if TR_SET != "reduced":
        tr["secondhalf"] = (max(1, plen // 2), plen)
    return tr


def parse(task, text):
    return B1.extract_code(text) if task.get("label_type") == "unit_tests" else (B1.parse_final(text) or "")


def verify(task, text):
    is_code = task.get("label_type") == "unit_tests"
    p = parse(task, text)
    if is_code:
        return (B1.verify_code_unit_tests(p, task.get("unit_tests", []), task.get("test_setup", "")), p)
    return (B1.verify_final(task, p)[1] > 0, p)


def score_text(task, answer):
    """Interpretable PROMPT+ANSWER format (matches the dominant tap training format)."""
    a = answer if str(answer).strip() else "(empty)"
    return f"Problem: {task.get('task_prompt', '')}\n\nSolution: {a}"


@torch.no_grad()
def sample_continue(model, cache, am_full, next_logits, n_steps, device, temp, top_p, seed):
    g = torch.Generator(device=device.type if device.type != "cuda" else "cuda")
    g.manual_seed(seed)
    tokens = []
    cur_am, nl = am_full, next_logits
    for _ in range(n_steps):
        probs = torch.softmax(nl[0].float() / max(temp, 1e-6), dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum_before = torch.cumsum(sp, dim=-1) - sp
        sp = torch.where(cum_before <= top_p, sp, torch.zeros_like(sp))
        tot = float(sp.sum())
        idx0 = int(torch.multinomial(sp / tot, 1, generator=g).item()) if tot > 0 else 0
        t = int(si[idx0].item())
        tokens.append(t)
        cur_am = H.update_attention_mask(cur_am, 1)
        nl, cache = C.decode_step(model, torch.tensor([[t]], device=device), cache, cur_am)
    return tokens


def tap_scores(ext, task, texts, da_policy, cc_policy):
    cands = [{"features": ext.encode_text_to_pooled_features(score_text(task, t), max_length=512),
              "reward": 0.0, "branch_id": str(i)} for i, t in enumerate(texts)]
    group = {"group_id": f"abl_{task.get('canary_id','x')}", "domain": task.get("domain", "unknown"),
             "task_id": task.get("canary_id", "x"), "candidates": cands, "kind": "tournament"}
    return M.score_group(group, da_policy), M.score_group(group, cc_policy)


def run_cell(model, tok, ext, tasks, dirs, da_policy, cc_policy, device, base, alpha, tr_name, decode, loci):
    """One cell: fork each (bloop, layer) in `loci` INDEPENDENTLY from the clean root, K branches each."""
    groups = []            # one per (task, locus): list of branch dicts
    per_task = {}          # task -> {base_correct, any_correct, selected_correct, ...}
    for task in tasks:
        tid = task["canary_id"]
        prompt = V.build_prompt(tok, task, "direct")
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
        ids, am = enc["input_ids"], enc["attention_mask"]
        plen = ids.shape[1]
        tr = token_ranges(plen)[tr_name]
        task_branches = []
        for (bloop, layer) in loci:
            rp = S.root_prefill_with_boundary(model, ids, am, bloop, layer)   # CLEAN root, single locus
            texts = []
            for k in range(K):
                d = dirs[layer][k % len(dirs[layer])]
                br = S.build_spliced_branch(model, ids, am, bloop, layer, alpha, d, tr, root_pack=rp)
                if decode == "greedy":
                    toks, _ = S.greedy_continue(model, br["spliced_cache"], br["am_full"], br["next_logits"], MNT, device)
                else:
                    seed = abs(hash((tid, bloop, layer, k))) % (2**31)
                    toks = sample_continue(model, br["spliced_cache"], br["am_full"], br["next_logits"],
                                           MNT, device, TEMP, TOP_P, seed)
                texts.append(tok.decode(toks, skip_special_tokens=True))
            del rp
            C.clear_cuda()
            da, cc = tap_scores(ext, task, texts, da_policy, cc_policy)
            bs = []
            for i, txt in enumerate(texts):
                correct, fin = verify(task, txt)
                bs.append({"text": txt, "final": fin or "", "correct": bool(correct),
                           "cc": float(cc[i]), "da": float(da[i]), "invalid": not bool((fin or "").strip())})
            groups.append({"task": tid, "locus": f"loop{bloop+1}_L{layer}", "branches": bs})
            task_branches.extend(bs)
        # per-task rollups
        any_correct = any(b["correct"] for b in task_branches)
        sel = max(task_branches, key=lambda b: (b["cc"] if b["cc"] == b["cc"] else -1e9))  # NaN-safe argmax
        per_task[tid] = {"base_correct": base[tid]["correct"], "any_correct": any_correct,
                         "selected_correct": bool(sel["correct"]), "selected_final": sel["final"][:60]}

    # cell aggregate
    def gmetrics(g):
        bs = g["branches"]
        finals = [b["final"] for b in bs]
        texts = [b["text"] for b in bs]
        ccs = [b["cc"] for b in bs if b["cc"] == b["cc"]]
        uniq_finals = len(set(finals)); uniq_texts = len(set(texts))
        uniq_outcomes = len(set(b["correct"] for b in bs))
        spread = (max(ccs) - min(ccs)) if ccs else 0.0
        return {"uniq_finals": uniq_finals, "uniq_texts": uniq_texts, "uniq_outcomes": uniq_outcomes,
                "spread": spread, "any_correct": any(b["correct"] for b in bs),
                "text_diverged": uniq_texts > 1, "final_diverged": uniq_finals > 1,
                "reconverged": (uniq_texts > 1 and uniq_finals == 1),
                "n_invalid": sum(b["invalid"] for b in bs), "n": len(bs)}
    gms = [gmetrics(g) for g in groups]
    ng = len(gms)
    base_missed = [t for t in per_task if not per_task[t]["base_correct"]]
    base_correct_tasks = [t for t in per_task if per_task[t]["base_correct"]]
    nbm = len(base_missed)
    text_div_groups = [m for m in gms if m["text_diverged"]]
    agg = {
        "alpha": alpha, "token_range": tr_name, "decode": decode,
        "loci": [f"loop{bl+1}_L{ly}" for (bl, ly) in loci],
        "base_acc": round(sum(per_task[t]["base_correct"] for t in per_task) / len(per_task), 3),
        "oracle_over_branches": round(sum(per_task[t]["any_correct"] for t in per_task) / len(per_task), 3),
        "new_correct_rate_on_base_missed": (round(sum(per_task[t]["any_correct"] for t in base_missed) / nbm, 3)
                                            if nbm else None),
        "new_correct_tasks": [t for t in base_missed if per_task[t]["any_correct"]],
        "selected_acc": round(sum(per_task[t]["selected_correct"] for t in per_task) / len(per_task), 3),
        "base_correct_preserved_rate": (round(sum(per_task[t]["any_correct"] for t in base_correct_tasks) /
                                              len(base_correct_tasks), 3) if base_correct_tasks else None),
        "mean_terminal_spread": round(sum(m["spread"] for m in gms) / ng, 4),
        "mean_unique_answers": round(sum(m["uniq_finals"] for m in gms) / ng, 3),
        "mean_unique_outcomes": round(sum(m["uniq_outcomes"] for m in gms) / ng, 3),
        "mean_distinct_branches": round(sum(m["uniq_texts"] for m in gms) / ng, 3),
        "loci_diverged_rate": round(sum(m["final_diverged"] for m in gms) / ng, 3),
        "text_diverged_rate": round(sum(m["text_diverged"] for m in gms) / ng, 3),
        "reconvergence_rate": (round(sum(m["reconverged"] for m in gms) / len(text_div_groups), 3)
                               if text_div_groups else 0.0),
        "parse_invalid_rate": round(sum(m["n_invalid"] for m in gms) / sum(m["n"] for m in gms), 3),
    }
    return agg, per_task


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]
    dirs = curated_dirs_by_layer(K)

    model, tok, info = C.load_model(attn_implementation="eager")
    device = next(model.parameters()).device
    ext = make_extractor_reusing(model, tok, device)
    da_policy = M.baseline_policies(include_diag=True).get("DualAnchor")
    cc_policy = M.crafted_v2_policies().get("CoreContent_v2_blockwise")
    assert da_policy and cc_policy, "missing tap policies"

    # base = clean greedy per task (cell-independent reference)
    base = {}
    for task in tasks:
        enc = tok(V.build_prompt(tok, task, "direct"), return_tensors="pt", truncation=True, max_length=1536).to(device)
        nl, cache, _cp, amf = C.prefill(model, enc["input_ids"], enc["attention_mask"])
        toks, _ = S.greedy_continue(model, cache, amf, nl, MNT, device)
        txt = tok.decode(toks, skip_special_tokens=True)
        c, fin = verify(task, txt)
        base[task["canary_id"]] = {"correct": bool(c), "final": (fin or "")[:60], "domain": task["domain"]}
        C.clear_cuda()

    loop1_loci = [(0, L) for L in LOOP1_LAYERS]        # early intervention (loop 1)
    cells = []
    tr_names = list(token_ranges(1024).keys())
    total = len(ALPHAS) * len(tr_names) * 2
    i = 0
    for alpha in ALPHAS:
        for tr_name in tr_names:
            for decode in ("greedy", "sample"):
                i += 1
                agg, _pt = run_cell(model, tok, ext, tasks, dirs, da_policy, cc_policy, device,
                                    base, alpha, tr_name, decode, loop1_loci)
                cells.append(agg)
                print(f"  [{i}/{total}] a={alpha} tr={tr_name:10} dec={decode:6} "
                      f"new_correct@base_missed={agg['new_correct_rate_on_base_missed']} "
                      f"oracle={agg['oracle_over_branches']} sel={agg['selected_acc']} "
                      f"loci_div={agg['loci_diverged_rate']} reconv={agg['reconvergence_rate']} "
                      f"spread={agg['mean_terminal_spread']}", flush=True)

    winners = [c for c in cells if c["new_correct_rate_on_base_missed"]]

    # LATER-LOOP SENTINEL: the single most aggressive cell (a=0.10, second-half, sample) forked at LOOP-4
    # loci (L4_36, L4_47). loop-1 tests EARLY intervention; a later-loop fork acts on a more semantically
    # formed reasoning state. Closes the "early-intervention-only" objection in the null case.
    sentinel = None
    if "secondhalf" in tr_names:
        print("  [sentinel] a=0.10 tr=secondhalf dec=sample loci=loop4_L36/L47", flush=True)
        sentinel, _pt = run_cell(model, tok, ext, tasks, dirs, da_policy, cc_policy, device,
                                 base, 0.10, "secondhalf", "sample", [(3, 36), (3, 47)])
        print(f"  [sentinel] new_correct@base_missed={sentinel['new_correct_rate_on_base_missed']} "
              f"oracle={sentinel['oracle_over_branches']} loci_div={sentinel['loci_diverged_rate']} "
              f"reconv={sentinel['reconvergence_rate']} spread={sentinel['mean_terminal_spread']}", flush=True)
    sentinel_new_correct = bool(sentinel and sentinel["new_correct_rate_on_base_missed"])

    any_new_correct = bool(winners) or sentinel_new_correct
    decision = {
        "metric": "new_correct_on_base_missed",
        "grid_any_cell_new_correct": bool(winners),
        "winning_cells": [{"alpha": c["alpha"], "token_range": c["token_range"], "decode": c["decode"],
                           "new_correct_tasks": c["new_correct_tasks"]} for c in winners],
        "sentinel_new_correct": sentinel_new_correct,
        "verdict": ("FORK_PARAM_SCREEN_FOUND_NEW_CORRECT__RUN_KMATCHED_SAMPLING_THEN_CHAINED_AT_WINNER"
                    if any_new_correct else
                    "NO_FROZEN_REACHABILITY_UNDER_TESTED_SINGLE_LOCUS_REGIMES"),
        "scope_caveat": ("NULL is scoped to SINGLE-LOCUS perturbation regimes — NOT a claim about chained "
                         "reachability anywhere. A single strong fork tests local magnitude, not accumulated "
                         "branch history or later-loop semantic maturity (the sentinel probes the latter). "
                         "Combined with the α=0.02 chained null, strong enough to stop local spending unless a "
                         "promising cell appears."),
        "next_if_new_correct": ["K-matched plain-sampling baseline (sampling alone may explain it)",
                                "chained loop at the winning regime"],
        "next_if_null": "treat frozen branching as effectively closed for LOCAL purposes; move to S3/backbone-training framing",
    }
    payload = {"verdict": "S1_4A_FORK_PARAMETER_SCREEN",
               "label": "S1.4a fork-parameter screen (single-locus fork probe; NOT the chained-loop result)",
               "config": {"N_PER": N_PER, "K": K, "MNT": MNT, "alphas": ALPHAS, "tr_set": TR_SET,
                          "token_ranges": tr_names, "decodes": ["greedy", "sample"], "temp": TEMP, "top_p": TOP_P,
                          "score_input_mode": "prompt_plus_answer", "grid_loci": "loop1_L24/L36/L47",
                          "sentinel_loci": "loop4_L36/L47 (a=0.10, secondhalf, sample)"},
               "base_per_task": base, "n_cells": len(cells), "cells": cells, "later_loop_sentinel": sentinel,
               "decision": decision, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4a_fork_param_screen.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"decision": decision, "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
    print(f"\n{decision['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
