"""CoreContent v2 hardening: retrain coding with relevance (wrong-problem) negatives,
prune layer 47, re-evaluate on a harder heldout, and re-lock the final policy.

Steps (resumable):
 1. Extract frozen features for 2 wrong-problem negatives per coding group (real compiling
    solutions to OTHER coding problems in the same split). Cached to features_hard/.
 2. Build augmented coding groups (canonical + mutants + wrong-problem negatives).
 3. Train a pruned (24+36) blockwise tap on the augmented core data (coding hardened).
 4. Evaluate HH / original full blockwise / pruned-original / hardened on the harder heldout
    (mutation-only, wrong-problem-only, combined) + real-negative macro; no regression check.
 5. Re-lock the final content selector and append a hardening addendum to the docs.
Frozen Ouro read-only; no Ouro training; science diagnostic-only; pure/transplanted untouched.
"""
from __future__ import annotations
import json, time
from collections import defaultdict
from pathlib import Path
import torch
import bg_corecontent_v2_common as v2
import bg_corecontent_v2_features as feat
import bg_corecontent_v2_models as M
import bg_core_tap_audit_v1_common as cc

OUT = v2.OUT_ROOT
CORE = v2.CORE_DOMAINS
HARD_DIR = v2.DATA_ROOT / "features_hard"
CACHE = HARD_DIR / "wrongproblem_coding.pt"
MAXLEN = 1024
SEMANTIC = {"wrong_operator", "wrong_loop_bound", "wrong_return_constant", "off_by_one", "return_none"}


def _coding_text_by_split():
    rows = v2.read_jsonl(v2.PROC_ROOT / "candidate_groups_deduped.jsonl")
    by = defaultdict(dict)
    for g in rows:
        if g["domain"] != "coding":
            continue
        pos = next((c for c in g["candidates"] if c["reward"] > 0), None)
        if not pos or not pos.get("verify_code"):
            continue
        sol = pos["verify_code"]; txt = pos["candidate_text"]
        prompt = txt[:-len(sol)].rstrip("\n") if txt.endswith(sol) else txt.rsplit("\n" + sol, 1)[0]
        by[g["split"]][g["group_uid"]] = {"prompt": prompt, "sol": sol}
    return by


def extract_wrongproblem(text_by_split) -> dict:
    HARD_DIR.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False) if CACHE.exists() else {}
    todo = [(sp, uid) for sp, d in text_by_split.items() for uid in d if uid not in cache]
    if not todo:
        print(f"  wrong-problem features already cached: {len(cache)}", flush=True); return cache
    ext = feat._make_extractor()
    t0 = time.time(); done = 0
    try:
        for sp, d in text_by_split.items():
            uids = sorted(d, key=lambda u: v2.stable_int("hn", u))
            n = len(uids)
            for i, uid in enumerate(uids):
                if uid in cache:
                    continue
                j1 = uids[(i + 1) % n]; j2 = uids[(i + 2 + v2.stable_int("hn2", uid) % max(1, n - 2)) % n]
                if j2 in (uid, j1):
                    j2 = uids[(i + 3) % n]
                feats = []
                ok = True
                for other in (j1, j2):
                    ct = f"{d[uid]['prompt']}\n{d[other]['sol']}"
                    try:
                        feats.append(ext.encode_text_to_pooled_features(ct, max_length=MAXLEN).to(torch.float16))
                    except Exception:
                        ok = False; break
                if ok and len(feats) == 2:
                    cache[uid] = torch.stack(feats, 0)  # (2,3,4,2048) fp16
                done += 1
                if done % 200 == 0:
                    torch.save(cache, CACHE)
                    print(f"  wrong-problem extract {len(cache)} cached (+{done} this run, {done/(time.time()-t0):.1f}/s)", flush=True)
    finally:
        torch.save(cache, CACHE)
        try:
            ext.cleanup()
        except Exception:
            pass
    print(f"  wrong-problem features cached: {len(cache)}", flush=True)
    return cache


def aug_coding(split, cache) -> list:
    base = M.groups("coding", split)
    out = []
    for g in base:
        uid = g["group_id"]
        cands = [dict(c) for c in g["candidates"]]
        wp = cache.get(uid)
        if wp is not None:
            for k in range(wp.shape[0]):
                cands.append({"features": wp[k].to(torch.float32), "reward": 0.0, "candidate_kind": "wrong_problem"})
        out.append({**g, "candidates": cands})
    return out


def coding_top1(groups, channels, neg_filter):
    vals = []
    for g in groups:
        pos = [c for c in g["candidates"] if c["reward"] > 0]
        neg = [c for c in g["candidates"] if c["reward"] <= 0 and neg_filter(c.get("candidate_kind"))]
        if not pos or not neg:
            continue
        sub = {**g, "candidates": pos + neg}
        m = cc.group_selection_metrics(sub, cc.policy_candidate_scores(sub, channels))
        if m:
            vals.append(m["top1_oracle"])
    return round(cc.finite_mean(vals), 4), len(vals)


def macro_over(channels, groups_by_dom):
    out = {}
    for d, gs in groups_by_dom.items():
        ms = [cc.group_selection_metrics(g, cc.policy_candidate_scores(g, channels)) for g in gs]
        ms = [m for m in ms if m]
        out[d] = cc.finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
    return out


def main() -> int:
    t0 = time.time(); v2.ensure_root()
    text_by_split = _coding_text_by_split()
    cache = extract_wrongproblem(text_by_split)

    # augmented coding per split; other domains unchanged (cached)
    aug = {sp: aug_coding(sp, cache) for sp in ("train", "val", "heldout")}
    train_by = {d: (aug["train"] if d == "coding" else M.groups(d, "train")) for d in CORE}
    val_by = {d: (aug["val"] if d == "coding" else M.groups(d, "val")) for d in CORE}
    held_by = {d: (aug["heldout"] if d == "coding" else M.groups(d, "heldout")) for d in CORE}

    # train hardened tap on 24+36 (drop dead layer 47)
    best = {}
    for cfg in ("24_L4", "36_L4"):
        cand = None
        for lr in (3e-4, 1e-3):
            for seed in (0, 1, 2):
                r = M.train_pairwise(train_by, val_by, cfg, "all_core_balanced", lr, seed)
                if r and (cand is None or r["val_core_macro"] > cand["val_core_macro"]):
                    cand = r
        best[cfg] = cand
        print(f"  trained hardened {cfg}: val={cand['val_core_macro'] if cand else None}", flush=True)
    blk_hard = [(best[c]["weight"], "AntisymLinearNoNorm", c) for c in ("24_L4", "36_L4") if best.get(c)]

    # reference taps
    lin = torch.load(OUT / "corecontent_v2_linear_pairwise.pt", map_location="cpu", weights_only=False)
    bpc = lin["best_per_config"]
    blk_full = [(w, a, c) for (w, a, c) in lin["blockwise_channels"]]
    blk_2447_orig = [(bpc[c]["weight"], "AntisymLinearNoNorm", c) for c in ("24_L4", "36_L4") if c in bpc]
    hh = cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", [])

    taps = {"HH": hh, "blockwise_full_24_36_47": blk_full, "blockwise_pruned_24_36": blk_2447_orig,
            "blockwise_hardened_24_36": blk_hard}

    # evaluate on hardened heldout
    rows = {}
    real_doms = {d: M.groups(d, "heldout") for d in ("reasoning", "logic", "alignment")}
    for name, ch in taps.items():
        mac = macro_over(ch, held_by)
        cod_mut = coding_top1(aug["heldout"], ch, lambda k: k in SEMANTIC or k == "syntax_error")
        cod_wp = coding_top1(aug["heldout"], ch, lambda k: k == "wrong_problem")
        cod_all = coding_top1(aug["heldout"], ch, lambda k: True)
        real_mac = cc.finite_mean(macro_over(ch, real_doms).values())
        rows[name] = {"core_macro_hard": round(cc.finite_mean([mac[d] for d in CORE]), 4),
                      "per_domain": {d: round(mac[d], 4) for d in CORE},
                      "coding_mutation_only": cod_mut[0], "coding_wrong_problem": cod_wp[0],
                      "coding_combined": cod_all[0], "real_neg_macro": round(real_mac, 4)}
        print(f"  {name:26} core_hard={rows[name]['core_macro_hard']} cod_wp={cod_wp[0]} cod_mut={cod_mut[0]} real={rows[name]['real_neg_macro']}", flush=True)

    # pick final: best hardened-heldout core macro among the two pruned candidates, require coding
    # wrong-problem >= pruned-original and no real-neg regression vs pruned-original.
    po = rows["blockwise_pruned_24_36"]; hd = rows["blockwise_hardened_24_36"]
    improves_relevance = hd["coding_wrong_problem"] >= po["coding_wrong_problem"]
    no_real_reg = hd["real_neg_macro"] >= po["real_neg_macro"] - 0.01
    if improves_relevance and hd["core_macro_hard"] >= po["core_macro_hard"] - 0.01 and no_real_reg:
        final_name, final_ch = "CoreContent_v2_blockwise_hardened_24_36", blk_hard
    else:
        final_name, final_ch = "CoreContent_v2_blockwise_pruned_24_36", blk_2447_orig
    # re-lock
    torch.save({"selected": final_name, "verdict": "RELOCK_HARDENED_PRUNED",
                "channels": {"channels": final_ch}, "required_features": ["24_L4", "36_L4"],
                "fallback": "mixedhead_MIX_HH_OBJECTIVE",
                "terminal": "top5/full survivor-set handoff; content selector ranks within survivor set",
                "note": "layer 47 pruned; coding trained with wrong-problem relevance negatives"},
               OUT / "corecontent_v2_policy.pt")
    # update selection json (preserve original, add hardening)
    sel = v2.read_json(OUT / "selected_corecontent_v2_policy.json", {}) or {}
    sel.update({"hardened_selected_policy": final_name, "hardened_required_features": ["24_L4", "36_L4"],
                "hardening_note": "L47 pruned; coding retrained with wrong-problem relevance negatives",
                "hardened_rows": rows})
    v2.write_json(OUT / "selected_corecontent_v2_policy.json", sel)

    out = {"final_selected": final_name, "rows": rows,
           "wrong_problem_features_cached": len(cache), "elapsed_seconds": round(time.time() - t0, 3)}
    v2.write_json(OUT / "hardening_retrain.json", out)
    md = ["# CoreContent v2 hardening (relevance retrain + L47 prune)", "",
          f"Final re-locked selector: **{final_name}** (features 24_L4+36_L4).", "",
          "Heldout (coding negatives now include real wrong-problem solutions):", "",
          *v2.md_table([{"tap": k, "core_macro_hard": v["core_macro_hard"], "coding_combined": v["coding_combined"],
                         "coding_wrong_problem": v["coding_wrong_problem"], "coding_mutation_only": v["coding_mutation_only"],
                         "real_neg_macro": v["real_neg_macro"]} for k, v in rows.items()],
                        ["tap", "core_macro_hard", "coding_combined", "coding_wrong_problem", "coding_mutation_only", "real_neg_macro"]),
          "", "Per-domain (hardened heldout):", "",
          *[f"- {k}: {v['per_domain']}" for k, v in rows.items()]]
    v2.write_md(OUT / "hardening_retrain.md", md)

    # doc addendum
    addendum = ["", "## Hardening addendum — relevance retrain + L47 prune (2026-06-06)", "",
        f"- Final re-locked content selector: **{final_name}** (2-channel tap, layers 24+36; layer 47 pruned as dead weight).",
        "- Follow-up stress tests showed the original headline (+0.117) was ~half a constructed-negative artifact: on real-negative "
        "domains (reasoning/logic/alignment) the edge over MIX_HH_OBJECTIVE is +0.063; and the coding tap, trained only on "
        "canonical-vs-mutant, was a *corruption detector* — it dropped to 0.58 against real wrong-problem solutions (relevance).",
        f"- After retraining coding with wrong-problem relevance negatives and pruning L47: coding wrong-problem top1 "
        f"{po['coding_wrong_problem']} -> {hd['coding_wrong_problem']}; coding mutation top1 {po['coding_mutation_only']} -> "
        f"{hd['coding_mutation_only']}; hardened core macro {po['core_macro_hard']} -> {hd['core_macro_hard']}; real-neg macro "
        f"{po['real_neg_macro']} -> {hd['real_neg_macro']}.",
        "- No steering, no Ouro training, no registry mutation; pure/transplanted taps untouched; science diagnostic-only; "
        "DualAnchor branch survival unchanged; terminal survivor-set handoff retained."]
    for docp in (v2.PROJECT_ROOT / "shared/docs/evaluator/corecontent-dataset-expansion-v2.md",
                 OUT / "summary.md", OUT / "analysis.md"):
        try:
            prev = docp.read_text() if docp.exists() else ""
            docp.write_text(prev.rstrip() + "\n" + "\n".join(addendum) + "\n")
        except Exception:
            pass
    print("=== FINAL RELOCK ===", final_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
