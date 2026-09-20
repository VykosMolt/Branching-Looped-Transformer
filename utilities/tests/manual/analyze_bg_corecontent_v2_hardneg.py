"""CoreContent v2 follow-up (b2): wrong-problem hard negatives for coding.

For each coding heldout group, build [own canonical (reward1), 2x other problems' canonical
solutions (reward0, real+compiling but wrong for this prompt)], extract frozen features, and
test whether the selected blockwise tap (and HH) rank the true canonical above plausible
wrong-problem code. Tests prompt-candidate RELEVANCE vs generic code-quality.
"""
from __future__ import annotations
import json, time
import torch
import bg_corecontent_v2_common as v2
import bg_corecontent_v2_features as feat
import bg_corecontent_v2_models as M
import bg_core_tap_audit_v1_common as cc

OUT = v2.OUT_ROOT
MAXLEN = 1024


def _coding_heldout_tasks():
    rows = v2.read_jsonl(v2.PROC_ROOT / "candidate_groups_deduped.jsonl")
    out = []
    for g in rows:
        if g["domain"] != "coding" or g["split"] != "heldout":
            continue
        pos = next((c for c in g["candidates"] if c["reward"] > 0), None)
        if not pos or not pos.get("verify_code"):
            continue
        sol = pos["verify_code"]
        txt = pos["candidate_text"]
        prompt = txt[:-len(sol)].rstrip("\n") if txt.endswith(sol) else txt.rsplit("\n" + sol, 1)[0]
        out.append({"group_uid": g["group_uid"], "prompt": prompt, "sol": sol})
    return out


def main() -> int:
    t0 = time.time(); v2.ensure_root()
    tasks = _coding_heldout_tasks()
    if len(tasks) < 4:
        v2.write_json(OUT / "followups_hardneg.json", {"error": "too few coding heldout tasks", "n": len(tasks)})
        print("too few tasks"); return 1
    ext = feat._make_extractor()
    blk = M.crafted_v2_policies().get("CoreContent_v2_blockwise", [])
    blk_no47 = [c for c in blk if c[2] != "47_L4"]
    hh = cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", [])
    n = len(tasks)
    res_blk = []; res_no47 = []; res_hh = []
    done = 0
    for i, t in enumerate(tasks):
        # two deterministic distinct other-problem solutions
        j1 = (i + 1 + v2.stable_int("hn1", t["group_uid"]) % (n - 1)) % n
        j2 = (i + 1 + v2.stable_int("hn2", t["group_uid"]) % (n - 1)) % n
        if j1 == i:
            j1 = (i + 1) % n
        if j2 == i or j2 == j1:
            j2 = (i + 2) % n
        cand_texts = [f"{t['prompt']}\n{t['sol']}",
                      f"{t['prompt']}\n{tasks[j1]['sol']}",
                      f"{t['prompt']}\n{tasks[j2]['sol']}"]
        rewards = [1.0, 0.0, 0.0]
        feats = []
        ok = True
        for ct in cand_texts:
            try:
                feats.append(ext.encode_text_to_pooled_features(ct, max_length=MAXLEN).to(torch.float32))
            except Exception:
                ok = False; break
        if not ok:
            continue
        group = {"group_id": t["group_uid"], "domain": "coding",
                 "candidates": [{"features": feats[k], "reward": rewards[k]} for k in range(3)]}
        for channels, store in ((blk, res_blk), (blk_no47, res_no47), (hh, res_hh)):
            m = cc.group_selection_metrics(group, cc.policy_candidate_scores(group, channels))
            if m:
                store.append(m["top1_oracle"])
        done += 1
        if done % 50 == 0:
            print(f"  hardneg {done}/{n}", flush=True)
    try:
        ext.cleanup()
    except Exception:
        pass
    out = {"groups": done,
           "blockwise_top1_wrongproblem": round(cc.finite_mean(res_blk), 4),
           "blockwise_no47_top1_wrongproblem": round(cc.finite_mean(res_no47), 4),
           "hh_top1_wrongproblem": round(cc.finite_mean(res_hh), 4),
           "note": "negatives = real compiling solutions to other coding problems (wrong for this prompt)",
           "elapsed_seconds": round(time.time() - t0, 3)}
    v2.write_json(OUT / "followups_hardneg.json", out)
    v2.write_md(OUT / "followups_hardneg.md", [
        "# CoreContent v2 follow-up (b2): wrong-problem hard negatives (coding heldout)", "",
        f"Negatives are real, compiling canonical solutions to *other* problems (relevance test). n={done}.", "",
        f"- blockwise top1: **{out['blockwise_top1_wrongproblem']}**",
        f"- blockwise (24+36, no L47) top1: {out['blockwise_no47_top1_wrongproblem']}",
        f"- mixedhead_MIX_HH_OBJECTIVE top1: {out['hh_top1_wrongproblem']}"])
    print("=== (b2) wrong-problem hard negatives (coding heldout) ===")
    print(f"  groups={done}")
    print(f"  blockwise={out['blockwise_top1_wrongproblem']}  blockwise_no47={out['blockwise_no47_top1_wrongproblem']}  HH={out['hh_top1_wrongproblem']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
