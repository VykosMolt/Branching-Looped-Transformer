"""PART G — offline branch-set preference data (no online generation).

Builds: (1) branch_set_rejection_sft — accepted (verifier-positive, parseable, correct-final,
budget-appropriate) branch sets to SFT on; (2) branch_set_dpo — (chosen good set, rejected bad set)
pairs over the SAME task (oracle / correct-final / useful-vs-superficial diversity / parseable /
budget / valid-vs-hallucinated); (3) generator_reward_groups — deterministic reward records via the
Part-H reward function. External labels only; DualAnchor/CoreContent never supply correctness.
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402
from utilities.branch_training.offline_reward_v2 import group_reward  # noqa: E402

RLOG = V.TRAIN_V2 / "rendered_logic_branch_sets.jsonl"
LAB = V.DATA_ROOT / "processed/branch_pools_labeled.jsonl"
MAX_RLOG = int(os.environ.get("G_MAX_RLOG", "23000"))
MAX_LAB = int(os.environ.get("G_MAX_LAB", "19000"))
rng = random.Random(29)


def _btext(b):
    return (b.get("branch_text") or "").split("FINAL ANSWER")[0].strip()


def render_set(branches, final, cap=4):
    lines = []
    for i, b in enumerate(branches[:cap], 1):
        body = _btext(b)[:300] or f"Approach {i}."
        lines.append(f"Branch {i}: {body}")
    return "\n".join(lines) + f"\nFINAL ANSWER: {final}"


def main() -> int:
    started = time.time()
    V.TRAIN_V2.mkdir(parents=True, exist_ok=True)
    rejection, dpo, reward_groups = [], [], []

    def consume(group):
        ba = group.get("branch_attempts") or []
        if not ba:
            return
        dom = group.get("domain", "logic")
        prompt = group["task_prompt"]
        pos = [b for b in ba if b.get("external_label") == "pass"]
        neg = [b for b in ba if b.get("external_label") == "fail"]
        # reward record (deterministic)
        if pos:
            group["selected_final"] = pos[0].get("final_answer")
        rr = group_reward(group)
        reward_groups.append({"prompt": prompt[:600], "domain": dom, "n_branches": rr["n_branches"],
                              "n_positive": rr["n_positive"], "reward": rr["reward"],
                              "components": rr["components"]})
        if pos:
            gold = pos[0].get("final_answer")
            good = render_set([pos[0]] + neg[:2], gold)
            # 1) rejection SFT: accept the good set
            rejection.append({"prompt": prompt, "completion": good, "domain": dom, "reward": rr["reward"]})
            # 2) DPO oracle/correct-final pair: good set vs an all-wrong set with wrong final
            if neg:
                bad = render_set(neg[:3], neg[0].get("final_answer", "Unknown"))
                dpo.append({"prompt": prompt, "chosen": good, "rejected": bad, "domain": dom, "pair": "oracle"})
            # 3) DPO budget pair: concise correct (when easy) vs overbranched redundant
            if len(pos) >= 2 and len(ba) >= 4:
                concise = f"FINAL ANSWER: {gold}"
                overbr = render_set(ba, gold, cap=8)
                dpo.append({"prompt": prompt, "chosen": concise, "rejected": overbr, "domain": dom, "pair": "overbranch"})
        elif neg:  # all-negative group -> DPO: defer/explore vs confidently wrong
            explore = render_set(neg[:3], "Insufficient information; defer to external search")
            wrong = render_set(neg[:1], neg[0].get("final_answer", "Unknown"))
            dpo.append({"prompt": prompt, "chosen": explore, "rejected": wrong, "domain": dom, "pair": "underbranch_defer"})

    n = 0
    with open(RLOG) as f:
        for line in f:
            if n >= MAX_RLOG:
                break
            consume(json.loads(line))
            n += 1
    n = 0
    with open(LAB) as f:
        for line in f:
            if n >= MAX_LAB:
                break
            d = json.loads(line)
            if d.get("split") == "train":
                consume(d)
                n += 1

    # fold F's contrastive pairs into DPO (already chosen/rejected)
    for name, pair in (("overbranch_negative_pairs", "overbranch_F"), ("underbranch_negative_pairs", "underbranch_F")):
        p = V.TRAIN_V2 / f"{name}.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                if r.get("chosen") and r.get("rejected"):
                    dpo.append({"prompt": r.get("prompt", ""), "chosen": r["chosen"], "rejected": r["rejected"],
                                "domain": r.get("domain", "logic"), "pair": pair})

    # dedup + write
    def dump(rows, name, keyf):
        seen, uniq = set(), []
        for r in rows:
            k = keyf(r)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        with open(V.TRAIN_V2 / f"{name}.jsonl", "w") as fo:
            for r in uniq:
                fo.write(json.dumps(r, default=V.v2.json_default) + "\n")
        return len(uniq)

    n_rej = dump(rejection, "branch_set_rejection_sft", lambda r: (r["prompt"][:160], r["completion"][:120]))
    n_dpo = dump(dpo, "branch_set_dpo", lambda r: (r["prompt"][:160], r["chosen"][:100], r["rejected"][:100]))
    n_rwd = dump(reward_groups, "generator_reward_groups", lambda r: (r["prompt"][:160], r["reward"], r["n_branches"]))

    import statistics
    rwd_vals = [r["reward"] for r in reward_groups]
    manifest = {"branch_set_rejection_sft": n_rej, "branch_set_dpo": n_dpo, "generator_reward_groups": n_rwd,
                "dpo_pair_types": {}, "reward_mean": round(statistics.mean(rwd_vals), 3) if rwd_vals else None,
                "reward_min": round(min(rwd_vals), 3) if rwd_vals else None,
                "reward_max": round(max(rwd_vals), 3) if rwd_vals else None}
    import collections
    manifest["dpo_pair_types"] = dict(collections.Counter(r["pair"] for r in dpo))
    V.write_json(V.TRAIN_V2 / "offline_training_manifest.json", manifest)

    if n_rej >= 50000 and n_dpo >= 100000 and n_rwd >= 50000:
        verdict = "OFFLINE_PREFS_READY"
    elif n_rej >= 20000 and n_dpo >= 50000 and n_rwd >= 20000:
        verdict = "OFFLINE_PREFS_READY"
    elif n_dpo >= 50000:
        verdict = "DPO_READY"
    elif n_rej >= 20000:
        verdict = "REJECTION_SFT_READY"
    elif n_rwd >= 20000:
        verdict = "REWARD_GROUPS_READY"
    else:
        verdict = "DATA_TOO_SMALL"

    payload = {"OFFLINE_BRANCH_SET_PREFERENCE_VERDICT": verdict, **manifest,
               "elapsed_seconds": round(time.time() - started, 1)}
    V.write_json(V.OUT_ROOT / "offline_branch_set_preferences.json", payload)
    V.write_md(V.OUT_ROOT / "offline_branch_set_preferences.md", [
        "# Offline Branch-Set Preferences (Part G)", "", V.status_line("OFFLINE_BRANCH_SET_PREFERENCE_VERDICT", verdict),
        f"- branch_set_rejection_sft: **{n_rej}** accepted (verifier-positive, correct-final) branch sets",
        f"- branch_set_dpo: **{n_dpo}** pairs — types {manifest['dpo_pair_types']}",
        f"- generator_reward_groups: **{n_rwd}** reward records (mean {manifest['reward_mean']}, "
        f"range [{manifest['reward_min']}, {manifest['reward_max']}])",
        "", "Built offline from E rendered logic + labeled train pools + F contrastive pairs; deterministic reward via "
        "`utilities/branch_training/offline_reward_v2.py`. No online generation. Diversity rewarded only when tied to "
        "verifier-positive branches; budget pairs penalize overbranching easy / underbranching hard.",
    ])
    V.set_stage("G_offline_prefs", verdict, manifest)
    V.prog("G_offline_prefs", {"verdict": verdict, **manifest})
    print(V.status_line("OFFLINE_BRANCH_SET_PREFERENCE_VERDICT", verdict))
    print(f"  rejection_sft {n_rej} | dpo {n_dpo} {manifest['dpo_pair_types']} | reward_groups {n_rwd} "
          f"(mean {manifest['reward_mean']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
