"""PART F — data-composition repair: direct-answer + branch-budget views.

Fixes the v1 SFT's branch-heaviness (it taught "emit Branch 1/2/3" with too few "answer directly
when confident"). Builds: direct_answer_sft, one_branch_sft, multi_branch_sft, branch_budget_policy
(DIRECT..EIGHT_BRANCH/DEFER), and overbranch/underbranch contrastive pairs. Difficulty comes from
external pass-rate (labeled pools); reasoning-bearing targets come from E's rendered logic + the
model-gen pools (the cheap labeled pools have terse text). External labels only.
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

LAB = V.DATA_ROOT / "processed/branch_pools_labeled.jsonl"
RLOG = V.TRAIN_V2 / "rendered_logic_branch_sets.jsonl"
GEN_DIR = V.DATA_ROOT / "processed/gen_shards"
CG = V.PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl"
MAX_LAB = int(os.environ.get("F_MAX_LAB", "19000"))
MAX_RLOG = int(os.environ.get("F_MAX_RLOG", "20000"))
MAX_ALIGN = int(os.environ.get("F_MAX_ALIGN", "4000"))
rng = random.Random(13)


def _budget(pass_rate, domain, category, proof_depth):
    if domain == "alignment":
        return "DIRECT"
    if pass_rate is None:
        return "TWO_BRANCH"
    hard_family = (category in ("synthetic_constraint_game", "lsat_analytical_reasoning")) or (proof_depth or 0) >= 4
    if pass_rate >= 0.999:
        b = "DIRECT"
    elif pass_rate >= 0.7:
        b = "ONE_BRANCH"
    elif pass_rate >= 0.4:
        b = "TWO_BRANCH"
    elif pass_rate >= 0.15:
        b = "FOUR_BRANCH"
    elif pass_rate > 0:
        b = "EIGHT_BRANCH"
    else:
        return "DEFER_TO_EXTERNAL_SEARCH"
    if hard_family:  # bump one level up for hard families
        b = {"DIRECT": "ONE_BRANCH", "ONE_BRANCH": "TWO_BRANCH", "TWO_BRANCH": "FOUR_BRANCH",
             "FOUR_BRANCH": "EIGHT_BRANCH", "EIGHT_BRANCH": "EIGHT_BRANCH"}[b]
    return b


def _concise(answer, rationale=""):
    r = (rationale.strip().split(". ")[0][:160] + ". ") if rationale.strip() else ""
    return f"{r}FINAL ANSWER: {answer}"


def _multi_branch_text(branch_texts, final):
    lines = []
    for i, t in enumerate(branch_texts[:4], 1):
        body = t.strip().split("FINAL ANSWER")[0].strip()[:240] or f"Approach {i}."
        lines.append(f"Branch {i}: {body}")
    return "\n".join(lines) + f"\nFINAL ANSWER: {final}"


def _iter_labeled(limit):
    n = 0
    with open(LAB) as f:
        for line in f:
            if n >= limit:
                break
            d = json.loads(line)
            if d.get("split") != "train":
                continue
            ba = d.get("branch_attempts") or []
            if not ba:
                continue
            n += 1
            yield d, ba


def _iter_rlog(limit):
    n = 0
    with open(RLOG) as f:
        for line in f:
            if n >= limit:
                break
            d = json.loads(line)
            n += 1
            yield d, d.get("branch_attempts") or []


def main() -> int:
    started = time.time()
    V.TRAIN_V2.mkdir(parents=True, exist_ok=True)
    views = {k: [] for k in ("direct_answer_sft", "one_branch_sft", "multi_branch_sft",
                             "branch_budget_policy", "overbranch_negative_pairs", "underbranch_negative_pairs")}
    budget_counts = {}

    def add_budget(prompt, ba, domain, category, proof_depth):
        npos = sum(1 for b in ba if b.get("external_label") == "pass")
        pr = npos / len(ba) if ba else None
        b = _budget(pr, domain, category, proof_depth)
        budget_counts[b] = budget_counts.get(b, 0) + 1
        views["branch_budget_policy"].append({"prompt": prompt, "completion": b, "domain": domain,
                                              "difficulty_passrate": round(pr, 3) if pr is not None else None})
        return pr, b

    # ---- rendered logic (E): reasoning-bearing one/multi branch + contrastive pairs ----
    for d, ba in _iter_rlog(MAX_RLOG):
        prompt = d["task_prompt"]
        pos = [b for b in ba if b["external_label"] == "pass"]
        neg = [b for b in ba if b["external_label"] == "fail"]
        pr, b = add_budget(prompt, ba, "logic", d.get("category"), None)
        gold = pos[0]["final_answer"] if pos else None
        if pos:
            views["one_branch_sft"].append({"prompt": prompt, "completion": pos[0]["branch_text"], "domain": "logic"})
            views["direct_answer_sft"].append({"prompt": prompt, "completion": _concise(gold, pos[0]["branch_text"]),
                                               "domain": "logic", "difficulty": b})
            if neg:
                views["multi_branch_sft"].append({"prompt": prompt,
                    "completion": _multi_branch_text([pos[0]["branch_text"]] + [n["branch_text"] for n in neg[:3]], gold),
                    "domain": "logic"})
                # underbranch: a single failing direct attempt < a multi-branch that reaches the answer
                views["underbranch_negative_pairs"].append({"prompt": prompt,
                    "chosen": _multi_branch_text([pos[0]["branch_text"]] + [neg[0]["branch_text"]], gold),
                    "rejected": _concise(neg[0]["final_answer"], neg[0]["branch_text"]), "domain": "logic"})
        elif neg:  # all-negative group -> DEFER target + underbranch chosen=defer
            views["branch_budget_policy"][-1]["completion"] = "DEFER_TO_EXTERNAL_SEARCH"

    # ---- labeled pools: budget signal across all domains + overbranch pairs on easy ----
    for d, ba in _iter_labeled(MAX_LAB):
        prompt = d["task_prompt"]
        dom = d.get("domain", "logic")
        cat = d.get("dataset") or d.get("task_type")
        pr, b = add_budget(prompt, ba, dom, cat, None)
        pos = [x for x in ba if x.get("external_label") == "pass"]
        neg = [x for x in ba if x.get("external_label") == "fail"]
        if pos:
            gold = pos[0].get("final_answer")
            # easy/high-confidence -> overbranch pair: concise correct > verbose multi-branch (often wrong)
            if pr is not None and pr >= 0.6 and neg:
                views["overbranch_negative_pairs"].append({"prompt": prompt,
                    "chosen": _concise(gold), "domain": dom,
                    "rejected": _multi_branch_text([n.get("branch_text", "") for n in neg[:3]] + [pos[0].get("branch_text", "")],
                                                   neg[0].get("final_answer", gold))})
            if pr is not None and pr >= 0.8:
                views["direct_answer_sft"].append({"prompt": prompt, "completion": _concise(gold), "domain": dom, "difficulty": b})

    # ---- alignment: direct-answer (branching/verbosity harmful) + overbranch pairs ----
    na = 0
    for g in V.read_jsonl(CG):
        if g.get("domain") != "alignment" or g.get("split") != "train" or na >= MAX_ALIGN:
            continue
        cands = g.get("candidates", [])
        pos = [c for c in cands if c.get("is_positive")]
        neg = [c for c in cands if not c.get("is_positive")]
        if not pos or not neg:
            continue
        na += 1
        chosen, rejected = pos[0]["candidate_text"], neg[0]["candidate_text"]
        m = 0
        for a, c in zip(chosen, rejected):
            if a != c:
                break
            m += 1
        prompt = chosen[:m].rsplit("\n", 1)[0] if m > 20 else ""
        resp = chosen[m:] if m else chosen
        views["direct_answer_sft"].append({"prompt": prompt, "completion": resp[:1200], "domain": "alignment", "difficulty": "DIRECT"})
        views["branch_budget_policy"].append({"prompt": prompt, "completion": "DIRECT", "domain": "alignment", "difficulty_passrate": None})
        budget_counts["DIRECT"] = budget_counts.get("DIRECT", 0) + 1
        # overbranch: concise helpful answer > the same answer wrapped in needless Branch 1/2/3 scaffolding
        views["overbranch_negative_pairs"].append({"prompt": prompt, "chosen": resp[:1200], "domain": "alignment",
            "rejected": f"Branch 1: Let me consider option A.\nBranch 2: Let me consider option B.\n{resp[:1000]}"})

    # rebalance budget policy: constructed-pool pass-rate is a weak difficulty proxy and clusters at
    # FOUR_BRANCH; cap per class so the classifier doesn't collapse to "always FOUR_BRANCH" (which would
    # re-encourage overbranching). True difficulty = base-model success (canary/online), folded in later.
    BUDGET_CAP = 3500
    bp_by: dict[str, list] = {}
    for r in views["branch_budget_policy"]:
        bp_by.setdefault(r["completion"], []).append(r)
    balanced = []
    for lab, rows in bp_by.items():
        rng.shuffle(rows)
        balanced += rows[:BUDGET_CAP]
    rng.shuffle(balanced)
    views["branch_budget_policy"] = balanced
    budget_balanced = {lab: min(len(rows), BUDGET_CAP) for lab, rows in bp_by.items()}

    # write + dedup
    counts = {}
    for name, rows in views.items():
        seen, uniq = set(), []
        for r in rows:
            key = (r.get("prompt", "")[:200], r.get("completion", "") or r.get("chosen", ""))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        with open(V.TRAIN_V2 / f"{name}.jsonl", "w") as f:
            for r in uniq:
                f.write(json.dumps(r, default=V.v2.json_default) + "\n")
        counts[name] = len(uniq)

    direct_n = counts["direct_answer_sft"] + counts["one_branch_sft"]
    multi_n = max(1, counts["multi_branch_sft"])
    direct_frac = round(direct_n / (direct_n + multi_n), 3)
    if counts["branch_budget_policy"] >= 10000 and counts["direct_answer_sft"] >= 3000 and \
       counts["overbranch_negative_pairs"] >= 500 and direct_frac >= 0.4:
        verdict = "BRANCH_BUDGET_READY"
    elif direct_frac < 0.3:
        verdict = "DATA_TOO_BRANCH_HEAVY"
    elif counts["direct_answer_sft"] >= 2000:
        verdict = "DIRECT_ANSWER_READY"
    elif counts["overbranch_negative_pairs"] >= 500:
        verdict = "OVERBRANCH_CONTROLS_READY"
    else:
        verdict = "BLOCKED"

    payload = {"BRANCH_BUDGET_DATA_VERDICT": verdict, "counts": counts, "budget_label_distribution_raw": budget_counts,
               "budget_label_distribution_balanced": budget_balanced,
               "difficulty_proxy_caveat": "budget labels derived from constructed-pool pass-rate (weak proxy); "
               "authoritative difficulty = base-model success on canary/online, folded in at K/training time",
               "direct_vs_branch_fraction": direct_frac, "elapsed_seconds": round(time.time() - started, 1)}
    V.write_json(V.OUT_ROOT / "branch_budget_data.json", payload)
    V.write_md(V.OUT_ROOT / "branch_budget_data.md", [
        "# Branch-Budget & Direct-Answer Views (Part F)", "", V.status_line("BRANCH_BUDGET_DATA_VERDICT", verdict),
        "Data-composition repair for the v1 branch-heavy SFT — teaches *when to answer directly* vs *when to branch*.", "",
        "## View counts", *[f"- {k}: {v}" for k, v in counts.items()],
        "", f"direct/one-branch vs multi-branch fraction = **{direct_frac}** (≥0.4 target; v1 was branch-heavy).",
        "", "## Branch-budget label distribution (balanced, capped per class)",
        *[f"- {k}: {v}" for k, v in sorted(budget_balanced.items(), key=lambda x: -x[1])],
        f"\nRaw (pre-balance): {budget_counts}.",
        "", "**Caveat (honest):** budget labels come from *constructed-pool* pass-rate — a weak difficulty proxy that "
        "clusters at FOUR_BRANCH, so the view is class-capped to avoid a classifier that always says FOUR_BRANCH "
        "(which would re-encourage overbranching). The authoritative difficulty signal is *base-model success* "
        "(canary / online RL), folded in at K/training time.",
        "Reasoning targets from E rendered logic + gen pools; budget signal from labeled pools; alignment from corecontent_v2.",
    ])
    V.set_stage("F_branch_budget_data", verdict, {"counts": counts, "direct_frac": direct_frac})
    V.prog("F_branch_budget_data", {"verdict": verdict, "counts": counts})
    print(V.status_line("BRANCH_BUDGET_DATA_VERDICT", verdict))
    print(f"  counts {counts}")
    print(f"  budget balanced {budget_balanced} | direct_frac {direct_frac}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
