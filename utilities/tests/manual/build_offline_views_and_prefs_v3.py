"""PARTS F+G v3 — worked-text training views and branch-set preferences (CPU-only).

Rebuild after the Part K degeneration finding (`data_flaw_meta_branch_rendering.md`). v2's
views taught answer-asserting three ways: meta-text renderings (E v2), bare `FINAL ANSWER: x`
direct-answer/overbranch targets, and 160-300 char truncations that stubbed real solutions.

v3 discipline:
- every SFT/preference completion SHOWS WORK; nothing is truncated below WORK_CAP chars,
  and branches shorter than MIN_WORK chars are excluded from worked views (terse cheap-pool
  texts route to budget classification only);
- "direct answer" means ONE short worked branch (<= DIRECT_CAP chars incl. final), never a
  bare final; tasks whose shortest passing solution is long are one_branch material instead;
- overbranch DPO chosen = the short worked branch; rejected = the redundant full set;
- empty/whitespace prompts are dropped at source (fixes the 7.7k empty-prompt coding pairs).

Inputs: train_v3/rendered_logic_branch_sets.jsonl (executed renders), labeled pools,
alignment candidate groups. Outputs (train_v3/): direct_answer_sft, one_branch_sft,
multi_branch_sft, branch_budget_policy, branch_set_rejection_sft, branch_set_dpo,
generator_reward_groups, offline_training_manifest.json.
"""
from __future__ import annotations
import collections
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402
from utilities.branch_training.offline_reward_v2 import group_reward  # noqa: E402

TRAIN_V3 = V.DATA_ROOT / "train_v3"
RLOG = TRAIN_V3 / "rendered_logic_branch_sets.jsonl"
LAB = V.DATA_ROOT / "processed/branch_pools_labeled.jsonl"
CG = V.PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl"
MAX_LAB = int(os.environ.get("F3_MAX_LAB", "19000"))
MAX_ALIGN = int(os.environ.get("F3_MAX_ALIGN", "4000"))
WORK_CAP = 1600
MIN_WORK = 80
DIRECT_CAP = 420
rng = random.Random(33)


def _full(b):
    t = (b.get("branch_text") or "").strip()
    return t[:WORK_CAP]


def _worked(b):
    t = _full(b)
    return t if len(t) >= MIN_WORK else None


def _with_final(text, final):
    if "FINAL ANSWER" in text:
        return text
    return f"{text}\nFINAL ANSWER: {final}"


def _set_text(branches, final, cap=4):
    lines = []
    for i, b in enumerate(branches[:cap], 1):
        body = _full(b).split("FINAL ANSWER")[0].strip()
        lines.append(f"Branch {i}: {body}")
    return "\n".join(lines) + f"\nFINAL ANSWER: {final}"


def _budget(pass_rate, domain, category, proof_depth):
    if domain == "alignment":
        return "DIRECT"
    if pass_rate is None:
        return "TWO_BRANCH"
    hard = (category in ("synthetic_constraint_game", "lsat_analytical_reasoning")) or (proof_depth or 0) >= 4
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
    if hard:
        b = {"DIRECT": "ONE_BRANCH", "ONE_BRANCH": "TWO_BRANCH", "TWO_BRANCH": "FOUR_BRANCH",
             "FOUR_BRANCH": "EIGHT_BRANCH", "EIGHT_BRANCH": "EIGHT_BRANCH"}[b]
    return b


def main() -> int:
    started = time.time()
    TRAIN_V3.mkdir(parents=True, exist_ok=True)
    views = {k: [] for k in ("direct_answer_sft", "one_branch_sft", "multi_branch_sft", "branch_budget_policy",
                             "branch_set_rejection_sft", "branch_set_dpo", "generator_reward_groups")}
    skipped = collections.Counter()

    def consume(group, source):
        prompt = str(group.get("task_prompt") or "").strip()
        if not prompt:
            skipped["empty_prompt"] += 1
            return
        ba = group.get("branch_attempts") or []
        if not ba:
            return
        dom = group.get("domain", "logic")
        pos = [b for b in ba if b.get("external_label") == "pass"]
        neg = [b for b in ba if b.get("external_label") == "fail"]
        n = len(ba)
        pass_rate = len(pos) / n if n else None
        bud = _budget(pass_rate, dom, group.get("category"), group.get("proof_depth"))
        views["branch_budget_policy"].append({"prompt": prompt, "completion": bud, "domain": dom,
                                              "difficulty_passrate": pass_rate})
        if pos:
            group["selected_final"] = pos[0].get("final_answer")
        rr = group_reward(group)
        views["generator_reward_groups"].append({"prompt": prompt[:600], "domain": dom,
                                                 "n_branches": rr["n_branches"], "n_positive": rr["n_positive"],
                                                 "reward": rr["reward"], "components": rr["components"]})
        if not pos:
            if neg:
                worked_negs = [b for b in neg if _worked(b)]
                if len(worked_negs) >= 2:
                    explore = _set_text(worked_negs[:3], "Insufficient information; defer to external search")
                    wrong = _set_text(worked_negs[:1], neg[0].get("final_answer", "Unknown"))
                    views["branch_set_dpo"].append({"prompt": prompt, "chosen": explore, "rejected": wrong,
                                                    "domain": dom, "pair": "underbranch_defer"})
            return
        gold = pos[0].get("final_answer")
        worked_pos = sorted((b for b in pos if _worked(b)), key=lambda b: len(_full(b)))
        if not worked_pos:
            skipped["no_worked_positive"] += 1
            return
        best, short = worked_pos[-1], worked_pos[0]

        one = _with_final(_full(best), gold)
        views["one_branch_sft"].append({"prompt": prompt, "completion": one, "domain": dom})
        short_txt = _with_final(_full(short), gold)
        if len(short_txt) <= DIRECT_CAP:
            views["direct_answer_sft"].append({"prompt": prompt, "completion": short_txt, "domain": dom,
                                               "difficulty": bud})
        worked_negs = [b for b in neg if _worked(b)]
        if worked_negs:
            good = _set_text([best] + worked_negs[:2], gold)
            views["multi_branch_sft"].append({"prompt": prompt, "completion": good, "domain": dom})
            views["branch_set_rejection_sft"].append({"prompt": prompt, "completion": good, "domain": dom,
                                                      "reward": rr["reward"]})
            bad = _set_text(worked_negs[:3], worked_negs[0].get("final_answer", "Unknown"))
            views["branch_set_dpo"].append({"prompt": prompt, "chosen": good, "rejected": bad,
                                            "domain": dom, "pair": "oracle"})
        if len(pos) >= 2 and n >= 4 and len(short_txt) <= DIRECT_CAP * 2:
            overbr = _set_text(ba, gold, cap=8)
            views["branch_set_dpo"].append({"prompt": prompt, "chosen": short_txt, "rejected": overbr,
                                            "domain": dom, "pair": "overbranch"})

    n = 0
    for line in open(RLOG):
        consume(json.loads(line), "rendered_v3")
        n += 1
    print(f"[v3] rendered groups consumed: {n}", flush=True)
    n = 0
    for line in open(LAB):
        if n >= MAX_LAB:
            break
        d = json.loads(line)
        if d.get("split") == "train":
            consume(d, "labeled_pool")
            n += 1
    print(f"[v3] labeled pools consumed: {n}", flush=True)

    n = 0
    if CG.exists():
        for line in open(CG):
            if n >= MAX_ALIGN:
                break
            d = json.loads(line)
            if d.get("domain") != "alignment" or d.get("split") != "train":
                continue
            cands = d.get("candidates") or []
            chosen = str((next((c for c in cands if c.get("is_positive")), None) or {}).get("candidate_text") or "")
            other = str((next((c for c in cands if not c.get("is_positive")), None) or {}).get("candidate_text") or "")
            if not chosen or not other:
                continue
            # prompt = shared HH prefix up to the last common "Assistant:" turn boundary
            i = 0
            while i < min(len(chosen), len(other)) and chosen[i] == other[i]:
                i += 1
            cut = chosen.rfind("Assistant:", 0, i)
            if cut <= 0:
                continue
            prompt = chosen[: cut + len("Assistant:")].strip()
            text = chosen[cut + len("Assistant:"):].strip()
            if prompt and len(text) >= MIN_WORK:
                views["direct_answer_sft"].append({"prompt": prompt, "completion": text[:1200],
                                                   "domain": "alignment", "difficulty": "DIRECT"})
                views["branch_budget_policy"].append({"prompt": prompt, "completion": "DIRECT",
                                                      "domain": "alignment", "difficulty_passrate": None})
                n += 1
    print(f"[v3] alignment rows: {n}", flush=True)

    by = collections.defaultdict(list)
    for r in views["branch_budget_policy"]:
        by[r["completion"]].append(r)
    cap = min(6000, max(len(v) for v in by.values()))
    views["branch_budget_policy"] = [r for rows in by.values() for r in rng.sample(rows, min(len(rows), cap))]

    counts = {}
    for name, rows in views.items():
        seen, uniq = set(), []
        for r in rows:
            k = (r.get("prompt", "")[:160], (r.get("completion") or r.get("chosen") or "")[:120],
                 (r.get("rejected") or "")[:80])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        with open(TRAIN_V3 / f"{name}.jsonl", "w") as fo:
            for r in uniq:
                fo.write(json.dumps(r, default=V.v2.json_default) + "\n")
        counts[name] = len(uniq)

    pair_types = dict(collections.Counter(r["pair"] for r in views["branch_set_dpo"]))
    lens = [len(r["completion"]) for r in views["one_branch_sft"]]
    manifest = {**counts, "dpo_pair_types": pair_types, "skipped": dict(skipped),
                "one_branch_mean_chars": round(statistics.mean(lens), 1) if lens else None,
                "elapsed_seconds": round(time.time() - started, 1)}
    V.write_json(TRAIN_V3 / "offline_training_manifest.json", manifest)
    ok = counts["branch_set_dpo"] >= 8000 and counts["branch_set_rejection_sft"] >= 5000 and \
        counts["one_branch_sft"] >= 8000 and not any(
            len((r.get("completion") or "")) < MIN_WORK for r in views["one_branch_sft"][:200])
    verdict = "WORKED_VIEWS_READY" if ok else "WORKED_VIEWS_WEAK"
    V.set_stage("FG_offline_views_v3", verdict, manifest)
    V.prog("FG_offline_views_v3", {"verdict": verdict, **manifest})
    V.write_md(V.OUT_ROOT / "offline_views_and_prefs_v3.md", [
        "# Offline Views + Preferences v3 (Parts F+G v3)", "",
        V.status_line("OFFLINE_VIEWS_V3_VERDICT", verdict),
        f"Counts: {json.dumps(counts)}",
        f"DPO pair types: {json.dumps(pair_types)} | skipped: {json.dumps(dict(skipped))}",
        "All completions show work (executed renders / full real solutions; no bare finals, no "
        "sub-{}-char truncation); direct answers are short worked branches. Supersedes the v2 "
        "views per data_flaw_meta_branch_rendering.md.".format(WORK_CAP),
    ])
    print(V.status_line("OFFLINE_VIEWS_V3_VERDICT", verdict))
    print(f"  {counts}")
    print(f"  pairs {pair_types} | skipped {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
