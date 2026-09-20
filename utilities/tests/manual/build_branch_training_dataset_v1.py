"""PART K — branch-training dataset construction (5 views).

From verifier-labeled GENERATED pools (+ DualAnchor teacher traces + v2 candidate texts):
  1 branch_format_sft      : prompt -> [Branch i]... [Final] (verifier-positive + diverse)
  2 branch_diversity_sft   : prompt -> structurally distinct attempts
  3 branch_policy_distillation : prompt+candidates -> DualAnchor keep/prune labels (POLICY, not correctness)
  4 final_self_selection   : prompt+branches -> the externally-correct branch
  5 verifier_reward_rl      : group reward records + DPO pairs (external reward)
Correctness from external verifiers only; teacher labels are policy/soft.
"""
from __future__ import annotations
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402

TRAIN = B.TRAIN
OUT = B.OUT_ROOT


def _fmt_branches(branches, limit=3):
    out = []
    for i, b in enumerate(branches[:limit]):
        out.append(f"[Branch {i+1}] ({b.get('strategy_label', 'attempt')})\n{b['branch_text'].strip()}")
    return "\n\n".join(out)


def main() -> int:
    started = time.time(); B.ensure_dirs()
    gen = [g for g in B.load_generated_groups() if g["split"] == "train"]
    counts = {}

    # 1 + 2 + 4 from generated pools (need branch text + a positive)
    fmt_rows, div_rows, sel_rows, rl_rows, dpo_rows = [], [], [], [], []
    for g in gen:
        br = g["branch_attempts"]
        prompt = g.get("task_prompt", "")
        if not prompt:
            continue
        pos = [b for b in br if b["objective_reward"] > 0]
        neg = [b for b in br if b["objective_reward"] <= 0]
        gold_final = pos[0]["final_answer"] if pos else None
        # 1 branch_format_sft: needs >=1 positive; show up to 3 branches (prefer including a positive) + final
        if pos:
            shown = (pos[:1] + neg[:2]) if neg else pos[:3]
            target = _fmt_branches(shown) + f"\n\n[Final]\n{gold_final}"
            fmt_rows.append({"prompt": prompt, "completion": target, "domain": g["domain"],
                             "text": f"{prompt}\n\n{target}"})
        # 2 branch_diversity_sft: >=2 distinct strategies
        bystrat = {}
        for b in br:
            bystrat.setdefault(b.get("strategy_label", "x"), b)
        if len(bystrat) >= 2:
            shown = list(bystrat.values())[:4]
            fin = gold_final if gold_final else (shown[0]["final_answer"])
            target = _fmt_branches(shown, limit=4) + f"\n\n[Final]\n{fin}"
            div_rows.append({"prompt": prompt, "completion": target, "domain": g["domain"],
                             "n_strategies": len(bystrat), "text": f"{prompt}\n\n{target}"})
        # 4 final_self_selection: prompt + numbered branches -> correct branch index
        if pos and len(br) >= 2:
            numbered = "\n\n".join(f"[Branch {i+1}]\n{b['branch_text'].strip()[:600]}" for i, b in enumerate(br))
            correct_idx = next(i for i, b in enumerate(br) if b["objective_reward"] > 0) + 1
            sel_prompt = f"{prompt}\n\nCandidate attempts:\n{numbered}\n\nWhich branch is correct?"
            sel_rows.append({"prompt": sel_prompt, "completion": f"Branch {correct_idx}", "domain": g["domain"],
                             "text": f"{sel_prompt}\nBranch {correct_idx}"})
        # 5 verifier_reward_rl: group reward record + DPO pairs (external reward only)
        rl_rows.append({"prompt": prompt, "domain": g["domain"],
                        "branch_rewards": [b["objective_reward"] for b in br],
                        "positive_oracle_present": bool(pos), "n_strategies": len(bystrat),
                        "parse_ok_rate": round(sum(1 for b in br if b["parse_ok"]) / len(br), 3),
                        "group_reward": round((1.0 if pos else 0.0) + 0.1 * min(len(bystrat), 4)
                                              + 0.1 * (sum(1 for b in br if b["parse_ok"]) / len(br)), 3)})
        for p in pos[:2]:
            for nb in neg[:2]:
                dpo_rows.append({"prompt": prompt, "chosen": p["branch_text"][:1000],
                                 "rejected": nb["branch_text"][:1000], "domain": g["domain"]})

    # 3 branch_policy_distillation from teacher traces joined to v2 candidate texts
    traces = B.read_jsonl(B.TEACHER / "dualanchor_teacher_traces.jsonl")
    v2_text = {}
    for grp in B.read_jsonl(B.PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl"):
        v2_text[grp["group_uid"]] = {c["candidate_uid"]: c.get("candidate_text", "") for c in grp["candidates"]}
    pol_rows = []
    for t in traces:
        if t.get("split") != "train":
            continue
        texts = v2_text.get(t["group_id"])
        if not texts:
            continue
        labels = t["teacher_policy_labels"]
        lines = []
        for lab in labels:
            txt = texts.get(lab["candidate_id"], "")
            if not txt:
                continue
            decision = "KEEP" if lab["keep_label"] else "PRUNE"
            if lab["rescue_label"]:
                decision = "RESCUE"
            lines.append(f"- {decision}: {txt.strip()[:200]}")
        if len(lines) >= 2:
            pol_rows.append({"domain": t["domain"], "teacher": "DualAnchor", "teacher_is_ground_truth": False,
                             "defer": t["stage_traces"][0]["defer"],
                             "prompt": "Manage these candidate branches (keep/prune/rescue):\n" + "\n".join(lines),
                             "completion": ("DEFER" if t["stage_traces"][0]["defer"] else "HANDOFF") +
                             f" survivors={t['stage_traces'][0]['survivor_ids']}"})

    B.write_jsonl(TRAIN / "branch_format_sft.jsonl", fmt_rows)
    B.write_jsonl(TRAIN / "branch_diversity_sft.jsonl", div_rows)
    B.write_jsonl(TRAIN / "branch_policy_distillation.jsonl", pol_rows)
    B.write_jsonl(TRAIN / "final_self_selection.jsonl", sel_rows)
    B.write_jsonl(TRAIN / "verifier_reward_rl.jsonl", rl_rows)
    B.write_jsonl(TRAIN / "branch_preference_dpo.jsonl", dpo_rows)
    counts = {"branch_format_sft": len(fmt_rows), "branch_diversity_sft": len(div_rows),
              "branch_policy_distillation": len(pol_rows), "final_self_selection": len(sel_rows),
              "verifier_reward_rl": len(rl_rows), "branch_preference_dpo": len(dpo_rows)}
    dom_fmt = Counter(r["domain"] for r in fmt_rows)
    card = ["# Branch Training Dataset Card", "",
            "Source: verifier-labeled model-generated branch pools (external labels only) + DualAnchor teacher "
            "traces (policy labels, NOT correctness) + v2 candidate texts.", "",
            "## Views", "", *[f"- `{k}.jsonl`: {v} rows" for k, v in counts.items()], "",
            f"branch_format_sft by domain: {dict(dom_fmt)}", "",
            "All correctness labels are external (answer keys / parsers / z3 / executed unit tests). "
            "DualAnchor/CoreContent are policy/soft teachers. Train-split only; heldout never used for training."]
    B.write_md(TRAIN / "dataset_card.md", card)
    sft_total = len(fmt_rows) + len(div_rows) + len(sel_rows)
    if sft_total >= 200 and pol_rows and dpo_rows:
        verdict = "TRAINING_DATA_READY"
    elif sft_total >= 50:
        verdict = "LOGIC_BRANCH_TRAINING_READY" if dom_fmt.get("logic", 0) else "FINAL_SELECTION_READY"
    elif sft_total > 0:
        verdict = "DIVERSITY_DATA_WEAK"
    else:
        verdict = "BLOCKED"
    payload = {"BRANCH_TRAINING_DATASET_VERDICT": verdict, "counts": counts,
               "branch_format_by_domain": dict(dom_fmt), "n_generated_train_groups": len(gen),
               "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(OUT / "branch_training_dataset.json", payload)
    B.write_md(OUT / "branch_training_dataset.md", [
        "# Branch Training Dataset (Part K)", "", B.status_line("BRANCH_TRAINING_DATASET_VERDICT", verdict), "",
        f"Built from {len(gen)} generated train groups. Views: {counts}.", "",
        f"branch_format_sft by domain: {dict(dom_fmt)}.", "",
        "External-label-only correctness; teacher labels are policy. Re-runnable as generation grows.",
    ])
    B.prog("K_branch_training_dataset", {"verdict": verdict, "counts": counts})
    print(B.status_line("BRANCH_TRAINING_DATASET_VERDICT", verdict))
    print(f"  counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
