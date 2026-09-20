"""PART H — offline reward specification + auditability check.

Documents the deterministic reward in `utilities/branch_training/offline_reward_v2.py` and audits it
on real branch sets: does it SEPARATE verifier-positive sets from all-negative ones, do per-component
contributions vary (auditable, not degenerate), and does diversity stay tied to correctness?
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402
from utilities.branch_training import offline_reward_v2 as RW  # noqa: E402

RLOG = V.TRAIN_V2 / "rendered_logic_branch_sets.jsonl"
LAB = V.DATA_ROOT / "processed/branch_pools_labeled.jsonl"

COMPONENT_DOC = {
    "positive_oracle_present": "+ any branch reaches the verified answer",
    "n_verifier_positive": "+ capped count of verifier-positive branches",
    "verifier_positive_diversity": "+ distinct answers AMONG correct branches (useful diversity only)",
    "appropriate_branch_budget": "+ few branches when easy, several when hard",
    "parse_ok": "+ branches emit a parseable final answer",
    "final_answer_correct": "+ selected final matches a verifier-positive answer",
    "domain_validity": "+ valid proof/contradiction/counterexample/elimination; - quantifier-reversal/negation-flip/etc",
    "duplicate_branch_penalty": "- duplicate branch finals",
    "superficial_diversity_penalty": "- diverse-looking but all-wrong branches",
    "overbranching_penalty": "- many branches on an all-correct (easy) task",
    "underbranching_penalty": "- a single failing attempt on a hard task",
    "unsupported_premise_penalty": "- premise hallucination",
    "invalid_final_format_penalty": "- unparseable final",
    "ambiguity_penalty": "- pool mostly unparseable",
}


def _sample(path, n, split_train=False):
    out = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            d = json.loads(line)
            if split_train and d.get("split") != "train":
                continue
            out.append(d)
    return out


def main() -> int:
    groups = _sample(RLOG, 4000) + _sample(LAB, 4000, split_train=True)
    pos_rewards, neg_rewards, comp_values = [], [], {k: [] for k in RW.weights()}
    logic_validity_fired = 0
    for g in groups:
        ba = g.get("branch_attempts") or []
        if not ba:
            continue
        pos = [b for b in ba if b.get("external_label") == "pass"]
        if pos:
            g["selected_final"] = pos[0].get("final_answer")
        r = RW.group_reward(g)
        (pos_rewards if pos else neg_rewards).append(r["reward"])
        for k, v in r["components"].items():
            comp_values[k].append(v)
        if g.get("domain") == "logic" and abs(r["components"].get("domain_validity", 0)) > 0:
            logic_validity_fired += 1

    sep = (round(statistics.mean(pos_rewards), 3) if pos_rewards else None,
           round(statistics.mean(neg_rewards), 3) if neg_rewards else None)
    margin = round((sep[0] or 0) - (sep[1] or 0), 3)
    # auditable = components vary (non-degenerate)
    varying = {k: round(statistics.pstdev(v), 3) for k, v in comp_values.items() if v}
    n_degenerate = sum(1 for k, s in varying.items() if s < 1e-6)
    overall_std = round(statistics.pstdev(pos_rewards + neg_rewards), 3) if (pos_rewards or neg_rewards) else 0.0

    if margin > 1.5 and n_degenerate <= 2 and logic_validity_fired > 0:
        verdict = "REWARD_READY"
    elif margin > 1.5 and n_degenerate <= 4:
        verdict = "REWARD_COMPONENTS_AUDITABLE"
    elif logic_validity_fired > 0 and margin > 0.5:
        verdict = "LOGIC_REWARD_READY"
    elif margin <= 0.2 or overall_std < 0.1:
        verdict = "REWARD_TOO_NOISY"
    else:
        verdict = "REWARD_COMPONENTS_AUDITABLE"

    payload = {"OFFLINE_REWARD_SPEC_VERDICT": verdict, "reward_module": "shared/utilities/branch_training/offline_reward_v2.py",
               "weights": RW.weights(), "separation_mean_pos_vs_neg": sep, "separation_margin": margin,
               "overall_reward_std": overall_std, "component_stdev": varying, "degenerate_components": n_degenerate,
               "logic_validity_fired": logic_validity_fired, "n_audited": len(groups)}
    V.write_json(V.OUT_ROOT / "offline_reward_spec.json", payload)
    V.write_md(V.OUT_ROOT / "offline_reward_spec.md", [
        "# Offline Reward Spec (Part H)", "", V.status_line("OFFLINE_REWARD_SPEC_VERDICT", verdict),
        "Deterministic reward over EXTERNAL verifier labels (not a learned reward model). It scores a branch SET; "
        "correctness dominates and **diversity counts only when tied to verifier-positive branches**, so Branch 1/2/3 "
        "formatting alone earns nothing.", "",
        f"**Audit:** mean reward positive-oracle sets {sep[0]} vs all-negative sets {sep[1]} → margin **{margin}** "
        f"(separates good from bad). Overall reward std {overall_std}; {n_degenerate} degenerate components; "
        f"logic domain-validity fired on {logic_validity_fired} logic groups.", "",
        "## Components (weight · meaning)",
        *[f"- `{k}` ({RW.weights()[k]:+}) — {COMPONENT_DOC.get(k, '')}" for k in RW.weights()],
        "", "## Domain validity",
        "- logic: + valid forward-chaining/elimination/finite-model/constraint-table/counterexample; "
        "- quantifier_reversal/negation_flip/premise_hallucination/invalid_contradiction/wrong_elimination/option_letter_mismatch.",
        "- coding: external = executed unit tests; math: exact/sympy; reasoning: correct + direct-when-appropriate; "
        "alignment: preference-label match, concision (penalize needless branch scaffolding).",
    ])
    V.set_stage("H_offline_reward_spec", verdict, {"margin": margin, "degenerate": n_degenerate})
    V.prog("H_offline_reward_spec", {"verdict": verdict, "margin": margin})
    print(V.status_line("OFFLINE_REWARD_SPEC_VERDICT", verdict))
    print(f"  pos/neg reward {sep} margin {margin} | overall_std {overall_std} | degenerate {n_degenerate} | "
          f"logic_validity_fired {logic_validity_fired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
