"""PART H core — deterministic offline generator reward over EXTERNAL verifier labels.

This is a reward *function*, not a learned reward model. It scores a whole branch SET (not branch
text) from fields already labeled by external verifiers (external_label / final_answer / parse_ok /
strategy_label / failure_modes). Design rules:
  - diversity counts ONLY when tied to verifier-positive (or structurally useful) branches;
  - correctness dominates; Branch 1/2/3 formatting alone earns nothing;
  - budget-awareness penalizes overbranching easy/all-correct and underbranching hard/all-wrong.

Auditable: returns per-component contributions so Part H can dump and inspect them.
"""
from __future__ import annotations
from typing import Any

POS_CAP = 4  # cap reward for piling on verifier-positive branches

# top-level component weights (correctness-dominant)
W = {
    "positive_oracle_present": 2.0,
    "n_verifier_positive": 0.25,
    "verifier_positive_diversity": 0.4,
    "appropriate_branch_budget": 0.5,
    "parse_ok": 0.5,
    "final_answer_correct": 1.5,
    "domain_validity": 1.0,
    "duplicate_branch_penalty": -0.6,
    "superficial_diversity_penalty": -0.6,
    "overbranching_penalty": -0.8,
    "underbranching_penalty": -0.6,
    "unsupported_premise_penalty": -0.5,
    "invalid_final_format_penalty": -0.4,
    "ambiguity_penalty": -0.3,
}

# logic failure modes (from rendered/labeled branches) that are explicitly penalized
_LOGIC_BAD_FM = {"quantifier_reversal", "negation_flip", "premise_hallucination",
                 "invalid_contradiction", "wrong_elimination", "option_letter_mismatch"}


def _branches(group):
    return group.get("branch_attempts") or group.get("branches") or []


def _is_pass(b):
    return (b.get("external_label") == "pass") or (b.get("label") == "pass") or (float(b.get("objective_reward", 0) or 0) > 0)


def _final(b):
    return str(b.get("final_answer") if b.get("final_answer") is not None else b.get("final", ""))


def _parse_ok(b):
    v = b.get("parse_ok")
    return 1.0 if v is True else (0.0 if v is False else (1.0 if b.get("final_answer") or b.get("final") else 0.0))


def _domain_validity(group, branches, pos):
    """+ valid structure, - domain-specific invalidities; in [-1, 1]."""
    dom = group.get("domain", "logic")
    if dom == "logic":
        good = sum(1 for b in pos if b.get("strategy_label") in
                   ("forward_chaining", "option_elimination", "finite_model_check", "constraint_table", "counterexample"))
        bad = sum(1 for b in branches for fm in (b.get("failure_modes") or []) if fm in _LOGIC_BAD_FM)
        return max(-1.0, min(1.0, (good - 0.5 * bad) / max(1, len(branches))))
    if dom == "coding":
        # parse/syntax already in external_label (unit tests executed); reward fn-name preserved if marked
        return 1.0 if pos else -0.3
    if dom == "math":
        return 1.0 if pos else -0.2
    if dom == "reasoning":
        return 1.0 if pos else -0.2
    if dom == "alignment":
        return 1.0 if pos else 0.0
    return 0.0


def _budget_fit(group, branches, pos):
    """appropriate_branch_budget in [0,1] minus over/under penalties (returned separately)."""
    n = len(branches)
    all_correct = len(pos) == n and n > 0
    all_wrong = len(pos) == 0
    over = 1.0 if (all_correct and n >= 4) else 0.0           # piled branches on an easy task
    under = 1.0 if (all_wrong and n <= 1) else 0.0             # gave up with a single failing attempt
    # "appropriate": few branches when easy, several when hard, at least one positive when solvable
    if all_correct and n <= 2:
        fit = 1.0
    elif (not all_correct and not all_wrong) and n >= 2:
        fit = 1.0
    elif all_wrong and n >= 4:
        fit = 0.7  # explored hard task adequately even if unsolved
    else:
        fit = 0.4
    return fit, over, under


def group_reward(group: dict[str, Any]) -> dict[str, Any]:
    branches = _branches(group)
    if not branches:
        return {"reward": 0.0, "components": {}, "n_branches": 0}
    pos = [b for b in branches if _is_pass(b)]
    finals = [_final(b) for b in branches if _final(b)]
    pos_finals = [_final(b) for b in pos if _final(b)]
    distinct = len(set(finals))
    pos_distinct = len(set(pos_finals))
    parse_rate = sum(_parse_ok(b) for b in branches) / len(branches)
    dup = (len(finals) - distinct) / max(1, len(finals)) if finals else 0.0
    superficial = (distinct - pos_distinct) / max(1, len(finals)) if (finals and not pos) else 0.0
    selected = group.get("selected_final")
    if selected is None and pos:
        selected = pos_finals[0] if pos_finals else None
    # majority-of-positive correctness of the selected final
    final_correct = 1.0 if (selected is not None and pos and str(selected) in set(pos_finals)) else 0.0
    ambiguous = 1.0 if (parse_rate < 0.5) else 0.0
    bad_format = sum(1 for b in branches if not _parse_ok(b)) / len(branches)

    fit, over, under = _budget_fit(group, branches, pos)
    dom_val = _domain_validity(group, branches, pos)
    unsupported = sum(1 for b in branches for fm in (b.get("failure_modes") or [])
                      if fm == "premise_hallucination") / len(branches)

    comp = {
        "positive_oracle_present": 1.0 if pos else 0.0,
        "n_verifier_positive": min(len(pos), POS_CAP) / POS_CAP,
        "verifier_positive_diversity": min(pos_distinct, 3) / 3.0,  # useful diversity only
        "appropriate_branch_budget": fit,
        "parse_ok": parse_rate,
        "final_answer_correct": final_correct,
        "domain_validity": dom_val,
        "duplicate_branch_penalty": dup,
        "superficial_diversity_penalty": superficial,
        "overbranching_penalty": over,
        "underbranching_penalty": under,
        "unsupported_premise_penalty": unsupported,
        "invalid_final_format_penalty": bad_format,
        "ambiguity_penalty": ambiguous,
    }
    contributions = {k: round(W[k] * v, 4) for k, v in comp.items()}
    reward = round(sum(contributions.values()), 4)
    return {"reward": reward, "components": comp, "contributions": contributions, "n_branches": len(branches),
            "n_positive": len(pos), "pos_distinct": pos_distinct, "parse_rate": round(parse_rate, 3)}


def weights() -> dict:
    return dict(W)
