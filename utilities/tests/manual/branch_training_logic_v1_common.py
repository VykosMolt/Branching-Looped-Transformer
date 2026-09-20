"""Logic generation + canonicalization for branch_training_logic_expansion_v1 (Part C).

Synthetic logic is generated from a symbolic source and verified before NL rendering; the
symbolic form is kept out of the model-visible prompt. Every item carries an external
verifier (truth-table / finite-model / forward-chaining / z3), never a tap/teacher label.

Categories:
  synthetic_propositional   - Horn facts+rules, classical entailment (entails/contradicts/unknown)
  proofwriter_deduction     - rule application with controlled proof depth (True/False/Unknown)
  synthetic_fol             - categorical syllogisms, finite-model-checked validity
  synthetic_constraint_game - z3-backed ordering puzzles ("which must be true")
  mcq_logical_reading       - canonicalized real MCQ logic (LogiQA etc.)
  fol_entailment            - canonicalized real FOL NL (FOLIO etc.)
"""
from __future__ import annotations
import hashlib
import itertools
import random
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENTAIL_OPTIONS = ["True", "False", "Unknown"]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi"]
PROPS = ["it is raining", "the alarm is on", "the door is locked", "the light is green",
         "the engine starts", "the battery is charged", "the gate is open", "the signal is sent",
         "the tank is full", "the valve is closed", "the fan is running", "the pump is active"]


def _sid(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")


# ===================================================== propositional / Horn entailment
def _eval_kb(assignment: dict[int, bool], facts, rules) -> bool:
    for v, pol in facts:
        if assignment[v] != pol:
            return False
    for body, (hv, hp) in rules:
        if all(assignment[v] == pol for v, pol in body) and assignment[hv] != hp:
            return False
    return True


def _classical_label(n_vars, facts, rules, query) -> str:
    qv, qp = query
    models = [a for a in (dict(zip(range(n_vars), bits))
                          for bits in itertools.product([False, True], repeat=n_vars))
              if _eval_kb(a, facts, rules)]
    if not models:
        return "Unknown"  # inconsistent KB -> avoid (filtered by caller)
    q_true = all(a[qv] == qp for a in models)
    q_false = all(a[qv] != qp for a in models)
    return "True" if q_true else ("False" if q_false else "Unknown")


def gen_propositional(seed: int, min_depth: int = 1) -> dict[str, Any] | None:
    rng = random.Random(seed)
    n = rng.randint(3, 5)
    props = rng.sample(PROPS, n)
    facts = []
    n_facts = rng.randint(1, 2)
    fact_vars = rng.sample(range(n), n_facts)
    for v in fact_vars:
        facts.append((v, rng.random() < 0.7))
    rules = []
    n_rules = rng.randint(2, 4)
    for _ in range(n_rules):
        body_size = rng.randint(1, 2)
        body_vars = rng.sample(range(n), body_size)
        body = [(v, rng.random() < 0.8) for v in body_vars]
        hv = rng.randrange(n)
        if hv in [v for v, _ in body]:
            continue
        rules.append((body, (hv, rng.random() < 0.85)))
    if not rules:
        return None
    qv = rng.randrange(n)
    label = _classical_label(n, facts, rules, (qv, True))
    # require a consistent KB
    models = [a for a in (dict(zip(range(n), bits)) for bits in itertools.product([False, True], repeat=n))
              if _eval_kb(a, facts, rules)]
    if not models:
        return None
    # NL render (symbolic hidden)
    def lit(v, pol):
        return f"{props[v]}" if pol else f"it is not the case that {props[v]}"
    fact_txt = "; ".join(lit(v, p) for v, p in facts)
    rule_txt = " ".join(f"If {' and '.join(lit(v, p) for v, p in body)}, then {lit(hv, hp)}."
                        for body, (hv, hp) in rules)
    prompt = (f"Facts: {fact_txt}.\nRules: {rule_txt}\n"
              f"Question: Based only on the facts and rules, is the following statement True, False, or Unknown?\n"
              f"Statement: {props[qv]}.")
    return {"category": "synthetic_propositional", "label_type": "deterministic_rubric", "prompt": prompt,
            "options": ENTAIL_OPTIONS, "answer_key": ENTAIL_OPTIONS.index(label), "gold_answer": label,
            "verifier": {"type": "truth_table", "n_vars": n}, "proof_depth": len(rules),
            "symbolic": {"facts": facts, "rules": rules, "query": [qv, True]}}


def gen_proof_depth(seed: int, depth: int) -> dict[str, Any] | None:
    """Forward chain of `depth` implications so the query needs `depth` rule applications."""
    rng = random.Random(seed)
    n = depth + rng.randint(1, 2) + 1
    n = min(n, 6)
    props = rng.sample(PROPS, n)
    chain = list(range(depth + 1))
    rng.shuffle(chain)
    facts = [(chain[0], True)]
    rules = [([(chain[i], True)], (chain[i + 1], True)) for i in range(depth)]
    # add a couple of distractor rules
    for _ in range(rng.randint(1, 2)):
        a, b = rng.sample(range(n), 2)
        rules.append(([(a, rng.random() < 0.5)], (b, rng.random() < 0.5)))
    query_target = chain[depth]
    label = _classical_label(n, facts, rules, (query_target, True))
    if label != "True":  # ensure the chain actually entails it (distractors may break it -> regenerate)
        return None

    def lit(v, pol):
        return f"{props[v]}" if pol else f"it is not the case that {props[v]}"
    fact_txt = "; ".join(lit(v, p) for v, p in facts)
    rng.shuffle(rules)
    rule_txt = " ".join(f"If {' and '.join(lit(v, p) for v, p in body)}, then {lit(hv, hp)}."
                        for body, (hv, hp) in rules)
    prompt = (f"Facts: {fact_txt}.\nRules: {rule_txt}\n"
              f"Question: Is the following statement True, False, or Unknown given the facts and rules?\n"
              f"Statement: {props[query_target]}.")
    return {"category": "proofwriter_deduction", "label_type": "deterministic_rubric", "prompt": prompt,
            "options": ENTAIL_OPTIONS, "answer_key": 0, "gold_answer": "True",
            "verifier": {"type": "forward_chaining", "depth": depth}, "proof_depth": depth,
            "symbolic": {"facts": facts, "rules": rules, "query": [query_target, True]}}


# ===================================================== categorical syllogisms (finite-model)
QUANT = ["All", "No", "Some", "Some-not"]


def _holds(quant, A, B, model):
    if quant == "All":
        return all((not A[x]) or B[x] for x in model)
    if quant == "No":
        return all(not (A[x] and B[x]) for x in model)
    if quant == "Some":
        return any(A[x] and B[x] for x in model)
    if quant == "Some-not":
        return any(A[x] and (not B[x]) for x in model)
    return False


def _syllogism_valid(p1, p2, concl, max_dom=4) -> bool:
    # predicates S,M,P over a finite domain; premises entail conclusion?
    for k in range(1, max_dom + 1):
        dom = list(range(k))
        for bits in itertools.product([False, True], repeat=3 * k):
            S = {x: bits[x] for x in dom}
            M = {x: bits[k + x] for x in dom}
            P = {x: bits[2 * k + x] for x in dom}
            preds = {"S": S, "M": M, "P": P}
            if _holds(p1[0], preds[p1[1]], preds[p1[2]], dom) and _holds(p2[0], preds[p2[1]], preds[p2[2]], dom):
                if not _holds(concl[0], preds[concl[1]], preds[concl[2]], dom):
                    return False
    return True


def gen_syllogism(seed: int) -> dict[str, Any] | None:
    rng = random.Random(seed)
    terms = rng.sample(["mammals", "dogs", "pets", "reptiles", "birds", "fish", "insects", "predators",
                        "herbivores", "vertebrates"], 3)
    S, M, P = "S", "M", "P"
    p1 = (rng.choice(QUANT), M, P)
    p2 = (rng.choice(QUANT), S, M)
    concl = (rng.choice(QUANT), S, P)
    valid = _syllogism_valid(p1, p2, concl)

    def render(q, a, b):
        amap = {S: terms[0], M: terms[1], P: terms[2]}
        x, y = amap[a], amap[b]
        return {"All": f"All {x} are {y}", "No": f"No {x} are {y}",
                "Some": f"Some {x} are {y}", "Some-not": f"Some {x} are not {y}"}[q]
    prompt = (f"Premise 1: {render(*p1)}.\nPremise 2: {render(*p2)}.\n"
              f"Question: Does the conclusion '{render(*concl)}' validly follow from the premises? "
              f"Answer Valid or Invalid.")
    label = "Valid" if valid else "Invalid"
    return {"category": "synthetic_fol", "label_type": "deterministic_rubric", "prompt": prompt,
            "options": ["Valid", "Invalid"], "answer_key": 0 if valid else 1, "gold_answer": label,
            "verifier": {"type": "finite_model_checking", "max_dom": 4},
            "symbolic": {"p1": p1, "p2": p2, "concl": concl, "terms": terms}}


# ===================================================== z3 ordering puzzles
def gen_constraint_game(seed: int) -> dict[str, Any] | None:
    try:
        import z3
    except Exception:
        return None
    rng = random.Random(seed)
    n = rng.randint(4, 5)
    people = NAMES[:n]
    pos = {p: z3.Int(f"pos_{p}") for p in people}
    base = [z3.And(pos[p] >= 1, pos[p] <= n) for p in people] + [z3.Distinct(*pos.values())]
    constraints = []
    texts = []
    for _ in range(rng.randint(3, 4)):
        kind = rng.choice(["before", "adjacent", "fixed", "not_fixed"])
        if kind == "before":
            a, b = rng.sample(people, 2)
            constraints.append(pos[a] < pos[b]); texts.append(f"{a} stands somewhere before {b}.")
        elif kind == "adjacent":
            a, b = rng.sample(people, 2)
            constraints.append(z3.Or(pos[a] - pos[b] == 1, pos[b] - pos[a] == 1))
            texts.append(f"{a} and {b} stand next to each other.")
        elif kind == "fixed":
            a = rng.choice(people); k = rng.randint(1, n)
            constraints.append(pos[a] == k); texts.append(f"{a} stands in position {k}.")
        else:
            a = rng.choice(people); k = rng.randint(1, n)
            constraints.append(pos[a] != k); texts.append(f"{a} does not stand in position {k}.")
    s = z3.Solver(); s.add(base + constraints)
    if s.check() != z3.sat:
        return None
    # find a fact that MUST be true (entailed): pos[x]==k for some x,k true in all models
    cand = []
    for p in people:
        for k in range(1, n + 1):
            s2 = z3.Solver(); s2.add(base + constraints + [pos[p] != k])
            if s2.check() == z3.unsat:
                cand.append((p, k))
    if not cand:
        return None
    p_true, k_true = rng.choice(cand)
    correct = f"{p_true} stands in position {k_true}."
    # distractors: statements NOT entailed (some model violates them)
    distractors = []
    tries = 0
    while len(distractors) < 3 and tries < 60:
        tries += 1
        p = rng.choice(people); k = rng.randint(1, n)
        if (p, k) in cand:
            continue
        stmt = f"{p} stands in position {k}."
        if stmt != correct and stmt not in distractors:
            distractors.append(stmt)
    if len(distractors) < 3:
        return None
    opts = [correct] + distractors
    order = list(range(4)); rng.shuffle(order)
    options = [opts[i] for i in order]
    answer_key = order.index(0)
    prompt = (f"{n} people ({', '.join(people)}) stand in a line, positions 1 (front) to {n} (back).\n"
              f"Constraints: {' '.join(texts)}\n"
              f"Question: Which of the following MUST be true?")
    return {"category": "synthetic_constraint_game", "label_type": "constraint_solver", "prompt": prompt,
            "options": options, "answer_key": answer_key, "gold_answer": options[answer_key],
            "verifier": {"type": "z3", "n": n}, "symbolic": {"constraints": texts, "people": people}}


# ===================================================== batch generation
GENERATORS = {
    "synthetic_propositional": (gen_propositional, {}),
    "proofwriter_deduction": (gen_proof_depth, {"depth_range": (2, 5)}),
    "synthetic_fol": (gen_syllogism, {}),
    "synthetic_constraint_game": (gen_constraint_game, {}),
}


def generate_batch(category: str, n: int, seed0: int = 0) -> list[dict[str, Any]]:
    out = []
    fn, cfg = GENERATORS[category]
    i = seed0
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        if category == "proofwriter_deduction":
            d = (i % (cfg["depth_range"][1] - cfg["depth_range"][0] + 1)) + cfg["depth_range"][0]
            rec = fn(_sid(category, i), d)
        else:
            rec = fn(_sid(category, i))
        i += 1
        if rec is None:
            continue
        rec["task_uid"] = f"synthlogic::{category}::{i}"
        rec["dataset"] = f"synthetic_{category}"
        out.append(rec)
    return out
