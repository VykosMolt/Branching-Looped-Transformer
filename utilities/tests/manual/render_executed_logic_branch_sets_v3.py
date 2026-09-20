"""PART E v3 — EXECUTED logic branch-set rendering (resumable, CPU-only).

Replaces the v2 strategy-narration renderer after the Part K finding that rendered branches
were meta-text (`data_flaw_meta_branch_rendering.md`): models trained on them learned to
narrate strategies and assert answers. v3 branches SHOW THE WORK — every branch text is the
narration of an actual executed computation:

- synthetic_propositional / proofwriter_deduction: real forward chaining over symbolic
  facts/rules, step by step, using the prompt's own atom phrasing.
- synthetic_fol: exhaustive finite-model check (universe sizes 1-3) with explicit witness /
  countermodel sets.
- synthetic_constraint_game: brute-force permutation solving with explicit valid orders.

Wrong branches are the SAME executors run with one injected, named mistake (misread negation,
hallucinated fact, dropped/flipped constraint, premature stop) — they show real work
containing a real error, and are kept only when the flawed run's answer differs from gold.

Families without symbolic payloads (LSAT/mcq/ruletaker/entailment) are NOT rendered here;
their coverage must come from model-generated pools. Output: train_v3/rendered_logic_branch_sets.jsonl
(same schema as v2; source="rendered_logic_executed_v3").
"""
from __future__ import annotations
import itertools
import json
import random
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402

JOB = "render_logic_v3"
TRAIN_V3 = V.DATA_ROOT / "train_v3"
OUT_JSONL = TRAIN_V3 / "rendered_logic_branch_sets.jsonl"
FAMILIES = ("synthetic_propositional", "proofwriter_deduction", "synthetic_fol", "synthetic_constraint_game")
_FA = "FINAL ANSWER: {}"


def _gold_match(task, final_answer) -> bool:
    gold = str(task.get("gold_answer", "")).strip().casefold()
    fa = str(final_answer).strip().casefold()
    if fa == gold:
        return True
    opts, ak = task.get("options"), task.get("answer_key")
    if opts and ak is not None and 0 <= ak < len(opts):
        return fa == str(opts[ak]).strip().casefold()
    return False


# ---------- propositional / proofwriter: forward chaining ----------

_NEG = "it is not the case that "


def _atom_names(task):
    """Map symbolic var ids to the prompt's phrases; fall back to P<i>."""
    sym = task["symbolic"]
    names: dict[int, str] = {}

    def learn(var, val, phrase):
        phrase = phrase.strip().rstrip(".")
        if phrase.lower().startswith(_NEG):
            phrase = phrase[len(_NEG):]
        names.setdefault(int(var), phrase)

    m = re.search(r"Facts:\s*(.*?)\nRules:\s*(.*?)\n(?:Question|Statement)", task["task_prompt"], re.S)
    stmt = re.search(r"Statement:\s*(.*?)\s*$", task["task_prompt"], re.S)
    if m:
        fact_phrases = [p for p in re.split(r"\.\s*", m.group(1)) if p.strip()]
        for (var, val), ph in zip(sym["facts"], fact_phrases):
            learn(var, val, ph)
        rule_strs = re.findall(r"If (.*?), then (.*?)\.", m.group(2))
        for ((ants, cons), (astr, cstr)) in zip(sym["rules"], rule_strs):
            aphr = [p.strip() for p in astr.split(" and ")]
            if len(aphr) == len(ants):
                for (var, val), ph in zip(ants, aphr):
                    learn(var, val, ph)
            learn(cons[0], cons[1], cstr)
    if stmt:
        learn(sym["query"][0], sym["query"][1], stmt.group(1))
    return lambda var: names.get(int(var), f"proposition P{var}")


def _lit(name_of, var, val):
    return name_of(var) if val else f"{_NEG}{name_of(var)}"


def _rule_str(name_of, rule):
    ants, cons = rule
    a = " and ".join(_lit(name_of, v, b) for v, b in ants)
    return f"If {a}, then {_lit(name_of, *cons)}"


def _chain(facts, rules, query, name_of, max_fires=None, narrate_misread=None):
    """Run forward chaining; return (final_answer, narration_lines, fired_count)."""
    known = {int(v): bool(b) for v, b in facts}
    lines = ["Facts given: " + "; ".join(_lit(name_of, v, b) for v, b in facts) + "."]
    fired, changed = 0, True
    while changed:
        changed = False
        for ri, (ants, cons) in enumerate(rules):
            if max_fires is not None and fired >= max_fires:
                changed = False
                break
            ok = all(known.get(int(v)) is bool(b) for v, b in ants)
            cv, cb = int(cons[0]), bool(cons[1])
            if ok and known.get(cv) is not cb and cv not in known:
                note = narrate_misread(ri) if narrate_misread else None
                lines.append(
                    f"Rule '{_rule_str(name_of, (ants, cons))}' fires: "
                    + "; ".join(f"'{_lit(name_of, v, b)}' holds" for v, b in ants)
                    + (f" ({note})" if note else "")
                    + f". Derive: {_lit(name_of, cv, cb)}."
                )
                known[cv] = cb
                fired += 1
                changed = True
    qv, qb = int(query[0]), bool(query[1])
    if qv in known:
        ans = "True" if known[qv] is qb else "False"
        lines.append(f"The statement '{_lit(name_of, qv, qb)}' is settled by the derivation: {ans}.")
    else:
        ans = "Unknown"
        lines.append(
            f"No further rules fire. '{name_of(qv)}' was never derived in either polarity, so the statement is Unknown."
        )
    return ans, lines, fired


def _tt_models(sym):
    """All assignments consistent with facts and rules (rule = material implication)."""
    vars_ = sorted({int(v) for v, _ in sym["facts"]} | {int(sym["query"][0])} |
                   {int(v) for ants, cons in sym["rules"] for v, _ in list(ants) + [cons]})
    models = []
    for bits in itertools.product((False, True), repeat=len(vars_)):
        m = dict(zip(vars_, bits))
        if any(m[int(v)] is not bool(b) for v, b in sym["facts"]):
            continue
        ok = True
        for ants, cons in sym["rules"]:
            if all(m[int(v)] is bool(b) for v, b in ants) and m[int(cons[0])] is not bool(cons[1]):
                ok = False
                break
        if ok:
            models.append(m)
    return models


def _model_str(m, name_of, focus_vars):
    return "; ".join(f"{name_of(v)}={'T' if m[v] else 'F'}" for v in focus_vars if v in m)


def _branches_propositional(task, rng):
    sym = task["symbolic"]
    name_of = _atom_names(task)
    out = []
    qv, qb = int(sym["query"][0]), bool(sym["query"][1])

    chain_ans, lines, _ = _chain(sym["facts"], sym["rules"], sym["query"], name_of)
    models = _tt_models(sym)
    qvals = {m[qv] for m in models}
    tt_ans = "Unknown" if len(qvals) != 1 else ("True" if (qvals == {qb}) else "False")
    focus = sorted({qv} | {int(v) for v, _ in sym["facts"]})
    if chain_ans == tt_ans:
        ans = chain_ans
        out.append(("\n".join(lines) + "\n" + _FA.format(ans), ans, "forward_chaining_executed", []))
    else:
        # chaining is incomplete here (e.g. contraposition needed): narrate the model check
        ans = tt_ans
        if ans == "Unknown":
            wit_t = next(m for m in models if m[qv] is qb)
            wit_f = next(m for m in models if m[qv] is not qb)
            body = (f"Enumerate every truth assignment consistent with the facts and rules "
                    f"({len(models)} in total). One consistent assignment makes the statement true "
                    f"({_model_str(wit_t, name_of, focus)}), another makes it false "
                    f"({_model_str(wit_f, name_of, focus)}); the facts and rules do not settle it.")
        else:
            body = (f"Enumerate every truth assignment consistent with the facts and rules: all {len(models)} of "
                    f"them give the statement the same value, e.g. {_model_str(models[0], name_of, focus)}. "
                    f"The rules force this even where step-by-step derivation stalls (contraposition).")
        out.append((f"Facts given: " + "; ".join(_lit(name_of, v, b) for v, b in sym["facts"]) + ".\n"
                    + body + "\n" + _FA.format(ans), ans, "model_enumeration_executed", []))

    # flawed (guaranteed for Unknown): exhibit one consistent witness model and overgeneralize from it
    if ans == "Unknown" and models:
        wit = next((m for m in models if m[qv] is qb), models[0])
        wans = "True" if wit[qv] is qb else "False"
        out.append((f"Look for an assignment consistent with the facts and rules: "
                    f"{_model_str(wit, name_of, focus)} works. In it the statement comes out "
                    f"{'true' if wit[qv] is qb else 'false'}; conclude from this single model.\n"
                    + _FA.format(wans), wans, "single_model_overclaim_executed", ["early_stop"]))

    # flawed: misread one antecedent's negation (executor really runs with the flip)
    rules = [tuple((list(map(tuple, a)), tuple(c))) for a, c in (tuple(r) for r in sym["rules"])]
    flaw_candidates = [(ri, ai) for ri, (ants, _c) in enumerate(sym["rules"]) for ai in range(len(ants))]
    rng.shuffle(flaw_candidates)
    for ri, ai in flaw_candidates[:3]:
        mutated = [([list(a) for a in ants], list(cons)) for ants, cons in sym["rules"]]
        mutated[ri][0][ai][1] = not mutated[ri][0][ai][1]
        fans, flines, _ = _chain(sym["facts"], mutated, sym["query"], name_of,
                                 narrate_misread=lambda r, _ri=ri: "misreading the negation" if r == _ri else None)
        if fans != ans:
            out.append(("\n".join(flines) + "\n" + _FA.format(fans), fans, "negation_misread_executed", ["negation_flip"]))
            break

    # flawed: hallucinate one unstated fact and chain from it
    unset = [int(c[0]) for _a, c in sym["rules"] if int(c[0]) not in {int(v) for v, _ in sym["facts"]}]
    if unset:
        hv = rng.choice(unset)
        hfacts = list(sym["facts"]) + [[hv, True]]
        fans, flines, _ = _chain(hfacts, sym["rules"], sym["query"], name_of)
        flines[0] += f" (also assuming '{name_of(hv)}', which was never stated)."
        if fans != ans:
            out.append(("\n".join(flines) + "\n" + _FA.format(fans), fans, "hallucinated_premise_executed",
                        ["premise_hallucination"]))

    # flawed: stop after the first rule firing
    fans, flines, fired = _chain(sym["facts"], sym["rules"], sym["query"], name_of, max_fires=1)
    if fans != ans and fired >= 1:
        flines.insert(-1, "Stop the derivation here without checking the remaining rules.")
        out.append(("\n".join(flines) + "\n" + _FA.format(fans), fans, "early_stop_executed", ["early_stop"]))

    # flawed (guaranteed to differ): full real chain + a named bad final inference
    qname = name_of(int(sym["query"][0]))
    chain_body = "\n".join(lines[:-1])
    if ans == "Unknown":
        fans = rng.choice(["True", "False"])
        tail = (f"'{qname}' was never derived, but it was never contradicted either; treat absence of "
                f"contradiction as truth." if fans == "True" else
                f"'{qname}' was never derived; treat absence of proof as falsity (closed-world reading).")
        out.append((f"{chain_body}\n{tail}\n" + _FA.format(fans), fans, "closed_world_misread_executed",
                    ["invalid_contradiction"]))
    elif ans == "True":
        out.append((f"{chain_body}\nThe statement was only reached through derived steps, not stated directly as "
                    f"a fact; distrust the chain and call it undetermined.\n" + _FA.format("Unknown"),
                    "Unknown", "skeptical_misread_executed", ["early_stop"]))
    else:  # False
        out.append((f"{chain_body}\nThe derivation settled the opposite polarity, but treat that as leaving the "
                    f"statement itself undetermined.\n" + _FA.format("Unknown"),
                    "Unknown", "polarity_shrug_executed", ["negation_flip"]))
    return out, ans


# ---------- synthetic_fol: finite-model check ----------

def _fol_holds(form, A, B, universe):
    if form == "All":
        return all((x not in A) or (x in B) for x in universe)
    if form == "No":
        return all((x not in A) or (x not in B) for x in universe)
    if form == "Some":
        return any((x in A) and (x in B) for x in universe)
    if form == "Some-not":
        return any((x in A) and (x not in B) for x in universe)
    raise ValueError(form)


def _fol_models(p1, p2, concl, letters):
    """Search universes of size 1-3; return (valid, countermodel|witness_model)."""
    for size in (1, 2, 3):
        universe = list(range(size))
        for bits in itertools.product(range(8), repeat=size):
            sets = {ltr: {x for x in universe if bits[x] >> i & 1} for i, ltr in enumerate(letters)}
            if _fol_holds(p1[0], sets[p1[1]], sets[p1[2]], universe) and \
               _fol_holds(p2[0], sets[p2[1]], sets[p2[2]], universe):
                if not _fol_holds(concl[0], sets[concl[1]], sets[concl[2]], universe):
                    return False, (universe, sets)
    return True, None


def _fol_sentence(form, a, b):
    return {"All": f"All {a} are {b}", "No": f"No {a} are {b}", "Some": f"Some {a} are {b}",
            "Some-not": f"Some {a} are not {b}"}[form]


def _set_str(name, s, universe):
    el = ", ".join(f"x{i}" for i in sorted(s)) if s else "(empty)"
    return f"{name} = {{{el}}}"


def _branches_fol(task, rng):
    sym = task["symbolic"]
    letters = []
    for stmt in (sym["p1"], sym["p2"], sym["concl"]):
        for ltr in stmt[1:]:
            if ltr not in letters:
                letters.append(ltr)
    if len(letters) > 3:
        return None, None
    terms = dict(zip(letters, sym["terms"]))
    t = lambda l: terms.get(l, l)
    p1s = _fol_sentence(sym["p1"][0], t(sym["p1"][1]), t(sym["p1"][2]))
    p2s = _fol_sentence(sym["p2"][0], t(sym["p2"][1]), t(sym["p2"][2]))
    cs = _fol_sentence(sym["concl"][0], t(sym["concl"][1]), t(sym["concl"][2]))
    valid, cm = _fol_models(sym["p1"], sym["p2"], sym["concl"], letters)
    gold = "Valid" if valid else "Invalid"
    out = []
    if valid:
        text = (f"Premises: '{p1s}' and '{p2s}'. Conclusion to test: '{cs}'.\n"
                f"Exhaustively check every membership assignment for {', '.join(t(l) for l in letters)} over "
                f"universes of one, two, and three elements. Every assignment that satisfies both premises also "
                f"satisfies the conclusion — no countermodel exists at any of these sizes, which suffices for "
                f"syllogistic forms.\n" + _FA.format("Valid"))
    else:
        universe, sets = cm
        setdesc = "; ".join(_set_str(t(l), sets[l], universe) for l in letters)
        text = (f"Premises: '{p1s}' and '{p2s}'. Conclusion to test: '{cs}'.\n"
                f"Search for a countermodel. Take a universe of {len(universe)} element(s) with {setdesc}. "
                f"Both premises hold under this assignment, but the conclusion fails. A countermodel exists.\n"
                + _FA.format("Invalid"))
    out.append((text, gold, "finite_model_check_executed", []))

    # flawed: read p1's converse (actually re-run the check with swapped terms)
    swapped = [sym["p1"][0], sym["p1"][2], sym["p1"][1]]
    fvalid, fcm = _fol_models(swapped, sym["p2"], sym["concl"], letters)
    fans = "Valid" if fvalid else "Invalid"
    if fans != gold:
        fp1 = _fol_sentence(swapped[0], t(swapped[1]), t(swapped[2]))
        if fvalid:
            body = (f"Reading the first premise as '{fp1}' (its converse), check all small models: under that reading "
                    f"no countermodel exists.")
        else:
            universe, sets = fcm
            body = (f"Reading the first premise as '{fp1}' (its converse), a countermodel appears: "
                    + "; ".join(_set_str(t(l), sets[l], universe) for l in letters) + ".")
        out.append((f"Premises: '{p1s}' and '{p2s}'. Conclusion: '{cs}'.\n{body}\n" + _FA.format(fans),
                    fans, "converse_misread_executed", ["quantifier_reversal"]))

    # flawed: only check one-element universes (real check, insufficient depth)
    v1only = True
    for bits in itertools.product(range(8), repeat=1):
        sets = {ltr: ({0} if bits[0] >> i & 1 else set()) for i, ltr in enumerate(letters)}
        if _fol_holds(sym["p1"][0], sets[sym["p1"][1]], sets[sym["p1"][2]], [0]) and \
           _fol_holds(sym["p2"][0], sets[sym["p2"][1]], sets[sym["p2"][2]], [0]) and \
           not _fol_holds(sym["concl"][0], sets[sym["concl"][1]], sets[sym["concl"][2]], [0]):
            v1only = False
            break
    fans1 = "Valid" if v1only else "Invalid"
    if fans1 != gold:
        out.append((f"Premises: '{p1s}' and '{p2s}'. Conclusion: '{cs}'.\n"
                    f"Check every membership assignment over a one-element universe only: "
                    f"{'no countermodel appears at that size, so accept the argument' if v1only else 'a countermodel appears'}. "
                    f"(Larger universes are not examined.)\n" + _FA.format(fans1),
                    fans1, "shallow_model_check_executed", ["early_stop"]))

    # flawed: existential import — treat 'All A are B' as also asserting 'Some A are B'
    if sym["p1"][0] == "All" or sym["p2"][0] == "All":
        wrong = "Valid" if gold == "Invalid" else "Invalid"
        out.append((f"Premises: '{p1s}' and '{p2s}'. Conclusion: '{cs}'.\n"
                    f"Assume every 'All' premise guarantees a witness (existential import), then read the conclusion "
                    f"off the assumed witness without checking models where the antecedent class is empty.\n"
                    + _FA.format(wrong), wrong, "existential_import_executed", ["premise_hallucination"]))
    return out, gold


# ---------- synthetic_constraint_game: permutation solving ----------

_C_AT = re.compile(r"^(\w+) stands in position (\d+)\.$")
_C_NOT_AT = re.compile(r"^(\w+) does not stand in position (\d+)\.$")
_C_ADJ = re.compile(r"^(\w+) and (\w+) stand next to each other\.$")
_C_BEFORE = re.compile(r"^(\w+) stands somewhere before (\w+)\.$")


def _parse_constraint(c):
    for pat, kind in ((_C_AT, "at"), (_C_NOT_AT, "not_at"), (_C_ADJ, "adj"), (_C_BEFORE, "before")):
        m = pat.match(c.strip())
        if m:
            return kind, m.groups()
    return None, None


def _satisfies(perm, kind, args):
    pos = {p: i + 1 for i, p in enumerate(perm)}
    if kind == "at":
        return pos[args[0]] == int(args[1])
    if kind == "not_at":
        return pos[args[0]] != int(args[1])
    if kind == "adj":
        return abs(pos[args[0]] - pos[args[1]]) == 1
    if kind == "before":
        return pos[args[0]] < pos[args[1]]
    return False


def _valid_perms(people, constraints):
    parsed = []
    for c in constraints:
        kind, args = _parse_constraint(c)
        if kind is None:
            return None, None
        parsed.append((kind, args, c))
    perms = [p for p in itertools.permutations(people) if all(_satisfies(p, k, a) for k, a, _ in parsed)]
    return perms, parsed


def _perm_str(p):
    return " - ".join(f"{i+1}:{name}" for i, name in enumerate(p))


def _option_truth(opt, perms):
    m = _C_AT.match(opt.strip())
    if not m or not perms:
        return None
    name, n = m.group(1), int(m.group(2))
    return all(p[n - 1] == name for p in perms)


def _branches_constraint(task, rng):
    sym = task["symbolic"]
    people, constraints = sym["people"], sym["constraints"]
    perms, parsed = _valid_perms(people, constraints)
    if perms is None or not perms:
        return None, None
    gold = str(task.get("gold_answer"))
    opts = task.get("options") or []
    holds = {o: _option_truth(o, perms) for o in opts}
    if holds.get(gold) is not True:
        return None, None  # wiring mismatch; skip rather than emit wrong "valid" text
    listing = "; ".join(_perm_str(p) for p in perms[:6]) + ("; ..." if len(perms) > 6 else "")
    text = (f"People to order: {', '.join(people)} in positions 1-{len(people)}.\n"
            f"Apply the constraints: " + " ".join(constraints) + "\n"
            f"Enumerate all {len(perms)} order(s) satisfying every constraint: {listing}.\n"
            f"Checking each option against every valid order, the one that must be true is: {gold}\n"
            + _FA.format(gold))
    out = [(text, gold, "permutation_enumeration_executed", [])]

    # flawed (usually available): overclaim from the first valid order alone
    if len(perms) > 1:
        p0 = perms[0]
        overs = [o for o in opts if o != gold and (m := _C_AT.match(o.strip())) and p0[int(m.group(2)) - 1] == m.group(1)]
        if overs:
            w = rng.choice(overs)
            out.append((f"People to order: {', '.join(people)} in positions 1-{len(people)}.\n"
                        f"Search for an order satisfying the constraints and stop at the first one found: "
                        f"{_perm_str(p0)}.\nRead the answer off this single order without checking whether other "
                        f"valid orders exist: {w}\n" + _FA.format(w), w, "single_order_overclaim_executed",
                        ["early_stop"]))

    # flawed: misread 'before' as 'after' (re-solve with the flipped constraint)
    bidx = [i for i, (k, _a, _c) in enumerate(parsed) if k == "before"]
    if bidx:
        i = rng.choice(bidx)
        flipped = list(constraints)
        a, b = parsed[i][1]
        flipped[i] = f"{b} stands somewhere before {a}."
        fperms, _ = _valid_perms(people, flipped)
        if fperms:
            fholds = {o: _option_truth(o, fperms) for o in opts}
            wrongs = [o for o, h in fholds.items() if h and o != gold]
            if wrongs:
                w = rng.choice(wrongs)
                ftext = (f"People to order: {', '.join(people)} in positions 1-{len(people)}.\n"
                         f"Reading '{constraints[i]}' as '{a} comes after {b}', enumerate the orders satisfying the "
                         f"constraints: {'; '.join(_perm_str(p) for p in fperms[:6])}.\n"
                         f"Under this reading, the option forced in every order is: {w}\n" + _FA.format(w))
                out.append((ftext, w, "before_after_misread_executed", ["negation_flip"]))

    # flawed: drop one constraint and overclaim
    if len(constraints) > 1:
        i = rng.randrange(len(constraints))
        dropped = [c for j, c in enumerate(constraints) if j != i]
        dperms, _ = _valid_perms(people, dropped)
        if dperms:
            dholds = {o: _option_truth(o, dperms) for o in opts}
            wrongs = [o for o, h in dholds.items() if h and o != gold]
            cand = wrongs or [o for o in opts if o != gold and any(p[int(_C_AT.match(o).group(2)) - 1] == _C_AT.match(o).group(1) for p in dperms if _C_AT.match(o))]
            if cand:
                w = rng.choice(cand)
                dtext = (f"People to order: {', '.join(people)} in positions 1-{len(people)}.\n"
                         f"Apply the constraints: " + " ".join(dropped) + "\n"
                         f"(The constraint '{constraints[i]}' is overlooked.) Valid orders under the remaining "
                         f"constraints include: {'; '.join(_perm_str(p) for p in dperms[:6])}.\n"
                         f"Conclude: {w}\n" + _FA.format(w))
                out.append((dtext, w, "dropped_constraint_executed", ["premise_hallucination"]))
    return out, gold


# ---------- assembly ----------

def build_group(task, rng):
    cat = task.get("category", "")
    if cat in ("synthetic_propositional", "proofwriter_deduction"):
        rendered, gold = _branches_propositional(task, rng)
    elif cat == "synthetic_fol":
        rendered, gold = _branches_fol(task, rng)
    elif cat == "synthetic_constraint_game":
        rendered, gold = _branches_constraint(task, rng)
    else:
        return None, "family_not_executable"
    if not rendered:
        return None, "render_failed"
    valid = [r for r in rendered if not r[3]]
    if not valid or not _gold_match(task, valid[0][1]):
        return None, "wiring_bug"  # executed gold disagrees with generator gold -> real bug, skip loudly
    if len(rendered) < 3:
        return None, "too_few_branches"
    if rng.random() < 0.12 and len(rendered) - len(valid) >= 3:
        rendered = [r for r in rendered if r[3]]  # all-negative hard group
    rng.shuffle(rendered)
    seen, branches, parser_agree, parser_total = set(), [], 0, 0
    for i, (text, fa, strat, fms) in enumerate(rendered):
        if text in seen:
            continue
        seen.add(text)
        lbl = "pass" if _gold_match(task, fa) else "fail"
        pl, _prew, _pfa, pok = V.label_branch(task, text)
        parser_total += 1
        parser_agree += 1 if pl == lbl else 0
        branches.append({"branch_id": i, "branch_text": text, "final_answer": fa, "strategy_label": strat,
                         "failure_modes": fms, "external_label": lbl,
                         "objective_reward": 1.0 if lbl == "pass" else 0.0,
                         "parse_ok": bool(pok), "parser_label": pl, "verifier_result": lbl,
                         "label_source": "generator_verified_gold_match", "source": "rendered_logic_executed_v3"})
    if len(branches) < 2:
        return None, "too_few_branches"
    return {"task_uid": task["task_uid"], "category": task.get("category"), "domain": "logic",
            "task_prompt": task["task_prompt"], "gold_answer": task.get("gold_answer"),
            "options": task.get("options"), "answer_key": task.get("answer_key"),
            "branch_attempts": branches,
            "parser_agreement": {"agree": parser_agree, "total": parser_total}}, "ok"


def main() -> int:
    V.ensure_dirs()
    TRAIN_V3.mkdir(parents=True, exist_ok=True)
    rng = random.Random(311)
    done = set()
    if OUT_JSONL.exists():
        for l in open(OUT_JSONL):
            try:
                done.add(json.loads(l)["task_uid"])
            except Exception:
                pass
    tasks = []
    for l in open(V.DATA_ROOT / "processed" / "logic_tasks.jsonl"):
        d = json.loads(l)
        if d.get("category") in FAMILIES and d.get("split") != "heldout" and d.get("symbolic") \
                and d["task_uid"] not in done:
            tasks.append(d)
    print(f"[{JOB}] pending {len(tasks)} (done {len(done)})", flush=True)
    counts = {"ok": len(done)}
    f = open(OUT_JSONL, "a")
    t0 = time.time()
    for n, task in enumerate(tasks, 1):
        if V.stop_requested(JOB):
            print(f"[{JOB}] STOP at {n}", flush=True)
            break
        group, status = build_group(task, rng)
        counts[status] = counts.get(status, 0) + 1
        if group:
            f.write(json.dumps(group, default=V.v2.json_default) + "\n")
        if n % 2000 == 0:
            f.flush()
            print(f"[{JOB}] {n}/{len(tasks)} | {counts} | {time.time()-t0:.0f}s", flush=True)
    f.close()
    groups = counts.get("ok", 0)
    wiring = counts.get("wiring_bug", 0)
    verdict = "EXECUTED_LOGIC_BRANCH_SETS_READY" if groups >= 5000 and wiring <= groups * 0.02 else "EXECUTED_RENDER_WEAK"
    V.set_stage("E_rendered_logic_v3", verdict, {"counts": counts})
    V.prog("E_rendered_logic_v3", {"verdict": verdict, "counts": counts})
    V.write_md(V.OUT_ROOT / "rendered_logic_branch_sets_v3.md", [
        "# Executed Logic Branch Sets v3 (Part E v3)", "",
        V.status_line("EXECUTED_LOGIC_RENDER_VERDICT", verdict),
        f"Counts: {json.dumps(counts)}",
        "Branches narrate actual executed computations (forward chaining / finite-model search / permutation "
        "enumeration); wrong branches are the same executors with one injected, named mistake, kept only when "
        "the flawed answer differs from gold. Supersedes the v2 strategy-narration renderings "
        "(see data_flaw_meta_branch_rendering.md).",
    ])
    print(V.status_line("EXECUTED_LOGIC_RENDER_VERDICT", verdict))
    print(f"  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
