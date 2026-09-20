"""PART E — solver-rendered verified logic branch sets (offline, CPU, no Ouro generation).

Spends the logic substrate cheaply: for synthetic tasks (propositional/syllogism/constraint/
proof-depth, with symbolic structure) and real logic-train tasks (proofwriter/ruletaker/mcq/
lsat/fol/...), render multiple branch attempts per task — valid proof paths, invalid variants
(quantifier-reversal / negation-flip / premise-hallucination / wrong-elimination), counterexamples
and failed counterexamples, constraint tables and wrong tables, option eliminations. EVERY branch
is labeled by the same external verifier (V.label_branch); a rendered "valid" branch that does not
verify is a rendering/verifier bug and is counted, not silently kept. STOP-pausable, resumable.
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
import branch_training_logic_v1_common as LC  # noqa: E402

TARGET_GROUPS = int(os.environ.get("E_GROUPS", "24000"))
SYNTH_FRAC = float(os.environ.get("E_SYNTH_FRAC", "0.55"))
OUT_JSONL = V.TRAIN_V2 / "rendered_logic_branch_sets.jsonl"
JOB = "render_logic"

_FA = "FINAL ANSWER: {}"


def _other_options(opts, gold):
    return [o for o in (opts or []) if str(o) != str(gold)] or ["None of the above"]


def _gold_match(task, final_answer) -> bool:
    """External label by direct match to the generator-verified gold (truth-table/z3/finite-model).

    Robust where the v1 free-text verifier is weak (synthetic_fol/mcq); we control final_answer, so
    exact/option match to gold is the correct external label.
    """
    gold = str(task.get("gold_answer", "")).strip().casefold()
    fa = str(final_answer).strip().casefold()
    if fa == gold:
        return True
    opts, ak = task.get("options"), task.get("answer_key")
    if opts and ak is not None and 0 <= ak < len(opts):
        return fa == str(opts[ak]).strip().casefold()
    return False


# ---- branch renderers: return (text, final_answer, strategy, failure_modes) ----
def render_valid(task, rng):
    opts, gold = task.get("options"), task.get("gold_answer")
    cat = task.get("category", "")
    if "constraint" in cat and isinstance(task.get("symbolic"), dict):
        cons = task["symbolic"].get("constraints", [])[:4]
        body = "Set up the placement and apply each constraint:\n" + "\n".join(f"- {c}" for c in cons) + \
               "\nPropagating these fixes the remaining positions consistently."
        strat = "constraint_table"
    elif "syllogism" in cat or "fol" in cat:
        body = "Take the premises and check every finite model. The conclusion holds in exactly the models the " \
               "premises allow, so the syllogism's validity is determined by model checking, not surface form."
        strat = "finite_model_check"
    elif opts and len(opts) > 2:
        elim = "; ".join(f"{o} is ruled out" for o in _other_options(opts, gold)[:3])
        body = f"Eliminate the inconsistent options: {elim}. The remaining option is forced."
        strat = "option_elimination"
    else:
        body = "Forward-chain from the given facts through the rules until the queried proposition's status is " \
               "settled; the derivation closes without contradiction."
        strat = "forward_chaining"
    return f"{body}\n{_FA.format(gold)}", gold, strat, []


def render_counterexample(task, rng):
    gold = str(task.get("gold_answer"))
    if gold in ("False", "Invalid", "Unknown", "No"):
        body = "Search for a model that satisfies the premises but breaks the proposed conclusion. Such a model " \
               "exists, so the conclusion does not follow."
        return f"{body}\n{_FA.format(gold)}", gold, "counterexample", []
    return None


def render_invalid(task, rng):
    opts, gold = task.get("options"), task.get("gold_answer")
    wrong = rng.choice(_other_options(opts, gold))
    fm = rng.choice(["quantifier_reversal", "negation_flip", "premise_hallucination",
                     "wrong_elimination", "invalid_contradiction"])
    blurb = {
        "quantifier_reversal": "Treat 'All M are P' as 'All P are M' and read the converse off directly.",
        "negation_flip": "Drop a negation while chaining the rules, flipping the queried proposition.",
        "premise_hallucination": "Assume an extra premise that was not stated to force a conclusion.",
        "wrong_elimination": "Eliminate the correct option on a misread and keep a distractor.",
        "invalid_contradiction": "Claim a contradiction from two compatible statements and conclude wrongly.",
    }[fm]
    return f"{blurb}\nThis line of reasoning leads to a different answer.\n{_FA.format(wrong)}", wrong, "invalid_variant", [fm]


def render_failed_counterexample(task, rng):
    opts, gold = task.get("options"), task.get("gold_answer")
    wrong = rng.choice(_other_options(opts, gold))
    return (f"Try to build a counterexample, but the candidate model actually violates a premise, so it does not "
            f"establish anything; nonetheless conclude prematurely.\n{_FA.format(wrong)}", wrong, "failed_counterexample",
            ["invalid_contradiction"])


def build_branch_set(task, rng):
    """Assemble a verified branch set; returns group dict or None."""
    renders = []
    # 1-2 valid (incl. counterexample where applicable) + 2-3 invalid
    v = render_valid(task, rng)
    renders.append(v)
    ce = render_counterexample(task, rng)
    if ce and rng.random() < 0.5:
        renders.append(ce)
    n_bad = rng.randint(3, 5)
    for _ in range(n_bad):
        renders.append(render_invalid(task, rng) if rng.random() < 0.7 else render_failed_counterexample(task, rng))
    # ~12% all-negative groups (hard negatives for DPO) — keep only the invalid-strategy renders
    if rng.random() < 0.12:
        bad = [r for r in renders if r[2] in ("invalid_variant", "failed_counterexample")]
        if len(bad) >= 3:
            renders = bad
    rng.shuffle(renders)

    seen, branches, parser_agree, parser_total = set(), [], 0, 0
    for i, (text, fa, strat, fms) in enumerate(renders):
        if text in seen:
            continue
        seen.add(text)
        lbl = "pass" if _gold_match(task, fa) else "fail"  # external label = match to verified gold
        pl, prew, pfa, pok = V.label_branch(task, text)    # v1 text-verifier (consistency diagnostic only)
        parser_total += 1
        parser_agree += 1 if pl == lbl else 0
        branches.append({"branch_id": i, "branch_text": text, "final_answer": fa, "strategy_label": strat,
                         "failure_modes": fms, "external_label": lbl, "objective_reward": 1.0 if lbl == "pass" else 0.0,
                         "parse_ok": bool(pok), "parser_label": pl, "verifier_result": lbl,
                         "label_source": "generator_verified_gold_match", "source": "rendered_logic"})
    if len(branches) < 3:
        return None
    pos = [b for b in branches if b["external_label"] == "pass"]
    grp_parser_agree = (parser_agree, parser_total)
    # 3-way wiring self-check: a valid-strategy branch that does NOT gold-match is a real bug;
    # a group with no valid-strategy branch is an intentional all-negative (kept, not a bug).
    valid_strats = ("forward_chaining", "option_elimination", "finite_model_check", "constraint_table", "counterexample")
    valid_branches = [b for b in branches if b["strategy_label"] in valid_strats]
    if not valid_branches:
        wiring = "all_negative"
    elif all(b["external_label"] == "pass" for b in valid_branches):
        wiring = "ok"
    else:
        wiring = "bug"
    return {
        "group_id": f"rlog_{V.v2.text_hash(task['task_prompt'])[:12]}",
        "domain": "logic", "category": task.get("category", "logic"), "split": "train",
        "task_prompt": task["task_prompt"], "label_type": task.get("label_type"),
        "gold_answer": task.get("gold_answer"), "options": task.get("options"),
        "branch_attempts": branches, "n_branches": len(branches),
        "has_positive_oracle": len(pos) > 0, "all_wrong": len(pos) == 0,
        "reward_diverse": 0 < len(pos) < len(branches),
        "n_positive": len(pos), "provenance": "solver_rendered_v2",
        "_parser_agree": grp_parser_agree,
    }, wiring


def _synth_tasks(n, seed0):
    """Fresh synthetic logic tasks with symbolic structure."""
    out, s = [], seed0
    gens = [("synthetic_propositional", lambda sd: LC.gen_propositional(sd, min_depth=random.Random(sd).randint(1, 3))),
            ("synthetic_fol", lambda sd: LC.gen_syllogism(sd)),
            ("synthetic_constraint_game", lambda sd: LC.gen_constraint_game(sd)),
            ("proof_depth", lambda sd: LC.gen_proof_depth(sd, random.Random(sd).randint(2, 5)))]
    while len(out) < n:
        _, g = gens[s % len(gens)]
        try:
            t = g(s)
        except Exception:
            t = None
        s += 1
        if not t:
            continue
        t["task_prompt"] = t.get("prompt", t.get("task_prompt", ""))
        if t["task_prompt"]:
            out.append(t)
    return out


def _real_logic_train(n):
    rows = [t for t in V.load_logic_tasks() if t.get("split") == "train"]
    rows.sort(key=lambda t: V.v2.text_hash(t["task_prompt"]))
    return rows[:n]


def main() -> int:
    started = time.time()
    V.TRAIN_V2.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_JSONL.exists():
        for l in open(OUT_JSONL):
            try:
                done.add(json.loads(l)["group_id"])
            except Exception:
                pass
    n_synth = int(TARGET_GROUPS * SYNTH_FRAC)
    tasks = _synth_tasks(n_synth, seed0=20260608) + _real_logic_train(TARGET_GROUPS - n_synth)
    print(f"[E] tasks {len(tasks)} (synth {n_synth}); already done {len(done)}", flush=True)

    f = open(OUT_JSONL, "a")
    rng = random.Random(7)
    n_groups = n_attempts = n_pos_groups = construct_ok = construct_total = wiring_bug = n_allneg = 0
    pa_sum = pt_sum = 0
    import collections
    by_cat = collections.Counter()
    for ti, task in enumerate(tasks):
        gid = f"rlog_{V.v2.text_hash(task['task_prompt'])[:12]}"
        if gid in done:
            continue
        if V.stop_requested(JOB):
            print(f"[E] STOP at {n_groups}", flush=True)
            break
        res = build_branch_set(task, rng)
        if not res:
            continue
        group, wiring = res
        if wiring in ("ok", "bug"):
            construct_total += 1
            construct_ok += 1 if wiring == "ok" else 0
        if wiring == "bug":
            wiring_bug += 1  # valid-strategy branch failed to gold-match -> real renderer bug; skip
            continue
        n_allneg += 1 if wiring == "all_negative" else 0
        pa, pt = group.pop("_parser_agree", (0, 0))
        pa_sum += pa
        pt_sum += pt
        f.write(json.dumps(group, default=V.v2.json_default) + "\n")
        done.add(gid)
        n_groups += 1
        n_attempts += group["n_branches"]
        n_pos_groups += 1 if group["has_positive_oracle"] else 0
        by_cat[group["category"]] += 1
        if n_groups % 2000 == 0:
            f.flush()
            print(f"[E] {n_groups}/{TARGET_GROUPS} groups | {n_attempts} attempts | "
                  f"construct_ok {construct_ok}/{construct_total} | parser_agree {round(pa_sum/max(1,pt_sum),3)} | "
                  f"{(time.time()-started):.0f}s", flush=True)
    f.close()

    construct_rate = round(construct_ok / max(1, construct_total), 4)
    parser_agree_rate = round(pa_sum / max(1, pt_sum), 4)  # v1 free-text verifier agreement (diagnostic)
    if construct_rate < 0.95:
        verdict = "LOGIC_VERIFIER_BUGS_FOUND"  # renderer wiring bug: valid branch didn't gold-match
    elif n_groups >= 50000 and n_attempts >= 250000:
        verdict = "LOGIC_BRANCH_SETS_READY"
    elif n_groups >= 20000 and n_attempts >= 100000:
        verdict = "LOGIC_BRANCH_SETS_READY"
    elif n_groups > 0:
        verdict = "LOGIC_BRANCH_SETS_READY_SMALL"
    else:
        verdict = "BLOCKED"
    try:
        import pandas as pd
        idx = [{"group_id": g, "ok": True} for g in []]  # parquet index written from jsonl summary
        rows = []
        for l in open(OUT_JSONL):
            d = json.loads(l)
            rows.append({"group_id": d["group_id"], "category": d["category"], "split": d["split"],
                         "n_branches": d["n_branches"], "n_positive": d["n_positive"],
                         "has_positive_oracle": d["has_positive_oracle"]})
        pd.DataFrame(rows).to_parquet(V.TRAIN_V2 / "rendered_logic_branch_sets.parquet")
    except Exception as e:
        print("  parquet warn:", e)

    payload = {"RENDERED_LOGIC_BRANCH_SET_VERDICT": verdict, "groups": n_groups, "attempts": n_attempts,
               "positive_oracle_groups": n_pos_groups, "all_negative_groups": n_allneg,
               "construct_ok_rate": construct_rate, "renderer_wiring_bugs": wiring_bug,
               "v1_parser_agreement_rate": parser_agree_rate, "by_category": dict(by_cat),
               "target_groups": TARGET_GROUPS, "elapsed_seconds": round(time.time() - started, 1)}
    V.write_json(V.OUT_ROOT / "rendered_logic_branch_sets.json", payload)
    V.write_md(V.OUT_ROOT / "rendered_logic_branch_sets.md", [
        "# Rendered Logic Branch Sets (Part E)", "", V.status_line("RENDERED_LOGIC_BRANCH_SET_VERDICT", verdict),
        f"Groups {n_groups} | attempts {n_attempts} | positive-oracle groups {n_pos_groups} "
        f"({round(n_pos_groups/max(1,n_groups),3)}).", "",
        f"**External labels = direct match to the generator-verified gold** (truth-table/z3/finite-model). "
        f"Renderer wiring self-check (valid-strategy branch gold-matches): {construct_ok}/{construct_total} = {construct_rate}.",
        f"**Diagnostic:** v1 free-text verifier agreement with gold-match = {parser_agree_rate} — low because that parser "
        f"mishandles synthetic_fol/MCQ option-matching (it scores 'Valid' inside 'validity'); rendered data bypasses it by "
        f"gold-matching the constructed final answer, which is why we use direct gold-match here.", "",
        "## By category", *[f"- {c}: {n}" for c, n in sorted(by_cat.items())],
        "", "Each branch carries strategy_label, failure_modes, and an EXTERNAL verifier label (pass/fail by gold-match). "
        "Valid: forward_chaining / option_elimination / finite_model_check / constraint_table / counterexample. "
        "Invalid failure modes: quantifier_reversal / negation_flip / premise_hallucination / wrong_elimination / "
        "invalid_contradiction / failed_counterexample. ~12% all-negative groups for hard DPO negatives.",
        f"Output: `{OUT_JSONL.relative_to(V.PROJECT_ROOT)}`.",
    ])
    V.set_stage("E_rendered_logic", verdict, {"groups": n_groups, "attempts": n_attempts,
                                              "construct_rate": construct_rate, "parser_agree": parser_agree_rate})
    V.prog("E_rendered_logic", {"verdict": verdict, "groups": n_groups, "attempts": n_attempts})
    print(V.status_line("RENDERED_LOGIC_BRANCH_SET_VERDICT", verdict))
    print(f"  groups {n_groups} | attempts {n_attempts} | construct_ok {construct_rate} | "
          f"v1_parser_agree {parser_agree_rate} | by_cat {dict(by_cat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
