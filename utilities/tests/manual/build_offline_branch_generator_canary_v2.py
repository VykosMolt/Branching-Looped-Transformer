"""PART C — fixed canary suite for branch_training_offline_verifier_generator_v2.

Builds a stable, statistically meaningful canary (target >=100/domain; logic >=150) that is
strictly task-disjoint from everything trained on (excludes the 81 logic cross-split leaks and
any heldout prompt that hashes into a train/val split). No generation here — just task
selection, stable ids/seeds, K, and slice tags. alignment = preference-pair sentinel.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402

TARGET = {"logic": 200, "math": 110, "reasoning": 110, "coding": 100, "alignment": 110}
HARD_LOGIC = {"lsat_analytical_reasoning", "proofwriter_deduction", "synthetic_constraint_game", "fol_entailment"}
CG = V.PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl"


def _h(s) -> str:
    return V.v2.text_hash(str(s))


def _train_hashes() -> set:
    """prompt-hashes of everything trained on / not-heldout — to enforce canary disjointness."""
    hs = set()
    import pandas as pd
    lt = pd.read_parquet(V.DATA_ROOT / "processed/logic_tasks.parquet")
    for _, r in lt[lt["split"].isin(["train", "val"])].iterrows():
        hs.add(_h(r["task_prompt"]))
    for g in V.read_jsonl(CG):
        if g.get("split") in ("train", "val"):
            rec = V._reconstruct_core_gen(g)
            if rec:
                hs.add(_h(rec["task_prompt"]))
    return hs


def _cid(domain, prompt) -> str:
    return f"{domain}_{_h(prompt)[:10]}"


def _alignment_pairs(n, train_hs):
    out = []
    for g in V.read_jsonl(CG):
        if g.get("domain") != "alignment" or g.get("split") != "heldout":
            continue
        cands = g.get("candidates", [])
        pos = [c for c in cands if c.get("is_positive")]
        neg = [c for c in cands if not c.get("is_positive")]
        if not pos or not neg:
            continue
        chosen, rejected = pos[0]["candidate_text"], neg[0]["candidate_text"]
        if not chosen or not rejected or chosen == rejected:
            continue
        # shared prompt prefix (HH context), if any
        m = 0
        for a, b in zip(chosen, rejected):
            if a != b:
                break
            m += 1
        prompt = chosen[:m].rsplit("\n", 1)[0] if m > 20 else ""
        if _h(prompt or chosen) in train_hs:
            continue
        out.append({"canary_id": _cid("alignment", g.get("task_uid", chosen[:40])), "domain": "alignment",
                    "category": g.get("subdomain", "preference"), "split": "canary", "label_type": "preference",
                    "task_prompt": prompt, "pref": {"chosen": chosen, "rejected": rejected}, "K": 1,
                    "slice": ["alignment_pref"], "source": "corecontent_v2", "seed": int(_h(chosen)[:8], 16) % (2**31)})
        if len(out) >= n:
            break
    return out


def main() -> int:
    train_hs = _train_hashes()
    canary, dropped_leak = [], 0

    # logic: stratified across 10 families, hard-tagged, leak-filtered
    per_fam = max(8, TARGET["logic"] // 10)
    by_fam: dict[str, list] = {}
    for t in V.logic_heldout():
        if _h(t["task_prompt"]) in train_hs:
            dropped_leak += 1
            continue
        by_fam.setdefault(t["category"], []).append(t)
    for fam in sorted(by_fam):
        for t in sorted(by_fam[fam], key=lambda x: _h(x["task_prompt"]))[:per_fam]:
            hard = fam in HARD_LOGIC
            canary.append({"canary_id": _cid("logic", t["task_prompt"]), "domain": "logic", "category": fam,
                           "split": "canary", "task_prompt": t["task_prompt"], "gold_answer": t.get("gold_answer"),
                           "options": t.get("options"), "answer_key": t.get("answer_key"),
                           "label_type": t.get("label_type"), "verifier": t.get("verifier"),
                           "proof_depth": t.get("proof_depth"), "K": 8 if hard else 4,
                           "slice": ["logic", "hard_logic" if hard else "logic_std"],
                           "source": "logic_tasks", "seed": int(_h(t["task_prompt"])[:8], 16) % (2**31)})

    # math / reasoning from corecontent_v2 heldout
    for dom in ("math", "reasoning"):
        got = []
        for t in V.core_heldout(dom, TARGET[dom] * 3):
            if _h(t["task_prompt"]) in train_hs:
                dropped_leak += 1
                continue
            got.append(t)
            if len(got) >= TARGET[dom]:
                break
        for i, t in enumerate(got):
            sl = [dom] + (["math_exact"] if dom == "math" else ["reasoning", ("direct_answer_candidate" if i % 3 == 0 else "reasoning_std")])
            canary.append({"canary_id": _cid(dom, t["task_prompt"]), "domain": dom, "category": t.get("dataset", dom),
                           "split": "canary", "task_prompt": t["task_prompt"], "gold_answer": t.get("gold_answer"),
                           "options": t.get("options"), "answer_key": t.get("answer_key"),
                           "label_type": t.get("label_type"), "verifier": t.get("verifier"),
                           "K": 8 if (dom == "math" and i % 4 == 0) else 4, "slice": sl,
                           "source": "corecontent_v2", "seed": int(_h(t["task_prompt"])[:8], 16) % (2**31)})

    # coding: MBPP heldout slice (offset beyond trained), unit-test verifiable
    cseen = 0
    for t in V.coding_heldout(TARGET["coding"] + 80, offset=200):
        if _h(t["task_prompt"]) in train_hs:
            dropped_leak += 1
            continue
        canary.append({"canary_id": _cid("coding", t["task_prompt"]), "domain": "coding", "category": "mbpp",
                       "split": "canary", "task_prompt": t["task_prompt"], "gold_answer": t.get("gold_answer"),
                       "label_type": "unit_tests", "unit_tests": t.get("unit_tests"), "test_setup": t.get("test_setup"),
                       "verifier": t.get("verifier"), "K": 4, "slice": ["coding", "coding_parse"],
                       "source": "mbpp", "seed": int(_h(t["task_prompt"])[:8], 16) % (2**31)})
        cseen += 1
        if cseen >= TARGET["coding"]:
            break

    # alignment preference-pair sentinel
    canary += _alignment_pairs(TARGET["alignment"], train_hs)

    # report
    import collections
    by_dom = collections.Counter(c["domain"] for c in canary)
    by_fam_logic = collections.Counter(c["category"] for c in canary if c["domain"] == "logic")
    k8 = sum(1 for c in canary if c.get("K") == 8)
    small = [d for d in V.CORE_DOMAINS if by_dom.get(d, 0) < 100]

    V.CANARY_DIR.mkdir(parents=True, exist_ok=True)
    with open(V.CANARY_DIR / "offline_branch_generator_canary_v2.jsonl", "w") as f:
        for c in canary:
            f.write(json.dumps(c, default=V.v2.json_default) + "\n")
    try:
        import pandas as pd
        pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in c.items()} for c in canary]) \
            .to_parquet(V.CANARY_DIR / "offline_branch_generator_canary_v2.parquet")
    except Exception as e:
        print("  parquet warn:", e)

    if any(by_dom.get(d, 0) == 0 for d in V.CORE_DOMAINS):
        verdict = "BLOCKED"
    elif by_dom.get("logic", 0) < 100 or len(by_fam_logic) < 8:
        verdict = "CANARY_LOGIC_WEAK"
    elif small:
        verdict = "CANARY_READY_SMALL"
    else:
        verdict = "CANARY_READY"

    payload = {"CANARY_SUITE_VERDICT": verdict, "total": len(canary), "by_domain": dict(by_dom),
               "logic_by_family": dict(by_fam_logic), "k8_hard_subset": k8, "dropped_leak": dropped_leak,
               "small_domains_lt100": small, "target": TARGET,
               "wilson_halfwidth_at_p0.7": {d: round((V.wilson_ci(int(0.7 * n), n)[1] - V.wilson_ci(int(0.7 * n), n)[0]) / 2, 3)
                                            for d, n in by_dom.items()}}
    V.write_json(V.OUT_ROOT / "canary_suite.json", payload)
    V.write_md(V.OUT_ROOT / "canary_suite.md", [
        "# Canary Suite (Part C)", "", V.status_line("CANARY_SUITE_VERDICT", verdict),
        f"Total {len(canary)} groups; dropped {dropped_leak} leak/train-overlap tasks.", "",
        "## Per domain (n, Wilson ±halfwidth @ p≈0.7)",
        *[f"- {d}: {by_dom[d]} (±{round((V.wilson_ci(int(0.7*by_dom[d]),by_dom[d])[1]-V.wilson_ci(int(0.7*by_dom[d]),by_dom[d])[0])/2,3)})" for d in V.CORE_DOMAINS if d in by_dom],
        "", f"## Logic families ({len(by_fam_logic)})", *[f"- {f}: {n}" for f, n in sorted(by_fam_logic.items())],
        "", f"## K=8 hard subset: {k8} groups (hard logic families + math sample).",
        "", "## Slices", "- logic: hard_logic vs logic_std; reasoning: direct_answer_candidate vs reasoning_std; "
        "math: math_exact; coding: coding_parse; alignment: preference-pair sentinel (no generation).",
        "", "Disjointness: every canary prompt-hash excluded if present in any train/val split (logic + corecontent_v2). "
        "Stable canary_id + seed per task; K=4 default, K=8 hard subset.",
    ])
    V.set_stage("C_canary_suite", verdict, {"total": len(canary), "by_domain": dict(by_dom)})
    V.prog("C_canary_suite", {"verdict": verdict, "by_domain": dict(by_dom)})
    print(V.status_line("CANARY_SUITE_VERDICT", verdict))
    print(f"  total {len(canary)} | by_domain {dict(by_dom)} | logic_families {len(by_fam_logic)} | "
          f"K8 {k8} | dropped_leak {dropped_leak} | small<100 {small}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
