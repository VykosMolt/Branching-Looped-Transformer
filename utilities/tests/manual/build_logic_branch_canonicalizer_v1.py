"""PART C — logic canonicalization: synthetic verified generation + real-dataset ingest.

Emits a unified logic_tasks set (one record per task) feeding branch-pool construction.
Every synthetic item is externally verified at generation; real MCQ/entailment items carry
the dataset answer key as the external label. Official test splits -> heldout/diagnostic.
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

import bg_corecontent_v2_common as v2  # noqa: E402
import branch_training_logic_v1_common as L  # noqa: E402

OUT_ROOT = v2.PROBE_ROOT / "branch_training_logic_expansion_terminal_v1_2026-06-06"
DATA_ROOT = v2.PROJECT_ROOT / "shared/data/branch_training_logic_expansion_v1"
PROC = DATA_ROOT / "processed"
PROGRESS = OUT_ROOT / "progress"

# synthetic targets (verified) — alone exceed the 20k logic train minimum
SYNTH_TARGETS = {"synthetic_propositional": 10000, "proofwriter_deduction": 7000,
                 "synthetic_fol": 5000, "synthetic_constraint_game": 3000}

# real logic datasets to ingest (best-effort, capped); family -> (hf_id, kw, kind)
REAL_LOGIC = [
    ("mcq_logical_reading", "lucasmccabe/logiqa", {"revision": "refs/convert/parquet"}, "mcq"),
    ("mcq_logical_reading", "tasksource/reclor", {}, "mcq"),
    ("mcq_logical_reading", "datatune/LogiQA2.0", {}, "mcq"),
    ("lsat_analytical_reasoning", "hails/agieval-lsat-ar", {}, "agieval"),
    ("lsat_logical_reasoning", "hails/agieval-lsat-lr", {}, "agieval"),
    ("logical_deduction_bbh", "tasksource/bigbench", {"name": "logical_deduction"}, "mcq"),
    ("fol_entailment", "tasksource/folio", {}, "entail"),
    ("proofwriter_deduction", "tasksource/proofwriter", {}, "entail"),
    ("ruletaker_deduction", "tasksource/ruletaker", {}, "entail"),
    ("logical_entailment", "tasksource/logical-entailment", {}, "entail"),
    ("prontoqa_reasoning", "renma/PrOntoQA", {}, "entail"),
]
REAL_CAP_TRAIN = 4000
REAL_CAP_EVAL = 1200
ENTAIL_LABELS = {"true": "True", "false": "False", "unknown": "Unknown", "entailment": "True",
                 "contradiction": "False", "neutral": "Unknown", "yes": "True", "no": "False",
                 "1": "True", "0": "False", "2": "Unknown"}


def _split_for(uid: str, prompt: str, force_heldout: bool = False) -> str:
    if force_heldout:
        return "heldout"
    h = L._sid("logic_split", uid, prompt) % 100
    return "train" if h < 80 else ("val" if h < 90 else "heldout")


def _rec(category, dataset, uid, prompt, options, answer_key, label_type, gold, verifier, split, extra=None):
    r = {"task_uid": uid, "dataset": dataset, "category": category, "domain": "logic",
         "task_type": category, "task_prompt": prompt, "options": options, "answer_key": int(answer_key),
         "gold_answer": gold, "label_type": label_type, "verifier": verifier, "split": split}
    if extra:
        r.update(extra)
    return r


def _ingest_mcq(family, hf_id, kw, split_name, rows, cap, force_heldout):
    out = []
    for i in range(min(cap, len(rows))):
        ex = rows[i]
        ctx = ex.get("context") or ex.get("article") or ex.get("passage") or ex.get("inputs") or ""
        q = ex.get("question") or ex.get("query") or ex.get("Question") or ""
        opts = ex.get("options") or ex.get("choices") or ex.get("answers")
        if isinstance(opts, dict):
            opts = opts.get("text") or list(opts.values())
        ans = ex.get("correct_option", ex.get("answer", ex.get("label", ex.get("gold", ex.get("answerKey")))))
        if not isinstance(opts, list) or len(opts) < 2 or ans is None:
            continue
        if isinstance(ans, list):
            ans = ans[0] if ans else None
        try:
            ai = int(ans)
        except Exception:
            ai = None
            if isinstance(ans, str) and len(ans) == 1 and ans.upper() in "ABCDEFG":
                ai = ord(ans.upper()) - 65
        if ai is None or not (0 <= ai < len(opts)):
            continue
        prompt = (f"{ctx}\n{q}" if ctx else q).strip()
        if not prompt:
            continue
        uid = f"{hf_id}::{split_name}::{i}"
        out.append(_rec(family, hf_id, uid, prompt, [str(o) for o in opts], ai, "mcq_answer_key",
                        str(opts[ai]), {"type": "mcq_answer_key"}, _split_for(uid, prompt, force_heldout)))
    return out


def _ingest_entail(family, hf_id, kw, split_name, rows, cap, force_heldout):
    out = []
    for i in range(min(cap, len(rows))):
        ex = rows[i]
        ctx = ex.get("theory") or ex.get("context") or ex.get("premises") or ex.get("premise") or ex.get("text") or ""
        q = ex.get("question") or ex.get("hypothesis") or ex.get("conclusion") or ex.get("query") or ""
        lab = ex.get("answer", ex.get("label", ex.get("gold", ex.get("validation"))))
        if isinstance(lab, list):
            lab = lab[0] if lab else None
        if lab is None:
            continue
        gold = ENTAIL_LABELS.get(str(lab).strip().lower())
        if gold is None:
            continue
        prompt = (f"{ctx}\nStatement: {q}\nIs the statement True, False, or Unknown?").strip()
        if len(prompt) < 8:
            continue
        uid = f"{hf_id}::{split_name}::{i}"
        out.append(_rec(family, hf_id, uid, prompt, L.ENTAIL_OPTIONS, L.ENTAIL_OPTIONS.index(gold),
                        "deterministic_rubric", gold, {"type": "entailment_label"},
                        _split_for(uid, prompt, force_heldout)))
    return out


def _ingest_agieval(family, hf_id, kw, split_name, rows, cap, force_heldout):
    out = []
    for i in range(min(cap, len(rows))):
        ex = rows[i]
        q = ex.get("query") or ex.get("question") or ""
        opts = ex.get("choices") or ex.get("options")
        gold = ex.get("gold") or ex.get("answer") or ex.get("label")
        if isinstance(gold, list):
            gold = gold[0] if gold else None
        if not isinstance(opts, list) or gold is None:
            continue
        try:
            ai = int(gold)
        except Exception:
            continue
        if not (0 <= ai < len(opts)):
            continue
        uid = f"{hf_id}::{split_name}::{i}"
        out.append(_rec(family, hf_id, uid, q.strip(), [str(o) for o in opts], ai, "mcq_answer_key",
                        str(opts[ai]), {"type": "mcq_answer_key"}, _split_for(uid, q, force_heldout)))
    return out


def ingest_real(family, hf_id, kw, kind):
    from datasets import load_dataset
    fn = {"mcq": _ingest_mcq, "entail": _ingest_entail, "agieval": _ingest_agieval}[kind]
    out = []
    for split_name, cap, force_held in (("train", REAL_CAP_TRAIN, False), ("validation", REAL_CAP_EVAL, True),
                                        ("test", REAL_CAP_EVAL, True)):
        try:
            ds = load_dataset(hf_id, split=split_name, **kw)
        except Exception:
            continue
        try:
            out += fn(family, hf_id, kw, split_name, ds, cap, force_held)
        except Exception:
            continue
    return out


def main() -> int:
    started = time.time()
    PROC.mkdir(parents=True, exist_ok=True); PROGRESS.mkdir(parents=True, exist_ok=True)
    tasks = []

    # 1) synthetic (verified)
    for cat, n in SYNTH_TARGETS.items():
        t0 = time.time()
        if cat == "synthetic_fol":
            # balance valid/invalid by capping each label
            raw = L.generate_batch(cat, n * 3, seed0=0)
            byl = defaultdict(list)
            for r in raw:
                byl[r["gold_answer"]].append(r)
            half = n // 2
            batch = byl.get("Valid", [])[:half] + byl.get("Invalid", [])[:n - len(byl.get("Valid", [])[:half])]
        else:
            batch = L.generate_batch(cat, n, seed0=0)
        for r in batch:
            uid = r["task_uid"]
            tasks.append(_rec(cat, r["dataset"], uid, r["prompt"], r["options"], r["answer_key"],
                              r["label_type"], r["gold_answer"], r["verifier"],
                              _split_for(uid, r["prompt"]), extra={"proof_depth": r.get("proof_depth"),
                                                                   "symbolic": r.get("symbolic")}))
        print(f"  synthetic {cat:28} {len(batch):6} in {time.time()-t0:.0f}s", flush=True)
        (PROGRESS / "C_logic.json").write_text(json.dumps({"synthetic_done": cat, "tasks": len(tasks)}) + "\n")

    # 2) real (best effort, capped, diversity/OOD)
    real_counts = {}
    for family, hf_id, kw, kind in REAL_LOGIC:
        try:
            got = ingest_real(family, hf_id, kw, kind)
        except Exception as e:
            got = []
            print(f"  real {hf_id} FAIL {type(e).__name__}", flush=True)
        real_counts[hf_id] = len(got)
        tasks += got
        print(f"  real {family:24} {hf_id:34} -> {len(got)}", flush=True)
        (PROGRESS / "C_logic.json").write_text(json.dumps({"real_done": hf_id, "tasks": len(tasks)}) + "\n")

    # write
    v2.write_jsonl(PROC / "logic_tasks.jsonl", tasks)
    flat = [{k: (json.dumps(v, default=v2.json_default) if isinstance(v, (list, dict)) else v) for k, v in t.items()}
            for t in tasks]
    try:
        import pandas as pd
        pd.DataFrame(flat).to_parquet(PROC / "logic_tasks.parquet", index=False)
    except Exception as e:
        print("  parquet skipped:", type(e).__name__, flush=True)

    by_cat = Counter(t["category"] for t in tasks)
    by_split = Counter(t["split"] for t in tasks)
    by_label = Counter(t["label_type"] for t in tasks)
    train_n = by_split.get("train", 0); held_n = by_split.get("heldout", 0)
    n_families = len(set(t["category"] for t in tasks))
    solver_cats = {"synthetic_constraint_game", "proofwriter_deduction", "synthetic_propositional", "synthetic_fol"}
    has_solver = any(t["category"] in solver_cats for t in tasks)
    if train_n >= 20000 and held_n >= 5000 and n_families >= 5 and has_solver:
        verdict = "LOGIC_VERIFIER_READY"
    elif has_solver and n_families >= 5:
        verdict = "LOGIC_SOLVER_PARTIAL" if train_n < 20000 else "LOGIC_VERIFIER_READY"
    elif n_families <= 2:
        verdict = "LOGIC_MCQ_ONLY"
    else:
        verdict = "LOGIC_SOLVER_PARTIAL"
    payload = {"LOGIC_CANONICALIZATION_VERDICT": verdict, "total_tasks": len(tasks),
               "by_category": dict(by_cat), "by_split": dict(by_split), "by_label_type": dict(by_label),
               "real_counts": real_counts, "train_groups": train_n, "heldout_groups": held_n,
               "n_categories": n_families, "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "logic_canonicalization.json", payload)
    v2.write_md(OUT_ROOT / "logic_canonicalization.md", [
        "# Logic Canonicalization (Part C)", "", v2.status_line("LOGIC_CANONICALIZATION_VERDICT", verdict), "",
        f"Total logic tasks: {len(tasks)} (train {train_n}, heldout {held_n}) across {n_families} categories. "
        "Synthetic items are verified at generation (truth-table / finite-model / forward-chaining / z3); "
        "real items carry dataset answer keys. Official test splits forced to heldout.", "",
        "## By category", "", *v2.md_table([{"category": k, "tasks": v} for k, v in by_cat.most_common()], ["category", "tasks"]),
        "", "## By split", "", *v2.md_table([{"split": k, "tasks": v} for k, v in by_split.items()], ["split", "tasks"]),
    ])
    (PROGRESS / "C_logic.json").write_text(json.dumps({"verdict": verdict, "tasks": len(tasks)}) + "\n")
    print(v2.status_line("LOGIC_CANONICALIZATION_VERDICT", verdict))
    print(f"  total={len(tasks)} train={train_n} heldout={held_n} categories={n_families}")
    print(f"  by_category={dict(by_cat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
