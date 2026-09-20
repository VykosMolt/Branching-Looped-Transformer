"""ROUND 3 data build — train_v4: v3 executed-logic views + WORKED math/coding (CPU-only).

Round 2 proved executed data repairs degeneration exactly where it exists (logic -0.233 -> 0.000)
and that v3 was 99.7% logic (`round2_executed_data_result.md`). v4 adds the missing domains:

- math: GSM8K + Hendrycks MATH dataset rationales (ground-truth worked solutions, gold finals
  from the dataset's own #### / \\boxed{} fields) + the 60 model-generated math pools (real
  solutions, externally labeled);
- coding: corecontent tasks' canonical solutions (code is the work) + the 60 model-generated
  coding pools (real reasoned code, unit-test labeled);
- anti-bare-answer DPO pairs ("show_work"): worked solution CHOSEN vs the literal degeneration
  format (`Branch 1: Answer: FINAL ANSWER: x`, answer correct) REJECTED — penalizes the
  failure mode itself, independent of correctness;
- wrong_final pairs: worked-correct chosen vs same work with a corrupted final.

No budget / direct-answer views are emitted for r3 (budget-as-text-style is retired per
round 2). Output: data/.../train_v4/{branch_set_rejection_sft,branch_set_dpo,one_branch_sft}.jsonl
(v3 rows merged in), manifest, stage FG_offline_views_v4.
"""
from __future__ import annotations
import collections
import json
import os
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

os.environ.setdefault("HF_HOME", str(V.PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(V.PROJECT_ROOT / "shared/hf_cache/datasets"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

TRAIN_V3 = V.DATA_ROOT / "train_v3"
TRAIN_V4 = V.DATA_ROOT / "train_v4"
TASKS_CC = V.PROJECT_ROOT / "shared/data/corecontent_v2/processed/tasks.jsonl"
GEN_GLOB = "shared/data/branch_training_logic_expansion_v1/processed/gen_shards/*.jsonl"
MAX_GSM = int(os.environ.get("V4_MAX_GSM", "6000"))
MAX_HEND = int(os.environ.get("V4_MAX_HEND", "3000"))
MAX_CODE = int(os.environ.get("V4_MAX_CODE", "4000"))
MAX_SHOW_WORK = int(os.environ.get("V4_MAX_SHOW_WORK", "4000"))
MIN_WORK = 80
WORK_CAP = 2400
HEND_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")
rng = random.Random(41)

VIEWS = ("branch_set_rejection_sft", "branch_set_dpo", "one_branch_sft")


def _branch_set(texts_with_finals, final):
    lines = []
    for i, (t, _f) in enumerate(texts_with_finals, 1):
        body = t.strip().split("FINAL ANSWER")[0].strip()[:WORK_CAP]
        lines.append(f"Branch {i}: {body}")
    return "\n".join(lines) + f"\nFINAL ANSWER: {final}"


def _bare(final):
    return f"Branch 1: Answer: FINAL ANSWER: {final}"


def _corrupt_number(final):
    m = re.search(r"-?\d+(?:\.\d+)?", str(final))
    if not m:
        return None
    n = m.group(0)
    try:
        new = str(int(float(n)) + rng.choice((1, -1, 2, 10)))
    except Exception:
        return None
    return str(final)[: m.start()] + new + str(final)[m.end():]


def _boxed(solution):
    i = solution.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    out = []
    while j < len(solution) and depth:
        ch = solution[j]
        depth += (ch == "{") - (ch == "}")
        if depth:
            out.append(ch)
        j += 1
    ans = "".join(out).strip()
    return ans or None


def gsm8k_rows():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        ans = r["answer"]
        if "####" not in ans:
            continue
        body, final = ans.rsplit("####", 1)
        body, final = body.strip(), final.strip()
        if len(body) < MIN_WORK or not final:
            continue
        rows.append({"prompt": r["question"].strip(), "work": body, "final": final, "src": "gsm8k"})
        if len(rows) >= MAX_GSM:
            break
    return rows


def hendrycks_rows():
    from datasets import load_dataset
    rows = []
    per = max(1, MAX_HEND // len(HEND_SUBJECTS))
    for subj in HEND_SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
        except Exception as e:
            print(f"[v4] hendrycks {subj} unavailable: {e}", flush=True)
            continue
        n = 0
        for r in ds:
            final = _boxed(r["solution"])
            work = r["solution"].strip()
            if not final or len(work) < MIN_WORK:
                continue
            rows.append({"prompt": r["problem"].strip(), "work": work, "final": final, "src": f"hendrycks_{subj}"})
            n += 1
            if n >= per:
                break
    return rows


def coding_rows():
    rows = []
    for l in open(TASKS_CC):
        d = json.loads(l)
        if d.get("domain") != "coding" or d.get("split") != "train" or d.get("force_heldout"):
            continue
        code = (d.get("canonical_solution") or "").strip()
        prompt = (d.get("prompt") or "").strip()
        if len(code) < MIN_WORK or not prompt:
            continue
        rows.append({"prompt": prompt, "work": code, "final": d.get("entry_point") or "solution implemented",
                     "src": d.get("source_dataset", "corecontent")})
        if len(rows) >= MAX_CODE:
            break
    return rows


def gen_pool_groups():
    import glob
    out = []
    for s in sorted(glob.glob(str(V.PROJECT_ROOT / GEN_GLOB))):
        for l in open(s):
            d = json.loads(l)
            if d.get("domain") in ("math", "coding") and str(d.get("task_prompt") or "").strip():
                out.append(d)
    return out


def main() -> int:
    started = time.time()
    TRAIN_V4.mkdir(parents=True, exist_ok=True)
    views = {k: [] for k in VIEWS}
    # carry v3 forward untouched
    carried = {}
    for name in VIEWS:
        p = TRAIN_V3 / f"{name}.jsonl"
        n = 0
        if p.exists():
            for l in open(p):
                views[name].append(json.loads(l))
                n += 1
        carried[name] = n

    math_rows = gsm8k_rows() + hendrycks_rows()
    code_rows = coding_rows()
    print(f"[v4] worked rows: math {len(math_rows)} | coding {len(code_rows)}", flush=True)
    show_work_budget = MAX_SHOW_WORK

    for r in math_rows:
        worked = _branch_set([(r["work"], r["final"])], r["final"])
        views["one_branch_sft"].append({"prompt": r["prompt"], "completion": worked, "domain": "math", "src": r["src"]})
        views["branch_set_rejection_sft"].append({"prompt": r["prompt"], "completion": worked, "domain": "math",
                                                  "src": r["src"]})
        if show_work_budget > 0:
            views["branch_set_dpo"].append({"prompt": r["prompt"], "chosen": worked, "rejected": _bare(r["final"]),
                                            "domain": "math", "pair": "show_work"})
            show_work_budget -= 1
        bad = _corrupt_number(r["final"])
        if bad:
            views["branch_set_dpo"].append({"prompt": r["prompt"], "chosen": worked,
                                            "rejected": _branch_set([(r["work"], bad)], bad),
                                            "domain": "math", "pair": "wrong_final"})

    for r in code_rows:
        worked = f"Branch 1: {r['work'][:WORK_CAP]}\nFINAL ANSWER: {r['final']}"
        views["one_branch_sft"].append({"prompt": r["prompt"], "completion": worked, "domain": "coding", "src": r["src"]})
        views["branch_set_rejection_sft"].append({"prompt": r["prompt"], "completion": worked, "domain": "coding",
                                                  "src": r["src"]})
        if show_work_budget > 0:
            views["branch_set_dpo"].append({"prompt": r["prompt"], "chosen": worked,
                                            "rejected": _bare(r["final"]), "domain": "coding", "pair": "show_work"})
            show_work_budget -= 1

    for g in gen_pool_groups():  # real model-generated pools: oracle pairs + worked sets
        prompt = g["task_prompt"].strip()
        ba = g.get("branch_attempts") or []
        pos = [(b.get("branch_text") or "", b.get("final_answer", "")) for b in ba
               if b.get("external_label") == "pass" and len(b.get("branch_text") or "") >= MIN_WORK]
        neg = [(b.get("branch_text") or "", b.get("final_answer", "")) for b in ba
               if b.get("external_label") == "fail" and len(b.get("branch_text") or "") >= MIN_WORK]
        dom = g["domain"]
        if pos:
            good = _branch_set(pos[:1] + neg[:2], pos[0][1])
            views["branch_set_rejection_sft"].append({"prompt": prompt, "completion": good, "domain": dom,
                                                      "src": "gen_pool"})
            if neg:
                bad = _branch_set(neg[:3], neg[0][1] or "Unknown")
                views["branch_set_dpo"].append({"prompt": prompt, "chosen": good, "rejected": bad,
                                                "domain": dom, "pair": "oracle"})

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
        rng.shuffle(uniq)
        with open(TRAIN_V4 / f"{name}.jsonl", "w") as fo:
            for r in uniq:
                fo.write(json.dumps(r, default=V.v2.json_default) + "\n")
        counts[name] = len(uniq)

    byd = collections.Counter(r.get("domain") for r in views["branch_set_rejection_sft"])
    pairs = collections.Counter(r["pair"] for r in views["branch_set_dpo"])
    manifest = {**counts, "carried_from_v3": carried, "rejection_sft_domains": dict(byd),
                "dpo_pair_types": dict(pairs), "elapsed_seconds": round(time.time() - started, 1)}
    V.write_json(TRAIN_V4 / "offline_training_manifest.json", manifest)
    ok = byd.get("math", 0) >= 5000 and byd.get("coding", 0) >= 2000 and byd.get("logic", 0) >= 10000
    verdict = "V4_MULTIDOMAIN_WORKED_READY" if ok else "V4_VIEWS_WEAK"
    V.set_stage("FG_offline_views_v4", verdict, manifest)
    V.prog("FG_offline_views_v4", {"verdict": verdict, **manifest})
    V.write_md(V.OUT_ROOT / "worked_math_coding_views_v4.md", [
        "# Worked Math/Coding Views v4 (Round 3 data)", "",
        V.status_line("OFFLINE_VIEWS_V4_VERDICT", verdict),
        f"Counts: {json.dumps(counts)} | rejection_sft domains: {json.dumps(dict(byd))}",
        f"DPO pair types: {json.dumps(dict(pairs))}",
        "v3 executed-logic views carried forward; math worked solutions from GSM8K/Hendrycks dataset "
        "rationales (gold finals from #### / \\boxed{}); coding from canonical solutions + unit-test-labeled "
        "model pools; show_work pairs penalize the bare-answer degeneration format directly. No budget / "
        "direct-answer views (budget-as-text-style retired per round2_executed_data_result.md).",
    ])
    print(V.status_line("OFFLINE_VIEWS_V4_VERDICT", verdict))
    print(f"  {counts}")
    print(f"  domains {dict(byd)} | pairs {dict(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
