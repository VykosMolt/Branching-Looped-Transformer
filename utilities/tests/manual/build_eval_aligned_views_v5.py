"""ROUND 4 data build — train_v5: train/eval FORMAT ALIGNMENT (CPU-only).

Rounds 1-3 (data_flaw_meta_branch_rendering.md, round2_executed_data_result.md,
round3_multidomain_result.md) repaired CONTENT domain by domain (logic executed renders
-> 0.000; math rationales -0.53 -> -0.13; canonical-code coding BACKFIRED to -0.467) but
never fixed the FORMAT channel: every round trained bare task prompts under the tokenizer's
default system line ("You are a helpful assistant."), with multi-branch
"Branch 1/2/3 ... FINAL ANSWER" completions, while the canary AND the real pool generator
render B1.build_prompt(tok, task, scaffold) — GEN_SYSTEM posture, chat template, and one
single-solution generation per (scaffold, sample).

v5 renders every row exactly as generation sees it:
1. prompt = B1.build_prompt(tok, task, scaffold): full chat-templated string incl.
   GEN_SYSTEM / GEN_SYSTEM_MATH / GEN_SYSTEM_CODE and the MBPP function-name note
   (unit tests rejoined by task_id). Gen-pool rows use the branch's OWN scaffold
   (inverse SCAFFOLD_FMT); dataset rationales use "direct".
2. completion = ONE single solution, no "Branch i:" prefixes:
   - logic: executed pass derivation + FINAL ANSWER (train_v3 rendered sets);
   - math: GSM8K/Hendrycks rationales with <<..>> calculator markup stripped +
     'FINAL ANSWER: <gold>', plus gen-pool pass solutions;
   - coding: gen-pool unit-test-verified REASONED code (deliberation + fenced ```python)
     VERBATIM — no appended FINAL ANSWER line (breaks extract_code's def-to-end fallback).
3. DPO single-solution pairs: oracle (pass vs fail, same group), wrong_final (work intact,
   corrupted final), reasoning_error (corrupted mid-work number + matching wrong final),
   show_work (worked solution vs bare 'FINAL ANSWER: <gold>').
4. Hygiene: canary-prompt exclusion (normalized-prompt hash), token gates (SFT
   prompt+completion <= 1000 of the trainer's 1024 so FINAL ANSWER never truncates; DPO
   prompt <= 640 / total <= 1020 matching DPOConfig), dedupe, fixed seed.

Reasoning is EXCLUDED: gen shards did not persist MCQ options, so those prompts cannot be
re-rendered faithfully (and reasoning never regressed). v3/v4 rows are NOT carried forward —
they are the misaligned format. The trainer consumes pre-rendered prompts via the _fmt patch
(prompt containing '<|im_start|>' is used verbatim); trl DPO already concatenates raw
prompt+completion, so DPO rows align with no trainer change.

Output: data/.../train_v5/{branch_set_rejection_sft,branch_set_dpo,one_branch_sft}.jsonl
(+ manifest), stage FG_offline_views_v5. View names kept for trainer compatibility; rows are
single-solution.
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

import branch_training_v1_common as B1  # noqa: E402
import branch_training_v2_common as V  # noqa: E402

os.environ.setdefault("HF_HOME", str(V.PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(V.PROJECT_ROOT / "shared/hf_cache/datasets"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

TRAIN_V5 = V.DATA_ROOT / "train_v5"
RLOG = V.DATA_ROOT / "train_v3/rendered_logic_branch_sets.jsonl"
GEN_GLOB = "shared/data/branch_training_logic_expansion_v1/processed/gen_shards/*.jsonl"
MAX_GSM = int(os.environ.get("V5_MAX_GSM", "6000"))
MAX_HEND = int(os.environ.get("V5_MAX_HEND", "3000"))
MAX_LOGIC_SFT = int(os.environ.get("V5_MAX_LOGIC_SFT", "12000"))
MAX_LOGIC_DPO = int(os.environ.get("V5_MAX_LOGIC_DPO", "12000"))
MAX_SHOW_WORK = int(os.environ.get("V5_MAX_SHOW_WORK", "4000"))
MAX_WRONG_FINAL = int(os.environ.get("V5_MAX_WRONG_FINAL", "6000"))
MAX_REASONING_ERR = int(os.environ.get("V5_MAX_REASONING_ERR", "6000"))
CODING_DUP = int(os.environ.get("V5_CODING_DUP", "12"))
MIN_WORK = 80
MIN_WORK_LOGIC = 60
WORK_CAP = 2400
SFT_TOK_BUDGET = 1000   # trainer SFT max_length=1024; margin so FINAL ANSWER is never cut
DPO_PROMPT_TOK = 640    # DPOConfig max_prompt_length (keep-end truncation otherwise)
DPO_TOTAL_TOK = 1020    # DPOConfig max_length=1024 minus EOS margin
HEND_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")
INV_SCAFFOLD = {v: k for k, v in B1.SCAFFOLD_FMT.items()}
rng = random.Random(57)

VIEWS = ("branch_set_rejection_sft", "branch_set_dpo", "one_branch_sft")
_CALC_RE = re.compile(r"<<[^>]*>>")


def _norm_prompt_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _canary_keys() -> set[str]:
    return {_norm_prompt_key(json.loads(l).get("task_prompt"))
            for l in open(V.CANARY_DIR / "offline_branch_generator_canary_v2.jsonl")}


def _with_final(text: str, final) -> str:
    if "FINAL ANSWER" in text:
        return text
    return f"{text}\nFINAL ANSWER: {final}"


def _corrupt_number(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    if not m:
        return None
    try:
        new = str(int(float(m.group(0))) + rng.choice((1, -1, 2, 10)))
    except Exception:
        return None
    return str(s)[: m.start()] + new + str(s)[m.end():]


def _corrupt_work(work: str):
    """Reasoning-error negative: perturb one numeric token in the latter half of the work."""
    hits = [m for m in re.finditer(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])", work)]
    hits = [m for m in hits if m.start() > len(work) // 3]
    if not hits:
        return None
    m = rng.choice(hits)
    try:
        new = str(int(float(m.group(0))) + rng.choice((1, -1, 2)))
    except Exception:
        return None
    return work[: m.start()] + new + work[m.end():]


def _boxed(solution: str):
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
        body = _CALC_RE.sub("", body).strip()  # strip <<48/2=24>> calculator markup (round-4 fix)
        final = final.strip()
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
            print(f"[v5] hendrycks {subj} unavailable: {e}", flush=True)
            continue
        n = 0
        for r in ds:
            final = _boxed(r["solution"])
            work = r["solution"].strip()
            if not final or len(work) < MIN_WORK:
                continue
            rows.append({"prompt": r["problem"].strip(), "work": work, "final": final,
                         "src": f"hendrycks_{subj}"})
            n += 1
            if n >= per:
                break
    return rows


def gen_pool_groups():
    import glob
    out = []
    for s in sorted(glob.glob(str(V.PROJECT_ROOT / GEN_GLOB))):
        for l in open(s):
            d = json.loads(l)
            if str(d.get("task_prompt") or "").strip():
                out.append(d)
    return out


def mbpp_tests_by_uid() -> dict[str, dict]:
    """Rejoin unit tests to gen-pool coding groups (tests were not persisted in the shards)."""
    return {t["task_uid"]: t for t in V.load_coding_gen_tasks(700)}


def main() -> int:
    started = time.time()
    TRAIN_V5.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    if not getattr(tok, "chat_template", None):
        raise RuntimeError("tokenizer has no chat template; v5 requires the eval-side rendering path")
    canary_keys = _canary_keys()
    skipped = collections.Counter()
    views = {k: [] for k in VIEWS}

    def add_sft(prompt, completion, domain, src, dup=1):
        row = {"prompt": prompt, "completion": completion, "domain": domain, "src": src}
        views["one_branch_sft"].append(row)
        for _ in range(max(1, dup)):
            views["branch_set_rejection_sft"].append(dict(row))

    def add_dpo(prompt, chosen, rejected, domain, pair, dup=1):
        if chosen.strip() == rejected.strip():
            skipped["dpo_identical"] += 1
            return
        for _ in range(max(1, dup)):
            views["branch_set_dpo"].append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                                            "domain": domain, "pair": pair})

    # ---------- logic: executed derivations from the v3 renderer (gold-match labels) ----------
    logic_groups = []
    for line in open(RLOG):
        g = json.loads(line)
        if _norm_prompt_key(g.get("task_prompt")) in canary_keys:
            skipped["logic_canary_overlap"] += 1
            continue
        logic_groups.append(g)
    rng.shuffle(logic_groups)
    n_logic_sft = n_logic_dpo = 0
    for g in logic_groups:
        ba = g.get("branch_attempts") or []
        pos = [b for b in ba if b.get("external_label") == "pass"
               and len((b.get("branch_text") or "").strip()) >= MIN_WORK_LOGIC]
        neg = [b for b in ba if b.get("external_label") == "fail"
               and len((b.get("branch_text") or "").strip()) >= MIN_WORK_LOGIC]
        if not pos:
            continue
        task = {"task_prompt": g["task_prompt"], "options": g.get("options"), "label_type": None}
        prompt = B1.build_prompt(tok, task, "direct")
        gold = pos[0].get("final_answer")
        chosen = _with_final(pos[0]["branch_text"].strip()[:WORK_CAP], gold)
        if n_logic_sft < MAX_LOGIC_SFT:
            add_sft(prompt, chosen, "logic", "rendered_v3")
            n_logic_sft += 1
        if neg and n_logic_dpo < MAX_LOGIC_DPO:
            rej = _with_final(neg[0]["branch_text"].strip()[:WORK_CAP],
                              neg[0].get("final_answer", "Unknown"))
            add_dpo(prompt, chosen, rej, "logic", "oracle")
            n_logic_dpo += 1
        if n_logic_sft >= MAX_LOGIC_SFT and n_logic_dpo >= MAX_LOGIC_DPO:
            break
    print(f"[v5] logic: sft {n_logic_sft} | dpo {n_logic_dpo} "
          f"(canary-overlap excluded {skipped['logic_canary_overlap']})", flush=True)

    # ---------- math: dataset rationales (direct scaffold) ----------
    math_rows = gsm8k_rows() + hendrycks_rows()
    show_work_budget, wrong_final_budget, reasoning_err_budget = MAX_SHOW_WORK, MAX_WRONG_FINAL, MAX_REASONING_ERR
    n_math = 0
    for r in math_rows:
        if _norm_prompt_key(r["prompt"]) in canary_keys:
            skipped["math_canary_overlap"] += 1
            continue
        task = {"task_prompt": r["prompt"], "label_type": "exact_answer"}
        prompt = B1.build_prompt(tok, task, "direct")
        work = r["work"][:WORK_CAP]
        chosen = f"{work}\nFINAL ANSWER: {r['final']}"
        add_sft(prompt, chosen, "math", r["src"])
        n_math += 1
        if show_work_budget > 0:
            add_dpo(prompt, chosen, f"FINAL ANSWER: {r['final']}", "math", "show_work")
            show_work_budget -= 1
        if wrong_final_budget > 0:
            bad_final = _corrupt_number(r["final"])
            if bad_final:
                add_dpo(prompt, chosen, f"{work}\nFINAL ANSWER: {bad_final}", "math", "wrong_final")
                wrong_final_budget -= 1
        if reasoning_err_budget > 0:
            bad_work = _corrupt_work(work)
            bad_final = _corrupt_number(r["final"])
            if bad_work and bad_final:
                add_dpo(prompt, chosen, f"{bad_work}\nFINAL ANSWER: {bad_final}", "math", "reasoning_error")
                reasoning_err_budget -= 1
    print(f"[v5] math rationales: sft {n_math}", flush=True)

    # ---------- gen pools: real model generations under their OWN scaffold ----------
    coding_join = mbpp_tests_by_uid()
    n_code_sft = n_code_dpo = n_genmath = 0
    for g in gen_pool_groups():
        dom = g.get("domain")
        if dom not in ("math", "coding"):
            continue  # reasoning excluded: options not persisted -> prompt not re-renderable
        if _norm_prompt_key(g.get("task_prompt")) in canary_keys:
            skipped["genpool_canary_overlap"] += 1
            continue
        ba = g.get("branch_attempts") or []
        pos = [b for b in ba if b.get("external_label") == "pass"
               and len((b.get("branch_text") or "").strip()) >= MIN_WORK]
        neg = [b for b in ba if b.get("external_label") == "fail"
               and len((b.get("branch_text") or "").strip()) >= MIN_WORK]
        if not pos:
            continue
        if dom == "coding":
            join = coding_join.get(str(g.get("task_id") or ""))
            if join is None:
                skipped["coding_no_mbpp_join"] += 1
                continue
            task = {"task_prompt": g["task_prompt"], "label_type": "unit_tests",
                    "unit_tests": join["unit_tests"]}
            prompt = B1.build_prompt(tok, task, "direct")  # scaffold ignored on unit_tests path
            for b in pos:
                text = b["branch_text"].strip()
                if "```" not in text or "FINAL ANSWER" in text:
                    skipped["coding_bad_shape"] += 1
                    continue
                add_sft(prompt, text, "coding", "gen_pool", dup=CODING_DUP)
                n_code_sft += 1
            if neg:
                good, bad = pos[0]["branch_text"].strip(), neg[0]["branch_text"].strip()
                if "```" in good:
                    add_dpo(prompt, good, bad, "coding", "oracle", dup=CODING_DUP)
                    n_code_dpo += 1
        else:
            task = {"task_prompt": g["task_prompt"], "label_type": "exact_answer"}
            for b in pos:
                scaffold = INV_SCAFFOLD.get(b.get("branch_format"), "direct")
                prompt = B1.build_prompt(tok, task, scaffold)
                add_sft(prompt, _with_final(b["branch_text"].strip()[:WORK_CAP], b.get("final_answer")),
                        "math", "gen_pool")
                n_genmath += 1
            if neg:
                b = pos[0]
                prompt = B1.build_prompt(tok, task, INV_SCAFFOLD.get(b.get("branch_format"), "direct"))
                add_dpo(prompt, _with_final(b["branch_text"].strip()[:WORK_CAP], b.get("final_answer")),
                        _with_final(neg[0]["branch_text"].strip()[:WORK_CAP],
                                    neg[0].get("final_answer", "Unknown")),
                        "math", "oracle")
    print(f"[v5] gen pools: coding sft {n_code_sft} (x{CODING_DUP} dup) dpo {n_code_dpo} | "
          f"math sft {n_genmath} | skipped {dict(skipped)}", flush=True)

    # ---------- token-length gates (vectorized; SFT 1000 / DPO 640+1020) ----------
    cache: dict[str, int] = {}

    def _tok_lens(texts: list[str]) -> list[int]:
        out = []
        for i in range(0, len(texts), 512):
            out += [len(x) for x in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
        return out

    def _lens_for(rows, fields):
        pending = []
        for r in rows:
            for f in fields:
                t = r.get(f)
                if t is not None and t not in cache:
                    cache[t] = -1
                    pending.append(t)
        for t, n in zip(pending, _tok_lens(pending)):
            cache[t] = n

    _lens_for(views["one_branch_sft"], ("prompt", "completion"))
    _lens_for(views["branch_set_rejection_sft"], ("prompt", "completion"))
    _lens_for(views["branch_set_dpo"], ("prompt", "chosen", "rejected"))

    def _sft_ok(r):
        return cache[r["prompt"]] + cache[r["completion"]] <= SFT_TOK_BUDGET

    def _dpo_ok(r):
        return (cache[r["prompt"]] <= DPO_PROMPT_TOK
                and cache[r["prompt"]] + max(cache[r["chosen"]], cache[r["rejected"]]) <= DPO_TOTAL_TOK)

    drops = {}
    for name, keep in (("one_branch_sft", _sft_ok), ("branch_set_rejection_sft", _sft_ok),
                       ("branch_set_dpo", _dpo_ok)):
        before = len(views[name])
        views[name] = [r for r in views[name] if keep(r)]
        drops[name] = before - len(views[name])
    print(f"[v5] token-gate drops: {drops}", flush=True)

    # ---------- dedupe (dup rows for coding survive: dedupe only one_branch_sft) ----------
    counts = {}
    for name, rows in views.items():
        if name == "one_branch_sft":
            seen, uniq = set(), []
            for r in rows:
                k = (r["prompt"][:200], r["completion"][:160])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(r)
            rows = uniq
        rng.shuffle(rows)
        with open(TRAIN_V5 / f"{name}.jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps(r, default=V.v2.json_default) + "\n")
        counts[name] = len(rows)
        views[name] = rows

    # ---------- format spot-checks (the actual round-4 contract) ----------
    sample = rng.sample(views["branch_set_rejection_sft"], min(200, len(views["branch_set_rejection_sft"])))
    fence = "```"
    contract = {
        "prompts_chat_templated": all(r["prompt"].startswith("<|im_start|>system") for r in sample),
        "prompts_have_gen_system": all("compact looped reasoning model" in r["prompt"] for r in sample),
        "prompts_end_with_assistant": all(r["prompt"].rstrip().endswith("<|im_start|>assistant") for r in sample),
        "no_branch_prefix_completions": all(not r["completion"].lstrip().startswith("Branch ") for r in sample),
        "noncoding_end_with_final": all("FINAL ANSWER" in r["completion"]
                                        for r in sample if r["domain"] != "coding"),
        "coding_fenced_no_final": all(fence in r["completion"] and "FINAL ANSWER" not in r["completion"]
                                      for r in sample if r["domain"] == "coding"),
    }
    byd = collections.Counter(r["domain"] for r in views["branch_set_rejection_sft"])
    pairs = collections.Counter(r["pair"] for r in views["branch_set_dpo"])
    uniq_coding = sum(1 for r in views["one_branch_sft"] if r["domain"] == "coding")
    manifest = {**counts, "rejection_sft_domains": dict(byd), "dpo_pair_types": dict(pairs),
                "unique_coding_sft": uniq_coding, "coding_dup": CODING_DUP,
                "token_gate_drops": drops, "skipped": dict(skipped), "format_contract": contract,
                "reasoning_excluded": "gen shards did not persist MCQ options; prompt not re-renderable",
                "v3_v4_not_carried": "old rows are the misaligned format round 4 removes",
                "elapsed_seconds": round(time.time() - started, 1)}
    V.write_json(TRAIN_V5 / "offline_training_manifest.json", manifest)
    ok = (all(contract.values()) and byd.get("logic", 0) >= 8000 and byd.get("math", 0) >= 5000
          and uniq_coding >= 60 and counts["branch_set_dpo"] >= 15000)
    verdict = "V5_EVAL_ALIGNED_READY" if ok else "V5_VIEWS_WEAK"
    V.set_stage("FG_offline_views_v5", verdict, manifest)
    V.prog("FG_offline_views_v5", {"verdict": verdict, **manifest})
    V.write_md(V.OUT_ROOT / "eval_aligned_views_v5.md", [
        "# Eval-Aligned Views v5 (Round 4 data)", "",
        V.status_line("OFFLINE_VIEWS_V5_VERDICT", verdict),
        f"Counts: {json.dumps(counts)} | rejection_sft domains: {json.dumps(dict(byd))}",
        f"DPO pair types: {json.dumps(dict(pairs))} | unique coding sft {uniq_coding} (dup x{CODING_DUP})",
        f"Format contract: {json.dumps(contract)} | token-gate drops {json.dumps(drops)} | "
        f"skipped {json.dumps(dict(skipped))}",
        "Round-4 fix: every prompt is the exact eval-side rendering (B1.build_prompt: GEN_SYSTEM posture, "
        "chat template, MBPP fn-note via unit-test rejoin; gen-pool rows under their own scaffold), every "
        "completion is ONE single solution (executed logic derivation / <<>>-stripped rationale + FINAL "
        "ANSWER / verbatim verified reasoned code, fenced, no FINAL ANSWER line). DPO adds reasoning_error "
        "negatives (corrupted mid-work number + wrong final). Reasoning excluded (options not persisted); "
        "v3/v4 rows not carried (misaligned format). Trainer must include the pre-rendered _fmt patch.",
    ])
    print(V.status_line("OFFLINE_VIEWS_V5_VERDICT", verdict))
    print(f"  {counts}")
    print(f"  domains {dict(byd)} | pairs {dict(pairs)} | contract {contract}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
