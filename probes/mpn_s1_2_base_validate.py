"""M+N S1.2 step one — base-accuracy validation + parser A/B (NO verification).

Re-establishes that the answering harness reproduces the model's known capability, BEFORE any
branch claim. Generates K samples per task with the validated evaluator harness (build_prompt +
GEN_SYSTEM + DOMAIN_MAXTOK + _B1_make_stop, do_sample=True temp 0.8 top_p 0.95 — same as
_gen_branches / the reachability run), keeps the RAW text, and parses it TWO ways:
  - BT  : branch_training parse_final (first FINAL ANSWER) / extract_code,
  - LA  : local-agent pure parser (normalize_model_output + extract_final_answer = LAST FINAL
          ANSWER; _extract_python_code_block) — REPLICATED here (pure regex, no oracle/verify).
Each parsed answer is labeled by the EXTERNAL verifier only (verify_final / verify_code_unit_tests).
Reports mean pass@1 (#correct/K) and oracle@K per domain per parser; adopts the better parser.
Target: math ~0.83 oracle@K, reasoning ~0.95 (reachability run). No self-verify, no oracle injection.

Run on GPU (backgrounded): venv/bin/python utilities/tests/manual/mpn_s1_2_base_validate.py
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v1_common as B1  # noqa: E402
import branch_training_v2_common as V  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")
N_PER = int(os.environ.get("S1_N_PER", "4"))
K = int(os.environ.get("S1_K", "4"))


# ---- local-agent PURE parser, replicated verbatim from ouro_policies.py / ouro_oracles.py ----
def _strip_think(t):
    t = t or ""
    t = re.sub(r"(?is)<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>", "", t).strip()
    if re.search(r"(?is)<think(?:ing)?\b", t):
        t = re.sub(r"(?is)<think(?:ing)?\b[^>]*>.*$", "", t).strip()
    return t.strip()


def _normalize(t):
    t = _strip_think(t).strip()
    t = re.sub(r"(?is)<\|im_start\|>\s*(?:assistant|user|system)?\s*", "", t).strip()
    t = re.sub(r"(?is)<\|im_end\|>.*$", "", t).strip()
    t = re.sub(r"(?is)</?s>", "", t).strip()
    for pat in (r"(?im)^\s*final\s*:\s*", r"(?im)^\s*answer\s*:\s*",
                r"(?im)^\s*(?:assistant|istant)\s*:\s*", r"(?im)\n\s*(?:assistant|istant)\s*$"):
        t = re.sub(pat, "", t).strip()
    return t


def la_extract_final(text):
    c = _normalize(text)
    up = c.upper()
    if "FINAL ANSWER:" in up:
        return c[up.rfind("FINAL ANSWER:") + len("FINAL ANSWER:"):].strip()
    return c.strip()


def la_extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m[0].strip()
    lines = (text or "").splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*(?:from\s+[A-Za-z_][\w.]*\s+import\s+|import\s+[A-Za-z_][\w.]*|def\s+\w+\s*\(|class\s+\w+\s*[:(])", ln):
            return "\n".join(lines[i:]).strip()
    mm = re.search(r"(?ms)^\s*def\s+\w+\s*\(.*", text or "")
    return mm.group(0).strip() if mm else ""


def gen_raw(model, tok, task, k):
    from transformers import StoppingCriteriaList
    d = task["domain"]
    mnt = V.DOMAIN_MAXTOK.get(d, 320)
    prompt = V.build_prompt(tok, task, "direct")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
    plen = enc["input_ids"].shape[1]
    texts, remaining = [], k
    chunk = 1 if mnt >= 1000 else (2 if mnt >= 600 else 4)
    guard = 0
    while remaining > 0 and guard < k + 6:
        guard += 1
        nb = min(chunk, remaining)
        try:
            stp = StoppingCriteriaList([V._B1_make_stop(tok, plen)])
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=mnt, do_sample=True, temperature=0.8, top_p=0.95,
                                     num_return_sequences=nb, pad_token_id=tok.pad_token_id, stopping_criteria=stp)
            for i in range(out.shape[0]):
                texts.append(tok.decode(out[i][plen:], skip_special_tokens=True))  # RAW (no trim)
            del out; remaining -= nb; torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower() and chunk > 1:
                chunk = max(1, chunk // 2); continue
            remaining -= 1
    return texts


def verify(task, parsed, is_code):
    if is_code:
        return B1.verify_code_unit_tests(parsed, task.get("unit_tests", []), task.get("test_setup", ""))
    _l, r = B1.verify_final(task, parsed)
    return r > 0


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(V.BASE_MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for task in tasks:
        d = task["domain"]; is_code = task.get("label_type") == "unit_tests"
        texts = gen_raw(model, tok, task, K)
        rec = {"task": task["canary_id"], "domain": d, "n": len(texts)}
        for tag, pf, cf in (("bt", B1.parse_final, B1.extract_code), ("la", la_extract_final, la_extract_code)):
            correct = []
            for t in texts:
                parsed = cf(t) if is_code else (pf(t) or "")
                correct.append(verify(task, parsed, is_code))
            rec[f"{tag}_pass1"] = round(sum(correct) / len(correct), 3) if correct else 0.0
            rec[f"{tag}_oracle"] = bool(any(correct))
        rows.append(rec)
        print(f"  {d:9} {task['canary_id']}: bt(p1={rec['bt_pass1']},orc={rec['bt_oracle']}) "
              f"la(p1={rec['la_pass1']},orc={rec['la_oracle']})", flush=True)

    def agg(tag):
        out = {}
        for d in DOMAINS:
            dr = [r for r in rows if r["domain"] == d]
            if dr:
                out[d] = {"pass1": round(sum(r[f"{tag}_pass1"] for r in dr) / len(dr), 3),
                          "oracle_at_K": round(sum(1 for r in dr if r[f"{tag}_oracle"]) / len(dr), 3), "n": len(dr)}
        return out
    bt, la = agg("bt"), agg("la")
    bt_macro = round(sum(v["oracle_at_K"] for v in bt.values()) / len(bt), 3) if bt else 0
    la_macro = round(sum(v["oracle_at_K"] for v in la.values()) / len(la), 3) if la else 0
    better = "la" if la_macro > bt_macro else ("bt" if bt_macro > la_macro else "tie")
    payload = {"verdict": "S1_2_BASE_VALIDATED", "N_PER": N_PER, "K": K, "n_tasks": len(rows),
               "bt_by_domain": bt, "la_by_domain": la, "bt_oracle_macro": bt_macro, "la_oracle_macro": la_macro,
               "better_parser": better, "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_base_validate.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(f"\nbetter_parser={better} | bt_oracle_macro={bt_macro} la_oracle_macro={la_macro}")
    print("target ~ math 0.83 / reasoning 0.95 oracle@K (reachability run). S1_2_BASE_VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
