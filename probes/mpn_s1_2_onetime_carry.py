"""M+N S1.4 step one — REAL one-time-per-locus + KV-carry branch baseline (vs sampling).

Earlier injection runs were SUSTAINED (hook fires every decode step). The architecture's real
branch is ONE-TIME: perturb once at a (loop,layer) boundary during prefill, bake into the KV
cache, continue decoding from that carried cache. `build_spliced_branch` (bit-exact validated v2
splice) does exactly that; `greedy_continue` decodes from the spliced branch cache. This measures
the REAL mechanism (single locus L1_24 for now) vs plain sampling, on the validated cache model:
  - SAMP : K plain samples (temp 0.8)
  - OTC  : K one-time-carry branches (curated L24 directions, alpha) — perturb@L1_24 + carry
Same BT parser + external verifier. Per task: distinct verified finals + oracle@K. Addresses the
"one-time vs sustained" confound on the real carry mechanism. (Multi-locus chaining + prune = step two.)

Run on GPU (backgrounded): venv/bin/python utilities/tests/manual/mpn_s1_2_onetime_carry.py
"""
from __future__ import annotations
import json
import os
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
import bg_autoregressive_cache_common_v1 as C  # noqa: E402
import bg_partial_cache_splice_v2_common as S  # noqa: E402
from mpn_s1_2_multilocus import curated_dirs_by_layer  # noqa: E402  (per-layer curated dirs)

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")
N_PER = int(os.environ.get("S1_N_PER", "2"))
K = int(os.environ.get("S1_K", "4"))
ALPHA = float(os.environ.get("S1_ALPHA", "0.02"))
BL, BLAYER = 0, 24  # boundary loop index 0 == loop 1; layer 24


def parse(task, text):
    return B1.extract_code(text) if task.get("label_type") == "unit_tests" else (B1.parse_final(text) or "")


def metrics(task, texts, is_code):
    finals, correct = [], []
    for t in texts:
        p = parse(task, t); finals.append((p or "")[:80]); correct.append(
            B1.verify_code_unit_tests(p, task.get("unit_tests", []), task.get("test_setup", "")) if is_code
            else B1.verify_final(task, p)[1] > 0)
    return {"distinct": len(set(finals)), "oracle": bool(any(correct)), "n_correct": int(sum(correct))}


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]
    dirs24 = curated_dirs_by_layer(K)[24]

    model, tok, info = C.load_model(attn_implementation="eager")
    device = next(model.parameters()).device

    rows = []
    for task in tasks:
        d = task["domain"]; is_code = task.get("label_type") == "unit_tests"
        mnt = min(V.DOMAIN_MAXTOK.get(d, 320), 768)
        prompt = V.build_prompt(tok, task, "direct")
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
        ids, am = enc["input_ids"], enc["attention_mask"]
        plen = ids.shape[1]
        # SAMP arm
        samp_texts = []
        with torch.inference_mode():
            for _ in range(K):
                o = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=mnt, do_sample=True,
                                   temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
                samp_texts.append(tok.decode(o[0][plen:], skip_special_tokens=True))
        # OTC arm: one-time-carry branches, amortized root
        tr = S.default_token_range(plen)
        root_pack = S.root_prefill_with_boundary(model, ids, am, BL, BLAYER)
        otc_texts = []
        for j in range(K):
            br = S.build_spliced_branch(model, ids, am, BL, BLAYER, ALPHA, dirs24[j % len(dirs24)], tr,
                                        root_pack=root_pack)
            toks, _ = S.greedy_continue(model, br["spliced_cache"], br["am_full"], br["next_logits"], mnt, device)
            otc_texts.append(tok.decode(toks, skip_special_tokens=True))
        rec = {"task": task["canary_id"], "domain": d,
               "samp": metrics(task, samp_texts, is_code), "otc": metrics(task, otc_texts, is_code)}
        rows.append(rec)
        print(f"  {d:9} {task['canary_id']}: SAMP(d={rec['samp']['distinct']},o={rec['samp']['oracle']}) "
              f"OTC(d={rec['otc']['distinct']},o={rec['otc']['oracle']})", flush=True)
        C.clear_cuda()

    def agg(arm):
        n = len(rows)
        return {"mean_distinct": round(sum(r[arm]["distinct"] for r in rows) / n, 3),
                "oracle_at_K": round(sum(1 for r in rows if r[arm]["oracle"]) / n, 3)}
    payload = {"verdict": "S1_4_ONETIME_CARRY", "N_PER": N_PER, "K": K, "alpha": ALPHA,
               "boundary": f"loop{BL+1}_L{BLAYER}", "n_tasks": len(rows),
               "sampling": agg("samp"), "one_time_carry": agg("otc"),
               "oracle_gap_samp_minus_otc": round(agg("samp")["oracle_at_K"] - agg("otc")["oracle_at_K"], 3),
               "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4_onetime_carry.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print("\nS1_4_ONETIME_CARRY — real one-time-carry mechanism vs sampling (single locus L1_24)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
