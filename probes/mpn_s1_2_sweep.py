"""M+N S1.2 — alpha/loci sweep mapping the distinctness-vs-stability frontier (MCQ cheap path).

For each (alpha, layer, loop) config, inject K branches and read each branch's MCQ outcome
(single-forward option-lean). Reports, aggregated over the MCQ tasks:
  - mean_distinct_outcomes / distinct_rate : do branches give DIFFERENT answers (vs ties),
  - branch_flip_rate                        : fraction of injected branches whose answer != base,
  - base_accuracy (const) & positive_oracle : does distinctness come WITH correctness (oracle>base)
                                              or is it just noise (flips to wrong answers)?
alpha 0.01/0.02 = safe envelope; 0.05/0.10 = diagnostic (allow_diagnostic_alpha), capped at 0.10.
Earlier loci (L1_24) have more downstream loops/layers to amplify a delta than later (L3_24).

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_2_sweep.py
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
import branch_training_v2_common as V  # noqa: E402
import mpn_s1_2_mcq_distinct as MCQ  # noqa: E402  (reuse _mcq_prompt/_letter_ids/_pred_letter)
from src.evaluator.bg_hidden_branching import make_branch_deltas, HiddenDeltaLayerHook, SAFE_ALPHA_CAP  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
ALPHAS = [0.01, 0.02, 0.05, 0.10]
LOCI = [(24, 1), (36, 1), (47, 1), (24, 2), (24, 3)]  # (layer, loop)
K = int(os.environ.get("S1_K", "4"))


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {t["canary_id"] for t in sel["tasks"] if t["domain"] in MCQ.MCQ_DOMAINS}
    tasks = [json.loads(l) for l in open(MCQ.CANARY) if json.loads(l)["canary_id"] in want]
    tasks = [t for t in tasks if t.get("options") and t.get("answer_key") is not None]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(V.BASE_MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = model.config.hidden_size

    # precompute per-task base prediction + encodings
    T = []
    for ti, task in enumerate(tasks):
        prompt, letters = MCQ._mcq_prompt(task)
        gold = chr(65 + int(task["answer_key"])) if int(task["answer_key"]) < len(letters) else None
        enc = {k: v.to("cuda") for k, v in tok(prompt, return_tensors="pt", truncation=True, max_length=1536).items()}
        lid = MCQ._letter_ids(tok, letters)
        base_pred, _ = MCQ._pred_letter(tok, model, enc, letters, lid)
        T.append({"ti": ti, "enc": enc, "letters": letters, "lid": lid, "gold": gold, "base": base_pred})

    base_acc = round(sum(1 for t in T if t["base"] == t["gold"]) / len(T), 3)
    grid = []
    for alpha in ALPHAS:
        allow = alpha > SAFE_ALPHA_CAP
        mrf = min(alpha, 0.10)
        for (layer, loop) in LOCI:
            distinct_counts, flips, oracle = [], 0, 0
            tot_branches = 0
            for t in T:
                deltas = make_branch_deltas(hidden, k=K + 1, alpha=alpha, seed=20260613 + t["ti"] * 17 + layer,
                                            include_clean=True, allow_diagnostic_alpha=allow)
                preds = []
                for d in deltas[1:K + 1]:
                    h = HiddenDeltaLayerHook(model, target_layer=layer, target_loops=[loop],
                                             delta=d, position=-1, max_rms_fraction=mrf)
                    try:
                        h.apply(); p, _ = MCQ._pred_letter(tok, model, t["enc"], t["letters"], t["lid"])
                    finally:
                        h.remove()
                    preds.append(p)
                    flips += int(p != t["base"]); tot_branches += 1
                allp = [t["base"]] + preds
                distinct_counts.append(len(set(allp)))
                oracle += int(t["gold"] in allp) if t["gold"] else 0
            n = len(T)
            grid.append({"alpha": alpha, "layer": layer, "loop": loop, "safe": not allow,
                         "mean_distinct": round(sum(distinct_counts) / n, 3),
                         "distinct_rate": round(sum(1 for c in distinct_counts if c > 1) / n, 3),
                         "branch_flip_rate": round(flips / max(tot_branches, 1), 3),
                         "positive_oracle": round(oracle / n, 3)})
            print(f"  a={alpha:<4} L{layer}_L{loop}: distinct={grid[-1]['mean_distinct']:.2f} "
                  f"flip={grid[-1]['branch_flip_rate']:.2f} oracle={grid[-1]['positive_oracle']:.3f}", flush=True)

    payload = {"verdict": "S1_2_FRONTIER_MAPPED", "K": K, "n_tasks": len(T), "base_accuracy": base_acc,
               "max_distinct": K + 1, "grid": grid, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_alpha_loci_sweep.json").write_text(json.dumps(payload, indent=2))
    print(f"\nbase_accuracy={base_acc} (oracle must EXCEED this for branches to add reachability)")
    print("S1_2_FRONTIER_MAPPED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
