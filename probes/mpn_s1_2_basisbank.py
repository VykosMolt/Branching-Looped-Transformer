"""M+N S1.2 — fair distinctness baseline using the CURATED basis-bank directions (not random).

The random-direction sweep gave 0 distinctness (frozen model is contractive). But the real
generator injects curated directions (high_yield_recipe / tap_aligned), which the hidden-origin
work found gave ~25% behavioral diversity. This runs the strongest external method: inject the
top-K curated L24 directions (rms-normalized tensor * alpha) at L24/L1 and measure MCQ
outcome-distinctness at safe (0.02) and diagnostic (0.10) alpha. This is the fair pre-training
baseline that S3 must beat.

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_2_basisbank.py
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
import mpn_s1_2_mcq_distinct as MCQ  # noqa: E402
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, rms_normalize  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
BANK = V.PROJECT_ROOT / "artifacts/reports/probes/bg_hidden_origin_branch_generator_v1_2026-05-18/basis_bank_v1.pt"
ALPHAS = [0.02, 0.10]
K = int(os.environ.get("S1_K", "8"))
PRIORITY = ["high_yield_recipe_direction", "v4_tap_aligned", "v2_tap_aligned", "v1_tap_aligned",
            "salvage_tap_aligned", "old_tap_aligned"]


def curated_l24_directions(k: int) -> list[tuple[str, torch.Tensor]]:
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    by_fam: dict[str, list] = {}
    for d in bank["directions"]:
        tl = d.get("target_layer") if isinstance(d, dict) else None
        if tl is None or int(tl) != 24:
            continue
        if not d.get("perturbation_usable") or d.get("family") == "random_orthogonal":
            continue
        t = d.get("tensor")
        if isinstance(t, torch.Tensor) and t.numel() == 2048:
            by_fam.setdefault(d["family"], []).append((d["name"], rms_normalize(t.float().flatten())))
    out = []
    for fam in PRIORITY:
        for name, vec in by_fam.get(fam, []):
            out.append((name, vec))
            if len(out) >= k:
                return out
    return out


def main() -> int:
    started = time.time()
    dirs = curated_l24_directions(K)
    if not dirs:
        print("NO curated L24 directions found"); return 3
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

    T = []
    for task in tasks:
        prompt, letters = MCQ._mcq_prompt(task)
        gold = chr(65 + int(task["answer_key"])) if int(task["answer_key"]) < len(letters) else None
        enc = {k: v.to("cuda") for k, v in tok(prompt, return_tensors="pt", truncation=True, max_length=1536).items()}
        lid = MCQ._letter_ids(tok, letters)
        base, _ = MCQ._pred_letter(tok, model, enc, letters, lid)
        T.append({"enc": enc, "letters": letters, "lid": lid, "gold": gold, "base": base})
    base_acc = round(sum(1 for t in T if t["base"] == t["gold"]) / len(T), 3)

    fams = sorted({n.split(":")[0] for n, _ in dirs})
    results = []
    for alpha in ALPHAS:
        mrf = min(alpha, 0.10)
        distinct, flips, oracle, tot = [], 0, 0, 0
        for t in T:
            preds = []
            for _name, vec in dirs:
                h = HiddenDeltaLayerHook(model, target_layer=24, target_loops=[1],
                                         delta=vec * alpha, position=-1, max_rms_fraction=mrf)
                try:
                    h.apply(); p, _ = MCQ._pred_letter(tok, model, t["enc"], t["letters"], t["lid"])
                finally:
                    h.remove()
                preds.append(p); flips += int(p != t["base"]); tot += 1
            allp = [t["base"]] + preds
            distinct.append(len(set(allp)))
            oracle += int(t["gold"] in allp) if t["gold"] else 0
        n = len(T)
        results.append({"alpha": alpha, "mean_distinct": round(sum(distinct) / n, 3),
                        "distinct_rate": round(sum(1 for c in distinct if c > 1) / n, 3),
                        "branch_flip_rate": round(flips / max(tot, 1), 3),
                        "positive_oracle": round(oracle / n, 3)})
        print(f"  curated a={alpha}: distinct={results[-1]['mean_distinct']:.2f} "
              f"flip={results[-1]['branch_flip_rate']:.2f} oracle={results[-1]['positive_oracle']:.3f}", flush=True)

    payload = {"verdict": "S1_2_CURATED_BASELINE", "K": K, "n_tasks": len(T), "n_dirs": len(dirs),
               "families_used": fams, "base_accuracy": base_acc, "max_distinct": K + 1,
               "results": results, "random_baseline_distinct": 1.0,
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_basisbank_baseline.json").write_text(json.dumps(payload, indent=2))
    print(f"\nfamilies: {fams} | n_dirs {len(dirs)} | base_acc {base_acc}")
    print("vs random-direction baseline: mean_distinct 1.0 (zero). S1_2_CURATED_BASELINE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
