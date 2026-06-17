"""M+N S1.2 (tiny-first) — live hidden-state injection smoke on the RLTT backbone.

Proves the branch-generation primitive works live and safely, on ONE reasoning task, before any
scaling. Reuses the validated injection (`HiddenDeltaLayerHook` + `make_branch_deltas`):
  - base forward (no hook),
  - ZERO-delta hook forward  -> must EQUAL base bit-for-bit (the hook is a verified no-op at a=0),
  - injected forward (+dir at L24/loop1, alpha 0.01, max_rms_fraction 0.02) -> must DIFFER, stay
    finite, and the hook must have fired on loop 1 ONLY.
Reads a cheap MCQ option-lean (single forward, no generation) base-vs-injected for interpretability
— this is the cheap branch-outcome signal relevant to the latent-labeling problem.

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_2_inject_smoke.py
"""
from __future__ import annotations
import json
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
from src.evaluator.bg_hidden_branching import make_branch_deltas, HiddenDeltaLayerHook, delta_rms  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
TASK_SET = OUT / "s1_1_task_set.json"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
TARGET_LAYER, TARGET_LOOP, ALPHA, MAX_RMS_FRAC = 24, 1, 0.01, 0.02


def _last_logits(model, enc) -> torch.Tensor:
    with torch.inference_mode():
        out = model(**enc, use_cache=False)
    lg = out.logits if hasattr(out, "logits") else out[0]
    return lg[0, -1].detach().float().cpu()


def _option_lean(tok, logits: torch.Tensor, options) -> dict:
    if not options:
        return {}
    probs = torch.softmax(logits, dim=-1)
    out = {}
    for j, _opt in enumerate(options):
        letter = chr(65 + j)
        ids = set()
        for t in (letter, " " + letter, letter + ".", " " + letter + ")"):
            enc = tok(t, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(int(enc[0]))
        out[letter] = round(float(sum(probs[i].item() for i in ids if 0 <= i < probs.numel())), 5)
    return out


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads(TASK_SET.read_text())
    rid = next(t["canary_id"] for t in sel["tasks"] if t["domain"] == "reasoning")
    task = next(json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] == rid)

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

    prompt = B1.build_prompt(tok, task, "direct")
    enc = {k: v.to("cuda") for k, v in tok(prompt, return_tensors="pt", truncation=True, max_length=1536).items()}
    opts = task.get("options") or []

    base = _last_logits(model, enc)
    deltas = make_branch_deltas(hidden, k=3, alpha=ALPHA, seed=20260613, include_clean=True)  # [zero,+dir,-dir]

    # zero-delta hook == base (no-op equivalence)
    hz = HiddenDeltaLayerHook(model, target_layer=TARGET_LAYER, target_loops=[TARGET_LOOP],
                              delta=deltas[0], position=-1, max_rms_fraction=MAX_RMS_FRAC)
    try:
        hz.apply(); zero = _last_logits(model, enc)
    finally:
        hz.remove()

    # injected (+dir)
    hi = HiddenDeltaLayerHook(model, target_layer=TARGET_LAYER, target_loops=[TARGET_LOOP],
                              delta=deltas[1], position=-1, max_rms_fraction=MAX_RMS_FRAC)
    try:
        hi.apply(); inj = _last_logits(model, enc)
    finally:
        hi.remove()
    diag = hi.diagnostics()

    zero_equiv = bool(torch.equal(base, zero))
    inj_maxabs = float((base - inj).abs().max())
    inj_argmax_changed = bool(base.argmax() != inj.argmax())
    finite = bool(torch.isfinite(base).all() and torch.isfinite(zero).all() and torch.isfinite(inj).all())
    fired_loops = sorted({r["loop"] for r in diag["records"]})
    hook_loop1_only = fired_loops == [TARGET_LOOP] and not diag["nan_or_inf"]

    verdict = ("S1_2_INJECTION_LIVE" if (finite and zero_equiv and inj_maxabs > 1e-3 and hook_loop1_only)
               else "S1_2_INJECTION_SMOKE_FAIL")
    payload = {"verdict": verdict, "task": rid, "domain": "reasoning",
               "hidden": hidden, "alpha": ALPHA, "delta_rms_injected": delta_rms(deltas[1]),
               "finite": finite, "zero_delta_equals_base": zero_equiv,
               "injected_maxabs_logit_diff": round(inj_maxabs, 4),
               "injected_argmax_changed": inj_argmax_changed,
               "hook_fired_loops": fired_loops, "hook_loop1_only": hook_loop1_only,
               "hook_modifications": diag["modifications"], "hook_nan_inf": diag["nan_or_inf"],
               "option_lean_base": _option_lean(tok, base, opts),
               "option_lean_injected": _option_lean(tok, inj, opts),
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_inject_smoke.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "option_lean_base"}, indent=2))
    print(f"\n{verdict}")
    return 0 if verdict == "S1_2_INJECTION_LIVE" else 4


if __name__ == "__main__":
    raise SystemExit(main())
