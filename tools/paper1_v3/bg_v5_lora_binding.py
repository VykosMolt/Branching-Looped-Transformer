"""Frozen conversion #5 -- LoRA direction-outcome binding pilot (S3A pilot).

Sealed plan: artifacts/reports/lora_s3a_pilot_20260727/EXPERIMENT_PLAN.md (do not edit).

Question: can a short LoRA pass BIND the fixed L24 writable direction to verifier
outcomes, so that injecting +alpha*d at the last prompt token (layer 24, loop 1,
alpha 0.02 = SAFE_ALPHA_CAP, RMS-scaled -- the verbatim §8 machinery) raises
verified success after training, where it does nothing on the frozen model?

Stages
  train --adapter binding|control
        Hand-rolled LoRA loop (bs=1, grad-accum 8, AdamW lr 1e-4 cosine, max 400
        steps, CE on candidate tokens only) over the #2 tournament table (slice
        [1080,1260), 6 verified candidates/task). Per-example hook sign:
        binding = verifier outcome (+ for success==True, - for success==False);
        control = coin flip from torch.Generator(seed 20260727), assigned once at
        dataset build and recorded in dataset_manifest.json.
  gate  Zero-delta hook forward == no-hook forward (bit-identical) on 3 probe
        prompts, adapter load check, training loss finite throughout. The
        synthetic absolute-position hook check always runs on CPU; model-on-GPU
        checks are marked SKIPPED_NO_GPU when CUDA is absent.
  eval  All 6 sealed arms on the hash-ordered pool slice [1260,1350) (90 fresh
        tasks), K=4 per arm, temp 0.7/top_p 0.95, max_new 320, FinalAnswerStop,
        marker-gated parse, external verifier only. Per-task seeds
        _sid(20260727, task_uid), arm-independent (paired). eval_records.pt is
        written after every completed arm so a crash resumes.
  analyze
        Sealed endpoints: task-clustered bootstrap 2000 rounds seed 20260727 for
        d_bind (arm1-arm2), d_ctrl (arm3-arm4), d_frozen (arm5-arm6), paired
        difference-of-differences bind-vs-ctrl, adapter-vs-frozen (arm2-arm6).
        Sealed verdict labels -> lora_binding_results.json.

GPU stages must not be launched while the GPU is occupied; run via
utilities/tests/manual/run_lora_s3a_v5.sh which waits for tournament_v4.
--out-root / --table overrides exist for CPU smoke tests only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    _sid, split_for, load_task_pool, options_dict, preanswer_prefix,
    FinalAnswerStop, token_cut_index,  # noqa: F401  (token_cut_index: convention import)
)
from src.evaluator.bg_hidden_branching import (  # noqa: E402
    HiddenDeltaLayerHook, SAFE_ALPHA_CAP, rms_normalize, cosine, _output_tensor,
)

OUT_ROOT_DEFAULT = PROJECT_ROOT / "artifacts/reports/lora_s3a_pilot_20260727"
TABLE_DEFAULT = PROJECT_ROOT / "artifacts/reports/tournament_v4_20260727/tournament_table.pt"
BANK = PROJECT_ROOT / "artifacts/reports/probes/bg_hidden_origin_branch_generator_v1_2026-05-18/basis_bank_v1.pt"
MODEL_PATH = PROJECT_ROOT / "models/ouro_rltt_local"

SEED = 20260727
TARGET_LAYER = 24
TARGET_LOOPS = [1]
ALPHA = 0.02  # == SAFE_ALPHA_CAP (asserted below); delta RMS and max_rms_fraction
HIDDEN_DIM = 2048
# Basis-bank selection convention (curated_dirs_by_layer in mpn_s1_2_multilocus.py),
# plus the sealed plan's heldout_leakage_free requirement; top-1 only.
FAMILY_PRIORITY = ["high_yield_recipe_direction", "v4_tap_aligned", "v2_tap_aligned",
                   "v1_tap_aligned", "salvage_tap_aligned", "old_tap_aligned"]

TRAIN_SLICE = (1080, 1260)   # provenance: the tournament table IS this slice
EVAL_OFFSET, EVAL_N = 1260, 90
K_EVAL = 4
MAX_NEW = 320
TEMPERATURE, TOP_P = 0.7, 0.95
MAX_PROMPT_TOK = 1536
CAND_TOK_CAP = 384           # candidates were generated with max_new 320; retokenization headroom

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
GRAD_ACCUM = 8
MAX_STEPS = 400
LR = 1e-4
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
ROUNDS = 2000
DEVICE = "cuda"

# Sealed arms, in run order (adapter wraps grouped to minimize model reloads).
ARMS = [
    ("arm1", "binding", True),
    ("arm2", "binding", False),
    ("arm3", "control", True),
    ("arm4", "control", False),
    ("arm5", "frozen", True),
    ("arm6", "frozen", False),
]

assert abs(ALPHA - SAFE_ALPHA_CAP) < 1e-12, "sealed alpha must equal SAFE_ALPHA_CAP"


# ------------------------------------------------------------------ hook subclass
class AbsolutePositionDeltaHook(HiddenDeltaLayerHook):
    """HiddenDeltaLayerHook pinned to an ABSOLUTE token index (plen-1).

    The base hook resolves negative positions relative to each forward's own
    seq_len, which during generate() would hit every decode step.  This subclass
    keeps the base class's RMS-scaled delta math untouched and only changes WHEN
    it fires: the delta is applied iff the forward's sequence covers the absolute
    index (seq_len > abs_position) -- once at generation prefill and once per
    teacher-forced training forward; single-token decode steps pass through
    bitwise-untouched.  Zero-delta forwards return the original output object
    (base-class early return), so the zero-delta gate checks a true no-op.
    """

    def __init__(self, model: Any, *, target_layer: int, target_loops: list[int],
                 delta: torch.Tensor, abs_position: int, max_rms_fraction: float) -> None:
        if int(abs_position) < 0:
            raise ValueError(f"abs_position must be >= 0, got {abs_position}")
        super().__init__(model, target_layer=target_layer, target_loops=target_loops,
                         delta=delta, position=int(abs_position),
                         max_rms_fraction=max_rms_fraction)

    def _hook_impl(self, module: Any, args: tuple, kwargs: dict | None, output: Any) -> Any:
        tensor = _output_tensor(output)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 3 and int(tensor.shape[1]) <= self.position:
            self.forward_call_count += 1  # keep loop-fallback parity with base class
            return output
        return super()._hook_impl(module, args, kwargs, output)


def hook_root(model: Any) -> Any:
    """Module whose .model.layers the hook targets (unwrap PEFT if present)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# ------------------------------------------------------------------ direction
def top1_curated_direction_l24() -> tuple[torch.Tensor, dict[str, Any]]:
    """Top-1 curated basis-bank direction for L24 (unit-RMS, 2048-d).

    Convention of curated_dirs_by_layer (mpn_s1_2_multilocus.py): bank order,
    grouped by family, families taken in FAMILY_PRIORITY order, perturbation_usable,
    family != random_orthogonal; plus the sealed plan's heldout_leakage_free filter.
    """
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    byf: dict[str, list[dict]] = {}
    for d in bank["directions"]:
        if not isinstance(d, dict):
            continue
        tl = d.get("target_layer")
        if tl is None or int(tl) != TARGET_LAYER:
            continue
        if not d.get("perturbation_usable") or not d.get("heldout_leakage_free"):
            continue
        if d.get("family") == "random_orthogonal":
            continue
        t = d.get("tensor")
        if isinstance(t, torch.Tensor) and int(t.numel()) == HIDDEN_DIM:
            byf.setdefault(d["family"], []).append(d)
    for fam in FAMILY_PRIORITY:
        if byf.get(fam):
            d = byf[fam][0]
            meta = {"family": d["family"], "name": d.get("name"), "source": d.get("source"),
                    "target_config": d.get("target_config"),
                    "recommended_alpha_bucket": d.get("recommended_alpha_bucket"),
                    "bank": str(BANK.relative_to(PROJECT_ROOT))}
            return rms_normalize(d["tensor"].float().flatten()), meta
    raise RuntimeError("no curated L24 direction found in basis bank")


# ------------------------------------------------------------------ dataset
def build_training_dataset(table_path: Path) -> list[dict[str, Any]]:
    """Deterministic class-balanced training sequence + per-example hook signs.

    Order: candidates sorted by (sha256(task_uid), draw_idx) -- the repo's
    hash-order convention.  Correct/incorrect alternate; nothing is dropped; the
    minority class cycles until the majority class is exhausted.  One-epoch cap =
    min(MAX_STEPS * GRAD_ACCUM, cycled epoch length).  binding sign = verifier
    outcome; control sign = torch.Generator(SEED) coin per example, assigned here
    once (deterministic, recorded in the manifest).
    """
    payload = torch.load(table_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    recs = sorted(records, key=lambda r: (hashlib.sha256(r["task_uid"].encode()).hexdigest(),
                                          int(r["draw_idx"])))
    correct = [r for r in recs if bool(r["success"])]
    incorrect = [r for r in recs if not bool(r["success"])]
    if not correct or not incorrect:
        raise RuntimeError(f"degenerate classes: {len(correct)} correct / {len(incorrect)} incorrect")
    n_pairs = max(len(correct), len(incorrect))
    seq: list[dict] = []
    for k in range(n_pairs):
        seq.append(correct[k % len(correct)])
        seq.append(incorrect[k % len(incorrect)])
    seq = seq[: MAX_STEPS * GRAD_ACCUM]
    gen = torch.Generator().manual_seed(SEED)
    examples = []
    for r in seq:
        ctrl = 1 if int(torch.randint(0, 2, (1,), generator=gen).item()) == 1 else -1
        examples.append({
            "task_uid": r["task_uid"], "draw_idx": int(r["draw_idx"]),
            "success": bool(r["success"]),
            "bind_sign": 1 if bool(r["success"]) else -1,
            "ctrl_sign": ctrl,
            "text": r["generated_text"],
        })
    return examples


def pool_sorted() -> list[dict]:
    pool = load_task_pool()
    pool.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    return pool


def prompt_for(task: dict) -> str:
    return mcq_prompt(task["task_prompt"], options_dict(task["options"]))


# ------------------------------------------------------------------ train stage
def stage_train(adapter_kind: str, out_root: Path, table_path: Path) -> int:
    t0 = time.time()
    out_root.mkdir(parents=True, exist_ok=True)
    direction, dmeta = top1_curated_direction_l24()
    (out_root / "direction.json").write_text(json.dumps(
        {**dmeta, "alpha": ALPHA, "target_layer": TARGET_LAYER, "target_loops": TARGET_LOOPS,
         "position": "last prompt token (absolute plen-1)"}, indent=2) + "\n")
    examples = build_training_dataset(table_path)
    manifest = [{k: e[k] for k in ("task_uid", "draw_idx", "success", "bind_sign", "ctrl_sign")}
                for e in examples]
    (out_root / "dataset_manifest.json").write_text(json.dumps(
        {"seed": SEED, "train_slice": list(TRAIN_SLICE), "n_examples": len(examples),
         "n_correct_in_table": sum(1 for e in examples if e["success"]),
         "examples": manifest}, indent=2) + "\n")
    n_steps = math.ceil(len(examples) / GRAD_ACCUM)
    print(f"[lora-train] adapter={adapter_kind} examples={len(examples)} planned_steps={n_steps} "
          f"direction={dmeta['name']}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("train stage requires CUDA (sealed bf16 GPU recipe) -- "
                           "do not run while the GPU is occupied")
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to(DEVICE)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    prompts: dict[str, list[int]] = {}
    for task in pool_sorted():
        prompts[task["task_uid"]] = tok(prompt_for(task), truncation=True,
                                        max_length=MAX_PROMPT_TOK)["input_ids"]

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(WARMUP_RATIO * n_steps)), n_steps)
    root = hook_root(model)
    loss_path = out_root / f"train_loss_{adapter_kind}.jsonl"
    logf = open(loss_path, "w")
    step, micro_losses, last_loss, unstable = 0, [], None, False
    opt.zero_grad(set_to_none=True)

    for i, ex in enumerate(examples):
        p_ids = prompts.get(ex["task_uid"])
        if p_ids is None:
            raise RuntimeError(f"task {ex['task_uid']} not found in pool for prompt rebuild")
        cand_ids = tok(ex["text"], add_special_tokens=False)["input_ids"][:CAND_TOK_CAP]
        cand_ids = cand_ids + [tok.eos_token_id]
        input_ids = torch.tensor([p_ids + cand_ids], dtype=torch.long, device=DEVICE)
        labels = torch.tensor([[-100] * len(p_ids) + cand_ids], dtype=torch.long, device=DEVICE)
        sign = ex["bind_sign"] if adapter_kind == "binding" else ex["ctrl_sign"]
        hook = AbsolutePositionDeltaHook(
            root, target_layer=TARGET_LAYER, target_loops=TARGET_LOOPS,
            delta=direction * (ALPHA * float(sign)), abs_position=len(p_ids) - 1,
            max_rms_fraction=ALPHA)
        hook.apply()
        try:
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                unstable = True
                print(f"[lora-train] NON-FINITE loss at example {i} (step {step}) -- "
                      f"TRAINING_UNSTABLE recorded, no retry per sealed plan", flush=True)
                logf.write(json.dumps({"step": step + 1, "loss": None, "finite": False,
                                       "example_idx": i}) + "\n")
                break
            (loss / GRAD_ACCUM).backward()  # hook stays attached: checkpoint recompute runs in backward
        finally:
            hook.remove()
        micro_losses.append(float(loss.detach().float()))
        if (i + 1) % GRAD_ACCUM == 0 or i == len(examples) - 1:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = sum(micro_losses) / len(micro_losses)
            logf.write(json.dumps({"step": step, "loss": last_loss, "lr": sched.get_last_lr()[0],
                                   "n_micro": len(micro_losses), "finite": True,
                                   "elapsed": round(time.time() - t0, 1)}) + "\n")
            logf.flush()
            if step == 1 or step % 10 == 0:
                print(f"[lora-train] {adapter_kind} step {step}/{n_steps} "
                      f"loss={last_loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
            micro_losses = []
            if step >= MAX_STEPS:
                break
    logf.close()

    adapter_dir = out_root / f"adapter_{adapter_kind}"
    if not unstable:
        model.save_pretrained(str(adapter_dir))
        print(f"[lora-train] adapter saved to {adapter_dir}", flush=True)
    status = {"adapter": adapter_kind, "n_examples": len(examples), "planned_steps": n_steps,
              "steps_done": step, "loss_finite": not unstable, "unstable": unstable,
              "final_loss": last_loss, "direction": dmeta, "alpha": ALPHA,
              "target_layer": TARGET_LAYER, "target_loops": TARGET_LOOPS,
              "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                       "target_modules": "all-linear"},
              "lr": LR, "grad_accum": GRAD_ACCUM, "max_steps": MAX_STEPS,
              "elapsed_seconds": round(time.time() - t0, 1)}
    (out_root / f"train_status_{adapter_kind}.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"[lora-train] {adapter_kind} {'TRAINING_UNSTABLE' if unstable else 'DONE'} "
          f"steps={step} final_loss={last_loss} elapsed={time.time() - t0:.0f}s", flush=True)
    return 1 if unstable else 0


# ------------------------------------------------------------------ gate stage
def _synthetic_hook_check() -> dict[str, Any]:
    """CPU check of AbsolutePositionDeltaHook on a fake 2-layer stack.

    Verifies: zero-delta is a bitwise no-op; nonzero delta hits ONLY index plen-1
    at loop 1; off-target loops and single-token decode steps are untouched; the
    residual is +/-direction at rms_fraction alpha; sign flips flip the residual.
    """
    import torch.nn as nn

    class _Layer(nn.Module):
        def forward(self, hidden, current_ut=None):  # mirrors decoder-layer loop kwarg
            return hidden

    class _Inner(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList(_Layer() for _ in range(n))

    class _Fake(nn.Module):
        def __init__(self, n=2):
            super().__init__()
            self.model = _Inner(n)

    fake = _Fake()
    layer = fake.model.layers[1]  # target_layer=2 -> index 1
    dim, plen = 16, 4
    gen = torch.Generator().manual_seed(0)
    d = rms_normalize(torch.randn(dim, generator=gen))
    x_prefill = torch.randn(1, plen, dim, generator=gen)
    x_train = torch.randn(1, plen + 5, dim, generator=gen)
    x_decode = torch.randn(1, 1, dim, generator=gen)

    def run(x, delta, cu):
        hook = AbsolutePositionDeltaHook(fake, target_layer=2, target_loops=TARGET_LOOPS,
                                         delta=delta, abs_position=plen - 1,
                                         max_rms_fraction=ALPHA)
        hook.apply()
        try:
            out = layer(x, current_ut=torch.tensor(cu))
        finally:
            hook.remove()
        return out, hook

    c: dict[str, Any] = {}
    out0, _ = run(x_prefill, d * 0.0, 0)
    c["zero_delta_bitwise_noop"] = bool(out0 is x_prefill and torch.equal(out0, x_prefill))

    out1, h1 = run(x_prefill, d * ALPHA, 0)  # current_ut=0 -> loop 1 (targeted)
    resid = (out1[0, plen - 1] - x_prefill[0, plen - 1]).float()
    base_rms = float(x_prefill[0, plen - 1].float().pow(2).mean().sqrt())
    c["prefill_only_last_prompt_token"] = bool(
        torch.equal(out1[:, : plen - 1], x_prefill[:, : plen - 1])
        and not torch.equal(out1[:, plen - 1], x_prefill[:, plen - 1])
        and len(h1.records) == 1)
    c["residual_cosine_to_direction"] = round(cosine(resid, d), 6)
    c["residual_rms_fraction"] = round(float(resid.pow(2).mean().sqrt()) / base_rms, 6)
    c["residual_math_ok"] = bool(c["residual_cosine_to_direction"] > 0.999
                                 and abs(c["residual_rms_fraction"] - ALPHA) < 1e-4)

    out_neg, _ = run(x_prefill, d * (-ALPHA), 0)
    resid_neg = (out_neg[0, plen - 1] - x_prefill[0, plen - 1]).float()
    c["negative_sign_flips_residual"] = bool(cosine(resid_neg, d) < -0.999)

    out2, _ = run(x_prefill, d * ALPHA, 1)  # loop 2 -> untouched
    c["off_target_loop_untouched"] = bool(out2 is x_prefill)

    out3, h3 = run(x_decode, d * ALPHA, 0)  # decode step: seq_len=1 <= plen-1 -> skipped
    c["decode_step_skipped"] = bool(out3 is x_decode and len(h3.records) == 0)

    out4, h4 = run(x_train, d * ALPHA, 0)  # teacher-forced: fires once at plen-1
    c["train_forward_pos_plen_minus_1_only"] = bool(
        torch.equal(out4[:, : plen - 1], x_train[:, : plen - 1])
        and torch.equal(out4[:, plen:], x_train[:, plen:])
        and not torch.equal(out4[:, plen - 1], x_train[:, plen - 1])
        and len(h4.records) == 1)
    c["pass"] = all(v for v in c.values() if isinstance(v, bool))
    return c


def stage_gate(out_root: Path) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    skipped: list[str] = []
    failed: list[str] = []

    syn = _synthetic_hook_check()
    checks["synthetic_abs_position_hook"] = syn
    print(f"[lora-gate] synthetic abs-position hook check: "
          f"{'PASS' if syn['pass'] else 'FAIL'} {json.dumps(syn)}", flush=True)
    if not syn["pass"]:
        failed.append("synthetic_abs_position_hook")

    for kind in ("binding", "control"):
        lp = out_root / f"train_loss_{kind}.jsonl"
        if not lp.exists():
            checks[f"loss_finite_{kind}"] = {"pass": False, "reason": "loss log missing"}
            failed.append(f"loss_finite_{kind}")
            print(f"[lora-gate] loss_finite_{kind}: FAIL (log missing)", flush=True)
            continue
        rows = [json.loads(line) for line in lp.read_text().splitlines() if line.strip()]
        finite = bool(rows) and all(r.get("finite", False) and r.get("loss") is not None
                                    and math.isfinite(r["loss"]) for r in rows)
        checks[f"loss_finite_{kind}"] = {"pass": finite, "n_steps": len(rows)}
        if not finite:
            failed.append(f"loss_finite_{kind}")
        print(f"[lora-gate] loss_finite_{kind}: {'PASS' if finite else 'FAIL'} "
              f"({len(rows)} steps)", flush=True)

    if not torch.cuda.is_available():
        skipped.extend(["zero_delta_noop_real_model", "adapter_load"])
        checks["zero_delta_noop_real_model"] = {"pass": None, "status": "SKIPPED_NO_GPU"}
        checks["adapter_load"] = {"pass": None, "status": "SKIPPED_NO_GPU"}
        print("[lora-gate] real-model zero-delta + adapter-load checks: SKIPPED_NO_GPU "
              "(gate_pass stays false until run on GPU)", flush=True)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        direction, _ = top1_curated_direction_l24()
        tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, low_cpu_mem_usage=True).to(DEVICE).eval()
        probes = pool_sorted()[EVAL_OFFSET: EVAL_OFFSET + 3]
        per_probe = []
        for task in probes:
            enc = tok(prompt_for(task), return_tensors="pt", truncation=True,
                      max_length=MAX_PROMPT_TOK)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            plen = int(enc["input_ids"].shape[1])
            with torch.inference_mode():
                ref = model(**enc).logits
            hook = AbsolutePositionDeltaHook(
                hook_root(model), target_layer=TARGET_LAYER, target_loops=TARGET_LOOPS,
                delta=direction * 0.0, abs_position=plen - 1, max_rms_fraction=ALPHA)
            hook.apply()
            try:
                with torch.inference_mode():
                    hooked = model(**enc).logits
            finally:
                hook.remove()
            per_probe.append(bool(torch.equal(ref, hooked)))
        ok = all(per_probe)
        checks["zero_delta_noop_real_model"] = {"pass": ok, "per_probe": per_probe,
                                                "probes": [t["task_uid"] for t in probes]}
        if not ok:
            failed.append("zero_delta_noop_real_model")
        print(f"[lora-gate] zero-delta no-op on 3 probe prompts: "
              f"{'PASS (bit-identical)' if ok else 'FAIL'} {per_probe}", flush=True)

        loads = {}
        for kind in ("binding", "control"):
            adir = out_root / f"adapter_{kind}"
            try:
                wrapped = PeftModel.from_pretrained(model, str(adir), is_trainable=False)
                wrapped.eval()
                model = wrapped.unload()
                loads[kind] = True
            except Exception as exc:  # noqa: BLE001
                loads[kind] = f"{type(exc).__name__}: {exc}"
        ok = all(v is True for v in loads.values())
        checks["adapter_load"] = {"pass": ok, "detail": loads}
        if not ok:
            failed.append("adapter_load")
        print(f"[lora-gate] adapter load check: {'PASS' if ok else 'FAIL'} {loads}", flush=True)

    gate_pass = not failed and not skipped
    result = {"gate_pass": gate_pass, "failed": failed, "skipped": skipped,
              "cpu_partial": bool(skipped), "checks": checks, "seed": SEED}
    (out_root / "gate_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[lora-gate] gate_pass={gate_pass} failed={failed} skipped={skipped}", flush=True)
    return 1 if failed else 0


# ------------------------------------------------------------------ eval stage
def _save_eval(path: Path, records: list[dict], done: set[str]) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"seed": SEED, "eval_offset": EVAL_OFFSET, "n_tasks": EVAL_N, "k": K_EVAL,
                "max_new": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P,
                "arms_done": sorted(done), "n_records": len(records), "records": records}, tmp)
    os.replace(tmp, path)


def stage_eval(out_root: Path) -> int:
    gate_path = out_root / "gate_results.json"
    if not gate_path.exists():
        raise RuntimeError("gate_results.json missing -- run --stage gate first")
    gate = json.loads(gate_path.read_text())
    if not gate.get("gate_pass"):
        raise RuntimeError(f"gate not passed (failed={gate.get('failed')} "
                           f"skipped={gate.get('skipped')}) -- eval refused")
    if not torch.cuda.is_available():
        raise RuntimeError("eval stage requires CUDA -- do not run while the GPU is occupied")

    t0 = time.time()
    direction, dmeta = top1_curated_direction_l24()
    tasks = pool_sorted()[EVAL_OFFSET: EVAL_OFFSET + EVAL_N]
    print(f"[lora-eval] {len(tasks)} tasks, slice [{EVAL_OFFSET},{EVAL_OFFSET + EVAL_N}), "
          f"K={K_EVAL}, direction={dmeta['name']}", flush=True)

    rec_path = out_root / "eval_records.pt"
    records: list[dict] = []
    done: set[str] = set()
    if rec_path.exists():
        prev = torch.load(rec_path, map_location="cpu", weights_only=False)
        records, done = prev["records"], set(prev["arms_done"])
        print(f"[lora-eval] resuming: arms done {sorted(done)}, {len(records)} records", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[lora-eval] model loaded in {time.time() - t0:.1f}s", flush=True)

    cur_kind = "frozen"
    for arm, kind, inject in ARMS:
        if arm in done:
            print(f"[lora-eval] {arm} already done -- skip", flush=True)
            continue
        if kind != cur_kind:
            if cur_kind != "frozen":
                model = model.unload()
            if kind != "frozen":
                model = PeftModel.from_pretrained(model, str(out_root / f"adapter_{kind}"),
                                                  is_trainable=False).to(DEVICE)
            model.eval()
            cur_kind = kind
            print(f"[lora-eval] model variant -> {kind}", flush=True)

        arm_t0, n_succ, n_gen = time.time(), 0, 0
        for ti, task in enumerate(tasks):
            opts = options_dict(task["options"])
            gold_letter = chr(65 + int(task["answer_key"]))
            prompt = prompt_for(task)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOK,
                      padding=False)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            plen = int(enc["input_ids"].shape[1])
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            seed_i = _sid(SEED, task["task_uid"])  # arm-independent -> paired across arms
            torch.manual_seed(seed_i)
            torch.cuda.manual_seed_all(seed_i)
            stopper = StoppingCriteriaList([FinalAnswerStop(tok, plen, K_EVAL)])
            hook = None
            if inject:
                hook = AbsolutePositionDeltaHook(
                    hook_root(model), target_layer=TARGET_LAYER, target_loops=TARGET_LOOPS,
                    delta=direction * ALPHA, abs_position=plen - 1, max_rms_fraction=ALPHA)
                hook.apply()
            try:
                with torch.inference_mode():
                    out = model.generate(
                        **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMPERATURE,
                        top_p=TOP_P, num_return_sequences=K_EVAL, pad_token_id=pad_id,
                        eos_token_id=tok.eos_token_id, use_cache=True, stopping_criteria=stopper)
            finally:
                if hook is not None:
                    hook.remove()
            n_mods = len(hook.records) if hook is not None else 0
            seqs = out.tolist()
            del out
            torch.cuda.empty_cache()
            for ci in range(K_EVAL):
                new_ids = seqs[ci][plen:]
                while new_ids and new_ids[-1] == pad_id:
                    new_ids.pop()
                text = tok.decode(new_ids, skip_special_tokens=True).strip()
                pre_text, found_marker = preanswer_prefix(text)
                parsed_raw = parse_mcq_answer(text, opts)
                parsed = parsed_raw if found_marker else None
                success = bool(found_marker and parsed == gold_letter)
                records.append({
                    "arm": arm, "adapter": kind, "inject": bool(inject),
                    "task_uid": task["task_uid"], "split": split_for(task["task_uid"]),
                    "proof_depth": task["proof_depth"], "draw_idx": ci,
                    "gold_letter": gold_letter, "generated_text": text,
                    "n_gen_tok": len(new_ids), "hit_max_tokens": len(new_ids) >= MAX_NEW,
                    "found_final_marker": bool(found_marker), "parsed_answer": parsed,
                    "success": success, "malformed": bool(not found_marker or parsed is None),
                    "seed": seed_i, "prompt_tok": plen, "n_hook_mods": n_mods,
                })
                n_succ += int(success)
                n_gen += 1
            if ti == 0 or (ti + 1) % 15 == 0:
                print(f"[lora-eval] {arm} {ti + 1}/{len(tasks)} tasks "
                      f"succ={n_succ}/{n_gen} elapsed={time.time() - arm_t0:.0f}s", flush=True)
        done.add(arm)
        _save_eval(rec_path, records, done)
        print(f"[lora-eval] {arm} DONE: success {n_succ}/{n_gen} "
              f"({time.time() - arm_t0:.0f}s) -- checkpoint saved", flush=True)

    if cur_kind != "frozen":
        model.unload()
    print(f"[lora-eval] all arms done: {len(records)} records, "
          f"elapsed={time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ analyze stage
def _ci(vals: list[float]) -> list[float]:
    s = sorted(vals)
    return [s[int(0.025 * len(s))], s[int(0.975 * len(s))]]


def sealed_verdict(d_bind: float, d_bind_ci: list[float], dod_ci: list[float],
                   adapter_ci: list[float], unstable_reasons: list[str]) -> str:
    """Sealed verdict labels, applied in sealed order."""
    if unstable_reasons:
        return "TRAINING_UNSTABLE"
    if d_bind > 0 and d_bind_ci[0] > 0 and dod_ci[0] > 0:
        return "BINDING_TRAINABLE"
    if adapter_ci[0] > 0 and d_bind_ci[0] <= 0 <= d_bind_ci[1]:
        return "ADAPTER_IMPROVES_UNCONDITIONALLY"
    return "BINDING_NOT_DETECTED"


def stage_analyze(out_root: Path) -> int:
    unstable_reasons: list[str] = []
    gate_path = out_root / "gate_results.json"
    if not gate_path.exists():
        unstable_reasons.append("gate_results_missing")
    else:
        gate = json.loads(gate_path.read_text())
        if not gate.get("gate_pass"):
            unstable_reasons.append(f"gate_failed: failed={gate.get('failed')} "
                                    f"skipped={gate.get('skipped')}")
    for kind in ("binding", "control"):
        sp = out_root / f"train_status_{kind}.json"
        if not sp.exists():
            unstable_reasons.append(f"train_status_{kind}_missing")
        else:
            st = json.loads(sp.read_text())
            if not st.get("loss_finite", False):
                unstable_reasons.append(f"non_finite_loss_{kind}")

    dmeta = None
    dpath = out_root / "direction.json"
    if dpath.exists():
        dmeta = json.loads(dpath.read_text())

    rec_path = out_root / "eval_records.pt"
    result: dict[str, Any] = {
        "seed": SEED, "rounds": ROUNDS, "k": K_EVAL, "alpha": ALPHA,
        "target_layer": TARGET_LAYER, "target_loops": TARGET_LOOPS,
        "eval_slice": [EVAL_OFFSET, EVAL_OFFSET + EVAL_N], "train_slice": list(TRAIN_SLICE),
        "direction": dmeta, "unstable_reasons": unstable_reasons,
        "optional_secondary_arm_minus_alpha": "not_run (optional in sealed plan)",
    }
    if not rec_path.exists():
        if not unstable_reasons:
            raise RuntimeError("eval_records.pt missing and no instability recorded -- "
                               "run --stage eval first")
        result["verdict"] = "TRAINING_UNSTABLE"
        (out_root / "lora_binding_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[lora-an] no eval records; VERDICT: TRAINING_UNSTABLE "
              f"({unstable_reasons})", flush=True)
        return 0

    payload = torch.load(rec_path, map_location="cpu", weights_only=False)
    recs = payload["records"]
    arm_names = [a for a, _, _ in ARMS]
    by: dict[str, dict[str, list[bool]]] = {a: {} for a in arm_names}
    for r in recs:
        if r["arm"] in by:
            by[r["arm"]].setdefault(r["task_uid"], []).append(bool(r["success"]))
    tasks = sorted(set.intersection(*(set(by[a]) for a in arm_names))) if all(by.values()) else []
    if not tasks:
        raise RuntimeError("no tasks covered by all 6 arms")
    rates = {a: torch.tensor([sum(by[a][t]) / len(by[a][t]) for t in tasks], dtype=torch.float64)
             for a in arm_names}
    n = len(tasks)
    arm_rate = {a: float(rates[a].mean()) for a in arm_names}

    d_bind = arm_rate["arm1"] - arm_rate["arm2"]
    d_ctrl = arm_rate["arm3"] - arm_rate["arm4"]
    d_frozen = arm_rate["arm5"] - arm_rate["arm6"]
    dod = d_bind - d_ctrl
    adapter_vs_frozen = arm_rate["arm2"] - arm_rate["arm6"]

    rng = torch.Generator().manual_seed(SEED)
    boot: dict[str, list[float]] = {k: [] for k in
                                    ("d_bind", "d_ctrl", "d_frozen", "dod", "adapter_vs_frozen")}
    for _ in range(ROUNDS):
        pick = torch.randint(0, n, (n,), generator=rng)
        m = {a: float(rates[a][pick].mean()) for a in arm_names}
        db, dc = m["arm1"] - m["arm2"], m["arm3"] - m["arm4"]
        boot["d_bind"].append(db)
        boot["d_ctrl"].append(dc)
        boot["d_frozen"].append(m["arm5"] - m["arm6"])
        boot["dod"].append(db - dc)
        boot["adapter_vs_frozen"].append(m["arm2"] - m["arm6"])
    ci = {k: _ci(v) for k, v in boot.items()}

    verdict = sealed_verdict(d_bind, ci["d_bind"], ci["dod"], ci["adapter_vs_frozen"],
                             unstable_reasons)
    result.update({
        "n_tasks": n, "arm_rates": arm_rate,
        "delta_bind": d_bind, "delta_bind_ci95": ci["d_bind"],
        "delta_ctrl": d_ctrl, "delta_ctrl_ci95": ci["d_ctrl"],
        "delta_frozen": d_frozen, "delta_frozen_ci95": ci["d_frozen"],
        "diff_of_diffs_bind_vs_ctrl": dod, "diff_of_diffs_ci95": ci["dod"],
        "adapter_vs_frozen": adapter_vs_frozen, "adapter_vs_frozen_ci95": ci["adapter_vs_frozen"],
        "verdict": verdict,
    })
    (out_root / "lora_binding_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[lora-an] n_tasks={n} rates=" +
          " ".join(f"{a}={arm_rate[a]:.3f}" for a in arm_names), flush=True)
    print(f"[lora-an] d_bind {d_bind:+.3f} [{ci['d_bind'][0]:+.3f},{ci['d_bind'][1]:+.3f}] "
          f"d_ctrl {d_ctrl:+.3f} [{ci['d_ctrl'][0]:+.3f},{ci['d_ctrl'][1]:+.3f}] "
          f"d_frozen {d_frozen:+.3f} [{ci['d_frozen'][0]:+.3f},{ci['d_frozen'][1]:+.3f}]", flush=True)
    print(f"[lora-an] diff-of-diffs {dod:+.3f} [{ci['dod'][0]:+.3f},{ci['dod'][1]:+.3f}] "
          f"adapter-vs-frozen {adapter_vs_frozen:+.3f} "
          f"[{ci['adapter_vs_frozen'][0]:+.3f},{ci['adapter_vs_frozen'][1]:+.3f}]", flush=True)
    print(f"[lora-an] VERDICT: {verdict}", flush=True)
    return 0


# ------------------------------------------------------------------ entry point
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["train", "gate", "eval", "analyze"], required=True)
    ap.add_argument("--adapter", choices=["binding", "control"])
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT,
                    help="override for CPU smoke tests only")
    ap.add_argument("--table", type=Path, default=TABLE_DEFAULT,
                    help="override for CPU smoke tests only")
    a = ap.parse_args()
    if a.stage == "train":
        if not a.adapter:
            ap.error("--adapter binding|control required for --stage train")
        return stage_train(a.adapter, a.out_root, a.table)
    if a.stage == "gate":
        return stage_gate(a.out_root)
    if a.stage == "eval":
        return stage_eval(a.out_root)
    return stage_analyze(a.out_root)


if __name__ == "__main__":
    raise SystemExit(main())
