"""PART L — branching SFT baseline (pausable + resumable).

Trains a LoRA/QLoRA adapter to emit structured multi-branch attempts + a final answer, from the
verifier-positive branch-training views (branch_format_sft + branch_diversity_sft + final_self_selection).
Base Ouro is NOT modified; adapter saved under the new model root only; heldout never used.

Pausability/resumability:
  - trains in step-budgeted chunks (STEPS_PER_RUN); STOP sentinel -> graceful checkpoint + exit.
  - auto-resumes from the latest checkpoint (full optimizer/scheduler/step state via Trainer).
  - re-run the same command to continue toward TARGET_STEPS.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402

JOB = "branching_sft"
OUT_DIR = B.MODEL_ROOT / JOB
SYSTEM = ("You are Ouro. For the task, produce a few DISTINCT solution branches (different strategies), "
          "then commit to one. End with 'FINAL ANSWER: <answer>'.")


def _latest_checkpoint(d: Path):
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if d.exists() else []
    return str(cks[-1]) if cks else None


def _load_dataset(tok):
    from datasets import Dataset
    rows = []
    for name in ("branch_format_sft", "branch_diversity_sft", "final_self_selection"):
        for r in B.read_jsonl(B.TRAIN / f"{name}.jsonl"):
            p, c = r.get("prompt"), r.get("completion")
            if not p or not c:
                continue
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p},
                    {"role": "assistant", "content": c}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False)
            except Exception:
                text = f"{SYSTEM}\n\n{p}\n\n{c}"
            rows.append({"text": text, "view": name})
    return Dataset.from_list(rows)


def main() -> int:
    started = time.time(); B.ensure_dirs(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_steps = int(os.environ.get("TARGET_STEPS", "400"))
    steps_per_run = int(os.environ.get("STEPS_PER_RUN", "150"))
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except Exception as e:
        B.write_json(B.OUT_ROOT / "branching_sft_training.json",
                     {"BRANCHING_SFT_VERDICT": "BLOCKED", "error": f"import: {type(e).__name__}: {e}"})
        print(B.status_line("BRANCHING_SFT_VERDICT", "BLOCKED")); return 1

    mp = str(B.PROJECT_ROOT / "shared/models/ouro_rltt_local")
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = _load_dataset(tok)
    if len(ds) < 32:
        B.write_json(B.OUT_ROOT / "branching_sft_training.json",
                     {"BRANCHING_SFT_VERDICT": "DATA_INSUFFICIENT", "n_examples": len(ds)})
        print(B.status_line("BRANCHING_SFT_VERDICT", "DATA_INSUFFICIENT"), "n=", len(ds)); return 0

    try:
        # Default: bf16 base + LoRA (same model used everywhere else in the run -> clean trained-vs-base
        # comparison in Part O; 2.67B fits 12GB with gradient checkpointing). 4-bit only as OOM fallback.
        use_4bit = os.environ.get("BNB_4BIT", "0") == "1"
        load_kw = dict(trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16)
        if use_4bit:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(mp, **load_kw)
        if use_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules="all-linear")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        print(f"  base precision: {'4bit-nf4' if use_4bit else 'bf16'}")
    except Exception as e:
        B.write_json(B.OUT_ROOT / "branching_sft_training.json",
                     {"BRANCHING_SFT_VERDICT": "RESOURCE_LIMITED", "error": f"model/peft: {type(e).__name__}: {str(e)[:300]}"})
        print(B.status_line("BRANCHING_SFT_VERDICT", "RESOURCE_LIMITED"), str(e)[:200]); return 0

    resume = _latest_checkpoint(OUT_DIR)
    prior_steps = int(resume.split("-")[-1]) if resume else 0
    run_stop = min(target_steps, prior_steps + steps_per_run)

    class StopCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if B.stop_requested(JOB) or state.global_step >= run_stop:
                control.should_training_stop = True
            return control

    cfg = SFTConfig(output_dir=str(OUT_DIR), max_steps=target_steps, per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, learning_rate=1e-4, lr_scheduler_type="cosine",
                    warmup_ratio=0.03, logging_steps=10, save_steps=50, save_total_limit=3,
                    bf16=True, gradient_checkpointing=True, max_length=1024, packing=False,
                    dataset_text_field="text", report_to=[], dataloader_num_workers=2)
    try:
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                             callbacks=[StopCallback()])
        out = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(OUT_DIR))  # adapter only (no base merge)
        final_step = trainer.state.global_step
        loss = out.training_loss if out else None
    except Exception as e:
        B.write_json(B.OUT_ROOT / "branching_sft_training.json",
                     {"BRANCHING_SFT_VERDICT": "RESOURCE_LIMITED", "error": f"train: {type(e).__name__}: {str(e)[:300]}",
                      "resumed_from": resume})
        print(B.status_line("BRANCHING_SFT_VERDICT", "RESOURCE_LIMITED"), str(e)[:200]); return 0

    done = final_step >= target_steps
    verdict = "SFT_TRAINED" if done else "SFT_TRAINED"  # partial chunk still = trained-so-far (resumable)
    payload = {"BRANCHING_SFT_VERDICT": verdict, "n_examples": len(ds), "target_steps": target_steps,
               "steps_this_run_to": run_stop, "global_step": final_step, "completed_target": done,
               "train_loss": loss, "adapter_dir": str(OUT_DIR.relative_to(B.PROJECT_ROOT)),
               "resumed_from": resume, "paused": (not done) or B.stop_requested(JOB),
               "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(B.OUT_ROOT / "branching_sft_training.json", payload)
    B.write_md(B.OUT_ROOT / "branching_sft_training.md", [
        "# Branching SFT (Part L) — pausable/resumable LoRA", "", B.status_line("BRANCHING_SFT_VERDICT", verdict), "",
        f"bf16 base + LoRA(all-linear), gradient-checkpointed (4-bit only as OOM fallback via BNB_4BIT=1); "
        f"base Ouro unmodified — same precision used everywhere else for a clean trained-vs-base eval. Examples: {len(ds)}. "
        f"global_step {final_step}/{target_steps} (this run capped at {run_stop}); train_loss {loss}. "
        f"{'Reached target.' if done else 'Paused/partial — re-run to continue (auto-resumes from checkpoint).'}", "",
        f"Adapter: `{OUT_DIR.relative_to(B.PROJECT_ROOT)}`. STOP sentinel: `{OUT_DIR/'STOP'}`.",
    ])
    B.prog("L_branching_sft", {"verdict": verdict, "global_step": final_step, "target": target_steps, "paused": payload["paused"]})
    print(B.status_line("BRANCHING_SFT_VERDICT", verdict))
    print(f"  step {final_step}/{target_steps} (run cap {run_stop}) loss={loss} paused={payload['paused']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
