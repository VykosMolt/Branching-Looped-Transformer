"""PART J — offline branch-generator training (bf16 LoRA, resumable, STOP-pausable).

One arm per invocation: ARM=<arm_name> (config from Part I). Two phases:
  PHASE_SFT  — prompt->completion SFT over the arm's SFT-type views (rejection_sft / direct_answer /
               one_branch / multi_branch / branch_budget_policy / teacher policy-distillation).
  PHASE_DPO  — branch-set DPO over branch_set_dpo (continues from the SFT adapter), if 'dpo' in losses.
Base/tokenizer untouched; adapter saved under the v2 model root only. Checkpoint-resumable;
STOP-pausable. In-training generation-based canary eval is deferred to Part K (generation is the
wall); J saves checkpoints and a loss trace. No heldout training, no online generation.
"""
from __future__ import annotations
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v2_common as V  # noqa: E402

ARM = os.environ.get("ARM", "C_offline_verifier_dpo")
SFT_CAP = int(os.environ.get("J_SFT_CAP", "30000"))
DPO_CAP = int(os.environ.get("J_DPO_CAP", "30000"))
rng = random.Random(101)
SFT_VIEWS = {"branch_set_rejection_sft", "direct_answer_sft", "one_branch_sft", "multi_branch_sft",
             "branch_budget_policy", "branch_format_sft", "branch_diversity_sft", "rendered_logic_branch_sets",
             "branch_policy_distillation"}


def _cfg():
    return json.loads((V.MODEL_ROOT / "configs" / f"{ARM}.json").read_text())


def _latest_ckpt(d: Path):
    if not d.exists():
        return None
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


def _view_path(view, cfg=None):
    dirs = []
    if cfg and cfg.get("data_root"):
        dirs.append(V.PROJECT_ROOT / cfg["data_root"])
    dirs += [V.TRAIN_V2, V.DATA_ROOT / "train"]  # default v2 views, then v1 views
    for d in dirs:
        p = d / f"{view}.jsonl"
        if p.exists():
            return p
    return None


def _load_sft_records(cfg):
    rows = []
    for view in cfg["data"]:
        if view not in SFT_VIEWS:
            continue
        p = _view_path(view, cfg)
        if p is None:
            continue
        for l in open(p):
            d = json.loads(l)
            if cfg.get("domain_filter") and d.get("domain") != cfg["domain_filter"]:
                continue
            if view == "rendered_logic_branch_sets":
                ba = d.get("branch_attempts") or []
                pos = [b for b in ba if b.get("external_label") == "pass"]
                if not pos:
                    continue
                rows.append({"prompt": d["task_prompt"], "completion": pos[0]["branch_text"]})
            elif view == "branch_policy_distillation":
                pr, comp = d.get("prompt"), d.get("completion") or d.get("policy")
                if pr and comp:
                    rows.append({"prompt": pr, "completion": str(comp)})
            elif str(d.get("prompt") or "").strip() and str(d.get("completion") or "").strip():
                rows.append({"prompt": d["prompt"], "completion": d["completion"]})
    rng.shuffle(rows)
    return rows[:SFT_CAP]


def _load_dpo_records(cfg):
    p = _view_path("branch_set_dpo", cfg) or V.TRAIN_V2 / "branch_set_dpo.jsonl"
    rows = []
    if "dpo" not in cfg["losses"] or not p.exists():
        return rows
    for l in open(p):
        d = json.loads(l)
        if cfg.get("domain_filter") and d.get("domain") != cfg["domain_filter"]:
            continue
        if str(d.get("prompt") or "").strip() and str(d.get("chosen") or "").strip() and str(d.get("rejected") or "").strip():
            rows.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    rng.shuffle(rows)
    return rows[:DPO_CAP]


def _base_and_tok(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(V.BASE_MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 local_files_only=True, low_cpu_mem_usage=True)
    if cfg["init"] == "prev_sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(V.PREV_SFT_ADAPTER), is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tok


from transformers import TrainerCallback  # noqa: E402


class StopCB(TrainerCallback):
    """trl callback: stop on STOP-file or step budget."""
    def __init__(self, job, max_steps):
        super().__init__()
        self.job, self.max_steps = job, max_steps

    def on_step_end(self, args, state, control, **kw):
        if V.stop_requested(self.job) or state.global_step >= self.max_steps:
            control.should_save = True
            control.should_training_stop = True
        return control


def _fmt(tok, prompt, completion):
    # v5 (round-4) prompts are already chat-templated (GEN_SYSTEM + ChatML, ending in
    # '<|im_start|>assistant\n'); re-wrapping them through apply_chat_template would nest the
    # whole ChatML string inside a fresh user turn under the default 'helpful assistant' system
    # line — exactly the train/eval mismatch round 4 fixes. Use them verbatim and close the turn.
    if "<|im_start|>" in prompt:
        text = prompt + completion
        return text if text.rstrip().endswith("<|im_end|>") else text + "<|im_end|>\n"
    try:
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
        return tok.apply_chat_template(msgs, tokenize=False)
    except Exception:
        return f"{prompt}\n{completion}{tok.eos_token}"


def main() -> int:
    started = time.time()
    cfg = _cfg()
    out_dir = V.MODEL_ROOT / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    job = f"train_{ARM}"
    gpu = V.gpu_audit()
    if gpu.get("orphans"):
        print("REFUSING: orphan GPU procs:", gpu["orphans"])
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    model, tok = _base_and_tok(cfg)
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], target_modules="all-linear",
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    target = cfg["target_steps"]
    sft_steps = int(target * (0.6 if "dpo" in cfg["losses"] else 1.0))
    log = {"arm": ARM, "phases": {}}

    # ---- PHASE SFT ----
    sft_rows = _load_sft_records(cfg)
    print(f"[{ARM}] SFT records {len(sft_rows)} | sft_steps {sft_steps}", flush=True)
    ds = Dataset.from_dict({"text": [_fmt(tok, r["prompt"], r["completion"]) for r in sft_rows]})
    sft_out = out_dir / "sft"
    args = SFTConfig(output_dir=str(sft_out), per_device_train_batch_size=1, gradient_accumulation_steps=8,
                     max_steps=sft_steps, learning_rate=1e-4, bf16=True, logging_steps=10,
                     save_steps=min(int(cfg["save_steps"]), 50),
                     lr_scheduler_type="cosine", warmup_ratio=0.03, max_length=1024, report_to=[],
                     gradient_checkpointing=True, dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=lcfg,
                         callbacks=[StopCB(job, sft_steps)])
    trainer.train(resume_from_checkpoint=_latest_ckpt(sft_out))
    sft_done = int(trainer.state.global_step) >= sft_steps
    if sft_done:
        trainer.save_model(str(sft_out))
    log["phases"]["sft"] = {"records": len(sft_rows), "steps": int(trainer.state.global_step),
                            "complete": sft_done,
                            "loss": trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None}
    del trainer
    torch.cuda.empty_cache()

    if not sft_done:
        verdict = "OFFLINE_TRAINING_PAUSED"
        log.update({"OFFLINE_TRAINING_VERDICT": verdict, "elapsed_seconds": round(time.time() - started, 1)})
        V.set_stage(f"J_train_{ARM}", verdict, {"phases": log["phases"]})
        V.prog(f"J_train_{ARM}", {"verdict": verdict, "phases": log["phases"]})
        print(V.status_line("OFFLINE_TRAINING_VERDICT", verdict))
        print(f"  {ARM}: paused in SFT at step {log['phases']['sft']['steps']}/{sft_steps}; re-run to resume")
        return 0

    # ---- PHASE DPO ----
    dpo_done = True
    if "dpo" in cfg["losses"] and not V.stop_requested(job):
        from trl import DPOTrainer, DPOConfig
        from peft import PeftModel
        dpo_rows = _load_dpo_records(cfg)
        dpo_steps = target - sft_steps
        print(f"[{ARM}] DPO records {len(dpo_rows)} | dpo_steps {dpo_steps}", flush=True)
        del model  # free the SFT-phase copy before loading the DPO base, or the GPU holds both
        gc.collect()
        torch.cuda.empty_cache()
        base2, _ = _base_and_tok({"init": "base"})
        model2 = PeftModel.from_pretrained(base2, str(sft_out), is_trainable=True)
        dds = Dataset.from_list(dpo_rows)
        dpo_out = out_dir / "dpo"
        dargs = DPOConfig(output_dir=str(dpo_out), per_device_train_batch_size=1, gradient_accumulation_steps=8,
                          max_steps=dpo_steps, learning_rate=5e-6, bf16=True, logging_steps=10,
                          save_steps=cfg["save_steps"], beta=0.1, max_length=1024, max_prompt_length=640,
                          report_to=[], gradient_checkpointing=True)
        dtrainer = DPOTrainer(model=model2, args=dargs, train_dataset=dds, processing_class=tok,
                              callbacks=[StopCB(job, dpo_steps)])
        dtrainer.train(resume_from_checkpoint=_latest_ckpt(dpo_out))
        dpo_done = int(dtrainer.state.global_step) >= dpo_steps
        if dpo_done:
            dtrainer.save_model(str(out_dir))  # final adapter = SFT+DPO
        log["phases"]["dpo"] = {"records": len(dpo_rows), "steps": int(dtrainer.state.global_step),
                                "complete": dpo_done,
                                "loss": dtrainer.state.log_history[-1].get("train_loss") if dtrainer.state.log_history else None}
    elif "dpo" in cfg["losses"]:
        dpo_done = False  # STOP arrived between phases; do not publish a final adapter
    else:
        import shutil
        for f in sft_out.glob("adapter_*"):
            shutil.copy(f, out_dir / f.name)

    verdict = "OFFLINE_TRAINING_COMPLETE" if dpo_done else "OFFLINE_TRAINING_PAUSED"
    log.update({"OFFLINE_TRAINING_VERDICT": verdict, "elapsed_seconds": round(time.time() - started, 1),
                "adapter_dir": str(out_dir.relative_to(V.PROJECT_ROOT))})
    # merge-append into the shared offline_training.json (one entry per arm)
    allrep = V.v2.read_json(V.OUT_ROOT / "offline_training.json", {}) or {}
    allrep.setdefault("arms", {})[ARM] = log
    V.write_json(V.OUT_ROOT / "offline_training.json", allrep)
    V.set_stage(f"J_train_{ARM}", verdict, {"phases": log["phases"]})
    V.prog(f"J_train_{ARM}", {"verdict": verdict, "phases": log["phases"]})
    print(V.status_line("OFFLINE_TRAINING_VERDICT", verdict))
    print(f"  {ARM}: {log['phases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
