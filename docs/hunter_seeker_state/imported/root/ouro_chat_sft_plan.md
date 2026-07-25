<!-- Imported from `ouro_chat_sft_plan.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 69f13ff4479c14c316e5363f9734bf6599cbfcfae3bb41a258ec5bfb1d1776db; original line count: 1424. -->

# Ouro training plan — 2026-05-11

Resume document for the chat + advanced-coding fine-tuning of the 2.67 B Ouro RLTT checkpoint. Author: Claude (Opus 4.7). The user (illjaesterhazy) asked for this doc before leaving for university; pick up here when they're back.

Treat this as a working plan, not a spec. No training has started.

**The plan has three layers, primary path now is Phase 3:**

- **Phase 3 — Full RLTT continuation on rented multi-GPU** (primary). Resumes the original FSDP training shards in `~/Downloads/RLTT/Downloads_RLTT/RLTT/`. Chat reward = the user's `pairwise_epoch2.pt` relational-preference evaluator (94.98 % HH-RLHF accuracy, arxiv:2604.09870). Code reward = differential-check primary + evaluator secondary.
- **Phase 1 — Chat LoRA-SFT on laptop** (fallback if rented compute is unavailable). Documented in full below; conservative LoRA on OASST1.
- **Phase 2 — Hard-coding LoRA-SFT on laptop** (fallback). Same shape, advanced-algorithms data.

The fallbacks remain accurate and runnable on the 12 GB laptop. Phase 3 is the path of record.

## Status

| Stage                                | State        |
|--------------------------------------|--------------|
| Wrapper-side conversational fix      | **Done**     |
| Decisions for SFT                    | **Locked**   |
| Sibling venv + dependencies          | Not started  |
| OASST1 data prep                     | Not started  |
| LoRA training script                 | Not started  |
| Training run                         | Not started  |
| Eval (DSA + chat + Hunter Seeker)    | Not started  |
| Deployment behind env-var gate       | Not started  |

## Background — what's already in place

This SFT pass is the second half of a two-part fix. The first half (wrapper) landed earlier today on 2026-05-11. Without it, the symptoms aren't fixable in weights alone.

### Wrapper fix that landed today (no rerun needed)

Edited files in `/home/moloch/local_agent/`:

- `ouro_task_profile.py`
  - Added `_looks_like_casual_conversation(task)` — short-circuits greetings/social input to `low|conversation|safe` regardless of keyword soup.
  - Rewrote `_looks_like_ml_theory_task` to use word-boundary regex (`re.compile(r"(?<![A-Za-z0-9_])(?:cs\.lg|dpo|ppo|sac|impala|v[ \-]trace|infonce)(?![A-Za-z0-9_])")`). Fixes a real bug: `"ppo"` substring-matched inside `"suPPOse"`.
  - Same casual-conversation override applied to qwen-router output in `PromptClassifier.classify`.

- `ouro_agent_improved.py:auto_multi_model`
  - Early branch: `if profile.task_type == "conversation" and profile.risk_profile == "safe": return direct_answer(...)`. Avoids judge_task's double-generate-and-compare for chat.

- `ouro_policies.py`
  - `looks_like_hard_technical_task` now excludes `conversation`/`creative`/`project_navigation`.
  - `trim_late_reasoning_restart` now also strips leading CoT preamble after a `FINAL ANSWER:` prefill, iterated up to 4 times.

- `ouro_backend.py:generate`
  - Plumbed `repetition_penalty` and `no_repeat_ngram_size` through `runtime_overrides`. Opt-in only — DSA harness and other callers unaffected.

- `ouro_direct.py`
  - New `_conversational_direct_answer` skips `OURO_SOLVER_POSTURE` and the `FINAL ANSWER:` prefill entirely.
  - Uses `repetition_penalty=1.1`, `no_repeat_ngram_size=6`, `temperature=0.3`.
  - New `_parse_conversational_output(raw)` splits raw output into `(thinking, reply)`. Cuts at fake `**Ouros:**`/`**Ouro:**` role markers, `user\n` follow-ups, and `*(Note: …)*` asides. Strips surrounding quotes around the reply.
  - Wraps CoT in `<thinking>…</thinking>` sentinels rather than discarding it.

- `ouro_ui.py`
  - Added `.thinking-block` CSS (collapsible `<details>` styled to match the rest of the UI).
  - `renderMarkdown` now extracts `<thinking>…</thinking>` blocks and renders them as `<details class="thinking-block"><summary>Thinking</summary>…</details>` — same dropdown UX as Claude's web app.

### Verified post-wrapper-fix on this Arch venv

```
greeting "Hello Ouro, …": low|conversation|safe; auto mode -> [Auto mode: conversation]
output: <thinking>...</thinking> + clean conversational reply, no [unverified] header,
        no '(' loop, no fake turns.
DSA harness (tools/test_local_agent_dsa_coding.py): 4/4 pass (subarray_sum_count,
        longest_increasing_subsequence, word_ladder_length, count_smaller_after_self),
        same as the pre-fix baseline in SESSION_CONTEXT_2026-05-10.md.
```

### What's still wrong (this is why we're doing SFT)

Even with a clean conversational system prompt and prefill stripped, the model itself:

1. **Third-person narrates the user** in CoT before producing a reply ("The user mentioned they're working on…").
2. **Hallucinates multi-turn dialogue**, including `**Ouros:**` role markers (note the typo — it spells the name wrong).
3. **Over-CoTs on simple greetings** — burns 200+ tokens of scratch before a 1-sentence reply.

The wrapper papers over this with the `<thinking>` dropdown and fake-turn cut, but the model's training distribution is clearly heavy on solver-style outputs and light on chat. SFT fixes the root.

## Decisions locked in (2026-05-11, from the user)

| Decision        | Choice                              | Rationale                                                                |
|-----------------|-------------------------------------|--------------------------------------------------------------------------|
| Data source     | **OASST1** (public, HuggingFace)    | User-selected. Diverse, human-curated, conversational.                   |
| Method          | **LoRA** (conservative)             | 12 GB GPU rules out full FT. Adapter is reversible.                      |
| Rank / scope    | **r=8** on attn+MLP                 | Smallest meaningful rank; minimizes drift from base.                     |
| Epochs / data   | **2 epochs / ~500 examples**        | Style nudge, not a re-pretrain.                                          |
| LR / opt        | **AdamW, lr=2e-4, bf16**            | Standard PEFT defaults for this size class.                              |
| Env             | **Sibling venv** (`venv_sft/`)      | Inference venv stays clean. Hunter Seeker imports unaffected.            |
| Deployment      | **Env-var-gated adapter**           | `LOCAL_AGENT_OURO_CHAT_LORA=1` opt-in. Default off → Hunter Seeker & DSA work untouched. |

The user's note: "Keep in mind there may be no issues with the models capabilities, ALSO Hunter Seeker needs to continue working." Interpretation: the SFT must not regress DSA or Hunter Seeker. Env-var gating + LoRA-on-demand handles this cleanly.

## Hardware / environment baseline (verified 2026-05-11)

```
Host          : arch-legion, Arch Linux, kernel 7.0.5-arch1-1
GPU           : RTX 5070 Ti Laptop, 12.32 GB total, driver 595.71.05, CUDA 12.8
Inference venv: /home/moloch/ouro_project/venv (Python 3.14.4)
  torch                  2.12.0.dev20260407+cu128
  transformers           4.54.1   (pinned; do NOT bump)
  safetensors            0.7.0
  accelerate             1.13.0
Model         : /home/moloch/ouro_project/models/ouro_rltt_local
  OuroForCausalLM, 2.67 B params, bf16, 48 layers, hidden 2048, 16 heads,
  total_ut_steps=4, vocab 49152, max_position_embeddings 65536, rope_theta 1e6
  Chat template: ChatML (<|im_start|>role\ncontent<|im_end|>\n)
  Forward signature accepts `labels` and computes loss internally
  (modeling_ouro.py:1098 forward, 1261 if labels is not None: loss = self.loss_function(...))
```

VRAM math (bf16, LoRA r=8 on attn+MLP):

| Item                  | Approx size |
|-----------------------|-------------|
| Base model weights    | 5.3 GB      |
| LoRA params           | ~30–60 MB   |
| LoRA optimizer state  | ~150 MB     |
| Activations (B=1)     | ~2–3 GB     |
| Headroom              | ~3–4 GB     |
| **Total**             | ~11 GB — fits in 12.3 GB with gradient checkpointing |

Use gradient checkpointing to be safe. Avoid Trainer's automatic FP16 fallback — bf16 only.

## Plan — step by step

### Step 1: sibling venv

```bash
python3.14 -m venv /home/moloch/ouro_project/venv_sft
source /home/moloch/ouro_project/venv_sft/bin/activate
pip install --upgrade pip

# Pin the same torch + transformers as the inference venv, then add SFT deps.
# torch 2.12.0.dev is from a custom index — match by version not URL.
pip install \
    'torch==2.12.0.dev20260407+cu128' \
    --index-url https://download.pytorch.org/whl/nightly/cu128
pip install \
    'transformers==4.54.1' \
    'safetensors==0.7.0' \
    'accelerate==1.13.0' \
    'peft>=0.13,<0.15' \
    'trl>=0.11,<0.13' \
    'datasets>=2.20,<3.1' \
    'bitsandbytes' \
    'sentencepiece' \
    'jinja2'

# Sanity check
python -c "
import torch, transformers, peft, trl, datasets, accelerate
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('peft        :', peft.__version__)
print('trl         :', trl.__version__)
print('datasets    :', datasets.__version__)
print('accelerate  :', accelerate.__version__)
print('cuda        :', torch.cuda.is_available(), torch.version.cuda)
"
```

If the torch nightly index URL has rotated, find the matching wheel via `pip index versions torch --index-url https://download.pytorch.org/whl/nightly/cu128`. The version string `2.12.0.dev20260407+cu128` is what the inference venv uses; matching it avoids ABI mismatch issues.

### Step 2: data prep

Download a small slice of OASST1, keep only English conversational exchanges, filter out solver-style turns, format for ChatML.

Target output: `/home/moloch/ouro_project/data/ouro_chat_sft/train.jsonl` + `val.jsonl`, each line one record `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.

```python
# /home/moloch/ouro_project/tools/prep_ouro_chat_sft.py
from datasets import load_dataset
import json, re, random
from pathlib import Path

OUT = Path("/home/moloch/ouro_project/data/ouro_chat_sft")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)
ds = load_dataset("OpenAssistant/oasst1", split="train")

# Build pairs: take 'prompter' messages whose top-rated child is an 'assistant'
# message, in English, length 40-1200 chars on both sides.
by_id = {row["message_id"]: row for row in ds}
children = {}
for row in ds:
    if row["parent_id"]:
        children.setdefault(row["parent_id"], []).append(row)

def best_child(row):
    kids = children.get(row["message_id"]) or []
    kids = [k for k in kids if k["role"] == "assistant" and k["lang"] == "en"]
    if not kids:
        return None
    return max(kids, key=lambda k: (k.get("rank") is None, -(k.get("rank") or 99)))

PAIRS = []
for row in ds:
    if row["role"] != "prompter" or row["lang"] != "en":
        continue
    if not (40 <= len(row["text"]) <= 1200):
        continue
    child = best_child(row)
    if child is None:
        continue
    if not (40 <= len(child["text"]) <= 1200):
        continue
    PAIRS.append((row["text"], child["text"]))

# Filter out solver-style assistant turns that would re-teach over-CoT.
SOLVER_PREFIXES = ("Okay,", "Alright,", "First,", "Let me ", "Let's think",
                   "The user", "Sure! Let's", "Sure, let's", "Step 1")
SOLVER_HEAD = ("```", "def ", "class ", "import ")
def is_solver_style(text):
    low = text.lstrip()
    if any(low.startswith(p) for p in SOLVER_PREFIXES):
        return True
    if any(low.startswith(p) for p in SOLVER_HEAD):
        return True
    if "the user" in low.lower()[:160]:
        return True
    return False

PAIRS = [p for p in PAIRS if not is_solver_style(p[1])]
random.shuffle(PAIRS)
PAIRS = PAIRS[:600]  # 500 train + 50 val + ~50 buffer
print(f"kept {len(PAIRS)} pairs")

SYS = ("You are Ouro, a compact looped reasoning model. "
       "Speak to the user directly in the first person. Keep your reply short and natural.")

def to_record(user, assistant):
    return {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}

split = int(len(PAIRS) * 0.9)
with (OUT / "train.jsonl").open("w") as f:
    for u, a in PAIRS[:split]:
        f.write(json.dumps(to_record(u, a)) + "\n")
with (OUT / "val.jsonl").open("w") as f:
    for u, a in PAIRS[split:]:
        f.write(json.dumps(to_record(u, a)) + "\n")
print("wrote", OUT)
```

Notes:
- The `is_solver_style` filter is intentional. We're trying to *unlearn* CoT preambles for chat, so we don't want OASST examples that already start with "Okay, let me…".
- Bonus: append ~20 hand-written Ouro-specific examples (introduction, identity, "what are you?", "who made you?") so the model picks up on its own identity. Put them in a separate `extras.jsonl` and concat before training. Keep them very short (1–2 sentences each).
- If you want to be extra careful about identity drift, also write 5–10 examples where the user asks "Why are you over-explaining?" and the assistant gives a terse 1-sentence answer. This anchors the style.

### Step 3: training script

```python
# /home/moloch/ouro_project/tools/train_ouro_chat_lora.py
import os, json, torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "/home/moloch/ouro_project/models/ouro_rltt_local"
DATA  = Path("/home/moloch/ouro_project/data/ouro_chat_sft")
OUT   = Path("/home/moloch/ouro_project/models/ouro_rltt_chat_lora")
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # required when gradient checkpointing is on

# LoRA: rank 8 on attn + MLP. Do NOT touch embeddings or lm_head.
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

MAX_LEN = 1024
ASSIST_OPEN  = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
TURN_CLOSE   = tok.encode("<|im_end|>",            add_special_tokens=False)

def encode(row):
    text = tok.apply_chat_template(row["messages"], tokenize=False,
                                   add_generation_prompt=False)
    ids = tok(text, truncation=True, max_length=MAX_LEN, return_tensors=None)["input_ids"]
    labels = [-100] * len(ids)
    # Find assistant span: from token after ASSIST_OPEN to next TURN_CLOSE.
    i = 0
    while i < len(ids):
        # Locate ASSIST_OPEN pattern
        if ids[i:i+len(ASSIST_OPEN)] == ASSIST_OPEN:
            start = i + len(ASSIST_OPEN)
            j = start
            while j < len(ids):
                if ids[j:j+len(TURN_CLOSE)] == TURN_CLOSE:
                    end = j + len(TURN_CLOSE)
                    for k in range(start, end):
                        labels[k] = ids[k]
                    i = end
                    break
                j += 1
            else:
                break
        else:
            i += 1
    return {"input_ids": ids, "labels": labels, "attention_mask": [1]*len(ids)}

train = load_dataset("json", data_files=str(DATA/"train.jsonl"), split="train").map(encode, remove_columns=["messages"])
val   = load_dataset("json", data_files=str(DATA/"val.jsonl"),   split="train").map(encode, remove_columns=["messages"])

def collate(batch):
    L = max(len(r["input_ids"]) for r in batch)
    pad = tok.pad_token_id
    def padlist(x, fill): return x + [fill]*(L - len(x))
    return {
        "input_ids":      torch.tensor([padlist(r["input_ids"], pad) for r in batch]),
        "labels":         torch.tensor([padlist(r["labels"],   -100) for r in batch]),
        "attention_mask": torch.tensor([padlist(r["attention_mask"], 0) for r in batch]),
    }

args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True,
    remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                  data_collator=collate)
trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("saved adapter ->", OUT)
```

Wall-clock estimate on the 5070 Ti Laptop: **~30–60 min** for 2 epochs of ~500 examples at gradient_accum=8.

**Known unknowns:**
- `OuroForCausalLM.forward` returns a custom `OuroRLTTCausalLMOutput` (or `CausalLMOutputWithPast`) — HF Trainer relies on `.loss` being on the output. Confirmed at modeling_ouro.py:1261 that `loss` is computed and attached when `labels` is provided. Should be fine, but watch the first 10 steps for `None` loss.
- Gradient checkpointing + the looped forward (`total_ut_steps=4`) means each step does 4 forward sub-passes. Memory will be tighter than for a vanilla 2.7 B model. If OOM, drop to `gradient_accumulation_steps=4` and reduce `MAX_LEN` to 768.
- PEFT may complain about the custom `OuroModel` not being a registered model type. If so, set `is_decoder_model=True` explicitly in `LoraConfig` or pass `target_modules` by exact name match (which I already do).

### Step 4: deployment behind env-var gate

Add to `/home/moloch/local_agent/ouro_backend.py`, inside `DeepThinkModel.load()` after the base model is on GPU:

```python
LORA_PATH = os.environ.get("LOCAL_AGENT_OURO_CHAT_LORA_PATH",
                          "/home/moloch/ouro_project/models/ouro_rltt_chat_lora")
if os.environ.get("LOCAL_AGENT_OURO_CHAT_LORA", "").strip() in ("1", "true", "yes"):
    try:
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
        self.model.eval()
        _log(f"{ANSI_GREEN}[Ouro] chat LoRA adapter loaded from {LORA_PATH}{ANSI_RST}")
    except Exception as exc:
        _log(f"{ANSI_YELLOW}[Ouro] chat LoRA load failed ({exc}); continuing with base model{ANSI_RST}")
```

Default: env var unset → base model only → Hunter Seeker and DSA paths see exactly the same weights they see today.

When debugging conversational quality, run `LOCAL_AGENT_OURO_CHAT_LORA=1 ./tools/launch_ouro_ui.sh` (or whatever the launch incantation is). When debugging DSA / Hunter Seeker, leave it unset.

The `peft` import is intentionally inside the try block — `local_agent`'s inference venv doesn't need `peft` installed unless the user opts in via the env var. If they do, `pip install peft` into the inference venv (small, no torch upgrade required).

### Step 5: validation checklist (mandatory before declaring done)

| Check                                                                   | Pass criterion                                                  |
|-------------------------------------------------------------------------|------------------------------------------------------------------|
| DSA harness with adapter OFF                                            | 4/4 pass (same as today's baseline)                              |
| DSA harness with adapter ON                                             | 4/4 pass (regression check)                                      |
| Conversation smoke with adapter ON                                      | Greeting produces 1–3 sentence first-person reply, no fake turns |
| Conversation smoke with adapter ON                                      | `<thinking>` block is short (≤ ~80 tokens) or absent             |
| Hunter Seeker (claude_sandbox) smoke                                    | Any one wa30 ladder run completes without VRAM regression        |
| Memory: adapter dir size                                                | ≤ 100 MB on disk                                                 |
| Re-run wrapper unit tests (187 wrapper+self-model tests, per session_context note) | All still pass |

Commands:

```bash
# DSA with adapter OFF (baseline)
unset LOCAL_AGENT_OURO_CHAT_LORA
source /home/moloch/ouro_project/venv/bin/activate
python /home/moloch/ouro_project/tools/test_local_agent_dsa_coding.py \
    --mode direct_first --ut-steps 2 --max-tokens 512

# DSA with adapter ON
LOCAL_AGENT_OURO_CHAT_LORA=1 python /home/moloch/ouro_project/tools/test_local_agent_dsa_coding.py \
    --mode direct_first --ut-steps 2 --max-tokens 512

# Conversation smoke
LOCAL_AGENT_OURO_CHAT_LORA=1 python -c "
from ouro_agent_improved import create_agent_runtime, run_task_mode
mm, qm, _, cls, proj, sm, hist = create_agent_runtime()
out = run_task_mode(mm, qm, proj, hist,
    'Hello Ouro, I thought it was about time to actually talk with you.',
    mode='auto', classifier=cls, self_model=sm, max_tokens=384)
print(out)
"

# Hunter Seeker smoke — pick the cheapest existing wa30 smoke target.
# (Need to confirm the canonical command with the user; the session_context note
#  mentions 'wa30 post-veto 8-run ladder' as the active line but that's a full
#  ladder, not a smoke. Likely `python claude_sandbox/sandbox_sweep_validate.py`
#  or a single-checkpoint live diagnostic; ask before running.)
```

If DSA regresses with adapter ON: do NOT enable the env var by default. Investigate by (a) reducing lr to 1e-4 and retraining; (b) tightening the dataset filter; (c) checking if any LoRA target overlaps a layer Hunter Seeker depends on.

## Risks and rollback

| Risk                                          | Mitigation                                                                 |
|-----------------------------------------------|----------------------------------------------------------------------------|
| Catastrophic forgetting on DSA                | Conservative LoRA + small data + adapter gate. Worst case: don't enable.   |
| Hunter Seeker depends on specific layer outputs | Adapter only loads when env var set. Hunter Seeker runs untouched.        |
| HF Trainer incompatible with custom OuroModel | Watch first 10 steps for `loss=None`. If broken, write a 30-line manual loop. |
| OOM during training                           | Lower `MAX_LEN` to 768; drop `gradient_accumulation_steps` to 4.            |
| Tokenizer chat-template subtly wrong          | Sanity-check by tokenizing one sample and decoding the assistant span.     |
| OASST1 license / content                      | Apache-2.0; safe for local research use. No redistribution implied.        |
| Model identity drift ("I am ChatGPT")         | Add 10–20 hand-written identity examples to `extras.jsonl`.                |

**Full rollback** (worst case): `rm -rf /home/moloch/ouro_project/models/ouro_rltt_chat_lora`, unset env var. Base model and wrapper are untouched.

## Open questions to resolve when resuming

1. **Hunter Seeker smoke command** — what's the cheapest way to confirm "still works"? The session_context note mentions a wa30 ladder, but that's a full run. Need the canonical 5-minute probe.
2. **Identity examples** — should the model self-identify as "Ouro", "Ouroboros", or something else when asked? Pick before writing `extras.jsonl`.
3. **Eval beyond DSA** — is there an existing chat-quality eval in the project, or should we add a small one? (10 prompts, manual rubric.)
4. **Backups before training** — none currently in plan. The base model dir is read-only-ish (we never write to it), and the adapter goes to a new dir, so the base is safe. If paranoid, `cp -r models/ouro_rltt_local models/ouro_rltt_local.bak` is ~5 GB.

## Files this plan will touch when executed

```
NEW: /home/moloch/ouro_project/venv_sft/                    (sibling venv)
NEW: /home/moloch/ouro_project/tools/prep_ouro_chat_sft.py
NEW: /home/moloch/ouro_project/tools/train_ouro_chat_lora.py
NEW: /home/moloch/ouro_project/data/ouro_chat_sft/{train,val}.jsonl
NEW: /home/moloch/ouro_project/models/ouro_rltt_chat_lora/  (LoRA adapter)
MOD: /home/moloch/local_agent/ouro_backend.py               (env-var gated PEFT load)
```

No base-model weights are modified. No inference-venv dependency changes (peft only installs into the inference venv if the user opts in by setting `LOCAL_AGENT_OURO_CHAT_LORA=1` and pip-installs peft).

## Resume cheat sheet

When picking this up, in order:

```bash
# 1. Confirm wrapper state is still clean
cd /home/moloch/local_agent && git status   # if under version control
source /home/moloch/ouro_project/venv/bin/activate
python tools/test_local_agent_dsa_coding.py --mode direct_first --ut-steps 2 --max-tokens 512
# Expect: 4/4 pass.

# 2. Build sibling venv (Step 1 above)

# 3. Prep data (Step 2)
source /home/moloch/ouro_project/venv_sft/bin/activate
python /home/moloch/ouro_project/tools/prep_ouro_chat_sft.py
wc -l /home/moloch/ouro_project/data/ouro_chat_sft/{train,val}.jsonl

# 4. Train (Step 3)
python /home/moloch/ouro_project/tools/train_ouro_chat_lora.py 2>&1 | tee /tmp/ouro_chat_sft.log

# 5. Validate (Step 5)

# 6. If everything passes, document the run in a new SESSION_CONTEXT_<date>.md.
```

---

# Phase 2 — Hard-coding LoRA (advanced algorithms)

Added 2026-05-11 after the user asked: *"I would also like to train it on a lot of this level of coding. Id probabably do them separately? As long as we keep the model working as it should."*

This is a **second, independent adapter**, not a merge into the chat LoRA. Trained, gated, and validated separately so the chat-LoRA roll-forward stays clean and Hunter Seeker / DSA can run with neither, either, or both adapters loaded.

## Why this phase exists

Per `/home/moloch/local_agent/PROJECT_STATE_LOCAL_AGENT.md:3608–3679`:

- The post-audit devils harness produces `outcome_class=honest_abstain` on `offline_dynamic_connectivity` and `minimum_xor_paths`.
- The wrapper is no longer the bottleneck. The residual blocker is **route-to-code execution**: even when a correct algorithmic route is supplied, the 2.67 B RLTT produces syntactically broken or semantically wrong implementations.
- Listed model-level remedies (line 3668–3671): retrain on adversarial route-to-code (segment-tree-over-time + rollback DSU, cycle-XOR linear basis, algebraic-walk decomposition); constrained / contrastive decoding; small draft-and-verify model.

User's framing: don't narrow this to the two devils tasks. The goal is to lift the model's competence across the **whole class of advanced algorithms** that currently sit beyond the 2.67 B's effective scale — not to memorize two solutions.

## Scope of "this level of coding"

Codeforces ≈ 2200+ / ICPC regional / IOI level. Target families (the breadth matters more than which specific problems we pick within each):

1. **Union-Find variants** — rollback DSU, persistent DSU, weighted / parity DSU, small-to-large merge.
2. **Segment-tree variants** — lazy propagation, persistent, **segment tree over time**, segment-tree beats, merge-sort tree, segment tree on Euler tour, Li Chao.
3. **Linear algebra over GF(2)** — **XOR linear basis**, Gauss–Jordan over bits, cycle-space arguments on graphs.
4. **Tree algorithms** — heavy-light decomposition, centroid decomposition, auxiliary trees / virtual trees, LCA via binary lifting and Euler tour.
5. **String algorithms** — suffix array + LCP, suffix automaton, Z-function, Aho–Corasick, palindromic tree, hashing.
6. **Graph theory beyond shortest path** — Dinic / ISAP max flow, min-cost flow, 2-SAT, dominator tree, biconnected components + block-cut tree, Eulerian circuit, Hopcroft–Karp, Hungarian.
7. **Offline query processing** — Mo's algorithm (with updates / on trees), CDQ divide-and-conquer, parallel binary search.
8. **DP optimization** — convex hull trick, Knuth's optimization, divide-and-conquer optimization, monotonic deque, bitmask DP, digit DP.
9. **Balanced BSTs / advanced structures** — treap (rotation and merge–split), splay, link–cut tree, Fenwick-on-Fenwick.
10. **Number theory** — extended Euclidean, CRT, Lucas, BSGS, Pollard ρ, Miller–Rabin, Möbius inversion.
11. **Polynomial / signal** — NTT, FFT convolution, polynomial multiplication mod p, multipoint evaluation (optional, larger code).

We are **not** trying to make Ouro a competitive programmer. We're trying to teach it that *when given a route, the corresponding code is a finite, learnable pattern.* That's why the data shape matters as much as the topic mix.

## Data plan

**Target volume:** 300–500 examples, distributed across the 11 families above (no family > ~15 % of the set). Larger than the chat phase because the surface is larger; still small enough to remain a style nudge.

**Per-example shape (mandatory, follows PROJECT_STATE line 3672–3679's brute-force-first curriculum):**

```python
# Route card (system or context turn): one-paragraph identification of the
# algorithmic family and the invariant the optimized solution relies on.

# Assistant turn (training target): two functions in one block.
def brute_force_<name>(<args>):
    # Obviously-correct O(n^k) implementation. No cleverness.
    ...

def <name>(<args>):
    # Optimized implementation matching the route card.
    ...

# Differential check on small random inputs:
if __name__ == "__main__":
    import random
    rng = random.Random(0)
    for _ in range(200):
        x = _random_small_case(rng)
        assert <name>(*x) == brute_force_<name>(*x), x
```

The brute-force half is what makes the example *learnable* and *checkable* at train time:

- It teaches the model that producing a checker is part of the answer, not optional.
- It gives our data-prep script a way to **filter the dataset** before training: any example whose own differential check fails on 200 small inputs is dropped. No silent garbage in the training set.
- It generalizes the wrapper-level brute-force-first curriculum down to weights, so the same shape works whether the wrapper is gating us or not.

**Sources (preferred → fallback):**

1. **CodeContests** (DeepMind, on HuggingFace `deepmind/code_contests`) — Codeforces / AtCoder problems with multiple solutions. Filter for difficulty ≥ 2200 *and* presence of an accepted Python or C++ solution. Translate C++ solutions to Python only when no Python is present; treat translated examples as a separate `provenance=translated` set so we can audit later.
2. **TACO** (`BAAI/TACO`) — competitive-programming corpus with metadata. Filter to `difficulty in {hard, very hard}`. ~ 200 examples max from here; quality is variable.
3. **Hand-curated** — for the families where (1) and (2) underrepresent (typically link–cut tree, persistent DSU, Aho–Corasick), write 5–10 canonical examples ourselves using KACTL / CP Handbook as reference. Hand-written examples should anchor each family with at least one known-good template.
4. **Synthetic augmentation** — for each curated example, optionally generate 1–2 small variations (different parameter names, different small case generator) via a strong external model, **gated through the differential-check filter**. Skip entirely if the curated set already gives us 300+ examples.

**Audit gate (mandatory before training):**

- Every example must pass its own embedded `assert` block on a fresh interpreter — i.e. the data-prep script runs each example in a sandboxed subprocess with a 10 s wallclock budget and discards failures.
- Spot-check 30 random survivors by hand for: route-card accuracy, brute-force actually-brute (no smuggled cleverness), variable-name consistency.

## Training plan

Same conservatism as Phase 1, with three deltas:

| Knob                          | Phase 1 (chat) | Phase 2 (hard-code) | Reason                                                                 |
|-------------------------------|----------------|----------------------|------------------------------------------------------------------------|
| LoRA rank                     | 8              | 8                    | Same — minimize drift.                                                 |
| Target modules                | attn + MLP     | attn + MLP           | Same. Do not touch embeddings / lm_head.                               |
| Epochs                        | 2              | **1**                | Larger dataset → fewer epochs to keep total update budget similar.     |
| Examples                      | ~500           | 300–500              | Comparable. Style nudge, not a re-pretrain.                            |
| Max sequence length           | ~1024          | **2048**             | Brute-force + optimized + tests routinely exceed 1024 tokens.          |
| Gradient accumulation         | 8              | **16**               | 2× sequence length → halve micro-batch → double accum to keep effective batch the same. |
| LR                            | 2e-4           | 2e-4                 | Same.                                                                  |
| Warmup                        | 3 % of steps   | 3 % of steps         | Same.                                                                  |
| Adapter path                  | `models/ouro_rltt_chat_lora/` | `models/ouro_rltt_code_lora/` | Separate dir, separate adapter, never co-located.       |

**No overlap with Phase 1's data.** Chat data goes through Phase 1's prep; code data goes through Phase 2's. We never train one adapter on a mix of the other adapter's data — that's what keeps them independently rollback-able.

## Adapter coexistence

Two env vars, both default off:

```bash
LOCAL_AGENT_OURO_CHAT_LORA=1     # Phase 1 adapter
LOCAL_AGENT_OURO_CODE_LORA=1     # Phase 2 adapter
```

Backend behavior in `ouro_backend.py:DeepThinkModel.load()`:

- Neither set → base model only. **Hunter Seeker mode** (and the current DSA harness path).
- Chat only → load chat LoRA via `PeftModel.from_pretrained`.
- Code only → load code LoRA via `PeftModel.from_pretrained`.
- Both → load chat first, then `model.load_adapter(<code_dir>, adapter_name="code")` and `model.set_adapter(["chat", "code"])` so PEFT activates both (additive low-rank deltas — supported since `peft>=0.7`). If at runtime the user's task profile is `conversation`, the wrapper can call `set_adapter("chat")` to deactivate code; for everything else, both stay on.

**Hunter Seeker is unaffected** because neither env var defaults on. The wa30 ladder runs base-model-only unless explicitly opted in.

## Validation checklist (mandatory before declaring done)

```bash
# 0. Baseline (no adapters): confirm DSA 4/4 + devils 2/2 honest_abstain still holds.
unset LOCAL_AGENT_OURO_CHAT_LORA LOCAL_AGENT_OURO_CODE_LORA
python tools/test_local_agent_dsa_coding.py --mode direct_first --ut-steps 2 --max-tokens 512
python tools/test_local_agent_hard_coding.py --full-capability --ut-steps 2 \
    --max-tokens 768 --react-steps 3 --repair-rounds 2 --task-wallclock-sec 360

# 1. Chat-only (Phase 1 baseline regression check).
LOCAL_AGENT_OURO_CHAT_LORA=1 python tools/test_local_agent_dsa_coding.py ...
# Expect: DSA still 4/4. Conversation smoke still clean.

# 2. Code-only.
LOCAL_AGENT_OURO_CODE_LORA=1 python tools/test_local_agent_dsa_coding.py ...
# Expect: DSA still 4/4. (If DSA regresses, the code adapter is too aggressive — back off.)
LOCAL_AGENT_OURO_CODE_LORA=1 python tools/test_local_agent_hard_coding.py ...
# Acceptance gate: at minimum, transition from honest_abstain → has_code=true on at
# least one of the two devils tasks, without the wrapper surfacing oracle leakage.
# Stretch goal: one of the two passes.

# 3. Both adapters.
LOCAL_AGENT_OURO_CHAT_LORA=1 LOCAL_AGENT_OURO_CODE_LORA=1 \
    python tools/test_local_agent_dsa_coding.py ...
# Expect: DSA still 4/4. Conversation smoke still clean. Hard harness same-or-better than (2).

# 4. Hunter Seeker smoke (env vars unset).
# Need the canonical cheap probe — same open question as Phase 1.
```

**Pass criteria (in order of importance):**

1. Base + chat-only DSA both stay 4/4. Non-negotiable.
2. Code-only DSA stays 4/4. If it regresses, the adapter is mis-trained — bisect: fewer steps, lower rank to 4, or drop the synthetic-augmentation slice.
3. Code-only devils harness moves *at minimum* from `honest_abstain` to `has_code=true,failed` on one task. Pure pass is a stretch goal, not a release gate.
4. Both-adapter combined run shows no new regression vs. each-alone.
5. Hunter Seeker smoke clean.

## Risks specific to Phase 2

| Risk                                                     | Mitigation                                                                                       |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Catastrophic forgetting on standard DSA                  | r=8, 1 epoch, validation gate (2) above. If it triggers, lower rank or drop epochs.              |
| Hard-code adapter teaches *wrong* solutions              | Differential-check filter at data-prep time + manual spot-check of 30 random survivors.          |
| Adapter stack (chat + code) interferes destructively     | Validation gate (3) above. Worst case: ship them as mutually exclusive instead of additive.      |
| Hunter Seeker indirectly affected via shared weights     | Hunter Seeker runs with env vars unset → base model only. The check is to *prove* that, not assume. |
| Sequence length 2048 OOMs on 12 GB                       | Drop to 1536 first; only then drop micro-batch and double accum again.                           |
| Synthetic / translated examples leak benchmark-like text | Provenance-tag synthetic and translated examples; if validation gate (3) shows benchmark gaming, hold them out and retrain. |
| Training-data licenses                                   | CodeContests is Apache-2.0; TACO is CC BY-NC 4.0 (research only; OK for local). Confirm before any redistribution. |

**Rollback:** `rm -rf /home/moloch/ouro_project/models/ouro_rltt_code_lora`, unset `LOCAL_AGENT_OURO_CODE_LORA`. Base model and chat adapter are untouched. The two adapters are independent.

## Open questions to resolve before starting Phase 2

1. **Volume ceiling** — 300, 500, or push to 1000? Recommend starting at 300 and only adding more if validation gate (3) underperforms.
2. **Translated C++ → Python solutions** — include from the start, or hold out as a v2 augmentation? Recommend: hold out. The pure-Python CodeContests slice should be enough for v1.
3. **Synthetic augmentation strength** — none, light (1 variation per example), or aggressive (3+)? Recommend: none for v1. Add only if the curated set under-fills a target family.
4. **Per-family balance** — 11 families × ~30 examples = ~330. Force equal-ish coverage, or weight by family difficulty / current failure rate? Recommend equal for v1; iterate after we see which families improve.
5. **Decoding-time companions** — constrained / contrastive decoding (PROJECT_STATE line 3670) are independent of training. Should we attempt one alongside the LoRA, or finish the adapter first and revisit? Recommend: adapter first; decoding-time is a Phase 3.

## Files Phase 2 will touch when executed

```
NEW: /home/moloch/ouro_project/tools/prep_ouro_code_sft.py
NEW: /home/moloch/ouro_project/tools/train_ouro_code_lora.py
NEW: /home/moloch/ouro_project/data/ouro_code_sft/{train,val}.jsonl
NEW: /home/moloch/ouro_project/data/ouro_code_sft/audit/   (failed-filter examples, kept for review)
NEW: /home/moloch/ouro_project/models/ouro_rltt_code_lora/
MOD: /home/moloch/local_agent/ouro_backend.py  (add second env-var gate; multi-adapter load)
```

No changes to Phase 1 files. The Phase 1 chat adapter is a hard prerequisite *only* in the sense that we want to validate (3) "both adapters loaded" — Phase 2 can be trained and validated standalone first.

## Resume cheat sheet — Phase 2

```bash
# 0. Phase 1 must be done first (chat adapter trained + validated).
ls /home/moloch/ouro_project/models/ouro_rltt_chat_lora/adapter_config.json

# 1. Reuse the same sibling venv (peft / trl / datasets already installed).
source /home/moloch/ouro_project/venv_sft/bin/activate

# 2. Prep data (writes train.jsonl + val.jsonl + audit/).
python /home/moloch/ouro_project/tools/prep_ouro_code_sft.py
wc -l /home/moloch/ouro_project/data/ouro_code_sft/{train,val}.jsonl
ls /home/moloch/ouro_project/data/ouro_code_sft/audit/ | wc -l   # failed-filter count

# 3. Manual spot-check 30 random survivors before training.
python /home/moloch/ouro_project/tools/prep_ouro_code_sft.py --spot-check 30

# 4. Train (~ longer than Phase 1 because of 2048 seq len; budget 2-4h).
python /home/moloch/ouro_project/tools/train_ouro_code_lora.py 2>&1 \
    | tee /tmp/ouro_code_sft.log

# 5. Validate using the four gates above, in order. Stop at the first failure.

# 6. Document the run in SESSION_CONTEXT_<date>.md alongside the Phase 1 entry.
```

Expected total wall-clock (with the user present): 1.5–2.5 hr. Most of that is training + waiting; data prep is ~5 min, validation is ~10 min.

---

# Phase 3 — Full RLTT continuation (primary plan)

Added 2026-05-11. The user opted for renting multi-GPU briefly and pointed at the FSDP shards in `~/Downloads/RLTT/Downloads_RLTT/RLTT/`. The user also pointed at their own pairwise evaluator (`artifacts/checkpoints/evaluator/pairwise_epoch2.pt`), which is the **arxiv:2604.09870 architecture** ("Relational Preference Encoding in Looped Transformer Internal States"), trained on HH-RLHF to 94.98 % accuracy on 8,552 test examples. The user notes it also tracks correctness on math tasks — extending the paper's pure-preference framing.

This phase supersedes Phases 1+2 if it runs successfully. Phases 1+2 remain the fallback if the rented run is blocked.

## Why this supersedes LoRA-SFT

| Concern                                                                | LoRA-SFT path (Phases 1+2)                             | Full RLTT (Phase 3)                                                       |
|------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------------|
| Trains the looped structure?                                           | No. Final-output loss only. All 4 loops get the same gradient. | Yes. Per-loop reward weighted by exit_pdf. Matches original training shape. |
| Continues from real optimizer state?                                   | No. LoRA params have fresh Adam moments.               | Yes. `optim_world_size_4_rank_*.pt` resume directly.                      |
| Reward fidelity                                                        | SFT has no reward; targets are just next-token labels. | Evaluator at 94.98 % accuracy (chat) + exact diff-check (code).           |
| Needs preference labels?                                                | OASST1 ships rankings, but Phase 1 SFT doesn't use them. | No. Evaluator gives labels on the fly from any prompt corpus.             |
| Risk of distribution drift                                             | Low (LoRA is reversible, frozen base).                 | Higher (full weights move). Mitigation: KL regularization against base.   |
| Hardware                                                               | Single 12 GB laptop.                                   | Rented 4× GPU box (matches `world_size=4` in `fsdp_config.json`).         |
| Hunter Seeker untouched?                                                | Yes — adapter env-var-gated, base weights frozen.      | **No** — base weights move. Pre-flight: snapshot base + run Hunter Seeker before & after; revert if regression. |

The "base weights move" point matters: Phase 3 produces a **new base checkpoint**, not an adapter. The original `models/ouro_rltt_local/` is preserved as `models/ouro_rltt_local_pre_phase3/` (renamed, not deleted). The Phase 3 output lands in `models/ouro_rltt_local_phase3/`. Hunter Seeker, the local agent wrapper, and the inference UI all keep pointing at the *original* path by default; the new checkpoint is opted into via an env var, the same gating pattern Phases 1+2 use for the LoRA adapters.

## What's in the Downloads checkpoint

Inventory at `/home/moloch/Downloads/RLTT/Downloads_RLTT/RLTT/`:

```
model_world_size_4_rank_{0,1,2,3}.pt    ~2.67 GB × 4   FSDP-sharded fp32 weights, ranks 0-3
optim_world_size_4_rank_{0,1,2,3}.pt    ~1.35 GB × 4   FSDP-sharded Adam moments (m, v)
extra_state_world_size_4_rank_{0,1,2,3}.pt  ~15 KB × 4 RNG state, scheduler step, etc.
fsdp_config.json                        46 B           {"FSDP_version": 1, "world_size": 4}
consolidated.pt                         2.67 GB        Single-rank bf16, inference-ready
consolidated_clean.pt                   10.7 GB        Single-rank fp32, cleaned base
huggingface/                                           HF-format export (config + modeling + tokenizer)
huggingface-20260504T064604Z-3-001.zip  ~1 MB          HF export archive
```

Dates: shards are 2026-05-04 (training output day); `consolidated_clean.pt` is 2026-05-10 (cleaned single-rank consolidation, the artifact `models/ouro_rltt_local/` was built from).

The shards encode **FSDP v1 world_size=4**. To resume training without rewriting, the rented box should also be 4× GPU. Other world sizes need a `torch.distributed.checkpoint` reshard step, which works but adds complexity. Start with 4× to match.

## Hardware and the rented box

| Provider option        | GPUs                | $/hr approx. | Notes                                                                    |
|------------------------|---------------------|--------------|--------------------------------------------------------------------------|
| RunPod 4×A100 80GB     | 4× A100 80 GB       | ~$6.50/hr    | World size matches; per-GPU 80 GB lets us drop activation checkpointing if we want. |
| Lambda 4×H100 80GB     | 4× H100 80 GB       | ~$13/hr      | Faster, ~2× throughput vs. A100. Best if total wall-clock budget is tight. |
| RunPod 4×L40S 48GB     | 4× L40S 48 GB       | ~$4/hr       | Cheapest 4× option. Fits 2.67 B + FSDP + optim state comfortably.        |

Recommended: **4× L40S 48 GB on RunPod** for the first run. ~$4/hr × ~6 hr ≈ $24 for a complete Phase 1 chat-RLTT pass; ~$50–$80 for the longer Phase 2 code-RLTT pass. Total cost: under $100 for both phases including ample buffer.

Memory math per GPU at world_size=4:
- 2.67 B params / 4 ranks × 4 bytes (fp32 weights) = ~2.67 GB per rank
- Adam state ≈ 2× params / 4 ranks = ~5.4 GB per rank (m + v in fp32)
- Activations at seq_len=2048, batch=2 per rank, 4 UT loops: ~10–14 GB per rank (loop-level checkpointing enabled, as `rltt_loop_level_checkpointing=True` in config)
- Total: ~20–25 GB per rank → fits 48 GB L40S with margin, fits 80 GB A100 easily.

## Data and reward design

### Phase 3a — Chat sub-phase

**Prompt source (per user decision 2026-05-11):** OASST1 user turns only. Ignore OASST1's preference labels. OASST1's labels are reserved for the pre-flight evaluator validation (see below), not for training.

Filtering: English, prompter turn with at least one rated assistant child (proxy for "real prompt that humans engaged with"), length 20–600 chars. Target ~5,000 unique prompts. We can re-sample many times; the dataset is the prompt corpus, not the response pairs (which are generated on the fly).

**Reward:** frozen `pairwise_epoch2.pt` evaluator. Per training step:

```
1. Sample prompt p from the prompt corpus.
2. Generate two responses (r_A, r_B) from the current policy with different
   temperature seeds. Same model, same step — both reflect current policy.
3. Run a forward pass on (p + r_A) and (p + r_B) to capture per-loop hidden
   states [h^1, h^2, h^3, h^4] for each. Capture is via the existing
   evaluator hook used in tests/manual/evaluate_pairwise_rltt.py.
4. score = evaluator(H_A, mask_A, H_B, mask_B)   ∈ ℝ
   target = +1 if score > 0 else -1   (which response the evaluator prefers)
5. Per-trajectory advantage for RLTT:
     winner gets +|score|, loser gets -|score|
   This is RLAIF-style — labels come from the AI evaluator, not from humans.
6. RLTT loss: standard policy-gradient with per-loop exit_pdf weighting,
   advantage as above. KL against a frozen reference copy of the pre-Phase-3
   base, coefficient 0.05.
```

The KL term is the safety belt: it stops the policy from drifting into "the evaluator's gameable surface" — i.e. wireheading on whatever artifact the evaluator picks up but humans wouldn't endorse.

### Phase 3a pre-flight: evaluator validation

Before any training, score 200 random OASST1 (chosen, rejected) pairs with `pairwise_epoch2.pt`. Compute agreement rate against OASST1's labels.

| Result                | Action                                                                    |
|-----------------------|---------------------------------------------------------------------------|
| ≥ 80 % agreement      | Ship. Evaluator transfers from HH-RLHF → OASST1 well enough for v1.       |
| 65 – 80 % agreement   | Train a thin adapter on top of the evaluator on OASST1 pairs (~1 hr).     |
| < 65 % agreement      | Re-train the evaluator from scratch on OASST1 (a few hours, separate run). |

This is a 30-line script using the existing `tests/manual/evaluate_pairwise_rltt.py` path with OASST1 instead of HH-RLHF.

### Phase 3b — Hard-coding sub-phase

**Prompt source:** the 300–500 hand-curated + CodeContests-filtered examples from Phase 2 (route-card + brute-force template + optimized template + diff-assertions). We use only the *prompt* side (the route card / problem statement). Responses are generated by the policy at training time.

**Reward (blended, per user decision):**

```
For each generated response r given prompt p:
  1. Parse r for `def <name>(` and `def brute_force_<name>(` blocks.
     If parsing fails → reward = -1.0  (hard failure, no syntactic code).
  2. Execute the differential check on 200 random small inputs in a
     sandboxed subprocess with 10 s wallclock.
     pass_rate ∈ [0, 1] = (# passed) / 200.
  3. diff_check_reward = 2 · pass_rate - 1   ∈ [-1, +1]   (primary signal).
  4. Generate a paired alternative response r' (different seed) and score
     the pair through the evaluator:
       eval_score = evaluator(loops(p+r), loops(p+r'))
     This tests whether the evaluator agrees with diff-check on code,
     which is the user's "evaluator reads loop convergence" hypothesis.
  5. Blended advantage:
       advantage = 0.8 · diff_check_reward + 0.2 · tanh(eval_score)
  6. RLTT loss + KL regularization, same shape as Phase 3a.
```

The 0.8 / 0.2 split keeps the differential check as the primary gradient driver — that's the signal we trust on code — while letting the evaluator contribute. **Logged side-by-side, the two signals' correlation is the test of the user's loop-convergence hypothesis.** If they correlate ≥ 0.6 on code, the user's intuition is vindicated and we could promote the evaluator weight later. If they correlate < 0.3, the evaluator isn't actually reading what we thought; back off to differential-check only.

This is built-in eval, not a separate experiment. Every training step contributes one data point to the correlation.

## Training script structure

```
/home/moloch/ouro_project/tools/train_ouro_rltt_phase3.py
  ├── load FSDP shards or consolidated.pt (--init-from {shards,consolidated})
  ├── frozen reference copy of base for KL regularization
  ├── frozen evaluator (PairwiseEvaluator + pairwise_epoch2.pt)
  ├── reward fn factory:
  │     phase3a_chat_reward(prompts, policy, evaluator, ref_policy)
  │     phase3b_code_reward(prompts, policy, evaluator, diff_check_runner)
  ├── per-loop weighted policy gradient loss (custom; not stock PPO)
  └── checkpoint to models/ouro_rltt_local_phase3/ every N steps
```

This is **not** a stock `trl.PPOTrainer` job. PPO trains on final logits; RLTT needs per-loop logits weighted by exit_pdf. The model already returns `OuroRLTTCausalLMOutput` with per-loop information (see `huggingface/modeling_ouro.py`), so the loss is custom but small (~50 lines around the existing model output).

Training config baseline:
- Phase 3a chat: 2,000 steps × batch 16 effective, ~6 hr on 4× L40S.
- Phase 3b code: 5,000 steps × batch 8 effective (longer sequences), ~16–20 hr on 4× L40S.
- LR: 5e-6 (full-model RLTT, much lower than LoRA's 2e-4).
- KL coefficient: 0.05, adaptive (raise if KL > 10, lower if < 1).

## Deployment

The Phase 3 output is a new full checkpoint, not an adapter. Deployment is a path swap:

```bash
# Default: original base (Hunter Seeker, DSA harness, everything stays on this).
ls /home/moloch/ouro_project/models/ouro_rltt_local/

# Opt-in: Phase 3 checkpoint.
export OURO_MODEL_PATH=/home/moloch/ouro_project/models/ouro_rltt_local_phase3
# (or whatever env var ouro_backend.py:DeepThinkModel.load() reads; if no
#  such var exists yet, add one — pattern matches the Phase 1/2 env vars.)
```

The original `ouro_rltt_local/` directory is **renamed to `ouro_rltt_local_pre_phase3/`** before deployment so it can never be silently overwritten. The new checkpoint goes to a new path. Reverting = path swap, no weight surgery.

## Validation gates

Same as Phase 1/2 but on the full checkpoint instead of the adapter:

```bash
# Gate 0: original base still passes DSA 4/4 (sanity that nothing was overwritten).
ls /home/moloch/ouro_project/models/ouro_rltt_local_pre_phase3/
OURO_MODEL_PATH=…_pre_phase3 python tools/test_local_agent_dsa_coding.py ...

# Gate 1: Phase 3 checkpoint passes DSA 4/4.
# If it regresses, the RLTT pass damaged general coding ability. Stop, debug.
OURO_MODEL_PATH=…_phase3 python tools/test_local_agent_dsa_coding.py ...

# Gate 2: conversation smoke clean on Phase 3 checkpoint.
# The fix the wrapper papers over should now be solved at the weight level —
# no third-person CoT, no fake **Ouros:** turns, sensible greetings.

# Gate 3: devils harness on Phase 3 checkpoint.
# Acceptance: at minimum, honest_abstain → has_code=true on one task.
# Stretch: pass on one task.
OURO_MODEL_PATH=…_phase3 python tools/test_local_agent_hard_coding.py ...

# Gate 4: Hunter Seeker smoke on Phase 3 checkpoint.
# This is the highest-stakes gate. Hunter Seeker depends on specific loop
# behavior; if Phase 3 damaged it, that's the moment we revert.

# Gate 5: Phase 3 + evaluator/diff-check correlation log.
# Plot or summarize the eval_score vs diff_check_reward correlation logged
# during Phase 3b. ≥ 0.6 → evaluator-on-code hypothesis confirmed.
# < 0.3 → evaluator-on-code is weaker than chat; consider promoting diff-check
# weight in any future run.
```

**Release gate:** all of Gate 0, 1, 4 must pass. Gate 2 must show qualitative improvement (no fake turns). Gate 3 acceptance is the stretch goal; gate 5 is informational, not blocking.

## Risks specific to Phase 3

| Risk                                                       | Mitigation                                                                                       |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Full-weight drift damages Hunter Seeker                    | Snapshot pre-rename, env-var-gated deploy, hard rollback path. Gate 4 is the explicit check.    |
| Evaluator "wireheading" — policy games the evaluator       | KL regularization at 0.05 against frozen base. Spot-check 30 random generations by hand pre-deploy. |
| FSDP shards won't load on a different world size           | First test: load shards on 4× rented GPU in inference-only mode, verify forward pass matches `consolidated.pt`. If mismatch, abort and use `consolidated_clean.pt` as init (loses optim state but keeps weights). |
| Evaluator overfits to HH-RLHF distribution                 | Pre-flight validation on 200 OASST1 pairs (the gate before any training).                       |
| Diff-check timeout (10 s) too tight for some optimizations | Adjustable per-family; some segment-tree problems are genuinely slow at small N.                |
| Rented box billing                                         | Set a hard wall-clock cap (8 hr Phase 3a, 24 hr Phase 3b). Stop early if validation flatlines.  |
| Network upload / download of the 28 GB checkpoint           | Pre-stage shards to a long-lived RunPod volume. Don't re-upload per run.                        |

**Rollback:** `mv models/ouro_rltt_local_phase3 models/ouro_rltt_local_phase3.failed; unset OURO_MODEL_PATH`. Original base in `…_pre_phase3` becomes the default again. The Downloads shards remain pristine — never write back to them.

## Open questions to resolve before Phase 3

1. **Cloud provider account** — does the user have a RunPod or Lambda account? If not, ~10 min to set up before booking a box. (Just account + payment; the launch is one CLI command on RunPod.)
2. **Checkpoint upload route** — `rsync` over ssh, or upload to S3 / GCS first then pull on the box? 28 GB at residential upload is hours. Recommend: pre-upload to a long-lived RunPod network volume once, then re-attach for each run.
3. **Evaluator on math** — the user said it works on math but I haven't found a math-eval run analogous to `eval_pairwise_rltt_epoch2_full.json`. Pre-flight: run the evaluator on 200 (correct, incorrect) math-trace pairs. If accuracy ≥ 70 %, the loop-convergence hypothesis on math is corroborated. If lower, the code-phase 0.2 evaluator weight should be lowered or zeroed.
4. **Hunter Seeker smoke command** — still open from Phase 1. This becomes critical for Phase 3 Gate 4. Resolve before booking the box.
5. **Reference policy snapshot timing** — do we KL-regularize against the pre-Phase-3 base, or against a moving target (e.g., the policy from N steps ago)? Recommend pre-Phase-3 base for stability; cheaper and more robust.

## Files Phase 3 will touch when executed

```
NEW: /home/moloch/ouro_project/tools/validate_evaluator_on_oasst1.py
NEW: /home/moloch/ouro_project/tools/train_ouro_rltt_phase3.py
NEW: /home/moloch/ouro_project/tools/runpod_launch_phase3.sh
NEW: /home/moloch/ouro_project/data/ouro_rltt_phase3/prompts_chat.jsonl
NEW: /home/moloch/ouro_project/data/ouro_rltt_phase3/prompts_code.jsonl
NEW: /home/moloch/ouro_project/runs/phase3_eval_validation.json
NEW: /home/moloch/ouro_project/runs/phase3_correlation_log.jsonl
NEW: /home/moloch/ouro_project/models/ouro_rltt_local_phase3/        (new base)
REN: /home/moloch/ouro_project/models/ouro_rltt_local ->
     /home/moloch/ouro_project/models/ouro_rltt_local_pre_phase3/    (rename, atomic)
MOD: /home/moloch/local_agent/ouro_backend.py                        (read OURO_MODEL_PATH env var)
```

The Downloads checkpoint is never written to. Reads only. If we need a reshard, write the resharded artifact to a new path.

## Resume cheat sheet — Phase 3

```bash
# Day 0 — local (laptop)

# 1. Pre-flight: validate evaluator on OASST1.
source /home/moloch/ouro_project/venv/bin/activate
python /home/moloch/ouro_project/tools/validate_evaluator_on_oasst1.py \
    --evaluator-ckpt artifacts/checkpoints/evaluator/pairwise_epoch2.pt \
    --n-pairs 200 \
    --out runs/phase3_eval_validation.json
# Expect ≥ 0.80 agreement. If < 0.65, retrain evaluator first (separate run).

# 2. Build prompt corpora.
python /home/moloch/ouro_project/tools/prep_ouro_chat_sft.py --emit-prompts-only \
    --out data/ouro_rltt_phase3/prompts_chat.jsonl
python /home/moloch/ouro_project/tools/prep_ouro_code_sft.py --emit-prompts-only \
    --out data/ouro_rltt_phase3/prompts_code.jsonl

# 3. Stage the 28 GB FSDP shards + evaluator to RunPod network volume.
#    (One-time. Subsequent runs re-attach the volume.)
rsync -avP ~/Downloads/RLTT/Downloads_RLTT/RLTT/ \
    runpod:/workspace/ouro_rltt_base/
rsync -avP /home/moloch/ouro_project/artifacts/checkpoints/evaluator/pairwise_epoch2.pt \
    runpod:/workspace/evaluator/

# Day 1 — rented box (4× L40S, ~$4/hr)

# 4. Sanity load shards on rented box. Forward pass must match consolidated.pt within rtol=1e-3.
bash /home/moloch/ouro_project/tools/runpod_launch_phase3.sh sanity

# 5. Phase 3a — chat RLTT.
bash /home/moloch/ouro_project/tools/runpod_launch_phase3.sh chat \
    --steps 2000 --kl-coef 0.05 --lr 5e-6
# Wall-clock: ~6 hr. Checkpoint to models/ouro_rltt_local_phase3/.

# 6. Phase 3b — code RLTT.
bash /home/moloch/ouro_project/tools/runpod_launch_phase3.sh code \
    --steps 5000 --kl-coef 0.05 --lr 5e-6 --eval-weight 0.2
# Wall-clock: ~16-20 hr. Continues from end-of-3a checkpoint, same model dir.

# 7. Pull final checkpoint back to laptop.
rsync -avP runpod:/workspace/ouro_rltt_local_phase3/ \
    /home/moloch/ouro_project/models/ouro_rltt_local_phase3/

# Day 2 — laptop, validation + deploy

# 8. Rename old base, validate new one.
mv /home/moloch/ouro_project/models/ouro_rltt_local \
   /home/moloch/ouro_project/models/ouro_rltt_local_pre_phase3
# Then run all five validation gates above. STOP at the first failure.

# 9. If gates pass: leave models/ouro_rltt_local_phase3/ in place;
#    point inference at it via OURO_MODEL_PATH. Document in SESSION_CONTEXT_<date>.md.
#
#    If a gate fails: mv ouro_rltt_local_phase3 ouro_rltt_local_phase3.failed;
#    mv ouro_rltt_local_pre_phase3 ouro_rltt_local;
#    Phase 1+2 LoRA fallback path remains available.
```

Expected total cost: **~$80** for both sub-phases at 4× L40S, plus a few dollars in network volume storage. Expected total wall-clock with the user present: **~3 days** (Day 0 prep, Day 1 training, Day 2 validation/deploy). Most of Day 1 is "machine grinds, occasionally check loss curves."

---

# Phase 4 — Evaluator-in-loops: expand, train, deploy

Added 2026-05-11. The user reframed: *"Basically the phase 4 would then be rltt plus evaluator training on even more data, math, logic, coding, reasoning, thinking."* And: *"something I already considered and proved as possible is pushing the evaluator into the loops and have it actually diverge the thinking processes and choose the better branch."*

So Phase 4 is the actual implementation of arxiv:2604.09870's future-work line ("basal ganglia integration for real-time steering"), and it has three sub-phases that can run independently:

- **4a — Inference-only search wrapper** (immediate, free, runs on laptop). Lift the basin/branch mechanism from `claude_sandbox/arc_agent_pairwise_stockfish_codex.py` ("stockfish-style" search with the pairwise evaluator as heuristic) into a general-purpose wrapper. Works on any base checkpoint, including the current one. This is the "cheap proof on a real task" step.
- **4b — Evaluator expansion** (laptop, hours). Broaden the pairwise evaluator's training data beyond HH-RLHF to include math, logic, code, reasoning, and general thinking pairs. Retrain the same 4.7 M-param architecture; ship as `pairwise_phase4b.pt`. Higher-fidelity reward signal for everything downstream.
- **4c — In-loop RLTT distillation** (rented compute, AlphaZero-style). Phase 3 trained the policy on final-output reward from the evaluator. Phase 4c trains the policy to *internalize the search itself*: during training, the search loop produces a target trajectory, the policy is updated to predict the search's output without search. This is the "policy distillation of MCTS" pattern. Strictly stronger than Phase 3, strictly more expensive.

The dependencies are linear: 4a is free and standalone, 4b is the data step that 4c needs, 4c is the big training run. Skipping any of them is fine; doing all three in order gives the strongest result.

## Should the evaluator stay in the loops after Phase 4c trains?

User question, 2026-05-11. Recommendation: **yes, task-gated.** Defaults:

| Task type            | In-loop search after 4c | Why                                                                                |
|----------------------|--------------------------|------------------------------------------------------------------------------------|
| `conversation`       | OFF                      | Greetings don't need search. Keep latency low. Policy alone handles this well.     |
| `creative`           | OFF                      | Subjective quality; search might converge on evaluator-pleasing-but-bland.         |
| `code` / `dsa`       | ON                       | Diff-check + evaluator concur on quality; search catches genuine bugs.             |
| `hard_technical`     | ON                       | Devils-tier. Where search pays off most.                                           |
| `math_reasoning`     | ON                       | Evaluator-on-math is the whole point of Phase 4b's data expansion.                 |
| `project_navigation` | OFF                      | Tool-call routing, not deep reasoning. Search would add latency without value.     |

Five reasons for "yes, gated":

1. **Distillation never perfectly captures search.** Search-at-inference is a strictly larger compute budget than a single forward pass. There will always be hard cases where the policy's first guess is wrong and the in-loop evaluator catches it. Those are the cases that matter — novel inputs, devils-tier tasks, math at the edge of competence.
2. **Evaluator is task-agnostic; policy is task-specific.** After 4c the policy is shaped by its training distribution. The evaluator's relational diff-reading is upstream of any specific corpus. Stripping it loses generalization for tasks 4c didn't cover.
3. **Latency only matters where it matters.** Free on cheap tasks (search off), affordable on expensive tasks (already minutes per response).
4. **Defense in depth.** Frozen evaluator post-training = stable second opinion that doesn't drift with the policy.
5. **The one real risk is wireheading.** Policy produces outputs the evaluator rates high that humans wouldn't. Mitigation: spot-check 30 random outputs periodically. Maintenance pattern, not an architectural argument for removal.

Per-task budget becomes a `TaskProfile` field, same shape as `max_tokens` / `temperature`.

---

## Phase 4a — Inference-only search wrapper (immediate, free)

This is the **first thing to actually run**, before any data collection or rented compute. Works with any base checkpoint. Proves the search mechanism on chat / code / reasoning, not just ARC. No training, no rented compute.

### What this is, mechanically

The Ouro forward pass runs 4 Universal-Transformer loops over the same 48-layer block. In standard inference, those 4 loops are linear: `h^1 → h^2 → h^3 → h^4 → logits`.

Phase 4 inserts branch-and-evaluate operations inside that chain. After some loop K, the hidden state is **perturbed N ways** (different noise seeds, different temperature, different exit-PDF sampling), each branch continues forward to loop 4, and the pairwise evaluator scores the completed branches against each other. The winning branch's logits are used; the others are discarded.

This is a tree search where:
- **Nodes** are per-loop hidden states.
- **Branching** is whatever produces meaningful perturbation at that loop step (the existing PoC uses noise + different exit-PDF samples; could also use small attention-mask permutations or activation perturbations).
- **Heuristic** is `pairwise_epoch2.pt`, applied pairwise on completed children (tournament) or pairwise against the best-so-far (single-elimination).
- **Budget** is `root_width × branch_width^depth ≤ max_expansions`, matching the PoC's existing config knobs.

The crucial property: the evaluator's input is **already** per-loop hidden states. So scoring partial-then-completed trajectories is the operation it was designed for. Nothing about the evaluator architecture needs to change. We're just calling it inside the forward pass instead of after.

### Reference implementation

`claude_sandbox/arc_agent_pairwise_stockfish_codex.py` (~4,000 lines, the ARC agent). Search the file for `model_basin_diag_branch_width`, `model_basin_diag_root_width`, `model_basin_diag_max_expansions`, `expansions += len(entries_sorted)`, and `_basin_diag_step_with_search` (or whatever the relevant method is named — naming is `_basin_*` throughout). That's the existing tree-search loop.

What we need to do for Phase 4 is **lift the basin/branch mechanism out of the ARC agent** and into a general-purpose inference wrapper that any task type can use. The ARC-specific parts (grid manipulation, action-space-aware scoring) get dropped; the loop-level search machinery is what survives.

### Architecture and integration

```
/home/moloch/local_agent/ouro_inloop_search.py        (NEW)
  ├── BasinSearchConfig: branch_width, root_width, max_expansions, branch_after_loop
  ├── BasinSearchWrapper(base_model, evaluator, config)
  │     forward(input_ids, ...) → logits chosen by search instead of stock forward
  │     internally uses the Ouro per-loop output (OuroRLTTCausalLMOutput.exit_pdfs)
  │     and the frozen evaluator
  └── load_evaluator(ckpt_path) → PairwiseEvaluator (reuse claude_sandbox import)

/home/moloch/local_agent/ouro_backend.py              (MOD)
  if LOCAL_AGENT_OURO_INLOOP_SEARCH=1:
      model = BasinSearchWrapper(model, evaluator, config_from_env())
```

Gating: `LOCAL_AGENT_OURO_INLOOP_SEARCH=1`. Default off so Hunter Seeker, DSA, and chat inference are unaffected. Same env-var pattern as Phases 1+2 use for LoRA adapters.

**Task-aware budget** (recommended once we've measured latency):

| Task type        | Default config                                              | Rationale                                                                 |
|------------------|-------------------------------------------------------------|---------------------------------------------------------------------------|
| `conversation`   | `root=1, branch=1, expansions=1` (search effectively off)   | Greetings don't need search. Keep latency low.                            |
| `code` / `dsa`   | `root=2, branch=2, expansions=4`                            | Coding benefits from divergent attempts; budget bounded.                  |
| `hard_technical` (devils) | `root=2, branch=2, expansions=8`                   | Where search should pay off most. Higher budget acceptable.               |
| `math_reasoning` | `root=2, branch=2, expansions=4`                            | Evaluator works on math (per user); use it.                               |

Defaults match the PoC's `branch_width=1, root_width=1, max_expansions=4` for ARC, but with `branch=2, root=2` so there's actually search happening.

### Why 4a runs first

- Works with **any** base checkpoint, including the current `models/ouro_rltt_local/`.
- **Free** — no extra compute, no rented box, no training.
- **Reversible** — env var off → stock inference.
- **Composable** — stacks on Phase 1+2 LoRA adapters *and* on Phase 3's new base.
- **Already proven on ARC** — the PoC is real; the work is generalizing past ARC.

Running 4a first also tells us *whether 4c is worth the rented-compute spend*. If 4a-on-Phase-3-base produces big Gate 2 wins (devils harness moves), then 4c distillation is high-EV. If 4a produces marginal wins, 4c might be too. Either way, the empirical data from 4a sizes the bet.

### Validation gates

The Phase 4 wrapper is correct if it satisfies all of:

```bash
# Gate 0: search off → bit-exact match to stock inference.
unset LOCAL_AGENT_OURO_INLOOP_SEARCH
python tools/test_local_agent_dsa_coding.py ... --max-tokens 256 --seed 0 > /tmp/baseline.json
LOCAL_AGENT_OURO_INLOOP_SEARCH=1 LOCAL_AGENT_OURO_SEARCH_BUDGET_OVERRIDE='root=1,branch=1,expansions=1' \
    python tools/test_local_agent_dsa_coding.py ... --max-tokens 256 --seed 0 > /tmp/search_off.json
diff /tmp/baseline.json /tmp/search_off.json   # MUST be empty.

# Gate 1: search on, DSA harness still 4/4.
LOCAL_AGENT_OURO_INLOOP_SEARCH=1 python tools/test_local_agent_dsa_coding.py ...
# If DSA regresses, the search is producing worse outputs than stock — the
# evaluator is mis-ranking on this distribution. Stop, debug.

# Gate 2: search on, devils harness.
LOCAL_AGENT_OURO_INLOOP_SEARCH=1 python tools/test_local_agent_hard_coding.py ...
# Acceptance: improvement vs. baseline on either task — even just from
# honest_abstain → has_code=true counts as a win, same gate as Phase 3.

# Gate 3: Hunter Seeker smoke. Two configurations:
LOCAL_AGENT_OURO_INLOOP_SEARCH=0 python <hunter_seeker_smoke>   # baseline
LOCAL_AGENT_OURO_INLOOP_SEARCH=1 python <hunter_seeker_smoke>   # with search
# Decide whether Hunter Seeker should default search-on or search-off based
# on the comparison. (Open question: ARC PoC used search, so probably on for HS.)

# Gate 4: latency budget honored.
# Each call should produce a token-per-second measurement; with search at
# root=2 branch=2 expansions=4, expect ~3-5× slowdown vs. stock. Document.
```

**Release gate:** Gates 0, 1, 3 (baseline configuration) must pass. Gate 2 acceptance is the win-condition. Gate 4 is informational; the trade-off is task-dependent.

### Risks specific to Phase 4a

| Risk                                                            | Mitigation                                                                                       |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Latency blowup makes the wrapper unusable for chat              | Task-aware budget (chat = search off). Budget caps surface as env vars; user can tune.           |
| Evaluator scores wrong on out-of-distribution inputs            | Same evaluator validation gate as Phase 3 (200 OASST1 pairs, 200 math pairs).                    |
| Search picks "evaluator-pleasing" but human-disliked outputs    | Spot-check 20 random conversational outputs hand-side-by-side with search on/off.                |
| ARC PoC's basin mechanism leaks ARC-specific assumptions         | Code review of the extraction. General-purpose wrapper references *only* loop hidden states + exit_pdfs + evaluator. No grid / action / coord references survive. |
| Stacking with LoRA / Phase 3: search interacts unexpectedly     | Test the 2×2 grid: (base | phase3) × (search on | off). Document regressions.                    |
| Determinism for tests                                            | Search uses RNG (noise perturbations). Seed it. Gate 0 bit-exact check uses `expansions=1`.       |

**Rollback:** `unset LOCAL_AGENT_OURO_INLOOP_SEARCH`. The wrapper is a strict additive layer; the underlying model is identical.

### Open questions before Phase 4a

1. **Hunter Seeker default policy** — does HS benefit from search? Probably yes (HS is closer to ARC-style search-amenable tasks), but verify before defaulting on.
2. **Branch perturbation source** — ARC PoC uses noise + different exit-PDF sampling. For chat / code, noise or sampling-temperature variation? Ablate.
3. **Single-elimination vs. tournament scoring** — O(N) vs. O(N²) evaluator calls. The PoC's `branch_width` semantics suggest single-elimination. Confirm by reading the basin code.
4. **Interaction with `early_exit_threshold=0.87`** — early-exit was the prior "stop looping when confident" mechanism. Phase 4a may want early-exit disabled during search (run all 4 loops for every branch so the evaluator gets full trajectories) and re-enabled outside.

### Files Phase 4a will touch

```
NEW: /home/moloch/local_agent/ouro_inloop_search.py
MOD: /home/moloch/local_agent/ouro_backend.py           (env-var-gated wrapper construction)
MOD: /home/moloch/local_agent/ouro_task_profile.py      (optional: per-task default budgets)
NEW: /home/moloch/ouro_project/tools/bench_inloop_search.py   (latency + win-rate benchmark)
```

No data, no model files, no rented compute. Pure code.

### Resume cheat sheet — Phase 4a

```bash
# 0. Read the basin code in the PoC, identify the lift-out boundary.
$EDITOR /home/moloch/ouro_project/claude_sandbox/arc_agent_pairwise_stockfish_codex.py
# Look for: model_basin_diag_*, _basin_*, entries_sorted, expansions += ...

# 1. Implement the general-purpose wrapper.
$EDITOR /home/moloch/local_agent/ouro_inloop_search.py

# 2. Wire it behind the env var.
$EDITOR /home/moloch/local_agent/ouro_backend.py

# 3. Validation gates (above), in order. Stop at the first failure.

# 4. Benchmark latency + win-rate on the four task families.
python /home/moloch/ouro_project/tools/bench_inloop_search.py \
    --tasks conversation code dsa hard_technical \
    --budgets default

# 5. Decide per-task defaults from the bench results. Update
#    ouro_task_profile.py with the chosen defaults.

# 6. Document in SESSION_CONTEXT_<date>.md.
```

Expected wall-clock: **half a day to a full day** of focused work. Most of the time is in step 0 (understanding the PoC's existing structure) and step 4 (running benchmarks across budgets). The wrapper itself is ~150–250 lines.

Total cost: **$0**. Runs on the laptop.

---

## Phase 4b — Evaluator expansion (math, logic, code, reasoning, thinking)

The HH-RLHF-trained `pairwise_epoch2.pt` evaluator scores 94.98 % on chat preference. The user reports it also tracks math correctness — which is the right inductive bias (it reads loop convergence, and a correct math trajectory converges differently from a wrong one). Phase 4b makes that the *training* distribution, not just an observed property: retrain the same 4.7 M-param architecture on a broader pairwise dataset, ship as `pairwise_phase4b.pt`.

### Data composition

Target ~30,000 pairwise examples, distributed across the five domains. Per-domain construction:

| Domain        | Source                                                           | Pair construction                                                                  | Target count |
|---------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------|--------------|
| **Chat**       | HH-RLHF (existing) + OASST1 ranked pairs                          | Existing chosen/rejected labels.                                                    | ~8,000        |
| **Math**       | MATH dataset, GSM8K + their model-generated wrong-trace siblings  | (correct trace) vs. (model-generated wrong trace with same final answer or wrong final). Filter to ones where the wrong trace is plausibly wrong, not obvious gibberish. | ~6,000        |
| **Logic**      | LogicQA, FOLIO, ProofWriter                                       | (valid deduction) vs. (one-step-fuzzed invalid deduction). Use existing wrong-answer choices where the dataset provides them. | ~4,000        |
| **Code**       | CodeContests + the diff-check-paired examples from Phase 2        | (passes diff-check) vs. (fails diff-check). The same brute-force-vs-broken pairing used as Phase 3b reward becomes evaluator training data. | ~6,000        |
| **Reasoning** / "thinking" | ARC-AGI training set (the data the user's existing PoC was built on) + chain-of-thought reasoning datasets (MMLU explanations) | Existing chosen/rejected from ARC PoC + (correct CoT) vs. (corrupted CoT). | ~6,000        |

The five domains together stress different aspects of "loop convergence": math = numerical correctness, logic = step validity, code = executable correctness, chat = stylistic + factual preference, reasoning = inferential coherence. If the user's hypothesis is right ("evaluator reads loop convergence"), training on all five should produce a single evaluator that works across all five — not five domain-specific evaluators.

### Training shape

Same as the original evaluator (arxiv 2604.09870 Section 3.4):

- Architecture: `PairwiseEvaluator` from `claude_sandbox/evaluator_pairwise_codex.py` (unchanged, 4.7 M params).
- Loss: `-logsigmoid(target · score) + 1e-4 · score²` with 50 % swap protocol.
- Base model frozen: forward pass through Ouro (or Phase 3's checkpoint) to get 4 loop hidden states per response.
- Training compute: single-GPU laptop. ~5–10 hr depending on dataset size. The evaluator itself is tiny; the bottleneck is the frozen Ouro forward pass for hidden states. Pre-compute and cache the hidden states once, then training is fast.

### Validation gates

```bash
# Gate 0: existing HH-RLHF benchmark holds.
# pairwise_phase4b.pt should match or exceed pairwise_epoch2.pt's 94.98% on
# the existing HH-RLHF test set. Drop >= 2pp is a red flag.

# Gate 1: held-out splits per domain.
# Math: ≥ 80% on held-out (correct, wrong-trace) pairs.
# Logic: ≥ 80%.
# Code: ≥ 85% (this should be near-perfect since the signal is exact).
# Reasoning: ≥ 75%.

# Gate 2: cross-domain transfer. Train without math, test on math.
# Tests whether the evaluator's loop-convergence reading generalizes,
# or whether it's domain-specific. User's hypothesis predicts cross-transfer.
```

### Output

`/home/moloch/ouro_project/artifacts/checkpoints/evaluator/pairwise_phase4b.pt` — drop-in replacement for `pairwise_epoch2.pt`. Phases 4a (search wrapper) and 4c (distillation training) consume this checkpoint instead of the original. Phase 3 was already complete by this point, so it doesn't retroactively benefit, but a Phase 3.5 mini-RLTT-pass with the upgraded evaluator is cheap and worth considering.

### Risks specific to Phase 4b

| Risk                                                          | Mitigation                                                                                |
|---------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Evaluator becomes domain-specialized, loses chat ability       | Gate 0 (existing HH-RLHF benchmark) is the firewall.                                       |
| Wrong-trace generation contaminates training distribution      | Manual spot-check 30 random wrong-traces per domain. Reject obviously-degenerate ones.    |
| Cross-domain leakage in eval splits                           | Strict source-level split (e.g., MATH problems by problem-id, not by sample).             |
| Replaces a known-good `pairwise_epoch2.pt`                    | Keep both. Env var selects which evaluator is loaded.                                     |

### Resume cheat sheet — Phase 4b

```bash
# 1. Pre-compute loop hidden states for all training examples. One pass.
python /home/moloch/ouro_project/tools/cache_eval_hidden_states.py \
    --base-model models/ouro_rltt_local \
    --datasets hh-rlhf oasst1 math logic code reasoning \
    --out cache/eval_hidden_states/

# 2. Train.
python /home/moloch/ouro_project/tools/train_pairwise_evaluator.py \
    --hidden-state-cache cache/eval_hidden_states/ \
    --out artifacts/checkpoints/evaluator/pairwise_phase4b.pt \
    --epochs 5

# 3. Validate.
python /home/moloch/ouro_project/tools/eval_pairwise_evaluator.py \
    --ckpt artifacts/checkpoints/evaluator/pairwise_phase4b.pt \
    --gates 0,1,2 \
    --out runs/pairwise_phase4b_validation.json

# 4. If all gates pass: ship as the default evaluator for Phases 4a, 4c.
```

Total wall-clock: 1 day (mostly hidden-state cache build). Total cost: $0.

---

## Phase 4c — In-loop RLTT distillation (rented compute, AlphaZero-style)

Phase 3 trained the policy on **final-output** reward from the evaluator. Phase 4c trains the policy on **search-output** as the supervision target — i.e. during training, run the search loop to find the best branch, then update the policy to predict that branch *without* the search at inference. This is policy distillation of MCTS, the AlphaZero training pattern adapted for the looped UT.

After Phase 4c, the policy's single-pass forward should approximate what Phase 4a's search-wrapped forward would have done — *most of the time*. Where the policy's first guess is still wrong, the Phase 4a wrapper remains as the inference-time safety net (task-gated per the table at the top of Phase 4).

### Why this is strictly more than Phase 3

Phase 3 reward signal: scalar per (prompt, response). Loss signal: per-token, weighted by exit_pdf, but the *target* is just "produce the better response."

Phase 4c reward signal: scalar per (prompt, branch), with multiple branches per step. Loss signal: per-loop-per-branch, with the search choosing which branch's logits become the target. The policy learns *which loop trajectories the search prefers*, not just which final outputs are better.

The implication: Phase 3 teaches "produce X." Phase 4c teaches "think like the search would." Bigger semantic update per step.

### Training shape

```
For each training step:
  1. Sample prompt p.
  2. Run Phase 4a search:
       - branch the forward pass after loop K with root_width × branch_width branches
       - score completed children with pairwise_phase4b.pt
       - select the winning trajectory
  3. The winning trajectory's exit_pdfs, per-loop logits, and final logits
     become the supervision target.
  4. Compute distillation loss:
       L_distill = sum over loops k:
                     exit_pdf[k] * KL(student_logits_k || teacher_logits_k)
       where teacher = the search-selected trajectory.
  5. Also include a small reward-modeling term (Phase 3 style) so the
     policy doesn't drift away from "outputs the evaluator prefers."
  6. KL regularization against pre-Phase-3 base (or pre-4c policy?), 
     coefficient 0.02 — lower than Phase 3 because the supervision is 
     already implicitly anchored to the evaluator's preferences.
```

### Hardware and cost

Search-augmented training is ~2–3× the per-step compute of Phase 3:
- Phase 3 step: 1 forward pass × policy + 1 forward pass × frozen reference + reward.
- Phase 4c step: (root_width × branch_width × number-of-search-stages) forward passes for search + 1 forward pass for student logit production + reward.

At `root=2, branch=2, expansions=4` (Phase 4a's default config), that's roughly 5–6× the forward-pass count vs. Phase 3. Stack with the KL regularization forward pass and we're at ~6–7× Phase 3's per-step cost.

Phase 3 was ~6 hr chat + ~18 hr code = 24 hr on 4× L40S at ~$4/hr = ~$96.
Phase 4c equivalent: ~24 × 6 = 144 hr on 4× L40S = **~$576**. Or rent bigger hardware (4× A100 80 GB at ~$6.50/hr, ~2× throughput) for ~72 hr = ~$468.

Either way, **Phase 4c is the single biggest spend in the plan**. Worth doing only if Phase 4a + Phase 3 show real wins.

### Validation gates

Same five gates as Phase 3 plus:

- **Gate 6 (distillation quality):** with Phase 4a search OFF, the Phase 4c policy should match or beat (Phase 3 policy + Phase 4a search ON) on the devils harness. That's the definition of "the policy internalized the search."
- **Gate 7 (residual search value):** with Phase 4a search ON, the Phase 4c policy should still improve over Phase 4c policy alone. If search adds nothing, the distillation captured everything (good); if search still adds something, leave it gated on for hard tasks (also good — the table at the top).

### Risks specific to Phase 4c

| Risk                                                       | Mitigation                                                                                       |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Distillation collapses to greedy (mode collapse)            | Sample-temperature on the policy at inference; KL term penalizes overly-peaked distributions.   |
| The search-selected trajectory is wrong (evaluator error)   | Gate 0 of Phase 4b is the firewall. If Phase 4b held up, Phase 4c can trust the search.         |
| Cost overrun                                                | Hard wall-clock cap. Stop and validate after every ~$100 spent.                                  |
| Phase 4c regresses chat (the search has less to add there)  | Phase 4c can run on code/math/reasoning only — don't include chat prompts. Chat policy stays at Phase 3 output. |

### Resume cheat sheet — Phase 4c

```bash
# Prerequisites: Phase 3 deployed, Phase 4a working on Phase 3 base, Phase 4b
# evaluator passes its gates.

# 1. Stage Phase 3 checkpoint + Phase 4b evaluator + Phase 4a search wrapper code
#    to RunPod network volume. (Reuse the volume from Phase 3.)
rsync -avP /home/moloch/ouro_project/models/ouro_rltt_local_phase3/ \
    runpod:/workspace/ouro_phase3/
rsync -avP /home/moloch/ouro_project/artifacts/checkpoints/evaluator/pairwise_phase4b.pt \
    runpod:/workspace/evaluator/

# 2. Launch Phase 4c training. Budget hard cap.
bash /home/moloch/ouro_project/tools/runpod_launch_phase4c.sh \
    --steps 3000 \
    --search-config root=2,branch=2,expansions=4 \
    --kl-coef 0.02 \
    --hard-wallclock 24h
# Cost: ~$100 per 24 hr at 4× L40S. Stop and validate every 24 hr.

# 3. Pull final checkpoint back.
rsync -avP runpod:/workspace/ouro_rltt_local_phase4c/ \
    /home/moloch/ouro_project/models/ouro_rltt_local_phase4c/

# 4. Validation gates 0-7 on laptop.

# 5. Deploy via OURO_MODEL_PATH=…_phase4c. Phase 4a wrapper stays available,
#    task-gated per the table at the top of Phase 4.
```

---

# Roadmap summary

| Phase | What                                                         | Where it runs              | Cost     | Status            |
|-------|--------------------------------------------------------------|----------------------------|----------|-------------------|
| 1     | Chat LoRA-SFT on OASST1                                       | 12 GB laptop               | $0       | Fallback only     |
| 2     | Hard-coding LoRA-SFT on advanced algorithms                  | 12 GB laptop               | $0       | Fallback only     |
| 3     | Full RLTT continuation (chat + code)                          | Rented 4× L40S, ~24 hr     | ~$80     | **Primary**       |
| 4a    | Search-augmented inference wrapper                            | 12 GB laptop               | $0       | Run after 3       |
| 4b    | Evaluator expansion (math + logic + code + reasoning + chat) | 12 GB laptop               | $0       | Run after 4a      |
| 4c    | In-loop RLTT distillation (AlphaZero-style)                  | Rented multi-GPU, ~72 hr   | ~$400–600 | Run after 4a + 4b |

Total budget if all phases run: **~$500–700** in compute + ~5 days of staged work (Phase 3 ≈ 3 days, Phase 4a ≈ 1 day, Phase 4b ≈ 1 day, Phase 4c ≈ 3 days). The phases can pause indefinitely between each other; each leaves a deployable artifact.

**Decision points along the way:**

- After Phase 3 deploys: does the new base eliminate the chat-CoT-leakage issue and improve the devils harness? If yes, continue. If no, debug before spending on Phase 4c.
- After Phase 4a runs on Phase 3 base: does search add measurable wins on devils? Sizes whether Phase 4c is worth the spend.
- After Phase 4b ships: does the expanded evaluator pass its gates? If transfer to math/code is weak, Phase 4c becomes less compelling (the supervision signal is the bottleneck).
- After Phase 4c trains: does the distilled policy match Phase 3 + search alone? If yes, you have a stronger single-pass model. If no, distillation didn't take, and Phase 3 + Phase 4a search-gated remains the deployed configuration.

Each decision point is a real off-ramp — none of the phases are sunk-cost-traps.
