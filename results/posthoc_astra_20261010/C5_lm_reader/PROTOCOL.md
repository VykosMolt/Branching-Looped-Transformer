# C5: does a fine-tuned language-model reader of the visible text match the hidden-state probe?

Written 2026-10-10, before any language-model reader was fit on real data. Its SHA-256 is in `PROTOCOL.sha256`; `lm_reader.py` refuses a real run if this file has changed.

## Status: post hoc

This check is **not preregistered**. The labels, the probe results and the TF-IDF reader results of both sets have been seen and published (OPI v4). What is fixed here, in advance, is everything about the language-model readers: data, models, revisions, hyperparameter grid, selection rule, evaluation, and how results will be worded. Nothing may change after the first real fit except through a dated amendment at the end of this file, reported with the result.

Before this file was written, only label-free and published-number work was done: extraction of texts, labels, ids and splits (`extract.py`), cohort counts (all equal the published ones, below), and token-length measurement. The CPU smoke test runs after this file is hashed, uses tiny random-weight models on a small subset, and its numbers mean nothing.

## Question

The paper says: "A stronger text reader, such as a fine-tuned language model, might close part of this gap; I did not test one." The gap: on the first set (Run 1) the hidden-state probe beat the TF-IDF reader across questions (0.758 vs 0.641, +0.118 [0.032, 0.206]) and within questions (+0.081 [0.000, 0.160], out of fold). On the second set (Run 4) the TF-IDF reader already matched the probe within questions (0.759 vs 0.709, 237 pairs). C5 replaces the TF-IDF reader with two fine-tuned language models and repeats the comparisons.

## Data

`data/answers.jsonl.gz` (SHA-256 `0d7dd546590efcc6d23ce6a90eb0eddda38be3d07f489550152c1a5452560d07`), built read-only by `extract.py` from the shards, one shard at a time.

| Cohort | Definition | All | Train+val | Held-out (questions, incorrect) | Mixed pairs all / held-out |
|---|---|---|---|---|---|
| `r1` (primary) | Run 1, published labels: parseable by the published parser and answer-free at 48 tokens (sealed `analyze_run1.py`) | 1,387 | 975 | 412 (165, 45) | 209 (108 q) / 57 (30 q) |
| `r1s` (secondary) | Run 1, strict labels (`gen_run4.strict_answer`, as `strict_labels_run1_final_rule.py`) and answer-free at 48 | 1,339 | 940 | 399 (163, 29) | 116 (61 q) / 37 (18 q) |
| `r4` (primary) | Run 4, strict labels (sealed `analyze_run4.py`): labelled and answer-free at 48 | 2,376 | 1,641 | 735 (326, 59) | 237 (130 q) / 73 (43 q) |

All counts equal the published ones. `r1s` has 29 held-out incorrect answers, one below the minimum of 30 the published analyses used; it is descriptive only.

- **Text**: exactly the TF-IDF reader's input, `"Prompt:\n{mcq_prompt(task_prompt, options)}\n\nVisible reasoning prefix:\n{prefix_texts[3]}"`, where `prefix_texts[3]` is the decoded first 48 generated tokens. Checks: (a) on every 25th record (147 records), decoding the saved `gen_ids[:48]` and `prompt_ids` with the Ouro tokenizer gives exactly `prefix_texts[3]` and the prompt (passed during extraction); (b) gate: `probes.py` refits the published TF-IDF readers on the stored texts and must reproduce the published held-out AUROC to 1e-9 (Run 1: 0.6405086285195277; Run 4, with its surface features appended: 0.7200882559422325).
- **Splits**: the published `split` field (train / val / heldout).
- **Folds** for selection and out-of-fold scoring: `fold = int(sha256("C5_lm_reader|" + task_uid), 16) mod 5`, at question level, the same for every reader and the probe. (Salted, because Run 1's questions are a contiguous range of the unsalted sha256 order.) Asserted: one fold and one split per question, all folds used, no question on both sides of any fit.

## Readers

| | (a) DeBERTa | (b) Qwen |
|---|---|---|
| Model | `microsoft/deberta-v3-large` @ `64a8c8eab3e352a784c658aef62be1662607476f` | `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` |
| Head | `AutoModelForSequenceClassification`, 2 labels (pooled [CLS]) | `Qwen2ForSequenceClassification`, 2 labels (last non-pad token) |
| Trained | all weights, fp32 master copy | LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down projections; classification head trained in full; base frozen in bf16; trainable weights fp32 |
| Tokenizer | slow sentencepiece (`DebertaV2Tokenizer`) | fast, pad = `<|endoftext|>` |
| Learning rates | 5e-6, 1e-5, 2e-5 | 1e-4, 2e-4, 4e-4 |

Common to both:
- **Length**: maximum 256 tokens. Measured over all 3,800 distinct texts: max 239 (DeBERTa), 237 (Qwen); median 175 / 176. Truncation is an error (asserted). Right padding, dynamic per batch.
- **Optimization**: AdamW (betas 0.9, 0.999; eps 1e-6; weight decay 0.01 on all trainable weights), effective batch 16 (micro-batch 4 x 4 accumulation steps), gradient clipping at 1.0, bf16 autocast, gradient checkpointing on (sized for the 12 GB laptop GPU; it does not change the math).
- **Schedule**: one 4-epoch schedule, linear warmup over the first 10% of its steps, then linear decay to zero. The epoch grid {2, 3, 4} is read as checkpoints of that schedule (the model after 2 or 3 epochs has not finished its decay).
- **Class weighting**: cross-entropy weighted by w_c = n / (2 n_c), from the training rows of each fit.
- **Score**: logit(correct) minus logit(incorrect).
- **Seed**: 20261010 for head/LoRA initialization and shuffling, in every fit. One seed, no reruns. A fit that crashes (out of memory, hardware) may be rerun unchanged; anything else is an amendment.

## Selection and fitting (per reader and cohort)

1. **Selection**, mirroring the probes (5-fold question-grouped CV within train+val): for each learning rate and each hash fold f, train on train+val rows outside f and score the rows in f after epochs 2, 3 and 4. For each (learning rate, epochs), pool the five folds' scores and compute AUROC over train+val. Take the highest; ties go to the smaller learning rate, then fewer epochs. 15 fits.
2. **Held-out**: refit on all of train+val with the selected learning rate, stop after the selected epochs, and score the held-out split once. 1 fit.
3. **Out of fold over all splits** (`r1`, `r4` only): for each hash fold f over all splits, fit on the other four folds with the selected configuration and score f. 5 fits. The configuration is not re-selected inside these folds; the probe is scored the same way (below).

No look at held-out labels, out-of-fold results, or any probe-versus-reader comparison happens until all fits for a reader are done.

## Probe scores compared with

All recomputed on CPU by `probes.py` with the published code and gated on the published numbers (to 1e-9). If a gate fails, C5 stops and is reported; no reader is fit.

| Use | `r1` | `r1s` | `r4` |
|---|---|---|---|
| Across questions, primary | `arm()` hidden+surface (log-prob mean/min), the probe behind the published +0.118; gate 0.7582803511958826 | same estimator, strict labels; gate 0.7771668219944082 | hidden-only probe (`analyze_run3` estimator, published config k=24, C=1); gate 0.6833567345301373 |
| Across questions, secondary | `analyze_run3` hidden-only (as `controls.py`) | same | combined probe (hidden + log-prob + six question features, config k=24, C=1); gate 0.7342041921572561 |
| Within questions, held-out pairs | `analyze_run3` hidden-only; gate: within-question AUROC 0.7192982456140351 | same (no published value) | hidden-only above; gate 0.7397260273972602 |
| Within questions, all pairs out of fold | `analyze_run3` hidden-only, hash folds | – | same |

Why hidden-only for `r4` across questions: the reader sees only text, and the combined probe also gets nine surface and question features. The paper's Run 4 TF-IDF comparison gave both sides those features; the language-model readers get none, so the combined comparison is reported as secondary.

The out-of-fold probe: its configuration (k in {16, 24}, C in {0.01, 0.1, 1}) is selected by the same hash-fold CV within train+val (pooled AUROC, first best in grid order), then each hash fold over all splits is scored by a fit on the other four. Its numbers will differ slightly from the published 0.679 / 0.709, which used nested selection and `GroupKFold` folds; both are reported.

## Evaluation (`results.py`)

For each reader and cohort:
- **E1, across questions, held-out**: AUROC of the reader and of the probe; probe minus reader, with the published paired question-clustered bootstrap (`bootstrap_task_clustered`, 2,000 draws, seed 20260724).
- **E2, within questions, held-out mixed pairs**: share of (correct, incorrect) pairs from the same question in which the correct answer scores higher (ties 1/2), pair-weighted; probe minus reader with the published paired bootstrap over questions (`paired_within.py`: 2,000 draws, `numpy` seed 20260724, 2.5% / 97.5% order statistics). The reader alone with `within_task_auroc`.
- **E3, within questions, all mixed pairs out of fold** (`r1`, `r4`): as E2 on the hash-fold out-of-fold scores of reader and probe.

`results.py --validate` must first reproduce, with the same code on published inputs: Run 1 probe minus TF-IDF +0.11777 [0.03205, 0.20621]; Run 4 combined minus TF-IDF+S +0.01412 [-0.02338, 0.05024]; Run 4 out-of-fold within, probe minus TF-IDF -0.05063 [-0.11715, 0.01660] over 237 pairs (from the saved `oof_scores.npy`).

## Wording, fixed now

Per reader, cohort and comparison: interval above zero, "the probe beats the reader"; below zero, "the reader beats the probe"; spanning zero, "no difference resolved at this sample size", which is not evidence of equality.

- The paper's sentence is answered by `r1` E1 (and E3). If either reader's E1 interval spans zero or lies below it, the paper can no longer say the hidden states beat a text reader across questions on the first set; it says which reader and gives both readers' results.
- Both readers are always reported. Picking the better reader after seeing held-out results is selection, so neither is called "the" reader.
- If a reader's held-out scores are constant or nearly so (a collapsed fit), the result is reported as is, with that fact; no rerun.
- `r1s` is descriptive (29 held-out incorrect).

## Compute

Planned on the laptop GPU (RTX 5070 Ti Laptop, 12 GB; budget about 11 GB), bf16. Also runs unchanged on a 48 GB A40.
- Memory (estimates, to be confirmed by `--preflight` before the run): DeBERTa-v3-large full fine-tune, fp32 weights + gradients + AdamW about 7 GB, activations at micro-batch 4 x 256 tokens with checkpointing under 1 GB; Qwen2.5-1.5B LoRA, bf16 base about 3.1 GB plus activations under 1 GB. If the preflight shows more than 11 GB, micro-batch 2 x 8 accumulation is used (same effective batch; recorded).
- Work per reader: 58 fits, at most 245,120 training sample-epochs (r1: 46,800 CV + 3,900 held-out + 22,192 out of fold; r1s: 45,120 + 3,760; r4: 78,768 + 6,564 + 38,016), plus scoring passes.
- Runtime (inferred, not measured; the preflight prints samples/s): DeBERTa about 20-40 samples/s on the laptop (2-3.5 h), 30-60 on an A40 (1-2.5 h); Qwen about 7-14 samples/s on the laptop (5-10 h), 15-30 on an A40 (2.5-5 h). Plus about 10 minutes of model loading per reader.

## Code and inputs (hashes at writing)

- Published code imported unchanged: `analyze_run3.py` 7d8b1d57…dafbb1c; `bg_v2_malformed_sibling_control.py` 52505aad…a06e7ba; `bg_v2_overnight_horizon_analysis.py` 20b3c624…e0ff; `proto_introspection_controls_analysis.py` 7cc11f8f…05da8; `bg_steering_suite_lib.py` 857747f1…82aca; `gen_run4.py` 4944e7b9…e5f; `answer_detector.py` 8be46be8…31dc5b4 (identical in both runs).
- Run 4 saved out-of-fold scores used only by `--validate`: `oof_scores.npy` b44f139c…f0bae.
- C5 code: `common.py`, `extract.py`, `probes.py`, `lm_reader.py`, `results.py`. Code fixes after hashing (plumbing found by the smoke test) are allowed if they do not change anything this file fixes; their final hashes are reported with the result.

## Amendments

### Amendment 1, 2026-10-10: Qwen trains without gradient checkpointing

Made before any real fit: `runs/` does not exist; only the CPU smoke test and the GPU preflights have run, and neither touches real labels with the real models' predictions.

- Change: Qwen2.5-1.5B LoRA trains with gradient checkpointing **off**. DeBERTa-v3-large keeps it on. Everything else is unchanged.
- Reason, measured by `lm_reader.py --preflight` on the idle laptop RTX 5070 Ti (`preflight.log`; timed fit: 12 steps on 192 random real texts, mean 174 tokens): with checkpointing off, peak 6.66 GiB (nvidia-smi 7,111 MiB, inside the 11 GiB budget) and 24.2 samples/s, against 3.60 GiB and 17.0 samples/s with it on. The run is about 1.3 h shorter (about 2.7–3.1 h instead of 3.7–4.4 h).
- Gradient checkpointing recomputes activations in the backward pass; it does not change the model, loss, data order or any hyperparameter, so the math is unchanged (results can differ only at floating-point level, as between any two GPU runs).
- Measured DeBERTa preflight for the record: peak 8.17 GiB (nvidia-smi 8,737 MiB), 26.6 samples/s, about 2.5–2.9 h.
- Hash of this file before this amendment: `4c74281e5b9d9529ffb4b480efd39734cbb4045fc5b17623fed8f5ab08ec2656` (kept in `PROTOCOL.sha256.orig`). The new hash is in `PROTOCOL.sha256`, which `lm_reader.py` checks.
