# C5: fine-tuned LM reader vs hidden-state probe (post hoc)

`PROTOCOL.md` fixes the whole check. Original hash `4c74281e5b9d9529ffb4b480efd39734cbb4045fc5b17623fed8f5ab08ec2656` (2026-10-10 11:03 CEST, `PROTOCOL.sha256.orig`); after Amendment 1 (2026-10-10, Qwen without gradient checkpointing, before any real fit) `577652b0f75f32232a03bb31e54a07a9a11ce7bf3ccd2f2d99010452f633f187` (`PROTOCOL.sha256`). `lm_reader.py` refuses a real run if `PROTOCOL.md` does not match `PROTOCOL.sha256`.

## Files

| File | What |
|---|---|
| `extract.py` | read-only extraction from the shards, one shard at a time → `data/answers.jsonl.gz` (texts, labels, ids, splits, folds; 0.25 MB) and large features to a scratch dir (CPU probe work only) |
| `probes.py` | CPU: TF-IDF and probe reproduction gates, hash-fold out-of-fold probe → `data/probe_scores.json`, `data/repro_checks.json`, log `probes.log` |
| `lm_reader.py` | fine-tuning and scoring (`--smoke`, `--preflight`); resumable, one JSON per finished fit |
| `results.py` | probe minus reader (E1–E3) → `runs/results.json`, `runs/RESULTS.md`; `--validate` replays published numbers |
| `common.py` | folds, pairing, paired within-question bootstrap; `python common.py` runs its self-check |
| `smoke/` | CPU smoke outputs (tiny random models; numbers meaningless) |

The data and probe files are already built; the GPU run needs only `data/` and the code.

## Environment

`/home/moloch/ouro_project/shared/venv/bin/python` has everything needed: torch 2.12.0.dev (cu128), transformers 4.54.1, peft 0.19.1, sentencepiece 0.2.1, accelerate 1.13.0, scikit-learn 1.7.2. `protobuf` is missing, so the DeBERTa fast tokenizer cannot be built; the code uses the slow sentencepiece tokenizer (the reference implementation), so nothing needs installing. Weights downloaded on first use: DeBERTa-v3-large `pytorch_model.bin` 0.87 GB, Qwen2.5-1.5B `model.safetensors` 3.1 GB (pinned revisions in `lm_reader.py`).

## Commands for the real run (laptop GPU)

```bash
cd /home/moloch/ouro_project/opi/posthoc_astra_20261010/C5_lm_reader
PY=/home/moloch/ouro_project/shared/venv/bin/python
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # weights are cached at the pinned revisions; offline stops transformers' background
                          # safetensors auto-conversion fetch (it downloaded DeBERTa refs/pr/13 during the preflight; unused)

# 0. Checks (CPU, ~1 min): self-check and the published-number replay. Both must pass.
$PY common.py
$PY results.py --validate

# 1. Preflight per reader: peak memory and samples/s on the 16 longest texts. Peak must be under ~11 GiB.
#    If not, add --micro-batch 2 to every command below (same effective batch of 16; record it).
$PY lm_reader.py --reader deberta --preflight
$PY lm_reader.py --reader qwen --preflight

# 2. Fit (resumable: rerunning the same command skips finished fits)
$PY lm_reader.py --reader deberta 2>&1 | tee runs_deberta.log
$PY lm_reader.py --reader qwen    2>&1 | tee runs_qwen.log

# 3. Results, only after both readers have finished
$PY results.py --reader runs/deberta/reader_scores.json --reader runs/qwen/reader_scores.json
```

Runtime: each reader does 58 fits and 207,904–245,120 training sample-epochs (depending on the selected epochs). Preflight on the idle laptop GPU, 2026-10-10 (`preflight.log`; timed fit = 12 steps on 192 random real texts, mean 173–174 tokens):

| Reader | Peak (torch / nvidia-smi) | Samples/s | Training time | Projected wall time (+~5% scoring, ~10 min loading) |
|---|---|---|---|---|
| DeBERTa-v3-large (checkpointing on) | 8.17 GiB / 8,737 MiB | 26.6 | 2.2–2.6 h | 2.5–2.9 h |
| Qwen2.5-1.5B LoRA, checkpointing off (protocol after Amendment 1) | 6.66 GiB / 7,111 MiB | 24.2 | 2.4–2.8 h | 2.7–3.1 h |
| Qwen, checkpointing on (original protocol; superseded) | 3.60 GiB / 4,063 MiB | 17.0 | 3.4–4.0 h | 3.7–4.4 h |

## Rebuilding the inputs (only if needed; reads the SSD)

```bash
$PY extract.py --feat-dir <scratch dir>     # ~3 min, peak RAM ~2 GB; gzip header timestamps change the file hash
$PY probes.py  --feat-dir <scratch dir>     # ~15 min CPU; exits 1 if a protocol gate fails
```

## Smoke test (CPU, tiny random models, 6–8 questions per split and cohort)

```bash
CUDA_VISIBLE_DEVICES= $PY lm_reader.py --reader deberta --smoke --device cpu
CUDA_VISIBLE_DEVICES= $PY lm_reader.py --reader qwen --smoke --device cpu
$PY results.py --reader smoke/deberta/reader_scores.json --reader smoke/qwen/reader_scores.json --out smoke
```

On CPU, bf16 autocast is off (oneDNN here has no bf16 backward); on CUDA it is on.
