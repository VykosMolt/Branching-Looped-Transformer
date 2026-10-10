"""C5: fine-tune a language-model reader on the prompt + first 48 generated tokens; implements PROTOCOL.md.

  --smoke      tiny random-weight model, small question subset, CPU; proves the plumbing only
  --preflight  GPU memory/throughput probe on the longest texts (writes nothing)
Each fit's scores are saved as soon as it finishes; a rerun skips finished fits (resume after a crash)."""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import COHORTS, DATA, HERE, N_FOLDS, OOF_COHORTS, check_folds, cohort, load_rows, mixed_pairs, sha256

READERS = {
    "deberta": {"model": "microsoft/deberta-v3-large", "revision": "64a8c8eab3e352a784c658aef62be1662607476f",
                "lrs": (5e-6, 1e-5, 2e-5), "lora": False, "grad_ckpt": True},
    "qwen": {"model": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323",
             "lrs": (1e-4, 2e-4, 4e-4), "lora": True, "grad_ckpt": False},  # grad_ckpt off: PROTOCOL.md Amendment 1
}
SMOKE_MODELS = {"deberta": ("hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification", "035f03d5411169990d9ac33970a084ad98c0769a"),
                "qwen": ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda")}
MAX_LEN = 256          # measured max 239 (DeBERTa) / 237 (Qwen) tokens; truncation is an error
EPOCHS = 4             # one schedule; epochs 2, 3, 4 are read as checkpoints of it
CKPTS = (2, 3, 4)
EFFECTIVE_BATCH = 16
WARMUP, WD, EPS, CLIP = 0.10, 0.01, 1e-6, 1.0
SEED = 20261010
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


class Reader:
    def __init__(self, name, smoke, device, micro, grad_ckpt):
        from transformers import AutoTokenizer
        self.spec = dict(READERS[name])
        if smoke:
            self.spec["model"], self.spec["revision"] = SMOKE_MODELS[name]
        self.name, self.device, self.micro, self.grad_ckpt, self.smoke = name, device, micro, grad_ckpt, smoke
        assert EFFECTIVE_BATCH % micro == 0
        self.accum = EFFECTIVE_BATCH // micro
        # DeBERTa-v3: the slow sentencepiece tokenizer (the fast one needs protobuf, not installed)
        self.tok = AutoTokenizer.from_pretrained(self.spec["model"], revision=self.spec["revision"], use_fast=name != "deberta")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def encode(self, texts):
        ids = [self.tok(t)["input_ids"] for t in texts]
        n_over = sum(len(x) > MAX_LEN for x in ids)
        if self.smoke:  # tiny test models have short position tables; the real run never truncates
            ids = [x[:MAX_LEN] for x in ids]
        assert n_over == 0 or self.smoke, f"{n_over} texts exceed MAX_LEN={MAX_LEN}"
        return ids

    def build(self):
        from transformers import AutoModelForSequenceClassification
        torch.manual_seed(SEED)
        kw = {"num_labels": 2, "revision": self.spec["revision"]}
        if self.spec["lora"]:
            from peft import LoraConfig, TaskType, get_peft_model
            # bf16 frozen base on CUDA (under autocast); fp32 on CPU, where autocast is off (smoke only)
            base_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            m = AutoModelForSequenceClassification.from_pretrained(self.spec["model"], torch_dtype=base_dtype, **kw)
            m.config.pad_token_id = self.tok.pad_token_id
            if self.grad_ckpt:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m = get_peft_model(m, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))  # score head trained in full
            for p in m.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        else:
            m = AutoModelForSequenceClassification.from_pretrained(self.spec["model"], torch_dtype=torch.float32, **kw)
            if self.grad_ckpt:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return m.to(self.device)

    def batch(self, ids, idx):
        L = max(len(ids[i]) for i in idx)
        pad = self.tok.pad_token_id
        x = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):  # right padding; Qwen2ForSequenceClassification reads the last non-pad token
            x[j, :len(ids[i])] = torch.tensor(ids[i]); att[j, :len(ids[i])] = 1
        return x.to(self.device), att.to(self.device)

    @torch.no_grad()
    def score(self, m, ids, idx):
        m.eval()
        out = []
        for b in range(0, len(idx), 32):
            x, att = self.batch(ids, idx[b:b + 32])
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):  # CPU: fp32 (no bf16 backward in oneDNN here)
                lg = m(input_ids=x, attention_mask=att).logits.float()
            out.append((lg[:, 1] - lg[:, 0]).cpu())
        m.train()
        return torch.cat(out).numpy().astype(float)

    def fit(self, ids, y, train, evals, lr, stop):
        """Train on positions `train` under the fixed 4-epoch schedule; after each epoch in `stop`, score each
        eval set. Returns {epoch: {eval name: scores}}."""
        m = self.build()
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=EPS, weight_decay=WD)
        steps_ep = math.ceil(len(train) / EFFECTIVE_BATCH)
        total, warm = EPOCHS * steps_ep, max(1, int(WARMUP * EPOCHS * steps_ep))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / (total - warm)))
        yt = torch.tensor(y[train])
        n1 = int(yt.sum()); n0 = len(yt) - n1
        assert n0 > 0 and n1 > 0, "a training set has one class"
        w = torch.tensor([len(yt) / (2 * n0), len(yt) / (2 * n1)], dtype=torch.float, device=self.device)
        gen = torch.Generator().manual_seed(SEED)
        res, t0, seen = {}, time.time(), 0
        m.train()
        for ep in range(1, max(stop) + 1):
            perm = [train[i] for i in torch.randperm(len(train), generator=gen).tolist()]
            for s in range(steps_ep):
                chunk = perm[s * EFFECTIVE_BATCH:(s + 1) * EFFECTIVE_BATCH]
                for b in range(0, len(chunk), self.micro):
                    mb = chunk[b:b + self.micro]
                    x, att = self.batch(ids, mb)
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):  # CPU: fp32 (no bf16 backward in oneDNN here)
                        lg = m(input_ids=x, attention_mask=att).logits.float()
                    loss = torch.nn.functional.cross_entropy(lg, torch.tensor(y[mb], device=self.device), weight=w, reduction="sum")
                    (loss / (w[torch.tensor(y[chunk], device=self.device)].sum())).backward()  # weighted mean over the step
                torch.nn.utils.clip_grad_norm_(params, CLIP)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                seen += len(chunk)
            if ep in stop:
                res[ep] = {k: self.score(m, ids, v) for k, v in evals.items()}
        mem = torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else float("nan")
        print(f"    fit: {len(train)} rows, lr {lr}, {max(stop)} epochs, {seen / (time.time() - t0):.1f} samples/s, "
              f"peak GPU {mem:.2f} GiB", flush=True)
        del m, opt
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return res


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def save(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def run_cohort(R, rows, c, out):
    rs, y, g, split, fold = cohort(rows, c)
    check_folds(g, split, fold)
    mixed_pairs(g, y)  # asserts pairing invariants
    keys = [r["key"] for r in rs]
    ids = R.encode([r["text"] for r in rs])
    tv = np.where(np.isin(split, ("train", "val")))[0]
    ho = np.where(split == "heldout")[0]
    assert not set(g[tv]) & set(g[ho]), "a question is in both train+val and held-out"
    d = out / c; d.mkdir(parents=True, exist_ok=True)
    print(f"[{c}] {len(rs)} rows, {len(tv)} train+val, {len(ho)} held-out, max {max(map(len, ids))} tokens", flush=True)

    # 1. selection: 5-fold CV over the hash folds within train+val, pooled OOF AUROC per (lr, epochs)
    table = {}
    for lr in R.spec["lrs"]:
        oof = {e: np.full(len(rs), np.nan) for e in CKPTS}
        for f in range(N_FOLDS):
            p = d / f"cv_lr{lr:g}_fold{f}.json"
            tr, ev = tv[fold[tv] != f], tv[fold[tv] == f]
            assert not set(g[tr]) & set(g[ev])
            if not p.exists():
                print(f"  [{c}] CV lr {lr:g} fold {f}", flush=True)
                res = R.fit(ids, y, list(tr), {"ev": list(ev)}, lr, CKPTS)
                save(p, {"ev": ev.tolist(), "scores": {str(e): res[e]["ev"].tolist() for e in CKPTS}})
            r = json.loads(p.read_text())
            assert r["ev"] == ev.tolist()
            for e in CKPTS:
                oof[e][ev] = r["scores"][str(e)]
        for e in CKPTS:
            assert not np.isnan(oof[e][tv]).any()
            table[(lr, e)] = auroc(y[tv], oof[e][tv])
    best = None
    for lr, e in itertools.product(R.spec["lrs"], CKPTS):  # strict >: ties go to the smaller lr, then fewer epochs
        if best is None or table[(lr, e)] > table[best]:
            best = (lr, e)
    lr, ep = best
    print(f"[{c}] selected lr {lr:g}, {ep} epochs (CV AUROC {table[best]:.4f})", flush=True)

    # 2. refit on all train+val, score held-out once
    p = d / "heldout.json"
    if not p.exists():
        res = R.fit(ids, y, list(tv), {"ho": list(ho)}, lr, (ep,))
        save(p, {"ho": ho.tolist(), "scores": res[ep]["ho"].tolist()})
    r = json.loads(p.read_text()); assert r["ho"] == ho.tolist()
    result = {"cv_table": {f"{a:g}|{b}": v for (a, b), v in table.items()}, "selected": {"lr": lr, "epochs": ep, "cv_auroc": table[best]},
              "heldout": dict(zip([keys[i] for i in ho], r["scores"])), "oof": {}}

    # 3. out of fold over all splits (hash folds), same config
    if c in OOF_COHORTS:
        oof = np.full(len(rs), np.nan)
        for f in range(N_FOLDS):
            p = d / f"oof_fold{f}.json"
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            assert not set(g[tr]) & set(g[te])
            if not p.exists():
                print(f"  [{c}] OOF fold {f}", flush=True)
                res = R.fit(ids, y, list(tr), {"te": list(te)}, lr, (ep,))
                save(p, {"te": te.tolist(), "scores": res[ep]["te"].tolist()})
            r = json.loads(p.read_text()); assert r["te"] == te.tolist()
            oof[te] = r["scores"]
        assert not np.isnan(oof).any()
        result["oof"] = dict(zip(keys, oof.tolist()))
    return result


def smoke_subset(rows):
    """Per cohort and split: up to 4 mixed questions and 4 others, by hash order (deterministic)."""
    import hashlib
    keep = set()
    for c in COHORTS:
        rs, y, g, split, _ = cohort(rows, c)
        for sp in ("train", "val", "heldout"):
            qs = sorted({t for t, s in zip(g, split) if s == sp}, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
            ys = {t: y[g == t] for t in qs}
            mixed = [t for t in qs if 0 < ys[t].mean() < 1][:4]
            other = [t for t in qs if not 0 < ys[t].mean() < 1][:4]
            keep |= {(c, t) for t in mixed + other}
    return [r for r in rows if (r["cohort"], r["task_uid"]) in keep]


def preflight(R, rows):
    """Peak memory and throughput on the 16 longest texts: 3 optimizer steps (2 x 16 x 3 = full-size batches)."""
    texts = sorted({r["text"] for r in rows}, key=lambda t: -len(R.tok(t)["input_ids"]))[:EFFECTIVE_BATCH]
    ids = R.encode(texts)
    y = np.array([i % 2 for i in range(len(ids))])
    R.fit(ids, y, list(range(len(ids))) * 3, {"x": list(range(len(ids)))}, R.spec["lrs"][-1], (1,))  # warm-up + memory
    rng = np.random.default_rng(0)  # timed fit: 192 random real texts (representative lengths), 12 steps, after warm-up
    sample = [rows[i]["text"] for i in rng.choice(len(rows), 12 * EFFECTIVE_BATCH, replace=False)]
    ids2 = R.encode(sample)
    print(f"  timed fit: 12 steps on 192 random texts, mean {np.mean([len(x) for x in ids2]):.0f} tokens", flush=True)
    R.fit(ids2, np.arange(len(ids2)) % 2, list(range(len(ids2))), {"x": list(range(EFFECTIVE_BATCH))}, R.spec["lrs"][-1], (1,))
    print(f"preflight {R.name}: longest text {max(map(len, ids))} tokens; budget 11 GiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=READERS, required=True)
    ap.add_argument("--cohorts", default=",".join(COHORTS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--micro-batch", type=int, default=4, help="per-step batch; accumulation keeps the effective batch at 16")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="force checkpointing off (protocol: on for DeBERTa, off for Qwen)")
    a = ap.parse_args()
    if not a.smoke:
        want = (HERE / "PROTOCOL.sha256").read_text().split()[0]
        assert sha256(HERE / "PROTOCOL.md") == want, "PROTOCOL.md changed after it was hashed"
    torch.set_num_threads(8)
    rows = load_rows()
    if a.smoke:
        rows = smoke_subset(rows)
    R = Reader(a.reader, a.smoke, torch.device(a.device), a.micro_batch, READERS[a.reader]["grad_ckpt"] and not a.no_grad_ckpt)
    if a.preflight:
        return preflight(R, rows)
    out = a.out or (HERE / ("smoke" if a.smoke else "runs") / a.reader)
    out.mkdir(parents=True, exist_ok=True)
    allres = {"reader": a.reader, "model": R.spec["model"], "revision": R.spec["revision"], "smoke": a.smoke,
              "data_sha256": sha256(DATA / "answers.jsonl.gz"), "cohorts": {}}
    for c in a.cohorts.split(","):
        allres["cohorts"][c] = run_cohort(R, rows, c, out)
        save(out / "reader_scores.json", allres)
    print("wrote", out / "reader_scores.json")


if __name__ == "__main__":
    main()
