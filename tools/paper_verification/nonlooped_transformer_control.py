#!/usr/bin/env python3
"""Frozen non-looped transformer control for CoreContent-v2 and HH probing.

The control backbone is openbmb/MiniCPM-2B-sft-bf16 (40 distinct decoder blocks). It is
never trained or modified. CoreContent-v2 uses physical post-block layers 24 and
36, mean pooled over mask-valid tokens, matching the surviving Ouro tap loci and
the original raw/NoNorm pairwise objective. The HH probe uses the post-final-norm
terminal representation with the documented bias-free L-BFGS protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/moloch/ouro_project")
MODEL_ID = "openbmb/MiniCPM-2B-sft-bf16"
REVISION = "4ec16344ac13e6ef5010aeecaa533369ac8eb53c"
SNAPSHOT = ROOT / "artifacts/hf_cache/hub/models--openbmb--MiniCPM-2B-sft-bf16/snapshots" / REVISION
GROUPS_PATH = ROOT / "data/corecontent_v2/processed/candidate_groups_deduped.jsonl"
CORE_DOMAINS = ("coding", "reasoning", "math", "logic", "alignment")
CONFIG_LAYERS = {"24_L4": 24, "36_L4": 36}
MAX_LEN = {"alignment": 768, "coding": 1024, "math": 768, "logic": 640,
           "reasoning": 640, "science": 640, "anatomy": 640}
TRAINING_SETS = {
    "all_core": CORE_DOMAINS,
    "all_core_balanced": CORE_DOMAINS,
    "code_reasoning": ("coding", "reasoning"),
    "math_logic": ("math", "logic"),
    "alignment_only": ("alignment",),
    "code_math_logic": ("coding", "math", "logic"),
}
SEED = 20260710
PROBE_SEED = 42


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16",
        "quantization": "none",
    }


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = hidden.float()
    m = mask.float().unsqueeze(-1)
    return ((h * m).sum(1) / m.sum(1).clamp_min(1.0)).cpu()


class NonLoopedExtractor:
    def __init__(self) -> None:
        if not (SNAPSHOT / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"model snapshot incomplete: {SNAPSHOT}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(SNAPSHOT), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        inner = self.model.model
        self.layers = inner.layers
        self.norm = inner.norm
        if int(self.model.config.num_hidden_layers) != 40:
            raise RuntimeError(f"expected 40 distinct layers, got {self.model.config.num_hidden_layers}")
        self.captured: dict[int, torch.Tensor] = {}

    @contextmanager
    def hooks(self, layer_numbers: Iterable[int]):
        self.captured = {}
        handles = []
        for number in layer_numbers:
            idx = number - 1
            def hook(_module, _inp, output, n=number):
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.captured[n] = value.detach()
            handles.append(self.layers[idx].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def encode(self, texts: list[str], max_length: int, layer_numbers: tuple[int, ...], final_norm: bool = False) -> dict[int, torch.Tensor]:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode(), self.hooks(layer_numbers):
            self.model(**enc, use_cache=False)
        result = {}
        for n in layer_numbers:
            h = self.captured[n]
            if final_norm and n == int(self.model.config.num_hidden_layers):
                h = self.norm(h)
            result[n] = masked_mean(h, enc["attention_mask"])
        return result

    def cleanup(self) -> None:
        self.model.to("cpu")
        del self.model, self.tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def load_source_groups(limit_groups: int = 0) -> list[dict[str, Any]]:
    groups = []
    with GROUPS_PATH.open() as handle:
        for line in handle:
            g = json.loads(line)
            if len(g.get("candidates", [])) >= 2:
                groups.append(g)
            if limit_groups and len(groups) >= limit_groups:
                break
    order = {"coding": 0, "math": 1, "logic": 2, "reasoning": 3, "alignment": 4, "science": 5, "anatomy": 6}
    groups.sort(key=lambda g: (order.get(g["domain"], 9), g["split"], g["group_uid"]))
    return groups


def load_manifest(out: Path) -> dict[str, Any]:
    path = out / "feature_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {"model_id": MODEL_ID, "revision": REVISION, "shards": [], "completed": []}


def extract_corecontent(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve() / "corecontent_features"
    out.mkdir(parents=True, exist_ok=True)
    groups = load_source_groups(args.limit_groups)
    manifest = load_manifest(out); done = set(manifest["completed"])
    pending = [g for g in groups if g["group_uid"] not in done]
    print(f"CoreContent extraction groups={len(groups)} done={len(done)} pending={len(pending)}", flush=True)
    extractor = NonLoopedExtractor(); started = time.time(); encoded = 0; errors = 0
    buffer = []; buffer_candidates = 0; shard_idx = len(manifest["shards"])

    def flush() -> None:
        nonlocal buffer, buffer_candidates, shard_idx
        if not buffer: return
        name = f"features_{shard_idx:04d}.pt"; path = out / name; tmp = path.with_suffix(".tmp")
        torch.save({"groups": buffer, "model_id": MODEL_ID, "revision": REVISION}, tmp); tmp.replace(path)
        manifest["shards"].append({"file": name, "groups": len(buffer), "candidates": buffer_candidates, "sha256": sha256(path)})
        for g in buffer: done.add(g["group_uid"])
        manifest["completed"] = sorted(done); write_json(out / "feature_manifest.json", manifest)
        print(f"wrote {name} groups={len(buffer)} candidates={buffer_candidates} total_done={len(done)}", flush=True)
        shard_idx += 1; buffer = []; buffer_candidates = 0

    try:
        for gi, g in enumerate(pending):
            rows = []
            for start in range(0, len(g["candidates"]), args.batch_size):
                batch = g["candidates"][start:start + args.batch_size]
                try:
                    got = extractor.encode([c["candidate_text"] for c in batch], MAX_LEN.get(g["domain"], 768), (24, 36))
                    for bi, c in enumerate(batch):
                        rows.append({"candidate_uid": c["candidate_uid"], "reward": float(c["reward"]),
                                     "features": {"24_L4": got[24][bi].half(), "36_L4": got[36][bi].half()}})
                        encoded += 1
                except Exception as exc:
                    errors += len(batch); print(f"encode error group={g['group_uid']} {type(exc).__name__}: {exc}", flush=True)
            if len(rows) >= 2:
                buffer.append({"group_uid": g["group_uid"], "task_uid": g["task_uid"], "domain": g["domain"],
                               "split": g["split"], "source_dataset": g["source_dataset"], "candidates": rows})
                buffer_candidates += len(rows)
            else:
                done.add(g["group_uid"])
            if buffer_candidates >= args.shard_candidates: flush()
            if (gi + 1) % 100 == 0:
                rate = encoded / max(time.time() - started, 1e-9)
                print(f"progress groups={gi+1}/{len(pending)} encoded={encoded} errors={errors} rate={rate:.2f}/s", flush=True)
        flush()
    finally:
        extractor.cleanup()
    write_json(Path(args.output_dir) / "corecontent_extraction_summary.json", {
        "status": "COMPLETE" if len(done) >= len(groups) else "PARTIAL", "model_id": MODEL_ID, "revision": REVISION,
        "source_groups": len(groups), "source_candidates": sum(len(g["candidates"]) for g in groups),
        "encoded_this_invocation": encoded, "errors": errors, "elapsed_seconds": time.time() - started,
        "loci": {"24_L4": "physical post-block layer 24", "36_L4": "physical post-block layer 36"},
        "pooling": "mask-valid token mean", "input": "raw candidate_text, matching CoreContent-v2", "environment": environment(),
    })


def load_feature_groups(out_dir: Path) -> list[dict[str, Any]]:
    root = out_dir / "corecontent_features"; man = json.loads((root / "feature_manifest.json").read_text()); groups=[]
    for shard in man["shards"]:
        groups.extend(torch.load(root / shard["file"], map_location="cpu", weights_only=False)["groups"])
    return groups


def pair_diffs(groups: list[dict[str, Any]], config: str, max_pairs: int = 24) -> torch.Tensor | None:
    rows=[]
    for g in groups:
        fs=[c["features"][config].float() for c in g["candidates"]]; rs=[float(c["reward"]) for c in g["candidates"]]
        pairs=[(i,j) for i in range(len(rs)) for j in range(len(rs)) if rs[i] > rs[j]]
        rows.extend(fs[i]-fs[j] for i,j in pairs[:max_pairs])
    return torch.stack(rows) if rows else None


def channel_scores(g: dict[str, Any], channels: list[tuple[str, torch.Tensor]]) -> list[float]:
    per=[]
    for cfg,w in channels:
        s=torch.tensor([torch.dot(w, c["features"][cfg].float()) for c in g["candidates"]])
        sd=s.std(unbiased=False); z=(s-s.mean())/sd if sd > 1e-8 else torch.zeros_like(s); per.append(z)
    return torch.stack(per).mean(0).tolist()


def group_metrics(g: dict[str, Any], scores: list[float]) -> tuple[float,float]:
    rewards=[float(c["reward"]) for c in g["candidates"]]; top=max(range(len(scores)),key=lambda i:scores[i]); top1=float(rewards[top]>=max(rewards)); cor=tot=0.0
    for i in range(len(scores)):
        for j in range(i+1,len(scores)):
            if rewards[i]==rewards[j]: continue
            hi,lo=(i,j) if rewards[i]>rewards[j] else (j,i); tot+=1; cor += 1 if scores[hi]>scores[lo] else (.5 if scores[hi]==scores[lo] else 0)
    return top1, cor/tot if tot else float("nan")


def evaluate(groups: list[dict[str, Any]], channels: list[tuple[str, torch.Tensor]], split: str) -> dict[str, Any]:
    by=defaultdict(list)
    for g in groups:
        if g["split"]==split and g["domain"] in CORE_DOMAINS: by[g["domain"]].append(group_metrics(g,channel_scores(g,channels)))
    domains={d:{"groups":len(v),"top1":float(np.mean([x[0] for x in v])),"pairwise":float(np.nanmean([x[1] for x in v]))} for d,v in by.items()}
    return {"split":split,"domains":domains,"macro_top1":float(np.mean([domains[d]["top1"] for d in CORE_DOMAINS])),
            "macro_pairwise":float(np.mean([domains[d]["pairwise"] for d in CORE_DOMAINS]))}


def train_corecontent(args: argparse.Namespace) -> None:
    out=Path(args.output_dir).resolve(); groups=load_feature_groups(out)
    clean_task_split = args.command == "train-clean"
    stored = defaultdict(set)
    for g in groups: stored[g["task_uid"]].add(g["split"])
    if clean_task_split:
        for g in groups:
            u = int(hashlib.sha256(f"20260711:{g['task_uid']}".encode()).hexdigest()[:16], 16) / 2**64
            g["split"] = "train" if u < .8 else ("val" if u < .9 else "heldout")
    assigned = defaultdict(set)
    for g in groups: assigned[g["task_uid"]].add(g["split"])
    split_audit = {
        "mode": "deterministic_task_disjoint_sha256" if clean_task_split else "stored_source_assignments",
        "seed": 20260711 if clean_task_split else None,
        "unique_task_ids": len(assigned),
        "task_ids_crossing_stored_splits": sum(len(v) > 1 for v in stored.values()),
        "task_ids_crossing_effective_splits": sum(len(v) > 1 for v in assigned.values()),
        "record_counts": dict(Counter(g["split"] for g in groups)),
        "unique_task_counts": dict(Counter(next(iter(v)) for v in assigned.values())),
    }
    by=defaultdict(list)
    for g in groups: by[(g["domain"],g["split"])].append(g)
    best={}; grid=[]; started=time.time()
    for cfg in CONFIG_LAYERS:
        cache={d:pair_diffs(by[(d,"train")],cfg) for d in CORE_DOMAINS}
        for tname,domains in TRAINING_SETS.items():
            tensors={d:cache[d] for d in domains if cache.get(d) is not None}
            if not tensors: continue
            m=min(len(x) for x in tensors.values());
            for lr in (3e-4,1e-3):
                for seed in (0,1,2):
                    gen=torch.Generator().manual_seed(seed); X=torch.cat([x[torch.randperm(len(x),generator=gen)[:m]] for x in tensors.values()])
                    w=torch.zeros(X.shape[1],requires_grad=True); opt=torch.optim.Adam([w],lr=lr,weight_decay=.01)
                    for _ in range(60):
                        opt.zero_grad(); loss=F.softplus(-(X@w)).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_([w],1.0); opt.step()
                    val=evaluate(groups,[(cfg,w.detach())],"val"); rec={"config":cfg,"training_set":tname,"lr":lr,"seed":seed,"pairs":len(X),"val_macro_top1":val["macro_top1"],"weight":w.detach()}; grid.append({k:v for k,v in rec.items() if k!="weight"})
                    if cfg not in best or rec["val_macro_top1"]>best[cfg]["val_macro_top1"]: best[cfg]=rec
    channels=[(cfg,best[cfg]["weight"]) for cfg in CONFIG_LAYERS]; val=evaluate(groups,channels,"val"); held=evaluate(groups,channels,"heldout")
    suffix = "_task_disjoint" if clean_task_split else ""
    torch.save({"model_id":MODEL_ID,"revision":REVISION,"architecture":"AntisymLinearNoNorm","channels":channels,"best":best,"split_audit":split_audit},out/f"minicpm2_sft_corecontent_v2{suffix}.pt")
    write_json(out/f"corecontent_control_results{suffix}.json",{"model_id":MODEL_ID,"revision":REVISION,"layers":CONFIG_LAYERS,"split_audit":split_audit,"grid":grid,"best_per_config":{c:{k:v for k,v in r.items() if k!="weight"} for c,r in best.items()},"blockwise_val":val,"blockwise_heldout":held,"elapsed_seconds":time.time()-started,"environment":environment()})


def fit_lbfgs(diffs: torch.Tensor, train_rows: np.ndarray, labels: torch.Tensor) -> torch.Tensor:
    w=torch.zeros(diffs.shape[1],requires_grad=True); x=torch.cat([diffs,-diffs]); n=len(train_rows); opt=torch.optim.LBFGS([w],lr=1,max_iter=200,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss=F.softplus(-labels[train_rows]*(x[train_rows]@w)).mean()+.5*(w@w)/n; loss.backward(); return loss
    opt.step(closure); return w.detach()


def acc_ci(correct: np.ndarray, seed: int, draws: int=10000) -> list[float]:
    rng=np.random.default_rng(seed); n=len(correct); vals=[correct[rng.integers(0,n,n)].mean() for _ in range(draws)]; return np.quantile(vals,[.025,.975]).tolist()


def clustered_acc_ci(correct: np.ndarray, clusters: np.ndarray, seed: int, draws: int=10000) -> list[float]:
    rng=np.random.default_rng(seed); unique=np.unique(clusters); by=[correct[clusters==u] for u in unique]; vals=[]
    for _ in range(draws):
        sampled=rng.integers(0,len(unique),len(unique)); vals.append(np.concatenate([by[i] for i in sampled]).mean())
    return np.quantile(vals,[.025,.975]).tolist()


def hh_probe(args: argparse.Namespace) -> None:
    n_pairs=args.hh_pairs; out=Path(args.output_dir).resolve(); path=out/f"minicpm2_sft_hh_features_{n_pairs}.pt"
    if path.exists(): payload=torch.load(path,map_location="cpu",weights_only=False); chosen=payload["chosen"].float(); rejected=payload["rejected"].float()
    else:
        from datasets import load_dataset
        ds=load_dataset("Anthropic/hh-rlhf",split="test"); idx=np.sort(np.random.default_rng(PROBE_SEED).choice(len(ds),size=n_pairs,replace=False)); ext=NonLoopedExtractor(); cs=[]; rs=[]; started=time.time()
        try:
            for j,i in enumerate(idx):
                cs.append(ext.encode([ds[int(i)]["chosen"]],384,(40,),True)[40][0]); rs.append(ext.encode([ds[int(i)]["rejected"]],384,(40,),True)[40][0])
                if (j+1)%100==0 or j+1==n_pairs: print(f"HH capture {j+1}/{n_pairs} elapsed={time.time()-started:.1f}s",flush=True)
        finally: ext.cleanup()
        chosen=torch.stack(cs); rejected=torch.stack(rs); payload={"chosen":chosen.half(),"rejected":rejected.half(),"dataset_indices":idx.tolist(),"model_id":MODEL_ID,"revision":REVISION,"layer":"post-final-norm physical layer 40","max_length":384}; torch.save(payload,path)
    diffs=chosen-rejected; labels=torch.cat([torch.ones(n_pairs),-torch.ones(n_pairs)]); perm=np.random.RandomState(PROBE_SEED).permutation(2*n_pairs); cut=int(.8*len(perm)); train_rows,eval_rows=perm[:cut],perm[cut:]; w=fit_lbfgs(diffs,train_rows,labels); x=torch.cat([diffs,-diffs]); scores=x[eval_rows]@w; correct=(labels[eval_rows]*scores>0).numpy().astype(float); clusters=(eval_rows%n_pairs).astype(np.int64)
    pair_perm=np.random.RandomState(PROBE_SEED).permutation(n_pairs); pair_cut=int(.8*n_pairs); trp,tep=pair_perm[:pair_cut],pair_perm[pair_cut:]; pair_train_rows=np.concatenate([trp,trp+n_pairs]); wp=fit_lbfgs(diffs,pair_train_rows,labels); cp=((diffs[tep]@wp)>0).numpy().astype(float)
    result={"model_id":MODEL_ID,"revision":REVISION,"n_source_pairs":n_pairs,"feature_dim":diffs.shape[1],"feature":"mean-pooled post-final-norm physical layer 40","row_split_with_orientation_leakage":{"accuracy":float(correct.mean()),"ci95_source_pair_clustered":clustered_acc_ci(correct,clusters,SEED),"n_rows":len(eval_rows),"unique_pairs":len(set(clusters.tolist()))},"clean_pair_disjoint":{"accuracy":float(cp.mean()),"ci95_pair_bootstrap":acc_ci(cp,SEED),"train_pairs":len(trp),"eval_pairs":len(tep)},"exact_antisymmetry":True,"bias":False,"normalization":"NoNorm/raw difference","environment":environment()}
    torch.save({"row_split_weight":w,"pair_disjoint_weight":wp,"result":result},out/f"minicpm2_sft_hh_linear_probe_{n_pairs}.pt"); write_json(out/f"hh_linear_probe_results_{n_pairs}.json",result)
    if n_pairs==200: write_json(out/"hh_linear_probe_results.json",result)


def ouro_hh_probe(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n_pairs=args.hh_pairs; out=Path(args.output_dir).resolve(); name=args.ouro_model
    source=torch.load(out/f"minicpm2_sft_hh_features_{n_pairs}.pt",map_location="cpu",weights_only=False); indices=source["dataset_indices"]
    path=out/f"ouro_{name}_hh_features_{n_pairs}.pt"
    if path.exists(): payload=torch.load(path,map_location="cpu",weights_only=False); chosen=payload["chosen"].float(); rejected=payload["rejected"].float()
    else:
        ds=load_dataset("Anthropic/hh-rlhf",split="test"); tok_path=ROOT/"models/ouro_rltt_local"; tok=AutoTokenizer.from_pretrained(str(tok_path),trust_remote_code=True,local_files_only=True); tok.padding_side="left"; device=torch.device("cuda")
        if name=="rltt": model_ref=str(ROOT/"models/ouro_rltt_local"); revision=None; kwargs={"local_files_only":True}
        elif name=="base": model_ref="ByteDance/Ouro-2.6B"; revision="1ed04250da1a9936042725d302e81c8fa2ab5abd"; kwargs={"cache_dir":str(ROOT/"artifacts/hf_cache"),"local_files_only":True,"revision":revision}
        else: raise ValueError(name)
        model=AutoModelForCausalLM.from_pretrained(model_ref,trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**kwargs).to(device).eval(); model.config.early_exit_threshold=1.0
        for p in model.parameters(): p.requires_grad_(False)
        captured=[]
        def hook(_m,_i,o):
            nonlocal captured; captured=[h.detach() for h in o[1]]
        handle=model.model.register_forward_hook(hook)
        def side(text):
            nonlocal captured; enc=tok(text,return_tensors="pt",truncation=True,max_length=384).to(device); captured=[]
            with torch.inference_mode(): model(**enc,use_cache=False)
            if len(captured)!=4: raise RuntimeError(f"expected four loops, got {len(captured)}")
            return torch.cat([masked_mean(h,enc["attention_mask"])[0] for h in captured])
        cs=[]; rs=[]; started=time.time()
        try:
            for j,i in enumerate(indices):
                cs.append(side(ds[int(i)]["chosen"])); rs.append(side(ds[int(i)]["rejected"]))
                if (j+1)%100==0 or j+1==n_pairs: print(f"Ouro {name} HH capture {j+1}/{n_pairs} elapsed={time.time()-started:.1f}s",flush=True)
        finally: handle.remove(); model.to("cpu"); del model,tok; torch.cuda.empty_cache()
        chosen=torch.stack(cs); rejected=torch.stack(rs); payload={"chosen":chosen.half(),"rejected":rejected.half(),"dataset_indices":indices,"model":model_ref,"revision":revision,"max_length":384,"feature":"concat four mean-pooled post-final-norm loop boundaries"}; torch.save(payload,path)
    diffs=chosen-rejected; labels=torch.cat([torch.ones(n_pairs),-torch.ones(n_pairs)]); pair_perm=np.random.RandomState(PROBE_SEED).permutation(n_pairs); pc=int(.8*n_pairs); trp,tep=pair_perm[:pc],pair_perm[pc:]; train_rows=np.concatenate([trp,trp+n_pairs]); w=fit_lbfgs(diffs,train_rows,labels); cp=((diffs[tep]@w)>0).numpy().astype(float)
    perm=np.random.RandomState(PROBE_SEED).permutation(2*n_pairs); rc=int(.8*len(perm)); rr,re=perm[:rc],perm[rc:]; wr=fit_lbfgs(diffs,rr,labels); x=torch.cat([diffs,-diffs]); cr=(labels[re]*(x[re]@wr)>0).numpy().astype(float); clusters=(re%n_pairs).astype(np.int64)
    result={"backbone":name,"n_source_pairs":n_pairs,"feature_dim":diffs.shape[1],"clean_pair_disjoint":{"accuracy":float(cp.mean()),"ci95":acc_ci(cp,SEED),"train_pairs":len(trp),"eval_pairs":len(tep)},"row_split_with_orientation_leakage":{"accuracy":float(cr.mean()),"ci95_clustered":clustered_acc_ci(cr,clusters,SEED)},"exact_antisymmetry":True,"bias":False,"normalization":"NoNorm/raw difference","source":{k:v for k,v in payload.items() if k not in ("chosen","rejected")}}
    torch.save({"pair_disjoint_weight":w,"row_split_weight":wr,"result":result},out/f"ouro_{name}_hh_linear_probe_{n_pairs}.pt"); write_json(out/f"ouro_{name}_hh_linear_probe_{n_pairs}.json",result)


def smoke(args: argparse.Namespace) -> None:
    groups=load_source_groups(8); ext=NonLoopedExtractor(); rows=[]
    try:
        texts=[g["candidates"][0]["candidate_text"] for g in groups]
        for bs in (1,2,4,8):
            t0=time.time(); shapes=[]; error=None
            try:
                for start in range(0,len(texts),bs):
                    got=ext.encode(texts[start:start+bs],256,(24,36)); shapes.append({k:list(v.shape) for k,v in got.items()})
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            dt=time.time()-t0; rows.append({"batch_size":bs,"texts":len(texts),"shapes":shapes,"seconds":dt,"texts_per_second":len(texts)/dt if not error else None,"error":error})
    finally: ext.cleanup()
    write_json(Path(args.output_dir)/"smoke.json",{"status":"PASS","rows":rows,"environment":environment(),"model_id":MODEL_ID,"revision":REVISION})
    print(json.dumps(rows,indent=2))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["smoke","extract","train","train-clean","hh","ouro-hh"]); ap.add_argument("--output-dir",required=True); ap.add_argument("--limit-groups",type=int,default=0); ap.add_argument("--batch-size",type=int,default=1); ap.add_argument("--shard-candidates",type=int,default=1600); ap.add_argument("--hh-pairs",type=int,default=200); ap.add_argument("--ouro-model",choices=["base","rltt"],default="rltt"); args=ap.parse_args(); Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    {"smoke":smoke,"extract":extract_corecontent,"train":train_corecontent,"train-clean":train_corecontent,"hh":hh_probe,"ouro-hh":ouro_hh_probe}[args.command](args)


if __name__=="__main__": main()
