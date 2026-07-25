#!/usr/bin/env python3
"""Resumable powered HH probe with indivisible source-pair train/eval assignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
SEED=20260711
MODEL_SPECS={
    "base":("ByteDance/Ouro-2.6B","1ed04250da1a9936042725d302e81c8fa2ab5abd"),
    "thinking":("ByteDance/Ouro-2.6B-Thinking","f1edd81e7ac41355db670500ceaf204e0f73af68"),
    "rltt":(str(ROOT/"models/ouro_rltt_local"),None),
}


def write_json(path:Path,x): path.write_text(json.dumps(x,indent=2)+"\n")
def masked_mean(x,mask):
    m=mask.to(x.dtype).unsqueeze(-1); return (x*m).sum(1)/m.sum(1).clamp_min(1)
def sha256(path:Path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def indices(out:Path,n:int):
    p=out/f"hh_train_indices_{n}.json"
    if p.exists(): return json.loads(p.read_text())["indices"]
    from datasets import load_dataset
    ds=load_dataset("Anthropic/hh-rlhf",split="train")
    idx=np.sort(np.random.default_rng(SEED).choice(len(ds),size=n,replace=False)).tolist()
    write_json(p,{"dataset":"Anthropic/hh-rlhf","split":"train","seed":SEED,"n":n,"dataset_length":len(ds),"indices":idx})
    return idx


def extract(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=Path(args.output_dir).resolve(); root=out/f"hh_{args.backbone}_{args.pairs}_diffs"; root.mkdir(parents=True,exist_ok=True)
    idx=indices(out,args.pairs); ds=load_dataset("Anthropic/hh-rlhf",split="train")
    existing=sorted(root.glob("diffs_*.pt")); done=sum(torch.load(p,map_location="cpu",weights_only=False)["diffs"].shape[0] for p in existing)
    print(f"backbone={args.backbone} pairs={args.pairs} done={done} pending={args.pairs-done}",flush=True)
    if done>=args.pairs: return
    model_ref,revision=MODEL_SPECS[args.backbone]; local=args.backbone=="rltt"
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/ouro_rltt_local"),trust_remote_code=True,local_files_only=True); tok.padding_side="left"
    kw={"trust_remote_code":True,"torch_dtype":torch.bfloat16,"low_cpu_mem_usage":True,"local_files_only":True}
    if not local: kw.update(cache_dir=str(ROOT/"artifacts/hf_cache"),revision=revision)
    model=AutoModelForCausalLM.from_pretrained(model_ref,**kw).to("cuda").eval(); model.config.early_exit_threshold=1.0
    for p in model.parameters(): p.requires_grad_(False)
    captured=[]
    def hook(_m,_i,o):
        nonlocal captured; captured=[h.detach() for h in o[1]]
    handle=model.model.register_forward_hook(hook); buf=[]; shard=len(existing); started=time.time()
    try:
        for pos in range(done,args.pairs):
            i=idx[pos]; texts=[ds[i]["chosen"],ds[i]["rejected"]]
            enc=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=384).to("cuda"); captured=[]
            with torch.inference_mode(): model(**enc,use_cache=False)
            if len(captured)!=4: raise RuntimeError(f"expected four loop boundaries, got {len(captured)}")
            pooled=torch.cat([masked_mean(h,enc["attention_mask"]) for h in captured],dim=1)
            buf.append((pooled[0]-pooled[1]).half().cpu())
            if len(buf)>=args.shard_pairs or pos+1==args.pairs:
                path=root/f"diffs_{shard:04d}.pt"; tmp=path.with_suffix(".tmp")
                arr=torch.stack(buf); start=pos+1-len(buf)
                torch.save({"diffs":arr,"positions":list(range(start,pos+1)),"dataset_indices":idx[start:pos+1],"backbone":args.backbone,"model":model_ref,"revision":revision,"max_length":384},tmp); tmp.replace(path)
                print(f"wrote {path.name} total={pos+1}/{args.pairs} rate={(pos+1-done)/max(time.time()-started,1e-9):.2f} pairs/s",flush=True)
                shard+=1; buf=[]
    finally:
        handle.remove(); model.to("cpu"); del model,tok; torch.cuda.empty_cache()
    files=sorted(root.glob("diffs_*.pt")); write_json(root/"manifest.json",{"status":"COMPLETE","backbone":args.backbone,"pairs":args.pairs,"shards":[{"file":p.name,"sha256":sha256(p)} for p in files],"model":model_ref,"revision":revision,"dtype":"bfloat16 forward; float16 saved","quantization":"none","device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu","seed":SEED})


def fit(args):
    out=Path(args.output_dir).resolve(); root=out/f"hh_{args.backbone}_{args.pairs}_diffs"; files=sorted(root.glob("diffs_*.pt"))
    diffs=torch.cat([torch.load(p,map_location="cpu",weights_only=False)["diffs"] for p in files]).float()
    if len(diffs)!=args.pairs: raise RuntimeError(f"expected {args.pairs} pairs, found {len(diffs)}")
    perm=np.random.RandomState(SEED).permutation(args.pairs); cut=int(.8*args.pairs); tr=perm[:cut]; ev=perm[cut:]
    x=diffs[tr]; n=len(x); w=torch.zeros(x.shape[1],requires_grad=True)
    opt=torch.optim.LBFGS([w],lr=1,max_iter=200,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss=F.softplus(-(x@w)).mean()+.5*(w@w)/n; loss.backward(); return loss
    started=time.time(); opt.step(closure)
    train_correct=(x@w.detach()>0).numpy(); correct=(diffs[ev]@w.detach()>0).numpy(); rng=np.random.default_rng(SEED); boots=[]
    for _ in range(10000): boots.append(correct[rng.integers(0,len(correct),len(correct))].mean())
    scores=diffs[ev]@w.detach(); swap_error=float((scores+(-scores)).abs().max())
    model_ref,revision=MODEL_SPECS[args.backbone]
    result={"status":"COMPLETE","backbone":args.backbone,"model":model_ref,"revision":revision,"dataset":"Anthropic/hh-rlhf","dataset_split":"train","n_source_pairs":args.pairs,"train_pairs":len(tr),"eval_pairs":len(ev),"pair_selection_seed":SEED,"split_seed":SEED,"feature_dim":diffs.shape[1],"feature":"four mean-pooled post-final-norm loop boundaries concatenated","max_length":384,"probe":"bias-free L-BFGS logistic linear","normalization":"NoNorm/raw difference","exact_antisymmetry":True,"max_swap_sum_abs":swap_error,"train_accuracy":float(train_correct.mean()),"eval_accuracy":float(correct.mean()),"eval_ci95_pair_bootstrap":np.quantile(boots,[.025,.975]).tolist(),"bootstrap_draws":10000,"fit_seconds":time.time()-started}
    torch.save({"weight":w.detach(),"train_positions":tr,"eval_positions":ev,"result":result},out/f"hh_{args.backbone}_{args.pairs}_pair_disjoint_probe.pt"); write_json(out/f"hh_{args.backbone}_{args.pairs}_pair_disjoint_results.json",result); print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["extract","fit"]); ap.add_argument("--backbone",choices=list(MODEL_SPECS),required=True); ap.add_argument("--pairs",type=int,default=40000); ap.add_argument("--shard-pairs",type=int,default=500); ap.add_argument("--output-dir",required=True); args=ap.parse_args(); Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    (extract if args.command=="extract" else fit)(args)
if __name__=="__main__": main()
