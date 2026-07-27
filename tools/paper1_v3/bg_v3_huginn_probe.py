"""V3 Huginn depth-recurrent readability probe.

Question: does candidate-quality readability improve with recurrence depth in a
looped model OUTSIDE the Ouro family? Huginn-0125 (Geiping et al., 2025) is a
3.5B depth-recurrent transformer: prelude (2 layers) -> recurrent core (4
layers, iterated) -> coda (2 layers), hidden 5280.

Design maps recurrence steps onto the sealed cross-loop machinery's loop axis:
features are stored [1, R_MAX, 5280] per candidate (one pseudo-layer = the
recurrent core boundary; "loops" 1..R_MAX = core iterations), after which the
SEALED pairwise tap training, grid selection, evaluation, and clustered
bootstrap run unchanged on patched module config. Same 2,150-group CoreContent
subset, same splits, same seeds.

Capture: one forward per candidate at num_steps=R_MAX with a hook on the last
core block; hook output i = the latent state after core iteration i+1. Masked
mean pooling identical to the sealed convention.

Stages: prep | extract | traineval | stats
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bg_xloop_early_v1_common as X  # noqa: E402

RUN_ROOT = PROJECT_ROOT / "artifacts/reports/huginn_probe_v3_20260726"
SEALED_ROOT = PROJECT_ROOT / "artifacts/reports/cross_loop_early_layer_taps_20260720"
HF_HUB = PROJECT_ROOT / "artifacts/hf_cache/hub"
R_MAX = 8
HIDDEN = 5280
SHARD_MAX_CANDS = 400  # 5280-d x 8 steps is ~4x the Ouro payload per candidate


def snapshot() -> Path:
    snaps = HF_HUB / "models--tomg-group-umd--huginn-0125" / "snapshots"
    return sorted(d for d in snaps.iterdir() if d.is_dir())[-1]


def patch_common() -> None:
    X.RUN_ROOT = RUN_ROOT
    X.FEATURE_ROOT = RUN_ROOT / "features"
    X.PRED_ROOT = RUN_ROOT / "predictions"
    X.LAYERS = (0,)                      # single pseudo-layer: recurrent core boundary
    X.LAYER_POS = {0: 0}
    X.LOOPS = tuple(range(1, R_MAX + 1))  # "loops" = core iterations 1..R_MAX
    X.HIDDEN = HIDDEN
    X.REFIT_CELLS = [(0, r) for r in X.LOOPS]
    X.TRANSFERS = [(0, u, v) for u in (1, 2, 4) for v in X.LOOPS if v > u]
    X.SHUFFLE_CELLS = [(0, 1), (0, R_MAX)]


def stage_prep() -> None:
    (RUN_ROOT / "features").mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "predictions").mkdir(parents=True, exist_ok=True)
    for fname in ("subset_groups.jsonl", "excluded_crossing_tasks.json"):
        src, dst = SEALED_ROOT / fname, RUN_ROOT / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        assert X.sha256_file(src) == X.sha256_file(dst), f"{fname} diverged"
    cfg = json.loads((snapshot() / "config.json").read_text())
    (RUN_ROOT / "model_plan.json").write_text(json.dumps({
        "model": "tomg-group-umd/huginn-0125", "snapshot": str(snapshot()),
        "n_embd": cfg["n_embd"], "r_max": R_MAX,
        "capture": "last core_block module output per recurrence step",
        "pooling": "masked mean, sealed convention", "dtype": "bfloat16"},
        indent=2) + "\n")
    print(f"[prep] ok; snapshot={snapshot().name}")


def load_subset() -> list[dict]:
    rows = []
    with open(RUN_ROOT / "subset_groups.jsonl") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def masked_mean_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    mask = attn.to(hidden.dtype).unsqueeze(-1)
    return ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).squeeze(0)


def stage_extract() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = snapshot()
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        snap, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    steps_states: list[torch.Tensor] = []
    last_core = model.transformer.core_block[-1]

    def hook(_m, _i, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        steps_states.append(out.detach())

    handle = last_core.register_forward_hook(hook)

    subset = load_subset()
    man_path = X.FEATURE_ROOT / "feature_manifest.json"
    man = X.read_json(man_path, {"schema_version": "huginn_probe.feat.1",
                                 "shards": [], "completed_group_uids": []})
    done = set(man["completed_group_uids"])
    order = {"coding": 0, "math": 1, "logic": 2, "reasoning": 3, "alignment": 4}
    pending = [g for g in subset if g["group_uid"] not in done]
    pending.sort(key=lambda g: (order[g["domain"]], g["split"], X.stable_int("ord", g["group_uid"])))
    total = sum(len(g["candidates"]) for g in subset)
    print(f"extract: {len(subset)} groups ({total} cands), {len(pending)} groups pending", flush=True)
    if not pending:
        handle.remove()
        return

    shard_next = max((s["idx"] for s in man["shards"]), default=-1) + 1
    buf, buf_cands, enc = [], 0, 0
    t0 = t_log = time.time()

    def flush():
        nonlocal buf, buf_cands, shard_next
        if not buf:
            return
        fname = f"huginn_{shard_next:04d}.pt"
        fpath = X.FEATURE_ROOT / fname
        tmp = fpath.with_suffix(".pt.tmp")
        torch.save({"schema_version": "huginn_probe.feat.1", "groups": buf,
                    "domain": buf[0]["domain"], "split": buf[0]["split"]}, tmp)
        tmp.replace(fpath)
        man["shards"].append({"file": fname, "idx": shard_next, "groups": len(buf),
                              "candidates": buf_cands, "domain": buf[0]["domain"],
                              "split": buf[0]["split"], "sha256": X.sha256_file(fpath)})
        for g in buf:
            done.add(g["group_uid"])
        man["completed_group_uids"] = sorted(done)
        X.write_json(man_path, man)
        print(f"  wrote {fname}: groups={len(buf)} cands={buf_cands} done={len(done)}", flush=True)
        shard_next += 1
        buf, buf_cands = [], 0

    prev_key = None
    try:
        for g in pending:
            key = (g["domain"], g["split"])
            if prev_key is not None and key != prev_key:
                flush()
            prev_key = key
            cands_out = []
            for c in g["candidates"]:
                text = c["candidate_text"]
                if not str(text).strip():
                    raise ValueError("empty candidate text")
                encd = tok(text, return_tensors="pt", truncation=True,
                           max_length=X.MAX_LEN[g["domain"]], padding=False)
                encd = {k: v.to("cuda") for k, v in encd.items()}
                if "attention_mask" not in encd:
                    encd["attention_mask"] = torch.ones_like(encd["input_ids"])
                steps_states.clear()
                with torch.inference_mode():
                    model(input_ids=encd["input_ids"],
                          attention_mask=encd["attention_mask"],
                          num_steps=R_MAX,
                          output_details={"return_logits": False, "return_latents": False,
                                          "return_head": False, "return_stats": False})
                if len(steps_states) != R_MAX:
                    raise RuntimeError(f"expected {R_MAX} core states, got {len(steps_states)}")
                pooled = torch.stack(
                    [masked_mean_pool(s.float().cpu(), encd["attention_mask"].cpu())
                     for s in steps_states], 0)          # [R_MAX, 5280]
                feats = pooled.unsqueeze(0)              # [1, R_MAX, 5280]
                cands_out.append({"candidate_uid": c["candidate_uid"],
                                  "reward": float(c["reward"]),
                                  "candidate_kind": c.get("candidate_kind"),
                                  "features": feats.to(torch.float16)})
                enc += 1
            buf.append({"group_uid": g["group_uid"], "task_uid": g["task_uid"],
                        "domain": g["domain"], "split": g["split"],
                        "source_dataset": g.get("source_dataset"), "candidates": cands_out})
            buf_cands += len(cands_out)
            if buf_cands >= SHARD_MAX_CANDS:
                flush()
            if time.time() - t_log > 120:
                t_log = time.time()
                rate = enc / (time.time() - t0)
                print(f"  ... encoded={enc} rate={rate:.2f}/s "
                      f"eta~{(total - enc) / max(rate, 1e-9) / 60:.0f}min", flush=True)
        flush()
    finally:
        handle.remove()
    man["extraction_seconds"] = round(time.time() - t0, 1)
    X.write_json(man_path, man)
    print(f"extraction complete: {enc} in {man['extraction_seconds']}s", flush=True)


def stage_traineval() -> None:
    import bg_xloop_early_v1_train_eval as TE
    TE.main()


def stage_stats() -> None:
    from bg_v3_family_xloop import generic_stats
    generic_stats([X.cell_name(0, R_MAX)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prep", "extract", "traineval", "stats", "all"])
    args = ap.parse_args()
    patch_common()
    stages = ["prep", "extract", "traineval", "stats"] if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        {"prep": stage_prep, "extract": stage_extract,
         "traineval": stage_traineval, "stats": stage_stats}[s]()
        print(f"[{s}] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
