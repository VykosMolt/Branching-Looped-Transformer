"""Cross-loop early-layer v1 — subset build, split integrity, and feature extraction.

Stage 1 (subset): deterministic bounded subset of the CoreContent v2 corrected dataset,
candidate identity taken from the cached v2 feature shards, texts joined by candidate_uid.
Stage 2 (integrity): hard zero-crossing assertion on task_uid across splits.
Stage 3 (extract): one frozen forward per candidate capturing layers {8,16,24,36,47} x
loops {1..4}, masked-mean pooled, fp16 shards with resume manifest.

Read-only w.r.t. the model and all production artifacts.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X

sys.path.insert(0, str(X.PROJECT_ROOT))
from src.evaluator.bg_transformer_features import (  # noqa: E402
    BGTransformerFeatureExtractor,
    _masked_mean_pool,
)

SHARD_MAX_CANDS = 1000


class _CaptureX:
    def __init__(self) -> None:
        self.inter: dict[int, list[torch.Tensor]] = {idx: [] for idx in X.CAPTURE_IDX.values()}
        self.boundary: list[torch.Tensor] = []

    def make_inter_hook(self, idx: int):
        def _hook(_module, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            self.inter[idx].append(t.detach())
        return _hook

    def boundary_hook(self, _module, _inp, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
            self.boundary = []
            return
        self.boundary = [h.detach() for h in output[1]]


class ExtractorX(BGTransformerFeatureExtractor):
    """Same model load / tokenizer / pooling as production; wider layer capture set."""

    def encode_multi_layer(self, text: str, max_length: int) -> torch.Tensor:
        if not str(text).strip():
            raise ValueError("text must be non-empty")
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length, padding=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=self.device)

        cap = _CaptureX()
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if inner is None or layers is None:
            raise RuntimeError("model does not expose model.model.layers")
        handles = [inner.register_forward_hook(cap.boundary_hook)]
        for idx in X.CAPTURE_IDX.values():
            handles.append(layers[idx].register_forward_hook(cap.make_inter_hook(idx)))
        try:
            with torch.inference_mode():
                self.model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        if len(cap.boundary) != len(X.LOOPS):
            raise RuntimeError(f"expected {len(X.LOOPS)} boundary states, got {len(cap.boundary)}")
        for layer, idx in X.CAPTURE_IDX.items():
            if len(cap.inter[idx]) != len(X.LOOPS):
                raise RuntimeError(f"layer {layer}: expected {len(X.LOOPS)} loop states, got {len(cap.inter[idx])}")

        per_layer = []
        for layer in X.LAYERS:
            states = cap.boundary if layer == 47 else cap.inter[X.CAPTURE_IDX[layer]]
            pooled = [_masked_mean_pool(states[i], enc["attention_mask"]) for i in range(len(X.LOOPS))]
            per_layer.append(torch.stack(pooled, 0))
        feats = torch.stack(per_layer, 0).to(device="cpu", dtype=torch.float32)
        expected = (len(X.LAYERS), len(X.LOOPS), X.HIDDEN)
        if tuple(feats.shape) != expected:
            raise ValueError(f"feature shape {tuple(feats.shape)} != {expected}")
        return feats


# ----------------------------------------------------------------- stage 1: subset
def build_subset() -> list[dict]:
    subset_path = X.RUN_ROOT / "subset_groups.jsonl"
    if subset_path.exists():
        print("subset_groups.jsonl exists; reusing", flush=True)
        return list(X.read_jsonl(subset_path))

    print("building subset ...", flush=True)
    text_by_uid: dict[str, str] = {}
    meta_by_gid: dict[str, dict] = {}
    task_split_full: dict[str, set] = defaultdict(set)
    for g in X.read_jsonl(X.PROC_JSONL):
        task_split_full[g["task_uid"]].add(g["split"])
        if g["domain"] not in X.CORE_DOMAINS:
            continue
        meta_by_gid[g["group_uid"]] = {"task_uid": g["task_uid"], "domain": g["domain"],
                                       "split": g["split"], "source_dataset": g.get("source_dataset")}
        for c in g["candidates"]:
            text_by_uid[c["candidate_uid"]] = c["candidate_text"]

    # Population-level check: split should be a pure function of task. 195 hendrycks_math
    # task_uids violate this because the v2 task id reuses the row index across subject
    # configs (an ID collision between distinct problems, not prompt leakage — identical
    # prompts were deduped in v2). These task_uids are unusable as disjointness/cluster
    # units, so they are excluded from the subset entirely and recorded.
    task_multi = {t: sorted(s) for t, s in task_split_full.items() if len(s) > 1}
    excluded_tasks = set(task_multi)
    if any(not t.startswith("EleutherAI/hendrycks_math") for t in excluded_tasks):
        raise AssertionError(f"unexpected non-hendrycks split-crossing tasks: "
                             f"{[t for t in excluded_tasks if not t.startswith('EleutherAI/hendrycks_math')][:5]}")
    X.write_json(X.RUN_ROOT / "excluded_crossing_tasks.json",
                 {"count": len(excluded_tasks), "reason": "task_uid row-index collision across "
                  "hendrycks_math subject configs -> same task_uid spans splits; excluded from subset",
                  "task_uids": sorted(excluded_tasks)})
    print(f"excluding {len(excluded_tasks)} split-crossing (colliding) task_uids", flush=True)

    # catalog cached v2 shard groups (candidate identity source)
    man = X.read_json(X.V2_FEATURE_ROOT / "feature_manifest.json", {})
    catalog: dict[tuple, list[dict]] = defaultdict(list)
    for s in man.get("shards", []):
        if s["domain"] not in X.CORE_DOMAINS or s["split"] not in X.SUBSET_SIZES:
            continue
        payload = torch.load(X.V2_FEATURE_ROOT / s["file"], map_location="cpu", weights_only=False)
        for g in payload["groups"]:
            cands = [{"candidate_uid": c["candidate_uid"], "reward": float(c["reward"]),
                      "candidate_kind": c.get("candidate_kind")} for c in g["candidates"]]
            if g["task_uid"] in excluded_tasks:
                continue
            if len(cands) < 2 or len({c["reward"] > 0 for c in cands}) < 2:
                continue
            if any(c["candidate_uid"] not in text_by_uid for c in cands):
                continue
            catalog[(g["domain"], g["split"])].append(
                {"group_uid": g["group_uid"], "task_uid": g["task_uid"], "domain": g["domain"],
                 "split": g["split"], "source_dataset": g.get("source_dataset"),
                 "shard_file": s["file"], "candidates": cands})
        del payload

    subset = []
    counts = {}
    dup_count = 0
    for (dom, split), gs in sorted(catalog.items()):
        # v2 shards contain a small number of duplicate group records (resume overlap);
        # keep the first occurrence per group_uid (deterministic: shard-file order)
        seen: set[str] = set()
        deduped = []
        for g in gs:
            if g["group_uid"] in seen:
                dup_count += 1
                continue
            seen.add(g["group_uid"])
            deduped.append(g)
        gs = deduped
        gs.sort(key=lambda g: X.stable_int(X.SUBSET_SEED_NS, g["group_uid"]))
        take = gs[: X.SUBSET_SIZES[split]]
        counts[f"{dom}/{split}"] = len(take)
        for g in take:
            for c in g["candidates"]:
                c["candidate_text"] = text_by_uid[c["candidate_uid"]]
            subset.append(g)

    with open(subset_path, "w") as f:
        import json
        for g in subset:
            f.write(json.dumps(g) + "\n")
    uniq = {g["group_uid"] for g in subset}
    assert len(uniq) == len(subset), f"duplicate group_uids in subset: {len(subset) - len(uniq)}"
    print(f"subset built: {len(subset)} groups (shard dupes skipped: {dup_count}), counts={counts}", flush=True)
    return subset


def split_integrity(subset: list[dict]) -> None:
    tasks_by_split: dict[str, set] = defaultdict(set)
    groups_by_split: dict[str, int] = defaultdict(int)
    cands = 0
    for g in subset:
        tasks_by_split[g["split"]].add(g["task_uid"])
        groups_by_split[g["split"]] += 1
        cands += len(g["candidates"])
    crossings = []
    splits = sorted(tasks_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            inter = tasks_by_split[a] & tasks_by_split[b]
            if inter:
                crossings.extend(sorted(inter)[:5])
    payload = {
        "zero_crossing_count": len(crossings),
        "crossing_examples": crossings[:10],
        "groups_by_split": dict(groups_by_split),
        "tasks_by_split": {k: len(v) for k, v in tasks_by_split.items()},
        "total_candidates": cands,
        "split_source": "existing deterministic v2 task/prompt-disjoint split (split_for); no new split constructed",
    }
    X.write_json(X.RUN_ROOT / "split_integrity.json", payload)
    assert not crossings, f"TASK SPLIT CROSSING DETECTED: {crossings[:5]}"
    print(f"split integrity OK: {payload['tasks_by_split']} candidates={cands}", flush=True)


# ----------------------------------------------------------------- stage 2: cached ref features
def dump_cached_ref(subset: list[dict]) -> None:
    """Cached v2 (3,4,2048) features for the heldout subset — feature-match + reproduction control."""
    out_path = X.RUN_ROOT / "cached_ref_heldout.pt"
    if out_path.exists():
        print("cached_ref_heldout.pt exists; reusing", flush=True)
        return
    want: dict[str, dict] = {g["group_uid"]: g for g in subset if g["split"] == "heldout"}
    by_shard: dict[str, list[str]] = defaultdict(list)
    for g in want.values():
        by_shard[g["shard_file"]].append(g["group_uid"])
    out_groups = []
    seen: set[str] = set()
    for shard_file, gids in sorted(by_shard.items()):
        payload = torch.load(X.V2_FEATURE_ROOT / shard_file, map_location="cpu", weights_only=False)
        gset = set(gids)
        for g in payload["groups"]:
            if g["group_uid"] not in gset or g["group_uid"] in seen:
                continue
            seen.add(g["group_uid"])
            out_groups.append({"group_uid": g["group_uid"], "task_uid": g["task_uid"],
                               "domain": g["domain"], "split": g["split"],
                               "candidates": [{"candidate_uid": c["candidate_uid"],
                                               "reward": float(c["reward"]),
                                               "features": c["features"].to(torch.float16)}
                                              for c in g["candidates"]]})
        del payload
    assert len(out_groups) == len(want), f"cached ref groups {len(out_groups)} != heldout subset {len(want)}"
    torch.save({"groups": out_groups, "layout": "v2 (3,4,2048) layers[24,36,47] x loops[1..4]"}, out_path)
    print(f"cached ref features saved: {len(out_groups)} heldout groups", flush=True)


# ----------------------------------------------------------------- stage 3: extraction
def extract(subset: list[dict]) -> None:
    man_path = X.FEATURE_ROOT / "feature_manifest.json"
    man = X.read_json(man_path, {"schema_version": "xloop_early.feat.1", "shards": [], "completed_group_uids": []})
    done = set(man["completed_group_uids"])
    order = {"coding": 0, "math": 1, "logic": 2, "reasoning": 3, "alignment": 4}
    pending = [g for g in subset if g["group_uid"] not in done]
    pending.sort(key=lambda g: (order[g["domain"]], g["split"], X.stable_int("ord", g["group_uid"])))
    total_cands = sum(len(g["candidates"]) for g in subset)
    print(f"extract: {len(subset)} groups ({total_cands} cands) total, {len(pending)} pending", flush=True)
    if not pending:
        return

    ext = ExtractorX(device="cuda", dtype="auto", force_all_loops=True)
    shard_next = max((s["idx"] for s in man["shards"]), default=-1) + 1
    buf: list[dict] = []
    buf_cands = 0
    enc = 0
    t0 = time.time()
    t_log = t0

    def flush():
        nonlocal buf, buf_cands, shard_next
        if not buf:
            return
        fname = f"xloop_{shard_next:04d}.pt"
        fpath = X.FEATURE_ROOT / fname
        tmp = fpath.with_suffix(".pt.tmp")
        # shards may mix domains/splits; group records carry their own identity
        torch.save({"schema_version": "xloop_early.feat.1", "groups": buf,
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
        buf = []
        buf_cands = 0

    prev_key = None
    try:
        for g in pending:
            key = (g["domain"], g["split"])
            if prev_key is not None and key != prev_key:
                flush()  # keep shards single-(domain,split) for sane manifests
            prev_key = key
            cands_out = []
            for c in g["candidates"]:
                f = ext.encode_multi_layer(c["candidate_text"], max_length=X.MAX_LEN[g["domain"]])
                cands_out.append({"candidate_uid": c["candidate_uid"], "reward": float(c["reward"]),
                                  "candidate_kind": c.get("candidate_kind"),
                                  "features": f.to(torch.float16)})
                enc += 1
            buf.append({"group_uid": g["group_uid"], "task_uid": g["task_uid"], "domain": g["domain"],
                        "split": g["split"], "source_dataset": g.get("source_dataset"),
                        "candidates": cands_out})
            buf_cands += len(cands_out)
            if buf_cands >= SHARD_MAX_CANDS:
                flush()
            if time.time() - t_log > 120:
                t_log = time.time()
                rate = enc / (time.time() - t0)
                eta_min = (total_cands - len([1 for gg in subset if gg["group_uid"] in done]) * 0 - enc) / max(rate, 1e-9) / 60
                print(f"  ... encoded={enc} rate={rate:.2f}/s eta~{eta_min:.0f}min", flush=True)
        flush()
    finally:
        try:
            ext.cleanup()
        except Exception:
            pass
    man["extraction_seconds"] = round(time.time() - t0, 1)
    man["encoded_this_run"] = enc
    X.write_json(man_path, man)
    print(f"extraction complete: {enc} encoded in {man['extraction_seconds']}s", flush=True)


def main() -> int:
    X.ensure_roots()
    subset = build_subset()
    split_integrity(subset)
    dump_cached_ref(subset)
    extract(subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
