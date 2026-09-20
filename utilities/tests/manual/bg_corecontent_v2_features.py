"""Feature extraction for CoreContent v2 (Parts G/H/I).

Frozen, read-only Ouro forward passes only. No training, no steering, no weight edits.
Each candidate text -> pooled BG features [layers=3, loops=4, hidden=2048] via
src.evaluator.bg_transformer_features.BGTransformerFeatureExtractor. Features are sharded
group-major (a group's candidates never split across shards) by domain/split, written
atomically, checksummed, and resumable via completed group_uids.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

import bg_corecontent_v2_common as v2

OUT_ROOT = v2.OUT_ROOT
FEATURE_ROOT = v2.FEATURE_ROOT
PROC_ROOT = v2.PROC_ROOT
ALL_DOMAINS = v2.ALL_DOMAINS
CORE_DOMAINS = v2.CORE_DOMAINS
HIDDEN_DIM = v2.HIDDEN_DIM
SCHEMA_VERSION = "ccv2.feat.1"

# per-domain encode max_length (alignment text is long; cap to control GPU time)
MAX_LEN = {"alignment": 768, "coding": 1024, "math": 768, "logic": 640, "reasoning": 640,
           "science": 640, "anatomy": 640}
SHARD_MAX_CANDS = 1800  # ~ 1800*3*4*2048*2 bytes ~= 90MB/shard in fp16


def _manifest_path() -> Path:
    return FEATURE_ROOT / "feature_manifest.json"


def _load_manifest() -> dict[str, Any]:
    p = _manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {"schema_version": SCHEMA_VERSION, "shards": [], "completed_group_uids": []}


def _save_manifest(man: dict[str, Any]) -> None:
    tmp = _manifest_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, default=v2.json_default))
    tmp.replace(_manifest_path())


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_groups() -> list[dict[str, Any]]:
    p = PROC_ROOT / "candidate_groups_deduped.jsonl"
    if not p.exists():
        p = PROC_ROOT / "candidate_groups.jsonl"
    return v2.read_jsonl(p)


def _make_extractor():
    import sys
    sys.path.insert(0, str(v2.PROJECT_ROOT))
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return BGTransformerFeatureExtractor(device=dev, dtype="auto", force_all_loops=True)


# ===================================================== PART G: plan + dry run
def feature_plan_main() -> int:
    started = time.time(); v2.ensure_root()
    groups = _load_groups()
    by = defaultdict(lambda: {"groups": 0, "cands": 0, "tok_sum": 0, "tok_n": 0})
    for g in groups:
        b = by[g["domain"]]; b["groups"] += 1; b["cands"] += len(g["candidates"])
    # rough token estimate by characters/4 on a sample
    sample = groups[:: max(1, len(groups) // 400)] if groups else []
    for g in sample:
        b = by[g["domain"]]
        for c in g["candidates"]:
            b["tok_sum"] += min(MAX_LEN.get(g["domain"], 768), len(c["candidate_text"]) // 4)
            b["tok_n"] += 1
    # throughput model from benchmark: enc/sec ~ piecewise by tokens
    def enc_per_sec(tok: float) -> float:
        if tok <= 30:
            return 15.0
        if tok <= 200:
            return 4.0
        if tok <= 600:
            return 2.0
        return 1.3
    plan_rows = []; total_cands = total_sec = total_bytes = 0
    for d in ALL_DOMAINS:
        b = by.get(d)
        if not b or not b["groups"]:
            continue
        avg_tok = (b["tok_sum"] / b["tok_n"]) if b["tok_n"] else 200
        eps = enc_per_sec(avg_tok)
        sec = b["cands"] / eps
        nbytes = b["cands"] * 3 * 4 * HIDDEN_DIM * 2
        plan_rows.append({"domain": d, "groups": b["groups"], "candidates": b["cands"],
                          "avg_tokens_est": round(avg_tok, 1), "enc_per_sec_est": eps,
                          "est_seconds": round(sec), "est_gb": round(nbytes / 1e9, 2)})
        total_cands += b["cands"]; total_sec += sec; total_bytes += nbytes
    est_hours = round(total_sec / 3600, 2)
    est_gb = round(total_bytes / 1e9, 1)
    # dry run: encode up to 100 candidates/domain for real, verify shapes + timing
    dry = {}
    ok = True
    try:
        ext = _make_extractor()
        for d in [x for x in ALL_DOMAINS if by.get(x, {}).get("groups")]:
            gs = [g for g in groups if g["domain"] == d][:30]
            n = 0; t0 = time.time(); shapes_ok = True
            for g in gs:
                for c in g["candidates"]:
                    if n >= 100:
                        break
                    f = ext.encode_text_to_pooled_features(c["candidate_text"], max_length=MAX_LEN.get(d, 768))
                    if tuple(f.shape) != (3, 4, HIDDEN_DIM):
                        shapes_ok = False
                    n += 1
                if n >= 100:
                    break
            dt = time.time() - t0
            dry[d] = {"encoded": n, "shapes_ok": shapes_ok, "enc_per_sec_real": round(n / dt, 2) if dt else None}
            print(f"  dry-run {d}: encoded={n} enc/sec={dry[d]['enc_per_sec_real']} shapes_ok={shapes_ok}", flush=True)
        try:
            ext.cleanup()
        except Exception:
            pass
    except Exception as ex:
        ok = False; dry["error"] = str(ex)[:200]
    if not ok:
        verdict = "DRY_RUN_FAILED"
    elif est_gb > 350 or est_hours > 36:
        verdict = "TOO_LARGE_NEEDS_SAMPLING"
    elif est_hours > 12 or est_gb > 120:
        verdict = "LARGE_BUT_FEASIBLE"
    else:
        verdict = "READY"
    payload = {"BG_CORECONTENT_V2_FEATURE_PLAN_VERDICT": verdict, "plan_rows": plan_rows,
               "total_candidates": total_cands, "est_hours": est_hours, "est_feature_gb": est_gb,
               "max_len": MAX_LEN, "shard_max_cands": SHARD_MAX_CANDS, "dry_run": dry,
               "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "feature_extraction_plan.json", payload)
    v2.write_json(v2.DATA_ROOT / "feature_extraction_plan.json", payload)
    v2.write_md(OUT_ROOT / "feature_extraction_plan.md", v2._md_top("CoreContent v2 Feature Extraction Plan",
        "BG_CORECONTENT_V2_FEATURE_PLAN_VERDICT", verdict, [
        f"Total candidates to encode: {total_cands}. Est time: {est_hours} h. Est feature storage: {est_gb} GB (fp16).", "",
        *v2.md_table(plan_rows, ["domain", "groups", "candidates", "avg_tokens_est", "enc_per_sec_est", "est_seconds", "est_gb"]),
        "", f"Dry run (real encodes): {dry}.",
    ]))
    v2.progress("partG_feature_plan", {"verdict": verdict, "est_hours": est_hours, "est_gb": est_gb})
    print(v2.status_line("BG_CORECONTENT_V2_FEATURE_PLAN_VERDICT", verdict))
    print(f"  total_candidates={total_cands} est_hours={est_hours} est_gb={est_gb}")
    return 1 if verdict == "DRY_RUN_FAILED" else 0


# ===================================================== PART H: extraction
def extract_features_main() -> int:
    started = time.time(); v2.ensure_root()
    groups = _load_groups()
    man = _load_manifest()
    done = set(man.get("completed_group_uids", []))
    shards = man.get("shards", [])
    # order: cheap domains first so partial runs still yield core coverage; alignment last
    order = {"coding": 0, "math": 1, "logic": 2, "reasoning": 3, "science": 4, "anatomy": 5, "alignment": 6}
    groups.sort(key=lambda g: (order.get(g["domain"], 9), g["split"], v2.stable_int("ord", g["group_uid"])))
    pending = [g for g in groups if g["group_uid"] not in done]
    print(f"  extraction: {len(groups)} groups total, {len(pending)} pending, {len(done)} done", flush=True)
    if not pending:
        return _finalize_extraction(man, started)
    ext = _make_extractor()
    shard_idx = {}
    for s in shards:
        key = (s["domain"], s["split"])
        shard_idx[key] = max(shard_idx.get(key, -1), s["idx"])
    buf: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    buf_cands: dict[tuple, int] = defaultdict(int)
    enc_total = 0; t_log = time.time(); errors = 0

    def flush(key: tuple) -> None:
        nonlocal shards
        if not buf[key]:
            return
        dom, split = key
        idx = shard_idx.get(key, -1) + 1
        shard_idx[key] = idx
        fname = f"{dom}_{split}_{idx:04d}.pt"
        fpath = FEATURE_ROOT / fname
        tmp = fpath.with_suffix(".pt.tmp")
        torch.save({"schema_version": SCHEMA_VERSION, "domain": dom, "split": split,
                    "groups": buf[key]}, tmp)
        tmp.replace(fpath)
        rec = {"file": fname, "domain": dom, "split": split, "idx": idx,
               "groups": len(buf[key]), "candidates": buf_cands[key], "checksum": _checksum(fpath)}
        shards.append(rec)
        for g in buf[key]:
            done.add(g["group_uid"])
        man["shards"] = shards; man["completed_group_uids"] = sorted(done)
        man["schema_version"] = SCHEMA_VERSION
        _save_manifest(man)
        v2.progress("partH_extract", {"shards": len(shards), "groups_done": len(done),
                                       "encoded": enc_total, "errors": errors})
        print(f"    wrote {fname}: groups={len(buf[key])} cands={buf_cands[key]} (total_done={len(done)})", flush=True)
        buf[key] = []; buf_cands[key] = 0

    try:
        for g in pending:
            key = (g["domain"], g["split"])
            enc_cands = []
            for c in g["candidates"]:
                try:
                    f = ext.encode_text_to_pooled_features(c["candidate_text"], max_length=MAX_LEN.get(g["domain"], 768))
                    enc_cands.append({"candidate_uid": c["candidate_uid"], "candidate_kind": c["candidate_kind"],
                                      "reward": float(c["reward"]), "features": f.to(torch.float16),
                                      "branch_id": c["candidate_uid"]})
                    enc_total += 1
                except Exception:
                    errors += 1
            if len(enc_cands) < 2 or len({c["reward"] > 0 for c in enc_cands}) < 2:
                done.add(g["group_uid"])  # mark handled but unusable
                continue
            buf[key].append({"group_uid": g["group_uid"], "task_uid": g["task_uid"], "domain": g["domain"],
                             "split": g["split"], "kind": g["kind"], "source_dataset": g["source_dataset"],
                             "candidates": enc_cands})
            buf_cands[key] += len(enc_cands)
            if buf_cands[key] >= SHARD_MAX_CANDS:
                flush(key)
            if time.time() - t_log > 120:
                t_log = time.time()
                rate = enc_total / (time.time() - started)
                print(f"  ... encoded={enc_total} errors={errors} rate={rate:.1f}/s done={len(done)}", flush=True)
        for key in list(buf.keys()):
            flush(key)
    finally:
        try:
            ext.cleanup()
        except Exception:
            pass
    return _finalize_extraction(man, started, enc_total=enc_total, errors=errors)


def _finalize_extraction(man: dict[str, Any], started: float, enc_total: int = 0, errors: int = 0) -> int:
    shards = man.get("shards", [])
    by_dom = Counter(); by_dom_cands = Counter()
    for s in shards:
        by_dom[s["domain"]] += s["groups"]; by_dom_cands[s["domain"]] += s["candidates"]
    core_min = {d: by_dom.get(d, 0) for d in CORE_DOMAINS}
    unmet = [d for d in CORE_DOMAINS if core_min[d] < v2.MIN_REWARD_DIVERSE[d]]
    if not shards:
        verdict = "BLOCKED"
    elif errors > 0 and not unmet:
        verdict = "READY"
    elif not unmet:
        verdict = "READY"
    elif len(unmet) <= 2 and "alignment" not in unmet:
        verdict = "LARGE_PARTIAL_USABLE"
    else:
        verdict = "DOMAIN_GAPS"
    total_gb = round(sum((FEATURE_ROOT / s["file"]).stat().st_size for s in shards if (FEATURE_ROOT / s["file"]).exists()) / 1e9, 2)
    payload = {"BG_CORECONTENT_V2_FEATURE_EXTRACTION_VERDICT": verdict, "shards": len(shards),
               "groups_by_domain": dict(by_dom), "candidates_by_domain": dict(by_dom_cands),
               "feature_gb": total_gb, "encoded_this_run": enc_total, "errors": errors,
               "unmet_minimum": unmet, "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "feature_extraction.json", payload)
    v2.write_md(OUT_ROOT / "feature_extraction.md", v2._md_top("CoreContent v2 Feature Extraction",
        "BG_CORECONTENT_V2_FEATURE_EXTRACTION_VERDICT", verdict, [
        f"Shards: {len(shards)}. Feature storage: {total_gb} GB. Encoded this run: {enc_total}. Errors: {errors}.", "",
        *v2.md_table([{"domain": d, "groups": by_dom.get(d, 0), "candidates": by_dom_cands.get(d, 0)} for d in ALL_DOMAINS],
                     ["domain", "groups", "candidates"]),
        "", f"Unmet core minimums: {unmet or 'none'}.",
    ]))
    v2.progress("partH_extract", {"verdict": verdict, "shards": len(shards), "groups_by_domain": dict(by_dom)})
    print(v2.status_line("BG_CORECONTENT_V2_FEATURE_EXTRACTION_VERDICT", verdict))
    print(f"  shards={len(shards)} groups_by_domain={dict(by_dom)} feature_gb={total_gb}")
    return 0


# ===================================================== group loading for models
_GROUP_CACHE: dict[str, list[dict[str, Any]]] = {}


def load_feature_groups(domains: Sequence[str] | None = None, splits: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Reconstruct cc-format groups from feature shards (candidate['features'] = (3,4,2048))."""
    man = _load_manifest()
    out: list[dict[str, Any]] = []
    for s in man.get("shards", []):
        if domains and s["domain"] not in domains:
            continue
        if splits and s["split"] not in splits:
            continue
        key = s["file"]
        if key not in _GROUP_CACHE:
            payload = torch.load(FEATURE_ROOT / s["file"], map_location="cpu", weights_only=False)
            grps = []
            for g in payload.get("groups", []):
                cands = [{"features": c["features"].to(torch.float32), "reward": float(c["reward"]),
                          "branch_id": c.get("branch_id"), "candidate_kind": c.get("candidate_kind")}
                         for c in g["candidates"] if hasattr(c.get("features"), "shape")]
                if len(cands) < 2:
                    continue
                grps.append({"group_id": g["group_uid"], "task_id": g["task_uid"], "domain": g["domain"],
                             "split": g["split"], "kind": g.get("kind"), "source": g.get("source_dataset"),
                             "source_dataset": g.get("source_dataset"), "candidates": cands})
            _GROUP_CACHE[key] = grps
        out.extend(_GROUP_CACHE[key])
    return out


# ===================================================== PART I: dataset balance
def dataset_balance_main() -> int:
    started = time.time(); v2.ensure_root()
    man = _load_manifest()
    groups = load_feature_groups()
    by_dom = defaultdict(lambda: {"groups": 0, "cands": 0, "reward_diverse": 0, "train": 0, "val": 0, "heldout": 0,
                                  "tie": 0})
    for g in groups:
        b = by_dom[g["domain"]]; b["groups"] += 1; b["cands"] += len(g["candidates"])
        if len({c["reward"] > 0 for c in g["candidates"]}) > 1:
            b["reward_diverse"] += 1
        b[g.get("split", "train")] = b.get(g.get("split", "train"), 0) + 1
        rs = [c["reward"] for c in g["candidates"]]
        if rs.count(max(rs)) > 1:
            b["tie"] += 1
    rows = []
    for d in ALL_DOMAINS:
        b = by_dom.get(d)
        if not b:
            rows.append({"domain": d, "groups": 0, "reward_diverse": 0, "v1_reward_diverse": v2.V1_COVERAGE.get(d, 0)})
            continue
        rows.append({"domain": d, "groups": b["groups"], "candidates": b["cands"],
                     "reward_diverse": b["reward_diverse"], "v1_reward_diverse": v2.V1_COVERAGE.get(d, 0),
                     "train": b.get("train", 0), "val": b.get("val", 0), "heldout": b.get("heldout", 0),
                     "tie_rate": round(b["tie"] / b["groups"], 3) if b["groups"] else None,
                     "avg_group_size": round(b["cands"] / b["groups"], 2) if b["groups"] else 0})
    core_div = {d: by_dom.get(d, {}).get("reward_diverse", 0) for d in CORE_DOMAINS}
    feat_gb = round(sum((FEATURE_ROOT / s["file"]).stat().st_size for s in man.get("shards", [])
                        if (FEATURE_ROOT / s["file"]).exists()) / 1e9, 2)
    unmet = [d for d in CORE_DOMAINS if core_div.get(d, 0) < v2.MIN_REWARD_DIVERSE[d]]
    improved = all(core_div.get(d, 0) >= v2.V1_COVERAGE.get(d, 0) for d in CORE_DOMAINS)
    if not unmet:
        verdict = "LARGE_CORE_DATA_READY"
    elif improved and len(unmet) <= 2:
        verdict = "MUCH_IMPROVED"
    elif unmet == ["reasoning"]:
        verdict = "STILL_REASONING_LIMITED"
    elif unmet == ["coding"]:
        verdict = "STILL_CODING_LIMITED"
    elif set(unmet) <= {"math", "logic"}:
        verdict = "STILL_MATH_LOGIC_LIMITED"
    elif by_dom.get("alignment", {}).get("reward_diverse", 0) > 5 * max(1, sum(core_div.get(d, 0) for d in ("coding", "reasoning", "math", "logic"))):
        verdict = "ALIGNMENT_DOMINATES"
    else:
        verdict = "DATA_LIMITED"
    payload = {"BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT": verdict, "rows": rows, "core_reward_diverse": core_div,
               "feature_gb": feat_gb, "shards": len(man.get("shards", [])), "unmet_minimum": unmet,
               "v1_vs_v2": {d: {"v1": v2.V1_COVERAGE.get(d, 0), "v2": core_div.get(d, 0)} for d in CORE_DOMAINS},
               "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "dataset_balance.json", payload)
    v2.write_csv(OUT_ROOT / "dataset_balance.csv", rows)
    v2.write_md(OUT_ROOT / "dataset_balance.md", v2._md_top("CoreContent v2 Dataset Balance (v1 vs v2)",
        "BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT", verdict, [
        f"Feature storage: {feat_gb} GB across {len(man.get('shards', []))} shards.", "",
        *v2.md_table(rows, ["domain", "groups", "reward_diverse", "v1_reward_diverse", "train", "val", "heldout",
                            "tie_rate", "avg_group_size"]),
        "", f"Core reward-diverse v2: {core_div}. Unmet minimums: {unmet or 'none'}.",
    ]))
    v2.progress("partI_dataset_balance", {"verdict": verdict, "core_reward_diverse": core_div})
    print(v2.status_line("BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT", verdict))
    print(f"  core_reward_diverse={core_div} feature_gb={feat_gb}")
    return 0
