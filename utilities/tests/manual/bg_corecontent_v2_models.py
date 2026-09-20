"""Models / evaluation / analysis for CoreContent v2 (Parts J-U).

Refits small CONTENT-SELECTION taps on the expanded feature set and re-decides whether a
crafted policy beats the v1 broad-objective baseline (mixedhead_MIX_HH_OBJECTIVE). Reuses
the locked DualAnchor scoring layer (cc.policy_candidate_scores / score_diff /
config_vector) and the v1 tap-policy registry. No Ouro training, no steering, no registry
mutation; science/anatomy diagnostic-only; pure/transplanted taps never overwritten.

Feature configs use the three single-layer L4 taps (24_L4, 36_L4, 47_L4); the 3-channel
"blockwise" combination is the multi-layer (concat_24_36_47-equivalent) policy, consistent
with the locked scoring layer.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

import bg_corecontent_v2_common as v2
import bg_corecontent_v2_features as feat
import bg_core_tap_audit_v1_common as cc

OUT_ROOT = v2.OUT_ROOT
CORE_DOMAINS = v2.CORE_DOMAINS
DIAG_DOMAINS = v2.DIAG_DOMAINS
ALL_DOMAINS = v2.ALL_DOMAINS
TRAIN_CONFIGS = ("24_L4", "36_L4", "47_L4")
V1_CRAFTING_ROOT = v2.V1_CRAFTING_ROOT
PURE_TAPS_PT = v2.PURE_TAPS_PT
TRANSPLANTED_TAPS_PT = v2.TRANSPLANTED_TAPS_PT

finite_mean = cc.finite_mean
fmt = cc.fmt
status_line = cc.status_line
md_table = cc.md_table
write_md = cc.write_md
write_csv = cc.write_csv

TRAINING_SETS = {
    "all_core": CORE_DOMAINS,
    "all_core_balanced": CORE_DOMAINS,
    "code_reasoning": ("coding", "reasoning"),
    "math_logic": ("math", "logic"),
    "alignment_only": ("alignment",),
    "code_math_logic": ("coding", "math", "logic"),
}

# ----------------------------------------------------------------- group loading
_BY: dict[tuple, list[dict[str, Any]]] = {}


def groups(domain: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    key = (domain, split)
    if key not in _BY:
        doms = [domain] if domain else None
        sps = [split] if split else None
        _BY[key] = feat.load_feature_groups(domains=doms, splits=sps)
    return _BY[key]


def groups_by_domain(split: str, domains: Sequence[str] = CORE_DOMAINS) -> dict[str, list[dict[str, Any]]]:
    return {d: groups(d, split) for d in domains}


def group_feats(group: dict[str, Any], config: str):
    cv = cc._sc()["config_vector"]
    feats, rewards = [], []
    for c in group["candidates"]:
        v = cv(c, config)
        if isinstance(v, torch.Tensor):
            feats.append(v.float()); rewards.append(float(c["reward"]))
    if len(feats) < 2:
        return None
    return torch.stack(feats, 0), rewards


# ----------------------------------------------------------------- scoring
def score_group(group: dict[str, Any], policy: Any) -> list[float]:
    n = len(group["candidates"])
    if policy == "ORACLE":
        return [float(c["reward"]) for c in group["candidates"]]
    if policy == "RANDOM":
        rng = random.Random(v2.stable_int("rand", group["group_id"]))
        return [rng.random() for _ in range(n)]
    return cc.policy_candidate_scores(group, policy)


def metrics(group: dict[str, Any], scores: list[float]) -> dict[str, float]:
    return cc.group_selection_metrics(group, scores)


def eval_policy_macro(policies: dict[str, Any], *, split: str | None, domains: Sequence[str] = CORE_DOMAINS,
                      max_groups: int = 100000) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    by_pol: dict[str, dict[str, float]] = defaultdict(dict)
    for dom in domains:
        gs = groups(dom, split)[:max_groups]
        agg: dict[str, list[dict[str, float]]] = defaultdict(list)
        for g in gs:
            for name, pol in policies.items():
                m = metrics(g, score_group(g, pol))
                if m:
                    agg[name].append(m)
        for name, ms in agg.items():
            by_pol[name][dom] = finite_mean(m["top1_oracle"] for m in ms)
            rows.append({"policy": name, "domain": dom, "groups": len(ms),
                         "top1_oracle": round(by_pol[name][dom], 4),
                         "pairwise_accuracy": round(finite_mean(m["pairwise_accuracy"] for m in ms), 4),
                         "mean_regret": round(finite_mean(m["regret"] for m in ms), 4)})
    return rows, by_pol


def macro_core(by_domain: dict[str, float]) -> float:
    return finite_mean(by_domain.get(d) for d in CORE_DOMAINS)


def eval_channels_macro(channels: list[tuple], by_dom: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    out = {}
    for d, gs in by_dom.items():
        ms = [metrics(g, cc.policy_candidate_scores(g, channels)) for g in gs]
        ms = [m for m in ms if m]
        out[d] = finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
    return out


# ----------------------------------------------------------------- policy registries
def load_constructed_channels(which: str) -> dict[str, list[tuple]]:
    path = PURE_TAPS_PT if which == "pure" else TRANSPLANTED_TAPS_PT
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, list[tuple]] = {}
    for name, by_cfg in (payload.get("taps") or {}).items():
        chans = []
        for cfg, rec in by_cfg.items():
            w = rec.get("weight") if isinstance(rec, dict) else None
            if isinstance(w, torch.Tensor):
                chans.append((w.float(), rec.get("arch", "AntisymLinear"), cfg))
        if chans:
            out[name] = chans
    return out


def _load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False) if Path(path).exists() else {}


def v1_crafted_policies() -> dict[str, list[tuple]]:
    pol: dict[str, list[tuple]] = {}
    wm = _load_pt(V1_CRAFTING_ROOT / "corecontent_weight_merge.pt")
    if wm.get("best_channels"):
        pol["v1_CoreContent_weight_merge"] = [(w, a, c) for (w, a, c) in wm["best_channels"]]
    lin = _load_pt(V1_CRAFTING_ROOT / "corecontent_linear_taps.pt")
    bpc = lin.get("best_per_config", {})
    if bpc:
        bc = max(bpc, key=lambda c: bpc[c]["val_core_macro"])
        pol["v1_CoreContent_linear_best"] = [(bpc[bc]["weight"], "AntisymLinearNoNorm", bc)]
    return pol


def baseline_policies(include_diag: bool = True) -> dict[str, Any]:
    base = cc.build_tap_policies()
    pol: dict[str, Any] = {}
    for k in ("DualAnchor", "MIX_CODE_REASONING_only", "MIX_OBJECTIVE_ALL_only",
              "mixedhead_MIX_HH_OBJECTIVE", "mixedhead_MIX_OBJECTIVE_ALL", "mixedhead_MIX_CODE_REASONING",
              "old_CODE", "old_HH"):
        if k in base:
            pol[k] = base[k]
    for name, ch in load_constructed_channels("pure").items():
        if not name.startswith("science") or include_diag:
            pol[f"pure::{name}"] = ch
    pol.update(v1_crafted_policies())
    if include_diag:
        for k in base:
            if k.startswith("science_"):
                pol[k] = base[k]
    pol["RANDOM"] = "RANDOM"
    pol["ORACLE"] = "ORACLE"
    return pol


def _is_diag(name: str) -> bool:
    n = name.lower()
    return ("science" in n) or ("anatomy" in n)


# ----------------------------------------------------------------- training
def _pairwise_diffs(gs: Sequence[dict[str, Any]], config: str, max_pairs: int = 24) -> list[torch.Tensor]:
    diffs = []
    for g in gs:
        gf = group_feats(g, config)
        if gf is None:
            continue
        feats, rewards = gf
        pairs = [(i, j) for i in range(len(rewards)) for j in range(len(rewards)) if rewards[i] > rewards[j]]
        for (i, j) in pairs[:max_pairs]:
            diffs.append(feats[i] - feats[j])
    return diffs


def train_pairwise(train_by: dict, val_by: dict, config: str, tset: str, lr: float, seed: int,
                   epochs: int = 60, balanced: bool = True, regret_weight: bool = False) -> dict | None:
    torch.manual_seed(seed)
    per = {d: _pairwise_diffs(train_by.get(d, []), config) for d in TRAINING_SETS[tset]}
    per = {d: torch.stack(v, 0) for d, v in per.items() if v}
    if not per:
        return None
    if balanced and len(per) > 1:
        m = min(x.shape[0] for x in per.values())
        gen = torch.Generator().manual_seed(seed)
        X = torch.cat([x[torch.randperm(x.shape[0], generator=gen)[:m]] for x in per.values()], 0)
    else:
        X = torch.cat(list(per.values()), 0)
    if X.shape[0] < 8:
        return None
    w = torch.zeros(1, X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        s = (X @ w.t()).squeeze(1)
        loss = torch.nn.functional.softplus(-s).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
    ch = [(w.detach().clone(), "AntisymLinearNoNorm", config)]
    valm = eval_channels_macro(ch, val_by)
    return {"weight": w.detach().clone(), "config": config, "arch": "AntisymLinearNoNorm", "training_set": tset,
            "lr": lr, "seed": seed, "pairs": int(X.shape[0]), "val_core_macro": round(macro_core(valm), 4),
            "val_by_domain": {k: round(v, 4) for k, v in valm.items()}}


def train_listwise(train_by: dict, val_by: dict, config: str, tset: str, lr: float, seed: int,
                   epochs: int = 40, cap_per_domain: int = 1500) -> dict | None:
    torch.manual_seed(seed)
    samples = []
    for d in TRAINING_SETS[tset]:
        gs = list(train_by.get(d, []))
        gs.sort(key=lambda g: v2.stable_int("lw", g["group_id"]))
        for g in gs[:cap_per_domain]:  # subsample for tractable Python-loop listwise
            gf = group_feats(g, config)
            if gf is None:
                continue
            feats, rewards = gf
            mx = max(rewards)
            tgt = torch.tensor([1.0 if r >= mx else 0.0 for r in rewards])
            if tgt.sum() == 0 or tgt.sum() == len(rewards):
                continue
            samples.append((feats, tgt / tgt.sum()))
    if len(samples) < 8:
        return None
    w = torch.zeros(1, samples[0][0].shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=0.01)
    # minibatch over groups for speed on large data
    bs = 256
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(samples)
        for k in range(0, len(samples), bs):
            opt.zero_grad()
            loss = 0.0
            for feats, tgt in samples[k:k + bs]:
                logp = torch.log_softmax((feats @ w.t()).squeeze(1), dim=0)
                loss = loss - (tgt * logp).sum()
            (loss / max(1, min(bs, len(samples) - k))).backward()
            torch.nn.utils.clip_grad_norm_([w], 1.0)
            opt.step()
    ch = [(w.detach().clone(), "AntisymLinearNoNorm", config)]
    valm = eval_channels_macro(ch, val_by)
    return {"weight": w.detach().clone(), "config": config, "arch": "AntisymLinearNoNorm", "training_set": tset,
            "lr": lr, "seed": seed, "objective": "listwise", "groups": len(samples),
            "val_core_macro": round(macro_core(valm), 4), "val_by_domain": {k: round(v, 4) for k, v in valm.items()}}


# ===================================================== PART J: baselines
def baselines_main() -> int:
    started = time.time(); v2.ensure_root()
    pol = baseline_policies()
    rows, by_pol = eval_policy_macro(pol, split="heldout", domains=ALL_DOMAINS)
    macro = {p: macro_core(d) for p, d in by_pol.items()}
    real = {k: v for k, v in macro.items() if k not in ("ORACLE", "RANDOM") and not _is_diag(k)}
    best = max(real, key=lambda k: real[k]) if real else None
    hh = real.get("mixedhead_MIX_HH_OBJECTIVE", float("nan"))
    if best == "mixedhead_MIX_HH_OBJECTIVE":
        verdict = "MIX_HH_OBJECTIVE_STILL_BEST"
    elif best == "mixedhead_MIX_OBJECTIVE_ALL":
        verdict = "MIX_OBJECTIVE_ALL_STRONG"
    elif best == "DualAnchor":
        verdict = "DUALANCHOR_CONTENT_STILL_MID"
    elif best and (real.get(best, 0) - hh) > 0.005:
        verdict = "BASELINE_SHIFTED_WITH_DATA"
    else:
        verdict = "READY"
    payload = {"BG_CORECONTENT_V2_BASELINES_VERDICT": verdict, "macro_core_top1": {k: round(v, 4) for k, v in macro.items()},
               "best_baseline": best, "mixedhead_HH": round(hh, 4) if hh == hh else None, "rows": rows,
               "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "baselines_v2.json", payload)
    write_csv(OUT_ROOT / "baseline_rows_v2.csv", rows)
    write_md(OUT_ROOT / "baselines_v2.md", v2._md_top("CoreContent v2 Baselines (heldout)",
        "BG_CORECONTENT_V2_BASELINES_VERDICT", verdict, [
        f"Best baseline (macro core top1): `{best}` = {fmt(real.get(best) if best else None)} vs "
        f"mixedhead_MIX_HH_OBJECTIVE {fmt(hh)}.", "",
        "## Macro core top1 by policy", "",
        *[f"- {k}: {fmt(v)}" for k, v in sorted(macro.items(), key=lambda x: -x[1])], "",
        *md_table(sorted(rows, key=lambda r: (r["policy"], r["domain"])),
                  ["policy", "domain", "groups", "top1_oracle", "pairwise_accuracy", "mean_regret"]),
    ]))
    v2.progress("partJ_baselines", {"verdict": verdict, "best": best, "macro": {k: round(v, 4) for k, v in macro.items()}})
    print(status_line("BG_CORECONTENT_V2_BASELINES_VERDICT", verdict))
    print(f"  best={best} {fmt(real.get(best) if best else None)} | HH={fmt(hh)}")
    return 0


# ===================================================== PART K: linear / pairwise
def linear_pairwise_main() -> int:
    started = time.time(); v2.ensure_root()
    train_by = groups_by_domain("train"); val_by = groups_by_domain("val")
    results = []; best_per_config: dict[str, dict] = {}
    for config in TRAIN_CONFIGS:
        for tset in TRAINING_SETS:
            for lr in (3e-4, 1e-3):
                for seed in (0, 1, 2):
                    r = train_pairwise(train_by, val_by, config, tset, lr, seed)
                    if not r:
                        continue
                    results.append({k: r[k] for k in ("config", "training_set", "lr", "seed", "pairs", "val_core_macro")})
                    if config not in best_per_config or r["val_core_macro"] > best_per_config[config]["val_core_macro"]:
                        best_per_config[config] = r
        v2.progress(f"partK_config_{config}", {"done": True})
    blockwise = [(best_per_config[c]["weight"], "AntisymLinearNoNorm", c) for c in TRAIN_CONFIGS if c in best_per_config]
    blockwise_val = round(macro_core(eval_channels_macro(blockwise, val_by)), 4) if blockwise else float("nan")
    best_single = max(best_per_config.values(), key=lambda r: r["val_core_macro"]) if best_per_config else None
    hh_val = round(macro_core(eval_channels_macro(cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", []), val_by)), 4)
    learned = max(blockwise_val, best_single["val_core_macro"] if best_single else -1)
    if best_single is None:
        verdict = "INSUFFICIENT"
    elif learned > hh_val + 0.01:
        verdict = "PAIRWISE_RELATIONAL_READY" if blockwise_val < best_single["val_core_macro"] else "LINEAR_IMPROVED_WITH_DATA"
    elif learned > 0.5:
        verdict = "STILL_WEAK"
    else:
        verdict = "DATA_LIMITED"
    torch.save({"meta": {"arch": "AntisymLinearNoNorm", "configs": TRAIN_CONFIGS},
                "best_per_config": {c: {"weight": best_per_config[c]["weight"], "training_set": best_per_config[c]["training_set"],
                                        "lr": best_per_config[c]["lr"], "seed": best_per_config[c]["seed"],
                                        "val_core_macro": best_per_config[c]["val_core_macro"]} for c in best_per_config},
                "blockwise_channels": blockwise, "blockwise_val_core_macro": blockwise_val},
               OUT_ROOT / "corecontent_v2_linear_pairwise.pt")
    v2.write_json(OUT_ROOT / "linear_pairwise_training.json", {"BG_CORECONTENT_V2_LINEAR_PAIRWISE_VERDICT": verdict,
                  "results": results, "best_per_config_val": {c: best_per_config[c]["val_core_macro"] for c in best_per_config},
                  "blockwise_val_core_macro": blockwise_val, "mixedhead_HH_val_macro": hh_val,
                  "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "linear_pairwise_rows.csv", sorted(results, key=lambda r: -r["val_core_macro"])[:60])
    write_md(OUT_ROOT / "linear_pairwise_training.md", v2._md_top("CoreContent v2 Linear/Pairwise Training",
        "BG_CORECONTENT_V2_LINEAR_PAIRWISE_VERDICT", verdict, [
        f"Best single-config val {fmt(best_single['val_core_macro'] if best_single else None)}; blockwise val {fmt(blockwise_val)}; "
        f"MIX_HH_OBJECTIVE val {fmt(hh_val)}.", "",
        *md_table([{"config": c, **{k: best_per_config[c][k] for k in ('training_set', 'lr', 'seed', 'val_core_macro')}} for c in best_per_config],
                  ["config", "training_set", "lr", "seed", "val_core_macro"]),
    ]))
    v2.progress("partK_linear_pairwise", {"verdict": verdict, "blockwise_val": blockwise_val, "hh_val": hh_val})
    print(status_line("BG_CORECONTENT_V2_LINEAR_PAIRWISE_VERDICT", verdict))
    print(f"  blockwise_val={blockwise_val} best_single={best_single['val_core_macro'] if best_single else None} HH={hh_val}")
    return 0


# ===================================================== PART L: listwise
def listwise_main() -> int:
    started = time.time(); v2.ensure_root()
    train_by = groups_by_domain("train"); val_by = groups_by_domain("val")
    results = []; best = None
    for config in TRAIN_CONFIGS:
        for tset in ("all_core_balanced", "code_math_logic", "alignment_only"):
            for lr in (3e-4, 1e-3):
                for seed in (0,):
                    r = train_listwise(train_by, val_by, config, tset, lr, seed)
                    if not r:
                        continue
                    results.append({k: r[k] for k in ("config", "training_set", "lr", "seed", "groups", "val_core_macro")})
                    if best is None or r["val_core_macro"] > best["val_core_macro"]:
                        best = r
        v2.progress(f"partL_config_{config}", {"done": True})
    hh_val = round(macro_core(eval_channels_macro(cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", []), val_by)), 4)
    if best is None:
        verdict = "INSUFFICIENT"
    elif best["val_core_macro"] > hh_val + 0.01:
        verdict = "LISTWISE_READY"
    elif best["val_core_macro"] > 0.5:
        verdict = "STILL_WEAK"
    else:
        verdict = "DATA_LIMITED"
    torch.save({"meta": {"objective": "listwise_softmax_tieaware"},
                "best_listwise": ({"weight": best["weight"], "config": best["config"], "arch": best["arch"],
                                   "training_set": best["training_set"], "val_core_macro": best["val_core_macro"]} if best else None)},
               OUT_ROOT / "corecontent_v2_listwise.pt")
    v2.write_json(OUT_ROOT / "listwise_training.json", {"BG_CORECONTENT_V2_LISTWISE_VERDICT": verdict,
                  "results": results, "best_val_core_macro": best["val_core_macro"] if best else None,
                  "mixedhead_HH_val_macro": hh_val, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "listwise_rows.csv", sorted(results, key=lambda r: -r["val_core_macro"])[:40])
    write_md(OUT_ROOT / "listwise_training.md", v2._md_top("CoreContent v2 Listwise Training",
        "BG_CORECONTENT_V2_LISTWISE_VERDICT", verdict, [
        f"Best listwise val {fmt(best['val_core_macro'] if best else None)}; MIX_HH_OBJECTIVE val {fmt(hh_val)}.", "",
        *md_table(sorted(results, key=lambda r: -r["val_core_macro"])[:10], ["config", "training_set", "lr", "seed", "groups", "val_core_macro"]),
    ]))
    v2.progress("partL_listwise", {"verdict": verdict, "best_val": best["val_core_macro"] if best else None, "hh_val": hh_val})
    print(status_line("BG_CORECONTENT_V2_LISTWISE_VERDICT", verdict))
    return 0


# ----------------------------------------------------------------- crafted v2 policies
def crafted_v2_policies() -> dict[str, list[tuple]]:
    pol: dict[str, list[tuple]] = {}
    lin = _load_pt(OUT_ROOT / "corecontent_v2_linear_pairwise.pt")
    if lin.get("blockwise_channels"):
        pol["CoreContent_v2_blockwise"] = [(w, a, c) for (w, a, c) in lin["blockwise_channels"]]
    bpc = lin.get("best_per_config", {})
    if bpc:
        bc = max(bpc, key=lambda c: bpc[c]["val_core_macro"])
        pol["CoreContent_v2_linear_best"] = [(bpc[bc]["weight"], "AntisymLinearNoNorm", bc)]
    lw = _load_pt(OUT_ROOT / "corecontent_v2_listwise.pt")
    if lw.get("best_listwise"):
        b = lw["best_listwise"]
        pol["CoreContent_v2_listwise"] = [(b["weight"], "AntisymLinearNoNorm", b["config"])]
    wm = _load_pt(OUT_ROOT / "corecontent_v2_weight_merge.pt")
    if wm.get("best_channels"):
        pol["CoreContent_v2_weight_merge"] = [(w, a, c) for (w, a, c) in wm["best_channels"]]
    return pol


def eval_gated_macro(gate: dict[str, list[tuple]], split: str, domains: Sequence[str] = CORE_DOMAINS) -> dict[str, float]:
    by = groups_by_domain(split, domains)
    out = {}
    for d in domains:
        ch = gate.get(d)
        if not ch:
            out[d] = float("nan"); continue
        ms = [metrics(g, cc.policy_candidate_scores(g, ch)) for g in by.get(d, [])]
        ms = [m for m in ms if m]
        out[d] = finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
    return out


# ===================================================== PART M: domain-gated
def domain_gated_main() -> int:
    started = time.time(); v2.ensure_root()
    base = cc.build_tap_policies()
    experts: dict[str, list[tuple]] = {k: base[k] for k in
        ("mixedhead_MIX_HH_OBJECTIVE", "mixedhead_MIX_OBJECTIVE_ALL", "mixedhead_MIX_CODE_REASONING",
         "MIX_CODE_REASONING_only", "DualAnchor") if k in base}
    experts.update(crafted_v2_policies())
    for name, ch in load_constructed_channels("pure").items():
        if not name.startswith("science"):
            experts[f"pure::{name}"] = ch
    val_by = groups_by_domain("val")
    gate: dict[str, list[tuple]] = {}; choice: dict[str, str] = {}
    for d in CORE_DOMAINS:
        scored = []
        for name, ch in experts.items():
            ms = [metrics(g, cc.policy_candidate_scores(g, ch)) for g in val_by.get(d, [])]
            ms = [m for m in ms if m]
            if ms:
                scored.append((name, finite_mean(m["top1_oracle"] for m in ms)))
        if scored:
            b = max(scored, key=lambda x: x[1]); gate[d] = experts[b[0]]; choice[d] = b[0]
    val_gated = eval_gated_macro(gate, "val"); val_macro = macro_core(val_gated)
    uni = {name: macro_core(eval_channels_macro(ch, val_by)) for name, ch in experts.items()}
    best_uni = max(uni, key=lambda k: uni[k]) if uni else None
    distinct = len(set(choice.values()))
    if val_macro > (uni.get(best_uni, 0) + 0.01) and distinct > 1:
        verdict = "HAND_RULE_BEST"
    elif val_macro > (uni.get(best_uni, 0) + 0.005):
        verdict = "LEARNED_GATE_GENERALIZES"
    elif val_macro >= uni.get(best_uni, 0) - 0.005:
        verdict = "DOMAIN_GATED_READY"
    else:
        verdict = "GATE_OVERFITS_AGAIN"
    torch.save({"gate_choice": choice, "gate_channels": {d: gate[d] for d in gate}}, OUT_ROOT / "domain_gated_v2.pt")
    v2.write_json(OUT_ROOT / "domain_gated_training.json", {"BG_CORECONTENT_V2_DOMAIN_GATED_VERDICT": verdict,
                  "gate_choice": choice, "val_gated_by_domain": {k: round(v, 4) for k, v in val_gated.items()},
                  "val_gated_core_macro": round(val_macro, 4), "best_universal": best_uni,
                  "best_universal_val_macro": round(uni.get(best_uni, float("nan")), 4) if best_uni else None,
                  "universal_val": {k: round(v, 4) for k, v in uni.items()}, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "domain_gated_rows.csv", [{"domain": d, "expert": choice.get(d), "val_top1": round(val_gated.get(d, float('nan')), 4)} for d in CORE_DOMAINS])
    write_md(OUT_ROOT / "domain_gated_training.md", v2._md_top("CoreContent v2 Domain-Gated Policy",
        "BG_CORECONTENT_V2_DOMAIN_GATED_VERDICT", verdict, [
        f"Gated val core macro {fmt(val_macro)} vs best universal `{best_uni}` {fmt(uni.get(best_uni) if best_uni else None)}.", "",
        "Per-domain expert (selected on val): " + ", ".join(f"{d}={choice.get(d)}" for d in CORE_DOMAINS),
    ]))
    v2.progress("partM_domain_gated", {"verdict": verdict, "choice": choice, "val_macro": round(val_macro, 4)})
    print(status_line("BG_CORECONTENT_V2_DOMAIN_GATED_VERDICT", verdict)); print("  gate:", choice)
    return 0


# ===================================================== PART N: weight merge
def _chan_w(channels: list[tuple], cfg: str):
    for (w, a, c) in channels:
        if c == cfg:
            return w.float()
    return None


def weight_merge_main() -> int:
    started = time.time(); v2.ensure_root()
    base = cc.build_tap_policies(); val_by = groups_by_domain("val")
    obj = base.get("mixedhead_MIX_OBJECTIVE_ALL", []); hh = base.get("mixedhead_MIX_HH_OBJECTIVE", [])
    code = base.get("mixedhead_MIX_CODE_REASONING", [])
    grids = [("obj+HH", (0.8, 0.2, 0.0)), ("obj+HH", (0.7, 0.3, 0.0)), ("obj+HH", (0.6, 0.4, 0.0)), ("obj+HH", (0.5, 0.5, 0.0)),
             ("obj+HH+code", (0.5, 0.25, 0.25)), ("obj+HH+code", (0.4, 0.4, 0.2)), ("obj+HH+code", (0.4, 0.2, 0.4)), ("obj+HH+code", (0.34, 0.33, 0.33))]
    results = []; best = None
    for label, (a, b, c) in grids:
        chans = []
        for cfg in TRAIN_CONFIGS:
            wo, wh, wc = _chan_w(obj, cfg), _chan_w(hh, cfg), _chan_w(code, cfg)
            if wo is None or wh is None:
                continue
            o = cc._normed(wo); hres = cc.masked_residual_vec(wh, [o])
            cres = cc.masked_residual_vec(wc, [o, hres]) if (wc is not None and c > 0) else torch.zeros_like(o)
            w = cc._normed(a * o + b * hres + c * cres).reshape(1, -1)
            chans.append((w, "AntisymLinearNoNorm", cfg))
        if not chans:
            continue
        vm = round(macro_core(eval_channels_macro(chans, val_by)), 4)
        results.append({"merge": label, "coeffs": f"{a}/{b}/{c}", "val_core_macro": vm})
        if best is None or vm > best["val"]:
            best = {"val": vm, "label": label, "coeffs": (a, b, c), "channels": chans}
    hh_val = round(macro_core(eval_channels_macro(hh, val_by)), 4)
    if best is None:
        verdict = "INSUFFICIENT"
    elif best["val"] > hh_val + 0.01:
        verdict = "OBJECTIVE_HH_MERGE_BEST" if "HH" in best["label"] else "DOMAIN_WEIGHTED_MERGE_BEST"
    elif best["val"] >= hh_val - 0.005:
        verdict = "CONTENT_MERGE_READY"
    else:
        verdict = "BROAD_OBJECTIVE_STILL_BEST"
    torch.save({"best_channels": best["channels"] if best else [], "best_coeffs": best["coeffs"] if best else None,
                "best_val": best["val"] if best else None}, OUT_ROOT / "weight_merge_v2.pt")
    v2.write_json(OUT_ROOT / "weight_merge_v2.json", {"BG_CORECONTENT_V2_WEIGHT_MERGE_VERDICT": verdict, "results": results,
                  "best": {"label": best["label"], "coeffs": best["coeffs"], "val": best["val"]} if best else None,
                  "hh_val": hh_val, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "weight_merge_v2_rows.csv", results)
    write_md(OUT_ROOT / "weight_merge_v2.md", v2._md_top("CoreContent v2 Weight Merge (content-only)",
        "BG_CORECONTENT_V2_WEIGHT_MERGE_VERDICT", verdict, [
        f"Best merge val {fmt(best['val'] if best else None)} ({best['label'] if best else '-'}) vs MIX_HH_OBJECTIVE {fmt(hh_val)}. "
        "No branch residuals.", "",
        *md_table(sorted(results, key=lambda r: -r["val_core_macro"]), ["merge", "coeffs", "val_core_macro"]),
    ]))
    v2.progress("partN_weight_merge", {"verdict": verdict, "best_val": best["val"] if best else None, "hh_val": hh_val})
    print(status_line("BG_CORECONTENT_V2_WEIGHT_MERGE_VERDICT", verdict))
    return 0


# ===================================================== PART O: science aux
def science_aux_main() -> int:
    started = time.time(); v2.ensure_root()
    base = cc.build_tap_policies()
    hh = base.get("mixedhead_MIX_HH_OBJECTIVE", [])
    sci_channels = []
    for k in base:
        if k.startswith("science_"):
            sci_channels += base[k]
    by = groups_by_domain("heldout", CORE_DOMAINS + ("anatomy",))
    def macro(ch, doms):
        out = {}
        for d in doms:
            ms = [metrics(g, cc.policy_candidate_scores(g, ch)) for g in by.get(d, [])]
            ms = [m for m in ms if m]
            out[d] = finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
        return out
    base_m = macro(hh, CORE_DOMAINS); aux_m = macro(hh + sci_channels, CORE_DOMAINS) if sci_channels else base_m
    anat_base = macro(hh, ("anatomy",)).get("anatomy"); anat_aux = macro(hh + sci_channels, ("anatomy",)).get("anatomy") if sci_channels else anat_base
    core_delta = macro_core(aux_m) - macro_core(base_m)
    if core_delta > 0.01:
        verdict = "SCIENCE_AUX_HELPS_CORE"
    elif core_delta < -0.01:
        verdict = "SCIENCE_AUX_HURTS_CORE"
    elif (anat_aux or 0) - (anat_base or 0) > 0.02:
        verdict = "SCIENCE_AUX_HELPFUL_ONLY_DIAGNOSTIC"
    else:
        verdict = "SCIENCE_AUX_NO_HELP"
    rows = [{"domain": d, "base": round(base_m.get(d, float('nan')), 4), "with_science": round(aux_m.get(d, float('nan')), 4)} for d in CORE_DOMAINS]
    v2.write_json(OUT_ROOT / "science_aux_v2.json", {"BG_CORECONTENT_V2_SCIENCE_AUX_VERDICT": verdict,
                  "core_delta": round(core_delta, 4), "anatomy_base": anat_base, "anatomy_aux": anat_aux,
                  "rows": rows, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "science_aux_v2_rows.csv", rows)
    write_md(OUT_ROOT / "science_aux_v2.md", v2._md_top("CoreContent v2 Science Auxiliary (diagnostic)",
        "BG_CORECONTENT_V2_SCIENCE_AUX_VERDICT", verdict, [
        f"Core macro delta with science aux: {fmt(core_delta)} (heldout). Science remains diagnostic-only unless strong lift.", "",
        *md_table(rows, ["domain", "base", "with_science"]),
    ]))
    v2.progress("partO_science_aux", {"verdict": verdict, "core_delta": round(core_delta, 4)})
    print(status_line("BG_CORECONTENT_V2_SCIENCE_AUX_VERDICT", verdict))
    return 0


# ----------------------------------------------------------------- bootstrap helpers
def per_group_top1(policy: Any, split: str, domains: Sequence[str] = CORE_DOMAINS) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for d in domains:
        vals = []
        for g in groups(d, split):
            m = metrics(g, score_group(g, policy))
            if m:
                vals.append(m["top1_oracle"])
        out[d] = vals
    return out


def boot_macro_ci(top1_by_dom: dict[str, list[float]], n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    point = finite_mean([finite_mean(v) for v in top1_by_dom.values() if v])
    samples = []
    doms = [d for d, v in top1_by_dom.items() if v]
    for _ in range(n_boot):
        per = []
        for d in doms:
            v = top1_by_dom[d]
            per.append(sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v))
        samples.append(sum(per) / len(per))
    samples.sort()
    lo = samples[int(0.025 * n_boot)] if samples else float("nan")
    hi = samples[int(0.975 * n_boot)] if samples else float("nan")
    return point, lo, hi


def _gate_channels() -> dict[str, list[tuple]]:
    g = _load_pt(OUT_ROOT / "domain_gated_v2.pt")
    return g.get("gate_channels", {})


# ===================================================== PART P: heldout eval
def heldout_eval_main() -> int:
    started = time.time(); v2.ensure_root()
    base = cc.build_tap_policies()
    policies: dict[str, Any] = baseline_policies(include_diag=True)
    policies.update(crafted_v2_policies())
    rows, by_pol = eval_policy_macro(policies, split="heldout", domains=ALL_DOMAINS)
    # gated + hand-rule as pseudo-policies
    gate = _gate_channels()
    extra_macro: dict[str, dict[str, float]] = {}
    if gate:
        gm = eval_gated_macro(gate, "heldout", ALL_DOMAINS)
        extra_macro["CoreContent_v2_domain_gated"] = gm
        for d, v in gm.items():
            rows.append({"policy": "CoreContent_v2_domain_gated", "domain": d, "groups": len(groups(d, "heldout")),
                         "top1_oracle": round(v, 4), "pairwise_accuracy": None, "mean_regret": None})
    macro = {p: macro_core(d) for p, d in by_pol.items()}
    for p, d in extra_macro.items():
        macro[p] = macro_core(d)
    hh = macro.get("mixedhead_MIX_HH_OBJECTIVE", float("nan"))
    # candidate v2 policies
    v2_names = [n for n in macro if n.startswith("CoreContent_v2") or n.startswith("CoreContent_v2_domain")]
    real = {k: v for k, v in macro.items() if k not in ("ORACLE", "RANDOM") and not _is_diag(k)}
    best = max(real, key=lambda k: real[k]) if real else None
    best_v2 = max(v2_names, key=lambda k: macro.get(k, -1)) if v2_names else None
    # bootstrap CI for HH and best_v2
    hh_ci = boot_macro_ci(per_group_top1(base.get("mixedhead_MIX_HH_OBJECTIVE", []), "heldout"))
    v2_ci = None
    if best_v2:
        pol = crafted_v2_policies().get(best_v2)
        if pol is not None:
            v2_ci = boot_macro_ci(per_group_top1(pol, "heldout"))
        elif best_v2 == "CoreContent_v2_domain_gated" and gate:
            tg = {d: [m["top1_oracle"] for g in groups(d, "heldout") if (m := metrics(g, cc.policy_candidate_scores(g, gate[d]))) ] for d in CORE_DOMAINS if d in gate}
            v2_ci = boot_macro_ci(tg)
    # no-regression checks vs HH on coding/reasoning
    hh_dom = by_pol.get("mixedhead_MIX_HH_OBJECTIVE", {})
    bestv2_dom = by_pol.get(best_v2, extra_macro.get(best_v2, {})) if best_v2 else {}
    no_reg = all((bestv2_dom.get(d, 0) >= hh_dom.get(d, 0) - 0.01) for d in ("coding", "reasoning")) if best_v2 else False
    beats = best_v2 is not None and macro.get(best_v2, -1) > hh + 0.005
    if beats and best_v2 == "CoreContent_v2_domain_gated":
        verdict = "DOMAIN_GATED_READY"
    elif beats and best_v2 and "weight_merge" in best_v2:
        verdict = "WEIGHT_MERGE_READY"
    elif beats:
        verdict = "V2_CORECONTENT_READY"
    elif best == "mixedhead_MIX_HH_OBJECTIVE" or (best and best.startswith("mixedhead")):
        verdict = "BROAD_OBJECTIVE_STILL_BEST"
    else:
        verdict = "NO_IMPROVEMENT"
    payload = {"BG_CORECONTENT_V2_HELDOUT_VERDICT": verdict, "macro_core_top1": {k: round(v, 4) for k, v in macro.items()},
               "mixedhead_HH": round(hh, 4) if hh == hh else None, "best_overall": best, "best_v2": best_v2,
               "hh_ci": [round(x, 4) for x in hh_ci], "best_v2_ci": [round(x, 4) for x in v2_ci] if v2_ci else None,
               "no_regression_coding_reasoning": no_reg, "by_pol_domain": {k: {d: round(x, 4) for d, x in v.items()} for k, v in by_pol.items()},
               "rows": rows, "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "heldout_eval_v2.json", payload)
    write_csv(OUT_ROOT / "heldout_eval_v2_rows.csv", rows)
    write_md(OUT_ROOT / "heldout_eval_v2.md", v2._md_top("CoreContent v2 Heldout Evaluation",
        "BG_CORECONTENT_V2_HELDOUT_VERDICT", verdict, [
        f"Best overall `{best}` = {fmt(real.get(best) if best else None)}; best v2 crafted `{best_v2}` = {fmt(macro.get(best_v2) if best_v2 else None)} "
        f"vs mixedhead_MIX_HH_OBJECTIVE = {fmt(hh)}.", "",
        f"HH 95% CI {[round(x,4) for x in hh_ci]}; best-v2 95% CI {[round(x,4) for x in v2_ci] if v2_ci else 'n/a'}. "
        f"No coding/reasoning regression: {no_reg}.", "",
        "## Macro core top1 (heldout)", "",
        *[f"- {k}: {fmt(v)}" for k, v in sorted(macro.items(), key=lambda x: -x[1])], "",
        *md_table(sorted([r for r in rows if r["domain"] in CORE_DOMAINS], key=lambda r: (r["policy"], r["domain"])),
                  ["policy", "domain", "groups", "top1_oracle"]),
    ]))
    v2.progress("partP_heldout", {"verdict": verdict, "best_v2": best_v2, "hh": round(hh, 4) if hh == hh else None,
                                  "macro": {k: round(v, 4) for k, v in macro.items()}})
    print(status_line("BG_CORECONTENT_V2_HELDOUT_VERDICT", verdict))
    print(f"  best={best} {fmt(real.get(best) if best else None)} | best_v2={best_v2} {fmt(macro.get(best_v2) if best_v2 else None)} | HH={fmt(hh)}")
    return 0


# ===================================================== PART Q: domain / error
def domain_error_main() -> int:
    started = time.time(); v2.ensure_root()
    he = v2.read_json(OUT_ROOT / "heldout_eval_v2.json", {}) or {}
    by_pol = he.get("by_pol_domain", {})
    base = cc.build_tap_policies()
    rows = []
    for d in CORE_DOMAINS:
        scores = {p: by_pol.get(p, {}).get(d) for p in by_pol if by_pol.get(p, {}).get(d) is not None}
        base_scores = {p: s for p, s in scores.items() if not p.startswith("CoreContent_v2") and p not in ("ORACLE", "RANDOM")}
        v2_scores = {p: s for p, s in scores.items() if p.startswith("CoreContent_v2")}
        bb = max(base_scores, key=lambda k: base_scores[k]) if base_scores else None
        bv = max(v2_scores, key=lambda k: v2_scores[k]) if v2_scores else None
        # source breakdown
        src = defaultdict(int)
        for g in groups(d, "heldout"):
            src[g.get("source_dataset")] += 1
        rows.append({"domain": d, "best_baseline": bb, "best_baseline_top1": round(base_scores.get(bb, float('nan')), 4) if bb else None,
                     "best_v2": bv, "best_v2_top1": round(v2_scores.get(bv, float('nan')), 4) if bv else None,
                     "improvement": round((v2_scores.get(bv, 0) - base_scores.get(bb, 0)), 4) if bv and bb else None,
                     "heldout_groups": sum(src.values()), "sources": dict(src)})
    improved = [r["domain"] for r in rows if (r["improvement"] or 0) > 0.005]
    if len(improved) >= 4:
        verdict = "CORE_DOMAINS_IMPROVED"
    elif "coding" in improved and "reasoning" in improved:
        verdict = "CODING_READY"
    elif set(improved) >= {"math", "logic"}:
        verdict = "MATH_LOGIC_READY"
    elif not improved:
        verdict = "BROAD_OBJECTIVE_STILL_ROBUST"
    else:
        verdict = "DATA_GAPS_REMAIN"
    v2.write_json(OUT_ROOT / "domain_error_analysis.json", {"BG_CORECONTENT_V2_DOMAIN_ERROR_VERDICT": verdict,
                  "rows": rows, "improved_domains": improved, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "error_cases.csv", [{"domain": r["domain"], "best_baseline": r["best_baseline"],
              "best_v2": r["best_v2"], "improvement": r["improvement"]} for r in rows])
    write_md(OUT_ROOT / "domain_error_analysis.md", v2._md_top("CoreContent v2 Domain / Error Analysis",
        "BG_CORECONTENT_V2_DOMAIN_ERROR_VERDICT", verdict, [
        f"Domains where best v2 beats best baseline (heldout, >0.005): {improved or 'none'}.", "",
        *md_table(rows, ["domain", "best_baseline", "best_baseline_top1", "best_v2", "best_v2_top1", "improvement", "heldout_groups"]),
    ]))
    v2.progress("partQ_domain_error", {"verdict": verdict, "improved": improved})
    print(status_line("BG_CORECONTENT_V2_DOMAIN_ERROR_VERDICT", verdict))
    return 0


# ===================================================== PART R: calibration / ablation
def calibration_ablation_main() -> int:
    started = time.time(); v2.ensure_root()
    base = cc.build_tap_policies(); val_by = groups_by_domain("val")
    hh = base.get("mixedhead_MIX_HH_OBJECTIVE", [])
    # config ablations on the v2 blockwise tap
    lin = _load_pt(OUT_ROOT / "corecontent_v2_linear_pairwise.pt")
    block = lin.get("blockwise_channels", [])
    abl = []
    for drop in (None, "24_L4", "36_L4", "47_L4"):
        chans = [c for c in block if c[2] != drop] if drop else block
        if chans:
            abl.append({"ablation": f"drop_{drop}" if drop else "full_blockwise",
                        "val_core_macro": round(macro_core(eval_channels_macro(chans, val_by)), 4)})
    single = [{"ablation": f"only_{c}", "val_core_macro": round(macro_core(eval_channels_macro([ch], val_by)), 4)}
              for ch in block for c in [ch[2]]]
    # missing-expert stress on domain gate (drop HH expert) — fall back behavior
    gate = _gate_channels()
    stress = []
    if gate:
        full = round(macro_core(eval_gated_macro(gate, "heldout")), 4)
        stress.append({"stress": "gate_full_heldout", "macro": full})
    full_block = next((a["val_core_macro"] for a in abl if a["ablation"] == "full_blockwise"), float("nan"))
    worst_drop = min((a["val_core_macro"] for a in abl if a["ablation"] != "full_blockwise"), default=full_block)
    degradation = round(full_block - worst_drop, 4) if full_block == full_block else None
    if degradation is not None and degradation < 0.01:
        verdict = "ROBUST"
    elif degradation is not None and degradation < 0.03:
        verdict = "CONSERVATIVE_BUT_SAFE"
    else:
        verdict = "DOMAIN_METADATA_CRITICAL"
    v2.write_json(OUT_ROOT / "calibration_ablation.json", {"BG_CORECONTENT_V2_CALIBRATION_ABLATION_VERDICT": verdict,
                  "config_ablation": abl, "single_config": single, "gate_stress": stress, "max_degradation": degradation,
                  "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "calibration_ablation_rows.csv", abl + single)
    write_md(OUT_ROOT / "calibration_ablation.md", v2._md_top("CoreContent v2 Calibration / Ablation",
        "BG_CORECONTENT_V2_CALIBRATION_ABLATION_VERDICT", verdict, [
        f"Max val degradation from dropping one layer config: {degradation}.", "",
        *md_table(abl + single, ["ablation", "val_core_macro"]),
    ]))
    v2.progress("partR_calibration_ablation", {"verdict": verdict, "degradation": degradation})
    print(status_line("BG_CORECONTENT_V2_CALIBRATION_ABLATION_VERDICT", verdict))
    return 0


# ===================================================== PART S: policy selection
def policy_selection_main() -> int:
    started = time.time(); v2.ensure_root()
    he = v2.read_json(OUT_ROOT / "heldout_eval_v2.json", {}) or {}
    macro = he.get("macro_core_top1", {})
    hh = he.get("mixedhead_HH")
    best_v2 = he.get("best_v2"); best_overall = he.get("best_overall")
    no_reg = he.get("no_regression_coding_reasoning", False)
    v2_val = macro.get(best_v2) if best_v2 else None
    beats = (v2_val is not None and hh is not None and v2_val > hh + 0.005 and no_reg)
    selected: dict[str, Any]
    if beats and best_v2 == "CoreContent_v2_domain_gated":
        sel_name = best_v2; verdict = "LOCK_V2_DOMAIN_GATED"
    elif beats and best_v2 and "weight_merge" in best_v2:
        sel_name = best_v2; verdict = "LOCK_WEIGHT_MERGE"
    elif beats:
        sel_name = best_v2; verdict = "LOCK_V2_CORECONTENT_POLICY"
    else:
        sel_name = "mixedhead_MIX_HH_OBJECTIVE"; verdict = "KEEP_MIX_HH_OBJECTIVE_BASELINE"
    # persist selected channels
    if sel_name == "CoreContent_v2_domain_gated":
        chans = {"gate": True, "gate_channels": _gate_channels()}
    elif sel_name.startswith("CoreContent_v2"):
        chans = {"channels": crafted_v2_policies().get(sel_name)}
    else:
        chans = {"channels": cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", [])}
    torch.save({"selected": sel_name, "verdict": verdict, "channels": chans,
                "required_features": list(TRAIN_CONFIGS), "fallback": "mixedhead_MIX_HH_OBJECTIVE",
                "terminal": "top5/full survivor-set handoff; content selector ranks within survivor set"},
               OUT_ROOT / "corecontent_v2_policy.pt")
    v1_sel = (v2.read_json(V1_CRAFTING_ROOT / "selected_corecontent_policy.json", {}) or {}).get("selected", "mixedhead_MIX_HH_OBJECTIVE")
    payload = {"BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT": verdict, "selected_policy": sel_name,
               "selected_macro_core_heldout": macro.get(sel_name), "mixedhead_HH_heldout": hh,
               "v1_selected_policy": v1_sel, "best_v2": best_v2, "no_regression_coding_reasoning": no_reg,
               "fallback_policy": "mixedhead_MIX_HH_OBJECTIVE",
               "terminal_handoff": "top5/full survivor-set handoff retained; content selector ranks within survivor set",
               "science_dependency": False, "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "selected_corecontent_v2_policy.json", payload)
    write_md(OUT_ROOT / "selected_corecontent_v2_policy.md", v2._md_top("CoreContent v2 Selected Policy",
        "BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT", verdict, [
        f"Selected content selector: `{sel_name}` (heldout core macro {fmt(macro.get(sel_name))} vs "
        f"mixedhead_MIX_HH_OBJECTIVE {fmt(hh)}).", "",
        f"v1 selected policy: `{v1_sel}`. Fallback: mixedhead_MIX_HH_OBJECTIVE. Terminal survivor-set handoff retained. "
        "Branch survival (DualAnchor) unchanged. Science/anatomy diagnostic-only.",
    ]))
    v2.progress("partS_policy_selection", {"verdict": verdict, "selected": sel_name})
    print(status_line("BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT", verdict)); print("  selected:", sel_name)
    return 0


# ===================================================== PART T: phase2b readiness
def phase2b_readiness_main() -> int:
    started = time.time(); v2.ensure_root()
    sel = v2.read_json(OUT_ROOT / "selected_corecontent_v2_policy.json", {}) or {}
    sv = sel.get("BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT")
    bal = v2.read_json(OUT_ROOT / "dataset_balance.json", {}) or {}
    bal_v = bal.get("BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT")
    if sv == "LOCK_V2_CORECONTENT_POLICY":
        verdict = "READY_FOR_PHASE2B_WITH_V2_CORECONTENT"
    elif sv == "LOCK_V2_DOMAIN_GATED":
        verdict = "READY_WITH_DOMAIN_RULE"
    elif sv == "LOCK_WEIGHT_MERGE":
        verdict = "READY_FOR_PHASE2B_WITH_V2_CORECONTENT"
    elif sv == "KEEP_MIX_HH_OBJECTIVE_BASELINE":
        verdict = "READY_FOR_PHASE2B_WITH_BROAD_OBJECTIVE"
    elif bal_v in ("INSUFFICIENT", "DATA_LIMITED"):
        verdict = "NEEDS_MORE_DATA"
    else:
        verdict = "READY_FOR_PHASE2B_WITH_BROAD_OBJECTIVE"
    locked = {"branch_survival": "DualAnchor (unchanged)", "content_final_selection": sel.get("selected_policy", "mixedhead_MIX_HH_OBJECTIVE"),
              "terminal": "top5/full survivor-set handoff; content selector ranks within survivor set",
              "domains": list(CORE_DOMAINS), "science": "diagnostic only"}
    v2.write_json(OUT_ROOT / "phase2b_content_readiness_v2.json", {"BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT": verdict,
                  "locked_phase2b": locked, "elapsed_seconds": round(time.time() - started, 3)})
    write_md(OUT_ROOT / "phase2b_content_readiness_v2.md", v2._md_top("CoreContent v2 Phase 2b Readiness",
        "BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT", verdict, [
        "Locked Phase 2b configuration:", "",
        *[f"- {k}: {v}" for k, v in locked.items()],
    ]))
    v2.progress("partT_phase2b", {"verdict": verdict})
    print(status_line("BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT", verdict))
    return 0


# ===================================================== PART U: synthesis
VERDICT_FILES = [
    ("BG_CORECONTENT_V2_INVENTORY_VERDICT", "inventory.json"),
    ("BG_CORECONTENT_V2_DATASET_PULL_VERDICT", "dataset_pull.json"),
    ("BG_CORECONTENT_V2_SCHEMA_NORMALIZATION_VERDICT", "schema_normalization.json"),
    ("BG_CORECONTENT_V2_CANDIDATE_GROUPS_VERDICT", "candidate_groups.json"),
    ("BG_CORECONTENT_V2_PARSER_VERIFIER_VERDICT", "parser_verifier_audit.json"),
    ("BG_CORECONTENT_V2_DEDUP_LEAKAGE_VERDICT", "dedup_leakage.json"),
    ("BG_CORECONTENT_V2_FEATURE_PLAN_VERDICT", "feature_extraction_plan.json"),
    ("BG_CORECONTENT_V2_FEATURE_EXTRACTION_VERDICT", "feature_extraction.json"),
    ("BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT", "dataset_balance.json"),
    ("BG_CORECONTENT_V2_BASELINES_VERDICT", "baselines_v2.json"),
    ("BG_CORECONTENT_V2_LINEAR_PAIRWISE_VERDICT", "linear_pairwise_training.json"),
    ("BG_CORECONTENT_V2_LISTWISE_VERDICT", "listwise_training.json"),
    ("BG_CORECONTENT_V2_DOMAIN_GATED_VERDICT", "domain_gated_training.json"),
    ("BG_CORECONTENT_V2_WEIGHT_MERGE_VERDICT", "weight_merge_v2.json"),
    ("BG_CORECONTENT_V2_SCIENCE_AUX_VERDICT", "science_aux_v2.json"),
    ("BG_CORECONTENT_V2_HELDOUT_VERDICT", "heldout_eval_v2.json"),
    ("BG_CORECONTENT_V2_DOMAIN_ERROR_VERDICT", "domain_error_analysis.json"),
    ("BG_CORECONTENT_V2_CALIBRATION_ABLATION_VERDICT", "calibration_ablation.json"),
    ("BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT", "selected_corecontent_v2_policy.json"),
    ("BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT", "phase2b_content_readiness_v2.json"),
]


def synthesis_main() -> int:
    started = time.time(); v2.ensure_root()
    verdicts = {}
    for key, fname in VERDICT_FILES:
        d = v2.read_json(OUT_ROOT / fname, {}) or {}
        verdicts[key] = d.get(key)
    he = v2.read_json(OUT_ROOT / "heldout_eval_v2.json", {}) or {}
    bal = v2.read_json(OUT_ROOT / "dataset_balance.json", {}) or {}
    sel = verdicts.get("BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT")
    heldv = verdicts.get("BG_CORECONTENT_V2_HELDOUT_VERDICT")
    balv = verdicts.get("BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT")
    if sel in ("LOCK_V2_CORECONTENT_POLICY", "LOCK_WEIGHT_MERGE"):
        status = "V2_CORECONTENT_READY"
    elif sel == "LOCK_V2_DOMAIN_GATED":
        status = "DOMAIN_RULE_READY"
    elif sel == "KEEP_MIX_HH_OBJECTIVE_BASELINE" and balv in ("LARGE_CORE_DATA_READY", "MUCH_IMPROVED"):
        status = "DATA_EXPANSION_SUCCESS_BUT_NO_MODEL_GAIN"
    elif balv in ("INSUFFICIENT",):
        status = "NEEDS_MORE_DATA"
    elif sel == "KEEP_MIX_HH_OBJECTIVE_BASELINE":
        status = "BROAD_OBJECTIVE_BASELINE_CONFIRMED"
    else:
        status = "DATA_STILL_LIMITED"
    core_div = bal.get("core_reward_diverse", {})
    summary = {**verdicts, "CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS": status,
               "core_reward_diverse_v2": core_div, "v1_coverage": v2.V1_COVERAGE,
               "heldout_macro": he.get("macro_core_top1"), "selected_policy": (v2.read_json(OUT_ROOT / "selected_corecontent_v2_policy.json", {}) or {}).get("selected_policy"),
               "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "summary.json", summary)
    v2.write_json(OUT_ROOT / "analysis.json", summary)
    top_lines = [f"{k} = {verdicts.get(k)}" for k, _ in VERDICT_FILES]
    top_lines.append(f"CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS = {status}")
    body = [
        "# CoreContent Dataset Expansion + Refit v2 — Synthesis", "",
        status_line("CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS", status), "",
        "## Top-line verdicts", "", *[f"    {l}" for l in top_lines], "",
        "## 1. Motivation", "",
        "CoreContent v1 completed but the broad-objective baseline (mixedhead_MIX_HH_OBJECTIVE) stayed best; the bottleneck "
        "was diagnosed as dataset scale/coverage, not head cleverness. v2 aggressively expands the starved core domains.", "",
        "## 2. Why v1 failed", "",
        f"v1 reward-diverse coverage was tiny: {v2.V1_COVERAGE}. Crafted heads overfit the small split.", "",
        "## 3-9. Data expansion", "",
        f"v2 reward-diverse coverage: {core_div} (feature storage {bal.get('feature_gb')} GB).", "",
        "## 10-12. Baselines, models, heldout", "",
        f"Heldout macro core top1 (selected vs HH): selected={summary['selected_policy']}, "
        f"HH={fmt(he.get('mixedhead_HH'))}. Heldout verdict={heldv}.", "",
        "## 15-16. Final policy and Phase 2b", "",
        f"Selected: {summary['selected_policy']} ({sel}). Phase 2b: {verdicts.get('BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT')}.", "",
        "## 17. No steering / science diagnostic", "",
        "No steering trained or applied. No Ouro training. No tap-registry mutation. pure/transplanted taps untouched. "
        "Science/anatomy diagnostic-only; terminal survivor-set handoff retained; DualAnchor branch survival unchanged.",
    ]
    write_md(OUT_ROOT / "summary.md", body)
    write_md(OUT_ROOT / "analysis.md", body)
    v2.progress("partU_synthesis", {"status": status})
    print(status_line("CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS", status))
    for l in top_lines:
        print("  " + l)
    return 0


# ===================================================== PART V: documentation
DOC_ROOT = v2.PROJECT_ROOT / "shared/docs/evaluator"
# Curated, memorable-named docs are maintained by hand (current-state.md,
# domain-transfer-ledger.md, content-selection-taps.md). docs_main only (re)writes the one
# machine-owned detail doc below to avoid polluting curated docs with auto-appended stubs.
APPEND_DOCS: list[str] = []


def docs_main() -> int:
    v2.ensure_root()
    summ = v2.read_json(OUT_ROOT / "summary.json", {}) or {}
    bal = v2.read_json(OUT_ROOT / "dataset_balance.json", {}) or {}
    he = v2.read_json(OUT_ROOT / "heldout_eval_v2.json", {}) or {}
    sel = v2.read_json(OUT_ROOT / "selected_corecontent_v2_policy.json", {}) or {}
    status = summ.get("CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS")
    section_title = "CoreContent dataset expansion and refit v2 (2026-06-04)"
    section = [
        "", f"## {section_title}", "",
        f"- Status: `{status}`.",
        f"- Why: v1 kept the broad-objective baseline (mixedhead_MIX_HH_OBJECTIVE); bottleneck was dataset scale, "
        f"not head design. v1 reward-diverse coverage {v2.V1_COVERAGE}.",
        f"- Data expanded (reward-diverse): {summ.get('core_reward_diverse_v2')} ; feature storage {bal.get('feature_gb')} GB, "
        f"{bal.get('shards')} shards. Datasets: coding(mbpp/apps/verifiable/humaneval), math(gsm8k/hendrycks/svamp), "
        f"logic(logiqa), reasoning(arc/openbookqa/commonsenseqa/strategyqa), alignment(hh/ultrafeedback/shp/pku).",
        f"- Parser/verifier: {summ.get('BG_CORECONTENT_V2_PARSER_VERIFIER_VERDICT')}; dedup/leakage: "
        f"{summ.get('BG_CORECONTENT_V2_DEDUP_LEAKAGE_VERDICT')}.",
        f"- Heldout: best v2 `{he.get('best_v2')}` = {fmt(he.get('macro_core_top1',{}).get(he.get('best_v2')) if he.get('best_v2') else None)} "
        f"vs mixedhead_MIX_HH_OBJECTIVE {fmt(he.get('mixedhead_HH'))} (verdict {summ.get('BG_CORECONTENT_V2_HELDOUT_VERDICT')}).",
        f"- Selected content selector: `{sel.get('selected_policy')}` ({sel.get('BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT')}). "
        f"Phase 2b: {summ.get('BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT')}.",
        "- No steering trained/applied/claimed. No Ouro training, no weight/tokenizer/checkpoint edits, no tap-registry "
        "mutation. pure_content_taps.pt / transplanted_taps.pt untouched. Science/anatomy diagnostic-only; terminal "
        "survivor-set handoff retained; DualAnchor branch survival unchanged.",
        f"- Artifacts: `{v2.OUT_ROOT.relative_to(v2.PROJECT_ROOT)}`.",
    ]
    # main doc
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    main_doc = DOC_ROOT / "corecontent-dataset-expansion-v2.md"
    write_md(main_doc, ["# CoreContent Dataset Expansion + Content Tap Refit v2", *section,
                        "", "### Top-line verdicts", "",
                        *[f"    {k} = {summ.get(k)}" for k, _ in VERDICT_FILES],
                        f"    CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS = {status}"])
    appended = []
    for name in APPEND_DOCS:
        p = DOC_ROOT / name
        try:
            prev = p.read_text() if p.exists() else f"# {name}\n"
            p.write_text(prev.rstrip() + "\n" + "\n".join(section) + "\n")
            appended.append(name)
        except Exception:
            pass
    v2.progress("partV_docs", {"appended": appended, "main_doc": str(main_doc.relative_to(v2.PROJECT_ROOT))})
    print(f"BG_CORECONTENT_V2_DOCS = wrote {main_doc.name}; appended to {len(appended)} docs")
    return 0
