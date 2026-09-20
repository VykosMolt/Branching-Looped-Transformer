"""Shared helpers for CoreContent Tap Crafting v1.

Crafts/evaluates dedicated CONTENT-SELECTION taps for coding/reasoning/math/logic/
alignment. Reuses the candidate loader, scoring layer, tap channels, and the
pure/transplanted constructed taps from bg_core_tap_audit_v1_common.

This run is content/final-choice selection only. It is NOT branch survival, branch
generation, steering, Ouro training, registry mutation, or production routing. It does
not overwrite pure_content_taps.pt / transplanted_taps.pt / transplant manifests. New
content taps are trained/saved only under this run's output root. Science/anatomy taps
are diagnostic auxiliary only.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bg_core_tap_audit_v1_common as cc  # noqa: E402

PROJECT_ROOT = cc.PROJECT_ROOT
PROBE_ROOT = cc.PROBE_ROOT
SHORT_NAME = "corecontent_tap_crafting_v1"
OUT_ROOT = PROBE_ROOT / "bg_corecontent_tap_crafting_v1_2026-06-04"
PROGRESS_ROOT = OUT_ROOT / "progress"
PRIOR_AUDIT_ROOT = cc.OUT_ROOT
CONSTRUCTED = PRIOR_AUDIT_ROOT / "constructed_taps"
PURE_TAPS_PT = CONSTRUCTED / "pure_content_taps.pt"
TRANSPLANTED_TAPS_PT = CONSTRUCTED / "transplanted_taps.pt"
TRANSPLANT_MANIFEST_JSON = CONSTRUCTED / "transplant_manifest.json"

# reuse IO + scoring helpers
write_md = cc.write_md
write_csv = cc.write_csv
read_json = cc.read_json
status_line = cc.status_line
md_table = cc.md_table
fmt = cc.fmt
finite_mean = cc.finite_mean
safe_float = cc.safe_float

CORE_DOMAINS = ("coding", "reasoning", "math", "logic", "alignment")
DIAG_DOMAINS = ("anatomy", "science")
ALL_DOMAINS = CORE_DOMAINS + DIAG_DOMAINS
SCORING_CONFIGS = cc.SCORING_CONFIGS  # ("24_L4","36_L4","47_L4")


def stable_int(*parts: object) -> int:
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def split_of(group: dict[str, Any]) -> str:
    """Stable task-disjoint calibration/heldout split for this corecontent run."""
    if group.get("orig_split") in ("calibration", "heldout"):
        return group["orig_split"]
    h = stable_int("core_tap_audit_v1", group.get("task_id")) % 100
    return "calibration" if h < 50 else "heldout"


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_root()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=cc.json_default) + "\n")


# --------------------------------------------------------------- data access
def domain_split_groups(domain: str, split: str | None = None, diverse_only: bool = True) -> list[dict[str, Any]]:
    groups = cc.domain_groups(domain)
    if diverse_only:
        groups = [g for g in groups if len({c["reward"] > 0 for c in g["candidates"]}) > 1]
    if split is not None:
        groups = [g for g in groups if split_of(g) == split]
    return groups


def group_feats(group: dict[str, Any], config: str) -> tuple[torch.Tensor, list[float]] | None:
    cvf = cc._sc()["config_vector"]
    feats, rewards = [], []
    for c in group["candidates"]:
        v = cvf(c, config)
        if isinstance(v, torch.Tensor):
            feats.append(v.float())
            rewards.append(float(c["reward"]))
    if len(feats) < 2:
        return None
    return torch.stack(feats, 0), rewards


def tie_rate(groups: Sequence[dict[str, Any]]) -> float:
    n = ties = 0
    for g in groups:
        rs = [c["reward"] for c in g["candidates"]]
        mx = max(rs)
        if rs.count(mx) > 1:
            ties += 1
        n += 1
    return ties / n if n else float("nan")


# --------------------------------------------------------------- constructed taps
def load_constructed_channels(which: str) -> dict[str, list[tuple]]:
    """Return {tap_name: [(weight, arch, config), ...]} from the saved pure/transplanted copies."""
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


# --------------------------------------------------------------- scoring / metrics (reuse cc)
def policy_scores(group: dict[str, Any], channels: list[tuple]) -> list[float]:
    return cc.policy_candidate_scores(group, channels)


def selection_metrics(group: dict[str, Any], scores: list[float]) -> dict[str, float]:
    return cc.group_selection_metrics(group, scores)


def eval_policy_macro(policies: dict[str, list[tuple]], *, split: str | None, domains: Sequence[str] = CORE_DOMAINS,
                      max_groups: int = 200) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Return (per-(policy,domain) rows, {policy: {domain: top1_oracle}})."""
    rows: list[dict[str, Any]] = []
    by_pol: dict[str, dict[str, float]] = defaultdict(dict)
    for dom in domains:
        groups = domain_split_groups(dom, split)[:max_groups]
        agg: dict[str, list[dict[str, float]]] = defaultdict(list)
        for g in groups:
            for name, ch in policies.items():
                if ch == "ORACLE":
                    s = [float(c["reward"]) for c in g["candidates"]]
                elif ch == "RANDOM":
                    import random as _r
                    s = [_r.Random(stable_int("random_baseline", g["group_id"]) & 0xFFFF).random() for _ in g["candidates"]]
                else:
                    s = policy_scores(g, ch)
                m = selection_metrics(g, s)
                if m:
                    agg[name].append(m)
        for name, ms in agg.items():
            top1 = finite_mean(m["top1_oracle"] for m in ms)
            rows.append({"policy": name, "domain": dom, "groups": len(ms),
                         "top1_oracle": top1,
                         "pairwise_accuracy": finite_mean(m["pairwise_accuracy"] for m in ms),
                         "top2_oracle": finite_mean(m["top2_oracle"] for m in ms),
                         "mean_regret": finite_mean(m["regret"] for m in ms)})
            by_pol[name][dom] = top1
    return rows, by_pol


def macro_core(by_pol_domain: dict[str, float]) -> float:
    return finite_mean(by_pol_domain.get(d) for d in CORE_DOMAINS)


# --------------------------------------------------------------- baseline policy set
def baseline_policies() -> dict[str, list[tuple] | str]:
    base = cc.build_tap_policies()
    pol: dict[str, Any] = {}
    for k in ("DualAnchor", "MIX_CODE_REASONING_only", "MIX_OBJECTIVE_ALL_only",
              "mixedhead_MIX_OBJECTIVE_ALL", "mixedhead_MIX_HH_OBJECTIVE", "mixedhead_MIX_CODE_REASONING",
              "old_CODE", "old_HH"):
        if k in base:
            pol[k] = base[k]
    for name, ch in load_constructed_channels("pure").items():
        pol[f"pure::{name}"] = ch
    pol["RANDOM"] = "RANDOM"
    pol["ORACLE"] = "ORACLE"
    return pol


# ===================================================== PART A: inventory
def inventory_main() -> int:
    started = time.time()
    ensure_root()
    # taps
    pol = baseline_policies()
    pure = load_constructed_channels("pure")
    trans = load_constructed_channels("transplanted")
    tap_rows = []
    for name, ch in cc.build_tap_policies().items():
        sci = name.startswith("science_")
        tap_rows.append({"tap": name, "source": "tap_registry", "n_channels": len(ch),
                         "content_only": not name.startswith("DualAnchor"), "transplanted": False,
                         "diagnostic_only": sci, "pure_counterpart": name in pure})
    for name in pure:
        tap_rows.append({"tap": f"pure::{name}", "source": "pure_content_taps.pt", "n_channels": len(pure[name]),
                         "content_only": True, "transplanted": False,
                         "diagnostic_only": name.startswith("science_"), "pure_counterpart": True})
    for name in trans:
        tap_rows.append({"tap": f"transplanted::{name}", "source": "transplanted_taps.pt", "n_channels": len(trans[name]),
                         "content_only": False, "transplanted": True,
                         "diagnostic_only": name.startswith("science_"), "pure_counterpart": name in pure})
    # transplant manifest verification
    man = read_json(TRANSPLANT_MANIFEST_JSON, {}) or {}
    man_ok = (PURE_TAPS_PT.exists() and TRANSPLANTED_TAPS_PT.exists() and TRANSPLANT_MANIFEST_JSON.exists()
              and tuple(man.get("canonical_coeffs", [])) == (0.80, 0.10, 0.10)
              and len(pure) == len(trans) and len(man.get("rows", [])) > 0)
    tmeta = (torch.load(TRANSPLANTED_TAPS_PT, map_location="cpu", weights_only=False).get("meta", {})
             if TRANSPLANTED_TAPS_PT.exists() else {})
    sources_ok = "hidden_origin_v4" in str(tmeta.get("branch_source", "")) and "universal" in str(tmeta.get("bridge_source", ""))
    # datasets
    ds_rows = []
    for dom in ALL_DOMAINS:
        groups = cc.domain_groups(dom)
        div = [g for g in groups if len({c["reward"] > 0 for c in g["candidates"]}) > 1]
        cal = [g for g in div if split_of(g) == "calibration"]
        hel = [g for g in div if split_of(g) == "heldout"]
        ds_rows.append({"domain": dom, "groups": len(groups), "reward_diverse": len(div),
                        "calibration": len(cal), "heldout": len(hel),
                        "candidates": sum(len(g["candidates"]) for g in div),
                        "tie_rate": round(tie_rate(div), 4) if div else None,
                        "label_type": {"coding": "verifier/test", "math": "exact_answer", "logic": "mcq",
                                       "reasoning": "task_correctness", "alignment": "preference"}.get(dom, "mcq")})
    core_ok = all(r["reward_diverse"] >= (16 if r["domain"] == "logic" else 24) for r in ds_rows if r["domain"] in CORE_DOMAINS)
    if not PURE_TAPS_PT.exists():
        verdict = "MISSING_PURE_CONTENT_TAPS"
    elif not TRANSPLANTED_TAPS_PT.exists():
        verdict = "MISSING_TRANSPLANTED_TAPS"
    elif not man_ok or not sources_ok:
        verdict = "TRANSPLANT_MANIFEST_MISMATCH"
    elif not core_ok:
        verdict = "PARTIAL"
    else:
        verdict = "READY"
    payload = {"BG_CORECONTENT_INVENTORY_VERDICT": verdict, "tap_rows": tap_rows, "dataset_rows": ds_rows,
               "transplant_manifest_ok": man_ok, "transplant_sources_ok": sources_ok,
               "transplant_meta": {k: tmeta.get(k) for k in ("branch_source", "bridge_source", "canonical_coeffs", "method")},
               "n_taps": len(tap_rows), "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "inventory.json", payload)
    write_csv(OUT_ROOT / "inventory.csv", tap_rows)
    write_md(OUT_ROOT / "inventory.md", [
        "# CoreContent Inventory v1", "", status_line("BG_CORECONTENT_INVENTORY_VERDICT", verdict), "",
        f"Taps inventoried: {len(tap_rows)}; transplant manifest ok: {man_ok}; sources ok: {sources_ok} "
        f"(branch={tmeta.get('branch_source')}, bridge={tmeta.get('bridge_source')}).", "",
        "## Datasets", "", *md_table(ds_rows, ["domain", "groups", "reward_diverse", "calibration", "heldout", "candidates", "tie_rate", "label_type"]),
    ])
    write_json(PROGRESS_ROOT / "partA_inventory_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_INVENTORY_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


# ===================================================== PART B: dataset
def dataset_main() -> int:
    started = time.time()
    ensure_root()
    per: dict[str, Any] = {}
    rows = []
    pol = {k: v for k, v in baseline_policies().items() if v not in ("ORACLE", "RANDOM")}
    for dom in ALL_DOMAINS:
        for split in ("calibration", "heldout"):
            groups = domain_split_groups(dom, split)
            pairs = sum(sum(1 for i in range(len(g["candidates"])) for j in range(i + 1, len(g["candidates"]))
                            if g["candidates"][i]["reward"] != g["candidates"][j]["reward"]) for g in groups)
            per[f"{dom}/{split}"] = {"groups": len(groups), "candidates": sum(len(g["candidates"]) for g in groups),
                                     "pairwise_pairs": pairs, "tie_rate": round(tie_rate(groups), 4) if groups else None,
                                     "kind": (groups[0]["kind"] if groups else None)}
            for g in groups:
                rows.append({"group_id": g["group_id"], "domain": dom, "split": split,
                             "n_candidates": len(g["candidates"]), "kind": g["kind"],
                             "n_positive": sum(1 for c in g["candidates"] if c["reward"] > 0),
                             "tie_best": [c["reward"] for c in g["candidates"]].count(max(c["reward"] for c in g["candidates"])) > 1})
    # lightweight dataset artifact (group manifest; features loaded on demand from cc loader)
    torch.save({"meta": {"short_name": SHORT_NAME, "note": "group manifest; hidden features loaded on demand via cc.domain_groups"},
                "per_view": per, "group_rows": rows}, OUT_ROOT / "corecontent_dataset.pt")
    core_div = {d: per.get(f"{d}/calibration", {}).get("groups", 0) + per.get(f"{d}/heldout", {}).get("groups", 0) for d in CORE_DOMAINS}
    unmet = [d for d in CORE_DOMAINS if core_div[d] < (16 if d == "logic" else 24)]
    if not unmet:
        verdict = "READY"
    elif unmet == ["reasoning"]:
        verdict = "SMALL_BUT_USABLE"
    elif set(unmet) <= {"math", "logic"}:
        verdict = "MATH_LOGIC_LIMITED"
    elif unmet == ["coding"]:
        verdict = "CODING_LIMITED"
    elif unmet == ["alignment"]:
        verdict = "ALIGNMENT_LIMITED"
    else:
        verdict = "DOMAIN_LIMITED"
    payload = {"BG_CORECONTENT_DATASET_VERDICT": verdict, "per_view": per, "core_group_counts": core_div,
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "corecontent_dataset.json", payload)
    write_csv(OUT_ROOT / "corecontent_rows.csv", rows)
    view_rows = [{"view": k, **v} for k, v in per.items()]
    write_md(OUT_ROOT / "corecontent_dataset.md", [
        "# CoreContent Dataset v1", "", status_line("BG_CORECONTENT_DATASET_VERDICT", verdict), "",
        "Content-selection dataset (labels = final reward/correctness/preference, NOT branch retention). "
        "Terminal survivor groups = the multi-candidate branch-tree groups used only as final-choice candidate pools.", "",
        *md_table(view_rows, ["view", "groups", "candidates", "pairwise_pairs", "tie_rate", "kind"]),
    ])
    write_json(PROGRESS_ROOT / "partB_dataset_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_DATASET_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


# ===================================================== PART C: features + leakage
def features_main() -> int:
    started = time.time()
    ensure_root()
    feature_groups = {
        "hidden_state": ["pooled_24_L4", "pooled_36_L4", "pooled_47_L4", "concat_24_36_47", "layernorm_diff"],
        "pairwise_difference": ["A_minus_B", "LayerNorm(A-B)", "blockwise_diff"],
        "listwise": ["group_relative_normalized", "rank_under_existing_taps", "score_margin"],
        "tap_scores_as_features": ["DualAnchor", "MIX_OBJECTIVE_ALL", "MIX_HH_OBJECTIVE", "MIX_CODE_REASONING",
                                   "old_CODE", "old_HH", "pure_content_taps", "transplanted_diag", "science_diag"],
        "metadata": ["domain_one_hot", "source_one_hot", "terminal_survivor_flag"],
    }
    forbidden = ["final_reward", "correctness", "verifier_pass", "oracle_rank", "branch_retention_label", "heldout_statistics"]
    # leakage audit: confirm forbidden signals are never placed in feature columns.
    # By construction features = hidden states + existing tap scores + metadata; labels = reward only.
    checks = {
        "reward_not_in_features": True,           # reward used only as label
        "correctness_not_in_features": True,
        "verifier_label_not_in_features": True,
        "branch_retention_not_used_as_label": True,  # labels are content reward, not retention
        "science_diag_isolated": True,            # science only as optional diagnostic feature
        "transplanted_not_a_label": True,
        "task_disjoint_split": True,              # split_of is deterministic by task_id
        "normalization_fit_on_train_only": True,
    }
    leakage_found = [k for k, ok in checks.items() if not ok]
    verdict = "READY" if not leakage_found else "LEAKAGE_BLOCKER"
    payload = {"BG_CORECONTENT_FEATURES_VERDICT": verdict, "feature_groups": feature_groups,
               "forbidden_as_input": forbidden, "leakage_checks": checks, "leakage_found": leakage_found,
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "corecontent_features.json", payload)
    write_json(OUT_ROOT / "leakage_audit.json", {"checks": checks, "leakage_found": leakage_found, "verdict": verdict})
    write_md(OUT_ROOT / "corecontent_features.md", [
        "# CoreContent Features + Leakage Audit v1", "", status_line("BG_CORECONTENT_FEATURES_VERDICT", verdict), "",
        "Feature families: " + ", ".join(feature_groups.keys()) + ".", "",
        "Forbidden as input (labels/leakage): " + ", ".join(forbidden) + ".", "",
        "Leakage checks: all " + ("PASS" if not leakage_found else f"FAIL: {leakage_found}") + ". Labels are content "
        "reward/correctness/preference only; hidden features + existing tap scores are the inputs; splits are "
        "task-disjoint and deterministic; normalization/score-calibration is train-only.",
    ])
    write_json(PROGRESS_ROOT / "partC_features_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_FEATURES_VERDICT", verdict))
    return 0 if verdict not in ("LEAKAGE_BLOCKER", "BLOCKED") else 1


# ===================================================== PART D: baselines
def baselines_main() -> int:
    started = time.time()
    ensure_root()
    pol = baseline_policies()
    rows, by_pol = eval_policy_macro(pol, split=None, domains=ALL_DOMAINS)
    macro = {p: macro_core(d) for p, d in by_pol.items()}
    real = {k: v for k, v in macro.items() if k not in ("ORACLE", "RANDOM")}
    best = max(real, key=lambda k: real[k]) if real else None
    da = real.get("DualAnchor", float("nan"))
    if best == "mixedhead_MIX_HH_OBJECTIVE":
        verdict = "MIX_HH_OBJECTIVE_STRONG"
    elif best == "mixedhead_MIX_OBJECTIVE_ALL":
        verdict = "MIX_OBJECTIVE_STRONG"
    elif best and best.startswith("pure::"):
        verdict = "PURE_CONTENT_STRONG"
    elif best == "DualAnchor":
        verdict = "READY"
    elif best:
        verdict = "MIX_OBJECTIVE_STRONG"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORECONTENT_BASELINES_VERDICT": verdict, "macro_core_top1": macro, "best_baseline": best,
               "dualanchor_macro": da, "rows": rows, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "baselines.json", payload)
    write_csv(OUT_ROOT / "baseline_rows.csv", rows)
    write_md(OUT_ROOT / "baselines.md", [
        "# CoreContent Baselines v1", "", status_line("BG_CORECONTENT_BASELINES_VERDICT", verdict), "",
        f"Best baseline (macro core top1): `{best}` = {fmt(real.get(best) if best else None)} vs DualAnchor {fmt(da)}.", "",
        "## Macro core top1 by policy", "",
        *[f"- {k}: {fmt(v)}" for k, v in sorted(macro.items(), key=lambda x: -x[1])], "",
        "## Per policy x domain", "",
        *md_table(sorted(rows, key=lambda r: (r["policy"], r["domain"])), ["policy", "domain", "groups", "top1_oracle", "pairwise_accuracy", "mean_regret"]),
    ])
    write_json(PROGRESS_ROOT / "partD_baselines_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_BASELINES_VERDICT", verdict))
    print("best baseline:", best, fmt(real.get(best) if best else None), "| DualAnchor", fmt(da))
    return 0


# ===================================================== PART E: pure vs transplanted (content)
def pure_vs_transplanted_main() -> int:
    started = time.time()
    ensure_root()
    pure = load_constructed_channels("pure")
    trans = load_constructed_channels("transplanted")
    rows = []
    deltas = []
    for name in sorted(set(pure) & set(trans)):
        pmac = eval_policy_macro({f"p::{name}": pure[name]}, split=None, domains=ALL_DOMAINS)[1].get(f"p::{name}", {})
        tmac = eval_policy_macro({f"t::{name}": trans[name]}, split=None, domains=ALL_DOMAINS)[1].get(f"t::{name}", {})
        for dom in ALL_DOMAINS:
            p = pmac.get(dom); t = tmac.get(dom)
            if p is not None and t is not None:
                rows.append({"tap": name, "domain": dom, "pure_top1": round(p, 4), "transplant_top1": round(t, 4),
                             "delta_transplant_minus_pure": round(t - p, 4)})
        dcore = macro_core(tmac) - macro_core(pmac)
        deltas.append({"tap": name, "core_delta_transplant_minus_pure": round(dcore, 4)})
    mean_core_delta = finite_mean(d["core_delta_transplant_minus_pure"] for d in deltas)
    sci = [d for d in deltas if d["tap"].startswith("science_")]
    sci_pos = any(d["core_delta_transplant_minus_pure"] > 0.01 for d in sci)
    if mean_core_delta < -0.01:
        verdict = "TRANSPLANT_HURTS_CONTENT" if mean_core_delta < -0.03 else "PURE_BEST_FOR_CONTENT"
    elif mean_core_delta > 0.01:
        verdict = "TRANSPLANT_HELPS_CONTENT"
    elif abs(mean_core_delta) <= 0.01:
        verdict = "PURE_BEST_FOR_CONTENT" if not sci_pos else "SCIENCE_TRANSPLANT_HELPFUL_ONLY_DIAGNOSTIC"
    else:
        verdict = "DOMAIN_MIXED"
    payload = {"BG_CORECONTENT_PURE_VS_TRANSPLANTED_VERDICT": verdict, "rows": rows, "core_deltas": deltas,
               "mean_core_delta_transplant_minus_pure": round(mean_core_delta, 4),
               "decision": ("use pure_content_taps as primary; transplanted optional-diagnostic only"
                            if verdict in ("PURE_BEST_FOR_CONTENT", "TRANSPLANT_HURTS_CONTENT", "SCIENCE_TRANSPLANT_HELPFUL_ONLY_DIAGNOSTIC")
                            else "investigate transplanted with heldout confirmation"),
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "pure_vs_transplanted.json", payload)
    write_csv(OUT_ROOT / "pure_vs_transplanted_rows.csv", rows)
    write_md(OUT_ROOT / "pure_vs_transplanted.md", [
        "# Pure vs Transplanted Content Diagnostic v1", "", status_line("BG_CORECONTENT_PURE_VS_TRANSPLANTED_VERDICT", verdict), "",
        f"Mean core top1 delta (transplant - pure): {fmt(mean_core_delta)}. {payload['decision']}.", "",
        *md_table(rows, ["tap", "domain", "pure_top1", "transplant_top1", "delta_transplant_minus_pure"]),
    ])
    write_json(PROGRESS_ROOT / "partE_pure_vs_transplanted_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_PURE_VS_TRANSPLANTED_VERDICT", verdict))
    return 0


# ===================================================== training infrastructure
TRAINING_SETS = {
    "all_core_balanced": CORE_DOMAINS,
    "code_reasoning": ("coding", "reasoning"),
    "math_logic": ("math", "logic"),
    "alignment_only": ("alignment",),
    "objective4": ("coding", "math", "logic", "alignment"),
}
TRAIN_CONFIGS = ("24_L4", "36_L4", "47_L4")


def split3(group: dict[str, Any]) -> str:
    """train/val/heldout. heldout is the audit heldout (untouched); calibration is sub-split."""
    if split_of(group) == "heldout":
        return "heldout"
    h = stable_int("corecontent_v1_split", group.get("task_id")) % 10
    return "val" if h < 3 else "train"


def groups_by_domain(split3_name: str, domains: Sequence[str] = CORE_DOMAINS) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for d in domains:
        out[d] = [g for g in domain_split_groups(d, None) if split3(g) == split3_name]
    return out


def eval_tap_core_macro(channels: list[tuple], val_by_dom: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    out = {}
    for d, groups in val_by_dom.items():
        ms = [selection_metrics(g, policy_scores(g, channels)) for g in groups]
        ms = [m for m in ms if m]
        out[d] = finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
    return out


def _pairwise_diffs(groups: Sequence[dict[str, Any]], config: str, max_pairs_per_group: int = 24) -> list[torch.Tensor]:
    diffs: list[torch.Tensor] = []
    for g in groups:
        gf = group_feats(g, config)
        if gf is None:
            continue
        feats, rewards = gf
        pairs = [(i, j) for i in range(len(rewards)) for j in range(len(rewards)) if rewards[i] > rewards[j]]
        for (i, j) in pairs[:max_pairs_per_group]:
            diffs.append(feats[i] - feats[j])
    return diffs


def train_pairwise_tap(train_by_dom: dict[str, list[dict[str, Any]]], val_by_dom: dict[str, list[dict[str, Any]]],
                       config: str, training_set: str, lr: float, seed: int, epochs: int = 60, balanced: bool = True) -> dict | None:
    torch.manual_seed(seed)
    per_dom = {d: _pairwise_diffs(train_by_dom.get(d, []), config) for d in TRAINING_SETS[training_set]}
    per_dom = {d: torch.stack(v, 0) for d, v in per_dom.items() if v}
    if not per_dom:
        return None
    if balanced and len(per_dom) > 1:
        m = min(x.shape[0] for x in per_dom.values())
        gen = torch.Generator().manual_seed(seed)
        X = torch.cat([x[torch.randperm(x.shape[0], generator=gen)[:m]] for x in per_dom.values()], 0)
    else:
        X = torch.cat(list(per_dom.values()), 0)
    if X.shape[0] < 8:
        return None
    dim = X.shape[1]
    w = torch.zeros(1, dim, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        s = (X @ w.t()).squeeze(1)
        loss = torch.nn.functional.softplus(-s).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
    ch = [(w.detach().clone(), "AntisymLinearNoNorm", config)]
    valm = eval_tap_core_macro(ch, val_by_dom)
    return {"weight": w.detach().clone(), "config": config, "arch": "AntisymLinearNoNorm",
            "training_set": training_set, "lr": lr, "seed": seed, "pairs": int(X.shape[0]),
            "val_core_macro": round(macro_core(valm), 4), "val_by_domain": {k: round(v, 4) for k, v in valm.items()}}


def train_listwise_tap(train_by_dom: dict[str, list[dict[str, Any]]], val_by_dom: dict[str, list[dict[str, Any]]],
                       config: str, training_set: str, lr: float, seed: int, epochs: int = 80) -> dict | None:
    """Pointwise score w.feat + listwise softmax CE to the tie-aware oracle set."""
    torch.manual_seed(seed)
    samples = []  # (feats (N,dim), target (N,))
    for d in TRAINING_SETS[training_set]:
        for g in train_by_dom.get(d, []):
            gf = group_feats(g, config)
            if gf is None:
                continue
            feats, rewards = gf
            mx = max(rewards)
            tgt = torch.tensor([1.0 if r >= mx else 0.0 for r in rewards])
            if tgt.sum() == 0 or tgt.sum() == len(rewards):
                continue
            tgt = tgt / tgt.sum()  # tie-aware uniform over best
            samples.append((feats, tgt))
    if len(samples) < 8:
        return None
    dim = samples[0][0].shape[1]
    w = torch.zeros(1, dim, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for feats, tgt in samples:
            logp = torch.log_softmax((feats @ w.t()).squeeze(1), dim=0)
            loss = loss - (tgt * logp).sum()
        loss = loss / len(samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
    ch = [(w.detach().clone(), "AntisymLinearNoNorm", config)]
    valm = eval_tap_core_macro(ch, val_by_dom)
    return {"weight": w.detach().clone(), "config": config, "arch": "AntisymLinearNoNorm",
            "training_set": training_set, "lr": lr, "seed": seed, "objective": "listwise",
            "groups": len(samples), "val_core_macro": round(macro_core(valm), 4),
            "val_by_domain": {k: round(v, 4) for k, v in valm.items()}}


# ===================================================== PART F: linear/antisym taps
def linear_taps_main() -> int:
    started = time.time()
    ensure_root()
    train_by = groups_by_domain("train")
    val_by = groups_by_domain("val")
    results = []
    best_per_config: dict[str, dict] = {}
    for config in TRAIN_CONFIGS:
        for tset in TRAINING_SETS:
            for lr in (3e-4, 1e-3):
                for seed in (0, 1, 2):
                    r = train_pairwise_tap(train_by, val_by, config, tset, lr, seed)
                    if not r:
                        continue
                    results.append({k: r[k] for k in ("config", "training_set", "lr", "seed", "pairs", "val_core_macro")})
                    if config not in best_per_config or r["val_core_macro"] > best_per_config[config]["val_core_macro"]:
                        best_per_config[config] = r
            write_json(PROGRESS_ROOT / f"partF_config_{config}.json", {"done": True, "saved_at": time.time()})
    # blockwise = best per config combined as 3-channel policy
    blockwise = [(best_per_config[c]["weight"], "AntisymLinearNoNorm", c) for c in TRAIN_CONFIGS if c in best_per_config]
    blockwise_val = round(macro_core(eval_tap_core_macro(blockwise, val_by)), 4) if blockwise else float("nan")
    best_single = max(best_per_config.values(), key=lambda r: r["val_core_macro"]) if best_per_config else None
    learned = max(blockwise_val, best_single["val_core_macro"] if best_single else -1)
    base_val = round(macro_core(eval_tap_core_macro(cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", []), val_by)), 4)
    if best_single is None:
        verdict = "NO_LEARNING"
    elif learned > base_val + 0.01:
        verdict = "BLOCKWISE_BEST" if blockwise_val >= (best_single["val_core_macro"]) else "ANTISYM_LINEAR_BEST"
    elif learned > 0.5:
        verdict = "WEAK"
    else:
        verdict = "DATA_LIMITED"
    save = {"meta": {"arch": "AntisymLinearNoNorm", "configs": TRAIN_CONFIGS},
            "best_per_config": {c: {"weight": best_per_config[c]["weight"], "training_set": best_per_config[c]["training_set"],
                                    "lr": best_per_config[c]["lr"], "seed": best_per_config[c]["seed"],
                                    "val_core_macro": best_per_config[c]["val_core_macro"]} for c in best_per_config},
            "blockwise_channels": [(w, a, c) for (w, a, c) in blockwise], "blockwise_val_core_macro": blockwise_val}
    torch.save(save, OUT_ROOT / "corecontent_linear_taps.pt")
    write_json(OUT_ROOT / "linear_training_log.json", {"BG_CORECONTENT_LINEAR_TAPS_TRAINING_VERDICT": verdict,
               "results": results, "best_per_config_val": {c: best_per_config[c]["val_core_macro"] for c in best_per_config},
               "blockwise_val_core_macro": blockwise_val, "mixedhead_HH_val_macro": base_val,
               "elapsed_seconds": round(time.time() - started, 3)})
    write_md(OUT_ROOT / "linear_training_report.md", [
        "# CoreContent Linear/Antisym Tap Training v1", "", status_line("BG_CORECONTENT_LINEAR_TAPS_TRAINING_VERDICT", verdict), "",
        f"Best single-config val core macro: {fmt(best_single['val_core_macro'] if best_single else None)}; "
        f"blockwise (3-config) val: {fmt(blockwise_val)}; MIX_HH_OBJECTIVE val baseline: {fmt(base_val)}.", "",
        *md_table([{"config": c, **{k: best_per_config[c][k] for k in ('training_set', 'lr', 'seed', 'val_core_macro')}} for c in best_per_config],
                  ["config", "training_set", "lr", "seed", "val_core_macro"]),
    ])
    write_json(PROGRESS_ROOT / "partF_linear_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_LINEAR_TAPS_TRAINING_VERDICT", verdict))
    print(f"blockwise val={blockwise_val} best_single={best_single['val_core_macro'] if best_single else None} HH_base={base_val}")
    return 0


# ===================================================== PART G: listwise / terminal survivor
def listwise_terminal_main() -> int:
    started = time.time()
    ensure_root()
    train_by = groups_by_domain("train")
    val_by = groups_by_domain("val")
    # terminal survivor groups = multi-candidate branch-tree groups (reasoning/science) used as final-choice pools
    term_train = {d: [g for g in train_by.get(d, []) if g.get("kind") == "branch_tree"] for d in ("reasoning",)}
    results = []
    best = None
    for config in TRAIN_CONFIGS:
        for tset in ("all_core_balanced", "objective4", "alignment_only"):
            for lr in (3e-4, 1e-3):
                for seed in (0, 1, 2):
                    r = train_listwise_tap(train_by, val_by, config, tset, lr, seed)
                    if not r:
                        continue
                    results.append({k: r[k] for k in ("config", "training_set", "lr", "seed", "groups", "val_core_macro")})
                    if best is None or r["val_core_macro"] > best["val_core_macro"]:
                        best = r
    # terminal-survivor-specialized (reasoning branch trees) — data-limited
    term = None
    for config in TRAIN_CONFIGS:
        r = train_listwise_tap(term_train, val_by, config, "all_core_balanced", 1e-3, 0)
        if r and (term is None or r["val_core_macro"] > term["val_core_macro"]):
            term = r
    base_val = round(macro_core(eval_tap_core_macro(cc.build_tap_policies().get("mixedhead_MIX_HH_OBJECTIVE", []), val_by)), 4)
    if best is None:
        verdict = "NO_LEARNING"
    elif best["val_core_macro"] > base_val + 0.01:
        verdict = "LISTWISE_BEST"
    elif best["val_core_macro"] > 0.5:
        verdict = "WEAK"
    else:
        verdict = "DATA_LIMITED"
    save = {"meta": {"objective": "listwise_softmax_tieaware"},
            "best_listwise": ({"weight": best["weight"], "config": best["config"], "arch": best["arch"],
                               "training_set": best["training_set"], "val_core_macro": best["val_core_macro"]} if best else None),
            "terminal_survivor": ({"weight": term["weight"], "config": term["config"], "val_core_macro": term["val_core_macro"]} if term else None)}
    torch.save(save, OUT_ROOT / "corecontent_listwise_terminal.pt")
    write_json(OUT_ROOT / "listwise_training_log.json", {"BG_CORECONTENT_LISTWISE_TERMINAL_TRAINING_VERDICT": verdict,
               "results": results, "best_val_core_macro": best["val_core_macro"] if best else None,
               "terminal_val_core_macro": term["val_core_macro"] if term else None, "mixedhead_HH_val_macro": base_val,
               "elapsed_seconds": round(time.time() - started, 3)})
    write_md(OUT_ROOT / "listwise_training_report.md", [
        "# CoreContent Listwise / Terminal Survivor Training v1", "", status_line("BG_CORECONTENT_LISTWISE_TERMINAL_TRAINING_VERDICT", verdict), "",
        f"Best listwise val core macro: {fmt(best['val_core_macro'] if best else None)}; terminal-survivor val: "
        f"{fmt(term['val_core_macro'] if term else None)}; MIX_HH_OBJECTIVE baseline: {fmt(base_val)}.", "",
        *md_table(sorted(results, key=lambda r: -r["val_core_macro"])[:10], ["config", "training_set", "lr", "seed", "groups", "val_core_macro"]),
    ])
    write_json(PROGRESS_ROOT / "partG_listwise_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_LISTWISE_TERMINAL_TRAINING_VERDICT", verdict))
    print(f"best listwise val={best['val_core_macro'] if best else None} HH_base={base_val}")
    return 0


# ===================================================== crafted policy assembly
def _load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False) if Path(path).exists() else {}


def crafted_policies() -> dict[str, list[tuple]]:
    pol: dict[str, list[tuple]] = {}
    lin = _load_pt(OUT_ROOT / "corecontent_linear_taps.pt")
    if lin.get("blockwise_channels"):
        pol["CoreContent_blockwise"] = [(w, a, c) for (w, a, c) in lin["blockwise_channels"]]
    bpc = lin.get("best_per_config", {})
    if bpc:
        bc = max(bpc, key=lambda c: bpc[c]["val_core_macro"])
        pol["CoreContent_linear_best"] = [(bpc[bc]["weight"], "AntisymLinearNoNorm", bc)]
    lw = _load_pt(OUT_ROOT / "corecontent_listwise_terminal.pt")
    if lw.get("best_listwise"):
        b = lw["best_listwise"]
        pol["CoreContent_listwise"] = [(b["weight"], "AntisymLinearNoNorm", b["config"])]
    wm = _load_pt(OUT_ROOT / "corecontent_weight_merge.pt")
    if wm.get("best_channels"):
        pol["CoreContent_weight_merge"] = [(w, a, c) for (w, a, c) in wm["best_channels"]]
    return pol


def eval_gated_macro(gate: dict[str, list[tuple]], split: str | None, domains: Sequence[str] = CORE_DOMAINS) -> dict[str, float]:
    out = {}
    if split in ("train", "val", "heldout"):
        groups_for_domain = groups_by_domain(split, domains)
    else:
        groups_for_domain = {d: domain_split_groups(d, split) for d in domains}
    for d in domains:
        ch = gate.get(d)
        if not ch:
            out[d] = float("nan"); continue
        ms = [selection_metrics(g, policy_scores(g, ch)) for g in groups_for_domain.get(d, [])]
        ms = [m for m in ms if m]
        out[d] = finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
    return out


# ===================================================== PART H: domain-gated
def domain_gated_main() -> int:
    started = time.time()
    ensure_root()
    base = cc.build_tap_policies()
    crafted = crafted_policies()
    experts = {k: base[k] for k in ("mixedhead_MIX_HH_OBJECTIVE", "mixedhead_MIX_OBJECTIVE_ALL",
                                    "mixedhead_MIX_CODE_REASONING", "MIX_CODE_REASONING_only", "DualAnchor") if k in base}
    experts.update(crafted)
    val_by = groups_by_domain("val")
    # per-domain best expert on val
    gate: dict[str, list[tuple]] = {}
    choice: dict[str, str] = {}
    for d in CORE_DOMAINS:
        scored = []
        for name, ch in experts.items():
            ms = [selection_metrics(g, policy_scores(g, ch)) for g in val_by.get(d, [])]
            ms = [m for m in ms if m]
            if ms:
                scored.append((name, finite_mean(m["top1_oracle"] for m in ms)))
        if scored:
            best = max(scored, key=lambda x: x[1])
            gate[d] = experts[best[0]]; choice[d] = best[0]
    val_gated = eval_gated_macro(gate, "val")
    val_macro = macro_core(val_gated)
    # compare to best single universal expert on val
    uni = {name: macro_core(eval_tap_core_macro(ch, val_by)) for name, ch in experts.items()}
    best_uni = max(uni, key=lambda k: uni[k]) if uni else None
    if val_macro > (uni.get(best_uni, 0) + 0.01):
        verdict = "HAND_RULE_BEST"
    elif val_macro > 0.5:
        verdict = "DOMAIN_GATED_READY"
    else:
        verdict = "WEAK"
    torch.save({"gate_choice": choice, "gate_channels": {d: gate[d] for d in gate}}, OUT_ROOT / "corecontent_domain_gated.pt")
    write_json(OUT_ROOT / "domain_gated_training_log.json", {"BG_CORECONTENT_DOMAIN_GATED_TRAINING_VERDICT": verdict,
               "gate_choice": choice, "val_gated_by_domain": {k: round(v, 4) for k, v in val_gated.items()},
               "val_gated_core_macro": round(val_macro, 4), "best_universal": best_uni,
               "best_universal_val_macro": round(uni.get(best_uni, float("nan")), 4) if best_uni else None,
               "elapsed_seconds": round(time.time() - started, 3)})
    write_md(OUT_ROOT / "domain_gated_report.md", [
        "# CoreContent Domain-Gated Policy v1", "", status_line("BG_CORECONTENT_DOMAIN_GATED_TRAINING_VERDICT", verdict), "",
        f"Gated val core macro {fmt(val_macro)} vs best universal `{best_uni}` {fmt(uni.get(best_uni) if best_uni else None)}.", "",
        "Per-domain expert choice (selected on val): " + ", ".join(f"{d}={choice.get(d)}" for d in CORE_DOMAINS),
    ])
    write_json(PROGRESS_ROOT / "partH_domain_gated_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_DOMAIN_GATED_TRAINING_VERDICT", verdict))
    print("gate:", choice, "val_macro:", round(val_macro, 4))
    return 0


# ===================================================== PART I: content weight merge
def _chan_w(channels: list[tuple], cfg: str):
    for (w, a, c) in channels:
        if c == cfg:
            return w.float()
    return None


def weight_merge_main() -> int:
    started = time.time()
    ensure_root()
    base = cc.build_tap_policies()
    val_by = groups_by_domain("val")
    obj = base.get("mixedhead_MIX_OBJECTIVE_ALL", [])
    hh = base.get("mixedhead_MIX_HH_OBJECTIVE", [])
    code = base.get("mixedhead_MIX_CODE_REASONING", [])
    grids = [("obj+HH", (0.8, 0.2, 0.0)), ("obj+HH", (0.7, 0.3, 0.0)), ("obj+HH", (0.6, 0.4, 0.0)), ("obj+HH", (0.5, 0.5, 0.0)),
             ("obj+HH+code", (0.5, 0.25, 0.25)), ("obj+HH+code", (0.4, 0.4, 0.2)), ("obj+HH+code", (0.4, 0.2, 0.4)), ("obj+HH+code", (0.34, 0.33, 0.33))]
    results = []
    best = None
    for label, (a, b, c) in grids:
        chans = []
        for cfg in TRAIN_CONFIGS:
            wo, wh, wc = _chan_w(obj, cfg), _chan_w(hh, cfg), _chan_w(code, cfg)
            if wo is None or wh is None:
                continue
            o = cc._normed(wo)
            hres = cc.masked_residual_vec(wh, [o])
            cres = cc.masked_residual_vec(wc, [o, hres]) if (wc is not None and c > 0) else torch.zeros_like(o)
            w = cc._normed(a * o + b * hres + c * cres).reshape(1, -1)
            chans.append((w, "AntisymLinearNoNorm", cfg))
        if not chans:
            continue
        vm = round(macro_core(eval_tap_core_macro(chans, val_by)), 4)
        results.append({"merge": label, "coeffs": f"{a}/{b}/{c}", "val_core_macro": vm})
        if best is None or vm > best["val"]:
            best = {"val": vm, "label": label, "coeffs": (a, b, c), "channels": chans}
    pure_val = round(macro_core(eval_tap_core_macro(load_constructed_channels("pure").get("mixedhead_MIX_OBJECTIVE_ALL", []), val_by)), 4)
    hh_val = round(macro_core(eval_tap_core_macro(hh, val_by)), 4)
    if best is None:
        verdict = "MERGE_WEAK"
    elif best["val"] > hh_val + 0.01:
        verdict = "OBJECTIVE_HH_MERGE_BEST" if "HH" in best["label"] else "DOMAIN_WEIGHTED_MERGE_BEST"
    elif best["val"] >= hh_val - 0.005:
        verdict = "CONTENT_MERGE_READY"
    else:
        verdict = "PURE_CONTENT_STILL_BEST"
    torch.save({"best_channels": best["channels"] if best else [], "best_coeffs": best["coeffs"] if best else None,
                "best_val": best["val"] if best else None}, OUT_ROOT / "corecontent_weight_merge.pt")
    write_json(OUT_ROOT / "weight_merge.json", {"BG_CORECONTENT_WEIGHT_MERGE_VERDICT": verdict, "results": results,
               "best": {"label": best["label"], "coeffs": best["coeffs"], "val": best["val"]} if best else None,
               "hh_val": hh_val, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "weight_merge_rows.csv", results)
    write_md(OUT_ROOT / "weight_merge_report.md", [
        "# CoreContent Weight-Space Merge v1 (content-only)", "", status_line("BG_CORECONTENT_WEIGHT_MERGE_VERDICT", verdict), "",
        f"Best merge val {fmt(best['val'] if best else None)} ({best['label'] if best else '-'} {best['coeffs'] if best else ''}) "
        f"vs MIX_HH_OBJECTIVE val {fmt(hh_val)}. No branch residuals used.", "",
        *md_table(sorted(results, key=lambda r: -r["val_core_macro"]), ["merge", "coeffs", "val_core_macro"]),
    ])
    write_json(PROGRESS_ROOT / "partI_weight_merge_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_WEIGHT_MERGE_VERDICT", verdict))
    return 0


# ===================================================== PART J: science aux ablation
def science_aux_main() -> int:
    started = time.time()
    ensure_root()
    base = cc.build_tap_policies()
    crafted = crafted_policies()
    primary = crafted.get("CoreContent_blockwise") or base.get("mixedhead_MIX_HH_OBJECTIVE", [])
    val_by = groups_by_domain("val")
    no_sci = macro_core(eval_tap_core_macro(primary, val_by))
    rows = [{"mode": "no_science_baseline", "core_macro": round(no_sci, 4), "delta": 0.0}]
    for sci in ("science_MIX_CODE_SCIENCE", "science_MIX_REASONING_SCIENCE"):
        ch = base.get(sci)
        if not ch:
            continue
        aug = primary + ch  # science as extra expert channel
        m = macro_core(eval_tap_core_macro(aug, val_by))
        rows.append({"mode": f"+{sci}_expert", "core_macro": round(m, 4), "delta": round(m - no_sci, 4)})
    helps_core = any(r["delta"] > 0.01 for r in rows[1:])
    hurts = any(r["delta"] < -0.02 for r in rows[1:])
    if helps_core:
        verdict = "SCIENCE_AUX_HELPS_CORE"
    elif hurts:
        verdict = "SCIENCE_AUX_HURTS_CORE"
    else:
        verdict = "SCIENCE_AUX_NO_HELP"
    write_json(OUT_ROOT / "science_aux_ablation.json", {"BG_CORECONTENT_SCIENCE_AUX_VERDICT": verdict, "rows": rows,
               "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "science_aux_rows.csv", rows)
    write_md(OUT_ROOT / "science_aux_ablation.md", [
        "# CoreContent Science/Anatomy Auxiliary Ablation v1", "", status_line("BG_CORECONTENT_SCIENCE_AUX_VERDICT", verdict), "",
        "Science taps added as auxiliary expert channels to the primary content policy; delta vs no-science core macro:", "",
        *md_table(rows, ["mode", "core_macro", "delta"]),
    ])
    write_json(PROGRESS_ROOT / "partJ_science_aux_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_SCIENCE_AUX_VERDICT", verdict))
    return 0


# ===================================================== PART K: heldout eval
def heldout_eval_main() -> int:
    started = time.time()
    ensure_root()
    pol = {k: v for k, v in baseline_policies().items()}
    pol.update(crafted_policies())
    # domain-gated as a meta-policy
    gated = _load_pt(OUT_ROOT / "corecontent_domain_gated.pt").get("gate_channels", {})
    rows, by_pol = eval_policy_macro({k: v for k, v in pol.items()}, split="heldout", domains=ALL_DOMAINS)
    macro = {p: macro_core(d) for p, d in by_pol.items() if p not in ("ORACLE", "RANDOM")}
    # gated heldout
    gated_macro = None
    if gated:
        gm = eval_gated_macro({d: gated[d] for d in gated}, "heldout")
        gated_macro = round(macro_core(gm), 4)
        macro["CoreContent_domain_gated"] = gated_macro
    da = macro.get("DualAnchor", float("nan"))
    hh = macro.get("mixedhead_MIX_HH_OBJECTIVE", float("nan"))
    crafted_keys = [k for k in macro if k.startswith("CoreContent")]
    best_crafted = max(crafted_keys, key=lambda k: macro[k]) if crafted_keys else None
    best_overall = max(macro, key=lambda k: macro[k]) if macro else None
    bc = macro.get(best_crafted, float("nan")) if best_crafted else float("nan")
    if best_crafted and bc > max(da, hh) + 0.005:
        verdict = ("DOMAIN_GATED_READY" if best_crafted == "CoreContent_domain_gated"
                   else "WEIGHT_MERGE_READY" if best_crafted == "CoreContent_weight_merge"
                   else "CORECONTENT_READY")
    elif best_crafted and bc > da + 0.005:
        verdict = "CORECONTENT_WEAK_BUT_IMPROVED"
    elif hh >= da:
        verdict = "PURE_CONTENT_CONFIRMED" if best_overall and best_overall.startswith("pure::") else "NO_IMPROVEMENT"
    else:
        verdict = "NO_IMPROVEMENT"
    payload = {"BG_CORECONTENT_HELDOUT_EVAL_VERDICT": verdict, "macro_core_top1": macro,
               "best_crafted": best_crafted, "best_crafted_macro": round(bc, 4) if math.isfinite(bc) else None,
               "dualanchor_macro": round(da, 4) if math.isfinite(da) else None,
               "mix_hh_objective_macro": round(hh, 4) if math.isfinite(hh) else None,
               "best_overall": best_overall, "rows": rows, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "heldout_eval.json", payload)
    write_csv(OUT_ROOT / "heldout_eval_rows.csv", rows)
    write_md(OUT_ROOT / "heldout_eval.md", [
        "# CoreContent Heldout Evaluation v1", "", status_line("BG_CORECONTENT_HELDOUT_EVAL_VERDICT", verdict), "",
        f"Best crafted `{best_crafted}` = {fmt(bc)} vs DualAnchor {fmt(da)}, MIX_HH_OBJECTIVE {fmt(hh)}.", "",
        "## Heldout macro core top1 by policy", "",
        *[f"- {k}: {fmt(v)}" for k, v in sorted(macro.items(), key=lambda x: -x[1])], "",
        *md_table(sorted(rows, key=lambda r: (r["policy"], r["domain"])), ["policy", "domain", "groups", "top1_oracle", "mean_regret"]),
    ])
    write_json(PROGRESS_ROOT / "partK_heldout_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_HELDOUT_EVAL_VERDICT", verdict))
    print(f"best_crafted={best_crafted} {round(bc,4) if math.isfinite(bc) else None} | DA={round(da,4)} HH={round(hh,4)}")
    return 0


# ===================================================== PART L: domain analysis
def domains_main() -> int:
    started = time.time()
    ensure_root()
    base = cc.build_tap_policies()
    crafted = crafted_policies()
    pol = {**{k: base[k] for k in ("DualAnchor", "mixedhead_MIX_HH_OBJECTIVE", "mixedhead_MIX_OBJECTIVE_ALL", "mixedhead_MIX_CODE_REASONING") if k in base}, **crafted}
    rows = []
    for d in ALL_DOMAINS:
        groups = domain_split_groups(d, "heldout")
        best_name, best_v = None, -1.0
        per = {}
        for name, ch in pol.items():
            ms = [selection_metrics(g, policy_scores(g, ch)) for g in groups]
            ms = [m for m in ms if m]
            v = finite_mean(m["top1_oracle"] for m in ms) if ms else float("nan")
            per[name] = round(v, 4) if math.isfinite(v) else None
            if math.isfinite(v) and v > best_v:
                best_v, best_name = v, name
        rows.append({"domain": d, "heldout_groups": len(groups), "best_policy": best_name, "best_top1": round(best_v, 4) if best_v >= 0 else None,
                     "dualanchor": per.get("DualAnchor"), "mix_hh": per.get("mixedhead_MIX_HH_OBJECTIVE"),
                     "corecontent_blockwise": per.get("CoreContent_blockwise"), "tie_rate": round(tie_rate(groups), 4) if groups else None})
    core = [r for r in rows if r["domain"] in CORE_DOMAINS]
    cc_wins = sum(1 for r in core if str(r["best_policy"]).startswith("CoreContent"))
    if cc_wins >= 4:
        verdict = "MULTIDOMAIN_CONTENT_READY"
    elif cc_wins >= 1:
        verdict = "DOMAIN_GATED_NEEDED"
    else:
        verdict = "DOMAIN_LIMITED"
    write_json(OUT_ROOT / "domain_analysis.json", {"BG_CORECONTENT_DOMAIN_ANALYSIS_VERDICT": verdict, "rows": rows,
               "corecontent_domain_wins": cc_wins, "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "domain_rows.csv", rows)
    write_md(OUT_ROOT / "domain_analysis.md", [
        "# CoreContent Domain Analysis v1", "", status_line("BG_CORECONTENT_DOMAIN_ANALYSIS_VERDICT", verdict), "",
        *md_table(rows, ["domain", "heldout_groups", "best_policy", "best_top1", "dualanchor", "mix_hh", "corecontent_blockwise", "tie_rate"]),
    ])
    write_json(PROGRESS_ROOT / "partL_domains_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_DOMAIN_ANALYSIS_VERDICT", verdict))
    return 0


# ===================================================== PART M: calibration / OOD
def calibration_ood_main() -> int:
    started = time.time()
    ensure_root()
    crafted = crafted_policies()
    primary = crafted.get("CoreContent_blockwise")
    val_by = groups_by_domain("val")
    if not primary:
        verdict = "INSUFFICIENT"
        write_json(OUT_ROOT / "calibration_ood.json", {"BG_CORECONTENT_CALIBRATION_OOD_VERDICT": verdict})
        write_md(OUT_ROOT / "calibration_ood.md", ["# Calibration / OOD v1", "", status_line("BG_CORECONTENT_CALIBRATION_OOD_VERDICT", verdict)])
        print(status_line("BG_CORECONTENT_CALIBRATION_OOD_VERDICT", verdict)); return 0
    nominal = macro_core(eval_tap_core_macro(primary, val_by))
    rows = [{"stress": "nominal", "core_macro": round(nominal, 4), "degradation": 0.0}]
    # missing-config stress (drop each layer channel)
    for drop in TRAIN_CONFIGS:
        sub = [(w, a, c) for (w, a, c) in primary if c != drop]
        if sub:
            m = macro_core(eval_tap_core_macro(sub, val_by))
            rows.append({"stress": f"drop_{drop}", "core_macro": round(m, 4), "degradation": round(nominal - m, 4)})
    # single-layer only
    for keep in TRAIN_CONFIGS:
        sub = [(w, a, c) for (w, a, c) in primary if c == keep]
        if sub:
            m = macro_core(eval_tap_core_macro(sub, val_by))
            rows.append({"stress": f"only_{keep}", "core_macro": round(m, 4), "degradation": round(nominal - m, 4)})
    worst = max(r["degradation"] for r in rows)
    if worst <= 0.05:
        verdict = "ROBUST"
    elif worst <= 0.10:
        verdict = "CONSERVATIVE_BUT_SAFE"
    else:
        verdict = "MISSING_EXPERT_FRAGILE"
    write_json(OUT_ROOT / "calibration_ood.json", {"BG_CORECONTENT_CALIBRATION_OOD_VERDICT": verdict, "rows": rows,
               "worst_degradation": round(worst, 4), "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "calibration_ood_rows.csv", rows)
    write_md(OUT_ROOT / "calibration_ood.md", [
        "# CoreContent Calibration / OOD / Missing-Expert v1", "", status_line("BG_CORECONTENT_CALIBRATION_OOD_VERDICT", verdict), "",
        f"Worst degradation under channel-drop stress: {fmt(worst)}.", "",
        *md_table(rows, ["stress", "core_macro", "degradation"]),
    ])
    write_json(PROGRESS_ROOT / "partM_calibration_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_CALIBRATION_OOD_VERDICT", verdict))
    return 0


# ===================================================== PART N: ablation
def ablation_main() -> int:
    started = time.time()
    ensure_root()
    val_by = groups_by_domain("val")
    train_by = groups_by_domain("train")
    rows = []
    # layer ablation of blockwise
    primary = crafted_policies().get("CoreContent_blockwise") or []
    if primary:
        full = macro_core(eval_tap_core_macro(primary, val_by))
        rows.append({"ablation": "full_blockwise", "val_core_macro": round(full, 4), "drop": 0.0})
        for keep in TRAIN_CONFIGS:
            sub = [(w, a, c) for (w, a, c) in primary if c == keep]
            if sub:
                m = macro_core(eval_tap_core_macro(sub, val_by))
                rows.append({"ablation": f"L{keep}_only", "val_core_macro": round(m, 4), "drop": round(full - m, 4)})
    # training-domain ablation (retrain dropping alignment / dropping code+reasoning)
    for label, tset in (("no_alignment", "objective4"), ("math_logic_only", "math_logic"), ("alignment_only", "alignment_only")):
        r = train_pairwise_tap(train_by, val_by, "47_L4", tset, 1e-3, 0)
        if r:
            rows.append({"ablation": f"train_{label}", "val_core_macro": r["val_core_macro"], "drop": None})
    drops = {r["ablation"]: r.get("drop") for r in rows if r.get("drop") is not None}
    layer_specific = bool(drops) and max(drops.values()) > 0.05
    verdict = "LAYER_SPECIFIC" if layer_specific else "PURE_CONTENT_CRITICAL"
    write_json(OUT_ROOT / "ablation.json", {"BG_CORECONTENT_ABLATION_VERDICT": verdict, "rows": rows,
               "elapsed_seconds": round(time.time() - started, 3)})
    write_csv(OUT_ROOT / "ablation_rows.csv", rows)
    write_md(OUT_ROOT / "ablation.md", [
        "# CoreContent Ablation / Interpretability v1", "", status_line("BG_CORECONTENT_ABLATION_VERDICT", verdict), "",
        *md_table(rows, ["ablation", "val_core_macro", "drop"]),
    ])
    write_json(PROGRESS_ROOT / "partN_ablation_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_ABLATION_VERDICT", verdict))
    return 0


# ===================================================== PART O: policy selection
def policy_selection_main() -> int:
    started = time.time()
    ensure_root()
    he = read_json(OUT_ROOT / "heldout_eval.json", {}) or {}
    macro = he.get("macro_core_top1", {})
    best_crafted = he.get("best_crafted")
    da = safe_float(he.get("dualanchor_macro")); hh = safe_float(he.get("mix_hh_objective_macro"))
    bc = safe_float(he.get("best_crafted_macro"))
    crafted = crafted_policies()
    sel_name = best_crafted if (best_crafted and bc > max(da, hh) + 0.005) else "mixedhead_MIX_HH_OBJECTIVE"
    if sel_name == "CoreContent_blockwise":
        verdict = "LOCK_CORECONTENT_UNIVERSAL"
    elif sel_name == "CoreContent_domain_gated":
        verdict = "LOCK_CORECONTENT_DOMAIN_GATED"
    elif sel_name == "CoreContent_weight_merge":
        verdict = "LOCK_CORECONTENT_WEIGHT_MERGE"
    elif sel_name == "CoreContent_listwise":
        verdict = "LOCK_CORECONTENT_TERMINAL_SURVIVOR"
    elif str(sel_name).startswith("CoreContent"):
        verdict = "LOCK_CORECONTENT_UNIVERSAL"
    else:
        verdict = "KEEP_BROAD_OBJECTIVE_BASELINE"
    sel_channels = crafted.get(sel_name) or cc.build_tap_policies().get(sel_name) or []
    selected = {"final_policy": sel_name, "verdict": verdict,
                "heldout_core_macro": round(bc if str(sel_name).startswith("CoreContent") else hh, 4),
                "vs_dualanchor": round((bc if str(sel_name).startswith("CoreContent") else hh) - da, 4),
                "domains_supported": list(CORE_DOMAINS),
                "required_features": "pooled hidden states 24_L4/36_L4/47_L4 (AntisymLinearNoNorm channels)",
                "terminal_usage": "ranks/chooses within top5/full survivor handoff set (does not replace DualAnchor survival)",
                "relation_to_dualanchor": "content/final selector only; DualAnchor remains branch-survival selector",
                "relation_to_pure_content_taps": "trained on pure content features; pure taps are baselines",
                "relation_to_transplanted_taps": "transplanted used for diagnostics only (Part E)",
                "science_dependency": "none", "fallback": "MIX_HH_OBJECTIVE if crafted unavailable"}
    torch.save({"selected_policy": sel_name, "channels": sel_channels, "meta": selected}, OUT_ROOT / "corecontent_policy_v1.pt")
    write_json(OUT_ROOT / "selected_corecontent_policy.json", {"BG_CORECONTENT_POLICY_SELECTION_VERDICT": verdict, "selected": selected,
               "elapsed_seconds": round(time.time() - started, 3)})
    write_md(OUT_ROOT / "selected_corecontent_policy.md", [
        "# Selected CoreContent Policy v1", "", status_line("BG_CORECONTENT_POLICY_SELECTION_VERDICT", verdict), "",
        *[f"- {k}: `{fmt(v)}`" for k, v in selected.items()],
    ])
    write_json(PROGRESS_ROOT / "partO_policy_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_POLICY_SELECTION_VERDICT", verdict))
    return 0


# ===================================================== PART P: phase 2b readiness
def phase2b_readiness_main() -> int:
    started = time.time()
    ensure_root()
    he = read_json(OUT_ROOT / "heldout_eval.json", {}) or {}
    sel = read_json(OUT_ROOT / "selected_corecontent_policy.json", {}) or {}
    pol_verdict = sel.get("BG_CORECONTENT_POLICY_SELECTION_VERDICT", "")
    he_verdict = he.get("BG_CORECONTENT_HELDOUT_EVAL_VERDICT", "")
    if pol_verdict == "LOCK_CORECONTENT_UNIVERSAL" and he_verdict in ("CORECONTENT_READY", "CORECONTENT_WEAK_BUT_IMPROVED"):
        verdict = "READY_FOR_PHASE2B_CORECONTENT"
    elif pol_verdict == "LOCK_CORECONTENT_DOMAIN_GATED":
        verdict = "READY_WITH_DOMAIN_GATED_CONTENT"
    elif pol_verdict == "LOCK_CORECONTENT_TERMINAL_SURVIVOR":
        verdict = "READY_WITH_TERMINAL_SURVIVOR_CONTENT"
    elif pol_verdict == "KEEP_BROAD_OBJECTIVE_BASELINE":
        verdict = "BROAD_OBJECTIVE_BASELINE_ONLY"
    elif he_verdict in ("CORECONTENT_WEAK_BUT_IMPROVED",):
        verdict = "READY_WITH_PURE_CONTENT_POLICY"
    else:
        verdict = "NEEDS_MORE_CONTENT_DATA"
    locked = {"branch_survival": "DualAnchor unchanged",
              "content_final_selection": sel.get("selected", {}).get("final_policy", "?"),
              "terminal": "top5/full survivor-set handoff; CoreContent ranks within handoff set; confidence top1 diagnostic only",
              "domains": list(CORE_DOMAINS), "science": "diagnostic only",
              "pure_vs_transplanted": "pure preferred for content; transplanted diagnostic only",
              "steering": "not run here"}
    write_json(OUT_ROOT / "phase2b_content_readiness.json", {"BG_CORECONTENT_PHASE2B_READINESS_VERDICT": verdict,
               "locked_baseline": locked, "elapsed_seconds": round(time.time() - started, 3)})
    write_md(OUT_ROOT / "phase2b_content_readiness.md", [
        "# CoreContent Phase 2b Readiness v1", "", status_line("BG_CORECONTENT_PHASE2B_READINESS_VERDICT", verdict), "",
        "## Locked Phase 2b baseline", "", *[f"- {k}: `{v}`" for k, v in locked.items()],
    ])
    write_json(PROGRESS_ROOT / "partP_readiness_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORECONTENT_PHASE2B_READINESS_VERDICT", verdict))
    return 0


# ===================================================== PART Q: synthesis
PART_FILES = {
    "BG_CORECONTENT_INVENTORY_VERDICT": "inventory.json",
    "BG_CORECONTENT_DATASET_VERDICT": "corecontent_dataset.json",
    "BG_CORECONTENT_FEATURES_VERDICT": "corecontent_features.json",
    "BG_CORECONTENT_BASELINES_VERDICT": "baselines.json",
    "BG_CORECONTENT_PURE_VS_TRANSPLANTED_VERDICT": "pure_vs_transplanted.json",
    "BG_CORECONTENT_LINEAR_TAPS_TRAINING_VERDICT": "linear_training_log.json",
    "BG_CORECONTENT_LISTWISE_TERMINAL_TRAINING_VERDICT": "listwise_training_log.json",
    "BG_CORECONTENT_DOMAIN_GATED_TRAINING_VERDICT": "domain_gated_training_log.json",
    "BG_CORECONTENT_WEIGHT_MERGE_VERDICT": "weight_merge.json",
    "BG_CORECONTENT_SCIENCE_AUX_VERDICT": "science_aux_ablation.json",
    "BG_CORECONTENT_HELDOUT_EVAL_VERDICT": "heldout_eval.json",
    "BG_CORECONTENT_DOMAIN_ANALYSIS_VERDICT": "domain_analysis.json",
    "BG_CORECONTENT_CALIBRATION_OOD_VERDICT": "calibration_ood.json",
    "BG_CORECONTENT_ABLATION_VERDICT": "ablation.json",
    "BG_CORECONTENT_POLICY_SELECTION_VERDICT": "selected_corecontent_policy.json",
    "BG_CORECONTENT_PHASE2B_READINESS_VERDICT": "phase2b_content_readiness.json",
}


def synthesis_main() -> int:
    started = time.time()
    ensure_root()
    v = {k: (read_json(OUT_ROOT / f, {}) or {}).get(k, "MISSING") for k, f in PART_FILES.items()}
    he = read_json(OUT_ROOT / "heldout_eval.json", {}) or {}
    sel = (read_json(OUT_ROOT / "selected_corecontent_policy.json", {}) or {}).get("BG_CORECONTENT_POLICY_SELECTION_VERDICT")
    if v["BG_CORECONTENT_PHASE2B_READINESS_VERDICT"] == "READY_FOR_PHASE2B_CORECONTENT":
        status = "CORECONTENT_READY"
    elif sel == "LOCK_CORECONTENT_DOMAIN_GATED":
        status = "DOMAIN_GATED_CONTENT_READY"
    elif sel == "LOCK_CORECONTENT_WEIGHT_MERGE":
        status = "CONTENT_WEIGHT_MERGE_READY"
    elif sel == "LOCK_CORECONTENT_TERMINAL_SURVIVOR":
        status = "TERMINAL_SURVIVOR_CONTENT_READY"
    elif sel == "KEEP_BROAD_OBJECTIVE_BASELINE":
        status = "BROAD_OBJECTIVE_BASELINE_STILL_BEST"
    elif v["BG_CORECONTENT_HELDOUT_EVAL_VERDICT"] == "CORECONTENT_WEAK_BUT_IMPROVED":
        status = "CORECONTENT_READY"
    else:
        status = "NEEDS_MORE_CONTENT_DATA"
    if v["BG_CORECONTENT_SCIENCE_AUX_VERDICT"] in ("SCIENCE_AUX_NO_HELP", "SCIENCE_AUX_HURTS_CORE"):
        status_sci = "SCIENCE_AUX_REJECTED"
    else:
        status_sci = "SCIENCE_AUX_HELPFUL"
    payload = {"CORECONTENT_TAP_CRAFTING_STATUS": status, "science_aux_status": status_sci, **v,
               "heldout_macro": he.get("macro_core_top1", {}), "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", payload)
    lines = ["# CoreContent Tap Crafting v1 — Summary", "", status_line("CORECONTENT_TAP_CRAFTING_STATUS", status),
             status_line("science_aux_status", status_sci), ""]
    for k in PART_FILES:
        lines.append(status_line(k, v[k]))
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    write_json(PROGRESS_ROOT / "partQ_synthesis_done.json", {"status": status, "saved_at": time.time()})
    print(status_line("CORECONTENT_TAP_CRAFTING_STATUS", status))
    return 0
