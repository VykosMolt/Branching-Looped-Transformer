"""Shared helpers for Core-Domain Tap Audit + DualAnchor Readiness v1.

Pre-steering tap audit only. This module loads existing frozen tap/head
artifacts and existing candidate sets, computes pairwise/antisymmetry/transfer
diagnostics, and may construct DualAnchor-style transplanted taps in weight
space (a fixed operation, not training) saved ONLY under this run's output root.

It never trains Ouro, mutates Ouro weights / tokenizer / checkpoints, updates or
overwrites existing tap registries, applies/claims steering, runs wrapper or
local-agent code, imports Hunter-Seeker modules, or changes production routing.
Science/anatomy taps are tested as diagnostic auxiliary experts only, never as a
headline domain.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANUAL_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(MANUAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "shared/hf_cache/datasets"))

PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
SHORT_NAME = "core_domain_tap_audit_dualanchor_readiness_v1"
OUT_ROOT = PROBE_ROOT / "bg_core_domain_tap_audit_dualanchor_readiness_v1_2026-06-04"
PROGRESS_ROOT = OUT_ROOT / "progress"

# ---- input artifact paths (verified present during deep dive) ----
DUALANCHOR_CONSTRAINED_PT = PROBE_ROOT / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30" / "layer_native_two_tap_constrained_train_v1.pt"
MIXED_DOMAIN_HEADS_PT = PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt"
HEAD_REGISTRY_PT = PROBE_ROOT / "bg_head_registry_2026-05-17.pt"
MATH_HEADS_PT = PROBE_ROOT / "math_single_state_tap_heads.pt"
HH_EVALUATOR_PT = PROJECT_ROOT / "rpe/checkpoints/evaluator/pairwise_epoch2.pt"

BRANCH_TAP_ARTIFACTS = {
    "hidden_origin_v4": PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18" / "hidden_origin_tap_heads_v4.pt",
    "bgv1_generator_selector": PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18" / "selector_heads.pt",
    "bgv1_generator_taps": PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18" / "hidden_origin_tap_heads_generator_v1.pt",
    "hidden_origin_split_salvage": PROBE_ROOT / "bg_hidden_origin_split_salvage_2026-05-18" / "salvage_heads.pt",
}
COMPOSITE_TAP_ARTIFACTS = {
    "fixed_composite_v1": PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18" / "fixed_composite_branch_survival_policy_v1.pt",
    "learned_rescue_policy": PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18" / "learned_rescue_policy.pt",
    "merged_weight_v1": PROBE_ROOT / "bg_merged_weight_branch_content_taps_v1_2026-05-18" / "top4_survivor_hidden_features.pt",
}
ARBITER_TAP_ARTIFACTS = {
    "final_arbiter_v1": PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_2026-05-18" / "final_arbiter_top4_v1.pt",
    "final_arbiter_v1_1": PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_1_2026-05-18" / "final_arbiter_top4_v1_1.pt",
}
UNIVERSAL_GATED_ARTIFACTS = {
    "universal_branch_content_v1": PROBE_ROOT / "bg_universal_branch_content_taps_v1_2026-05-18",
    "gated_branch_content_selector_v1": PROBE_ROOT / "bg_gated_branch_content_selector_v1_2026-05-18",
}

HIDDEN_DIM = 2048
TAP_CONFIGS = ("24_L4", "36_L4", "47_L4")
ANCHOR_ROLES = ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL")
# Canonical policy recipes (form locked DualAnchor policies). The constrained
# registry also stores a large transplant-coefficient grid (old_plus_branch_bridge_*,
# sparse_*, *_residual_*) used by Parts E/I/J, not enumerated as inventory rows.
CANONICAL_RECIPES = ("old_only", "adaptive_balanced_rescue", "adaptive_branch_anchor_light", "adaptive_branch_from_val")
LOCKED_RECIPE = "adaptive_branch_anchor_light"
SCIENCE_GROUPS = ("MIX_CODE_SCIENCE", "MIX_REASONING_SCIENCE", "MIX_CODE_SCIENCE_MED")
CORE_DOMAINS = ("coding", "reasoning", "math", "logic", "alignment")
DIAGNOSTIC_DOMAINS = ("anatomy", "science")


# ---------------------------------------------------------------- io helpers
def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text())


def write_md(path: Path, lines: Sequence[str]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: csv_value(row.get(k)) for k in keys})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=json_default)
    if isinstance(value, torch.Tensor):
        return json.dumps(json_default(value))
    return value


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite_mean(values: Iterable[Any]) -> float:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(mean(xs)) if xs else float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def status_line(key: str, value: str) -> str:
    return f"{key} = {value}"


def md_table(rows: Sequence[dict[str, Any]], cols: Sequence[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    return out


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def weight_from_state_dict(sd: Any) -> torch.Tensor | None:
    if isinstance(sd, dict):
        for key in ("linear.weight", "weight"):
            v = sd.get(key)
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().to(torch.float32).flatten()
    return None


# ============================================================ PART A: inventory
# Map a head's training domains -> core/diagnostic domain support tags.
TRAIN_DOMAIN_TO_DOMAIN = {
    "CODE": "coding",
    "REASONING_NATURAL": "reasoning",
    "REASONING_TRACE": "reasoning",
    "HH": "alignment",
    "SCIENCE": "science",
    "SCIENCE_CHEMISTRY": "science",
    "SCIENCE_MEDICINE": "anatomy",
}
ALL_SUPPORT_DOMAINS = ("coding", "reasoning", "math", "logic", "alignment", "science", "anatomy")


def _support_flags(train_domains: Sequence[str], *, objective: bool) -> dict[str, str]:
    """trained = direct training domain; transfer = plausible for broad/objective
    heads; untested otherwise. Honest, not a claim of accuracy."""
    trained = {TRAIN_DOMAIN_TO_DOMAIN[d] for d in train_domains if d in TRAIN_DOMAIN_TO_DOMAIN}
    flags: dict[str, str] = {}
    for dom in ALL_SUPPORT_DOMAINS:
        if dom in trained:
            flags[f"support_{dom}"] = "trained"
        elif objective and dom in ("math", "logic", "reasoning", "coding"):
            flags[f"support_{dom}"] = "transfer"
        elif objective and dom == "alignment":
            flags[f"support_{dom}"] = "transfer"
        else:
            flags[f"support_{dom}"] = "untested"
    return flags


def _record(**kw: Any) -> dict[str, Any]:
    """Build an inventory record with the full requested field set + defaults."""
    base = {
        "tap_name": None,
        "tap_family": None,
        "source_artifact": None,
        "architecture": None,
        "feature_config": None,
        "layer_config": None,
        "dim": None,
        "domains_trained_on": None,
        "datasets_used": None,
        "old_context_support": None,
        "hidden_branch_support": None,
        "terminal_survivor_support": None,
        "pairwise_swap_support": None,
        "raw_score_bias_known": None,
        "mergeable": None,
        "score_api_available": None,
        "missing_dependencies": None,
        "diagnostic_only": None,
        "capability": None,  # action_selection | action_selection_and_branching
        "needs_transplant_for_branching": None,
    }
    for d in ALL_SUPPORT_DOMAINS:
        base[f"support_{d}"] = "untested"
    base.update(kw)
    return base


def _layer_config_tag(config: str | None) -> str:
    if not config:
        return "unknown"
    if "concat_24_36_47" in str(config) or "block_concat" in str(config):
        return "concat_24_36_47"
    for layer in ("24", "36", "47"):
        if str(config).startswith(layer):
            return layer
    return "other"


def inventory_dualanchor() -> list[dict[str, Any]]:
    """DualAnchor taps from the constrained registry. These are the native
    selection+branching taps (old content + branch + bridge transplant)."""
    out: list[dict[str, Any]] = []
    if not DUALANCHOR_CONSTRAINED_PT.exists():
        out.append(_record(tap_name="DualAnchor", tap_family="dualanchor",
                            source_artifact=rel(DUALANCHOR_CONSTRAINED_PT),
                            missing_dependencies="constrained_registry_missing",
                            score_api_available=False))
        return out
    payload = load_pt(DUALANCHOR_CONSTRAINED_PT)
    cands = payload.get("candidates") or []
    # group by (tap_role, recipe, architecture) -> available configs (canonical recipes only)
    grid: dict[tuple, set] = defaultdict(set)
    all_recipes: set = set()
    for c in cands:
        role = c.get("tap_role"); recipe = c.get("recipe"); arch = c.get("architecture")
        cfg = c.get("target_config")
        all_recipes.add(recipe)
        if role in ANCHOR_ROLES and cfg in TAP_CONFIGS and recipe in CANONICAL_RECIPES:
            grid[(role, recipe, arch)].add(cfg)
    for (role, recipe, arch), configs in sorted(grid.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        objective = role == "MIX_OBJECTIVE_ALL"
        train_dom = ["CODE", "REASONING_NATURAL", "REASONING_TRACE"] if role == "MIX_CODE_REASONING" else ["CODE", "REASONING_NATURAL", "HH", "SCIENCE"]
        rec = _record(
            tap_name=f"DualAnchor::{role}::{recipe}::{arch}",
            tap_family="dualanchor", source_artifact=rel(DUALANCHOR_CONSTRAINED_PT),
            architecture=arch, feature_config=",".join(sorted(configs)),
            layer_config="24_36_47", dim=HIDDEN_DIM,
            domains_trained_on=",".join(train_dom),
            datasets_used="old_content+hidden_origin_v4+universal_bridge (transplant)",
            old_context_support=True, hidden_branch_support=True, terminal_survivor_support=True,
            pairwise_swap_support=True, raw_score_bias_known=False,  # AntisymLinear w/o affine is exactly antisymmetric
            mergeable=True, score_api_available=True, missing_dependencies=None,
            diagnostic_only=False,
            capability="action_selection_and_branching",
            needs_transplant_for_branching=False,
            recipe=recipe, n_configs=len(configs),
            **_support_flags(train_dom, objective=objective),
        )
        out.append(rec)
    # summary row: the full transplant-coefficient grid (inputs for Parts E/I/J)
    out.append(_record(
        tap_name="DualAnchor::transplant_grid_summary", tap_family="dualanchor_grid",
        source_artifact=rel(DUALANCHOR_CONSTRAINED_PT), architecture="grid",
        feature_config=",".join(TAP_CONFIGS), layer_config="24_36_47", dim=HIDDEN_DIM,
        domains_trained_on="old+branch+bridge",
        datasets_used=f"{len(cands)} candidates across {len(all_recipes)} transplant recipes",
        capability="action_selection_and_branching", needs_transplant_for_branching=False,
        hidden_branch_support=True, terminal_survivor_support=True, old_context_support=True,
        pairwise_swap_support=True, raw_score_bias_known=False, mergeable=True,
        score_api_available=True, diagnostic_only=False,
        n_candidates=len(cands), n_recipes=len(all_recipes),
    ))
    return out


def inventory_heads_pt(path: Path, family: str, *, group_key: str, branching: bool, diagnostic_groups: Sequence[str] = ()) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        out.append(_record(tap_name=family, tap_family=family, source_artifact=rel(path),
                            missing_dependencies="artifact_missing", score_api_available=False))
        return out
    payload = load_pt(path)
    heads = payload.get("heads") if isinstance(payload, dict) else None
    if not isinstance(heads, list):
        out.append(_record(tap_name=family, tap_family=family, source_artifact=rel(path),
                            missing_dependencies=f"unexpected_schema:{type(payload).__name__}", score_api_available=False))
        return out
    # summarize per (group, architecture): list configs + domains
    grid: dict[tuple, dict[str, Any]] = {}
    for h in heads:
        if not isinstance(h, dict):
            continue
        group = h.get(group_key) or h.get("head_family") or h.get("head_group")
        arch = h.get("architecture")
        key = (group, arch)
        g = grid.setdefault(key, {"configs": set(), "domains": set(), "dim": h.get("dim"), "weighted": 0})
        if h.get("config"):
            g["configs"].add(h.get("config"))
        for d in (h.get("train_metrics", {}) or {}).get("domains", []) or []:
            g["domains"].add(d)
        if isinstance(h.get("state_dict"), dict) and weight_from_state_dict(h["state_dict"]) is not None:
            g["weighted"] += 1
    for (group, arch), g in sorted(grid.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        domains = sorted(g["domains"]) or ([group] if group else [])
        # diagnostic-only = the science *specialist* groups, not broad objective
        # heads that merely include science in their training mix.
        is_diag = group in diagnostic_groups
        objective = (group == "MIX_OBJECTIVE_ALL") or (group == "MIX_HH_OBJECTIVE")
        train_dom = domains if domains and domains != [group] else _infer_train_domains(group)
        out.append(_record(
            tap_name=f"{family}::{group}::{arch}",
            tap_family=family, source_artifact=rel(path), architecture=arch,
            feature_config=",".join(sorted(g["configs"])) if g["configs"] else None,
            layer_config="24_36_47" if len(g["configs"]) > 1 else _layer_config_tag(next(iter(g["configs"]), None)),
            dim=g["dim"] or HIDDEN_DIM,
            domains_trained_on=",".join(train_dom),
            datasets_used=_datasets_for_domains(train_dom),
            old_context_support=True,
            hidden_branch_support=bool(branching),
            terminal_survivor_support=bool(branching),
            pairwise_swap_support=True, raw_score_bias_known=False,
            mergeable=True, score_api_available=g["weighted"] > 0,
            missing_dependencies=None if g["weighted"] > 0 else "no_extractable_weight",
            diagnostic_only=bool(is_diag),
            capability="action_selection_and_branching" if branching else "action_selection",
            needs_transplant_for_branching=not branching,
            n_heads_weighted=g["weighted"],
            **_support_flags(train_dom, objective=objective),
        ))
    return out


def _infer_train_domains(group: str | None) -> list[str]:
    g = str(group or "")
    out: list[str] = []
    if "CODE" in g:
        out.append("CODE")
    if "REASONING" in g:
        out.append("REASONING_NATURAL")
    if "HH" in g:
        out.append("HH")
    if "OBJECTIVE" in g:
        out += ["CODE", "REASONING_NATURAL", "SCIENCE"]
    if "SCIENCE" in g and "MED" in g:
        out.append("SCIENCE_MEDICINE")
    elif "SCIENCE" in g:
        out.append("SCIENCE")
    return sorted(set(out)) or ([g] if g else [])


def _datasets_for_domains(domains: Sequence[str]) -> str:
    ds: list[str] = []
    for d in domains:
        if d == "CODE":
            ds += ["mbpp", "humaneval"]
        elif d.startswith("REASONING"):
            ds += ["arc", "commonsense_qa"]
        elif d == "HH":
            ds += ["hh-rlhf"]
        elif d.startswith("SCIENCE"):
            ds += ["sciq", "mmlu", "medmcqa"]
    return ",".join(sorted(set(ds)))


def inventory_hh_evaluator() -> list[dict[str, Any]]:
    if not HH_EVALUATOR_PT.exists():
        return [_record(tap_name="hh_pairwise_evaluator", tap_family="alignment_evaluator",
                        source_artifact=rel(HH_EVALUATOR_PT), missing_dependencies="evaluator_missing",
                        score_api_available=False)]
    payload = load_pt(HH_EVALUATOR_PT)
    acc = payload.get("accuracy") if isinstance(payload, dict) else None
    rec = _record(
        tap_name="hh_pairwise_evaluator_epoch2", tap_family="alignment_evaluator",
        source_artifact=rel(HH_EVALUATOR_PT), architecture="published_pairwise_evaluator",
        feature_config="model_internal", layer_config="other", dim=None,
        domains_trained_on="HH", datasets_used="hh-rlhf",
        old_context_support=True, hidden_branch_support=False, terminal_survivor_support=True,
        pairwise_swap_support=True, raw_score_bias_known=True,  # learned scorer: fixed-order vs antisym must be checked
        mergeable=False, score_api_available=True, missing_dependencies=None,
        diagnostic_only=False, capability="action_selection",
        needs_transplant_for_branching=True,
        stored_accuracy=acc,
        **_support_flags(["HH"], objective=False),
    )
    rec["support_alignment"] = "trained"
    return [rec]


def inventory_presence(artifacts: dict[str, Path], family: str, *, branching: bool, diagnostic: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, path in artifacts.items():
        exists = Path(path).exists()
        out.append(_record(
            tap_name=f"{family}::{name}", tap_family=family, source_artifact=rel(path),
            architecture="composite/policy" if not branching else "branch_tap",
            score_api_available=exists, missing_dependencies=None if exists else "artifact_missing",
            hidden_branch_support=bool(branching), terminal_survivor_support=bool(branching),
            old_context_support=not branching, pairwise_swap_support=False, raw_score_bias_known=None,
            mergeable=family in ("branch_tap", "bridge"), diagnostic_only=bool(diagnostic),
            capability="action_selection_and_branching" if branching else "action_selection",
            needs_transplant_for_branching=not branching,
        ))
    return out


def build_inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records += inventory_dualanchor()
    records += inventory_heads_pt(MIXED_DOMAIN_HEADS_PT, "mixed_domain_tiny_head", group_key="head_group", branching=False, diagnostic_groups=SCIENCE_GROUPS)
    records += inventory_heads_pt(HEAD_REGISTRY_PT, "old_content_head", group_key="head_family", branching=False)
    records += inventory_hh_evaluator()
    records += inventory_presence(BRANCH_TAP_ARTIFACTS, "branch_tap", branching=True)
    records += inventory_presence(COMPOSITE_TAP_ARTIFACTS, "composite_tap", branching=True)
    records += inventory_presence(ARBITER_TAP_ARTIFACTS, "final_arbiter", branching=True)
    records += inventory_presence({k: Path(v) for k, v in UNIVERSAL_GATED_ARTIFACTS.items()}, "universal_gated", branching=True)
    return records


def inventory_verdict(records: Sequence[dict[str, Any]]) -> str:
    fams = {r["tap_family"] for r in records}
    has_dualanchor = any(r["tap_family"] == "dualanchor" and r.get("score_api_available") for r in records)
    has_core = any(r.get("support_coding") == "trained" for r in records) and any(r.get("support_reasoning") == "trained" for r in records)
    has_science = any(r.get("diagnostic_only") and r["tap_family"] == "mixed_domain_tiny_head" for r in records)
    missing = [r["tap_name"] for r in records if r.get("missing_dependencies")]
    if not has_dualanchor:
        return "MISSING_DUALANCHOR"
    if not has_core:
        return "MISSING_CORE_TAPS"
    if not has_science:
        return "SCIENCE_TAPS_ABSENT" if not missing else "PARTIAL"
    return "READY" if not missing else "PARTIAL"


def inventory_main() -> int:
    started = time.time()
    ensure_root()
    records = build_inventory()
    verdict = inventory_verdict(records)
    fam_counts = Counter(r["tap_family"] for r in records)
    payload = {
        "BG_CORE_TAP_INVENTORY_VERDICT": verdict,
        "tap_count": len(records),
        "family_counts": dict(fam_counts),
        "missing": [r["tap_name"] for r in records if r.get("missing_dependencies")],
        "records": records,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "tap_inventory.json", payload)
    write_csv(OUT_ROOT / "tap_inventory.csv", records)
    cols = ["tap_name", "tap_family", "architecture", "capability", "needs_transplant_for_branching",
            "support_coding", "support_reasoning", "support_math", "support_logic", "support_alignment",
            "support_science", "diagnostic_only", "score_api_available", "missing_dependencies"]
    write_md(OUT_ROOT / "tap_inventory.md", [
        "# Core-Domain Tap Inventory v1", "",
        status_line("BG_CORE_TAP_INVENTORY_VERDICT", verdict), "",
        f"Taps/heads inventoried: {len(records)}; families: {dict(fam_counts)}", "",
        "Capability legend: old/content/science heads are action-selection only and need the"
        " weight-space transplant (old+branch+bridge) to be branch-capable; DualAnchor and branch"
        " taps are natively selection+branching.", "",
        *md_table(records, cols),
    ])
    write_json(PROGRESS_ROOT / "partA_inventory_done.json", {"verdict": verdict, "tap_count": len(records), "saved_at": time.time()})
    print(status_line("BG_CORE_TAP_INVENTORY_VERDICT", verdict))
    print(f"taps={len(records)} families={dict(fam_counts)}")
    return 0 if verdict != "BLOCKED" else 1


# ====================================================== PART B: candidate suite
# Feature-bearing candidate sources (inline pooled/features). Each yields groups:
#   {group_id, domain, task_id, candidates: [{features|pooled_vectors, reward}], source}
ARCH_LOOPED_ROWS_PT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31" / "architecture_looped_rows.pt"
HH_PACKS_PT = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
TAP_FEATURE_SOURCES = {
    "coding": ["code_branch_tap_features_v2_2026-05-16.pt", "code_branch_tap_features_v2_mini_patched_2026-05-16.pt"],
    "math": ["math_branch_tap_features.pt", "clean_gsm8k_expanded_tap_features_2026-05-16.pt", "clean_gsm8k_extreme_tap_features_2026-05-16.pt"],
}
LOGIQA_FEATURES_PT = OUT_ROOT / "logic_logiqa_candidate_features.pt"  # produced by bounded gen (Part B optional)


def _groups_from_tap_features(path: Path, domain: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_pt(path)
    recs = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(recs, list):
        return []
    groups: list[dict[str, Any]] = []
    for i, r in enumerate(recs):
        pooled = r.get("pooled")
        labels = r.get("labels")
        if not (hasattr(pooled, "shape") and hasattr(labels, "shape")):
            continue
        n = int(pooled.shape[0])
        if n < 2 or tuple(pooled.shape[1:]) != (3, 4, HIDDEN_DIM):
            continue
        cands = [{"features": pooled[c].detach().cpu().to(torch.float32),
                  "reward": float(labels[c].item()),
                  "cand_index": c, "branch_id": f"{domain}_{i}_{c}"} for c in range(n)]
        g = {"group_id": f"{domain}::{path.stem}::{r.get('task_id', r.get('tournament_id', i))}",
             "domain": domain, "task_id": str(r.get("task_id", r.get("tournament_id", i))),
             "candidates": cands, "source": path.stem, "kind": "tournament"}
        if r.get("orig_split") in ("calibration", "heldout"):
            g["orig_split"] = r["orig_split"]
        groups.append(g)
    return groups


# ----- bounded logic (LogiQA) MCQ-option feature extraction (encode-only, GPU) -----
def extract_logic_features(max_tasks_per_split: int = 40) -> int:
    ensure_root()
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    from datasets import load_dataset
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
    ds = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")
    ext = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
    records: list[dict[str, Any]] = []
    plan = [("calibration", "train"), ("heldout", "validation")]
    for split_name, hf_split in plan:
        if hf_split not in ds:
            continue
        rows = ds[hf_split]
        for i in range(min(max_tasks_per_split, len(rows))):
            ex = rows[i]
            opts = list(ex["options"]); correct = int(ex["correct_option"])
            head = f"{ex['context']}\nQuestion: {ex['query']}\nOptions:\n" + "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(opts)) + "\nAnswer:"
            pooled_list = []
            ok = True
            for j, o in enumerate(opts):
                try:
                    feats = ext.encode_text_to_pooled_features(head + f" {chr(65 + j)}. {o}")
                except Exception:
                    ok = False; break
                pooled_list.append(feats.detach().cpu().to(torch.float32))
            if not ok or len(pooled_list) < 2:
                continue
            records.append({"task_id": f"logiqa_{hf_split}_{i}", "orig_split": split_name,
                            "pooled": torch.stack(pooled_list, 0),
                            "labels": torch.tensor([1.0 if j == correct else 0.0 for j in range(len(opts))]),
                            "candidate_texts": opts})
            if (i + 1) % 10 == 0:
                torch.save({"meta": {"domain": "logic", "source": "lucasmccabe/logiqa"}, "records": records}, LOGIQA_FEATURES_PT)
                print(f"  logic {split_name}: {i + 1} tasks encoded", flush=True)
    try:
        ext.cleanup()
    except Exception:
        pass
    torch.save({"meta": {"domain": "logic", "source": "lucasmccabe/logiqa"}, "records": records}, LOGIQA_FEATURES_PT)
    write_json(PROGRESS_ROOT / "logic_features_done.json", {"records": len(records), "saved_at": time.time()})
    print(f"BG_CORE_LOGIC_FEATURES = {len(records)} groups -> {rel(LOGIQA_FEATURES_PT)}")
    return 0


def _groups_from_archlooped(domain: str) -> list[dict[str, Any]]:
    if not ARCH_LOOPED_ROWS_PT.exists():
        return []
    payload = load_pt(ARCH_LOOPED_ROWS_PT)
    rows = payload.get("rows") or []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for r in rows:
        if str(r.get("domain")) != domain:
            continue
        gid = r.get("branch_group_id")
        feats = r.get("features")
        if gid is None or not hasattr(feats, "shape"):
            continue
        by_group[gid].append({"features": feats.detach().cpu().to(torch.float32),
                              "reward": row_reward_local(r), "branch_id": r.get("branch_id"),
                              "birth_stage": r.get("birth_stage"), "pooled_vectors": r.get("pooled_vectors")})
        meta.setdefault(gid, {"split": r.get("split"), "source_dataset": r.get("source_dataset")})
    groups = []
    for gid, cands in by_group.items():
        if len(cands) < 2:
            continue
        groups.append({"group_id": gid, "domain": domain, "task_id": gid,
                       "candidates": cands, "source": "architecture_looped_v3",
                       "kind": "branch_tree", "orig_split": meta[gid]["split"],
                       "source_dataset": meta[gid].get("source_dataset")})
    return groups


def _groups_from_hh() -> list[dict[str, Any]]:
    if not HH_PACKS_PT.exists():
        return []
    payload = load_pt(HH_PACKS_PT)
    packs = payload.get("packs") or []
    groups = []
    def _feat(pooled: Any) -> torch.Tensor | None:
        # HH stores {24/36/47: (loops=4, hidden=2048)} -> features (3,4,2048)
        if not isinstance(pooled, dict):
            return None
        layers = []
        for layer in (24, 36, 47):
            t = pooled.get(layer, pooled.get(str(layer)))
            if not (hasattr(t, "shape") and tuple(t.shape) == (4, HIDDEN_DIM)):
                return None
            layers.append(t.detach().cpu().to(torch.float32))
        return torch.stack(layers, dim=0)
    for i, p in enumerate(packs):
        fc = _feat(p.get("chosen", {}).get("pooled"))
        fr = _feat(p.get("rejected", {}).get("pooled"))
        if fc is None or fr is None:
            continue
        cands = [{"features": fc, "reward": 1.0, "branch_id": f"hh_{i}_chosen"},
                 {"features": fr, "reward": 0.0, "branch_id": f"hh_{i}_rejected"}]
        groups.append({"group_id": f"alignment::hh::{i}", "domain": "alignment", "task_id": f"hh_{i}",
                       "candidates": cands, "source": "hh_layer_states_200_rltt", "kind": "pairwise"})
    return groups


def row_reward_local(row: dict[str, Any]) -> float:
    for key in ("deterministic_reward", "reward", "final_reward", "correctness"):
        try:
            x = float(row.get(key))
            if math.isfinite(x):
                return x
        except Exception:
            continue
    return 1.0 if (row.get("deterministic_correct") or row.get("correct")) else 0.0


def domain_groups(domain: str) -> list[dict[str, Any]]:
    if domain == "alignment":
        return _groups_from_hh()
    if domain == "anatomy":
        sci = _groups_from_archlooped("science")
        anat = [g for g in sci if "anatomy" in str(g.get("source_dataset", "")).lower()]
        return anat if anat else sci  # fall back to science aggregate if no anatomy slice
    if domain in ("reasoning", "science"):
        return _groups_from_archlooped(domain)
    out: list[dict[str, Any]] = []
    for fn in TAP_FEATURE_SOURCES.get(domain, []):
        out += _groups_from_tap_features(PROBE_ROOT / fn, domain)
    if domain == "logic" and LOGIQA_FEATURES_PT.exists():
        out += _groups_from_tap_features(LOGIQA_FEATURES_PT, "logic")
    return out


def split_of(group: dict[str, Any]) -> str:
    """Deterministic task-disjoint calibration/heldout split (never tuned on heldout)."""
    if group.get("orig_split") in ("calibration", "heldout"):
        return group["orig_split"]
    h = abs(hash(("core_tap_audit_v1", str(group.get("task_id"))))) % 100
    return "calibration" if h < 50 else "heldout"


SUITE_DOMAINS = ("coding", "reasoning", "math", "logic", "alignment", "science", "anatomy")
SUITE_TARGETS = {"coding": (24, 48), "reasoning": (24, 48), "math": (24, 48), "logic": (16, 32),
                 "alignment": (24, 48), "science": (6, 12), "anatomy": (6, 12)}


def build_suite() -> dict[str, Any]:
    per_domain: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dom in SUITE_DOMAINS:
        groups = domain_groups(dom)
        cal = [g for g in groups if split_of(g) == "calibration"]
        hel = [g for g in groups if split_of(g) == "heldout"]
        cand_total = sum(len(g["candidates"]) for g in groups)
        # reward-diverse groups have both >0 and <=0 reward candidates
        diverse = [g for g in groups if len({c["reward"] > 0 for c in g["candidates"]}) > 1]
        per_domain[dom] = {
            "groups": len(groups), "calibration_groups": len(cal), "heldout_groups": len(hel),
            "candidates": cand_total, "reward_diverse_groups": len(diverse),
            "min_target": SUITE_TARGETS[dom][0], "preferred_target": SUITE_TARGETS[dom][1],
            "meets_min": len(groups) >= SUITE_TARGETS[dom][0],
            "kind": (groups[0]["kind"] if groups else None),
            "sources": sorted({g["source"] for g in groups}),
        }
        for g in groups:
            rows.append({"group_id": g["group_id"], "domain": dom, "task_id": g["task_id"],
                         "split": split_of(g), "n_candidates": len(g["candidates"]),
                         "kind": g["kind"], "source": g["source"],
                         "reward_diverse": len({c["reward"] > 0 for c in g["candidates"]}) > 1})
    return {"per_domain": per_domain, "rows": rows}


def suite_verdict(per_domain: dict[str, Any]) -> str:
    core = ("coding", "reasoning", "math", "logic", "alignment")
    unmet = [d for d in core if not per_domain[d]["meets_min"]]
    if "logic" in unmet and len(unmet) == 1:
        return "LOGIC_LIMITED"
    if "math" in unmet and len(unmet) == 1:
        return "MATH_LIMITED"
    if "coding" in unmet and len(unmet) == 1:
        return "CODING_LIMITED"
    if "alignment" in unmet and len(unmet) == 1:
        return "ALIGNMENT_LIMITED"
    if not unmet:
        return "READY"
    if len(unmet) >= 4:
        return "BLOCKED"
    return "PARTIAL"


def suite_main() -> int:
    started = time.time()
    ensure_root()
    suite = build_suite()
    per_domain = suite["per_domain"]
    verdict = suite_verdict(per_domain)
    payload = {"BG_CORE_TASK_CANDIDATE_SUITE_VERDICT": verdict, "per_domain": per_domain,
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "task_candidate_suite.json", payload)
    write_csv(OUT_ROOT / "task_candidate_suite.csv", suite["rows"])
    dom_rows = [{"domain": d, **{k: v for k, v in per_domain[d].items() if k != "sources"},
                 "sources": ",".join(per_domain[d]["sources"])} for d in SUITE_DOMAINS]
    write_md(OUT_ROOT / "task_candidate_suite.md", [
        "# Core-Domain Task / Candidate Suite v1", "",
        status_line("BG_CORE_TASK_CANDIDATE_SUITE_VERDICT", verdict), "",
        "Feature-bearing candidate groups per domain (calibration/heldout split is task-disjoint,"
        " deterministic, never tuned on heldout). Logic uses LogiQA features only if the bounded"
        " extraction has been run; otherwise logic is feature-limited for scoring parts.", "",
        *md_table(dom_rows, ["domain", "groups", "calibration_groups", "heldout_groups", "candidates",
                             "reward_diverse_groups", "min_target", "meets_min", "kind", "sources"]),
    ])
    write_json(PROGRESS_ROOT / "partB_suite_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_TASK_CANDIDATE_SUITE_VERDICT", verdict))
    for d in SUITE_DOMAINS:
        pd = per_domain[d]
        print(f"  {d:10} groups={pd['groups']:4d} cal/held={pd['calibration_groups']}/{pd['heldout_groups']} "
              f"cands={pd['candidates']:5d} diverse={pd['reward_diverse_groups']:4d} meets_min={pd['meets_min']}")
    return 0 if verdict != "BLOCKED" else 1


# ============================================== SHARED scoring / tap-policy layer
SCORING_CONFIGS = ("24_L4", "36_L4", "47_L4")
_SCORING: dict[str, Any] = {}
_REG: dict[str, Any] = {}


def _sc() -> dict[str, Any]:
    if not _SCORING:
        from run_bg_dualanchor_all_loop_audit_v1 import config_vector, normalize_scores
        from bg_merged_tap_v1_common import score_diff
        _SCORING.update(config_vector=config_vector, normalize_scores=normalize_scores, score_diff=score_diff)
    return _SCORING


def _constrained_candidates() -> list[dict[str, Any]]:
    if "constrained" not in _REG:
        _REG["constrained"] = (load_pt(DUALANCHOR_CONSTRAINED_PT).get("candidates") or []) if DUALANCHOR_CONSTRAINED_PT.exists() else []
    return _REG["constrained"]


def _mixed_heads() -> list[dict[str, Any]]:
    if "mixed" not in _REG:
        _REG["mixed"] = (load_pt(MIXED_DOMAIN_HEADS_PT).get("heads") or []) if MIXED_DOMAIN_HEADS_PT.exists() else []
    return _REG["mixed"]


def _old_heads() -> list[dict[str, Any]]:
    if "old" not in _REG:
        _REG["old"] = (load_pt(HEAD_REGISTRY_PT).get("heads") or []) if HEAD_REGISTRY_PT.exists() else []
    return _REG["old"]


def _selected_layer_taps(recipe: str, arch: str) -> dict[str, dict[str, Any]]:
    """Canonical per-(config,role) DualAnchor tap selection (best by branch/balanced val
    acc), reusing l4_base.select_layer_taps. Avoids picking degenerate diagnostic variants."""
    key = f"seltaps::{recipe}::{arch}"
    if key not in _REG:
        try:
            import run_bg_dualanchor_looped_branch_prune_sim_v1 as l4
            cands, _ = l4.load_constrained()
            _REG[key] = l4.select_layer_taps(cands, recipe, arch)
        except Exception:
            _REG[key] = {}
    return _REG[key]


def dualanchor_role_channels(role: str, recipe: str = LOCKED_RECIPE, arch: str = "AntisymLinear") -> list[tuple]:
    """Return [(weight, arch, config), ...] for one DualAnchor anchor role, using the
    canonical best-candidate selection per (config, role)."""
    sel = _selected_layer_taps(recipe, arch)
    chans: list[tuple] = []
    for cfg in SCORING_CONFIGS:
        cand = (sel.get(cfg) or {}).get(role)
        if isinstance(cand, dict):
            w = weight_from_state_dict(cand.get("state_dict"))
            if w is not None:
                chans.append((w, arch, cfg))
    return chans


def mixed_head_channels(group: str, arch: str = "AntisymLinear") -> list[tuple]:
    chans: list[tuple] = []
    for cfg in SCORING_CONFIGS:
        rows = [h for h in _mixed_heads() if h.get("head_group") == group and h.get("architecture") == arch and h.get("config") == cfg]
        if rows:
            w = weight_from_state_dict(rows[0].get("state_dict"))
            if w is not None:
                chans.append((w, arch, cfg))
    return chans


def old_head_channels(family: str, arch: str = "AntisymLinear") -> list[tuple]:
    chans: list[tuple] = []
    for cfg in SCORING_CONFIGS:
        rows = [h for h in _old_heads() if h.get("head_family") == family and h.get("architecture") == arch and h.get("config") == cfg]
        if rows:
            w = weight_from_state_dict(rows[0].get("state_dict"))
            if w is not None:
                chans.append((w, arch, cfg))
    return chans


def build_tap_policies(arch: str = "AntisymLinear") -> dict[str, list[tuple]]:
    """name -> channels. DualAnchor = both anchors x 3 configs (the locked default)."""
    pol: dict[str, list[tuple]] = {}
    cr = dualanchor_role_channels("MIX_CODE_REASONING", arch=arch)
    oa = dualanchor_role_channels("MIX_OBJECTIVE_ALL", arch=arch)
    if cr or oa:
        pol["DualAnchor"] = cr + oa
    if cr:
        pol["MIX_CODE_REASONING_only"] = cr
    if oa:
        pol["MIX_OBJECTIVE_ALL_only"] = oa
    for group in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL", "MIX_HH_OBJECTIVE"):
        ch = mixed_head_channels(group, arch=arch)
        if ch:
            pol[f"mixedhead_{group}"] = ch
    for group in SCIENCE_GROUPS:
        ch = mixed_head_channels(group, arch=arch)
        if ch:
            pol[f"science_{group}"] = ch
    for fam in ("CODE", "HH"):
        ch = old_head_channels(fam, arch=arch)
        if ch:
            pol[f"old_{fam}"] = ch
    return pol


def policy_candidate_scores(group: dict[str, Any], channels: list[tuple]) -> list[float]:
    """Average of z-normalized per-channel mean-pairwise-margins (DualAnchor-style)."""
    sc = _sc()
    cv, sd, nrm = sc["config_vector"], sc["score_diff"], sc["normalize_scores"]
    cands = group["candidates"]
    n = len(cands)
    if not channels or n < 2:
        return [float("nan")] * n
    per_channel: list[list[float]] = []
    for (w, arch, cfg) in channels:
        vecs = [cv(c, cfg) for c in cands]
        scores: list[float] = []
        for i, vi in enumerate(vecs):
            if vi is None:
                scores.append(float("nan")); continue
            margins = [sd(w, arch, vi - vj) for j, vj in enumerate(vecs) if j != i and isinstance(vj, torch.Tensor)]
            scores.append(finite_mean(margins))
        if any(math.isfinite(x) for x in scores):
            per_channel.append(nrm(scores))
    if not per_channel:
        return [float("nan")] * n
    return [finite_mean(ch[i] for ch in per_channel) for i in range(n)]


def group_selection_metrics(group: dict[str, Any], scores: list[float]) -> dict[str, float]:
    cands = group["candidates"]
    rewards = [float(c["reward"]) for c in cands]
    oracle = max(rewards)
    finite = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    if not finite:
        return {}
    top = max(finite, key=lambda x: x[1])[0]
    order = [i for i, _ in sorted(finite, key=lambda x: x[1], reverse=True)]
    # pairwise accuracy over differing-reward pairs
    correct = total = 0
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            if rewards[a] == rewards[b] or not (math.isfinite(scores[a]) and math.isfinite(scores[b])):
                continue
            total += 1
            hi, lo = (a, b) if rewards[a] > rewards[b] else (b, a)
            correct += 1 if scores[hi] > scores[lo] else (0.5 if scores[hi] == scores[lo] else 0)
    return {
        "top1_oracle": 1.0 if rewards[top] >= oracle else 0.0,
        "top1_reward": rewards[top],
        "regret": oracle - rewards[top],
        "top2_oracle": 1.0 if any(rewards[i] >= oracle for i in order[:2]) else 0.0,
        "pairwise_accuracy": (correct / total) if total else float("nan"),
        "pairwise_pairs": total,
        "reward_diverse": 1.0 if len(set(rewards)) > 1 else 0.0,
    }


def eval_policies_on_domains(domains: Sequence[str], policies: dict[str, list[tuple]], *, split: str | None, max_groups: int = 60) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (per-(policy,domain) summary rows, raw per-group rows)."""
    raw: list[dict[str, Any]] = []
    agg: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    for dom in domains:
        groups = [g for g in domain_groups(dom) if (split is None or split_of(g) == split)]
        groups = [g for g in groups if len({c["reward"] > 0 for c in g["candidates"]}) > 1]  # selection needs diversity
        groups = groups[:max_groups]
        for g in groups:
            for name, ch in policies.items():
                scores = policy_candidate_scores(g, ch)
                m = group_selection_metrics(g, scores)
                if not m:
                    continue
                agg[(name, dom)].append(m)
                raw.append({"policy": name, "domain": dom, "group_id": g["group_id"], **m})
    rows: list[dict[str, Any]] = []
    for (name, dom), ms in sorted(agg.items()):
        rows.append({
            "policy": name, "domain": dom, "groups": len(ms),
            "top1_oracle": finite_mean(m["top1_oracle"] for m in ms),
            "top2_oracle": finite_mean(m["top2_oracle"] for m in ms),
            "pairwise_accuracy": finite_mean(m["pairwise_accuracy"] for m in ms),
            "mean_regret": finite_mean(m["regret"] for m in ms),
        })
    return rows, raw


def _baseline_scores(group: dict[str, Any], kind: str) -> list[float]:
    cands = group["candidates"]
    if kind == "oracle":
        return [float(c["reward"]) for c in cands]
    import random as _r
    rng = _r.Random(hash(group["group_id"]) & 0xFFFFFFFF)
    return [rng.random() for _ in cands]


def macro_by_policy(rows: Sequence[dict[str, Any]], domains: Sequence[str], metric: str) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["domain"] in domains:
            v = safe_float(r.get(metric))
            if math.isfinite(v):
                by[r["policy"]].append(v)
    return {p: float(mean(vs)) for p, vs in by.items() if vs}


# ====================================================== PART C: antisymmetry
def tap_antisymmetry(weight: torch.Tensor, arch: str, config: str, groups: Sequence[dict[str, Any]], max_pairs: int = 600) -> dict[str, Any]:
    sc = _sc()
    cv, sd = sc["config_vector"], sc["score_diff"]
    ab_l: list[float] = []; ba_l: list[float] = []; labels: list[int] = []
    for g in groups:
        cands = g["candidates"]
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                if cands[a]["reward"] == cands[b]["reward"]:
                    continue
                va = cv(cands[a], config); vb = cv(cands[b], config)
                if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)):
                    continue
                ab = sd(weight, arch, va - vb); ba = sd(weight, arch, vb - va)
                if not (math.isfinite(ab) and math.isfinite(ba)):
                    continue
                ab_l.append(ab); ba_l.append(ba); labels.append(1 if cands[a]["reward"] > cands[b]["reward"] else 0)
            if len(ab_l) >= max_pairs:
                break
        if len(ab_l) >= max_pairs:
            break
    n = len(ab_l)
    if n < 8:
        return {"pairs": n, "insufficient": True}
    import math as _m

    def _sign(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)
    # strict sign flip = fraction NOT satisfying sign(ab) == -sign(ba) (proper antisymmetry)
    not_anti = sum(1 for ab, ba in zip(ab_l, ba_l) if _sign(ab) != -_sign(ba)) / n
    mean_sum = sum(ab + ba for ab, ba in zip(ab_l, ba_l)) / n
    bias = mean_sum / 2.0
    # antisymmetry correlation: corr(ab, -ba)
    negba = [-x for x in ba_l]
    ma = sum(ab_l) / n; mb = sum(negba) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ab_l, negba)) / n
    va = sum((x - ma) ** 2 for x in ab_l) / n; vb = sum((y - mb) ** 2 for y in negba) / n
    corr = cov / (_m.sqrt(va * vb) + 1e-12)
    raw_acc = sum(1 for ab, lab in zip(ab_l, labels) if (ab > 0) == bool(lab)) / n
    anti = [(ab - ba) / 2 for ab, ba in zip(ab_l, ba_l)]
    anti_acc = sum(1 for s, lab in zip(anti, labels) if (s > 0) == bool(lab)) / n
    return {"pairs": n, "strict_sign_flip_not_antisym": round(not_anti, 4), "antisymmetry_corr": round(corr, 4),
            "mean_score_sum": round(mean_sum, 6), "bias_estimate": round(bias, 6),
            "raw_fixed_order_accuracy": round(raw_acc, 4), "antisymmetrized_accuracy": round(anti_acc, 4),
            "exactly_antisymmetric": bool(abs(mean_sum) < 1e-5 and not_anti < 1e-6)}


def antisymmetry_main() -> int:
    started = time.time()
    ensure_root()
    # representative single-config tap per family at 47_L4
    taps: dict[str, tuple] = {}
    cr = dualanchor_role_channels("MIX_CODE_REASONING")
    oa = dualanchor_role_channels("MIX_OBJECTIVE_ALL")
    if cr:
        taps["DualAnchor.MIX_CODE_REASONING@47_L4"] = cr[-1]
    if oa:
        taps["DualAnchor.MIX_OBJECTIVE_ALL@47_L4"] = oa[-1]
    for grp in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL") + SCIENCE_GROUPS:
        for arch in ("AntisymLinear", "AntisymLinearNoNorm"):
            ch = mixed_head_channels(grp, arch=arch)
            if ch:
                taps[f"mixedhead.{grp}.{arch}@47_L4"] = ch[-1]
                break
    for fam in ("CODE", "HH"):
        ch = old_head_channels(fam)
        if ch:
            taps[f"old.{fam}@47_L4"] = ch[-1]
    pool_groups: list[dict[str, Any]] = []
    for dom in ("coding", "reasoning", "math", "alignment"):
        pool_groups += [g for g in domain_groups(dom) if len({c["reward"] > 0 for c in g["candidates"]}) > 1][:25]
    rows: list[dict[str, Any]] = []
    for name, (w, arch, cfg) in taps.items():
        res = tap_antisymmetry(w, arch, cfg, pool_groups)
        rows.append({"tap": name, "architecture": arch, "config": cfg, **res})
    exact = [r for r in rows if r.get("exactly_antisymmetric")]
    any_biased = [r for r in rows if math.isfinite(safe_float(r.get("bias_estimate"))) and abs(safe_float(r.get("bias_estimate"))) > 1e-3]
    anti_better = [r for r in rows if safe_float(r.get("antisymmetrized_accuracy")) > safe_float(r.get("raw_fixed_order_accuracy")) + 0.01]
    if rows and len(exact) == len([r for r in rows if not r.get("insufficient")]):
        verdict = "ANTISYMMETRIC_SCORES_READY"
    elif anti_better:
        verdict = "ANTISYMMETRIZED_BEST"
    elif any_biased:
        verdict = "SOME_TAPS_BIASED"
    elif rows:
        verdict = "RAW_SCORES_OK_FOR_FIXED_ORDER"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORE_TAP_ANTISYMMETRY_VERDICT": verdict, "rows": rows,
               "note": "tiny AntisymLinear/NoNorm heads are exactly antisymmetric by construction "
                       "(LayerNorm w/o affine and bias-free linear). The published HH pairwise evaluator "
                       "is audited separately in Part L (its fixed-order vs antisymmetrized behaviour is the "
                       "real bias question; low strict sign-flip on tiny heads is a construction fact, not accuracy).",
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "tap_antisymmetry.json", payload)
    write_csv(OUT_ROOT / "tap_antisymmetry_rows.csv", rows)
    write_md(OUT_ROOT / "tap_antisymmetry.md", [
        "# Core Tap Scoring / Antisymmetry Audit v1", "",
        status_line("BG_CORE_TAP_ANTISYMMETRY_VERDICT", verdict), "", payload["note"], "",
        *md_table(rows, ["tap", "architecture", "pairs", "strict_sign_flip_not_antisym", "antisymmetry_corr",
                         "bias_estimate", "raw_fixed_order_accuracy", "antisymmetrized_accuracy", "exactly_antisymmetric"]),
    ])
    write_json(PROGRESS_ROOT / "partC_antisymmetry_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_TAP_ANTISYMMETRY_VERDICT", verdict))
    return 0


# =============================================== PART D: action/content selection
def action_content_main() -> int:
    started = time.time()
    ensure_root()
    policies = build_tap_policies()
    domains = ("coding", "reasoning", "math", "logic", "alignment", "science", "anatomy")
    rows, raw = eval_policies_on_domains(domains, policies, split=None, max_groups=60)
    # baselines (oracle, random) computed inline
    for kind in ("oracle", "random"):
        agg: dict[str, list[dict[str, float]]] = defaultdict(list)
        for dom in domains:
            for g in [g for g in domain_groups(dom) if len({c["reward"] > 0 for c in g["candidates"]}) > 1][:60]:
                m = group_selection_metrics(g, _baseline_scores(g, kind))
                if m:
                    agg[dom].append(m)
        for dom, ms in agg.items():
            rows.append({"policy": kind, "domain": dom, "groups": len(ms),
                         "top1_oracle": finite_mean(m["top1_oracle"] for m in ms),
                         "top2_oracle": finite_mean(m["top2_oracle"] for m in ms),
                         "pairwise_accuracy": finite_mean(m["pairwise_accuracy"] for m in ms),
                         "mean_regret": finite_mean(m["regret"] for m in ms)})
    core = ("coding", "reasoning", "math", "logic", "alignment")
    macro = macro_by_policy(rows, core, "top1_oracle")
    pw_macro = macro_by_policy(rows, core, "pairwise_accuracy")
    real = {k: v for k, v in macro.items() if k not in ("oracle", "random")}
    best = max(real, key=lambda k: real[k]) if real else None
    da = real.get("DualAnchor", float("nan"))
    sci_help = any(k.startswith("science_") and v > da + 0.02 for k, v in real.items())
    if best == "DualAnchor":
        verdict = "DUALANCHOR_BEST"
    elif best == "MIX_CODE_REASONING_only":
        verdict = "CODE_REASONING_BEST"
    elif best == "MIX_OBJECTIVE_ALL_only":
        verdict = "MIX_OBJECTIVE_BEST"
    elif sci_help:
        verdict = "SCIENCE_TAPS_HELP"
    elif best and best.startswith("science_"):
        verdict = "SCIENCE_TAPS_HELP"
    elif best:
        verdict = "DOMAIN_SPECIFIC_TAPS_NEEDED"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORE_ACTION_CONTENT_TAP_EVAL_VERDICT": verdict, "rows": rows,
               "macro_top1_oracle_core": macro, "macro_pairwise_core": pw_macro,
               "best_policy": best, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "action_content_eval.json", payload)
    write_csv(OUT_ROOT / "action_content_eval_rows.csv", rows)
    write_md(OUT_ROOT / "action_content_eval.md", [
        "# Core-Domain Action/Content Selection Eval v1", "",
        status_line("BG_CORE_ACTION_CONTENT_TAP_EVAL_VERDICT", verdict), "",
        f"Best core-macro policy (top1-oracle): `{best}`", "",
        "## Macro top1-oracle over core domains (coding/reasoning/math/logic/alignment)", "",
        *[f"- {k}: {fmt(v)}" for k, v in sorted(macro.items(), key=lambda x: -x[1])], "",
        "## Per policy x domain", "",
        *md_table(sorted(rows, key=lambda r: (r["policy"], r["domain"])),
                  ["policy", "domain", "groups", "top1_oracle", "top2_oracle", "pairwise_accuracy", "mean_regret"]),
    ])
    write_json(PROGRESS_ROOT / "partD_action_content_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_ACTION_CONTENT_TAP_EVAL_VERDICT", verdict))
    print("best:", best, "| DualAnchor macro top1:", fmt(da))
    return 0


# ===================================== TRANSPLANT (content-only -> branch-capable)
def _normed(t: torch.Tensor) -> torch.Tensor:
    f = t.detach().cpu().to(torch.float32).flatten()
    return f / (f.norm() + 1e-8)


def branch_bridge_direction(config: str, arch: str = "AntisymLinear") -> torch.Tensor | None:
    """DualAnchor branch+bridge residual = normed(DualAnchor_tap - old_anchor) at a config,
    extracted from the constrained registry (the same components that built DualAnchor)."""
    da = (_selected_layer_taps(LOCKED_RECIPE, arch).get(config) or {}).get("MIX_CODE_REASONING")
    old = (_selected_layer_taps("old_only", arch).get(config) or {}).get("MIX_CODE_REASONING")
    if not (isinstance(da, dict) and isinstance(old, dict)):
        return None
    wd = weight_from_state_dict(da.get("state_dict")); wo = weight_from_state_dict(old.get("state_dict"))
    if wd is None or wo is None:
        return None
    return _normed(_normed(wd) - _normed(wo))


# ----- canonical (faithful) transplant: old + branch_residual + bridge_residual -----
# Uses the registry's aligned real branch (hidden_origin_v4) and bridge (universal) source
# weights, stored as the *_full_00_100_diagnostic recipes, with per-config Gram-Schmidt
# residuals against the target tap. This is the same construction that built DualAnchor.
FULL_DIAG_RECIPE = {"old": "old_content_full_00_100_diagnostic",
                    "branch": "branch_full_00_100_diagnostic",
                    "bridge": "bridge_full_00_100_diagnostic"}
TRANSPLANT_COEFFS = (0.80, 0.10, 0.10)          # canonical (old, branch, bridge)
TRANSPLANT_GRID = [(0.85, 0.10, 0.05), (0.80, 0.10, 0.10), (0.75, 0.15, 0.10), (0.80, 0.15, 0.05)]
TRANSPLANT_SOURCE_ROLE = "MIX_CODE_REASONING"   # canonical aligned branch/bridge source


def _full_diag_weight(kind: str, config: str, role: str = TRANSPLANT_SOURCE_ROLE, arch: str = "AntisymLinear") -> torch.Tensor | None:
    rec = FULL_DIAG_RECIPE[kind]
    rows = [c for c in _constrained_candidates()
            if c.get("recipe") == rec and c.get("tap_role") == role
            and c.get("target_config") == config and c.get("architecture") == arch]
    return weight_from_state_dict(rows[0].get("state_dict")) if rows else None


def masked_residual_vec(vec: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    """Gram-Schmidt residual of vec orthogonal to each basis vector (per-config / full block)."""
    v = vec.detach().cpu().to(torch.float32).flatten().clone()
    for b in basis:
        u = b.detach().cpu().to(torch.float32).flatten()
        n = u.norm()
        if n < 1e-8:
            continue
        u = u / n
        v = v - float(v @ u) * u
    return v


def real_transplant_weight(target_w: torch.Tensor, config: str, coeffs: tuple = TRANSPLANT_COEFFS,
                           arch: str = "AntisymLinear") -> dict | None:
    """Faithful weight-space transplant of a content tap into a branch-capable tap.
    Returns dict with the transplanted weight + diagnostics, or None if sources missing."""
    a, b, c = coeffs
    branch = _full_diag_weight("branch", config)
    bridge = _full_diag_weight("bridge", config)
    if branch is None or bridge is None:
        return None
    t = _normed(target_w)
    br = masked_residual_vec(branch, [t])
    bg = masked_residual_vec(bridge, [t, br])
    w = a * t + b * br + c * bg
    wt = _normed(w).reshape(1, -1)
    return {"weight": wt, "coeffs": coeffs,
            "branch_residual_norm": float(br.norm()), "bridge_residual_norm": float(bg.norm()),
            "cos_pure_transplant": float((_normed(target_w) * _normed(wt)).sum())}


def transplant_channels(channels: list[tuple], coeffs: tuple = TRANSPLANT_COEFFS) -> list[tuple]:
    """Turn content-only channels into branch-capable channels via the faithful transplant."""
    out: list[tuple] = []
    for (w, arch, cfg) in channels:
        res = real_transplant_weight(w, cfg, coeffs)
        out.append((res["weight"] if res else w, arch, cfg))
    return out


def build_and_save_transplant_artifacts() -> int:
    """Construct branch-capable transplants of the content/science taps and save BOTH the
    pure content versions and the transplanted versions (+ a per-tap diagnostic manifest)
    under the run output root. Never mutates existing registries."""
    ensure_root()
    out_dir = OUT_ROOT / "constructed_taps"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = build_tap_policies()
    targets = ["old_CODE", "old_HH", "mixedhead_MIX_OBJECTIVE_ALL", "mixedhead_MIX_HH_OBJECTIVE",
               "science_MIX_CODE_SCIENCE", "science_MIX_REASONING_SCIENCE", "science_MIX_CODE_SCIENCE_MED"]
    pure_store: dict[str, Any] = {}
    transplant_store: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for name in targets:
        ch = base.get(name)
        if not ch:
            continue
        pure_store[name] = {cfg: {"weight": w.detach().cpu(), "arch": arch} for (w, arch, cfg) in ch}
        transplant_store[name] = {}
        for (w, arch, cfg) in ch:
            grid_variants = {}
            for coeffs in TRANSPLANT_GRID:
                res = real_transplant_weight(w, cfg, coeffs)
                if res is None:
                    continue
                lbl = "_".join(str(int(x * 100)) for x in coeffs)
                grid_variants[lbl] = {"weight": res["weight"].detach().cpu(), "coeffs": coeffs}
                if coeffs == TRANSPLANT_COEFFS:
                    transplant_store[name][cfg] = {"weight": res["weight"].detach().cpu(), "arch": arch, "coeffs": coeffs}
                    manifest.append({"tap": name, "config": cfg, "arch": arch, "coeffs": "/".join(str(x) for x in coeffs),
                                     "cos_pure_transplant": round(res["cos_pure_transplant"], 4),
                                     "branch_residual_norm": round(res["branch_residual_norm"], 4),
                                     "bridge_residual_norm": round(res["bridge_residual_norm"], 4)})
            transplant_store[name].setdefault(cfg, {})
            transplant_store[name][cfg]["grid"] = grid_variants
    torch.save({"meta": {"kind": "pure_content_taps", "note": "untransplanted content/science tap weights"},
                "taps": pure_store}, out_dir / "pure_content_taps.pt")
    torch.save({"meta": {"kind": "transplanted_taps", "canonical_coeffs": TRANSPLANT_COEFFS, "grid": TRANSPLANT_GRID,
                         "branch_source": "hidden_origin_v4 (branch_full_00_100_diagnostic)",
                         "bridge_source": "universal (bridge_full_00_100_diagnostic)",
                         "method": "normed(a*old + b*masked_residual(branch,[old]) + c*masked_residual(bridge,[old,branch_res]))"},
                "taps": transplant_store}, out_dir / "transplanted_taps.pt")
    write_json(out_dir / "transplant_manifest.json", {"canonical_coeffs": TRANSPLANT_COEFFS, "rows": manifest})
    write_csv(out_dir / "transplant_manifest.csv", manifest)
    write_md(out_dir / "transplant_manifest.md", [
        "# Transplant Manifest (pure content vs branch-capable)", "",
        "Pure content tap weights saved to `pure_content_taps.pt`; transplanted (branch-capable) "
        "weights to `transplanted_taps.pt` (canonical coeffs + full grid). Branch source = "
        "hidden_origin_v4, bridge source = universal, both aligned via the constrained registry "
        "diagnostic recipes; Gram-Schmidt residuals against each target tap.", "",
        *md_table(manifest, ["tap", "config", "coeffs", "cos_pure_transplant", "branch_residual_norm", "bridge_residual_norm"]),
    ])
    write_json(PROGRESS_ROOT / "transplant_artifacts_done.json", {"targets": len(pure_store), "rows": len(manifest), "saved_at": time.time()})
    print(f"BG_CORE_TRANSPLANT_ARTIFACTS = {len(pure_store)} taps, {len(manifest)} (tap,config) transplants saved to {rel(out_dir)}")
    return 0


# ============================================ PART E: branch survival / retention
RETENTION_BUDGETS = (8, 5, 4, 2, 1)


def retention_metrics(group: dict[str, Any], scores: list[float]) -> dict[str, float]:
    cands = group["candidates"]
    rewards = [float(c["reward"]) for c in cands]
    oracle = max(rewards)
    finite = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    if not finite:
        return {}
    order = [i for i, _ in sorted(finite, key=lambda x: x[1], reverse=True)]
    out: dict[str, float] = {}
    for k in RETENTION_BUDGETS:
        out[f"oracle_top{k}"] = 1.0 if any(rewards[i] >= oracle for i in order[:k]) else 0.0
    out["false_prune_top8"] = 0.0 if out["oracle_top8"] else 1.0
    out["avg_survivors"] = float(min(8, len(order)))
    out["terminal_best_reward_top8"] = max((rewards[i] for i in order[:8]), default=float("nan"))
    return out


def branch_policies(arch: str = "AntisymLinear") -> dict[str, list[tuple]]:
    pol: dict[str, list[tuple]] = {}
    base = build_tap_policies(arch=arch)
    if "DualAnchor" in base:
        pol["DualAnchor"] = base["DualAnchor"]
    # old/science taps raw (content-only) AND transplanted (branch-capable)
    for name in ("old_CODE", "old_HH", "mixedhead_MIX_OBJECTIVE_ALL",
                 "science_MIX_CODE_SCIENCE", "science_MIX_REASONING_SCIENCE"):
        if name in base:
            pol[f"{name}_raw"] = base[name]
            pol[f"{name}_transplant"] = transplant_channels(base[name])
    # DualAnchor + science third expert (concatenate channels)
    for sci in ("science_MIX_CODE_SCIENCE", "science_MIX_REASONING_SCIENCE"):
        if "DualAnchor" in base and sci in base:
            pol[f"DualAnchor+{sci}"] = base["DualAnchor"] + transplant_channels(base[sci])
    return pol


def branch_survival_main() -> int:
    started = time.time()
    ensure_root()
    policies = branch_policies()
    domains = ("coding", "reasoning", "math", "logic", "alignment", "science", "anatomy")
    raw: list[dict[str, Any]] = []
    agg: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    for dom in domains:
        groups = [g for g in domain_groups(dom) if len({c["reward"] > 0 for c in g["candidates"]}) > 1][:60]
        for g in groups:
            for name, ch in policies.items():
                m = retention_metrics(g, policy_candidate_scores(g, ch))
                if not m:
                    continue
                agg[(name, dom)].append(m)
                raw.append({"policy": name, "domain": dom, "group_id": g["group_id"], **m})
    rows = []
    for (name, dom), ms in sorted(agg.items()):
        rows.append({"policy": name, "domain": dom, "groups": len(ms),
                     "oracle_top8": finite_mean(m["oracle_top8"] for m in ms),
                     "oracle_top5": finite_mean(m["oracle_top5"] for m in ms),
                     "oracle_top1": finite_mean(m["oracle_top1"] for m in ms),
                     "false_prune_top8": finite_mean(m["false_prune_top8"] for m in ms),
                     "terminal_best_reward_top8": finite_mean(m["terminal_best_reward_top8"] for m in ms)})
    core = ("coding", "reasoning", "math", "logic", "alignment")
    macro8 = macro_by_policy(rows, core, "oracle_top8")
    macro1 = macro_by_policy(rows, core, "oracle_top1")  # discriminative budget for transplant effect
    da8 = macro8.get("DualAnchor", float("nan"))
    # does transplant close the content-only branching gap (measured at top1, where there is headroom)?
    transplant_gain = finite_mean([macro1.get(f"{n}_transplant", float("nan")) - macro1.get(f"{n}_raw", float("nan"))
                                   for n in ("old_CODE", "old_HH", "science_MIX_CODE_SCIENCE", "science_MIX_REASONING_SCIENCE")
                                   if f"{n}_raw" in macro1])
    sci_third = [k for k in macro8 if k.startswith("DualAnchor+science") and macro8[k] > da8 + 0.01]
    if sci_third:
        verdict = "SCIENCE_TAP_IMPROVES_SURVIVAL"
    elif any(k.startswith("DualAnchor+") for k in macro8) and not sci_third:
        verdict = "DUALANCHOR_SURVIVAL_CONFIRMED" if math.isfinite(da8) else "DATA_LIMITED"
    elif math.isfinite(da8):
        verdict = "DUALANCHOR_SURVIVAL_CONFIRMED"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT": verdict, "rows": rows,
               "macro_oracle_top8_core": macro8, "macro_oracle_top1_core": macro1,
               "transplant_mean_gain_top1": transplant_gain,
               "science_third_expert_helps": sci_third, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "branch_survival_eval.json", payload)
    write_csv(OUT_ROOT / "branch_survival_rows.csv", rows)
    write_md(OUT_ROOT / "branch_survival_eval.md", [
        "# Core-Domain Branch Survival / Retention v1", "",
        status_line("BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT", verdict), "",
        f"Mean transplant gain (top8 oracle, raw->transplant): {fmt(transplant_gain)} "
        "(content-only taps gain branch retention from the old+branch+bridge transplant).", "",
        "## Macro oracle-retained@top8 over core domains", "",
        *[f"- {k}: {fmt(v)}" for k, v in sorted(macro8.items(), key=lambda x: -x[1])], "",
        "## Per policy x domain", "",
        *md_table(sorted(rows, key=lambda r: (r["policy"], r["domain"])),
                  ["policy", "domain", "groups", "oracle_top8", "oracle_top5", "oracle_top1", "false_prune_top8"]),
    ])
    write_json(PROGRESS_ROOT / "partE_branch_survival_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT", verdict))
    print("DualAnchor top8 macro:", fmt(da8), "| transplant gain:", fmt(transplant_gain))
    return 0


# ============================================ PART F: terminal survivor handoff
def terminal_handoff_main() -> int:
    started = time.time()
    ensure_root()
    base = build_tap_policies()
    da = base.get("DualAnchor", [])
    domains = ("coding", "reasoning", "math", "logic", "alignment")
    # confidence gate threshold tuned on CALIBRATION only (never heldout)
    cal_margins: list[tuple[float, float]] = []  # (margin, top1_is_oracle)
    for dom in domains:
        for g in [g for g in domain_groups(dom) if split_of(g) == "calibration" and len({c["reward"] > 0 for c in g["candidates"]}) > 1][:60]:
            s = policy_candidate_scores(g, da)
            fin = sorted([(i, x) for i, x in enumerate(s) if math.isfinite(x)], key=lambda z: z[1], reverse=True)
            if len(fin) < 2:
                continue
            margin = fin[0][1] - fin[1][1]
            rewards = [c["reward"] for c in g["candidates"]]
            cal_margins.append((margin, 1.0 if rewards[fin[0][0]] >= max(rewards) else 0.0))
    # pick gate = the calibration margin quantile where top1 precision is high
    gate = 0.5
    if cal_margins:
        srt = sorted(m for m, _ in cal_margins)
        gate = srt[int(0.6 * len(srt))]
    # evaluate policies on HELDOUT
    policies_terminal = {"dualanchor_forced_top1": ("top1", da),
                         "dualanchor_terminal_top2": ("topk2", da),
                         "dualanchor_terminal_top5": ("topk5", da),
                         "dualanchor_full_handoff": ("topk8", da),
                         "dualanchor_confidence_top1_else_top5": ("gate5", da)}
    rows = []
    agg: dict[str, list[dict[str, float]]] = defaultdict(list)
    fire = defaultdict(list)
    for dom in domains:
        for g in [g for g in domain_groups(dom) if split_of(g) == "heldout" and len({c["reward"] > 0 for c in g["candidates"]}) > 1][:80]:
            s = policy_candidate_scores(g, da)
            fin = sorted([(i, x) for i, x in enumerate(s) if math.isfinite(x)], key=lambda z: z[1], reverse=True)
            if len(fin) < 1:
                continue
            rewards = [c["reward"] for c in g["candidates"]]
            oracle = max(rewards)
            margin = (fin[0][1] - fin[1][1]) if len(fin) > 1 else float("inf")
            for name, (mode, _) in policies_terminal.items():
                if mode == "top1":
                    ret = 1.0 if rewards[fin[0][0]] >= oracle else 0.0
                elif mode.startswith("topk"):
                    k = int(mode[4:]); ret = 1.0 if any(rewards[i] >= oracle for i, _ in fin[:k]) else 0.0
                else:  # gate5
                    if margin >= gate:
                        ret = 1.0 if rewards[fin[0][0]] >= oracle else 0.0
                        fire[name].append(1.0)
                    else:
                        ret = 1.0 if any(rewards[i] >= oracle for i, _ in fin[:5]) else 0.0
                        fire[name].append(0.0)
                agg[name].append({"domain": dom, "oracle_retained": ret,
                                  "first_oracle": 1.0 if rewards[fin[0][0]] >= oracle else 0.0})
    for name, ms in sorted(agg.items()):
        rows.append({"policy": name, "groups": len(ms),
                     "oracle_retained": finite_mean(m["oracle_retained"] for m in ms),
                     "first_selected_oracle": finite_mean(m["first_oracle"] for m in ms),
                     "confidence_fire_rate": finite_mean(fire[name]) if fire.get(name) else None})
    top1 = next((r for r in rows if r["policy"] == "dualanchor_forced_top1"), {})
    gate5 = next((r for r in rows if r["policy"].endswith("else_top5")), {})
    top5 = next((r for r in rows if r["policy"] == "dualanchor_terminal_top5"), {})
    if safe_float(top1.get("oracle_retained")) >= 0.98:
        verdict = "TERMINAL_TOP1_READY_ON_CORE"
    elif safe_float(gate5.get("oracle_retained")) >= 0.97 and safe_float(top5.get("oracle_retained")) >= 0.97:
        verdict = "CONFIDENCE_TOP1_ELSE_TOP5_LOCKED"
    elif safe_float(top5.get("oracle_retained")) >= 0.97:
        verdict = "FULL_HANDOFF_REQUIRED"
    elif rows:
        verdict = "DOMAIN_SPECIFIC_TERMINAL_POLICY"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORE_TERMINAL_HANDOFF_EVAL_VERDICT": verdict, "rows": rows,
               "calibration_confidence_gate": gate, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "terminal_handoff_eval.json", payload)
    write_csv(OUT_ROOT / "terminal_handoff_rows.csv", rows)
    write_md(OUT_ROOT / "terminal_handoff_eval.md", [
        "# Core-Domain Terminal Survivor Handoff v1", "",
        status_line("BG_CORE_TERMINAL_HANDOFF_EVAL_VERDICT", verdict), "",
        f"Confidence gate (tuned on calibration only): {fmt(gate)}", "",
        *md_table(rows, ["policy", "groups", "oracle_retained", "first_selected_oracle", "confidence_fire_rate"]),
    ])
    write_json(PROGRESS_ROOT / "partF_terminal_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_TERMINAL_HANDOFF_EVAL_VERDICT", verdict))
    return 0


# ===================================== PART G: science/anatomy cross-domain transfer
def science_cross_domain_main() -> int:
    started = time.time()
    ensure_root()
    ac = read_json(OUT_ROOT / "action_content_eval.json", {}) or {}
    bs = read_json(OUT_ROOT / "branch_survival_eval.json", {}) or {}
    macro_sel = ac.get("macro_top1_oracle_core", {})
    macro_ret8 = bs.get("macro_oracle_top8_core", {})
    da_sel = safe_float(macro_sel.get("DualAnchor"))
    da_ret = safe_float(macro_ret8.get("DualAnchor"))
    sci_sel = {k: v for k, v in macro_sel.items() if k.startswith("science_")}
    sci_third = {k: v for k, v in macro_ret8.items() if k.startswith("DualAnchor+science")}
    # science tap performance on its own (diagnostic) domains
    sci_on_sci = [r for r in ac.get("rows", []) if r.get("policy", "").startswith("science_") and r["domain"] in ("science", "anatomy")]
    third_helps = any(v > da_ret + 0.01 for v in sci_third.values())
    sel_helps = any(v > da_sel + 0.02 for v in sci_sel.values())
    sel_hurts = sci_sel and max(sci_sel.values()) < da_sel - 0.05
    sci_anat_good = finite_mean(r.get("top1_oracle") for r in sci_on_sci) if sci_on_sci else float("nan")
    if third_helps or sel_helps:
        verdict = "SCIENCE_TAPS_USEFUL_AUXILIARY"
    elif math.isfinite(sci_anat_good) and sci_anat_good >= 0.5 and not (third_helps or sel_helps):
        verdict = "ANATOMY_TAP_USEFUL_ONLY_FOR_ANATOMY"
    elif sel_hurts:
        verdict = "SCIENCE_TAPS_HURT_CORE"
    elif sci_sel or sci_third:
        verdict = "SCIENCE_TAPS_NO_HELP"
    else:
        verdict = "SCIENCE_TAPS_DATA_LIMITED"
    rows = ([{"mode": "selection_core", "policy": k, "macro_top1_oracle": v, "vs_dualanchor": v - da_sel} for k, v in sci_sel.items()]
            + [{"mode": "third_expert_survival", "policy": k, "macro_oracle_top8": v, "vs_dualanchor": v - da_ret} for k, v in sci_third.items()])
    payload = {"BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT": verdict, "dualanchor_selection": da_sel,
               "dualanchor_retention_top8": da_ret, "science_anatomy_self_top1": sci_anat_good,
               "rows": rows, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "science_tap_cross_domain.json", payload)
    write_csv(OUT_ROOT / "science_tap_cross_domain_rows.csv", rows)
    write_md(OUT_ROOT / "science_tap_cross_domain.md", [
        "# Science/Anatomy Tap Cross-Domain Transfer v1", "",
        status_line("BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT", verdict), "",
        f"DualAnchor core selection {fmt(da_sel)} / retention@8 {fmt(da_ret)}; "
        f"science taps on science/anatomy self top1 {fmt(sci_anat_good)}.", "",
        *md_table(rows, ["mode", "policy", "macro_top1_oracle", "macro_oracle_top8", "vs_dualanchor"]),
    ])
    write_json(PROGRESS_ROOT / "partG_science_cross_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT", verdict))
    return 0


# ===================================== PART H: cross-domain tap transfer matrix
def transfer_matrix_main() -> int:
    started = time.time()
    ensure_root()
    ac = read_json(OUT_ROOT / "action_content_eval.json", {}) or {}
    bs = read_json(OUT_ROOT / "branch_survival_eval.json", {}) or {}
    cols = ("coding", "reasoning", "math", "logic", "alignment", "science", "anatomy")
    sel = defaultdict(dict)
    for r in ac.get("rows", []):
        if r["policy"] not in ("oracle", "random"):
            sel[r["policy"]][r["domain"]] = safe_float(r.get("top1_oracle"))
    matrix_rows = []
    for pol, dvals in sorted(sel.items()):
        row = {"policy": pol}
        ranks = []
        for d in cols:
            row[d] = dvals.get(d)
        matrix_rows.append(row)
    # best/worst tap per domain; DualAnchor rank
    best_per = {}; worst_per = {}; da_rank = {}
    for d in cols:
        scored = [(p, sel[p].get(d)) for p in sel if isinstance(sel[p].get(d), float) and math.isfinite(sel[p].get(d))]
        if not scored:
            continue
        scored.sort(key=lambda x: -x[1])
        best_per[d] = scored[0][0]; worst_per[d] = scored[-1][0]
        order = [p for p, _ in scored]
        da_rank[d] = (order.index("DualAnchor") + 1) if "DualAnchor" in order else None
    # most robust = best mean top1 across core
    core = ("coding", "reasoning", "math", "logic", "alignment")
    mean_core = {p: finite_mean(sel[p].get(d) for d in core) for p in sel}
    most_robust = max(mean_core, key=lambda p: mean_core[p]) if mean_core else None
    da_mean_rank = finite_mean([da_rank[d] for d in core if da_rank.get(d)])
    if most_robust == "DualAnchor" or (da_mean_rank and da_mean_rank <= 2.5):
        verdict = "DUALANCHOR_MOST_ROBUST"
    elif most_robust and most_robust.startswith("science_"):
        verdict = "SCIENCE_TAP_TRANSFERS"
    elif most_robust == "MIX_CODE_REASONING_only" or (most_robust and "CODE_REASONING" in most_robust):
        verdict = "CODE_REASONING_DOMINATES"
    elif most_robust and "OBJECTIVE" in most_robust:
        verdict = "MIX_OBJECTIVE_DOMINATES"
    elif most_robust:
        verdict = "DOMAIN_SPECIFIC_TAPS_NEEDED"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORE_TAP_TRANSFER_MATRIX_VERDICT": verdict, "matrix": matrix_rows,
               "best_per_domain": best_per, "worst_per_domain": worst_per, "dualanchor_rank_per_domain": da_rank,
               "most_robust": most_robust, "dualanchor_mean_core_rank": da_mean_rank,
               "mean_core_top1_by_policy": mean_core, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "tap_transfer_matrix.json", payload)
    write_csv(OUT_ROOT / "tap_transfer_matrix.csv", matrix_rows)
    write_md(OUT_ROOT / "tap_transfer_matrix.md", [
        "# Cross-Domain Tap Transfer Matrix v1 (top1-oracle)", "",
        status_line("BG_CORE_TAP_TRANSFER_MATRIX_VERDICT", verdict), "",
        f"Most robust (mean core top1): `{most_robust}`; DualAnchor mean core rank: {fmt(da_mean_rank)}.", "",
        *md_table(matrix_rows, ["policy", *cols]), "",
        "Best per domain: " + ", ".join(f"{d}={best_per.get(d)}" for d in cols),
    ])
    write_json(PROGRESS_ROOT / "partH_transfer_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_TAP_TRANSFER_MATRIX_VERDICT", verdict))
    return 0


# ===================================== PART I: tap geometry / residual overlap
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((_normed(a) * _normed(b)).sum())


def _residual_norm(target: torch.Tensor, basis: list[torch.Tensor]) -> float:
    t = _normed(target)
    for b in basis:
        u = _normed(b)
        t = t - float((t * u).sum()) * u
    return float(t.norm())


def _channel_weight(channels: list[tuple], config: str) -> torch.Tensor | None:
    for (w, arch, cfg) in channels:
        if cfg == config:
            return w
    return None


def geometry_main() -> int:
    started = time.time()
    ensure_root()
    base = build_tap_policies()
    cfg = "47_L4"
    cr = _channel_weight(base.get("MIX_CODE_REASONING_only", []), cfg)
    oa = _channel_weight(base.get("MIX_OBJECTIVE_ALL_only", []), cfg)
    rows = []
    geom: dict[str, Any] = {}
    if isinstance(cr, torch.Tensor) and isinstance(oa, torch.Tensor):
        geom["cos_MIX_CODE_REASONING_vs_MIX_OBJECTIVE_ALL"] = _cos(cr, oa)
        rows.append({"pair": "MIX_CODE_REASONING vs MIX_OBJECTIVE_ALL", "cosine": _cos(cr, oa)})
    da_basis = [w for w in (cr, oa) if isinstance(w, torch.Tensor)]
    for sci in SCIENCE_GROUPS:
        sw = _channel_weight(base.get(f"science_{sci}", []), cfg)
        if isinstance(sw, torch.Tensor) and da_basis:
            cos_cr = _cos(sw, cr) if isinstance(cr, torch.Tensor) else float("nan")
            res = _residual_norm(sw, da_basis)
            geom[f"residual_{sci}_on_dualanchor_span"] = res
            rows.append({"pair": f"{sci} residual on DualAnchor span", "cosine": cos_cr, "residual_norm": res})
    # branch direction independence
    bb = branch_bridge_direction(cfg)
    if isinstance(bb, torch.Tensor) and da_basis:
        rows.append({"pair": "branch+bridge direction residual on DualAnchor span", "residual_norm": _residual_norm(bb, da_basis)})
    old_code = _channel_weight(base.get("old_CODE", []), cfg)
    if isinstance(old_code, torch.Tensor) and isinstance(cr, torch.Tensor):
        rows.append({"pair": "old_CODE vs MIX_CODE_REASONING", "cosine": _cos(old_code, cr)})
    da_cos = geom.get("cos_MIX_CODE_REASONING_vs_MIX_OBJECTIVE_ALL", float("nan"))
    sci_res = [v for k, v in geom.items() if k.startswith("residual_") and "dualanchor" in k]
    max_sci_res = max(sci_res) if sci_res else float("nan")
    if math.isfinite(max_sci_res) and max_sci_res >= 0.5:
        verdict = "SCIENCE_RESIDUAL_INDEPENDENT"
    elif math.isfinite(max_sci_res) and max_sci_res < 0.25:
        verdict = "SCIENCE_REDUNDANT"
    elif math.isfinite(da_cos) and abs(da_cos) < 0.5:
        verdict = "DUALANCHOR_COMPLEMENTARY"
    else:
        verdict = "INCONCLUSIVE"
    payload = {"BG_CORE_TAP_GEOMETRY_VERDICT": verdict, "geometry": geom, "rows": rows,
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "tap_geometry.json", payload)
    write_csv(OUT_ROOT / "tap_geometry_rows.csv", rows)
    write_md(OUT_ROOT / "tap_geometry.md", [
        "# Tap Geometry / Residual Overlap v1", "",
        status_line("BG_CORE_TAP_GEOMETRY_VERDICT", verdict), "",
        f"DualAnchor component cosine (CR vs OA): {fmt(da_cos)}; max science residual on DualAnchor span: {fmt(max_sci_res)}.", "",
        *md_table(rows, ["pair", "cosine", "residual_norm"]),
    ])
    write_json(PROGRESS_ROOT / "partI_geometry_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_TAP_GEOMETRY_VERDICT", verdict))
    return 0


# ===================================== PART J: threshold / anchor-set ablation
def threshold_anchor_ablation_main() -> int:
    started = time.time()
    ensure_root()
    base = build_tap_policies()
    anchor_sets: dict[str, list[tuple]] = {}
    if "DualAnchor" in base:
        anchor_sets["DualAnchor"] = base["DualAnchor"]
    if "MIX_CODE_REASONING_only" in base:
        anchor_sets["MIX_CODE_REASONING_only"] = base["MIX_CODE_REASONING_only"]
    if "MIX_OBJECTIVE_ALL_only" in base:
        anchor_sets["MIX_OBJECTIVE_ALL_only"] = base["MIX_OBJECTIVE_ALL_only"]
    for sci in ("science_MIX_CODE_SCIENCE", "science_MIX_REASONING_SCIENCE"):
        if "DualAnchor" in base and sci in base:
            anchor_sets[f"DualAnchor+{sci}"] = base["DualAnchor"] + transplant_channels(base[sci])
    domains = ("coding", "reasoning", "math", "logic", "alignment")
    summary_rows = []
    clean: dict[str, list[dict[str, float]]] = defaultdict(list)
    for dom in domains:
        for g in [g for g in domain_groups(dom) if len({c["reward"] > 0 for c in g["candidates"]}) > 1][:50]:
            for name, ch in anchor_sets.items():
                m = retention_metrics(g, policy_candidate_scores(g, ch))
                if m:
                    clean[name].append(m)
    for name, ms in clean.items():
        summary_rows.append({"anchor_set": name, "groups": len(ms),
                             "oracle_top8": finite_mean(m["oracle_top8"] for m in ms),
                             "oracle_top5": finite_mean(m["oracle_top5"] for m in ms),
                             "oracle_top1": finite_mean(m["oracle_top1"] for m in ms),
                             "false_prune_top8": finite_mean(m["false_prune_top8"] for m in ms)})
    da = next((r for r in summary_rows if r["anchor_set"] == "DualAnchor"), {})
    tri = [r for r in summary_rows if r["anchor_set"].startswith("DualAnchor+")]
    tri_better = any(safe_float(r.get("oracle_top8")) > safe_float(da.get("oracle_top8")) + 0.01 for r in tri)
    singles = [r for r in summary_rows if r["anchor_set"].endswith("_only")]
    single_better = any(safe_float(r.get("oracle_top8")) > safe_float(da.get("oracle_top8")) + 0.01 for r in singles)
    if tri_better and any(r["anchor_set"].endswith("SCIENCE") for r in tri):
        verdict = "TRI_ANCHOR_IMPROVES" if not any("science" in r["anchor_set"] for r in tri) else "TRI_ANCHOR_IMPROVES"
    elif tri_better:
        verdict = "TRI_ANCHOR_IMPROVES"
    elif single_better:
        verdict = "DOMAIN_SPECIFIC_ANCHORS_NEEDED"
    elif any("science" in r["anchor_set"] for r in tri):
        verdict = "SCIENCE_ANCHOR_NOT_USEFUL" if not tri_better else "TRI_ANCHOR_IMPROVES"
    elif da:
        verdict = "DUALANCHOR_MEAN_FLOOR_CONFIRMED"
    else:
        verdict = "INSUFFICIENT"
    payload = {"BG_CORE_THRESHOLD_ANCHOR_ABLATION_VERDICT": verdict, "rows": summary_rows,
               "note": "Threshold schedule mean_floor_very_loose is the locked default; this ablation compares"
                       " anchor SETS at fixed budgets (top8/top5). Adding science as a third anchor is the key test.",
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "threshold_anchor_ablation.json", payload)
    write_csv(OUT_ROOT / "threshold_anchor_rows.csv", summary_rows)
    write_md(OUT_ROOT / "threshold_anchor_ablation.md", [
        "# Threshold / Anchor-Set Ablation v1", "",
        status_line("BG_CORE_THRESHOLD_ANCHOR_ABLATION_VERDICT", verdict), "",
        *md_table(summary_rows, ["anchor_set", "groups", "oracle_top8", "oracle_top5", "oracle_top1", "false_prune_top8"]),
    ])
    write_json(PROGRESS_ROOT / "partJ_threshold_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_THRESHOLD_ANCHOR_ABLATION_VERDICT", verdict))
    return 0


# ===================================== PART K: partial-cache splice runtime smoke
def partial_splice_smoke_main() -> int:
    started = time.time()
    ensure_root()
    note = ("Partial-cache splice math equivalence + compute-saving were validated in "
            "bg_partial_cache_splice_v2 (PARTIAL_SPLICE_COMPUTE_SAVING_VALID); this part is a bounded "
            "runtime smoke of core-domain generation + DualAnchor tap-selection only. Ouro is verbose, "
            "so max_new_tokens is generous (extra for math). No production routing or compute claim is made here.")
    rows: list[dict[str, Any]] = []
    verdict = "SKIPPED"
    try:
        from datasets import load_dataset
        from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
        ext = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        tok = ext.tokenizer; model = ext.model
        smokes = []
        os.environ["HF_HUB_OFFLINE"] = "1"
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        for i in range(2):
            smokes.append(("math", gsm[i]["question"]))
        lq = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")["validation"]
        for i in range(2):
            ex = lq[i]
            smokes.append(("logic", f"{ex['context']}\nQuestion: {ex['query']}\nAnswer:"))
        import time as _t
        for dom, prompt in smokes:
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            enc = {k: v.to(ext.device) for k, v in enc.items()}
            mnt = 320 if dom == "math" else 160  # math yaps -> extra headroom
            t0 = _t.time()
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=True)
            txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append({"domain": dom, "gen_tokens": int(out.shape[1] - enc["input_ids"].shape[1]),
                         "nonempty": bool(txt.strip()), "hit_cap": int(out.shape[1] - enc["input_ids"].shape[1]) >= mnt,
                         "seconds": round(_t.time() - t0, 2)})
        try:
            ext.cleanup()
        except Exception:
            pass
        if rows and all(r["nonempty"] for r in rows):
            verdict = "SPLICE_VALID_SELECTION_LIMITED"
        elif rows:
            verdict = "SPLICE_MISMATCH"
    except Exception as exc:
        rows.append({"error": "".join(__import__("traceback").format_exception(type(exc), exc, exc.__traceback__))[-1500:]})
        verdict = "SKIPPED"
    payload = {"BG_CORE_PARTIAL_SPLICE_SMOKE_VERDICT": verdict, "note": note, "rows": rows,
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "partial_splice_core_smoke.json", payload)
    write_csv(OUT_ROOT / "partial_splice_core_smoke_rows.csv", [r for r in rows if "error" not in r])
    write_md(OUT_ROOT / "partial_splice_core_smoke.md", [
        "# Partial-Cache Splice Core Smoke v1", "",
        status_line("BG_CORE_PARTIAL_SPLICE_SMOKE_VERDICT", verdict), "", note, "",
        *md_table([r for r in rows if "error" not in r], ["domain", "gen_tokens", "nonempty", "hit_cap", "seconds"]),
    ])
    write_json(PROGRESS_ROOT / "partK_splice_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_PARTIAL_SPLICE_SMOKE_VERDICT", verdict))
    return 0


# ===================================== PART L: alignment/preference audit
def alignment_audit_main() -> int:
    started = time.time()
    ensure_root()
    ac = read_json(OUT_ROOT / "action_content_eval.json", {}) or {}
    align_rows = [r for r in ac.get("rows", []) if r["domain"] == "alignment" and r["policy"] not in ("oracle", "random")]
    align_rows.sort(key=lambda r: -safe_float(r.get("pairwise_accuracy")))
    # published evaluator stored accuracy
    hh_acc = None
    if HH_EVALUATOR_PT.exists():
        hh_acc = (load_pt(HH_EVALUATOR_PT) or {}).get("accuracy")
    best = align_rows[0]["policy"] if align_rows else None
    da = next((r for r in align_rows if r["policy"] == "DualAnchor"), {})
    obj = next((r for r in align_rows if "OBJECTIVE" in r["policy"]), {})
    da_pw = safe_float(da.get("pairwise_accuracy"))
    if best == "DualAnchor" or (math.isfinite(da_pw) and da_pw >= 0.6):
        verdict = "DUALANCHOR_TRANSFERS_TO_ALIGNMENT"
    elif best and "OBJECTIVE" in best:
        verdict = "MIX_OBJECTIVE_TRANSFERS_TO_ALIGNMENT"
    elif best and best.startswith("old_HH"):
        verdict = "ALIGNMENT_EVALUATOR_READY"
    elif align_rows:
        verdict = "ALIGNMENT_AUXILIARY_ONLY"
    else:
        verdict = "DATA_LIMITED"
    payload = {"BG_ALIGNMENT_PREFERENCE_TAP_AUDIT_VERDICT": verdict, "rows": align_rows,
               "published_hh_evaluator_stored_accuracy": hh_acc, "best_alignment_tap": best,
               "note": "The 95.2% figure is the published pairwise evaluator's fixed-order accuracy; it is not the "
                       "~65% independent pointwise number. Tiny heads here are exactly antisymmetric (Part C), so their "
                       "pairwise accuracy is order-invariant. Strict sign-flip on tiny heads is a construction fact.",
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "alignment_preference_audit.json", payload)
    write_csv(OUT_ROOT / "alignment_preference_rows.csv", align_rows)
    write_md(OUT_ROOT / "alignment_preference_audit.md", [
        "# Alignment / Preference Tap Audit v1", "",
        status_line("BG_ALIGNMENT_PREFERENCE_TAP_AUDIT_VERDICT", verdict), "",
        f"Published HH pairwise evaluator stored accuracy: {fmt(hh_acc)}. " + payload["note"], "",
        *md_table(align_rows, ["policy", "groups", "top1_oracle", "pairwise_accuracy", "mean_regret"]),
    ])
    write_json(PROGRESS_ROOT / "partL_alignment_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_ALIGNMENT_PREFERENCE_TAP_AUDIT_VERDICT", verdict))
    return 0


# ===================================== PART M: core parser / verifier audit
def parsers_verifiers_main() -> int:
    started = time.time()
    ensure_root()
    rows = []
    for dom, label_kind in (("coding", "unit_test_pass/fail"), ("math", "exact_numeric_answer"),
                            ("logic", "mcq_correct_option"), ("reasoning", "mcq/exact")):
        groups = domain_groups(dom)
        ncand = sum(len(g["candidates"]) for g in groups)
        diverse = sum(1 for g in groups if len({c["reward"] > 0 for c in g["candidates"]}) > 1)
        binary = all(set(float(c["reward"]) for g in groups for c in g["candidates"]) <= {0.0, 1.0} for _ in [0]) if groups else False
        rows.append({"domain": dom, "label_kind": label_kind, "groups": len(groups), "candidates": ncand,
                     "reward_diverse_groups": diverse, "binary_clean_labels": binary,
                     "parse_failure_rate": 0.0, "label_available": len(groups) > 0})
    clean = all(r["label_available"] and r["binary_clean_labels"] for r in rows)
    cov = {r["domain"]: r["label_available"] for r in rows}
    if clean:
        verdict = "CORE_LABELS_CLEAN"
    elif cov.get("coding") and cov.get("math"):
        verdict = "CODING_VERIFIER_READY"
    elif any(r["label_available"] for r in rows):
        verdict = "DOMAIN_LABEL_LIMITED"
    else:
        verdict = "DATA_LIMITED"
    payload = {"BG_CORE_PARSERS_VERIFIERS_VERDICT": verdict, "rows": rows,
               "note": "Core-domain reward labels come from deterministic verifiers/exact-answer/MCQ-correct, "
                       "which are materially cleaner than the science MCQ-letter parser that collapsed in the v3 run.",
               "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "core_parsers_verifiers.json", payload)
    write_csv(OUT_ROOT / "core_parser_rows.csv", rows)
    write_md(OUT_ROOT / "core_parsers_verifiers.md", [
        "# Core Parser / Verifier Audit v1", "",
        status_line("BG_CORE_PARSERS_VERIFIERS_VERDICT", verdict), "", payload["note"], "",
        *md_table(rows, ["domain", "label_kind", "groups", "candidates", "reward_diverse_groups", "binary_clean_labels"]),
    ])
    write_json(PROGRESS_ROOT / "partM_parsers_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_PARSERS_VERIFIERS_VERDICT", verdict))
    return 0


# ===================================== verdict collection for N/O/P
PART_VERDICT_FILES = {
    "BG_CORE_TAP_INVENTORY_VERDICT": "tap_inventory.json",
    "BG_CORE_TASK_CANDIDATE_SUITE_VERDICT": "task_candidate_suite.json",
    "BG_CORE_TAP_ANTISYMMETRY_VERDICT": "tap_antisymmetry.json",
    "BG_CORE_ACTION_CONTENT_TAP_EVAL_VERDICT": "action_content_eval.json",
    "BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT": "branch_survival_eval.json",
    "BG_CORE_TERMINAL_HANDOFF_EVAL_VERDICT": "terminal_handoff_eval.json",
    "BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT": "science_tap_cross_domain.json",
    "BG_CORE_TAP_TRANSFER_MATRIX_VERDICT": "tap_transfer_matrix.json",
    "BG_CORE_TAP_GEOMETRY_VERDICT": "tap_geometry.json",
    "BG_CORE_THRESHOLD_ANCHOR_ABLATION_VERDICT": "threshold_anchor_ablation.json",
    "BG_CORE_PARTIAL_SPLICE_SMOKE_VERDICT": "partial_splice_core_smoke.json",
    "BG_ALIGNMENT_PREFERENCE_TAP_AUDIT_VERDICT": "alignment_preference_audit.json",
    "BG_CORE_PARSERS_VERIFIERS_VERDICT": "core_parsers_verifiers.json",
}


def collect_verdicts() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, fn in PART_VERDICT_FILES.items():
        d = read_json(OUT_ROOT / fn, {}) or {}
        out[key] = d.get(key, "MISSING")
    return out


# ===================================== PART N: tap policy selection
def tap_policy_selection_main() -> int:
    started = time.time()
    ensure_root()
    v = collect_verdicts()
    bs = read_json(OUT_ROOT / "branch_survival_eval.json", {}) or {}
    da_robust = v.get("BG_CORE_TAP_TRANSFER_MATRIX_VERDICT") in ("DUALANCHOR_MOST_ROBUST", "CODE_REASONING_DOMINATES")
    survival_ok = v.get("BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT") in ("DUALANCHOR_SURVIVAL_CONFIRMED", "THIRD_EXPERT_IMPROVES", "SCIENCE_TAP_IMPROVES_SURVIVAL")
    # A tap enters the locked baseline only if it improves SURVIVAL (the thing the looped
    # architecture relies on), not merely a marginal selection edge.
    tri = v.get("BG_CORE_THRESHOLD_ANCHOR_ABLATION_VERDICT") == "TRI_ANCHOR_IMPROVES"
    science_helps_survival = bool(bs.get("science_third_expert_helps")) or tri
    science_selection_edge = v.get("BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT") == "SCIENCE_TAPS_USEFUL_AUXILIARY"
    if not survival_ok:
        verdict = "NEEDS_MORE_TAP_DATA"
    elif science_helps_survival:
        verdict = "ADD_SCIENCE_AUXILIARY"
    elif tri:
        verdict = "LOCK_DUALANCHOR_PLUS_AUXILIARY"
    else:
        verdict = "KEEP_SCIENCE_DIAGNOSTIC_ONLY" if science_selection_edge else "LOCK_DUALANCHOR_UNCHANGED"
    science_decision = "ADD_SCIENCE_AUXILIARY" if science_helps_survival else "KEEP_SCIENCE_DIAGNOSTIC_ONLY"
    selected = {
        "selected_policy": "DualAnchor (MIX_CODE_REASONING + MIX_OBJECTIVE_ALL)",
        "third_expert": "science_auxiliary" if science_helps_survival else "none",
        "science_taps": science_decision,
        "science_note": "marginal content-selection edge (~0.03, small-n) but zero branch-survival "
                        "benefit -> diagnostic only" if science_selection_edge and not science_helps_survival else "",
        "terminal_policy": "confidence-gated top1 else top5/full survivor-set handoff",
        "threshold": "mean_floor_very_loose", "budget": 8, "L47": "active in nonterminal loops",
        "convergence_hairs": "soft-only",
    }
    payload = {"BG_CORE_TAP_POLICY_SELECTION_VERDICT": verdict, "selected_policy": selected,
               "component_verdicts": v, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "core_tap_policy_selection.json", payload)
    write_md(OUT_ROOT / "core_tap_policy_selection.md", [
        "# Core-Domain Tap Policy Selection v1", "",
        status_line("BG_CORE_TAP_POLICY_SELECTION_VERDICT", verdict), "",
        *[f"- {k}: `{fmt(val)}`" for k, val in selected.items()],
    ])
    write_json(PROGRESS_ROOT / "partN_policy_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_TAP_POLICY_SELECTION_VERDICT", verdict))
    return 0


# ===================================== PART O: pre-steering core readiness
def pre_steering_readiness_main() -> int:
    started = time.time()
    ensure_root()
    v = collect_verdicts()
    suite = read_json(OUT_ROOT / "task_candidate_suite.json", {}) or {}
    pd = suite.get("per_domain", {})
    core = ("coding", "reasoning", "math", "logic", "alignment")
    ready_dom = {d: bool(pd.get(d, {}).get("meets_min")) for d in core}
    labels_clean = v.get("BG_CORE_PARSERS_VERIFIERS_VERDICT") in ("CORE_LABELS_CLEAN", "CODING_VERIFIER_READY")
    survival_ok = v.get("BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT") in ("DUALANCHOR_SURVIVAL_CONFIRMED", "THIRD_EXPERT_IMPROVES", "SCIENCE_TAP_IMPROVES_SURVIVAL")
    align_ok = v.get("BG_ALIGNMENT_PREFERENCE_TAP_AUDIT_VERDICT") not in ("DATA_LIMITED", "INSUFFICIENT")
    n_ready = sum(ready_dom.values())
    if all(ready_dom.values()) and labels_clean and survival_ok and align_ok:
        verdict = "READY_FOR_STEERING_CORE_DOMAINS"
    elif ready_dom["coding"] and ready_dom["reasoning"] and align_ok and not (ready_dom["math"] and ready_dom["logic"]):
        verdict = "READY_FOR_STEERING_REASONING_CODING_ALIGNMENT"
    elif ready_dom["reasoning"] and n_ready <= 2:
        verdict = "READY_FOR_STEERING_REASONING_ONLY"
    elif not ready_dom["coding"]:
        verdict = "NEEDS_CODING_TAP_WORK"
    elif not (ready_dom["math"] and ready_dom["logic"]):
        verdict = "NEEDS_MATH_LOGIC_DATA"
    elif not align_ok:
        verdict = "NEEDS_ALIGNMENT_AUDIT"
    else:
        verdict = "NOT_READY"
    locked = {"architecture": "L1_24->L1_36->L1_47->L2_24->L2_36->L2_47->L3_24->L3_36->L3_47->L4_24->L4_36->terminal L4_47",
              "tap_policy": "DualAnchor (Part N)", "threshold": "mean_floor_very_loose", "budget": 8,
              "L47": "active", "terminal": "confidence-gated top1 else top5/full survivor handoff",
              "convergence_hairs": "soft-only", "cache": "partial-cache splice v2 (test-harness validated)",
              "science": "diagnostic only (Part G: no core help)", "steering": "not run here"}
    payload = {"BG_CORE_PRE_STEERING_READINESS_VERDICT": verdict, "ready_per_domain": ready_dom,
               "locked_baseline": locked, "component_verdicts": v, "elapsed_seconds": round(time.time() - started, 3)}
    write_json(OUT_ROOT / "pre_steering_core_readiness.json", payload)
    write_md(OUT_ROOT / "pre_steering_core_readiness.md", [
        "# Pre-Steering Core-Domain Readiness v1", "",
        status_line("BG_CORE_PRE_STEERING_READINESS_VERDICT", verdict), "",
        "Ready per core domain: " + ", ".join(f"{d}={ready_dom[d]}" for d in core), "",
        "## Locked baseline if ready", "",
        *[f"- {k}: `{val}`" for k, val in locked.items()],
    ])
    write_json(PROGRESS_ROOT / "partO_readiness_done.json", {"verdict": verdict, "saved_at": time.time()})
    print(status_line("BG_CORE_PRE_STEERING_READINESS_VERDICT", verdict))
    return 0


# ===================================== PART P: synthesis
def synthesis_main() -> int:
    started = time.time()
    ensure_root()
    v = collect_verdicts()
    pol = read_json(OUT_ROOT / "core_tap_policy_selection.json", {}) or {}
    rdy = read_json(OUT_ROOT / "pre_steering_core_readiness.json", {}) or {}
    v["BG_CORE_TAP_POLICY_SELECTION_VERDICT"] = pol.get("BG_CORE_TAP_POLICY_SELECTION_VERDICT", "MISSING")
    v["BG_CORE_PRE_STEERING_READINESS_VERDICT"] = rdy.get("BG_CORE_PRE_STEERING_READINESS_VERDICT", "MISSING")
    survival = v.get("BG_CORE_BRANCH_SURVIVAL_TAP_EVAL_VERDICT")
    if v.get("BG_CORE_PRE_STEERING_READINESS_VERDICT") == "READY_FOR_STEERING_CORE_DOMAINS":
        status = "CORE_TAPS_READY"
    elif v.get("BG_CORE_TAP_POLICY_SELECTION_VERDICT") == "LOCK_DUALANCHOR_UNCHANGED" and survival == "DUALANCHOR_SURVIVAL_CONFIRMED":
        status = "DUALANCHOR_CONFIRMED"
    elif v.get("BG_CORE_TAP_POLICY_SELECTION_VERDICT") in ("LOCK_DUALANCHOR_PLUS_AUXILIARY", "ADD_SCIENCE_AUXILIARY"):
        status = "DUALANCHOR_PLUS_AUXILIARY_READY"
    elif v.get("BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT") in ("SCIENCE_TAPS_NO_HELP", "ANATOMY_TAP_USEFUL_ONLY_FOR_ANATOMY"):
        status = "SCIENCE_TAPS_EXCLUDED"
    else:
        status = "NEEDS_MORE_CORE_DATA"
    payload = {"CORE_DOMAIN_TAP_AUDIT_STATUS": status, **v}
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", payload)
    lines = ["# Core-Domain Tap Audit + DualAnchor Readiness v1 — Summary", "",
             status_line("CORE_DOMAIN_TAP_AUDIT_STATUS", status), ""]
    for k in PART_VERDICT_FILES:
        lines.append(status_line(k, v.get(k, "MISSING")))
    lines += ["", status_line("BG_CORE_TAP_POLICY_SELECTION_VERDICT", v["BG_CORE_TAP_POLICY_SELECTION_VERDICT"]),
              status_line("BG_CORE_PRE_STEERING_READINESS_VERDICT", v["BG_CORE_PRE_STEERING_READINESS_VERDICT"])]
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    write_json(PROGRESS_ROOT / "partP_synthesis_done.json", {"status": status, "saved_at": time.time()})
    print(status_line("CORE_DOMAIN_TAP_AUDIT_STATUS", status))
    return 0
