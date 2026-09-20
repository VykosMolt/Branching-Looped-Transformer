"""Shared harness for branch_training_offline_verifier_generator_v2.

Thin layer over branch_training_v1_common (B1): same model/verifier/generation/io machinery,
new v2 roots and a few offline-first utilities (GPU audit, run-state ledger, Wilson CI,
STOP-file, deterministic heldout/canary sampling). Reuses v1 data under data/.../ ; writes
new data under train_v2/ and canary/. Never overwrites v1 artifacts or base model.
"""
from __future__ import annotations
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import branch_training_v1_common as B1  # noqa: E402  (the v1 harness)

# ---- re-exports (single import surface for the v2 scripts) ----
v2 = B1.v2
L = B1.L
PROJECT_ROOT = B1.PROJECT_ROOT
write_json = B1.write_json
write_md = B1.write_md
write_csv = B1.write_csv
read_jsonl = B1.read_jsonl
status_line = B1.status_line
finite_mean = B1.finite_mean
md_table = B1.md_table
label_branch = B1.label_branch
verify_final = B1.verify_final
build_prompt = B1.build_prompt
trim_generation = B1.trim_generation
load_logic_tasks = B1.load_logic_tasks
load_coding_gen_tasks = B1.load_coding_gen_tasks
_B1_make_stop = B1._make_stop
_reconstruct_core_gen = B1._reconstruct_core_gen
DOMAIN_MAXTOK = B1.DOMAIN_MAXTOK

# ---- v2 roots ----
RUN_NAME = "branch_training_offline_verifier_generator_v2"
OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/branch_training_offline_verifier_generator_v2_2026-06-07"
MODEL_ROOT = PROJECT_ROOT / "opi/taps/models/branch_training_offline_verifier_generator_v2"
DATA_ROOT = B1.DATA_ROOT  # shared/data/branch_training_logic_expansion_v1
TRAIN_V2 = DATA_ROOT / "train_v2"
CANARY_DIR = DATA_ROOT / "canary"
LOG_DIR = PROJECT_ROOT / "artifacts/logs/branch_training_v2"
PREV_SFT_ADAPTER = PROJECT_ROOT / "opi/taps/models/branch_training_logic_expansion_v1/branching_sft"
BASE_MODEL = PROJECT_ROOT / "shared/models/ouro_rltt_local"
RUN_STATE = OUT_ROOT / "run_state.json"
STOP_DIR = MODEL_ROOT / "_control"

CORE_DOMAINS = ["coding", "reasoning", "math", "logic", "alignment"]


def ensure_dirs() -> None:
    for d in (OUT_ROOT, OUT_ROOT / "progress", MODEL_ROOT, MODEL_ROOT / "configs", TRAIN_V2,
              CANARY_DIR, LOG_DIR, STOP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def prog(name: str, payload: dict) -> None:
    ensure_dirs()
    (OUT_ROOT / "progress" / f"{name}.json").write_text(json.dumps(payload, default=v2.json_default, indent=2))


# ---- STOP-file (pausable) ----
def stop_path(job: str) -> Path:
    return STOP_DIR / f"STOP_{job}"


def stop_requested(job: str) -> bool:
    return stop_path(job).exists() or (STOP_DIR / "STOP").exists()


# ---- GPU / orphan audit ----
def gpu_audit() -> dict:
    info: dict[str, Any] = {"orphans": []}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        used, total, util = [x.strip() for x in out.split(",")]
        info.update(used_mib=int(used), total_mib=int(total), util_pct=int(util), free_mib=int(total) - int(used))
    except Exception as e:
        info.update(error=str(e), free_mib=None)
    try:
        ps = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True, timeout=20).stdout
        for line in ps.splitlines():
            if any(k in line for k in ("generate_branch_pools", "train_offline_branch", "train_branching_sft",
                                       "evaluate_offline_branch", "render_verified_logic")) and "grep" not in line:
                info["orphans"].append(line.strip()[:120])
    except Exception:
        pass
    return info


# ---- run-state ledger (single source of truth) ----
def read_state() -> dict:
    return v2.read_json(RUN_STATE, {}) or {}


def write_state(updates: dict) -> dict:
    st = read_state()
    st.setdefault("run", RUN_NAME)
    st.setdefault("stages", {})
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    for k, val in updates.items():
        if k == "stages":
            st["stages"].update(val)
        else:
            st[k] = val
    ensure_dirs()
    RUN_STATE.write_text(json.dumps(st, default=v2.json_default, indent=2))
    return st


def set_stage(stage: str, verdict: str, extra: dict | None = None) -> None:
    rec = {"verdict": verdict, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    write_state({"stages": {stage: rec}})


# ---- statistics ----
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def flag(delta: float, ci_excludes_zero: bool | None = None) -> str:
    """Label a delta conservatively. Strong claims only when CI excludes zero."""
    if ci_excludes_zero is True:
        return "MEASURED_GAIN" if delta > 0 else "MEASURED_REGRESSION"
    if abs(delta) < 1e-9:
        return "INCONCLUSIVE"
    return "FLAG_UP" if delta > 0 else "FLAG_DOWN"


# ---- deterministic heldout pools (task-disjoint, stable ids) ----
def logic_heldout(per_family: int | None = None) -> list[dict]:
    out = [t for t in load_logic_tasks() if t.get("split") == "heldout"]
    if per_family is None:
        return out
    by: dict[str, list] = {}
    for t in out:
        by.setdefault(t["category"], []).append(t)
    sel = []
    for c in sorted(by):
        sel += sorted(by[c], key=lambda t: v2.text_hash(t["task_prompt"]))[:per_family]
    return sel


def core_heldout(domain: str, n: int) -> list[dict]:
    """math/reasoning heldout reconstructed from corecontent_v2 deduped groups."""
    rows = read_jsonl(PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl")
    got = []
    for g in rows:
        if g.get("domain") == domain and g.get("split") == "heldout":
            r = _reconstruct_core_gen(g)
            if r:
                got.append(r)
    got.sort(key=lambda t: v2.text_hash(t["task_prompt"]))
    return got[:n]


def coding_heldout(n: int, offset: int = 200) -> list[dict]:
    """MBPP tasks beyond the v1-trained slice (task-disjoint), name-fix prompt applied downstream."""
    cod = load_coding_gen_tasks(offset + n + 50)
    return cod[offset:offset + n]
