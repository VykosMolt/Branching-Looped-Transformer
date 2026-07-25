"""Cross-loop early-layer tap experiment v1 (2026-07-20) — shared constants and IO.

Read-only experiment on frozen Ouro-RLTT: does CoreContent process-quality signal become
readable at physical layers 8/16 in later recurrent loops, and do tap directions transfer
frozen across loops? No steering, no backbone training, no production artifact mutation.

New scripts only; nothing here replaces production code. Output root is isolated:
`artifacts/reports/cross_loop_early_layer_taps_20260720/`.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT_ROOT / "artifacts/reports/cross_loop_early_layer_taps_20260720"
LOG_ROOT = RUN_ROOT / "logs"
PRED_ROOT = RUN_ROOT / "predictions"
FIG_ROOT = RUN_ROOT / "figures"
FEATURE_ROOT = RUN_ROOT / "features"

DATA_ROOT = PROJECT_ROOT / "data/corecontent_v2"
PROC_JSONL = DATA_ROOT / "processed/candidate_groups_deduped.jsonl"
V2_FEATURE_ROOT = DATA_ROOT / "features"
V2_OUT_ROOT = PROJECT_ROOT / "artifacts/reports/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04"
V2_POLICY_PT = V2_OUT_ROOT / "corecontent_v2_policy.pt"

CORE_DOMAINS = ("coding", "reasoning", "math", "logic", "alignment")
LAYERS = (8, 16, 24, 36, 47)
LAYER_POS = {l: i for i, l in enumerate(LAYERS)}
CAPTURE_IDX = {8: 7, 16: 15, 24: 23, 36: 35}  # post-block hook index = layer - 1; 47 = loop boundary
LOOPS = (1, 2, 3, 4)
HIDDEN = 2048
V2_LAYER_POS = {24: 0, 36: 1, 47: 2}  # cached v2 shard tensor layout (3,4,2048)

# identical to bg_corecontent_v2_features.MAX_LEN (feature-match depends on this)
MAX_LEN = {"alignment": 768, "coding": 1024, "math": 768, "logic": 640, "reasoning": 640}

SUBSET_SEED_NS = "xloop_early_v1"
SUBSET_SIZES = {"train": 250, "val": 60, "heldout": 120}
BOOT_SEED = 20260720
BOOT_DRAWS = 10000

# v2 train_pairwise grid, reproduced verbatim (selection on val only)
TRAINING_SETS = {
    "all_core": CORE_DOMAINS,
    "all_core_balanced": CORE_DOMAINS,
    "code_reasoning": ("coding", "reasoning"),
    "math_logic": ("math", "logic"),
    "alignment_only": ("alignment",),
    "code_math_logic": ("coding", "math", "logic"),
}
GRID_LRS = (3e-4, 1e-3)
GRID_SEEDS = (0, 1, 2)

REFIT_CELLS = [(l, u) for l in (8, 16, 24) for u in LOOPS] + [(36, 4), (47, 4)]
TRANSFERS = [(l, u, v) for l in (8, 16, 24) for u in LOOPS for v in LOOPS if v > u]
SHUFFLE_CELLS = [(8, 1), (8, 4), (16, 2), (24, 4)]


def cell_name(layer: int, loop: int) -> str:
    return f"{layer}_L{loop}"


def stable_int(*parts: object) -> int:
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return default


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_roots() -> None:
    for p in (RUN_ROOT, LOG_ROOT, PRED_ROOT, FIG_ROOT, FEATURE_ROOT):
        p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- scoring (v2-equivalent)
def finite_mean(vals) -> float:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def normalize_scores(vals: list[float]) -> list[float]:
    xs = [float(v) if math.isfinite(float(v)) else 0.0 for v in vals]
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]


def cell_vector(cand: dict[str, Any], layer: int, loop: int) -> torch.Tensor:
    """Feature vector for one (layer, loop) cell from the (5,4,2048) fresh tensor."""
    f = cand["features"]
    assert tuple(f.shape) == (len(LAYERS), len(LOOPS), HIDDEN), f"bad feature shape {tuple(f.shape)}"
    return f[LAYER_POS[layer], loop - 1].to(torch.float32)


def score_group_cell(group: dict[str, Any], weight: torch.Tensor, layer: int, loop: int) -> list[float]:
    """Single-channel AntisymLinearNoNorm scoring, exactly the locked v2 convention:
    per-candidate mean pairwise margin w·(vi - vj), then z-normalized within group."""
    cands = group["candidates"]
    n = len(cands)
    if n < 2:
        return [float("nan")] * n
    w = weight.detach().to(torch.float32).flatten()
    vecs = [cell_vector(c, layer, loop) for c in cands]
    raw = torch.stack(vecs, 0) @ w  # [n]; margin mean_j w·(vi-vj) = w·vi - mean_j w·vj (j != i)
    tot = raw.sum()
    scores = [float(raw[i] - (tot - raw[i]) / (n - 1)) for i in range(n)]
    return normalize_scores(scores)


def group_selection_metrics(group: dict[str, Any], scores: list[float]) -> dict[str, float]:
    """Verbatim logic of cc.group_selection_metrics."""
    cands = group["candidates"]
    rewards = [float(c["reward"]) for c in cands]
    oracle = max(rewards)
    finite = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    if not finite:
        return {}
    top = max(finite, key=lambda x: x[1])[0]
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
        "pairwise_accuracy": (correct / total) if total else float("nan"),
        "pairwise_pairs": total,
    }


def chance_top1(group: dict[str, Any]) -> float:
    rewards = [float(c["reward"]) for c in group["candidates"]]
    mx = max(rewards)
    return sum(1 for r in rewards if r >= mx) / len(rewards)


# ----------------------------------------------------------------- fresh feature shards
def load_fresh_groups(splits=None, domains=None) -> list[dict[str, Any]]:
    """Load extracted (5,4,2048) feature groups from this run's shards."""
    man = read_json(FEATURE_ROOT / "feature_manifest.json", {})
    out = []
    for s in man.get("shards", []):
        if splits and s["split"] not in splits:
            continue
        if domains and s["domain"] not in domains:
            continue
        payload = torch.load(FEATURE_ROOT / s["file"], map_location="cpu", weights_only=False)
        for g in payload["groups"]:
            for c in g["candidates"]:
                c["features"] = c["features"].to(torch.float32)
            out.append(g)
    return out
