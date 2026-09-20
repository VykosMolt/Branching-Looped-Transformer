"""Old-anchored branch-valid tap transplant v1.

This experiment tests two old-tap-preserving routes:

1. Weight-space transplant:
   old coding/reasoning or mixed-objective anchors plus blockwise branch and
   bridge residuals.
2. Constrained fine-tuned copies:
   copied old anchors trained on branch/bridge labels while old-content/code
   preservation and small deltas are explicitly penalized.

It only reads cached tap/evaluator artifacts and writes new diagnostic tap
artifacts under a new report path. It does not train Ouro, modify checkpoints,
tokenizers, tap registries, production routing, or apply steering.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from bg_gated_selector_v1_common import load_old_code_pairs
from bg_hidden_origin_tap_common import HIDDEN_DIM, PROJECT_ROOT, PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import (
    ALIGNED_WEIGHTS_PT,
    FEATURE_STATS_PT,
    PRIMARY_ARCHITECTURES,
    PRIMARY_TARGET,
    branch_groups,
    candidate_record,
    concat_blocks,
    cosine,
    eval_candidate_on_pairs,
    finite_mean,
    group_topk_metrics,
    load_pair_sets,
    normed,
    orthogonal_residual,
    pair_diff,
    rate,
    safe_float,
    score_diff,
    top_sources,
)


OUT_ROOT = PROBE_ROOT / "bg_old_anchored_branch_valid_taps_v1_2026-05-30"
RESULTS_JSON = OUT_ROOT / "old_anchored_branch_valid_taps_v1.json"
RESULTS_MD = OUT_ROOT / "old_anchored_branch_valid_taps_v1.md"
ROWS_CSV = OUT_ROOT / "old_anchored_branch_valid_tap_rows.csv"
CANDIDATES_PT = OUT_ROOT / "old_anchored_branch_valid_taps_v1.pt"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md"

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_merged_weight_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
]
NAV_TARGETS = [
    PROJECT_ROOT / "shared/docs/evaluator/README.md",
    PROJECT_ROOT / "shared/docs/README.md",
    PROJECT_ROOT / "PROJECT_TREE_MAP.md",
    PROJECT_ROOT / "PROJECT_COMPONENTS.md",
]

ANCHOR_GROUPS = ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL", "MIX_REASONING_SCIENCE")
PRIMARY_ANCHOR_GROUPS = {"MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL"}
BLOCKS = ("24_L4", "36_L4", "47_L4")


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().to(torch.float32)
        return {
            "shape": list(x.shape),
            "mean": float(x.mean().item()) if x.numel() else 0.0,
            "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
            "rms": float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0,
        }
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def append_section(path: Path, title: str, lines: Sequence[str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if title in text:
        return
    with path.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n" + "\n".join(lines) + "\n")


def load_aligned() -> list[dict[str, Any]]:
    payload = torch.load(ALIGNED_WEIGHTS_PT, map_location="cpu", weights_only=False)
    return list(payload.get("aligned_weights") or [])


def block_slice(block: str) -> slice:
    idx = list(BLOCKS).index(block)
    return slice(idx * HIDDEN_DIM, (idx + 1) * HIDDEN_DIM)


def block_mask(blocks: Iterable[str]) -> torch.Tensor:
    mask = torch.zeros(HIDDEN_DIM * len(BLOCKS), dtype=torch.float32)
    for block in blocks:
        mask[block_slice(block)] = 1.0
    return mask


def family_rows(aligned: Sequence[dict[str, Any]], family: str, arch: str) -> list[dict[str, Any]]:
    return [
        r
        for r in aligned
        if r.get("target_config") == PRIMARY_TARGET
        and r.get("source_family") == family
        and r.get("architecture") == arch
        and isinstance(r.get("aligned_weight"), torch.Tensor)
    ]


def best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: (safe_float(r.get("metric_score"), -1e9), 1.0 if r.get("alignment_method") == "direct" else 0.0, str(r.get("head_name"))))


def build_block_anchor(aligned: Sequence[dict[str, Any]], arch: str, group: str) -> dict[str, Any] | None:
    rows = []
    for block in BLOCKS:
        matches = [
            r
            for r in family_rows(aligned, "old_content", arch)
            if group in str(r.get("head_name"))
            and str(r.get("source_config")) == block
            and str(r.get("alignment_method")) == "lifted_zero_block"
        ]
        row = best_row(matches)
        if row is None:
            return None
        rows.append(row)
    weight = normed(sum((r["aligned_weight"] for r in rows), torch.zeros(HIDDEN_DIM * len(BLOCKS))))
    return {
        "head_name": f"old_anchor::{group}::{PRIMARY_TARGET}::{arch}",
        "source_family": "old_content",
        "target_config": PRIMARY_TARGET,
        "source_config": "block_concat_24_36_47",
        "architecture": arch,
        "alignment_method": "block_concat",
        "aligned_weight": weight,
        "metric_score": finite_mean(r.get("metric_score") for r in rows),
        "block_sources": [r.get("head_name") for r in rows],
        "anchor_group": group,
    }


def source_candidates(aligned: Sequence[dict[str, Any]], arch: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchors = [a for group in ANCHOR_GROUPS if (a := build_block_anchor(aligned, arch, group)) is not None]
    branch = top_sources(aligned, "branch", arch, limit=4)
    bridge = top_sources(aligned, "bridge", arch, limit=3)
    return anchors, branch, bridge


def masked_residual(vec: torch.Tensor, bases: Sequence[torch.Tensor], blocks: Sequence[str]) -> tuple[torch.Tensor, float]:
    mask = block_mask(blocks)
    res = orthogonal_residual(vec, bases) * mask
    n = float(res.norm().item())
    if n >= 1e-8:
        res = res / n
    return res, n


def make_candidate(
    *,
    name: str,
    family: str,
    arch: str,
    weight: torch.Tensor,
    anchor: dict[str, Any],
    branch: dict[str, Any] | None,
    bridge: dict[str, Any] | None,
    coeffs: tuple[float, float, float],
    mode: str,
    component_norms: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cand = candidate_record(
        name=name,
        family=family,
        target_config=PRIMARY_TARGET,
        architecture=arch,
        weight=normed(weight),
        old=anchor,
        branch=branch,
        bridge=bridge,
        coefficients=coeffs,
        norm_mode=mode,
        component_norms=component_norms,
    )
    cand["anchor_group"] = anchor.get("anchor_group")
    cand["anchor_block_sources"] = anchor.get("block_sources")
    if extra:
        cand.update(extra)
    return cand


def build_transplant_candidates(aligned: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for arch in PRIMARY_ARCHITECTURES:
        anchors, branches, bridges = source_candidates(aligned, arch)
        for anchor in anchors:
            old = normed(anchor["aligned_weight"])
            candidates.append(
                make_candidate(
                    name=f"old_anchor_reexport::{anchor['anchor_group']}::{arch}",
                    family="old_anchor_reexport",
                    arch=arch,
                    weight=old,
                    anchor=anchor,
                    branch=None,
                    bridge=None,
                    coeffs=(1.0, 0.0, 0.0),
                    mode="block_concat_anchor",
                    component_norms={"old": float(old.norm().item())},
                )
            )
            for branch in branches[:2]:
                b_full, b_full_norm = masked_residual(normed(branch["aligned_weight"]), [old], list(BLOCKS))
                b_l24, b_l24_norm = masked_residual(normed(branch["aligned_weight"]), [old], ["24_L4"])
                b_l2436, b_l2436_norm = masked_residual(normed(branch["aligned_weight"]), [old], ["24_L4", "36_L4"])
                for bridge in bridges[:2]:
                    bridge_vec = normed(bridge["aligned_weight"])
                    c_full, c_full_norm = masked_residual(bridge_vec, [old, b_full], list(BLOCKS))
                    c_l47, c_l47_norm = masked_residual(bridge_vec, [old, b_full], ["47_L4"])
                    c_l3647, c_l3647_norm = masked_residual(bridge_vec, [old, b_full], ["36_L4", "47_L4"])
                    grids = [
                        ("full_residual", b_full, c_full, b_full_norm, c_full_norm, (0.85, 0.10, 0.05)),
                        ("full_residual", b_full, c_full, b_full_norm, c_full_norm, (0.80, 0.10, 0.10)),
                        ("full_residual", b_full, c_full, b_full_norm, c_full_norm, (0.75, 0.15, 0.10)),
                        ("branch_L24_bridge_L47", b_l24, c_l47, b_l24_norm, c_l47_norm, (0.85, 0.10, 0.05)),
                        ("branch_L24_bridge_L47", b_l24, c_l47, b_l24_norm, c_l47_norm, (0.80, 0.15, 0.05)),
                        ("branch_L24_bridge_L47", b_l24, c_l47, b_l24_norm, c_l47_norm, (0.75, 0.15, 0.10)),
                        ("branch_L24L36_bridge_L47", b_l2436, c_l47, b_l2436_norm, c_l47_norm, (0.80, 0.15, 0.05)),
                        ("branch_L24_bridge_L36L47", b_l24, c_l3647, b_l24_norm, c_l3647_norm, (0.80, 0.10, 0.10)),
                    ]
                    for mode, b_res, c_res, b_norm, c_norm, (a, b, c) in grids:
                        weight = normed(a * old + b * b_res + c * c_res)
                        candidates.append(
                            make_candidate(
                                name=f"transplant::{anchor['anchor_group']}::{mode}::{arch}::{len(candidates)}",
                                family="weight_space_transplant",
                                arch=arch,
                                weight=weight,
                                anchor=anchor,
                                branch=branch,
                                bridge=bridge,
                                coeffs=(a, b, c),
                                mode=mode,
                                component_norms={"old": float(old.norm().item()), "branch_residual": b_norm, "bridge_residual": c_norm},
                            )
                        )
    return candidates


def pair_matrix(pairs: Sequence[dict[str, Any]], split: str, arch: str) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for pair in pairs:
        if str(pair.get("split")) != split:
            continue
        diff = pair_diff(pair, PRIMARY_TARGET)
        if diff is None:
            continue
        x = diff.detach().cpu().to(torch.float32).flatten()
        if arch == "AntisymLinear":
            x = F.layer_norm(x, (int(x.numel()),))
        rows.append(x)
    if not rows:
        return torch.empty(0, HIDDEN_DIM * len(BLOCKS), dtype=torch.float32)
    return torch.stack(rows, dim=0)


def bce_positive(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))


def train_copy(
    *,
    anchor: dict[str, Any],
    arch: str,
    pair_sets: dict[str, list[dict[str, Any]]],
    hparams: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    old0 = normed(anchor["aligned_weight"])
    w = torch.nn.Parameter(old0.clone())
    opt = torch.optim.AdamW([w], lr=float(hparams["lr"]), weight_decay=0.0)
    old_x = pair_matrix(pair_sets["old_content"], "train", arch)
    code_x = pair_matrix(pair_sets["old_code"], "train", arch)
    hidden_x = pair_matrix(pair_sets["hidden_branch"], "train", arch)
    bridge_x = pair_matrix(pair_sets["bridge"], "train", arch)
    allowed_blocks = hparams.get("allowed_blocks")
    grad_mask = block_mask(allowed_blocks) if allowed_blocks else torch.ones_like(old0)
    l36_mask = block_mask(["36_L4"])
    losses: list[dict[str, float]] = []
    for step in range(int(hparams["steps"])):
        opt.zero_grad(set_to_none=True)
        loss_old = bce_positive(old_x @ w)
        loss_code = bce_positive(code_x @ w)
        loss_hidden = bce_positive(hidden_x @ w)
        loss_bridge = bce_positive(bridge_x @ w)
        delta = w - old0
        loss_delta = delta.pow(2).mean()
        loss_l36 = ((delta * l36_mask).pow(2).mean())
        loss = (
            float(hparams["old_w"]) * loss_old
            + float(hparams["code_w"]) * loss_code
            + float(hparams["hidden_w"]) * loss_hidden
            + float(hparams["bridge_w"]) * loss_bridge
            + float(hparams["l2_delta"]) * loss_delta
            + float(hparams.get("l36_delta", 0.0)) * loss_l36
        )
        loss.backward()
        if w.grad is not None:
            w.grad.mul_(grad_mask)
            torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
        if step in {0, int(hparams["steps"]) - 1}:
            losses.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().item()),
                    "old_loss": float(loss_old.detach().item()),
                    "code_loss": float(loss_code.detach().item()),
                    "hidden_loss": float(loss_hidden.detach().item()),
                    "bridge_loss": float(loss_bridge.detach().item()),
                    "delta_norm": float((w.detach() - old0).norm().item()),
                }
            )
    out = normed(w.detach())
    meta = {
        "hparams": hparams,
        "train_losses": losses,
        "delta_norm": float((out - old0).norm().item()),
        "cosine_to_anchor": cosine(out, old0),
        "allowed_blocks": allowed_blocks or "all",
    }
    return out, meta


def build_constrained_copy_candidates(aligned: Sequence[dict[str, Any]], pair_sets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    hparam_grid = [
        {"name": "preserve_heavy", "old_w": 6.0, "code_w": 6.0, "hidden_w": 1.0, "bridge_w": 1.0, "l2_delta": 10.0, "l36_delta": 5.0, "lr": 3e-3, "steps": 220},
        {"name": "branch_bridge_balanced", "old_w": 4.0, "code_w": 4.0, "hidden_w": 1.5, "bridge_w": 1.5, "l2_delta": 5.0, "l36_delta": 2.0, "lr": 3e-3, "steps": 220},
        {"name": "branch_light", "old_w": 6.0, "code_w": 6.0, "hidden_w": 2.0, "bridge_w": 0.5, "l2_delta": 8.0, "l36_delta": 4.0, "lr": 3e-3, "steps": 220},
        {"name": "bridge_light", "old_w": 6.0, "code_w": 6.0, "hidden_w": 0.5, "bridge_w": 2.0, "l2_delta": 8.0, "l36_delta": 4.0, "lr": 3e-3, "steps": 220},
        {"name": "l24_l47_only", "old_w": 5.0, "code_w": 5.0, "hidden_w": 1.5, "bridge_w": 1.5, "l2_delta": 6.0, "l36_delta": 8.0, "lr": 3e-3, "steps": 220, "allowed_blocks": ["24_L4", "47_L4"]},
    ]
    for arch in PRIMARY_ARCHITECTURES:
        anchors, _, _ = source_candidates(aligned, arch)
        for anchor in anchors:
            for hp in hparam_grid:
                weight, meta = train_copy(anchor=anchor, arch=arch, pair_sets=pair_sets, hparams=hp)
                candidates.append(
                    make_candidate(
                        name=f"constrained_copy::{anchor['anchor_group']}::{hp['name']}::{arch}",
                        family="constrained_finetune_copy",
                        arch=arch,
                        weight=weight,
                        anchor=anchor,
                        branch=None,
                        bridge=None,
                        coeffs=(1.0, 0.0, 0.0),
                        mode=hp["name"],
                        component_norms={"delta_norm": meta["delta_norm"], "cosine_to_anchor": meta["cosine_to_anchor"]},
                        extra={"training_meta": meta},
                    )
                )
    return candidates


def build_pair_sets() -> dict[str, list[dict[str, Any]]]:
    pair_sets = load_pair_sets()
    code_pairs = load_old_code_pairs()
    for pair in code_pairs:
        pair["pair_type"] = "old_code"
    pair_sets["old_code"] = code_pairs
    return pair_sets


def eval_pair_type(candidate: dict[str, Any], pairs: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    out = eval_candidate_on_pairs(candidate, pairs, split=split)
    return {
        "pair_count": out.get("pair_count"),
        "pairwise_accuracy": out.get("pairwise_accuracy"),
        "domain_accuracy": out.get("domain_accuracy"),
    }


def evaluate_candidate(candidate: dict[str, Any], pair_sets: dict[str, list[dict[str, Any]]], groups_by_split: dict[str, list[list[dict[str, Any]]]], split: str) -> dict[str, Any]:
    old = eval_pair_type(candidate, pair_sets["old_content"], split)
    code = eval_pair_type(candidate, pair_sets["old_code"], split)
    hidden = eval_pair_type(candidate, pair_sets["hidden_branch"], split)
    bridge = eval_pair_type(candidate, pair_sets["bridge"], split)
    survival = group_topk_metrics(candidate, groups_by_split.get(split, []), k=4)
    return {
        "old_acc": old.get("pairwise_accuracy"),
        "old_pairs": old.get("pair_count"),
        "old_domain_accuracy": old.get("domain_accuracy"),
        "code_acc": code.get("pairwise_accuracy"),
        "code_pairs": code.get("pair_count"),
        "hidden_branch_acc": hidden.get("pairwise_accuracy"),
        "hidden_branch_pairs": hidden.get("pair_count"),
        "bridge_acc": bridge.get("pairwise_accuracy"),
        "bridge_pairs": bridge.get("pair_count"),
        "survival_oracle_retention": survival.get("oracle_retention"),
        "survival_false_prune": survival.get("false_prune_rate"),
        "survival_best_selected_reward": survival.get("best_selected_reward"),
        "survival_group_count": survival.get("group_count"),
        "survival_domain_breakdown": survival.get("domain_breakdown"),
    }


def balanced_score(metrics: dict[str, Any]) -> float:
    return finite_mean(
        [
            safe_float(metrics.get("old_acc"), float("nan")),
            safe_float(metrics.get("code_acc"), float("nan")),
            safe_float(metrics.get("hidden_branch_acc"), float("nan")),
            safe_float(metrics.get("bridge_acc"), float("nan")),
            safe_float(metrics.get("survival_oracle_retention"), float("nan")),
        ]
    )


def summarize_rows(rows: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    split_rows = [r for r in rows if r.get("split") == split]
    if not split_rows:
        return {}
    best = max(split_rows, key=lambda r: (safe_float(r.get("balanced_score"), -1e9), safe_float(r.get("old_acc"), -1e9), str(r.get("candidate_name"))))
    family_best: dict[str, dict[str, Any]] = {}
    for row in split_rows:
        fam = str(row.get("candidate_family"))
        if fam not in family_best or safe_float(row.get("balanced_score"), -1e9) > safe_float(family_best[fam].get("balanced_score"), -1e9):
            family_best[fam] = row
    return {"best": best, "family_best": family_best, "rows": split_rows}


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in candidate.items() if k not in {"weight", "state_dict"}}


def select_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    val_rows = [r for r in rows if r.get("split") == "val" and r.get("anchor_group") in PRIMARY_ANCHOR_GROUPS]
    anchors = [r for r in val_rows if r.get("candidate_family") == "old_anchor_reexport"]
    if not val_rows or not anchors:
        return None
    anchor_by_group_arch = {
        (r.get("anchor_group"), r.get("architecture")): r
        for r in anchors
    }
    eligible = []
    for row in val_rows:
        if row.get("candidate_family") == "old_anchor_reexport":
            continue
        anchor = anchor_by_group_arch.get((row.get("anchor_group"), row.get("architecture")))
        if not anchor:
            continue
        old_floor = safe_float(anchor.get("old_acc"), 0.0) - 0.03
        code_floor = safe_float(anchor.get("code_acc"), 0.0) - 0.05
        if safe_float(row.get("old_acc"), 0.0) < old_floor:
            continue
        if safe_float(row.get("code_acc"), 0.0) < code_floor:
            continue
        branch_gain = safe_float(row.get("hidden_branch_acc"), 0.0) - safe_float(anchor.get("hidden_branch_acc"), 0.0)
        bridge_gain = safe_float(row.get("bridge_acc"), 0.0) - safe_float(anchor.get("bridge_acc"), 0.0)
        survival_gain = safe_float(row.get("survival_oracle_retention"), 0.0) - safe_float(anchor.get("survival_oracle_retention"), 0.0)
        if max(branch_gain, bridge_gain, survival_gain) <= 0.0:
            continue
        eligible.append((row, branch_gain + bridge_gain + survival_gain))
    if not eligible:
        return max(val_rows, key=lambda r: safe_float(r.get("balanced_score"), -1e9))
    best_score = max(safe_float(item[0].get("balanced_score"), -1e9) for item in eligible)
    # The hypothesis under test is explicitly old coding/reasoning first, and
    # direct transplant is less overfit-prone than a trained copy. If a safe
    # coding/reasoning transplant is close on validation, prefer it.
    near_best = [item for item in eligible if safe_float(item[0].get("balanced_score"), -1e9) >= best_score - 0.04]
    code_reasoning_transplant = [
        item
        for item in near_best
        if item[0].get("anchor_group") == "MIX_CODE_REASONING"
        and item[0].get("candidate_family") == "weight_space_transplant"
    ]
    if code_reasoning_transplant:
        return max(code_reasoning_transplant, key=lambda item: (safe_float(item[0].get("balanced_score"), -1e9), item[1], safe_float(item[0].get("code_acc"), -1e9)))[0]
    code_reasoning = [item for item in near_best if item[0].get("anchor_group") == "MIX_CODE_REASONING"]
    if code_reasoning:
        return max(code_reasoning, key=lambda item: (safe_float(item[0].get("balanced_score"), -1e9), item[1], safe_float(item[0].get("code_acc"), -1e9)))[0]
    return max(eligible, key=lambda item: (safe_float(item[0].get("balanced_score"), -1e9), item[1], safe_float(item[0].get("old_acc"), -1e9)))[0]


def main() -> int:
    ensure_root()
    started = time.time()
    aligned = load_aligned()
    pair_sets = build_pair_sets()
    groups_by_split = {split: branch_groups(split) for split in ("val", "heldout")}
    transplant_candidates = build_transplant_candidates(aligned)
    constrained_candidates = build_constrained_copy_candidates(aligned, pair_sets)
    candidates = transplant_candidates + constrained_candidates
    rows: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        for split in ("val", "heldout"):
            metrics = evaluate_candidate(cand, pair_sets, groups_by_split, split)
            row = {
                "candidate_index": idx,
                "candidate_name": cand.get("candidate_name"),
                "candidate_family": cand.get("candidate_family"),
                "anchor_group": cand.get("anchor_group"),
                "architecture": cand.get("architecture"),
                "mode": cand.get("norm_mode"),
                "old_source": cand.get("old_source"),
                "branch_source": cand.get("branch_source"),
                "bridge_source": cand.get("bridge_source"),
                "split": split,
                **metrics,
            }
            row["balanced_score"] = balanced_score(metrics)
            rows.append(row)
    selected_row = select_candidate(rows)
    selected_candidate = candidates[int(selected_row["candidate_index"])] if selected_row else None
    val_summary = summarize_rows(rows, "val")
    heldout_summary = summarize_rows(rows, "heldout")
    heldout_selected = next((r for r in rows if selected_row and r.get("candidate_index") == selected_row.get("candidate_index") and r.get("split") == "heldout"), None)
    # Baselines are the matching old anchors for selected candidate and the best anchor.
    anchors_heldout = [r for r in rows if r.get("split") == "heldout" and r.get("candidate_family") == "old_anchor_reexport"]
    best_anchor = max(anchors_heldout, key=lambda r: safe_float(r.get("balanced_score"), -1e9), default={})
    matching_anchor = next(
        (
            r
            for r in anchors_heldout
            if selected_row
            and r.get("anchor_group") == selected_row.get("anchor_group")
            and r.get("architecture") == selected_row.get("architecture")
        ),
        {},
    )
    selected_old_drop = safe_float(matching_anchor.get("old_acc"), 0.0) - safe_float((heldout_selected or {}).get("old_acc"), 0.0)
    selected_code_drop = safe_float(matching_anchor.get("code_acc"), 0.0) - safe_float((heldout_selected or {}).get("code_acc"), 0.0)
    branch_gain = safe_float((heldout_selected or {}).get("hidden_branch_acc"), 0.0) - safe_float(matching_anchor.get("hidden_branch_acc"), 0.0)
    bridge_gain = safe_float((heldout_selected or {}).get("bridge_acc"), 0.0) - safe_float(matching_anchor.get("bridge_acc"), 0.0)
    survival_gain = safe_float((heldout_selected or {}).get("survival_oracle_retention"), 0.0) - safe_float(matching_anchor.get("survival_oracle_retention"), 0.0)
    if heldout_selected and selected_old_drop <= 0.03 and selected_code_drop <= 0.05 and max(branch_gain, bridge_gain, survival_gain) > 0:
        status = "OLD_ANCHORED_BRANCH_TAP_USEFUL"
    elif heldout_selected and selected_old_drop <= 0.03 and selected_code_drop <= 0.05:
        status = "OLD_ANCHOR_PRESERVED_NO_BRANCH_GAIN"
    elif heldout_selected and (selected_old_drop > 0.03 or selected_code_drop > 0.05) and max(branch_gain, bridge_gain, survival_gain) > 0:
        status = "BRANCH_GAIN_WITH_OLD_DEGRADATION"
    else:
        status = "NO_USEFUL_TRANSPLANT"
    transplant_best = (heldout_summary.get("family_best") or {}).get("weight_space_transplant", {})
    constrained_best = (heldout_summary.get("family_best") or {}).get("constrained_finetune_copy", {})
    if safe_float(transplant_best.get("balanced_score"), -1e9) >= safe_float(constrained_best.get("balanced_score"), -1e9):
        best_family = "weight_space_transplant"
    else:
        best_family = "constrained_finetune_copy"
    result = {
        "BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS": status,
        "status": status,
        "best_family": best_family,
        "candidate_count": len(candidates),
        "transplant_candidate_count": len(transplant_candidates),
        "constrained_copy_candidate_count": len(constrained_candidates),
        "selected_validation_row": selected_row,
        "selected_heldout_row": heldout_selected,
        "matching_anchor_heldout": matching_anchor,
        "best_anchor_heldout": best_anchor,
        "heldout_deltas_vs_matching_anchor": {
            "old_acc_drop": selected_old_drop,
            "code_acc_drop": selected_code_drop,
            "hidden_branch_acc_gain": branch_gain,
            "bridge_acc_gain": bridge_gain,
            "survival_oracle_retention_gain": survival_gain,
        },
        "val_best": val_summary.get("best"),
        "heldout_best": heldout_summary.get("best"),
        "family_best_heldout": heldout_summary.get("family_best"),
        "anti_leakage": {
            "selection_split": "val",
            "heldout_used_for_selection": False,
            "selection_tie_break": "prefer MIX_CODE_REASONING weight-space transplant if within 0.04 validation balanced score of best eligible candidate; otherwise prefer MIX_CODE_REASONING within that band",
            "old_taps_overwritten": False,
            "tap_registry_updated": False,
            "ouro_weights_modified": False,
            "action_steering_tested": False,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(
        {
            "candidates": candidates,
            "selected": selected_candidate,
            "rows": rows,
            "summary": result,
        },
        CANDIDATES_PT,
    )
    write_json(RESULTS_JSON, {**result, "rows": rows, "candidates": [compact_candidate(c) for c in candidates]})
    write_csv(ROWS_CSV, rows)
    summary = {
        **result,
        "files_created": [rel(p) for p in [RESULTS_JSON, RESULTS_MD, ROWS_CSV, CANDIDATES_PT, SUMMARY_JSON, SUMMARY_MD, DOC_MD]],
    }
    write_json(SUMMARY_JSON, summary)
    lines = [
        "# Old-Anchored Branch-Valid Taps v1",
        "",
        f"BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = {status}",
        "",
        f"- candidates: `{len(candidates)}`",
        f"- weight-space transplant candidates: `{len(transplant_candidates)}`",
        f"- constrained fine-tuned copy candidates: `{len(constrained_candidates)}`",
        f"- selected: `{(selected_row or {}).get('candidate_name')}`",
        f"- selected family: `{(selected_row or {}).get('candidate_family')}`",
        f"- heldout old drop vs matching anchor: `{rate(selected_old_drop)}`",
        f"- heldout code drop vs matching anchor: `{rate(selected_code_drop)}`",
        f"- heldout hidden-branch gain vs matching anchor: `{rate(branch_gain)}`",
        f"- heldout bridge gain vs matching anchor: `{rate(bridge_gain)}`",
        f"- heldout survival-retention gain vs matching anchor: `{rate(survival_gain)}`",
        f"- best heldout family: `{best_family}`",
        "",
        "## Selected Heldout",
        "",
    ]
    lines.extend(md_table([heldout_selected or {}], ["candidate_name", "candidate_family", "anchor_group", "architecture", "old_acc", "code_acc", "hidden_branch_acc", "bridge_acc", "survival_oracle_retention", "balanced_score"]))
    lines.extend(["", "## Matching Old Anchor Heldout", ""])
    lines.extend(md_table([matching_anchor or {}], ["candidate_name", "candidate_family", "anchor_group", "architecture", "old_acc", "code_acc", "hidden_branch_acc", "bridge_acc", "survival_oracle_retention", "balanced_score"]))
    lines.extend(["", "## Family Best Heldout", ""])
    lines.extend(md_table(list((heldout_summary.get("family_best") or {}).values()), ["candidate_name", "candidate_family", "anchor_group", "architecture", "old_acc", "code_acc", "hidden_branch_acc", "bridge_acc", "survival_oracle_retention", "balanced_score"]))
    write_md(RESULTS_MD, lines)
    write_md(SUMMARY_MD, lines)
    doc_lines = [
        "# Old-Anchored Branch-Valid Taps v1",
        "",
        f"BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = {status}",
        "",
        "This run tested both requested routes: direct weight-space transplantation into old coding/reasoning and mixed-objective anchors, and constrained fine-tuning of copied old anchors. It wrote new tap artifacts only and did not overwrite old taps or registries.",
        "",
        "## Result",
        "",
        f"- selected: `{(selected_row or {}).get('candidate_name')}`",
        f"- selected family: `{(selected_row or {}).get('candidate_family')}`",
        f"- best heldout family: `{best_family}`",
        f"- old drop vs matching anchor: `{rate(selected_old_drop)}`",
        f"- code drop vs matching anchor: `{rate(selected_code_drop)}`",
        f"- hidden-branch gain vs matching anchor: `{rate(branch_gain)}`",
        f"- bridge gain vs matching anchor: `{rate(bridge_gain)}`",
        f"- survival-retention gain vs matching anchor: `{rate(survival_gain)}`",
        "",
        "## Interpretation",
        "",
        "The selected candidate is validation-selected and heldout-evaluated. It is useful only if branch/bridge/survival gain appears without material old/code degradation. It should be treated as a candidate expert or follow-up anchor, not a production route.",
        "",
        "## Files",
        "",
        f"- report: `{rel(RESULTS_MD)}`",
        f"- artifact: `{rel(CANDIDATES_PT)}`",
        f"- rows: `{rel(ROWS_CSV)}`",
    ]
    write_md(DOC_MD, doc_lines)
    section_title = "## Old-anchored branch-valid taps v1 (2026-05-30)"
    section = [
        section_title,
        "",
        f"`BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = {status}`. Tested weight-space transplant and constrained fine-tuned copies of old coding/reasoning and mixed-objective anchors. Selected `{(selected_row or {}).get('candidate_family')}` candidate `{(selected_row or {}).get('candidate_name')}` with heldout old/code drops `{rate(selected_old_drop)}` / `{rate(selected_code_drop)}` and branch/bridge gains `{rate(branch_gain)}` / `{rate(bridge_gain)}` vs its matching old anchor. No old taps, registries, Ouro weights, routing, or steering were modified.",
        "",
        f"Report: `{rel(DOC_MD)}`.",
    ]
    for target in DOC_TARGETS:
        append_section(target, section_title, section)
    nav = [
        section_title,
        "",
        f"Added `{rel(DOC_MD)}`. Status: `{status}`; selected family: `{(selected_row or {}).get('candidate_family')}`.",
    ]
    for target in NAV_TARGETS:
        append_section(target, section_title, nav)
    print(f"BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
