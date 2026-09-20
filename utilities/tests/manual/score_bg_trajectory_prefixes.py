"""Score trajectory prefixes with existing BG heads/configs."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from bg_trajectory_prediction_lib import REPORT_ROOT, load_task_suite, rel, task_by_id, write_json, write_md
from src.evaluator.bg_controller import (
    BGController,
    HEAD_CLASSES,
    PROJECT_ROOT,
    config_dim,
    config_vector,
    normalize_domain_hint,
)


OUT_JSON = REPORT_ROOT / "prefix_scores.json"
OUT_MD = REPORT_ROOT / "prefix_scores.md"
FEATURE_PT = REPORT_ROOT / "prefix_features.pt"
REGISTRY_PT = PROJECT_ROOT / "opi/taps/probes/bg_head_registry_2026-05-17.pt"
MIXED_PT = PROJECT_ROOT / "opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.pt"

TARGET_CONFIGS = {
    "24_L4",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_concat_L1_L4",
    "47_concat_all_loops",
}
TARGET_ARCHS = {"AntisymLinear", "AntisymLinearNoNorm"}


def _load_head_rows(path: Path, artifact_label: str) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for idx, row in enumerate(payload.get("heads") or []):
        config = str(row.get("config"))
        arch = str(row.get("architecture"))
        if config not in TARGET_CONFIGS or arch not in TARGET_ARCHS:
            continue
        dim = int(row.get("dim", config_dim(config)))
        if dim != config_dim(config):
            continue
        family = str(row.get("head_group") or row.get("head_family") or row.get("family") or "UNKNOWN")
        head_id = f"{artifact_label}::{family}::{config}::{arch}"
        rows.append(
            {
                "head_id": head_id,
                "family": family,
                "config": config,
                "architecture": arch,
                "artifact": rel(path),
                "row_index": idx,
                "dim": dim,
                "state_dict": row["state_dict"],
            }
        )
    return rows


def _build_head(row: dict[str, Any]) -> torch.nn.Module:
    head = HEAD_CLASSES[row["architecture"]](int(row["dim"]))
    head.load_state_dict(row["state_dict"])
    head.eval()
    return head


def _rank_matrix(mat: torch.Tensor) -> tuple[list[int], list[int], list[float]]:
    n = int(mat.shape[0])
    mask = ~torch.eye(n, dtype=torch.bool)
    wins = ((mat > 0) & mask).sum(dim=1).to(torch.int64)
    margin_sum = (mat * mask.to(mat.dtype)).sum(dim=1)
    ranking = sorted(range(n), key=lambda idx: (-int(wins[idx]), -float(margin_sum[idx]), idx))
    ranks = [0] * n
    for rank, idx in enumerate(ranking, start=1):
        ranks[idx] = rank
    return ranks, wins.tolist(), [float(x) for x in margin_sum.tolist()]


def _matrix_for_head(head: torch.nn.Module, config: str, features: list[torch.Tensor]) -> torch.Tensor:
    n = len(features)
    mat = torch.zeros((n, n), dtype=torch.float32)
    with torch.no_grad():
        for i in range(n):
            left = config_vector(features[i], config).to(dtype=torch.float32)
            for j in range(i + 1, n):
                right = config_vector(features[j], config).to(dtype=torch.float32)
                score = float(head(left, right).detach().cpu().item())
                mat[i, j] = score
                mat[j, i] = -score
    return mat


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    by_task = task_by_id(tasks)
    feature_payload = torch.load(FEATURE_PT, map_location="cpu", weights_only=False)
    records = list(feature_payload.get("records") or [])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["task_id"]), int(row["prefix_length"]))].append(row)

    controller = BGController.from_artifacts(device="cpu")
    locked_specs = []
    for name, spec in controller.specs.items():
        locked_specs.append(
            {
                "head_id": f"locked::{name}",
                "family": spec.family,
                "config": spec.config,
                "architecture": spec.architecture,
                "artifact": spec.artifact_path,
                "row_index": None,
                "dim": spec.dim,
                "locked_role": name,
                "module": controller.heads[name],
            }
        )
    artifact_specs = _load_head_rows(REGISTRY_PT, "registry") + _load_head_rows(MIXED_PT, "mixed")
    head_specs = locked_specs + artifact_specs
    score_rows = []
    pairwise_rows = []
    group_count = 0
    for (task_id, prefix_length), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        task = by_task.get(task_id)
        if task is None:
            continue
        rows = sorted(rows, key=lambda r: int(r["branch_id"]))
        if len(rows) < 2:
            continue
        group_count += 1
        branch_ids = [int(row["branch_id"]) for row in rows]
        features = [row["features"] for row in rows]
        for spec in head_specs:
            if "module" in spec:
                head = spec["module"]
            else:
                head = _build_head(spec)
            mat = _matrix_for_head(head, spec["config"], features)
            ranks, wins, margin_sum = _rank_matrix(mat)
            for i, left_branch in enumerate(branch_ids):
                for j in range(i + 1, len(branch_ids)):
                    pairwise_rows.append(
                        {
                            "task_id": task_id,
                            "domain": task["domain"],
                            "prefix_length": int(prefix_length),
                            "left_branch_id": int(left_branch),
                            "right_branch_id": int(branch_ids[j]),
                            "score_left_beats_right": float(mat[i, j].item()),
                            "head_id": spec["head_id"],
                            "family": spec["family"],
                            "config": spec["config"],
                            "architecture": spec["architecture"],
                        }
                    )
            for idx, branch_id in enumerate(branch_ids):
                score_rows.append(
                    {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "prefix_length": int(prefix_length),
                        "branch_id": int(branch_id),
                        "head_id": spec["head_id"],
                        "family": spec["family"],
                        "config": spec["config"],
                        "architecture": spec["architecture"],
                        "rank": int(ranks[idx]),
                        "wins": int(wins[idx]),
                        "margin_sum": float(margin_sum[idx]),
                        "mean_pairwise_margin": float(margin_sum[idx]) / max(len(rows) - 1, 1),
                        "branch_count": len(rows),
                        "domain_hint": normalize_domain_hint(task["domain"]),
                    }
                )
            if "module" not in spec:
                del head

    if score_rows:
        verdict = "READY"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_TRAJECTORY_PREFIX_SCORE_VERDICT": verdict,
        "verdict": verdict,
        "score_row_count": len(score_rows),
        "pairwise_score_row_count": len(pairwise_rows),
        "scored_task_prefix_groups": group_count,
        "head_count": len(head_specs),
        "head_specs": [
            {k: v for k, v in spec.items() if k not in {"state_dict", "module"}}
            for spec in head_specs
        ],
        "counts_by_domain": dict(Counter(row["domain"] for row in score_rows)),
        "prefix_scores": score_rows,
        "pairwise_scores": pairwise_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Trajectory Prefix Scores (2026-05-18)",
        "",
        f"BG_TRAJECTORY_PREFIX_SCORE_VERDICT = {verdict}",
        "",
        f"- score_row_count: `{len(score_rows)}`",
        f"- pairwise_score_row_count: `{len(pairwise_rows)}`",
        f"- scored_task_prefix_groups: `{group_count}`",
        f"- head_count: `{len(head_specs)}`",
        f"- counts_by_domain: `{payload['counts_by_domain']}`",
        "",
        "## Heads",
        "",
        "| Head | Family | Config | Architecture | Artifact |",
        "|---|---|---|---|---|",
    ]
    for spec in payload["head_specs"][:80]:
        lines.append(f"| `{spec['head_id']}` | `{spec['family']}` | `{spec['config']}` | `{spec['architecture']}` | `{spec.get('artifact')}` |")
    if len(payload["head_specs"]) > 80:
        lines.append(f"| ... | ... | ... | ... | `{len(payload['head_specs']) - 80} more` |")
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_PREFIX_SCORE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
