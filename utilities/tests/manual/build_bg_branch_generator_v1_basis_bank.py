"""Build heldout-clean basis/direction bank for Branch Generator v1."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_branch_generator_v1_common import (
    AUDIT_PLAN_JSON,
    BASIS_BANK_JSON,
    BASIS_BANK_MD,
    BASIS_BANK_PT,
    HEADS_V4_PT,
    HIDDEN_DIM,
    compact_direction_entry,
    direction_entry,
    direction_from_state_dict,
    ensure_bgv1_root,
    load_json,
    load_head_rows,
    md_table,
    rel,
    split_task_ids,
    tensor_stats,
    write_json,
    write_md,
)
from build_bg_hidden_origin_direction_bank_v4 import (
    add_adapter_entries,
    add_empirical_entries,
    add_head_entries,
    add_prior_v3_bank_entries,
    add_random_entries,
    attach_alignment,
    family_status,
)


def add_v4_head_entries(entries: list[dict[str, Any]], heldout_ids: set[str], trained_task_ids: list[str]) -> None:
    for row in load_head_rows(HEADS_V4_PT, variant="v4_only_primary_safe", only_passing=True):
        direction = row.get("direction")
        if not isinstance(direction, torch.Tensor):
            direction = direction_from_state_dict(row.get("state_dict") or {})
        if not isinstance(direction, torch.Tensor):
            continue
        config = str(row.get("config") or "")
        layer = None
        if int(direction.numel()) == HIDDEN_DIM:
            if config.startswith("24_"):
                layer = 24
            elif config.startswith("36_"):
                layer = 36
            elif config.startswith("47_"):
                layer = 47
        entries.append(
            direction_entry(
                name=f"v4_tap_aligned:{config}:{row.get('architecture')}:{row.get('variant') or 'head'}",
                family="v4_tap_aligned",
                tensor=direction,
                source=rel(HEADS_V4_PT),
                target_layer=layer,
                target_config=config,
                count=1,
                trained_on_task_ids=trained_task_ids,
                heldout_task_ids=heldout_ids,
                recommended_alpha_bucket="alpha_0_005",
            )
        )


def compact_basis_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = [row["tensor"].detach().cpu().flatten().to(torch.float32) for row in rows if isinstance(row.get("tensor"), torch.Tensor)]
    if not tensors:
        return {"vector_count": 0}
    mat = torch.stack(tensors, dim=0)
    return {
        "vector_count": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "matrix_stats": tensor_stats(mat),
        "families": dict(Counter(str(row.get("family")) for row in rows)),
    }


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    plan = load_json(AUDIT_PLAN_JSON, {}) or {}
    if plan.get("verdict") == "BLOCKED" or not plan:
        payload = {"BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing or blocked audit plan"}
        write_json(BASIS_BANK_JSON, payload)
        write_md(BASIS_BANK_MD, ["# Branch Generator V1 Basis Bank", "", "BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = BLOCKED"])
        print("BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = BLOCKED", flush=True)
        return 1
    train_ids = set(split_task_ids(plan, "train"))
    heldout_ids = set(split_task_ids(plan, "heldout"))
    val_ids = set(split_task_ids(plan, "val"))
    entries: list[dict[str, Any]] = []
    add_random_entries(entries, heldout_ids)
    add_head_entries(entries, heldout_ids)
    add_v4_head_entries(entries, heldout_ids, sorted(train_ids))
    add_prior_v3_bank_entries(entries, heldout_ids)
    add_empirical_entries(entries, train_ids, heldout_ids)
    add_adapter_entries(entries, heldout_ids)
    # Duplicate metadata tags for the v4 high-yield path without duplicating tensors.
    for row in list(entries):
        if row.get("family") in {"old_tap_aligned", "v2_tap_aligned", "v3_tap_aligned", "salvage_tap_aligned"} and row.get("target_layer") in {24, 36}:
            cloned = dict(row)
            cloned["name"] = f"high_yield_recipe_direction:{row.get('name')}"
            cloned["family"] = "high_yield_recipe_direction"
            cloned["source"] = f"v4_driver_l24_l36_alpha_0_005::{row.get('source')}"
            cloned["recommended_alpha_bucket"] = "alpha_0_005"
            entries.append(cloned)
    attach_alignment(entries)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        if row.get("perturbation_usable") and row.get("target_layer") in {24, 36, 47}:
            by_layer[str(int(row["target_layer"]))].append(row)
    basis_matrices = {layer: compact_basis_matrix(vals) for layer, vals in sorted(by_layer.items())}
    statuses = family_status(entries)
    ready_non_random = [
        family
        for family, info in statuses.items()
        if family != "random_orthogonal" and int(info.get("perturbation_usable_entries") or 0) > 0
    ]
    if not any(row.get("family") == "random_orthogonal" and row.get("perturbation_usable") for row in entries):
        verdict = "BLOCKED"
    elif len(ready_non_random) >= 3:
        verdict = "READY"
    elif len(ready_non_random) >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "RANDOM_ONLY"
    payload = {
        "BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT": verdict,
        "verdict": verdict,
        "directions": entries,
        "directions_by_layer": dict(by_layer),
        "basis_matrices": basis_matrices,
        "family_counts": dict(Counter(row["family"] for row in entries)),
        "family_status": statuses,
        "ready_non_random_heldout_clean_families": ready_non_random,
        "train_task_ids_used_for_empirical_directions": sorted(train_ids),
        "val_task_ids_excluded_from_empirical_directions": sorted(val_ids),
        "heldout_task_ids_excluded_from_empirical_directions": sorted(heldout_ids),
        "direction_compatibility_guard": {
            "concat_projection_forbidden_without_validated_projection": True,
            "single_layer_dim_required_for_perturbation": HIDDEN_DIM,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, BASIS_BANK_PT)
    compact_payload = {k: v for k, v in payload.items() if k not in {"directions", "directions_by_layer"}}
    compact_payload["directions"] = [compact_direction_entry(row) for row in entries]
    compact_payload["directions_by_layer"] = {layer: [compact_direction_entry(row) for row in vals] for layer, vals in by_layer.items()}
    write_json(BASIS_BANK_JSON, compact_payload)
    direction_rows = [
        {
            "family": row["family"],
            "name": str(row["name"])[:80],
            "layer": row.get("target_layer"),
            "config": row.get("target_config"),
            "dim": row["direction_dim"],
            "usable": row["perturbation_usable"],
            "heldout_clean": row["heldout_leakage_free"],
            "count": row.get("count"),
        }
        for row in entries
    ]
    direction_rows.sort(key=lambda row: (str(row["family"]), str(row["layer"]), str(row["name"])))
    lines = [
        "# Branch Generator V1 Basis Bank",
        "",
        f"BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = {verdict}",
        "",
        f"- ready_non_random_heldout_clean_families: `{ready_non_random}`",
        f"- family_counts: `{payload['family_counts']}`",
        f"- basis_matrices: `{basis_matrices}`",
        "",
        "Concat directions are scoring-only unless they already match a single target layer. No projection is trained or inferred.",
        "",
        "## Family Status",
        "",
    ]
    lines.extend(md_table([{"family": k, **v} for k, v in sorted(statuses.items())], ["family", "entries", "perturbation_usable_entries", "heldout_leakage_free_entries", "layers", "status"]))
    lines.extend(["", "## Directions", ""])
    lines.extend(md_table(direction_rows[:320], ["family", "name", "layer", "config", "dim", "usable", "heldout_clean", "count"]))
    lines.extend(["", "## Outputs", "", f"- PT: `{rel(BASIS_BANK_PT)}`", f"- JSON: `{rel(BASIS_BANK_JSON)}`"])
    write_md(BASIS_BANK_MD, lines)
    print(f"BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(BASIS_BANK_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
