"""Build v4 hidden-origin branch-generation direction bank with heldout guard."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_hidden_origin_quota_v4_common import (
    CONFIGS,
    DIRECTION_BANK_V4_JSON,
    DIRECTION_BANK_V4_MD,
    DIRECTION_BANK_V4_PT,
    HIDDEN_DIM,
    PROBE_ROOT,
    SALVAGE_HEADS_PT,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    compact_direction_entry,
    config_dim,
    config_layer,
    config_vector_from_row,
    cosine,
    direction_entry,
    direction_from_state_dict,
    ensure_v4_root,
    group_rows,
    load_all_v3_branch_rows,
    load_head_rows,
    load_pt,
    load_quota_plan,
    md_table,
    primary_safe_v3_row,
    rel,
    rms_normalize,
    split_task_ids,
    tensor_stats,
    write_json,
    write_md,
)


def add_random_entries(entries: list[dict[str, Any]], heldout_ids: set[str]) -> None:
    for layer in (24, 36, 47):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260518 + 4000 + layer)
        basis: list[torch.Tensor] = []
        for idx in range(6):
            raw = torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32)
            for base in basis:
                unit = rms_normalize(base)
                raw = raw - torch.dot(raw, unit) / torch.dot(unit, unit).clamp(min=1e-8) * unit
            vec = rms_normalize(raw)
            basis.append(vec)
            entries.append(
                direction_entry(
                    name=f"random_orthogonal:L{layer}:v4:idx={idx}",
                    family="random_orthogonal",
                    tensor=vec,
                    source="deterministic_random_v4",
                    target_layer=layer,
                    count=1,
                    heldout_task_ids=heldout_ids,
                )
            )


def task_ids_from_dataset(path: Any, splits: tuple[str, ...] = ("train", "val", "test", "heldout")) -> list[str]:
    payload = load_pt(path, {}) or {}
    task_sets = payload.get("tasks_by_split") or {}
    ids: set[str] = set()
    for split in splits:
        ids.update(str(x) for x in task_sets.get(split, []))
    return sorted(ids)


def add_head_entries(entries: list[dict[str, Any]], heldout_ids: set[str]) -> None:
    sources = [
        ("old_tap_aligned", PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt", None, []),
        ("old_tap_aligned", PROBE_ROOT / "bg_head_registry_2026-05-17.pt", None, []),
        ("v1_tap_aligned", V1_ROOT / "hidden_origin_tap_heads.pt", None, task_ids_from_dataset(V1_ROOT / "hidden_origin_tap_dataset.pt", ("train", "val"))),
        ("v2_tap_aligned", V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic", task_ids_from_dataset(V2_ROOT / "hidden_origin_tap_dataset_v2.pt", ("train", "val"))),
        ("v3_tap_aligned", V3_ROOT / "hidden_origin_tap_heads_v3.pt", "primary_safe_deterministic", task_ids_from_dataset(V3_ROOT / "hidden_origin_tap_dataset_v3.pt", ("train", "val"))),
        ("salvage_tap_aligned", SALVAGE_HEADS_PT, None, []),
    ]
    for family, path, variant, trained_task_ids in sources:
        for row in load_head_rows(path, variant=variant, only_passing=True):
            direction = row.get("direction")
            if not isinstance(direction, torch.Tensor):
                direction = direction_from_state_dict(row.get("state_dict") or {})
            if not isinstance(direction, torch.Tensor):
                continue
            config = str(row.get("config") or "")
            layer = config_layer(config)
            usable_layer = layer if int(direction.numel()) == HIDDEN_DIM else None
            entries.append(
                direction_entry(
                    name=f"{family}:{config}:{row.get('architecture')}:{row.get('variant') or row.get('head_group') or 'head'}",
                    family=family,
                    tensor=direction,
                    source=rel(path),
                    target_layer=usable_layer,
                    target_config=config,
                    count=1,
                    trained_on_task_ids=trained_task_ids,
                    heldout_task_ids=heldout_ids,
                )
            )


def add_prior_v3_bank_entries(entries: list[dict[str, Any]], heldout_ids: set[str]) -> None:
    path = V3_ROOT / "direction_bank_v3.pt"
    payload = load_pt(path, {}) or {}
    for row in list(payload.get("directions") or []):
        tensor = row.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            continue
        family = str(row.get("family") or "v3_bank")
        translated = {
            "hidden_origin_empirical": "hidden_origin_empirical_train_only",
            "hidden_origin_whitened": "hidden_origin_whitened_train_only",
        }.get(family, family)
        entries.append(
            direction_entry(
                name=f"v3_bank:{row.get('name')}",
                family=translated,
                tensor=tensor,
                source=rel(path),
                target_layer=int(row["target_layer"]) if row.get("target_layer") is not None and int(tensor.numel()) == HIDDEN_DIM else None,
                target_config=row.get("target_config"),
                count=int(row.get("count") or 0),
                trained_on_task_ids=row.get("trained_on_task_ids") or [],
                heldout_task_ids=heldout_ids,
                recommended_alpha_bucket=str(row.get("recommended_alpha_bucket") or "alpha_0_01"),
            )
        )


def add_empirical_entries(entries: list[dict[str, Any]], train_ids: set[str], heldout_ids: set[str]) -> None:
    rows = [
        row
        for row in load_all_v3_branch_rows(include_prior=True)
        if primary_safe_v3_row(row) and str(row.get("task_id")) in train_ids
    ]
    diffs_by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    vals_by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    for vals in group_rows(rows).values():
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        for row in ordered:
            for cfg in CONFIGS:
                vec = config_vector_from_row(row, cfg)
                if isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(cfg):
                    vals_by_config[cfg].append(vec.detach().cpu().to(torch.float32))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                ra = float(a.get("deterministic_reward", a.get("reward", 0.0)))
                rb = float(b.get("deterministic_reward", b.get("reward", 0.0)))
                if ra == rb:
                    continue
                pref, rej = (a, b) if ra > rb else (b, a)
                for cfg in CONFIGS:
                    left = config_vector_from_row(pref, cfg)
                    right = config_vector_from_row(rej, cfg)
                    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and int(left.numel()) == config_dim(cfg) and int(right.numel()) == config_dim(cfg):
                        diffs_by_config[cfg].append(left.detach().cpu().to(torch.float32) - right.detach().cpu().to(torch.float32))
    for cfg, diffs in sorted(diffs_by_config.items()):
        if not diffs:
            continue
        mean_diff = torch.stack(diffs, dim=0).mean(dim=0)
        layer = config_layer(cfg)
        usable_layer = layer if int(mean_diff.numel()) == HIDDEN_DIM else None
        entries.append(
            direction_entry(
                name=f"hidden_origin_empirical_train_only:{cfg}",
                family="hidden_origin_empirical_train_only",
                tensor=mean_diff,
                source="v4_train_only_prior_hidden_origin_pairs",
                target_layer=usable_layer,
                target_config=cfg,
                count=len(diffs),
                trained_on_task_ids=sorted(train_ids),
                heldout_task_ids=heldout_ids,
            )
        )
        vals = vals_by_config.get(cfg) or []
        if len(diffs) >= 20 and len(vals) >= 20:
            std = torch.stack(vals, dim=0).std(dim=0, unbiased=False).clamp(min=1e-4)
            entries.append(
                direction_entry(
                    name=f"hidden_origin_whitened_train_only:{cfg}",
                    family="hidden_origin_whitened_train_only",
                    tensor=mean_diff / std,
                    source="v4_train_only_prior_hidden_origin_pairs",
                    target_layer=usable_layer,
                    target_config=cfg,
                    count=len(diffs),
                    trained_on_task_ids=sorted(train_ids),
                    heldout_task_ids=heldout_ids,
                    recommended_alpha_bucket="alpha_0_005",
                )
            )


def add_adapter_entries(entries: list[dict[str, Any]], heldout_ids: set[str]) -> None:
    paths = [
        (PROBE_ROOT / "bg_empirical_steering_direction_2026-05-18/directions.pt", "adapter_proxy"),
        (PROBE_ROOT / "bg_sequence_adapter_2026-05-18/sequence_adapter.pt", "sequence_adapter_proxy"),
        (PROBE_ROOT / "bg_sequence_level_adapter_2026-05-18/sequence_adapter.pt", "sequence_adapter_proxy"),
    ]
    for path, family in paths:
        payload = load_pt(path, None)
        if payload is None:
            continue
        stack = [("", payload)]
        added = 0
        while stack and added < 16:
            name, value = stack.pop()
            if isinstance(value, torch.Tensor) and int(value.numel()) in {HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 3}:
                lname = name.lower()
                if int(value.numel()) != HIDDEN_DIM:
                    layer = None
                elif "36" in lname:
                    layer = 36
                elif "47" in lname:
                    layer = 47
                else:
                    layer = 24
                entries.append(
                    direction_entry(
                        name=f"{family}:{name or 'tensor'}",
                        family=family,
                        tensor=value,
                        source=rel(path),
                        target_layer=layer,
                        target_config=f"L{layer}" if layer else None,
                        count=1,
                        heldout_task_ids=heldout_ids,
                        recommended_alpha_bucket="alpha_0_005",
                    )
                )
                added += 1
            elif isinstance(value, dict):
                for key, val in value.items():
                    stack.append((f"{name}.{key}" if name else str(key), val))
            elif isinstance(value, (list, tuple)):
                for idx, val in enumerate(value):
                    stack.append((f"{name}[{idx}]", val))


def attach_alignment(entries: list[dict[str, Any]]) -> None:
    refs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in entries:
        if row["family"] in {"old_tap_aligned", "v1_tap_aligned", "v2_tap_aligned", "v3_tap_aligned", "salvage_tap_aligned"}:
            refs[row["family"]].append(row["tensor"])
    for row in entries:
        tensor = row["tensor"]
        for label, items in refs.items():
            vals = [abs(cosine(tensor, ref)) for ref in items if int(ref.numel()) == int(tensor.numel())]
            finite = [v for v in vals if isinstance(v, float) and v == v]
            row[f"max_abs_cosine_to_{label}"] = max(finite) if finite else float("nan")


def family_status(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for family in sorted({row["family"] for row in entries}):
        vals = [row for row in entries if row["family"] == family]
        usable = [row for row in vals if row.get("perturbation_usable")]
        out[family] = {
            "entries": len(vals),
            "perturbation_usable_entries": len(usable),
            "heldout_leakage_free_entries": sum(1 for row in vals if row.get("heldout_leakage_free")),
            "layers": sorted({row.get("target_layer") for row in usable if row.get("target_layer") is not None}),
            "status": "ready" if usable else "weak" if vals else "unavailable",
        }
    return out


def main() -> int:
    started = time.time()
    ensure_v4_root()
    plan = load_quota_plan()
    if plan.get("verdict") == "BLOCKED" or not plan:
        payload = {"BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing or blocked quota plan"}
        write_json(DIRECTION_BANK_V4_JSON, payload)
        write_md(DIRECTION_BANK_V4_MD, ["# Hidden-Origin Direction Bank V4", "", "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT = BLOCKED", flush=True)
        return 1
    train_ids = split_task_ids(plan, "train")
    heldout_ids = split_task_ids(plan, "heldout")
    entries: list[dict[str, Any]] = []
    add_random_entries(entries, heldout_ids)
    add_head_entries(entries, heldout_ids)
    add_prior_v3_bank_entries(entries, heldout_ids)
    add_empirical_entries(entries, train_ids, heldout_ids)
    add_adapter_entries(entries, heldout_ids)
    attach_alignment(entries)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        if row.get("perturbation_usable") and row.get("target_layer") in {24, 36, 47}:
            by_layer[str(int(row["target_layer"]))].append(row)
    statuses = family_status(entries)
    ready_non_random = [
        family
        for family, info in statuses.items()
        if family != "random_orthogonal" and int(info["perturbation_usable_entries"]) > 0
    ]
    if not any(row["family"] == "random_orthogonal" and row.get("perturbation_usable") for row in entries):
        verdict = "BLOCKED"
    elif len(ready_non_random) >= 3:
        verdict = "READY"
    elif len(ready_non_random) >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "RANDOM_ONLY"
    payload = {
        "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT": verdict,
        "verdict": verdict,
        "directions": entries,
        "directions_by_layer": dict(by_layer),
        "family_counts": dict(Counter(row["family"] for row in entries)),
        "family_status": statuses,
        "ready_non_random_heldout_clean_families": ready_non_random,
        "train_task_ids_used_for_empirical_directions": sorted(train_ids),
        "heldout_task_ids_excluded_from_empirical_directions": sorted(heldout_ids),
        "direction_compatibility_guard": {
            "concat_projection_forbidden_without_validated_projection": True,
            "single_layer_dim_required_for_perturbation": HIDDEN_DIM,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, DIRECTION_BANK_V4_PT)
    compact_payload = {k: v for k, v in payload.items() if k not in {"directions", "directions_by_layer"}}
    compact_payload["directions"] = [compact_direction_entry(row) for row in entries]
    compact_payload["directions_by_layer"] = {layer: [compact_direction_entry(row) for row in vals] for layer, vals in by_layer.items()}
    write_json(DIRECTION_BANK_V4_JSON, compact_payload)
    direction_rows = [
        {
            "family": row["family"],
            "name": str(row["name"])[:80],
            "layer": row.get("target_layer"),
            "config": row.get("target_config"),
            "dim": row["direction_dim"],
            "usable": row["perturbation_usable"],
            "scoring_only": row["scoring_only"],
            "heldout_clean": row["heldout_leakage_free"],
            "count": row.get("count"),
        }
        for row in entries
    ]
    direction_rows.sort(key=lambda row: (str(row["family"]), str(row["layer"]), str(row["name"])))
    lines = [
        "# Hidden-Origin Direction Bank V4",
        "",
        f"BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT = {verdict}",
        "",
        f"- ready_non_random_heldout_clean_families: `{ready_non_random}`",
        f"- family_counts: `{payload['family_counts']}`",
        f"- layers: `{sorted(by_layer)}`",
        f"- empirical train task IDs: `{len(train_ids)}`",
        f"- heldout excluded task IDs: `{len(heldout_ids)}`",
        "",
        "Concat directions are kept scoring-only unless they already match a single target layer. No projection is trained or inferred.",
        "",
        "## Family Status",
        "",
    ]
    lines.extend(md_table([{"family": k, **v} for k, v in sorted(statuses.items())], ["family", "entries", "perturbation_usable_entries", "heldout_leakage_free_entries", "layers", "status"]))
    lines.extend(["", "## Directions", ""])
    lines.extend(md_table(direction_rows[:260], ["family", "name", "layer", "config", "dim", "usable", "scoring_only", "heldout_clean", "count"]))
    lines.extend(["", "## Outputs", "", f"- PT: `{rel(DIRECTION_BANK_V4_PT)}`", f"- JSON: `{rel(DIRECTION_BANK_V4_JSON)}`"])
    write_md(DIRECTION_BANK_V4_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(DIRECTION_BANK_V4_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
