"""Build v3 branch-generation direction bank with leakage/compatibility guards."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import torch

from bg_hidden_origin_diversity_v3_common import (
    CONFIGS,
    DIRECTION_BANK_V3_PT,
    HIDDEN_DIM,
    PROBE_ROOT,
    SEED,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    available_configs_for_rows,
    branch_group_metrics,
    config_dim,
    config_layer,
    config_vector_from_row,
    cosine,
    deterministic_reward,
    direction_from_state_dict,
    ensure_v3_root,
    group_rows,
    load_all_v3_branch_rows,
    load_head_rows,
    load_json,
    md_table,
    primary_safe_v3_row,
    rel,
    rms_normalize,
    stable_v2_row,
    tensor_stats,
    write_json,
    write_md,
)


OUT_PT = DIRECTION_BANK_V3_PT
OUT_JSON = V3_ROOT / "direction_bank_v3.json"
OUT_MD = V3_ROOT / "direction_bank_v3.md"


def compatible_branch_points(layer: int | None) -> list[str]:
    if layer == 24:
        return ["L24"]
    if layer == 36:
        return ["L36"]
    if layer == 47:
        return ["L47"]
    return []


def entry(
    *,
    name: str,
    family: str,
    tensor: torch.Tensor,
    source: str,
    target_layer: int | None,
    target_config: str | None = None,
    count: int = 0,
    recommended_alpha_bucket: str = "alpha_0_01",
) -> dict[str, Any]:
    vec = tensor.detach().cpu().flatten().to(torch.float32)
    usable = target_layer in {24, 36, 47} and int(vec.numel()) == HIDDEN_DIM
    return {
        "name": name,
        "family": family,
        "source": source,
        "tensor": rms_normalize(vec) if int(vec.numel()) > 0 else vec,
        "direction_dim": int(vec.numel()),
        "target_layer": int(target_layer) if target_layer is not None else None,
        "target_config": target_config,
        "compatible_branch_points": compatible_branch_points(target_layer),
        "perturbation_usable": bool(usable),
        "scoring_only": not bool(usable),
        "count": int(count),
        "recommended_alpha_bucket": recommended_alpha_bucket,
        "stats": tensor_stats(vec) if int(vec.numel()) else {},
    }


def add_random_entries(entries: list[dict[str, Any]]) -> None:
    for layer in (24, 36, 47):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + 1000 + layer)
        basis: list[torch.Tensor] = []
        for idx in range(4):
            raw = torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32)
            for base in basis:
                unit = rms_normalize(base)
                raw = raw - torch.dot(raw, unit) / torch.dot(unit, unit).clamp(min=1e-8) * unit
            vec = rms_normalize(raw)
            basis.append(vec)
            entries.append(
                entry(
                    name=f"random_orthogonal:L{layer}:seed={SEED + 1000 + layer}:idx={idx}",
                    family="random_orthogonal",
                    tensor=vec,
                    source="deterministic_random_v3",
                    target_layer=layer,
                    count=1,
                    recommended_alpha_bucket="alpha_0_01",
                )
            )


def add_head_entries(entries: list[dict[str, Any]]) -> None:
    sources = [
        ("old_tap_aligned", PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt", None),
        ("old_tap_aligned", PROBE_ROOT / "bg_head_registry_2026-05-17.pt", None),
        ("v1_tap_aligned", V1_ROOT / "hidden_origin_tap_heads.pt", None),
        ("v2_tap_aligned", V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic"),
    ]
    for family, path, variant in sources:
        for row in load_head_rows(path, variant=variant, only_passing=True):
            direction = row.get("direction")
            if not isinstance(direction, torch.Tensor):
                direction = direction_from_state_dict(row.get("state_dict") or {})
            if not isinstance(direction, torch.Tensor):
                continue
            config = str(row.get("config") or "")
            layer = config_layer(config)
            dim = int(direction.numel())
            # Concat heads are intentionally kept scoring-only unless a validated
            # projection exists.  The target layer is recorded only for single-layer
            # configs.
            usable_layer = layer if dim == HIDDEN_DIM else None
            entries.append(
                entry(
                    name=f"{family}:{config}:{row.get('architecture')}:{row.get('head_group') or row.get('variant') or 'head'}",
                    family=family,
                    tensor=direction,
                    source=rel(path),
                    target_layer=usable_layer,
                    target_config=config,
                    count=1,
                    recommended_alpha_bucket="alpha_0_01",
                )
            )


def empirical_entries(entries: list[dict[str, Any]], train_eligible_task_ids: set[str]) -> None:
    rows = [
        row
        for row in load_all_v3_branch_rows(include_prior=True)
        if primary_safe_v3_row(row) and str(row.get("task_id")) in train_eligible_task_ids
    ]
    diffs_by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    values_by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    for vals in group_rows(rows).values():
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        if len(ordered) < 2:
            continue
        for row in ordered:
            for cfg in CONFIGS:
                vec = config_vector_from_row(row, cfg)
                if isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(cfg):
                    values_by_config[cfg].append(vec.detach().cpu().to(torch.float32))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                ra = deterministic_reward(a)
                rb = deterministic_reward(b)
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
        layer = config_layer(cfg)
        mean_diff = torch.stack(diffs, dim=0).mean(dim=0)
        usable_layer = layer if int(mean_diff.numel()) == HIDDEN_DIM else None
        entries.append(
            entry(
                name=f"hidden_origin_empirical_mean_diff:{cfg}",
                family="hidden_origin_empirical",
                tensor=mean_diff,
                source="v1_v2_train_eligible_hidden_origin_pairs",
                target_layer=usable_layer,
                target_config=cfg,
                count=len(diffs),
                recommended_alpha_bucket="alpha_0_01",
            )
        )
        vals = values_by_config.get(cfg) or []
        if len(diffs) >= 20 and len(vals) >= 20:
            std = torch.stack(vals, dim=0).std(dim=0, unbiased=False).clamp(min=1e-4)
            entries.append(
                entry(
                    name=f"hidden_origin_whitened_mean_diff:{cfg}",
                    family="hidden_origin_whitened",
                    tensor=mean_diff / std,
                    source="v1_v2_train_eligible_hidden_origin_pairs",
                    target_layer=usable_layer,
                    target_config=cfg,
                    count=len(diffs),
                    recommended_alpha_bucket="alpha_0_005",
                )
            )


def find_adapter_tensors() -> list[tuple[str, str, torch.Tensor, str]]:
    out: list[tuple[str, str, torch.Tensor, str]] = []
    paths = [
        (PROBE_ROOT / "bg_empirical_steering_direction_2026-05-18/directions.pt", "adapter_proxy"),
        (PROBE_ROOT / "bg_sequence_adapter_2026-05-18/sequence_adapter.pt", "sequence_adapter_proxy"),
        (PROBE_ROOT / "bg_sequence_level_adapter_2026-05-18/sequence_adapter.pt", "sequence_adapter_proxy"),
    ]
    for path, family in paths:
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        stack = [("", payload)]
        while stack:
            name, value = stack.pop()
            if isinstance(value, torch.Tensor) and int(value.numel()) in {HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 3}:
                out.append((rel(path), name or "tensor", value.detach().cpu().to(torch.float32), family))
            elif isinstance(value, dict):
                for key, val in value.items():
                    stack.append((f"{name}.{key}" if name else str(key), val))
            elif isinstance(value, (list, tuple)):
                for idx, val in enumerate(value):
                    stack.append((f"{name}[{idx}]", val))
    return out[:24]


def add_adapter_entries(entries: list[dict[str, Any]]) -> None:
    for source, name, tensor, family in find_adapter_tensors():
        lname = name.lower()
        if int(tensor.numel()) != HIDDEN_DIM:
            layer = None
        elif "24" in lname:
            layer = 24
        elif "36" in lname:
            layer = 36
        elif "47" in lname:
            layer = 47
        else:
            layer = 24
        entries.append(
            entry(
                name=f"{family}:{name}",
                family=family,
                tensor=tensor,
                source=source,
                target_layer=layer,
                target_config=f"L{layer}" if layer else None,
                count=1,
                recommended_alpha_bucket="alpha_0_005",
            )
        )
        if family == "adapter_proxy" and layer == 24 and int(tensor.numel()) == HIDDEN_DIM:
            entries.append(
                entry(
                    name=f"{family}:{name}:L36_copy_diagnostic",
                    family=family,
                    tensor=tensor,
                    source=source,
                    target_layer=36,
                    target_config="L36",
                    count=1,
                    recommended_alpha_bucket="alpha_0_005",
                )
            )


def add_v2_bank_entries(entries: list[dict[str, Any]]) -> None:
    path = V2_ROOT / "direction_bank.pt"
    if not path.exists():
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return
    for layer_key, vals in (payload.get("directions_by_layer") or {}).items():
        try:
            layer = int(layer_key)
        except Exception:
            continue
        for row in list(vals or []):
            tensor = row.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                continue
            family = str(row.get("family") or "v2_bank")
            translated = {
                "v1_hidden_origin_tap": "v1_tap_aligned",
                "old_tap_aligned": "old_tap_aligned",
                "hidden_origin_empirical": "hidden_origin_empirical",
                "hidden_origin_whitened": "hidden_origin_whitened",
                "adapter_proxy": "adapter_proxy",
            }.get(family, family)
            entries.append(
                entry(
                    name=f"v2_bank:{row.get('name')}",
                    family=translated,
                    tensor=tensor,
                    source=rel(path),
                    target_layer=layer if int(tensor.numel()) == HIDDEN_DIM else None,
                    target_config=f"L{layer}",
                    count=int(row.get("count") or 0),
                    recommended_alpha_bucket="alpha_0_01",
                )
            )


def reference_dirs(entries: list[dict[str, Any]]) -> dict[str, list[torch.Tensor]]:
    refs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in entries:
        if row["family"] in {"old_tap_aligned", "v1_tap_aligned", "v2_tap_aligned"}:
            refs[row["family"]].append(row["tensor"])
    return refs


def attach_alignment(entries: list[dict[str, Any]]) -> None:
    refs = reference_dirs(entries)
    for row in entries:
        tensor = row["tensor"]
        for label, items in refs.items():
            vals = [abs(cosine(tensor, ref)) for ref in items if int(ref.numel()) == int(tensor.numel())]
            finite = [v for v in vals if math.isfinite(v)]
            row[f"max_abs_cosine_to_{label}"] = max(finite) if finite else float("nan")
        same_family = [
            abs(cosine(tensor, other["tensor"]))
            for other in entries
            if other is not row and other["family"] == row["family"] and int(other["tensor"].numel()) == int(tensor.numel())
        ]
        finite_same = [v for v in same_family if math.isfinite(v)]
        row["mean_abs_cosine_within_family"] = float(mean(finite_same)) if finite_same else float("nan")


def family_status(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for family in sorted({row["family"] for row in entries}):
        vals = [row for row in entries if row["family"] == family]
        usable = [row for row in vals if row.get("perturbation_usable")]
        if usable:
            status = "ready"
        elif vals:
            status = "weak"
        else:
            status = "unavailable"
        out[family] = {
            "entries": len(vals),
            "perturbation_usable_entries": len(usable),
            "layers": sorted({row.get("target_layer") for row in usable if row.get("target_layer") is not None}),
            "status": status,
            "recommended_alpha_bucket": Counter(str(row.get("recommended_alpha_bucket")) for row in usable).most_common(1)[0][0] if usable else "alpha_0_01",
        }
    return out


def compact_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "tensor"}


def main() -> int:
    started = time.time()
    ensure_v3_root()
    split = load_json(V3_ROOT / "split_guard_v3.json", {}) or {}
    if split.get("verdict") == "BLOCKED":
        payload = {
            "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "split guard blocked",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Direction Bank V3", "", "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT = BLOCKED", flush=True)
        return 1
    train_eligible = set(str(x) for x in split.get("v3_empirical_direction_train_eligible_task_ids") or split.get("v3_train_candidate_task_ids") or [])
    if not train_eligible:
        train_eligible = {str(row.get("task_id")) for row in load_all_v3_branch_rows(include_prior=True) if primary_safe_v3_row(row)}

    entries: list[dict[str, Any]] = []
    add_random_entries(entries)
    add_head_entries(entries)
    add_v2_bank_entries(entries)
    empirical_entries(entries, train_eligible)
    add_adapter_entries(entries)
    attach_alignment(entries)

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        if row.get("perturbation_usable") and row.get("target_layer") in {24, 36, 47}:
            by_layer[str(int(row["target_layer"]))].append(row)
    # Keep L47 available only as diagnostic; primary scripts will not use it for
    # selector readiness target accounting.
    statuses = family_status(entries)
    ready_non_random = [
        family
        for family, info in statuses.items()
        if family != "random_orthogonal" and info["status"] == "ready"
    ]
    if "random_orthogonal" not in statuses:
        verdict = "BLOCKED"
    elif len(ready_non_random) >= 3:
        verdict = "READY"
    elif len(ready_non_random) >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "RANDOM_ONLY"
    family_counts = Counter(row["family"] for row in entries)
    payload = {
        "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT": verdict,
        "verdict": verdict,
        "directions": entries,
        "directions_by_layer": dict(by_layer),
        "family_counts": dict(family_counts),
        "family_status": statuses,
        "layers": sorted(by_layer),
        "train_eligible_task_ids_used_for_empirical_directions": sorted(train_eligible),
        "heldout_task_ids_excluded_from_empirical_directions": split.get("v3_heldout_candidate_task_ids", []),
        "direction_compatibility_guard": {
            "concat_projection_forbidden_without_validated_projection": True,
            "single_layer_dim_required_for_perturbation": HIDDEN_DIM,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    compact_payload = {k: v for k, v in payload.items() if k not in {"directions", "directions_by_layer"}}
    compact_payload["directions"] = [compact_entry(row) for row in entries]
    compact_payload["directions_by_layer"] = {layer: [compact_entry(row) for row in vals] for layer, vals in by_layer.items()}
    write_json(OUT_JSON, compact_payload)

    direction_rows = [
        {
            "family": row["family"],
            "name": row["name"][:72],
            "layer": row.get("target_layer"),
            "config": row.get("target_config"),
            "dim": row["direction_dim"],
            "usable": row["perturbation_usable"],
            "scoring_only": row["scoring_only"],
            "old_cos": row.get("max_abs_cosine_to_old_tap_aligned"),
            "v1_cos": row.get("max_abs_cosine_to_v1_tap_aligned"),
            "v2_cos": row.get("max_abs_cosine_to_v2_tap_aligned"),
        }
        for row in entries
    ]
    direction_rows.sort(key=lambda row: (str(row["family"]), str(row["layer"]), str(row["name"])))
    lines = [
        "# Hidden-Origin Direction Bank V3",
        "",
        f"BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT = {verdict}",
        "",
        f"- family_counts: `{dict(family_counts)}`",
        f"- ready_non_random_families: `{ready_non_random}`",
        f"- layers: `{sorted(by_layer)}`",
        f"- empirical_direction_heldout_exclusion_count: `{len(split.get('v3_heldout_candidate_task_ids', []))}`",
        "",
        "Directions with concat dimensionality are kept as scoring/geometry diagnostics only. No concat direction is projected into a single layer.",
        "",
        "## Family Status",
        "",
    ]
    lines.extend(md_table([{"family": k, **v} for k, v in sorted(statuses.items())], ["family", "entries", "perturbation_usable_entries", "layers", "status", "recommended_alpha_bucket"]))
    lines.extend(["", "## Directions", ""])
    lines.extend(md_table(direction_rows[:220], ["family", "name", "layer", "config", "dim", "usable", "scoring_only", "old_cos", "v1_cos", "v2_cos"]))
    lines.extend(["", "## Outputs", "", f"- PT: `{rel(OUT_PT)}`", f"- JSON: `{rel(OUT_JSON)}`"])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

