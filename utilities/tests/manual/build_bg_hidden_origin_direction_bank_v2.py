"""Build a perturbation direction bank for hidden-origin diversity v2."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import torch

from bg_hidden_origin_diversity_v2_common import (
    DIRECTION_BANK_PT,
    HIDDEN_DIM,
    PROBE_ROOT,
    V1_ROOT,
    V2_ROOT,
    config_vector_from_row,
    cosine,
    deterministic_reward,
    ensure_v2_root,
    group_rows,
    load_all_branch_rows,
    md_table,
    rel,
    rms_normalize,
    safe_primary_row,
    stable_v2_row,
    tensor_stats,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import direction_from_state_dict


OUT_PT = DIRECTION_BANK_PT
OUT_JSON = V2_ROOT / "direction_bank.json"
OUT_MD = V2_ROOT / "direction_bank.md"


def add_direction(
    directions_by_layer: dict[str, list[dict[str, Any]]],
    *,
    layer: int,
    name: str,
    family: str,
    tensor: torch.Tensor,
    source: str,
    count: int = 0,
) -> None:
    if not isinstance(tensor, torch.Tensor) or int(tensor.numel()) != HIDDEN_DIM:
        return
    vec = rms_normalize(tensor.flatten().detach().cpu().to(torch.float32))
    directions_by_layer[str(int(layer))].append(
        {
            "layer": int(layer),
            "name": name,
            "family": family,
            "tensor": vec,
            "source": source,
            "count": int(count),
            "stats": tensor_stats(vec),
        }
    )


def config_layer(config: str) -> int | None:
    for prefix, layer in (("24_", 24), ("36_", 36), ("47_", 47), ("30_", 30), ("42_", 42)):
        if config.startswith(prefix):
            return layer
    return None


def load_head_payloads() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for path in (
        PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt",
        PROBE_ROOT / "bg_head_registry_2026-05-17.pt",
        V1_ROOT / "hidden_origin_tap_heads.pt",
    ):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        out.append((rel(path), payload))
    return out


def add_head_directions(directions_by_layer: dict[str, list[dict[str, Any]]]) -> None:
    for source, payload in load_head_payloads():
        for row in list(payload.get("heads") or []):
            state = row.get("state_dict") if isinstance(row, dict) else None
            direction = direction_from_state_dict(state) if isinstance(state, dict) else None
            if not isinstance(direction, torch.Tensor) or int(direction.numel()) != HIDDEN_DIM:
                continue
            cfg = str(row.get("config") or "")
            layer = config_layer(cfg)
            if layer not in {24, 36, 47}:
                continue
            group = str(row.get("head_group") or row.get("head_family") or row.get("group") or "")
            family = "v1_hidden_origin_tap" if "hidden_origin" in source else "old_tap_aligned"
            name = f"{family}:{group}:{cfg}:{row.get('architecture')}"
            add_direction(directions_by_layer, layer=layer, name=name, family=family, tensor=direction, source=source)


def add_empirical_directions(directions_by_layer: dict[str, list[dict[str, Any]]]) -> None:
    rows = [row for row in load_all_branch_rows() if safe_primary_row(row) and stable_v2_row(row)]
    groups = group_rows(rows)
    diffs_by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    values_by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    for vals in groups.values():
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        if len(ordered) < 2:
            continue
        for row in ordered:
            for cfg in ("24_L4", "24_mean", "36_L4", "36_mean", "47_L4", "47_mean"):
                vec = config_vector_from_row(row, cfg)
                if isinstance(vec, torch.Tensor) and int(vec.numel()) == HIDDEN_DIM:
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
                for cfg in ("24_L4", "24_mean", "36_L4", "36_mean", "47_L4", "47_mean"):
                    left = config_vector_from_row(pref, cfg)
                    right = config_vector_from_row(rej, cfg)
                    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                        diffs_by_config[cfg].append(left.detach().cpu().to(torch.float32) - right.detach().cpu().to(torch.float32))
    for cfg, diffs in diffs_by_config.items():
        layer = config_layer(cfg)
        if layer not in {24, 36, 47} or not diffs:
            continue
        mean_diff = torch.stack(diffs, dim=0).mean(dim=0)
        add_direction(
            directions_by_layer,
            layer=layer,
            name=f"hidden_origin_empirical_mean_diff:{cfg}",
            family="hidden_origin_empirical",
            tensor=mean_diff,
            source="prior_hidden_origin_outcomes",
            count=len(diffs),
        )
        vals = values_by_config.get(cfg) or []
        if len(diffs) >= 20 and len(vals) >= 20:
            std = torch.stack(vals, dim=0).std(dim=0, unbiased=False).clamp(min=1e-4)
            add_direction(
                directions_by_layer,
                layer=layer,
                name=f"hidden_origin_whitened_mean_diff:{cfg}",
                family="hidden_origin_whitened",
                tensor=mean_diff / std,
                source="prior_hidden_origin_outcomes",
                count=len(diffs),
            )


def find_adapter_like_tensors() -> list[tuple[str, str, torch.Tensor]]:
    out: list[tuple[str, str, torch.Tensor]] = []
    for path in (
        PROBE_ROOT / "bg_empirical_steering_direction_2026-05-18/directions.pt",
    ):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        stack = [("", payload)]
        while stack:
            name, value = stack.pop()
            if isinstance(value, torch.Tensor) and int(value.numel()) == HIDDEN_DIM:
                out.append((rel(path), name or "tensor", value.detach().cpu().to(torch.float32)))
            elif isinstance(value, dict):
                for key, val in value.items():
                    stack.append((f"{name}.{key}" if name else str(key), val))
            elif isinstance(value, (list, tuple)):
                for idx, val in enumerate(value):
                    stack.append((f"{name}[{idx}]", val))
    return out[:12]


def add_adapter_proxy_directions(directions_by_layer: dict[str, list[dict[str, Any]]]) -> None:
    for source, name, tensor in find_adapter_like_tensors():
        lower = name.lower()
        if "24" in lower:
            layers = [24]
        elif "36" in lower:
            layers = [36]
        elif "47" in lower:
            layers = [47]
        else:
            layers = [24, 36]
        for layer in layers:
            add_direction(
                directions_by_layer,
                layer=layer,
                name=f"adapter_proxy:{name}",
                family="adapter_proxy",
                tensor=tensor,
                source=source,
                count=1,
            )


def summarize_alignment(directions_by_layer: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for layer, items in sorted(directions_by_layer.items()):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                rows.append(
                    {
                        "layer": layer,
                        "left": items[i]["name"],
                        "right": items[j]["name"],
                        "left_family": items[i]["family"],
                        "right_family": items[j]["family"],
                        "cosine": cosine(items[i]["tensor"], items[j]["tensor"]),
                    }
                )
    return rows


def main() -> int:
    started = time.time()
    ensure_v2_root()
    directions_by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    add_head_directions(directions_by_layer)
    add_empirical_directions(directions_by_layer)
    add_adapter_proxy_directions(directions_by_layer)

    directions_by_layer = {layer: vals for layer, vals in directions_by_layer.items() if vals}
    family_counts = Counter(item["family"] for vals in directions_by_layer.values() for item in vals)
    non_random_families = set(family_counts)
    if len(non_random_families) >= 2:
        verdict = "READY"
    elif len(non_random_families) == 1:
        verdict = "PARTIAL"
    elif directions_by_layer:
        verdict = "PARTIAL"
    else:
        verdict = "RANDOM_ONLY"
    alignment_rows = summarize_alignment(directions_by_layer)
    payload = {
        "BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT": verdict,
        "verdict": verdict,
        "directions_by_layer": directions_by_layer,
        "family_counts": dict(family_counts),
        "layers": sorted(directions_by_layer),
        "alignment_rows": alignment_rows,
        "random_orthogonal_available": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    compact = {
        k: v
        for k, v in payload.items()
        if k not in {"directions_by_layer", "alignment_rows"}
    }
    compact["directions_by_layer"] = {
        layer: [
            {
                "layer": item["layer"],
                "name": item["name"],
                "family": item["family"],
                "source": item["source"],
                "count": item.get("count"),
                "stats": item.get("stats"),
            }
            for item in vals
        ]
        for layer, vals in directions_by_layer.items()
    }
    compact["alignment_rows"] = sorted(alignment_rows, key=lambda row: abs(float(row["cosine"])) if math.isfinite(float(row["cosine"])) else -1, reverse=True)[:120]
    write_json(OUT_JSON, compact)

    direction_rows = [
        {
            "layer": layer,
            "name": item["name"],
            "family": item["family"],
            "source": item["source"],
            "count": item.get("count"),
            "rms": item["stats"]["rms"],
        }
        for layer, vals in sorted(directions_by_layer.items())
        for item in vals
    ]
    align_display = [
        {
            "layer": row["layer"],
            "left_family": row["left_family"],
            "right_family": row["right_family"],
            "cosine": f"{row['cosine']:.3f}" if math.isfinite(float(row["cosine"])) else "NA",
        }
        for row in sorted(alignment_rows, key=lambda r: abs(float(r["cosine"])) if math.isfinite(float(r["cosine"])) else -1, reverse=True)[:40]
    ]
    lines = [
        "# Hidden-Origin Direction Bank V2",
        "",
        f"BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT = {verdict}",
        "",
        f"- family_counts: `{dict(family_counts)}`",
        f"- random_orthogonal_available: `True`",
        "",
        "## Directions",
        "",
    ]
    lines.extend(md_table(direction_rows, ["layer", "name", "family", "source", "count", "rms"]))
    lines.extend(["", "## Alignment Snapshot", ""])
    lines.extend(md_table(align_display, ["layer", "left_family", "right_family", "cosine"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

