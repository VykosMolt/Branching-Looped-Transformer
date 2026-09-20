"""Shared helpers for DualAnchor architecture-looped stratified probe v3."""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from bg_hidden_origin_tap_common import PROBE_ROOT

OUT_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31"
V2_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v2_2026-05-31"
V1_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_lineage_probe_v1_2026-05-31"
TRUE_CARRY_ROOT = PROBE_ROOT / "bg_dualanchor_true_carry_equivalence_v1_2026-05-31"
PERTURB_LIFT_ROOT = PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31"
GUARDED_ROOT = PROBE_ROOT / "bg_dualanchor_all_loop_guarded_policy_v1_2026-05-31"

RUN_JSON = OUT_ROOT / "run_report.json"
RUN_MD = OUT_ROOT / "run_report.md"
ROWS_CSV = OUT_ROOT / "architecture_looped_rows.csv"
ROWS_JSON = OUT_ROOT / "architecture_looped_rows.json"
ROWS_PT = OUT_ROOT / "architecture_looped_rows.pt"
TASK_ROWS_CSV = OUT_ROOT / "task_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "stage_decisions.csv"
TERMINAL_POLICY_CSV = OUT_ROOT / "terminal_policy_rows.csv"
L47_ABLATION_CSV = OUT_ROOT / "l47_ablation_rows.csv"
RECOVERY_CSV = OUT_ROOT / "false_prune_recovery_rows.csv"
LINEAGE_JSONL = OUT_ROOT / "lineage_rows.jsonl"
STAGE_JSONL = OUT_ROOT / "stage_decisions.jsonl"


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        x = safe_float(value)
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def write_json(path: Path, payload: Any) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_md(path: Path, lines: Sequence[str]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def md_table(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        vals = [str(row.get(key, "")) for key in keys]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def grouped_mean(rows: Sequence[dict[str, Any]], group_key: str, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_key)].append(row)
    out = []
    for value, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        rec = {group_key: value, "count": len(vals)}
        for key in keys:
            rec[key] = finite_mean(row.get(key) for row in vals)
        out.append(rec)
    return out


def slice_summary(rows: Sequence[dict[str, Any]], name: str, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    keys = [
        "terminal_oracle_retained",
        "terminal_forced_top1_oracle",
        "terminal_forced_top1_reward",
        "terminal_best_reward",
        "terminal_confident",
        "terminal_deferred",
        "terminal_reward_diverse",
        "positive_oracle",
        "stage_false_prunes",
        "final_candidate_count",
        "final_perturbed_fraction",
    ]
    out = {"slice": name, "count": len(selected)}
    for key in keys:
        out[key] = finite_mean(row.get(key) for row in selected)
    return out


def load_run() -> dict[str, Any]:
    return read_json(RUN_JSON, {}) or {}


def load_task_rows() -> list[dict[str, str]]:
    return read_csv(TASK_ROWS_CSV)


def load_terminal_rows() -> list[dict[str, str]]:
    return read_csv(TERMINAL_POLICY_CSV)


def load_l47_rows() -> list[dict[str, str]]:
    return read_csv(L47_ABLATION_CSV)


def load_stage_rows() -> list[dict[str, str]]:
    return read_csv(STAGE_ROWS_CSV)


def load_lineage_rows() -> list[dict[str, str]]:
    return read_csv(ROWS_CSV)


def summarize_slices(task_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        slice_summary(task_rows, "all", lambda row: True),
        slice_summary(task_rows, "reasoning", lambda row: row.get("domain") == "reasoning"),
        slice_summary(task_rows, "science", lambda row: row.get("domain") == "science"),
        slice_summary(task_rows, "positive_oracle", lambda row: safe_float(row.get("positive_oracle"), 0.0) > 0),
        slice_summary(task_rows, "reward_diverse", lambda row: safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
        slice_summary(
            task_rows,
            "positive_and_reward_diverse",
            lambda row: safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0,
        ),
        slice_summary(task_rows, "heldout", lambda row: row.get("split") == "heldout"),
        slice_summary(task_rows, "calibration_or_val", lambda row: row.get("split") in {"calibration", "val"}),
        slice_summary(task_rows, "train_or_diagnostic", lambda row: row.get("split") in {"train", "diagnostic"}),
        slice_summary(
            task_rows,
            "tie_heavy_or_not_reward_diverse",
            lambda row: safe_float(row.get("terminal_reward_diverse"), 0.0) <= 0,
        ),
    ]


def report_lines(title: str, verdict_key: str, verdict: str, sections: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
    lines = [f"# {title}", "", f"{verdict_key} = {verdict}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", ""])
        lines.extend(body)
        lines.append("")
    return lines


def latest_verdict(path: Path, key: str) -> str:
    data = read_json(path, {}) or {}
    return str(data.get(key) or data.get("status") or "MISSING")

