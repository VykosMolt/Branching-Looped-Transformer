#!/usr/bin/env python3
"""Aggregate the BG sequence-adapter quick preflight gate."""
from __future__ import annotations

import time

from bg_sequence_adapter_common import QUICK_OUT_ROOT, load_json, rel, write_json, write_md


OUT_JSON = QUICK_OUT_ROOT / "summary.json"
OUT_MD = QUICK_OUT_ROOT / "summary.md"


def main() -> int:
    started = time.time()
    parser = load_json(QUICK_OUT_ROOT / "parser_audit.json", {})
    reward = load_json(QUICK_OUT_ROOT / "reward_distribution.json", {})
    micro = load_json(QUICK_OUT_ROOT / "optimizer_sanity_micro.json", {})
    throughput = load_json(QUICK_OUT_ROOT / "gpu_throughput.json", {})
    parser_v = parser.get("BG_SEQUENCE_PARSER_AUDIT_VERDICT", "BLOCKED")
    reward_v = reward.get("BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT", "BLOCKED")
    micro_v = micro.get("BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT", "BLOCKED")
    throughput_v = throughput.get("BG_SEQUENCE_GPU_THROUGHPUT_VERDICT", "BLOCKED")

    blockers: list[str] = []
    warnings: list[str] = []
    if parser_v == "BLOCKED":
        blockers.append("parser audit blocked")
    elif parser_v == "PARTIAL":
        warnings.append("parser audit partial")
    if reward_v in {"BLOCKED", "REWARD_SIGNAL_FLAT"}:
        blockers.append(f"reward distribution {reward_v}")
    elif reward_v == "REWARD_SIGNAL_WEAK":
        warnings.append("reward signal weak but nonflat")
    if micro_v in {"BLOCKED", "OPTIMIZER_NO_MOVEMENT"}:
        blockers.append(f"optimizer micro {micro_v}")
    elif micro_v == "OPTIMIZER_WEAK":
        warnings.append("optimizer micro weak")
    if throughput_v in {"BLOCKED", "TOO_SLOW_FOR_FULL_SCOPE"}:
        blockers.append(f"throughput {throughput_v}")
    elif throughput_v == "LONG_BUT_FEASIBLE":
        warnings.append("throughput long but feasible")

    if blockers:
        readiness = "NOT_READY"
    elif warnings:
        readiness = "READY_WITH_WARNINGS"
    else:
        readiness = "READY"

    payload = {
        "BG_SEQUENCE_PARSER_AUDIT_VERDICT": parser_v,
        "BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT": reward_v,
        "BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT": micro_v,
        "BG_SEQUENCE_GPU_THROUGHPUT_VERDICT": throughput_v,
        "OVERNIGHT_SEQUENCE_ADAPTER_READINESS": readiness,
        "warnings": warnings,
        "blockers": blockers,
        "recommended_next": "continue_to_phase1_full_sequence_adapter_run" if readiness in {"READY", "READY_WITH_WARNINGS"} else "stop_and_fix_preflight_blockers",
        "component_paths": {
            "parser_audit": rel(QUICK_OUT_ROOT / "parser_audit.json"),
            "reward_distribution": rel(QUICK_OUT_ROOT / "reward_distribution.json"),
            "optimizer_sanity_micro": rel(QUICK_OUT_ROOT / "optimizer_sanity_micro.json"),
            "gpu_throughput": rel(QUICK_OUT_ROOT / "gpu_throughput.json"),
        },
        "commands_run": [
            "venv/bin/python -u utilities/tests/manual/bg_sequence_adapter_parser_audit.py",
            "venv/bin/python -u utilities/tests/manual/bg_sequence_adapter_reward_distribution_check.py",
            "venv/bin/python -u utilities/tests/manual/bg_sequence_adapter_optimizer_sanity_micro.py",
            "venv/bin/python -u utilities/tests/manual/bg_sequence_adapter_gpu_throughput_benchmark.py",
            "venv/bin/python -u utilities/tests/manual/analyze_bg_sequence_adapter_quick_preflight.py",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Quick Preflight Summary",
        "",
        f"BG_SEQUENCE_PARSER_AUDIT_VERDICT = {parser_v}",
        f"BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT = {reward_v}",
        f"BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT = {micro_v}",
        f"BG_SEQUENCE_GPU_THROUGHPUT_VERDICT = {throughput_v}",
        f"OVERNIGHT_SEQUENCE_ADAPTER_READINESS = {readiness}",
        "",
        "## Parser audit",
        "",
        f"- parse rate: `{parser.get('parse_rate')}`",
        f"- generations: `{parser.get('generation_count')}`",
        "",
        "## Reward distribution",
        "",
        f"- tasks with nonzero reward variance: `{reward.get('tasks_with_nonzero_reward_variance')}`",
        f"- nonzero reward variance rate: `{reward.get('nonzero_reward_variance_rate')}`",
        "",
        "## Optimizer sanity",
        "",
        f"- selected reward lift: `{(micro.get('selected_result') or {}).get('reward_lift')}`",
        f"- changed outputs: `{(micro.get('selected_result') or {}).get('changed_output_count')}`",
        f"- relaxed alpha used: `{micro.get('SANITY_RELAXED_ALPHA_USED')}`",
        "",
        "## GPU throughput",
        "",
        f"- baseline sec/generation: `{throughput.get('baseline_sec_per_generation')}`",
        f"- intervention sec/generation: `{throughput.get('intervention_sec_per_generation')}`",
        f"- estimates: `{throughput.get('runtime_estimates')}`",
        "",
        "## Overnight readiness",
        "",
        f"- warnings: `{warnings}`",
        f"- blockers: `{blockers}`",
        f"- recommended next: `{payload['recommended_next']}`",
        "",
        "## Files created",
        "",
    ]
    for path in payload["component_paths"].values():
        lines.append(f"- `{path}`")
    lines.extend(["", "## Commands run", ""])
    for cmd in payload["commands_run"]:
        lines.append(f"- `{cmd}`")
    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    write_md(OUT_MD, lines)
    print(f"OVERNIGHT_SEQUENCE_ADAPTER_READINESS = {readiness}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if readiness in {"READY", "READY_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
