"""Summarize layer/config choices for mixed-domain head winners."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402


EVAL_JSON = REPORT_DIR / "mixed_domain_head_evaluation_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "mixed_head_layer_choice_analysis_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "mixed_head_layer_choice_analysis_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(EVAL_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["pairwise_acc"],
            row["metrics"]["top1_tournament_acc"],
            -float(row["metrics"].get("cycle_rate", 0.0) or 0.0),
        ),
    )


def layer_bucket(config: str) -> str:
    if config.startswith("24_"):
        return "24"
    if config.startswith("36_"):
        return "36"
    if config.startswith("47_concat"):
        return "47_concat"
    if config.startswith("47_"):
        return "47"
    return "other"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mixed Head Layer Choice Analysis (2026-05-17)",
        "",
        f"MIXED_HEAD_LAYER_ANALYSIS_VERDICT = {payload['meta']['mixed_head_layer_analysis_verdict']}",
        "",
        "## Winner Counts",
        f"- config winners: `{json.dumps(payload['winner_counts']['config'], default=str)}`",
        f"- layer winners: `{json.dumps(payload['winner_counts']['layer'], default=str)}`",
        f"- architecture winners: `{json.dumps(payload['winner_counts']['architecture'], default=str)}`",
        "",
        "## Per Eval Winner",
    ]
    for row in payload["per_eval_winners"]:
        lines.append(f"- {row['eval_set']}: {row['head_group']} / {row['config']} / {row['architecture']} pairwise={row['pairwise']:.3f} top1={row['top1']:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_json(args.input)
    rows = payload.get("rows", []) or []
    eval_sets = sorted({row["eval_set"] for row in rows})
    per_eval = []
    config_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    arch_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    family_layer: dict[str, Counter[str]] = defaultdict(Counter)
    for eval_set in eval_sets:
        winner = best([row for row in rows if row["eval_set"] == eval_set])
        if not winner:
            continue
        config = winner["config"]
        arch = winner["architecture"]
        group = winner["head_group"]
        config_counts[config] += 1
        layer_counts[layer_bucket(config)] += 1
        arch_counts[arch] += 1
        group_counts[group] += 1
        family_layer[group][layer_bucket(config)] += 1
        per_eval.append(
            {
                "eval_set": eval_set,
                "head_group": group,
                "config": config,
                "architecture": arch,
                "layer_bucket": layer_bucket(config),
                "pairwise": winner["metrics"]["pairwise_acc"],
                "top1": winner["metrics"]["top1_tournament_acc"],
            }
        )
    verdict = "READY" if per_eval else "BLOCKED"
    out = {
        "meta": {
            "mixed_head_layer_analysis_verdict": verdict,
            "source": repo_path(output_path(args.input)),
        },
        "winner_counts": {
            "config": dict(config_counts),
            "layer": dict(layer_counts),
            "architecture": dict(arch_counts),
            "head_group": dict(group_counts),
            "family_layer": {k: dict(v) for k, v in family_layer.items()},
        },
        "per_eval_winners": per_eval,
        "interpretation": {
            "earlier_layers_prominent": layer_counts["24"] + layer_counts["36"] >= layer_counts["47"] + layer_counts["47_concat"],
            "nonorm_prominent": arch_counts["AntisymLinearNoNorm"] >= arch_counts["AntisymLinear"],
            "all_loop_prominent": config_counts["47_concat_all_loops"] > 0,
        },
    }
    write_json(output_path(args.output), out)
    write_markdown(output_path(args.output_md), out)
    print(f"MIXED_HEAD_LAYER_ANALYSIS_VERDICT = {verdict}")
    print(f"wrote {repo_path(output_path(args.output))}")


if __name__ == "__main__":
    main()
