"""Download/cache compact coding, reasoning, logic, and thinking datasets.

This does not assume every Hugging Face dataset alias still exists. It tries a
small list of useful benchmarks and writes a manifest recording successes,
failures, columns, and example counts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    domain: str
    path: str
    config: Optional[str]
    split: Optional[str]
    notes: str


SPECS = [
    DatasetSpec("humaneval", "coding", "openai/openai_humaneval", None, "test", "Python function synthesis."),
    DatasetSpec("mbpp_sanitized", "coding", "google-research-datasets/mbpp", "sanitized", "test", "Python programming problems."),
    DatasetSpec("mbpp_plain", "coding", "mbpp", "sanitized", "test", "Fallback MBPP alias."),
    DatasetSpec("arc_challenge", "reasoning", "allenai/ai2_arc", "ARC-Challenge", "validation", "Multiple-choice science reasoning."),
    DatasetSpec("arc_easy", "reasoning", "allenai/ai2_arc", "ARC-Easy", "validation", "Easier ARC control."),
    DatasetSpec("bbh_boolean", "reasoning", "lukaemon/bbh", "boolean_expressions", "test", "BBH symbolic boolean reasoning."),
    DatasetSpec("bbh_multistep", "reasoning", "lukaemon/bbh", "multistep_arithmetic_two", "test", "BBH multistep arithmetic."),
    DatasetSpec("bbh_logical_five", "logic", "lukaemon/bbh", "logical_deduction_five_objects", "test", "BBH logical deduction."),
    DatasetSpec("bbh_tracking", "logic", "lukaemon/bbh", "tracking_shuffled_objects_five_objects", "test", "BBH object tracking."),
    DatasetSpec("logiqa_eleuther", "logic", "EleutherAI/logiqa", None, "validation", "Reading-comprehension logic QA."),
    DatasetSpec("logiqa_alt", "logic", "lucasmccabe/logiqa", None, "test", "Fallback LogiQA alias."),
    DatasetSpec("strategyqa", "thinking", "ChilleD/StrategyQA", None, "test", "Implicit multi-hop yes/no reasoning."),
    DatasetSpec("gpqa_diamond", "thinking", "Idavidrein/gpqa", "gpqa_diamond", "train", "Graduate-level Google-proof QA."),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/eval_dataset_download_manifest.json")
    p.add_argument("--only-domain", choices=["coding", "reasoning", "logic", "thinking"], default=None)
    return p.parse_args()


def split_count(ds) -> int | dict:
    if isinstance(ds, DatasetDict):
        return {name: len(split) for name, split in ds.items()}
    return len(ds)


def split_columns(ds) -> list[str] | dict:
    if isinstance(ds, DatasetDict):
        return {name: list(split.column_names) for name, split in ds.items()}
    return list(ds.column_names)


def try_load(spec: DatasetSpec) -> dict:
    kwargs = {}
    if spec.config is not None:
        args = (spec.path, spec.config)
    else:
        args = (spec.path,)
    if spec.split is not None:
        kwargs["split"] = spec.split

    try:
        ds = load_dataset(*args, **kwargs)
        return {
            "ok": True,
            "spec": asdict(spec),
            "count": split_count(ds),
            "columns": split_columns(ds),
            "cache_files": getattr(ds, "cache_files", None),
        }
    except Exception as exc:
        return {
            "ok": False,
            "spec": asdict(spec),
            "error": repr(exc),
        }


def main() -> None:
    args = parse_args()
    specs = [spec for spec in SPECS if args.only_domain is None or spec.domain == args.only_domain]
    rows = []
    for spec in specs:
        print(f"Loading {spec.tag}: {spec.path} {spec.config or ''} {spec.split or ''}".strip())
        row = try_load(spec)
        rows.append(row)
        if row["ok"]:
            print(f"  OK count={row['count']} columns={row['columns']}")
        else:
            print(f"  FAIL {row['error']}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"datasets": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
