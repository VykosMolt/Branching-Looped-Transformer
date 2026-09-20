#!/usr/bin/env python3
"""Build teacher-forced MCQ dataset for causal BG intervention adapter."""
from __future__ import annotations

import time
from collections import Counter

from bg_causal_adapter_common import OUT_ROOT, build_dataset_rows, rel, write_json, write_md


OUT_JSON = OUT_ROOT / "adapter_dataset.json"
OUT_MD = OUT_ROOT / "adapter_dataset.md"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, split, tokenization = build_dataset_rows()
    example_counts = Counter(row["split"] for row in rows)
    task_counts = {name: len(ids) for name, ids in split.items()}
    domain_counts = Counter(row["domain"] for row in rows)
    if task_counts.get("train", 0) >= 20 and task_counts.get("heldout", 0) >= 8 and rows:
        verdict = "READY"
    elif task_counts.get("train", 0) >= 10 and task_counts.get("heldout", 0) >= 4 and rows:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_CAUSAL_ADAPTER_DATASET_VERDICT": verdict,
        "examples": rows,
        "split_task_ids": split,
        "task_counts": task_counts,
        "example_counts": dict(example_counts),
        "domain_counts": dict(domain_counts),
        "target_configs": {
            "reasoning": {"prefix_length": 64},
            "science": {"prefix_length": 32},
        },
        "tokenization": tokenization,
        "KL_ANSWER_POSITION_MASKED": True,
        "splitting": "by_task_id",
        "heldout_task_overlap_with_train": sorted(set(split.get("heldout", [])) & set(split.get("train", []))),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Causal Adapter Teacher-Forced Dataset",
        "",
        f"BG_CAUSAL_ADAPTER_DATASET_VERDICT = {verdict}",
        "",
        f"- examples: `{len(rows)}`",
        f"- task_counts: `{task_counts}`",
        f"- example_counts: `{dict(example_counts)}`",
        f"- domain_counts: `{dict(domain_counts)}`",
        f"- target_configs: `reasoning @ 64`, `science @ 32`",
        f"- option_token_format: `{tokenization['option_token_format']}`",
        f"- GSM8K included: `{tokenization['gsm8k_included']}`",
        "",
        "## Splits",
        "",
    ]
    for split_name in ["train", "val", "heldout"]:
        lines.append(f"- {split_name}: `{split.get(split_name, [])}`")
    lines.extend(
        [
            "",
            "## Example Preview",
            "",
            "| example | split | domain | task | prefix | branch | correct | success |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in rows[:24]:
        lines.append(
            f"| `{row['example_id']}` | `{row['split']}` | `{row['domain']}` | `{row['task_id']}` | "
            f"{row['prefix_length']} | {row['source_branch_id']} | `{row['correct_option']}` | `{row['continuation_success']}` |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_CAUSAL_ADAPTER_DATASET_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
