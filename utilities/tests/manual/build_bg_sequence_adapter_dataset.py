#!/usr/bin/env python3
"""Build task-level splits for BG sequence-level adapter reward training."""
from __future__ import annotations

import time
from collections import Counter

from bg_sequence_adapter_common import OUT_ROOT, build_sequence_dataset_rows, rel, write_json, write_md


OUT_JSON = OUT_ROOT / "sequence_adapter_dataset.json"
OUT_MD = OUT_ROOT / "sequence_adapter_dataset.md"


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, split = build_sequence_dataset_rows()
    task_counts = {key: len(vals) for key, vals in split.items()}
    domain_counts = Counter(row["domain"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    heldout_overlap = sorted(set(split.get("heldout", [])) & set(split.get("train", [])))
    if task_counts.get("train", 0) >= 24 and task_counts.get("val", 0) >= 8 and task_counts.get("heldout", 0) >= 12:
        verdict = "READY"
    elif task_counts.get("train", 0) >= 12 and task_counts.get("val", 0) >= 4 and task_counts.get("heldout", 0) >= 8 and not heldout_overlap:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_SEQUENCE_ADAPTER_DATASET_VERDICT": verdict,
        "tasks": rows,
        "split_task_ids": split,
        "task_counts": task_counts,
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "target_split": {"train": "24-40", "val": "8-12", "heldout": "12-24"},
        "minimum_split": {"train": 12, "val": 4, "heldout": 8},
        "splitting": "by_task_id",
        "heldout_task_overlap_with_train": heldout_overlap,
        "prompt_format": "Question/options/Think briefly/FINAL ANSWER",
        "parser_type": "mcq_final_answer_letter",
        "gsm8k_included": False,
        "prefix_policy": "no generated prefix_text used; intervention position is prompt_last_token",
        "INTERVENTION_POSITION_WARNING": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence-Level Adapter Dataset",
        "",
        f"BG_SEQUENCE_ADAPTER_DATASET_VERDICT = {verdict}",
        "",
        f"- tasks: `{len(rows)}`",
        f"- task counts: `{task_counts}`",
        f"- domain counts: `{dict(domain_counts)}`",
        f"- heldout overlap with train: `{heldout_overlap}`",
        f"- prompt format: `Question/options/Think briefly/FINAL ANSWER`",
        f"- parser type: `mcq_final_answer_letter`",
        f"- intervention position warning: `true`",
        "",
        "## Splits",
        "",
    ]
    for name in ["train", "val", "heldout"]:
        lines.append(f"- {name}: `{split.get(name, [])}`")
    lines.extend(["", "## Preview", "", "| split | domain | task_id | correct | source |", "|---|---|---|---|---|"])
    for row in rows[:40]:
        lines.append(f"| `{row['split']}` | `{row['domain']}` | `{row['task_id']}` | `{row['correct_option']}` | `{row['source_dataset']}` |")
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_ADAPTER_DATASET_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
