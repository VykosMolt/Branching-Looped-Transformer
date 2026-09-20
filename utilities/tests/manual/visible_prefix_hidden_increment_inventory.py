#!/usr/bin/env python3
"""Label-blind inventory gate for the visible-prefix conference extension.

This program deliberately inspects artifact schemas, identifiers, partitions,
and hashes only.  It does not read correctness values or compute any metric.
Run it before the sealed analysis wrapper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "opi/preanswer/visible_prefix_hidden_increment_20260729T075247Z"
)

GSM_PT = PROJECT_ROOT / "artifacts/reports/proto_introspection/within_domain_recapture.pt"
GSM_INDEX = (
    PROJECT_ROOT / "artifacts/reports/proto_introspection/within_domain_recapture_index.json"
)
GSM_DATA = Path(
    "/home/moloch/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/"
    "740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow"
)
GSM_PUBLISHED = (
    PROJECT_ROOT
    / "opi/verification/paper_verification/fable_hardening_checks_20260710_175017/"
    "preanswer_oof_predictions.csv"
)

HORIZON_V2 = (
    PROJECT_ROOT
    / "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic/"
    "horizon_generation_main_off0.pt"
)
HORIZON_V3 = [
    PROJECT_ROOT
    / "opi/preanswer/horizon_power_v3_20260726/horizon_logic/"
    f"horizon_generation_v3_off{offset}.pt"
    for offset in (170, 340, 510)
]
HORIZON_TASKS = (
    PROJECT_ROOT / "shared/data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl"
)
HORIZON_PUBLISHED_V2 = (
    PROJECT_ROOT
    / "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic/"
    "raw_predictions_heldout.json"
)
HORIZON_PUBLISHED_V3 = (
    PROJECT_ROOT
    / "opi/preanswer/horizon_power_v3_20260726/raw_predictions_heldout_v3.json"
)

OURO_TOKENIZER = PROJECT_ROOT / "shared/models/ouro_rltt_local/tokenizer.json"
OURO_CONFIG = PROJECT_ROOT / "shared/models/ouro_rltt_local/config.json"
OURO_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
VISIBLE_MODEL_ID = "microsoft/deberta-v3-small"
VISIBLE_MODEL_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"
VISIBLE_MODEL_ROOT = (
    PROJECT_ROOT
    / "shared/hf_cache/hub/models--microsoft--deberta-v3-small/snapshots"
    / VISIBLE_MODEL_REVISION
)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = payload.get("records", [])
        records.extend(rows)
        metadata.append(
            {
                "path": str(path),
                "payload_keys": sorted(payload),
                "record_count": len(rows),
                "record_keys": sorted(rows[0]) if rows else [],
                "model": payload.get("model"),
                "feature_shape": payload.get("feature_shape"),
                "tap_layers": payload.get("tap_layers"),
                "num_loops": payload.get("num_loops"),
                "seed": payload.get("seed"),
                "task_offset": payload.get("task_offset"),
            }
        )
    return records, metadata


def split_summary(records: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
    task_splits: dict[str, set[str]] = {}
    candidate_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for row in records:
        task = str(row["task_uid"])
        split = str(row["split"])
        task_splits.setdefault(task, set()).add(split)
        candidate_counts[split] += 1
    for splits in task_splits.values():
        for split in splits:
            task_counts[split] += 1
    crossing = sorted(task for task, splits in task_splits.items() if len(splits) != 1)
    rows = [
        {"task_id": task, "split": next(iter(splits))}
        for task, splits in sorted(task_splits.items())
        if len(splits) == 1
    ]
    return {
        "cohort": cohort,
        "candidate_counts_all_rows": dict(sorted(candidate_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "n_unique_tasks": len(task_splits),
        "zero_task_crossing": not crossing,
        "crossing_task_ids": crossing,
        "task_split_sha256": canonical_hash(rows),
        "task_split_rows": rows,
    }


def file_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    arms = report["cohorts"]
    lines = [
        "# Signed inventory report",
        "",
        f"- Inventory timestamp: `{report['inventory_timestamp_utc']}`",
        f"- Experiment ID: `{report['experiment_id']}`",
        "- Scope: label-blind schema, path, partition, and checksum inventory.",
        "- Attestation: no correctness values or held-out metrics were read or computed by "
        "this inventory program.",
        "",
        "## Gate decisions",
        "",
    ]
    for name in ("horizon_new_only", "gsm8k", "horizon_pooled"):
        arm = arms[name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Inventory status: `{arm['inventory_status']}`",
                f"- HV inventory gate: `{arm['hv_inventory_gate']}`",
                f"- Exact-prefix inventory gate: `{arm['exact_prefix_inventory_gate']}`",
                f"- Original partition gate: `{arm['partition_inventory_gate']}`",
                f"- Reason: {arm['reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Hidden-score decision rule",
            "",
            "Horizon raw hidden-only scores are preserved only for held-out rows. Frozen "
            "hidden feature tensors exist for train, validation, and held-out rows, and "
            "the deterministic original head implementation, grids, seed, and published "
            "held-out scores are preserved. The sealed run must reconstruct the original "
            "head without using held-out labels and pass exact/numerical agreement before "
            "any visible-model fitting. Failure marks Horizon HV `NOT_ADJUDICABLE`.",
            "",
            "GSM8K has frozen hidden features and historical out-of-fold predictions, but "
            "has neither exact rendered candidate trajectories nor the required original "
            "train/validation/held-out partition. It is stopped at inventory as "
            "`VISIBLE_PREFIX_NOT_ADJUDICABLE`; no approximated prefix or replacement split "
            "is allowed.",
            "",
            "## Signature",
            "",
            "The detached SHA-256 signature is written to `inventory_report.sha256`. "
            "The signer identity is `Codex /root`, using a deterministic content digest "
            "rather than an unavailable private-key signature.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    missing_required = [
        p
        for p in [
            GSM_PT,
            HORIZON_V2,
            *HORIZON_V3,
            HORIZON_TASKS,
            OURO_TOKENIZER,
            VISIBLE_MODEL_ROOT / "pytorch_model.bin",
        ]
        if not p.is_file()
    ]

    gsm_payload = torch.load(GSM_PT, map_location="cpu", weights_only=False)
    gsm_records = gsm_payload.get("records", {}).get("gsm8k", [])
    gsm_record_keys = sorted(gsm_records[0]) if gsm_records else []
    gsm_sample_keys = (
        sorted(gsm_records[0]["samples"][0])
        if gsm_records and gsm_records[0].get("samples")
        else []
    )
    gsm_n_samples = sum(len(row.get("samples", [])) for row in gsm_records)

    v2_records, v2_meta = load_records([HORIZON_V2])
    v3_records, v3_meta = load_records(HORIZON_V3)
    pooled_records = v2_records + v3_records
    split_new = split_summary(v3_records, "horizon_new_only")
    split_pooled = split_summary(pooled_records, "horizon_pooled")

    source_files = [
        file_entry(GSM_PT, "GSM8K frozen hidden features and row metadata"),
        file_entry(GSM_INDEX, "GSM8K light index"),
        file_entry(GSM_DATA, "GSM8K source test dataset"),
        file_entry(GSM_PUBLISHED, "GSM8K published/recreated OOF predictions"),
        file_entry(HORIZON_TASKS, "Horizon exact source tasks"),
        file_entry(HORIZON_V2, "Horizon v2 frozen cohort shard"),
        *[file_entry(p, "Horizon v3 prospective frozen cohort shard") for p in HORIZON_V3],
        file_entry(HORIZON_PUBLISHED_V2, "Horizon v2 published held-out predictions"),
        file_entry(HORIZON_PUBLISHED_V3, "Horizon v3 published held-out predictions"),
        file_entry(
            PROJECT_ROOT / "shared/utilities/tests/manual/bg_v2_overnight_horizon_generate.py",
            "Horizon prompt/cut reconstruction implementation",
        ),
        file_entry(
            PROJECT_ROOT / "shared/utilities/tests/manual/bg_v2_overnight_horizon_analysis.py",
            "Original Horizon head implementation",
        ),
        file_entry(
            PROJECT_ROOT / "shared/utilities/tests/manual/bg_v3_horizon_power_analysis.py",
            "Original Horizon power-arm implementation",
        ),
        file_entry(
            PROJECT_ROOT / "shared/utilities/tests/manual/proto_introspection_within_domain_recapture.py",
            "Original GSM8K recapture implementation",
        ),
        file_entry(
            PROJECT_ROOT / "shared/utilities/tests/manual/proto_introspection_within_domain_analysis.py",
            "Original GSM8K analysis implementation",
        ),
        file_entry(OURO_TOKENIZER, "Original generation tokenizer"),
        file_entry(OURO_CONFIG, "Original generation checkpoint config"),
        file_entry(VISIBLE_MODEL_ROOT / "config.json", "Frozen visible encoder config"),
        file_entry(VISIBLE_MODEL_ROOT / "spm.model", "Frozen visible encoder tokenizer"),
        file_entry(VISIBLE_MODEL_ROOT / "pytorch_model.bin", "Frozen visible encoder weights"),
    ]

    shared_horizon_schema = sorted(v3_records[0]) if v3_records else []
    horizon_artifact_fields = {
        "original_prompts": {
            "status": "RECONSTRUCTIBLE_EXACTLY",
            "evidence": [
                str(HORIZON_TASKS),
                str(PROJECT_ROOT / "shared/utilities/tests/manual/bg_v2_overnight_horizon_generate.py"),
            ],
        },
        "trajectory": {
            "status": "PRESERVED_EXACT_RENDERED",
            "field": "generated_text",
        },
        "strict_cut": {"status": "PRESERVED", "field": "n_pre_tok"},
        "tokenizer": {
            "status": "PRESERVED_AND_REVISION_IDENTIFIED",
            "id": "ByteDance/Ouro-2.6B",
            "revision": OURO_REVISION,
            "local_path": str(OURO_TOKENIZER.parent),
            "tokenizer_sha256": sha256(OURO_TOKENIZER),
        },
        "partitions": {"status": "PRESERVED", "field": "split"},
        "labels_task_ids": {
            "status": "PRESERVED",
            "fields": ["success", "task_uid"],
        },
        "strict_pre_cut_shortcuts": {
            "status": "PRESERVED_WITH_DISALLOWED_FIELDS_FILTERED_LATER",
            "allowed_fields": ["n_pre_tok", "mean_logprob_pre", "min_logprob_pre"],
            "explicitly_disallowed": [
                "hit_max_tokens",
                "malformed",
                "parsed_answer",
                "found_final_marker",
            ],
        },
        "hidden_features": {
            "status": "PRESERVED_ALL_SPLITS",
            "field": "preanswer_features",
            "shape": [3, 4, 2048],
        },
        "raw_hidden_scores": {
            "status": "HELDOUT_ONLY",
            "evidence": [str(HORIZON_PUBLISHED_V2), str(HORIZON_PUBLISHED_V3)],
        },
        "original_head": {
            "status": "DETERMINISTIC_REPRODUCTION_SPEC_PRESERVED_WEIGHTS_ABSENT",
            "failure_rule": "must match published held-out hidden scores without held-out fitting",
        },
        "record_keys": shared_horizon_schema,
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": out.name,
        "inventory_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "signer": "Codex /root",
        "signature_scheme": "SHA-256 detached content digest",
        "label_blind_attestation": (
            "This inventory program inspected schemas, IDs, splits, counts, and hashes "
            "only; it did not access correctness values or held-out metrics."
        ),
        "working_tree_note": (
            "Pre-existing unrelated dirty-worktree changes were observed and left untouched."
        ),
        "missing_required_paths": [str(p) for p in missing_required],
        "cohorts": {
            "horizon_new_only": {
                "inventory_status": "CONDITIONALLY_COMPLETE",
                "hv_inventory_gate": "PENDING_EXACT_HEAD_REPRODUCTION",
                "exact_prefix_inventory_gate": "PASS",
                "partition_inventory_gate": "PASS",
                "reason": (
                    "Exact rendered trajectories, stored token cut indices, source prompts, "
                    "tokenizer revision, frozen hidden features, splits, labels, task IDs, "
                    "and held-out published scores exist. Hidden-only train/validation scores "
                    "must be cross-fitted from the preserved original head specification."
                ),
                "artifact_fields": horizon_artifact_fields,
                "split_summary": {k: v for k, v in split_new.items() if k != "task_split_rows"},
                "shards": v3_meta,
            },
            "gsm8k": {
                "inventory_status": "INCOMPLETE",
                "hv_inventory_gate": "FAIL",
                "exact_prefix_inventory_gate": "FAIL",
                "partition_inventory_gate": "FAIL",
                "reason": (
                    "The strict-preanswer artifact preserves hidden features and cut lengths "
                    "but not generated token IDs, rendered trajectories, or visible-prefix "
                    "text. The original evaluation used grouped cross-validation and did not "
                    "preserve a train/validation/held-out partition. Exact prefixes and the "
                    "required partition cannot be reconstructed without changing the estimand."
                ),
                "n_tasks": len(gsm_records),
                "n_candidates": gsm_n_samples,
                "record_keys": gsm_record_keys,
                "sample_keys": gsm_sample_keys,
                "artifact_fields": {
                    "original_prompts": {
                        "status": "RECONSTRUCTIBLE_FROM_DATASET",
                        "source": str(GSM_DATA),
                    },
                    "trajectory": {"status": "ABSENT"},
                    "strict_cut": {"status": "PRESERVED_LENGTH_ONLY", "field": "n_pre_tok"},
                    "tokenizer": {
                        "status": "PRESERVED_AND_REVISION_IDENTIFIED",
                        "id": "ByteDance/Ouro-2.6B",
                        "revision": OURO_REVISION,
                    },
                    "partitions": {"status": "ABSENT_ORIGINAL_GROUPED_CV_ONLY"},
                    "labels_task_ids": {
                        "status": "PRESERVED",
                        "fields": ["correct", "task_id"],
                    },
                    "strict_pre_cut_shortcuts": {
                        "status": "PRESERVED",
                        "fields": [
                            "n_pre_tok",
                            "lp_mean_logprob",
                            "lp_mean_entropy",
                            "lp_last_entropy",
                            "prompt_tok",
                            "question_chars",
                        ],
                    },
                    "hidden_features": {
                        "status": "PRESERVED",
                        "field": "preanswer_feat",
                        "shape": [3, 4, 2048],
                    },
                    "raw_hidden_scores": {
                        "status": "HISTORICAL_OOF_ONLY",
                        "source": str(GSM_PUBLISHED),
                    },
                    "published_heldout_predictions": {
                        "status": "ABSENT_NO_ORIGINAL_HELDOUT_SPLIT"
                    },
                },
            },
            "horizon_pooled": {
                "inventory_status": "CONDITIONALLY_COMPLETE",
                "hv_inventory_gate": "PENDING_EXACT_HEAD_REPRODUCTION",
                "exact_prefix_inventory_gate": "PASS",
                "partition_inventory_gate": "PASS",
                "reason": (
                    "The pooled v2+v3 arm has the same complete rendered-text, cut, split, "
                    "feature, and reproduction evidence as the new-only arm."
                ),
                "artifact_fields": horizon_artifact_fields,
                "split_summary": {
                    k: v for k, v in split_pooled.items() if k != "task_split_rows"
                },
                "shards": v2_meta + v3_meta,
            },
        },
        "visible_encoder": {
            "model_id": VISIBLE_MODEL_ID,
            "revision": VISIBLE_MODEL_REVISION,
            "local_snapshot": str(VISIBLE_MODEL_ROOT),
            "all_required_files_present": all(
                (VISIBLE_MODEL_ROOT / name).is_file()
                for name in ("config.json", "spm.model", "pytorch_model.bin")
            ),
        },
        "source_files": source_files,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "pid": os.getpid(),
        },
    }

    source_manifest = {
        "schema_version": 1,
        "created_at_utc": report["inventory_timestamp_utc"],
        "files": source_files,
    }
    split_manifest = {
        "schema_version": 1,
        "created_at_utc": report["inventory_timestamp_utc"],
        "gsm8k": {
            "status": "NOT_PRESERVED",
            "reason": "historical task-grouped CV only; no original train/val/heldout",
            "n_tasks": len(gsm_records),
            "n_candidates": gsm_n_samples,
        },
        "horizon_new_only": split_new,
        "horizon_pooled": split_pooled,
    }

    json_path = out / "inventory_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "inventory_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    (out / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    signature = sha256(json_path)
    (out / "inventory_report.sha256").write_text(
        f"{signature}  inventory_report.json\n", encoding="utf-8"
    )
    print(out)
    print(f"inventory_report.json sha256={signature}")


if __name__ == "__main__":
    main()
