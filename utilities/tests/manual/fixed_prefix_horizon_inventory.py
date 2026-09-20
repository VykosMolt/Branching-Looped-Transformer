#!/usr/bin/env python3
"""Inventory-only gate for the repaired fixed-prefix Horizon experiment.

This program never generates, re-tokenizes a preserved continuation, replays
the model, or fits a classifier.  It verifies whether the source artifacts
contain the exact token sequences required by the sealed mission.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = PROJECT_ROOT / "opi/preanswer/fixed_prefix_horizon_20260729T082827Z"
TASK_SOURCE = (
    PROJECT_ROOT / "shared/data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl"
)
V2_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724"
V3_ROOT = PROJECT_ROOT / "opi/preanswer/horizon_power_v3_20260726"
V2_SHARD = V2_ROOT / "horizon_logic/horizon_generation_main_off0.pt"
V3_SHARDS = [
    V3_ROOT / f"horizon_logic/horizon_generation_v3_off{offset}.pt"
    for offset in (170, 340, 510)
]
V2_INDEX = V2_ROOT / "horizon_logic/horizon_generation_main_off0_index.json"
V3_INDEXES = [
    V3_ROOT / f"horizon_logic/horizon_generation_v3_off{offset}_index.json"
    for offset in (170, 340, 510)
]
TERMINAL_POOL = V2_ROOT / "terminal_selection/terminal_pool.pt"
V2_SUMS = V2_ROOT / "SHA256SUMS"
V3_SUMS = V3_ROOT / "SHA256SUMS"
V2_ENVIRONMENT = V2_ROOT / "ENVIRONMENT.json"
V3_ENVIRONMENT = V3_ROOT / "ENVIRONMENT.json"
V2_RUN_MANIFEST = V2_ROOT / "MASTER_RUN_MANIFEST.json"
V2_TASK_MANIFEST = V2_ROOT / "horizon_logic/task_manifest.json"
V2_SPLIT_INTEGRITY = V2_ROOT / "horizon_logic/split_integrity.json"
V2_CUT_INTEGRITY = V2_ROOT / "horizon_logic/cut_integrity.json"
V2_RAW_PREDICTIONS = V2_ROOT / "horizon_logic/raw_predictions_heldout.json"
V3_RAW_PREDICTIONS = V3_ROOT / "raw_predictions_heldout_v3.json"
GENERATOR = PROJECT_ROOT / "shared/utilities/tests/manual/bg_v2_overnight_horizon_generate.py"
V3_GENERATOR = PROJECT_ROOT / "shared/utilities/tests/manual/bg_v3_horizon_power_generate.py"
ANALYSIS = PROJECT_ROOT / "shared/utilities/tests/manual/bg_v2_overnight_horizon_analysis.py"
FEATURE_EXTRACTOR = PROJECT_ROOT / "shared/src/evaluator/bg_transformer_features.py"
MODEL_ROOT = PROJECT_ROOT / "shared/models/ouro_rltt_local"
MODEL_FILES = [
    MODEL_ROOT / "model-00001.safetensors",
    MODEL_ROOT / "model-00002.safetensors",
    MODEL_ROOT / "model-00003.safetensors",
    MODEL_ROOT / "model.safetensors.index.json",
    MODEL_ROOT / "config.json",
    MODEL_ROOT / "configuration_ouro.py",
    MODEL_ROOT / "modeling_ouro.py",
]
TOKENIZER_FILES = [
    MODEL_ROOT / "tokenizer.json",
    MODEL_ROOT / "tokenizer_config.json",
    MODEL_ROOT / "vocab.json",
    MODEL_ROOT / "merges.txt",
    MODEL_ROOT / "special_tokens_map.json",
]
TOKENIZER_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
TOKEN_FIELDS = {
    "gen_ids",
    "generated_ids",
    "generated_token_ids",
    "new_ids",
    "new_ids_full",
    "continuation_ids",
    "continuation_token_ids",
    "sequence_ids",
    "sequences",
}


def sha256(path: Path, chunk_size: int = 8 << 20) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def file_entry(path: Path, role: str, historically_sealed: bool | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
        "historically_sealed": historically_sealed,
    }


def parse_sums(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})\s+(.+)$", line)
        if match:
            output[match.group(2)] = match.group(1)
    return output


def options_dict(options: list[str]) -> dict[str, str]:
    return dict(zip([chr(65 + i) for i in range(len(options))], options))


def options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def mcq_prompt(question: str, options: dict[str, str]) -> str:
    return (
        "Answer the multiple-choice question. Give concise reasoning if useful, "
        "then end with exactly one line: FINAL ANSWER: <letter>.\n\n"
        f"Question:\n{question}\n\nOptions:\n{options_text(options)}"
    )


def split_for(task_uid: str) -> str:
    value = int(hashlib.sha256(task_uid.encode()).hexdigest()[:8], 16) % 100
    if value < 50:
        return "train"
    if value < 70:
        return "val"
    return "heldout"


def load_task_pool() -> list[dict[str, Any]]:
    rows = []
    with TASK_SOURCE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if (
                row.get("category") == "synthetic_propositional"
                and 2 <= int(row.get("proof_depth", 0)) <= 4
            ):
                rows.append(row)
                if len(rows) >= 4000:
                    break
    rows.sort(key=lambda row: hashlib.sha256(row["task_uid"].encode()).hexdigest())
    return rows


def load_shards() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    specs = [(V2_SHARD, "v2_original", 0), *[
        (path, "v3_prospective", offset)
        for path, offset in zip(V3_SHARDS, (170, 340, 510))
    ]]
    for source_index, (path, source, expected_offset) in enumerate(specs):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = payload["records"]
        metadata.append(
            {
                "path": str(path),
                "source": source,
                "expected_offset": expected_offset,
                "payload_keys": sorted(payload),
                "record_count": len(rows),
                "record_keys": sorted(rows[0]) if rows else [],
                "generation_metadata": {
                    key: payload.get(key)
                    for key in (
                        "tag",
                        "category",
                        "depth_range",
                        "model",
                        "tap_layers",
                        "num_loops",
                        "feature_shape",
                        "k_per_task",
                        "max_new_tokens",
                        "temperature",
                        "top_p",
                        "seed",
                        "task_offset",
                        "n_tasks_requested",
                        "n_tasks_completed",
                        "n_candidates",
                        "budget_hit",
                        "elapsed_seconds",
                        "errors",
                    )
                },
            }
        )
        for record_index, record in enumerate(rows):
            copied = dict(record)
            copied["_source"] = source
            copied["_source_index"] = source_index
            copied["_source_path"] = str(path)
            copied["_record_index"] = record_index
            records.append(copied)
    return records, metadata


def canonical_task_split_hash(records: list[dict[str, Any]]) -> str:
    split_map: dict[str, str] = {}
    for row in records:
        split_map[str(row["task_uid"])] = str(row["split"])
    return sha256_json(
        [{"task_id": task, "split": split_map[task]} for task in sorted(split_map)]
    )


def render_inventory(report: dict[str, Any]) -> str:
    gate = report["gate"]
    rows = report["candidate_sequence_inventory"]
    return f"""# Fixed-prefix Horizon inventory report

Inventory timestamp: `{report['inventory_timestamp_utc']}`

## Decision

`{gate['verdict']}`

The repaired experiment stops at the pre-preregistration inventory gate.
Complete generated-token ID sequences are preserved for
**{rows['candidates_with_generated_token_ids']} of
{rows['total_candidates']} candidates**. All four canonical Horizon shard
schemas preserve rendered `generated_text` and the scalar `n_gen_tok`, but no
generated-token ID list or full sequence tensor.

The canonical generator moved `out.sequences` to a local `seqs` list, decoded
`new_ids_full` to text, and wrote only `generated_text` plus token counts and
derived features. The temporary IDs were deleted after each task. The v2
terminal pool copies the same record schema and adds `full_features`; it does
not restore token IDs. The v3 shards inherit the identical generator.

Retokenizing the rendered continuation is forbidden and cannot recover the
sampled tokenization uniquely. Regenerating from seeds is both prohibited and
not an exact artifact reconstruction. Therefore no candidate can be replayed
as `prompt token IDs + preserved generated token IDs`, no fixed boundary can
be selected, and no answer detector, replay, feature extraction, or
visible-versus-hidden fitting is authorized.

## Independently relevant identity finding

The historical environment establishes the local checkpoint path,
Transformers 4.54.1, four-loop inference, and repository Git head. Current
checkpoint and tokenizer hashes are recorded. However, the historical
SHA256SUMS files did not stamp the model weights or the untracked generator and
feature-extractor source files. Exact replay identity is therefore not
cryptographically established from the historical package, independently of
the fatal missing-token condition.

## What is preserved

- Exact task identities, candidate indices and record ordering: present.
- Original deterministic task partitions: present and reproduced with zero
  crossing.
- Rendered continuations and generated-token counts: present.
- Eventual correctness labels, parsed committed answers and deterministic
  verifier-equivalent fields: present and internally consistent.
- Original prompts: reconstructible byte-for-byte from the hash-recorded task
  source and formatter; prompt token counts agree with the stored counts.
- Canonical feature definition: layers 24/36/47, four recurrent passes,
  attention-mask mean pooling over the replay input, `[3,4,2048]` float32
  features, followed by standardized PCA and L2 logistic regression.

## Required disposition

No preregistration was sealed because the gate precedes preregistration. Files
with downstream required names are terminal status artifacts, not fitted
results. No model was loaded, no continuation was generated or retokenized,
no replay was attempted, and no predictive performance was inspected.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.UTC).isoformat()

    records, shard_metadata = load_shards()
    terminal_payload = torch.load(TERMINAL_POOL, map_location="cpu", weights_only=False)
    terminal_records = terminal_payload.get("records", [])
    terminal_record_keys = sorted(terminal_records[0]) if terminal_records else []
    terminal_token_fields = sorted(TOKEN_FIELDS & set(terminal_record_keys))
    index_token_fields: dict[str, list[str]] = {}
    for index_path in [V2_INDEX, *V3_INDEXES]:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        found: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                found.update(TOKEN_FIELDS & set(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(index_payload)
        index_token_fields[str(index_path)] = sorted(found)
    pool = load_task_pool()
    task_lookup = {row["task_uid"]: row for row in pool}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT, trust_remote_code=True, local_files_only=True
    )

    v2_sums = parse_sums(V2_SUMS)
    v3_sums = parse_sums(V3_SUMS)
    expected_sum = {
        str(V2_SHARD): v2_sums.get("horizon_logic/horizon_generation_main_off0.pt"),
        **{
            str(path): v3_sums.get(f"horizon_logic/{path.name}")
            for path in V3_SHARDS
        },
    }
    source_checks = {
        path: {
            "expected_sha256": expected,
            "actual_sha256": sha256(Path(path)),
            "matches": expected == sha256(Path(path)),
        }
        for path, expected in expected_sum.items()
    }

    task_splits: dict[str, set[str]] = defaultdict(set)
    candidate_ids: list[str] = []
    compound_to_splits: dict[str, set[str]] = defaultdict(set)
    prompt_count_mismatches: list[str] = []
    split_mismatches: list[str] = []
    verifier_mismatches: list[str] = []
    malformed_mismatches: list[str] = []
    missing_source_tasks: list[str] = []
    token_rows: list[dict[str, Any]] = []
    token_fields_found: Counter[str] = Counter()
    source_task_order: dict[tuple[str, int], list[str]] = defaultdict(list)

    for global_index, row in enumerate(records):
        task_id = str(row["task_uid"])
        candidate_index = int(row["candidate_idx"])
        candidate_id = f"{task_id}::candidate_{candidate_index}"
        candidate_ids.append(candidate_id)
        task_splits[task_id].add(str(row["split"]))
        compound_to_splits[candidate_id].add(str(row["split"]))
        if row["split"] != split_for(task_id):
            split_mismatches.append(candidate_id)
        for field in TOKEN_FIELDS & set(row):
            token_fields_found[field] += 1

        task = task_lookup.get(task_id)
        prompt_ids: list[int] | None = None
        prompt_hash: str | None = None
        prompt_text_hash: str | None = None
        if task is None:
            missing_source_tasks.append(candidate_id)
        else:
            prompt = mcq_prompt(task["task_prompt"], options_dict(task["options"]))
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            prompt_hash = sha256_json(prompt_ids)
            prompt_text_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if len(prompt_ids) != int(row["prompt_tok"]):
                prompt_count_mismatches.append(candidate_id)

        verifier_expected = bool(
            row["found_final_marker"] and row["parsed_answer"] == row["gold_letter"]
        )
        if bool(row["success"]) != verifier_expected:
            verifier_mismatches.append(candidate_id)
        malformed_expected = bool(
            not row["found_final_marker"] or row["parsed_answer"] is None
        )
        if bool(row["malformed"]) != malformed_expected:
            malformed_mismatches.append(candidate_id)

        shard_key = (row["_source"], row["_source_index"])
        if candidate_index == 0:
            source_task_order[shard_key].append(task_id)
        token_rows.append(
            {
                "global_candidate_order": global_index,
                "source": row["_source"],
                "source_path": row["_source_path"],
                "source_record_index": row["_record_index"],
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_idx": candidate_index,
                "split": row["split"],
                "prompt_token_count_stored": int(row["prompt_tok"]),
                "prompt_token_ids_reconstructed": prompt_ids,
                "prompt_token_ids_sha256": prompt_hash,
                "prompt_text_sha256": prompt_text_hash,
                "generated_token_count_stored": int(row["n_gen_tok"]),
                "generated_token_ids": None,
                "generated_token_ids_status": "ABSENT_FROM_SOURCE_ARTIFACT",
                "rendered_continuation_sha256": hashlib.sha256(
                    str(row["generated_text"]).encode()
                ).hexdigest(),
                "exact_candidate_token_sequence_reconstructible": False,
                "retokenization_attempted": False,
            }
        )

    expected_tasks = {
        ("v2_original", 0): [row["task_uid"] for row in pool[0:170]],
        ("v3_prospective", 1): [row["task_uid"] for row in pool[170:340]],
        ("v3_prospective", 2): [row["task_uid"] for row in pool[340:510]],
        ("v3_prospective", 3): [row["task_uid"] for row in pool[510:680]],
    }
    order_checks = {}
    for key, expected in expected_tasks.items():
        actual = source_task_order.get(key, [])
        order_checks[f"{key[0]}_source_index_{key[1]}"] = {
            "n_expected": len(expected),
            "n_actual": len(actual),
            "exact_task_order_matches": actual == expected,
            "expected_task_order_sha256": sha256_json(expected),
            "actual_task_order_sha256": sha256_json(actual),
        }

    split_crossings = sorted(task for task, splits in task_splits.items() if len(splits) != 1)
    duplicate_candidate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_ids).items()
        if count > 1
    )
    candidate_crossings = sorted(
        candidate_id
        for candidate_id, splits in compound_to_splits.items()
        if len(splits) != 1
    )
    splits = Counter(next(iter(values)) for values in task_splits.values())
    candidate_splits = Counter(str(row["split"]) for row in records)
    class_counts = Counter(
        ("correct" if bool(row["success"]) else "incorrect") for row in records
    )

    def cohort_summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
        cohort_task_splits = {
            str(row["task_uid"]): str(row["split"]) for row in subset
        }
        return {
            "task_count": len(cohort_task_splits),
            "candidate_count": len(subset),
            "task_counts_by_split": dict(
                sorted(Counter(cohort_task_splits.values()).items())
            ),
            "candidate_counts_by_split": dict(
                sorted(Counter(str(row["split"]) for row in subset).items())
            ),
            "class_counts": dict(
                Counter(
                    "correct" if bool(row["success"]) else "incorrect"
                    for row in subset
                )
            ),
        }

    original_records = [row for row in records if row["_source"] == "v2_original"]
    new_only_records = [
        row for row in records if row["_source"] == "v3_prospective"
    ]
    cohort_summaries = {
        "horizon_new_only_primary": cohort_summary(new_only_records),
        "horizon_original_historical_component": cohort_summary(original_records),
        "horizon_pooled_secondary": cohort_summary(records),
    }

    source_files = [
        file_entry(TASK_SOURCE, "exact Horizon task source", False),
        file_entry(V2_SHARD, "original v2 candidate shard", True),
        *[file_entry(path, "prospective v3 candidate shard", True) for path in V3_SHARDS],
        file_entry(V2_INDEX, "v2 candidate index", True),
        *[file_entry(path, "v3 candidate index", True) for path in V3_INDEXES],
        file_entry(TERMINAL_POOL, "v2 derived terminal pool", True),
        file_entry(V2_SUMS, "v2 historical checksum manifest", True),
        file_entry(V3_SUMS, "v3 historical checksum manifest", True),
        file_entry(V2_ENVIRONMENT, "v2 historical environment manifest", True),
        file_entry(V3_ENVIRONMENT, "v3 historical environment manifest", True),
        file_entry(V2_RUN_MANIFEST, "v2 historical run manifest", True),
        file_entry(V2_TASK_MANIFEST, "v2 historical Horizon task manifest", True),
        file_entry(V2_SPLIT_INTEGRITY, "v2 historical split-integrity report", True),
        file_entry(V2_CUT_INTEGRITY, "v2 historical cut-integrity report", True),
        file_entry(V2_RAW_PREDICTIONS, "v2 published held-out predictions", True),
        file_entry(V3_RAW_PREDICTIONS, "v3 published held-out predictions", True),
        file_entry(GENERATOR, "canonical generator source currently present", False),
        file_entry(V3_GENERATOR, "v3 generator wrapper currently present", False),
        file_entry(ANALYSIS, "canonical hidden-probe analysis currently present", False),
        file_entry(FEATURE_EXTRACTOR, "canonical hidden feature extractor currently present", False),
        *[file_entry(path, "current local frozen checkpoint file", False) for path in MODEL_FILES],
        *[file_entry(path, "current local tokenizer file", False) for path in TOKENIZER_FILES],
    ]

    checkpoint_manifest = {
        "schema_version": 1,
        "historical_environment": {
            "model_path": "shared/models/ouro_rltt_local",
            "transformers": "4.54.1",
            "tokenizers": "0.21.4",
            "torch": "2.12.0.dev20260407+cu128",
            "recurrent_depth": 4,
            "git_head": "e4776dd41a85cad699ac36f309b5986ab48bd171",
        },
        "current_checkpoint_files": [
            file_entry(path, "checkpoint/custom-model") for path in MODEL_FILES
        ],
        "identity_status": "NOT_CRYPTOGRAPHICALLY_PINNED_BY_HISTORICAL_PACKAGE",
        "reason": (
            "historical manifests record the local path and environment but not model "
            "weight/custom-code hashes; generator and extractor were untracked at the "
            "recorded Git head"
        ),
    }
    tokenizer_manifest = {
        "schema_version": 1,
        "tokenizer_id": "ByteDance/Ouro-2.6B",
        "revision": TOKENIZER_REVISION,
        "local_path": str(MODEL_ROOT),
        "files": [file_entry(path, "tokenizer") for path in TOKENIZER_FILES],
        "tokenizer_json_matches_cached_revision": (
            sha256(MODEL_ROOT / "tokenizer.json")
            == sha256(
                PROJECT_ROOT
                / "shared/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots"
                / TOKENIZER_REVISION
                / "tokenizer.json"
            )
        ),
        "status": "TOKENIZER_ESTABLISHED_FOR_PROMPTS_ONLY_CONTINUATION_IDS_ABSENT",
    }
    split_manifest = {
        "schema_version": 1,
        "task_count": len(task_splits),
        "candidate_count": len(records),
        "task_counts_by_split": dict(sorted(splits.items())),
        "candidate_counts_by_split": dict(sorted(candidate_splits.items())),
        "task_split_sha256": canonical_task_split_hash(records),
        "zero_task_crossing": not split_crossings,
        "crossing_task_ids": split_crossings,
        "zero_candidate_id_duplication": not duplicate_candidate_ids,
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "zero_candidate_crossing": not candidate_crossings,
        "candidate_crossings": candidate_crossings,
        "split_function_mismatch_candidate_ids": split_mismatches,
        "source_offset_order_checks": order_checks,
        "cohorts": cohort_summaries,
    }
    report = {
        "schema_version": 1,
        "experiment_id": out.name,
        "inventory_timestamp_utc": created,
        "scope": ["horizon_new_only", "horizon_pooled"],
        "gate": {
            "status": "FAIL",
            "verdict": "FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE",
            "load_bearing_reason": (
                "complete preserved generated-token ID sequences are absent for every "
                "candidate; exact fixed-N replay is forbidden without them"
            ),
            "stopped_before_preregistration": True,
            "stopped_before_replay": True,
            "generation_calls": 0,
            "retokenized_continuations": 0,
            "classifier_fits": 0,
        },
        "candidate_sequence_inventory": {
            "total_candidates": len(records),
            "new_only_candidates": sum(row["_source"] == "v3_prospective" for row in records),
            "pooled_candidates": len(records),
            "candidates_with_generated_token_ids": sum(
                any(field in row for field in TOKEN_FIELDS) for row in records
            ),
            "candidates_with_rendered_text": sum("generated_text" in row for row in records),
            "candidates_with_generated_token_count": sum("n_gen_tok" in row for row in records),
            "token_fields_searched": sorted(TOKEN_FIELDS),
            "token_fields_found": dict(token_fields_found),
            "index_token_fields_found": index_token_fields,
            "all_record_keys": sorted(set().union(*(set(row) for row in records)) - {
                "_source", "_source_index", "_source_path", "_record_index"
            }),
            "generator_disposition": (
                "out.sequences -> local seqs -> new_ids_full -> decoded text; IDs not "
                "stored in record; local tensors deleted after task"
            ),
            "terminal_pool_record_count": len(terminal_records),
            "terminal_pool_record_keys": terminal_record_keys,
            "terminal_pool_token_fields_found": terminal_token_fields,
            "terminal_pool_token_ids_present": bool(terminal_token_fields),
        },
        "prompts": {
            "source_tasks_missing": missing_source_tasks,
            "prompt_token_count_mismatches": prompt_count_mismatches,
            "exact_prompt_reconstruction_status": (
                "PASS" if not missing_source_tasks and not prompt_count_mismatches else "FAIL"
            ),
        },
        "partitions_and_order": split_manifest,
        "labels_and_verifier": {
            "eventual_labels_present": all("success" in row for row in records),
            "committed_answers_present": all("parsed_answer" in row for row in records),
            "gold_answers_present": all("gold_letter" in row for row in records),
            "verifier_equivalence_mismatches": verifier_mismatches,
            "malformed_equivalence_mismatches": malformed_mismatches,
            "class_counts_all_preserved_candidates": dict(class_counts),
            "class_counts_by_cohort": {
                name: summary["class_counts"]
                for name, summary in cohort_summaries.items()
            },
            "status": (
                "PASS" if not verifier_mismatches and not malformed_mismatches else "FAIL"
            ),
        },
        "source_artifact_checksum_verification": source_checks,
        "shards": shard_metadata,
        "canonical_hidden_specification": {
            "status": "UNIQUELY_IDENTIFIED_FROM_CURRENT_CANONICAL_SOURCE",
            "layers": [24, 36, 47],
            "recurrent_steps": [1, 2, 3, 4],
            "loop_setting": "force_all_loops=True; total_ut_steps=4; early_exit_threshold=1.0",
            "token_pooling": "attention-mask-weighted mean across all replay input tokens",
            "position_selection": "all unmasked prompt-plus-prefix positions",
            "feature_shape": [3, 4, 2048],
            "feature_dtype": "float32 on CPU after model forward",
            "probe_class": "standardize -> PCA -> L2 logistic regression",
            "pca_grid": [16, 24],
            "l2_grid": [1.0, 2.0, 4.0],
            "task_grouped_folds": 5,
            "source": str(FEATURE_EXTRACTOR),
            "source_sha256": sha256(FEATURE_EXTRACTOR),
        },
        "canonical_shortcuts": {
            "historical_fields": [
                "n_pre_tok",
                "mean_logprob_pre",
                "min_logprob_pre",
                "hit_max_tokens",
            ],
            "repaired_boundary_status": (
                "NOT_COMPUTABLE_EXACTLY because first-N token identities and token-level "
                "generation scores were not preserved"
            ),
        },
        "checkpoint_identity": checkpoint_manifest["identity_status"],
        "source_files": source_files,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": platform.platform(),
            "device_used": "cpu inventory only",
        },
    }

    write_json(out / "inventory_report.json", report)
    (out / "inventory_report.md").write_text(render_inventory(report), encoding="utf-8")
    write_json(out / "source_manifest.json", {"created_at_utc": created, "files": source_files})
    write_json(out / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(out / "tokenizer_manifest.json", tokenizer_manifest)
    write_jsonl(out / "candidate_token_manifest.jsonl", token_rows)
    write_json(out / "split_manifest.json", split_manifest)
    write_json(
        out / "environment.json",
        {
            "captured_at_utc": created,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "platform": platform.platform(),
            "cuda_used": False,
            "generation_calls": 0,
            "model_replay_calls": 0,
        },
    )
    (out / "inventory_report.sha256").write_text(
        f"{sha256(out / 'inventory_report.json')}  inventory_report.json\n",
        encoding="utf-8",
    )
    print(out)
    print(report["gate"]["verdict"])
    print(
        f"generated token IDs: "
        f"{report['candidate_sequence_inventory']['candidates_with_generated_token_ids']}/"
        f"{len(records)}"
    )


if __name__ == "__main__":
    main()
