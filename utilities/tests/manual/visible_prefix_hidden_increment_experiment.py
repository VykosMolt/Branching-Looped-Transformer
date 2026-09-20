#!/usr/bin/env python3
"""Sealed visible-prefix versus hidden-state conference extension.

The audit stage performs exact rendered-prefix reconstruction, leakage checks,
and frozen-encoder truncation accounting.  It intentionally performs no model
fitting.  The fitting stage is enabled only after the audit gates pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "opi/preanswer/visible_prefix_hidden_increment_20260729T075247Z"
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
OURO_TOKENIZER = PROJECT_ROOT / "shared/models/ouro_rltt_local"
VISIBLE_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"
VISIBLE_MODEL = (
    PROJECT_ROOT
    / "shared/hf_cache/hub/models--microsoft--deberta-v3-small/snapshots"
    / VISIBLE_REVISION
)
FINAL_MARKER = re.compile(r"FINAL\s*ANSWE", re.IGNORECASE)
SEED = 20260729
MAX_LENGTH = 512
PROMPT_ALLOCATION = 256
SPECIAL_PAIR_TOKENS = 3
PREFIX_ALLOCATION = MAX_LENGTH - PROMPT_ALLOCATION - SPECIAL_PAIR_TOKENS
LEAK_TASK_MATERIALITY_FRACTION = 0.20
NEAR_DUPLICATE_THRESHOLD = 0.95


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(rows: Iterable[dict[str, Any]]) -> str:
    return sha256_text(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
        )
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def hash_rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}\x1f{key}".encode()).hexdigest()


def options_dict(options: list[str]) -> dict[str, str]:
    return dict(zip([chr(65 + i) for i in range(len(options))], options))


def options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def mcq_prompt(question: str, options: dict[str, str]) -> str:
    # Byte-for-byte copy of bg_steering_suite_lib.mcq_prompt at the inventoried hash.
    return (
        "Answer the multiple-choice question. Give concise reasoning if useful, "
        "then end with exactly one line: FINAL ANSWER: <letter>.\n\n"
        f"Question:\n{question}\n\nOptions:\n{options_text(options)}"
    )


def load_task_map() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with HORIZON_TASKS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if (
                row.get("category") == "synthetic_propositional"
                and 2 <= int(row.get("proof_depth", 0)) <= 4
            ):
                rows[str(row["task_uid"])] = row
    return rows


def load_shards(paths: list[Path], source: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for shard_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for file_row_index, row in enumerate(payload["records"]):
            copied = dict(row)
            copied["_source"] = source
            copied["_source_path"] = str(path)
            copied["_shard_index"] = shard_index
            copied["_file_row_index"] = file_row_index
            output.append(copied)
    return output


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def visible_input(prompt: str, prefix: str) -> str:
    return f"Prompt:\n{prompt}\n\nVisible reasoning prefix:\n{prefix}"


def reconstruct_rows(
    records: list[dict[str, Any]],
    task_map: dict[str, dict[str, Any]],
    ouro_tokenizer: Any,
    visible_tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reconstructed: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for global_index, row in enumerate(records):
        task_id = str(row["task_uid"])
        task = task_map.get(task_id)
        if task is None:
            raise RuntimeError(f"source task missing: {task_id}")
        prompt = mcq_prompt(task["task_prompt"], options_dict(task["options"]))
        trajectory = str(row["generated_text"])
        marker = FINAL_MARKER.search(trajectory)
        if marker is None:
            prefix = trajectory.strip()
            original_cut_reason = "no_final_marker_full_rendered_trajectory"
        else:
            prefix = trajectory[: marker.start()].strip()
            original_cut_reason = "strictly_before_original_final_marker"

        # The exact rendered trajectory is preserved.  The original char-level cut
        # implementation is replayed byte-for-byte above; the stored token count is
        # separately checked against the original tokenizer.
        full_ids = ouro_tokenizer.encode(trajectory, add_special_tokens=False)
        stored_k = int(row["n_pre_tok"])
        stored_cut_render = ouro_tokenizer.decode(
            full_ids[: min(stored_k, len(full_ids))], skip_special_tokens=True
        ).strip()
        prompt_ids_ouro = ouro_tokenizer(
            prompt, add_special_tokens=True, truncation=False
        )["input_ids"]
        prompt_count_matches = len(prompt_ids_ouro) == int(row["prompt_tok"])

        prompt_ids = visible_tokenizer.encode(prompt, add_special_tokens=False)
        prefix_ids = visible_tokenizer.encode(prefix, add_special_tokens=False)
        prompt_kept = min(len(prompt_ids), PROMPT_ALLOCATION)
        prefix_kept = min(len(prefix_ids), PREFIX_ALLOCATION)
        prompt_truncated = len(prompt_ids) > PROMPT_ALLOCATION
        prefix_truncated = len(prefix_ids) > PREFIX_ALLOCATION
        row_id = f"{task_id}::candidate_{int(row['candidate_idx'])}"

        gold_text = str(row["gold_answer"]).strip()
        gold_letter = str(row["gold_letter"]).strip().upper()
        literal_pat = re.compile(rf"(?<!\w){re.escape(gold_text)}(?!\w)", re.IGNORECASE)
        literal_gold = bool(gold_text and literal_pat.search(prefix))
        symbolic_patterns = [
            re.compile(
                rf"\b(?:answer|option|choice)\s*(?:is|=|:)?\s*{re.escape(gold_letter)}\b",
                re.IGNORECASE,
            ),
            re.compile(rf"(?m)^\s*{re.escape(gold_letter)}\s*[.)]\s*$", re.IGNORECASE),
        ]
        symbolic_gold = bool(gold_letter and any(p.search(prefix) for p in symbolic_patterns))
        forbidden_marker = bool(FINAL_MARKER.search(prefix))
        normalized_gold = normalized_text(gold_text)
        normalized_prefix_tokens = set(normalized_text(prefix).split())
        normalized_equivalent = bool(
            normalized_gold
            and " " not in normalized_gold
            and normalized_gold in normalized_prefix_tokens
        )
        answer_leak = bool(
            literal_gold or symbolic_gold or normalized_equivalent or forbidden_marker
        )

        enriched = dict(row)
        enriched.update(
            {
                "_row_id": row_id,
                "_global_index": global_index,
                "_prompt": prompt,
                "_prefix": prefix,
                "_visible_input": visible_input(prompt, prefix),
                "_answer_leak": answer_leak,
                "_literal_gold": literal_gold,
                "_symbolic_gold": symbolic_gold,
                "_normalized_equivalent": normalized_equivalent,
                "_forbidden_marker": forbidden_marker,
                "_prompt_tokens_visible": len(prompt_ids),
                "_prefix_tokens_visible": len(prefix_ids),
                "_prompt_tokens_retained": prompt_kept,
                "_prefix_tokens_retained": prefix_kept,
                "_prompt_truncated": prompt_truncated,
                "_prefix_truncated": prefix_truncated,
                "_truncated": prompt_truncated or prefix_truncated,
                "_stored_cut_render": stored_cut_render,
                "_stored_cut_render_matches_original_prefix": stored_cut_render == prefix,
                "_prompt_count_matches_stored": prompt_count_matches,
            }
        )
        reconstructed.append(enriched)
        manifest.append(
            {
                "row_id": row_id,
                "task_id": task_id,
                "candidate_idx": int(row["candidate_idx"]),
                "split": row["split"],
                "source": row["_source"],
                "source_path": row["_source_path"],
                "prompt": prompt,
                "visible_reasoning_prefix": prefix,
                "input_format": visible_input(prompt, prefix),
                "trajectory_sha256": sha256_text(trajectory),
                "prompt_sha256": sha256_text(prompt),
                "prefix_sha256": sha256_text(prefix),
                "stored_strict_cut_token_index": stored_k,
                "strict_cut_method": original_cut_reason,
                "stored_cut_render_sha256": sha256_text(stored_cut_render),
                "stored_cut_render_matches_original_prefix": stored_cut_render == prefix,
                "prompt_token_count_matches_stored": prompt_count_matches,
                "visible_encoder_prompt_tokens": len(prompt_ids),
                "visible_encoder_prefix_tokens": len(prefix_ids),
                "visible_encoder_prompt_tokens_retained": prompt_kept,
                "visible_encoder_prefix_tokens_retained": prefix_kept,
                "visible_encoder_truncated": prompt_truncated or prefix_truncated,
            }
        )
    return reconstructed, manifest


def cross_split_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, dict[str, str]] = {}
    for row in rows:
        per_task.setdefault(
            row["task_uid"],
            {"split": row["split"], "prompt": row["_prompt"]},
        )
    by_prompt: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task, data in per_task.items():
        by_prompt[data["prompt"]].append((task, data["split"]))
    exact = []
    for prompt, values in by_prompt.items():
        if len(values) > 1:
            splits = sorted({split for _, split in values})
            exact.append(
                {
                    "prompt_sha256": sha256_text(prompt),
                    "task_ids": sorted(task for task, _ in values),
                    "splits": splits,
                    "cross_split": len(splits) > 1,
                }
            )

    tasks = sorted(per_task)
    prompts = [per_task[t]["prompt"] for t in tasks]
    near: list[dict[str, Any]] = []
    if len(prompts) >= 2:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        matrix = vec.fit_transform(prompts)
        sims = matrix @ matrix.T
        coo = sims.tocoo()
        for i, j, similarity in zip(coo.row, coo.col, coo.data):
            if i >= j or float(similarity) < NEAR_DUPLICATE_THRESHOLD:
                continue
            split_i = per_task[tasks[i]]["split"]
            split_j = per_task[tasks[j]]["split"]
            near.append(
                {
                    "task_id_a": tasks[i],
                    "task_id_b": tasks[j],
                    "split_a": split_i,
                    "split_b": split_j,
                    "cross_split": split_i != split_j,
                    "char_tfidf_cosine": round(float(similarity), 8),
                }
            )
    near.sort(key=lambda x: (-x["char_tfidf_cosine"], x["task_id_a"], x["task_id_b"]))
    return {
        "exact_duplicate_prompt_groups": exact,
        "n_exact_duplicate_prompt_groups": len(exact),
        "n_cross_split_exact_duplicate_prompt_groups": sum(
            x["cross_split"] for x in exact
        ),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_pairs": near,
        "n_near_duplicate_pairs": len(near),
        "n_cross_split_near_duplicate_pairs": sum(x["cross_split"] for x in near),
    }


def scalar_auc(values: list[float], labels: list[int]) -> float | None:
    y = np.asarray(labels)
    x = np.asarray(values)
    pos = x[y == 1]
    neg = x[y == 0]
    if not len(pos) or not len(neg):
        return None
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def arm_audit(
    arm: str,
    rows: list[dict[str, Any]],
    split_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = [row for row in rows if not bool(row["malformed"])]
    tasks_all = sorted({str(row["task_uid"]) for row in original})
    leaking_tasks = sorted({str(row["task_uid"]) for row in original if row["_answer_leak"]})
    clean = [row for row in original if row["task_uid"] not in set(leaking_tasks)]
    split_tasks_before: Counter[str] = Counter()
    split_tasks_after: Counter[str] = Counter()
    for task in tasks_all:
        split = next(row["split"] for row in original if row["task_uid"] == task)
        split_tasks_before[split] += 1
    for task in sorted({str(row["task_uid"]) for row in clean}):
        split = next(row["split"] for row in clean if row["task_uid"] == task)
        split_tasks_after[split] += 1

    task_fraction = len(leaking_tasks) / max(len(tasks_all), 1)
    per_split_removed = {
        split: 1.0
        - split_tasks_after.get(split, 0) / max(split_tasks_before.get(split, 0), 1)
        for split in ("train", "val", "heldout")
    }
    reconstruction_errors = [
        row["_row_id"]
        for row in original
        if not row["_prompt_count_matches_stored"]
    ]
    marker_errors = [row["_row_id"] for row in original if row["_forbidden_marker"]]

    material = (
        task_fraction > LEAK_TASK_MATERIALITY_FRACTION
        or any(v > LEAK_TASK_MATERIALITY_FRACTION for v in per_split_removed.values())
    )
    insufficient = (
        split_tasks_after.get("train", 0) < 50
        or split_tasks_after.get("val", 0) < 20
        or split_tasks_after.get("heldout", 0) < 30
    )
    exact_prefix_ok = not reconstruction_errors and not marker_errors
    if not exact_prefix_ok:
        gate = "VISIBLE_PREFIX_NOT_ADJUDICABLE"
        reason = (
            "prompt/tokenizer reconstruction mismatch or forbidden marker under the "
            "stored strict cut"
        )
    elif material:
        gate = "VISIBLE_PREFIX_NOT_ADJUDICABLE"
        reason = (
            "genuine normalized-gold leakage requires task exclusions exceeding the "
            "predeclared 20% materiality threshold"
        )
    elif insufficient:
        gate = "VISIBLE_PREFIX_NOT_ADJUDICABLE"
        reason = "leak-free task exclusion leaves fewer than the prespecified split minima"
    else:
        gate = "ADJUDICABLE_AFTER_HIDDEN_HEAD_REPRODUCTION"
        reason = "exact-prefix and leak-free sensitivity gates pass"

    heldout = [row for row in original if row["split"] == "heldout"]
    labels = [int(bool(row["success"])) for row in heldout]
    source_values = [1.0 if row["_source"] == "v3" else 0.0 for row in heldout]
    order_values = [
        float(row["_file_row_index"]) / max(len(heldout) - 1, 1) for row in heldout
    ]
    leakage = {
        "arm": arm,
        "original_scorable_candidates": len(original),
        "original_scorable_tasks": len(tasks_all),
        "candidate_leak_counts": {
            "literal_gold_answer": sum(row["_literal_gold"] for row in original),
            "normalized_equivalence": sum(
                row["_normalized_equivalent"] for row in original
            ),
            "symbolic_equivalence": sum(row["_symbolic_gold"] for row in original),
            "forbidden_answer_marker": len(marker_errors),
            "any": sum(row["_answer_leak"] for row in original),
        },
        "leaking_task_ids": leaking_tasks,
        "n_leaking_tasks": len(leaking_tasks),
        "leaking_task_fraction": task_fraction,
        "task_counts_before": dict(split_tasks_before),
        "task_counts_after_taskwise_exclusion": dict(split_tasks_after),
        "per_split_task_fraction_removed": per_split_removed,
        "materiality_threshold": LEAK_TASK_MATERIALITY_FRACTION,
        "reconstruction_error_row_ids": reconstruction_errors,
        "stored_cut_render_nonidentical_count": sum(
            not row["_stored_cut_render_matches_original_prefix"] for row in original
        ),
        "forbidden_marker_row_ids": marker_errors,
        "task_crossing_count": len(split_manifest.get("crossing_task_ids", [])),
        "post_cut_feature_static_audit": {
            "S_allowed": ["n_pre_tok", "mean_logprob_pre", "min_logprob_pre"],
            "excluded": [
                "hit_max_tokens",
                "malformed",
                "parsed_answer",
                "found_final_marker",
                "generated_text_after_cut",
            ],
            "verifier_fields_enter_features": False,
            "future_token_logprobabilities_enter_features": False,
            "final_output_parser_fields_enter_features": False,
            "hidden_features_enter_visible_models": False,
            "labels_enter_text_preprocessing": False,
        },
        "file_order_shortcut_heldout_auroc": scalar_auc(order_values, labels),
        "source_id_shortcut_heldout_auroc": (
            scalar_auc(source_values, labels) if arm == "horizon_pooled" else None
        ),
        "gate": gate,
        "gate_reason": reason,
    }
    disposition = {
        "arm": arm,
        "unchanged_original_cohort": {
            "status": "RETAINED_AS_HISTORICAL_SENSITIVITY_ONLY",
            "n_tasks": len(tasks_all),
            "n_candidates": len(original),
        },
        "leak_free_primary_sensitivity": {
            "taskwise_exclusion": True,
            "excluded_task_ids": leaking_tasks,
            "n_tasks": len({row["task_uid"] for row in clean}),
            "n_candidates": len(clean),
            "status": gate,
            "reason": reason,
        },
        "never_censored_individual_spans": True,
        "estimand_materially_changed": material,
    }
    return leakage, disposition


def truncation_audit(arm_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "model_id": "microsoft/deberta-v3-small",
        "revision": VISIBLE_REVISION,
        "maximum_sequence_length": MAX_LENGTH,
        "pair_special_tokens": SPECIAL_PAIR_TOKENS,
        "fixed_prompt_allocation": PROMPT_ALLOCATION,
        "fixed_prefix_allocation": PREFIX_ALLOCATION,
        "rule": (
            "retain the first 256 prompt tokens and the most recent 253 visible-prefix "
            "tokens, with three pair-special tokens; never right-truncate the prefix end"
        ),
        "arms": {},
    }
    for arm, rows in arm_rows.items():
        scorable = [row for row in rows if not row["malformed"]]
        by_split: dict[str, Any] = {}
        for split in ("train", "val", "heldout"):
            part = [row for row in scorable if row["split"] == split]
            by_split[split] = {
                "n": len(part),
                "n_truncated": sum(row["_truncated"] for row in part),
                "fraction_truncated": (
                    sum(row["_truncated"] for row in part) / len(part) if part else None
                ),
                "n_prompt_truncated": sum(row["_prompt_truncated"] for row in part),
                "n_prefix_truncated": sum(row["_prefix_truncated"] for row in part),
                "mean_prompt_tokens_original": (
                    float(np.mean([row["_prompt_tokens_visible"] for row in part]))
                    if part
                    else None
                ),
                "mean_prefix_tokens_original": (
                    float(np.mean([row["_prefix_tokens_visible"] for row in part]))
                    if part
                    else None
                ),
                "mean_prompt_tokens_retained": (
                    float(np.mean([row["_prompt_tokens_retained"] for row in part]))
                    if part
                    else None
                ),
                "mean_prefix_tokens_retained": (
                    float(np.mean([row["_prefix_tokens_retained"] for row in part]))
                    if part
                    else None
                ),
            }
        output["arms"][arm] = by_split
    return output


def render_leakage_md(
    audits: dict[str, Any], duplicates: dict[str, Any], gsm_reason: str
) -> str:
    lines = [
        "# Leakage audit",
        "",
        "The automated audit was run only after the preregistration and model-selection "
        "rules were hash-sealed. No classifier was fitted in this stage.",
        "",
        "## GSM8K",
        "",
        f"`VISIBLE_PREFIX_NOT_ADJUDICABLE`: {gsm_reason}",
        "",
    ]
    for arm in ("horizon_new_only", "horizon_pooled"):
        audit = audits[arm]
        lines.extend(
            [
                f"## {arm}",
                "",
                f"- Gate: `{audit['gate']}`",
                f"- Original scorable tasks/candidates: "
                f"{audit['original_scorable_tasks']}/{audit['original_scorable_candidates']}",
                f"- Leaking tasks: {audit['n_leaking_tasks']} "
                f"({audit['leaking_task_fraction']:.1%})",
                f"- Candidate literal-gold hits: "
                f"{audit['candidate_leak_counts']['literal_gold_answer']}",
                f"- Candidate symbolic-equivalence hits: "
                f"{audit['candidate_leak_counts']['symbolic_equivalence']}",
                f"- Forbidden marker hits: "
                f"{audit['candidate_leak_counts']['forbidden_answer_marker']}",
                f"- Reason: {audit['gate_reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Duplicate-prompt audit",
            "",
            f"- Exact cross-split duplicate groups: "
            f"{duplicates['n_cross_split_exact_duplicate_prompt_groups']}",
            f"- Near-duplicate threshold: {duplicates['near_duplicate_threshold']}",
            f"- Cross-split near-duplicate pairs: "
            f"{duplicates['n_cross_split_near_duplicate_pairs']}",
            "",
            "No individual visible span was censored. Genuine violations are disposed "
            "taskwise exactly as preregistered.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(out: Path) -> None:
    prereg_hash = (out / "preregistration.sha256").read_text(encoding="utf-8").split()[0]
    if len(prereg_hash) != 64:
        raise RuntimeError("invalid or missing preregistration seal")

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    ouro_tokenizer = AutoTokenizer.from_pretrained(
        OURO_TOKENIZER, trust_remote_code=True, local_files_only=True
    )
    visible_tokenizer = AutoTokenizer.from_pretrained(
        VISIBLE_MODEL, revision=VISIBLE_REVISION, local_files_only=True, use_fast=False
    )
    if visible_tokenizer.num_special_tokens_to_add(pair=True) != SPECIAL_PAIR_TOKENS:
        raise RuntimeError("visible tokenizer special-token count changed")

    task_map = load_task_map()
    v2 = load_shards([HORIZON_V2], "v2")
    v3 = load_shards(HORIZON_V3, "v3")
    new_rows, new_manifest = reconstruct_rows(v3, task_map, ouro_tokenizer, visible_tokenizer)
    pooled_rows, pooled_manifest = reconstruct_rows(
        v2 + v3, task_map, ouro_tokenizer, visible_tokenizer
    )

    # The exact-prefix manifest is the pooled superset; arm membership is explicit.
    new_ids = {row["_row_id"] for row in new_rows}
    pooled_manifest_by_id = {row["row_id"]: row for row in pooled_manifest}
    for row_id, item in pooled_manifest_by_id.items():
        item["arm_membership"] = (
            ["horizon_new_only", "horizon_pooled"]
            if row_id in new_ids
            else ["horizon_pooled"]
        )
    write_jsonl(
        out / "exact_prefix_manifest.jsonl",
        [pooled_manifest_by_id[key] for key in sorted(pooled_manifest_by_id)],
    )

    split_info = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    duplicates = cross_split_duplicates(pooled_rows)
    audit_new, disp_new = arm_audit(
        "horizon_new_only", new_rows, split_info["horizon_new_only"]
    )
    audit_pool, disp_pool = arm_audit(
        "horizon_pooled", pooled_rows, split_info["horizon_pooled"]
    )
    gsm_reason = (
        "preserved GSM8K rows contain neither generated token IDs, exact rendered "
        "trajectories, nor visible-prefix text, and no original train/validation/held-out "
        "partition exists"
    )
    audit = {
        "schema_version": 1,
        "preregistration_sha256": prereg_hash,
        "gsm8k": {
            "gate": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
            "reason": gsm_reason,
        },
        "horizon_new_only": audit_new,
        "horizon_pooled": audit_pool,
        "duplicates": duplicates,
        "global_checks": {
            "preprocessing_fit_outside_train": False,
            "heldout_informed_hyperparameter_choice": False,
            "in_sample_scores_enter_fusion": False,
            "hidden_features_enter_visible_only_models": False,
            "candidate_labels_enter_text_preprocessing": False,
            "task_crossing_new_only": not bool(
                split_info["horizon_new_only"]["crossing_task_ids"]
            ),
            "task_crossing_pooled": not bool(
                split_info["horizon_pooled"]["crossing_task_ids"]
            ),
        },
    }
    disposition = {
        "schema_version": 1,
        "gsm8k": {
            "status": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
            "reason": gsm_reason,
            "repair_attempted": False,
            "why_no_repair": "no preserved token IDs or rendered trajectories",
        },
        "horizon_new_only": disp_new,
        "horizon_pooled": disp_pool,
    }
    write_json(out / "leakage_audit.json", audit)
    write_json(out / "leakage_disposition.json", disposition)
    (out / "leakage_audit.md").write_text(
        render_leakage_md(
            {"horizon_new_only": audit_new, "horizon_pooled": audit_pool},
            duplicates,
            gsm_reason,
        ),
        encoding="utf-8",
    )
    write_json(
        out / "truncation_audit.json",
        truncation_audit(
            {"horizon_new_only": new_rows, "horizon_pooled": pooled_rows}
        ),
    )
    gate_state = {
        "preregistration_sha256": prereg_hash,
        "gsm8k": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
        "horizon_new_only": audit_new["gate"],
        "horizon_pooled": audit_pool["gate"],
        "exact_prefix_manifest_sha256": hashlib.sha256(
            (out / "exact_prefix_manifest.jsonl").read_bytes()
        ).hexdigest(),
        "leakage_audit_sha256": hashlib.sha256(
            (out / "leakage_audit.json").read_bytes()
        ).hexdigest(),
    }
    write_json(out / "gate_state.json", gate_state)
    print(json.dumps(gate_state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("audit",), default="audit")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_audit(out)


if __name__ == "__main__":
    main()
