#!/usr/bin/env python3
"""Assemble the required terminal package after the fixed-token inventory fails.

This script is packaging-only.  It performs no tokenization of continuations,
generation, model loading, replay, detector development, or classifier fitting.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = PROJECT_ROOT / "opi/preanswer/fixed_prefix_horizon_20260729T082827Z"
INVENTORY_SCRIPT = PROJECT_ROOT / "shared/utilities/tests/manual/fixed_prefix_horizon_inventory.py"
FINALIZER_SCRIPT = PROJECT_ROOT / "shared/utilities/tests/manual/fixed_prefix_horizon_finalize.py"
WRAPPER = PROJECT_ROOT / "shared/utilities/tests/manual/reproduce_fixed_prefix_horizon_gate.sh"
V2_SHARD = (
    PROJECT_ROOT
    / "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic/"
    "horizon_generation_main_off0.pt"
)
V3_SHARDS = [
    PROJECT_ROOT
    / "opi/preanswer/horizon_power_v3_20260726/horizon_logic/"
    f"horizon_generation_v3_off{offset}.pt"
    for offset in (170, 340, 510)
]
VERDICT = "FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE"
GRID = [16, 32, 48, 64, 96, 128]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, source in [(V2_SHARD, "v2_original"), *[
        (item, "v3_prospective") for item in V3_SHARDS
    ]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload["records"]:
            copied = dict(row)
            copied["_source"] = source
            records.append(copied)
    return records


def status(reason: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN_INVENTORY_GATE_FAILURE",
        "verdict": VERDICT,
        "reason": reason,
        "generation_calls": 0,
        "replay_calls": 0,
        "classifier_fits": 0,
    }


def write_terminal_preregistration(out: Path, inventory_hash: str) -> None:
    text = f"""# Preregistration status: not reached

This is a terminal gate record, not a predictive-analysis preregistration.

The inventory gate ran before detector implementation, checkpoint selection,
replay, validation fitting, and held-out inspection. It found exact preserved
generated-token sequences for 0/2,720 Horizon candidates. The mission's
stopping rule therefore assigned `{VERDICT}` and prohibited all downstream
work.

- Inventory report SHA-256: `{inventory_hash}`
- Fixed grid that was not evaluated: `{GRID}`
- Selected primary N: none
- Detector frozen: no
- Replay performed: no
- Model selection performed: no
- Held-out predictive metrics opened: no

Retokenizing rendered continuations, regenerating from seeds, or inventing a
boundary would violate the repaired estimand. This file exists only so the
terminal evidence package has an explicit preregistration disposition.
"""
    path = out / "preregistration.md"
    path.write_text(text, encoding="utf-8")
    (out / "preregistration.sha256").write_text(
        f"{sha256(path)}  preregistration.md\n", encoding="utf-8"
    )


def write_detector_artifacts(out: Path) -> None:
    (out / "answer_detector_spec.md").write_text(
        f"""# Answer-bearing detector specification: not instantiated

Status: `{VERDICT}`

The detector was required to operate on the first N **preserved generated
tokens**. Because none of the 2,720 candidates preserves those IDs, no candidate
boundary can be rendered exactly without prohibited retokenization. Detector
implementation and training-only hand correction therefore did not begin.

The requested detection categories and grid remain design requirements, not
executed analyses. No answer spans were censored and no final-marker cut was
reused.
""",
        encoding="utf-8",
    )
    write_json(
        out / "answer_detector_training_audit.json",
        {
            **status("exact first-N generated-token prefixes do not exist"),
            "training_rows_reviewed": 0,
            "detector_code_corrections": [],
            "false_positive_count": None,
            "false_negative_count": None,
        },
    )
    (out / "answer_detector_hand_audit.md").write_text(
        f"""# Answer-detector hand audit

`{VERDICT}`

No detector hand audit was performed. Sampling candidate text at N would first
require exact preserved token IDs; using re-tokenized rendered text is
explicitly forbidden. Numeric, symbolic, categorical, marker-only,
false-positive, false-negative, intermediate-calculation, and near-boundary
categories are therefore all `NOT_RUN_INVENTORY_GATE_FAILURE`.
""",
        encoding="utf-8",
    )


def write_checkpoint_artifacts(out: Path, inventory: dict[str, Any]) -> None:
    with (out / "checkpoint_grid_training_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "N",
            "status",
            "training_candidates_with_exact_ids",
            "training_strict_clean_fraction",
            "training_retained_tasks",
            "training_correct_candidates",
            "training_incorrect_candidates",
            "selected",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for boundary in GRID:
            writer.writerow(
                {
                    "N": boundary,
                    "status": "NOT_EVALUABLE_GENERATED_TOKEN_IDS_ABSENT",
                    "training_candidates_with_exact_ids": 0,
                    "training_strict_clean_fraction": "",
                    "training_retained_tasks": "",
                    "training_correct_candidates": "",
                    "training_incorrect_candidates": "",
                    "selected": False,
                }
            )
    write_json(
        out / "checkpoint_selection.json",
        {
            **status("fixed grid cannot be applied to absent generated token IDs"),
            "fixed_grid": GRID,
            "selected_primary_N": None,
            "training_only_rule_applied": False,
            "failure_stage": "PRE_SELECTION_ARTIFACT_GATE",
        },
    )
    write_json(
        out / "fixed_primary_task_manifest.json",
        {
            **status("no primary N was selectable"),
            "selected_primary_N": None,
            "tasks": [],
        },
    )
    write_jsonl(
        out / "fixed_primary_candidate_manifest.jsonl",
        [
            {
                **status("no primary N was selectable"),
                "selected_primary_N": None,
                "candidate_count": 0,
            }
        ],
    )
    write_json(
        out / "truncation_audit.json",
        {
            **status("semantic-model inputs were never constructed"),
            "maximum_sequence_length": None,
            "fraction_truncated": None,
        },
    )


def write_replay_and_features(out: Path, inventory: dict[str, Any]) -> None:
    replay = {
        **status("exact prompt-plus-preserved-prefix sequences are unavailable"),
        "training_sample_size": 0,
        "token_agreement": None,
        "max_absolute_error": None,
        "rms_error": None,
        "checkpoint_identity": inventory["checkpoint_identity"],
        "model_loaded": False,
    }
    write_json(out / "replay_validation.json", replay)
    (out / "replay_validation.md").write_text(
        f"""# Replay validation

`{VERDICT}`

Replay was not attempted. The required generated-token IDs are absent for all
2,720 candidates, and the historical package did not cryptographically stamp
checkpoint weights/custom model code. Maximum error, RMS error and token
agreement are therefore undefined—not zero.
""",
        encoding="utf-8",
    )
    write_json(
        out / "hidden_feature_manifest.json",
        {
            **status("replay was prohibited"),
            "canonical_definition_inventoried": inventory[
                "canonical_hidden_specification"
            ],
            "features_extracted": 0,
            "feature_shards": [],
        },
    )
    table = pa.table(
        {
            "status": [VERDICT],
            "reason": ["no exact preserved generated-token sequences; no replay"],
            "candidate_id": [None],
            "feature_vector": pa.array([None], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, out / "hidden_features.parquet")


def write_model_artifacts(out: Path) -> None:
    with (out / "visible_model_validation_leaderboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = ["cohort", "family", "validation_auroc", "selected", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cohort in ("horizon_new_only", "horizon_pooled"):
            for family in ("V0", "V1", "V2", "V3"):
                writer.writerow(
                    {
                        "cohort": cohort,
                        "family": family,
                        "validation_auroc": "",
                        "selected": False,
                        "status": "NOT_RUN_INVENTORY_GATE_FAILURE",
                    }
                )
    write_json(
        out / "model_selection.json",
        {
            **status("no primary boundary or replay features exist"),
            "selected_visible_family": None,
            "heldout_used": False,
        },
    )
    write_json(
        out / "seed_level_results.json",
        {
            "V3": [
                {
                    "seed": seed,
                    "validation_auroc": None,
                    "heldout_auroc": None,
                    "status": "NOT_RUN_INVENTORY_GATE_FAILURE",
                }
                for seed in (20260729, 20260730, 20260731)
            ],
            "technical_execution_failure": False,
            "upstream_artifact_gate_failure": True,
        },
    )
    write_jsonl(
        out / "out_of_fold_predictions.jsonl",
        [status("no base score was fitted")],
    )
    write_jsonl(
        out / "heldout_predictions.jsonl",
        [status("held-out predictive evaluation was not opened")],
    )
    write_json(
        out / "nested_model_coefficients.json",
        {
            **status("V, H and HV were not fitted"),
            "V": None,
            "H": None,
            "HV": None,
        },
    )


def write_metric_artifacts(out: Path) -> None:
    endpoint = {
        "V0": None,
        "V1": None,
        "V2": None,
        "V3": None,
        "V": None,
        "H": None,
        "HV": None,
        "H_minus_V0": None,
        "H_minus_V": None,
        "HV_minus_V": None,
        "status": VERDICT,
    }
    write_json(
        out / "primary_metrics.json",
        {
            "cohort": "horizon_new_only",
            "selected_primary_N": None,
            "adequacy": None,
            "metrics": endpoint,
            "verdict": VERDICT,
        },
    )
    write_json(
        out / "secondary_metrics.json",
        {
            "cohort": "horizon_pooled",
            "role": "secondary precision estimate; not independent replication",
            "selected_primary_N": None,
            "metrics": endpoint,
            "verdict": VERDICT,
        },
    )
    write_json(
        out / "checkpoint_curve_metrics.json",
        {
            "fixed_grid": GRID,
            "checkpoints": [
                {
                    "N": boundary,
                    "status": "NOT_EVALUABLE_GENERATED_TOKEN_IDS_ABSENT",
                    "strict_clean_retention": None,
                    "V": None,
                    "H": None,
                    "HV": None,
                    "HV_minus_V": None,
                    "H_minus_V0": None,
                }
                for boundary in GRID
            ],
        },
    )
    write_json(
        out / "bootstrap_results.json",
        {
            **status("no held-out score pairs exist"),
            "replicates_preregistered": None,
            "replicates_run": 0,
            "HV_minus_V_ci95": None,
            "H_minus_V0_ci95": None,
        },
    )
    write_json(
        out / "simultaneous_checkpoint_bands.json",
        {
            **status("checkpoint curve was not run"),
            "bands": None,
        },
    )
    write_json(
        out / "equivalence_results.json",
        {
            **status("primary HV-minus-V endpoint does not exist"),
            "margin": [-0.02, 0.02],
            "equivalence_passed": False,
            "ci90": None,
            "not_equivalent_claim": False,
        },
    )
    write_json(
        out / "shuffled_label_results.json",
        {
            **status("no predictive pipeline was fitted"),
            "replicates_run": 0,
        },
    )
    write_json(
        out / "influence_analysis.json",
        {
            **status("no held-out endpoint exists"),
            "task_influence": None,
            "source_family_influence": None,
        },
    )


def write_integrity_artifacts(out: Path, inventory: dict[str, Any]) -> None:
    split = inventory["partitions_and_order"]
    leakage = {
        "status": "TERMINATED_BEFORE_PREFIX_LEAKAGE_ANALYSIS",
        "verdict": VERDICT,
        "zero_task_crossing": split["zero_task_crossing"],
        "zero_candidate_id_duplication": split["zero_candidate_id_duplication"],
        "zero_candidate_crossing": split["zero_candidate_crossing"],
        "exact_generated_token_sequences_available": False,
        "continuations_retokenized": False,
        "post_N_feature_leakage": "NOT_APPLICABLE_NO_N",
        "verifier_leakage": "NOT_APPLICABLE_NO_MODELS",
        "malformedness_leakage": "NOT_APPLICABLE_NO_MODELS",
        "future_token_logprobability_leakage": "NOT_APPLICABLE_NO_MODELS",
        "total_length_leakage": "NOT_APPLICABLE_NO_MODELS",
        "preprocessing_outside_train": False,
        "heldout_informed_selection": False,
        "in_sample_fusion_scores": False,
        "checkpoint_mismatch": inventory["checkpoint_identity"],
        "replay_mismatch": "NOT_TESTED_REPLAY_PROHIBITED",
    }
    write_json(out / "leakage_audit.json", leakage)
    (out / "leakage_audit.md").write_text(
        f"""# Leakage and integrity audit

`{VERDICT}`

The inventory confirms zero task crossing, zero duplicate candidate IDs and
zero candidate crossing. The original shard checksums match their historical
manifests. The repaired boundary leakage audit cannot be run because no exact
first-N generated-token sequence exists. Continuations were not retokenized.

All post-boundary leakage checks are `NOT_APPLICABLE_NO_N`; no feature matrix,
preprocessor, base score or fusion score was constructed. Replay mismatch is
undefined because replay was prohibited. Checkpoint identity is additionally
not cryptographically pinned by the historical package.
""",
        encoding="utf-8",
    )


def write_per_task(out: Path, records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["task_uid"])].append(row)
    fields = [
        "task_id",
        "split",
        "source",
        "candidate_count",
        "correct_candidates",
        "incorrect_candidates",
        "generated_token_ids_available",
        "selected_primary_task",
        "status",
    ]
    with (out / "per_task_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task_id, rows in sorted(grouped.items()):
            writer.writerow(
                {
                    "task_id": task_id,
                    "split": rows[0]["split"],
                    "source": rows[0]["_source"],
                    "candidate_count": len(rows),
                    "correct_candidates": sum(bool(row["success"]) for row in rows),
                    "incorrect_candidates": sum(not bool(row["success"]) for row in rows),
                    "generated_token_ids_available": False,
                    "selected_primary_task": False,
                    "status": VERDICT,
                }
            )


def write_hand_audit(out: Path) -> None:
    (out / "hand_audit.md").write_text(
        f"""# Post-model hand audit

`{VERDICT}`

No post-model hand audit categories exist because no primary boundary was
selected and no hidden, visible or fused model was fitted. Hidden-only correct,
visible-only correct, both-correct, both-wrong, disagreement, HV-gain, margin,
truncation and replay-drift samples are all undefined.

The detector hand audit was also not attempted: exact first-N prefixes cannot
be constructed without prohibited continuation retokenization. This is a
missing-artifact disposition, not a claim of zero false positives or false
negatives.
""",
        encoding="utf-8",
    )


def make_figures(out: Path) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    specs = [
        ("checkpoint_cleanliness_retention.png", "Checkpoint cleanliness and retention"),
        ("visible_hidden_fused_auroc.png", "Visible versus hidden versus fused AUROC"),
        ("paired_hv_minus_v_bootstrap.png", "Paired HV − V bootstrap"),
        ("h_minus_v0.png", "H − V0"),
        ("checkpoint_readout_curve.png", "Checkpoint readout curve"),
        ("calibration.png", "Calibration"),
        ("risk_coverage.png", "Risk–coverage"),
        ("answer_bearing_attrition.png", "Answer-bearing attrition"),
    ]
    for filename, title in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.axis("off")
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=15)
        ax.text(
            0.5,
            0.48,
            "NOT ADJUDICABLE",
            ha="center",
            va="center",
            fontsize=14,
            color="#a33a3a",
            weight="bold",
        )
        ax.text(
            0.5,
            0.34,
            "0 / 2,720 candidates preserve generated token IDs.\n"
            "No fixed-N boundary, replay, fitting, or held-out evaluation was permitted.",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(figdir / filename)
        plt.close(fig)


def final_report(out: Path, inventory: dict[str, Any], elapsed: float) -> str:
    split = inventory["partitions_and_order"]
    new_only = split["cohorts"]["horizon_new_only_primary"]
    original = split["cohorts"]["horizon_original_historical_component"]
    pooled = split["cohorts"]["horizon_pooled_secondary"]
    inventory_hash = sha256(out / "inventory_report.json")
    candidate_hash = sha256(out / "candidate_token_manifest.jsonl")
    return f"""# Final report: repaired fixed-prefix Horizon experiment

## Verdict

`{VERDICT}`

The repaired experiment is non-adjudicable at the mandatory artifact inventory
gate. This is not an underpowered result, a null result, evidence of
equivalence, or evidence for either hidden or visible advantage.

## Why the final-marker experiment was invalid

The prior sealed audit found that the final-marker prefix already contained
the normalized answer in 451/503 scorable new-only tasks and 598/669 pooled
tasks. “Before the final marker” was therefore not a genuinely answer-free
boundary. Those historical hidden-versus-shortcut results remain only
non-conditional sensitivities and are not reused as evidence for the repaired
question.

## Inventory gate

The four canonical, historically checksum-matched Horizon shards contain 2,720
candidates: 2,040 prospectively generated v3/new-only candidates plus 680
original v2 candidates. Exact prompts, prompt token counts, task IDs,
candidate indices/order, deterministic splits, rendered continuations,
eventual labels, committed parsed answers, generation metadata, and canonical
hidden-feature definitions are present.

The fatal field is absent:

- candidates with complete preserved generated-token IDs: **0/2,720**;
- candidates with only rendered continuation text and `n_gen_tok`: **2,720/2,720**;
- token ID fields in shard records: none;
- token ID fields in light indexes: none;
- token ID fields in the v2 derived terminal pool: none.

The canonical generator converted `out.sequences` to a transient list, decoded
each `new_ids_full` sequence into `generated_text`, stored only the text and
counts, then deleted the transient IDs. Retokenizing the rendered text cannot
establish the sampled tokenization uniquely and is explicitly forbidden.
Regenerating from seeds is generation, can drift, and is also forbidden.

The current tokenizer matches `ByteDance/Ouro-2.6B` revision
`1ed04250da1a9936042725d302e81c8fa2ab5abd`, and all reconstructed prompt token
counts match. However, the historical package did not stamp checkpoint weight
or custom-code hashes; it records only the local path, environment and Git
head. This independently prevents fully authenticated replay identity.

## Boundary selection and detector

No N was selected. The fixed grid `{GRID}` was never evaluated because its
inputs do not exist. Accordingly:

- strict-clean retention at every checkpoint: not estimable;
- training retained-class counts: not estimable;
- projected held-out adequacy: not estimable;
- answer-detector false positives/negatives: not estimable;
- hand detector audit: not run.

No detector code was frozen or corrected, no answer strings were censored, and
the contaminated final-marker cut was not reused.

## Replay and predictive results

Replay was not attempted and the model was not loaded. Maximum absolute replay
error, RMS error and token agreement are undefined. No hidden feature was
extracted.

Therefore all of the following are not estimable for both Horizon new-only and
Horizon pooled:

- AUROC(V0), AUROC(V1), AUROC(V2), AUROC(V3);
- AUROC(V), AUROC(H), AUROC(HV);
- H−V0, H−V and HV−V;
- calibration, Brier score, ECE, AUARC and risk–coverage;
- checkpoint curves, clustered confidence intervals and equivalence.

No visible family won validation. V3 seeds 20260729, 20260730 and 20260731 were
all `NOT_RUN_INVENTORY_GATE_FAILURE`, not technical resource failures. Practical
equivalence was not tested; visible advantage and conditional hidden increment
were not tested.

## Cohort integrity

- primary new-only tasks/candidates:
  {new_only['task_count']}/{new_only['candidate_count']}; task splits train
  {new_only['task_counts_by_split']['train']}, validation
  {new_only['task_counts_by_split']['val']}, held-out
  {new_only['task_counts_by_split']['heldout']};
- original historical component tasks/candidates:
  {original['task_count']}/{original['candidate_count']}; task splits train
  {original['task_counts_by_split']['train']}, validation
  {original['task_counts_by_split']['val']}, held-out
  {original['task_counts_by_split']['heldout']};
- secondary pooled tasks/candidates:
  {pooled['task_count']}/{pooled['candidate_count']}; task splits train
  {pooled['task_counts_by_split']['train']}, validation
  {pooled['task_counts_by_split']['val']}, held-out
  {pooled['task_counts_by_split']['heldout']};
- zero task crossing: {str(split['zero_task_crossing']).lower()};
- zero duplicate candidate IDs: {str(split['zero_candidate_id_duplication']).lower()};
- zero candidate crossing: {str(split['zero_candidate_crossing']).lower()};
- exact source offset/task ordering reproduced for all four shards: true;
- verifier-equivalent label mismatches: 0;
- malformed-field equivalence mismatches: 0.

The primary new-only and secondary pooled endpoints receive the same
`{VERDICT}`. No pooled precision estimate exists and pooling cannot replace the
missing primary result.

## Limitations and required repair

This experiment can be made adjudicable only by prospectively preserving the
exact prompt IDs, generated IDs, per-token log probabilities, tokenizer and
checkpoint hashes, custom model code hash, deterministic split, and verifier
record for every continuation. Existing rendered text cannot be upgraded into
the missing token artifact without changing the estimand.

The outcome neither restores nor rejects the original strict pre-answer
interpretation. It leaves that interpretation unsupported by the existing
Horizon artifacts.

## Artifacts, hashes and runtime

Artifact root:
`{out}`

- inventory report SHA-256: `{inventory_hash}`
- candidate token manifest SHA-256: `{candidate_hash}`
- inventory-to-package runtime: {elapsed:.1f} seconds

The checksum manifest records every payload hash. One-command reproduction of
the inventory gate and terminal package:

```bash
bash utilities/tests/manual/reproduce_fixed_prefix_horizon_gate.sh {out}
```

No generation, continuation retokenization, replay, fitting, or manuscript
modification occurs in that command.
"""


def integration_sketch() -> str:
    return f"""# Conference integration/correction sketch

## Repaired estimand

The desired extension asks whether hidden states at a fixed, genuinely
answer-free generated-token boundary add eventual-correctness information
beyond a strong reader of the identical prompt and prefix.

## Protocol disposition

The intended grid was N ∈ {GRID}, with training-only selection, conservative
task-complete answer cleanliness, frozen four-loop Ouro replay, and nested
V/H/HV evaluation. It could not be executed because the historical Horizon
artifacts preserve rendered continuations but no generated-token IDs.

## Conference claim boundary

Report `{VERDICT}`. The repaired primary result does not exist. Do not claim a
hidden increment, visible sufficiency, equivalence, visible advantage, or a
repaired fixed-prefix readout.

The historical final-marker hidden-versus-shortcut association may be retained
only as a visibly answer-contaminated, non-conditional sensitivity. It must not
be called strict pre-answer evidence and must not anchor the conference
proto-introspection claim.

## Effect on the conference argument

The conference extension should treat fixed-boundary conditionality as an open
evidence requirement. A future prospective run must preserve exact generated
IDs and cryptographically pin replay code/checkpoint identity. No manuscript
edit is made by this package.
"""


def correction_sketch() -> str:
    return f"""# Frozen arXiv manuscript correction sketch

This is a correction plan only. The public manuscript, source package, figures
and metadata were not edited.

## Historical claim categories affected

1. Claims that the Horizon final-marker cut is “strict pre-answer” or
   “answer-free.”
2. Claims that Horizon hidden states predict success before any visible answer
   commitment.
3. Claims that hidden information survives comparison with the exact visible
   reasoning prefix.
4. Cross-domain or replication language that treats Horizon as independent
   confirmation of the GSM8K strict-preanswer interpretation.
5. Figure captions, abstracts, summaries or limitations that use the Horizon
   hidden-versus-shortcut increment as hidden-specific conditional evidence.

## Narrowest accurate replacement

> On preserved Horizon continuations, a hidden-state-plus-shortcut model was
> associated with eventual correctness at a final-marker-defined capture.
> A later audit found that the visible prefix often already stated the
> normalized answer, so this result is a historical non-conditional sensitivity
> rather than a strict answer-free test.

## Results that remain valid

- The preserved continuations, deterministic verifier labels and task splits.
- The numerical hidden-versus-shortcut association for its historical,
  answer-contaminated construction.
- Engineering descriptions of four-loop feature capture, provided they are not
  interpreted as strict answer-free predictive evidence.

## Results whose interpretation changes

- Horizon final-marker AUROC and bootstrap results become historical
  non-conditional sensitivities.
- They cannot establish conditional hidden information beyond visible text.
- They cannot serve as a strict pre-answer domain replication.

## Repaired experiment disposition

The fixed-token repair neither restores nor rejects the strict pre-answer
interpretation. It is `{VERDICT}` because 0/2,720 candidates preserve generated
token IDs, and checkpoint/custom-code identity was not historically
cryptographically pinned. The strict interpretation remains unsupported, not
falsified.
"""


def write_protocol_manifest(out: Path, inventory_hash: str) -> None:
    write_json(
        out / "sealed_protocol_manifest.json",
        {
            "status": "TERMINAL_INVENTORY_GATE_MANIFEST_NOT_PREDICTIVE_PROTOCOL",
            "verdict": VERDICT,
            "inventory_report_sha256": inventory_hash,
            "inventory_script": str(INVENTORY_SCRIPT.relative_to(PROJECT_ROOT)),
            "inventory_script_sha256": sha256(INVENTORY_SCRIPT),
            "terminal_finalizer": str(FINALIZER_SCRIPT.relative_to(PROJECT_ROOT)),
            "terminal_finalizer_sha256": sha256(FINALIZER_SCRIPT),
            "reproduction_wrapper": str(WRAPPER.relative_to(PROJECT_ROOT)),
            "reproduction_wrapper_sha256": sha256(WRAPPER),
            "artifact_reproduction_wrapper": "reproduce.sh",
            "artifact_reproduction_wrapper_sha256": sha256(out / "reproduce.sh"),
            "fixed_grid_not_evaluated": GRID,
            "selected_primary_N": None,
            "detector_sealed": False,
            "replay_performed": False,
            "heldout_predictive_metrics_opened": False,
            "generation_calls": 0,
            "continuation_retokenization_calls": 0,
            "classifier_fits": 0,
        },
    )


def write_artifact_wrapper(out: Path) -> None:
    path = out / "reproduce.sh"
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
artifact_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${artifact_dir}/../../.." && pwd)"
exec bash "${project_root}/utilities/tests/manual/reproduce_fixed_prefix_horizon_gate.sh" "${artifact_dir}"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def checksum_package(out: Path) -> None:
    files = sorted(
        path
        for path in out.rglob("*")
        if path.is_file()
        and path.name not in {"checksum_manifest.json", "checksum_manifest.sha256"}
    )
    manifest = {
        "schema_version": 1,
        "root": str(out),
        "files": [
            {
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
        "self_hash_file": "checksum_manifest.sha256",
    }
    write_json(out / "checksum_manifest.json", manifest)
    (out / "checksum_manifest.sha256").write_text(
        f"{sha256(out / 'checksum_manifest.json')}  checksum_manifest.json\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    started = time.time()
    out = args.output_dir.resolve()
    inventory = json.loads((out / "inventory_report.json").read_text(encoding="utf-8"))
    if inventory["gate"]["verdict"] != VERDICT:
        raise RuntimeError("terminal finalizer called without the required failed gate")
    inventory_hash = sha256(out / "inventory_report.json")
    records = load_records()

    write_terminal_preregistration(out, inventory_hash)
    write_detector_artifacts(out)
    write_checkpoint_artifacts(out, inventory)
    write_replay_and_features(out, inventory)
    write_model_artifacts(out)
    write_metric_artifacts(out)
    write_integrity_artifacts(out, inventory)
    write_per_task(out, records)
    write_hand_audit(out)
    make_figures(out)
    write_artifact_wrapper(out)
    write_protocol_manifest(out, inventory_hash)

    inventory_created = dt.datetime.fromisoformat(inventory["inventory_timestamp_utc"])
    overall_elapsed = (
        dt.datetime.now(dt.UTC) - inventory_created.astimezone(dt.UTC)
    ).total_seconds()
    (out / "FINAL_REPORT.md").write_text(
        final_report(out, inventory, overall_elapsed), encoding="utf-8"
    )
    (out / "CONFERENCE_INTEGRATION_SKETCH.md").write_text(
        integration_sketch(), encoding="utf-8"
    )
    (out / "MANUSCRIPT_CORRECTION_SKETCH.md").write_text(
        correction_sketch(), encoding="utf-8"
    )
    write_json(
        out / "runtime.json",
        {
            "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "inventory_to_package_seconds": overall_elapsed,
            "finalizer_seconds": time.time() - started,
            "generation_calls": 0,
            "continuation_retokenization_calls": 0,
            "replay_calls": 0,
            "classifier_fits": 0,
            "python": sys.version,
            "platform": platform.platform(),
        },
    )
    checksum_package(out)
    print(VERDICT)
    print(out)


if __name__ == "__main__":
    main()
