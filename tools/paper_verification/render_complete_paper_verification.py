#!/usr/bin/env python3
"""Combine the Fable checks and remaining TODO evidence into final MD/JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/moloch/ouro_project")
FABLE = ROOT / "artifacts/reports/paper_verification/fable_hardening_checks_20260710_175017/fable_hardening_checks.json"
SLICE = ROOT / "artifacts/reports/proto_introspection/final_prewrite_hardening_2026-06-17/antisymmetrized_hh_pairwise_audit_2026-06-17.json"


def read(path: Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True, type=Path)
    args = ap.parse_args()
    report_dir = args.report_dir.resolve()
    evidence_path = report_dir / "remaining_todos_evidence.json"
    full_path = report_dir / "full_hh_audit/full_hh_antisymmetry.json"
    if not full_path.exists():
        raise FileNotFoundError(f"full audit not complete: {full_path}")
    fable, evidence, slice_audit, full = map(read, (FABLE, evidence_path, SLICE, full_path))
    loc = fable["check1"]
    pre = fable["check2"]
    s3 = evidence["s3b2_matched_random"]
    execution_segments = [
        {"batch_size": 2, "length_bucketed": False, "rows_completed": 884},
        {"batch_size": 4, "length_bucketed": False, "rows_completed": 164},
        {"batch_size": 2, "length_bucketed": False, "rows_completed": 156},
        {"batch_size": 1, "length_bucketed": False, "rows_completed": 314},
        {"batch_size": 4, "length_bucketed": True, "rows_completed": 7034},
    ]
    full["resume_execution_segments"] = execution_segments

    patches = {
        "section_3_4_full_audit": (
            f"A strict swap audit on the full 8,552-pair HH-RLHF test set confirms that the original evaluator's high canonical-order accuracy is substantially order-artifacted. "
            f"Fixed-order accuracy is {full['metrics']['fixed_order_accuracy']:.3f}, whereas strict antisymmetrized accuracy is {full['metrics']['strict_antisym_accuracy']:.3f} "
            f"(95% pair-bootstrap CI [{full['metrics']['strict_antisym_accuracy_ci95'][0]:.3f}, {full['metrics']['strict_antisym_accuracy_ci95'][1]:.3f}]). "
            "We therefore report the historical 95.2% only as fixed-order discovery-stage accuracy, not as relational discrimination."
        ),
        "section_3_7_replace": (
            "In a targeted replication using a bias-free antisymmetric linear difference probe, the same held-out HH accuracy was obtained for base Ouro-2.6B, Ouro-2.6B-Thinking, and Ouro-RLTT "
            "(83.75% for each; exact swap consistency 1.0). Thus the earlier 24%/95.2%/95.0% separation is specific to an unrecovered historical fixed-order evaluation and does not establish training-stage localization in clean relational units. "
            "We therefore do not claim that reasoning/RL training installs the linearly readable relational preference signal. Because the original probe weights were unavailable and the reconstructed row split permits opposite orientations of a source pair to cross train and evaluation, resolving training-stage localization requires a larger independently pair-split antisymmetric replication."
        ),
        "section_5": (
            "At the strict pre-answer cut, adding hidden features to the length-and-log-probability baseline raises AUROC from 0.731 to 0.797 (incremental ΔAUROC +0.066). "
            "A paired task-clustered bootstrap over 170 GSM8K problems (10,000 draws; all four candidates retained per sampled problem) gives a 95% CI of [+0.021, +0.112], which excludes zero. "
            "The improvement is statistically significant under task-clustered resampling, while remaining a modest incremental effect demonstrated in one powered domain."
        ),
        "section_6_4_s3b2": (
            f"Among the eight oracle-present S3B2 groups, the forced selector succeeds in 5/8 groups (0.625). Under uniform random choice within each group's ten-candidate pool, the group-specific success probabilities imply a matched expectation of {s3['matched_uniform_random_expectation']:.4f} ({s3['expected_random_hits']:.1f}/8 hits), not 0.625. "
            f"The exact Poisson-binomial probability of at least five random successes is p={s3['exact_null_probability_at_least_observed']:.4f}. The selector therefore exceeds the matched-random point expectation, but N=8 does not establish a statistically significant selection advantage; correctness is decodable, while reliable forced selection remains unestablished."
        ),
        "section_8_1_steering": (
            "Across seven pinned frozen-backbone intervention methods, no method produced reliable signed behavioral steering at safe magnitudes. The teacher-forced adapter proxy is nearly orthogonal to the empirical success direction (−0.0043) and raw NoNorm readout (−0.0006), while raw and empirical directions have cosine +0.1010. Independently trained teacher-forced and sequence-level adapters converge to cosine 0.951094, but the shared direction does not transfer to adapter-specific held-out free-generation control."
        ),
        "section_8_2_s1": (
            "In the four-task K-matched control, plain stochastic decoding used 12 samples per task (temperature 0.7, top-p 0.95, 96-token budget) and reached oracle 0.750. Sampled forks reached 0.611 (−0.139 versus plain sampling), while deterministic/greedy forks produced zero new-correct tasks. Frozen branch injection therefore adds no demonstrated reachability beyond matched stochastic decoding in this bounded test."
        ),
        "second_domain_limitation": (
            "Cross-domain pre-answer generality remains partial. A powered second-domain run was rejected at preflight: SVAMP front-loaded answers and failed parser/label checks, while parseable MATH generations were approximately 21/22 correct and the remaining failures were predominantly truncations, leaving no clean negative class. No full second-domain run was performed."
        ),
        "math_transfer_origin": (
            "The historical 47/100 Hendrycks MATH observation is retained only as explicitly unarchived, non-load-bearing motivation. Its raw denominator table and the exact single-shot baseline underlying the reported ~2.7× ratio were not recovered; the exact multiplier should be removed if publication policy requires every number to be artifact-reproducible."
        ),
    }
    combined = {
        "report": "complete_paper_verification_and_remaining_todos",
        "draft_read_only": evidence["draft_read_only"],
        "hard_rules": {"ouro_trained": False, "checkpoints_modified": False, "broad_generation_run": False, "paper_draft_modified": False},
        "prior_fable_checks": {
            "localization_verdicts": fable["verdict_constants"],
            "localization_results": loc["results"],
            "localization_identity_audit": loc["identity_audit"],
            "fixed_order_decomposition": fable["followup_evaluator_decomposition"],
            "preanswer": pre,
        },
        "hh_antisymmetry": {"saved_512_slice": slice_audit, "full_8552": full},
        "remaining_todos": evidence,
        "recommended_paper_patches": patches,
        "paper_actions": {
            "training_stage_localization": "REMOVE_AS_STRONG_RESULT_OR_RECAST_AS_FAILED_REPLICATION",
            "preanswer_primary": "KEEP_PRIMARY_CLAIM_VERIFIED",
            "fixed_order_evaluator": "KEEP_ONLY_WITH_FULL_ANTISYMMETRY_CAVEAT",
            "s3b2_selection": "REPORT_ABOVE_EXPECTATION_BUT_NOT_SIGNIFICANT",
            "s1_kmatched": "CLOSE_VERIFIED",
            "steering": "CLOSE_VERIFIED",
            "second_powered_domain": "CLOSE_AS_DOCUMENTED_PREFLIGHT_REJECTION_AND_LIMITATION",
            "math_transfer": "NON_LOAD_BEARING_OR_REMOVE_EXACT_MULTIPLIER",
            "latex_conversion": "DEFER_UNTIL_EVIDENCE_PATCHES_ARE_ACCEPTED",
        },
        "blockers": [
            "Original 84.5% probe weights and historical base=24% evaluation artifact were not recovered.",
            "The historical math-transfer denominator table and ~2.7x baseline artifact were not recovered.",
            "The powered second pre-answer domain has no clean accepted dataset after two failed preflights.",
            "The draft references a missing companion changelog.md.",
            f"A publication commit was not created because the shared worktree contains {evidence['git_state']['dirty_entry_count']} pre-existing dirty entries; HEAD and the status manifest are pinned instead.",
            "Markdown-to-LaTeX conversion is deferred because the no-draft-modification constraint remains active and v3.16 contains claims now requiring evidence patches.",
        ],
    }
    json_path = report_dir / "complete_paper_verification.json"
    json_path.write_text(json.dumps(combined, indent=2) + "\n")

    m = full["metrics"]
    md = f"""# Complete paper verification and remaining TODO resolution

Draft inspected read-only: `{evidence['draft_read_only']}`  
No Ouro training, checkpoint mutation, or broad generation was performed.

## Executive verdict

- **Training-stage localization:** not reproduced in clean antisymmetric-linear units. Base, Thinking, and RLTT each score 0.8375 in the recovered reconstruction. Remove the 24%/95.2%/95.0% localization claim as a strong result.
- **Strict pre-answer GSM8K:** verified. ΔAUROC = +0.065790 with fresh paired task-clustered 95% CI [+0.020707, +0.112254], excluding zero.
- **Full HH evaluator swap audit:** fixed-order {m['fixed_order_accuracy']:.4f}; strict antisymmetric {m['strict_antisym_accuracy']:.4f} (95% CI [{m['strict_antisym_accuracy_ci95'][0]:.4f}, {m['strict_antisym_accuracy_ci95'][1]:.4f}]). The fixed-order result is substantially order-artifacted.
- **S3B2 matched random:** observed 5/8 = 0.625 versus native uniform-random expectation 0.3625; exact P(X≥5) = 0.0869. Above expectation, not significant at 0.05.
- **S1 and steering spot-checks:** verified exactly against source JSON/diagnostic artifacts.
- **Second powered domain:** closed as a documented preflight rejection, not silently left untested.

## 1. Why the evaluator produced the old-looking high numbers

The evaluator genuinely outputs high canonical-order accuracy, but that is not the same measurement as an antisymmetric relational probe. On the controlled 200-pair cross-backbone slice, the original attention+normalization head gives canonical accuracies 0.950 (base), 0.950 (Thinking), and 0.945 (RLTT), while strict antisymmetrized accuracies are only 0.580, 0.595, and 0.600. Swapped-direction correctness is 0.110–0.130 and the symmetric order component has mean about +1.26 to +1.29, demonstrating a strong learned preference for the first argument. Attention pooling is not required for the canonical high result, and normalization mitigates rather than creates the order dominance.

This explains how the evaluator could surface ~95% numbers: chosen was canonically placed first, and the high-capacity fixed-order head exploited that regularity. It does **not** explain or recover the historical base=24% cell. The current controlled application of the same saved head gives base=95%, and no artifact for the 24% run was found. That cell most likely depended on an unrecovered mismatch in extraction locus, preprocessing/tokenization, dataset ordering, or checkpoint/run identity; the exact cause cannot be proved without the missing artifact. The safe conclusion is that the historical cross-stage localization pattern is not reproducible, not that the evaluator never produced the logged numbers.

## 2. Fable checks

### Antisymmetric-linear localization

The original ~84.5% weights were not saved, so the closest documented probe was reconstructed: raw/NoNorm candidate differences, layer-47 loop-boundary means for L1–L4 concatenated (8192 dimensions), bias-free score `w·(h_left-h_right)`, L-BFGS logistic fit, exact swap antisymmetry. All three backbones score 0.8375 on 80 held-out ordered rows (74 source pairs), source-pair-clustered CI [0.7349, 0.9351], with swap consistency 1.0. Features are not identical: Thinking–RLTT score Pearson is 0.9989 and Base–Thinking 0.9118, but decisions are essentially unchanged.

Caveat: the reconstructed historical protocol splits positive/negative orientations at row level, so opposite orientations of a source pair can cross train/evaluation. This is a failed targeted replication, not a definitive proof of equality across stages.

### Task-clustered pre-answer CI

Raw data were recovered for 170 tasks × 4 candidates = 680 examples (407 positive, 273 negative). Five-fold task-grouped out-of-fold predictions were recreated, then 170 task IDs were sampled with replacement in each of 10,000 paired bootstrap draws, retaining all four candidates for every sampled task occurrence. No draw was skipped.

- hidden+all AUROC: 0.796973
- length+logprob AUROC: 0.731183
- ΔAUROC: +0.065790
- task-clustered 95% CI: [+0.020707, +0.112254]
- bootstrap SD: 0.023464
- leave-one-task-out Δ range: [0.056000, 0.070557]

The original code path for the historical [+0.017, +0.114] interval is not fully preserved, but the fresh task-clustered recomputation independently verifies the significance claim.

## 3. HH strict swap audit

The pinned 512-pair artifact gives fixed-order 0.9668 and strict antisymmetric 0.6367 (95% CI [0.5957, 0.6777]). The new full-test audit evaluates all 8,552 HH test pairs in both orders with local `models/ouro_rltt_local`, `pairwise_epoch2.pt`, max length 768, bfloat16 backbone, float32 head, no quantization, and exact dataset indices.

- fixed-order accuracy: {m['fixed_order_accuracy']:.6f} (95% CI [{m['fixed_order_accuracy_ci95'][0]:.6f}, {m['fixed_order_accuracy_ci95'][1]:.6f}])
- swapped-direction accuracy: {m['swapped_direction_accuracy']:.6f}
- strict antisymmetric accuracy: {m['strict_antisym_accuracy']:.6f} (95% CI [{m['strict_antisym_accuracy_ci95'][0]:.6f}, {m['strict_antisym_accuracy_ci95'][1]:.6f}])
- strict sign-flip rate: {m['strict_sign_flip_rate']:.6f}
- normal/flipped score correlation: {m['normal_flipped_pearson']:.6f}
- symmetric/antisymmetric absolute-magnitude ratio: {m['symmetric_to_antisym_abs_ratio']:.6f}

The run used resume-safe inference segments recorded in JSON; all 8,552 rows are unique dataset indices. Model shard, evaluator, library, dtype, device, seed, and cache revision metadata are pinned in the full audit JSON.

## 4. Other empirical TODOs

### S3B2 native matched-random control

The eight oracle-present groups contain 10 candidates each, with positive counts 7, 1, 3, 2, 4, 2, 8, and 2. Uniform within-group random success therefore averages 0.3625 (expected 2.9/8), not 0.625 and not the inherited S3B1 macro 0.5833. The exact heterogeneous Bernoulli tail for at least five successes is 0.0869024. Report this as above random expectation but statistically unresolved at N=8.

### S1 K-matched control

Exact artifact values: four tasks; 12 plain samples/task; temperature 0.7; top-p 0.95; 96-token sampling budget (160-token long reference). Plain-sampling oracle is 0.750, sampled-fork screen mean is 0.611 (fork minus sampling −0.139), and greedy fork produces 0 new-correct tasks. Status: `FROZEN_FORK_CLOSED__SAMPLING_EXPLAINS_SCREEN__S3_IS_LEVER`.

### Steering closure

Exactly seven methods are enumerated in the consolidation artifact. Geometry values are adapter-vs-empirical −0.0042941, adapter-vs-raw −0.0005530, raw-vs-empirical +0.1010016, and independently trained adapter convergence +0.9510944. The draft's method count and cosine are verified.

### Second powered pre-answer domain

Status: `REJECTED_AT_PREFLIGHT_BOTH_CANDIDATES`; `CROSS_DOMAIN_VERDICT = PARTIAL_NOT_FULLY_ESTABLISHED`; action `STOP_AND_DOCUMENT`. SVAMP failed strict-cut validity and parser/label checks. MATH supplied long reasoning, but parseable generations were ~21/22 correct and failures were largely truncations, making the negative class either absent or a length artifact. No full recapture was warranted.

### Historical math-transfer origin

The 47/100 and ~2.7× values remain unarchived. No raw denominator table was recovered. They may remain only as explicitly non-load-bearing historical motivation; otherwise remove the exact multiplier.

## 5. Production TODOs

- Six requested figures/tables are generated under `figures/`: pre-answer AUROC, domain-transfer matrix, CoreContent/DualAnchor roles, cache/branch schematic, S3B2 detection-vs-selection, and the two-null orthogonality audit.
- All five explicit project-relative paths in the draft resolve. Of 23 backticked artifact-file tokens, all resolve by exact/suffix search except the genuinely absent companion `changelog.md`.
- Reproducibility state is pinned to Git HEAD `{evidence['git_state']['head_at_verification']}` plus `git_status_short.txt`. No commit was created because the shared worktree has {evidence['git_state']['dirty_entry_count']} dirty entries.
- Markdown-to-LaTeX conversion is deferred: the instruction forbids modifying the draft, and v3.16 first needs the evidence patches below accepted.

## 6. Exact recommended wording

### §3.4 full audit

{patches['section_3_4_full_audit']}

### §3.7 replacement

{patches['section_3_7_replace']}

### §5

{patches['section_5']}

### §6.4 S3B2

{patches['section_6_4_s3b2']}

### §8.1 steering

{patches['section_8_1_steering']}

### §8.2 matched sampling

{patches['section_8_2_s1']}

### Second-domain limitation

{patches['second_domain_limitation']}

### Historical math-transfer note

{patches['math_transfer_origin']}

## 7. Remaining blockers

""" + "\n".join(f"- {x}" for x in combined["blockers"]) + "\n"
    (report_dir / "complete_paper_verification.md").write_text(md)
    print(json.dumps({"markdown": str(report_dir / 'complete_paper_verification.md'), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
