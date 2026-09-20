#!/usr/bin/env python3
"""Finalize the evidence package after the sealed leakage gates terminate the run."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

import visible_prefix_hidden_increment_experiment as E


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "opi/preanswer/visible_prefix_hidden_increment_20260729T075247Z"
)
HIST_HORIZON = (
    PROJECT_ROOT / "opi/preanswer/horizon_power_v3_20260726/auroc_results_v3.json"
)
HIST_GSM = (
    PROJECT_ROOT
    / "opi/verification/paper_verification/fable_hardening_checks_20260710_175017/"
    "preanswer_task_clustered_ci.json"
)
GSM_PT = PROJECT_ROOT / "artifacts/reports/proto_introspection/within_domain_recapture.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def selected(rows: list[dict[str, Any]], predicate, n: int = 5) -> list[dict[str, Any]]:
    candidates = [row for row in rows if predicate(row)]
    candidates.sort(key=lambda row: E.hash_rank(row["_row_id"]))
    return candidates[:n]


def prepare_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    ouro = AutoTokenizer.from_pretrained(
        E.OURO_TOKENIZER, trust_remote_code=True, local_files_only=True
    )
    visible = AutoTokenizer.from_pretrained(
        E.VISIBLE_MODEL,
        revision=E.VISIBLE_REVISION,
        local_files_only=True,
        use_fast=False,
    )
    tasks = E.load_task_map()
    v2 = E.load_shards([E.HORIZON_V2], "v2")
    v3 = E.load_shards(E.HORIZON_V3, "v3")
    new, _ = E.reconstruct_rows(v3, tasks, ouro, visible)
    pooled, _ = E.reconstruct_rows(v2 + v3, tasks, ouro, visible)
    return new, pooled


def historical_results() -> dict[str, Any]:
    horizon = json.loads(HIST_HORIZON.read_text(encoding="utf-8"))
    gsm = json.loads(HIST_GSM.read_text(encoding="utf-8"))
    return {
        "gsm8k": {
            "construction": (
                "historical grouped-OOF hidden+length/logprob result; no visible-prefix "
                "reader and no original held-out split"
            ),
            "shortcut_auroc": gsm["point_estimates"]["length_plus_logprob_auroc"],
            "hidden_plus_shortcut_auroc": gsm["point_estimates"][
                "hidden_plus_all_auroc"
            ],
            "delta": gsm["point_estimates"]["delta"],
            "ci95": gsm["task_clustered_bootstrap"]["ci95_percentile"],
            "n_tasks": gsm["n_tasks"],
            "n_examples": gsm["n_examples"],
        },
        "horizon_new_only": {
            "construction": (
                "historical four-shortcut result; includes disallowed post-cut "
                "hit_max_tokens and is not the conference endpoint"
            ),
            "shortcut_auroc": horizon["new_only_4sc"]["auroc_shortcut_only"],
            "hidden_only_auroc": horizon["new_only_4sc"]["auroc_hidden_only"],
            "hidden_plus_shortcut_auroc": horizon["new_only_4sc"][
                "auroc_hidden_plus_shortcuts"
            ],
            "delta": horizon["new_only_4sc"][
                "incremental_auroc_combined_minus_shortcut"
            ],
            "ci95": horizon["new_only_4sc"][
                "paired_task_clustered_bootstrap_combined_minus_shortcut"
            ]["ci95"],
            "n_heldout": horizon["new_only_4sc"]["n_heldout"],
            "n_heldout_tasks": horizon["new_only_4sc"]["n_heldout_tasks"],
        },
        "horizon_pooled": {
            "construction": (
                "historical four-shortcut result; includes disallowed post-cut "
                "hit_max_tokens and is not the conference endpoint"
            ),
            "shortcut_auroc": horizon["pooled_4sc"]["auroc_shortcut_only"],
            "hidden_only_auroc": horizon["pooled_4sc"]["auroc_hidden_only"],
            "hidden_plus_shortcut_auroc": horizon["pooled_4sc"][
                "auroc_hidden_plus_shortcuts"
            ],
            "delta": horizon["pooled_4sc"][
                "incremental_auroc_combined_minus_shortcut"
            ],
            "ci95": horizon["pooled_4sc"][
                "paired_task_clustered_bootstrap_combined_minus_shortcut"
            ]["ci95"],
            "n_heldout": horizon["pooled_4sc"]["n_heldout"],
            "n_heldout_tasks": horizon["pooled_4sc"]["n_heldout_tasks"],
        },
    }


def write_model_placeholders(out: Path) -> None:
    leaderboard_rows = []
    for arm in ("horizon_new_only", "gsm8k", "horizon_pooled"):
        for family in ("V0", "V1", "V2", "V3"):
            leaderboard_rows.append(
                {
                    "arm": arm,
                    "family": family,
                    "input": (
                        "S" if family == "V0" else "prompt_plus_exact_visible_prefix"
                    ),
                    "validation_auroc": "",
                    "status": "NOT_RUN_UPSTREAM_VISIBLE_PREFIX_GATE",
                    "selected": False,
                }
            )
    with (out / "visible_model_validation_leaderboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard_rows[0]))
        writer.writeheader()
        writer.writerows(leaderboard_rows)

    write_json(
        out / "model_selection.json",
        {
            "selected_visible_family": None,
            "status": "NOT_RUN_UPSTREAM_VISIBLE_PREFIX_GATE",
            "reason": (
                "all arms were terminally non-adjudicable before classifier fitting; "
                "selection was not permitted"
            ),
            "heldout_used_for_selection": False,
        },
    )
    write_json(
        out / "seed_level_results.json",
        {
            "V3": [
                {
                    "seed": seed,
                    "status": "NOT_RUN_UPSTREAM_VISIBLE_PREFIX_GATE",
                    "validation_auroc": None,
                    "heldout_auroc": None,
                }
                for seed in (20260729, 20260730, 20260731)
            ],
            "execution_failure_type": "PREREGISTERED_UPSTREAM_STOPPING_RULE",
            "resource_failure": False,
        },
    )
    status_row = {
        "status": "NOT_RUN_UPSTREAM_VISIBLE_PREFIX_GATE",
        "contains_predictions": False,
    }
    write_jsonl(out / "out_of_fold_predictions.jsonl", [status_row])
    write_jsonl(out / "heldout_predictions.jsonl", [status_row])
    write_json(
        out / "nested_model_coefficients.json",
        {
            "V": None,
            "H": None,
            "HV": None,
            "status": "NOT_FITTED_UPSTREAM_VISIBLE_PREFIX_GATE",
        },
    )
    write_json(
        out / "shuffled_label_results.json",
        {
            "status": "NOT_RUN_NO_ADJUDICABLE_HELDOUT_ENDPOINT",
            "replicates_preregistered": 1000,
            "replicates_run": 0,
        },
    )


def write_per_task(out: Path, new: list[dict[str, Any]], pooled: list[dict[str, Any]]) -> None:
    leakage = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
    excluded_new = set(leakage["horizon_new_only"]["leaking_task_ids"])
    excluded_pool = set(leakage["horizon_pooled"]["leaking_task_ids"])
    rows_out: list[dict[str, Any]] = []

    def append_arm(arm: str, rows: list[dict[str, Any]], excluded: set[str]) -> None:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_uid"])].append(row)
        for task_id, items in sorted(by_task.items()):
            scorable = [row for row in items if not row["malformed"]]
            rows_out.append(
                {
                    "arm": arm,
                    "task_id": task_id,
                    "split": items[0]["split"],
                    "source": items[0]["_source"],
                    "proof_depth": items[0]["proof_depth"],
                    "n_candidates_all": len(items),
                    "n_candidates_scorable": len(scorable),
                    "n_positive_scorable": sum(bool(row["success"]) for row in scorable),
                    "n_negative_scorable": sum(
                        not bool(row["success"]) for row in scorable
                    ),
                    "n_visible_prefix_leak_candidates": sum(
                        row["_answer_leak"] for row in scorable
                    ),
                    "excluded_from_leak_free_sensitivity": task_id in excluded,
                    "conference_endpoint_auroc": "",
                    "status": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
                }
            )

    append_arm("horizon_new_only", new, excluded_new)
    append_arm("horizon_pooled", pooled, excluded_pool)
    gsm = torch.load(GSM_PT, map_location="cpu", weights_only=False)["records"]["gsm8k"]
    for record in sorted(gsm, key=lambda x: x["task_id"]):
        samples = record["samples"]
        rows_out.append(
            {
                "arm": "gsm8k",
                "task_id": record["task_id"],
                "split": "",
                "source": "within_domain_recapture",
                "proof_depth": "",
                "n_candidates_all": len(samples),
                "n_candidates_scorable": len(samples),
                "n_positive_scorable": sum(bool(row["correct"]) for row in samples),
                "n_negative_scorable": sum(
                    not bool(row["correct"]) for row in samples
                ),
                "n_visible_prefix_leak_candidates": "",
                "excluded_from_leak_free_sensitivity": "",
                "conference_endpoint_auroc": "",
                "status": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
            }
        )
    with (out / "per_task_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)


def audit_row_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Row: `{row['_row_id']}`",
            f"  - task/split/source: `{row['task_uid']}` / `{row['split']}` / `{row['_source']}`",
            f"  - correct: `{bool(row['success'])}`; malformed: `{bool(row['malformed'])}`; "
            f"truncated: `{bool(row['_truncated'])}`; leak: `{bool(row['_answer_leak'])}`",
            f"  - gold answer: `{row['gold_answer']}`; gold letter: `{row['gold_letter']}`",
            f"  - prompt SHA-256: `{E.sha256_text(row['_prompt'])}`",
            f"  - prefix SHA-256: `{E.sha256_text(row['_prefix'])}`",
            f"  - visible prefix reviewed:\n\n"
            + "\n".join(f"    {line}" for line in row["_prefix"].splitlines())
            + "\n",
        ]
    )


def write_hand_audit(out: Path, new: list[dict[str, Any]], pooled: list[dict[str, Any]]) -> None:
    leakage = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
    near_tasks = {
        item["task_id_a"]
        for item in leakage["duplicates"]["near_duplicate_pairs"]
    } | {
        item["task_id_b"]
        for item in leakage["duplicates"]["near_duplicate_pairs"]
    }
    sections: list[str] = [
        "# Deterministic hand audit",
        "",
        "Sampling rule: within each named cohort and category, sort candidate row IDs "
        "by SHA-256 of `20260729 + unit-separator + row_id` and take the first five. "
        "Rows may appear in more than one category. Every prefix below was reviewed "
        "without editing or censoring.",
        "",
        "The review confirmed that the dominant leakage is genuine: reasoning prefixes "
        "state the semantic conclusion (`True`, `False`, or `Unknown`) before the frozen "
        "`FINAL ANSWER` marker. No forbidden marker remained in a reconstructed prefix. "
        "The deterministic stored-cut reconstruction therefore did not cause the gold "
        "occurrences.",
        "",
    ]
    for arm, rows in (("horizon_new_only", new), ("horizon_pooled_v2_only", [
        row for row in pooled if row["_source"] == "v2"
    ])):
        categories = {
            "positive": lambda r: not r["malformed"] and bool(r["success"]),
            "negative": lambda r: not r["malformed"] and not bool(r["success"]),
            "malformed_or_structurally_unusual": lambda r: bool(r["malformed"])
            or bool(r["hit_max_tokens"]),
            "truncated_encoder_input": lambda r: bool(r["_truncated"]),
            "detected_gold_leak": lambda r: not r["malformed"] and bool(r["_answer_leak"]),
            "near_duplicate_candidate": lambda r: r["task_uid"] in near_tasks,
        }
        sections.extend([f"## {arm}", ""])
        for category, predicate in categories.items():
            sample = selected(rows, predicate)
            sections.extend([f"### {category}", ""])
            if not sample:
                sections.extend(["No eligible row.", ""])
            else:
                for row in sample:
                    sections.extend([audit_row_text(row), ""])
        sections.extend(
            [
                "### model-margin categories",
                "",
                "Largest visible margin, hidden margin, H-versus-V disagreement, and "
                "HV-gain categories are unavailable because the preregistered upstream "
                "leakage gate prohibited fitting V, H, and HV.",
                "",
                "### exact duplicate candidates",
                "",
                "No exact duplicate prompt group belonging to distinct task IDs was "
                "detected.",
                "",
            ]
        )

    gsm = torch.load(GSM_PT, map_location="cpu", weights_only=False)["records"]["gsm8k"]
    gsm_items = []
    for record in gsm:
        for sample in record["samples"]:
            gsm_items.append(
                {
                    "row_id": f"{record['task_id']}::candidate_{sample['sample']}",
                    "task_id": record["task_id"],
                    "correct": bool(sample["correct"]),
                    "parsed": bool(sample["parsed"]),
                    "n_pre_tok": int(sample["n_pre_tok"]),
                    "reason": sample["preanswer_reason"],
                }
            )
    sections.extend(
        [
            "## GSM8K",
            "",
            "The artifact has no visible text to hand-review. Deterministic metadata-only "
            "rows are recorded below; this inability is itself the exact-prefix blocker.",
            "",
        ]
    )
    for category, predicate in (
        ("positive", lambda r: r["correct"]),
        ("negative", lambda r: not r["correct"]),
        ("structurally_unusual", lambda r: not r["parsed"] or r["n_pre_tok"] == 0),
    ):
        candidates = [row for row in gsm_items if predicate(row)]
        candidates.sort(key=lambda row: E.hash_rank(row["row_id"]))
        sections.extend([f"### {category}", ""])
        for row in candidates[:5]:
            sections.append(f"- `{json.dumps(row, sort_keys=True)}`")
        sections.append("")
    sections.extend(
        [
            "GSM8K truncated-input, text-margin, disagreement, leakage-candidate, and "
            "duplicate-text hand-audit categories are not adjudicable because no exact "
            "candidate prefix text was preserved.",
            "",
        ]
    )
    (out / "hand_audit.md").write_text("\n".join(sections), encoding="utf-8")


def make_figures(out: Path, leakage: dict[str, Any], hist: dict[str, Any], trunc: dict[str, Any]) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

    # Historical non-conforming construction only.
    arms = ["GSM8K", "Horizon\nnew-only", "Horizon\npooled"]
    shortcuts = [
        hist["gsm8k"]["shortcut_auroc"],
        hist["horizon_new_only"]["shortcut_auroc"],
        hist["horizon_pooled"]["shortcut_auroc"],
    ]
    hidden = [
        hist["gsm8k"]["hidden_plus_shortcut_auroc"],
        hist["horizon_new_only"]["hidden_plus_shortcut_auroc"],
        hist["horizon_pooled"]["hidden_plus_shortcut_auroc"],
    ]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.18, shortcuts, 0.36, label="historical shortcuts")
    ax.bar(x + 0.18, hidden, 0.36, label="historical hidden + shortcuts")
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x, arms)
    ax.legend()
    ax.set_title("Historical sensitivity only — conference V/HV endpoints not adjudicable")
    fig.tight_layout()
    fig.savefig(figdir / "auroc_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, key in enumerate(("horizon_new_only", "horizon_pooled")):
        value = leakage[key]["leaking_task_fraction"]
        ax.bar(i, value, color="#bd4b4b")
        ax.text(i, value + 0.015, f"{value:.1%}", ha="center")
    ax.axhline(E.LEAK_TASK_MATERIALITY_FRACTION, color="black", ls="--", label="20% gate")
    ax.set_xticks([0, 1], ["Horizon new-only", "Horizon pooled"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of scorable tasks with pre-marker gold occurrence")
    ax.legend()
    ax.set_title("HV versus V bootstrap not run: leakage gate")
    fig.tight_layout()
    fig.savefig(figdir / "hv_minus_v_paired_bootstrap.png")
    plt.close(fig)

    for filename, title in (
        ("risk_coverage.png", "Risk–coverage"),
        ("calibration.png", "Calibration"),
        ("prompt_only_vs_prompt_prefix.png", "Prompt-only versus prompt-plus-prefix"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            f"{title} not estimated",
            ha="center",
            va="center",
            fontsize=15,
        )
        ax.text(
            0.5,
            0.42,
            "All classifier fitting stopped at the preregistered visible-prefix leakage gate.",
            ha="center",
            va="center",
            wrap=True,
        )
        fig.tight_layout()
        fig.savefig(figdir / filename)
        plt.close(fig)

    # Prefix-length/truncation diagnostic, explicitly not performance.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["new train", "new val", "new heldout", "pool train", "pool val", "pool heldout"]
    values = []
    for arm in ("horizon_new_only", "horizon_pooled"):
        for split in ("train", "val", "heldout"):
            values.append(trunc["arms"][arm][split]["fraction_truncated"])
    ax.bar(np.arange(len(values)), values, color="#547aa5")
    ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    ax.set_ylabel("fraction truncated")
    ax.set_title("Performance by prefix length not estimated; truncation frequency shown")
    fig.tight_layout()
    fig.savefig(figdir / "performance_by_prefix_length.png")
    plt.close(fig)


def report_text(
    out: Path,
    leakage: dict[str, Any],
    trunc: dict[str, Any],
    hist: dict[str, Any],
    elapsed: float,
) -> str:
    new = leakage["horizon_new_only"]
    pool = leakage["horizon_pooled"]
    new_tr = trunc["arms"]["horizon_new_only"]["heldout"]
    pool_tr = trunc["arms"]["horizon_pooled"]["heldout"]
    return f"""# Final report: visible-prefix versus hidden-state conference gate

## Outcome

The conference endpoint is **not adjudicable**. This is not a null result and
not evidence that visible and hidden constructions are equivalent.

The preserved GSM8K strict-preanswer cohort lacks exact candidate trajectories
or prefix text and lacks an original train/validation/held-out partition. The
Horizon cohorts preserve exact rendered trajectories and splits, but they fail
the required strict visible-prefix contract: the pre-marker reasoning prefix
literally or symbolically states the normalized gold conclusion in
{new['n_leaking_tasks']} of {new['original_scorable_tasks']} new-only tasks
({new['leaking_task_fraction']:.1%}) and {pool['n_leaking_tasks']} of
{pool['original_scorable_tasks']} pooled tasks
({pool['leaking_task_fraction']:.1%}). Taskwise removal leaves only
{new['task_counts_after_taskwise_exclusion'].get('train', 0)} train,
{new['task_counts_after_taskwise_exclusion'].get('val', 0)} validation, and
{new['task_counts_after_taskwise_exclusion'].get('heldout', 0)} held-out
new-only tasks. This exceeds the sealed 20% materiality threshold in every
split and falls below the sealed minimum task counts.

Consequently, fitting a strong visible reader on the unchanged Horizon text
would answer a different, answer-contaminated question. Fitting on the small
post-exclusion residue would also change the estimand materially. The sealed
stopping rule therefore prohibited V1/V2/V3, T, R reproduction, V/H/HV fusion,
held-out endpoint evaluation, bootstrapping, equivalence testing, calibration,
and selective-prediction analyses.

## Direct answers to the conference questions

1. **Does visible prompt-plus-prefix text explain the published hidden-state
   effect?** Not adjudicable from these preserved strict cohorts. Horizon
   visible prefixes often contain the semantic answer, while GSM8K exact
   prefixes are absent.
2. **Do hidden states retain a statistically supported conditional increment
   beyond the strongest visible reader?** Not established; HV−V was not validly
   estimable.
3. **Are visible and hidden constructions practically equivalent?** Not
   established. The clustered equivalence test was not run.
4. **Does visible text outperform the hidden construction?** Not established.
5. **Was HV fully adjudicable?** No, in every arm.
6. **Which visible family won validation?** None; selection was prohibited by
   the upstream gate.
7. **V3 seed-level results:** seeds 20260729, 20260730, and 20260731 were all
   `NOT_RUN_UPSTREAM_VISIBLE_PREFIX_GATE`; this was not a resource failure.

## Endpoint hierarchy and verdicts

| Arm | Endpoint status | Verdict |
|---|---|---|
| Horizon Logic new-only (designated primary) | HV−V not estimable | `VISIBLE_PREFIX_NOT_ADJUDICABLE` |
| GSM8K (replication) | exact prefix and original split absent | `VISIBLE_PREFIX_NOT_ADJUDICABLE` |
| Horizon Logic pooled (secondary precision) | HV−V not estimable | `VISIBLE_PREFIX_NOT_ADJUDICABLE` |
| Aggregate | designated primary not adjudicable | `VISIBLE_PREFIX_AGGREGATE_NOT_ADJUDICABLE` |

No fallback H−V verdict is assigned: GSM8K has no valid V construction, and
Horizon's leak-free V estimand is materially changed and under the minimum
split sizes. A fallback comparison would not restore the requested conditional
question.

## Historical sensitivity, not conference evidence

The unchanged historical analyses are retained, as required, but they do not
answer HV−V:

- GSM8K grouped OOF: shortcuts AUROC
  {hist['gsm8k']['shortcut_auroc']:.3f}, hidden+shortcuts
  {hist['gsm8k']['hidden_plus_shortcut_auroc']:.3f}, delta
  {hist['gsm8k']['delta']:+.3f}, task-clustered 95% CI
  [{hist['gsm8k']['ci95'][0]:+.3f}, {hist['gsm8k']['ci95'][1]:+.3f}]. This has
  no preserved exact visible prefix and no original held-out split.
- Horizon new-only historical four-shortcut result: shortcuts AUROC
  {hist['horizon_new_only']['shortcut_auroc']:.3f}, hidden+shortcuts
  {hist['horizon_new_only']['hidden_plus_shortcut_auroc']:.3f}, delta
  {hist['horizon_new_only']['delta']:+.3f}, 95% CI
  [{hist['horizon_new_only']['ci95'][0]:+.3f},
  {hist['horizon_new_only']['ci95'][1]:+.3f}].
- Horizon pooled historical four-shortcut result: shortcuts AUROC
  {hist['horizon_pooled']['shortcut_auroc']:.3f}, hidden+shortcuts
  {hist['horizon_pooled']['hidden_plus_shortcut_auroc']:.3f}, delta
  {hist['horizon_pooled']['delta']:+.3f}, 95% CI
  [{hist['horizon_pooled']['ci95'][0]:+.3f},
  {hist['horizon_pooled']['ci95'][1]:+.3f}].

The Horizon historical construction includes `hit_max_tokens`, which this
conference protocol forbids as post-cut generation metadata. None of these
historical deltas is conditional on a strong visible-prefix reader.

## Leakage and reconstruction

No forbidden `FINAL ANSWER` marker survived in a reconstructed Horizon prefix,
and no prompt token-count reconstruction error was found. The gold occurrences
are genuine stored-prefix content, not a tokenizer/rendering repair error.
The hand audit confirms that reasoning commonly states `True`, `False`, or
`Unknown` before the final marker. Structurally unusual prefixes sometimes
contain incomplete marker fragments such as `FINAL ANSW`; these do not satisfy
the frozen original `FINAL\\s*ANSWE` cut regex and are recorded without repair.
No individual span was censored.

There were
{leakage['duplicates']['n_cross_split_exact_duplicate_prompt_groups']} exact
cross-split duplicate-prompt groups and
{leakage['duplicates']['n_cross_split_near_duplicate_pairs']} cross-split
near-duplicate pairs at the sealed 0.95 character-TFIDF threshold. File-order
held-out AUROC was {new['file_order_shortcut_heldout_auroc']:.3f} new-only and
{pool['file_order_shortcut_heldout_auroc']:.3f} pooled; pooled source-ID AUROC
was {pool['source_id_shortcut_heldout_auroc']:.3f}. These diagnostics did not
enter any model.

## Truncation

Under the frozen DeBERTa-v3-small allocation, new-only held-out truncation was
{new_tr['fraction_truncated']:.1%} ({new_tr['n_truncated']}/{new_tr['n']});
pooled held-out truncation was {pool_tr['fraction_truncated']:.1%}
({pool_tr['n_truncated']}/{pool_tr['n']}). All truncation was prefix overflow;
zero prompts exceeded the 256-token prompt allocation. Because no classifier
was fitted, truncation-performance sensitivity is not estimable.

## Domain-specific limitations

- GSM8K is blocked by missing exact visible candidate text and missing original
  three-way partitions.
- Horizon is blocked by pervasive genuine answer content before the stored
  final-answer marker.
- The historical Horizon scorable cohort was itself selected using
  post-generation malformedness; this selection is preserved only for
  historical comparability and is not used as a feature.
- Raw hidden-only train/validation scores were not preserved. Frozen hidden
  features and a deterministic reproduction specification exist, but the
  reproduction was not run after the earlier terminal leakage gate.
- Re-encoding the exact rendered trajectories and applying the stored token
  count did not byte-match the original char-cut prefix on
  {new['stored_cut_render_nonidentical_count']} new-only and
  {pool['stored_cut_render_nonidentical_count']} pooled scorable candidates.
  This reflects loss of the original generated token IDs and reinforces the
  need for prospective token-ID preservation. The original hidden feature
  input itself remains exactly reconstructible because the frozen generator
  implementation built it from the preserved rendered text and the frozen
  char-level cut function.
- A new prospective cohort with an answer-semantic cut or a separately
  annotated answer-free reasoning boundary is required. This package does not
  authorize generation and therefore cannot repair the missing estimand.

## Artifacts and reproducibility

Artifact root:
`{out}`

Key seals:

- inventory JSON SHA-256:
  `879fd166fd2d2512dd4f2527a3cc483808bfaf946adc400c143c14e4b6a6c84a`
- preregistration SHA-256:
  `1cf90fe1c26ca462c261489dba207af82828d6ff648b15d0a930d0db76f629b0`
- exact-prefix manifest SHA-256:
  `{sha256(out / 'exact_prefix_manifest.jsonl')}`
- leakage audit SHA-256:
  `{sha256(out / 'leakage_audit.json')}`

The checksum manifest contains the hash of every payload file. The
inventory-to-final-package runtime was {elapsed:.1f} seconds; preserved
source-generation runtimes remain in the source payloads and are not counted
as new runtime.

Reproduction from the sealed artifact root:

```bash
bash utilities/tests/manual/reproduce_visible_prefix_hidden_increment_evidence_package.sh \
  {out}
```

No frozen manuscript or arXiv tarball file was modified.
"""


def integration_text() -> str:
    return """# Conference integration sketch

Add a short “Visible-prefix conditionality gate” subsection to the conference
extension, not to the frozen arXiv package.

Suggested text:

> We preregistered a zero-generation comparison between a strong external
> prompt-plus-prefix reader and the frozen hidden-state readout. The endpoint
> could not be adjudicated from the preserved cohorts. GSM8K did not preserve
> exact candidate prefix text or the required original three-way split. In
> Horizon Logic, deterministic reconstruction showed that almost 90% of
> scorable tasks had at least one pre-marker prefix that explicitly stated the
> normalized gold conclusion. Taskwise exclusion exceeded the preregistered
> materiality threshold and left an under-sized held-out arm. We therefore did
> not fit or compare V, H, and HV, did not test equivalence, and retain the
> published hidden-versus-shortcut results only as historical sensitivities.

Use the exact verdict labels:

- Horizon new-only: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- GSM8K: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- Horizon pooled: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- aggregate: `VISIBLE_PREFIX_AGGREGATE_NOT_ADJUDICABLE`

Do not write that the visible reader explains the hidden effect, that the
hidden readout has a conditional increment, that the constructions are
equivalent, or that visible text wins. The next valid experiment needs
prospective preservation of token IDs, a task-disjoint three-way split, and a
cut placed before any semantic commitment to the answer—not merely before the
literal `FINAL ANSWER` marker.
"""


def checksum_package(out: Path) -> None:
    files = sorted(
        path
        for path in out.rglob("*")
        if path.is_file()
        and path.name not in {"checksum_manifest.json", "checksum_manifest.sha256"}
    )
    payload = {
        "schema_version": 1,
        "root": str(out),
        "self_hash_file": "checksum_manifest.sha256",
        "files": [
            {
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    write_json(out / "checksum_manifest.json", payload)
    (out / "checksum_manifest.sha256").write_text(
        f"{sha256(out / 'checksum_manifest.json')}  checksum_manifest.json\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()
    out = args.output_dir.resolve()
    gate = json.loads((out / "gate_state.json").read_text(encoding="utf-8"))
    expected = {
        "gsm8k": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
        "horizon_new_only": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
        "horizon_pooled": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
    }
    if any(gate[key] != value for key, value in expected.items()):
        raise RuntimeError("finalizer is only valid for the observed terminal gate state")

    new, pooled = prepare_rows()
    leakage = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
    trunc = json.loads((out / "truncation_audit.json").read_text(encoding="utf-8"))
    hist = historical_results()

    write_model_placeholders(out)
    write_per_task(out, new, pooled)
    write_hand_audit(out, new, pooled)

    verdicts = {
        "horizon_new_only": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
        "gsm8k": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
        "horizon_pooled": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
        "aggregate": "VISIBLE_PREFIX_AGGREGATE_NOT_ADJUDICABLE",
    }
    metrics = {
        "schema_version": 1,
        "conference_endpoints": {
            arm: {
                "V0": None,
                "V1": None,
                "V2": None,
                "V3": None,
                "V": None,
                "H": None,
                "HV": None,
                "H_minus_V": None,
                "HV_minus_V": None,
                "calibration": None,
                "risk_coverage": None,
                "class_balance": None,
                "status": "VISIBLE_PREFIX_NOT_ADJUDICABLE",
            }
            for arm in ("horizon_new_only", "gsm8k", "horizon_pooled")
        },
        "historical_sensitivity_only": hist,
        "verdicts": verdicts,
        "null_interpretation_prohibited": True,
        "equivalence_claimed": False,
        "conditional_hidden_increment_claimed": False,
    }
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "bootstrap_results.json",
        {
            "conference_HV_minus_V": {
                arm: {
                    "status": "NOT_RUN_NOT_ADJUDICABLE",
                    "replicates_preregistered": 10000,
                    "replicates_run": 0,
                    "ci95": None,
                }
                for arm in ("horizon_new_only", "gsm8k", "horizon_pooled")
            },
            "historical_sensitivity_only": hist,
        },
    )
    write_json(
        out / "equivalence_results.json",
        {
            arm: {
                "status": "NOT_RUN_NOT_ADJUDICABLE",
                "smallest_effect_of_interest": [-0.02, 0.02],
                "clustered_tost_alpha": 0.05,
                "ci90": None,
                "equivalent": False,
            }
            for arm in ("horizon_new_only", "gsm8k", "horizon_pooled")
        },
    )
    write_json(
        out / "influence_analysis.json",
        {
            "conference_endpoint": {
                "status": "NOT_RUN_NOT_ADJUDICABLE",
                "leave_one_task_family_out": None,
                "leave_one_task_out": None,
                "per_source": None,
                "per_difficulty": None,
            },
            "historical_sensitivity_only": hist,
            "leakage_task_exclusion_influence": {
                "horizon_new_only_task_fraction_removed": leakage[
                    "horizon_new_only"
                ]["leaking_task_fraction"],
                "horizon_pooled_task_fraction_removed": leakage["horizon_pooled"][
                    "leaking_task_fraction"
                ],
            },
        },
    )
    make_figures(out, leakage, hist, trunc)
    inventory_created = dt.datetime.fromisoformat(
        json.loads((out / "inventory_report.json").read_text(encoding="utf-8"))[
            "inventory_timestamp_utc"
        ]
    )
    overall_elapsed = (
        dt.datetime.now(dt.UTC) - inventory_created.astimezone(dt.UTC)
    ).total_seconds()
    (out / "FINAL_REPORT.md").write_text(
        report_text(out, leakage, trunc, hist, overall_elapsed), encoding="utf-8"
    )
    (out / "CONFERENCE_INTEGRATION_SKETCH.md").write_text(
        integration_text(), encoding="utf-8"
    )
    finalizer_elapsed = time.time() - started
    write_json(
        out / "finalization_runtime.json",
        {
            "inventory_to_final_package_seconds": overall_elapsed,
            "finalizer_elapsed_seconds": finalizer_elapsed,
            "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "verdicts": verdicts,
        },
    )
    checksum_package(out)
    print(json.dumps(verdicts, indent=2))
    print(out)


if __name__ == "__main__":
    main()
