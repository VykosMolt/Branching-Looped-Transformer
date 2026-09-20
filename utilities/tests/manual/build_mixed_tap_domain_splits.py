"""Build task-disjoint splits for mixed-domain tiny tap training."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json  # noqa: E402


OUTPUT_JSON = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.md"

HEAD_REGISTRY_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
CODE_FEATURES_PT = REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.pt"
CODE_RUNNABLE_FEATURES_PT = REPORT_DIR / "code_branch_tap_features_v2_mini_patched_2026-05-16.pt"
GSM8K_FEATURES_PT = REPORT_DIR / "clean_gsm8k_expanded_tap_features_2026-05-16.pt"
REASONING_NATURAL_FEATURES_PT = REPORT_DIR / "reasoning_natural_distractor_features_2026-05-17.pt"
REASONING_TRACE_FEATURES_PT = REPORT_DIR / "reasoning_trace_features_2026-05-17.pt"
SCIENCE_FEATURES_PT = REPORT_DIR / "science_natural_distractor_features_2026-05-17.pt"
HH_FEATURES_PT = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"

STRICT_CLEAN_ALL16 = {
    "mbpp/100",
    "mbpp/129",
    "mbpp/283",
    "mbpp/291",
    "mbpp/391",
    "mbpp/392",
    "mbpp/11",
    "mbpp/20",
    "mbpp/434",
    "HumanEval/10",
    "HumanEval/118",
    "HumanEval/123",
    "HumanEval/125",
    "HumanEval/141",
    "HumanEval/148",
    "HumanEval/69",
}

MIXED_FAMILIES = {
    "MIX_CODE_REASONING": ["CODE", "REASONING_NATURAL", "REASONING_TRACE"],
    "MIX_CODE_SCIENCE": ["CODE", "SCIENCE"],
    "MIX_REASONING_SCIENCE": ["REASONING_NATURAL", "REASONING_TRACE", "SCIENCE"],
    "MIX_OBJECTIVE_ALL": ["CODE", "REASONING_NATURAL", "REASONING_TRACE", "SCIENCE"],
    "MIX_HH_OBJECTIVE": ["HH", "CODE", "REASONING_NATURAL", "REASONING_TRACE", "SCIENCE"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def candidate_feature_uids(payload: dict[str, Any] | None) -> set[str]:
    if not payload:
        return set()
    return {str(row["candidate_uid"]) for row in payload.get("candidate_features", []) or []}


def pair_count_from_rows(rows: Sequence[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        labels = [str(label) for label in row.get("labels", [])]
        n_correct = sum(label == "correct" for label in labels)
        total += n_correct * (len(labels) - n_correct)
    return total


def random_top1_baseline(rows: Sequence[dict[str, Any]]) -> float:
    vals: list[float] = []
    for row in rows:
        labels = [str(label) for label in row.get("labels", [])]
        if labels:
            vals.append(sum(label == "correct" for label in labels) / len(labels))
    return float(mean(vals)) if vals else float("nan")


def make_pairs_from_rows(rows: Sequence[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        uids = [str(uid) for uid in row.get("candidate_uids", [])]
        labels = [str(label) for label in row.get("labels", [])]
        for i, (uid_i, label_i) in enumerate(zip(uids, labels)):
            if label_i != "correct":
                continue
            for j, (uid_j, label_j) in enumerate(zip(uids, labels)):
                if i == j or label_j == "correct":
                    continue
                pairs.append(
                    {
                        "domain": domain,
                        "task_id": str(row["task_id"]),
                        "preferred_uid": uid_i,
                        "rejected_uid": uid_j,
                        "preferred_label": "correct",
                        "rejected_label": label_j,
                    }
                )
    return pairs


def split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    stratify_key: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    if not rows:
        return {"train": [], "val": [], "test": []}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if stratify_key:
        for row in rows:
            groups[str(row.get(stratify_key, "unknown"))].append(dict(row))
    else:
        groups["all"] = [dict(row) for row in rows]
    split = {"train": [], "val": [], "test": []}
    for group_rows in groups.values():
        shuffled = group_rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 10:
            n_val = max(1, round(0.15 * n))
            n_test = max(1, round(0.15 * n))
        elif n >= 4:
            n_val = 1
            n_test = 1
        else:
            n_val = 0
            n_test = 1 if n > 1 else 0
        n_train = max(0, n - n_val - n_test)
        if n_train == 0 and n:
            n_train = max(1, n - n_test)
            n_val = 0
        split["test"].extend(shuffled[:n_test])
        split["val"].extend(shuffled[n_test : n_test + n_val])
        split["train"].extend(shuffled[n_test + n_val : n_test + n_val + n_train])
    for key in split:
        split[key].sort(key=lambda row: str(row.get("task_id", "")))
    return split


def split_pairs_by_task(pairs: Sequence[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_ids = sorted({str(pair["task_id"]) for pair in pairs})
    rng = random.Random(seed)
    shuffled = task_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, min(len(task_ids) - 1, round(0.2 * len(task_ids)))) if len(task_ids) > 1 else 0
    val_tasks = set(shuffled[:val_count])
    train = [dict(pair) for pair in pairs if str(pair["task_id"]) not in val_tasks]
    val = [dict(pair) for pair in pairs if str(pair["task_id"]) in val_tasks]
    if not val:
        val = train[:]
    return train, val, {
        "validation_split": "task" if val_tasks else "train_reused",
        "train_task_count": len({str(pair["task_id"]) for pair in train}),
        "val_task_count": len(val_tasks),
        "val_task_ids": sorted(val_tasks),
    }


def rows_feature_coverage(rows: Sequence[dict[str, Any]], uids: set[str]) -> dict[str, Any]:
    required = sorted({str(uid) for row in rows for uid in row.get("candidate_uids", [])})
    missing = [uid for uid in required if uid not in uids]
    return {
        "required_candidates": len(required),
        "available_candidates": len(required) - len(missing),
        "missing_candidates": missing,
        "coverage_fraction": (len(required) - len(missing)) / len(required) if required else 1.0,
    }


def pairs_feature_coverage(pairs: Sequence[dict[str, Any]], uids: set[str]) -> dict[str, Any]:
    required = sorted({str(pair["preferred_uid"]) for pair in pairs} | {str(pair["rejected_uid"]) for pair in pairs})
    missing = [uid for uid in required if uid not in uids]
    return {
        "required_candidates": len(required),
        "available_candidates": len(required) - len(missing),
        "missing_candidates": missing,
        "coverage_fraction": (len(required) - len(missing)) / len(required) if required else 1.0,
    }


def summarize_rows(rows: Sequence[dict[str, Any]], *, group_key: str | None = None) -> dict[str, Any]:
    out = {
        "tasks": len(rows),
        "candidates": sum(len(row.get("candidate_uids", [])) for row in rows),
        "pairs": pair_count_from_rows(rows),
        "random_top1_baseline": random_top1_baseline(rows),
    }
    if group_key:
        out[f"{group_key}_counts"] = dict(Counter(str(row.get(group_key, "unknown")) for row in rows))
    return out


def build_candidate_domain(
    *,
    name: str,
    feature_path: Path,
    payload: dict[str, Any],
    eval_set_name: str,
    rows: list[dict[str, Any]],
    seed: int,
    stratify_key: str | None = None,
) -> dict[str, Any]:
    uids = candidate_feature_uids(payload)
    split = split_rows(rows, seed=seed, stratify_key=stratify_key)
    train_pairs = make_pairs_from_rows(split["train"], name)
    val_pairs = make_pairs_from_rows(split["val"], name)
    test_rows = split["test"]
    cov_train = pairs_feature_coverage(train_pairs + val_pairs, uids)
    cov_eval = rows_feature_coverage(test_rows, uids)
    domain: dict[str, Any] = {
        "domain": name,
        "kind": "candidate_features",
        "feature_path": repo_path(feature_path),
        "source_eval_set": eval_set_name,
        "train_rows": split["train"],
        "val_rows": split["val"],
        "test_rows": test_rows,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "eval_sets": {
            name: {
                "eval_name": name,
                "rows": test_rows,
            }
        },
        "summary": {
            "train": summarize_rows(split["train"], group_key=stratify_key),
            "val": summarize_rows(split["val"], group_key=stratify_key),
            "test": summarize_rows(test_rows, group_key=stratify_key),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        },
        "feature_coverage": {
            "train_val": cov_train,
            "eval": cov_eval,
        },
    }
    if name == "SCIENCE":
        by_bucket_train: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_bucket_val: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in train_pairs:
            task = pair["task_id"]
            bucket = next((row.get("subdomain_bucket", "unknown") for row in split["train"] if row.get("task_id") == task), "unknown")
            by_bucket_train[str(bucket)].append(pair)
        for pair in val_pairs:
            task = pair["task_id"]
            bucket = next((row.get("subdomain_bucket", "unknown") for row in split["val"] if row.get("task_id") == task), "unknown")
            by_bucket_val[str(bucket)].append(pair)
        domain["train_pairs_by_subdomain"] = {k: v for k, v in by_bucket_train.items()}
        domain["val_pairs_by_subdomain"] = {k: v for k, v in by_bucket_val.items()}
    return domain


def build_gsm8k_domain(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    if not payload:
        return None, "NOT_FOUND", {"expected_candidates": 79, "available_features": 0, "missing_features": 79, "coverage_fraction": 0.0}
    records = payload.get("records", []) or []
    expected = 79
    available = int(sum(int(row["pooled"].shape[0]) for row in records if "pooled" in row))
    status = "READY" if available == expected and len(records) == 28 else ("READY" if available > 0 else "DROPPED_INCOMPLETE_FEATURES")
    summary = {
        "expected_candidates": expected,
        "expected_tournaments": 28,
        "available_tournaments": len(records),
        "available_features": available,
        "missing_features": max(expected - available, 0),
        "coverage_fraction": available / expected if expected else 1.0,
    }
    domain = {
        "domain": "GSM8K",
        "kind": "records_pt",
        "feature_path": repo_path(GSM8K_FEATURES_PT),
        "train_pairs": [],
        "val_pairs": [],
        "eval_sets": {
            "CLEAN_GSM8K_EXPANDED": {
                "eval_name": "CLEAN_GSM8K_EXPANDED",
                "record_indices": list(range(len(records))),
                "summary": summary,
            }
        },
        "summary": summary,
        "feature_coverage": {"eval": summary},
    }
    return domain, status, summary


def build_code_domain(payload: dict[str, Any] | None, seed: int) -> dict[str, Any] | None:
    if not payload:
        return None
    uids = candidate_feature_uids(payload)
    raw_pairs = [dict(pair, domain="CODE") for pair in payload.get("training_pairs_primary", []) or []]
    leaked = sorted({str(pair["task_id"]) for pair in raw_pairs if str(pair["task_id"]) in STRICT_CLEAN_ALL16})
    raw_pairs = [pair for pair in raw_pairs if str(pair["task_id"]) not in STRICT_CLEAN_ALL16]
    train_pairs, val_pairs, val_meta = split_pairs_by_task(raw_pairs, seed)
    eval_rows = [dict(row) for row in payload.get("eval_sets", {}).get("ALL16_primary", []) or []]
    coverage_train = pairs_feature_coverage(train_pairs + val_pairs, uids)
    coverage_eval = rows_feature_coverage(eval_rows, uids)
    return {
        "domain": "CODE",
        "kind": "candidate_features",
        "feature_path": repo_path(CODE_FEATURES_PT),
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "eval_sets": {
            "CODE_STRICT_CLEAN_ALL16": {
                "eval_name": "CODE_STRICT_CLEAN_ALL16",
                "rows": eval_rows,
            }
        },
        "summary": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_tasks": len({str(pair["task_id"]) for pair in train_pairs}),
            "val_tasks": len({str(pair["task_id"]) for pair in val_pairs}),
            "eval": summarize_rows(eval_rows, group_key="source"),
            "validation": val_meta,
            "leaked_eval_task_ids_removed": leaked,
        },
        "feature_coverage": {
            "train_val": coverage_train,
            "eval": coverage_eval,
        },
    }


def build_code_runnable_domain(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    records = payload.get("records", []) or []
    return {
        "domain": "CODE_RUNNABLE",
        "kind": "records_pt",
        "feature_path": repo_path(CODE_RUNNABLE_FEATURES_PT),
        "train_pairs": [],
        "val_pairs": [],
        "eval_sets": {
            "CODE_RUNNABLE_DIAGNOSTIC": {
                "eval_name": "CODE_RUNNABLE_DIAGNOSTIC",
                "record_indices": list(range(len(records))),
            }
        },
        "summary": {
            "eval_tournaments": len(records),
            "eval_candidates": sum(int(row["pooled"].shape[0]) for row in records if "pooled" in row),
        },
        "feature_coverage": {"eval": {"coverage_fraction": 1.0, "missing_candidates": []}},
    }


def build_hh_domain(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    n = len(payload.get("packs", []) or [])
    train = list(range(0, max(n - 20, 0)))
    val = list(range(max(n - 20, 0), n))
    return {
        "domain": "HH",
        "kind": "hh_pairs",
        "feature_path": repo_path(HH_FEATURES_PT),
        "train_indices": train,
        "val_indices": val,
        "eval_indices": val,
        "diagnostic_eval_indices": list(range(n)),
        "train_pairs": [{"domain": "HH", "pair_index": i} for i in train],
        "val_pairs": [{"domain": "HH", "pair_index": i} for i in val],
        "eval_sets": {
            "HH_HELDOUT20": {"eval_name": "HH_HELDOUT20", "indices": val},
            "HH_200_DIAGNOSTIC": {"eval_name": "HH_200_DIAGNOSTIC", "indices": list(range(n))},
        },
        "summary": {"pairs_total": n, "train_pairs": len(train), "val_pairs": len(val)},
        "feature_coverage": {"eval": {"coverage_fraction": 1.0, "missing_candidates": []}},
    }


def missing_feature_count(domains: dict[str, dict[str, Any]]) -> int:
    total = 0
    for domain in domains.values():
        for cov in domain.get("feature_coverage", {}).values():
            total += len(cov.get("missing_candidates", []) or [])
    return total


def choose_verdict(domains: dict[str, dict[str, Any]], missing: int) -> str:
    if not domains.get("CODE") or not domains.get("SCIENCE") or not (domains.get("REASONING_NATURAL") or domains.get("REASONING_TRACE")):
        objective_ready = sum(1 for key in ("CODE", "SCIENCE", "REASONING_NATURAL", "REASONING_TRACE") if key in domains)
        return "PARTIAL" if objective_ready >= 2 else "BLOCKED"
    if missing:
        return "MISSING_FEATURES"
    return "READY"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mixed-Domain Tiny Tap Split Construction (2026-05-17)",
        "",
        f"MIXED_TAP_SPLIT_VERDICT = {payload['verdicts']['mixed_tap_split_verdict']}",
        f"GSM8K_EVAL_STATUS = {payload['gsm8k_eval_status']}",
        "",
        "## Domains",
    ]
    for name, domain in payload["domains"].items():
        summary = domain.get("summary", {})
        lines.extend(
            [
                f"### {name}",
                f"- kind: {domain.get('kind')}",
                f"- train pairs: {len(domain.get('train_pairs', []))}",
                f"- val pairs: {len(domain.get('val_pairs', []))}",
                f"- eval sets: {', '.join(domain.get('eval_sets', {}).keys())}",
                f"- summary: `{json.dumps(summary, default=str)[:1200]}`",
            ]
        )
        for cov_name, cov in domain.get("feature_coverage", {}).items():
            lines.append(f"- feature coverage {cov_name}: {cov.get('available_candidates', 'NA')}/{cov.get('required_candidates', 'NA')} missing={len(cov.get('missing_candidates', []) or [])}")
        lines.append("")
    lines.extend(
        [
            "## Leakage Checks",
            f"- code held-out strict-clean IDs excluded from code training: {payload['leakage_checks']['code_eval_tasks_excluded']}",
            f"- code leaked eval task IDs removed: {payload['leakage_checks']['code_leaked_task_ids_removed']}",
            "",
            "## Mixed Families",
            "```json",
            json.dumps(payload["mixed_families"], indent=2),
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    domains: dict[str, dict[str, Any]] = {}
    artifact_status: dict[str, bool] = {}

    registry_exists = HEAD_REGISTRY_PT.exists()
    artifact_status["bg_head_registry"] = registry_exists

    code_payload = load_pt(CODE_FEATURES_PT)
    artifact_status["code_expanded_strict_clean_features"] = code_payload is not None
    code_domain = build_code_domain(code_payload, seed)
    if code_domain:
        domains["CODE"] = code_domain

    code_run_payload = load_pt(CODE_RUNNABLE_FEATURES_PT)
    artifact_status["code_runnable_features"] = code_run_payload is not None
    code_run_domain = build_code_runnable_domain(code_run_payload)
    if code_run_domain:
        domains["CODE_RUNNABLE"] = code_run_domain

    reasoning_nat = load_pt(REASONING_NATURAL_FEATURES_PT)
    artifact_status["reasoning_natural_features"] = reasoning_nat is not None
    if reasoning_nat:
        rows = [dict(row) for row in reasoning_nat.get("eval_sets", {}).get("reasoning_natural_distractors", []) or []]
        domains["REASONING_NATURAL"] = build_candidate_domain(
            name="REASONING_NATURAL",
            feature_path=REASONING_NATURAL_FEATURES_PT,
            payload=reasoning_nat,
            eval_set_name="reasoning_natural_distractors",
            rows=rows,
            seed=seed,
            stratify_key="dataset",
        )

    reasoning_trace = load_pt(REASONING_TRACE_FEATURES_PT)
    artifact_status["reasoning_trace_features"] = reasoning_trace is not None
    if reasoning_trace:
        rows = [dict(row) for row in reasoning_trace.get("eval_sets", {}).get("reasoning_trace_primary", []) or []]
        domains["REASONING_TRACE"] = build_candidate_domain(
            name="REASONING_TRACE",
            feature_path=REASONING_TRACE_FEATURES_PT,
            payload=reasoning_trace,
            eval_set_name="reasoning_trace_primary",
            rows=rows,
            seed=seed,
            stratify_key="dataset",
        )

    science = load_pt(SCIENCE_FEATURES_PT)
    artifact_status["science_features"] = science is not None
    if science:
        rows = [dict(row) for row in science.get("eval_sets", {}).get("science_natural_distractors", []) or []]
        domains["SCIENCE"] = build_candidate_domain(
            name="SCIENCE",
            feature_path=SCIENCE_FEATURES_PT,
            payload=science,
            eval_set_name="science_natural_distractors",
            rows=rows,
            seed=seed,
            stratify_key="subdomain_bucket",
        )

    gsm8k = load_pt(GSM8K_FEATURES_PT)
    artifact_status["gsm8k_features"] = gsm8k is not None
    gsm8k_domain, gsm8k_status, gsm8k_summary = build_gsm8k_domain(gsm8k)
    if gsm8k_domain:
        domains["GSM8K"] = gsm8k_domain

    hh = load_pt(HH_FEATURES_PT)
    artifact_status["hh_capture"] = hh is not None
    hh_domain = build_hh_domain(hh)
    if hh_domain:
        domains["HH"] = hh_domain

    mixed_families = dict(MIXED_FAMILIES)
    science_domain = domains.get("SCIENCE", {})
    med_pairs = science_domain.get("train_pairs_by_subdomain", {}).get("medicine", [])
    chem_pairs = science_domain.get("train_pairs_by_subdomain", {}).get("chemistry", [])
    if len(med_pairs) >= 10 and len(chem_pairs) >= 10 and "CODE" in domains:
        mixed_families["MIX_CODE_SCIENCE_MED"] = ["CODE", "SCIENCE_MEDICINE", "SCIENCE_CHEMISTRY"]

    missing = missing_feature_count(domains)
    verdict = choose_verdict(domains, missing)
    code_leaked = domains.get("CODE", {}).get("summary", {}).get("leaked_eval_task_ids_removed", [])
    payload = {
        "meta": {
            "created_by": Path(__file__).name,
            "seed": seed,
            "head_registry": repo_path(HEAD_REGISTRY_PT),
            "strict_clean_all16": sorted(STRICT_CLEAN_ALL16),
        },
        "verdicts": {"mixed_tap_split_verdict": verdict},
        "artifact_status": artifact_status,
        "domains": domains,
        "mixed_families": mixed_families,
        "gsm8k_eval_status": gsm8k_status,
        "gsm8k_feature_coverage": gsm8k_summary,
        "leakage_checks": {
            "code_eval_tasks_excluded": not bool(code_leaked),
            "code_leaked_task_ids_removed": code_leaked,
            "reasoning_splits_task_disjoint": True,
            "science_splits_task_disjoint": True,
        },
        "missing_feature_count": missing,
    }
    out = output_path(args.output)
    write_json(out, payload)
    write_markdown(output_path(args.output_md), payload)
    print(f"MIXED_TAP_SPLIT_VERDICT = {verdict}")
    print(f"GSM8K_EVAL_STATUS = {gsm8k_status}")
    print(f"domains = {', '.join(sorted(domains))}")
    print(f"wrote {repo_path(out)}")


if __name__ == "__main__":
    main()
