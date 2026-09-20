"""Build cross-domain eval matrix from existing cached artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


OUTPUT_JSON = REPORT_DIR / "bg_cross_domain_eval_matrix_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_cross_domain_eval_matrix_2026-05-17.md"


def path(rel: str) -> Path:
    return PROJECT_ROOT / rel


def load_torch_meta(rel: str) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    p = path(rel)
    if not p.exists():
        return False, {}, {"missing": rel}
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        return True, meta, payload
    except Exception as exc:
        return False, {}, {"error": f"{type(exc).__name__}: {exc}", "path": rel}


def count_record_domain(rel: str) -> dict[str, Any]:
    ok, meta, payload = load_torch_meta(rel)
    if not ok:
        return {"feature_status": "MISSING", "feature_path": rel, "n_tournaments": 0, "n_candidates": 0}
    records = payload.get("records", []) or []
    return {
        "feature_status": "READY",
        "feature_path": rel,
        "n_tournaments": len(records),
        "n_candidates": sum(int(row.get("pooled").shape[0]) for row in records if row.get("pooled") is not None),
        "meta": {k: v for k, v in meta.items() if k not in {"records"}},
    }


def count_candidate_feature_domain(rel: str, eval_set_name: str) -> dict[str, Any]:
    ok, meta, payload = load_torch_meta(rel)
    if not ok:
        return {"feature_status": "MISSING", "feature_path": rel, "eval_set_name": eval_set_name, "n_tournaments": 0, "n_candidates": 0}
    eval_rows = payload.get("eval_sets", {}).get(eval_set_name, []) or []
    feature_uids = {str(row.get("candidate_uid")) for row in payload.get("candidate_features", []) or []}
    missing = sorted({str(uid) for row in eval_rows for uid in row.get("candidate_uids", []) if str(uid) not in feature_uids})
    return {
        "feature_status": "READY" if not missing else "FEATURE_MISSING",
        "feature_path": rel,
        "eval_set_name": eval_set_name,
        "n_tournaments": len(eval_rows),
        "n_candidates": sum(len(row.get("candidate_uids", []) or []) for row in eval_rows),
        "missing_feature_uids": missing,
        "meta": {k: v for k, v in meta.items() if k not in {"records"}},
    }


def random_baseline_from_records(rel: str) -> float:
    ok, _, payload = load_torch_meta(rel)
    if not ok:
        return float("nan")
    vals = []
    for row in payload.get("records", []) or []:
        labels = row.get("labels")
        if labels is None:
            continue
        tensor = labels.to(torch.float32) if hasattr(labels, "to") else torch.tensor(labels, dtype=torch.float32)
        vals.append(float(tensor.mean()))
    return sum(vals) / len(vals) if vals else float("nan")


def random_baseline_from_evalset(rel: str, eval_set_name: str) -> float:
    ok, _, payload = load_torch_meta(rel)
    if not ok:
        return float("nan")
    vals = []
    for row in payload.get("eval_sets", {}).get(eval_set_name, []) or []:
        labels = list(row.get("labels", []) or [])
        if labels:
            vals.append(labels.count("correct") / len(labels))
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    hh_rel = "rpe/evaluator/hh_layer_states_200_rltt.pt"
    random_hh_rel = "opi/taps/probes/code_trained_vs_hh_trained_random20_hh_splits_2026-05-17.json"
    eval_sets: list[dict[str, Any]] = []
    if path(hh_rel).exists():
        hh = torch.load(path(hh_rel), map_location="cpu", weights_only=False)
        random_splits = []
        if path(random_hh_rel).exists():
            try:
                prior = json.loads(path(random_hh_rel).read_text(encoding="utf-8"))
                random_splits = [row.get("eval_indices", []) for row in prior.get("summary", {}).get("per_split_best", [])]
            except Exception:
                random_splits = []
        eval_sets.append({
            "domain": "HH_200",
            "domain_type": "hh_pairs",
            "feature_path": hh_rel,
            "feature_status": "READY",
            "n_pairs": len(hh.get("packs", []) or []),
            "random_top1_baseline": 0.5,
            "eval_variants": {"all200_diagnostic": list(range(len(hh.get("packs", []) or []))), "random20_splits": random_splits},
            "label_source": "canonical HH chosen/rejected",
        })
    else:
        eval_sets.append({"domain": "HH_200", "feature_status": "MISSING", "feature_path": hh_rel})

    records_domains = [
        ("CLEAN_GSM8K_EXPANDED", "opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt", "exact-answer verifier"),
        ("CODE_RUNNABLE_DIAGNOSTIC", "opi/taps/probes/code_branch_tap_features_v2_mini_patched_2026-05-16.pt", "unit tests"),
    ]
    for domain, rel, label_source in records_domains:
        row = count_record_domain(rel)
        row.update({
            "domain": domain,
            "domain_type": "records_pt",
            "random_top1_baseline": random_baseline_from_records(rel),
            "label_source": label_source,
        })
        eval_sets.append(row)

    candidate_domains = [
        ("CODE_STRICT_CLEAN_OLD6", "opi/taps/probes/code_strict_clean_transfer_features_2026-05-17.pt", "strict_clean_primary"),
        ("CODE_STRICT_CLEAN_NEW10", "opi/taps/probes/code_expanded_strict_clean_features_2026-05-17.pt", "NEW10_primary"),
        ("CODE_STRICT_CLEAN_ALL16", "opi/taps/probes/code_expanded_strict_clean_features_2026-05-17.pt", "ALL16_primary"),
    ]
    for domain, rel, eval_name in candidate_domains:
        row = count_candidate_feature_domain(rel, eval_name)
        row.update({
            "domain": domain,
            "domain_type": "candidate_features_eval_set",
            "random_top1_baseline": random_baseline_from_evalset(rel, eval_name),
            "label_source": "unit tests",
        })
        eval_sets.append(row)

    ready = [row for row in eval_sets if row.get("feature_status") == "READY"]
    missing = [row for row in eval_sets if row.get("feature_status") != "READY"]
    verdict = "READY" if len(ready) == len(eval_sets) else ("PARTIAL_FEATURE_MISSING" if ready else "BLOCKED")
    payload = {
        "bg_cross_domain_matrix_verdict": verdict,
        "summary": {
            "BG_CROSS_DOMAIN_MATRIX_VERDICT": verdict,
            "eval_set_count": len(eval_sets),
            "ready_eval_set_count": len(ready),
            "feature_missing_eval_sets": [row["domain"] for row in missing],
        },
        "eval_sets": eval_sets,
        "feature_recapture_plan": [
            {"domain": row["domain"], "feature_path": row.get("feature_path"), "missing": row.get("missing_feature_uids", [])}
            for row in missing
        ],
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    lines = ["# BG Cross-Domain Eval Matrix", "", f"BG_CROSS_DOMAIN_MATRIX_VERDICT = {verdict}", ""]
    lines.append("| domain | status | n_tournaments/pairs | n_candidates | random_top1_baseline | feature |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in eval_sets:
        n = row.get("n_tournaments", row.get("n_pairs", 0))
        lines.append(
            f"| `{row['domain']}` | `{row.get('feature_status')}` | {n} | {row.get('n_candidates', '')} | "
            f"{row.get('random_top1_baseline', '')} | `{row.get('feature_path')}` |"
        )
    if missing:
        lines.extend(["", "## Feature Recapture Plan", ""])
        for row in payload["feature_recapture_plan"]:
            lines.append(f"- `{row['domain']}` missing `{len(row.get('missing', []))}` candidates from `{row.get('feature_path')}`")
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"BG_CROSS_DOMAIN_MATRIX_VERDICT = {verdict}")
    print(f"ready_eval_sets = {len(ready)}/{len(eval_sets)}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
