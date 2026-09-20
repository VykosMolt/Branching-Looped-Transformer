"""Merge cached feature stores for mixed-domain tiny tap training/eval."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402
from build_mixed_tap_domain_splits import (  # noqa: E402
    GSM8K_FEATURES_PT,
    CODE_RUNNABLE_FEATURES_PT,
)


SPLITS_JSON = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "mixed_tap_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "mixed_tap_features_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(output_path(path), map_location="cpu", weights_only=False)


def rows_required_uids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(uid) for row in rows for uid in row.get("candidate_uids", [])}


def pairs_required_uids(pairs: list[dict[str, Any]]) -> set[str]:
    return {str(pair["preferred_uid"]) for pair in pairs} | {str(pair["rejected_uid"]) for pair in pairs}


def merge_candidate_features(splits: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    by_uid: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for domain in splits.get("domains", {}).values():
        if domain.get("kind") != "candidate_features":
            continue
        source = str(domain["feature_path"])
        if source in sources:
            continue
        sources.append(source)
        payload = load_pt(source)
        for row in payload.get("candidate_features", []) or []:
            uid = str(row["candidate_uid"])
            if uid not in by_uid:
                by_uid[uid] = {
                    "candidate_uid": uid,
                    "task_id": str(row.get("task_id", "")),
                    "candidate_metadata": row.get("candidate_metadata", {}),
                    "feature_text": row.get("feature_text", ""),
                    "pooled": row["pooled"].detach().cpu().to(torch.float32),
                    "source_feature_artifact": source,
                }
    return list(by_uid.values()), by_uid, sources


def collect_required(splits: dict[str, Any]) -> dict[str, Any]:
    required: dict[str, set[str]] = {}
    for name, domain in splits.get("domains", {}).items():
        if domain.get("kind") != "candidate_features":
            continue
        need = set()
        need |= pairs_required_uids(domain.get("train_pairs", []) or [])
        need |= pairs_required_uids(domain.get("val_pairs", []) or [])
        for eval_spec in domain.get("eval_sets", {}).values():
            need |= rows_required_uids(eval_spec.get("rows", []) or [])
        if name == "SCIENCE":
            for pairs in domain.get("train_pairs_by_subdomain", {}).values():
                need |= pairs_required_uids(pairs)
            for pairs in domain.get("val_pairs_by_subdomain", {}).values():
                need |= pairs_required_uids(pairs)
        required[name] = need
    return required


def load_record_domains(splits: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    if "GSM8K" in splits.get("domains", {}) and output_path(GSM8K_FEATURES_PT).exists():
        records["GSM8K"] = load_pt(GSM8K_FEATURES_PT).get("records", []) or []
    if "CODE_RUNNABLE" in splits.get("domains", {}) and output_path(CODE_RUNNABLE_FEATURES_PT).exists():
        records["CODE_RUNNABLE"] = load_pt(CODE_RUNNABLE_FEATURES_PT).get("records", []) or []
    return records


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mixed-Domain Feature Merge (2026-05-17)",
        "",
        f"MIXED_TAP_FEATURE_VERDICT = {payload['meta']['mixed_tap_feature_verdict']}",
        f"GSM8K_EVAL_STATUS = {payload['meta']['gsm8k_eval_status']}",
        f"candidate feature rows = {payload['meta']['candidate_feature_rows']}",
        "",
        "## Coverage",
    ]
    for domain, cov in payload["coverage"].items():
        frac = cov.get("coverage_fraction")
        frac_text = "NA" if frac is None or (isinstance(frac, float) and math.isnan(frac)) else f"{float(frac):.3f}"
        lines.append(f"- {domain}: {cov['available']}/{cov['required']} ({frac_text}), missing={len(cov.get('missing', []))}")
    lines.extend(["", "## Feature Sources"])
    for source in payload["meta"]["candidate_feature_sources"]:
        lines.append(f"- `{source}`")
    if payload["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in payload["blockers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    splits = load_json(args.splits)
    features, by_uid, sources = merge_candidate_features(splits)
    required = collect_required(splits)
    coverage: dict[str, Any] = {}
    blockers: list[str] = []
    for domain, need in required.items():
        missing = sorted(uid for uid in need if uid not in by_uid)
        if missing:
            blockers.append(f"{domain} missing {len(missing)} cached feature candidates")
        coverage[domain] = {
            "required": len(need),
            "available": len(need) - len(missing),
            "missing": missing,
            "coverage_fraction": (len(need) - len(missing)) / len(need) if need else 1.0,
        }

    record_domains = load_record_domains(splits)
    if "GSM8K" in splits.get("domains", {}) and not record_domains.get("GSM8K"):
        blockers.append("GSM8K records requested but unavailable")
    if "CODE_RUNNABLE" in splits.get("domains", {}) and not record_domains.get("CODE_RUNNABLE"):
        blockers.append("CODE_RUNNABLE records requested but unavailable")

    verdict = "BLOCKED" if blockers else "READY"
    out_payload = {
        "meta": {
            "mixed_tap_feature_verdict": verdict,
            "splits_json": repo_path(output_path(args.splits)),
            "candidate_feature_rows": len(features),
            "candidate_feature_sources": sources,
            "record_domains": {key: len(val) for key, val in record_domains.items()},
            "gsm8k_eval_status": splits.get("gsm8k_eval_status", "NOT_FOUND"),
            "device_requested": args.device,
            "recaptured_candidates": 0,
        },
        "candidate_features": features,
        "record_domains": record_domains,
        "hh_path": splits.get("domains", {}).get("HH", {}).get("feature_path"),
        "domains": splits.get("domains", {}),
        "mixed_families": splits.get("mixed_families", {}),
        "coverage": coverage,
        "blockers": blockers,
    }
    out = output_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out)

    report_payload = {
        "meta": out_payload["meta"],
        "coverage": coverage,
        "blockers": blockers,
    }
    write_json(out.with_suffix(".json"), report_payload)
    write_markdown(output_path(args.output_md), report_payload)
    print(f"MIXED_TAP_FEATURE_VERDICT = {verdict}")
    print(f"candidate_feature_rows = {len(features)}")
    if blockers:
        for blocker in blockers:
            print(f"BLOCKER: {blocker}")
    print(f"wrote {repo_path(out)}")


if __name__ == "__main__":
    main()
