"""HH-RLHF same-content comparison for old BG taps vs new two taps.

Samples Anthropic HH-RLHF chosen/rejected test pairs with a non-42 seed,
captures local Ouro-RLTT layer-native features once, and evaluates old frozen
BG taps against strict MIX_CODE_REASONING / MIX_OBJECTIVE_ALL two-tap
candidates on the exact same pairs.

This does not train Ouro, modify checkpoints/tokenizers/tap registries, run
wrapper/local-agent or Hunter-Seeker code, apply steering, or change routing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
OUT_ROOT = PROBE_ROOT / "bg_two_tap_hh_rlhf_comparison_v1_2026-05-30"
DATASET_JSON = OUT_ROOT / "hh_rlhf_pairs.json"
DATASET_MD = OUT_ROOT / "hh_rlhf_pairs.md"
FEATURES_PT = OUT_ROOT / "hh_rlhf_features.pt"
ROWS_CSV = OUT_ROOT / "hh_rlhf_comparison_rows.csv"
SUMMARY_JSON = OUT_ROOT / "hh_rlhf_comparison_summary.json"
SUMMARY_MD = OUT_ROOT / "hh_rlhf_comparison_summary.md"
ARTIFACT_PT = OUT_ROOT / "two_tap_hh_rlhf_comparison_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_two_tap_hh_rlhf_comparison_v1.md"

MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
LAYER_CONFIGS = ("24_L4", "36_L4", "47_L4")
FEATURE_CONFIGS = (
    "24_L1",
    "24_L4",
    "24_mean",
    "36_L1",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "47_concat_L1_L4",
    "47_concat_all_loops",
)

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "shared/hf_cache/hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "shared/hf_cache/datasets"))

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import capture_pooled_taps, config_vector  # noqa: E402
from run_bg_two_tap_fresh_dataset_comparison_v1 import (  # noqa: E402
    accuracy_from_scores,
    finite_mean,
    json_default,
    load_new_two_taps,
    rel,
    safe_float,
    score_vector,
    strip_scores,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--target-pairs", type=int, default=512)
    parser.add_argument("--skip-first", type=int, default=200, help="Avoid likely prior HH_200 capture slice.")
    parser.add_argument("--max-feature-chars", type=int, default=6000)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--force-capture", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Anthropic/hh-rlhf", split="test", cache_dir=os.environ["HF_DATASETS_CACHE"])
    rng = random.Random(int(args.seed))
    indices = [idx for idx in range(len(ds)) if idx >= int(args.skip_first)]
    rng.shuffle(indices)
    pairs: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for idx in indices:
        row = ds[idx]
        chosen = clean_text(row.get("chosen"))
        rejected = clean_text(row.get("rejected"))
        if not chosen or not rejected:
            skipped["empty"] += 1
            continue
        if len(chosen) > int(args.max_feature_chars) or len(rejected) > int(args.max_feature_chars):
            skipped["too_long"] += 1
            continue
        pair_id = f"hh_rlhf_test/{idx}"
        chosen_uid = f"{pair_id}/chosen"
        rejected_uid = f"{pair_id}/rejected"
        pairs.append(
            {
                "pair_id": pair_id,
                "dataset_index": idx,
                "source_dataset": "Anthropic/hh-rlhf",
                "source_split": "test",
                "source_status": "hh_rlhf_test_fresh_seed_non42",
                "chosen_uid": chosen_uid,
                "rejected_uid": rejected_uid,
                "chosen_chars": len(chosen),
                "rejected_chars": len(rejected),
            }
        )
        candidates.extend(
            [
                {
                    "candidate_uid": chosen_uid,
                    "pair_id": pair_id,
                    "dataset_index": idx,
                    "label": "chosen",
                    "is_preferred": True,
                    "feature_text": chosen,
                },
                {
                    "candidate_uid": rejected_uid,
                    "pair_id": pair_id,
                    "dataset_index": idx,
                    "label": "rejected",
                    "is_preferred": False,
                    "feature_text": rejected,
                },
            ]
        )
        if len(pairs) >= int(args.target_pairs):
            break
    verdict = "READY" if len(pairs) >= min(int(args.target_pairs), 200) else "PARTIAL" if pairs else "BLOCKED"
    payload = {
        "BG_TWO_TAP_HH_RLHF_DATASET_VERDICT": verdict,
        "summary": {
            "seed": int(args.seed),
            "target_pairs": int(args.target_pairs),
            "skip_first": int(args.skip_first),
            "pair_count": len(pairs),
            "candidate_count": len(candidates),
            "dataset_total_test_rows": len(ds),
            "source_status": "hh_rlhf_test_fresh_seed_non42",
            "skipped": dict(skipped),
            "reuse_policy": "Prior local HH capture covered only 200 pairs; this samples the HH-RLHF test split with a non-42 seed and skips the likely first-200 prior slice.",
        },
        "pairs": pairs,
        "candidates": candidates,
    }
    write_json(DATASET_JSON, payload)
    lines = [
        "# HH-RLHF Pair Dataset",
        "",
        f"BG_TWO_TAP_HH_RLHF_DATASET_VERDICT = {verdict}",
        "",
        f"- seed: `{payload['summary']['seed']}`",
        f"- pair_count: `{len(pairs)}`",
        f"- candidate_count: `{len(candidates)}`",
        f"- skip_first: `{payload['summary']['skip_first']}`",
        f"- skipped: `{payload['summary']['skipped']}`",
        f"- reuse_policy: `{payload['summary']['reuse_policy']}`",
        "",
    ]
    write_md(DATASET_MD, lines)
    return payload


def capture_features(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    if FEATURES_PT.exists() and not args.force_capture:
        return torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    candidates = list(payload.get("candidates") or [])
    texts = [row["feature_text"] for row in candidates]
    device = torch.device(args.device)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    pooled = capture_pooled_taps(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=int(args.max_length),
        device=device,
        report_every=int(args.report_every),
    ).cpu()
    elapsed = time.time() - start
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rows = []
    for idx, row in enumerate(candidates):
        rows.append({**row, "pooled": pooled[idx].to(torch.float32).contiguous()})
    out = {
        "meta": {
            "BG_TWO_TAP_HH_RLHF_FEATURE_CAPTURE_VERDICT": "READY",
            "elapsed_seconds": elapsed,
            "candidate_count": len(rows),
            "pair_count": len(payload.get("pairs") or []),
            "tap_layers": [24, 36, 47],
            "feature_configs": list(FEATURE_CONFIGS),
            "model_path": rel(args.model_path),
        },
        "candidate_features": rows,
        "pairs": payload.get("pairs") or [],
        "dataset_summary": payload.get("summary") or {},
    }
    torch.save(out, FEATURES_PT)
    return out


def build_pair_diffs(feature_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_uid = {str(row.get("candidate_uid")): row for row in feature_payload.get("candidate_features") or []}
    out = []
    for pair in feature_payload.get("pairs") or []:
        chosen = by_uid.get(str(pair.get("chosen_uid")))
        rejected = by_uid.get(str(pair.get("rejected_uid")))
        if not chosen or not rejected:
            continue
        feats = {}
        for config in FEATURE_CONFIGS:
            try:
                cv = config_vector(chosen["pooled"], config).detach().cpu().to(torch.float32).flatten()
                rv = config_vector(rejected["pooled"], config).detach().cpu().to(torch.float32).flatten()
                feats[config] = cv - rv
            except Exception:
                pass
        if feats:
            out.append({**pair, "features": feats})
    return out


def load_old_bg_taps_hh() -> list[dict[str, Any]]:
    from run_bg_two_tap_full_readiness_v1 import source_candidates

    out = []
    for row in source_candidates():
        if row.get("candidate_family") != "source_old_content":
            continue
        if row.get("source_run") not in {"old_registry", "old_mixed_domain"}:
            continue
        if str(row.get("target_config")) not in FEATURE_CONFIGS:
            continue
        weight = tensor_weight(row)
        if isinstance(weight, torch.Tensor):
            out.append({**row, "eval_family": "old_bg_all_available"})
    return out


def evaluate_candidates(candidates: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_config: dict[str, list[torch.Tensor]] = {config: [] for config in FEATURE_CONFIGS}
    for pair in pairs:
        for config in FEATURE_CONFIGS:
            diff = (pair.get("features") or {}).get(config)
            if isinstance(diff, torch.Tensor):
                by_config[config].append(diff)
    for cand in candidates:
        config = str(cand.get("target_config"))
        if config not in by_config or not by_config[config]:
            continue
        weight = tensor_weight(cand)
        if not isinstance(weight, torch.Tensor):
            continue
        arch = str(cand.get("architecture"))
        diffs = torch.stack(by_config[config], dim=0)
        scores = [float(x) for x in score_vector(weight, arch, diffs).tolist()]
        rows.append(
            {
                "candidate_name": cand.get("candidate_name"),
                "candidate_family": cand.get("candidate_family"),
                "eval_family": cand.get("eval_family"),
                "source_run": cand.get("source_run"),
                "source_family": cand.get("source_family"),
                "tap_role": cand.get("tap_role"),
                "recipe": cand.get("recipe"),
                "target_config": config,
                "architecture": arch,
                "pair_count": len(scores),
                "pairwise_accuracy": accuracy_from_scores(scores),
                "mean_margin": finite_mean(scores),
            }
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    old = [r for r in rows if r.get("eval_family") == "old_bg_all_available"]
    old_layer = [r for r in old if r.get("target_config") in LAYER_CONFIGS]
    new = [r for r in rows if r.get("eval_family") == "new_two_tap_primary"]
    best_old = max(old, key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
    best_old_layer = max(old_layer, key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
    best_new = max(new, key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
    old_acc = safe_float(best_old.get("pairwise_accuracy"), float("nan"))
    old_layer_acc = safe_float(best_old_layer.get("pairwise_accuracy"), float("nan"))
    new_acc = safe_float(best_new.get("pairwise_accuracy"), float("nan"))
    if math.isfinite(old_acc) and math.isfinite(new_acc):
        if new_acc + 1e-12 >= old_acc:
            verdict = "TWO_TAP_MATCHES_OR_BEATS_ALL_OLD_BG_ON_HH_RLHF"
        elif math.isfinite(old_layer_acc) and new_acc + 1e-12 >= old_layer_acc:
            verdict = "TWO_TAP_BEATS_LAYER_NATIVE_OLD_BG_BUT_ALL_OLD_BG_BEST_ON_HH_RLHF"
        else:
            verdict = "OLD_BG_BEST_ON_HH_RLHF"
    else:
        verdict = "DATA_LIMITED"
    return {
        "BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT": verdict,
        "pair_count": len(pairs),
        "best_old_bg": best_old.get("candidate_name"),
        "best_old_bg_accuracy": old_acc,
        "best_old_bg_family": best_old.get("candidate_family"),
        "best_old_bg_layer_native": best_old_layer.get("candidate_name"),
        "best_old_bg_layer_native_accuracy": old_layer_acc,
        "best_new_two_tap": best_new.get("candidate_name"),
        "best_new_two_tap_accuracy": new_acc,
        "best_new_two_tap_family": best_new.get("candidate_family"),
        "delta_new_minus_old": new_acc - old_acc if math.isfinite(old_acc) and math.isfinite(new_acc) else float("nan"),
        "old_candidate_count": len(old),
        "old_layer_native_candidate_count": len(old_layer),
        "new_candidate_count": len(new),
    }


def write_reports(summary: dict[str, Any], dataset: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    write_csv(ROWS_CSV, rows)
    write_json(SUMMARY_JSON, {"summary": summary, "dataset_summary": dataset.get("summary"), "rows": rows})
    lines = [
        "# Two-Tap HH-RLHF Comparison v1",
        "",
        f"BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT = {summary['BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT']}",
        "",
        "## Scope",
        "",
        "- Dataset: `Anthropic/hh-rlhf`, test split.",
        "- Prior HH work used a 200-pair local capture; this run uses a non-42 seed and skips the likely first-200 slice.",
        "- Old BG taps and strict two-tap candidates were evaluated on identical chosen/rejected pairs.",
        "- Primary two-tap candidates are restricted to `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` anchors.",
        "- Old BG comparison includes all available old frozen configs from the same pooled states, with layer-native old BG also reported separately.",
        "- No targeted-rehost diagnostics were counted.",
        "",
        "## Result",
        "",
        f"- pair_count: `{summary['pair_count']}`",
        f"- best_old_bg_accuracy: `{summary['best_old_bg_accuracy']}`",
        f"- best_old_bg_layer_native_accuracy: `{summary['best_old_bg_layer_native_accuracy']}`",
        f"- best_new_two_tap_accuracy: `{summary['best_new_two_tap_accuracy']}`",
        f"- delta_new_minus_old: `{summary['delta_new_minus_old']}`",
        f"- best_old_bg: `{summary['best_old_bg']}`",
        f"- best_old_bg_layer_native: `{summary['best_old_bg_layer_native']}`",
        f"- best_new_two_tap: `{summary['best_new_two_tap']}`",
        "",
        "## Dataset",
        "",
        f"- seed: `{dataset['summary']['seed']}`",
        f"- skip_first: `{dataset['summary']['skip_first']}`",
        f"- candidate_count: `{dataset['summary']['candidate_count']}`",
        f"- skipped: `{dataset['summary']['skipped']}`",
        "",
        "## Files",
        "",
        f"- dataset: `{rel(DATASET_JSON)}`",
        f"- features: `{rel(FEATURES_PT)}`",
        f"- rows: `{rel(ROWS_CSV)}`",
        f"- summary: `{rel(SUMMARY_JSON)}`",
        f"- artifact: `{rel(ARTIFACT_PT)}`",
        "",
    ]
    write_md(SUMMARY_MD, lines)
    doc_lines = [
        "# BG Two-Tap HH-RLHF Comparison v1",
        "",
        "This probe compares the old frozen BG taps and the strict layer-native two-tap candidates on fresh Anthropic HH-RLHF chosen/rejected pairs.",
        "",
        f"BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT = {summary['BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT']}",
        "",
        f"- pairs: `{summary['pair_count']}`",
        f"- best old BG accuracy: `{summary['best_old_bg_accuracy']}`",
        f"- best layer-native old BG accuracy: `{summary['best_old_bg_layer_native_accuracy']}`",
        f"- best new two-tap accuracy: `{summary['best_new_two_tap_accuracy']}`",
        f"- delta: `{summary['delta_new_minus_old']}`",
        f"- best old BG: `{summary['best_old_bg']}`",
        f"- best layer-native old BG: `{summary['best_old_bg_layer_native']}`",
        f"- best new two-tap: `{summary['best_new_two_tap']}`",
        "",
        "No Ouro weights, tokenizer files, checkpoints, old tap registries, production routing, wrapper/local-agent code, Hunter-Seeker modules, or steering modules were modified or executed.",
        "",
        "Artifacts:",
        "",
        f"- `{rel(SUMMARY_MD)}`",
        f"- `{rel(FEATURES_PT)}`",
        f"- `{rel(ARTIFACT_PT)}`",
        "",
    ]
    write_md(DOC_MD, doc_lines)


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    if dataset.get("BG_TWO_TAP_HH_RLHF_DATASET_VERDICT") == "BLOCKED":
        raise SystemExit("BG_TWO_TAP_HH_RLHF_DATASET_VERDICT=BLOCKED")
    features = capture_features(args, dataset)
    pairs = build_pair_diffs(features)
    old_taps = load_old_bg_taps_hh()
    new_taps, _bundles = load_new_two_taps(False)
    rows = evaluate_candidates(old_taps + new_taps, pairs)
    summary = summarize(rows, pairs)
    write_reports(summary, dataset, rows)
    torch.save(
        {
            "summary": summary,
            "dataset_summary": dataset.get("summary"),
            "rows": rows,
            "paths": {
                "dataset_json": rel(DATASET_JSON),
                "features_pt": rel(FEATURES_PT),
                "rows_csv": rel(ROWS_CSV),
                "summary_json": rel(SUMMARY_JSON),
                "summary_md": rel(SUMMARY_MD),
            },
        },
        ARTIFACT_PT,
    )
    print(f"BG_TWO_TAP_HH_RLHF_DATASET_VERDICT = {dataset['BG_TWO_TAP_HH_RLHF_DATASET_VERDICT']}", flush=True)
    print(f"BG_TWO_TAP_HH_RLHF_FEATURE_CAPTURE_VERDICT = {features['meta']['BG_TWO_TAP_HH_RLHF_FEATURE_CAPTURE_VERDICT']}", flush=True)
    print(f"BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT = {summary['BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT']}", flush=True)
    print(f"Wrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
