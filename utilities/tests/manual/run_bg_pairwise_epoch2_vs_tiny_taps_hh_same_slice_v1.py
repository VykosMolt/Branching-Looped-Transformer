"""Compare original CLT PairwiseEvaluator with tiny taps on the same HH slice.

This diagnostic reuses the exact 512 HH-RLHF chosen/rejected pairs from the
two-tap HH comparison, then:

1. streams those pairs through local Ouro-RLTT and the frozen
   ``pairwise_epoch2.pt`` evaluator;
2. evaluates loop-pattern ablations through the same frozen evaluator;
3. reuses cached pooled features to score old BG tiny taps and the new two taps.

No model weights, tokenizer files, checkpoints, tap registries, routing, wrapper
code, Hunter-Seeker modules, or steering paths are modified or executed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SRC_ROOT = PROJECT_ROOT / "shared/src"
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator, validate_hook_output  # noqa: E402
from math_bg_probe_lib import config_vector  # noqa: E402
from run_bg_two_tap_fresh_dataset_comparison_v1 import (  # noqa: E402
    accuracy_from_scores,
    finite_mean,
    load_new_two_taps,
    rel,
    safe_float,
    score_vector,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)
from run_bg_two_tap_hh_rlhf_comparison_v1 import (  # noqa: E402
    FEATURE_CONFIGS,
    load_old_bg_taps_hh,
)


SOURCE_ROOT = PROBE_ROOT / "bg_two_tap_hh_rlhf_comparison_v1_2026-05-30"
SOURCE_PAIRS_JSON = SOURCE_ROOT / "hh_rlhf_pairs.json"
SOURCE_FEATURES_PT = SOURCE_ROOT / "hh_rlhf_features.pt"

OUT_ROOT = PROBE_ROOT / "bg_pairwise_epoch2_vs_tiny_taps_hh_same_slice_v1_2026-05-30"
CLT_ROWS_CSV = OUT_ROOT / "pairwise_epoch2_rows.csv"
TINY_ROWS_CSV = OUT_ROOT / "tiny_tap_rows.csv"
PAIR_ROWS_CSV = OUT_ROOT / "per_pair_selected_scores.csv"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ARTIFACT_PT = OUT_ROOT / "pairwise_epoch2_vs_tiny_taps_hh_same_slice_v1.pt"

MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
CHECKPOINT_PATH = PROJECT_ROOT / "rpe/checkpoints/evaluator/pairwise_epoch2.pt"

CLT_CONFIGS = (
    "canonical_boundary",
    "only_loop_1",
    "only_loop_2",
    "only_loop_3",
    "only_loop_4",
    "mean_all_replicated",
    "mean_loops_23_replicated",
    "mean_loops_234_replicated",
    "loop2_doublenorm",
)

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "shared/hf_cache/hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "shared/hf_cache/datasets"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--early-exit-threshold", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--force-clt", action="store_true")
    return parser.parse_args()


class LoopStateCapture:
    def __init__(self) -> None:
        self.boundary_states: list[torch.Tensor] = []
        self.validated = False

    def hook_fn(self, _module, _inputs, output) -> None:  # type: ignore[no-untyped-def]
        hidden_states = output[1]
        if not self.validated:
            validate_hook_output(hidden_states)
            self.validated = True
        self.boundary_states = [h.detach() for h in hidden_states]

    def clear(self) -> None:
        self.boundary_states = []


def load_source_payload(max_pairs: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(SOURCE_PAIRS_JSON.read_text(encoding="utf-8"))
    pairs = list(data.get("pairs") or [])[:max_pairs]
    candidates = list(data.get("candidates") or [])
    wanted = {row["chosen_uid"] for row in pairs} | {row["rejected_uid"] for row in pairs}
    candidates = [row for row in candidates if row.get("candidate_uid") in wanted]
    by_uid = {row["candidate_uid"]: row for row in candidates}
    for pair in pairs:
        if pair["chosen_uid"] not in by_uid or pair["rejected_uid"] not in by_uid:
            raise RuntimeError(f"missing source text for pair {pair.get('pair_id')}")
    return pairs, candidates


def build_configurations(
    states_c: Sequence[torch.Tensor],
    states_r: Sequence[torch.Tensor],
    norm_fn,
) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    c = list(states_c)
    r = list(states_r)
    c_mean_all = torch.stack(c, dim=0).mean(dim=0)
    r_mean_all = torch.stack(r, dim=0).mean(dim=0)
    c_mean_23 = torch.stack(c[1:3], dim=0).mean(dim=0)
    r_mean_23 = torch.stack(r[1:3], dim=0).mean(dim=0)
    c_mean_234 = torch.stack(c[1:], dim=0).mean(dim=0)
    r_mean_234 = torch.stack(r[1:], dim=0).mean(dim=0)
    c_loop2_norm = norm_fn(c[1])
    r_loop2_norm = norm_fn(r[1])
    return {
        "canonical_boundary": (c, r),
        "only_loop_1": ([c[0]] * 4, [r[0]] * 4),
        "only_loop_2": ([c[1]] * 4, [r[1]] * 4),
        "only_loop_3": ([c[2]] * 4, [r[2]] * 4),
        "only_loop_4": ([c[3]] * 4, [r[3]] * 4),
        "mean_all_replicated": ([c_mean_all] * 4, [r_mean_all] * 4),
        "mean_loops_23_replicated": ([c_mean_23] * 4, [r_mean_23] * 4),
        "mean_loops_234_replicated": ([c_mean_234] * 4, [r_mean_234] * 4),
        "loop2_doublenorm": ([c_loop2_norm] * 4, [r_loop2_norm] * 4),
    }


def summarize_scores(scores: Sequence[float], flips: Sequence[float]) -> dict[str, Any]:
    finite = [s for s in scores if math.isfinite(float(s))]
    flip_finite = [s for s in flips if math.isfinite(float(s))]
    n = len(finite)
    centered = [s - f for s, f in zip(scores, flips) if math.isfinite(float(s)) and math.isfinite(float(f))]
    sign_flip = [
        1.0 if (float(s) > 0 and float(f) < 0) or (float(s) < 0 and float(f) > 0) else 0.0
        for s, f in zip(scores, flips)
        if math.isfinite(float(s)) and math.isfinite(float(f)) and abs(float(s)) > 1e-12 and abs(float(f)) > 1e-12
    ]
    return {
        "pair_count": n,
        "accuracy": accuracy_from_scores(finite),
        "centered_accuracy": accuracy_from_scores(centered),
        "mean_score": finite_mean(finite),
        "score_std": float(torch.tensor(finite, dtype=torch.float32).std(unbiased=False).item()) if finite else float("nan"),
        "positive_rate": float(mean([1.0 if s > 0 else 0.0 for s in finite])) if finite else float("nan"),
        "flip_positive_rate": float(mean([1.0 if s > 0 else 0.0 for s in flip_finite])) if flip_finite else float("nan"),
        "strict_sign_flip_rate": float(mean(sign_flip)) if sign_flip else float("nan"),
    }


def run_pairwise_epoch2(args: argparse.Namespace) -> dict[str, Any]:
    if ARTIFACT_PT.exists() and not args.force_clt:
        cached = torch.load(ARTIFACT_PT, map_location="cpu", weights_only=False)
        if cached.get("clt") and cached.get("clt_pair_scores"):
            return {"clt": cached["clt"], "clt_pair_scores": cached["clt_pair_scores"]}

    pairs, candidates = load_source_payload(int(args.max_pairs))
    by_uid = {row["candidate_uid"]: row for row in candidates}

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": str(device)},
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = float(args.early_exit_threshold)

    capture = LoopStateCapture()
    hook = model.model.register_forward_hook(capture.hook_fn)

    evaluator = PairwiseEvaluator().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    @torch.no_grad()
    def capture_states(texts: Sequence[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        tokens = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
        ).to(device)
        capture.clear()
        model(**tokens, use_cache=False)
        if len(capture.boundary_states) != 4:
            raise RuntimeError(f"expected 4 boundary states, got {len(capture.boundary_states)}")
        states = [h.to(device=device, dtype=torch.float32) for h in capture.boundary_states]
        return states, tokens["attention_mask"]

    @torch.no_grad()
    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        if not hasattr(model.model, "norm"):
            return x
        return model.model.norm(x.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)).to(dtype=torch.float32)

    score_store: dict[str, list[float]] = {name: [] for name in CLT_CONFIGS}
    flip_store: dict[str, list[float]] = {name: [] for name in CLT_CONFIGS}
    pair_rows: list[dict[str, Any]] = []

    start = time.time()
    for start_idx in range(0, len(pairs), int(args.batch_size)):
        batch_pairs = pairs[start_idx : start_idx + int(args.batch_size)]
        chosen_texts = [by_uid[pair["chosen_uid"]]["feature_text"] for pair in batch_pairs]
        rejected_texts = [by_uid[pair["rejected_uid"]]["feature_text"] for pair in batch_pairs]

        states_c, mask_c = capture_states(chosen_texts)
        states_r, mask_r = capture_states(rejected_texts)
        cfgs = build_configurations(states_c, states_r, norm_fn)

        batch_pair_rows = [{"pair_id": pair["pair_id"], "dataset_index": pair["dataset_index"]} for pair in batch_pairs]
        for config_name, (cs, rs) in cfgs.items():
            with torch.no_grad():
                scores = evaluator(cs, mask_c, rs, mask_r).view(-1).detach().cpu().to(torch.float32).tolist()
                flips = evaluator(rs, mask_r, cs, mask_c).view(-1).detach().cpu().to(torch.float32).tolist()
            score_store[config_name].extend(float(s) for s in scores)
            flip_store[config_name].extend(float(s) for s in flips)
            for row, score, flip in zip(batch_pair_rows, scores, flips):
                row[f"clt_{config_name}_score"] = float(score)
                row[f"clt_{config_name}_flip_score"] = float(flip)
        pair_rows.extend(batch_pair_rows)

        done = min(start_idx + int(args.batch_size), len(pairs))
        if int(args.report_every) > 0 and (done % int(args.report_every) == 0 or done == len(pairs)):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"pairwise_epoch2 scored {done}/{len(pairs)} rate={rate:.2f} pairs/s", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    hook.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    clt_rows = []
    for config_name in CLT_CONFIGS:
        row = {
            "family": "pairwise_epoch2",
            "config": config_name,
            "checkpoint": rel(args.checkpoint),
            "max_length": int(args.max_length),
            "early_exit_threshold": float(args.early_exit_threshold),
        }
        row.update(summarize_scores(score_store[config_name], flip_store[config_name]))
        clt_rows.append(row)
    return {"clt": clt_rows, "clt_pair_scores": pair_rows}


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


def tiny_domain(row: dict[str, Any]) -> str:
    name = str(row.get("candidate_name") or "")
    if "source::old_registry::HH::" in name:
        return "HH_GENERAL"
    for token in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL", "MIX_HH_OBJECTIVE", "MIX_REASONING_SCIENCE", "HH", "CODE"):
        if f"::{token}::" in name or token in name:
            return token
    return "unknown"


def evaluate_tiny_taps() -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    feature_payload = torch.load(SOURCE_FEATURES_PT, map_location="cpu", weights_only=False)
    pairs = build_pair_diffs(feature_payload)
    old_taps = load_old_bg_taps_hh()
    new_taps, _bundles = load_new_two_taps(False)
    candidates = []
    for row in old_taps:
        candidates.append({**row, "eval_family": "old_bg_tiny", "domain_role": tiny_domain(row)})
    for row in new_taps:
        candidates.append({**row, "eval_family": "new_two_tap", "domain_role": tiny_domain(row)})

    by_config: dict[str, list[torch.Tensor]] = {config: [] for config in FEATURE_CONFIGS}
    for pair in pairs:
        for config in FEATURE_CONFIGS:
            diff = (pair.get("features") or {}).get(config)
            if isinstance(diff, torch.Tensor):
                by_config[config].append(diff)

    rows: list[dict[str, Any]] = []
    scores_by_name: dict[str, list[float]] = {}
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
        name = str(cand.get("candidate_name"))
        scores_by_name[name] = scores
        rows.append(
            {
                "family": cand.get("eval_family"),
                "domain_role": cand.get("domain_role"),
                "candidate_name": name,
                "target_config": config,
                "architecture": arch,
                "pair_count": len(scores),
                "accuracy": accuracy_from_scores(scores),
                "centered_accuracy": accuracy_from_scores([2.0 * s for s in scores]),
                "mean_score": finite_mean(scores),
                "strict_sign_flip_rate": 1.0,
                "candidate_family": cand.get("candidate_family"),
                "source_run": cand.get("source_run"),
            }
        )
    return rows, scores_by_name


def best_row(rows: Sequence[dict[str, Any]], pred) -> dict[str, Any]:
    subset = [row for row in rows if pred(row)]
    return max(subset, key=lambda row: safe_float(row.get("accuracy"), -1.0), default={})


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("family", "config", "candidate_name", "domain_role", "target_config", "architecture", "accuracy", "centered_accuracy", "strict_sign_flip_rate")
    return {key: row.get(key) for key in keys if key in row}


def write_pair_rows(
    clt_pair_rows: list[dict[str, Any]],
    tiny_scores: dict[str, list[float]],
    selected: dict[str, str],
) -> None:
    rows = []
    for idx, row in enumerate(clt_pair_rows):
        out = dict(row)
        for label, name in selected.items():
            if name and name in tiny_scores and idx < len(tiny_scores[name]):
                out[f"{label}_score"] = tiny_scores[name][idx]
        rows.append(out)
    write_csv(PAIR_ROWS_CSV, rows)


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    clt_payload = run_pairwise_epoch2(args)
    clt_rows = clt_payload["clt"]
    tiny_rows, tiny_scores = evaluate_tiny_taps()

    best_clt = best_row(clt_rows, lambda _row: True)
    best_clt_centered = max(
        clt_rows,
        key=lambda row: safe_float(row.get("centered_accuracy"), -1.0),
        default={},
    )
    best_old_all = best_row(tiny_rows, lambda row: row.get("family") == "old_bg_tiny")
    best_old_hh_general = best_row(tiny_rows, lambda row: row.get("family") == "old_bg_tiny" and row.get("domain_role") == "HH_GENERAL")
    best_old_hh_any = best_row(tiny_rows, lambda row: row.get("family") == "old_bg_tiny" and row.get("domain_role") in {"HH_GENERAL", "MIX_HH_OBJECTIVE", "HH"})
    best_old_mix_code = best_row(tiny_rows, lambda row: row.get("family") == "old_bg_tiny" and row.get("domain_role") == "MIX_CODE_REASONING")
    best_old_mix_obj = best_row(tiny_rows, lambda row: row.get("family") == "old_bg_tiny" and row.get("domain_role") == "MIX_OBJECTIVE_ALL")
    best_new_all = best_row(tiny_rows, lambda row: row.get("family") == "new_two_tap")
    best_new_code = best_row(tiny_rows, lambda row: row.get("family") == "new_two_tap" and row.get("domain_role") == "MIX_CODE_REASONING")
    best_new_obj = best_row(tiny_rows, lambda row: row.get("family") == "new_two_tap" and row.get("domain_role") == "MIX_OBJECTIVE_ALL")

    selected_scores = {
        "best_old_all": str(best_old_all.get("candidate_name") or ""),
        "best_old_hh_general": str(best_old_hh_general.get("candidate_name") or ""),
        "best_old_hh_any": str(best_old_hh_any.get("candidate_name") or ""),
        "best_new_all": str(best_new_all.get("candidate_name") or ""),
    }
    write_pair_rows(clt_payload["clt_pair_scores"], tiny_scores, selected_scores)

    summary = {
        "BG_PAIRWISE_EPOCH2_VS_TINY_TAPS_HH_SAME_SLICE_VERDICT": "DIAGNOSTIC_COMPLETE",
        "pair_count": len(clt_payload["clt_pair_scores"]),
        "source_pair_slice": rel(SOURCE_PAIRS_JSON),
        "source_pooled_features": rel(SOURCE_FEATURES_PT),
        "model_path": rel(args.model_path),
        "pairwise_checkpoint": rel(args.checkpoint),
        "max_length": int(args.max_length),
        "early_exit_threshold": float(args.early_exit_threshold),
        "best_pairwise_epoch2_any": compact(best_clt),
        "best_pairwise_epoch2_centered": compact(best_clt_centered),
        "canonical_pairwise_epoch2": compact(best_row(clt_rows, lambda row: row.get("config") == "canonical_boundary")),
        "best_old_bg_tiny_all": compact(best_old_all),
        "best_old_bg_tiny_hh_general": compact(best_old_hh_general),
        "best_old_bg_tiny_hh_any": compact(best_old_hh_any),
        "best_old_bg_tiny_mix_code_reasoning": compact(best_old_mix_code),
        "best_old_bg_tiny_mix_objective_all": compact(best_old_mix_obj),
        "best_new_two_tap_all": compact(best_new_all),
        "best_new_two_tap_mix_code_reasoning": compact(best_new_code),
        "best_new_two_tap_mix_objective_all": compact(best_new_obj),
        "interpretation": {
            "primary_caveat": "Tiny-tap best rows are diagnostic selections on this same HH slice; do not use as readiness/model-selection evidence.",
            "main_question": "Whether no-GRU relational readouts recover the old CLT HH preference signal on the identical two-tap HH slice.",
            "result_rule": "Compare pairwise_epoch2 loop-pattern rows to old/new tiny taps; exact-antisymmetric tiny taps have strict sign flip by construction.",
        },
    }

    write_csv(CLT_ROWS_CSV, clt_rows)
    write_csv(TINY_ROWS_CSV, tiny_rows)
    write_json(SUMMARY_JSON, {"summary": summary, "pairwise_epoch2_rows": clt_rows, "tiny_tap_rows": tiny_rows})
    torch.save(
        {
            "summary": summary,
            "clt": clt_rows,
            "tiny": tiny_rows,
            "clt_pair_scores": clt_payload["clt_pair_scores"],
            "paths": {
                "summary_md": rel(SUMMARY_MD),
                "summary_json": rel(SUMMARY_JSON),
                "clt_rows_csv": rel(CLT_ROWS_CSV),
                "tiny_rows_csv": rel(TINY_ROWS_CSV),
                "pair_rows_csv": rel(PAIR_ROWS_CSV),
            },
        },
        ARTIFACT_PT,
    )

    lines = [
        "# Pairwise Epoch2 vs Tiny Taps on Same HH Slice v1",
        "",
        "BG_PAIRWISE_EPOCH2_VS_TINY_TAPS_HH_SAME_SLICE_VERDICT = DIAGNOSTIC_COMPLETE",
        "",
        "## Scope",
        "",
        "- Reuses the exact 512 HH-RLHF test pairs from the two-tap HH comparison.",
        "- Scores original `pairwise_epoch2.pt` through local Ouro-RLTT loop-boundary states.",
        "- Scores old BG tiny taps and new two taps from the cached pooled layer features.",
        "- All scores are pairwise chosen-vs-rejected comparator scores.",
        "- No training, checkpoint edits, routing changes, wrapper/local-agent code, Hunter-Seeker execution, or steering were performed.",
        "",
        "## Headline",
        "",
        f"- pair_count: `{summary['pair_count']}`",
        f"- canonical_pairwise_epoch2: `{summary['canonical_pairwise_epoch2']}`",
        f"- best_pairwise_epoch2_any: `{summary['best_pairwise_epoch2_any']}`",
        f"- best_pairwise_epoch2_centered: `{summary['best_pairwise_epoch2_centered']}`",
        f"- best_old_bg_tiny_all: `{summary['best_old_bg_tiny_all']}`",
        f"- best_old_bg_tiny_hh_general: `{summary['best_old_bg_tiny_hh_general']}`",
        f"- best_old_bg_tiny_hh_any: `{summary['best_old_bg_tiny_hh_any']}`",
        f"- best_new_two_tap_all: `{summary['best_new_two_tap_all']}`",
        "",
        "## Interpretation",
        "",
        "The original CLT evaluator and its loop-pattern ablations are evaluated on the same HH slice as the recent two-tap run. The tiny-tap rows remain diagnostic same-slice selections, not readiness-bearing model selection.",
        "",
        "## Files",
        "",
        f"- pairwise rows: `{rel(CLT_ROWS_CSV)}`",
        f"- tiny tap rows: `{rel(TINY_ROWS_CSV)}`",
        f"- selected per-pair scores: `{rel(PAIR_ROWS_CSV)}`",
        f"- summary JSON: `{rel(SUMMARY_JSON)}`",
        f"- artifact: `{rel(ARTIFACT_PT)}`",
        "",
    ]
    write_md(SUMMARY_MD, lines)
    print("BG_PAIRWISE_EPOCH2_VS_TINY_TAPS_HH_SAME_SLICE_VERDICT = DIAGNOSTIC_COMPLETE", flush=True)
    print(f"Wrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
