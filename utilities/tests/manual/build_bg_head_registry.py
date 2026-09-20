"""Build a reusable registry of HH-trained and code-trained tiny heads."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json
from evaluate_hh_transfer_on_clean_gsm8k_extreme import build_hh_features, split_indices, train_hh_head
from math_bg_probe_lib import MATH_CONFIGS, config_dim
from train_code_specific_tiny_heads_and_eval import split_pairs_by_task, train_head, feature_map


HH_CAPTURE = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
CODE_FEATURES = REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.pt"
OUTPUT_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "bg_head_registry_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_head_registry_2026-05-17.md"

ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")
KNOWN_STRICT_CLEAN_EVAL_IDS = {
    "mbpp/100", "mbpp/129", "mbpp/283", "mbpp/291", "mbpp/391", "mbpp/392",
    "mbpp/11", "mbpp/20", "mbpp/434",
    "HumanEval/10", "HumanEval/118", "HumanEval/123", "HumanEval/125",
    "HumanEval/141", "HumanEval/148", "HumanEval/69",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hh-capture", default=str(HH_CAPTURE))
    parser.add_argument("--code-features", default=str(CODE_FEATURES))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_hh_registry(hh_payload: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), int(args.heldout), int(args.seed))
    heads = []
    for config in MATH_CONFIGS:
        print(f"training HH registry heads {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        for architecture in ARCHITECTURES:
            head, metrics = train_hh_head(architecture, config, chosen, rejected, train_idx, eval_idx, args, device)
            heads.append({
                "head_family": "HH",
                "architecture": architecture,
                "family_architecture": "HH_NoNorm" if architecture == "AntisymLinearNoNorm" else "HH_AntisymLinear",
                "config": config,
                "dim": config_dim(config),
                "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "train_metrics": metrics,
            })
    return heads, {"pairs_total": len(hh_payload["packs"]), "train_pairs": len(train_idx), "heldout_pairs": len(eval_idx)}


def train_code_registry(feature_payload: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pooled_by_uid = feature_map(feature_payload)
    pairs = [
        pair for pair in feature_payload.get("training_pairs_primary", []) or []
        if pair.get("preferred_uid") in pooled_by_uid and pair.get("rejected_uid") in pooled_by_uid
    ]
    leakage = sorted({str(pair.get("task_id")) for pair in pairs if str(pair.get("task_id")) in KNOWN_STRICT_CLEAN_EVAL_IDS})
    if leakage:
        raise SystemExit(f"known strict-clean eval task leakage in code pairs: {leakage}")
    task_ids = sorted({str(pair["task_id"]) for pair in pairs})
    if len(task_ids) < 8 or len(pairs) < 30:
        raise SystemExit(f"insufficient code train pairs: tasks={len(task_ids)} pairs={len(pairs)}")
    train_pairs, val_pairs, validation = split_pairs_by_task(pairs, int(args.seed))
    heads = []
    for config in MATH_CONFIGS:
        print(f"training CODE registry heads {config}", flush=True)
        for architecture in ARCHITECTURES:
            head, metrics = train_head(
                architecture=architecture,
                config=config,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                pooled_by_uid=pooled_by_uid,
                args=args,
                device=device,
            )
            heads.append({
                "head_family": "CODE",
                "architecture": architecture,
                "family_architecture": "CODE_NoNorm" if architecture == "AntisymLinearNoNorm" else "CODE_AntisymLinear",
                "config": config,
                "dim": config_dim(config),
                "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "train_metrics": metrics,
            })
    return heads, {
        "training_task_count": len(task_ids),
        "primary_training_pair_count": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "validation": validation,
        "excluded_strict_clean_eval_task_ids": sorted(KNOWN_STRICT_CLEAN_EVAL_IDS),
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# BG Head Registry",
        "",
        f"BG_HEAD_REGISTRY_VERDICT = {payload['bg_head_registry_verdict']}",
        "",
        f"- output_pt: `{payload['outputs']['pt']}`",
        f"- heads_total: `{s['heads_total']}`",
        f"- hh_heads: `{s['hh_heads']}`",
        f"- code_heads: `{s['code_heads']}`",
        f"- configs: `{s['configs']}`",
        f"- architectures: `{s['architectures']}`",
        f"- hh_training: `{payload['hh_training']}`",
        f"- code_training: `{payload['code_training']}`",
        "",
        "Saved heads are tiny AntisymLinear or AntisymLinearNoNorm state_dicts only; no backbone weights are modified.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    hh_path = output_path(args.hh_capture)
    code_path = output_path(args.code_features)
    out_pt = output_path(args.output)
    out_json = output_path(args.output_json)
    out_md = output_path(args.output_md)
    blockers = []
    if not hh_path.exists():
        blockers.append(f"missing HH capture: {repo_path(hh_path)}")
    if not code_path.exists():
        blockers.append(f"missing code features: {repo_path(code_path)}")
    if blockers:
        verdict = "BLOCKED"
        payload = {"bg_head_registry_verdict": verdict, "blockers": blockers}
        write_json(out_json, payload)
        raise SystemExit("; ".join(blockers))
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    code_payload = torch.load(code_path, map_location="cpu", weights_only=False)
    hh_heads, hh_training = train_hh_registry(hh_payload, args, device)
    code_heads, code_training = train_code_registry(code_payload, args, device)
    heads = hh_heads + code_heads
    verdict = "RETRAINED"
    registry = {
        "meta": {
            "bg_head_registry_verdict": verdict,
            "hh_capture": repo_path(hh_path),
            "code_features": repo_path(code_path),
            "configs": list(MATH_CONFIGS),
            "architectures": list(ARCHITECTURES),
            "seed": int(args.seed),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "swap_augmentation": False,
            "lambda_sym": 0.0,
            "hh_training": hh_training,
            "code_training": code_training,
        },
        "heads": heads,
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(registry, out_pt)
    payload = {
        "bg_head_registry_verdict": verdict,
        "summary": {
            "BG_HEAD_REGISTRY_VERDICT": verdict,
            "heads_total": len(heads),
            "hh_heads": len(hh_heads),
            "code_heads": len(code_heads),
            "configs": list(MATH_CONFIGS),
            "architectures": list(ARCHITECTURES),
        },
        "hh_training": hh_training,
        "code_training": code_training,
        "outputs": {"pt": repo_path(out_pt), "json": repo_path(out_json), "md": repo_path(out_md)},
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"BG_HEAD_REGISTRY_VERDICT = {verdict}")
    print(f"heads_total = {len(heads)}")
    print(f"Wrote {out_pt}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
