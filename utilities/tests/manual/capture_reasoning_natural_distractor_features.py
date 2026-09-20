"""Capture pooled Ouro-RLTT tap features for natural MCQ distractor candidates."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_branch_pilot_lib import REPORT_DIR, RLTT_MODEL_PATH, repo_path, write_json
from math_bg_probe_lib import MATH_CONFIGS, TAP_LAYERS, capture_pooled_taps


INPUT_JSON = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "reasoning_natural_distractor_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "reasoning_natural_distractor_features_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_JSON))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=1800)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_text(row: dict[str, Any]) -> str:
    return f"Question:\n{row['question']}\n\nCandidate answer:\n{row['option_letter']}. {row['option_text']}"


def write_md(path: Path, verdict: str, args: argparse.Namespace, n_candidates: int, details: dict[str, Any] | None = None) -> None:
    details = details or {}
    lines = [
        "# Reasoning Natural Distractor Features",
        "",
        f"REASONING_DISTRACTOR_FEATURE_VERDICT = {verdict}",
        "",
        f"- input: `{repo_path(args.input)}`",
        f"- output: `{repo_path(args.output)}`",
        f"- candidates: `{n_candidates}`",
        f"- details: `{details}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    set_verdict = data.get("reasoning_distractor_set_verdict", "BLOCKED")
    if set_verdict not in {"READY", "PARTIAL"}:
        write_md(Path(args.output_md), "BLOCKED", args, 0, {"set_verdict": set_verdict})
        raise SystemExit(f"REASONING_DISTRACTOR_SET_VERDICT={set_verdict}; feature capture skipped")
    if args.device == "cuda" and not torch.cuda.is_available():
        write_md(Path(args.output_md), "BLOCKED", args, 0, {"error": "CUDA unavailable"})
        raise SystemExit("--device cuda requested but CUDA unavailable")
    candidates = list(data.get("candidates", []) or [])
    start = time.time()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True)
    model.to(device)
    model.eval()
    texts = [feature_text(row) for row in candidates]
    if time.time() - start > int(args.max_seconds):
        write_md(Path(args.output_md), "TIMEOUT", args, 0, {"elapsed_before_capture": time.time() - start})
        raise SystemExit("REASONING_DISTRACTOR_FEATURE_VERDICT=TIMEOUT")
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
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if elapsed > int(args.max_seconds):
        write_md(Path(args.output_md), "TIMEOUT", args, len(candidates), {"elapsed_seconds": elapsed})
        raise SystemExit("REASONING_DISTRACTOR_FEATURE_VERDICT=TIMEOUT")
    features = []
    for idx, row in enumerate(candidates):
        features.append({
            "candidate_uid": row["candidate_uid"],
            "task_id": row["task_id"],
            "candidate_metadata": row,
            "feature_text": texts[idx],
            "pooled": pooled[idx].to(torch.float32).contiguous(),
        })
    payload = {
        "meta": {
            "reasoning_distractor_feature_verdict": "READY",
            "input_json": repo_path(args.input),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "n_tournaments": len(data.get("tournaments", [])),
            "n_candidates": len(features),
            "elapsed_seconds": elapsed,
        },
        "candidate_features": features,
        "eval_sets": {
            "reasoning_natural_distractors": data.get("tournaments", []),
        },
    }
    torch.save(payload, args.output)
    write_md(Path(args.output_md), "READY", args, len(features), {"elapsed_seconds": elapsed, "n_tournaments": len(data.get("tournaments", []))})
    print("REASONING_DISTRACTOR_FEATURE_VERDICT = READY")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
