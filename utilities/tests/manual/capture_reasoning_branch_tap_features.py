"""Capture pooled tap features for the reasoning branch pilot."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_branch_pilot_lib import REPORT_DIR, RLTT_MODEL_PATH, repo_path, write_json
from math_bg_probe_lib import MATH_CONFIGS, TAP_LAYERS, capture_pooled_taps


INPUT_JSON = REPORT_DIR / "reasoning_branch_pilot_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "reasoning_branch_tap_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "reasoning_branch_tap_features_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(INPUT_JSON))
    p.add_argument("--output", default=str(OUTPUT_PT))
    p.add_argument("--output-md", default=str(OUTPUT_MD))
    p.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--report-every", type=int, default=20)
    return p.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_text(row: dict[str, Any]) -> str:
    opts = "\n".join(f"{label}. {text}" for label, text in row["options"].items())
    return f"Question:\n{row['question']}\n\nOptions:\n{opts}\n\nCandidate answer:\nFINAL ANSWER: {row['parsed_answer']}"


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    verdict = data.get("reasoning_branch_data_verdict", "BLOCKED")
    if verdict != "READY":
        raise SystemExit(f"REASONING_BRANCH_DATA_VERDICT={verdict}; feature capture skipped")
    wanted = {uid for t in data.get("tournaments", []) for uid in t.get("candidate_uids", [])}
    candidates = [row for row in data.get("candidates", []) if row.get("candidate_uid") in wanted and row.get("label") in {"correct", "incorrect"}]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True)
    model.to(device)
    model.eval()
    texts = [feature_text(row) for row in candidates]
    pooled = capture_pooled_taps(model=model, tokenizer=tokenizer, texts=texts, max_length=int(args.max_length), device=device, report_every=int(args.report_every)).cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    by_uid = {row["candidate_uid"]: row for row in candidates}
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
            "reasoning_feature_verdict": "READY",
            "input_json": repo_path(args.input),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "n_tournaments": len(data.get("tournaments", [])),
            "n_candidates": len(features),
        },
        "candidate_features": features,
        "eval_sets": {
            "reasoning_primary": [
                {**t, "labels": ["correct" if by_uid[uid]["label"] == "correct" else "incorrect" for uid in t["candidate_uids"]]}
                for t in data.get("tournaments", [])
            ],
        },
    }
    torch.save(payload, args.output)
    md = [
        "# Reasoning Branch Tap Features",
        "",
        "REASONING_FEATURE_VERDICT = READY",
        "",
        f"- input: `{repo_path(args.input)}`",
        f"- output: `{repo_path(args.output)}`",
        f"- tournaments: `{len(data.get('tournaments', []))}`",
        f"- candidates: `{len(features)}`",
        "",
    ]
    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")
    print("REASONING_FEATURE_VERDICT = READY")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
