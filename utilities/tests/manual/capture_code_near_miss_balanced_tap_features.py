"""Capture Ouro-RLTT tap features for balanced near-miss code tournaments."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, RLTT_MODEL_PATH, output_path, repo_path, write_json
from math_bg_probe_lib import TAP_LAYERS, capture_pooled_taps


INPUT_JSON = REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "code_branch_near_miss_balanced_tap_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "code_branch_near_miss_balanced_tap_features_2026-05-17.md"
OUTPUT_JSON = REPORT_DIR / "code_branch_near_miss_balanced_tap_features_2026-05-17.json"
CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_JSON))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--tokenizer-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=10)
    return parser.parse_args()


def feature_text(prompt: str, code: str) -> str:
    return (
        "Problem:\n"
        + str(prompt).strip()
        + "\n\nCandidate solution:\n```python\n"
        + str(code).strip()
        + "\n```"
    )


def primary_tournaments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = payload.get("balanced_tournament_verdict")
    if verdict == "GREEN":
        return [t for t in payload.get("tournaments", []) if t.get("strict_clean")]
    if verdict == "YELLOW":
        return [t for t in payload.get("tournaments", []) if t.get("strict_clean") or t.get("diagnostic_runnable")]
    return []


def candidates_for(tournament: dict[str, Any], verdict: str) -> list[dict[str, Any]]:
    if verdict == "GREEN":
        return list(tournament.get("strict_candidates", []))
    if tournament.get("strict_clean"):
        return list(tournament.get("strict_candidates", []))
    return list(tournament.get("diagnostic_runnable_candidates", []))


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Code Near-Miss Balanced Tap Feature Capture",
        "",
        f"BALANCED_FEATURE_VERDICT = {result['balanced_feature_verdict']}",
        "",
        f"- input: `{result['input']}`",
        f"- output: `{result['output']}`",
        f"- primary_eval_set: `{result['primary_eval_set']}`",
        f"- n_tournaments: `{result['n_tournaments']}`",
        f"- n_candidates: `{result['n_candidates']}`",
        f"- tap_layers: `{result['tap_layers']}`",
        "",
        "Feature texts contain only problem prompts and candidate code. Unit-test labels/results and candidate stage are excluded from model input.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise SystemExit("--device cuda requested but CUDA is not available")
    input_path = output_path(args.input)
    output_pt = output_path(args.output)
    output_md = output_path(args.output_md)
    output_json = output_path(args.output_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    verdict = payload.get("balanced_tournament_verdict")
    if verdict not in {"GREEN", "YELLOW"}:
        raise SystemExit(f"BALANCED_TOURNAMENT_VERDICT={verdict}; feature capture blocked")
    tournaments = primary_tournaments(payload)
    primary = payload.get("primary_eval_set", "none")
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    candidate_lists: list[list[dict[str, Any]]] = []
    for tournament in tournaments:
        cands = candidates_for(tournament, str(verdict))
        start = len(texts)
        for cand in cands:
            texts.append(feature_text(tournament["prompt"], cand.get("final_code", "")))
        spans.append((start, len(texts)))
        candidate_lists.append(cands)

    device = torch.device(args.device)
    print(f"loading tokenizer: {args.tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading Ouro-RLTT model: {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0

    pooled_flat = capture_pooled_taps(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=int(args.max_length),
        device=device,
        report_every=int(args.report_every),
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = []
    for tournament, cands, (start, end) in zip(tournaments, candidate_lists, spans):
        records.append({
            "tournament_id": int(tournament["tournament_id"]),
            "task_id": tournament["task_id"],
            "source": tournament["source"],
            "difficulty": tournament.get("difficulty", "unknown"),
            "function_name": tournament["function_name"],
            "prompt": tournament["prompt"],
            "labels": torch.tensor([bool(c["is_correct"]) for c in cands], dtype=torch.bool),
            "candidate_texts": [feature_text(tournament["prompt"], c.get("final_code", "")) for c in cands],
            "candidate_codes": [str(c.get("final_code", "")) for c in cands],
            "candidate_metadata": cands,
            "strict_clean": bool(tournament.get("strict_clean")),
            "diagnostic_runnable": bool(tournament.get("diagnostic_runnable")),
            "pooled": pooled_flat[start:end].contiguous().cpu(),
        })

    feature_payload = {
        "meta": {
            "input_json": repo_path(input_path),
            "model_path_resolved": str(Path(args.model_path).resolve()),
            "tokenizer_path_resolved": str(Path(args.tokenizer_path).resolve()),
            "max_length": int(args.max_length),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "n_tournaments": len(records),
            "n_candidates": len(texts),
            "primary_eval_set": primary,
            "balanced_tournament_verdict": verdict,
            "balanced_feature_verdict": "RECAPTURED",
        },
        "records": records,
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_payload, output_pt)
    result = {
        "balanced_feature_verdict": "RECAPTURED",
        "input": repo_path(input_path),
        "output": repo_path(output_pt),
        "output_json": repo_path(output_json),
        "primary_eval_set": primary,
        "n_tournaments": len(records),
        "n_candidates": len(texts),
        "tap_layers": list(TAP_LAYERS),
        "pooled_shape_per_record": list(records[0]["pooled"].shape) if records else [],
    }
    write_json(output_json, result)
    write_md(output_md, result)
    print("BALANCED_FEATURE_VERDICT = RECAPTURED", flush=True)
    print(f"Wrote {output_pt}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
