"""Capture pooled Ouro-RLTT tap features for v2 code branch candidates."""
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
CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(THIS_DIR))

try:
    from utilities.tests.manual.code_branch_pilot_lib import REPORT_DIR, RLTT_MODEL_PATH, output_path, repo_path
except ModuleNotFoundError:
    from code_branch_pilot_lib import REPORT_DIR, RLTT_MODEL_PATH, output_path, repo_path

from math_bg_probe_lib import TAP_LAYERS, capture_pooled_taps  # noqa: E402


DEFAULT_INPUT = REPORT_DIR / "code_branch_tournaments_v2_2026-05-16.json"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_tap_features_v2_2026-05-16.pt"
DEFAULT_MD = REPORT_DIR / "code_branch_tap_features_v2_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default="")
    parser.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--tokenizer-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    return args


def feature_text(prompt: str, code: str) -> str:
    return (
        "Problem:\n"
        + str(prompt).strip()
        + "\n\nCandidate solution:\n```python\n"
        + str(code).strip()
        + "\n```"
    )


def candidate_list(tournament: dict[str, Any], primary: str) -> list[dict[str, Any]]:
    if primary == "strict_clean":
        return list(tournament.get("strict_candidates", []))
    if primary == "diagnostic_runnable":
        return list(tournament.get("diagnostic_runnable_candidates", []))
    if primary == "diagnostic_mixed":
        return list(tournament.get("diagnostic_mixed_primary_candidates", tournament.get("diagnostic_candidates", [])))
    return []


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Code Branch v2 Tap Feature Capture",
        "",
        f"- input: `{result['input']}`",
        f"- output: `{result['output']}`",
        f"- model_path: `{result['model_path_resolved']}`",
        f"- tokenizer_path: `{result['tokenizer_path_resolved']}`",
        f"- primary_eval_set: `{result['primary_eval_set']}`",
        f"- n_tournaments: `{result['n_tournaments']}`",
        f"- n_candidates: `{result['n_candidates']}`",
        f"- tap_layers: `{result['tap_layers']}`",
        f"- pooled_shape_per_record: `{result['pooled_shape_per_record']}`",
        "",
        "Feature texts contain the problem prompt and candidate code only. Unit-test labels/results and candidate stage are not included in model input.",
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
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    verdict = payload.get("code_v2_tournament_verdict")
    if verdict not in {"CLEAN", "RUNNABLE_DIAGNOSTIC", "BROAD_DIAGNOSTIC", "DIAGNOSTIC_ONLY"}:
        raise SystemExit(f"CODE_V2_TOURNAMENT_VERDICT={verdict}; feature capture blocked")
    primary = str(payload.get("primary_eval_set"))
    tournaments = [
        t for t in payload.get("tournaments", [])
        if (
            (primary == "strict_clean" and t.get("strict_clean"))
            or (primary == "diagnostic_runnable" and t.get("diagnostic_runnable"))
            or (primary == "diagnostic_mixed" and t.get("diagnostic_mixed"))
        )
    ]
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for tournament in tournaments:
        start = len(texts)
        for cand in candidate_list(tournament, primary):
            texts.append(feature_text(tournament["prompt"], cand.get("final_code", "")))
        spans.append((start, len(texts)))

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
    for tournament, (start, end) in zip(tournaments, spans):
        candidates = candidate_list(tournament, primary)
        labels = torch.tensor([bool(c["is_correct"]) for c in candidates], dtype=torch.bool)
        records.append({
            "tournament_id": int(tournament["tournament_id"]),
            "task_id": tournament["task_id"],
            "source": tournament["source"],
            "difficulty": tournament.get("difficulty", "unknown"),
            "function_name": tournament["function_name"],
            "prompt": tournament["prompt"],
            "labels": labels,
            "candidate_texts": [feature_text(tournament["prompt"], c.get("final_code", "")) for c in candidates],
            "candidate_codes": [str(c.get("final_code", "")) for c in candidates],
            "candidate_metadata": candidates,
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
            "code_v2_tournament_verdict": verdict,
        },
        "records": records,
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_payload, output_pt)
    result = {
        "input": repo_path(input_path),
        "output": repo_path(output_pt),
        "model_path_resolved": str(Path(args.model_path).resolve()),
        "tokenizer_path_resolved": str(Path(args.tokenizer_path).resolve()),
        "primary_eval_set": primary,
        "n_tournaments": len(records),
        "n_candidates": len(texts),
        "tap_layers": list(TAP_LAYERS),
        "pooled_shape_per_record": list(records[0]["pooled"].shape) if records else [],
    }
    write_md(output_md, result)
    print(f"Wrote {output_pt}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
