"""Capture pooled Ouro-RLTT tap features for clean GSM8K tournaments."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    DEFAULT_RLTT_PATH,
    PROJECT_ROOT,
    TAP_LAYERS,
    capture_pooled_taps,
    output_path,
    resolve_local,
)


DEFAULT_INPUT = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_micro_2026-05-16.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_tap_features_2026-05-16.pt"
DEFAULT_MD = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_tap_features_2026-05-16.md"
FINAL_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.json"
FINAL_MD = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--model-path", default=DEFAULT_RLTT_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_RLTT_PATH)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=10)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_clean_tournaments(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload.get("clean_gsm8k_verdict")
    expanded_verdict = payload.get("expanded_clean_gsm8k_verdict")
    if verdict != "CLEAN" and expanded_verdict not in {"CLEAN_30", "CLEAN_MINIMUM"}:
        raise SystemExit(
            f"clean verdict is {verdict or expanded_verdict}; feature capture is blocked"
        )
    tournaments = payload.get("tournaments", [])
    min_required = 20 if expanded_verdict in {"CLEAN_30", "CLEAN_MINIMUM"} else 5
    if len(tournaments) < min_required:
        raise SystemExit(f"fewer than {min_required} clean tournaments; feature capture is blocked")
    return payload


def write_md(path: Path, result: Dict[str, object]) -> None:
    lines = [
        "# Clean GSM8K Extreme Tap Feature Capture",
        "",
        f"- input: `{result['input']}`",
        f"- output: `{result['output']}`",
        f"- model_path: `{result['model_path_resolved']}`",
        f"- tokenizer_path: `{result['tokenizer_path_resolved']}`",
        f"- n_tournaments: `{result['n_tournaments']}`",
        f"- n_candidates: `{result['n_candidates']}`",
        f"- tap_layers: `{result['tap_layers']}`",
        f"- pooled_shape_per_record: `{result['pooled_shape_per_record']}`",
        "",
        "Pooled features are masked means over valid tokens for loops L1-L4 at layers 24, 36, and 47.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_final_report_stub(capture_result: Dict[str, object]) -> None:
    if capture_result["input"] != repo_path(DEFAULT_INPUT):
        return
    if not FINAL_JSON.exists():
        return
    payload = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
    payload["feature_capture"] = capture_result
    FINAL_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Clean GSM8K Extreme Transfer",
        "",
        f"CLEAN_GSM8K_VERDICT = {payload.get('clean_gsm8k_verdict', 'UNKNOWN')}",
        "CLEAN_TRANSFER_VERDICT = NOT_RUN",
        "RECOMMENDED_NEXT = feature_capture_complete_transfer_pending",
        "",
        "## Feature Capture Summary",
        "",
        f"- n_tournaments: `{capture_result['n_tournaments']}`",
        f"- n_candidates: `{capture_result['n_candidates']}`",
        f"- output: `{capture_result['output']}`",
        "",
    ]
    FINAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = output_path(args.input)
    output_pt = output_path(args.output)
    output_md = output_path(args.output_md) if args.output_md else output_pt.with_suffix(".md")
    payload = load_clean_tournaments(input_path)
    tournaments = payload["tournaments"]

    texts: List[str] = []
    spans = []
    for tournament in tournaments:
        start = len(texts)
        for attempt in tournament["attempts"]:
            texts.append(str(attempt["candidate_text"]))
        spans.append((start, len(texts)))

    device = torch.device(args.device)
    model_path = resolve_local(args.model_path)
    tokenizer_path = resolve_local(args.tokenizer_path)

    print(f"loading tokenizer: {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading Ouro-RLTT model: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
        max_length=args.max_length,
        device=device,
        report_every=args.report_every,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = []
    for tournament, (start, end) in zip(tournaments, spans):
        attempts = tournament["attempts"]
        labels = torch.tensor([bool(a["is_correct"]) for a in attempts], dtype=torch.bool)
        records.append({
            "tournament_id": int(tournament["tournament_id"]),
            "source": "gsm8k",
            "problem_id": int(tournament["problem_id"]),
            "dataset_index": int(tournament.get("dataset_index", tournament["problem_id"])),
            "question": tournament["question"],
            "gold_answer": tournament["gold_answer"],
            "labels": labels,
            "candidate_texts": [str(a["candidate_text"]) for a in attempts],
            "attempt_metadata": attempts,
            "pooled": pooled_flat[start:end].contiguous().cpu(),
        })

    feature_payload = {
        "meta": {
            "input_json": repo_path(input_path),
            "model_path_resolved": model_path,
            "tokenizer_path_resolved": tokenizer_path,
            "max_length": int(args.max_length),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "n_tournaments": len(records),
            "n_candidates": len(texts),
        },
        "records": records,
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_payload, output_pt)
    result = {
        "input": repo_path(input_path),
        "output": repo_path(output_pt),
        "model_path_resolved": model_path,
        "tokenizer_path_resolved": tokenizer_path,
        "n_tournaments": len(records),
        "n_candidates": len(texts),
        "tap_layers": list(TAP_LAYERS),
        "pooled_shape_per_record": list(records[0]["pooled"].shape) if records else [],
    }
    write_md(output_md, result)
    update_final_report_stub(result)
    print(f"Wrote {output_pt}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
