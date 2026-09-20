"""Capture Ouro-RLTT pooled features for strict-clean code transfer candidates."""
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

from math_bg_probe_lib import MATH_CONFIGS, TAP_LAYERS, capture_pooled_taps  # noqa: E402


INPUT_JSON = REPORT_DIR / "code_strict_clean_transfer_set_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "code_strict_clean_transfer_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_transfer_features_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_JSON))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-md", default="")
    parser.add_argument("--model-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--tokenizer-path", default=str(RLTT_MODEL_PATH))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=5)
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


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Strict-Clean Code Transfer Feature Capture",
        "",
        f"STRICT_CLEAN_FEATURE_VERDICT = {result['strict_clean_feature_verdict']}",
        "",
        f"- input: `{result['input']}`",
        f"- output: `{result['output']}`",
        f"- model_path: `{result['model_path_resolved']}`",
        f"- tokenizer_path: `{result['tokenizer_path_resolved']}`",
        f"- n_tournaments_primary: `{result['n_tournaments_primary']}`",
        f"- n_candidates_unique: `{result['n_candidates_unique']}`",
        f"- eval_sets: `{result['eval_sets']}`",
        f"- tap_layers: `{result['tap_layers']}`",
        f"- feature_configs_supported: `{result['feature_configs_supported']}`",
        "",
        "Feature texts contain only the problem prompt and candidate code. Unit-test labels/results, candidate stage, source verdicts, and evaluator scores are excluded from model input.",
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
    verdict = payload.get("strict_clean_transfer_set_verdict")
    n_tournaments = int(payload.get("summary", {}).get("n_tasks", 0))
    if verdict not in {"READY", "TOO_SMALL"}:
        raise SystemExit(f"STRICT_CLEAN_TRANSFER_SET_VERDICT={verdict}; feature capture blocked")
    if verdict == "TOO_SMALL" and n_tournaments < 3:
        raise SystemExit("STRICT_CLEAN_FEATURE_VERDICT=BLOCKED: fewer than 3 tournaments")

    unique: dict[str, dict[str, Any]] = {}
    eval_sets: dict[str, list[dict[str, Any]]] = {"strict_clean_primary": [], "strict_clean_plus_wrong_code": []}
    for tournament in payload.get("tournaments", []):
        for set_name, candidate_key in (
            ("strict_clean_primary", "strict_clean_primary_candidates"),
            ("strict_clean_plus_wrong_code", "strict_clean_plus_wrong_code_candidates"),
        ):
            candidate_uids = []
            labels = []
            for cand in tournament.get(candidate_key, []):
                uid = str(cand["candidate_uid"])
                candidate_uids.append(uid)
                labels.append(bool(cand.get("is_correct")))
                unique.setdefault(uid, {
                    "candidate_uid": uid,
                    "task_id": cand.get("task_id"),
                    "prompt": tournament.get("prompt", ""),
                    "code": cand.get("final_code", ""),
                    "feature_text": feature_text(tournament.get("prompt", ""), cand.get("final_code", "")),
                    "candidate_metadata": cand,
                })
            eval_sets[set_name].append({
                "tournament_id": int(tournament["tournament_id"]),
                "task_id": tournament["task_id"],
                "source": tournament.get("source", "unknown"),
                "difficulty": tournament.get("difficulty", "unknown"),
                "function_name": tournament.get("function_name", ""),
                "prompt": tournament.get("prompt", ""),
                "candidate_uids": candidate_uids,
                "labels": labels,
            })

    candidate_items = list(unique.values())
    texts = [item["feature_text"] for item in candidate_items]
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

    pooled = capture_pooled_taps(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=int(args.max_length),
        device=device,
        report_every=int(args.report_every),
    ).cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate_features = []
    for idx, item in enumerate(candidate_items):
        candidate_features.append({
            "candidate_uid": item["candidate_uid"],
            "task_id": item["task_id"],
            "candidate_metadata": item["candidate_metadata"],
            "feature_text": item["feature_text"],
            "pooled": pooled[idx].contiguous().to(torch.float32),
        })
    feature_payload = {
        "meta": {
            "strict_clean_feature_verdict": "READY",
            "input_json": repo_path(input_path),
            "model_path_resolved": str(Path(args.model_path).resolve()),
            "tokenizer_path_resolved": str(Path(args.tokenizer_path).resolve()),
            "max_length": int(args.max_length),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "n_tournaments_primary": n_tournaments,
            "n_candidates_unique": len(candidate_features),
            "eval_sets": {name: len(rows) for name, rows in eval_sets.items()},
        },
        "candidate_features": candidate_features,
        "eval_sets": eval_sets,
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_payload, output_pt)
    result = {
        **feature_payload["meta"],
        "input": repo_path(input_path),
        "output": repo_path(output_pt),
    }
    write_md(output_md, result)
    print("STRICT_CLEAN_FEATURE_VERDICT = READY", flush=True)
    print(f"Wrote {output_pt}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
