"""Merge or recapture features for the expanded strict-clean code comparison."""
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

from build_code_specific_training_control_splits import feature_text, relevant_feature_paths  # noqa: E402
from ensure_code_specific_training_features import load_existing_features, make_candidate_feature  # noqa: E402
from math_bg_probe_lib import MATH_CONFIGS, TAP_LAYERS, capture_pooled_taps  # noqa: E402


EVAL_SET_JSON = REPORT_DIR / "code_expanded_strict_clean_eval_set_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "code_expanded_strict_clean_features_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=str(EVAL_SET_JSON))
    parser.add_argument("--output", default=str(OUTPUT_PT))
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


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path: Path | str) -> str:
    return repo_path(Path(path))


def primary_eval_uids(eval_payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for name in ("OLD6_primary", "NEW10_primary", "ALL16_primary"):
        for tournament in eval_payload.get("eval_sets", {}).get(name, []) or []:
            out.update(str(uid) for uid in tournament.get("candidate_uids", []) or [])
    return out


def recapture_missing(
    *,
    missing_uids: list[str],
    candidate_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not missing_uids:
        return []
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise RuntimeError("--device cuda requested but CUDA is not available")
    rows = [candidate_index[uid] for uid in missing_uids]
    texts = [feature_text(row.get("prompt", ""), row.get("final_code", "")) for row in rows]
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

    out: list[dict[str, Any]] = []
    for idx, uid in enumerate(missing_uids):
        out.append(make_candidate_feature(
            candidate_uid=uid,
            pooled=pooled[idx],
            candidate_index=candidate_index,
            source_feature_artifact="recaptured:code_expanded_strict_clean_features_2026-05-17",
        ))
    return out


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Expanded Strict-Clean Feature Coverage",
        "",
        f"EXPANDED_STRICT_CLEAN_FEATURE_VERDICT = {result['expanded_strict_clean_feature_verdict']}",
        "",
        f"- eval_set: `{result['eval_set']}`",
        f"- output: `{result['output']}`",
        f"- eval_set_verdict: `{result['eval_set_verdict']}`",
        f"- required_candidates_total: `{result['required_candidates_total']}`",
        f"- required_primary_eval_candidates: `{result['required_primary_eval_candidates']}`",
        f"- existing_feature_candidates_total: `{result['existing_feature_candidates_total']}`",
        f"- recaptured_candidates: `{result['recaptured_candidates']}`",
        f"- blocked_missing_candidates: `{result['blocked_missing_candidates']}`",
        f"- tap_layers: `{result['tap_layers']}`",
        f"- loops: `{result['loops']}`",
        f"- feature_configs_supported: `{result['feature_configs_supported']}`",
        "",
        "Feature input text contains only the problem prompt and candidate code. Unit-test results, labels, candidate stage, source fields, verdicts, and evaluator outputs are excluded from model input.",
        "",
        "## Existing Feature Artifacts",
        "",
    ]
    for row in result["feature_artifact_inventory"]:
        lines.append(f"- `{row['path']}` loaded=`{row['loaded']}` entries=`{row.get('entries_added', 0)}`")
    if result.get("blocked_candidates"):
        lines.extend(["", "## Blocked Candidates", ""])
        for row in result["blocked_candidates"]:
            lines.append(f"- `{row['candidate_uid']}` task=`{row.get('task_id', '')}` reason=`{row['reason']}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_set_path = output_path(args.eval_set)
    output_pt = output_path(args.output)
    output_md = output_path(args.output_md)
    eval_payload = load_json(eval_set_path)
    eval_verdict = str(eval_payload.get("expanded_strict_clean_set_verdict", "BLOCKED"))
    if eval_verdict not in {"READY", "PARTIAL"}:
        result = {
            "expanded_strict_clean_feature_verdict": "BLOCKED",
            "eval_set": rel(eval_set_path),
            "output": rel(output_pt),
            "eval_set_verdict": eval_verdict,
            "required_candidates_total": 0,
            "required_primary_eval_candidates": 0,
            "existing_feature_candidates_total": 0,
            "recaptured_candidates": 0,
            "blocked_missing_candidates": 1,
            "blocked_candidates": [{"candidate_uid": "", "reason": f"eval set verdict {eval_verdict} is not runnable"}],
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "feature_artifact_inventory": [],
        }
        write_md(output_md, result)
        raise SystemExit("EXPANDED_STRICT_CLEAN_FEATURE_VERDICT=BLOCKED")

    candidate_index = {str(uid): dict(row) for uid, row in eval_payload.get("candidate_index", {}).items()}
    required_uids = [str(uid) for uid in eval_payload.get("required_feature_candidate_uids", [])]
    primary_uids = primary_eval_uids(eval_payload)
    feature_paths = relevant_feature_paths()
    if output_pt.exists() and output_pt not in feature_paths:
        feature_paths.append(output_pt)
    existing_features, inventory = load_existing_features(sorted(set(feature_paths)), candidate_index)

    missing_uids = [uid for uid in required_uids if uid not in existing_features]
    blocked: list[dict[str, Any]] = []
    recapturable: list[str] = []
    for uid in missing_uids:
        row = candidate_index.get(uid, {})
        if str(row.get("prompt") or "").strip() and str(row.get("final_code") or "").strip():
            recapturable.append(uid)
        else:
            blocked.append({
                "candidate_uid": uid,
                "task_id": row.get("task_id", ""),
                "label": row.get("label", ""),
                "reason": "missing prompt or final_code for recapture",
            })

    recaptured: list[dict[str, Any]] = []
    verdict = "READY"
    if blocked:
        verdict = "BLOCKED"
    elif recapturable:
        recaptured = recapture_missing(missing_uids=recapturable, candidate_index=candidate_index, args=args)
        verdict = "RECAPTURED"

    final_features: dict[str, dict[str, Any]] = {}
    for uid in required_uids:
        if uid in existing_features:
            entry = dict(existing_features[uid])
            entry["candidate_uid"] = uid
            meta = dict(candidate_index.get(uid, entry.get("candidate_metadata", {})))
            meta["candidate_uid"] = uid
            entry["candidate_metadata"] = meta
            entry["task_id"] = meta.get("task_id", entry.get("task_id", ""))
            final_features[uid] = entry
    for entry in recaptured:
        final_features[str(entry["candidate_uid"])] = entry

    if verdict != "BLOCKED":
        still_missing = [uid for uid in required_uids if uid not in final_features]
        if still_missing:
            verdict = "BLOCKED"
            for uid in still_missing:
                blocked.append({"candidate_uid": uid, "reason": "feature still missing after merge/recapture"})

    feature_payload = {
        "meta": {
            "expanded_strict_clean_feature_verdict": verdict,
            "eval_set_json": rel(eval_set_path),
            "model_path_resolved": str(Path(args.model_path).resolve()),
            "tokenizer_path_resolved": str(Path(args.tokenizer_path).resolve()),
            "max_length": int(args.max_length),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "required_candidates_total": len(required_uids),
            "required_primary_eval_candidates": len(primary_uids),
            "existing_feature_candidates_total": len([uid for uid in required_uids if uid in existing_features]),
            "existing_primary_eval_feature_candidates": len([uid for uid in primary_uids if uid in existing_features]),
            "recaptured_candidates": len(recaptured),
            "blocked_missing_candidates": len(blocked),
            "feature_artifact_inventory": inventory,
        },
        "candidate_features": [final_features[uid] for uid in required_uids if uid in final_features],
        "eval_sets": eval_payload.get("eval_sets", {}),
        "training_pairs_primary": eval_payload.get("training_pairs_primary", []),
        "training_pairs_near_miss_only": eval_payload.get("training_pairs_near_miss_only", []),
        "training_tasks": eval_payload.get("training_tasks", []),
        "candidate_index": {uid: candidate_index[uid] for uid in required_uids if uid in candidate_index},
    }
    if verdict != "BLOCKED":
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_payload, output_pt)

    result = {
        "expanded_strict_clean_feature_verdict": verdict,
        "eval_set": rel(eval_set_path),
        "output": rel(output_pt),
        "eval_set_verdict": eval_verdict,
        "required_candidates_total": len(required_uids),
        "required_primary_eval_candidates": len(primary_uids),
        "existing_feature_candidates_total": len([uid for uid in required_uids if uid in existing_features]),
        "recaptured_candidates": len(recaptured),
        "blocked_missing_candidates": len(blocked),
        "blocked_candidates": blocked,
        "tap_layers": list(TAP_LAYERS),
        "loops": ["L1", "L2", "L3", "L4"],
        "feature_configs_supported": list(MATH_CONFIGS),
        "feature_artifact_inventory": inventory,
    }
    write_md(output_md, result)
    print(f"EXPANDED_STRICT_CLEAN_FEATURE_VERDICT = {verdict}", flush=True)
    print(f"required_candidates_total = {len(required_uids)}", flush=True)
    print(f"required_primary_eval_candidates = {len(primary_uids)}", flush=True)
    print(f"recaptured_candidates = {len(recaptured)}", flush=True)
    print(f"Wrote {output_pt}", flush=True)
    print(f"Wrote {output_md}", flush=True)
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
