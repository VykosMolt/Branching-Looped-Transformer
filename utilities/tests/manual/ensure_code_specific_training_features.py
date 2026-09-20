"""Merge or recapture pooled features for code-specific tiny-head training."""
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

from build_code_specific_training_control_splits import (  # noqa: E402
    candidate_aliases,
    candidate_code,
    feature_text,
    relevant_feature_paths,
    stable_candidate_uid,
)
from math_bg_probe_lib import MATH_CONFIGS, TAP_LAYERS, capture_pooled_taps  # noqa: E402


SPLITS_JSON = REPORT_DIR / "code_specific_training_control_splits_2026-05-17.json"
OUTPUT_PT = REPORT_DIR / "code_specific_training_features_2026-05-17.pt"
OUTPUT_MD = REPORT_DIR / "code_specific_training_features_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
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


def make_candidate_feature(
    *,
    candidate_uid: str,
    pooled: torch.Tensor,
    candidate_index: dict[str, dict[str, Any]],
    source_feature_artifact: str,
) -> dict[str, Any]:
    meta = dict(candidate_index.get(candidate_uid, {"candidate_uid": candidate_uid}))
    meta["candidate_uid"] = candidate_uid
    prompt = str(meta.get("prompt") or "")
    code = str(meta.get("final_code") or "")
    return {
        "candidate_uid": candidate_uid,
        "task_id": meta.get("task_id", ""),
        "candidate_metadata": meta,
        "feature_text": feature_text(prompt, code),
        "pooled": pooled.detach().cpu().contiguous().to(torch.float32),
        "source_feature_artifact": source_feature_artifact,
    }


def add_feature_entry(
    features: dict[str, dict[str, Any]],
    aliases: set[str],
    entry: dict[str, Any],
) -> None:
    primary_uid = str(entry["candidate_uid"])
    features.setdefault(primary_uid, entry)
    for alias in aliases:
        if alias not in features:
            alias_entry = dict(entry)
            alias_entry["candidate_uid"] = alias
            alias_entry["alias_of"] = primary_uid
            features[alias] = alias_entry


def load_existing_features(
    paths: list[Path],
    candidate_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    features: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            inventory.append({"path": rel(path), "loaded": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        added = 0
        if isinstance(payload, dict) and "candidate_features" in payload:
            for row in payload.get("candidate_features", []) or []:
                meta = dict(row.get("candidate_metadata") or {})
                uid = str(row.get("candidate_uid") or meta.get("candidate_uid") or "").strip()
                if not uid:
                    code = candidate_code(meta)
                    uid = stable_candidate_uid(meta, str(meta.get("task_id") or ""), code)
                pooled = row["pooled"].detach().cpu().to(torch.float32)
                known_meta = dict(candidate_index.get(uid, meta))
                known_meta.setdefault("candidate_uid", uid)
                entry = {
                    "candidate_uid": uid,
                    "task_id": known_meta.get("task_id", meta.get("task_id", "")),
                    "candidate_metadata": known_meta,
                    "feature_text": row.get("feature_text") or feature_text(known_meta.get("prompt", ""), known_meta.get("final_code", "")),
                    "pooled": pooled.contiguous(),
                    "source_feature_artifact": rel(path),
                }
                aliases = candidate_aliases(meta, str(meta.get("task_id") or known_meta.get("task_id") or ""), candidate_code(meta))
                aliases.add(uid)
                add_feature_entry(features, aliases, entry)
                added += 1
        if isinstance(payload, dict) and "records" in payload:
            for record in payload.get("records", []) or []:
                task_id = str(record.get("task_id") or "")
                metadata = list(record.get("candidate_metadata", []) or [])
                codes = list(record.get("candidate_codes", []) or [])
                pooled_block = record.get("pooled")
                if pooled_block is None:
                    continue
                for idx, meta_any in enumerate(metadata):
                    meta = dict(meta_any)
                    code = codes[idx] if idx < len(codes) else candidate_code(meta)
                    uid = stable_candidate_uid(meta, task_id, code)
                    pooled = pooled_block[idx].detach().cpu().to(torch.float32)
                    known_meta = dict(candidate_index.get(uid, meta))
                    known_meta.setdefault("candidate_uid", uid)
                    if code and not known_meta.get("final_code"):
                        known_meta["final_code"] = code
                    if record.get("prompt") and not known_meta.get("prompt"):
                        known_meta["prompt"] = record.get("prompt")
                    entry = {
                        "candidate_uid": uid,
                        "task_id": known_meta.get("task_id", task_id),
                        "candidate_metadata": known_meta,
                        "feature_text": feature_text(known_meta.get("prompt", ""), known_meta.get("final_code", code)),
                        "pooled": pooled.contiguous(),
                        "source_feature_artifact": rel(path),
                    }
                    aliases = candidate_aliases(meta, task_id, code)
                    aliases.add(uid)
                    add_feature_entry(features, aliases, entry)
                    added += 1
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        inventory.append({
            "path": rel(path),
            "loaded": True,
            "entries_added": added,
            "meta": {k: v for k, v in dict(meta).items() if k not in {"records", "candidate_features"}},
        })
    return features, inventory


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
            source_feature_artifact="recaptured:code_specific_training_features_2026-05-17",
        ))
    return out


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Code-Specific Training Feature Coverage",
        "",
        f"CODE_SPECIFIC_FEATURE_VERDICT = {result['code_specific_feature_verdict']}",
        "",
        f"- splits: `{result['splits']}`",
        f"- output: `{result['output']}`",
        f"- split_verdict: `{result['split_verdict']}`",
        f"- required_candidates: `{result['required_candidates']}`",
        f"- existing_feature_candidates: `{result['existing_feature_candidates']}`",
        f"- recaptured_candidates: `{result['recaptured_candidates']}`",
        f"- blocked_missing_candidates: `{result['blocked_missing_candidates']}`",
        f"- tap_layers: `{result['tap_layers']}`",
        f"- loops: `{result['loops']}`",
        f"- feature_configs_supported: `{result['feature_configs_supported']}`",
        "",
        "Feature input text contains only the problem prompt and candidate code. Unit-test results, labels, candidate stage, source verdicts, and evaluator scores are excluded from model input.",
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
    splits_path = output_path(args.splits)
    output_pt = output_path(args.output)
    output_md = output_path(args.output_md)
    splits = load_json(splits_path)
    split_verdict = str(splits.get("code_specific_split_verdict", "BLOCKED"))
    if split_verdict not in {"READY", "MISSING_FEATURES"}:
        result = {
            "code_specific_feature_verdict": "BLOCKED",
            "splits": rel(splits_path),
            "output": rel(output_pt),
            "split_verdict": split_verdict,
            "required_candidates": 0,
            "existing_feature_candidates": 0,
            "recaptured_candidates": 0,
            "blocked_missing_candidates": 0,
            "blocked_candidates": [{"candidate_uid": "", "reason": f"split verdict {split_verdict} is not runnable"}],
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "feature_artifact_inventory": [],
        }
        write_md(output_md, result)
        raise SystemExit("CODE_SPECIFIC_FEATURE_VERDICT=BLOCKED")

    candidate_index = {str(uid): dict(row) for uid, row in splits.get("candidate_index", {}).items()}
    required_uids = [str(uid) for uid in splits.get("required_feature_candidate_uids", [])]
    feature_paths = relevant_feature_paths()
    existing_features, inventory = load_existing_features(feature_paths, candidate_index)

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
            "code_specific_feature_verdict": verdict,
            "splits_json": rel(splits_path),
            "model_path_resolved": str(Path(args.model_path).resolve()),
            "tokenizer_path_resolved": str(Path(args.tokenizer_path).resolve()),
            "max_length": int(args.max_length),
            "tap_layers": list(TAP_LAYERS),
            "loops": ["L1", "L2", "L3", "L4"],
            "feature_configs_supported": list(MATH_CONFIGS),
            "required_candidates": len(required_uids),
            "existing_feature_candidates": len([uid for uid in required_uids if uid in existing_features]),
            "recaptured_candidates": len(recaptured),
            "blocked_missing_candidates": len(blocked),
            "feature_artifact_inventory": inventory,
        },
        "candidate_features": [final_features[uid] for uid in required_uids if uid in final_features],
        "training_pairs_primary": splits.get("training_pairs_primary", []),
        "training_pairs_near_miss_only": splits.get("training_pairs_near_miss_only", []),
        "training_tasks": splits.get("training_tasks", []),
        "eval_sets": splits.get("eval_sets", {}),
        "candidate_index": {uid: candidate_index[uid] for uid in required_uids if uid in candidate_index},
    }
    if verdict != "BLOCKED":
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_payload, output_pt)

    result = {
        "code_specific_feature_verdict": verdict,
        "splits": rel(splits_path),
        "output": rel(output_pt),
        "split_verdict": split_verdict,
        "required_candidates": len(required_uids),
        "existing_feature_candidates": len([uid for uid in required_uids if uid in existing_features]),
        "recaptured_candidates": len(recaptured),
        "blocked_missing_candidates": len(blocked),
        "blocked_candidates": blocked,
        "tap_layers": list(TAP_LAYERS),
        "loops": ["L1", "L2", "L3", "L4"],
        "feature_configs_supported": list(MATH_CONFIGS),
        "feature_artifact_inventory": inventory,
    }
    write_md(output_md, result)
    print(f"CODE_SPECIFIC_FEATURE_VERDICT = {verdict}", flush=True)
    print(f"required_candidates = {len(required_uids)}", flush=True)
    print(f"recaptured_candidates = {len(recaptured)}", flush=True)
    print(f"Wrote {output_pt}", flush=True)
    print(f"Wrote {output_md}", flush=True)
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
