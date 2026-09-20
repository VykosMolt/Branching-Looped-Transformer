"""Run the standard post-RLTT acquisition diagnostic probe bundle.

The bundle is intentionally measurement-only. It does not train, tune, or
change checkpoint state. Each child script writes its own JSON artifact; this
wrapper writes a manifest with command lines, return codes, and model metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = PROJECT_ROOT / "shared/venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-RLTT diagnostic probe bundle.")
    parser.add_argument("--model-path", default="shared/models/ouro_rltt_local")
    parser.add_argument("--model-label", default="rltt")
    parser.add_argument("--base-model-path", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--pairwise-checkpoint", default="rpe/checkpoints/evaluator/pairwise_epoch2.pt")
    parser.add_argument("--encoder-checkpoint", default="artifacts/checkpoints/running/sprint4_encoder_reverted.pt")
    parser.add_argument("--encoder-label", default="")
    parser.add_argument("--output-dir", default="rpe/evaluator/post_rltt_probe_bundle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--flip-samples", type=int, default=100)
    parser.add_argument("--spatial-samples", type=int, default=100)
    parser.add_argument("--anti-saturation-samples", type=int, default=80)
    parser.add_argument("--arc-samples", type=int, default=100)
    parser.add_argument("--grid-samples", type=int, default=32)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--skip-evaluator", action="store_true")
    parser.add_argument("--skip-base-compare", action="store_true")
    parser.add_argument("--hash-model-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def path_arg(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def path_metadata(path: str, hash_files: bool = False) -> Dict[str, Any]:
    p = Path(path_arg(path))
    info: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return info
    if p.is_file():
        info["size_bytes"] = int(p.stat().st_size)
        if hash_files:
            info["sha256"] = file_sha256(p)
        return info
    files = sorted(child for child in p.iterdir() if child.is_file())
    info["files"] = []
    total = 0
    manifest_digest = hashlib.sha256()
    for child in files:
        size = int(child.stat().st_size)
        total += size
        row: Dict[str, Any] = {"name": child.name, "size_bytes": size}
        if hash_files:
            row["sha256"] = file_sha256(child)
        manifest_digest.update(child.name.encode("utf-8"))
        manifest_digest.update(str(size).encode("utf-8"))
        info["files"].append(row)
    info["total_size_bytes"] = int(total)
    info["size_manifest_sha256"] = manifest_digest.hexdigest()
    return info


def command_specs(args: argparse.Namespace, output_dir: Path) -> List[List[str]]:
    py = str(PYTHON if PYTHON.exists() else Path(sys.executable))
    model_path = path_arg(args.model_path)
    base_model_path = args.base_model_path
    pairwise = path_arg(args.pairwise_checkpoint)
    encoder = path_arg(args.encoder_checkpoint)
    offline_flag = ["--offline"] if args.offline else []
    allow_remote_flag = ["--allow-remote"] if args.allow_remote else []

    commands: List[List[str]] = []
    if not args.skip_evaluator:
        commands.append([
            py,
            "shared/utilities/evaluator/probes/evaluate_pairwise_rltt.py",
            "--model-path", model_path,
            "--checkpoint", pairwise,
            "--max-examples", str(args.eval_examples),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--output-json", str(output_dir / f"evaluate_pairwise_{args.model_label}.json"),
            *offline_flag,
        ])
        commands.append([
            py,
            "shared/utilities/evaluator/probes/flip_pairwise_rltt.py",
            "--model-path", model_path,
            "--checkpoint", pairwise,
            "--n-samples", str(args.flip_samples),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--output-json", str(output_dir / f"flip_pairwise_{args.model_label}.json"),
            *offline_flag,
        ])
    commands.append([
        py,
        "shared/utilities/evaluator/probes/probe_spatial_rltt.py",
        "--model-path", model_path,
        "--grid-samples", str(args.spatial_samples),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--output-json", str(output_dir / f"probe_spatial_{args.model_label}.json"),
        *offline_flag,
    ])
    commands.append([
        py,
        "shared/utilities/evaluator/probes/probe_spatial_antisaturation.py",
        "--model-path", model_path,
        "--model-label", args.model_label,
        "--samples-per-task", str(args.anti_saturation_samples),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--output-json", str(output_dir / f"probe_spatial_antisaturation_{args.model_label}.json"),
        *offline_flag,
    ])
    commands.append([
        py,
        "shared/utilities/evaluator/probes/probe_arc_candidate_separate.py",
        "--model-path", model_path,
        "--model-label", args.model_label,
        "--samples-per-task", str(args.arc_samples),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--output", str(output_dir / f"probe_arc_candidate_separate_{args.model_label}.json"),
        *allow_remote_flag,
    ])
    model_paths = model_path
    model_labels = args.model_label
    if not args.skip_base_compare:
        model_paths = f"{base_model_path},{model_path}"
        model_labels = f"{args.base_label},{args.model_label}"
    commands.append([
        py,
        "shared/utilities/evaluator/probes/probe_gridencoder_loop_signature.py",
        "--encoder-checkpoint", encoder,
        "--encoder-label", args.encoder_label or Path(args.encoder_checkpoint).stem,
        "--model-paths", model_paths,
        "--model-labels", model_labels,
        "--samples", str(args.grid_samples),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--output", str(output_dir / f"probe_gridencoder_loop_signature_{args.model_label}.json"),
        *allow_remote_flag,
    ])
    return commands


def run_command(
    cmd: Sequence[str],
    env: Dict[str, str],
    dry_run: bool,
    log_path: Path,
) -> Dict[str, Any]:
    started = time.time()
    print("RUN", " ".join(cmd), flush=True)
    if dry_run:
        return {
            "cmd": list(cmd),
            "returncode": None,
            "duration_sec": 0.0,
            "dry_run": True,
            "log_path": str(log_path),
        }
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    log_text = "\n".join(
        [
            "$ " + " ".join(cmd),
            f"returncode={int(proc.returncode)}",
            "",
            "## stdout",
            proc.stdout or "",
            "",
            "## stderr",
            proc.stderr or "",
        ]
    )
    log_path.write_text(log_text, encoding="utf-8")
    combined_tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:]
    if combined_tail.strip():
        print(combined_tail, flush=True)
    return {
        "cmd": list(cmd),
        "returncode": int(proc.returncode),
        "duration_sec": float(time.time() - started),
        "dry_run": False,
        "log_path": str(log_path),
        "log_tail": combined_tail,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(path_arg(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / ".hf_modules_cache"))
    if args.offline:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")

    manifest: Dict[str, Any] = {
        "created_at_unix": float(time.time()),
        "project_root": str(PROJECT_ROOT),
        "model": path_metadata(args.model_path, hash_files=args.hash_model_files),
        "pairwise_checkpoint": path_metadata(
            args.pairwise_checkpoint,
            hash_files=args.hash_model_files,
        ),
        "encoder_checkpoint": path_metadata(
            args.encoder_checkpoint,
            hash_files=args.hash_model_files,
        ),
        "commands": [],
    }
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for index, cmd in enumerate(command_specs(args, output_dir), start=1):
        script_name = Path(cmd[1]).stem if len(cmd) > 1 else "command"
        log_path = logs_dir / f"{index:02d}_{script_name}.log"
        result = run_command(
            cmd,
            env=env,
            dry_run=bool(args.dry_run),
            log_path=log_path,
        )
        manifest["commands"].append(result)
        if result["returncode"] not in (0, None):
            break

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    failed = [row for row in manifest["commands"] if row.get("returncode") not in (0, None)]
    if failed:
        raise SystemExit(int(failed[0]["returncode"]))


if __name__ == "__main__":
    main()
