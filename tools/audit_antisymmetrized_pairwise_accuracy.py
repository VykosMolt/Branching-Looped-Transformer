#!/usr/bin/env python3
"""Compute strict antisymmetrized pairwise accuracy from saved HH state packs.

This is a cheap audit only when the saved state pack already exists. It does
not train, generate, or modify checkpoints.

Expected pack format is the one written by
utilities/evaluator/probes/probe_loop_geometry_hh.py:

  {
    "packs": [
      {
        "chosen_states": [Tensor[seq, hidden], ...],
        "chosen_mask": Tensor[seq],
        "rejected_states": [Tensor[seq, hidden], ...],
        "rejected_mask": Tensor[seq],
      },
      ...
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "This audit requires numpy and torch. Run inside the project venv if "
        "system python does not provide them."
    ) from exc

from evaluator_core.pairwise_evaluator import PairwiseEvaluator


DEFAULT_STATES = (
    "artifacts/reports/evaluator/hh_loop_states_200_thinking.pt"
)
DEFAULT_CHECKPOINT = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
DEFAULT_JSON = (
    "artifacts/reports/probes/antisymmetrized_accuracy_audit_2026-06-17.json"
)
DEFAULT_MD = (
    "artifacts/reports/probes/antisymmetrized_accuracy_audit_2026-06-17.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_MD)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_packs(path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "packs" not in payload:
        raise ValueError(f"{path} does not contain a 'packs' list")
    packs = payload["packs"]
    if not isinstance(packs, list) or not packs:
        raise ValueError(f"{path} has no usable packs")
    required = {"chosen_states", "chosen_mask", "rejected_states", "rejected_mask"}
    missing = required.difference(packs[0].keys())
    if missing:
        raise ValueError(f"{path} pack entries missing keys: {sorted(missing)}")
    return packs


def load_head(path: Path, device: torch.device) -> PairwiseEvaluator:
    head = PairwiseEvaluator().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    head.load_state_dict(state_dict)
    head.eval()
    return head


def to_batched_states(states: Iterable[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [
        tensor.to(device=device, dtype=torch.float32).unsqueeze(0)
        for tensor in states
    ]


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def score_pack(head: PairwiseEvaluator, pack: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    chosen_states = to_batched_states(pack["chosen_states"], device)
    rejected_states = to_batched_states(pack["rejected_states"], device)
    chosen_mask = pack["chosen_mask"].to(device=device, dtype=torch.float32).unsqueeze(0)
    rejected_mask = pack["rejected_mask"].to(device=device, dtype=torch.float32).unsqueeze(0)

    normal = head(chosen_states, chosen_mask, rejected_states, rejected_mask)
    flipped = head(rejected_states, rejected_mask, chosen_states, chosen_mask)
    normal_score = float(normal.view(-1).item())
    flipped_score = float(flipped.view(-1).item())
    return {
        "normal_score": normal_score,
        "flipped_score": flipped_score,
        "antisym_score": 0.5 * (normal_score - flipped_score),
        "bias_score": 0.5 * (normal_score + flipped_score),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    normal = np.asarray([r["normal_score"] for r in rows], dtype=np.float64)
    flipped = np.asarray([r["flipped_score"] for r in rows], dtype=np.float64)
    antisym = np.asarray([r["antisym_score"] for r in rows], dtype=np.float64)
    bias = np.asarray([r["bias_score"] for r in rows], dtype=np.float64)
    valid = np.isfinite(normal) & np.isfinite(flipped) & np.isfinite(antisym)
    normal = normal[valid]
    flipped = flipped[valid]
    antisym = antisym[valid]
    bias = bias[valid]
    n = int(normal.size)
    sign_flip = (normal > 0) != (flipped > 0)
    return {
        "n_valid": n,
        "normal_accuracy": float(np.mean(normal > 0)) if n else float("nan"),
        "flipped_accuracy": float(np.mean(flipped < 0)) if n else float("nan"),
        "antisym_acc": float(np.mean(antisym > 0)) if n else float("nan"),
        "strict_sign_flip_rate": float(np.mean(sign_flip)) if n else float("nan"),
        "flip_correlation": finite_corr(normal, flipped),
        "normal_mean": float(np.mean(normal)) if n else float("nan"),
        "flipped_mean": float(np.mean(flipped)) if n else float("nan"),
        "antisym_mean": float(np.mean(antisym)) if n else float("nan"),
        "bias_mean": float(np.mean(bias)) if n else float("nan"),
        "bias_abs_mean": float(np.mean(np.abs(bias))) if n else float("nan"),
        "normal_std": float(np.std(normal)) if n else float("nan"),
        "flipped_std": float(np.std(flipped)) if n else float("nan"),
        "antisym_std": float(np.std(antisym)) if n else float("nan"),
        "bias_std": float(np.std(bias)) if n else float("nan"),
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    s = result["summary"]
    lines = [
        "# Antisymmetrized Pairwise Accuracy Audit",
        "",
        f"States: `{result['states_path']}`",
        f"Checkpoint: `{result['checkpoint_path']}`",
        f"Examples scored: {s['n_valid']}",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| normal_accuracy | {s['normal_accuracy']:.6f} |",
        f"| flipped_accuracy | {s['flipped_accuracy']:.6f} |",
        f"| antisym_acc | {s['antisym_acc']:.6f} |",
        f"| strict_sign_flip_rate | {s['strict_sign_flip_rate']:.6f} |",
        f"| flip_correlation | {s['flip_correlation']:.6f} |",
        f"| normal_mean | {s['normal_mean']:.6f} |",
        f"| flipped_mean | {s['flipped_mean']:.6f} |",
        f"| antisym_mean | {s['antisym_mean']:.6f} |",
        f"| bias_mean | {s['bias_mean']:.6f} |",
        "",
        "Definitions: antisym_score = (normal_score - flipped_score) / 2; "
        "bias_score = (normal_score + flipped_score) / 2.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    states_path = Path(args.states)
    checkpoint_path = Path(args.checkpoint)
    packs = load_packs(states_path)
    if args.max_examples > 0:
        packs = packs[: args.max_examples]

    head = load_head(checkpoint_path, device)
    rows = [score_pack(head, pack, device) for pack in packs]
    result = {
        "states_path": str(states_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "rows_scored": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), result)
    print(f"Wrote {output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
