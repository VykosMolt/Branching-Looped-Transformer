"""Run the paper's pairwise flip diagnostic against local RLTT loop states."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator, validate_hook_output


DEFAULT_MODEL_PATH = "models/ouro_rltt_local"
DEFAULT_CHECKPOINT_PATH = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flip-test the epoch-2 pairwise evaluator on local RLTT loop states."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--early-exit-threshold", type=float, default=0.87)
    parser.add_argument("--expected-mean-sum", type=float, default=2.51)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


class LoopStateCapture:
    def __init__(self) -> None:
        self.captured: Dict[str, List[torch.Tensor]] = {}
        self.validated = False

    def hook_fn(self, module, inputs, output) -> None:  # type: ignore[no-untyped-def]
        hidden_states = output[1]
        if not self.validated:
            validate_hook_output(hidden_states)
            self.validated = True
        self.captured["hidden_states_list"] = [h.detach() for h in hidden_states]

    @torch.no_grad()
    def get_all_hidden_states(
        self,
        model: torch.nn.Module,
        tokens: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        self.captured.clear()
        model(**tokens, use_cache=False)
        hidden_states_list = self.captured["hidden_states_list"]
        device = tokens["input_ids"].device
        raw_list = [h.to(device=device, dtype=torch.float32) for h in hidden_states_list]
        return raw_list, tokens["attention_mask"]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def load_eval_split(dataset_name: str, split: str, n_samples: int):
    ds = load_dataset(dataset_name, split=split)
    return ds.select(range(min(n_samples, len(ds))))


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / ".hf_modules_cache"))

    device = resolve_device(args.device)
    model_path = str(Path(args.model_path))
    checkpoint_path = str(Path(args.checkpoint))

    print(f"Transformers model: {model_path}")
    print(f"Evaluator checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Samples: {args.n_samples}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": str(device)},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.config.early_exit_threshold = args.early_exit_threshold

    capture = LoopStateCapture()
    handle = model.model.register_forward_hook(capture.hook_fn)

    evaluator = PairwiseEvaluator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    evaluator.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    evaluator.eval()

    ds = load_eval_split(args.dataset, args.split, args.n_samples)
    normal_scores: List[float] = []
    flipped_scores: List[float] = []

    print("\nFlip test")
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        tokens_c = tokenizer(
            batch["chosen"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        tokens_r = tokenizer(
            batch["rejected"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        hidden_c, mask_c = capture.get_all_hidden_states(model, tokens_c)
        hidden_r, mask_r = capture.get_all_hidden_states(model, tokens_r)

        with torch.no_grad():
            score_normal = evaluator(hidden_c, mask_c, hidden_r, mask_r).view(-1)
            score_flipped = evaluator(hidden_r, mask_r, hidden_c, mask_c).view(-1)

        for n_score, f_score in zip(score_normal.tolist(), score_flipped.tolist()):
            normal_scores.append(float(n_score))
            flipped_scores.append(float(f_score))
            idx = len(normal_scores)
            if idx <= 20:
                print(f"  Example {idx}: Normal={n_score:+.4f} | Flipped={f_score:+.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(normal_scores) % 25 == 0 or len(normal_scores) == len(ds):
            print(f"  Processed {len(normal_scores)}/{len(ds)}")

    handle.remove()

    normal_arr = np.array(normal_scores, dtype=np.float32)
    flipped_arr = np.array(flipped_scores, dtype=np.float32)
    sign_flips = int(np.sum((normal_arr > 0) != (flipped_arr > 0)))
    corr = float(np.corrcoef(normal_arr, flipped_arr)[0, 1])
    mean_sum = float(np.mean(normal_arr + flipped_arr))
    result = {
        "model_path": model_path,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "samples": int(len(normal_arr)),
        "normal_mean": float(np.mean(normal_arr)),
        "normal_std": float(np.std(normal_arr)),
        "flipped_mean": float(np.mean(flipped_arr)),
        "flipped_std": float(np.std(flipped_arr)),
        "sign_flips": sign_flips,
        "sign_flip_rate": float(sign_flips / max(len(normal_arr), 1)),
        "antisymmetry_correlation": corr,
        "mean_sum": mean_sum,
        "expected_mean_sum": args.expected_mean_sum,
        "mean_sum_delta_from_expected": float(mean_sum - args.expected_mean_sum),
        "normal_min": float(np.min(normal_arr)),
        "normal_max": float(np.max(normal_arr)),
        "flipped_min": float(np.min(flipped_arr)),
        "flipped_max": float(np.max(flipped_arr)),
        "device": str(device),
    }

    print("\nFlip test results")
    print(f"Normal mean:              {result['normal_mean']:+.4f}")
    print(f"Flipped mean:             {result['flipped_mean']:+.4f}")
    print(f"Sign flips:               {sign_flips}/{len(normal_arr)} ({result['sign_flip_rate'] * 100:.1f}%)")
    print(f"Antisymmetry correlation: {corr:+.4f}")
    print(f"Mean normal + flipped:    {mean_sum:+.4f}")
    print(f"Delta from +{args.expected_mean_sum:.2f}:       {result['mean_sum_delta_from_expected']:+.4f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
