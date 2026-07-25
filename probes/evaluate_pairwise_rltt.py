"""Run the original HH-RLHF pairwise evaluator against a local Ouro/RLTT model.

This is the archived ``evaluate_pairwise.py`` path with arguments added for
local model/checkpoint selection and capped sanity runs. The scoring path is:

    HH-RLHF chosen/rejected text -> Ouro loop states -> PairwiseEvaluator score

Positive score means the evaluator prefers the chosen response.
"""
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
        description="Evaluate epoch-2 pairwise CLT weights on local RLTT/Ouro loop states."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--detail-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--early-exit-threshold", type=float, default=0.87)
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
        attention_mask = tokens["attention_mask"]
        return raw_list, attention_mask


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def load_text_dataset(dataset_name: str, split: str, max_examples: int):
    ds = load_dataset(dataset_name, split=split)
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


def maybe_limit_detail_samples(detail_samples: int, total: int) -> int:
    return max(0, min(detail_samples, total))


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault(
        "HF_MODULES_CACHE",
        str(Path.cwd() / ".hf_modules_cache"),
    )

    device = resolve_device(args.device)
    model_path = str(Path(args.model_path))
    checkpoint_path = str(Path(args.checkpoint))

    print(f"Transformers model: {model_path}")
    print(f"Evaluator checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Max examples: {args.max_examples if args.max_examples > 0 else 'full split'}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
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
    if hasattr(model, "config"):
        model.config.early_exit_threshold = args.early_exit_threshold

    capture = LoopStateCapture()
    hook_handle = model.model.register_forward_hook(capture.hook_fn)

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    print("Loading HH-RLHF split...")
    ds = load_text_dataset(args.dataset, args.split, args.max_examples)
    print(f"Evaluation examples: {len(ds)}")

    correct = 0
    total = 0
    scores_list: List[float] = []

    detail_count = maybe_limit_detail_samples(args.detail_samples, len(ds))
    if detail_count:
        print(f"\nPer-example analysis (first {detail_count} examples)")
    for idx in range(detail_count):
        tokens_chosen = tokenizer(
            ds[idx]["chosen"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        tokens_rejected = tokenizer(
            ds[idx]["rejected"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        hidden_c, mask_c = capture.get_all_hidden_states(model, tokens_chosen)
        hidden_r, mask_r = capture.get_all_hidden_states(model, tokens_rejected)

        with torch.no_grad():
            score = evaluator(hidden_c, mask_c, hidden_r, mask_r).item()

        result = "OK" if score > 0 else "MISS"
        print(f"[{result}] Example {idx + 1} | Score: {score:.4f}")

    print("\nBatched evaluation")
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]

        tokens_chosen = tokenizer(
            chosen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        tokens_rejected = tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        hidden_c, mask_c = capture.get_all_hidden_states(model, tokens_chosen)
        hidden_r, mask_r = capture.get_all_hidden_states(model, tokens_rejected)

        with torch.no_grad():
            scores = evaluator(hidden_c, mask_c, hidden_r, mask_r)

        batch_scores = [float(s) for s in scores.view(-1).tolist()]
        scores_list.extend(batch_scores)
        correct += sum(1 for s in batch_scores if s > 0)
        total += len(batch_scores)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if total == len(ds) or total % max(args.batch_size, 50) == 0:
            acc = correct / max(total, 1)
            avg_score = float(np.mean(scores_list)) if scores_list else 0.0
            print(f"Progress: {total}/{len(ds)} | Acc: {acc:.4f} | Avg score: {avg_score:.4f}")

    hook_handle.remove()

    scores_arr = np.array(scores_list, dtype=np.float32)
    final_acc = correct / max(total, 1)
    result = {
        "model_path": model_path,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "max_examples": args.max_examples,
        "examples_evaluated": total,
        "accuracy": final_acc,
        "average_score": float(np.mean(scores_arr)) if total else 0.0,
        "score_std": float(np.std(scores_arr)) if total else 0.0,
        "min_score": float(np.min(scores_arr)) if total else 0.0,
        "max_score": float(np.max(scores_arr)) if total else 0.0,
        "positive_rate": float(np.mean(scores_arr > 0)) if total else 0.0,
        "early_exit_threshold": args.early_exit_threshold,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
    }

    print("\nFinal results")
    print(f"Test examples:       {total}")
    print(f"Accuracy:            {final_acc:.4f} ({final_acc * 100:.1f}%)")
    print(f"Average score:       {result['average_score']:.4f}")
    print(f"Score std:           {result['score_std']:.4f}")
    print(f"Min score:           {result['min_score']:.4f}")
    print(f"Max score:           {result['max_score']:.4f}")
    print(f"Positive rate:       {result['positive_rate'] * 100:.1f}%")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
