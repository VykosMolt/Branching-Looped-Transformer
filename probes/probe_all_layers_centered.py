"""Full all-layer evaluator insertion probe.

Captures every decoder layer inside every UT loop and scores the frozen
pairwise evaluator on:

  - natural per-loop sequence at layer L: [h_L_loop1, ..., h_L_loop4]
  - per-loop replicated taps: [h_L_loop_i] * 4
  - both raw layer output and post-final-OuroRMSNorm output
  - boundary natural and boundary per-loop controls

Reports swap-centered/debiased metrics, not just canonical chosen-first
accuracy. This is the insertion-map probe for basal-ganglia-style control.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
NUM_UT_LOOPS = 4
NUM_DECODER_LAYERS = 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--dataset", default="Anthropic/hh-rlhf")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--score-chunk-size", type=int, default=32)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_all_layers_centered.json")
    return p.parse_args()


class AllLayerCapture:
    def __init__(self) -> None:
        self.boundary_states: List[torch.Tensor] = []
        self.layer_states: Dict[int, List[torch.Tensor]] = {i: [] for i in range(NUM_DECODER_LAYERS)}
        self.boundary_validated = False

    def boundary_hook(self, module, inputs, output) -> None:
        hidden_states = output[1]
        if not self.boundary_validated:
            validate_hook_output(hidden_states)
            self.boundary_validated = True
        self.boundary_states = [h.detach() for h in hidden_states]

    def make_layer_hook(self, layer_idx: int):
        def hook(module, inputs, output) -> None:
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            self.layer_states[layer_idx].append(tensor.detach())
        return hook

    def clear(self) -> None:
        self.boundary_states = []
        for i in range(NUM_DECODER_LAYERS):
            self.layer_states[i] = []


def attach_hooks(model, capture: AllLayerCapture) -> List:
    handles = [model.model.register_forward_hook(capture.boundary_hook)]
    for i in range(NUM_DECODER_LAYERS):
        handles.append(model.model.layers[i].register_forward_hook(capture.make_layer_hook(i)))
    return handles


def to_fp32(states: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [h.to(device=device, dtype=torch.float32) for h in states]


def metric_block(normal: Iterable[float], flipped: Iterable[float]) -> Dict[str, float | int]:
    s = np.array(list(normal), dtype=np.float64)
    sf = np.array(list(flipped), dtype=np.float64)
    valid = np.isfinite(s) & np.isfinite(sf)
    s = s[valid]
    sf = sf[valid]
    if s.size == 0:
        return {"n_valid": 0}

    centered = s - sf
    bias = 0.5 * (s + sf)
    antisym = 0.5 * centered
    strict = np.sign(s) != np.sign(sf)
    corr = 0.0
    if np.std(s) > 1e-9 and np.std(sf) > 1e-9:
        corr = float(np.corrcoef(s, -sf)[0, 1])

    return {
        "n_valid": int(s.size),
        "canonical_acc": float(np.mean(s > 0)),
        "flipped_acc": float(np.mean(sf < 0)),
        "normal_pos_rate": float(np.mean(s > 0)),
        "flipped_pos_rate": float(np.mean(sf > 0)),
        "strict_sign_reversal_rate": float(np.mean(strict)),
        "same_sign_rate": float(1.0 - np.mean(strict)),
        "antisym_corr": corr,
        "centered_acc": float(np.mean(centered > 0)),
        "centered_mean": float(np.mean(centered)),
        "centered_std": float(np.std(centered)),
        "bias_mean": float(np.mean(bias)),
        "bias_std": float(np.std(bias)),
        "antisym_mean": float(np.mean(antisym)),
        "antisym_std": float(np.std(antisym)),
        "bias_to_signal": float(abs(np.mean(bias)) / (abs(np.mean(antisym)) + 1e-8)),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        "flipped_mean": float(np.mean(sf)),
        "flipped_std": float(np.std(sf)),
    }


def build_specs() -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    specs: List[Dict[str, object]] = []

    def add(name: str, **meta: object) -> None:
        item = {"config": name}
        item.update(meta)
        specs.append(item)

    add("boundary_natural", family="boundary_natural", layer=None, loop=None, normed=True, tap_kind="boundary")
    for loop_idx in range(NUM_UT_LOOPS):
        add(
            f"boundary_loop_{loop_idx+1}_replicated",
            family="boundary_loop",
            layer=None,
            loop=loop_idx,
            normed=True,
            tap_kind="boundary_loop",
        )

    for layer in range(NUM_DECODER_LAYERS):
        for normed in (False, True):
            suffix = "normed" if normed else "raw"
            add(
                f"layer_{layer:02d}_natural_{suffix}",
                family="layer_natural",
                layer=layer,
                loop=None,
                normed=normed,
                tap_kind="layer_natural",
            )
            for loop_idx in range(NUM_UT_LOOPS):
                add(
                    f"layer_{layer:02d}_loop_{loop_idx+1}_{suffix}",
                    family="layer_loop",
                    layer=layer,
                    loop=loop_idx,
                    normed=normed,
                    tap_kind="layer_loop",
                )

    return specs, {str(item["config"]): item for item in specs}


def get_states(spec: Dict[str, object], data: Dict[str, object]) -> List[torch.Tensor]:
    family = str(spec["family"])
    if family == "boundary_natural":
        return list(data["boundary"])
    if family == "boundary_loop":
        loop_idx = int(spec["loop"])
        return [data["boundary"][loop_idx]] * NUM_UT_LOOPS

    layer = int(spec["layer"])
    table = data["layers_normed"] if bool(spec["normed"]) else data["layers_raw"]
    if family == "layer_natural":
        return list(table[layer])
    if family == "layer_loop":
        loop_idx = int(spec["loop"])
        return [table[layer][loop_idx]] * NUM_UT_LOOPS
    raise ValueError(f"unknown family: {family}")


@torch.no_grad()
def score_specs(
    evaluator: PairwiseEvaluator,
    specs: List[Dict[str, object]],
    c_data: Dict[str, object],
    c_mask: torch.Tensor,
    r_data: Dict[str, object],
    r_mask: torch.Tensor,
    chunk_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    normal: Dict[str, float] = {}
    flipped: Dict[str, float] = {}

    for start in range(0, len(specs), chunk_size):
        chunk = specs[start:start + chunk_size]
        c_slots: List[List[torch.Tensor]] = [[] for _ in range(NUM_UT_LOOPS)]
        r_slots: List[List[torch.Tensor]] = [[] for _ in range(NUM_UT_LOOPS)]

        for spec in chunk:
            cs = get_states(spec, c_data)
            rs = get_states(spec, r_data)
            for slot in range(NUM_UT_LOOPS):
                c_slots[slot].append(cs[slot])
                r_slots[slot].append(rs[slot])

        c_batch = [torch.cat(c_slots[slot], dim=0) for slot in range(NUM_UT_LOOPS)]
        r_batch = [torch.cat(r_slots[slot], dim=0) for slot in range(NUM_UT_LOOPS)]
        c_mask_batch = c_mask.expand(len(chunk), -1).contiguous()
        r_mask_batch = r_mask.expand(len(chunk), -1).contiguous()

        s = evaluator(c_batch, c_mask_batch, r_batch, r_mask_batch).view(-1).detach().float().cpu().numpy()
        sf = evaluator(r_batch, r_mask_batch, c_batch, c_mask_batch).view(-1).detach().float().cpu().numpy()

        for offset, spec in enumerate(chunk):
            name = str(spec["config"])
            sv = float(s[offset])
            sfv = float(sf[offset])
            normal[name] = sv if math.isfinite(sv) else float("nan")
            flipped[name] = sfv if math.isfinite(sfv) else float("nan")

    return normal, flipped


def summarize_top(summary: Dict[str, Dict[str, float | int]], predicate, key: str, n: int = 12) -> List[str]:
    rows = [(name, metrics) for name, metrics in summary.items() if predicate(name, metrics)]
    rows.sort(key=lambda item: (-float(item[1][key]), -float(item[1]["flipped_acc"])))
    return [name for name, _ in rows[:n]]


def print_rows(title: str, names: List[str], summary: Dict[str, Dict[str, float | int]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'config':>34} | {'canon':>5} | {'center':>6} | {'flip':>5} | {'strict':>6} | {'corr':>6} | {'bias/sig':>8}")
    for name in names:
        row = summary[name]
        print(
            f"  {name:>32} | {row['canonical_acc']:.3f} | {row['centered_acc']:.3f} | "
            f"{row['flipped_acc']:.3f} | {row['strict_sign_reversal_rate']:.3f} | "
            f"{row['antisym_corr']:+.3f} | {row['bias_to_signal']:.2f}"
        )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Examples: {args.max_examples}, max_length={args.max_length}, device={device}")
    print(f"Layers: 0-{NUM_DECODER_LAYERS - 1}, chunk_size={args.score_chunk_size}\n")

    specs, spec_meta = build_specs()
    print(f"Configs per pair: {len(specs)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": str(device)},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = args.early_exit_threshold

    @torch.no_grad()
    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return model.model.norm(x.to(dtype=torch.bfloat16)).to(dtype=torch.float32)

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    capture = AllLayerCapture()
    handles = attach_hooks(model, capture)

    print("Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    n_total = len(ds)
    print(f"Examples loaded: {n_total}\n")

    raw_scores: Dict[str, List[float]] = {str(spec["config"]): [] for spec in specs}
    raw_scores_flipped: Dict[str, List[float]] = {str(spec["config"]): [] for spec in specs}

    for idx in range(n_total):
        tc = tokenizer(ds[idx]["chosen"], return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        tr = tokenizer(ds[idx]["rejected"], return_tensors="pt", truncation=True, max_length=args.max_length).to(device)

        capture.clear()
        with torch.no_grad():
            model(**tc, use_cache=False)
        c_boundary = to_fp32(capture.boundary_states, device)
        c_layers_raw = {L: to_fp32(capture.layer_states[L], device) for L in range(NUM_DECODER_LAYERS)}

        capture.clear()
        with torch.no_grad():
            model(**tr, use_cache=False)
        r_boundary = to_fp32(capture.boundary_states, device)
        r_layers_raw = {L: to_fp32(capture.layer_states[L], device) for L in range(NUM_DECODER_LAYERS)}

        for L in range(NUM_DECODER_LAYERS):
            if len(c_layers_raw[L]) != NUM_UT_LOOPS or len(r_layers_raw[L]) != NUM_UT_LOOPS:
                raise RuntimeError(f"Layer {L} captured wrong loop count")
        if len(c_boundary) != NUM_UT_LOOPS or len(r_boundary) != NUM_UT_LOOPS:
            raise RuntimeError("Boundary capture wrong loop count")

        c_layers_normed = {L: [norm_fn(h) for h in c_layers_raw[L]] for L in range(NUM_DECODER_LAYERS)}
        r_layers_normed = {L: [norm_fn(h) for h in r_layers_raw[L]] for L in range(NUM_DECODER_LAYERS)}
        c_data = {"boundary": c_boundary, "layers_raw": c_layers_raw, "layers_normed": c_layers_normed}
        r_data = {"boundary": r_boundary, "layers_raw": r_layers_raw, "layers_normed": r_layers_normed}

        normal, flipped = score_specs(
            evaluator,
            specs,
            c_data,
            tc["attention_mask"],
            r_data,
            tr["attention_mask"],
            args.score_chunk_size,
        )
        for name in raw_scores:
            raw_scores[name].append(normal[name])
            raw_scores_flipped[name].append(flipped[name])

        if (idx + 1) % args.report_every == 0 or idx == n_total - 1:
            n_so_far = idx + 1
            preview = []
            for name in (
                "boundary_natural",
                "layer_24_natural_normed",
                "layer_36_natural_normed",
                "layer_44_natural_normed",
                "layer_24_loop_2_normed",
                "layer_36_loop_2_normed",
            ):
                acc = sum(1 for x in raw_scores[name] if x > 0) / n_so_far
                preview.append(f"{name}={acc:.3f}")
            print(f"[{n_so_far}/{n_total}] " + "  ".join(preview))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()

    summary = {
        name: metric_block(raw_scores[name], raw_scores_flipped[name])
        for name in raw_scores
    }

    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "examples_evaluated": n_total,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "num_ut_loops": NUM_UT_LOOPS,
        "metrics_definition": {
            "canonical_acc": "mean(score(chosen, rejected) > 0)",
            "flipped_acc": "mean(score(rejected, chosen) < 0)",
            "centered_acc": "mean(score(chosen, rejected) - score(rejected, chosen) > 0)",
            "strict_sign_reversal_rate": "mean(sign(score_normal) != sign(score_flipped))",
            "bias_to_signal": "abs(mean(0.5*(s+sf))) / abs(mean(0.5*(s-sf)))",
        },
        "config_metadata": spec_meta,
        "per_config": summary,
        "raw_scores": {k: [float(x) for x in v] for k, v in raw_scores.items()},
        "raw_scores_flipped": {k: [float(x) for x in v] for k, v in raw_scores_flipped.items()},
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print_rows(
        "Top natural layer taps by centered_acc",
        summarize_top(summary, lambda n, _: "_natural_" in n and n.startswith("layer_"), "centered_acc"),
        summary,
    )
    print_rows(
        "Top per-loop layer taps by centered_acc",
        summarize_top(summary, lambda n, _: "_loop_" in n and n.startswith("layer_"), "centered_acc"),
        summary,
    )
    print_rows(
        "Top natural layer taps by canonical_acc",
        summarize_top(summary, lambda n, _: "_natural_" in n and n.startswith("layer_"), "canonical_acc"),
        summary,
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
