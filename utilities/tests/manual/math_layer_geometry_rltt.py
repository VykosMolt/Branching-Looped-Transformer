"""Math layer-geometry probe for Ouro-RLTT.

Measures hidden-state loop geometry at layers 24, 36, and 47 on math text.
No evaluator checkpoint is loaded. The goal is to check whether the HH result
generalizes: layers 24/36 converged, layer 47 bipartite.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RLTT_PATH,
    DEFAULT_TOKENIZER_PATH,
    NUM_LOOPS,
    TAP_LAYERS,
    capture_pooled_taps,
    cosine,
    gold_solution_text,
    load_math_problems,
    output_path,
    perturb_answer,
    resolve_local,
    stats,
    wrong_answer_text,
)

LOOP_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_RLTT_PATH)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--source", choices=("gsm8k", "math", "mixed"), default="mixed")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--min-math-level", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--output-json", default=f"{DEFAULT_OUTPUT_DIR}/math_layer_geometry_rltt.json")
    p.add_argument("--output-md", default=f"{DEFAULT_OUTPUT_DIR}/math_layer_geometry_rltt.md")
    p.add_argument("--output-pt", default=f"{DEFAULT_OUTPUT_DIR}/math_layer_geometry_rltt_features.pt")
    return p.parse_args()


def layer_verdict(summary: Dict[str, float]) -> str:
    l1l4 = summary["L1_L4_cos"]
    l2l4 = summary["L2_L4_cos"]
    off = summary["mean_off_diag_cos"]
    min_off = summary["min_off_diag_cos"]
    if off < 0.40:
        return "fully distributed"
    if min_off > 0.90:
        return "fully converged"
    if l1l4 < 0.85 and l2l4 > 0.90:
        return "bipartite"
    return "intermediate / unclear"


def analyse(features: torch.Tensor) -> Dict[int, Dict[str, object]]:
    # features: [N, 2, layers=3, loops=4, H], side 0=gold, side 1=wrong
    out: Dict[int, Dict[str, object]] = {}
    for layer_pos, layer in enumerate(TAP_LAYERS):
        pair_cos: Dict[str, Dict[str, float | int]] = {}
        for i, j in LOOP_PAIRS:
            vals = []
            for n in range(features.shape[0]):
                vals.append(cosine(features[n, 0, layer_pos, i - 1], features[n, 0, layer_pos, j - 1]))
                vals.append(cosine(features[n, 1, layer_pos, i - 1], features[n, 1, layer_pos, j - 1]))
            pair_cos[f"L{i}_L{j}"] = stats(vals)

        norm_per_loop = {}
        for loop in range(NUM_LOOPS):
            vals = []
            for n in range(features.shape[0]):
                vals.append(float(features[n, 0, layer_pos, loop].norm()))
                vals.append(float(features[n, 1, layer_pos, loop].norm()))
            norm_per_loop[f"L{loop+1}"] = stats(vals)

        diff_cos = {}
        for i, j in LOOP_PAIRS:
            vals = []
            for n in range(features.shape[0]):
                di = features[n, 0, layer_pos, i - 1] - features[n, 1, layer_pos, i - 1]
                dj = features[n, 0, layer_pos, j - 1] - features[n, 1, layer_pos, j - 1]
                vals.append(cosine(di, dj))
            diff_cos[f"d{i}_d{j}"] = stats(vals)

        means = [float(pair_cos[f"L{i}_L{j}"]["mean"]) for i, j in LOOP_PAIRS]
        summary = {
            "L1_L4_cos": float(pair_cos["L1_L4"]["mean"]),
            "L2_L4_cos": float(pair_cos["L2_L4"]["mean"]),
            "mean_off_diag_cos": float(np.mean(means)),
            "min_off_diag_cos": float(np.min(means)),
        }
        summary["verdict"] = layer_verdict(summary)
        out[layer] = {
            "pair_cos": pair_cos,
            "norm_per_loop": norm_per_loop,
            "diff_vector_cross_loop_cos": diff_cos,
            "summary": summary,
        }
    return out


def write_md(path: Path, result: Dict[str, object]) -> None:
    rows = result["per_layer"]
    lines = [
        "# Math Layer Geometry on Ouro-RLTT",
        "",
        "No evaluator checkpoint was loaded. Geometry is measured on paired math texts:",
        "gold/reference solution text and a deterministic wrong-answer control.",
        "",
        "| Layer | L1-L4 cos | L2-L4 cos | mean off-diag | min off-diag | Verdict |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for layer in TAP_LAYERS:
        s = rows[str(layer)]["summary"]
        lines.append(
            f"| {layer} | {s['L1_L4_cos']:+.4f} | {s['L2_L4_cos']:+.4f} | "
            f"{s['mean_off_diag_cos']:+.4f} | {s['min_off_diag_cos']:+.4f} | "
            f"{s['verdict']} |"
        )
    lines.extend([
        "",
        "Architecture check:",
        f"- {result['architecture_verdict']}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def architecture_verdict(per_layer: Dict[int, Dict[str, object]]) -> str:
    v = {layer: per_layer[layer]["summary"]["verdict"] for layer in TAP_LAYERS}
    if v[24] == "fully converged" and v[36] == "fully converged" and v[47] == "bipartite":
        return "math matches HH geometry: 24/36 converged, 47 bipartite. Heterogeneous BG tap interface remains protected."
    return (
        "math geometry differs from the HH blocker result. Review before treating "
        "the heterogeneous tap interface as domain-stable."
    )


def main() -> None:
    args = parse_args()
    rng = __import__("random").Random(args.seed)
    problems, source_meta = load_math_problems(
        args.source, args.max_examples, args.seed, args.min_math_level)
    examples = []
    skipped = 0
    for problem in problems:
        wrong = perturb_answer(problem.gold_answer, rng)
        if wrong is None:
            skipped += 1
            continue
        examples.append({
            "problem": problem,
            "gold_text": gold_solution_text(problem),
            "wrong_text": wrong_answer_text(problem, wrong),
            "wrong_answer": wrong,
        })
        if len(examples) >= args.max_examples:
            break
    if not examples:
        raise SystemExit("No verifier-clean math examples available for geometry probe.")

    texts: List[str] = []
    for ex in examples:
        texts.extend([ex["gold_text"], ex["wrong_text"]])

    device = torch.device(args.device)
    model_path = resolve_local(args.model_path)
    tokenizer_path = resolve_local(args.tokenizer_path)

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading RLTT model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0

    flat_features = capture_pooled_taps(
        model, tokenizer, texts, args.max_length, device, args.report_every)
    features = flat_features.view(len(examples), 2, len(TAP_LAYERS), NUM_LOOPS, -1).contiguous()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_layer_int = analyse(features)
    verdict = architecture_verdict(per_layer_int)
    per_layer = {str(k): v for k, v in per_layer_int.items()}

    meta = {
        "args": vars(args),
        "model_path_resolved": model_path,
        "tokenizer_path_resolved": tokenizer_path,
        "source_meta": source_meta,
        "examples_used": len(examples),
        "examples_skipped": skipped,
        "feature_shape": list(features.shape),
        "tap_layers": list(TAP_LAYERS),
        "layer_24_36_convention": "post-block raw residual",
        "layer_47_convention": "post-final-norm boundary output[1]",
    }
    result = {
        "meta": meta,
        "per_layer": per_layer,
        "architecture_verdict": verdict,
    }

    out_json = output_path(args.output_json)
    out_md = output_path(args.output_md)
    out_pt = output_path(args.output_pt)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    torch.save({
        "meta": meta,
        "features": features,
        "examples": [
            {
                "source": ex["problem"].source,
                "dataset_index": ex["problem"].dataset_index,
                "question": ex["problem"].question,
                "gold_answer": ex["problem"].gold_answer,
                "wrong_answer": ex["wrong_answer"],
            }
            for ex in examples
        ],
    }, out_pt)
    write_md(out_md, result)

    print("\n=== Math geometry summary ===")
    for layer in TAP_LAYERS:
        s = per_layer[str(layer)]["summary"]
        print(
            f"Layer {layer}: L1-L4={s['L1_L4_cos']:+.4f} "
            f"L2-L4={s['L2_L4_cos']:+.4f} "
            f"mean_off={s['mean_off_diag_cos']:+.4f} verdict={s['verdict']}"
        )
    print(f"\nArchitecture verdict: {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_pt}")


if __name__ == "__main__":
    main()
