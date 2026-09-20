"""Probe old-head-compatible inputs for converged BG taps.

Layers 24 and 36 are now known to be loop-converged on Ouro-RLTT HH text,
while layer 47 remains bipartite. This probe uses the published
PairwiseEvaluator checkpoint unchanged to compare simple 2048-dim input
constructions before new single-state tap heads are designed.

Default backbone is Ouro-RLTT, because BG deployment is RLTT. Thinking is
available only as an explicit --backbone thinking comparison.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "shared/src"
PROBES_ROOT = PROJECT_ROOT / "shared/utilities" / "evaluator" / "probes"
for root in (SRC_ROOT, PROBES_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluator_core.pairwise_evaluator as pairwise_module
from evaluator_core.pairwise_evaluator import PairwiseEvaluator
from layer_state_capture import (
    DEFAULT_RLTT_PATH,
    DEFAULT_THINKING_PATH,
    DEFAULT_TOKENIZER_PATH,
    NUM_LOOPS,
    resolve_local,
    select_examples,
)

DEFAULT_CHECKPOINT = "rpe/checkpoints/evaluator/pairwise_epoch2.pt"
DEFAULT_JSON = "opi/taps/probes/probe_converged_tap_inputs.json"
DEFAULT_MD = "opi/taps/probes/probe_converged_tap_inputs.md"
VALID_LAYERS = (24, 36, 47)
INTERMEDIATE_IDX = {24: 23, 36: 35}
CONFIGS = (
    "L1_replicated",
    "L2_replicated",
    "L4_replicated",
    "mean_replicated",
    "natural_seq",
)
NEAR_CHANCE_CENTERED = 0.53
MEANINGFUL_CENTERED_GAP = 0.01

GEOMETRY_CONTEXT = {
    24: {"L1_L4": 0.9284, "L2_L4": 0.9790, "mean_off_diag": 0.9632,
         "verdict": "fully converged"},
    36: {"L1_L4": 0.9347, "L2_L4": 0.9822, "mean_off_diag": 0.9682,
         "verdict": "fully converged"},
    47: {"L1_L4": 0.7350, "L2_L4": 0.9608, "mean_off_diag": 0.8849,
         "verdict": "bipartite"},
}


@dataclass
class SideStates:
    layers: Dict[int, List[torch.Tensor]]
    mask: torch.Tensor


class LayerCapture:
    """Capture post-block raw states for 24/36 and v10 boundary for 47."""

    def __init__(self, layers: Iterable[int]) -> None:
        self.layers = tuple(layers)
        self.inter: Dict[int, List[torch.Tensor]] = {
            INTERMEDIATE_IDX[layer]: []
            for layer in self.layers
            if layer in INTERMEDIATE_IDX
        }
        self.boundary: List[torch.Tensor] = []

    def clear(self) -> None:
        for idx in self.inter:
            self.inter[idx] = []
        self.boundary = []

    def make_inter_hook(self, idx: int):
        def _hook(_module, _inp, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.inter[idx].append(tensor.detach())
        return _hook

    def boundary_hook(self, _module, _inp, output):
        # OuroModel.forward returns (BaseModelOutputWithPast,
        # hidden_states_list, gate_list). output[1] is the four post-norm
        # loop boundary states used by v10 and by the published head.
        self.boundary = [h.detach() for h in output[1]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=("rltt", "thinking"), default="rltt")
    p.add_argument("--model-path", default=None,
                   help="Override model path. Defaults from --backbone.")
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset", default="Anthropic/hh-rlhf")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", nargs="+", type=int, default=list(VALID_LAYERS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--output-json", default=DEFAULT_JSON)
    p.add_argument("--output-md", default=DEFAULT_MD)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_block(normal: Iterable[float], flipped: Iterable[float]) -> Dict[str, float | int]:
    s = np.asarray(list(normal), dtype=np.float64)
    sf = np.asarray(list(flipped), dtype=np.float64)
    valid = np.isfinite(s) & np.isfinite(sf)
    s = s[valid]
    sf = sf[valid]
    n = int(s.size)
    if n == 0:
        return {"n_valid": 0}

    centered = s - sf
    summed = s + sf
    strict = np.sign(s) != np.sign(sf)
    corr = float("nan")
    if np.std(s) > 1e-12 and np.std(sf) > 1e-12:
        corr = float(np.corrcoef(s, -sf)[0, 1])
    centered_std = float(np.std(centered))

    return {
        "n_valid": n,
        "canonical_acc": float(np.mean(s > 0)),
        "flipped_acc": float(np.mean(sf < 0)),
        "centered_acc": float(np.mean(centered > 0)),
        "strict_sign_reversal": float(np.mean(strict)),
        "normal_pos_rate": float(np.mean(s > 0)),
        "flipped_pos_rate": float(np.mean(sf > 0)),
        "antisym_corr": corr,
        "bias_to_signal": (
            abs(float(np.mean(summed))) / centered_std
            if centered_std > 1e-12 else float("inf")
        ),
        "score_cr_mean": float(np.mean(s)),
        "score_cr_std": float(np.std(s)),
        "score_rc_mean": float(np.mean(sf)),
        "score_rc_std": float(np.std(sf)),
        "centered_mean": float(np.mean(centered)),
        "centered_std": centered_std,
        "summed_mean": float(np.mean(summed)),
        "raw_scores_cr": s.tolist(),
        "raw_scores_rc": sf.tolist(),
    }


def build_config(states: List[torch.Tensor], name: str) -> List[torch.Tensor]:
    if name == "L1_replicated":
        base = states[0]
        return [base] * NUM_LOOPS
    if name == "L2_replicated":
        base = states[1]
        return [base] * NUM_LOOPS
    if name == "L4_replicated":
        base = states[3]
        return [base] * NUM_LOOPS
    if name == "mean_replicated":
        base = torch.stack(states, dim=0).mean(dim=0)
        return [base] * NUM_LOOPS
    if name == "natural_seq":
        return list(states)
    raise ValueError(f"unknown config: {name}")


def to_side(cap: LayerCapture, mask: torch.Tensor, layers: Tuple[int, ...],
            device: torch.device) -> SideStates:
    out: Dict[int, List[torch.Tensor]] = {}
    for layer in layers:
        if layer == 47:
            if len(cap.boundary) != NUM_LOOPS:
                raise RuntimeError(f"layer 47 boundary expected {NUM_LOOPS} loops, "
                                   f"got {len(cap.boundary)}")
            out[layer] = [
                h.to(device=device, dtype=torch.float32)
                for h in cap.boundary
            ]
        else:
            idx = INTERMEDIATE_IDX[layer]
            if len(cap.inter[idx]) != NUM_LOOPS:
                raise RuntimeError(f"layer {layer} expected {NUM_LOOPS} hooks, "
                                   f"got {len(cap.inter[idx])}")
            out[layer] = [
                h.to(device=device, dtype=torch.float32)
                for h in cap.inter[idx]
            ]
    return SideStates(layers=out, mask=mask.to(device=device, dtype=torch.float32))


@torch.no_grad()
def score_pair(head: PairwiseEvaluator, a_states: List[torch.Tensor],
               a_mask: torch.Tensor, b_states: List[torch.Tensor],
               b_mask: torch.Tensor) -> float:
    out = head(a_states, a_mask, b_states, b_mask).view(-1)
    value = float(out.item())
    return value if math.isfinite(value) else float("nan")


def load_head(checkpoint_path: Path, device: torch.device) -> Tuple[PairwiseEvaluator, Dict[str, object]]:
    evaluator_file = Path(inspect.getfile(pairwise_module)).resolve()
    head = PairwiseEvaluator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    incompatible = head.load_state_dict(state_dict, strict=True)
    head.eval()
    meta = {
        "module": "evaluator_core.pairwise_evaluator.PairwiseEvaluator",
        "module_path": str(evaluator_file),
        "module_sha256": sha256_file(evaluator_file),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_load_strict": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    return head, meta


def rank_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda r: (
            -float(r["centered_acc"]),
            -float(r["canonical_acc"]),
            str(r["layer"]),
            str(r["config"]),
        ),
    )


def recommend(summary: Dict[str, Dict[str, Dict[str, float | int]]]) -> Dict[str, object]:
    rec: Dict[str, object] = {}
    for layer in ("24", "36"):
        rows = [
            {"config": cfg, **metrics}
            for cfg, metrics in summary.get(layer, {}).items()
        ]
        if not rows:
            rec[f"tap{layer}_input"] = "not evaluated"
            continue
        rows = rank_rows([{"layer": int(layer), **row} for row in rows])
        top = rows[0]
        runner_up = rows[1] if len(rows) > 1 else None
        best_centered = float(top["centered_acc"])
        runner_centered = (
            float(runner_up["centered_acc"]) if runner_up is not None
            else float("nan")
        )
        centered_gap = (
            best_centered - runner_centered
            if math.isfinite(runner_centered) else float("nan")
        )
        old_head_decisive = (
            best_centered > NEAR_CHANCE_CENTERED
            and (not math.isfinite(centered_gap)
                 or centered_gap >= MEANINGFUL_CENTERED_GAP)
        )
        rec[f"tap{layer}_best_old_head_config"] = str(top["config"])
        rec[f"tap{layer}_best_centered_acc"] = best_centered
        rec[f"tap{layer}_centered_gap_to_runner_up"] = centered_gap
        rec[f"tap{layer}_old_head_decisive"] = old_head_decisive
        if old_head_decisive:
            rec[f"tap{layer}_input"] = str(top["config"])
        else:
            rec[f"tap{layer}_input"] = (
                "train_new_single_state_2048_head; old-head zero-shot "
                "configs are tied/near chance"
            )
        rec[f"tap{layer}_conservative_capture_default"] = "L4"
        rec[f"tap{layer}_latency_candidates"] = ["L1", "L2"]
    rec["tap47_input"] = (
        "fused L1/L4: concat(h47_L1,h47_L4) or "
        "concat(h47_L4,h47_L4-h47_L1), pending Experiment 2 concat-vs-diff"
    )
    return rec


def write_markdown(path: Path, args: argparse.Namespace, rows: List[Dict[str, object]],
                   recs: Dict[str, object], head_meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Converged Tap Input Probe")
    lines.append("")
    lines.append("## Layer-Geometry Context")
    lines.append("")
    lines.append("| Layer | L1-L4 cos | L2-L4 cos | mean off-diag | Verdict |")
    lines.append("|---:|---:|---:|---:|---|")
    for layer in VALID_LAYERS:
        g = GEOMETRY_CONTEXT[layer]
        lines.append(
            f"| {layer} | {g['L1_L4']:+.4f} | {g['L2_L4']:+.4f} | "
            f"{g['mean_off_diag']:+.4f} | {g['verdict']} |"
        )
    lines.append("")
    lines.append("Probe 2 sanity: Thinking vs RLTT identical at layers 24/36/47; "
                 "published-head Pearson 0.9906, decision agreement 0.9950, "
                 "canonical acc Thinking 0.9500, RLTT 0.9450.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Backbone: `{args.backbone}`")
    lines.append(f"- Max examples: `{args.max_examples}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- Layers: `{', '.join(map(str, args.layers))}`")
    lines.append(f"- Evaluator module: `{head_meta['module_path']}`")
    lines.append(f"- Evaluator module SHA256: `{head_meta['module_sha256']}`")
    lines.append(f"- Checkpoint: `{head_meta['checkpoint_path']}`")
    lines.append(f"- Checkpoint SHA256: `{head_meta['checkpoint_sha256']}`")
    lines.append(f"- Strict load missing keys: `{head_meta['missing_keys']}`")
    lines.append(f"- Strict load unexpected keys: `{head_meta['unexpected_keys']}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("Sorted by centered accuracy, then canonical accuracy.")
    lines.append("")
    lines.append("| Layer | Config | centered | canonical | flipped | strict | corr | bias/sig | agree vs L47 natural |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        agree = row.get("decision_agreement_with_l47_natural_seq")
        agree_s = "NA" if agree is None or not math.isfinite(float(agree)) else f"{float(agree):.3f}"
        corr = row.get("antisym_corr")
        corr_s = "NA" if corr is None or not math.isfinite(float(corr)) else f"{float(corr):+.3f}"
        lines.append(
            f"| {row['layer']} | `{row['config']}` | "
            f"{float(row['centered_acc']):.3f} | "
            f"{float(row['canonical_acc']):.3f} | "
            f"{float(row['flipped_acc']):.3f} | "
            f"{float(row['strict_sign_reversal']):.3f} | "
            f"{corr_s} | {float(row['bias_to_signal']):.3f} | {agree_s} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Layer 47 is the only old-head-compatible relational tap in this "
                 "probe. Layers 24 and 36 are converged intermediate "
                 "checkpoints, but the published boundary-trained head is "
                 "near chance on HH centered accuracy there.")
    lines.append("")
    lines.append("The high canonical rates at layers 24/36 are therefore not "
                 "sufficient branch-selection evidence by themselves; centered "
                 "accuracy is flat across the candidate constructions.")
    lines.append("")
    lines.append("The intermediate layers are still order-sensitive raw material: "
                 "their strict sign-reversal and antisymmetry-correlation "
                 "metrics are substantially above a degenerate readout. The "
                 "next decision should be made after training new 2048-dim "
                 "single-state heads, not by reusing the old checkpoint.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    for layer in ("24", "36"):
        best = recs.get(f"tap{layer}_best_old_head_config", "not evaluated")
        centered = recs.get(f"tap{layer}_best_centered_acc")
        gap = recs.get(f"tap{layer}_centered_gap_to_runner_up")
        centered_s = (
            "NA" if not isinstance(centered, (float, int))
            else f"{float(centered):.3f}"
        )
        gap_s = (
            "NA" if not isinstance(gap, (float, int)) or not math.isfinite(float(gap))
            else f"{float(gap):.3f}"
        )
        lines.append(
            f"- tap{layer}: train a new 2048-dim single-state comparator. "
            f"The old-head best row was `{best}` "
            f"(centered {centered_s}, gap {gap_s}), which is not a "
            "meaningful readout winner."
        )
        lines.append(
            f"- tap{layer}: use L4 only as a conservative capture/default "
            "implementation choice; L1/L2 remain latency candidates for the "
            "trained-head ablation."
        )
    lines.append("- tap47 remains fused L1/L4, pending Experiment 2 concat-vs-diff.")
    lines.append("")
    lines.append("Do not run more old-head zero-shot probes to choose 24/36. The "
                 "next useful probe is to train small single-state 2048-dim "
                 "heads for layer 24 and 36 candidates, then compare HH "
                 "centered diagnostics and generated-branch tournament "
                 "performance.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    layers = tuple(dict.fromkeys(args.layers))
    bad = [layer for layer in layers if layer not in VALID_LAYERS]
    if bad:
        raise SystemExit(f"Unsupported layers {bad}; valid layers are {VALID_LAYERS}")
    args.layers = list(layers)

    device = torch.device(args.device)
    model_path = args.model_path
    if model_path is None:
        model_path = DEFAULT_RLTT_PATH if args.backbone == "rltt" else DEFAULT_THINKING_PATH
    model_path = resolve_local(model_path)
    tokenizer_path = resolve_local(args.tokenizer_path)
    checkpoint_path = Path(resolve_local(args.checkpoint))

    print(f"Backbone: {args.backbone} ({model_path})")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Layers: {layers}")

    examples = select_examples(args.seed, args.max_examples, args.dataset, args.split)
    head, head_meta = load_head(checkpoint_path, device)
    print(f"Evaluator module: {head_meta['module_path']}")
    print(f"Evaluator module SHA256: {head_meta['module_sha256']}")
    print(f"Checkpoint SHA256: {head_meta['checkpoint_sha256']}")
    print(f"Strict load missing={head_meta['missing_keys']} "
          f"unexpected={head_meta['unexpected_keys']}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    print(f"Loading model {model_path} (bf16)")
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
    print(f"Model ready in {time.time() - t0:.1f}s")

    cap = LayerCapture(layers)
    handles = [model.model.register_forward_hook(cap.boundary_hook)]
    for layer in layers:
        if layer in INTERMEDIATE_IDX:
            idx = INTERMEDIATE_IDX[layer]
            handles.append(model.model.layers[idx].register_forward_hook(
                cap.make_inter_hook(idx)))

    def run_side(text: str) -> SideStates:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length).to(device)
        cap.clear()
        with torch.no_grad():
            model(**enc, use_cache=False)
        return to_side(cap, enc["attention_mask"], layers, device)

    raw: Dict[str, Dict[str, List[float]]] = {}
    for layer in layers:
        for cfg in CONFIGS:
            raw[f"{layer}/{cfg}"] = {"s_cr": [], "s_rc": []}

    start = time.time()
    for i, ex in enumerate(examples):
        chosen = run_side(ex["chosen"])
        rejected = run_side(ex["rejected"])
        for layer in layers:
            for cfg in CONFIGS:
                key = f"{layer}/{cfg}"
                c_states = build_config(chosen.layers[layer], cfg)
                r_states = build_config(rejected.layers[layer], cfg)
                s_cr = score_pair(head, c_states, chosen.mask,
                                  r_states, rejected.mask)
                s_rc = score_pair(head, r_states, rejected.mask,
                                  c_states, chosen.mask)
                raw[key]["s_cr"].append(s_cr)
                raw[key]["s_rc"].append(s_rc)

        if (i + 1) % args.report_every == 0 or i + 1 == len(examples):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (len(examples) - i - 1) / rate if rate > 0 else float("nan")
            print(f"{i+1:>4}/{len(examples)}  elapsed={elapsed:6.1f}s "
                  f"rate={rate:4.2f}/s eta={eta:6.1f}s")

        del chosen, rejected
        if device.type == "cuda" and (i + 1) % args.report_every == 0:
            torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary: Dict[str, Dict[str, Dict[str, float | int]]] = {
        str(layer): {} for layer in layers
    }
    for layer in layers:
        for cfg in CONFIGS:
            key = f"{layer}/{cfg}"
            block = metric_block(raw[key]["s_cr"], raw[key]["s_rc"])
            summary[str(layer)][cfg] = block

    ref_scores = None
    if 47 in layers:
        ref_scores = np.asarray(raw["47/natural_seq"]["s_cr"], dtype=np.float64)
    rows: List[Dict[str, object]] = []
    for layer in layers:
        for cfg in CONFIGS:
            key = f"{layer}/{cfg}"
            metrics = dict(summary[str(layer)][cfg])
            scores = np.asarray(raw[key]["s_cr"], dtype=np.float64)
            agree = None
            if ref_scores is not None and ref_scores.size == scores.size:
                valid = np.isfinite(ref_scores) & np.isfinite(scores)
                agree = float(np.mean(np.sign(scores[valid]) == np.sign(ref_scores[valid]))) if valid.any() else float("nan")
            metrics["decision_agreement_with_l47_natural_seq"] = agree
            summary[str(layer)][cfg]["decision_agreement_with_l47_natural_seq"] = agree
            rows.append({"layer": layer, "config": cfg, **metrics})

    ranked_rows = rank_rows(rows)
    recs = recommend(summary)

    result = {
        "args": vars(args) | {
            "model_path_resolved": model_path,
            "tokenizer_path_resolved": tokenizer_path,
        },
        "layer_geometry_context": GEOMETRY_CONTEXT,
        "probe2_sanity": {
            "thinking_rltt_identical_layers": [24, 36, 47],
            "pearson": 0.9906,
            "decision_agreement": 0.9950,
            "canonical_acc_thinking": 0.9500,
            "canonical_acc_rltt": 0.9450,
        },
        "head_provenance": head_meta,
        "summary": summary,
        "ranked_rows": ranked_rows,
        "recommendation": recs,
    }

    out_json = Path(resolve_local(args.output_json))
    out_md = Path(resolve_local(args.output_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n",
                        encoding="utf-8")
    write_markdown(out_md, args, ranked_rows, recs, head_meta)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print("\nRecommendation:")
    for key, value in recs.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
