"""Prompt-only hidden carry equivalence probe for DualAnchor branch mechanics.

This checks the narrow mechanical question raised by the architecture-looped
lineage run:

    Does cumulative hook replay match an actual manual hidden-state carry path
    for prompt-only forwards?

The probe does not train Ouro, does not modify weights/checkpoints, does not run
action steering, and does not claim generation-ready true fork/carry.  It runs
deterministic prompt-only forwards and compares:

1. normal model forward vs a manual layer/loop runner,
2. HiddenDeltaLayerHook replay vs manual post-layer hidden carry at L24/L36/L47,
3. manual post-loop boundary carry after L47/L1, which has no current hook replay
   equivalent in the standard generation path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, delta_rms, rms_normalize  # noqa: E402
from src.evaluator.bg_transformer_features import DEFAULT_MODEL_PATH  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_dualanchor_true_carry_equivalence_v1_2026-05-31"
OUT_JSON = OUT_ROOT / "true_carry_equivalence.json"
OUT_MD = OUT_ROOT / "true_carry_equivalence.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ROWS_CSV = OUT_ROOT / "true_carry_equivalence_rows.csv"
ARTIFACT_PT = OUT_ROOT / "true_carry_equivalence.pt"

NUM_LOOPS = 4
NUM_LAYERS = 48
HIDDEN_DIM = 2048
TAP_LAYERS = (24, 36, 47)
DEFAULT_PROMPTS = (
    "Question: Which object is usually used to write on paper?\nA. Spoon\nB. Pencil\nC. Pillow\nD. Shoe\nAnswer:",
    "Question: Plants generally need which of these to make food?\nA. Sunlight\nB. Plastic\nC. Iron nails\nD. Sandpaper\nAnswer:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def dtype_from_arg(value: str, device: torch.device) -> torch.dtype:
    v = str(value).lower()
    if v == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def encode(tokenizer: Any, prompt: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_length), padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    return enc


def make_delta(seed: int, alpha: float, dim: int = HIDDEN_DIM) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return rms_normalize(torch.randn(dim, generator=gen, dtype=torch.float32)) * float(alpha)


def apply_delta_like_hook(hidden_states: torch.Tensor, delta: torch.Tensor, position: int = -1, max_rms_fraction: float = 0.02) -> torch.Tensor:
    changed = hidden_states.clone()
    seq_len = int(changed.shape[1])
    pos = int(position) if int(position) >= 0 else seq_len + int(position)
    target = changed[:, pos : pos + 1, :]
    base_rms = target.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    delta_dev = delta.detach().to(device=target.device, dtype=target.dtype).view(1, 1, -1)
    residual = delta_dev * base_rms.to(dtype=target.dtype)
    residual_rms = residual.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    max_allowed = float(max_rms_fraction) * base_rms
    residual = residual * torch.clamp(max_allowed / residual_rms, max=1.0).to(dtype=target.dtype)
    changed[:, pos : pos + 1, :] = target + residual
    return changed


def build_model_inputs(inner: Any, enc: dict[str, torch.Tensor], inputs_embeds: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    attention_mask = enc.get("attention_mask")
    cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    mask_kwargs = {
        "config": inner.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(inner, "has_sliding_layers", False):
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    position_embeddings = inner.rotary_emb(inputs_embeds, position_ids)
    return causal_mask_mapping, position_ids, cache_position, position_embeddings


def clone_last(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :].detach().to(device="cpu", dtype=torch.float32).clone()


def run_manual(
    model: Any,
    enc: dict[str, torch.Tensor],
    *,
    layer_intervention: dict[str, Any] | None = None,
    boundary_intervention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inner = model.model
    inputs_embeds = inner.embed_tokens(enc["input_ids"])
    causal_mask_mapping, position_ids, cache_position, position_embeddings = build_model_inputs(inner, enc, inputs_embeds)
    hidden_states = inputs_embeds
    captures: dict[str, torch.Tensor] = {}
    gate_list: list[torch.Tensor] = []
    with torch.inference_mode():
        for current_ut in range(NUM_LOOPS):
            loop = current_ut + 1
            for layer_idx, decoder_layer in enumerate(inner.layers[:NUM_LAYERS], start=1):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    current_ut=current_ut,
                )
                if layer_intervention and int(layer_intervention["loop"]) == loop and int(layer_intervention["layer"]) == layer_idx:
                    hidden_states = apply_delta_like_hook(
                        hidden_states,
                        layer_intervention["delta"],
                        position=int(layer_intervention.get("position", -1)),
                        max_rms_fraction=float(layer_intervention.get("max_rms_fraction", 0.02)),
                    )
                if layer_idx in TAP_LAYERS:
                    captures[f"decoder_L{loop}_{layer_idx}"] = clone_last(hidden_states)
            hidden_states = inner.norm(hidden_states)
            if boundary_intervention and int(boundary_intervention["loop"]) == loop:
                hidden_states = apply_delta_like_hook(
                    hidden_states,
                    boundary_intervention["delta"],
                    position=int(boundary_intervention.get("position", -1)),
                    max_rms_fraction=float(boundary_intervention.get("max_rms_fraction", 0.02)),
                )
            captures[f"boundary_L{loop}"] = clone_last(hidden_states)
            gate_list.append(inner.early_exit_gate(hidden_states))
        logits = model.lm_head(hidden_states[:, -1:, :]).detach().to(device="cpu", dtype=torch.float32)
    return {
        "last_hidden": clone_last(hidden_states),
        "logits": logits,
        "captures": captures,
        "gate_list": [g.detach().to(device="cpu", dtype=torch.float32) for g in gate_list],
    }


class HookCapture:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.decoder: dict[str, list[torch.Tensor]] = {str(layer): [] for layer in TAP_LAYERS}
        self.boundary: list[torch.Tensor] = []
        self.handles: list[Any] = []

    def _decoder_hook(self, layer: int):
        def hook(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any] | None, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.decoder[str(layer)].append(clone_last(tensor))

        return hook

    def _boundary_hook(self, _module: Any, _inp: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, (tuple, list)) and len(output) >= 2 and output[1] is not None:
            self.boundary = [clone_last(t) for t in output[1]]

    def __enter__(self) -> "HookCapture":
        self.handles.append(self.inner.register_forward_hook(self._boundary_hook))
        for layer in TAP_LAYERS:
            self.handles.append(self.inner.layers[layer - 1].register_forward_hook(self._decoder_hook(layer), with_kwargs=True))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def captures(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for layer in TAP_LAYERS:
            vals = self.decoder[str(layer)]
            for idx, tensor in enumerate(vals[:NUM_LOOPS], start=1):
                out[f"decoder_L{idx}_{layer}"] = tensor
        for idx, tensor in enumerate(self.boundary[:NUM_LOOPS], start=1):
            out[f"boundary_L{idx}"] = tensor
        return out


def run_model_with_optional_hook(model: Any, enc: dict[str, torch.Tensor], hook_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    active: HiddenDeltaLayerHook | None = None
    inner = model.model
    try:
        if hook_spec:
            active = HiddenDeltaLayerHook(
                model,
                target_layer=int(hook_spec["layer"]),
                target_loops=[int(hook_spec["loop"])],
                delta=hook_spec["delta"],
                position=int(hook_spec.get("position", -1)),
                max_rms_fraction=float(hook_spec.get("max_rms_fraction", 0.02)),
            )
            active.apply()
        # Register capture after the perturbation hook so the target-layer row
        # observes the carried post-hook tensor rather than the pre-hook output.
        with HookCapture(inner) as cap:
            with torch.inference_mode():
                out = model(**enc, use_cache=False, return_dict=True, logits_to_keep=1)
    finally:
        hook_diag = active.diagnostics() if active is not None else None
        if active is not None:
            active.remove()
    return {
        "logits": out.logits.detach().to(device="cpu", dtype=torch.float32),
        "last_hidden": None,
        "captures": cap.captures(),
        "hook_diagnostics": hook_diag,
    }


def tensor_metrics(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, float]:
    if a is None or b is None:
        return {"max_abs": float("nan"), "rms": float("nan"), "cosine": float("nan")}
    aa = a.detach().flatten().to(dtype=torch.float32)
    bb = b.detach().flatten().to(dtype=torch.float32)
    diff = aa - bb
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    cosine = torch.dot(aa, bb) / denom.clamp(min=1e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "rms": float(diff.pow(2).mean().sqrt().item()),
        "cosine": float(cosine.item()),
    }


def compare_capture_sets(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    return {key: tensor_metrics(left.get(key), right.get(key)) for key in keys}


def flatten_capture_metrics(prefix: str, metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, vals in metrics.items():
        for metric, value in vals.items():
            out[f"{prefix}_{key}_{metric}"] = value
    return out


def main() -> int:
    args = parse_args()
    ensure_root()
    started = time.time()
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_from_arg(str(args.dtype), device)
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_PATH), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(DEFAULT_MODEL_PATH),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.config.total_ut_steps = NUM_LOOPS
    model.config.early_exit_threshold = 1.0
    if hasattr(model.model, "total_ut_steps"):
        model.model.total_ut_steps = NUM_LOOPS

    rows: list[dict[str, Any]] = []
    payload_cases: list[dict[str, Any]] = []
    prompts = DEFAULT_PROMPTS[: max(1, int(args.max_prompts))]
    landmarks = [
        "decoder_L1_24",
        "decoder_L1_36",
        "decoder_L1_47",
        "boundary_L1",
        "decoder_L2_24",
        "decoder_L2_36",
        "decoder_L2_47",
        "boundary_L2",
        "boundary_L4",
    ]
    target_specs = [
        {"kind": "post_layer", "loop": 1, "layer": 24},
        {"kind": "post_layer", "loop": 1, "layer": 36},
        {"kind": "post_layer", "loop": 1, "layer": 47},
        {"kind": "post_loop_boundary", "loop": 1, "layer": "boundary"},
    ]

    try:
        for prompt_index, prompt in enumerate(prompts):
            enc = encode(tokenizer, prompt, device, int(args.max_length))
            normal_model = run_model_with_optional_hook(model, enc, None)
            normal_manual = run_manual(model, enc)
            normal_logits = tensor_metrics(normal_model["logits"], normal_manual["logits"])
            normal_caps = compare_capture_sets(normal_model["captures"], normal_manual["captures"], landmarks)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "case": "manual_full_vs_model_forward",
                    "target": "none",
                    "logits_max_abs": normal_logits["max_abs"],
                    "logits_rms": normal_logits["rms"],
                    "logits_cosine": normal_logits["cosine"],
                    **flatten_capture_metrics("capture", normal_caps),
                }
            )

            for spec in target_specs:
                delta = make_delta(int(args.seed) + 101 * prompt_index + int(spec["loop"]) * 17 + (0 if spec["layer"] == "boundary" else int(spec["layer"])), float(args.alpha))
                common = {
                    "loop": int(spec["loop"]),
                    "position": -1,
                    "delta": delta,
                    "max_rms_fraction": max(float(delta_rms(delta)), 0.02),
                }
                if spec["kind"] == "post_layer":
                    hook_spec = {**common, "layer": int(spec["layer"])}
                    manual = run_manual(model, enc, layer_intervention=hook_spec)
                    hook = run_model_with_optional_hook(model, enc, hook_spec)
                    logits = tensor_metrics(hook["logits"], manual["logits"])
                    caps = compare_capture_sets(hook["captures"], manual["captures"], landmarks)
                    equivalent = bool(logits["rms"] <= 1e-4 and logits["cosine"] >= 0.999999)
                    row = {
                        "prompt_index": prompt_index,
                        "case": "manual_post_layer_carry_vs_hook_replay",
                        "target": f"L{spec['loop']}_{spec['layer']}",
                        "hook_replay_equivalent_available": True,
                        "equivalence_pass": equivalent,
                        "delta_rms": float(delta_rms(delta)),
                        "logits_max_abs": logits["max_abs"],
                        "logits_rms": logits["rms"],
                        "logits_cosine": logits["cosine"],
                        "hook_modifications": None if hook.get("hook_diagnostics") is None else hook["hook_diagnostics"].get("modifications"),
                        **flatten_capture_metrics("capture", caps),
                    }
                    rows.append(row)
                    payload_cases.append(row)
                else:
                    manual_boundary = run_manual(model, enc, boundary_intervention=common)
                    manual_decoder47 = run_manual(model, enc, layer_intervention={**common, "layer": 47})
                    logits = tensor_metrics(manual_boundary["logits"], manual_decoder47["logits"])
                    caps = compare_capture_sets(manual_boundary["captures"], manual_decoder47["captures"], landmarks)
                    row = {
                        "prompt_index": prompt_index,
                        "case": "manual_boundary_carry_vs_decoder47_layer_carry",
                        "target": f"L{spec['loop']}_boundary_after_loop",
                        "hook_replay_equivalent_available": False,
                        "equivalence_pass": False,
                        "delta_rms": float(delta_rms(delta)),
                        "logits_max_abs": logits["max_abs"],
                        "logits_rms": logits["rms"],
                        "logits_cosine": logits["cosine"],
                        **flatten_capture_metrics("capture", caps),
                    }
                    rows.append(row)
                    payload_cases.append(row)
    finally:
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    post_layer_rows = [r for r in rows if r.get("case") == "manual_post_layer_carry_vs_hook_replay"]
    no_hook_rows = [r for r in rows if r.get("case") == "manual_full_vs_model_forward"]
    boundary_rows = [r for r in rows if r.get("case") == "manual_boundary_carry_vs_decoder47_layer_carry"]
    manual_runner_pass = bool(no_hook_rows) and all(float(r.get("logits_rms", 999.0)) <= 1e-4 for r in no_hook_rows)
    post_layer_pass = bool(post_layer_rows) and all(bool(r.get("equivalence_pass")) for r in post_layer_rows)
    boundary_distinct = bool(boundary_rows) and any(float(r.get("logits_rms", 0.0)) > 1e-6 for r in boundary_rows)
    verdict = "PROMPT_ONLY_LAYER_CARRY_EQUIVALENT"
    if not manual_runner_pass:
        verdict = "MANUAL_RUNNER_MISMATCH"
    elif not post_layer_pass:
        verdict = "HOOK_REPLAY_LAYER_CARRY_MISMATCH"
    elif boundary_distinct:
        verdict = "PROMPT_ONLY_LAYER_CARRY_EQUIVALENT_BOUNDARY_NOT_HOOKED"

    payload = {
        "BG_DUALANCHOR_TRUE_CARRY_EQUIVALENCE_VERDICT": verdict,
        "status": verdict,
        "mode": "PROMPT_ONLY_MANUAL_LAYER_CARRY_EQUIVALENCE",
        "true_generation_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "model_path": rel(DEFAULT_MODEL_PATH),
        "args": vars(args),
        "manual_runner_pass": manual_runner_pass,
        "post_layer_carry_matches_hook_replay": post_layer_pass,
        "post_loop_boundary_has_hook_replay_equivalent": False,
        "boundary_carry_distinct_from_decoder47_layer_carry": boundary_distinct,
        "row_count": len(rows),
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(ROWS_CSV, rows)
    torch.save(payload, ARTIFACT_PT)

    lines = [
        "# DualAnchor True Carry Equivalence Probe v1",
        "",
        f"BG_DUALANCHOR_TRUE_CARRY_EQUIVALENCE_VERDICT = {verdict}",
        "",
        "## What Was Tested",
        "",
        "- Manual full forward vs standard model forward.",
        "- Manual post-layer carry vs `HiddenDeltaLayerHook` replay at L24, L36, and decoder-layer L47.",
        "- Manual post-loop boundary carry after L1 vs decoder-layer L47 carry.",
        "",
        "This is prompt-only. It does not prove generation-ready true fork/carry or compute savings.",
        "",
        "## Results",
        "",
        f"- manual runner pass: `{manual_runner_pass}`",
        f"- post-layer carry matches hook replay: `{post_layer_pass}`",
        f"- post-loop boundary hook replay equivalent available: `False`",
        f"- boundary carry distinct from decoder-layer L47 carry: `{boundary_distinct}`",
        "",
        "## Key Rows",
        "",
        "| case | target | prompt | pass | logits_rms | logits_cosine | hook_modifications |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('case')}` | `{row.get('target')}` | `{row.get('prompt_index')}` | "
            f"`{row.get('equivalence_pass', '')}` | `{row.get('logits_rms')}` | `{row.get('logits_cosine')}` | `{row.get('hook_modifications', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If post-layer rows pass, cumulative hook replay is mechanically equivalent to manually carrying the perturbed hidden state forward for prompt-only forwards at those decoder layers.",
            "- The current layer hook can perturb decoder layer 47 and that perturbation flows through layer 48 and later loops.",
            "- A post-loop boundary perturbation is a different intervention surface and currently has no standard hook/generation replay equivalent.",
            "- Generation-ready true fork/carry still requires branch-specific hidden/cache continuation during autoregressive decoding.",
            "",
            "## Files",
            "",
            f"- report json: `{rel(OUT_JSON)}`",
            f"- rows: `{rel(ROWS_CSV)}`",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
        ]
    )
    write_md(OUT_MD, lines)
    write_md(SUMMARY_MD, lines)
    print(f"BG_DUALANCHOR_TRUE_CARRY_EQUIVALENCE_VERDICT = {verdict}", flush=True)
    print(f"manual_runner_pass = {manual_runner_pass}", flush=True)
    print(f"post_layer_carry_matches_hook_replay = {post_layer_pass}", flush=True)
    print(f"boundary_carry_distinct_from_decoder47_layer_carry = {boundary_distinct}", flush=True)
    print(f"report = {rel(OUT_MD)}", flush=True)
    return 0 if manual_runner_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
