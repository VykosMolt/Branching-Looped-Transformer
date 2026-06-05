"""Shared capture machinery for the layer-24/36/47 × 4-loop probes.

Both `probe_bipartite_layers_24_36.py` (Probe 1) and
`probe_cross_backbone_layers.py` (Probe 2) read the same Ouro hidden
states. To avoid capturing twice this module owns:

  - example selection (mirrors probe_loop_geometry_hh.py exactly: seeded
    permutation of the HH-RLHF test split, then sorted indices)
  - the forward-hook capture of per-loop hidden states at layers 24, 36
    and 47 for one backbone
  - save / load of the captured packs

Architecture facts (models/ouro_rltt_local/modeling_ouro.py, verified):
  - num_hidden_layers = 48, total_ut_steps = 4 (R = 4 loops),
    hidden_size = 2048.
  - One "loop" = one full pass through all 48 shared decoder layers.
    `_run_single_ut_loop` applies `self.norm` at the end of each loop
    and appends the result to `hidden_states_list`, which `OuroModel`
    returns as output[1]. That list is exactly v10's per-loop
    "boundary" state (post-final-norm). We therefore take **layer 47**
    from output[1] (identical convention to probe_loop_geometry_hh.py's
    BoundaryCapture, so the L47 numbers reproduce v10 as a sanity check).
  - **layers 24 and 36** have no per-loop norm; the analogous "post-block"
    state is the raw output of `model.model.layers[23]` / `layers[35]`
    (after that decoder layer's MLP residual add, before the next layer).
    Each fires once per loop, in loop order, so a forward hook collects
    exactly 4 tensors = loops 1..4. This is the intermediate-layer
    convention the probe spec requests.

  Consequence (documented, intentional): L47 is post-final-norm while
  L24/L36 are pre-norm residual-stream states. That is what the BG taps
  actually see in deployment (intermediate taps read the raw residual
  stream; the L47 tap reads the normed boundary), and it matches v10 for
  the L47 sanity check.

Numerical contract (identical to probe_loop_geometry_hh.py / v10):
  bf16 model forward (two 2.6B Ouro backbones cannot hold fp32 in 12 GB
  VRAM), captured states cast to fp32 immediately, all geometry/head math
  in fp32. Single Ouro tokenizer (models/ouro_rltt_local) for both
  backbones to guarantee identical input IDs.

Path reconciliation: the probe spec's `tests/manual/` + `runs/` wording
predates the public repo reorg; probes live under the top-level probes/ folder.
and artifacts under artifacts/reports/evaluator/ (v10 convention). The
spec's intent is honored, the paths are the current ones.
"""
from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUM_LOOPS = 4
TAP_LAYERS = (24, 36, 47)            # 1-indexed, spec language
_INTERMEDIATE_LAYER_IDX = {24: 23, 36: 35}  # 0-indexed module positions
_BOUNDARY_LAYER = 47                  # taken from OuroModel output[1] (post-norm)

DEFAULT_THINKING_PATH = "ByteDance/Ouro-2.6B-Thinking"
DEFAULT_RLTT_PATH = "models/ouro_rltt_local"
DEFAULT_TOKENIZER_PATH = "models/ouro_rltt_local"
DEFAULT_HEAD_PATH = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
DEFAULT_OUTPUT_DIR = "artifacts/reports/evaluator"
DEFAULT_DATASET = "Anthropic/hh-rlhf"
DEFAULT_SPLIT = "test"


def resolve_local(path: str) -> str:
    """Return an absolute path if `path` names something under the project
    root (model dirs, checkpoints, output dirs); otherwise return it
    unchanged so HF repo ids like 'ByteDance/Ouro-2.6B-Thinking' still
    resolve via the hub. Makes the probes CWD-independent."""
    if Path(path).is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    return path


def select_examples(seed: int, max_examples: int,
                    dataset: str = DEFAULT_DATASET,
                    split: str = DEFAULT_SPLIT) -> List[Dict[str, str]]:
    """Mirror probe_loop_geometry_hh.py::select_examples exactly so the
    200 pairs are byte-identical to the v10 / L1-ablation set."""
    print(f"Loading dataset {dataset} ({split})")
    ds = load_dataset(dataset, split=split)
    n_total = len(ds)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)[:max_examples].tolist()
    indices.sort()
    print(f"Total {n_total} examples; sampled {len(indices)} with seed {seed}")
    return [{"chosen": ds[int(i)]["chosen"], "rejected": ds[int(i)]["rejected"]}
            for i in indices]


def _mean_pool(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over mask-valid tokens. states [T, H] fp32, mask [T]."""
    m = mask.to(dtype=states.dtype).unsqueeze(-1)
    denom = m.sum(dim=0).clamp(min=1.0)
    return (states * m).sum(dim=0) / denom


@dataclass
class SidePack:
    pooled: Dict[int, torch.Tensor]      # layer -> [NUM_LOOPS, H] fp32
    l47_tokens: List[torch.Tensor]       # NUM_LOOPS tensors [T, H] fp32
    mask: torch.Tensor                   # [T] fp32


@dataclass
class ExamplePack:
    chosen: SidePack
    rejected: SidePack


class _LayerCapture:
    """Forward hooks: layers[23]/[35] (per-loop post-block) + model.model
    output[1] (the 4 post-norm boundary states = layer 47)."""

    def __init__(self) -> None:
        self.inter: Dict[int, List[torch.Tensor]] = {23: [], 35: []}
        self.boundary: List[torch.Tensor] = []

    def clear(self) -> None:
        self.inter = {23: [], 35: []}
        self.boundary = []

    def make_inter_hook(self, idx: int):
        def _hook(_module, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            self.inter[idx].append(t.detach())
        return _hook

    def boundary_hook(self, _module, _inp, output):
        # OuroModel.forward returns (BaseModelOutputWithPast,
        # hidden_states_list, gate_list); output[1] is the per-loop list.
        self.boundary = [h.detach() for h in output[1]]


def _to_sidepack(cap: _LayerCapture, attn_mask: torch.Tensor) -> SidePack:
    assert len(cap.boundary) == NUM_LOOPS, (
        f"expected {NUM_LOOPS} boundary states, got {len(cap.boundary)}")
    for idx in (23, 35):
        assert len(cap.inter[idx]) == NUM_LOOPS, (
            f"layer idx {idx}: expected {NUM_LOOPS} loop captures, "
            f"got {len(cap.inter[idx])} (adaptive early-exit must be off)")

    mask = attn_mask.squeeze(0).to(device="cpu", dtype=torch.float32).clone()

    l47_tokens = [h.squeeze(0).to(device="cpu", dtype=torch.float32).clone()
                  for h in cap.boundary]

    pooled: Dict[int, torch.Tensor] = {}
    for layer, idx in _INTERMEDIATE_LAYER_IDX.items():
        per_loop = [
            _mean_pool(cap.inter[idx][k].squeeze(0).to("cpu", torch.float32),
                       mask)
            for k in range(NUM_LOOPS)
        ]
        pooled[layer] = torch.stack(per_loop, dim=0)            # [4, H]
    pooled[_BOUNDARY_LAYER] = torch.stack(
        [_mean_pool(t, mask) for t in l47_tokens], dim=0)        # [4, H]

    return SidePack(pooled=pooled, l47_tokens=l47_tokens, mask=mask)


def capture_backbone(model_path: str, tokenizer_path: str,
                     examples: List[Dict[str, str]], max_length: int,
                     device: torch.device, label: str,
                     report_every: int = 25) -> Tuple[List[ExamplePack], Dict]:
    model_path = resolve_local(model_path)
    tokenizer_path = resolve_local(tokenizer_path)
    print(f"\n[{label}] tokenizer from {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[{label}] loading model {model_path} (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # threshold == 1.0 keeps adaptive_compute disabled -> all 4 loops run.
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0
    print(f"[{label}] ready in {time.time() - t0:.1f}s "
          f"(num_hidden_layers={model.config.num_hidden_layers}, "
          f"total_ut_steps={getattr(model.config, 'total_ut_steps', '?')})")
    assert model.config.num_hidden_layers == 48, "expected 48 layers"

    cap = _LayerCapture()
    handles = [model.model.register_forward_hook(cap.boundary_hook)]
    for layer1, idx in _INTERMEDIATE_LAYER_IDX.items():
        handles.append(
            model.model.layers[idx].register_forward_hook(
                cap.make_inter_hook(idx)))

    def _run(text: str) -> SidePack:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_length).to(device)
        cap.clear()
        with torch.no_grad():
            model(**enc, use_cache=False)
        return _to_sidepack(cap, enc["attention_mask"])

    packs: List[ExamplePack] = []
    n = len(examples)
    t_start = time.time()
    for i, ex in enumerate(examples):
        c = _run(ex["chosen"])
        r = _run(ex["rejected"])
        packs.append(ExamplePack(chosen=c, rejected=r))
        if (i + 1) % report_every == 0 or i == n - 1:
            el = time.time() - t_start
            rate = (i + 1) / el if el > 0 else 0.0
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[{label}] {i+1:>3}/{n}  el={el:5.1f}s "
                  f"rate={rate:4.2f}/s eta={eta:5.1f}s")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    meta = {
        "label": label,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "n_examples": n,
        "max_length": max_length,
        "num_loops": NUM_LOOPS,
        "tap_layers": list(TAP_LAYERS),
        "l47_convention": "OuroModel output[1] (post-final-norm); == v10 boundary",
        "l24_l36_convention": "post-block raw residual (layers[23]/[35] output, pre-norm)",
        "dtype": "fp32 states (model fwd bf16)",
    }
    return packs, meta


def save_packs(path: Path, packs: List[ExamplePack], meta: Dict) -> None:
    payload = {
        "meta": meta,
        "packs": [
            {
                "chosen": {"pooled": p.chosen.pooled,
                           "l47_tokens": p.chosen.l47_tokens,
                           "mask": p.chosen.mask},
                "rejected": {"pooled": p.rejected.pooled,
                             "l47_tokens": p.rejected.l47_tokens,
                             "mask": p.rejected.mask},
            }
            for p in packs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"[save] {path}  ({mb:.1f} MB, {len(packs)} examples)")


def load_packs(path: Path) -> Tuple[List[ExamplePack], Dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    packs = [
        ExamplePack(
            chosen=SidePack(d["chosen"]["pooled"], d["chosen"]["l47_tokens"],
                            d["chosen"]["mask"]),
            rejected=SidePack(d["rejected"]["pooled"],
                              d["rejected"]["l47_tokens"], d["rejected"]["mask"]),
        )
        for d in payload["packs"]
    ]
    return packs, payload.get("meta", {})
