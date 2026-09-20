"""Shared helpers for the BG causal intervention adapter experiment."""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_causal_intervention_adapter_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
STAGE2_LAYERHOOK_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
EMPIRICAL_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_empirical_steering_direction_2026-05-18"
PRECONSOLIDATION_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_preconsolidation_control_probes_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518
HIDDEN_DIM = 2048
NUM_LOOPS = 4
TARGET_LAYER = 36
TARGET_LAYER_INDEX = 35
REQUIRED_CUDA_DEVICE_INDEX = 0
REQUIRED_GPU_NAME_FRAGMENT = "5070 Ti"
PRIMARY_MODE = "multi_loop_decayed"
COMPARISON_MODE = "single_loop_L1"
TRAIN_ALPHAS = [0.005, 0.01, 0.02]
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_md(path: str | Path, lines: Iterable[str]) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def append_once(path: str | Path, title: str, lines: list[str]) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    marker = f"## {title}"
    if marker in existing:
        return
    p.write_text(existing.rstrip() + "\n\n" + marker + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def avg(values: Iterable[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            vals.append(numeric)
    return sum(vals) / len(vals) if vals else None


def rate(values: Iterable[bool]) -> float | None:
    vals = [bool(v) for v in values]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def set_seed(seed: int = SEED) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def require_adapter_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("BG causal adapter requires CUDA; torch.cuda.is_available() is false")
    if torch.cuda.device_count() <= REQUIRED_CUDA_DEVICE_INDEX:
        raise RuntimeError(
            f"BG causal adapter requires cuda:{REQUIRED_CUDA_DEVICE_INDEX}; "
            f"only {torch.cuda.device_count()} CUDA devices are visible"
        )
    name = torch.cuda.get_device_name(REQUIRED_CUDA_DEVICE_INDEX)
    if REQUIRED_GPU_NAME_FRAGMENT not in name:
        raise RuntimeError(
            f"BG causal adapter requires the 5070 Ti on cuda:{REQUIRED_CUDA_DEVICE_INDEX}; "
            f"found {name!r}"
        )
    torch.cuda.set_device(REQUIRED_CUDA_DEVICE_INDEX)
    return torch.device(f"cuda:{REQUIRED_CUDA_DEVICE_INDEX}")


def load_model_tokenizer(device: str | None = None, dtype: str = "auto") -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is not None and str(device) != f"cuda:{REQUIRED_CUDA_DEVICE_INDEX}":
        raise RuntimeError(
            f"BG causal adapter is hardcoded to cuda:{REQUIRED_CUDA_DEVICE_INDEX}; got device={device!r}"
        )
    dev = require_adapter_cuda_device()
    if dtype == "auto":
        torch_dtype = torch.bfloat16
    elif dtype in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif dtype in {"fp16", "float16"}:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(dev)
    actual_device = next(model.parameters()).device
    if actual_device.type != "cuda" or int(actual_device.index or 0) != REQUIRED_CUDA_DEVICE_INDEX:
        raise RuntimeError(f"model is not on required cuda:{REQUIRED_CUDA_DEVICE_INDEX}; first parameter is on {actual_device}")
    actual_name = torch.cuda.get_device_name(actual_device.index or REQUIRED_CUDA_DEVICE_INDEX)
    if REQUIRED_GPU_NAME_FRAGMENT not in actual_name:
        raise RuntimeError(f"model landed on wrong GPU {actual_name!r}; expected fragment {REQUIRED_GPU_NAME_FRAGMENT!r}")
    model.eval()
    freeze_model(model)
    force_four_loops(model)
    return model, tokenizer, dev


def freeze_model(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def force_four_loops(model: Any) -> None:
    inner = getattr(model, "model", None)
    if hasattr(model, "config") and hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = NUM_LOOPS
    if hasattr(model, "config") and hasattr(model.config, "early_exit_threshold"):
        model.config.early_exit_threshold = 1.0
    if inner is not None and hasattr(inner, "total_ut_steps"):
        inner.total_ut_steps = NUM_LOOPS
    if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "total_ut_steps"):
        inner.config.total_ut_steps = NUM_LOOPS


def all_frozen(model: Any) -> bool:
    return all(not p.requires_grad for p in model.parameters())


def loop_scales(mode: str) -> tuple[list[int], dict[int, float]]:
    if mode == "single_loop_L1":
        return [1], {1: 1.0}
    if mode == "single_loop_L4":
        return [4], {4: 1.0}
    if mode == "multi_loop_uniform":
        return [1, 2, 3, 4], {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    if mode == "multi_loop_decayed":
        return [1, 2, 3, 4], {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
    raise ValueError(f"unsupported intervention mode {mode}")


def tensor_rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)


def rms_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / tensor_rms(x, eps=eps).to(device=x.device, dtype=x.dtype)


def build_teacher_forced_text(prompt: str, prefix_text: str) -> tuple[str, str]:
    prefix_part = f"{str(prompt).rstrip()}\n\nPartial answer attempt:\n{str(prefix_text).strip()}"
    full_text = prefix_part + "\nFINAL ANSWER:"
    return prefix_part, full_text


def option_token_ids(tokenizer: Any, options: dict[str, Any] | None = None) -> dict[str, int]:
    letters = sorted((options or {letter: "" for letter in OPTION_LETTERS}).keys())
    out: dict[str, int] = {}
    for letter in letters:
        ids = tokenizer.encode(str(letter), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"option letter {letter!r} is not single-token: {ids}")
        out[str(letter)] = int(ids[0])
    return out


def encode_example(tokenizer: Any, example: dict[str, Any], device: torch.device, max_length: int = 1536) -> dict[str, Any]:
    prefix_part, full_text = build_teacher_forced_text(example["prompt"], example["prefix_text"])
    prefix_ids = tokenizer(prefix_part, return_tensors="pt", truncation=True, max_length=max_length, padding=False)["input_ids"]
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    intervention_token_index = min(max(int(prefix_ids.shape[1]) - 1, 0), int(enc["input_ids"].shape[1]) - 1)
    answer_logit_token_index = int(enc["input_ids"].shape[1]) - 1
    ids_by_letter = option_token_ids(tokenizer, example.get("options"))
    option_letters = sorted(ids_by_letter)
    option_ids = torch.tensor([ids_by_letter[letter] for letter in option_letters], dtype=torch.long, device=device)
    correct_letter = str(example["correct_option"])
    correct_index = option_letters.index(correct_letter)
    wrong_indices = [idx for idx, letter in enumerate(option_letters) if letter != correct_letter]
    return {
        "enc": enc,
        "full_text": full_text,
        "prefix_part": prefix_part,
        "intervention_token_index": intervention_token_index,
        "answer_logit_token_index": answer_logit_token_index,
        "option_letters": option_letters,
        "option_ids": option_ids,
        "correct_index": correct_index,
        "wrong_indices": wrong_indices,
        "INTERVENTION_POSITION_WARNING": False,
        "intervention_position_kind": "prefix_last_token",
    }


def forward_logits(model: Any, enc: dict[str, torch.Tensor], logits_to_keep: int = 33) -> torch.Tensor:
    try:
        out = model(**enc, use_cache=False, return_dict=True, logits_to_keep=int(logits_to_keep))
    except TypeError:
        out = model(**enc, use_cache=False, return_dict=True)
    return out.logits


def option_metrics_from_logits(logits: torch.Tensor, encoded: dict[str, Any]) -> dict[str, Any]:
    answer_rel_idx = -1 if logits.shape[1] <= encoded["answer_logit_token_index"] else encoded["answer_logit_token_index"]
    option_logits = logits[0, answer_rel_idx, encoded["option_ids"]].float()
    correct = option_logits[encoded["correct_index"]]
    wrong = option_logits[encoded["wrong_indices"]]
    margin = correct - torch.logsumexp(wrong, dim=0)
    ce = F.cross_entropy(option_logits.unsqueeze(0), torch.tensor([encoded["correct_index"]], device=option_logits.device))
    pred_idx = int(torch.argmax(option_logits).detach().cpu().item())
    return {
        "option_logits": option_logits,
        "margin": margin,
        "ce": ce,
        "accuracy": 1.0 if pred_idx == int(encoded["correct_index"]) else 0.0,
        "predicted_option": encoded["option_letters"][pred_idx],
    }


def non_answer_kl(baseline_logits: torch.Tensor, intervened_logits: torch.Tensor, window: int = 16) -> torch.Tensor:
    b = baseline_logits.detach().float()
    i = intervened_logits.float()
    if b.shape[1] <= 1 or i.shape[1] <= 1:
        return torch.zeros((), device=i.device, dtype=torch.float32)
    b = b[:, :-1, :]
    i = i[:, :-1, :]
    if b.shape[1] > window:
        b = b[:, -window:, :]
        i = i[:, -window:, :]
    target = torch.softmax(b, dim=-1)
    log_probs = torch.log_softmax(i, dim=-1)
    return F.kl_div(log_probs, target, reduction="batchmean")


class AdapterInterventionHook:
    """Differentiable adapter layer hook for training/evaluation."""

    def __init__(
        self,
        model: Any,
        adapter: torch.nn.Module,
        *,
        alpha: float,
        intervention_mode: str = PRIMARY_MODE,
        target_layer: int = TARGET_LAYER,
        max_rms_fraction: float = 0.02,
    ) -> None:
        if float(alpha) > 0.02:
            raise ValueError("alpha must be <= 0.02")
        self.model = model
        self.adapter = adapter
        self.alpha = float(alpha)
        self.target_layer = int(target_layer)
        self.target_loops, self.scales = loop_scales(intervention_mode)
        self.intervention_mode = intervention_mode
        self.max_rms_fraction = float(max_rms_fraction)
        self.position = -1
        self.handle: Any | None = None
        self.forward_call_count = 0
        self.records: list[dict[str, Any]] = []
        self.delta_fraction_tensors: list[torch.Tensor] = []
        self.loop_index_source = "uninitialized"
        self.nan_or_inf = False

    def _target_module(self) -> Any:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("model does not expose model.layers")
        return layers[max(0, self.target_layer - 1)]

    def _loop(self, kwargs: dict[str, Any] | None) -> int:
        if kwargs and "current_ut" in kwargs:
            value = kwargs["current_ut"]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().item())
            self.loop_index_source = "current_ut"
            return int(value) + 1
        self.loop_index_source = "validated_call_counter"
        return ((self.forward_call_count - 1) % NUM_LOOPS) + 1

    def _hook(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any] | None, output: Any) -> Any:
        self.forward_call_count += 1
        loop = self._loop(kwargs)
        if loop not in self.target_loops or self.alpha == 0.0:
            return output
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.isfinite(tensor).all():
            self.nan_or_inf = True
            return output
        seq_len = int(tensor.shape[1])
        pos = self.position if self.position >= 0 else seq_len + self.position
        pos = max(0, min(pos, seq_len - 1))
        changed = tensor.clone()
        target = changed[:, pos, :]
        adapter_input = target.detach().float()
        direction = self.adapter(adapter_input)
        direction = rms_normalize(direction).to(device=target.device, dtype=target.dtype)
        alpha_eff = self.alpha * float(self.scales.get(loop, 1.0))
        delta = alpha_eff * tensor_rms(target).to(dtype=target.dtype) * direction
        delta_rms = tensor_rms(delta).to(dtype=torch.float32)
        hidden_rms = tensor_rms(target).to(dtype=torch.float32).clamp(min=1e-8)
        max_rms = self.max_rms_fraction * hidden_rms
        scale = torch.minimum(torch.ones_like(delta_rms), max_rms / delta_rms.clamp(min=1e-8))
        delta = delta * scale.to(device=delta.device, dtype=delta.dtype)
        changed[:, pos, :] = target + delta
        frac = (tensor_rms(delta).to(dtype=torch.float32) / hidden_rms).mean()
        self.delta_fraction_tensors.append(frac)
        self.records.append(
            {
                "layer": self.target_layer,
                "loop": loop,
                "position": int(pos),
                "alpha_eff": float(alpha_eff),
                "rms_fraction": float(frac.detach().cpu().item()),
                "forward_call_index": self.forward_call_count,
            }
        )
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        if isinstance(output, list):
            return [changed] + list(output[1:])
        return changed

    def apply(self, position: int) -> "AdapterInterventionHook":
        self.remove()
        self.position = int(position)
        module = self._target_module()
        try:
            self.handle = module.register_forward_hook(
                lambda mod, args, kwargs, out: self._hook(mod, args, kwargs, out),
                with_kwargs=True,
            )
        except TypeError:
            self.handle = module.register_forward_hook(lambda mod, args, out: self._hook(mod, args, None, out))
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        per_loop: dict[str, list[float]] = defaultdict(list)
        for row in self.records:
            per_loop[str(row["loop"])].append(float(row["rms_fraction"]))
        return {
            "hook_forward_call_count": self.forward_call_count,
            "hook_modifications": len(self.records),
            "hook_loop_index_source": self.loop_index_source,
            "nan_or_inf_activations": self.nan_or_inf,
            "activation_rms_change": avg(row["rms_fraction"] for row in self.records) or 0.0,
            "per_loop_activation_rms_change": {key: avg(vals) for key, vals in sorted(per_loop.items())},
            "records": self.records,
        }


@contextmanager
def adapter_hook(model: Any, adapter: torch.nn.Module, *, alpha: float, intervention_mode: str, position: int):
    hook = AdapterInterventionHook(model, adapter, alpha=alpha, intervention_mode=intervention_mode).apply(position)
    try:
        yield hook
    finally:
        hook.remove()


def load_stage1_mq_tasks() -> list[dict[str, Any]]:
    tasks = load_json(STAGE1_ROOT / "task_suite.json", {}).get("tasks") or []
    return [row for row in tasks if row.get("domain") in {"reasoning", "science"} and row.get("evaluator_type") == "mcq_letter"]


def continued_rows_for_task(task_id: str, prefix_length: int) -> list[dict[str, Any]]:
    rows = load_json(STAGE1_ROOT / "continued_prefixes.json", {}).get("continued_prefixes") or []
    return [
        row
        for row in rows
        if str(row.get("task_id")) == str(task_id)
        and int(row.get("prefix_length", -1)) == int(prefix_length)
        and bool(row.get("evaluable", True))
    ]


def split_task_ids(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for row in tasks:
        by_domain[str(row["domain"])].append(str(row["task_id"]))
    split = {"train": [], "val": [], "heldout": []}
    for domain, task_ids in sorted(by_domain.items()):
        ordered = sorted(task_ids)
        split["heldout"].extend(ordered[:4])
        split["val"].extend(ordered[4:8])
        split["train"].extend(ordered[8:])
    return {key: sorted(vals) for key, vals in split.items()}


def build_dataset_rows() -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    tasks = {str(row["task_id"]): row for row in load_stage1_mq_tasks()}
    split = split_task_ids(list(tasks.values()))
    split_by_task = {task_id: name for name, ids in split.items() for task_id in ids}
    rows: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        domain = str(task["domain"])
        prefix_length = 64 if domain == "reasoning" else 32
        for cont in continued_rows_for_task(task_id, prefix_length):
            branch_id = int(cont.get("branch_id", 0))
            correct = str(task.get("answer_key") or task.get("gold_answer") or "").upper()
            options = {str(k).upper(): str(v) for k, v in (task.get("options") or {}).items()}
            if correct not in options:
                continue
            rows.append(
                {
                    "example_id": f"{task_id}::p{prefix_length}::b{branch_id}",
                    "task_id": task_id,
                    "domain": domain,
                    "prompt": str(task.get("prompt") or task.get("question") or ""),
                    "prefix_text": str(cont.get("prefix_text") or ""),
                    "options": options,
                    "correct_option": correct,
                    "wrong_options": [letter for letter in sorted(options) if letter != correct],
                    "correct_answer_text": correct,
                    "answer_format": "FINAL ANSWER: <letter>",
                    "prefix_length": prefix_length,
                    "source_branch_id": branch_id,
                    "continuation_success": bool(cont.get("is_correct") or cont.get("evaluation", {}).get("success")),
                    "split": split_by_task.get(task_id, "train"),
                }
            )
    tokenization = {
        "option_token_format": "bare_single_letter",
        "gsm8k_included": False,
        "gsm8k_exclusion_reason": "numeric answer sequence targets are optional and omitted from this MCQ-only causal adapter run",
    }
    return rows, split, tokenization


def load_adapter_dataset() -> dict[str, Any]:
    return load_json(OUT_ROOT / "adapter_dataset.json", {})


def rows_for_split(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [row for row in dataset.get("examples", []) if row.get("split") == split]


def load_empirical_direction(name: str = "RAW_NONORM_READOUT") -> torch.Tensor | None:
    path = EMPIRICAL_ROOT / "directions.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for target in payload.get("targets", []):
        if str(target.get("domain")) == "reasoning" and int(target.get("prefix_length", -1)) == 64:
            for row in target.get("directions", []):
                if row.get("direction_name") == name:
                    vec = row["vector"].detach().flatten().float()
                    return vec / vec.pow(2).mean().sqrt().clamp(min=1e-8)
    return None
