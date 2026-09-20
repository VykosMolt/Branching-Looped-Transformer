"""Shared helpers for BG sequence-level adapter experiments."""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
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

QUICK_OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_sequence_adapter_quick_preflight_2026-05-18"
OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_sequence_level_adapter_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
PRECONSOLIDATION_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_preconsolidation_control_probes_2026-05-18"
EMPIRICAL_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_empirical_steering_direction_2026-05-18"
CAUSAL_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_causal_intervention_adapter_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518
HIDDEN_DIM = 2048
PRIMARY_MODE = "multi_loop_decayed"
COMPARISON_MODE = "single_loop_L1"
TARGET_LAYER = 36
SAFE_ALPHAS = [0.005, 0.01, 0.02]
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def out_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: str | Path, default: Any = None) -> Any:
    p = out_path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_md(path: str | Path, lines: Iterable[str]) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def append_once(path: str | Path, title: str, lines: list[str]) -> None:
    p = out_path(path)
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


def rate(values: Iterable[Any]) -> float | None:
    vals = [bool(v) for v in values]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def set_seed(seed: int = SEED) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_model_tokenizer(dtype: str = "auto") -> tuple[Any, Any, torch.device]:
    from bg_causal_adapter_common import load_model_tokenizer as _load

    return _load(dtype=dtype)


def all_frozen(model: Any) -> bool:
    return all(not p.requires_grad for p in model.parameters())


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def options_text(options: dict[str, Any]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def format_mcq_prompt(task: dict[str, Any]) -> str:
    question = str(task.get("question") or task.get("prompt") or "").strip()
    options = {str(k).upper(): str(v) for k, v in (task.get("options") or {}).items()}
    return (
        f"Question: {question}\n"
        "Options:\n"
        f"{options_text(options)}\n"
        "Think briefly if needed.\n"
        "FINAL ANSWER:"
    )


def parse_mcq_answer(text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = str(text or "")
    letters = sorted(str(k).upper() for k in (options or {letter: "" for letter in OPTION_LETTERS}))
    allowed = "".join(re.escape(letter) for letter in letters)
    if not raw.strip():
        return {"parsed_answer": None, "parse_success": False, "parse_failure_reason": "empty_output"}
    final_matches = list(re.finditer(r"FINAL\s+ANSWER\s*:", raw, flags=re.IGNORECASE))
    if final_matches:
        tail = raw[final_matches[-1].end() :]
        match = re.search(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", tail, flags=re.IGNORECASE)
        if match:
            return {
                "parsed_answer": match.group(1).upper(),
                "parse_success": True,
                "parse_failure_reason": "",
                "parser_path": "after_final_answer",
            }
        return {
            "parsed_answer": None,
            "parse_success": False,
            "parse_failure_reason": "final_answer_marker_without_option",
            "parser_path": "after_final_answer",
        }
    fallback = re.findall(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", raw, flags=re.IGNORECASE)
    if fallback:
        return {
            "parsed_answer": str(fallback[-1]).upper(),
            "parse_success": True,
            "parse_failure_reason": "",
            "parser_path": "last_standalone_option",
        }
    return {
        "parsed_answer": None,
        "parse_success": False,
        "parse_failure_reason": "no_standalone_option",
        "parser_path": "fallback",
    }


def repetition_rate(text: str, ngram: int = 4) -> float:
    toks = re.findall(r"\S+", str(text or "").lower())
    if len(toks) < ngram * 2:
        return 0.0
    grams = [tuple(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(grams), 1)


def reward_for_output(task: dict[str, Any], text: str, error: str = "") -> dict[str, Any]:
    if error:
        return {
            "reward": -1.0,
            "parsed_answer": None,
            "parse_success": False,
            "correct": False,
            "parse_failure_reason": "generation_error",
            "empty_output": False,
            "severe_repetition": False,
        }
    if not str(text or "").strip():
        return {
            "reward": -0.5,
            "parsed_answer": None,
            "parse_success": False,
            "correct": False,
            "parse_failure_reason": "empty_output",
            "empty_output": True,
            "severe_repetition": False,
        }
    parsed = parse_mcq_answer(text, task.get("options"))
    rep = repetition_rate(text) >= 0.30
    correct = bool(parsed["parse_success"] and parsed.get("parsed_answer") == str(task.get("correct_option") or task.get("answer_key")).upper())
    if rep:
        reward = -0.3
    elif not parsed["parse_success"]:
        reward = -0.2
    else:
        reward = 1.0 if correct else 0.0
    return {
        "reward": float(reward),
        "parsed_answer": parsed.get("parsed_answer"),
        "parse_success": bool(parsed["parse_success"]),
        "correct": correct,
        "parse_failure_reason": parsed.get("parse_failure_reason", ""),
        "parser_path": parsed.get("parser_path", ""),
        "empty_output": False,
        "severe_repetition": rep,
    }


def load_stage1_mcq_tasks() -> list[dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "task_suite.json", {})
    tasks: list[dict[str, Any]] = []
    for row in payload.get("tasks") or []:
        domain = str(row.get("domain"))
        options = {str(k).upper(): str(v) for k, v in (row.get("options") or {}).items()}
        correct = str(row.get("answer_key") or row.get("gold_answer") or "").upper()
        if domain not in {"reasoning", "science"}:
            continue
        if str(row.get("evaluator_type")) != "mcq_letter":
            continue
        if correct not in options:
            continue
        task = dict(row)
        task["options"] = options
        task["correct_option"] = correct
        task["prompt"] = format_mcq_prompt(task)
        task["parser_type"] = "mcq_final_answer_letter"
        tasks.append(task)
    return sorted(tasks, key=lambda r: (str(r.get("domain")), str(r.get("task_id"))))


def split_task_ids(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    split = {"train": [], "val": [], "heldout": []}
    by_domain: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        by_domain[str(task["domain"])].append(str(task["task_id"]))
    for _domain, ids in sorted(by_domain.items()):
        ordered = sorted(ids)
        split["heldout"].extend(ordered[:6])
        split["val"].extend(ordered[6:10])
        split["train"].extend(ordered[10:])
    return {key: sorted(vals) for key, vals in split.items()}


def build_sequence_dataset_rows() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    tasks = load_stage1_mcq_tasks()
    split = split_task_ids(tasks)
    split_by_task = {task_id: name for name, ids in split.items() for task_id in ids}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        domain = str(task["domain"])
        prefix_length = 64 if domain == "reasoning" else 32
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "domain": domain,
                "source_dataset": str(task.get("source_dataset") or ""),
                "prompt": str(task["prompt"]),
                "question": str(task.get("question") or ""),
                "prefix_text": "",
                "options": task["options"],
                "correct_option": task["correct_option"],
                "expected_answer_text": str(task.get("gold_answer") or task["correct_option"]),
                "parser_type": "mcq_final_answer_letter",
                "prefix_length": prefix_length,
                "split": split_by_task.get(str(task["task_id"]), "train"),
                "intervention_position_kind": "prompt_last_token",
                "INTERVENTION_POSITION_WARNING": True,
            }
        )
    return rows, split


def load_sequence_dataset() -> dict[str, Any]:
    return load_json(OUT_ROOT / "sequence_adapter_dataset.json", {})


def rows_for_split(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [row for row in dataset.get("tasks", []) if row.get("split") == split]


def select_balanced_tasks(tasks: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task.get("domain"))].append(task)
    selected: list[dict[str, Any]] = []
    domains = sorted(grouped)
    while len(selected) < int(total):
        progressed = False
        for domain in domains:
            if grouped[domain]:
                selected.append(grouped[domain].pop(0))
                progressed = True
                if len(selected) >= int(total):
                    break
        if not progressed:
            break
    return selected


def generation_prompt(task: dict[str, Any]) -> str:
    return str(task.get("prompt") or format_mcq_prompt(task))


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device, max_length: int = 1536) -> dict[str, torch.Tensor]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    return enc


class AdapterGenerationHook:
    def __init__(self, model: Any, adapter: torch.nn.Module, alpha: float, mode: str, position: int, max_alpha: float = 0.02) -> None:
        from src.evaluator.bg_sequence_adapter import sequence_adapter_hook

        self.inner = sequence_adapter_hook(
            model,
            adapter,
            alpha=float(alpha),
            intervention_mode=mode,
            position=int(position),
            max_rms_fraction=0.02,
            max_alpha=float(max_alpha),
        )
        self.hook = None

    def __enter__(self) -> "AdapterGenerationHook":
        self.hook = self.inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.inner.__exit__(exc_type, exc, tb)

    def diagnostics(self) -> dict[str, Any]:
        return self.hook.diagnostics() if self.hook is not None else {}


class StaticGenerationHook:
    def __init__(self, model: Any, direction: torch.Tensor, alpha: float, mode: str, position: int) -> None:
        from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode

        spec = build_intervention_mode(mode, float(alpha))
        self.hook = BGLayerHookSteering(
            model,
            target_layer=TARGET_LAYER,
            target_loops=spec["target_loops"],
            direction=rms_direction(direction),
            alpha=float(alpha),
            loop_alpha_scales=spec["loop_alpha_scales"],
            max_rms_fraction=0.02,
            direction_normalization="rms",
        )
        self.position = int(position)

    def __enter__(self) -> "StaticGenerationHook":
        self.hook.apply(position=self.position)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.hook.remove()

    def diagnostics(self) -> dict[str, Any]:
        raw = self.hook.diagnostics()
        return {
            "hook_forward_call_count": raw.get("forward_call_count", 0),
            "hook_modifications": raw.get("modifications", 0),
            "hook_loop_index_source": raw.get("loop_index_source", ""),
            "activation_rms_change": raw.get("activation_rms_change", 0.0),
            "per_loop_activation_rms_change": raw.get("per_loop_activation_rms_change", {}),
            "nan_or_inf_activations": raw.get("nan_or_inf", False),
        }


def generate_completion(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    hook_context: Any | None = None,
) -> dict[str, Any]:
    enc = encode_prompt(tokenizer, prompt, device)
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    started = time.time()
    error = ""
    generated = enc["input_ids"]
    ctx = hook_context if hook_context is not None else nullcontext()
    diagnostics: dict[str, Any] = {}
    try:
        with ctx as active_hook:
            kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "pad_token_id": pad_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False,
            }
            if do_sample:
                kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
            with torch.no_grad():
                generated = model.generate(**enc, **kwargs)
            if active_hook is not None and hasattr(active_hook, "diagnostics"):
                diagnostics = active_hook.diagnostics()
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
        generated = enc["input_ids"]
    new_ids = generated[0, prompt_len:].detach()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    cleanup_cuda()
    return {
        "prompt_len": prompt_len,
        "prompt_input_ids": enc["input_ids"].detach(),
        "new_ids": new_ids,
        "output_text": text,
        "output_length": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= int(max_new_tokens),
        "seconds": round(time.time() - started, 3),
        "generation_error": error,
        "diagnostics": diagnostics,
    }


def rms_direction(vec: torch.Tensor) -> torch.Tensor:
    flat = vec.detach().flatten().to(dtype=torch.float32, device="cpu")
    return flat / flat.pow(2).mean().sqrt().clamp(min=1e-8)


def random_rms_direction(seed: int, dim: int = HIDDEN_DIM) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return rms_direction(torch.randn(int(dim), generator=gen))


def load_empirical_direction(name: str = "RAW_NONORM_READOUT") -> torch.Tensor | None:
    path = EMPIRICAL_ROOT / "directions.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for target in payload.get("targets", []):
        if str(target.get("domain")) in {"reasoning", "science"}:
            for row in target.get("directions", []):
                if str(row.get("direction_name")) == name and int(row["vector"].numel()) == HIDDEN_DIM:
                    return rms_direction(row["vector"])
    return None


def load_teacher_adapter(device: torch.device) -> torch.nn.Module | None:
    path = CAUSAL_ROOT / "adapter_checkpoints/best_adapter.pt"
    if not path.exists():
        return None
    from src.evaluator.bg_sequence_adapter import LowRankDeltaAdapter

    adapter = LowRankDeltaAdapter(rank=32).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.eval()
    return adapter


def load_sequence_adapter(device: torch.device, path: str | Path | None = None) -> torch.nn.Module | None:
    ckpt = out_path(path) if path is not None else OUT_ROOT / "sequence_adapter_checkpoints/best_sequence_adapter.pt"
    if not ckpt.exists():
        return None
    from src.evaluator.bg_sequence_adapter import build_sequence_adapter

    payload = torch.load(ckpt, map_location=device, weights_only=False)
    adapter = build_sequence_adapter(
        kind=str(payload.get("adapter_kind", "low_rank")),
        rank=int(payload.get("adapter_rank", 32)),
        device=device,
    )
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.eval()
    return adapter


def run_generation_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    task: dict[str, Any],
    *,
    method: str,
    seed: int,
    alpha: float = 0.0,
    mode: str = PRIMARY_MODE,
    adapter: torch.nn.Module | None = None,
    direction: torch.Tensor | None = None,
    max_new_tokens: int = 96,
    do_sample: bool = False,
    sample_index: int = 0,
    max_alpha: float = 0.02,
) -> dict[str, Any]:
    prompt = generation_prompt(task)
    prompt_len = int(encode_prompt(tokenizer, prompt, device)["input_ids"].shape[1])
    position = max(0, prompt_len - 1)
    hook_context = None
    if method in {"trained_sequence_adapter", "teacher_forced_adapter_checkpoint", "sanity_adapter"}:
        if adapter is None:
            raise ValueError(f"{method} requires adapter")
        hook_context = AdapterGenerationHook(model, adapter, alpha, mode, position, max_alpha=max_alpha)
    elif method in {"random_same_rms", "raw_nonorm_static", "empirical_mean_diff"}:
        if direction is None:
            raise ValueError(f"{method} requires direction")
        hook_context = StaticGenerationHook(model, direction, alpha, mode, position)
    result = generate_completion(
        model,
        tokenizer,
        device,
        prompt,
        seed=seed,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        hook_context=hook_context,
    )
    reward = reward_for_output(task, result["output_text"], result["generation_error"])
    diagnostics = result.get("diagnostics") or {}
    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "source": task.get("source_dataset", ""),
        "method": method,
        "mode": mode,
        "alpha": float(alpha),
        "seed": int(seed),
        "sample_index": int(sample_index),
        "decode": "sampled" if do_sample else "deterministic",
        "gold_answer": task.get("correct_option") or task.get("answer_key"),
        "generated_output": result["output_text"],
        "parsed_answer": reward["parsed_answer"],
        "correct": bool(reward["correct"]),
        "parse_success": bool(reward["parse_success"]),
        "parse_failure_reason": reward["parse_failure_reason"],
        "reward": float(reward["reward"]),
        "output_length": int(result["output_length"]),
        "hit_max_tokens": bool(result["hit_max_tokens"]),
        "empty_output": bool(reward["empty_output"]),
        "repetition_rate": repetition_rate(result["output_text"]),
        "severe_repetition": bool(reward["severe_repetition"]),
        "generation_seconds": result["seconds"],
        "generation_error": result["generation_error"],
        "cuda_error": "cuda" in str(result["generation_error"]).lower(),
        "use_cache": False,
        "intervention_position_kind": task.get("intervention_position_kind", "prompt_last_token"),
        "INTERVENTION_POSITION_WARNING": bool(task.get("INTERVENTION_POSITION_WARNING", True)),
        "intervention_token_index": position,
        "hook_forward_call_count": diagnostics.get("hook_forward_call_count", 0),
        "hook_modifications": diagnostics.get("hook_modifications", 0),
        "hook_loop_index_source": diagnostics.get("hook_loop_index_source", diagnostics.get("loop_index_source", "")),
        "activation_rms_change": diagnostics.get("activation_rms_change", 0.0),
        "per_loop_activation_rms_change": diagnostics.get("per_loop_activation_rms_change", {}),
        "nan_or_inf_activations": bool(diagnostics.get("nan_or_inf_activations", diagnostics.get("nan_or_inf", False))),
    }


def aggregate_rows(rows: list[dict[str, Any]], by: tuple[str, ...] = ("method", "alpha", "decode")) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in by)].append(row)
    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items(), key=lambda item: str(item[0])):
        label = "|".join(f"{k}={v}" for k, v in zip(by, key))
        out[label] = {
            **{k: v for k, v in zip(by, key)},
            "n": len(vals),
            "reward_mean": avg(row.get("reward") for row in vals),
            "reward_std": pstdev([finite(row.get("reward")) for row in vals]),
            "success_rate": avg(row.get("correct") for row in vals),
            "parse_rate": avg(row.get("parse_success") for row in vals),
            "repetition_rate": avg(row.get("repetition_rate") for row in vals),
            "empty_output_rate": avg(row.get("empty_output") for row in vals),
            "hit_max_tokens_rate": avg(row.get("hit_max_tokens") for row in vals),
            "output_length_mean": avg(row.get("output_length") for row in vals),
            "activation_rms_change": avg(row.get("activation_rms_change") for row in vals),
            "cuda_error_count": sum(1 for row in vals if row.get("cuda_error")),
            "nan_or_inf_count": sum(1 for row in vals if row.get("nan_or_inf_activations")),
        }
    return out


def pstdev(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5


def task_reward_variance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id"))].append(finite(row.get("reward")))
    nonzero = {task: vals for task, vals in by_task.items() if len(set(vals)) > 1}
    return {
        "task_count": len(by_task),
        "tasks_with_nonzero_reward_variance": len(nonzero),
        "nonzero_reward_variance_rate": len(nonzero) / max(len(by_task), 1),
        "invariant_task_count": len(by_task) - len(nonzero),
        "moved_task_ids": sorted(nonzero),
    }


def policy_logprob_terms(
    model: Any,
    adapter: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    new_ids: torch.Tensor,
    *,
    alpha: float,
    mode: str,
    max_alpha: float = 0.02,
    lambda_kl: float = 0.02,
    lambda_delta: float = 0.05,
) -> dict[str, Any]:
    if new_ids.numel() == 0:
        zero = torch.zeros((), device=prompt_input_ids.device, dtype=torch.float32)
        return {"objective": zero, "sum_logprob": zero, "kl": zero, "entropy": zero, "delta_loss": zero, "diagnostics": {}}
    prompt_ids = prompt_input_ids.to(device=prompt_input_ids.device)
    full_ids = torch.cat([prompt_ids, new_ids.view(1, -1).to(prompt_ids.device)], dim=1)
    attention_mask = torch.ones_like(full_ids, device=full_ids.device)
    prompt_len = int(prompt_ids.shape[1])
    position = max(0, prompt_len - 1)
    enc = {"input_ids": full_ids, "attention_mask": attention_mask}
    with torch.no_grad():
        base_out = model(**enc, use_cache=False, return_dict=True)
        base_logits = base_out.logits.detach().float()
    from src.evaluator.bg_sequence_adapter import sequence_adapter_hook

    with sequence_adapter_hook(
        model,
        adapter,
        alpha=float(alpha),
        intervention_mode=mode,
        position=position,
        max_rms_fraction=0.02,
        max_alpha=float(max_alpha),
    ) as hook:
        out = model(**enc, use_cache=False, return_dict=True)
        logits = out.logits.float()
        diagnostics = hook.diagnostics()
        delta_loss = torch.stack(hook.delta_fraction_tensors).pow(2).mean() if hook.delta_fraction_tensors else torch.zeros((), device=full_ids.device)
    target_positions = torch.arange(prompt_len, full_ids.shape[1], device=full_ids.device)
    logit_positions = target_positions - 1
    target_ids = full_ids[0, target_positions]
    token_logits = logits[0, logit_positions, :]
    log_probs = F.log_softmax(token_logits, dim=-1)
    token_logprobs = log_probs.gather(1, target_ids.view(-1, 1)).squeeze(1)
    sum_logprob = token_logprobs.sum()
    probs = torch.softmax(token_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    base_probs = torch.softmax(base_logits[0, logit_positions, :], dim=-1)
    kl = F.kl_div(log_probs, base_probs, reduction="batchmean")
    stability_penalty = float(lambda_kl) * kl + float(lambda_delta) * delta_loss
    return {
        "objective": sum_logprob - stability_penalty,
        "sum_logprob": sum_logprob,
        "kl": kl,
        "entropy": entropy,
        "delta_loss": delta_loss,
        "diagnostics": diagnostics,
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 40) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:max_rows]:
        vals = [str(row.get(col, "")) for col in columns]
        vals = [val.replace("\n", " ")[:160] for val in vals]
        lines.append("| " + " | ".join(vals) + " |")
    return lines
