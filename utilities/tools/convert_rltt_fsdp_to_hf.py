#!/usr/bin/env python3
"""Convert local RLTT FSDP/DTensor rank shards into a HF-style checkpoint.

The downloaded RLTT package contains:
  - huggingface-*.zip with config/tokenizer/custom model code;
  - model_world_size_4_rank_*.pt files, each a torch.save'd OrderedDict of
    DTensor values sharded along placement dim 0.

This script extracts the HF files, gathers the model shards on CPU, and writes
sharded safetensors plus model.safetensors.index.json so Transformers can load
the result with from_pretrained(local_path, trust_remote_code=True).
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Required for torch.load(..., weights_only=True) to allow DTensor globals.
import torch.distributed.tensor  # noqa: F401
from safetensors.torch import save_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ouro RLTT FSDP/DTensor rank shards to HF safetensors."
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="Directory containing model_world_size_*_rank_*.pt and huggingface zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shared/models/ouro_rltt_local"),
        help="Output HF-style model directory.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Expected FSDP world size. Defaults to fsdp_config.json or shard count.",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=1.8,
        help="Approximate max output safetensors shard size.",
    )
    parser.add_argument(
        "--dtype",
        choices=("preserve", "float32", "bfloat16", "float16"),
        default="preserve",
        help="Output tensor dtype. Preserve is lossless and matches downloaded shards.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def _find_hf_zip(downloads_dir: Path) -> Path:
    zips = sorted(downloads_dir.glob("huggingface*.zip"))
    if not zips:
        raise FileNotFoundError(f"no huggingface*.zip found in {downloads_dir}")
    return zips[0]


def _world_size(downloads_dir: Path, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    cfg_path = downloads_dir / "fsdp_config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        raw = cfg.get("world_size")
        if isinstance(raw, int) and raw > 0:
            return int(raw)
    shards = sorted(downloads_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no model_world_size_*_rank_*.pt found in {downloads_dir}")
    return len(shards)


def _extract_hf_files(hf_zip: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(hf_zip) as archive:
        for member in archive.infolist():
            name = member.filename
            if not name.startswith("huggingface/") or name.endswith("/"):
                continue
            rel = Path(name).relative_to("huggingface")
            target = output_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _load_rank_state_dicts(downloads_dir: Path, world_size: int) -> List[OrderedDict[str, Any]]:
    state_dicts: List[OrderedDict[str, Any]] = []
    for rank in range(int(world_size)):
        path = downloads_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[load] rank {rank}: {path}", flush=True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, OrderedDict):
            state = OrderedDict(state)
        state_dicts.append(state)
    keys0 = list(state_dicts[0].keys())
    for rank, state in enumerate(state_dicts[1:], start=1):
        if list(state.keys()) != keys0:
            raise ValueError(f"rank {rank} state_dict keys differ from rank 0")
    return state_dicts


def _placement_shard_dim(value: Any) -> Optional[int]:
    placements = getattr(value, "placements", None)
    if placements is None:
        return None
    dims: List[int] = []
    for placement in placements:
        dim = getattr(placement, "dim", None)
        if callable(dim):
            dim = dim()
        if isinstance(dim, int):
            dims.append(int(dim))
            continue
        if placement.__class__.__name__ == "Shard":
            raw = getattr(placement, "_dim", None)
            if isinstance(raw, int):
                dims.append(int(raw))
    if not dims:
        return None
    if len(set(dims)) != 1:
        raise ValueError(f"multi-dim DTensor placements are not supported: {placements!r}")
    return int(dims[0])


def _convert_dtype(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "preserve":
        return tensor
    target = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    return tensor.to(dtype=target)


def _local_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_local"):
        value = value.to_local()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected Tensor/DTensor, got {type(value)!r}")
    return value.detach().cpu().contiguous()


def _gather_tensor(key: str, values: Sequence[Any], dtype: str) -> torch.Tensor:
    first = values[0]
    shard_dim = _placement_shard_dim(first)
    locals_ = [_local_tensor(value) for value in values]
    if shard_dim is None:
        tensor = locals_[0]
    else:
        tensor = torch.cat(locals_, dim=int(shard_dim)).contiguous()
    expected_shape = tuple(getattr(first, "shape", tuple(tensor.shape)))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{key}: gathered shape {tuple(tensor.shape)} != expected {expected_shape}"
        )
    return _convert_dtype(tensor, dtype).contiguous()


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _save_shard(
    output_dir: Path,
    shard_idx: int,
    tensors: Dict[str, torch.Tensor],
    weight_map: Dict[str, str],
) -> int:
    filename = f"model-{shard_idx:05d}.safetensors"
    path = output_dir / filename
    size = sum(_tensor_nbytes(t) for t in tensors.values())
    print(f"[save] {filename}: {len(tensors)} tensors, {size / 1024**3:.3f} GiB", flush=True)
    save_file(tensors, str(path), metadata={"format": "pt"})
    for key in tensors:
        weight_map[key] = filename
    return int(size)


def _write_safetensors(
    state_dicts: Sequence[OrderedDict[str, Any]],
    output_dir: Path,
    *,
    max_shard_size_gb: float,
    dtype: str,
) -> None:
    max_bytes = max(1, int(float(max_shard_size_gb) * 1024**3))
    keys = list(state_dicts[0].keys())
    current: Dict[str, torch.Tensor] = {}
    current_size = 0
    total_size = 0
    shard_idx = 1
    weight_map: Dict[str, str] = {}

    for i, key in enumerate(keys, start=1):
        tensor = _gather_tensor(key, [state[key] for state in state_dicts], dtype)
        size = _tensor_nbytes(tensor)
        if current and current_size + size > max_bytes:
            total_size += _save_shard(output_dir, shard_idx, current, weight_map)
            shard_idx += 1
            current = {}
            current_size = 0
            gc.collect()
        current[key] = tensor
        current_size += size
        if i % 25 == 0 or i == len(keys):
            print(f"[gather] {i}/{len(keys)} tensors", flush=True)

    if current:
        total_size += _save_shard(output_dir, shard_idx, current, weight_map)

    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": weight_map,
    }
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[done] total safetensors payload: {total_size / 1024**3:.3f} GiB", flush=True)


def _patch_config(output_dir: Path, dtype: str) -> None:
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        return
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if dtype != "preserve":
        cfg["dtype"] = dtype
        cfg["torch_dtype"] = dtype
    # The project is pinned to Transformers 4.54.1. The downloaded config was
    # exported with 4.57.3 metadata, but the included remote-code files are what
    # define the custom model. Keep the metadata aligned with the runtime pin so
    # local loading does not imply a package upgrade.
    cfg["transformers_version"] = "4.54.1"
    cfg.setdefault("local_conversion", {})["source_format"] = "fsdp_dtensor_world_size_4"
    with cfg_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = _parse_args()
    downloads_dir = args.downloads_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world_size = _world_size(downloads_dir, args.world_size)
    hf_zip = _find_hf_zip(downloads_dir)
    print(f"[config] downloads={downloads_dir}", flush=True)
    print(f"[config] output={output_dir}", flush=True)
    print(f"[config] world_size={world_size}", flush=True)
    print(f"[config] hf_zip={hf_zip.name}", flush=True)
    print(f"[config] dtype={args.dtype}", flush=True)

    _extract_hf_files(hf_zip, output_dir)
    _patch_config(output_dir, args.dtype)
    state_dicts = _load_rank_state_dicts(downloads_dir, world_size)
    _write_safetensors(
        state_dicts,
        output_dir,
        max_shard_size_gb=args.max_shard_size_gb,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
