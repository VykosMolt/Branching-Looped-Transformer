"""Tiny causal intervention adapters for BG layer-hook experiments.

This module is deliberately lightweight: it defines adapter modules and tensor
utilities only.  It does not load Ouro, tokenizer files, checkpoints, or BG
heads at import time.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_DIM = 2048


def rms(x: torch.Tensor, dim: int = -1, keepdim: bool = True, eps: float = 1e-8) -> torch.Tensor:
    return x.float().pow(2).mean(dim=dim, keepdim=keepdim).sqrt().clamp(min=float(eps))


def rms_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / rms(v, dim=-1, keepdim=True, eps=eps).to(device=v.device, dtype=v.dtype)


def clip_delta_rms(delta: torch.Tensor, h: torch.Tensor, max_fraction: float = 0.02, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip ``delta`` so its RMS is at most ``max_fraction * rms(h)``.

    Returns the clipped delta and the resulting per-sample RMS fraction.
    """
    if max_fraction <= 0:
        raise ValueError("max_fraction must be positive")
    h_rms = rms(h, dim=-1, keepdim=True, eps=eps).to(device=delta.device, dtype=delta.dtype)
    delta_rms = rms(delta, dim=-1, keepdim=True, eps=eps).to(device=delta.device, dtype=delta.dtype)
    max_rms = float(max_fraction) * h_rms
    scale = torch.minimum(torch.ones_like(delta_rms), max_rms / delta_rms.clamp(min=eps))
    clipped = delta * scale
    frac = rms(clipped, dim=-1, keepdim=False, eps=eps).to(dtype=torch.float32) / h_rms.squeeze(-1).to(dtype=torch.float32).clamp(min=eps)
    return clipped, frac


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


class Rank1GatedDirectionAdapter(nn.Module):
    """A learned global direction with an input-dependent scalar gate."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, gate_hidden: int = 128, zero_init: bool = False) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.direction = nn.Parameter(torch.empty(self.hidden_dim))
        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), 1),
        )
        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init: bool = False) -> None:
        if zero_init:
            nn.init.zeros_(self.direction)
            for module in self.gate:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
            return
        nn.init.normal_(self.direction, mean=0.0, std=1.0 / math.sqrt(self.hidden_dim))
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_at_position: torch.Tensor) -> torch.Tensor:
        h = hidden_at_position.float()
        gate = torch.sigmoid(self.gate(h))
        direction = rms_normalize(self.direction.to(device=h.device, dtype=h.dtype).unsqueeze(0))
        return gate * direction.expand(h.shape[0], -1)


class LowRankDeltaAdapter(nn.Module):
    """A small low-rank non-linear delta generator."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, rank: int = 32, zero_init: bool = False) -> None:
        super().__init__()
        if int(rank) not in {16, 32, 64}:
            raise ValueError("rank must be one of {16, 32, 64}")
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, self.rank)
        self.up = nn.Linear(self.rank, self.hidden_dim)
        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init: bool = False) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            nn.init.xavier_uniform_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden_at_position: torch.Tensor) -> torch.Tensor:
        h = hidden_at_position.float()
        z = self.down(self.norm(h))
        delta = self.up(F.gelu(z))
        return rms_normalize(delta)


class HyperDirectionAdapter(nn.Module):
    """Mixture over learned basis directions with input-dependent weights."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_basis: int = 8, hidden: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.basis = nn.Parameter(torch.randn(self.num_basis, self.hidden_dim) / math.sqrt(self.hidden_dim))
        self.router = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.num_basis),
        )

    def forward(self, hidden_at_position: torch.Tensor) -> torch.Tensor:
        h = hidden_at_position.float()
        weights = torch.softmax(self.router(h), dim=-1)
        basis = rms_normalize(self.basis.to(device=h.device, dtype=h.dtype))
        return rms_normalize(weights @ basis)


def assert_adapter_budget(adapter: nn.Module, max_params: int = 2_000_000) -> None:
    count = parameter_count(adapter)
    if count > int(max_params):
        raise ValueError(f"adapter has {count} parameters, exceeds max_params={max_params}")


def trainable_parameters(modules: Iterable[nn.Module]) -> int:
    return int(sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad))

