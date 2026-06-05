"""Read-only BG Phase 1 branch-selection controller.

The controller loads existing tiny pairwise heads and applies the locked v8.1
domain routing policy over already captured pooled candidate features. It does
not generate candidates, train heads, capture features, or modify checkpoints.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEAD_REGISTRY_PATH = PROJECT_ROOT / "artifacts/reports/probes/bg_head_registry_2026-05-17.pt"
DEFAULT_MIXED_HEADS_PATH = PROJECT_ROOT / "artifacts/reports/probes/mixed_domain_tiny_heads_2026-05-17.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "artifacts/reports/probes/bg_policy_sim_eval_bundle_2026-05-17.pt"

TAP_LAYERS = (24, 36, 47)
LAYER_TO_POS = {layer: idx for idx, layer in enumerate(TAP_LAYERS)}
NUM_LOOPS = 4
HIDDEN_DIM = 2048

SUPPORTED_CONFIGS = (
    "24_L1",
    "24_L4",
    "24_mean",
    "36_L1",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "47_concat_L1_L4",
    "47_concat_all_loops",
)
SUPPORTED_MODES = ("conservative", "experimental_vote", "code_backup", "diagnostic_all")

HH_DOMAINS = {"hh", "preference", "unknown"}
OBJECTIVE_DOMAINS = {"code", "strict_clean_code", "reasoning", "science", "math", "gsm8k", "objective"}

LOCKED_HEAD_REQUESTS = {
    "hh_general": {
        "source": "registry",
        "group": "HH",
        "config": "47_concat_L1_L4",
        "architecture": "AntisymLinearNoNorm",
        "family": "HH",
        "domain_role": "HH/preference/unknown/default semantic preference head",
    },
    "objective_mixed": {
        "source": "mixed",
        "group": "MIX_CODE_REASONING",
        "config": "36_L4",
        "architecture": "AntisymLinearNoNorm",
        "family": "MIX_CODE_REASONING",
        "domain_role": "default objective selector for code/reasoning/science/math/GSM8K/objective branch selection",
    },
    "code_specialist_backup": {
        "source": "registry",
        "group": "CODE",
        "config": "36_L4",
        "architecture": "AntisymLinear",
        "family": "CODE",
        "domain_role": "strict-clean code backup/diagnostic/escalation, not default conservative route",
    },
}

ROLE_TO_HEAD_NAME = {
    "HH_GENERAL": "hh_general",
    "OBJECTIVE_MIXED_PRIMARY": "objective_mixed",
    "CODE_SPECIALIST": "code_specialist_backup",
}


class AntisymLinear(nn.Module):
    """LayerNorm(no affine)(left - right) -> Linear(no bias)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.linear = nn.Linear(self.dim, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(left - right)).squeeze(-1)


class AntisymLinearNoNorm(nn.Module):
    """Linear(left - right)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.linear(left - right).squeeze(-1)


HEAD_CLASSES = {
    "AntisymLinear": AntisymLinear,
    "AntisymLinearNoNorm": AntisymLinearNoNorm,
}


@dataclass
class BGHeadSpec:
    name: str
    family: str
    config: str
    architecture: str
    artifact_path: str | None
    domain_role: str
    dim: int
    calibration_std: float | None = None


def _repo_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def _resolve_path(path: str | Path | None, default: Path | None = None) -> Path | None:
    if path is None:
        return default
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def config_dim(config: str) -> int:
    if config not in SUPPORTED_CONFIGS:
        raise ValueError(f"unknown config: {config}; supported configs: {', '.join(SUPPORTED_CONFIGS)}")
    if config == "47_concat_L1_L4":
        return HIDDEN_DIM * 2
    if config == "47_concat_all_loops":
        return HIDDEN_DIM * 4
    return HIDDEN_DIM


def _validate_pooled(pooled: torch.Tensor, *, context: str = "pooled") -> torch.Tensor:
    if not isinstance(pooled, torch.Tensor):
        raise ValueError(f"{context} must be a torch.Tensor, got {type(pooled).__name__}")
    expected = (len(TAP_LAYERS), NUM_LOOPS, HIDDEN_DIM)
    actual = tuple(int(x) for x in pooled.shape)
    if actual != expected:
        raise ValueError(f"{context} has shape {actual}; expected {expected} ([layers=3, loops=4, hidden=2048])")
    return pooled


def config_vector(pooled: torch.Tensor, config: str) -> torch.Tensor:
    """Return one config vector from a pooled candidate tensor [3, 4, 2048]."""
    pooled = _validate_pooled(pooled)
    if config == "24_L1":
        return pooled[LAYER_TO_POS[24], 0]
    if config == "24_L4":
        return pooled[LAYER_TO_POS[24], 3]
    if config == "24_mean":
        return pooled[LAYER_TO_POS[24]].mean(dim=0)
    if config == "36_L1":
        return pooled[LAYER_TO_POS[36], 0]
    if config == "36_L4":
        return pooled[LAYER_TO_POS[36], 3]
    if config == "36_mean":
        return pooled[LAYER_TO_POS[36]].mean(dim=0)
    if config == "47_L4":
        return pooled[LAYER_TO_POS[47], 3]
    if config == "47_mean":
        return pooled[LAYER_TO_POS[47]].mean(dim=0)
    if config == "47_concat_L1_L4":
        return torch.cat([pooled[LAYER_TO_POS[47], 0], pooled[LAYER_TO_POS[47], 3]], dim=-1)
    if config == "47_concat_all_loops":
        return torch.cat([pooled[LAYER_TO_POS[47], i] for i in range(NUM_LOOPS)], dim=-1)
    raise ValueError(f"unknown config: {config}; supported configs: {', '.join(SUPPORTED_CONFIGS)}")


def normalize_domain_hint(domain_hint: str | None) -> str:
    raw = "unknown" if domain_hint is None else str(domain_hint).strip().lower()
    key = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "": "unknown",
        "none": "unknown",
        "unknown": "unknown",
        "default": "unknown",
        "hh": "hh",
        "hh_rlhf": "hh",
        "hh_heldout20": "hh",
        "hh_200_diagnostic": "hh",
        "preference": "preference",
        "preferences": "preference",
        "pref": "preference",
        "human_preference": "preference",
        "code": "code",
        "code_runnable": "code",
        "code_runnable_diagnostic": "code",
        "strict_clean": "strict_clean_code",
        "code_strict_clean": "strict_clean_code",
        "strict_clean_code": "strict_clean_code",
        "code_strict_clean_all16": "strict_clean_code",
        "reasoning": "reasoning",
        "reasoning_natural": "reasoning",
        "reasoning_natural_distractor": "reasoning",
        "reasoning_trace": "reasoning",
        "science": "science",
        "science_overall": "science",
        "science_biology": "science",
        "science_chemistry": "science",
        "science_medicine": "science",
        "science_general": "science",
        "science_other": "science",
        "biology": "science",
        "chemistry": "science",
        "medicine": "science",
        "math": "math",
        "gsm8k": "gsm8k",
        "clean_gsm8k": "gsm8k",
        "clean_gsm8k_expanded": "gsm8k",
        "objective": "objective",
        "objective_qa": "objective",
        "qa": "objective",
    }
    return aliases.get(key, "unknown")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    return json.loads(path.read_text(encoding="utf-8"))


def _head_key(group: str, config: str, architecture: str) -> str:
    return f"{group}::{config}::{architecture}"


def _row_group(row: dict[str, Any]) -> str:
    return str(row.get("head_group") or row.get("head_family") or "")


def _find_head_row(rows: Sequence[dict[str, Any]], *, group: str, config: str, architecture: str) -> dict[str, Any] | None:
    for row in rows:
        if _row_group(row) == group and row.get("config") == config and row.get("architecture") == architecture:
            return row
    return None


def _build_head(architecture: str, dim: int, state_dict: dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    if architecture not in HEAD_CLASSES:
        raise RuntimeError(f"unsupported head architecture {architecture}; supported: {', '.join(HEAD_CLASSES)}")
    head = HEAD_CLASSES[architecture](int(dim))
    head.load_state_dict(state_dict)
    head.to(device=device, dtype=torch.float32)
    head.eval()
    return head


def _sign(score: float) -> int:
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


class BGController:
    """Thin read-only controller over existing pairwise BG heads."""

    def __init__(
        self,
        *,
        heads: dict[str, nn.Module],
        specs: dict[str, BGHeadSpec],
        device: str | torch.device = "cpu",
        calibration_by_domain: dict[str, dict[str, float]] | None = None,
        role_keys: dict[str, str] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.heads = heads
        self.specs = specs
        self.calibration_by_domain = calibration_by_domain or {}
        self.role_keys = role_keys or {}
        for head in self.heads.values():
            head.to(self.device)
            head.eval()

    @classmethod
    def from_artifacts(
        cls,
        head_registry_path: str | Path | None = None,
        mixed_heads_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> "BGController":
        registry_path = _resolve_path(head_registry_path, DEFAULT_HEAD_REGISTRY_PATH)
        mixed_path = _resolve_path(mixed_heads_path, DEFAULT_MIXED_HEADS_PATH)
        cfg_path = _resolve_path(config_path, DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)
        missing_paths = [str(p) for p in (registry_path, mixed_path) if p is None or not p.exists()]
        if missing_paths:
            raise RuntimeError(f"missing BG controller artifact path(s): {missing_paths}")

        assert registry_path is not None
        assert mixed_path is not None
        registry = _load_payload(registry_path)
        mixed = _load_payload(mixed_path)
        config_payload = _load_payload(cfg_path) if cfg_path is not None and cfg_path.exists() else {}

        registry_rows = list(registry.get("heads", []) or [])
        mixed_rows = list(mixed.get("heads", []) or [])
        role_primary = dict(config_payload.get("role_primary") or config_payload.get("meta", {}).get("role_primary") or {})

        heads: dict[str, nn.Module] = {}
        specs: dict[str, BGHeadSpec] = {}
        role_keys: dict[str, str] = {}
        missing_heads: list[str] = []
        dev = torch.device(device)

        for name, request in LOCKED_HEAD_REQUESTS.items():
            rows = registry_rows if request["source"] == "registry" else mixed_rows
            artifact_path = registry_path if request["source"] == "registry" else mixed_path
            row = _find_head_row(
                rows,
                group=request["group"],
                config=request["config"],
                architecture=request["architecture"],
            )
            if row is None:
                missing_heads.append(
                    f"{name} ({request['group']}::{request['config']}::{request['architecture']}) in {_repo_path(artifact_path)}"
                )
                continue
            dim = int(row.get("dim", config_dim(str(row["config"]))))
            if dim != config_dim(str(row["config"])):
                raise RuntimeError(
                    f"head {name} has dim {dim}, expected {config_dim(str(row['config']))} for config {row['config']}"
                )
            heads[name] = _build_head(str(row["architecture"]), dim, row["state_dict"], dev)
            specs[name] = BGHeadSpec(
                name=name,
                family=str(request["family"]),
                config=str(row["config"]),
                architecture=str(row["architecture"]),
                artifact_path=_repo_path(artifact_path),
                domain_role=str(request["domain_role"]),
                dim=dim,
                calibration_std=None,
            )
            role_keys[name] = _head_key(str(request["group"]), str(row["config"]), str(row["architecture"]))

        if missing_heads:
            raise RuntimeError("missing required BG head(s): " + "; ".join(missing_heads))

        for role, name in ROLE_TO_HEAD_NAME.items():
            if role in role_primary:
                role_keys[name] = str(role_primary[role])

        calibration_by_domain = cls._load_calibration(config_payload, role_keys)
        for name, by_domain in calibration_by_domain.items():
            vals = [float(v) for v in by_domain.values() if float(v) > 1e-8 and not math.isnan(float(v))]
            if name in specs and vals:
                specs[name].calibration_std = float(mean(vals))

        return cls(
            heads=heads,
            specs=specs,
            device=dev,
            calibration_by_domain=calibration_by_domain,
            role_keys=role_keys,
        )

    @staticmethod
    def _load_calibration(config_payload: dict[str, Any], role_keys: dict[str, str]) -> dict[str, dict[str, float]]:
        head_scores = config_payload.get("head_scores") or {}
        if not isinstance(head_scores, dict):
            return {}
        out: dict[str, dict[str, float]] = {}
        for name, key in role_keys.items():
            if key not in head_scores:
                continue
            by_domain: dict[str, float] = {}
            for domain, row in head_scores[key].items():
                metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
                std = metrics.get("margin_std")
                try:
                    value = float(std)
                except Exception:
                    continue
                if not math.isnan(value):
                    by_domain[str(domain)] = value
            if by_domain:
                out[name] = by_domain
        return out

    def _require_head(self, name: str) -> tuple[nn.Module, BGHeadSpec]:
        if name not in self.heads or name not in self.specs:
            raise RuntimeError(f"required BG head is missing: {name}")
        return self.heads[name], self.specs[name]

    def _route_conservative(self, domain_hint: str | None) -> str:
        domain = normalize_domain_hint(domain_hint)
        if domain in HH_DOMAINS:
            return "hh_general"
        if domain in OBJECTIVE_DOMAINS:
            return "objective_mixed"
        return "hh_general"

    def _score_with_head(self, head_name: str, left_pooled: torch.Tensor, right_pooled: torch.Tensor) -> float:
        head, spec = self._require_head(head_name)
        left_vec = config_vector(left_pooled, spec.config).to(device=self.device, dtype=torch.float32)
        right_vec = config_vector(right_pooled, spec.config).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            score = head(left_vec, right_vec)
        return float(score.detach().cpu().item())

    def _calibration_std(self, head_name: str, domain_hint: str | None = None) -> tuple[float, str]:
        raw = "" if domain_hint is None else str(domain_hint).strip()
        by_domain = self.calibration_by_domain.get(head_name, {})
        for key in (raw, raw.upper(), raw.lower()):
            if key in by_domain and float(by_domain[key]) > 1e-8:
                return float(by_domain[key]), f"simulator_domain:{key}"
        spec = self.specs.get(head_name)
        if spec is not None and spec.calibration_std is not None and float(spec.calibration_std) > 1e-8:
            return float(spec.calibration_std), "simulator_fallback"
        return 1.0, "uncalibrated"

    def _normalized_margin(self, head_name: str, score: float, domain_hint: str | None) -> tuple[float, float, str]:
        std, status = self._calibration_std(head_name, domain_hint)
        return float(score) / std, std, status

    def _experimental_vote(self, left_pooled: torch.Tensor, right_pooled: torch.Tensor, domain_hint: str | None) -> dict[str, Any]:
        score_hh = self._score_with_head("hh_general", left_pooled, right_pooled)
        score_obj = self._score_with_head("objective_mixed", left_pooled, right_pooled)
        z_hh, std_hh, cal_hh = self._normalized_margin("hh_general", score_hh, domain_hint)
        z_obj, std_obj, cal_obj = self._normalized_margin("objective_mixed", score_obj, domain_hint)
        sign_hh = _sign(score_hh)
        sign_obj = _sign(score_obj)
        if sign_hh == sign_obj:
            chosen_head = "hh_general" if abs(z_hh) >= abs(z_obj) else "objective_mixed"
            selected_sign = sign_hh
            vote_case = "agreement"
        else:
            chosen_head = "hh_general" if abs(z_hh) >= abs(z_obj) else "objective_mixed"
            selected_sign = sign_hh if chosen_head == "hh_general" else sign_obj
            vote_case = "disagreement"
        selected_z = z_hh if chosen_head == "hh_general" else z_obj
        selected_score = float(selected_sign * abs(selected_z)) if selected_sign != 0 else 0.0
        calibration = "uncalibrated" if cal_hh == "uncalibrated" or cal_obj == "uncalibrated" else "simulator_fallback"
        if cal_hh.startswith("simulator_domain") and cal_obj.startswith("simulator_domain"):
            calibration = "simulator_domain"
        return {
            "score": selected_score,
            "mode": "experimental_vote",
            "selected_head": chosen_head,
            "vote_case": vote_case,
            "domain_hint": domain_hint,
            "normalized_domain": normalize_domain_hint(domain_hint),
            "calibration": calibration,
            "heads": {
                "hh_general": {"score": score_hh, "z": z_hh, "std": std_hh, "calibration": cal_hh},
                "objective_mixed": {"score": score_obj, "z": z_obj, "std": std_obj, "calibration": cal_obj},
            },
        }

    def score_pair(
        self,
        left_pooled: torch.Tensor,
        right_pooled: torch.Tensor,
        domain_hint: str = "unknown",
        mode: str = "conservative",
        return_details: bool = False,
    ) -> float | dict[str, Any]:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"unknown BG controller mode {mode!r}; supported modes: {', '.join(SUPPORTED_MODES)}")
        _validate_pooled(left_pooled, context="left_pooled")
        _validate_pooled(right_pooled, context="right_pooled")

        if mode == "diagnostic_all":
            scores = {
                name: self._score_with_head(name, left_pooled, right_pooled)
                for name in ("hh_general", "objective_mixed", "code_specialist_backup")
            }
            routed = self._route_conservative(domain_hint)
            return {
                "mode": mode,
                "domain_hint": domain_hint,
                "normalized_domain": normalize_domain_hint(domain_hint),
                "conservative_head": routed,
                "scores": scores,
            }

        if mode == "experimental_vote":
            details = self._experimental_vote(left_pooled, right_pooled, domain_hint)
            return details if return_details else float(details["score"])

        head_name = "code_specialist_backup" if mode == "code_backup" else self._route_conservative(domain_hint)
        score = self._score_with_head(head_name, left_pooled, right_pooled)
        if not return_details:
            return score
        return {
            "score": score,
            "mode": mode,
            "selected_head": head_name,
            "domain_hint": domain_hint,
            "normalized_domain": normalize_domain_hint(domain_hint),
            "head_spec": asdict(self.specs[head_name]),
        }

    def _candidate_list(self, candidates_pooled: list[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
        if isinstance(candidates_pooled, torch.Tensor):
            if candidates_pooled.ndim == 3:
                return [_validate_pooled(candidates_pooled, context="candidate")]
            if candidates_pooled.ndim == 4:
                return [_validate_pooled(candidates_pooled[i], context=f"candidate[{i}]") for i in range(candidates_pooled.shape[0])]
            raise ValueError(
                f"candidates_pooled tensor has shape {tuple(candidates_pooled.shape)}; expected [N,3,4,2048] or [3,4,2048]"
            )
        return [_validate_pooled(item, context=f"candidate[{idx}]") for idx, item in enumerate(candidates_pooled)]

    def _matrix_for_head(self, candidates: list[torch.Tensor], head_name: str) -> torch.Tensor:
        n = len(candidates)
        mat = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                score = self._score_with_head(head_name, candidates[i], candidates[j])
                mat[i, j] = float(score)
                mat[j, i] = -float(score)
        return mat

    def tournament_scores(
        self,
        candidates_pooled: list[torch.Tensor] | torch.Tensor,
        domain_hint: str = "unknown",
        mode: str = "conservative",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"unknown BG controller mode {mode!r}; supported modes: {', '.join(SUPPORTED_MODES)}")
        candidates = self._candidate_list(candidates_pooled)
        if mode == "diagnostic_all":
            return {
                name: self._matrix_for_head(candidates, name)
                for name in ("hh_general", "objective_mixed", "code_specialist_backup")
            }
        if mode == "conservative":
            return self._matrix_for_head(candidates, self._route_conservative(domain_hint))
        if mode == "code_backup":
            return self._matrix_for_head(candidates, "code_specialist_backup")

        n = len(candidates)
        mat = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(self.score_pair(candidates[i], candidates[j], domain_hint=domain_hint, mode=mode))
                mat[i, j] = score
                mat[j, i] = -score
        return mat

    @staticmethod
    def _rank_from_matrix(mat: torch.Tensor) -> dict[str, Any]:
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"score matrix must be square, got shape {tuple(mat.shape)}")
        n = int(mat.shape[0])
        mask = ~torch.eye(n, dtype=torch.bool)
        wins = ((mat > 0) & mask).sum(dim=1).to(torch.int64)
        margin_sum = (mat * mask.to(mat.dtype)).sum(dim=1)
        ranking = sorted(range(n), key=lambda idx: (-int(wins[idx].item()), -float(margin_sum[idx].item()), idx))
        return {
            "ranking": ranking,
            "wins": wins,
            "margin_sum": margin_sum,
        }

    def rank_candidates(
        self,
        candidates_pooled: list[torch.Tensor] | torch.Tensor,
        domain_hint: str = "unknown",
        mode: str = "conservative",
        return_details: bool = False,
    ) -> list[int] | dict[str, Any]:
        matrices = self.tournament_scores(candidates_pooled, domain_hint=domain_hint, mode=mode)
        if mode == "diagnostic_all":
            assert isinstance(matrices, dict)
            head_results = {}
            for name, mat in matrices.items():
                ranked = self._rank_from_matrix(mat)
                head_results[name] = {
                    "ranking": ranked["ranking"],
                    "wins": ranked["wins"],
                    "margin_sum": ranked["margin_sum"],
                    "score_matrix": mat,
                }
            return {
                "mode": mode,
                "domain_hint": domain_hint,
                "normalized_domain": normalize_domain_hint(domain_hint),
                "conservative_head": self._route_conservative(domain_hint),
                "head_results": head_results,
            }

        assert isinstance(matrices, torch.Tensor)
        ranked = self._rank_from_matrix(matrices)
        if not return_details:
            return ranked["ranking"]
        selected_head = "experimental_vote" if mode == "experimental_vote" else (
            "code_specialist_backup" if mode == "code_backup" else self._route_conservative(domain_hint)
        )
        return {
            "mode": mode,
            "domain_hint": domain_hint,
            "normalized_domain": normalize_domain_hint(domain_hint),
            "selected_head": selected_head,
            "ranking": ranked["ranking"],
            "score_matrix": matrices,
            "wins": ranked["wins"],
            "margin_sum": ranked["margin_sum"],
        }

    def select_best(
        self,
        candidates_pooled: list[torch.Tensor] | torch.Tensor,
        domain_hint: str = "unknown",
        mode: str = "conservative",
        return_details: bool = False,
    ) -> int | dict[str, Any]:
        ranked = self.rank_candidates(candidates_pooled, domain_hint=domain_hint, mode=mode, return_details=True)
        if mode == "diagnostic_all":
            assert isinstance(ranked, dict)
            selected_by_head = {
                name: row["ranking"][0] if row["ranking"] else -1
                for name, row in ranked["head_results"].items()
            }
            ranked["selected_by_head"] = selected_by_head
            return ranked
        assert isinstance(ranked, dict)
        selected = int(ranked["ranking"][0]) if ranked["ranking"] else -1
        if not return_details:
            return selected
        ranked["selected_index"] = selected
        return ranked


def check_antisymmetry(
    controller: BGController,
    sample_features: list[torch.Tensor] | torch.Tensor,
    domain_hint: str = "unknown",
    mode: str = "conservative",
    atol: float = 1e-5,
) -> bool:
    candidates = controller._candidate_list(sample_features)
    if len(candidates) < 2:
        raise ValueError("check_antisymmetry requires at least two candidate feature tensors")
    ab = controller.score_pair(candidates[0], candidates[1], domain_hint=domain_hint, mode=mode)
    ba = controller.score_pair(candidates[1], candidates[0], domain_hint=domain_hint, mode=mode)
    if isinstance(ab, dict):
        ab = float(ab["score"])
    if isinstance(ba, dict):
        ba = float(ba["score"])
    if abs(float(ab) + float(ba)) > atol:
        raise AssertionError(f"pairwise antisymmetry failed: score(a,b)={ab}, score(b,a)={ba}, atol={atol}")
    return True

