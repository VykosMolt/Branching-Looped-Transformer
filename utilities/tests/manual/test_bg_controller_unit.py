"""Unit and smoke tests for the read-only BG controller."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_controller import (  # noqa: E402
    AntisymLinear,
    AntisymLinearNoNorm,
    BGController,
    BGHeadSpec,
    check_antisymmetry,
    config_dim,
    config_vector,
)


def fake_pooled(value: float = 0.0) -> torch.Tensor:
    pooled = torch.zeros(3, 4, 2048, dtype=torch.float32)
    pooled[1, 3, 0] = float(value)
    pooled[2, 0, 0] = float(value)
    pooled[2, 3, 0] = float(value)
    return pooled


class ConstantHead(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        return torch.full(shape, self.value, dtype=torch.float32, device=left.device)


def make_specs(calibrated: bool = True) -> dict[str, BGHeadSpec]:
    hh_std = 2.0 if calibrated else None
    obj_std = 1.0 if calibrated else None
    return {
        "hh_general": BGHeadSpec(
            name="hh_general",
            family="HH",
            config="36_L4",
            architecture="AntisymLinearNoNorm",
            artifact_path=None,
            domain_role="test HH",
            dim=2048,
            calibration_std=hh_std,
        ),
        "objective_mixed": BGHeadSpec(
            name="objective_mixed",
            family="MIX_CODE_REASONING",
            config="36_L4",
            architecture="AntisymLinearNoNorm",
            artifact_path=None,
            domain_role="test objective",
            dim=2048,
            calibration_std=obj_std,
        ),
        "code_specialist_backup": BGHeadSpec(
            name="code_specialist_backup",
            family="CODE",
            config="36_L4",
            architecture="AntisymLinear",
            artifact_path=None,
            domain_role="test code backup",
            dim=2048,
            calibration_std=None,
        ),
    }


def make_constant_controller(hh: float, obj: float, code: float, calibrated: bool = True) -> BGController:
    return BGController(
        heads={
            "hh_general": ConstantHead(hh),
            "objective_mixed": ConstantHead(obj),
            "code_specialist_backup": ConstantHead(code),
        },
        specs=make_specs(calibrated=calibrated),
        device="cpu",
    )


def test_config_vector_shapes() -> None:
    pooled = torch.randn(3, 4, 2048)
    assert config_vector(pooled, "36_L4").shape == (2048,)
    assert config_vector(pooled, "47_concat_L1_L4").shape == (4096,)
    assert config_vector(pooled, "47_concat_all_loops").shape == (8192,)
    assert config_dim("36_L4") == 2048
    assert config_dim("47_concat_L1_L4") == 4096
    assert config_dim("47_concat_all_loops") == 8192


def test_head_antisymmetry() -> None:
    for cls in (AntisymLinear, AntisymLinearNoNorm):
        torch.manual_seed(7)
        head = cls(16)
        left = torch.randn(5, 16)
        right = torch.randn(5, 16)
        ab = head(left, right)
        ba = head(right, left)
        assert torch.allclose(ab, -ba, atol=1e-5), cls.__name__


def test_conservative_routing() -> None:
    controller = make_constant_controller(hh=1.0, obj=2.0, code=3.0)
    left = fake_pooled(1.0)
    right = fake_pooled(0.0)
    expected = {
        "hh": "hh_general",
        "preference": "hh_general",
        "unknown": "hh_general",
        "code": "objective_mixed",
        "strict_clean_code": "objective_mixed",
        "reasoning": "objective_mixed",
        "science": "objective_mixed",
        "math": "objective_mixed",
        "gsm8k": "objective_mixed",
        "objective": "objective_mixed",
    }
    for domain, head_name in expected.items():
        result = controller.score_pair(left, right, domain_hint=domain, mode="conservative", return_details=True)
        assert result["selected_head"] == head_name, domain


def test_code_backup_mode() -> None:
    controller = make_constant_controller(hh=1.0, obj=2.0, code=3.0)
    result = controller.score_pair(fake_pooled(1.0), fake_pooled(0.0), domain_hint="code", mode="code_backup", return_details=True)
    assert result["selected_head"] == "code_specialist_backup"
    assert result["score"] == 3.0


def test_experimental_vote() -> None:
    left = fake_pooled(1.0)
    right = fake_pooled(0.0)

    agreement = make_constant_controller(hh=2.0, obj=1.0, code=0.0)
    agreement.specs["objective_mixed"].calibration_std = 0.25
    result = agreement.score_pair(left, right, domain_hint="code", mode="experimental_vote", return_details=True)
    assert result["vote_case"] == "agreement"
    assert result["selected_head"] == "objective_mixed"
    assert result["score"] == 4.0

    disagreement = make_constant_controller(hh=1.0, obj=-3.0, code=0.0)
    result = disagreement.score_pair(left, right, domain_hint="code", mode="experimental_vote", return_details=True)
    assert result["vote_case"] == "disagreement"
    assert result["selected_head"] == "objective_mixed"
    assert result["score"] == -3.0

    missing_calibration = make_constant_controller(hh=0.5, obj=-0.25, code=0.0, calibrated=False)
    result = missing_calibration.score_pair(left, right, domain_hint="code", mode="experimental_vote", return_details=True)
    assert result["calibration"] == "uncalibrated"
    assert result["selected_head"] == "hh_general"


def test_tournament_ranking() -> None:
    head = AntisymLinearNoNorm(2048)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.weight[0, 0] = 1.0
    specs = make_specs()
    controller = BGController(
        heads={
            "hh_general": head,
            "objective_mixed": head,
            "code_specialist_backup": head,
        },
        specs=specs,
        device="cpu",
    )
    candidates = torch.stack([fake_pooled(3.0), fake_pooled(1.0), fake_pooled(2.0)], dim=0)
    details = controller.rank_candidates(candidates, domain_hint="code", mode="conservative", return_details=True)
    assert details["ranking"] == [0, 2, 1]
    assert details["wins"].tolist() == [2, 0, 1]
    assert torch.allclose(details["margin_sum"], torch.tensor([3.0, -3.0, 0.0]))
    assert controller.select_best(candidates, domain_hint="code") == 0
    assert check_antisymmetry(controller, candidates[:2], domain_hint="code")


def main() -> None:
    tests = [
        test_config_vector_shapes,
        test_head_antisymmetry,
        test_conservative_routing,
        test_code_backup_mode,
        test_experimental_vote,
        test_tournament_ranking,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("BG_CONTROLLER_UNIT_TEST_VERDICT = PASS")


if __name__ == "__main__":
    main()

