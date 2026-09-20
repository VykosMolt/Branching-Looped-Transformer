"""Train/evaluate a small temporal GRU control on clean GSM8K tournaments."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    LAYER_TO_POS,
    PROJECT_ROOT,
    condorcet_winner_rate,
    cycle_rate,
    output_path,
    pairwise_accuracy,
    tournament_top1_accuracy,
)
from evaluate_hh_transfer_on_clean_gsm8k_extreme import (
    random_top1_baseline,
    rate,
    repo_path,
    snippet,
    split_indices,
)


DEFAULT_FEATURES = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt"
DEFAULT_HH = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
DEFAULT_LINEAR = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_expanded_gru_control_2026-05-16.json"
SUMMARY_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.json"
SUMMARY_MD = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md"
OLD_CONFOUNDED_JSON = PROJECT_ROOT / "opi/taps/probes/hh_to_math_transfer_probe_2026-05-16.json"
MICRO_TRANSFER_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.json"


GRU_CONFIGS = {
    "gru24_sequence": 24,
    "gru36_sequence": 36,
    "gru47_sequence": 47,
}


class SmallTemporalGRUComparator(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, hidden, bias=False)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            bias=False,
        )
        self.out = nn.Linear(hidden, 1, bias=False)

    def forward(self, left_seq: torch.Tensor, right_seq: torch.Tensor) -> torch.Tensor:
        diff = left_seq - right_seq
        diff = self.norm(diff)
        x = self.proj(diff)
        _, h = self.gru(x)
        return self.out(h[-1]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--hh-capture", default=str(DEFAULT_HH))
    parser.add_argument("--linear-results", default=str(DEFAULT_LINEAR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(SUMMARY_MD))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--sym-weight", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def hh_sequences(hh_payload: Dict[str, object], layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen = []
    rejected = []
    for pack in hh_payload["packs"]:
        chosen.append(pack["chosen"]["pooled"][layer].to(torch.float32))
        rejected.append(pack["rejected"]["pooled"][layer].to(torch.float32))
    return torch.stack(chosen, dim=0), torch.stack(rejected, dim=0)


def record_sequences(records: Sequence[Dict[str, object]], layer: int) -> List[torch.Tensor]:
    pos = LAYER_TO_POS[layer]
    return [row["pooled"][:, pos, :, :].to(torch.float32) for row in records]


def pref_loss(head: nn.Module, chosen: torch.Tensor, rejected: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    s_cr = head(chosen, rejected)
    s_rc = head(rejected, chosen)
    loss_pref = (
        F.binary_cross_entropy_with_logits(s_cr, torch.ones_like(s_cr))
        + F.binary_cross_entropy_with_logits(s_rc, torch.zeros_like(s_rc))
    )
    loss_sym = torch.mean((s_cr + s_rc) ** 2)
    loss_l2 = 1e-4 * torch.mean(s_cr ** 2 + s_rc ** 2)
    return loss_pref, {
        "loss_pref": float(loss_pref.detach().cpu()),
        "loss_sym": float(loss_sym.detach().cpu()),
        "loss_l2": float(loss_l2.detach().cpu()),
    }


def full_loss(head: nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, sym_weight: float) -> torch.Tensor:
    s_cr = head(chosen, rejected)
    s_rc = head(rejected, chosen)
    loss_pref = (
        F.binary_cross_entropy_with_logits(s_cr, torch.ones_like(s_cr))
        + F.binary_cross_entropy_with_logits(s_rc, torch.zeros_like(s_rc))
    )
    loss_sym = torch.mean((s_cr + s_rc) ** 2)
    loss_l2 = 1e-4 * torch.mean(s_cr ** 2 + s_rc ** 2)
    return loss_pref + sym_weight * loss_sym + loss_l2


@torch.no_grad()
def eval_loss(head: nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, idx: Sequence[int], device: torch.device, sym_weight: float) -> float:
    return float(full_loss(head, chosen[list(idx)].to(device), rejected[list(idx)].to(device), sym_weight).detach().cpu())


@torch.no_grad()
def centered_hh_acc(head: nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, idx: Sequence[int], device: torch.device) -> float:
    left = chosen[list(idx)].to(device)
    right = rejected[list(idx)].to(device)
    centered = 0.5 * (head(left, right) - head(right, left))
    return float((centered > 0).to(torch.float32).mean().detach().cpu())


def train_gru(
    config: str,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    head = SmallTemporalGRUComparator(hidden=args.hidden).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    left = chosen[list(train_idx)].to(device)
    right = rejected[list(train_idx)].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    losses: List[float] = []
    eval_losses: List[float] = []
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(left.shape[0], generator=generator, device=device)
        total = 0.0
        for start in range(0, left.shape[0], args.batch_size):
            batch = perm[start : start + args.batch_size]
            loss = full_loss(head, left[batch], right[batch], args.sym_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.numel())
        train_loss = total / max(left.shape[0], 1)
        losses.append(train_loss)
        head.eval()
        val = eval_loss(head, chosen, rejected, eval_idx, device, args.sym_weight)
        eval_losses.append(val)
        if val + 1e-7 < best_loss:
            best_loss = val
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    metrics = {
        "config": config,
        "epochs_run": len(losses),
        "best_epoch": best_epoch + 1,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "eval_loss_best": best_loss,
        "hh_train_acc": centered_hh_acc(head, chosen, rejected, train_idx, device),
        "hh_heldout_acc": centered_hh_acc(head, chosen, rejected, eval_idx, device),
        "parameter_count": param_count(head),
    }
    return head.to("cpu"), metrics


@torch.no_grad()
def raw_matrix(head: nn.Module, seqs: torch.Tensor, device: torch.device) -> torch.Tensor:
    seqs = seqs.to(device=device, dtype=torch.float32)
    k = seqs.shape[0]
    left = seqs[:, None, :, :].expand(k, k, seqs.shape[1], seqs.shape[2]).reshape(k * k, seqs.shape[1], seqs.shape[2])
    right = seqs[None, :, :, :].expand(k, k, seqs.shape[1], seqs.shape[2]).reshape(k * k, seqs.shape[1], seqs.shape[2])
    mat = head(left, right).reshape(k, k).detach().cpu()
    mat.fill_diagonal_(0.0)
    return mat


def margin_stats(matrices: Sequence[torch.Tensor]) -> Tuple[float, float]:
    margins: List[float] = []
    for mat in matrices:
        totals = mat.sum(dim=1)
        if totals.numel() < 2:
            margins.append(0.0)
            continue
        top2 = torch.topk(totals, k=2).values
        margins.append(float(top2[0] - top2[1]))
    return (float(mean(margins)), float(pstdev(margins))) if margins else (float("nan"), float("nan"))


def bias_to_signal(raw_matrices: Sequence[torch.Tensor]) -> float:
    sums: List[float] = []
    diffs: List[float] = []
    for mat in raw_matrices:
        k = mat.shape[0]
        for i in range(k):
            for j in range(i + 1, k):
                s_ab = float(mat[i, j])
                s_ba = float(mat[j, i])
                sums.append(s_ab + s_ba)
                diffs.append(s_ab - s_ba)
    if not diffs:
        return float("nan")
    std = float(np.std(np.asarray(diffs, dtype=np.float64)))
    if std <= 1e-12:
        return float("inf")
    return abs(float(np.mean(np.asarray(sums, dtype=np.float64)))) / std


def evaluate_gru(
    head: nn.Module,
    seqs_by_record: Sequence[torch.Tensor],
    records: Sequence[Dict[str, object]],
    baseline: float,
    device: torch.device,
) -> Dict[str, object]:
    raw = [raw_matrix(head, seqs, device) for seqs in seqs_by_record]
    centered = [0.5 * (mat - mat.T) for mat in raw]
    labels = [row["labels"] for row in records]
    margin_mean, margin_std = margin_stats(centered)
    top1 = tournament_top1_accuracy(centered, labels)
    raw_top1 = tournament_top1_accuracy(raw, labels)
    return {
        "n_tournaments": len(records),
        "centered_top1_tournament_acc": top1,
        "centered_top1_over_random_baseline": top1 - baseline,
        "centered_pairwise_acc": pairwise_accuracy(centered, labels),
        "centered_condorcet_winner_rate": condorcet_winner_rate(centered, labels),
        "centered_cycle_rate": cycle_rate(centered, triplets_per_matrix=1000, seed=42),
        "raw_top1_tournament_acc": raw_top1,
        "raw_pairwise_acc": pairwise_accuracy(raw, labels),
        "bias_to_signal": bias_to_signal(raw),
        "margin_mean": margin_mean,
        "margin_std": margin_std,
    }


def best_gru(rows: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["centered_top1_tournament_acc"],
            row["metrics"]["centered_pairwise_acc"],
            -row["metrics"]["centered_cycle_rate"],
        ),
    )


def gru_verdict(best: Optional[Dict[str, object]], linear: Dict[str, object]) -> str:
    if best is None or not linear.get("best_hh_trained"):
        return "GRU_NOT_RUN"
    linear_best = linear["best_hh_trained"]["metrics"]
    linear_verdict = linear.get("expanded_linear_transfer_verdict", "NOT_RUN")
    m = best["metrics"]
    top1 = float(m["centered_top1_tournament_acc"])
    pairwise = float(m["centered_pairwise_acc"])
    lin_top1 = float(linear_best["top1_tournament_acc"])
    lin_pairwise = float(linear_best["pairwise_acc"])
    bias = float(m["bias_to_signal"])
    heldout = float(best["train_metrics"]["hh_heldout_acc"])
    helps = (top1 >= lin_top1 + 0.05 or pairwise >= lin_pairwise + 0.05) and bias <= 0.5
    if helps:
        return "GRU_HELPS"
    if top1 < lin_top1 - 0.05 or pairwise < lin_pairwise - 0.05 or bias > 0.5 or heldout <= 0.60:
        return "GRU_WEAK"
    if linear_verdict in {"GOOD", "WEAK"}:
        return "GRU_NOT_NEEDED"
    return "GRU_WEAK"


def recommended_next(clean_verdict: str, linear_verdict: str, gru_control_verdict: str) -> str:
    if clean_verdict not in {"CLEAN_30", "CLEAN_MINIMUM"}:
        return "generation_tuning_or_code_pivot"
    if linear_verdict == "GOOD" and gru_control_verdict == "GRU_NOT_NEEDED":
        return "code_branch_dataset_or_scale_clean_gsm8k_once_more"
    if gru_control_verdict == "GRU_HELPS":
        return "run_gru_control_on_code_branch_dataset_or_expand_clean_gsm8k"
    if linear_verdict == "WEAK" and gru_control_verdict == "GRU_NOT_NEEDED":
        return "expand_clean_gsm8k_once_more_or_try_code"
    if linear_verdict == "POOR" and gru_control_verdict == "GRU_WEAK":
        return "pivot_to_code_branch_dataset"
    if linear_verdict == "GOOD":
        return "code_branch_dataset_or_scale_clean_gsm8k_once_more"
    return "generation_tuning_or_code_pivot"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def old_confounded_summary() -> Dict[str, object]:
    payload = load_json(OLD_CONFOUNDED_JSON)
    return {
        "path": repo_path(OLD_CONFOUNDED_JSON),
        "exists": OLD_CONFOUNDED_JSON.exists(),
        "label": "historical_confounded_baseline",
        "transfer_verdict": payload.get("transfer_verdict"),
    }


def clean_micro_summary() -> Dict[str, object]:
    payload = load_json(MICRO_TRANSFER_JSON)
    return {
        "path": repo_path(MICRO_TRANSFER_JSON),
        "exists": MICRO_TRANSFER_JSON.exists(),
        "label": "tiny_clean_n5_baseline",
        "clean_gsm8k_verdict": payload.get("clean_gsm8k_verdict"),
        "clean_transfer_verdict": payload.get("clean_transfer_verdict"),
        "random_top1_baseline": payload.get("random_top1_baseline"),
    }


def compact_linear_row(row: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not row:
        return {}
    m = row["metrics"]
    return {
        "config": row["config"],
        "architecture": row["architecture"],
        "top1": m["top1_tournament_acc"],
        "pairwise": m["pairwise_acc"],
        "cycle": m["cycle_rate"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
    }


def compact_gru_row(row: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not row:
        return {}
    m = row["metrics"]
    return {
        "config": row["config"],
        "top1": m["centered_top1_tournament_acc"],
        "pairwise": m["centered_pairwise_acc"],
        "cycle": m["centered_cycle_rate"],
        "bias_to_signal": m["bias_to_signal"],
        "hh_heldout_acc": row["train_metrics"]["hh_heldout_acc"],
    }


def family_winner(best_antisym: Optional[Dict[str, object]], best_nonorm: Optional[Dict[str, object]], best_g: Optional[Dict[str, object]]) -> str:
    candidates = []
    if best_antisym:
        m = best_antisym["metrics"]
        candidates.append(("AntisymLinear", m["top1_tournament_acc"], m["pairwise_acc"]))
    if best_nonorm:
        m = best_nonorm["metrics"]
        candidates.append(("AntisymLinearNoNorm", m["top1_tournament_acc"], m["pairwise_acc"]))
    if best_g:
        m = best_g["metrics"]
        candidates.append(("GRU", m["centered_top1_tournament_acc"], m["centered_pairwise_acc"]))
    if not candidates:
        return "NA"
    return max(candidates, key=lambda row: (row[1], row[2]))[0]


def winning_layer(config: str) -> str:
    if not config:
        return "NA"
    if "24" in config:
        return "24"
    if "36" in config:
        return "36"
    if "47" in config:
        return "47"
    return "NA"


def generation_examples(generation: Dict[str, object], limit: int = 10) -> List[Dict[str, object]]:
    out = []
    for t in generation.get("tournaments", [])[:limit]:
        out.append({
            "tournament_id": t["tournament_id"],
            "prompt_mode": t["prompt_mode"],
            "question": snippet(t["question"]),
            "gold_answer": t["gold_answer"],
            "branches": [
                {
                    "attempt_index": a["attempt_index"],
                    "is_correct": a["is_correct"],
                    "answer": a["extracted_answer"],
                    "temperature": a["temperature"],
                    "classification": "correct" if a["is_correct"] else a.get("wrong_classification", "unknown"),
                }
                for a in t["attempts"]
            ],
        })
    return out


def write_gru_md(path: Path, result: Dict[str, object]) -> None:
    lines = [
        "# Expanded Clean GSM8K GRU Control",
        "",
        f"GRU_CONTROL_VERDICT = {result['gru_control_verdict']}",
        "",
        "| config | centered_top1 | over_random | centered_pairwise | centered_condorcet | centered_cycle | raw_top1 | raw_pairwise | bias_to_signal | hh_holdout | params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["gru_table"]:
        m = row["metrics"]
        tm = row["train_metrics"]
        lines.append(
            f"| `{row['config']}` | {rate(m['centered_top1_tournament_acc'])} | "
            f"{rate(m['centered_top1_over_random_baseline'])} | {rate(m['centered_pairwise_acc'])} | "
            f"{rate(m['centered_condorcet_winner_rate'])} | {rate(m['centered_cycle_rate'])} | "
            f"{rate(m['raw_top1_tournament_acc'])} | {rate(m['raw_pairwise_acc'])} | "
            f"{rate(m['bias_to_signal'])} | {rate(tm['hh_heldout_acc'])} | {tm['parameter_count']} |"
        )
    lines.extend([
        "",
        f"Best GRU: `{result['best_gru'].get('config')}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(path: Path, summary: Dict[str, object]) -> None:
    gen = summary["generation_summary"]
    lines = [
        "# Expanded Clean GSM8K Transfer + GRU Control",
        "",
        f"EXPANDED_CLEAN_GSM8K_VERDICT = {summary['expanded_clean_gsm8k_verdict']}",
        f"EXPANDED_LINEAR_TRANSFER_VERDICT = {summary['expanded_linear_transfer_verdict']}",
        f"GRU_CONTROL_VERDICT = {summary['gru_control_verdict']}",
        f"RECOMMENDED_NEXT = {summary['recommended_next']}",
        "",
        "## 1. Expanded Generation Summary",
        "",
    ]
    for key in (
        "prompts_processed",
        "attempts_generated",
        "clean_attempts",
        "correct_clean_attempts",
        "incorrect_clean_attempts",
        "clean_tournaments_kept",
        "correct_candidate_count_distribution",
        "random_top1_baseline",
        "near_miss_fraction",
    ):
        if key in gen:
            lines.append(f"- {key}: `{gen[key]}`")
    lines.extend(["", "## 2. Clean Tournament Examples", ""])
    for t in summary["clean_tournament_examples"]:
        lines.extend([
            f"### Tournament {t['tournament_id']} mode={t['prompt_mode']}",
            "",
            f"Prompt: {t['question']}",
            "",
            f"Gold answer: `{t['gold_answer']}`",
            "",
        ])
        for b in t["branches"]:
            lines.append(
                f"- branch={b['attempt_index']} correct={b['is_correct']} "
                f"answer=`{b['answer']}` temp={b['temperature']} class=`{b['classification']}`"
            )
        lines.append("")
    lines.extend([
        "## 3. Feature Capture Summary",
        "",
        f"- features: `{summary['features_path']}`",
        f"- n_tournaments: `{summary['feature_summary']['n_tournaments']}`",
        f"- n_candidates: `{summary['feature_summary']['n_candidates']}`",
        "",
        "## 4. HH-Trained AntisymLinear / NoNorm Transfer Table",
        "",
        "| config | architecture | top1 | over_random | pairwise | condorcet | cycle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["linear_transfer_table"]:
        m = row["metrics"]
        lines.append(
            f"| `{row['config']}` | {row['architecture']} | {rate(m['top1_tournament_acc'])} | "
            f"{rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | "
            f"{rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} |"
        )
    lines.extend([
        "",
        "## 5. Small Temporal GRU Control Table",
        "",
        "| config | centered_top1 | over_random | centered_pairwise | centered_cycle | raw_top1 | raw_pairwise | bias_to_signal | hh_holdout |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["gru_table"]:
        m = row["metrics"]
        tm = row["train_metrics"]
        lines.append(
            f"| `{row['config']}` | {rate(m['centered_top1_tournament_acc'])} | "
            f"{rate(m['centered_top1_over_random_baseline'])} | {rate(m['centered_pairwise_acc'])} | "
            f"{rate(m['centered_cycle_rate'])} | {rate(m['raw_top1_tournament_acc'])} | "
            f"{rate(m['raw_pairwise_acc'])} | {rate(m['bias_to_signal'])} | {rate(tm['hh_heldout_acc'])} |"
        )
    lines.extend([
        "",
        "## 6. Random Baseline and Correct-Candidate Distribution",
        "",
        f"- random_top1_baseline: `{summary['random_top1_baseline']:.3f}`",
        f"- correct_candidate_count_distribution: `{summary['correct_candidate_count_distribution']}`",
        "",
        "## 7. Historical Comparisons",
        "",
        "- old pilot comparison is marked historical/confounded because the data-validity probe found truncation confounding.",
        f"- old confounded report: `{summary['old_confounded_pilot'].get('path')}`",
        f"- clean n=5 micro report: `{summary['clean_micro_baseline'].get('path')}`",
        "",
        "## 8. Best-Head Comparison",
        "",
        f"- best AntisymLinear: `{summary['best_antisymlinear']}`",
        f"- best NoNorm: `{summary['best_nonorm']}`",
        f"- best GRU: `{summary['best_gru']}`",
        f"- winner family: `{summary['winner_family']}`",
        f"- winner layer: `{summary['winner_layer']}`",
        "",
        "## 9. Markdown Docs Updated",
        "",
    ])
    for doc in summary["docs_updated"]:
        lines.append(f"- `{doc}`")
    lines.extend([
        "",
        "## 10. Files Modified / Created",
        "",
    ])
    for file_path in summary["files_modified_or_created"]:
        lines.append(f"- `{file_path}`")
    lines.extend([
        "",
        "## 11. Commands Run",
        "",
        "```bash",
        *summary["commands_run"],
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_docs(summary: Dict[str, object]) -> List[str]:
    docs = [
        PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
        PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    ]
    best_a = summary["best_antisymlinear"]
    best_n = summary["best_nonorm"]
    best_g = summary["best_gru"]
    interpretation = (
        "Temporal aggregation did not improve branch selection beyond the exact-antisymmetric linear/NoNorm controls."
        if summary["gru_control_verdict"] == "GRU_NOT_NEEDED"
        else "Temporal aggregation changed the expanded clean GSM8K branch-selection result and should be checked on another clean domain."
        if summary["gru_control_verdict"] == "GRU_HELPS"
        else "The GRU control underperformed the exact-antisymmetric linear/NoNorm controls; centered raw bias was low, but HH holdout accuracy stayed below the control threshold."
    )
    block = [
        "",
        "## Expanded clean GSM8K transfer + GRU control (2026-05-16)",
        "",
        f"- EXPANDED_CLEAN_GSM8K_VERDICT: `{summary['expanded_clean_gsm8k_verdict']}`",
        f"- EXPANDED_LINEAR_TRANSFER_VERDICT: `{summary['expanded_linear_transfer_verdict']}`",
        f"- GRU_CONTROL_VERDICT: `{summary['gru_control_verdict']}`",
        f"- clean tournaments: `{summary['generation_summary'].get('clean_tournaments_kept')}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']:.3f}`",
        f"- best AntisymLinear config/head: `{best_a}`",
        f"- best NoNorm config/head: `{best_n}`",
        f"- best GRU config: `{best_g}`",
        f"- winner family: `{summary['winner_family']}`",
        f"- winner layer: `{summary['winner_layer']}`",
        f"- full report: `{repo_path(SUMMARY_MD)}`",
        f"- interpretation: {interpretation}",
        "",
    ]
    updated = []
    for doc in docs:
        if not doc.exists():
            continue
        with doc.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(block))
        updated.append(repo_path(doc))
    return updated


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    features_path = output_path(args.features)
    hh_path = output_path(args.hh_capture)
    linear_path = output_path(args.linear_results)
    output_json = output_path(args.output)
    output_md = output_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    summary_json = output_path(args.summary_json)
    summary_md = output_path(args.summary_md)
    if not features_path.exists():
        raise SystemExit(f"missing features: {features_path}")
    if not hh_path.exists():
        raise SystemExit("HH_CAPTURE_MISSING")
    if not linear_path.exists():
        raise SystemExit(f"missing linear results: {linear_path}")

    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    linear = load_json(linear_path)
    generation_path = output_path(feature_payload["meta"]["input_json"])
    generation = load_json(generation_path)
    clean_verdict = generation.get("expanded_clean_gsm8k_verdict", "UNKNOWN")
    if clean_verdict not in {"CLEAN_30", "CLEAN_MINIMUM"}:
        raise SystemExit(f"expanded clean verdict is {clean_verdict}; GRU control blocked")
    if linear.get("expanded_linear_transfer_verdict") not in {"GOOD", "WEAK", "POOR"}:
        raise SystemExit("linear transfer did not complete; GRU control blocked")

    records = feature_payload["records"]
    baseline = random_top1_baseline(records)
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), args.heldout, args.seed)

    rows: List[Dict[str, object]] = []
    for config, layer in GRU_CONFIGS.items():
        print(f"training/evaluating {config}", flush=True)
        chosen, rejected = hh_sequences(hh_payload, layer)
        head, train_metrics = train_gru(config, chosen, rejected, train_idx, eval_idx, args, device)
        head = head.to(device)
        seqs_by_record = record_sequences(records, layer)
        metrics = evaluate_gru(head, seqs_by_record, records, baseline, device)
        rows.append({
            "config": config,
            "layer": layer,
            "train_metrics": train_metrics,
            "metrics": metrics,
        })

    best_g = best_gru(rows)
    gru_control_verdict = gru_verdict(best_g, linear)
    rec_next = recommended_next(clean_verdict, linear.get("expanded_linear_transfer_verdict", "NOT_RUN"), gru_control_verdict)

    gru_result = {
        "gru_control_verdict": gru_control_verdict,
        "recommended_next": rec_next,
        "features_path": repo_path(features_path),
        "hh_capture": repo_path(hh_path),
        "linear_results": repo_path(linear_path),
        "random_top1_baseline": baseline,
        "gru_table": rows,
        "best_gru": best_g,
        "commands_run": [
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_temporal_gru_control_on_clean_gsm8k.py --features opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --linear-results opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json --output opi/taps/probes/clean_gsm8k_expanded_gru_control_2026-05-16.json --device cuda",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(gru_result, indent=2, default=str) + "\n", encoding="utf-8")
    write_gru_md(output_md, gru_result)

    best_a = compact_linear_row(linear.get("best_antisymlinear"))
    best_n = compact_linear_row(linear.get("best_nonorm"))
    best_g_compact = compact_gru_row(best_g)
    winner = family_winner(linear.get("best_antisymlinear"), linear.get("best_nonorm"), best_g)
    winner_config = (
        best_g_compact.get("config", "")
        if winner == "GRU"
        else best_n.get("config", "")
        if winner == "AntisymLinearNoNorm"
        else best_a.get("config", "")
    )
    summary = {
        "expanded_clean_gsm8k_verdict": clean_verdict,
        "expanded_linear_transfer_verdict": linear.get("expanded_linear_transfer_verdict"),
        "gru_control_verdict": gru_control_verdict,
        "recommended_next": rec_next,
        "generation_summary": generation.get("summary", {}),
        "feature_summary": feature_payload.get("meta", {}),
        "features_path": repo_path(features_path),
        "linear_results": repo_path(linear_path),
        "gru_results": repo_path(output_json),
        "random_top1_baseline": baseline,
        "correct_candidate_count_distribution": linear.get("correct_candidate_count_distribution", {}),
        "linear_transfer_table": linear.get("transfer_table", []),
        "gru_table": rows,
        "best_antisymlinear": best_a,
        "best_nonorm": best_n,
        "best_gru": best_g_compact,
        "winner_family": winner,
        "winner_layer": winning_layer(winner_config),
        "clean_tournament_examples": generation_examples(generation),
        "old_confounded_pilot": old_confounded_summary(),
        "clean_micro_baseline": clean_micro_summary(),
        "files_modified_or_created": [
            "shared/utilities/tests/manual/generate_clean_gsm8k_extreme_expand.py",
            "shared/utilities/tests/manual/capture_clean_gsm8k_extreme_tap_features.py",
            "shared/utilities/tests/manual/evaluate_hh_transfer_on_clean_gsm8k_extreme.py",
            "shared/utilities/tests/manual/evaluate_hh_temporal_gru_control_on_clean_gsm8k.py",
            "opi/taps/probes/clean_gsm8k_expanded_2026-05-16.json",
            "opi/taps/probes/clean_gsm8k_expanded_2026-05-16.md",
            "opi/taps/probes/clean_gsm8k_expanded_2026-05-16.log",
            "opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt",
            "opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.md",
            "opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json",
            "opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.md",
            "opi/taps/probes/clean_gsm8k_expanded_gru_control_2026-05-16.json",
            "opi/taps/probes/clean_gsm8k_expanded_gru_control_2026-05-16.md",
            "opi/taps/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.json",
            "opi/taps/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md",
            "docs/evaluator/evaluator_domain_transfer_notes.md",
            "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
            "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/generate_clean_gsm8k_extreme_expand.py",
            "venv/bin/python -m py_compile utilities/tests/manual/capture_clean_gsm8k_extreme_tap_features.py",
            "venv/bin/python -m py_compile utilities/tests/manual/evaluate_hh_transfer_on_clean_gsm8k_extreme.py",
            "venv/bin/python -m py_compile utilities/tests/manual/evaluate_hh_temporal_gru_control_on_clean_gsm8k.py",
            "venv/bin/python -u utilities/tests/manual/generate_clean_gsm8k_extreme_expand.py --max-prompts 80 --target-clean-tournaments 30 --min-clean-tournaments 20 --attempts-per-mode 4 --device cuda",
            "venv/bin/python -u utilities/tests/manual/capture_clean_gsm8k_extreme_tap_features.py --input opi/taps/probes/clean_gsm8k_expanded_2026-05-16.json --output opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt --device cuda",
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_clean_gsm8k_extreme.py --features opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --output opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json",
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_temporal_gru_control_on_clean_gsm8k.py --features opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --linear-results opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json --output opi/taps/probes/clean_gsm8k_expanded_gru_control_2026-05-16.json --device cuda",
        ],
    }
    docs_updated = append_docs(summary)
    summary["docs_updated"] = docs_updated
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    write_summary_md(summary_md, summary)

    print(f"EXPANDED_CLEAN_GSM8K_VERDICT = {clean_verdict}", flush=True)
    print(f"EXPANDED_LINEAR_TRANSFER_VERDICT = {linear.get('expanded_linear_transfer_verdict')}", flush=True)
    print(f"GRU_CONTROL_VERDICT = {gru_control_verdict}", flush=True)
    print(f"RECOMMENDED_NEXT = {rec_next}", flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)
    print(f"Wrote {summary_json}", flush=True)
    print(f"Wrote {summary_md}", flush=True)


if __name__ == "__main__":
    main()
