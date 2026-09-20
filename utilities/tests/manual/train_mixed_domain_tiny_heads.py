"""Train balanced mixed-domain tiny pairwise heads from cached features."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402
from evaluate_hh_transfer_on_clean_gsm8k_extreme import build_hh_features  # noqa: E402
from math_bg_probe_lib import MATH_CONFIGS, config_dim, config_vector  # noqa: E402
from train_code_specific_tiny_heads_and_eval import HEAD_CLASSES  # noqa: E402


SPLITS_JSON = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "mixed_tap_features_2026-05-17.pt"
OUTPUT_PT = REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.md"

MIXED_CONFIGS = (
    "24_L4",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "47_concat_L1_L4",
    "47_concat_all_loops",
    "24_L1",
    "24_mean",
    "36_L1",
)
ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--output", default=str(OUTPUT_PT))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(output_path(path), map_location="cpu", weights_only=False)


def feature_map(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32)
        for row in payload.get("candidate_features", []) or []
    }


def pair_tensors_from_uids(pairs: list[dict[str, Any]], pooled_by_uid: dict[str, torch.Tensor], config: str) -> tuple[torch.Tensor, torch.Tensor]:
    left = [config_vector(pooled_by_uid[str(pair["preferred_uid"])], config).to(torch.float32) for pair in pairs]
    right = [config_vector(pooled_by_uid[str(pair["rejected_uid"])], config).to(torch.float32) for pair in pairs]
    return torch.stack(left, dim=0), torch.stack(right, dim=0)


def domain_pairs(features: dict[str, Any], domain_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domains = features["domains"]
    if domain_name == "SCIENCE_MEDICINE":
        science = domains.get("SCIENCE", {})
        return science.get("train_pairs_by_subdomain", {}).get("medicine", []) or [], science.get("val_pairs_by_subdomain", {}).get("medicine", []) or []
    if domain_name == "SCIENCE_CHEMISTRY":
        science = domains.get("SCIENCE", {})
        return science.get("train_pairs_by_subdomain", {}).get("chemistry", []) or [], science.get("val_pairs_by_subdomain", {}).get("chemistry", []) or []
    domain = domains.get(domain_name, {})
    return domain.get("train_pairs", []) or [], domain.get("val_pairs", []) or []


def build_domain_tensors(
    *,
    domain_name: str,
    config: str,
    features: dict[str, Any],
    pooled_by_uid: dict[str, torch.Tensor],
    hh_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if domain_name == "HH":
        hh_path = features["hh_path"]
        if config not in hh_cache:
            hh_payload = load_pt(hh_path)
            hh_cache[config] = build_hh_features(hh_payload, config)
        chosen, rejected = hh_cache[config]
        hh_domain = features["domains"]["HH"]
        train_idx = list(hh_domain.get("train_indices", []))
        val_idx = list(hh_domain.get("val_indices", [])) or train_idx
        meta = {"raw_train_pairs": len(train_idx), "raw_val_pairs": len(val_idx)}
        return chosen[train_idx], rejected[train_idx], chosen[val_idx], rejected[val_idx], meta
    train_pairs, val_pairs = domain_pairs(features, domain_name)
    if not val_pairs:
        val_pairs = train_pairs
    left_train, right_train = pair_tensors_from_uids(train_pairs, pooled_by_uid, config)
    left_val, right_val = pair_tensors_from_uids(val_pairs, pooled_by_uid, config)
    meta = {"raw_train_pairs": len(train_pairs), "raw_val_pairs": len(val_pairs)}
    return left_train, right_train, left_val, right_val, meta


@torch.no_grad()
def binary_acc(head: torch.nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    logits = head(left.to(device), right.to(device))
    return float((logits > 0).to(torch.float32).mean().detach().cpu()) if logits.numel() else float("nan")


@torch.no_grad()
def binary_loss(head: torch.nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    logits = head(left.to(device), right.to(device))
    return float(F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).detach().cpu()) if logits.numel() else float("inf")


def train_balanced_head(
    *,
    family: str,
    domains: list[str],
    architecture: str,
    config: str,
    features: dict[str, Any],
    pooled_by_uid: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    hh_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    raw_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}
    for domain in domains:
        try:
            lt, rt, lv, rv, meta = build_domain_tensors(
                domain_name=domain,
                config=config,
                features=features,
                pooled_by_uid=pooled_by_uid,
                hh_cache=hh_cache,
            )
        except Exception as exc:
            return None, {"blocked": True, "blocker": f"{domain}: {exc}", "family": family, "architecture": architecture, "config": config}
        if lt.shape[0] == 0:
            return None, {"blocked": True, "blocker": f"{domain}: no training pairs", "family": family, "architecture": architecture, "config": config}
        tensors[domain] = {"left_train": lt, "right_train": rt, "left_val": lv, "right_val": rv}
        raw_counts[domain] = int(meta["raw_train_pairs"])
        val_counts[domain] = int(meta["raw_val_pairs"])

    target_per_domain = max(raw_counts.values())
    effective_pairs = target_per_domain * len(domains)
    batch_size = max(1, min(int(args.batch_size), effective_pairs))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    head = HEAD_CLASSES[architecture](config_dim(config)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(int(args.epochs)):
        sampled_left: list[torch.Tensor] = []
        sampled_right: list[torch.Tensor] = []
        for domain in domains:
            left = tensors[domain]["left_train"].to(device)
            right = tensors[domain]["right_train"].to(device)
            n = left.shape[0]
            if n >= target_per_domain:
                idx = torch.randperm(n, generator=generator, device=device)[:target_per_domain]
            else:
                idx = torch.randint(0, n, (target_per_domain,), generator=generator, device=device)
            sampled_left.append(left[idx])
            sampled_right.append(right[idx])
        left_epoch = torch.cat(sampled_left, dim=0)
        right_epoch = torch.cat(sampled_right, dim=0)
        target = torch.ones(left_epoch.shape[0], device=device)
        perm = torch.randperm(left_epoch.shape[0], generator=generator, device=device)
        head.train()
        total = 0.0
        for start in range(0, left_epoch.shape[0], batch_size):
            batch = perm[start : start + batch_size]
            logits = head(left_epoch[batch], right_epoch[batch])
            loss = F.binary_cross_entropy_with_logits(logits, target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.numel())
        train_loss = total / max(int(left_epoch.shape[0]), 1)
        losses.append(train_loss)
        head.eval()
        val_left = torch.cat([tensors[d]["left_val"] for d in domains], dim=0)
        val_right = torch.cat([tensors[d]["right_val"] for d in domains], dim=0)
        val_loss = binary_loss(head, val_left, val_right, device)
        val_losses.append(val_loss)
        if val_loss + 1e-7 < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= int(args.patience):
            break

    head.load_state_dict(best_state)
    head.eval()
    domain_balance: dict[str, Any] = {}
    small_domain = min(raw_counts, key=raw_counts.get)
    large_domain = max(raw_counts, key=raw_counts.get)
    small_domain_overfit = False
    domain_overfit_warning = False
    for domain in domains:
        train_acc = binary_acc(head, tensors[domain]["left_train"], tensors[domain]["right_train"], device)
        val_acc = binary_acc(head, tensors[domain]["left_val"], tensors[domain]["right_val"], device)
        gap = train_acc - val_acc if not (math.isnan(train_acc) or math.isnan(val_acc)) else float("nan")
        if domain == small_domain and train_acc > 0.95 and val_acc < 0.70:
            small_domain_overfit = True
        if not math.isnan(gap) and gap > 0.20:
            domain_overfit_warning = True
        domain_balance[domain] = {
            "raw_available_training_pairs": raw_counts[domain],
            "raw_validation_pairs": val_counts[domain],
            "effective_sampled_pairs_per_epoch": target_per_domain,
            "sample_factor": target_per_domain / raw_counts[domain] if raw_counts[domain] else float("inf"),
            "train_pairwise_accuracy": train_acc,
            "validation_pairwise_accuracy": val_acc,
            "train_val_gap": gap,
        }
    metrics = {
        "family": family,
        "domains": domains,
        "architecture": architecture,
        "config": config,
        "dim": config_dim(config),
        "epochs_run": len(losses),
        "best_epoch": best_epoch + 1,
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "val_loss_best": best_loss,
        "batch_size": batch_size,
        "target_pairs_per_domain_per_epoch": target_per_domain,
        "effective_pairs_per_epoch": effective_pairs,
        "smallest_domain": small_domain,
        "smallest_domain_pair_count": raw_counts[small_domain],
        "largest_domain": large_domain,
        "largest_domain_pair_count": raw_counts[large_domain],
        "small_domain_overfit": small_domain_overfit,
        "domain_overfit_warning": domain_overfit_warning,
        "domain_balance": domain_balance,
    }
    return head.to("cpu"), metrics


def compact_head(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["train_metrics"]
    return {
        "head_group": row["head_group"],
        "architecture": row["architecture"],
        "family_architecture": row["family_architecture"],
        "config": row["config"],
        "dim": row["dim"],
        "domains": metrics.get("domains", []),
        "epochs_run": metrics.get("epochs_run"),
        "val_loss_best": metrics.get("val_loss_best"),
        "small_domain_overfit": metrics.get("small_domain_overfit"),
        "domain_overfit_warning": metrics.get("domain_overfit_warning"),
        "domain_balance": metrics.get("domain_balance", {}),
    }


def summarize_domain_balance(heads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in heads:
        family = row["head_group"]
        by_family.setdefault(family, {})
        for domain, info in (row.get("domain_balance") or {}).items():
            by_family[family].setdefault(domain, []).append(info)
    summary: dict[str, list[dict[str, Any]]] = {}
    for family, domains in by_family.items():
        rows: list[dict[str, Any]] = []
        for domain, items in sorted(domains.items()):
            train_acc = [float(item["train_pairwise_accuracy"]) for item in items if not math.isnan(float(item["train_pairwise_accuracy"]))]
            val_acc = [float(item["validation_pairwise_accuracy"]) for item in items if not math.isnan(float(item["validation_pairwise_accuracy"]))]
            gaps = [float(item["train_val_gap"]) for item in items if not math.isnan(float(item["train_val_gap"]))]
            rows.append(
                {
                    "domain": domain,
                    "raw_available_training_pairs": int(items[0]["raw_available_training_pairs"]),
                    "raw_validation_pairs": int(items[0]["raw_validation_pairs"]),
                    "effective_sampled_pairs_per_epoch": int(items[0]["effective_sampled_pairs_per_epoch"]),
                    "sample_factor": float(items[0]["sample_factor"]),
                    "mean_train_pairwise_accuracy": float(mean(train_acc)) if train_acc else float("nan"),
                    "mean_validation_pairwise_accuracy": float(mean(val_acc)) if val_acc else float("nan"),
                    "max_train_val_gap": max(gaps) if gaps else float("nan"),
                    "heads_count": len(items),
                }
            )
        summary[family] = rows
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mixed-Domain Tiny Head Training (2026-05-17)",
        "",
        f"MIXED_TAP_TRAINING_VERDICT = {payload['meta']['mixed_tap_training_verdict']}",
        f"SMALL_DOMAIN_OVERFIT = {payload['meta']['small_domain_overfit']}",
        f"DOMAIN_OVERFIT_WARNING = {payload['meta']['domain_overfit_warning']}",
        f"trained heads = {payload['meta']['trained_head_count']}",
        "",
        "## Families",
    ]
    for family, info in payload["family_summary"].items():
        lines.extend(
            [
                f"### {family}",
                f"- domains: {', '.join(info['domains'])}",
                f"- trained heads: {info['trained_heads']}",
                f"- blocked heads: {info['blocked_heads']}",
                f"- small-domain overfit: {info['small_domain_overfit']}",
                f"- domain overfit warning: {info['domain_overfit_warning']}",
                "",
            ]
        )
    lines.extend(["## Domain Balance By Family"])
    for family, rows in payload.get("domain_balance_summary", {}).items():
        lines.extend(
            [
                f"### {family}",
                "| domain | raw train pairs | raw val pairs | effective/epoch | sample factor | mean train acc | mean val acc | max gap |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| {domain} | {raw_available_training_pairs} | {raw_validation_pairs} | "
                "{effective_sampled_pairs_per_epoch} | {sample_factor:.3f} | "
                "{mean_train_pairwise_accuracy:.3f} | {mean_validation_pairwise_accuracy:.3f} | {max_train_val_gap:.3f} |".format(**row)
            )
        lines.append("")
    lines.extend(["## Per-Head Domain Balance Details"])
    for row in payload["heads"]:
        lines.append(f"- {row['head_group']} / {row['config']} / {row['architecture']}: `{json.dumps(row['domain_balance'], default=str)[:900]}`")
    if payload.get("blockers"):
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in payload["blockers"][:40])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_verdict(trained_families: set[str], requested: set[str]) -> str:
    major = {"MIX_CODE_REASONING", "MIX_CODE_SCIENCE", "MIX_OBJECTIVE_ALL"}
    if requested and requested.issubset(trained_families):
        return "READY"
    if major.issubset(trained_families):
        return "PARTIAL"
    return "BLOCKED"


def main() -> None:
    args = parse_args()
    splits = load_json(args.splits)
    features = load_pt(args.features)
    pooled_by_uid = feature_map(features)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    heads: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    blockers: list[str] = []
    family_summary: dict[str, Any] = {}
    hh_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    mixed_families = features.get("mixed_families") or splits.get("mixed_families", {})

    for family, domains in mixed_families.items():
        trained = 0
        blocked = 0
        fam_small_overfit = False
        fam_domain_warning = False
        for config in MIXED_CONFIGS:
            if config not in MATH_CONFIGS:
                continue
            for architecture in ARCHITECTURES:
                head, metrics = train_balanced_head(
                    family=family,
                    domains=list(domains),
                    architecture=architecture,
                    config=config,
                    features=features,
                    pooled_by_uid=pooled_by_uid,
                    args=args,
                    device=device,
                    hh_cache=hh_cache,
                )
                if head is None:
                    blocked += 1
                    blockers.append(str(metrics.get("blocker", metrics)))
                    continue
                row = {
                    "head_group": family,
                    "architecture": architecture,
                    "family_architecture": f"{family}_{architecture}",
                    "config": config,
                    "dim": config_dim(config),
                    "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "train_metrics": metrics,
                }
                heads.append(row)
                compact.append(compact_head(row))
                trained += 1
                fam_small_overfit = fam_small_overfit or bool(metrics.get("small_domain_overfit"))
                fam_domain_warning = fam_domain_warning or bool(metrics.get("domain_overfit_warning"))
        family_summary[family] = {
            "domains": list(domains),
            "trained_heads": trained,
            "blocked_heads": blocked,
            "small_domain_overfit": fam_small_overfit,
            "domain_overfit_warning": fam_domain_warning,
        }

    trained_families = {family for family, info in family_summary.items() if info["trained_heads"] > 0}
    verdict = choose_verdict(trained_families, set(mixed_families))
    small_overfit = any(info["small_domain_overfit"] for info in family_summary.values())
    domain_warning = any(info["domain_overfit_warning"] for info in family_summary.values())
    pt_payload = {
        "meta": {
            "mixed_tap_training_verdict": verdict,
            "splits_json": repo_path(output_path(args.splits)),
            "features_pt": repo_path(output_path(args.features)),
            "configs": list(MIXED_CONFIGS),
            "architectures": list(ARCHITECTURES),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "small_domain_overfit": small_overfit,
            "domain_overfit_warning": domain_warning,
            "trained_head_count": len(heads),
        },
        "heads": heads,
        "family_summary": family_summary,
        "blockers": blockers,
    }
    out = output_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_payload, out)
    json_payload = {
        "meta": pt_payload["meta"],
        "family_summary": family_summary,
        "heads": compact,
        "domain_balance_summary": summarize_domain_balance(compact),
        "blockers": blockers,
    }
    write_json(output_path(args.output_json), json_payload)
    write_markdown(output_path(args.output_md), json_payload)
    print(f"MIXED_TAP_TRAINING_VERDICT = {verdict}")
    print(f"trained_head_count = {len(heads)}")
    print(f"SMALL_DOMAIN_OVERFIT = {small_overfit}")
    print(f"DOMAIN_OVERFIT_WARNING = {domain_warning}")
    print(f"wrote {repo_path(out)}")


if __name__ == "__main__":
    main()
