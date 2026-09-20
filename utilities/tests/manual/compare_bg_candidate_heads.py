"""Compare candidate BG controller heads on the policy simulation bundle."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402


BUNDLE_PT = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "bg_candidate_head_comparison_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_candidate_head_comparison_2026-05-17.md"

OBJECTIVE_DOMAINS = {
    "CLEAN_GSM8K_EXPANDED",
    "CODE_RUNNABLE_DIAGNOSTIC",
    "CODE_STRICT_CLEAN_ALL16",
    "REASONING_NATURAL_DISTRACTOR",
    "REASONING_TRACE",
    "SCIENCE_OVERALL",
}
MAIN_ROLES = [
    "HH_GENERAL",
    "CODE_SPECIALIST",
    "OBJECTIVE_MIXED_PRIMARY",
    "OBJECTIVE_MIXED_BROAD",
    "SCIENCE_AWARE_MIXED",
    "RISKY_HH_OBJECTIVE",
]
PAIR_DIAGNOSTICS = [
    ("HH_GENERAL", "OBJECTIVE_MIXED_PRIMARY"),
    ("CODE_SPECIALIST", "OBJECTIVE_MIXED_PRIMARY"),
    ("HH_GENERAL", "CODE_SPECIALIST"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(BUNDLE_PT))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(output_path(path), map_location="cpu", weights_only=False)


def safe_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return float(mean(vals)) if vals else float("nan")


def ci(vals: list[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in vals if not math.isnan(float(v)))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan")}
    lo = vals[max(0, int(0.025 * (len(vals) - 1)))]
    hi = vals[min(len(vals) - 1, int(0.975 * (len(vals) - 1)))]
    return {"lo": lo, "hi": hi}


def head_metrics(bundle: dict[str, Any], key: str, domain: str) -> dict[str, Any]:
    return bundle["head_scores"][key][domain]["metrics"]


def candidate_keys(bundle: dict[str, Any]) -> list[str]:
    keys = [key for key, row in bundle["heads"].items() if row.get("candidate_head")]
    for key in bundle.get("role_primary", {}).values():
        if key not in keys:
            keys.append(key)
    return sorted(keys)


def primary_role_keys(bundle: dict[str, Any]) -> dict[str, str]:
    return {role: key for role, key in bundle.get("role_primary", {}).items() if key in bundle["heads"]}


def flatten_scores(bundle: dict[str, Any], key: str, domains: list[str] | None = None) -> tuple[list[float], list[bool], list[str]]:
    xs: list[float] = []
    correct: list[bool] = []
    ids: list[str] = []
    use_domains = domains or sorted(bundle["domains"])
    for domain in use_domains:
        records = bundle["domains"][domain]["records"]
        scored = bundle["head_scores"][key][domain]["records"]
        for rec, score_rec in zip(records, scored):
            for pair_idx, score in enumerate(score_rec["pair_scores"]):
                xs.append(float(score))
                correct.append(bool(score > 0))
                ids.append(f"{domain}::{rec['record_id']}::{pair_idx}")
    return xs, correct, ids


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def performance_table(bundle: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        info = bundle["heads"][key]
        per_domain = {}
        objective_vals = []
        for domain in sorted(bundle["domains"]):
            metrics = head_metrics(bundle, key, domain)
            oracle = bundle["oracle_domain_best"][domain]["pairwise_acc"]
            existing = bundle["existing_specialist_best"][domain]["pairwise_acc"]
            row = {
                "pairwise_acc": metrics["pairwise_acc"],
                "top1": metrics["top1"],
                "top1_over_random": metrics["top1_over_random"],
                "random_top1_baseline": metrics["random_top1_baseline"],
                "cycle_rate": float("nan"),
                "condorcet_winner_rate": float("nan"),
                "margin_mean": metrics["margin_mean"],
                "margin_std": metrics["margin_std"],
                "regret_vs_oracle_pairwise": metrics["pairwise_acc"] - oracle,
                "regret_vs_existing_specialist_pairwise": metrics["pairwise_acc"] - existing,
                "pair_total": metrics["pair_total"],
            }
            per_domain[domain] = row
            if domain in OBJECTIVE_DOMAINS:
                objective_vals.append(float(metrics["pairwise_acc"]))
        rows.append(
            {
                "head_key": key,
                "head_group": info["head_group"],
                "config": info["config"],
                "architecture": info["architecture"],
                "role_tags": info.get("role_tags", []),
                "deployable": bool(info.get("deployable", True)),
                "average_objective_pairwise": safe_mean(objective_vals),
                "hh_heldout_pairwise": per_domain.get("HH_HELDOUT20", {}).get("pairwise_acc"),
                "strict_clean_pairwise": per_domain.get("CODE_STRICT_CLEAN_ALL16", {}).get("pairwise_acc"),
                "per_domain": per_domain,
            }
        )
    return rows


def correlation_matrix(bundle: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    flat = {key: flatten_scores(bundle, key) for key in keys}
    rows: dict[str, Any] = {}
    for a in keys:
        rows[a] = {}
        xs, ac, _ = flat[a]
        for b in keys:
            ys, bc, _ = flat[b]
            n = min(len(xs), len(ys))
            if n == 0:
                rows[a][b] = {"score_correlation": float("nan"), "decision_agreement": float("nan"), "disagreement_rate": float("nan")}
                continue
            agreement = sum((xs[i] > 0) == (ys[i] > 0) for i in range(n)) / n
            rows[a][b] = {
                "score_correlation": corr(xs[:n], ys[:n]),
                "decision_agreement": agreement,
                "disagreement_rate": 1.0 - agreement,
                "margin_correlation": corr([abs(x) for x in xs[:n]], [abs(y) for y in ys[:n]]),
            }
    return rows


def error_overlap(bundle: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in sorted(bundle["domains"]):
        out[domain] = {}
        for i, a in enumerate(keys):
            _, ac, _ = flatten_scores(bundle, a, [domain])
            for b in keys[i + 1 :]:
                _, bc, _ = flatten_scores(bundle, b, [domain])
                n = min(len(ac), len(bc))
                both = sum(ac[j] and bc[j] for j in range(n))
                a_only = sum(ac[j] and not bc[j] for j in range(n))
                b_only = sum((not ac[j]) and bc[j] for j in range(n))
                neither = sum((not ac[j]) and (not bc[j]) for j in range(n))
                out[domain][f"{a}__VS__{b}"] = {
                    "both_correct": both,
                    "head_a_correct_head_b_wrong": a_only,
                    "head_a_wrong_head_b_correct": b_only,
                    "both_wrong": neither,
                    "n_pairs": n,
                    "head_a_fix_rate_total": a_only / n if n else float("nan"),
                    "head_b_fix_rate_total": b_only / n if n else float("nan"),
                }
    return out


def role_pair_overlap(bundle: dict[str, Any], role_keys: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pairs = [
        ("HH_GENERAL", "OBJECTIVE_MIXED_PRIMARY"),
        ("CODE_SPECIALIST", "OBJECTIVE_MIXED_PRIMARY"),
        ("HH_GENERAL", "CODE_SPECIALIST"),
    ]
    for domain in sorted(bundle["domains"]):
        out[domain] = {}
        for a_role, b_role in pairs:
            if a_role not in role_keys or b_role not in role_keys:
                continue
            a = role_keys[a_role]
            b = role_keys[b_role]
            _, ac, ids = flatten_scores(bundle, a, [domain])
            _, bc, _ = flatten_scores(bundle, b, [domain])
            n = min(len(ac), len(bc))
            a_only_examples = [ids[j] for j in range(n) if ac[j] and not bc[j]][:10]
            b_only_examples = [ids[j] for j in range(n) if (not ac[j]) and bc[j]][:10]
            out[domain][f"{a_role}__VS__{b_role}"] = {
                f"{a_role}_fixes_{b_role}_errors": sum(ac[j] and not bc[j] for j in range(n)),
                f"{b_role}_fixes_{a_role}_errors": sum((not ac[j]) and bc[j] for j in range(n)),
                "n_pairs": n,
                f"{a_role}_unique_examples": a_only_examples,
                f"{b_role}_unique_examples": b_only_examples,
            }
    return out


def margin_reliability(bundle: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    bins = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, float("inf")]
    out: dict[str, Any] = {}
    for key in keys:
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            correct = 0.0
            total = 0.0
            count = 0
            for domain in bundle["domains"]:
                for rec in bundle["head_scores"][key][domain]["records"]:
                    margin = float(rec["margin"])
                    if margin < lo or margin >= hi:
                        continue
                    correct += float(rec["pair_correct"])
                    total += float(rec["pair_total"])
                    count += 1
            acc = correct / total if total else float("nan")
            center = hi if math.isfinite(hi) else lo + 1.0
            rows.append({"bin": f"[{lo},{hi})", "records": count, "pairwise_acc": acc, "confidence_proxy": center})
        ece_terms = [
            abs(float(row["pairwise_acc"]) - min(float(row["confidence_proxy"]) / 2.0, 1.0)) * row["records"]
            for row in rows
            if row["records"] and not math.isnan(float(row["pairwise_acc"]))
        ]
        n = sum(row["records"] for row in rows)
        out[key] = {"bins": rows, "ece_like": sum(ece_terms) / n if n else float("nan")}
    return out


def disagreement_diagnostics(bundle: dict[str, Any], role_keys: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a_role, b_role in PAIR_DIAGNOSTICS:
        if a_role not in role_keys or b_role not in role_keys:
            continue
        a = role_keys[a_role]
        b = role_keys[b_role]
        out[f"{a_role}__VS__{b_role}"] = {}
        for domain in sorted(bundle["domains"]):
            ax, ac, _ = flatten_scores(bundle, a, [domain])
            bx, bc, _ = flatten_scores(bundle, b, [domain])
            n = min(len(ax), len(bx))
            disagreements = [(ax[i] > 0) != (bx[i] > 0) for i in range(n)]
            d_n = sum(disagreements)
            either_error_when_disagree = sum(disagreements[i] and (not ac[i] or not bc[i]) for i in range(n))
            both_error_when_agree = sum((not disagreements[i]) and (not ac[i]) and (not bc[i]) for i in range(n))
            out[f"{a_role}__VS__{b_role}"][domain] = {
                "n_pairs": n,
                "disagreements": d_n,
                "disagreement_rate": d_n / n if n else float("nan"),
                "either_head_error_given_disagreement": either_error_when_disagree / d_n if d_n else float("nan"),
                "both_error_given_agreement": both_error_when_agree / (n - d_n) if n > d_n else float("nan"),
            }
    return out


def bootstrap_domain_pairwise(bundle: dict[str, Any], key: str, domain: str, samples: int, rng: random.Random) -> dict[str, Any]:
    scored = bundle["head_scores"][key][domain]["records"]
    if not scored:
        return {"mean": float("nan"), "ci95": {"lo": float("nan"), "hi": float("nan")}}
    vals = []
    n = len(scored)
    for _ in range(samples):
        correct = 0
        total = 0
        for _j in range(n):
            rec = scored[rng.randrange(n)]
            correct += int(rec["pair_correct"])
            total += int(rec["pair_total"])
        vals.append(correct / total if total else float("nan"))
    return {"mean": safe_mean(vals), "ci95": ci(vals), "samples": samples}


def bootstrap_report(bundle: dict[str, Any], role_keys: dict[str, str], samples: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    out: dict[str, Any] = {}
    for role, key in role_keys.items():
        if role not in MAIN_ROLES:
            continue
        out[role] = {}
        for domain in sorted(bundle["domains"]):
            if domain in {"HH_200_DIAGNOSTIC"}:
                continue
            out[role][domain] = bootstrap_domain_pairwise(bundle, key, domain, samples, rng)
    return out


def complementarity_verdict(bundle: dict[str, Any], role_keys: dict[str, str], overlaps: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    roles = ["HH_GENERAL", "CODE_SPECIALIST", "OBJECTIVE_MIXED_PRIMARY"]
    fixes: dict[str, float] = {role: 0.0 for role in roles}
    for domain, domain_pairs in overlaps.items():
        for label, item in domain_pairs.items():
            for role in roles:
                key = f"{role}_fixes_"
                for metric, value in item.items():
                    if metric.startswith(key) and item.get("n_pairs", 0):
                        fixes[role] = max(fixes[role], float(value) / float(item["n_pairs"]))
    high = all(fixes[role] >= 0.05 for role in roles)
    obj_key = role_keys.get("OBJECTIVE_MIXED_PRIMARY")
    code_key = role_keys.get("CODE_SPECIALIST")
    hh_key = role_keys.get("HH_GENERAL")
    objective_dominates = False
    if obj_key and code_key and hh_key:
        strict_gap = (
            head_metrics(bundle, obj_key, "CODE_STRICT_CLEAN_ALL16")["pairwise_acc"]
            - head_metrics(bundle, code_key, "CODE_STRICT_CLEAN_ALL16")["pairwise_acc"]
        )
        better_than_hh = 0
        for domain in OBJECTIVE_DOMAINS:
            if domain in bundle["domains"]:
                if head_metrics(bundle, obj_key, domain)["pairwise_acc"] > head_metrics(bundle, hh_key, domain)["pairwise_acc"]:
                    better_than_hh += 1
        objective_dominates = strict_gap >= -0.03 and better_than_hh >= 2
    if high:
        verdict = "HIGH_COMPLEMENTARITY"
    elif objective_dominates:
        verdict = "OBJECTIVE_MIXED_DOMINATES_OBJECTIVE"
    elif bundle.get("domains"):
        verdict = "LOW_COMPLEMENTARITY"
    else:
        verdict = "INSUFFICIENT"
    return verdict, {"max_fix_rate_by_role": fixes, "objective_mixed_dominates_objective": objective_dominates}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BG Candidate Head Comparison (2026-05-17)",
        "",
        f"BG_HEAD_COMPARISON_VERDICT = {payload['meta']['bg_head_comparison_verdict']}",
        f"HEAD_COMPLEMENTARITY_VERDICT = {payload['meta']['head_complementarity_verdict']}",
        "",
        "## Candidate Heads",
        "| role/head | config | architecture | objective avg | HH heldout | strict clean |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(payload["per_head_performance"], key=lambda r: (not r["role_tags"], r["head_group"], r["config"], r["architecture"])):
        role = ",".join(row["role_tags"]) or row["head_group"]
        lines.append(
            f"| {role} | {row['config']} | {row['architecture']} | "
            f"{row['average_objective_pairwise']:.3f} | {row['hh_heldout_pairwise']:.3f} | {row['strict_clean_pairwise']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Complementarity",
            f"- max fix rates: `{json.dumps(payload['complementarity']['max_fix_rate_by_role'], default=str)}`",
            f"- objective mixed dominates objective: {payload['complementarity']['objective_mixed_dominates_objective']}",
            "",
            "## Disagreement Diagnostics",
        ]
    )
    for pair, by_domain in payload["disagreement_diagnostics"].items():
        vals = [row.get("disagreement_rate") for row in by_domain.values() if not math.isnan(float(row.get("disagreement_rate", float("nan"))))]
        lines.append(f"- {pair}: mean disagreement={safe_mean(vals):.3f}")
    lines.extend(["", "## Bootstrap CI Examples"])
    for role, by_domain in payload["bootstrap_ci"].items():
        for domain, item in by_domain.items():
            if domain in {"CODE_STRICT_CLEAN_ALL16", "HH_HELDOUT20", "REASONING_TRACE", "SCIENCE_OVERALL"}:
                ci95 = item["ci95"]
                lines.append(f"- {role} / {domain}: mean={item['mean']:.3f}, ci95=[{ci95['lo']:.3f},{ci95['hi']:.3f}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle = load_pt(args.bundle)
    if bundle.get("meta", {}).get("bg_policy_eval_bundle_verdict") == "BLOCKED":
        payload = {
            "meta": {
                "bg_head_comparison_verdict": "BLOCKED",
                "head_complementarity_verdict": "INSUFFICIENT",
                "bundle": repo_path(output_path(args.bundle)),
            },
            "blockers": ["eval bundle blocked"],
        }
        write_json(output_path(args.output), payload)
        write_markdown(output_path(args.output_md), payload)
        print("BG_HEAD_COMPARISON_VERDICT = BLOCKED")
        return
    keys = candidate_keys(bundle)
    role_keys = primary_role_keys(bundle)
    perf = performance_table(bundle, keys)
    corr_matrix = correlation_matrix(bundle, keys)
    overlaps = error_overlap(bundle, list(role_keys.values()))
    unique = role_pair_overlap(bundle, role_keys)
    margin = margin_reliability(bundle, list(role_keys.values()))
    disagreement = disagreement_diagnostics(bundle, role_keys)
    boot = bootstrap_report(bundle, role_keys, int(args.bootstrap_samples), int(args.seed))
    comp_verdict, comp_details = complementarity_verdict(bundle, role_keys, unique)
    verdict = "READY" if keys and role_keys else ("PARTIAL" if keys else "BLOCKED")
    payload = {
        "meta": {
            "bg_head_comparison_verdict": verdict,
            "head_complementarity_verdict": comp_verdict,
            "bundle": repo_path(output_path(args.bundle)),
            "candidate_head_count": len(keys),
            "role_primary": role_keys,
            "bootstrap_samples": int(args.bootstrap_samples),
        },
        "per_head_performance": perf,
        "head_correlation_matrix": corr_matrix,
        "error_overlap": overlaps,
        "unique_win_analysis": unique,
        "margin_reliability": margin,
        "disagreement_diagnostics": disagreement,
        "bootstrap_ci": boot,
        "complementarity": comp_details,
        "blockers": [],
    }
    write_json(output_path(args.output), payload)
    write_markdown(output_path(args.output_md), payload)
    print(f"BG_HEAD_COMPARISON_VERDICT = {verdict}")
    print(f"HEAD_COMPLEMENTARITY_VERDICT = {comp_verdict}")
    print(f"candidate_heads = {len(keys)}")
    print(f"wrote {repo_path(output_path(args.output))}")


if __name__ == "__main__":
    main()
