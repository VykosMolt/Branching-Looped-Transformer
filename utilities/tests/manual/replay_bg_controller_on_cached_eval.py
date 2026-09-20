"""Replay the read-only BG controller over the cached policy simulator eval set."""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_controller import BGController  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "opi/taps/probes"
BUNDLE_PT = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.pt"
BUNDLE_JSON = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.json"
SIM_JSON = REPORT_DIR / "bg_controller_policy_simulation_2026-05-17.json"
HEAD_REGISTRY_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "bg_controller_replay_2026-05-18.json"
OUTPUT_MD = REPORT_DIR / "bg_controller_replay_2026-05-18.md"

REPLAY_MODES = ("conservative", "experimental_vote", "code_backup")
TOL = 0.03


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def resolve_repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return float(mean(vals)) if vals else float("nan")


def hh_pooled_to_tensor(pooled: dict[int, torch.Tensor]) -> torch.Tensor:
    return torch.stack([pooled[24], pooled[36], pooled[47]], dim=0).detach().cpu().to(torch.float32)


def feature_maps(features: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, Any]]]]:
    by_uid = {
        str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32)
        for row in features.get("candidate_features", []) or []
    }
    record_domains = {
        name: list(rows or [])
        for name, rows in (features.get("record_domains") or {}).items()
    }
    return by_uid, record_domains


def candidates_for_record(
    *,
    domain: str,
    record: dict[str, Any],
    by_uid: dict[str, torch.Tensor],
    record_domains: dict[str, list[dict[str, Any]]],
    hh_payload: dict[str, Any],
) -> tuple[list[torch.Tensor], list[bool], list[tuple[int, int]]]:
    labels = [bool(x) for x in record.get("labels", [])]
    pair_indices = [tuple(pair) for pair in record.get("pair_indices", [])]
    if domain.startswith("HH_"):
        pair_index = int(record["pair_index"])
        pack = hh_payload["packs"][pair_index]
        candidates = [
            hh_pooled_to_tensor(pack["chosen"]["pooled"]),
            hh_pooled_to_tensor(pack["rejected"]["pooled"]),
        ]
        return candidates, labels, pair_indices
    if domain == "CLEAN_GSM8K_EXPANDED":
        row = record_domains["GSM8K"][int(record["record_index"])]
        pooled = row["pooled"].detach().cpu().to(torch.float32)
        return [pooled[i] for i in range(int(pooled.shape[0]))], labels, pair_indices
    if domain == "CODE_RUNNABLE_DIAGNOSTIC":
        row = record_domains["CODE_RUNNABLE"][int(record["record_index"])]
        pooled = row["pooled"].detach().cpu().to(torch.float32)
        return [pooled[i] for i in range(int(pooled.shape[0]))], labels, pair_indices
    candidates = []
    missing = []
    for uid in record.get("candidate_uids", []) or []:
        key = str(uid)
        if key not in by_uid:
            missing.append(key)
            continue
        candidates.append(by_uid[key])
    if missing:
        raise RuntimeError(f"missing candidate feature uid(s) for {domain}/{record.get('record_id')}: {missing[:5]}")
    return candidates, labels, pair_indices


def has_cycle(mat: torch.Tensor, a: int, b: int, c: int) -> bool:
    return bool(
        (mat[a, b] > 0 and mat[b, c] > 0 and mat[c, a] > 0)
        or (mat[a, c] > 0 and mat[c, b] > 0 and mat[b, a] > 0)
    )


def cycle_counts(mat: torch.Tensor) -> tuple[int, int]:
    n = int(mat.shape[0])
    if n < 3:
        return 0, 0
    total = 0
    cyclic = 0
    for a, b, c in itertools.combinations(range(n), 3):
        total += 1
        cyclic += int(has_cycle(mat, a, b, c))
    return cyclic, total


def ranking_from_matrix(mat: torch.Tensor) -> list[int]:
    n = int(mat.shape[0])
    mask = ~torch.eye(n, dtype=torch.bool)
    wins = ((mat > 0) & mask).sum(dim=1)
    margin_sum = (mat * mask.to(mat.dtype)).sum(dim=1)
    return sorted(range(n), key=lambda idx: (-int(wins[idx].item()), -float(margin_sum[idx].item()), idx))


def score_record(mat: torch.Tensor, labels: list[bool], pair_indices: list[tuple[int, int]]) -> dict[str, Any]:
    pair_scores = [float(mat[i, j]) for i, j in pair_indices]
    pair_correct = sum(1 for score in pair_scores if score > 0.0)
    if len(labels) == 2 and pair_indices == [(0, 1)]:
        pred = 0 if float(mat[0, 1]) > 0.0 else 1
    else:
        ranking = ranking_from_matrix(mat)
        pred = int(ranking[0]) if ranking else -1
    cyclic, triplets = cycle_counts(mat)
    return {
        "pair_correct": int(pair_correct),
        "pair_total": int(len(pair_indices)),
        "top1_correct": bool(labels[pred]) if pred >= 0 and pred < len(labels) else False,
        "pred_index": pred,
        "cycle_count": cyclic,
        "triplet_count": triplets,
    }


def summarize_records(rows: list[dict[str, Any]], random_top1_baseline: float) -> dict[str, Any]:
    pair_total = sum(int(row["pair_total"]) for row in rows)
    pair_correct = sum(int(row["pair_correct"]) for row in rows)
    top_total = len(rows)
    top_correct = sum(1 for row in rows if row.get("top1_correct"))
    triplets = sum(int(row["triplet_count"]) for row in rows)
    cycles = sum(int(row["cycle_count"]) for row in rows)
    return {
        "pairwise_acc": pair_correct / pair_total if pair_total else float("nan"),
        "top1": top_correct / top_total if top_total else float("nan"),
        "top1_over_random": top_correct / top_total - random_top1_baseline if top_total else float("nan"),
        "random_top1_baseline": random_top1_baseline,
        "pair_correct": pair_correct,
        "pair_total": pair_total,
        "record_total": top_total,
        "cycle_rate": cycles / triplets if triplets else 0.0,
        "cycle_count": cycles,
        "triplet_count": triplets,
    }


def evaluate_matrix_mode(
    *,
    controller: BGController,
    bundle: dict[str, Any],
    by_uid: dict[str, torch.Tensor],
    record_domains: dict[str, list[dict[str, Any]]],
    hh_payload: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    route_checks: dict[str, str] = {}
    for domain, spec in bundle["domains"].items():
        rows = []
        for record in spec["records"]:
            candidates, labels, pair_indices = candidates_for_record(
                domain=domain,
                record=record,
                by_uid=by_uid,
                record_domains=record_domains,
                hh_payload=hh_payload,
            )
            mat = controller.tournament_scores(candidates, domain_hint=domain, mode=mode)
            assert isinstance(mat, torch.Tensor)
            rows.append(score_record(mat, labels, pair_indices))
            if domain not in route_checks and len(candidates) >= 2 and mode == "conservative":
                detail = controller.score_pair(candidates[0], candidates[1], domain_hint=domain, mode=mode, return_details=True)
                route_checks[domain] = str(detail["selected_head"])
        by_domain[domain] = summarize_records(rows, float(spec["random_top1_baseline"]))
    objective_domains = set(bundle.get("meta", {}).get("objective_domains") or [])
    objective_avg = safe_mean([row["pairwise_acc"] for name, row in by_domain.items() if name in objective_domains])
    all_rows = []
    for domain, metrics in by_domain.items():
        all_rows.append(
            {
                "pair_correct": metrics["pair_correct"],
                "pair_total": metrics["pair_total"],
                "top1_correct": False,
                "cycle_count": metrics["cycle_count"],
                "triplet_count": metrics["triplet_count"],
            }
        )
    total_pairs = sum(row["pair_total"] for row in by_domain.values())
    total_correct = sum(row["pair_correct"] for row in by_domain.values())
    return {
        "domain_breakdown": by_domain,
        "objective_average_pairwise": objective_avg,
        "overall_pairwise": total_correct / total_pairs if total_pairs else float("nan"),
        "route_checks": route_checks,
    }


def evaluate_diagnostic_all(
    *,
    controller: BGController,
    bundle: dict[str, Any],
    by_uid: dict[str, torch.Tensor],
    record_domains: dict[str, list[dict[str, Any]]],
    hh_payload: dict[str, Any],
) -> dict[str, Any]:
    head_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for domain, spec in bundle["domains"].items():
        for record in spec["records"]:
            candidates, labels, pair_indices = candidates_for_record(
                domain=domain,
                record=record,
                by_uid=by_uid,
                record_domains=record_domains,
                hh_payload=hh_payload,
            )
            matrices = controller.tournament_scores(candidates, domain_hint=domain, mode="diagnostic_all")
            assert isinstance(matrices, dict)
            for head_name, mat in matrices.items():
                head_rows.setdefault(head_name, {}).setdefault(domain, []).append(score_record(mat, labels, pair_indices))
    out: dict[str, Any] = {}
    for head_name, by_domain_rows in head_rows.items():
        by_domain = {
            domain: summarize_records(rows, float(bundle["domains"][domain]["random_top1_baseline"]))
            for domain, rows in by_domain_rows.items()
        }
        total_pairs = sum(row["pair_total"] for row in by_domain.values())
        total_correct = sum(row["pair_correct"] for row in by_domain.values())
        out[head_name] = {
            "domain_breakdown": by_domain,
            "overall_pairwise": total_correct / total_pairs if total_pairs else float("nan"),
        }
    return out


def expected_conservative_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    role_primary = bundle["role_primary"]
    by_domain = {}
    routes = {}
    for domain in bundle["domains"]:
        role = "HH_GENERAL" if domain.startswith("HH_") else "OBJECTIVE_MIXED_PRIMARY"
        key = role_primary[role]
        routes[domain] = "hh_general" if role == "HH_GENERAL" else "objective_mixed"
        by_domain[domain] = dict(bundle["head_scores"][key][domain]["metrics"])
        by_domain[domain]["head_key"] = key
    objective_domains = set(bundle.get("meta", {}).get("objective_domains") or [])
    objective_avg = safe_mean([row["pairwise_acc"] for name, row in by_domain.items() if name in objective_domains])
    total_pairs = sum(int(row["pair_total"]) for row in by_domain.values())
    total_correct = sum(int(row["pair_correct"]) for row in by_domain.values())
    return {
        "domain_breakdown": by_domain,
        "objective_average_pairwise": objective_avg,
        "overall_pairwise": total_correct / total_pairs if total_pairs else float("nan"),
        "routes": routes,
    }


def compare_conservative(replay: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    domain_diffs = {}
    max_pairwise_diff = 0.0
    route_mismatches = []
    for domain, metrics in expected["domain_breakdown"].items():
        got = replay["domain_breakdown"].get(domain, {})
        pair_diff = abs(float(got.get("pairwise_acc", float("nan"))) - float(metrics.get("pairwise_acc", float("nan"))))
        top1_diff = abs(float(got.get("top1", float("nan"))) - float(metrics.get("top1", float("nan"))))
        max_pairwise_diff = max(max_pairwise_diff, pair_diff)
        domain_diffs[domain] = {
            "expected_pairwise": metrics.get("pairwise_acc"),
            "replay_pairwise": got.get("pairwise_acc"),
            "pairwise_abs_diff": pair_diff,
            "expected_top1": metrics.get("top1"),
            "replay_top1": got.get("top1"),
            "top1_abs_diff": top1_diff,
        }
        expected_route = expected["routes"][domain]
        got_route = replay.get("route_checks", {}).get(domain)
        if got_route is not None and got_route != expected_route:
            route_mismatches.append({"domain": domain, "expected": expected_route, "got": got_route})
    return {
        "max_pairwise_abs_diff": max_pairwise_diff,
        "within_tolerance": bool(max_pairwise_diff <= TOL and not route_mismatches),
        "route_mismatches": route_mismatches,
        "domain_diffs": domain_diffs,
    }


def compare_policy_to_sim(replay: dict[str, Any], sim: dict[str, Any], policy_name: str) -> dict[str, Any]:
    policy = (sim.get("policy_results") or {}).get(policy_name)
    if not policy:
        return {"available": False}
    diffs = {}
    max_diff = 0.0
    for domain, got in replay["domain_breakdown"].items():
        sim_metrics = (policy.get("domain_breakdown") or {}).get(domain)
        if not sim_metrics:
            continue
        diff = abs(float(got["pairwise_acc"]) - float(sim_metrics["pairwise_acc"]))
        max_diff = max(max_diff, diff)
        diffs[domain] = {
            "replay_pairwise": got["pairwise_acc"],
            "sim_pairwise": sim_metrics["pairwise_acc"],
            "pairwise_abs_diff": diff,
        }
    return {"available": True, "policy": policy_name, "max_pairwise_abs_diff": max_diff, "domain_diffs": diffs}


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    return "NA" if math.isnan(x) else f"{x:.3f}"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BG Controller Replay (2026-05-18)",
        "",
        f"BG_CONTROLLER_REPLAY_VERDICT = {payload['top_lines']['BG_CONTROLLER_REPLAY_VERDICT']}",
        "",
        "## Conservative Replay",
        f"- max pairwise abs diff vs locked route cached scores: `{payload['comparisons']['conservative']['max_pairwise_abs_diff']:.6f}`",
        f"- within tolerance: `{payload['comparisons']['conservative']['within_tolerance']}`",
        "",
        "| domain | replay pairwise | expected pairwise | replay top1 | expected top1 | route |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    conservative = payload["modes"]["conservative"]
    expected = payload["expected_conservative"]
    for domain in sorted(conservative["domain_breakdown"]):
        got = conservative["domain_breakdown"][domain]
        exp = expected["domain_breakdown"][domain]
        route = conservative["route_checks"].get(domain, "")
        lines.append(
            f"| {domain} | {rate(got['pairwise_acc'])} | {rate(exp['pairwise_acc'])} | "
            f"{rate(got['top1'])} | {rate(exp['top1'])} | {route} |"
        )
    lines.extend(
        [
            "",
            "## Other Modes",
            f"- experimental_vote objective average pairwise: `{rate(payload['modes']['experimental_vote']['objective_average_pairwise'])}`",
            f"- code_backup objective average pairwise: `{rate(payload['modes']['code_backup']['objective_average_pairwise'])}`",
            f"- experimental vote simulator comparison max diff: `{rate(payload['comparisons']['experimental_vote_vs_sim'].get('max_pairwise_abs_diff'))}`",
            "",
            "## Interpretation",
            payload["interpretation"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not BUNDLE_PT.exists() or not BUNDLE_JSON.exists():
        raise SystemExit("missing replay bundle")
    bundle = load_pt(BUNDLE_PT)
    _bundle_json = load_json(BUNDLE_JSON)
    sim = load_json(SIM_JSON) if SIM_JSON.exists() else {}
    registry = load_pt(HEAD_REGISTRY_PT)
    features_path = resolve_repo(bundle["meta"]["features_pt"])
    features = load_pt(features_path)
    by_uid, record_domains = feature_maps(features)
    hh_path = resolve_repo(registry["meta"]["hh_capture"])
    hh_payload = load_pt(hh_path)

    controller = BGController.from_artifacts(device="cpu")
    modes = {
        mode: evaluate_matrix_mode(
            controller=controller,
            bundle=bundle,
            by_uid=by_uid,
            record_domains=record_domains,
            hh_payload=hh_payload,
            mode=mode,
        )
        for mode in REPLAY_MODES
    }
    diagnostic_all = evaluate_diagnostic_all(
        controller=controller,
        bundle=bundle,
        by_uid=by_uid,
        record_domains=record_domains,
        hh_payload=hh_payload,
    )
    expected = expected_conservative_from_bundle(bundle)
    conservative_cmp = compare_conservative(modes["conservative"], expected)
    vote_cmp = compare_policy_to_sim(modes["experimental_vote"], sim, "GENERAL_AND_OBJECTIVE_VOTE_margin")
    if not conservative_cmp["within_tolerance"]:
        verdict = "FAIL"
        interpretation = "Conservative routing or scoring diverged from the locked head scores; debug controller scoring before integration."
    else:
        verdict = "PASS"
        interpretation = (
            "Conservative mode reproduces the locked v8.1 domain routing over cached features within tolerance. "
            "Experimental vote is replayed for validation only and is not part of the default routing gate."
        )

    payload = {
        "top_lines": {"BG_CONTROLLER_REPLAY_VERDICT": verdict},
        "inputs": {
            "bundle_pt": repo_path(BUNDLE_PT),
            "bundle_json": repo_path(BUNDLE_JSON),
            "simulation_json": repo_path(SIM_JSON),
            "features_pt": repo_path(features_path),
            "hh_capture": repo_path(hh_path),
        },
        "modes": modes,
        "diagnostic_all": diagnostic_all,
        "expected_conservative": expected,
        "comparisons": {
            "conservative": conservative_cmp,
            "experimental_vote_vs_sim": vote_cmp,
        },
        "interpretation": interpretation,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_md(OUTPUT_MD, payload)
    print(f"BG_CONTROLLER_REPLAY_VERDICT = {verdict}")
    print(f"conservative_max_pairwise_abs_diff = {conservative_cmp['max_pairwise_abs_diff']:.6f}")
    if vote_cmp.get("available"):
        print(f"experimental_vote_vs_sim_max_pairwise_abs_diff = {vote_cmp['max_pairwise_abs_diff']:.6f}")
    print(f"wrote {repo_path(OUTPUT_JSON)}")
    print(f"wrote {repo_path(OUTPUT_MD)}")
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
