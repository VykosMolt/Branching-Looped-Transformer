"""Compute-matched partial-trajectory routing experiment for BG."""
from __future__ import annotations

import itertools
import json
import random
import time
import traceback
from collections import Counter, defaultdict

import torch

from bg_steering_suite_lib import (
    REPORT_ROOT,
    SEED,
    OuroTextGenerator,
    branch_rows_for_task,
    continuation_budget,
    continuation_prompt,
    domain_hint_for_task,
    domains_allowed_by_reachability,
    evaluate_output,
    load_branch_pools,
    load_reachability,
    load_task_suite,
    rel,
    summarize_success,
    task_by_id,
    verdict_from_delta,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "partial_routing_results.json"
OUT_PARTIAL = REPORT_ROOT / "partial_routing_results.partial.json"
OUT_MD = REPORT_ROOT / "partial_routing_results.md"
FEATURE_PT = REPORT_ROOT / "partial_branch_features.pt"


def _feature_map() -> dict[tuple[str, int], torch.Tensor]:
    payload = torch.load(FEATURE_PT, map_location="cpu", weights_only=False)
    return {(str(r["task_id"]), int(r["branch_id"])): r["features"] for r in payload.get("records", [])}


def _expected_random_topk(successes: list[bool], k: int) -> float:
    n = len(successes)
    if n == 0:
        return 0.0
    k = min(k, n)
    combos = list(itertools.combinations(range(n), k))
    return sum(any(successes[i] for i in combo) for combo in combos) / max(len(combos), 1)


def _rank_with_controller(controller, feats: list[torch.Tensor], task: dict) -> dict:
    result = controller.rank_candidates(torch.stack(feats), domain_hint=domain_hint_for_task(task), mode="conservative", return_details=True)
    diag = controller.select_best(torch.stack(feats), domain_hint=domain_hint_for_task(task), mode="diagnostic_all", return_details=True)
    vote = controller.rank_candidates(torch.stack(feats), domain_hint=domain_hint_for_task(task), mode="experimental_vote", return_details=True)
    return {"conservative": result, "diagnostic_all": diag, "experimental_vote": vote}


def main() -> int:
    started = time.time()
    reachability = load_reachability()
    allowed = domains_allowed_by_reachability()
    if not allowed:
        payload = {
            "BG_PARTIAL_ROUTING_VERDICT": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
            "skipped_reason": "reachability gate did not pass any non-devil domain",
            "reachability_verdict": reachability.get("BG_REACHABILITY_GATE_VERDICT"),
            "GENERATOR_REACHABILITY_LIMITED": True,
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Partial Trajectory Routing", "", "BG_PARTIAL_ROUTING_VERDICT = INSUFFICIENT", "", "- skipped: reachability gate did not pass any non-devil domain"])
        print("BG_PARTIAL_ROUTING_VERDICT = INSUFFICIENT")
        return 0

    tasks = [t for t in load_task_suite() if t["domain"] in allowed or t.get("is_devil")]
    branch_payload = load_branch_pools()
    fmap = _feature_map()
    from src.evaluator.bg_controller import BGController

    controller = BGController.from_artifacts(device="cpu")
    completed = {}
    if OUT_PARTIAL.exists():
        try:
            old = json.loads(OUT_PARTIAL.read_text(encoding="utf-8"))
            completed = {str(r["task_id"]): r for r in old.get("task_results", [])}
        except Exception:
            completed = {}
    task_results = list(completed.values())
    generator: OuroTextGenerator | None = None
    try:
        generator = OuroTextGenerator(device="cuda")
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in completed:
                continue
            branches = branch_rows_for_task(branch_payload, task_id)
            branches = [b for b in branches if (task_id, int(b["branch_id"])) in fmap and b.get("initial_partial_text")]
            if len(branches) < 2:
                continue
            feats = [fmap[(task_id, int(b["branch_id"]))] for b in branches]
            rankings = _rank_with_controller(controller, feats, task)
            ranking = list(rankings["conservative"]["ranking"])
            continuations = []
            for branch in branches:
                prompt = continuation_prompt(task, branch.get("initial_partial_text", ""))
                try:
                    gen = generator.generate(
                        prompt,
                        max_new_tokens=continuation_budget(task),
                        temperature=0.7,
                        top_p=0.95,
                        seed=SEED + int(task.get("suite_index", 0)) * 211 + int(branch["branch_id"]),
                    )
                    final_text = (branch.get("initial_partial_text", "") + "\n" + gen["text"]).strip()
                    evaluation = evaluate_output(task, final_text)
                    continuations.append(
                        {
                            "branch_id": int(branch["branch_id"]),
                            "final_text": final_text,
                            "continuation": gen,
                            "evaluation": evaluation,
                        }
                    )
                except Exception as exc:
                    continuations.append(
                        {
                            "branch_id": int(branch["branch_id"]),
                            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                            "evaluation": {"evaluable": False, "success": False},
                        }
                    )
            success_by_branch = {int(row["branch_id"]): bool(row.get("evaluation", {}).get("success")) for row in continuations}
            successes_ordered = [success_by_branch.get(int(b["branch_id"]), False) for b in branches]
            first_branch = int(branches[0]["branch_id"])
            bg_top1 = int(branches[ranking[0]]["branch_id"])
            bg_top2 = [int(branches[i]["branch_id"]) for i in ranking[:2]]
            vote_rank = list(rankings["experimental_vote"]["ranking"])
            vote_top1 = int(branches[vote_rank[0]]["branch_id"]) if vote_rank else -1
            diag_selected = {
                name: int(branches[idx]["branch_id"]) if 0 <= idx < len(branches) else -1
                for name, idx in rankings["diagnostic_all"].get("selected_by_head", {}).items()
            }
            row = {
                "task_id": task_id,
                "domain": task["domain"],
                "is_devil": bool(task.get("is_devil")),
                "branch_ids": [int(b["branch_id"]) for b in branches],
                "rankings": {
                    "conservative": {
                        "ranking_branch_ids": [int(branches[i]["branch_id"]) for i in ranking],
                        "score_matrix": rankings["conservative"]["score_matrix"].tolist(),
                        "wins": rankings["conservative"]["wins"].tolist(),
                        "margin_sum": rankings["conservative"]["margin_sum"].tolist(),
                    },
                    "diagnostic_selected": diag_selected,
                    "experimental_vote_top1": vote_top1,
                },
                "continuations": continuations,
                "policy_success": {
                    "oracle_continue_all": any(successes_ordered),
                    "random_top1_expected": _expected_random_topk(successes_ordered, 1),
                    "random_top2_expected": _expected_random_topk(successes_ordered, 2),
                    "first_branch": success_by_branch.get(first_branch, False),
                    "bg_top1_conservative": success_by_branch.get(bg_top1, False),
                    "bg_top2_conservative": any(success_by_branch.get(bid, False) for bid in bg_top2),
                    "experimental_vote_top1": success_by_branch.get(vote_top1, False),
                },
                "token_usage": {
                    "continued_tokens_all": sum(int(c.get("continuation", {}).get("token_count", 0)) for c in continuations),
                    "bg_top1_tokens": next((int(c.get("continuation", {}).get("token_count", 0)) for c in continuations if c["branch_id"] == bg_top1), 0),
                },
            }
            task_results.append(row)
            write_json(OUT_PARTIAL, {"complete": False, "task_results": task_results})
    finally:
        if generator is not None:
            generator.cleanup()

    evaluable = [r for r in task_results if any(c.get("evaluation", {}).get("evaluable") for c in r.get("continuations", [])) and not r.get("is_devil")]
    bg1 = [1.0 if r["policy_success"]["bg_top1_conservative"] else 0.0 for r in evaluable]
    rand1 = [float(r["policy_success"]["random_top1_expected"]) for r in evaluable]
    bg2 = [1.0 if r["policy_success"]["bg_top2_conservative"] else 0.0 for r in evaluable]
    rand2 = [float(r["policy_success"]["random_top2_expected"]) for r in evaluable]
    bg1_rate = sum(bg1) / max(len(bg1), 1)
    rand1_rate = sum(rand1) / max(len(rand1), 1)
    bg2_rate = sum(bg2) / max(len(bg2), 1)
    rand2_rate = sum(rand2) / max(len(rand2), 1)
    oracle_rate = sum(1.0 if r["policy_success"]["oracle_continue_all"] else 0.0 for r in evaluable) / max(len(evaluable), 1)
    delta = max(bg1_rate - rand1_rate, bg2_rate - rand2_rate)
    verdict = "INSUFFICIENT" if oracle_rate < 0.10 else verdict_from_delta(delta, len(evaluable), min_n=20)
    by_domain = {}
    for domain in sorted({r["domain"] for r in evaluable}):
        rows = [r for r in evaluable if r["domain"] == domain]
        by_domain[domain] = {
            "n": len(rows),
            "oracle_success_rate": sum(r["policy_success"]["oracle_continue_all"] for r in rows) / max(len(rows), 1),
            "bg_top1_success": sum(r["policy_success"]["bg_top1_conservative"] for r in rows) / max(len(rows), 1),
            "random_top1_success": sum(float(r["policy_success"]["random_top1_expected"]) for r in rows) / max(len(rows), 1),
            "bg_top2_success": sum(r["policy_success"]["bg_top2_conservative"] for r in rows) / max(len(rows), 1),
            "random_top2_success": sum(float(r["policy_success"]["random_top2_expected"]) for r in rows) / max(len(rows), 1),
        }
    payload = {
        "BG_PARTIAL_ROUTING_VERDICT": verdict,
        "verdict": verdict,
        "GENERATOR_REACHABILITY_LIMITED": oracle_rate < 0.10,
        "metrics": {
            "evaluable_tasks": len(evaluable),
            "oracle_success_rate": oracle_rate,
            "random_top1_success": rand1_rate,
            "first_branch_success": sum(r["policy_success"]["first_branch"] for r in evaluable) / max(len(evaluable), 1),
            "bg_top1_success": bg1_rate,
            "random_top2_success": rand2_rate,
            "bg_top2_success": bg2_rate,
            "bg_top1_lift_vs_random": bg1_rate - rand1_rate,
            "bg_top2_lift_vs_random": bg2_rate - rand2_rate,
            "bg_topk_oracle_gap": oracle_rate - bg2_rate,
        },
        "by_domain": by_domain,
        "task_results": task_results,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_json(OUT_PARTIAL, payload)
    lines = [
        "# BG Partial-Trajectory Routing Results (2026-05-18)",
        "",
        f"BG_PARTIAL_ROUTING_VERDICT = {verdict}",
        "",
        f"- metrics: `{payload['metrics']}`",
        f"- by_domain: `{by_domain}`",
    ]
    write_md(OUT_MD, lines)
    print(f"BG_PARTIAL_ROUTING_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
