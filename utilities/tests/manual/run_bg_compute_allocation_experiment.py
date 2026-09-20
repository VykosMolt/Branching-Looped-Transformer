"""BG compute allocation experiment over shared partial branch pools."""
from __future__ import annotations

import json
import random
import time
import traceback

import torch

from bg_steering_suite_lib import (
    REPORT_ROOT,
    SEED,
    OuroTextGenerator,
    branch_rows_for_task,
    domain_hint_for_task,
    domains_allowed_by_reachability,
    equal_budget_per_branch,
    continuation_prompt,
    evaluate_output,
    load_branch_pools,
    load_reachability,
    load_task_suite,
    rel,
    verdict_from_delta,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "compute_allocation_results.json"
OUT_PARTIAL = REPORT_ROOT / "compute_allocation_results.partial.json"
OUT_MD = REPORT_ROOT / "compute_allocation_results.md"
FEATURE_PT = REPORT_ROOT / "partial_branch_features.pt"


def _feature_map() -> dict[tuple[str, int], torch.Tensor]:
    payload = torch.load(FEATURE_PT, map_location="cpu", weights_only=False)
    return {(str(r["task_id"]), int(r["branch_id"])): r["features"] for r in payload.get("records", [])}


def _weighted_budgets(branch_ids: list[int], ranking: list[int], total: int) -> dict[int, int]:
    if not branch_ids:
        return {}
    n = len(branch_ids)
    top1 = branch_ids[ranking[0]] if ranking else branch_ids[0]
    top2 = branch_ids[ranking[1]] if len(ranking) > 1 else None
    budgets = {bid: 0 for bid in branch_ids}
    budgets[top1] += max(1, int(total * 0.50))
    if top2 is not None:
        budgets[top2] += max(1, int(total * 0.30))
    remaining = max(0, total - sum(budgets.values()))
    rest = [bid for bid in branch_ids if bid not in {top1, top2}]
    targets = rest or branch_ids
    for idx in range(remaining):
        budgets[targets[idx % len(targets)]] += 1
    return budgets


def _run_policy(generator, task, branches, budgets, seed_offset: int) -> dict:
    rows = []
    for branch in branches:
        bid = int(branch["branch_id"])
        budget = max(1, int(budgets.get(bid, 1)))
        try:
            gen = generator.generate(
                continuation_prompt(task, branch.get("initial_partial_text", "")),
                max_new_tokens=budget,
                temperature=0.7,
                top_p=0.95,
                seed=SEED + seed_offset + int(task.get("suite_index", 0)) * 307 + bid,
            )
            final_text = (branch.get("initial_partial_text", "") + "\n" + gen["text"]).strip()
            evaluation = evaluate_output(task, final_text)
            rows.append({"branch_id": bid, "budget": budget, "generation": gen, "evaluation": evaluation})
        except Exception as exc:
            rows.append(
                {
                    "branch_id": bid,
                    "budget": budget,
                    "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    "evaluation": {"evaluable": False, "success": False},
                }
            )
    return {
        "budgets": budgets,
        "branches": rows,
        "success": any(bool(row.get("evaluation", {}).get("success")) for row in rows),
        "tokens": sum(int(row.get("generation", {}).get("token_count", 0)) for row in rows),
    }


def main() -> int:
    started = time.time()
    allowed = domains_allowed_by_reachability()
    if not allowed:
        payload = {
            "BG_COMPUTE_ALLOCATION_VERDICT": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
            "skipped_reason": "reachability gate did not pass any non-devil domain",
            "reachability_verdict": load_reachability().get("BG_REACHABILITY_GATE_VERDICT"),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Compute Allocation Results", "", "BG_COMPUTE_ALLOCATION_VERDICT = INSUFFICIENT", "", "- skipped: reachability gate did not pass"])
        print("BG_COMPUTE_ALLOCATION_VERDICT = INSUFFICIENT")
        return 0

    tasks = [t for t in load_task_suite() if t["domain"] in allowed and not t.get("is_devil")]
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
            branches = [b for b in branch_rows_for_task(branch_payload, task_id) if (task_id, int(b["branch_id"])) in fmap and b.get("initial_partial_text")]
            if len(branches) < 2:
                continue
            branch_ids = [int(b["branch_id"]) for b in branches]
            feats = [fmap[(task_id, bid)] for bid in branch_ids]
            rank_result = controller.rank_candidates(torch.stack(feats), domain_hint=domain_hint_for_task(task), mode="conservative", return_details=True)
            ranking = list(rank_result["ranking"])
            n = len(branch_ids)
            small = equal_budget_per_branch(task)
            total = n * small
            equal_budgets = {bid: small for bid in branch_ids}
            rng = random.Random(SEED + int(task.get("suite_index", 0)))
            random_rank = list(range(n))
            rng.shuffle(random_rank)
            random_budgets = _weighted_budgets(branch_ids, random_rank, total)
            bg_budgets = _weighted_budgets(branch_ids, ranking, total)
            policies = {
                "EQUAL_BUDGET": _run_policy(generator, task, branches, equal_budgets, 1),
                "RANDOM_WEIGHTED_BUDGET": _run_policy(generator, task, branches, random_budgets, 2),
                "BG_WEIGHTED_BUDGET": _run_policy(generator, task, branches, bg_budgets, 3),
            }
            successful = [idx for idx, bid in enumerate(branch_ids) if policies["EQUAL_BUDGET"]["branches"][idx].get("evaluation", {}).get("success")]
            oracle_rank = successful + [i for i in range(n) if i not in successful]
            oracle_budgets = _weighted_budgets(branch_ids, oracle_rank, total)
            policies["ORACLE_WEIGHTED_BUDGET"] = _run_policy(generator, task, branches, oracle_budgets, 4)
            row = {
                "task_id": task_id,
                "domain": task["domain"],
                "branch_ids": branch_ids,
                "ranking_branch_ids": [branch_ids[i] for i in ranking],
                "total_budget": total,
                "policies": policies,
            }
            task_results.append(row)
            write_json(OUT_PARTIAL, {"complete": False, "task_results": task_results})
    finally:
        if generator is not None:
            generator.cleanup()

    evaluable = task_results
    equal = [1.0 if r["policies"]["EQUAL_BUDGET"]["success"] else 0.0 for r in evaluable]
    random_weighted = [1.0 if r["policies"]["RANDOM_WEIGHTED_BUDGET"]["success"] else 0.0 for r in evaluable]
    bg = [1.0 if r["policies"]["BG_WEIGHTED_BUDGET"]["success"] else 0.0 for r in evaluable]
    oracle = [1.0 if r["policies"]["ORACLE_WEIGHTED_BUDGET"]["success"] else 0.0 for r in evaluable]
    equal_rate = sum(equal) / max(len(equal), 1)
    random_rate = sum(random_weighted) / max(len(random_weighted), 1)
    bg_rate = sum(bg) / max(len(bg), 1)
    oracle_rate = sum(oracle) / max(len(oracle), 1)
    bg_tokens = sum(r["policies"]["BG_WEIGHTED_BUDGET"]["tokens"] for r in evaluable)
    eq_tokens = sum(r["policies"]["EQUAL_BUDGET"]["tokens"] for r in evaluable)
    bg_spt = sum(bg) / max(bg_tokens, 1)
    eq_spt = sum(equal) / max(eq_tokens, 1)
    relative_spt = (bg_spt / eq_spt - 1.0) if eq_spt > 0 else 0.0
    delta = bg_rate - equal_rate
    verdict = "HELPS" if (len(evaluable) >= 20 and (delta >= 0.05 or relative_spt >= 0.10)) else verdict_from_delta(delta, len(evaluable), min_n=20)
    payload = {
        "BG_COMPUTE_ALLOCATION_VERDICT": verdict,
        "verdict": verdict,
        "metrics": {
            "evaluable_tasks": len(evaluable),
            "equal_budget_success": equal_rate,
            "random_weighted_success": random_rate,
            "bg_weighted_success": bg_rate,
            "oracle_weighted_success": oracle_rate,
            "bg_lift_vs_equal": delta,
            "bg_success_per_token": bg_spt,
            "equal_success_per_token": eq_spt,
            "relative_success_per_token_lift": relative_spt,
            "oracle_gap": oracle_rate - bg_rate,
        },
        "task_results": task_results,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_json(OUT_PARTIAL, payload)
    write_md(OUT_MD, ["# BG Compute Allocation Results (2026-05-18)", "", f"BG_COMPUTE_ALLOCATION_VERDICT = {verdict}", "", f"- metrics: `{payload['metrics']}`"])
    print(f"BG_COMPUTE_ALLOCATION_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
