"""PART E — branch pool generation.

Two entry points (so model generation is resumable/pausable independent of the cheap build):
  default / "cheap" : build cheap candidate pools (dataset options / mutations / solver) for all
                      logic + core tasks, write branch_pools_raw.{jsonl,parquet} + report.
  "gen"             : resumable, pausable model branch generation (bounded subset, logic-priority);
                      re-run to make more progress; STOP sentinel or GEN_TIME_BUDGET pauses gracefully.
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402
import branch_training_logic_v1_common as L  # noqa: E402

OUT = B.OUT_ROOT
# how many cheap pools to build (logic-priority); generation subset is smaller/bounded
LOGIC_TRAIN_CAP = 12000
CORE_PER_DOMAIN = 3000


def build_pools_main() -> int:
    started = time.time(); B.ensure_dirs()
    logic = B.load_logic_tasks()
    # keep all heldout logic + a capped train/val sample (logic priority)
    held = [t for t in logic if t["split"] == "heldout"]
    other = [t for t in logic if t["split"] != "heldout"]
    other.sort(key=lambda t: L._sid("poolcap", t["task_uid"]))
    logic_sel = held + other[:LOGIC_TRAIN_CAP]
    core = B.load_core_tasks(max_per_domain=CORE_PER_DOMAIN)
    tasks = logic_sel + core
    groups = []
    for t in tasks:
        try:
            branches = B.cheap_branch_pool(t)
        except Exception:
            continue
        if len(branches) < 2:
            continue
        groups.append(B.make_group(t, branches))
    B.write_jsonl(B.PROC / "branch_pools_raw.jsonl", groups)
    flat = [{"group_id": g["group_id"], "split": g["split"], "domain": g["domain"], "dataset": g["dataset"],
             "category": g["provenance"]["category"], "n_branches": len(g["branch_attempts"]),
             "has_positive_oracle": g["external_oracle"]["has_positive_oracle"],
             "reward_diverse": g["external_oracle"]["reward_diverse"]} for g in groups]
    try:
        import pandas as pd
        pd.DataFrame(flat).to_parquet(B.PROC / "branch_pools_raw.parquet", index=False)
    except Exception:
        pass
    by_dom = Counter(g["domain"] for g in groups)
    by_split = Counter(g["split"] for g in groups)
    n_branches = sum(len(g["branch_attempts"]) for g in groups)
    rdiv = sum(1 for g in groups if g["external_oracle"]["reward_diverse"])
    logic_groups = sum(1 for g in groups if g["domain"] == "logic")
    avg = round(n_branches / len(groups), 2) if groups else 0
    verdict = "LOGIC_BRANCH_POOLS_READY" if logic_groups >= 10000 else ("BRANCH_POOLS_READY" if groups else "BLOCKED")
    payload = {"BRANCH_POOL_GENERATION_VERDICT": verdict, "groups": len(groups), "branches": n_branches,
               "avg_branches_per_group": avg, "reward_diverse_groups": rdiv, "logic_groups": logic_groups,
               "by_domain": dict(by_dom), "by_split": dict(by_split),
               "note": "cheap candidate pools (dataset options/mutations/solver); model-generated pools added by 'gen'.",
               "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(OUT / "branch_pool_generation.json", payload)
    B.write_md(OUT / "branch_pool_generation.md", [
        "# Branch Pool Generation (Part E)", "", B.status_line("BRANCH_POOL_GENERATION_VERDICT", verdict), "",
        f"Cheap candidate pools: {len(groups)} groups ({logic_groups} logic), {n_branches} branches, "
        f"avg {avg}/group, {rdiv} reward-diverse. Model-generated branch pools are produced by the "
        "resumable/pausable `gen` entry point (logic-priority, bounded).", "",
        "## By domain", "", *B.md_table([{"domain": k, "groups": v} for k, v in by_dom.most_common()], ["domain", "groups"]),
        "## By split", "", *B.md_table([{"split": k, "groups": v} for k, v in by_split.items()], ["split", "groups"]),
    ])
    B.prog("E_cheap_pools", {"verdict": verdict, "groups": len(groups), "logic_groups": logic_groups})
    print(B.status_line("BRANCH_POOL_GENERATION_VERDICT", verdict))
    print(f"  cheap pools: {len(groups)} groups ({logic_groups} logic), {n_branches} branches")
    return 0


def generate_main() -> int:
    """Resumable/pausable bounded model generation, DOMAIN-BALANCED (branching is general; not logic-weighted)."""
    B.ensure_dirs()
    n_gen = int(os.environ.get("GEN_N_TASKS", "1600"))
    k = int(os.environ.get("GEN_K", "4"))
    # No time-based auto-stop: runs until done or manual STOP sentinel. Override GEN_TIME_BUDGET only if wanted.
    budget = float(os.environ.get("GEN_TIME_BUDGET", "1e18"))
    # balanced across logic/math/reasoning/coding (all externally verifiable)
    subset = B.load_gen_tasks_balanced(n_total=n_gen)
    import collections
    print(f"  gen subset by domain: {dict(collections.Counter(t['domain'] for t in subset))}", flush=True)
    man = B.generate_branch_pools(subset, k=k, job="branch_gen", time_budget_s=budget)
    gen_groups = B.load_generated_groups()
    parse_ok = sum(1 for g in gen_groups for b in g["branch_attempts"] if b["parse_ok"])
    n_b = sum(len(g["branch_attempts"]) for g in gen_groups)
    pos_oracle = sum(1 for g in gen_groups if g["external_oracle"]["has_positive_oracle"])
    payload = {"generated_groups": len(gen_groups), "generated_branches": n_b,
               "parse_ok_rate": round(parse_ok / n_b, 4) if n_b else 0,
               "positive_oracle_groups": pos_oracle,
               "positive_oracle_rate": round(pos_oracle / len(gen_groups), 4) if gen_groups else 0,
               "paused": man.get("paused", False), "saved_at": time.time()}
    B.write_json(OUT / "branch_pool_generation_gen.json", payload)
    B.prog("E_generate_summary", payload)
    print(f"  generated groups={len(gen_groups)} parse_ok_rate={payload['parse_ok_rate']} "
          f"pos_oracle_rate={payload['positive_oracle_rate']} paused={payload['paused']}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cheap"
    raise SystemExit(generate_main() if mode == "gen" else build_pools_main())
