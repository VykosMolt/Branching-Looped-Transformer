"""PART G — deduplication, leakage guard, and split policy for branch pools.

Task-disjoint splits are already assigned deterministically per task at canonicalization. This
stage removes exact duplicate groups, detects/repairs cross-split prompt leakage (reassign to
first-seen split), and writes the split manifest. Official benchmark test rows already routed to
heldout in Part C.
"""
from __future__ import annotations
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402


def _prompt_key(g: dict) -> str:
    p = g.get("task_prompt") or ""
    if not p and g.get("branch_attempts"):
        # core groups: use the shared prefix of the first branch text
        p = str(g["branch_attempts"][0].get("branch_text", ""))[:400]
    return f"{g['domain']}::{B.v2.text_hash(p)}"


def main() -> int:
    started = time.time(); B.ensure_dirs()
    groups = B.read_jsonl(B.PROC / "branch_pools_labeled.jsonl")
    if not groups:
        groups = B.read_jsonl(B.PROC / "branch_pools_raw.jsonl")
    seen_group: set = set()
    seen_prompt: dict[str, str] = {}
    kept = []
    n_dup = n_leak = 0
    for g in groups:
        gkey = f"{g['domain']}::{B.v2.text_hash(str(g.get('task_id')))}"
        if gkey in seen_group:
            n_dup += 1
            continue
        seen_group.add(gkey)
        pk = _prompt_key(g)
        if pk in seen_prompt:
            if g["split"] != seen_prompt[pk]:
                g["split"] = seen_prompt[pk]
                n_leak += 1
        else:
            seen_prompt[pk] = g["split"]
        kept.append(g)
    B.write_jsonl(B.PROC / "branch_pools_deduped.jsonl", kept)
    flat = [{"group_id": g["group_id"], "domain": g["domain"], "split": g["split"],
             "category": g.get("provenance", {}).get("category"), "n_branches": len(g["branch_attempts"])}
            for g in kept]
    try:
        import pandas as pd
        pd.DataFrame(flat).to_parquet(B.PROC / "branch_pools_deduped.parquet", index=False)
    except Exception:
        pass
    # residual leakage check
    split_of: dict[str, set] = defaultdict(set)
    for g in kept:
        split_of[_prompt_key(g)].add(g["split"])
    residual = sum(1 for s in split_of.values() if len(s) > 1)
    by_split = Counter(g["split"] for g in kept)
    by_dom_split = defaultdict(Counter)
    for g in kept:
        by_dom_split[g["domain"]][g["split"]] += 1
    split_manifest = {"by_split": dict(by_split),
                      "by_domain_split": {d: dict(c) for d, c in by_dom_split.items()},
                      "kept_groups": len(kept), "policy": "task-disjoint; official test->heldout; first-seen split wins"}
    B.write_json(B.PROC / "split_manifest.json", split_manifest)
    if residual > 0:
        verdict = "LEAKAGE_RISK_REMAINS"
    elif n_leak > 0:
        verdict = "LEAKAGE_FOUND_FIXED"
    elif n_dup > 0:
        verdict = "DUPLICATES_REMOVED"
    else:
        verdict = "READY"
    payload = {"DEDUP_LEAKAGE_VERDICT": verdict, "input_groups": len(groups), "kept_groups": len(kept),
               "duplicates_removed": n_dup, "leakage_reassigned": n_leak, "residual_cross_split_prompts": residual,
               "by_split": dict(by_split), "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(B.OUT_ROOT / "dedup_leakage.json", payload)
    B.write_md(B.OUT_ROOT / "dedup_leakage.md", [
        "# Branch-Pool Dedup / Leakage (Part G)", "", B.status_line("DEDUP_LEAKAGE_VERDICT", verdict), "",
        f"Input {len(groups)} -> kept {len(kept)}. Duplicates removed: {n_dup}. Leakage reassigned: {n_leak}. "
        f"Residual cross-split prompts: {residual}.", "",
        f"Split balance: {dict(by_split)}.", "",
        "## By domain x split", "",
        *B.md_table([{"domain": d, **{s: c.get(s, 0) for s in ("train", "val", "heldout", "diagnostic")}}
                     for d, c in by_dom_split.items()], ["domain", "train", "val", "heldout", "diagnostic"]),
    ])
    B.prog("G_dedup", {"verdict": verdict, "kept_groups": len(kept), "by_split": dict(by_split)})
    print(B.status_line("DEDUP_LEAKAGE_VERDICT", verdict))
    print(f"  kept={len(kept)} dups={n_dup} leak_fixed={n_leak} residual={residual} splits={dict(by_split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
