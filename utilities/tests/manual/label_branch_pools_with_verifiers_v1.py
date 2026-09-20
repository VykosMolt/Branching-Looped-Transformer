"""PART F — external verifier labeling audit.

Branch pools are labeled with EXTERNAL verifiers only (dataset keys / answer parsers / z3 /
executed unit tests). Cheap pools carry dataset labels; generated branches were labeled at
generation by label_branch. This stage merges cheap + generated pools, (re)computes oracle
fields, and audits label coverage/quality. Re-runnable as generation adds shards.
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

OUT = B.OUT_ROOT


def _recompute_oracle(g: dict) -> dict:
    rewards = [b["objective_reward"] for b in g["branch_attempts"]]
    has_pos = any(r > 0 for r in rewards)
    mx = max(rewards) if rewards else 0.0
    g["external_oracle"] = {
        "has_positive_oracle": has_pos,
        "best_branch_ids": [b["branch_id"] for b in g["branch_attempts"] if b["objective_reward"] >= mx and mx > 0],
        "reward_diverse": len(set(r > 0 for r in rewards)) > 1,
        "all_wrong": not has_pos, "all_correct": bool(rewards) and all(r > 0 for r in rewards)}
    return g


def main() -> int:
    started = time.time(); B.ensure_dirs()
    cheap = B.read_jsonl(B.PROC / "branch_pools_raw.jsonl")
    gen = B.load_generated_groups()
    for g in gen:
        g.setdefault("provenance", {})["source_kind"] = "model_generated"
    for g in cheap:
        g.setdefault("provenance", {})["source_kind"] = "cheap"
    groups = [_recompute_oracle(g) for g in (cheap + gen)]
    B.write_jsonl(B.PROC / "branch_pools_labeled.jsonl", groups)
    flat = []
    for g in groups:
        o = g["external_oracle"]
        flat.append({"group_id": g["group_id"], "domain": g["domain"], "category": g["provenance"].get("category"),
                     "split": g["split"], "source_kind": g["provenance"].get("source_kind"),
                     "n_branches": len(g["branch_attempts"]), "has_positive_oracle": o["has_positive_oracle"],
                     "reward_diverse": o["reward_diverse"], "all_wrong": o["all_wrong"]})
    try:
        import pandas as pd
        pd.DataFrame(flat).to_parquet(B.PROC / "branch_pools_labeled.parquet", index=False)
    except Exception:
        pass

    # audit
    def slice_metrics(gs):
        n = len(gs)
        if not n:
            return {}
        nb = sum(len(g["branch_attempts"]) for g in gs)
        parse_ok = sum(1 for g in gs for b in g["branch_attempts"] if b["parse_ok"])
        unknown = sum(1 for g in gs for b in g["branch_attempts"] if b["external_label"] == "unknown")
        return {"groups": n,
                "positive_oracle_rate": round(sum(1 for g in gs if g["external_oracle"]["has_positive_oracle"]) / n, 4),
                "reward_diverse_rate": round(sum(1 for g in gs if g["external_oracle"]["reward_diverse"]) / n, 4),
                "all_wrong_rate": round(sum(1 for g in gs if g["external_oracle"]["all_wrong"]) / n, 4),
                "all_correct_rate": round(sum(1 for g in gs if g["external_oracle"]["all_correct"]) / n, 4),
                "parse_ok_rate": round(parse_ok / nb, 4) if nb else 0,
                "ambiguous_rate": round(unknown / nb, 4) if nb else 0}
    by_domain = {d: slice_metrics([g for g in groups if g["domain"] == d]) for d in sorted({g["domain"] for g in groups})}
    by_logic_family = {c: slice_metrics([g for g in groups if g["provenance"].get("category") == c and g["domain"] == "logic"])
                       for c in sorted({g["provenance"].get("category") for g in groups if g["domain"] == "logic"})}
    by_source = {s: slice_metrics([g for g in groups if g["provenance"].get("source_kind") == s])
                 for s in ("cheap", "model_generated")}
    overall = slice_metrics(groups)

    logic_cov = by_domain.get("logic", {}).get("groups", 0)
    gen_cov = by_source.get("model_generated", {}).get("groups", 0)
    core_ok = all(by_domain.get(d, {}).get("groups", 0) > 0 for d in ("coding", "math", "reasoning", "logic"))
    if core_ok and logic_cov >= 10000:
        verdict = "LOGIC_LABELS_READY" if by_domain.get("logic", {}).get("all_wrong_rate", 1) < 0.5 else "CORE_LABELS_READY"
    elif core_ok:
        verdict = "CORE_LABELS_READY"
    else:
        verdict = "PARSER_LIMITED"
    audit = {"EXTERNAL_VERIFIER_LABELING_VERDICT": verdict, "overall": overall, "by_domain": by_domain,
             "by_logic_family": by_logic_family, "by_source": by_source, "generated_groups": gen_cov,
             "total_groups": len(groups), "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(B.PROC / "verifier_audit.json", audit)
    B.write_json(OUT / "verifier_labeling.json", audit)
    B.write_md(OUT / "verifier_labeling.md", [
        "# External Verifier Labeling (Part F)", "", B.status_line("EXTERNAL_VERIFIER_LABELING_VERDICT", verdict), "",
        "All labels are external (dataset keys / parsers / z3 / executed unit tests). DualAnchor/CoreContent "
        "are NOT used for correctness here.", "",
        f"Overall: {overall}", "", "## By domain", "",
        *B.md_table([{"domain": d, **m} for d, m in by_domain.items()],
                    ["domain", "groups", "positive_oracle_rate", "reward_diverse_rate", "all_wrong_rate", "parse_ok_rate"]),
        "", "## By logic family", "",
        *B.md_table([{"family": c, **m} for c, m in by_logic_family.items()],
                    ["family", "groups", "positive_oracle_rate", "reward_diverse_rate", "all_wrong_rate"]),
        "", f"Generated-pool coverage so far: {gen_cov} groups ({by_source.get('model_generated', {})}).",
    ])
    B.prog("F_verifier_labeling", {"verdict": verdict, "total_groups": len(groups), "generated_groups": gen_cov})
    print(B.status_line("EXTERNAL_VERIFIER_LABELING_VERDICT", verdict))
    print(f"  total={len(groups)} generated={gen_cov} overall={overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
