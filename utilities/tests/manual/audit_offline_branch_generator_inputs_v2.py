"""PART B — input audit for branch_training_offline_verifier_generator_v2.

Verifies the v1 substrate: counts, schema, splits, uid uniqueness, prompt-hash leakage,
external-label/reward/parse availability, teacher_is_ground_truth=false, training views,
logic-family coverage, DPO validity. Read-only; writes input_audit.{md,json}.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import branch_training_v2_common as V  # noqa: E402

D = V.DATA_ROOT


def _pq(rel):
    return pd.read_parquet(D / rel)


def _first_jsonl(rel, n=200):
    rows = []
    p = D / rel
    if not p.exists():
        return rows
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main() -> int:
    rep: dict = {"warnings": [], "errors": [], "checks": {}}
    C = rep["checks"]

    # 1. logic tasks
    lt = _pq("processed/logic_tasks.parquet")
    splits = lt["split"].value_counts().to_dict()
    fams = sorted(lt["category"].unique().tolist())
    # prompt-hash leakage across splits
    lt = lt.copy()
    lt["_ph"] = lt["task_prompt"].map(lambda s: V.v2.text_hash(str(s)))
    ph_by_split = {s: set(g["_ph"]) for s, g in lt.groupby("split")}
    leak_pairs = {}
    sl = list(ph_by_split)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            ov = len(ph_by_split[sl[i]] & ph_by_split[sl[j]])
            if ov:
                leak_pairs[f"{sl[i]}∩{sl[j]}"] = ov
    C["logic_tasks"] = {"rows": int(len(lt)), "splits": {k: int(v) for k, v in splits.items()},
                        "families": fams, "n_families": len(fams), "uid_unique": bool(lt["task_uid"].is_unique),
                        "prompt_hash_leakage": leak_pairs, "gold_present": bool(lt["gold_answer"].notna().any()),
                        "verifier_present": bool(lt["verifier"].notna().any())}
    if leak_pairs:
        rep["warnings"].append(f"logic prompt-hash overlap across splits: {leak_pairs}")
    if len(fams) < 10:
        rep["warnings"].append(f"only {len(fams)} logic families (<10)")

    # 2. labeled branch pools (parquet index + jsonl branch-level fields)
    bl = _pq("processed/branch_pools_labeled.parquet")
    pos_rate = float(bl["has_positive_oracle"].mean()) if "has_positive_oracle" in bl else None
    blj = _first_jsonl("processed/branch_pools_labeled.jsonl", 50)
    branch_fields = []
    label_src_ok = ext_label_ok = reward_ok = parse_ok = False
    if blj:
        g0 = blj[0]
        battempts = g0.get("branch_attempts") or g0.get("branches") or []
        if battempts:
            branch_fields = sorted(battempts[0].keys())
            ext_label_ok = any("external_label" in b for b in battempts)
            reward_ok = any(("reward" in b) or ("external_reward" in b) for b in battempts)
            parse_ok = any(("parse_ok" in b) or ("final_answer" in b) for b in battempts)
        # label source must be external (verifier/answer_key/unit_tests/...) not tap
        srcs = set()
        for b in battempts:
            srcs.add(str(b.get("label_source") or g0.get("label_type") or ""))
        label_src_ok = not any("tap" in s.lower() or "dualanchor" in s.lower() or "corecontent" in s.lower() for s in srcs)
    C["branch_pools_labeled"] = {"groups": int(len(bl)), "positive_oracle_rate": round(pos_rate, 4) if pos_rate is not None else None,
                                 "branch_fields": branch_fields, "external_label_present": ext_label_ok,
                                 "reward_present": reward_ok, "parse_or_final_present": parse_ok,
                                 "label_source_external_only": label_src_ok,
                                 "by_domain": {k: int(v) for k, v in bl["domain"].value_counts().items()}}
    if not ext_label_ok:
        rep["warnings"].append("branch attempts: external_label field not found in sampled labeled jsonl")

    # 3. deduped + group-split leakage
    bd = _pq("processed/branch_pools_deduped.parquet")
    gsplit = bd.groupby("group_id")["split"].nunique()
    multi = int((gsplit > 1).sum()) if len(gsplit) else 0
    C["branch_pools_deduped"] = {"groups": int(len(bd)), "group_ids_in_multiple_splits": multi,
                                 "splits": {k: int(v) for k, v in bd["split"].value_counts().items()}}
    if multi:
        rep["errors"].append(f"{multi} dedup group_ids appear in >1 split (leakage)")

    # 4. terminal survivor sets
    ts = _pq("terminal_survivor_sets.parquet")
    C["terminal_survivor_sets"] = {"rows": int(len(ts)),
                                   "survivor_oracle_retention": round(float(ts["survivor_has_oracle"].mean()), 4) if "survivor_has_oracle" in ts else None}

    # 5. teacher traces (policy only, not ground truth)
    tt = _pq("teacher/dualanchor_teacher_traces.parquet")
    teacher_cols = list(tt.columns)
    has_correctness_cols = any(c in teacher_cols for c in ("reward", "correct", "gold_answer", "final_reward"))
    schema = V.v2.read_json(D / "schema" / "teacher_trace_schema.json", {}) or {}
    gt_false = (schema.get("properties", {}).get("teacher_is_ground_truth", {}).get("const") is False) or \
               ("teacher_is_ground_truth" in json.dumps(schema) and "false" in json.dumps(schema).lower())
    C["teacher_traces"] = {"rows": int(len(tt)), "policy_cols": teacher_cols,
                           "has_correctness_cols": has_correctness_cols, "schema_teacher_is_ground_truth_false": bool(gt_false)}
    if has_correctness_cols:
        rep["errors"].append("teacher traces contain correctness-like columns (violates ground-truth rule)")

    # 6. training views
    views = ["branch_format_sft", "branch_diversity_sft", "branch_policy_distillation",
             "final_self_selection", "verifier_reward_rl", "branch_preference_dpo"]
    vinfo = {}
    for v in views:
        p = D / "train" / f"{v}.jsonl"
        n = sum(1 for _ in open(p)) if p.exists() else 0
        sample = _first_jsonl(f"train/{v}.jsonl", 1)
        vinfo[v] = {"rows": n, "keys": sorted(sample[0].keys()) if sample else []}
    # DPO validity
    dpo = _first_jsonl("train/branch_preference_dpo.jsonl", 500)
    dpo_valid = sum(1 for r in dpo if r.get("chosen") and r.get("rejected") and r.get("chosen") != r.get("rejected"))
    vinfo["branch_preference_dpo"]["valid_pairs_in_sample"] = f"{dpo_valid}/{len(dpo)}"
    C["training_views"] = vinfo
    missing_views = [v for v in views if vinfo[v]["rows"] == 0]
    if missing_views:
        rep["warnings"].append(f"training views with 0 rows: {missing_views}")

    # 7. gen shards
    gm = V.v2.read_json(D / "processed/gen_shards/gen_manifest.json", {}) or {}
    gen_uids = gm.get("completed_task_uids", [])
    C["gen_shards"] = {"completed_task_uids": len(gen_uids), "shards": len(gm.get("shards", []))}

    # verdict
    if rep["errors"]:
        verdict = "LEAKAGE_RISK" if any("leak" in e or "split" in e for e in rep["errors"]) else "BLOCKED"
    elif vinfo and not missing_views and ext_label_ok and not has_correctness_cols:
        verdict = "INPUTS_READY_WITH_WARNINGS" if rep["warnings"] else "INPUTS_READY"
    elif missing_views:
        verdict = "TRAINING_VIEWS_INCOMPLETE"
    elif not ext_label_ok:
        verdict = "VERIFIER_LABELS_MISSING"
    else:
        verdict = "INPUTS_READY_WITH_WARNINGS"
    logic_ok = len(fams) >= 10 and splits.get("train", 0) >= 30000 and splits.get("heldout", 0) >= 8000 and not leak_pairs

    payload = {"INPUT_AUDIT_VERDICT": verdict, "logic_substrate_ready": bool(logic_ok), **rep}
    V.write_json(V.OUT_ROOT / "input_audit.json", payload)
    V.write_md(V.OUT_ROOT / "input_audit.md", [
        "# Input Audit (Part B)", "", V.status_line("INPUT_AUDIT_VERDICT", verdict),
        f"logic_substrate_ready = {logic_ok}", "",
        "## Logic tasks",
        f"- {C['logic_tasks']['rows']} tasks; splits {C['logic_tasks']['splits']}; {C['logic_tasks']['n_families']} families; "
        f"uid_unique={C['logic_tasks']['uid_unique']}; prompt-hash leakage {C['logic_tasks']['prompt_hash_leakage'] or 'none'}.",
        f"- families: {', '.join(fams)}",
        "## Branch pools",
        f"- labeled {C['branch_pools_labeled']['groups']} groups (pos-oracle {C['branch_pools_labeled']['positive_oracle_rate']}); "
        f"external_label={C['branch_pools_labeled']['external_label_present']}, label_source_external_only={C['branch_pools_labeled']['label_source_external_only']}.",
        f"- deduped {C['branch_pools_deduped']['groups']} groups; multi-split group_ids {C['branch_pools_deduped']['group_ids_in_multiple_splits']}.",
        f"- terminal survivor sets {C['terminal_survivor_sets']['rows']} (oracle retention {C['terminal_survivor_sets']['survivor_oracle_retention']}).",
        "## Teacher traces (policy, not ground truth)",
        f"- {C['teacher_traces']['rows']} rows; cols {C['teacher_traces']['policy_cols']}; "
        f"has_correctness_cols={C['teacher_traces']['has_correctness_cols']}; schema gt-false={C['teacher_traces']['schema_teacher_is_ground_truth_false']}.",
        "## Training views", *[f"- {v}: {info['rows']} rows; keys {info['keys']}" for v, info in vinfo.items()],
        "## Warnings / errors",
        *( [f"- WARN: {w}" for w in rep["warnings"]] + [f"- ERROR: {e}" for e in rep["errors"]] or ["- none"]),
    ])
    V.set_stage("B_input_audit", verdict, {"logic_substrate_ready": bool(logic_ok)})
    V.prog("B_input_audit", {"verdict": verdict})
    print(V.status_line("INPUT_AUDIT_VERDICT", verdict))
    print(f"  logic {C['logic_tasks']['rows']} ({C['logic_tasks']['splits']}) {C['logic_tasks']['n_families']} fam | "
          f"labeled {C['branch_pools_labeled']['groups']}g | teacher {C['teacher_traces']['rows']} | "
          f"views_ok={not missing_views} | warnings={len(rep['warnings'])} errors={len(rep['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
