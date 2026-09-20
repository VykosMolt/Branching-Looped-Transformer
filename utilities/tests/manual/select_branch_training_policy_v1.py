"""PART Q — final branch-training policy decision + run synthesis.

Aggregates every stage verdict (A..O) and decides the next architecture state. Honest about
the bounded, proof-of-capability nature of the L/O training: the PRIMARY deliverable of this
run is the branch-training DATA + EVALUATION HARNESS; converged training (M/N + scale) is a
separate effort. External verifiers are the only ground truth; DualAnchor/CoreContent stay
teachers/baselines; science diagnostic-only; nothing overwrites base/tokenizer/registry.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402

OUT = B.OUT_ROOT


def J(name, key=None):
    d = B.v2.read_json(OUT / name, {}) or {}
    return (d.get(key) if key else d)


def main() -> int:
    started = time.time()
    v = {
        "INIT": J("run_state.json", "BRANCH_TRAINING_LOGIC_EXPANSION_INIT_VERDICT"),
        "DATA_PULL": J("source_ledger.json", "BRANCH_TRAINING_DATA_PULL_VERDICT"),
        "LOGIC_CANON": J("logic_canonicalization.json", "LOGIC_CANONICALIZATION_VERDICT"),
        "BRANCH_POOLS": J("branch_pool_generation.json", "BRANCH_POOL_GENERATION_VERDICT"),
        "VERIFIER_LABELING": J("verifier_labeling.json", "EXTERNAL_VERIFIER_LABELING_VERDICT"),
        "DEDUP": J("dedup_leakage.json", "DEDUP_LEAKAGE_VERDICT"),
        "INTEGRATED_TERMINAL": J("integrated_terminal_eval.json", "INTEGRATED_DUALANCHOR_CORECONTENT_TERMINAL_VERDICT"),
        "REACHABILITY": J("branch_pool_reachability.json", "BRANCH_POOL_REACHABILITY_VERDICT"),
        "TEACHER_TRACES": J("dualanchor_teacher_traces.json", "DUALANCHOR_TEACHER_TRACE_VERDICT"),
        "TRAINING_DATASET": J("branch_training_dataset.json", "BRANCH_TRAINING_DATASET_VERDICT"),
        "BRANCHING_SFT": J("branching_sft_training.json", "BRANCHING_SFT_VERDICT"),
        "TEACHER_DISTILL": J("dualanchor_teacher_distillation.json", "DUALANCHOR_TEACHER_DISTILLATION_VERDICT") or "NOT_RUN",
        "VERIFIER_REWARD": J("verifier_rewarded_training.json", "VERIFIER_REWARDED_BRANCHING_VERDICT") or "NOT_RUN",
        "TRAINED_EVAL": J("trained_branching_eval.json", "TRAINED_BRANCHING_EVAL_VERDICT"),
    }
    o = J("trained_branching_eval.json")
    lift = o.get("oracle_lift_sft_minus_base")
    macro = o.get("macro_positive_oracle", {})
    reach = J("branch_pool_reachability.json", "generated_by_domain") or {}
    logic_train = J("logic_canonicalization.json", "train_groups")

    # decision: bounded SFT vs external baseline
    if v["TRAINED_EVAL"] == "MODEL_INTERNAL_BRANCHING_IMPROVES" and (lift or 0) > 0.05:
        decision = "USE_BRANCH_TRAINED_MODEL_AS_GENERATOR"
    elif v["TRAINED_EVAL"] in ("MODEL_INTERNAL_BRANCHING_PARTIAL",) and (lift or 0) >= 0:
        decision = "NEEDS_BETTER_GENERATOR"
    elif v["TRAINED_EVAL"] in ("NO_GAIN",):
        decision = "KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE"
    else:
        decision = "KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE"

    # overall status: the data+harness is the headline deliverable
    data_ready = (v["LOGIC_CANON"] in ("LOGIC_VERIFIER_READY",) and (logic_train or 0) >= 20000
                  and v["TRAINING_DATASET"] == "TRAINING_DATA_READY")
    converged_training = v["TEACHER_DISTILL"] != "NOT_RUN" and v["VERIFIER_REWARD"] != "NOT_RUN"
    if decision == "USE_BRANCH_TRAINED_MODEL_AS_GENERATOR":
        status = "MODEL_INTERNAL_BRANCHING_READY"
    elif data_ready and not converged_training:
        status = "LOGIC_EXPANSION_READY_TRAINING_NOT_READY"
    elif decision == "NEEDS_BETTER_GENERATOR":
        status = "BRANCH_TRAINING_PARTIAL_READY"
    elif decision == "KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE":
        status = "EXTERNAL_BASELINE_STILL_BEST"
    else:
        status = "NOT_READY"

    locked = {
        "branch_survival": "DualAnchor (MIX_CODE_REASONING + MIX_OBJECTIVE_ALL) — unchanged (teacher + external baseline)",
        "content_final_selection": "CoreContent_v2_blockwise_pruned_24_36 — unchanged (validated terminal ranker; H confirmed it beats DualAnchor forced-top1 within survivors)",
        "terminal": "top5/full survivor-set handoff (selection, not survival, is the bottleneck per H)",
        "branch_trained_model": f"branching_sft LoRA adapter (bounded 300-step proof-of-capability); decision={decision}",
        "logic": f"expanded: {logic_train} train groups across 10 families, verifier-backed",
        "science": "diagnostic only", "steering": "not run, not claimed",
    }
    decision_doc = {
        "BRANCH_TRAINING_POLICY_DECISION": decision,
        "BRANCH_TRAINING_LOGIC_EXPANSION_STATUS": status,
        "stage_verdicts": v,
        "trained_eval": {"macro_positive_oracle": macro, "oracle_lift_sft_minus_base": lift,
                         "diversity_lift": o.get("diversity_lift")},
        "headline_findings": {
            "math_reachability_fix": "0.31 -> 0.83 via brutal tool-free prompt + 1400 math budget + early-stop + LaTeX verifier",
            "terminal_bottleneck": "selection not survival (H: CoreContent_v2 0.658 vs DualAnchor forced-top1 0.379, survivor oracle retention 1.0)",
            "teacher_useful_except_logic": "J: DualAnchor pruning beats random +0.11 coding/reasoning, only +0.03 logic",
            "failures_audited_genuine": "math/logic/reasoning/coding fails hand-verified after fixing math-LaTeX, coding-name, hendrycks-uid artifacts",
        },
        "locked_state": locked,
        "primary_deliverable": "branch-training DATA + EVALUATION HARNESS (logic-expanded, verifier-labeled, 5 training views) — READY" if data_ready else "data/harness incomplete",
        "next": ("scale + run M (teacher distillation) and N (verifier-reward RL) to convergence for a real internal-branching model"
                 if status in ("LOGIC_EXPANSION_READY_TRAINING_NOT_READY", "BRANCH_TRAINING_PARTIAL_READY") else "—"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    B.write_json(OUT / "final_policy_decision.json", decision_doc)
    B.write_md(OUT / "final_policy_decision.md", [
        "# Branch-Training Final Policy Decision (Part Q)", "",
        B.status_line("BRANCH_TRAINING_POLICY_DECISION", decision),
        B.status_line("BRANCH_TRAINING_LOGIC_EXPANSION_STATUS", status), "",
        "## Primary deliverable", "",
        f"- {decision_doc['primary_deliverable']}.",
        "- The L/O training is a **bounded, 300-step proof-of-capability** (bf16 LoRA on Ouro-RLTT), not a converged model.", "",
        "## Trained vs Ouro-RLTT (no adapter)", "",
        f"- macro positive_oracle@K: {macro}; lift (sft-base): {lift}; diversity lift: {o.get('diversity_lift')}.", "",
        "## Locked state", "", *[f"- **{k}**: {val}" for k, val in locked.items()], "",
        "## Stage verdicts", "", *[f"    {k} = {val}" for k, val in v.items()], "",
        "## Headline findings", "", *[f"- {k}: {val}" for k, val in decision_doc["headline_findings"].items()], "",
        f"## Next\n\n{decision_doc['next']}",
    ])
    B.prog("Q_policy_decision", {"decision": decision, "status": status})
    print(B.status_line("BRANCH_TRAINING_POLICY_DECISION", decision))
    print(B.status_line("BRANCH_TRAINING_LOGIC_EXPANSION_STATUS", status))
    print(f"  lift(sft-base)={lift} macro={macro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
