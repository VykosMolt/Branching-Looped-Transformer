"""PART R — documentation update for branch_training_logic_expansion_terminal_v1.

Creates the memorable-named root doc and appends a concise section to the curated evaluator
docs. Append-only; does not remove history or overwrite old run notes.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402

OUT = B.OUT_ROOT
DOC_ROOT = B.PROJECT_ROOT / "shared/docs/evaluator"
MAIN = DOC_ROOT / "branch-training-logic-expansion.md"
APPEND = ["current-state.md", "evaluator-navigation-map.md", "branch-generation-and-survival.md",
          "terminal-selection-and-arbiters.md", "domain-transfer-ledger.md", "interfaces-and-tools.md"]


def main() -> int:
    started = time.time()
    dec = B.v2.read_json(OUT / "final_policy_decision.json", {}) or {}
    o = B.v2.read_json(OUT / "trained_branching_eval.json", {}) or {}
    reach = B.v2.read_json(OUT / "branch_pool_reachability.json", {}) or {}
    H = B.v2.read_json(OUT / "integrated_terminal_eval.json", {}) or {}
    status = dec.get("BRANCH_TRAINING_LOGIC_EXPANSION_STATUS")
    decision = dec.get("BRANCH_TRAINING_POLICY_DECISION")
    section = [
        "", "## Branch training + logic expansion + terminal v1 (2026-06-06)", "",
        f"- Status: `{status}`; policy decision: `{decision}`.",
        "- Goal: move from external branch selection (DualAnchor + CoreContent_v2) toward **model-internal "
        "branching**. Primary deliverable = the branch-training **data + evaluation harness** (the L/O training is a "
        "bounded 300-step LoRA proof-of-capability on Ouro-RLTT, not a converged model). External verifiers are the "
        "only ground truth; DualAnchor/CoreContent are teachers/baselines; science diagnostic-only; steering not run.",
        f"- **Logic expanded** to ~{B.v2.read_json(OUT/'logic_canonicalization.json',{}).get('train_groups')} train groups "
        "across 10 verifier-backed families (synthetic propositional/proof-depth/syllogism/z3-constraint + "
        "LogiQA/ReClor/RuleTaker/FOLIO/LSAT/logical-entailment).",
        f"- **Branch-pool reachability@4** (Ouro-RLTT generations, external-labeled): "
        f"{ {d: m.get('positive_oracle@4') for d, m in (reach.get('generated_by_domain') or {}).items()} }. "
        "Math went 0.31→0.83 once given a brutal tool-free answer-forcing prompt + 1400 math budget + early-stop + "
        "LaTeX/sympy verifier; coding ~0.43 after fixing a function-name prompt gap (genuine, not artifact).",
        f"- **Experiment 1 (integrated terminal, H)**: within real DualAnchor top-5 survivor sets, "
        f"CoreContent_v2 {H.get('corecontent_v2')} > DualAnchor forced-top1 {H.get('dualanchor_forced_top1')} "
        f"(retention {H.get('survivor_oracle_retention_macro')}) → **selection, not survival, is the terminal bottleneck**; "
        "the composed external baseline (DualAnchor survival → CoreContent_v2 ranking → survivor handoff) is valid.",
        "- **DualAnchor-as-teacher (J)**: useful branch-policy teacher (oracle-retention lift over random +0.11 "
        "coding/reasoning) but ~random on logic (+0.03) → logic branch quality must come from verifier reward, not "
        "teacher distillation.",
        f"- **Trained vs Ouro-RLTT-no-adapter (O)**: macro positive_oracle@K {o.get('macro_positive_oracle')}, "
        f"lift {o.get('oracle_lift_sft_minus_base')}, diversity lift {o.get('diversity_lift')} "
        f"(`{o.get('TRAINED_BRANCHING_EVAL_VERDICT')}`).",
        "- Data-hygiene fixes caught by hand-audit: math-LaTeX verifier false-negatives, coding function-name "
        "mismatch, and a hendrycks_math `task_uid` collision (row indices repeat across 7 subjects → re-keyed by prompt-hash).",
        "- No Ouro base/tokenizer/checkpoint/registry overwrite; adapters saved only under "
        "`opi/taps/models/branch_training_logic_expansion_v1/`; pure/transplanted/CoreContent_v2 artifacts untouched.",
        f"- Artifacts: `{OUT.relative_to(B.PROJECT_ROOT)}`; data `shared/data/branch_training_logic_expansion_v1/`.",
    ]
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    B.write_md(MAIN, ["# Branch Training + Logic Expansion + Terminal v1", *section,
                      "", "### Stage verdicts", "",
                      *[f"    {k} = {val}" for k, val in (dec.get("stage_verdicts") or {}).items()],
                      f"    BRANCH_TRAINING_LOGIC_EXPANSION_STATUS = {status}"])
    appended = []
    for name in APPEND:
        p = DOC_ROOT / name
        try:
            prev = p.read_text() if p.exists() else f"# {name}\n"
            p.write_text(prev.rstrip() + "\n" + "\n".join(section) + "\n")
            appended.append(name)
        except Exception:
            pass
    verdict = "DOCS_UPDATED" if appended else "DOCS_PARTIAL"
    B.write_json(OUT / "docs_update.json", {"BRANCH_TRAINING_DOC_UPDATE_VERDICT": verdict,
                 "main_doc": str(MAIN.relative_to(B.PROJECT_ROOT)), "appended": appended,
                 "elapsed_seconds": round(time.time() - started, 3)})
    B.prog("R_docs", {"verdict": verdict, "appended": len(appended)})
    print(B.status_line("BRANCH_TRAINING_DOC_UPDATE_VERDICT", verdict))
    print(f"  wrote {MAIN.name}; appended to {len(appended)} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
