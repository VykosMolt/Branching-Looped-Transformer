"""PART A — run-state ledger for branch_training_logic_expansion_terminal_v1.

Locks the current selected architecture state and verifies inputs before the run that moves
from external branch selection (DualAnchor + CoreContent_v2) toward model-internal branching.
No training here; no overwrites of existing taps/checkpoints/registries; science diagnostic-only.
"""
from __future__ import annotations
import importlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bg_corecontent_v2_common as v2  # reuse io + roots + dataset infra  # noqa: E402

PROJECT_ROOT = v2.PROJECT_ROOT
SHORT_NAME = "branch_training_logic_expansion_terminal_v1"
OUT_ROOT = v2.PROBE_ROOT / "branch_training_logic_expansion_terminal_v1_2026-06-06"
PROGRESS = OUT_ROOT / "progress"
DATA_ROOT = PROJECT_ROOT / "shared/data/branch_training_logic_expansion_v1"
MODEL_ROOT = PROJECT_ROOT / "opi/taps/models/branch_training_logic_expansion_v1"

# locked selected state carried in from corecontent v2 + dualanchor audit
LOCKED_STATE = {
    "branch_survival": {"policy": "DualAnchor", "anchors": ["MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL"],
                        "status": "unchanged (teacher + external baseline)"},
    "content_final_selection": {"policy": "CoreContent_v2_blockwise_pruned_24_36",
                                "prev_baseline": "mixedhead_MIX_HH_OBJECTIVE (beaten on expanded heldout)",
                                "role": "terminal-ranking baseline + optional soft teacher; NOT ground truth"},
    "terminal": "top5/full survivor-set handoff (locked); content selector ranks within survivors; no unconditional top1",
    "science": "diagnostic only (never a promotion gate)",
    "steering": "not run, not claimed",
    "priority_domain": "logic (largest expansion; weakest absolute core domain in v2)",
    "training": "allowed ONLY under opi/taps/models/branch_training_logic_expansion_v1/ (new adapters/LoRA/SFT/RL)",
    "label_policy": "external verifiers ONLY for correctness; DualAnchor/CoreContent are policy/soft teachers, never correctness labels",
}

DO_NOT_OVERWRITE = [
    "constructed_taps/pure_content_taps.pt", "constructed_taps/transplanted_taps.pt",
    "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/",
    "shared/data/corecontent_v2/", "base Ouro checkpoint", "tokenizer", "any tap registry",
]


def _exists(p: Path) -> bool:
    return Path(p).exists()


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    # input checks
    v2_root = v2.OUT_ROOT
    v2_policy = v2_root / "corecontent_v2_policy.pt"
    pure = v2.PURE_TAPS_PT
    model_dir = PROJECT_ROOT / "shared/models/ouro_rltt_local"
    dualanchor_common = _HERE / "bg_dualanchor_arch_loop_v3_common.py"
    cc_audit = v2.PRIOR_AUDIT_ROOT
    libs = {}
    for m in ("peft", "trl", "z3", "sympy", "nltk", "bitsandbytes", "torch", "transformers", "accelerate"):
        try:
            libs[m] = getattr(importlib.import_module(m), "__version__", "ok")
        except Exception:
            libs[m] = None
    inputs = {
        "ouro_model": _exists(model_dir),
        "corecontent_v2_policy_pt": _exists(v2_policy),
        "corecontent_v2_features": _exists(PROJECT_ROOT / "shared/data/corecontent_v2/features/feature_manifest.json"),
        "pure_content_taps": _exists(pure),
        "dualanchor_arch_loop_common": _exists(dualanchor_common),
        "core_tap_audit_root": _exists(cc_audit),
    }
    missing_inputs = [k for k, ok in inputs.items() if not ok]
    train_libs_ok = all(libs.get(m) for m in ("peft", "trl"))
    solver_ok = bool(libs.get("z3"))
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False

    if missing_inputs or not inputs["ouro_model"]:
        verdict = "MISSING_INPUTS" if not inputs["ouro_model"] else "MISSING_INPUTS"
    elif not gpu:
        verdict = "BLOCKED"
    else:
        verdict = "READY"

    manifest = {
        "short_name": SHORT_NAME, "created_at": time.time(),
        "output_root": str(OUT_ROOT.relative_to(PROJECT_ROOT)),
        "data_root": str(DATA_ROOT.relative_to(PROJECT_ROOT)),
        "model_root": str(MODEL_ROOT.relative_to(PROJECT_ROOT)),
        "locked_state": LOCKED_STATE, "do_not_overwrite": DO_NOT_OVERWRITE,
        "core_domains": list(v2.CORE_DOMAINS), "priority_domain": "logic",
        "logic_targets": {"min": {"train_groups": 20000, "branch_attempts": 100000, "heldout_groups": 5000},
                          "preferred": {"train_groups": 50000, "branch_attempts": 250000, "heldout_groups": 10000}},
        "stages": ["A_init", "B_pull", "C_logic_canonicalization", "D_schema", "E_branch_pools",
                   "F_verifier_labeling", "G_dedup", "H_integrated_terminal", "I_reachability",
                   "J_teacher_traces", "K_branch_training_dataset", "L_branching_sft",
                   "M_teacher_distillation", "N_verifier_rewarded", "O_eval", "P_ablations",
                   "Q_policy_decision", "R_docs"],
    }
    (DATA_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=v2.json_default) + "\n")

    payload = {"BRANCH_TRAINING_LOGIC_EXPANSION_INIT_VERDICT": verdict, "locked_state": LOCKED_STATE,
               "inputs_present": inputs, "missing_inputs": missing_inputs, "libraries": libs,
               "training_libs_ok": train_libs_ok, "solver_ok": solver_ok, "gpu": gpu,
               "do_not_overwrite": DO_NOT_OVERWRITE, "elapsed_seconds": round(time.time() - started, 3)}
    v2.write_json(OUT_ROOT / "run_state.json", payload)
    v2.write_md(OUT_ROOT / "run_state.md", [
        "# Branch Training + Logic Expansion + Terminal v1 — Run State", "",
        v2.status_line("BRANCH_TRAINING_LOGIC_EXPANSION_INIT_VERDICT", verdict), "",
        "Goal: move from external branch selection (DualAnchor + CoreContent_v2) toward **model-internal "
        "branching**, with **logic** as the priority expansion. DualAnchor is a *teacher*, not a permanent crutch; "
        "correctness comes ONLY from external verifiers.", "",
        "## Locked selected state", "",
        *[f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}" for k, v in LOCKED_STATE.items()], "",
        "## Inputs present", "",
        *[f"- {k}: {ok}" for k, ok in inputs.items()], "",
        f"Training libs (peft/trl): {train_libs_ok}; z3 solver: {solver_ok}; GPU: {gpu}.", "",
        "## Do not overwrite", "", *[f"- `{p}`" for p in DO_NOT_OVERWRITE],
    ])
    (PROGRESS / "A_init.json").write_text(json.dumps({"verdict": verdict, "saved_at": time.time()}) + "\n")
    print(v2.status_line("BRANCH_TRAINING_LOGIC_EXPANSION_INIT_VERDICT", verdict))
    print(f"  inputs missing: {missing_inputs or 'none'} | train_libs={train_libs_ok} z3={solver_ok} gpu={gpu}")
    print(f"  libs: " + ", ".join(f"{k}={v}" for k, v in libs.items() if v))
    return 1 if verdict == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
