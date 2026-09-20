"""PART I — controlled offline-first training matrix.

Defines arms A..I (data mix, losses, init, domain-gated teacher weights, bf16 LoRA). Honest about
the single-12GB-GPU budget: all configs are written, but only a runnable subset is marked ON
(MATRIX_REDUCED) — the spec's preferred candidates C/G/H + the B control + A control.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402

TEACHER_WEIGHTS = {"coding": 0.7, "reasoning": 0.5, "math": 0.25, "logic": 0.0, "alignment": 0.3}
COMMON = {"precision": "bf16", "lora": True, "lora_target": "all-linear", "lora_r": 16, "lora_alpha": 32,
          "grad_checkpointing": True, "base_untouched": True, "tokenizer_untouched": True,
          "save_steps": 100, "eval_canary_every": 200, "stop_pausable": True, "no_heldout_training": True,
          "no_online_generation": True}

ARMS = [
    ("A_format_control", {"init": "base", "data": ["branch_format_sft", "branch_diversity_sft"],
        "losses": ["sft"], "teacher": False, "purpose": "reproduce v1 branch-format SFT (control)", "on": True,
        "target_steps": 300}),
    ("B_direct_budget_sft", {"init": "base", "data": ["direct_answer_sft", "one_branch_sft", "multi_branch_sft",
        "branch_budget_policy"], "losses": ["sft", "budget_cls"], "teacher": False,
        "purpose": "fix overbranching via direct-answer + budget", "on": True, "target_steps": 400}),
    ("C_offline_verifier_dpo", {"init": "base", "data": ["branch_set_rejection_sft", "branch_set_dpo"],
        "losses": ["sft", "dpo"], "teacher": False, "purpose": "MAIN generator-improvement arm (verifier prefs)",
        "on": True, "target_steps": 500}),
    ("D_logic_rendered_verifier", {"init": "base", "data": ["rendered_logic_branch_sets", "branch_set_dpo"],
        "losses": ["sft", "dpo"], "teacher": False, "domain_filter": "logic",
        "purpose": "spend the logic substrate", "on": False, "target_steps": 500}),
    ("E_teacher_coding_reasoning", {"init": "base", "data": ["branch_policy_distillation"], "losses": ["teacher_policy"],
        "teacher": True, "teacher_weights": {k: (v if k in ("coding", "reasoning") else 0.0) for k, v in TEACHER_WEIGHTS.items()},
        "purpose": "test teacher where v1 lift was real (coding/reasoning)", "on": False, "target_steps": 300}),
    ("F_teacher_plus_verifier", {"init": "base", "data": ["branch_set_rejection_sft", "branch_set_dpo",
        "branch_policy_distillation"], "losses": ["sft", "dpo", "teacher_policy"], "teacher": True,
        "teacher_weights": TEACHER_WEIGHTS, "purpose": "does teacher add after verifier signal?", "on": False,
        "target_steps": 500}),
    ("G_budget_plus_verifier", {"init": "base", "data": ["branch_set_rejection_sft", "branch_set_dpo",
        "direct_answer_sft", "branch_budget_policy"], "losses": ["sft", "dpo", "budget_cls"], "teacher": False,
        "purpose": "improve reachability WITHOUT overbranching (preferred)", "on": True, "target_steps": 500}),
    ("H_from_base_best_recipe", {"init": "base", "data": ["branch_set_rejection_sft", "branch_set_dpo",
        "direct_answer_sft", "multi_branch_sft", "branch_budget_policy"], "losses": ["sft", "dpo", "budget_cls"],
        "teacher": False, "purpose": "best recipe from clean base init (preferred)", "on": True, "target_steps": 600}),
    ("I_continue_previous_sft", {"init": "prev_sft", "data": ["branch_set_rejection_sft", "branch_set_dpo",
        "direct_answer_sft"], "losses": ["sft", "dpo"], "teacher": False,
        "purpose": "can the v1 300-step adapter be rescued?", "on": False, "target_steps": 300}),
]


def main() -> int:
    V.ensure_dirs()
    cfg_dir = V.MODEL_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    on, written = [], []
    for name, spec in ARMS:
        cfg = {"arm": name, **COMMON, **spec, "out_dir": str((V.MODEL_ROOT / name).relative_to(V.PROJECT_ROOT)),
               "data_root": str(V.TRAIN_V2.relative_to(V.PROJECT_ROOT)),
               "prev_adapter": str(V.PREV_SFT_ADAPTER.relative_to(V.PROJECT_ROOT)) if spec["init"] == "prev_sft" else None}
        (cfg_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2, default=V.v2.json_default))
        written.append(name)
        if spec.get("on"):
            on.append(name)

    # single 12GB GPU + slow Ouro generation => can't run all 9 to convergence; run the reduced ON set
    verdict = "MATRIX_REDUCED" if len(on) < len(ARMS) else "MATRIX_READY"
    payload = {"TRAINING_MATRIX_VERDICT": verdict, "arms_total": len(ARMS), "arms_on": on,
               "arms_off_resource_limited": [n for n, _ in ARMS if n not in on],
               "teacher_weights_default": TEACHER_WEIGHTS, "common": COMMON,
               "rationale": "single 12GB GPU + ~150s/task Ouro generation: train the spec's preferred candidates "
               "(C/G/H) + controls (A format, B direct-budget); others configured but OFF (resource-limited)."}
    V.write_json(V.OUT_ROOT / "training_matrix.json", payload)
    V.write_md(V.OUT_ROOT / "training_matrix.md", [
        "# Offline Training Matrix (Part I)", "", V.status_line("TRAINING_MATRIX_VERDICT", verdict),
        f"Configs written for all {len(ARMS)} arms under `{cfg_dir.relative_to(V.PROJECT_ROOT)}/`; "
        f"**{len(on)} ON** (resource-reduced), rest OFF.", "",
        "## Arms",
        *[f"- {'✅' if spec.get('on') else '⬜'} **{name}** — {spec['purpose']} (init {spec['init']}, "
          f"losses {spec['losses']}, {spec['target_steps']} steps)" for name, spec in ARMS],
        "", "## Domain-gated teacher weights (when teacher on)",
        *[f"- {d}: {w}" for d, w in TEACHER_WEIGHTS.items()],
        "", "Common: bf16 LoRA (r=16, all-linear), grad-checkpointing, base/tokenizer untouched, adapters separate, "
        "no heldout training, no online generation, canary eval every 200 steps, STOP-pausable, save every 100.",
        "", "Honest scope: one 12GB GPU + slow generation precludes converging all 9; the ON set covers the spec's "
        "preferred candidates (C verifier-DPO, G budget+verifier, H best-recipe) plus the A/B controls.",
    ])
    V.set_stage("I_training_matrix", verdict, {"arms_on": on})
    V.prog("I_training_matrix", {"verdict": verdict, "arms_on": on})
    print(V.status_line("TRAINING_MATRIX_VERDICT", verdict))
    print(f"  {len(written)} configs written | ON: {on}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
