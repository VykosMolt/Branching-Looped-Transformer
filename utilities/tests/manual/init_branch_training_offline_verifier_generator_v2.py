"""PART A — run-state initialization for branch_training_offline_verifier_generator_v2.

Creates run/model roots, the state ledger, STOP-file path, resume plan, and artifact manifest.
Audits inputs, training stack, and GPU before anything heavy runs. Idempotent.
"""
from __future__ import annotations
import importlib
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402

REQUIRED_INPUTS = [
    "processed/logic_tasks.parquet",
    "processed/branch_pools_labeled.parquet",
    "processed/branch_pools_deduped.parquet",
    "terminal_survivor_sets.parquet",
    "teacher/dualanchor_teacher_traces.parquet",
    "train/branch_format_sft.jsonl",
    "train/branch_diversity_sft.jsonl",
    "train/branch_policy_distillation.jsonl",
    "train/final_self_selection.jsonl",
    "train/verifier_reward_rl.jsonl",
    "train/branch_preference_dpo.jsonl",
    "processed/gen_shards/gen_manifest.json",
]
STACK = ["torch", "transformers", "peft", "trl", "bitsandbytes", "z3", "datasets"]

PARTS = [
    ("A", "init", "run state + manifest"),
    ("B", "input_audit", "verify v1 substrate"),
    ("C", "canary_suite", "fixed >=100/domain canary"),
    ("D", "canary_baselines", "base / prev-SFT / oracle / random baselines (GPU)"),
    ("E", "rendered_logic_branch_sets", "solver-rendered verified logic branch sets (offline)"),
    ("F", "branch_budget_data", "direct-answer + branch-budget views (data-composition repair)"),
    ("G", "offline_branch_set_preferences", "rejection-SFT + branch-set DPO + reward groups"),
    ("H", "offline_reward_spec", "deterministic verifier-backed reward function"),
    ("I", "training_matrix", "controlled offline arms A..I"),
    ("J", "offline_training", "bf16 LoRA offline training (resumable)"),
    ("K", "canary_checkpoint_eval", "select <=3 adapters for heldout"),
    ("L", "final_heldout_eval", "task-disjoint heldout, once"),
    ("M", "online_rl_gate", "decide if online RL is justified (no GRPO here)"),
    ("N", "error_audit", "hand-audit failures vs artifacts"),
    ("O", "policy_decision", "next architecture state"),
    ("P", "doc_update", "docs (append-only)"),
]


def main() -> int:
    started = time.time()
    V.ensure_dirs()
    gpu = V.gpu_audit()

    inputs = {p: (V.DATA_ROOT / p).exists() for p in REQUIRED_INPUTS}
    missing_inputs = [p for p, ok in inputs.items() if not ok]
    prev_adapter = (V.PREV_SFT_ADAPTER / "adapter_model.safetensors").exists()
    base_ok = (V.BASE_MODEL / "config.json").exists()

    stack = {}
    for m in STACK:
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, "__version__", None) or (mod.get_version_string() if hasattr(mod, "get_version_string") else "ok")
            stack[m] = str(ver)
        except Exception as e:
            stack[m] = f"MISSING:{e}"
    missing_stack = [m for m, v in stack.items() if v.startswith("MISSING")]

    gpu_ok = (gpu.get("free_mib") or 0) >= 8000 and not gpu.get("orphans")

    if not base_ok or missing_inputs:
        verdict = "MISSING_INPUTS"
    elif missing_stack:
        verdict = "MISSING_TRAINING_STACK"
    elif not gpu_ok:
        verdict = "GPU_NOT_READY"
    else:
        verdict = "READY"

    manifest = {
        "run": V.RUN_NAME, "out_root": str(V.OUT_ROOT.relative_to(V.PROJECT_ROOT)),
        "model_root": str(V.MODEL_ROOT.relative_to(V.PROJECT_ROOT)),
        "data_root": str(V.DATA_ROOT.relative_to(V.PROJECT_ROOT)),
        "train_v2": str(V.TRAIN_V2.relative_to(V.PROJECT_ROOT)),
        "canary_dir": str(V.CANARY_DIR.relative_to(V.PROJECT_ROOT)),
        "log_dir": str(V.LOG_DIR.relative_to(V.PROJECT_ROOT)),
        "prev_sft_adapter": str(V.PREV_SFT_ADAPTER.relative_to(V.PROJECT_ROOT)),
        "stop_dir": str(V.STOP_DIR.relative_to(V.PROJECT_ROOT)),
        "inputs_present": inputs, "prev_adapter_present": prev_adapter, "base_model_present": base_ok,
        "stack": stack, "gpu": gpu,
        "do_not_overwrite": [
            "shared/models/ouro_rltt_local/", "tokenizer files", "constructed_taps/pure_content_taps.pt",
            "constructed_taps/transplanted_taps.pt", "CoreContent_v2 artifacts",
            "branch_training_logic_expansion_v1 reports/adapters/checkpoints", "tap registries",
        ],
        "parts": [{"part": p, "stage": s, "desc": d} for p, s, d in PARTS],
    }
    V.write_json(V.OUT_ROOT / "artifact_manifest.json", manifest)
    V.write_json(V.OUT_ROOT / "run_state.json", {
        "run": V.RUN_NAME, "init_verdict": verdict, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {"A_init": {"verdict": verdict, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}},
        "active_pid": None, "stop_dir": str(V.STOP_DIR),
    })
    V.write_md(V.OUT_ROOT / "run_state.md", [
        "# Offline Verifier-Backed Branch-Generator v2 — Run State", "",
        V.status_line("OFFLINE_BRANCH_GENERATOR_INIT_VERDICT", verdict), "",
        "Goal: train a useful **branch generator** (raise positive_oracle@K) **offline-first** from the v1 "
        "verifier-backed substrate, before paying for online RL. Near-term target: `USE_BRANCH_TRAINED_MODEL_AS_GENERATOR` "
        "(better pools; DualAnchor still prunes; CoreContent_v2 still ranks). Self-selection beating CoreContent not required.",
        "", "## Locked external baseline (unchanged unless heldout evidence demotes it)",
        "- Branch survival: DualAnchor (MIX_CODE_REASONING + MIX_OBJECTIVE_ALL).",
        "- Content/final selection: CoreContent_v2_blockwise.",
        "- Terminal: top5/full survivor-set handoff; no unconditional terminal top1. Science: diagnostic only.",
        "- Steering: claimed only if a trained write-path adapter wins on heldout. Tap/readout selection is NOT steering.",
        "", "## Ground truth", "- Correctness only from external verifiers (unit tests / exact / MCQ keys / parser / z3 / "
        "symbolic / rubrics / preference labels). DualAnchor = keep/prune/rescue/expand/defer policy teacher; "
        "CoreContent_v2 = terminal soft ranker. Neither gives correctness/reward/answer labels.",
        "", f"## Inputs ({len(REQUIRED_INPUTS) - len(missing_inputs)}/{len(REQUIRED_INPUTS)} present)",
        *( [f"- MISSING: `{p}`" for p in missing_inputs] or ["- all required v1 inputs present."]),
        f"- base model: {base_ok}; prev SFT adapter: {prev_adapter}.",
        "", "## Stack", *[f"- {m}: {v}" for m, v in stack.items()],
        "", f"## GPU\n- free {gpu.get('free_mib')} MiB / util {gpu.get('util_pct')}% / orphans {len(gpu.get('orphans', []))}.",
        "", "## Operational", "- STOP-file pausable (`" + str(V.STOP_DIR.relative_to(V.PROJECT_ROOT)) + "/STOP` or `STOP_<job>`).",
        "- Resumable checkpoints + shard manifests; frequent logging; no cron unless asked; single active-pid source of truth.",
        "- Online RL is GATED (Part M); not run here.",
    ])
    V.write_md(V.OUT_ROOT / "resume_plan.md", [
        "# Resume Plan", "",
        "Run is sequential A→P; each stage writes a verdict to `run_state.json` `stages`. Re-running a stage is "
        "idempotent. Training/eval stages are STOP-file pausable and checkpoint-resumable.", "",
        "## Order & gates",
        "1. A init → B input audit → C canary suite → D canary baselines. **Gate: do not train until C+D saved.**",
        "2. E rendered logic + F budget/direct-answer + G offline prefs + H reward spec (offline data; CPU-cheap).",
        "3. I matrix → J offline training (resumable) → K canary checkpoint eval → L final heldout (once).",
        "4. M online-RL gate (decision only) → N error audit → O policy → P docs.", "",
        "## Parts", *[f"- {p} — {s}: {d}" for p, s, d in PARTS],
        "", "## Cost note",
        "- Ouro-RLTT generation is the wall (~150–170 s/task at v1 budgets). Offline data (E/F/G) needs no generation. "
        "Generation-based stages (D baselines, K/L evals) are resumable long runs; canary GEN size is a cost/precision knob.",
    ])
    V.set_stage("A_init", verdict, {"missing_inputs": missing_inputs, "missing_stack": missing_stack,
                                    "gpu_free_mib": gpu.get("free_mib")})
    V.prog("A_init", {"verdict": verdict})
    print(V.status_line("OFFLINE_BRANCH_GENERATOR_INIT_VERDICT", verdict))
    print(f"  inputs {len(REQUIRED_INPUTS)-len(missing_inputs)}/{len(REQUIRED_INPUTS)} | stack_missing={missing_stack} | "
          f"gpu_free={gpu.get('free_mib')}MiB orphans={len(gpu.get('orphans', []))} | elapsed {round(time.time()-started,2)}s")
    return 0 if verdict == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
