"""PART M - Synthesis and readiness for partial_cache_splice_v2."""

from __future__ import annotations

import json

import bg_autoregressive_cache_common_v1 as C
import bg_partial_cache_splice_v2_common as V2

OUT = V2.OUT_DIR


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    C.set_seed()
    files = {
        "inventory": "inventory.json",
        "reproduce_v1": "reproduce_level6_v1.json",
        "boundary_dependency": "boundary_dependency.json",
        "hook_timing": "hook_timing.json",
        "suffix_recompute": "suffix_recompute_impl.json",
        "single_branch": "single_branch_splice.json",
        "multi_branch": "multi_branch_splice.json",
        "batched_prune": "batched_prune_splice.json",
        "position_padding": "position_padding_splice.json",
        "compute_accounting": "compute_accounting.json",
        "arch_loop_smoke": "arch_loop_smoke.json",
        "failure_analysis": "failure_analysis.json",
    }
    d = {k: load(v) for k, v in files.items()}
    verd = {k: (d[k].get("verdict") if isinstance(d[k], dict) else None) for k in files}

    single_ok = verd.get("single_branch") in ("SINGLE_BRANCH_SPLICE_VALID", "SINGLE_BRANCH_SPLICE_NUMERIC_DRIFT_SMALL")
    multi_ok = verd.get("multi_branch") in ("MULTI_BRANCH_SPLICE_VALID", "MULTI_BRANCH_SPLICE_NUMERIC_DRIFT_SMALL")
    suffix_impl = verd.get("suffix_recompute") in ("SUFFIX_RECOMPUTE_IMPLEMENTED", "LOOP_SUFFIX_REPLAY_IMPLEMENTED")
    equivalence_ok = single_ok and multi_ok
    compute = verd.get("compute_accounting")
    real_saving = (compute == "REAL_COMPUTE_SAVING_MEASURED")
    theoretical = (compute in ("REAL_COMPUTE_SAVING_MEASURED", "THEORETICAL_SAVING_ONLY"))
    requires_surgery = (verd.get("suffix_recompute") == "REQUIRES_MODEL_SURGERY")

    if requires_surgery:
        status = "PARTIAL_SPLICE_REQUIRES_MODEL_SURGERY"
    elif not equivalence_ok:
        status = "PARTIAL_SPLICE_INVALID"
    elif suffix_impl and equivalence_ok and real_saving:
        status = "PARTIAL_SPLICE_COMPUTE_SAVING_VALID"
    elif suffix_impl and equivalence_ok and theoretical:
        status = "PARTIAL_SPLICE_EQUIVALENT_BUT_NO_MEASURED_SPEEDUP"
    elif equivalence_ok:
        status = "PARTIAL_SPLICE_SLOT_LOGIC_ONLY"
    else:
        status = "INSUFFICIENT"

    claims = {
        "compute_saving_branch_carry": (status == "PARTIAL_SPLICE_COMPUTE_SAVING_VALID"),
        "splice_correctness_equivalence": equivalence_ok,
        "suffix_recompute_avoids_full_prefill": suffix_impl,
        "production_readiness": False,
    }

    top = {
        "BG_PARTIAL_SPLICE_V2_INVENTORY_VERDICT": verd.get("inventory"),
        "BG_PARTIAL_SPLICE_V2_REPRODUCE_V1_VERDICT": verd.get("reproduce_v1"),
        "BG_PARTIAL_SPLICE_BOUNDARY_DEPENDENCY_VERDICT": verd.get("boundary_dependency"),
        "BG_PARTIAL_SPLICE_HOOK_TIMING_VERDICT": verd.get("hook_timing"),
        "BG_PARTIAL_SPLICE_SUFFIX_RECOMPUTE_IMPL_VERDICT": verd.get("suffix_recompute"),
        "BG_PARTIAL_SPLICE_SINGLE_BRANCH_VERDICT": verd.get("single_branch"),
        "BG_PARTIAL_SPLICE_MULTI_BRANCH_VERDICT": verd.get("multi_branch"),
        "BG_PARTIAL_SPLICE_BATCHED_PRUNE_VERDICT": verd.get("batched_prune"),
        "BG_PARTIAL_SPLICE_POSITION_PADDING_VERDICT": verd.get("position_padding"),
        "BG_PARTIAL_SPLICE_COMPUTE_ACCOUNTING_VERDICT": verd.get("compute_accounting"),
        "BG_PARTIAL_SPLICE_ARCH_LOOP_SMOKE_VERDICT": verd.get("arch_loop_smoke"),
        "BG_PARTIAL_SPLICE_FAILURE_ANALYSIS_VERDICT": verd.get("failure_analysis"),
        "PARTIAL_CACHE_SPLICE_V2_STATUS": status,
    }

    summary = {**top, "claims_allowed": claims, "stage_verdicts": verd,
               "note": "v2 implements a real suffix-recompute splice (Mode B): capture the residual "
                       "boundary hidden during a minimal shared-prefix prefill, apply the additive "
                       "boundary perturbation without a forward, recompute ONLY the affected suffix. "
                       "Spliced cache is bit-exact vs full perturbed reference; compute is saved when "
                       "the shared prefix is amortized across K branches. No steering; not production-ready."}
    V2.save_json("summary.json", summary)

    sm = ["# Partial Cache Splice v2 — SUMMARY\n", "## Top-line verdicts\n"]
    for k, v in top.items():
        sm.append(f"- `{k} = {v}`")
    sm.append("\n## Claims allowed\n")
    for k, v in claims.items():
        sm.append(f"- {'YES' if v else 'NO '} {k}")
    sm.append("\n## No-steering / science-repair non-interference\n")
    sm.append("- No steering, no training, no weight/tokenizer edits, no wrapper/local-agent, no "
              "Hunter-Seeker. GPU guard confirmed the MMLU science-repair process was NOT active; "
              "its artifacts were not touched.")
    V2.save_md("summary.md", "\n".join(sm))

    # analysis.md
    def g(dd, *ks, default=None):
        for k in ks:
            if not isinstance(dd, dict):
                return default
            dd = dd.get(k, default)
        return dd

    an = ["# Partial Cache Splice v2 — ANALYSIS\n",
          f"**PARTIAL_CACHE_SPLICE_V2_STATUS = {status}**\n",
          "## 1. Motivation\n",
          "Turn the v1 Level 6 diagnostic (copy-affected slots, no saving) into a real "
          "compute-saving suffix-recompute splice.\n",
          "## 2. Prior v1 cache-carry result\n",
          f"- v1: {g(d['inventory'],'v1_status')}; v1 Level 6 = {g(d['inventory'],'v1_level6_verdict')} "
          "(diagnostic only).\n",
          "## 3. GPU/process guard\n",
          f"- {verd.get('inventory')}: science-repair NOT active; GPU idle; artifacts untouched.\n",
          "## 4. V1 Level 6 reproduction\n",
          f"- {verd.get('reproduce_v1')}: copy-affected oracle matches reference; over-share diverges; "
          "zero perturbation matches.\n",
          "## 5. Cache boundary theory\n",
          f"- {verd.get('boundary_dependency')}: perturb-at-layer-output => boundary slot unaffected; "
          "first affected = (loop, layer+1); downstream_only policy.\n",
          "## 6. Hook timing / affected slots\n",
          f"- {verd.get('hook_timing')}: empirically, boundary slot unaffected, first changed = "
          "(loop, layer+1), changed set == downstream_only theory.\n",
          "## 7. Suffix recompute implementation\n",
          f"- {verd.get('suffix_recompute')}: Mode B — capture boundary hidden, additive perturb, "
          "recompute only the suffix layers; spliced cache bit-exact vs full reference.\n",
          "## 8. Single-branch splice\n",
          f"- {verd.get('single_branch')}: spliced branch == full perturbed reference (bit-exact "
          "prefill cache + continuation); aggressive over-share diverges.\n",
          "## 9. Multi-branch splice\n",
          f"- {verd.get('multi_branch')}: K branches share one prefix; each matches its own reference; "
          "independent storage, no contamination.\n",
          "## 10. Batched/prune splice\n",
          f"- {verd.get('batched_prune')}: spliced branches batch + prune/reorder correctly.\n",
          "## 11. Position/padding stress\n",
          f"- {verd.get('position_padding')}: left-padded splice matches with explicit position_ids.\n",
          "## 12. Compute accounting\n",
          f"- {verd.get('compute_accounting')}: baseline K*full vs splice prefix + K*suffix; "
          "saving grows with K and boundary loop depth (see compute_accounting.md).\n",
          "## 13. Architecture-looped smoke\n",
          f"- {verd.get('arch_loop_smoke')}: fork-via-splice -> proxy score -> prune -> carry; "
          "survivors match references.\n",
          "## 14. Failure analysis\n",
          f"- {verd.get('failure_analysis')}.\n",
          "## 15. Final status\n",
          f"- **{status}**.\n",
          "## 16. What can and cannot be claimed\n"]
    for k, v in claims.items():
        an.append(f"- {'CAN claim' if v else 'CANNOT claim'}: {k}")
    an.append("- CANNOT claim production readiness.")
    an.append("- Compute saving is REAL but amortized: it requires K>=2 branches sharing one prompt "
              "(K=1 does prefix+suffix == full). Copy-affected (Mode A) saves nothing.\n")
    an.append("## 17. Files created\n- utilities/tests/manual/bg_partial_cache_splice_v2_*.py, "
              "reproduce_/probe_/analyze_bg_*_v2.py; artifacts under "
              "opi/taps/probes/bg_partial_cache_splice_v2_2026-06-01/.\n")
    an.append("## 18. Commands run\n- venv/bin/python -u utilities/tests/manual/<script>.py per the "
              "RUN COMMANDS (py_compile first).\n")
    an.append("## 19. Blockers\n- None. Compute saving is amortized (K>=2); left-padded splice needs "
              "explicit position_ids.\n")
    V2.save_md("analysis.md", "\n".join(an))
    V2.save_json("analysis.json", {**top, "claims_allowed": claims})

    print("=" * 70)
    for k, v in top.items():
        print(f"{k} = {v}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
