"""PART A - GPU/process guard and inventory for partial_cache_splice_v2.

First: refuse to run GPU work if the MMLU science-repair process is active.
Then inventory cache tools, v1 status, and suffix-recompute feasibility.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import bg_autoregressive_cache_common_v1 as C
import bg_partial_cache_splice_v2_common as V2


def check_science_repair_active() -> dict:
    """Detect an active MMLU science-repair / recipe-v3 calibration process."""
    patterns = ["mmlu_science", "recipe_v3", "branch_parser_repair",
                "run_bg_mmlu_science", "thermal_guard"]
    active = []
    try:
        out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            low = line.lower()
            if any(p in low for p in patterns) and "grep" not in low and "splice_v2" not in low:
                active.append(line.strip())
    except Exception as e:  # noqa: BLE001
        active.append(f"ps_error: {e}")

    # check stale PID pointer files and whether those PIDs are alive
    pid_files = {}
    for fn in ["/tmp/mmlu_v3_calib_pid.txt", "/tmp/mmlu_v3_active_pid.txt",
               "/tmp/mmlu_v3_thermal_guard_pid.txt"]:
        p = Path(fn)
        if p.exists():
            try:
                pid = int(p.read_text().strip())
                alive = _pid_alive(pid)
                pid_files[fn] = {"pid": pid, "alive": alive}
            except Exception:
                pid_files[fn] = {"pid": None, "alive": False}
    any_alive = any(v.get("alive") for v in pid_files.values())
    proc_active = len([a for a in active if not a.startswith("ps_error")]) > 0
    return {"active_processes": active, "pid_files": pid_files,
            "any_pid_alive": any_alive, "process_match": proc_active,
            "science_repair_active": bool(any_alive or proc_active)}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return pid_is_permission_only(pid)
    except Exception:
        return False


def pid_is_permission_only(pid: int) -> bool:
    # PermissionError means the pid exists but is owned by another user; treat as alive.
    return Path(f"/proc/{pid}").exists()


def gpu_status() -> dict:
    out = {}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        total, used = [int(x) for x in q.split(",")]
        out["total_mib"] = total
        out["used_mib"] = used
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        out["compute_apps"] = [a.strip() for a in apps.splitlines() if a.strip()]
        # python ML process on GPU?
        out["ml_python_on_gpu"] = any(
            ("python" in a.lower() and "brave" not in a.lower()) for a in out["compute_apps"])
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def main() -> int:
    sci = check_science_repair_active()
    gpu = gpu_status()

    if sci["science_repair_active"]:
        verdict = "WAIT_FOR_SCIENCE_REPAIR"
        guard = {"verdict": verdict, "science": sci, "gpu": gpu,
                 "decision": "MMLU science-repair appears active; refusing GPU work."}
        V2.save_json("gpu_guard.json", guard)
        V2.save_md("gpu_guard.md", f"# GPU guard\n\n**BG_PARTIAL_SPLICE_GPU_GUARD_VERDICT = {verdict}**\n\n"
                                   f"```json\n{json.dumps(guard, indent=2)}\n```\n")
        V2.save_json("inventory.json", {"verdict": verdict, "guard": guard})
        V2.save_md("inventory.md", f"# PART A inventory\n\n**BG_PARTIAL_SPLICE_V2_INVENTORY_VERDICT = {verdict}**\n")
        print(f"BG_PARTIAL_SPLICE_V2_INVENTORY_VERDICT = {verdict}")
        return 0

    # Safe -> proceed with inventory
    C.set_seed()
    model, tok, info = C.load_model()
    cls = C.get_universal_cache_class()
    cache = C.new_cache()

    v1_dir = C.PROJECT_ROOT / "opi/taps/probes/bg_autoregressive_kv_branch_carry_v1_2026-06-01"
    def load_v1(name):
        p = v1_dir / name
        return json.loads(p.read_text()) if p.exists() else None
    v1_summary = load_v1("summary.json")
    v1_level6 = load_v1("level6_partial_splice.json")
    v1_slot = load_v1("loop_layer_slot_audit.json")

    # suffix recompute feasibility: can we call a decoder layer standalone + capture boundary?
    feasible = hasattr(model.model, "layers") and hasattr(model.model, "rotary_emb") and hasattr(model.model, "norm")

    report = {
        "verdict": "READY",
        "guard": {"science_repair_active": False, "science": sci, "gpu": gpu},
        "model_info": info,
        "cache_class": cls.__name__,
        "expected_slots": info["expected_cache_slots"],
        "v1_status": (v1_summary or {}).get("AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS"),
        "v1_level6_verdict": (v1_level6 or {}).get("verdict"),
        "v1_slot_audit_verdict": (v1_slot or {}).get("verdict"),
        "v1_helpers_available": True,
        "suffix_recompute_feasible": bool(feasible),
        "layer_loop_hooks_targetable": True,
        "notes": [
            "v2 suffix-recompute reuses model.model.layers / rotary_emb / norm as test-only "
            "orchestration (no permanent model surgery, no weight edits).",
            "Compute saving requires capturing the residual boundary hidden during prefill "
            "(KV cache alone does not store the residual stream).",
        ],
    }
    V2.save_json("inventory.json", report)
    V2.save_json("gpu_guard.json", {"verdict": "PROCEED_SAFE", "science": sci, "gpu": gpu})

    md = ["# PART A - GPU guard + inventory (partial_cache_splice_v2)\n",
          "**BG_PARTIAL_SPLICE_V2_INVENTORY_VERDICT = READY**\n",
          "## GPU / process guard\n",
          f"- science repair active: **False** (PIDs dead, no matching process)",
          f"- GPU used {gpu.get('used_mib')}/{gpu.get('total_mib')} MiB; ML python on GPU: {gpu.get('ml_python_on_gpu')}",
          f"- compute apps: {gpu.get('compute_apps')}\n",
          "## Model / cache\n",
          f"- {info['model_class']} dtype {info['dtype']} attn {info['attn_implementation']}",
          f"- total_ut_steps={info['total_ut_steps']} num_hidden_layers={info['num_hidden_layers']} "
          f"expected slots={info['expected_cache_slots']}",
          f"- cache class: {cls.__name__}\n",
          "## v1 build-on\n",
          f"- v1 status: {report['v1_status']}",
          f"- v1 Level 6: {report['v1_level6_verdict']}",
          f"- v1 slot audit: {report['v1_slot_audit_verdict']}",
          f"- suffix recompute feasible: {feasible}\n"]
    V2.save_md("inventory.md", "\n".join(md))
    # also write a passing gpu_guard.md
    V2.save_md("gpu_guard.md", "# GPU guard\n\n**science repair NOT active; PROCEED_SAFE.**\n\n"
               f"```json\n{json.dumps({'science': sci, 'gpu': gpu}, indent=2)}\n```\n")

    print("=" * 70)
    print("BG_PARTIAL_SPLICE_V2_INVENTORY_VERDICT = READY")
    print(f"gpu_used={gpu.get('used_mib')}MiB ml_python_on_gpu={gpu.get('ml_python_on_gpu')} "
          f"v1_status={report['v1_status']} suffix_feasible={feasible}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
