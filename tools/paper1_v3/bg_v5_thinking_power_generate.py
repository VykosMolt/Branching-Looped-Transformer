"""V5 Thinking pre-answer power extension -- generation wrapper (path-only).

Calls the EXISTING sealed Thinking generator (bg_v3_thinking_preanswer_generate,
which itself wraps the sealed v2 Horizon protocol: k=4, max_new=448, T=0.7,
top_p=0.95, seed 20260724, strict pre-answer cut, hash-ordered pool, cap 4000)
on fresh task offsets. NOTHING in the sealed generator files is modified; only
two module attributes are overridden at call time:

  - OUT_ROOT          -> the v5 run directory (path-only change)
  - MAX_OFFSET_EXCL   -> raised from the pilot's mirror cap (170) to the end of
                         the v5 slice, so the sealed guard admits offsets
                         >= 1500 instead of refusing everything past the pilot
                         slice. The disjointness guard this wrapper enforces
                         (START_OFFSET >= 1500) is strictly stronger than the
                         one it relaxes.

Task-slice ledger (hash-ordered pool, offsets): [0,170) v2/pilot, [170,680) v3,
[680,860) C2, [860,1080)+[1350,1500) C4, [1080,1260) C3+C5-train,
[1260,1350) C5-eval. The v5 slice starts at 1500 -- the first unused offset --
and must stay within the pool cap 4000 (DO NOT raise the cap: membership
changes).

Generation is chunked into shards of CHUNK_TASKS tasks (one sealed-generator
invocation each, resumable: existing shards are skipped). ~39 s/task on the
pilot hardware.

Usage:
  venv/bin/python -u utilities/tests/manual/bg_v5_thinking_power_generate.py
  venv/bin/python utilities/tests/manual/bg_v5_thinking_power_generate.py --plan-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# N_TASKS: set by the operator after reviewing bg_v5_thinking_power_sim output.
# Sim (seed 20260727, 500 reps/cell): see sizing_simulation_v5.json.
# ---------------------------------------------------------------------------
N_TASKS = 900                # <-- OPERATOR-SET total new tasks for the v5 slice
START_OFFSET = 1500          # first unused offset in the hash-ordered pool
POOL_CAP = 4000              # sealed pool cap -- membership changes if raised
CHUNK_TASKS = 150            # tasks per shard / sealed-generator invocation
SEC_PER_TASK = 39.0          # pilot wall-clock rate
TAG = "think_pow"
SEED = 20260724              # sealed protocol seed (per-task seeds derive from task_uid)
K = 4
MAX_NEW = 448                # sealed at 448 by the pilot plan; NOT to be revised

V5_ROOT = PROJECT_ROOT / "artifacts/reports/thinking_preanswer_power_v5_20260727/horizon_logic"


def chunk_plan():
    assert START_OFFSET >= 1500, (
        f"refusing: START_OFFSET {START_OFFSET} < 1500 overlaps prior sealed slices "
        "(ledger: [0,1500) consumed)")
    assert N_TASKS > 0, "N_TASKS must be positive"
    assert START_OFFSET + N_TASKS <= POOL_CAP, (
        f"refusing: slice [{START_OFFSET}, {START_OFFSET + N_TASKS}) exceeds the "
        f"sealed pool cap {POOL_CAP}")
    chunks = []
    off = START_OFFSET
    while off < START_OFFSET + N_TASKS:
        n = min(CHUNK_TASKS, START_OFFSET + N_TASKS - off)
        chunks.append((off, n))
        off += n
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true",
                    help="print the chunk plan and guards without loading anything")
    args = ap.parse_args()

    chunks = chunk_plan()
    est_h = N_TASKS * SEC_PER_TASK / 3600
    print(f"[v5-gen] slice [{START_OFFSET}, {START_OFFSET + N_TASKS}) -- {N_TASKS} tasks, "
          f"{len(chunks)} shards of <= {CHUNK_TASKS}, est {est_h:.1f} h at "
          f"{SEC_PER_TASK:.0f} s/task", flush=True)
    for off, n in chunks:
        print(f"[v5-gen]   shard offset={off} n_tasks={n} -> "
              f"horizon_generation_{TAG}_off{off}.pt", flush=True)
    if args.plan_only:
        print("[v5-gen] --plan-only: guards passed, exiting", flush=True)
        return

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    V5_ROOT.mkdir(parents=True, exist_ok=True)

    import bg_v3_thinking_preanswer_generate as tgen  # noqa: E402  (heavy: transformers)
    import src.evaluator.bg_transformer_features as feat  # noqa: E402

    base_extractor = feat.BGTransformerFeatureExtractor
    tgen.OUT_ROOT = V5_ROOT                      # path-only override
    tgen.MAX_OFFSET_EXCL = START_OFFSET + N_TASKS  # admit the v5 slice (see docstring)

    for off, n in chunks:
        shard = V5_ROOT / "horizon_logic" / f"horizon_generation_{TAG}_off{off}.pt"
        if shard.exists():
            print(f"[v5-gen] shard offset={off} exists -- skipping (resume)", flush=True)
            continue
        # reset any extractor patch from the previous chunk to avoid subclass nesting
        feat.BGTransformerFeatureExtractor = base_extractor
        budget = int(n * SEC_PER_TASK * 1.6) + 300
        sys.argv = [
            "bg_v3_thinking_preanswer_generate.py",
            "--task-offset", str(off), "--n-tasks", str(n),
            "--k", str(K), "--max-new", str(MAX_NEW), "--seed", str(SEED),
            "--tag", TAG, "--budget-seconds", str(budget),
        ]
        print(f"[v5-gen] START shard offset={off} n_tasks={n} budget={budget}s", flush=True)
        tgen.main()
        idx_path = V5_ROOT / "horizon_logic" / f"horizon_generation_{TAG}_off{off}_index.json"
        if idx_path.exists():
            light = json.loads(idx_path.read_text())
            if light.get("budget_hit"):
                print(f"[v5-gen] WARN shard offset={off} hit its budget: "
                      f"{light['n_tasks_completed']}/{light['n_tasks_requested']} tasks",
                      flush=True)
        print(f"[v5-gen] DONE shard offset={off}", flush=True)

    print(f"[v5-gen] ALL SHARDS DONE: slice [{START_OFFSET}, {START_OFFSET + N_TASKS})",
          flush=True)


if __name__ == "__main__":
    main()
