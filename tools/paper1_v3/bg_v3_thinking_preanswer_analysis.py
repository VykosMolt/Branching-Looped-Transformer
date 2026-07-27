"""V3 Thinking pre-answer replication -- analysis (SEALED PROTOCOL).

Within-checkpoint analysis of the Thinking-generated Horizon candidates,
mirroring bg_v3_horizon_power_analysis: same fit family, task-grouped CV,
hyperparameter grid, deterministic split, seed, and 2000-round paired
task-clustered bootstrap. No cross-checkpoint AUROC comparisons.

SEALED ENDPOINTS AND VERDICTS (written before generation):
  Primary: within-Thinking heldout increment (combined - shortcut) under the
  adversarial 5-shortcut composite (incl. malformed-sibling count).
  Secondary: the published 4-shortcut composite.
  Integrity gate: the Thinking task_uid set must be exactly the RLTT v2 main
  task set (same slice), with zero split mismatches.

  Verdicts:
    THINKING_PREANSWER_REPLICATED_AND_ROBUST  (4sc and 5sc exclude zero)
    THINKING_PREANSWER_POSITIVE_NOT_ROBUST    (4sc excludes zero, 5sc does not)
    THINKING_PREANSWER_NULL                   (4sc does not exclude zero)
  Any insufficient-data condition (single-class heldout, <20 scorable in a
  split) is reported as THINKING_PREANSWER_UNDERPOWERED with counts, not
  massaged.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bg_v2_overnight_horizon_analysis import HORIZON_ROOT, SEED, shortcut_vec  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402
from bg_v2_malformed_sibling_control import shortcut_vec_aug, malformed_count_by_task  # noqa: E402
from bg_v3_horizon_power_analysis import load_records, run_arm  # noqa: E402

T_ROOT = PROJECT_ROOT / "artifacts/reports/thinking_preanswer_v3_20260726/horizon_logic"
OUT_ROOT = PROJECT_ROOT / "artifacts/reports/thinking_preanswer_v3_20260726"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    recs = load_records(T_ROOT)
    if not recs:
        raise SystemExit("no Thinking shards found")
    rltt = load_records(HORIZON_ROOT)
    t_tasks, r_tasks = {r["task_uid"] for r in recs}, {r["task_uid"] for r in rltt}
    assert t_tasks == r_tasks, (
        f"task set mismatch vs RLTT v2 main slice: only_thinking={len(t_tasks - r_tasks)} "
        f"only_rltt={len(r_tasks - t_tasks)}")
    bad = [r["task_uid"] for r in recs if r["split"] != split_for(r["task_uid"])]
    assert not bad, f"split mismatch: {bad[:5]}"
    print(f"[thinking] records={len(recs)} tasks={len(t_tasks)} (== RLTT v2 slice) · gate OK")

    n_mal = sum(r["malformed"] for r in recs)
    scor = [r for r in recs if not r["malformed"]]
    ho = [r for r in scor if r["split"] == "heldout"]
    results = {
        "seed": SEED, "n_candidates": len(recs), "n_malformed": n_mal,
        "malformed_rate": n_mal / len(recs),
        "n_scorable": len(scor),
        "n_heldout_scorable": len(ho),
        "n_heldout_negatives": sum(1 for r in ho if not r["success"]),
        "rltt_reference_malformed_rate": sum(r["malformed"] for r in rltt) / len(rltt),
        "task_set_gate": "PASSED",
    }
    tv = [r for r in scor if r["split"] in ("train", "val")]
    single_class = (len({bool(r["success"]) for r in ho}) < 2)
    if len(tv) < 20 or len(ho) < 20 or single_class:
        results["verdict"] = "THINKING_PREANSWER_UNDERPOWERED"
        (OUT_ROOT / "thinking_preanswer_results.json").write_text(
            json.dumps(results, indent=2, default=str) + "\n")
        print("[thinking] UNDERPOWERED:", {k: results[k] for k in
              ("n_scorable", "n_heldout_scorable", "n_heldout_negatives")})
        return

    mal = malformed_count_by_task(recs)
    results["arm_4sc"], preds4 = run_arm("thinking_4sc", recs, shortcut_vec)
    results["arm_5sc"], preds5 = run_arm("thinking_5sc",
                                         recs, lambda r: shortcut_vec_aug(r, mal))
    b4 = results["arm_4sc"]["paired_task_clustered_bootstrap_combined_minus_shortcut"]
    b5 = results["arm_5sc"]["paired_task_clustered_bootstrap_combined_minus_shortcut"]
    if b4["excludes_zero"] and b5["excludes_zero"]:
        v = "THINKING_PREANSWER_REPLICATED_AND_ROBUST"
    elif b4["excludes_zero"]:
        v = "THINKING_PREANSWER_POSITIVE_NOT_ROBUST"
    else:
        v = "THINKING_PREANSWER_NULL"
    results["verdict"] = v
    (OUT_ROOT / "thinking_preanswer_results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n")
    (OUT_ROOT / "raw_predictions_heldout.json").write_text(
        json.dumps({"arm_4sc": preds4, "arm_5sc": preds5}, indent=2) + "\n")
    print(f"[thinking] verdict: {v}")
    print(f"[thinking] 4sc: {b4['mean_delta']:+.4f} {b4['ci95']}")
    print(f"[thinking] 5sc: {b5['mean_delta']:+.4f} {b5['ci95']}")


if __name__ == "__main__":
    main()
