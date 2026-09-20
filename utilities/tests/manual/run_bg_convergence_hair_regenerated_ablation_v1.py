from __future__ import annotations

import time

from bg_convergence_hairs_rs_v1_common import OUT_ROOT, REPLAY_JSON, read_json, status_line, write_csv, write_json, write_md


def main() -> int:
    started = time.time()
    replay = read_json(REPLAY_JSON, {}) or {}
    verdict = "SKIPPED"
    reason = (
        "Skipped by default because the v3 .pt artifact already contains L30/L42 hidden states for replay, "
        "and a live regenerated ablation would require overnight-scale model generation. This preserves the "
        "no-steering/no-training/no-wrapper constraints and avoids making compute-savings claims."
    )
    payload = {
        "BG_CONVERGENCE_HAIR_REGENERATED_VERDICT": verdict,
        "reason": reason,
        "replay_verdict": replay.get("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT"),
        "modes_requested": [
            "no_hair_baseline",
            "soft_cluster_diagnostic",
            "L30_representative_merge",
            "L42_representative_merge",
            "L30_L42_representative_merge",
            "conservative_hair_best_policy_from_replay",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "convergence_hair_regenerated.json", payload)
    write_csv(OUT_ROOT / "convergence_hair_regenerated_rows.csv", [])
    lines = [
        "# DualAnchor Convergence Hair Regenerated Ablation v1",
        "",
        status_line("BG_CONVERGENCE_HAIR_REGENERATED_VERDICT", verdict),
        "",
        reason,
        "",
        "A regenerated ablation can be run later with explicit runtime approval/parameters if the replay result warrants live confirmation.",
    ]
    write_md(OUT_ROOT / "convergence_hair_regenerated.md", lines)
    print(status_line("BG_CONVERGENCE_HAIR_REGENERATED_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

