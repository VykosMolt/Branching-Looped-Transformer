"""Prompt-only probe on the Horizon replication set (see PLAN.md in this directory).

Reuses the published analysis code unchanged; only the feature input differs.
Writes only into this directory.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
MANUAL = PROJECT_ROOT / "shared/utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    sys.path.insert(0, p)

from bg_v2_overnight_horizon_analysis import shortcut_vec  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402
from bg_v2_malformed_sibling_control import arm  # noqa: E402
from bg_v3_horizon_power_analysis import load_records, run_arm  # noqa: E402

V3 = PROJECT_ROOT / "opi/preanswer/horizon_power_v3_20260726"
records = load_records(V3 / "horizon_logic")
assert len(records) == 2040, len(records)
assert all(r["split"] == split_for(r["task_uid"]) for r in records)

# Gate: reproduce the published new-only 4-shortcut arm exactly.
published = json.loads((V3 / "auroc_results_v3.json").read_text())["new_only_4sc"]
gate, _ = run_arm("gate_new_only_4sc", records, shortcut_vec)
for key in ("auroc_shortcut_only", "auroc_hidden_only", "auroc_hidden_plus_shortcuts",
            "incremental_auroc_combined_minus_shortcut"):
    if abs(gate[key] - published[key]) > 1e-9:
        raise SystemExit(f"GATE FAILED on {key}: {gate[key]!r} vs {published[key]!r}")
print("GATE PASSED: published new_only_4sc reproduced to 1e-9")

# Prompt-only arm.
scorable = [r for r in records if not r["malformed"]]
Xh = torch.stack([r["prompt_only_features"].flatten().float() for r in scorable])
Xb = torch.tensor([[r["prompt_tok"], float(r["proof_depth"])] for r in scorable])
y = torch.tensor([1.0 if r["success"] else 0.0 for r in scorable])
tasks = [r["task_uid"] for r in scorable]
tv = torch.tensor([r["split"] in ("train", "val") for r in scorable])
ho = torch.tensor([r["split"] == "heldout" for r in scorable])
t_tv = [t for t, m in zip(tasks, tv.tolist()) if m]
t_ho = [t for t, m in zip(tasks, ho.tolist()) if m]
res = arm("prompt_only", Xh[tv], Xb[tv], y[tv], t_tv, Xh[ho], Xb[ho], y[ho], t_ho)
scores = res.pop("_scores")
res.update({
    "baseline_features": ["prompt_tok", "proof_depth"],
    "hidden_features": "prompt_only_features flattened (layers 24/36/47 x 4 loops x 2048)",
    "n_heldout": int(ho.sum()), "n_heldout_tasks": len(set(t_ho)),
    "n_heldout_negatives": int((y[ho] == 0).sum()), "heldout_base_rate": float(y[ho].mean()),
    "gate": "PASSED",
})
(HERE / "results.json").write_text(json.dumps(res, indent=2))
(HERE / "heldout_predictions.json").write_text(json.dumps(
    {"tasks": t_ho, "y": y[ho].tolist(), "scores": {k: v.tolist() for k, v in scores.items()}}))
print(json.dumps({k: v for k, v in res.items() if k != "hyperparams"}, indent=2))
