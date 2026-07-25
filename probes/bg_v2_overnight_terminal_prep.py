"""Paper 1 v2 overnight programme -- Part II prep: add full-candidate terminal features.

Terminal selection scores COMPLETED candidates, so it needs a pooled hidden-state
feature over the full (prompt + generated text), not the pre-answer cut used for Part I.
This script loads the already-generated Horizon Logic pool (bg_v2_overnight_horizon_generate.py
outputs), adds a `full_features` tensor per candidate via one more forward pass through the
same canonical extractor, and writes a consolidated pool file. It does not regenerate any
text and does not touch the original per-shard generation files.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))

from bg_steering_suite_lib import mcq_prompt  # noqa: E402

HORIZON_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic"
TERMINAL_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724/terminal_selection"
LOGIC_TASKS_PATH = PROJECT_ROOT / "data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl"


def load_prompt_lookup(task_uids: set[str]) -> dict[str, str]:
    """task_uid -> task_prompt, looked up from the same deterministic source pool used at
    generation time (records themselves do not store the raw prompt, only options/gold)."""
    import json
    out = {}
    with open(LOGIC_TASKS_PATH) as f:
        for line in f:
            d = json.loads(line)
            uid = d.get("task_uid")
            if uid in task_uids:
                out[uid] = d["task_prompt"]
                if len(out) == len(task_uids):
                    break
    return out


def main() -> None:
    TERMINAL_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(glob.glob(str(HORIZON_ROOT / "horizon_generation_*.pt"))):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
    print(f"[terminal-prep] loaded {len(records)} candidate records")
    if not records:
        print("[terminal-prep] no records found -- aborting")
        return

    prompt_lookup = load_prompt_lookup({r["task_uid"] for r in records})
    missing = [r["task_uid"] for r in records if r["task_uid"] not in prompt_lookup]
    if missing:
        print(f"[terminal-prep] WARNING: {len(missing)} records have no prompt lookup match "
              f"(e.g. {missing[:3]}); these will be skipped")

    t0 = time.time()
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    print(f"[terminal-prep] model loaded in {time.time()-t0:.1f}s")

    out_records = []
    errors = []
    for i, r in enumerate(records):
        task_prompt = prompt_lookup.get(r["task_uid"])
        if task_prompt is None:
            errors.append(f"{r['task_uid']}/{r['candidate_idx']}: no prompt lookup match")
            continue
        prompt = mcq_prompt(task_prompt, r["options"])
        full_text = f"{prompt}\n{r['generated_text']}".strip()
        try:
            feats = extractor.encode_text_to_pooled_features(full_text)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{r['task_uid']}/{r['candidate_idx']}: {exc}")
            continue
        rr = dict(r)
        rr["full_features"] = feats
        out_records.append(rr)
        if (i + 1) % 50 == 0:
            print(f"[terminal-prep] {i+1}/{len(records)} done, elapsed={time.time()-t0:.1f}s")

    extractor.cleanup()
    out_path = TERMINAL_ROOT / "terminal_pool.pt"
    torch.save({"records": out_records, "errors": errors, "elapsed_seconds": round(time.time() - t0, 1)}, out_path)
    print(f"[terminal-prep] wrote {out_path} ({len(out_records)} records, {len(errors)} errors)")


if __name__ == "__main__":
    main()
