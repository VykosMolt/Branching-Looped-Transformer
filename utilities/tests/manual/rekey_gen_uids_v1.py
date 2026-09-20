"""One-time re-key of already-generated math/reasoning branch groups to the unique-by-prompt
uid scheme (fixes the hendrycks_math row-index collision). Idempotent. Run AFTER generation
finishes, BEFORE Part K. Coding/logic uids are already unique and left untouched.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B  # noqa: E402

GD = B.GEN_DIR
HASH_SUFFIX = re.compile(r"::[0-9a-f]{8}$")  # already re-keyed -> skip (idempotent)


def main() -> int:
    if any("gen" in c for c in os.popen("ps -eo cmd").read().split("\n") if "generate_branch_pools_v1.py" in c):
        print("REKEY_ABORTED: generation still running — run after it completes."); return 1
    mp = GD / "gen_manifest.json"
    m = json.loads(mp.read_text())
    n_rekeyed = 0
    all_uids = set()
    for s in m["shards"]:
        fp = GD / s["file"]
        if not fp.exists():
            continue
        groups = [json.loads(l) for l in open(fp)]
        changed = False
        for g in groups:
            if g["domain"] in ("math", "reasoning") and not HASH_SUFFIX.search(g["task_id"]):
                prompt = g.get("task_prompt", "")
                new = f"{g['task_id']}::{B.v2.text_hash(prompt)[:8]}"
                g["task_id"] = new
                g["group_id"] = f"branch_v1_{g['split']}_{g['domain']}_{g['dataset']}_{B.L._sid(new) % 10**8}"
                n_rekeyed += 1
                changed = True
            all_uids.add(g["task_id"])
        if changed:
            with open(fp, "w") as f:
                for g in groups:
                    f.write(json.dumps(g, default=B.v2.json_default) + "\n")
    # rebuild completed_task_uids from the (now-unique) flushed group ids + keep any non-group completed
    m["completed_task_uids"] = sorted(all_uids)
    mp.write_text(json.dumps(m, default=B.v2.json_default))
    print(f"REKEY_DONE: re-keyed {n_rekeyed} math/reasoning groups; {len(all_uids)} flushed groups now have "
          f"{len(all_uids)} unique uids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
