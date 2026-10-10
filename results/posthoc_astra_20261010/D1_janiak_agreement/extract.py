"""Read-only extraction of Runs 4, 6 and 1 into cache/ (deleted at the end): per-answer metadata (pickle) and the
pooled features at N* for answer-free answers (float32 .npy). Shards are loaded one at a time with the sealed
loaders' asserts. Run with python -B so nothing is written next to imported modules on the SSD.

usage: python -B extract.py run4|run6|run1
"""
import glob
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = "/home/moloch/ouro_project"
SSD = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer"
R4 = ROOT + "/opi/preanswer/horizon_cohort2_20261003"
R3 = ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
sys.path[:0] = [ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual", R3, R4]

CLASSES = ("True", "False", "Unknown")
RUNS = {  # shard glob, N*, index of N* in the run's prefix grid
    "run4": (SSD + "/horizon_cohort2_20261003/run4_shard_*.pt", 48, 3),
    "run6": (SSD + "/gsm8k_train_20261003/run6_shard_*.pt", 96, 5),
    "run1": (SSD + "/fixed_prefix_rerun_20260927/run1_shard_off*.pt", 48, 3),
}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def horizon_task_lookup():
    from bg_v2_overnight_horizon_generate import LOGIC_TASKS_PATH
    out = {}
    for line in open(LOGIC_TASKS_PATH):
        d = json.loads(line)
        if d.get("symbolic") and "n_vars" in (d.get("verifier") or {}):  # only propositional tasks carry these
            out[d["task_uid"]] = (d["verifier"]["n_vars"], len(d["symbolic"]["facts"]), len(d["symbolic"]["rules"]))
    return out


def main(run):
    pattern, n_star, gi = RUNS[run]
    os.makedirs(CACHE, exist_ok=True)
    if run == "run6":
        from answer_detector_gsm8k import first_commitment, is_answer_bearing, value
    else:
        from answer_detector import is_answer_bearing, task_statement
        from gen_run4 import strict_answer
        # Run 1's sealed detector is byte-identical to Run 4's (checked here, not assumed)
        assert sha(R4 + "/answer_detector.py") == sha(SSD + "/fixed_prefix_rerun_20260927/answer_detector.py")
        tasks = horizon_task_lookup() if run == "run1" else None
    meta, H = [], []
    for f in sorted(glob.glob(pattern)):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert not d.get("smoke", False), f
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        assert list(d["prefix_grid"])[gi] == n_star, f
        for r in d["records"]:
            qid = r.get("task_uid", r.get("task_id"))
            lp = r["gen_logprobs"]
            win = lp[:n_star] or [0.0]  # as the sealed lp_at(n)
            m = {"qid": qid, "split": r["split"], "cand": int(r["candidate_idx"]), "n_gen_tok": int(r["n_gen_tok"]),
                 "hit_max": float(r["hit_max_tokens"]), "prompt_tok": int(r["prompt_tok"]),
                 "lp_full": float(np.mean(lp)) if lp else float("nan"),
                 "lp_win_mean": sum(win) / len(win), "lp_win_min": min(win), "shard": os.path.basename(f)}
            text = r["prefix_texts"][gi]
            if run == "run6":
                v = value(r["own_answer"])
                m["pred"] = None if v is None else str(v)
                m["gold"] = str(value(r["gold"]))
                m["label"] = None if v is None else int(v == value(r["gold"]))
                assert (v is not None) == r["parseable"] and (m["label"] == 1) == r["success"]
                m["free"] = not is_answer_bearing(text, r["own_answer"])
                m["in_window_commit"] = first_commitment(text) is not None
                m["qfeat"] = [float(r["prompt_tok"]), float(r["ref_steps"])]
            else:
                strict = strict_answer(r["generated_text"], r["options"])
                if run == "run4":
                    assert strict == r["strict_answer"]
                    nv, nf, nr = r["n_vars"], r["n_facts"], r["n_rules"]
                else:
                    nv, nf, nr = tasks[qid]
                m["pred"] = strict
                m["gold"] = r["gold_letter"]
                m["label"] = None if strict is None else int(strict == r["gold_letter"])
                m["free"] = not is_answer_bearing(text, task_statement(r["task_prompt"]))
                m["in_window_commit"] = strict_answer(text, r["options"]) is not None
                m["qfeat"] = [float(r["prompt_tok"]), float(r["proof_depth"]), float(nv), float(nf), float(nr)] + \
                             [float(r["gold_answer"] == c) for c in CLASSES]
            m["h_row"] = -1
            if m["free"]:
                m["h_row"] = len(H)
                H.append(r["prefix_features"][gi].flatten().float().numpy().copy())
            meta.append(m)
        del d
        print(run, os.path.basename(f), len(meta), "answers,", len(H), "answer-free", flush=True)
    np.save(os.path.join(CACHE, f"{run}_H.npy"), np.stack(H).astype(np.float32))
    pickle.dump(meta, open(os.path.join(CACHE, f"{run}_meta.pkl"), "wb"))
    tok = sum(m["prompt_tok"] + m["n_gen_tok"] for m in meta)
    print(run, "done:", len(meta), "answers;", len(H), "answer-free;", tok, "prompt+generated tokens", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
