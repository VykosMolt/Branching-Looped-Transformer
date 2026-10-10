"""Read-only extraction of the C5 cohorts from the Run 1 and Run 4 shards, one shard at a time.

Writes data/answers.jsonl.gz (texts, labels, ids, splits, folds; small) and, for the CPU probe reproduction
only, <feat_dir>/{r1,r4}.npz (24,576-dim features at N=48 plus the probes' surface features; ~0.4 GB, not kept).
Cohorts:
  r1  Run 1, published labels: parseable (published parser) and answer-free at 48 tokens (analyze_run1, sealed)
  r1s Run 1, strict labels: strict_answer() not None and answer-free at 48 (strict_labels_run1_final_rule.py)
  r4  Run 4: labelled (strict) and answer-free at 48 (analyze_run4, sealed)
"""
import argparse
import glob
import gzip
import json
import sys

from common import DATA, R1, R4, R4CODE, ROOT, fold_of, lib_paths, tfidf_text

sys.dont_write_bytecode = True
lib_paths()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from bg_steering_suite_lib import mcq_prompt  # noqa: E402

sys.path.insert(0, R4CODE)
from answer_detector import is_answer_bearing, task_statement  # noqa: E402  (same file, same hash, in both runs)
from gen_run4 import strict_answer  # noqa: E402

N, I48 = 48, 3  # PREFIX_GRID = (0, 16, 32, 48, 64, 96, 128)
CLASSES = ("True", "False", "Unknown")


def free48(r):
    return not is_answer_bearing(r["prefix_texts"][I48], task_statement(r["task_prompt"]))


def lp48(r):
    lp = r["gen_logprobs"][:N] or [0.0]
    return [sum(lp) / len(lp), min(lp)]


def s_diff(r):  # analyze_run4.s_diff
    return [float(r["prompt_tok"]), float(r["proof_depth"]), float(r["n_vars"]), float(r["n_facts"]),
            float(r["n_rules"])] + [float(r["gold_answer"] == c) for c in CLASSES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True, help="where the large feature files go (scratch)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT + "/shared/models/ouro_rltt_local")
    rows, decode_checked = [], 0
    for run, pattern in (("r1", f"{R1}/run1_shard_off*.pt"), ("r4", f"{R4}/run4_shard_*.pt")):
        H, LP, SD, k = [], [], [], 0
        for f in sorted(glob.glob(pattern)):
            d = torch.load(f, map_location="cpu", weights_only=False)
            assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
            for r in d["records"]:
                k += 1
                assert list(r.get("prefix_grid", (0, 16, 32, 48, 64, 96, 128))) == [0, 16, 32, 48, 64, 96, 128]
                if not free48(r):
                    continue
                prompt = mcq_prompt(r["task_prompt"], r["options"])
                if run == "r1":
                    strict = strict_answer(r["generated_text"], r["options"])
                    labs = {"r1": None if r["malformed"] else int(r["success"]),
                            "r1s": None if strict is None else int(strict == r["gold_letter"])}
                else:
                    labs = {"r4": int(r["success"]) if r["labelled"] else None}
                labs = {c: v for c, v in labs.items() if v is not None}
                if not labs:
                    continue
                if k % 25 == 0:  # independent reconstruction from the saved token ids
                    assert tok.decode(r["gen_ids"][:N], skip_special_tokens=True) == r["prefix_texts"][I48]
                    assert tok.decode(r["prompt_ids"], skip_special_tokens=True) == prompt
                    decode_checked += 1
                fi = len(H)
                H.append(r["prefix_features"][I48].flatten().float().numpy())
                LP.append(lp48(r))
                SD.append(s_diff(r) if run == "r4" else [float(r["prompt_tok"]), float(r["proof_depth"])])
                for c, y in labs.items():
                    rows.append({"cohort": c, "key": f"{r['task_uid']}#{r['candidate_idx']}", "task_uid": r["task_uid"],
                                 "candidate_idx": r["candidate_idx"], "split": r["split"], "fold": fold_of(r["task_uid"]),
                                 "y": y, "feat_idx": fi, "text": tfidf_text(prompt, r["prefix_texts"][I48])})
            del d
            print(f, k, len(H), flush=True)
        np.savez(f"{a.feat_dir}/{run}.npz", H=np.stack(H), lp=np.array(LP), sdiff=np.array(SD))
        del H
    DATA.mkdir(exist_ok=True)
    with gzip.open(DATA / "answers.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("rows", len(rows), "decode checks passed", decode_checked)


if __name__ == "__main__":
    main()
