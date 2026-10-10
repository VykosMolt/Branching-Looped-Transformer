import glob, pickle, collections, sys, torch
sys.path[:0] = ["/home/moloch/ouro_project/opi/preanswer/horizon_cohort2_20261003", "/home/moloch/ouro_project", "/home/moloch/ouro_project/shared", "/home/moloch/ouro_project/shared/utilities/tests/manual"]
from answer_detector import is_answer_bearing, task_statement
from gen_run4 import strict_answer
R1 = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927"
meta = []
for f in sorted(glob.glob(R1 + "/run1_shard_off*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    for r in d["records"]:
        meta.append({k: r[k] for k in ("task_uid", "candidate_idx", "split", "malformed", "success", "parsed_answer", "gold_letter", "options",
                                        "generated_text", "prefix_texts", "task_prompt", "gen_ids", "prompt_ids")})
    del d; print(f, len(meta), flush=True)
pickle.dump(meta, open("r1_meta.pkl", "wb"))
prim = [r for r in meta if not r["malformed"] and not is_answer_bearing(r["prefix_texts"][3], task_statement(r["task_prompt"]))]
inw = [r for r in prim if strict_answer(r["prefix_texts"][3], r["options"]) is not None]
print("Run 1 primary", len(prim), "label-like answer inside 48-token window", len(inw), collections.Counter((r["split"], r["success"]) for r in inw))
print("of these, in-window answer == published label:", sum(strict_answer(r["prefix_texts"][3], r["options"]) == r["parsed_answer"] for r in inw))
