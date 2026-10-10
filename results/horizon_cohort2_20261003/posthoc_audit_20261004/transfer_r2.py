"""Post-hoc, exploratory: hidden-only correctness probe trained on ALL Run 4 answer-free rows (N=64 or 96), applied unchanged to
Run 2's 80-token prefix features (180 disjoint depth-4 tasks, 6 samples each), strict labels, answer-free at 80 tokens by the
frozen detector. Within-question AUROC on Run 2 mixed tasks, task-clustered bootstrap."""
from rows import *
import json
from fastarm import Arm
from sklearn.model_selection import GroupKFold
from gen_run4 import strict_answer
d = torch.load("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/branching/tournament_rerun_depth4_20260927/tournament_table.pt", map_location="cpu", weights_only=False)
r2 = []
for x in d["records"]:
    opts = {"A": "True", "B": "False", "C": "Unknown"}
    a = strict_answer(x["generated_text"], opts)
    st = task_statement(x["task_prompt"])
    from transformers import AutoTokenizer
    r2.append({"task_id": x["task_uid"], "success": a == x["gold_letter"], "labelled": a is not None, "feat": x["prefix_features"].flatten().float().numpy(),
               "text80": x["generated_text"], "gen_ids": x["gen_ids"], "st": st})
tok = AutoTokenizer.from_pretrained("/home/moloch/ouro_project/shared/models/ouro_rltt_local")
for x in r2:
    x["free80"] = not is_answer_bearing(tok.decode(x["gen_ids"][:80], skip_special_tokens=True), x["st"])
del d
assert len({t for t in (x["task_id"] for x in r2)} & {r["task_uid"] for r in meta}) == 0, "task overlap"
R2 = [x for x in r2 if x["labelled"] and x["free80"]]
print("Run 2 records", len(r2), "labelled", sum(x["labelled"] for x in r2), "labelled & answer-free at 80:", len(R2), "accuracy", np.mean([x["success"] for x in R2]))
out = {}
for n in (64, 96):
    rows4 = [r for r in lab if r["free"][n]]
    H4 = np.stack([np.asarray(F[f"f{n}"][r["i"]], dtype=np.float32) for r in rows4])
    H = np.concatenate([H4, np.stack([x["feat"] for x in R2])]).astype(np.float32)
    y = np.array([int(r["success"]) for r in rows4] + [int(x["success"]) for x in R2])
    g = np.array([r["task_id"] for r in rows4] + [x["task_id"] for x in R2])
    itv, iho = np.arange(len(rows4)), np.arange(len(rows4), len(y))
    arm = Arm([H], np.zeros((len(y), 0), np.float32), y, list(GroupKFold(5).split(itv, y[itv], g[itv])), itv, iho)
    p, hp = arm.fit_score([0], False)
    from sklearn.metrics import roc_auc_score
    out[n] = {"hp": hp, "across_auroc_run2": roc_auc_score(y[iho], p), "within": A3.within_task_auroc(p, y[iho], list(g[iho]))}
    print(n, json.dumps(out[n], default=float), flush=True)
json.dump(out, open("transfer_r2.json", "w"), indent=1, default=float)
