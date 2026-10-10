"""Post-hoc (NOT preregistered), 2026-09-27: the end-of-prompt control on Horizon Run 1, in the form Run 3
prespecifies (amendment 1), plus a scalar sensitivity version."""
import json, sys
import numpy as np
import torch

ROOT = "/home/moloch/ouro_project"
R1 = ROOT + "/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path[:0] = [R1, ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"]
import analyze_run1 as A1  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

PE = torch.load(R1 + "/posthoc_20260927/prompt_end_states.pt")
recs = A1.load()
for r in recs:
    r["task_id"] = r["task_uid"]
    r["prompt_end_state"] = PE[r["task_uid"]]
A1.answer_free_table(recs)
N = 48
parse = [r for r in recs if not r["malformed"]]
P = [r for r in parse if r["free"][N]]
diff = lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])])
S = lambda r: torch.cat([A1.prefix_lp_vec(N)(r), diff(r)])
pend = lambda r: r["prompt_end_state"]
out = {}


def brief(res):
    b = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    return {k: round(res[k], 3) for k in ("auroc_shortcut_only", "auroc_hidden_only", "auroc_hidden_plus_shortcuts",
                                          "incremental_auroc_combined_minus_shortcut")} | {
        "ci95": [round(b[0], 3), round(b[1], 3)], "hyperparams": res["hyperparams"]}


# 1. prespecified form (Run 3 amendment 1): baseline = PCA(end-of-prompt state) + S; full adds PCA(H at N=48)
out["control_prompt_end_block"] = brief(A3.proper_arm("prompt_end_control", P, [A1.feat_at(N)], S, base=[pend])[0])
# 2. end-of-prompt state alone against prompt tokens + proof depth (all parseable answers, and the 48-token cohort)
out["prompt_end_vs_difficulty_parseable"] = brief(A3.proper_arm("prompt_end_only", parse, [pend], diff)[0])
out["prompt_end_vs_difficulty_primary_cohort"] = brief(A3.proper_arm("prompt_end_only_P", P, [pend], diff)[0])
# 3. sensitivity: the end-of-prompt probe as one cross-fitted score in S
y = np.array([int(r["success"]) for r in P]); g = np.array([r["task_id"] for r in P])
tv = np.array([r["split"] in ("train", "val") for r in P]); ho = ~tv
E = np.stack([pend(r).flatten().float().numpy() for r in P]); none = np.zeros((len(P), 0))
score = np.zeros(len(P))
itv = np.where(tv)[0]
for a_, b_ in GroupKFold(5).split(itv, y[itv], g[itv]):
    m = np.zeros(len(P), bool); m[itv[a_]] = True
    t = np.zeros(len(P), bool); t[itv[b_]] = True
    score[itv[b_]] = A3._fit_score([E], none, y, g, m, t)[0]
score[ho] = A3._fit_score([E], none, y, g, tv, ho)[0]
sidx = {id(r): s for r, s in zip(P, score)}
S2 = lambda r: torch.cat([S(r), torch.tensor([float(sidx[id(r)])])])
out["control_prompt_end_scalar"] = brief(A3.proper_arm("prompt_end_scalar_control", P, [A1.feat_at(N)], S2)[0])
json.dump(out, open(R1 + "/posthoc_20260927/prompt_control.json", "w"), indent=2, default=float)
for k, v in out.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "hyperparams"}, "| hp:", json.dumps(v["hyperparams"]))
