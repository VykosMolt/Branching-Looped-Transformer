"""Post-hoc (NOT preregistered), 2026-09-27: controls requested after the +0.243 question.
1. Final-answer line with difficulty in the baseline and the fixed combined model, on every Horizon cohort.
2. 48-token cohort: within-question AUROC, and a mean-pooled-prompt control (interim; the end-of-prompt
   state needs a GPU pass).
3. Verdict dissociation: on wrong answers, does the 48-token state predict the model's own (wrong) verdict or
   the gold one? Across samples of one question that end in different verdicts, does it track each sample's own?
Uses the fixed combined model and within-question AUROC from the amended Run 3 analysis, unchanged."""
import glob, json, sys
import numpy as np
import torch

ROOT = "/home/moloch/ouro_project"
R1 = ROOT + "/opi/preanswer/fixed_prefix_rerun_20260927"
R3 = ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"
ARCH = ("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260729/historical_experiments/paper_v3_audit_evidence/"
        "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic/horizon_generation_main_off0.pt")
sys.path[:0] = [R1, R3]
import analyze_run1 as A1  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from bg_v2_overnight_horizon_analysis import SEED, shortcut_vec  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

out = {}
diff = lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])])


def rows_of(paths):
    rs = [r for f in paths for r in torch.load(f, map_location="cpu", weights_only=False)["records"]]
    for r in rs:
        r["task_id"] = r["task_uid"]
    return [r for r in rs if not r["malformed"]]


def brief(res):
    if "incremental_auroc_combined_minus_shortcut" not in res:
        return res
    b = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    return {k: round(res[k], 3) for k in ("auroc_shortcut_only", "auroc_hidden_only", "auroc_hidden_plus_shortcuts",
                                          "incremental_auroc_combined_minus_shortcut")} | {
        "ci95": [round(b[0], 3), round(b[1], 3)], "n_heldout": res["n_heldout"], "neg": res["n_heldout_negatives"]}


# 1. final-answer line, four cohorts
first = rows_of([ARCH])
new = rows_of(sorted(glob.glob(ROOT + "/opi/preanswer/horizon_power_v3_20260726/horizon_logic/horizon_generation_v3_off*.pt")))
recs1 = A1.load()
for r in recs1:
    r["task_id"] = r["task_uid"]
A1.answer_free_table(recs1)
fresh = [r for r in recs1 if not r["malformed"]]
pre = lambda r: r["preanswer_features"]
for name, rows in (("first170", first), ("new510", new), ("pooled680", first + new), ("fresh599", fresh)):
    out[f"final_line_{name}"] = {
        "sealed_arm_4_shortcuts": brief(A3.guarded(name, rows, pre, shortcut_vec)[0]),
        "fixed_model_4_shortcuts": brief(A3.proper_arm(name, rows, [pre], shortcut_vec)[0]),
        "fixed_model_4_shortcuts_plus_difficulty": brief(A3.proper_arm(name, rows, [pre], lambda r: torch.cat([shortcut_vec(r), diff(r)]))[0])}
    print(name, json.dumps(out[f"final_line_{name}"]), flush=True)

# 2. 48-token cohort
N = 48
P = [r for r in fresh if r["free"][N]]
S = lambda r: torch.cat([A1.prefix_lp_vec(N)(r), diff(r)])
res, sc = A3.proper_arm("primary_difficulty", P, [A1.feat_at(N)], S)
out["primary_fixed_model_with_difficulty"] = brief(res)
out["within_question_heldout"] = {k: A3.within_task_auroc(sc[k].numpy(), sc["y"].numpy(), sc["tasks"]) for k in ("combined", "hidden", "shortcut")}
out["interim_prompt_control_meanpooled"] = brief(A3.proper_arm("prompt_meanpool_control", P, [A1.feat_at(N)], S, base=[A1.feat_at(0)])[0])
print("within-question (held-out):", json.dumps(out["within_question_heldout"]), flush=True)
print("interim prompt control:", json.dumps(out["interim_prompt_control_meanpooled"]), flush=True)

# out-of-fold scores over all splits (more questions): outer 5-fold by question, inner selection as in proper_arm
y = np.array([int(r["success"]) for r in P]); g = np.array([r["task_id"] for r in P])
H = np.stack([A1.feat_at(N)(r).flatten().float().numpy() for r in P])
Sm = np.stack([S(r).float().numpy() for r in P])
oof = {"combined": np.zeros(len(P)), "hidden": np.zeros(len(P)), "shortcut": np.zeros(len(P))}
for a_, b_ in GroupKFold(5).split(H, y, g):
    tv = np.zeros(len(P), bool); tv[a_] = True
    oof["combined"][b_] = A3._fit_score([H], Sm, y, g, tv, ~tv)[0]
    oof["hidden"][b_] = A3._fit_score([H], np.zeros((len(P), 0)), y, g, tv, ~tv)[0]
    oof["shortcut"][b_] = A3._fit_score([], Sm, y, g, tv, ~tv)[0]
out["within_question_oof_all_splits"] = {k: A3.within_task_auroc(v, y, list(g)) for k, v in oof.items()}
print("within-question (OOF, all splits):", json.dumps(out["within_question_oof_all_splits"]), flush=True)

# 3. verdict dissociation, out-of-fold by question
V = ["True", "False", "Unknown"]
norm = lambda t: t.strip().rstrip(".").capitalize()
own = np.array([V.index(norm(r["options"][r["parsed_answer"]])) for r in P])
gold = np.array([V.index(norm(r["options"][r["gold_letter"]])) for r in P])
rng = np.random.default_rng(SEED)


def oof_verdict(train_mask):
    pr = np.zeros((len(P), 3))
    for a_, b_ in GroupKFold(5).split(H, own, g):
        tr = np.intersect1d(a_, np.where(train_mask)[0])
        m = make_pipeline(StandardScaler(), PCA(24, random_state=0), LogisticRegression(C=0.1, max_iter=5000)).fit(H[tr], own[tr])
        p = np.zeros((len(b_), 3)); p[:, m.classes_] = m.predict_proba(H[b_]); pr[b_] = p
    return pr


def boot(stat_fn, tasks):
    u = np.unique(tasks); ix = {t: np.where(tasks == t)[0] for t in u}
    vals = sorted(stat_fn(np.concatenate([ix[t] for t in rng.choice(u, len(u))])) for _ in range(2000))
    return [round(vals[50], 3), round(vals[1949], 3)]


wrong = np.where(y == 0)[0]
for label, mask in (("trained_on_all", np.ones(len(P), bool)), ("trained_on_correct_only", y == 1)):
    pr = oof_verdict(mask)
    pw, ow, gw, tw = pr[wrong], own[wrong], gold[wrong], g[wrong]
    acc_own = lambda ix: float((pw[ix].argmax(1) == ow[ix]).mean())
    acc_gold = lambda ix: float((pw[ix].argmax(1) == gw[ix]).mean())
    allix = np.arange(len(wrong))
    out[f"verdict_on_wrong_answers_{label}"] = {
        "n_wrong": len(wrong), "argmax_is_own_verdict": round(acc_own(allix), 3), "ci_own": boot(acc_own, tw),
        "argmax_is_gold_verdict": round(acc_gold(allix), 3), "ci_gold": boot(acc_gold, tw),
        "mean_p_own": round(float(pw[allix, ow].mean()), 3), "mean_p_gold": round(float(pw[allix, gw].mean()), 3)}
    print("wrong answers,", label, json.dumps(out[f"verdict_on_wrong_answers_{label}"]), flush=True)
    if label == "trained_on_all":
        pairs = [(i, j) for t in np.unique(g) for i in np.where(g == t)[0] for j in np.where(g == t)[0] if i < j and own[i] != own[j]]
        tr_ = lambda ps: np.mean([(pr[i, own[i]] > pr[j, own[i]]) + (pr[j, own[j]] > pr[i, own[j]]) for i, j in ps]) / 2
        by = {}
        for i, j in pairs:
            by.setdefault(g[i], []).append((i, j))
        ts = sorted(by)
        vals = sorted(tr_([p for t in rng.choice(ts, len(ts)) for p in by[t]]) for _ in range(2000))
        out["verdict_siblings_different_verdicts"] = {"n_questions": len(ts), "n_pairs": len(pairs),
                                                      "share_tracking_own_verdict": round(float(tr_(pairs)), 3),
                                                      "ci95": [round(vals[50], 3), round(vals[1949], 3)]}
        print("siblings with different verdicts:", json.dumps(out["verdict_siblings_different_verdicts"]), flush=True)

json.dump(out, open(R1 + "/posthoc_20260927/controls.json", "w"), indent=2, default=float)
print("saved")
