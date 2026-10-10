"""Post-hoc checks on Run 4 (laptop GroupKFold folds, sealed estimator math via fastarm)."""
from rows import *
import json, pickle
from fastarm import Arm
from sklearn.model_selection import GroupKFold
from proto_introspection_controls_analysis import auroc
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered
T = lambda p: torch.tensor(p, dtype=torch.double)
out = {}
def setup(rows, blocks_fns, Sfn):
    y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
    tv = np.array([r["split"] in ("train", "val") for r in rows]); itv, iho = np.where(tv)[0], np.where(~tv)[0]
    S = np.stack([Sfn(r).float().numpy() for r in rows]) if Sfn else np.zeros((len(rows), 0), np.float32)
    B = [np.stack([np.asarray(F[k][r["i"]], dtype=np.float32) for r in rows]) for k in blocks_fns]
    folds = list(GroupKFold(5).split(itv, y[itv], g[itv]))
    return Arm(B, S, y, folds, itv, iho), y, g, iho
def inc(arm, y, g, iho, base, full, label):
    yh, th = torch.tensor(y[iho]).double(), list(g[iho])
    pb, hb = arm.fit_score(base, True); pc, hc = arm.fit_score(full, True)
    b = bootstrap_task_clustered(T(pc), T(pb), yh, th, rounds=2000)["ci95"]
    r = {"base": round(auroc(T(pb), yh), 4), "full": round(auroc(T(pc), yh), 4), "inc": round(auroc(T(pc), yh) - auroc(T(pb), yh), 4),
         "ci": [round(b[0], 4), round(b[1], 4)], "hp": [hb, hc], "n_ho": len(iho), "neg": int((y[iho] == 0).sum())}
    out[label] = r; print(label, json.dumps(r), flush=True); return pb, pc
# 1. pooled-prompt (N=0) control: blocks [f48, pend, f0]
arm, y, g, iho = setup(primary, ["f48", "pend", "f0"], S48)
inc(arm, y, g, iho, [], [0], "primary_laptop_folds")
inc(arm, y, g, iho, [1], [1, 0], "pend_control_laptop_folds")
inc(arm, y, g, iho, [2], [2, 0], "pooled_prompt_control")
inc(arm, y, g, iho, [1, 2], [1, 2, 0], "pend_plus_pooled_prompt_control")
# 2. apples-to-apples for 0.153: proper estimator with S = log-prob only
arm2, y2, g2, iho2 = setup(primary, ["f48"], lp_at(48))
inc(arm2, y2, g2, iho2, [], [0], "proper_estimator_lp_only")
# 3. exclude the 9 answers whose label sits inside the 48-token window
bad = set(pickle.load(open("inwindow.pkl", "rb"))[48])
prim_x = [r for r in primary if (r["task_uid"], r["candidate_idx"]) not in bad]
arm3, y3, g3, iho3 = setup(prim_x, ["f48"], S48)
pb, pc = inc(arm3, y3, g3, iho3, [], [0], "primary_excl_inwindow")
ph, hh = arm3.fit_score([0], False)
out["wq_excl_inwindow"] = {k: A3.within_task_auroc(v, y3[iho3], list(g3[iho3])) for k, v in (("hidden", ph), ("combined", pc), ("surface", pb))}
print("wq_excl_inwindow", json.dumps(out["wq_excl_inwindow"]), flush=True)
json.dump(out, open("checks2.json", "w"), indent=1, default=str)
