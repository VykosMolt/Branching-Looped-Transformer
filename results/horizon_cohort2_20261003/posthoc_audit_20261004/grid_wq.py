"""Post-hoc: held-out within-question AUROC of the hidden-only probe (sealed selection) at every cut."""
from rows import *
import json
from fastarm import Arm
from sklearn.model_selection import GroupKFold
out = {}
for n in (16, 32, 48, 64, 96, 128):
    rows = [r for r in lab if r["free"][n]]
    y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
    tv = np.array([r["split"] in ("train", "val") for r in rows]); itv, iho = np.where(tv)[0], np.where(~tv)[0]
    H = np.stack([np.asarray(F[f"f{n}"][r["i"]], dtype=np.float32) for r in rows])
    Sn = np.stack([torch.cat([lp_at(n)(r), s_diff(r)]).float().numpy() for r in rows])
    arm = Arm([H], Sn, y, list(GroupKFold(5).split(itv, y[itv], g[itv])), itv, iho)
    ph, hh = arm.fit_score([0], False); pc, hc = arm.fit_score([0], True); pb, hb = arm.fit_score([], True)
    out[n] = {k: A3.within_task_auroc(v, y[iho], list(g[iho])) for k, v in (("hidden", ph), ("combined", pc), ("surface", pb))}
    print(n, json.dumps({k: (round(v["estimate"], 3), [round(x, 3) for x in v["ci95"]], v["n_pairs"], v["n_mixed_tasks"]) for k, v in out[n].items()}), flush=True)
json.dump(out, open("grid_wq.json", "w"), indent=1)
