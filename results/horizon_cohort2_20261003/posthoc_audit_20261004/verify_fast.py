from rows import *
from fastarm import Arm
from sklearn.model_selection import GroupKFold
from proto_introspection_controls_analysis import auroc
rows = primary
y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
tv = np.array([r["split"] in ("train", "val") for r in rows]); itv, iho = np.where(tv)[0], np.where(~tv)[0]
S = np.stack([S48(r).float().numpy() for r in rows])
H = np.stack([np.asarray(F["f48"][r["i"]], dtype=np.float32) for r in rows])
PE = np.stack([np.asarray(F["pend"][r["i"]], dtype=np.float32) for r in rows])
folds = list(GroupKFold(5).split(itv, y[itv], g[itv]))
arm = Arm([H, PE], S, y, folds, itv, iho)
A = lambda p: auroc(torch.tensor(p), torch.tensor(y[iho]).double())
pb, hb = arm.fit_score([], True); pc, hc = arm.fit_score([0], True); ph, hh = arm.fit_score([0], False)
print("laptop folds, fast impl:", hb, round(A(pb), 4), hc, round(A(pc), 4), hh, round(A(ph), 4))
# force the pod's selections
pb2, _ = arm.fit_score([], True, force=((), 0.1)); pc2, _ = arm.fit_score([0], True, force=((24,), 1.0))
print("pod configs forced: baseline %.10f combined %.10f (published 0.6809246816, 0.7342041922)" % (A(pb2), A(pc2)))
pb3, _ = arm.fit_score([1], True, force=((16,), 0.1)); pc3, _ = arm.fit_score([1, 0], True, force=((16, 16), 0.1))
print("pod pend configs forced: baseline %.10f combined %.10f (published 0.7092066994, 0.7580734129)" % (A(pb3), A(pc3)))
