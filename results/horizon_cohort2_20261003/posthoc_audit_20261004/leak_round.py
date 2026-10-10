from common import *
from collections import defaultdict
by = defaultdict(list)
for r in meta: by[r["task_uid"]].append(r["i"])
P, Z = F["pend"], F["f0"]
wq_p, wq_z, bq_p, bq_z, ident_p, ident_z = [], [], [], [], 0, 0
for t, ix in by.items():
    a, b = np.asarray(P[ix]), np.asarray(Z[ix])
    wq_p.append(np.abs(a - a[0]).max()); wq_z.append(np.abs(b - b[0]).max())
    ident_p += int(np.all(a == a[0])); ident_z += int(np.all(b == b[0]))
print("questions", len(by), "pend bit-identical within question:", ident_p, " f0 bit-identical:", ident_z)
print("max within-question |diff| pend: median %.4g max %.4g" % (np.median(wq_p), np.max(wq_p)))
print("max within-question |diff| f0  : median %.4g max %.4g" % (np.median(wq_z), np.max(wq_z)))
# scale: between-question SD per dim (median over dims)
print("between-question per-dim SD, median over dims: pend %.4g  f0 %.4g" % (np.median(np.asarray(P[::4]).std(0)), np.median(np.asarray(Z[::4]).std(0))))
# within-question per-dim SD vs between
idx = np.array([r["i"] for r in meta])
wsd_p = np.sqrt(np.mean([np.asarray(P[ix]).var(0) for ix in list(by.values())[:300]], 0))
print("within-question per-dim SD, median over dims: pend %.4g" % np.median(wsd_p))
# does within-question variation depend on total length?
r0 = meta[0]
