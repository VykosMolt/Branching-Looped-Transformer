import torch
d = torch.load("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/horizon_cohort2_20261003/run4_shard_1100.pt", map_location="cpu", weights_only=False)
r = d["records"][0]
for k, v in r.items():
    if torch.is_tensor(v): print(k, tuple(v.shape), v.dtype, float(v.abs().max()))
    elif isinstance(v, list): print(k, "list", len(v), type(v[0]) if v else None)
    else: print(k, repr(v)[:80])
print(len(d["records"]), d["equivalence_checks_maxabs"][:5])
