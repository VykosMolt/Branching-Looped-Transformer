import torch, collections
d = torch.load("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/branching/tournament_rerun_depth4_20260927/tournament_table.pt", map_location="cpu", weights_only=False)
r = d["records"][0]
for k, v in r.items():
    print(k, tuple(v.shape) if torch.is_tensor(v) else (repr(v)[:70] if not isinstance(v, list) else f"list {len(v)}"))
recs = d["records"]
by = collections.defaultdict(list)
for x in recs:
    if not x["malformed"]: by[x["task_uid"]].append(x["success"])
mixed = [t for t, v in by.items() if 0 < sum(v) < len(v)]
print("records", len(recs), "parseable", sum(not x["malformed"] for x in recs), "mixed tasks", len(mixed), "acc", sum(x["success"] for x in recs if not x["malformed"]) / sum(not x["malformed"] for x in recs))
print(collections.Counter(x["split"] for x in recs))
