"""Read-only extraction of Run 4 shards into a compact cache (metadata pickle + memmapped features)."""
import glob, pickle, numpy as np, torch
SRC = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/horizon_cohort2_20261003"
OUT = "/tmp/claude-1000/-home-moloch-Documents-Research-ouro-Operational-Proto-Introspection/07561c8b-32be-4f56-b8e6-e1f42d49fe0a/scratchpad/audit_opi/r4cache"
import os; os.makedirs(OUT, exist_ok=True)
NTOT, D = 4656, 3 * 4 * 2048
mm = lambda name, dt: np.lib.format.open_memmap(f"{OUT}/{name}.npy", mode="w+", dtype=dt, shape=(NTOT, D))
pend, f0, f48 = mm("pend", np.float32), mm("f0", np.float32), mm("f48", np.float32)
fo = {n: mm(f"f{n}", np.float16) for n in (16, 32, 64, 96, 128)}
GRID = (0, 16, 32, 48, 64, 96, 128)
meta, i = [], 0
KEEP = ("task_uid", "split", "proof_depth", "n_vars", "n_facts", "n_rules", "gold_answer", "gold_letter", "options", "task_prompt",
        "candidate_idx", "generated_text", "n_gen_tok", "hit_max_tokens", "prompt_tok", "gen_logprobs", "strict_answer", "labelled",
        "success", "parsed_published", "malformed_published", "success_published", "prefix_texts", "gen_ids")
for f in sorted(glob.glob(f"{SRC}/run4_shard_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    assert not d["smoke"] and max(d["equivalence_checks_maxabs"] or [0]) <= 1e-3
    for r in d["records"]:
        assert list(r["prefix_grid"]) == list(GRID)
        meta.append({k: r[k] for k in KEEP})
        pend[i] = r["prompt_end_state"].flatten().numpy(); f0[i] = r["prefix_features"][0].flatten().numpy()
        if r["labelled"]:
            f48[i] = r["prefix_features"][3].flatten().numpy()
            for n in fo:
                fo[n][i] = r["prefix_features"][GRID.index(n)].flatten().numpy().astype(np.float16)
        i += 1
    del d
    print(f, i, flush=True)
assert i == NTOT
for a in [pend, f0, f48, *fo.values()]:
    a.flush()
pickle.dump(meta, open(f"{OUT}/meta.pkl", "wb"))
print("done", i)
