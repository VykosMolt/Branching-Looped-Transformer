"""Post hoc (2026-10-18): Run 8's Thinking-minus-RLTT interval with independent pairing.

within_oof.unpaired_diff zips the two sides' *sorted* bootstrap replicates, pairing the k-th smallest of one with the
k-th smallest of the other; that comonotone pairing narrows the interval. This script rebuilds the same out-of-fold
scores with Run 8's own code, redraws the same replicates in the same RNG order, checks that sorting them reproduces
the sealed interval exactly, and reports the interval from the draws paired in draw order (independent)."""
import json, sys
from pathlib import Path
import numpy as np

R8 = Path("/home/moloch/ouro_project/opi/preanswer/horizon_thinking_20261010")
SSD = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer")
sys.path.insert(0, str(R8))
import analyze_run8 as A  # noqa: E402  (sealed; imports within_oof as A.W)
W = A.W


def draws(x, rng):
    mixed, pairs = W._pairs(np.asarray(x["y"]), list(x["tasks"]))
    w = {t: (sum(W._win(x["s"], p, n) for p, n in pairs[t]), len(pairs[t])) for t in mixed}
    stat = lambda ts: sum(w[t][0] for t in ts) / sum(w[t][1] for t in ts)
    return stat(mixed), np.array([stat(rng.choice(mixed, len(mixed))) for _ in range(W.ROUNDS)])


def both(a, b):
    rng = np.random.default_rng(W.SEED)  # one stream, sides drawn in the sealed order
    (ea, da), (eb, db) = draws(a, rng), draws(b, rng)
    q = lambda d: [float(np.sort(d)[int(0.025 * W.ROUNDS)]), float(np.sort(d)[int(0.975 * W.ROUNDS)])]
    return {"diff": ea - eb, "a": ea, "b": eb, "ci95_sealed_sorted_pairing": q(np.sort(da) - np.sort(db)),
            "ci95_independent_pairing": q(da - db), "ci95_a": q(da), "ci95_b": q(db)}


recs, _ = A.load(SSD / "horizon_thinking_20261010")
A.answer_free_table(recs)
n_star, _ = A.choose_n_star(recs)
lab = [r for r in recs if r["labelled"]]
r4, _ = A.load(SSD / "horizon_cohort2_20261003", "run4")
A.answer_free_table(r4)
sealed = json.loads((R8 / "results.json").read_text())["secondary_rltt_comparison"]
out = {"n_star": n_star}
for name, nt, nr in (("own_n_star", n_star, A.RLTT_N_STAR), ("both_at_48", A.RLTT_N_STAR, A.RLTT_N_STAR)):
    st, _, yt, tt = A.oof_within([r for r in lab if r["free"][nt]], nt, arms=("hidden",))
    sr, _, yr, tr = A.oof_within([r for r in r4 if r["labelled"] and r["free"][nr]], nr, arms=("hidden",))
    res = both({"s": st["hidden"], "y": yt, "tasks": tt}, {"s": sr["hidden"], "y": yr, "tasks": tr})
    s = sealed[name]["thinking_minus_rltt"]
    assert abs(res["diff"] - s["diff"]) < 1e-12, (res["diff"], s["diff"])
    assert np.allclose(res["ci95_sealed_sorted_pairing"], s["ci95"], atol=1e-12), (res["ci95_sealed_sorted_pairing"], s["ci95"])
    out[name] = res
    print(name, json.dumps(res))
(Path(__file__).parent / "results.json").write_text(json.dumps(out, indent=2))
