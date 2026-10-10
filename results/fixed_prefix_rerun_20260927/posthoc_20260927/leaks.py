"""Post-hoc (not preregistered) checks on Run 1: verdict identity, margin to commitment, composition."""
import sys, json, collections, torch
R1 = "/home/moloch/ouro_project/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path.insert(0, R1)
import analyze_run1 as A
from answer_detector import answer_hits, task_statement
from proto_introspection_controls_analysis import auroc
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/moloch/ouro_project/shared/models/ouro_rltt_local")
recs = A.load()
A.answer_free_table(recs)
N = 48
parse = [r for r in recs if not r["malformed"]]
primary = [r for r in parse if r["free"][N]]
V = ("True", "False", "Unknown")
verdict = lambda r: r["options"][r["parsed_answer"]].strip().rstrip(".").capitalize()
print("verdict values in primary:", collections.Counter(verdict(r) for r in primary))
vec_v = lambda r: torch.tensor([float(verdict(r) == v) for v in V])
lp, diff = A.prefix_lp_vec(N), (lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])]))
out = {}

# 1. composition before/after the detector (label-free counts, then error rates)
for split in ("train", "val", "heldout"):
    rs = [r for r in parse if r["split"] == split]
    kept, drop = [r for r in rs if r["free"][N]], [r for r in rs if not r["free"][N]]
    err = lambda xs: sum(not r["success"] for r in xs) / max(len(xs), 1)
    out[f"composition_{split}"] = {"parseable": len(rs), "kept": len(kept), "dropped": len(drop),
                                   "err_kept": round(err(kept), 3), "err_dropped": round(err(drop), 3)}
    print(split, out[f"composition_{split}"])
pat = collections.Counter(h for r in parse if not r["free"][N]
                          for h in answer_hits(r["prefix_texts"][A.GRID.index(N)], task_statement(r["task_prompt"])))
print("detector patterns that dropped answers at N=48:", dict(pat))
dv = collections.Counter((verdict(r), r["success"]) for r in primary)
print("verdict x correct (primary):", dict(dv))

# 2. verdict identity as a baseline (it uses the future answer, so it is a diagnostic, not a usable baseline)
for name, sc in (("verdict only", vec_v), ("lp + verdict", lambda r: torch.cat([lp(r), vec_v(r)])),
                 ("lp + depth + prompt + verdict", lambda r: torch.cat([lp(r), diff(r), vec_v(r)]))):
    res = A.run_arm(name, primary, A.feat_at(N), sc)[0]
    b = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    out[name] = res
    print(f"S = {name}: surface {res['auroc_shortcut_only']:.3f} hidden {res['auroc_hidden_only']:.3f} both "
          f"{res['auroc_hidden_plus_shortcuts']:.3f} increase {res['incremental_auroc_combined_minus_shortcut']:+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]", flush=True)

# 3. margin: the sealed primary model, scored separately on answers that commit soon after the cut vs much later
res, sc, y_ho, t_ho = A.run_arm("primary", primary, A.feat_at(N), lp)
ho = [r for r in primary if r["split"] == "heldout"]
def first_commit_after(r, limit=160):
    st = task_statement(r["task_prompt"])
    for n in range(N + 4, limit + 1, 4):
        if answer_hits(tok.decode(r["gen_ids"][:n], skip_special_tokens=True), st):
            return n
    return None
fc = [first_commit_after(r) for r in ho]
groups = {"commits in tokens 49-64": [c is not None and c <= 64 for c in fc],
          "commits in tokens 65-128": [c is not None and 64 < c <= 128 for c in fc],
          "no commitment through 160": [c is None or c > 160 for c in fc]}
for g, mask in groups.items():
    m = torch.tensor(mask)
    yy, tt = y_ho[m], [t for t, k in zip(t_ho, mask) if k]
    if len(tt) == 0 or yy.float().mean() in (0.0, 1.0):
        print(g, "n", int(m.sum()), "one class only"); continue
    b = bootstrap_task_clustered(sc["combined"][m], sc["shortcut"][m], yy, tt, rounds=2000)
    out[f"margin {g}"] = {"n": int(m.sum()), "neg": int((yy == 0).sum()), "auroc_both": auroc(sc["combined"][m], yy),
                          "auroc_hidden": auroc(sc["hidden"][m], yy), "auroc_surface": auroc(sc["shortcut"][m], yy), "boot": b}
    print(f"{g}: n {int(m.sum())} incorrect {int((yy == 0).sum())} hidden {auroc(sc['hidden'][m], yy):.3f} surface "
          f"{auroc(sc['shortcut'][m], yy):.3f} both {auroc(sc['combined'][m], yy):.3f} increase ci {b['ci95'] if b else None}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2, default=float)
