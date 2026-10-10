"""Post-hoc (not preregistered): how much of Run 1's +0.243 is a weaker baseline?"""
import sys, json, torch
R1 = "/home/moloch/ouro_project/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path.insert(0, R1)
import analyze_run1 as A
from bg_v2_overnight_horizon_analysis import shortcut_vec

recs = A.load()
A.answer_free_table(recs)
n_star, _ = A.choose_n_star(recs)
parse = [r for r in recs if not r["malformed"]]
primary = [r for r in parse if r["free"][n_star]]
lp = A.prefix_lp_vec(n_star)
diff = lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])])
arms = {
    "A_sealed_primary (S = lp mean/min)": (primary, A.feat_at(n_star), lp),
    "B_S = lp + prompt length + proof depth": (primary, A.feat_at(n_star), lambda r: torch.cat([lp(r), diff(r)])),
    "C_S = prompt length + proof depth only": (primary, A.feat_at(n_star), diff),
    "D_final-line features + 4 shortcuts, SAME cohort": (primary, lambda r: r["preanswer_features"], shortcut_vec),
    "E_final-line + 4 shortcuts + depth, same cohort": (primary, lambda r: r["preanswer_features"], lambda r: torch.cat([shortcut_vec(r), diff(r)])),
    "F_prompt-only features, same cohort, S = lp + depth": (primary, A.feat_at(0), lambda r: torch.cat([lp(r), diff(r)])),
}
out = {}
for name, (rows, feat, sc) in arms.items():
    res = A.run_arm(name[:1], rows, feat, sc)[0]
    b = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    out[name] = res
    print(f"{name}: surface {res['auroc_shortcut_only']:.3f} hidden {res['auroc_hidden_only']:.3f} both {res['auroc_hidden_plus_shortcuts']:.3f} "
          f"increase {res['incremental_auroc_combined_minus_shortcut']:+.3f} [{b[0]:+.3f}, {b[1]:+.3f}] n_ho {res['n_heldout']} neg {res['n_heldout_negatives']}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2, default=float)
