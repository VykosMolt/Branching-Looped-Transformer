from rows import *
import json, pickle
print("labelled", len(lab), "primary", len(primary))
res, sc = A3.proper_arm("primary", primary, [feat(48)], S48)
wq = {k: A3.within_task_auroc(sc[k].numpy(), sc["y"].numpy(), sc["tasks"]) for k in ("combined", "hidden", "shortcut")}
print(json.dumps(wq))
pickle.dump({"res": res, "sc": {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in sc.items()},
             "ho_idx": [r["i"] for r in primary if r["split"] == "heldout"]}, open("repro_primary.pkl", "wb"))
pe, _ = A3.proper_arm("prompt_end_control", primary, [feat(48)], S48, base=[pend])
print("pend control", pe["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"])
