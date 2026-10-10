from common import *
import analyze_run3 as A3
from answer_detector import is_answer_bearing, task_statement
GRID = (0, 16, 32, 48, 64, 96, 128)
CLASSES = ("True", "False", "Unknown")
for r in meta:
    st = task_statement(r["task_prompt"])
    r["free"] = {n: (n == 0 or not is_answer_bearing(r["prefix_texts"][k], st)) for k, n in enumerate(GRID)}
lab = [r for r in meta if r["labelled"]]
def feat(n):
    key = f"f{n}"
    return lambda r: torch.from_numpy(np.asarray(F[key][r["i"]], dtype=np.float32))
pend = lambda r: torch.from_numpy(np.asarray(F["pend"][r["i"]], dtype=np.float32))
def lp_at(n):
    def f(r):
        lp = r["gen_logprobs"][:n] or [0.0]
        return torch.tensor([sum(lp) / len(lp), min(lp)])
    return f
def s_diff(r):
    return torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"]), float(r["n_vars"]), float(r["n_facts"]),
                         float(r["n_rules"])] + [float(r["gold_answer"] == c) for c in CLASSES])
S48 = lambda r: torch.cat([lp_at(48)(r), s_diff(r)])
primary = [r for r in lab if r["free"][48]]
