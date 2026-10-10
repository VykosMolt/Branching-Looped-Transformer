"""Check janiak_vendor.py against Denis's code at commit 26c48be, by import.

1. Every vendored function's source equals the original's source.
2. aggregate_question (restated) matches aggregate_prompt_features on random questions with ties, unparsed traces
   and all-unparsed questions.
Run: venv/bin/python test_vendor.py [path to reasoning-geometry checkout]
"""
import inspect
import subprocess
import sys

import numpy as np

import janiak_vendor as V

REPO = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-1000/-home-moloch-Documents-Research-ouro-Operational-Proto-Introspection/"
    "07561c8b-32be-4f56-b8e6-e1f42d49fe0a/scratchpad/reasoning-geometry")
assert subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip() == "26c48be"
sys.path.insert(0, REPO)
from applications import incremental_abstention as IA  # noqa: E402
from baselines import deepconf_asymmetry as DA  # noqa: E402
from controls import orgad_agreement_control as OC  # noqa: E402

SOURCES = {
    IA: ("_finite", "is_parseable_answer", "_row_logprob", "_winning_answer", "_plurality_outcome", "_mean_field",
         "_impute_and_scale", "crossfit_logistic_predictions", "_auacc", "prompt_metrics", "paired_bootstrap_delta"),
    DA: ("_average_ranks", "auroc", "bootstrap_auroc", "_pearson"),
    OC: ("spearman", "crossfit_residuals", "agreement_strata", "stratum_readout", "agreement_level_counts"),
}
for module, names in SOURCES.items():
    for name in names:
        assert inspect.getsource(getattr(module, name)) == inspect.getsource(getattr(V, name)), name
assert IA.METRIC_NAMES == V.METRIC_NAMES and OC.MIN_STRATUM_CLASS == V.MIN_STRATUM_CLASS

rng = np.random.default_rng(0)
rows = []
for pid in range(300):
    gold = "A"
    mode = pid % 5  # 0: all unparsed; 1: two-way tie; else random
    for t in range(4):
        ans = None if mode == 0 else ("AB"[t % 2] if mode == 1 else rng.choice(["A", "B", "C", None]))
        rows.append({"prompt_id": pid, "trace_id": t, "fold": pid % 5, "trace_length": int(rng.integers(50, 449)),
                     "predicted_answer": ans, "gold_answer": gold, "length_score": float(rng.integers(50, 449)),
                     "logprob_score": float(rng.normal()), "entropy_score": float(rng.random())})
ref = IA.aggregate_prompt_features(rows, max_new_tokens=448, expected_traces=4)
n_checked = 0
for pid, entry in ref.items():
    mine = V.aggregate_question([r for r in rows if r["prompt_id"] == pid])
    for key in ("length", "entropy", "logprob", "vote_agreement", "outcome", "automatic_failure", "valid_plurality"):
        assert np.isclose(mine[key], entry[key]) if isinstance(entry[key], float) else mine[key] == entry[key], (pid, key)
    n_checked += 1
print(f"vendor check passed: {sum(map(len, SOURCES.values()))} functions source-identical; "
      f"aggregate_question matches aggregate_prompt_features on {n_checked} questions")
