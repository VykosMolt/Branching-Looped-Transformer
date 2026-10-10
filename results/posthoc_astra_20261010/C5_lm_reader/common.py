"""Shared helpers for C5 (LM text reader vs hidden-state probe). See PROTOCOL.md."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only run directories we import from
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = "/home/moloch/ouro_project"
R1 = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927"
R4 = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/horizon_cohort2_20261003"
R4CODE = ROOT + "/opi/preanswer/horizon_cohort2_20261003"
R3CODE = ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"
COHORTS = ("r1", "r1s", "r4")  # Run 1 published labels, Run 1 strict labels, Run 4 (strict labels)
OOF_COHORTS = ("r1", "r4")
N_FOLDS = 5
BOOT_SEED = 20260724  # the published bootstraps' seed
ROUNDS = 2000


def lib_paths():
    for p in (ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual", R3CODE):
        if p not in sys.path:
            sys.path.insert(0, p)


def fold_of(task_uid: str) -> int:
    """Question-level fold. Salted: Run 1's questions are a contiguous range of the unsalted sha256 order."""
    return int(hashlib.sha256(f"C5_lm_reader|{task_uid}".encode()).hexdigest(), 16) % N_FOLDS


def tfidf_text(prompt: str, prefix: str) -> str:
    """Exactly the TF-IDF reader's input (analyze_run1.visible_reader, analyze_run4.visible_reader)."""
    return f"Prompt:\n{prompt}\n\nVisible reasoning prefix:\n{prefix}"


def load_rows(path=DATA / "answers.jsonl.gz"):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def cohort(rows, name):
    """Rows of one cohort, in the published record order, with numpy views used everywhere."""
    rs = [r for r in rows if r["cohort"] == name]
    y = np.array([r["y"] for r in rs], int)
    g = np.array([r["task_uid"] for r in rs])
    split = np.array([r["split"] for r in rs])
    fold = np.array([r["fold"] for r in rs])
    return rs, y, g, split, fold


def check_folds(g, split, fold):
    """Fold assignment invariants: one fold per question, hash rule respected, every fold used, one split per question."""
    seen_f, seen_s = {}, {}
    for t, s, f in zip(g, split, fold):
        assert seen_f.setdefault(t, f) == f, f"question {t} in two folds"
        assert seen_s.setdefault(t, s) == s, f"question {t} in two splits"
        assert f == fold_of(t), "fold does not follow the hash rule"
    assert set(fold.tolist()) == set(range(N_FOLDS)), "a fold is empty"


def mixed_pairs(g, y, idx=None):
    """{question: [(i_correct, i_incorrect), ...]} over questions with both outcomes, restricted to positions idx."""
    idx = np.arange(len(y)) if idx is None else np.asarray(idx)
    by = {}
    for i in idx:
        by.setdefault(g[i], []).append(int(i))
    pairs = {t: [(p, n) for p in ix if y[p] == 1 for n in ix if y[n] == 0] for t, ix in sorted(by.items())}
    pairs = {t: v for t, v in pairs.items() if v}
    for t, v in pairs.items():
        for p, n in v:
            assert g[p] == g[n] == t and y[p] == 1 and y[n] == 0 and p != n
    return pairs


def wq(s, pairs, ts):
    """Pair-weighted within-question AUROC over questions ts (ties 1/2), as paired_within.py."""
    v = [1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for t in ts for p, n in pairs[t]]
    return float(np.mean(v))


def paired_within(a, b, pairs):
    """a minus b, within question, with the published question-clustered bootstrap (paired_within.py)."""
    mixed = sorted(pairs)
    rng = np.random.default_rng(BOOT_SEED)
    d = sorted(wq(a, pairs, ts) - wq(b, pairs, ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(ROUNDS)))
    return {"a": wq(a, pairs, mixed), "b": wq(b, pairs, mixed), "diff": wq(a, pairs, mixed) - wq(b, pairs, mixed),
            "ci95": [d[50], d[1949]], "n_questions": len(mixed), "n_pairs": sum(len(v) for v in pairs.values())}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":  # self-check of fold and pairing logic on a toy set
    g = np.array(["a", "a", "a", "b", "b", "c", "c", "d"])
    y = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    p = mixed_pairs(g, y)
    assert p == {"a": [(0, 1), (0, 2)], "c": [(6, 5)]}, p
    assert mixed_pairs(g, y, idx=[0, 1, 3, 4]) == {"a": [(0, 1)]}
    s = np.array([0.9, 0.1, 0.9, 0.5, 0.5, 0.2, 0.2, 0.0])
    assert wq(s, p, sorted(p)) == (1.0 + 0.5 + 0.5) / 3
    r = paired_within(s, -s, p)
    assert abs(r["diff"] - (2 / 3 - 1 / 3)) < 1e-12 and r["n_pairs"] == 3
    assert fold_of("x") == fold_of("x") and 0 <= fold_of("x") < N_FOLDS
    print("common.py self-check ok")
