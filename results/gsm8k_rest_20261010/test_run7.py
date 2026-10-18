"""CPU dry run of Run 7 (no GPU, no network): problem selection and overlap asserts, fold assignment, label parsing
(Run 3's detector self-check and the chronological variant), the probe against analyze_run3._fit_score, and both
analysis stages end to end on synthetic shards (including the pooled Run 6 + Run 7 path).
Run: shared/venv/bin/python opi/preanswer/gsm8k_rest_20261010/test_run7.py  (temporary files go to $TMPDIR)."""
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_run7 as A7  # noqa: E402  (sets up the paths)
import within_oof as W  # noqa: E402
from gen_run7 import PREFIX_GRID, earlier_runs, select_tasks  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402


def test_selection():
    c = select_tasks()  # asserts 6,016 problems, no id or normalized-text overlap, no duplicate text
    assert [t["task_id"] for t in c] == [t["task_id"] for t in select_tasks()]
    n_train = sum(t["dataset_split"] == "train" for t in c)
    runs = earlier_runs()
    assert (n_train, len(c) - n_train) == (7473 - 2000 - 3, 1319 - 170 - 600 - 3)
    sizes = {k: len(v) for k, v in runs.items() if not k.startswith("_")}
    assert sizes == {"original_170": 170, "run3_cohort": 600, "run3_smoke": 3, "run6_cohort": 2000, "run6_smoke": 3}
    assert {t["dataset_index"] for t in runs["original_170"]} == set(range(170))
    ids = {t["task_id"] for t in c}
    assert not ids & {t["task_id"] for t in runs["run6_cohort"][:3]}  # the smoke problems are not in the cohort
    from gen_run7 import norm_q
    assert not {norm_q(t["question"]) for t in c} & {norm_q(t["question"]) for k, v in runs.items() if not k.startswith("_") for t in v}
    print(f"selection ok: {len(c)} problems ({n_train} train, {len(c) - n_train} test); earlier runs {sizes}; overlap 0 by id and text")


def test_folds():
    assert [W.hash_fold(t) for t in ("gsm8k/0", "gsm8k-train/0", "gsm8k-train/7472")] == [1, 1, 2]  # pinned
    ids = [t["task_id"] for t in select_tasks()]
    sizes = np.bincount([W.hash_fold(t) for t in ids], minlength=5)
    assert sizes.sum() == len(ids) and abs(sizes / len(ids) - 0.2).max() < 0.02, sizes
    rows = [t for t in ids for _ in range(4)]  # four answers per problem land in the same fold
    for tr, te in W.hash_folds(rows, "oof5"):
        assert not {rows[i] for i in tr} & {rows[i] for i in te}
    inner = W.hash_folds(np.array(rows)[W.hash_folds(rows, "oof5")[0][0]], "inner5")
    assert all(len(te) > 0 for _, te in inner)
    print("folds ok: outer fold sizes", sizes.tolist())


def test_labels():
    runpy.run_path(str(A7.R3 / "answer_detector_gsm8k.py"), run_name="__main__")  # Run 3's 29-case self-check
    from answer_detector_gsm8k import first_commitment
    cases = [  # (text, primary = priority order, chronological = earliest match, primary match offset)
        ("Answer: 16 eggs\nFINAL ANSWER: 20", "20", "16", 16),
        ("the answer is 5. #### 6", "6", "5", 17),
        ("so \\boxed{3} then FINAL ANSWER: 4\n\nFINAL ANSWER: 9", "4", "3", 18),
        ("FINAL ANSWER: 18\n#### 7", "18", "18", 0),
        ("no number here", None, None, None),
        ("**FINAL ANSWER: 18**", "18", "18", 2),
    ]
    for text, prim, chrono, pos in cases:
        assert first_commitment(text) == prim, (text, first_commitment(text))
        assert A7.first_commitment_chrono(text) == chrono, (text, A7.first_commitment_chrono(text))
        assert A7.commit_pos(text) == pos, (text, A7.commit_pos(text))
    print("labels ok: detector self-check +", len(cases), "chronological cases")


def test_probe_matches_run3():
    """within_oof.fit_score (ARPACK PCA, folds passed in) against analyze_run3._fit_score (full SVD, GroupKFold)."""
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(0)
    n, g = 600, np.repeat(np.arange(150), 4).astype(str)
    z = rng.standard_normal(n)
    B = [(z[:, None] * rng.standard_normal(300) + rng.standard_normal((n, 300)) * 3).astype(np.float32)]
    S = rng.standard_normal((n, 3)).astype(np.float32)
    y = (z + rng.standard_normal(n) > 0).astype(int)
    tv = np.array([split_for(f"t{x}") != "heldout" for x in g]); ho = ~tv
    p3, h3 = A7.A3._fit_score(B, S, y, g, tv, ho)
    itv = np.where(tv)[0]
    folds = list(GroupKFold(5).split(itv, y[itv], g[itv]))
    pw, hw = W.fit_score(B, S, y, itv, np.where(ho)[0], folds, {}, 0)
    assert (h3["k"], h3["C"]) == (hw["k"], hw["C"]) and np.abs(p3 - pw).max() < 1e-4, (h3, hw, np.abs(p3 - pw).max())
    print(f"probe ok: matches analyze_run3._fit_score (max |diff| {np.abs(p3 - pw).max():.1e}, k={hw['k']}, C={hw['C']})")


def synth_shards(tasks, prefix, out: Path, seed: int, dim: int = 16):
    """Shards in the generator's format with a planted per-answer signal; some texts commit twice (chronological
    variant differs) and some never commit (unparseable)."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    w = rng.standard_normal((3, 4, dim))
    for s0 in range(0, len(tasks), 100):
        recs = []
        for t in tasks[s0:s0 + 100]:
            diff = rng.normal()
            for ci in range(4):
                z = rng.normal()
                ok = z + diff + rng.normal() * 0.5 > -0.8
                own = t["gold"] if ok else str(int(float(t["gold"])) + 1)
                pre = "Solution: Okay, let's see. We read the problem and set up the steps carefully."
                late = " more careful work" * (6 if ok else 2)
                kind = rng.integers(10)
                tail = (f" so FINAL ANSWER: {own}\nmore" if kind > 1 else f" Answer: {own}0 then FINAL ANSWER: {own}" if kind == 1 else " and it stops")
                txt = pre + late + tail
                own_p = None if kind == 0 else own
                feats = torch.tensor(z * w + rng.normal(size=(len(PREFIX_GRID), 3, 4, dim)), dtype=torch.float32)
                recs.append({**t, "split": split_for(t["task_id"]), "candidate_idx": ci, "seed": 0, "prompt_tok": 100 + len(t["question"]) // 5,
                             "question_chars": len(t["question"]), "prompt_ids": [1], "gen_ids": [2] * 120, "n_gen_tok": 120,
                             "hit_max_tokens": bool(rng.random() < 0.1), "generated_text": txt,
                             "gen_logprobs": list(rng.normal(-0.5, 0.2, 120)), "decode_roundtrip_ok": True, "own_answer": own_p,
                             "parseable": own_p is not None, "success": own_p is not None and own == t["gold"],
                             "prefix_grid": list(PREFIX_GRID), "prefix_features": feats,
                             "prompt_end_state": torch.tensor(diff * w + rng.normal(size=(3, 4, dim)), dtype=torch.float32),
                             "prefix_texts": [pre[: 6 * n] for n in PREFIX_GRID]})
        torch.save({"shard_start": s0, "seed": 0, "k": 4, "max_new_tokens": 448, "prefix_grid": list(PREFIX_GRID), "model": "synthetic",
                    "smoke": False, "elapsed_seconds": 0.0, "equivalence_checks_maxabs": [0.0], "errors": [], "records": recs},
                   out / f"{prefix}_shard_{s0:04d}.pt")


def test_end_to_end():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        d, d6 = Path(tmp) / "r7", Path(tmp) / "r6"
        synth_shards(select_tasks()[:300], "run7", d, 1)
        synth_shards(earlier_runs()["run6_cohort"][:150], "run6", d6, 2)
        for stage in ("audit", "full"):
            cmd = [sys.executable, str(HERE / "analyze_run7.py"), "--stage", stage, "--data", str(d), "--out", str(d), "--synthetic", "--run6-data", str(d6)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
        res = json.loads((d / "results.json").read_text())
        assert (d / "hand_audit.md").exists() and (d / "RESULTS.md").exists() and (d / "audit_stage.json").exists()
        p = res["primary"]
        assert p["verdict"] in ("WITHIN_QUESTION_SIGNAL", "NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55", "UNRESOLVED", "UNDERPOWERED")
        assert p["within_question_oof"]["hidden"]["estimate"] > 0.6, p["within_question_oof"]["hidden"]  # planted signal found
        assert res["secondary_chronological_labels"]["answers_own_answer_differs"] > 0 and "within_question_hidden_oof" in res["secondary_chronological_labels"]
        assert "within_question_hidden_oof" in res["secondary_pooled_run6_run7"]
        assert res["integrity"]["n_tasks"] == 300 and res["run6_rule_would_choose"] in PREFIX_GRID
        print("end to end ok:", (d / "RESULTS.md").read_text().splitlines()[2])
        print("   primary hidden", round(p["within_question_oof"]["hidden"]["estimate"], 3), "| pooled",
              round(res["secondary_pooled_run6_run7"]["within_question_hidden_oof"]["estimate"], 3),
              "| across verdict", res["secondary_across_question"].get("verdict"))


def test_null_oof():
    """No signal planted: the out-of-fold within-question interval must cover 0.5 (fixed seed)."""
    rng = np.random.default_rng(3)
    tasks = np.repeat([f"gsm8k-train/{i}" for i in range(400)], 4)
    diff = np.repeat(rng.normal(size=400), 4)
    y = (diff + rng.normal(size=len(tasks)) > -0.5).astype(int)
    Hm = (diff[:, None] + rng.normal(size=(len(tasks), 120))).astype(np.float32)  # difficulty only, no within-question signal
    sc, _ = W.oof_scores([Hm], rng.normal(size=(len(tasks), 2)), y, tasks, ("hidden",))
    wq = W.within_task_auroc(sc["hidden"], y, list(tasks))
    assert wq["ci95"][0] < 0.5 < wq["ci95"][1], wq
    print(f"null ok: within-question {wq['estimate']:.3f} [{wq['ci95'][0]:.3f}, {wq['ci95'][1]:.3f}] on {wq['n_mixed_tasks']} mixed problems")


def test_seal_list_and_imports():
    """seal_run7.sh seals exactly analyze_run7.SEALED, and every imported earlier-run file is byte-identical to the
    hash in that run's own SEAL.sha256."""
    import hashlib
    import re
    files = re.search(r'FILES="([^"]+)"', (HERE / "seal_run7.sh").read_text()).group(1).split()
    assert tuple(files) == A7.SEALED, files
    known = {}
    for d in (A7.R6, A7.R3):
        known.update({(d / n).resolve(): h for h, n in (l.split() for l in (d / "SEAL.sha256").read_text().splitlines())})
    for f in files:
        p = (HERE / f).resolve()
        if p.parent != HERE:
            assert known[p] == hashlib.sha256(p.read_bytes()).hexdigest(), f
    print("seal list ok:", len(files), "files; imported files match their runs' seals")


def test_verdict_rule():
    v = lambda lo, hi, n=150: W.verdict({"n_mixed_tasks": n, "ci95": [lo, hi]}, 100)  # noqa: E731
    assert v(0.51, 0.7) == "WITHIN_QUESTION_SIGNAL" and v(0.45, 0.54) == "NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55"
    assert v(0.48, 0.6) == "UNRESOLVED" and v(0.6, 0.9, 99) == "UNDERPOWERED" and v(0.51, 0.54) == "WITHIN_QUESTION_SIGNAL"
    print("verdict rule ok")


if __name__ == "__main__":
    for f in (test_selection, test_folds, test_labels, test_seal_list_and_imports, test_verdict_rule, test_null_oof,
              test_probe_matches_run3, test_end_to_end):
        f()
    print("ALL RUN 7 DRY-RUN TESTS PASSED")
