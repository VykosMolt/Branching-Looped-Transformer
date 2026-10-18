"""CPU dry run of Run 8 (no GPU, no network): questions and protocol constants identical to Run 4, model pins,
label parsing (Run 4's strict_answer cases plus the chronological variant), fold assignment, and both analysis stages
end to end on synthetic shards, including the comparison with synthetic Run 4 (RLTT) shards.
Run: shared/venv/bin/python opi/preanswer/horizon_thinking_20261010/test_run8.py  (temporary files go to $TMPDIR)."""
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_run8 as A8  # noqa: E402  (sets up the paths)
import gen_run4 as G4  # noqa: E402
import gen_run8 as G8  # noqa: E402
import within_oof as W  # noqa: E402
from bg_v2_overnight_horizon_generate import _sid, split_for  # noqa: E402


def test_same_inputs_as_run4():
    t8, t4 = G8.select_tasks(), G4.select_tasks()
    assert [t["task_uid"] for t in t8] == [t["task_uid"] for t in t4] and len(t8) == 1164
    for k in ("SEED", "K", "MAX_NEW", "SHARD", "PREFIX_GRID", "CHECK_EVERY", "TASKS_SHA256"):
        assert getattr(G8, k) == getattr(G4, k), k
    for f in ("select_tasks", "strict_answer", "forward_all"):
        assert inspect.getsource(getattr(G8, f)) == inspect.getsource(getattr(G4, f)), f
    main8, main4 = inspect.getsource(G8.main), inspect.getsource(G4.main)  # main differs only in model and names
    allowed = {"    model_sha = ensure_model()",
               '    extractor = BGTransformerFeatureExtractor(model_path=MODEL, device="cuda", dtype="auto", force_all_loops=True)',
               '    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)'}
    d8 = {l for l in main8.splitlines() if l not in main4.splitlines()} - allowed
    d4 = {l for l in main4.splitlines() if l not in main8.splitlines()} - allowed
    norm = lambda s: {l.replace("run8", "runX").replace("run4", "runX").replace('"model": f"{REPO}@{REVISION}", "model_sha256": model_sha,', '"model": M,')
                      .replace('"model": "shared/models/ouro_rltt_local",', '"model": M,') for l in s}  # noqa: E731
    assert norm(d8) == norm(d4), (d8, d4)
    print("inputs ok: same 1,164 questions in the same order; seeds, sampling, grid, labels, features identical to Run 4")


def test_model_pins():
    assert G8.MODEL_FILES["modeling_ouro.py"] == hashlib.sha256((HERE / "modeling_ouro_shim.py").read_bytes()).hexdigest()
    assert G8.MODEL_FILES["model.safetensors"] == "c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080"
    shim = (HERE / "modeling_ouro_shim.py").read_text()
    assert "def key_cache(self, value" in shim and "self._key_cache" in shim
    print("model pins ok:", len(G8.MODEL_FILES), "files pinned")


def test_labels():
    sys.path.insert(0, str(A8.R4))
    import test_strict_answer as T  # Run 4's 13 cases, against gen_run8.strict_answer
    for text, want in T.CASES:
        assert G8.strict_answer(text, T.OPTS) == want, text
    O = T.OPTS
    cases = [  # (text, strict = letter priority, chronological = earliest match)
        ("FINAL ANSWER: Unknown\n... FINAL ANSWER: B", "B", "C"),
        ("Final answer: True then \\boxed{B}", "B", "A"),
        ("FINAL ANSWER: B\nFINAL ANSWER: Unknown", "B", "B"),
        ("no commitment (A) here", None, None),
    ]
    for text, strict, chrono in cases:
        assert G8.strict_answer(text, O) == strict and A8.chrono_answer(text, O) == chrono, (text, G8.strict_answer(text, O), A8.chrono_answer(text, O))
    assert A8.strict_pos("x FINAL ANSWER: Unknown\n... FINAL ANSWER: B", O) == 28 and A8.strict_pos("ab\\boxed{A}", O) == 2
    for text, want in T.CASES:  # the chronological variant agrees wherever only one commitment exists
        if want is None:
            assert A8.chrono_answer(text, O) is None, text
    print("labels ok: Run 4's", len(T.CASES), "cases +", len(cases), "chronological cases")


def test_seal_list_and_imports():
    """seal_run8.sh seals exactly analyze_run8.SEALED; imported earlier-run files match their runs' seals (Run 7's
    within_oof.py is checked once Run 7 is sealed)."""
    import re
    files = re.search(r'FILES="([^"]+)"', (HERE / "seal_run8.sh").read_text()).group(1).split()
    assert tuple(files) == A8.SEALED, files
    known = {}
    for d in (A8.R4, A8.R3, A8.R7):
        if (d / "SEAL.sha256").exists():
            known.update({(d / n).resolve(): h for h, n in (l.split() for l in (d / "SEAL.sha256").read_text().splitlines())})
    unchecked = []
    for f in files:
        p = (HERE / f).resolve()
        if p.parent != HERE:
            if p in known:
                assert known[p] == hashlib.sha256(p.read_bytes()).hexdigest(), f
            else:
                unchecked.append(f)
    assert unchecked in ([], ["../gsm8k_rest_20261010/within_oof.py"]), unchecked
    print("seal list ok:", len(files), "files; imported files match their runs' seals; not yet sealed elsewhere:", unchecked)


def test_folds():
    ids = [t["task_uid"] for t in G8.select_tasks()]
    sizes = np.bincount([W.hash_fold(t) for t in ids], minlength=5)
    assert abs(sizes / len(ids) - 0.2).max() < 0.03, sizes
    print("folds ok:", sizes.tolist())


def synth(tasks, prefix, out: Path, seed: int, dim: int = 16, model_sha=None):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    w = rng.standard_normal((3, 4, dim))
    G = G8.PREFIX_GRID
    for s0 in range(0, len(tasks), 100):
        recs = []
        for t in tasks[s0:s0 + 100]:
            opts, gold = G8.options_dict(t["options"]), chr(65 + int(t["answer_key"]))
            diff = rng.normal()
            for ci in range(4):
                z = rng.normal()
                ok = z + diff + rng.normal() * 0.5 > -0.8
                letter = gold if ok else "ABC"[("ABC".index(gold) + 1) % 3]
                pre = "Let me parse the facts and rules carefully, one at a time, and keep track of each step."
                kind = rng.integers(10)
                tail = (f" so FINAL ANSWER: {letter}\nmore" if kind > 1 else f" Final answer: {opts['ABC'[('ABC'.index(letter) + 2) % 3]]} then FINAL ANSWER: {letter}" if kind == 1 else " and it stops")
                txt = pre + " more careful work" * (6 if ok else 2) + tail
                strict = G8.strict_answer(txt, opts)
                recs.append({"task_uid": t["task_uid"], "split": split_for(t["task_uid"]), "proof_depth": t["proof_depth"],
                             "n_vars": t["verifier"]["n_vars"], "n_facts": len(t["symbolic"]["facts"]), "n_rules": len(t["symbolic"]["rules"]),
                             "gold_answer": t["gold_answer"], "gold_letter": gold, "options": opts, "task_prompt": t["task_prompt"],
                             "candidate_idx": ci, "seed": _sid(G8.SEED, t["task_uid"]), "generated_text": txt, "n_gen_tok": 120,
                             "hit_max_tokens": bool(rng.random() < 0.3), "prompt_tok": 150, "prompt_ids": [1, 2, 3], "gen_ids": [4] * 120,
                             "gen_logprobs": list(rng.normal(-0.5, 0.2, 120)), "strict_answer": strict, "labelled": strict is not None,
                             "success": strict == gold, "parsed_published": letter if kind else None, "malformed_published": kind == 0,
                             "success_published": bool(kind and letter == gold), "found_final_marker": bool(kind), "n_pre_tok": 100,
                             "mean_logprob_pre": -0.5, "min_logprob_pre": -2.0,
                             "preanswer_features": torch.tensor(z * w + rng.normal(size=(3, 4, dim)), dtype=torch.float32),
                             "prefix_grid": list(G), "prefix_features": torch.tensor(z * w + rng.normal(size=(len(G), 3, 4, dim)), dtype=torch.float32),
                             "prompt_end_state": torch.tensor(diff * w + rng.normal(size=(3, 4, dim)), dtype=torch.float32),
                             "prefix_texts": [pre[: 2 * n] for n in G], "decode_roundtrip_ok": True})
        d = {"shard_start": s0, "seed": G8.SEED, "k": 4, "max_new_tokens": 448, "prefix_grid": list(G), "model": "synthetic", "smoke": False,
             "elapsed_seconds": 0.0, "equivalence_checks_maxabs": [0.0], "errors": [], "records": recs}
        if model_sha is not None:
            d["model_sha256"] = model_sha
        torch.save(d, out / f"{prefix}_shard_{s0:04d}.pt")


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        d8, d4 = Path(tmp) / "r8", Path(tmp) / "r4"
        tasks = G8.select_tasks()[:300]
        synth(tasks, "run8", d8, 1, model_sha=G8.MODEL_FILES)
        synth(tasks, "run4", d4, 2)
        for stage in ("audit", "full"):
            r = subprocess.run([sys.executable, str(HERE / "analyze_run8.py"), "--stage", stage, "--data", str(d8), "--out", str(d8),
                                "--synthetic", "--rltt-data", str(d4)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
        res = json.loads((d8 / "results.json").read_text())
        p = res["primary"]
        assert p["verdict"] in ("WITHIN_QUESTION_SIGNAL", "NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55", "UNRESOLVED", "UNDERPOWERED")
        assert p["within_question_oof"]["hidden"]["estimate"] > 0.6
        cm = res["secondary_rltt_comparison"]
        assert cm["inputs_identical"]["answers_compared"] == 1200 and cm["inputs_identical"]["seeds_equal"] == 1200
        assert "thinking_minus_rltt" in cm["own_n_star"] and res["secondary_chronological_labels"]["answers_label_differs"] > 0
        assert "within_question_hidden_oof" in res["secondary_chronological_labels"] and "run4_primary_across_question" in res
        print("end to end ok:", (d8 / "RESULTS.md").read_text().splitlines()[2])
        print("   primary hidden", round(p["within_question_oof"]["hidden"]["estimate"], 3), "| thinking minus rltt",
              round(cm["own_n_star"]["thinking_minus_rltt"]["diff"], 3), "| N* =", res["n_star"])
        # a shard generated with other weights must be refused
        f = sorted(d8.glob("run8_shard_*.pt"))[0]
        x = torch.load(f, weights_only=False); x["model_sha256"] = {**x["model_sha256"], "model.safetensors": "0" * 64}; torch.save(x, f)
        r = subprocess.run([sys.executable, str(HERE / "analyze_run8.py"), "--stage", "audit", "--data", str(d8), "--out", str(d8), "--synthetic"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "other weights" in r.stderr
        print("   shard with other weights refused")


if __name__ == "__main__":
    for f in (test_same_inputs_as_run4, test_model_pins, test_labels, test_seal_list_and_imports, test_folds, test_end_to_end):
        f()
    print("ALL RUN 8 DRY-RUN TESTS PASSED")
