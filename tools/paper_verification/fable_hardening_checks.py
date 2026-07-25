#!/usr/bin/env python3
"""Bounded verification helpers for Fable's paper-hardening checks.

This script never trains or modifies an Ouro backbone.  It has three narrow
operations:

* capture-base: capture mean-pooled loop-boundary states for the already
  selected 200 HH-RLHF pairs from ByteDance/Ouro-2.6B;
* localization: reconstruct the documented bias-free L-BFGS difference probe
  on saved Thinking features and apply the same weights/split to all backbones;
* preanswer-ci: recreate the saved task-grouped OOF predictions and bootstrap
  the paired AUROC increment by task cluster.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260710
PROBE_SEED = 42
BASE_MODEL_ID = "ByteDance/Ouro-2.6B"
BASE_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
THINKING_MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
THINKING_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
RLTT_PATH = ROOT / "models/ouro_rltt_local"
THINKING_FEATURES = ROOT / "artifacts/reports/evaluator/hh_layer_states_200_thinking.pt"
RLTT_FEATURES = ROOT / "artifacts/reports/evaluator/hh_layer_states_200_rltt.pt"
PREANSWER_RAW = ROOT / "artifacts/reports/proto_introspection/within_domain_recapture.pt"
ORIGINAL_ANALYSIS = ROOT / "utilities/tests/manual/proto_introspection_within_domain_analysis.py"
ORIGINAL_UTILS = ROOT / "utilities/tests/manual/proto_introspection_controls_analysis.py"


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict[str, Any]:
    import transformers

    dev = "cpu"
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": dev,
        "dtype": "torch.bfloat16 model forward; captured/analysis features torch.float32",
        "quantization": "none",
    }


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


def capture_base(args: argparse.Namespace) -> None:
    """Capture only pooled L47 boundary states for the fixed 200-pair slice."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output).resolve()
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = str(RLTT_PATH)

    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    rng = np.random.default_rng(PROBE_SEED)
    indices = sorted(rng.permutation(len(ds))[: args.max_examples].tolist())
    examples = [{"dataset_index": int(i), "chosen": ds[int(i)]["chosen"], "rejected": ds[int(i)]["rejected"]} for i in indices]

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading {BASE_MODEL_ID}@{BASE_REVISION} from project cache {cache}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_REVISION,
        cache_dir=str(cache),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.early_exit_threshold = 1.0
    if int(model.config.num_hidden_layers) != 48 or int(model.config.total_ut_steps) != 4:
        raise RuntimeError(f"unexpected base config: layers={model.config.num_hidden_layers}, loops={model.config.total_ut_steps}")

    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal captured
        captured = [h.detach() for h in output[1]]

    handle = model.model.register_forward_hook(hook)

    def run_side(text: str) -> torch.Tensor:
        nonlocal captured
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        captured = []
        with torch.no_grad():
            model(**enc, use_cache=False)
        if len(captured) != 4:
            raise RuntimeError(f"expected 4 boundary loop states, got {len(captured)}")
        pooled = [_mean_pool(h.float(), enc["attention_mask"]).squeeze(0).cpu() for h in captured]
        return torch.stack(pooled)

    packs = []
    started = time.time()
    try:
        for j, ex in enumerate(examples):
            packs.append({
                "dataset_index": ex["dataset_index"],
                "chosen_pooled_l47": run_side(ex["chosen"]),
                "rejected_pooled_l47": run_side(ex["rejected"]),
            })
            if (j + 1) % 20 == 0 or j + 1 == len(examples):
                print(f"base capture {j+1}/{len(examples)} elapsed={time.time()-started:.1f}s", flush=True)
    finally:
        handle.remove()

    payload = {
        "meta": {
            "model_id": BASE_MODEL_ID,
            "revision": BASE_REVISION,
            "tokenizer_path": tokenizer_path,
            "dataset": "Anthropic/hh-rlhf",
            "split": "test",
            "dataset_selection_seed": PROBE_SEED,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "feature": "mean-pooled post-final-norm L47 boundary, loops 1..4",
            "feature_shape_per_side": [4, 2048],
            "early_exit_threshold": 1.0,
            "elapsed_seconds": time.time() - started,
            "environment": environment(),
            "cache_dir": str(cache),
        },
        "packs": packs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Wrote {out}", flush=True)


def load_full_boundary(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    chosen, rejected = [], []
    for pack in payload["packs"]:
        if "chosen_pooled_l47" in pack:
            c, r = pack["chosen_pooled_l47"], pack["rejected_pooled_l47"]
        else:
            c = pack["chosen"]["pooled"][47]
            r = pack["rejected"]["pooled"][47]
        chosen.append(torch.cat([c[i].float() for i in range(4)]))
        rejected.append(torch.cat([r[i].float() for i in range(4)]))
    return torch.stack(chosen), torch.stack(rejected), payload.get("meta", {})


def fit_bias_free_lbfgs(diffs: torch.Tensor, train_rows: np.ndarray, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    # Mirrors sklearn C=1 scaling: mean logistic loss + ||w||^2/(2*C*n).
    w = torch.zeros(diffs.shape[1], dtype=torch.float32, requires_grad=True)
    x = torch.cat([diffs, -diffs], dim=0)
    n_train = len(train_rows)
    opt = torch.optim.LBFGS(
        [w], lr=1.0, max_iter=200, tolerance_grad=1e-7,
        tolerance_change=1e-9, line_search_fn="strong_wolfe"
    )
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        opt.zero_grad()
        margin = labels[train_rows] * (x[train_rows] @ w)
        loss = F.softplus(-margin).mean() + 0.5 * (w @ w) / n_train
        loss.backward()
        return loss

    final_loss = float(opt.step(closure).detach())
    return w.detach(), {"closure_calls": calls, "optimizer_return_loss": final_loss, "weight_norm": float(w.detach().norm())}


def bootstrap_accuracy(correct: np.ndarray, cluster_ids: np.ndarray, seed: int, draws: int = 10000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(cluster_ids)
    values = np.empty(draws, dtype=np.float64)
    by_cluster = [correct[cluster_ids == u] for u in unique]
    for b in range(draws):
        sampled = rng.integers(0, len(unique), len(unique))
        vals = np.concatenate([by_cluster[i] for i in sampled])
        values[b] = vals.mean()
    return {
        "method": "percentile bootstrap clustered by source HH pair",
        "seed": seed,
        "draws": draws,
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


def localization(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_features).resolve()
    sources = {
        "Base Ouro 2.6B": base_path,
        "Ouro 2.6B Thinking": THINKING_FEATURES,
        "Ouro RLTT local": RLTT_FEATURES,
    }
    loaded: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
    failures = {}
    for name, path in sources.items():
        try:
            loaded[name] = load_full_boundary(path)
        except Exception as exc:
            failures[name] = {"path": str(path), "error_type": type(exc).__name__, "error": str(exc)}
    if "Ouro 2.6B Thinking" not in loaded:
        raise RuntimeError(f"Thinking reconstruction source unavailable: {failures.get('Ouro 2.6B Thinking')}")

    c_t, r_t, _ = loaded["Ouro 2.6B Thinking"]
    if len(c_t) != 200 or c_t.shape[1] != 8192:
        raise RuntimeError(f"unexpected Thinking feature shape {tuple(c_t.shape)}")
    diffs_t = c_t - r_t
    labels = torch.cat([torch.ones(200), -torch.ones(200)]).float()
    permutation = np.random.RandomState(PROBE_SEED).permutation(400)
    split = int(0.8 * len(permutation))
    train_rows, eval_rows = permutation[:split], permutation[split:]
    w, opt_meta = fit_bias_free_lbfgs(diffs_t, train_rows, labels)
    probe_path = out_dir / "reconstructed_antisym_linear_probe.pt"
    torch.save({
        "weight": w,
        "bias": None,
        "status": "RECONSTRUCTED_LINEAR_PROBE",
        "basis": "concat mean-pooled post-final-norm L47 boundary loops 1..4",
        "normalization": "NoNorm/raw candidate difference",
        "feature_dim": 8192,
        "probe_seed": PROBE_SEED,
        "train_rows": train_rows.tolist(),
        "eval_rows": eval_rows.tolist(),
        "optimizer": "torch.optim.LBFGS full batch, strong_wolfe",
        "regularization": "mean logistic + ||w||^2/(2*C*n), C=1",
        "optimizer_meta": opt_meta,
    }, probe_path)

    rows_out = []
    results = []
    for name, (chosen, rejected, meta) in loaded.items():
        if len(chosen) != 200 or chosen.shape[1] != 8192:
            failures[name] = {"path": str(sources[name]), "error": f"shape mismatch {tuple(chosen.shape)}"}
            continue
        diffs = chosen - rejected
        x = torch.cat([diffs, -diffs], dim=0)
        scores_ab = x[eval_rows] @ w
        scores_ba = -scores_ab
        target = labels[eval_rows]
        correct = ((target * scores_ab) > 0).cpu().numpy().astype(np.float64)
        source_pair = (eval_rows % 200).astype(np.int64)
        swap_error = torch.max(torch.abs(scores_ab + scores_ba)).item()
        swap_consistency = float((torch.sign(scores_ba) == -torch.sign(scores_ab)).float().mean())
        ci = bootstrap_accuracy(correct, source_pair, SEED, args.bootstrap_draws)
        unique_pairs = np.unique(source_pair)
        canonical_correct = ((diffs[unique_pairs] @ w) > 0).float()
        result = {
            "backbone": name,
            "model_path_or_id": meta.get("model_id") or meta.get("model_path") or str(sources[name]),
            "revision": meta.get("revision") or (THINKING_REVISION if name == "Ouro 2.6B Thinking" else None),
            "n_ordered_eval_rows": len(eval_rows),
            "n_unique_source_pairs": int(len(unique_pairs)),
            "feature_shape": [len(chosen), chosen.shape[1]],
            "probe_path": str(probe_path),
            "probe_status": "RECONSTRUCTED_LINEAR_PROBE",
            "strict_antisym_accuracy": float(correct.mean()),
            "strict_antisym_accuracy_ci95": ci["ci95"],
            "unique_pair_canonical_accuracy_diagnostic": float(canonical_correct.mean()),
            "swap_consistency": swap_consistency,
            "max_abs_swap_sum": swap_error,
            "fixed_order_accuracy_diagnostic": None,
            "notes": "Same bias-free weight and original-style held-out ordered rows; CI clusters duplicate orientations by source pair.",
        }
        results.append(result)
        for j, row_idx in enumerate(eval_rows):
            rows_out.append({
                "backbone": name,
                "eval_row": int(row_idx),
                "source_pair": int(row_idx % 200),
                "orientation": "chosen_minus_rejected" if row_idx < 200 else "rejected_minus_chosen",
                "target_sign": float(target[j]),
                "score_ab": float(scores_ab[j]),
                "score_ba": float(scores_ba[j]),
                "correct": int(correct[j]),
            })

    with (out_dir / "localization_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else ["backbone"])
        writer.writeheader()
        writer.writerows(rows_out)
    json_dump(out_dir / "localization_results.json", {
        "probe_recovery": {
            "status": "RECONSTRUCTED_LINEAR_PROBE",
            "saved_weights_found": False,
            "documented_source_code": str(ROOT / "probe_test.py"),
            "source_code_recovery": "tracked file deleted in worktree; inspected with git show HEAD:probe_test.py",
            "basis": "raw NoNorm difference of mean-pooled post-final-norm L47 boundary loops 1..4, concatenated",
            "feature_dim": 8192,
            "bias": False,
            "exact_antisymmetry": True,
            "dataset": "Anthropic/hh-rlhf test; fixed 200-pair seeded slice from saved captures",
            "split": "original-style symmetric +/- rows; numpy RandomState(42), 80/20 row split",
            "known_split_caveat": "the +/- orientations are split at row level, so opposite orientations of a source pair can cross train/eval; CI clusters by source pair",
            "probe_path": str(probe_path),
            "optimizer_meta": opt_meta,
        },
        "results": results,
        "failures": failures,
        "environment": environment(),
    })
    print(json.dumps({"results": results, "failures": failures}, indent=2), flush=True)


def _auc_weighted_presorted(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    s, y, w = scores[order], labels[order], weights[order]
    total_pos = float(w[y == 1].sum())
    total_neg = float(w[y == 0].sum())
    if total_pos <= 0 or total_neg <= 0:
        return float("nan")
    wins = 0.0
    cum_neg = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        wp = float(w[i:j][y[i:j] == 1].sum())
        wn = float(w[i:j][y[i:j] == 0].sum())
        wins += wp * (cum_neg + 0.5 * wn)
        cum_neg += wn
        i = j
    return wins / (total_pos * total_neg)


def _bootstrap_delta(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
                     cluster_index: np.ndarray, n_clusters: int, draws: int,
                     seed: int, clustered: bool) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    vals = []
    skipped = 0
    n = len(labels)
    for _ in range(draws):
        if clustered:
            counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
            weights = counts[cluster_index].astype(np.float64)
        else:
            weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        aa = _auc_weighted_presorted(scores_a, labels, weights)
        ab = _auc_weighted_presorted(scores_b, labels, weights)
        if not math.isfinite(aa) or not math.isfinite(ab):
            skipped += 1
            continue
        vals.append(aa - ab)
    return np.asarray(vals, dtype=np.float64), skipped


def preanswer_ci(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT / "utilities/tests/manual"))
    import proto_introspection_controls_analysis as C

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(PREANSWER_RAW, map_location="cpu", weights_only=False)
    recs = payload["records"]["gsm8k"]
    x_hid, x_len, x_lp, ys, groups, samples = [], [], [], [], [], []
    for r in recs:
        for s in r["samples"]:
            if not s.get("has_preanswer") or "preanswer_feat" not in s:
                continue
            lp = s.get("lp_mean_logprob")
            if lp is None or (isinstance(lp, float) and math.isnan(lp)):
                continue
            x_hid.append(s["preanswer_feat"].flatten().float())
            x_len.append(torch.tensor([r["question_chars"] / 1000.0, r["prompt_tok"] / 512.0, s["n_pre_tok"] / 256.0]))
            x_lp.append(torch.tensor([float(lp), float(s["lp_mean_entropy"]), float(s["lp_last_entropy"])]))
            ys.append(1 if s["correct"] else 0)
            groups.append(str(r["task_id"]))
            samples.append(int(s["sample"]))
    Xhid = torch.stack(x_hid)
    Xsc = torch.cat([torch.stack(x_len), torch.stack(x_lp)], dim=1)
    y = torch.tensor(ys).long()
    print(f"Recomputing grouped-CV OOF predictions: n={len(y)}, groups={len(set(groups))}", flush=True)
    score_composite = C.cv_oof_scores(Xsc, y, groups, k_pca=6, l2=1.0).double()
    score_hidden_all = C.cv_oof_combined(Xhid, Xsc, y, groups, k_pca=24, l2=2.0).double()
    labels = y.numpy().astype(np.int64)
    s_a = score_hidden_all.numpy()
    s_b = score_composite.numpy()
    point_a = _auc_weighted_presorted(s_a, labels, np.ones(len(y)))
    point_b = _auc_weighted_presorted(s_b, labels, np.ones(len(y)))
    point_delta = point_a - point_b
    unique_groups = sorted(set(groups))
    group_to_i = {g: i for i, g in enumerate(unique_groups)}
    cluster_index = np.asarray([group_to_i[g] for g in groups], dtype=np.int64)
    vals, skipped = _bootstrap_delta(s_a, s_b, labels, cluster_index, len(unique_groups), args.bootstrap_draws, args.seed, True)
    candidate_vals, candidate_skipped = _bootstrap_delta(s_a, s_b, labels, cluster_index, len(unique_groups), args.bootstrap_draws, args.seed, False)

    # Leave-one-task-out influence on fixed OOF predictions (diagnostic).
    loo = []
    for gi, gid in enumerate(unique_groups):
        keep = cluster_index != gi
        d = _auc_weighted_presorted(s_a[keep], labels[keep], np.ones(int(keep.sum()))) - _auc_weighted_presorted(s_b[keep], labels[keep], np.ones(int(keep.sum())))
        loo.append({"task_id": gid, "delta_without_task": d, "change_from_full": d - point_delta})

    with (out_dir / "preanswer_oof_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["task_id", "sample", "label", "length_logprob_score", "hidden_all_score"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(y)):
            writer.writerow({"task_id": groups[i], "sample": samples[i], "label": int(labels[i]), "length_logprob_score": float(s_b[i]), "hidden_all_score": float(s_a[i])})
    with (out_dir / "preanswer_leave_one_task_out.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(loo[0].keys()))
        writer.writeheader()
        writer.writerows(loo)

    def summarize(v: np.ndarray, skip: int, method: str) -> dict[str, Any]:
        return {
            "method": method,
            "draws_requested": args.bootstrap_draws,
            "draws_valid": int(len(v)),
            "draws_skipped_degenerate": int(skip),
            "seed": args.seed,
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "ci95_percentile": np.quantile(v, [0.025, 0.975]).tolist(),
            "excludes_zero": bool(np.quantile(v, 0.025) > 0 or np.quantile(v, 0.975) < 0),
        }

    result = {
        "raw_artifact": str(PREANSWER_RAW),
        "raw_artifact_sha256": sha256(PREANSWER_RAW),
        "original_analysis_script": str(ORIGINAL_ANALYSIS),
        "original_bootstrap_utility": str(ORIGINAL_UTILS),
        "n_tasks": len(unique_groups),
        "n_examples": len(y),
        "class_balance": {"positive": int(y.sum()), "negative": int((1-y).sum())},
        "split": "5-fold task-grouped CV; exact existing grouped_folds seed 20260617",
        "point_estimates": {"hidden_plus_all_auroc": point_a, "length_plus_logprob_auroc": point_b, "delta": point_delta},
        "task_clustered_bootstrap": summarize(vals, skipped, "paired task-clustered percentile bootstrap; sample 170 tasks with replacement and include all 4 candidates per sampled task"),
        "candidate_bootstrap_diagnostic_only": summarize(candidate_vals, candidate_skipped, "paired candidate-level percentile bootstrap (diagnostic only; not primary)"),
        "leave_one_task_out": {
            "min_delta": min(x["delta_without_task"] for x in loo),
            "max_delta": max(x["delta_without_task"] for x in loo),
            "max_abs_change": max(abs(x["change_from_full"]) for x in loo),
            "table": str(out_dir / "preanswer_leave_one_task_out.csv"),
        },
        "environment": environment(),
    }
    json_dump(out_dir / "preanswer_task_clustered_ci.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture-base")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default=str(ROOT / "artifacts/hf_cache"))
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=capture_base)
    p = sub.add_parser("localization")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-features", required=True)
    p.add_argument("--bootstrap-draws", type=int, default=10000)
    p.set_defaults(func=localization)
    p = sub.add_parser("preanswer-ci")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap-draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(func=preanswer_ci)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    args.func(args)


if __name__ == "__main__":
    main()
