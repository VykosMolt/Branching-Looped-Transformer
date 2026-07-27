"""Frozen conversion #2 -- matched-budget prefix-prune tournament.

Sealed plan: artifacts/reports/tournament_v4_20260727/EXPERIMENT_PLAN.md.
Stage generate: one batched 6-candidate sample per task run to completion; pooled
features for the P=80-token prefix cut and the full text. Stage analyze: simulate
tournament / matched best-of-3 / random-prune policies over the same table with
exact token accounting; frozen DualAnchor locked channels are the only scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    _sid, split_for, load_task_pool, options_dict, preanswer_prefix,
    FinalAnswerStop, token_cut_index,
)

OUT_ROOT = PROJECT_ROOT / "artifacts/reports/tournament_v4_20260727"
SEED = 20260727
TASK_OFFSET, N_TASKS = 1080, 180
B = 6            # branches sampled per task
ROUND_B = 3      # branches per generate call (AMENDMENT 2: 2 rounds of 3, OOM fix)
ROUNDS_PER_TASK = B // ROUND_B
S = 2            # survivors kept by the tournament
P = 80           # prefix cut in generated tokens
MAX_NEW = 320
N_RANDOM_REPLICATES = 500
ROUNDS = 2000


# ---------------------------------------------------------------- frozen scorer
def load_channels():
    from bg_core_tap_audit_v1_common import dualanchor_role_channels
    chans = []
    for role in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL"):
        chans.extend(dualanchor_role_channels(role))
    assert len(chans) == 6, f"expected 6 locked channels, got {len(chans)}"
    return chans


def config_vec(pooled: torch.Tensor, config: str) -> torch.Tensor:
    from src.evaluator.bg_controller import config_vector
    return config_vector(pooled, config)


def cohort_scores(feats: list[torch.Tensor], channels) -> list[float]:
    """Per-candidate score: per-channel mean antisymmetric margin vs cohort,
    z-scored per channel within the cohort, averaged across channels."""
    from bg_merged_tap_v1_common import score_diff
    n = len(feats)
    if n == 1:
        return [0.0]
    per_chan = []
    for (w, arch, cfg) in channels:
        vecs = [config_vec(f, cfg).flatten().float() for f in feats]
        margins = []
        for i in range(n):
            vals = [score_diff(w, arch, vecs[i] - vecs[j]) for j in range(n) if j != i]
            vals = [v for v in vals if v == v]  # drop NaN
            margins.append(sum(vals) / len(vals) if vals else 0.0)
        t = torch.tensor(margins)
        sd = float(t.std())
        t = (t - t.mean()) / (sd if sd > 1e-9 else 1.0)
        per_chan.append(t)
    return torch.stack(per_chan).mean(dim=0).tolist()


# ---------------------------------------------------------------- generate stage
def stage_generate() -> None:
    assert TASK_OFFSET >= 1080, "sealed: slice must not overlap prior runs"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pool = load_task_pool()
    pool.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    tasks = pool[TASK_OFFSET: TASK_OFFSET + N_TASKS]
    print(f"[tour-gen] {len(tasks)} tasks, slice [{TASK_OFFSET},{TASK_OFFSET + N_TASKS})")

    t0 = time.time()
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    print(f"[tour-gen] model loaded in {time.time() - t0:.1f}s")

    records, errors = [], []
    for ti, task in enumerate(tasks):
        try:
            opts = options_dict(task["options"])
            gold_letter = chr(65 + int(task["answer_key"]))
            prompt = mcq_prompt(task["task_prompt"], opts)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
            enc = {kk: vv.to(extractor.device) for kk, vv in enc.items()}
            plen = int(enc["input_ids"].shape[1])
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            # AMENDMENT 2 (pre-data, recorded in the sealed plan): B=6 drawn as
            # ROUNDS_PER_TASK seeded rounds of ROUND_B -- a single k=6 batch OOMs
            # the 12GB card. Sampling params unchanged; per-round seed is
            # _sid(SEED, task_uid, r).
            seqs = []
            for rnd in range(ROUNDS_PER_TASK):
                seed_r = _sid(SEED, task["task_uid"], rnd)
                torch.manual_seed(seed_r)
                if extractor.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed_r)
                stopper = StoppingCriteriaList([FinalAnswerStop(tok, plen, ROUND_B)])
                with torch.inference_mode():
                    out = model.generate(
                        **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7, top_p=0.95,
                        num_return_sequences=ROUND_B, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                        use_cache=True, stopping_criteria=stopper,
                    )
                seqs.extend(out.sequences.tolist() if hasattr(out, "sequences") else out.tolist())
                del out
                if extractor.device.type == "cuda":
                    torch.cuda.empty_cache()

            for ci in range(B):
                new_ids = seqs[ci][plen:]
                while new_ids and new_ids[-1] == pad_id:
                    new_ids.pop()
                text = tok.decode(new_ids, skip_special_tokens=True).strip()
                pre_text, found_marker = preanswer_prefix(text)
                parsed_raw = parse_mcq_answer(text, opts)
                parsed = parsed_raw if found_marker else None
                prefix_ids = new_ids[:P]
                prefix_text = tok.decode(prefix_ids, skip_special_tokens=True).strip()
                full_feats = extractor.encode_text_to_pooled_features(f"{prompt}\n{text}".strip())
                prefix_feats = extractor.encode_text_to_pooled_features(f"{prompt}\n{prefix_text}".strip())
                records.append({
                    "task_uid": task["task_uid"], "split": split_for(task["task_uid"]),
                    "proof_depth": task["proof_depth"], "draw_idx": ci,
                    "gold_letter": gold_letter, "generated_text": text,
                    "n_gen_tok": len(new_ids), "hit_max_tokens": len(new_ids) >= MAX_NEW,
                    "found_final_marker": bool(found_marker), "parsed_answer": parsed,
                    "success": bool(found_marker and parsed == gold_letter),
                    "malformed": bool(not found_marker or parsed is None),
                    "full_features": full_feats, "prefix_features": prefix_feats,
                    "prefix_tok": len(prefix_ids), "seed": _sid(SEED, task["task_uid"], ci // ROUND_B),
                })
            if (ti + 1) % 10 == 0 or ti == 0:
                print(f"[tour-gen] {ti + 1}/{len(tasks)} tasks elapsed={time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task['task_uid']}: {type(exc).__name__}: {exc}")
            print(f"[tour-gen] ERROR {task['task_uid']}: {exc}", flush=True)
            if extractor.device.type == "cuda":
                torch.cuda.empty_cache()

    extractor.cleanup()
    torch.save({"seed": SEED, "task_offset": TASK_OFFSET, "n_tasks": N_TASKS,
                "B": B, "P": P, "max_new": MAX_NEW, "model": "models/ouro_rltt_local",
                "n_records": len(records), "errors": errors,
                "elapsed_seconds": round(time.time() - t0, 1), "records": records},
               OUT_ROOT / "tournament_table.pt")
    print(f"[tour-gen] wrote table: {len(records)} records, {len(errors)} errors")


# ---------------------------------------------------------------- analyze stage
def stage_analyze() -> None:
    payload = torch.load(OUT_ROOT / "tournament_table.pt", map_location="cpu", weights_only=False)
    records = payload["records"]
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_uid"], []).append(r)
    tasks = sorted(tu for tu, g in by_task.items() if len(g) == B)
    print(f"[tour-an] {len(tasks)} complete tasks of {len(by_task)}")
    channels = load_channels()
    print(f"[tour-an] loaded {len(channels)} frozen DualAnchor channels: "
          f"{[(c[1], c[2]) for c in channels]}")

    def pre_cost(r): return min(r["n_gen_tok"], P)
    def cont_cost(r): return max(0, r["n_gen_tok"] - P)

    tour_win, base_win, base4_win, per_task = [], [], [], []
    oracle6, oracle3, surv_oracle = [], [], []
    tour_cost, base_cost, base4_cost = [], [], []
    prefix_scores_all, success_all = [], []
    rng = torch.Generator().manual_seed(SEED)
    rand_wins = torch.zeros(N_RANDOM_REPLICATES)

    for tu in tasks:
        g = sorted(by_task[tu], key=lambda r: r["draw_idx"])
        succ = [bool(r["success"]) for r in g]
        oracle6.append(any(succ))
        oracle3.append(any(succ[:3]))

        pscores = cohort_scores([r["prefix_features"] for r in g], channels)
        prefix_scores_all.extend(pscores)
        success_all.extend(succ)
        order = sorted(range(B), key=lambda i: -pscores[i])
        survivors = order[:S]
        surv_oracle.append(any(succ[i] for i in survivors))

        fscores_surv = cohort_scores([g[i]["full_features"] for i in survivors], channels)
        pick_t = survivors[max(range(S), key=lambda k: fscores_surv[k])]
        tour_win.append(int(succ[pick_t]))
        tour_cost.append(sum(pre_cost(r) for r in g) + sum(cont_cost(g[i]) for i in survivors))

        fscores_base = cohort_scores([g[i]["full_features"] for i in range(3)], channels)
        pick_b = max(range(3), key=lambda k: fscores_base[k])
        base_win.append(int(succ[pick_b]))
        base_cost.append(sum(g[i]["n_gen_tok"] for i in range(3)))

        fscores_b4 = cohort_scores([g[i]["full_features"] for i in range(4)], channels)
        pick_b4 = max(range(4), key=lambda k: fscores_b4[k])
        base4_win.append(int(succ[pick_b4]))
        base4_cost.append(sum(g[i]["n_gen_tok"] for i in range(4)))
        per_task.append(tu)

        # random-prune replicates: random survivors, tap-pick among them
        for rep in range(N_RANDOM_REPLICATES):
            perm = torch.randperm(B, generator=rng).tolist()
            rs = perm[:S]
            fs = cohort_scores([g[i]["full_features"] for i in rs], channels) \
                if S > 1 else [0.0]
            rand_wins[rep] += succ[rs[max(range(S), key=lambda k: fs[k])]]

    n = len(tasks)
    rand_rates = (rand_wins / n).tolist()
    rand_sorted = sorted(rand_rates)
    rand_p95 = rand_sorted[int(0.95 * len(rand_sorted))]
    tour_rate, base_rate = sum(tour_win) / n, sum(base_win) / n
    base4_rate = sum(base4_win) / n

    # paired task-clustered bootstraps: tournament - bestof3, tournament - bestof4
    rng_b = torch.Generator().manual_seed(SEED)
    diffs, diffs4 = [], []
    for _ in range(ROUNDS):
        pick = torch.randint(0, n, (n,), generator=rng_b).tolist()
        diffs.append(sum(tour_win[i] - base_win[i] for i in pick) / n)
        diffs4.append(sum(tour_win[i] - base4_win[i] for i in pick) / n)
    diffs.sort()
    diffs4.sort()
    ci_lo, ci_hi = diffs[int(0.025 * ROUNDS)], diffs[int(0.975 * ROUNDS)]
    ci4_lo, ci4_hi = diffs4[int(0.025 * ROUNDS)], diffs4[int(0.975 * ROUNDS)]

    # powered prefix-score AUROC replication
    from proto_introspection_controls_analysis import auroc
    y = torch.tensor([1.0 if s else 0.0 for s in success_all])
    prefix_auroc = auroc(torch.tensor(prefix_scores_all), y) if 0 < y.mean() < 1 else None

    ceiling = sum(oracle6) / n
    result = {
        "seed": SEED, "n_tasks": n, "B": B, "S": S, "P": P,
        "pool_ceiling_any_of_6": ceiling,
        "oracle_any_of_first3": sum(oracle3) / n,
        "survivor_oracle_retention": sum(surv_oracle) / n,
        "tournament_rate": tour_rate, "baseline_bestof3_rate": base_rate,
        "baseline_bestof4_rate": base4_rate,
        "random_prune_mean": sum(rand_rates) / len(rand_rates),
        "random_prune_p95": rand_p95,
        "paired_diff_vs_bestof3": tour_rate - base_rate,
        "paired_diff_vs_bestof3_ci95": [ci_lo, ci_hi],
        "paired_diff_vs_bestof4": tour_rate - base4_rate,
        "paired_diff_vs_bestof4_ci95": [ci4_lo, ci4_hi],
        "mean_tokens_tournament": sum(tour_cost) / n,
        "mean_tokens_bestof3": sum(base_cost) / n,
        "mean_tokens_bestof4": sum(base4_cost) / n,
        "prefix_score_final_success_auroc": prefix_auroc,
        "channels": [(c[1], c[2]) for c in channels],
    }
    spend_mismatch = (sum(tour_cost) / n) > 1.05 * (sum(base_cost) / n)
    result["spend_guard_triggered"] = spend_mismatch
    if not (0.15 <= ceiling <= 0.95):
        verdict = "POOL_CANNOT_DIFFERENTIATE"
    elif tour_rate <= rand_p95:
        verdict = "PRUNING_NOT_SIGNAL_DRIVEN"
    elif tour_rate - base_rate > 0 and ci_lo > 0:
        if spend_mismatch and tour_rate - base4_rate < 0:
            verdict = "TOURNAMENT_BEATS_BUT_SPEND_MISMATCH"
        else:
            verdict = "TOURNAMENT_BEATS_MATCHED_BASELINE"
    elif tour_rate - base_rate < 0 and ci_hi < 0:
        verdict = "TOURNAMENT_WORSE_THAN_MATCHED_BASELINE"
    else:
        verdict = "TOURNAMENT_MATCHES_MATCHED_BASELINE"
    result["verdict"] = verdict
    (OUT_ROOT / "tournament_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[tour-an] tournament {tour_rate:.3f} vs bestof3 {base_rate:.3f} "
          f"(Δ {tour_rate - base_rate:+.3f} [{ci_lo:+.3f},{ci_hi:+.3f}]) "
          f"vs bestof4 {base4_rate:.3f} (Δ {tour_rate - base4_rate:+.3f} "
          f"[{ci4_lo:+.3f},{ci4_hi:+.3f}]) random-prune p95 {rand_p95:.3f} ceiling {ceiling:.3f}")
    print(f"[tour-an] tokens/task: tournament {result['mean_tokens_tournament']:.0f} "
          f"bestof3 {result['mean_tokens_bestof3']:.0f} bestof4 {result['mean_tokens_bestof4']:.0f} "
          f"spend_guard={spend_mismatch}; prefix AUROC {prefix_auroc}")
    print(f"[tour-an] VERDICT: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "analyze"], required=True)
    a = ap.parse_args()
    stage_generate() if a.stage == "generate" else stage_analyze()
