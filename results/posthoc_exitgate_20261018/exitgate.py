"""Post hoc (2026-10-18): which loop did the threshold-1.0 exit rule pick when Runs 4 and 8 generated?

For a fixed-seed sample of generated answers, one teacher-forced forward of prompt + generated tokens through the same
extractor the runs generated with (BGTransformerFeatureExtractor, force_all_loops=True, bf16, the same model files).
A hook on the inner model reads the hidden_states_list and gate_list that forward computes; the exit rule is applied to
them with the code copied from the model file (same ops, same dtype). Checks:
  - forward's own choice: which loop's lm_head logits equal, bit for bit, the logits the forward returned;
  - stored log-probs: gen_logprobs were log_softmax of the processed scores (the run's generate() processors) of the
    loop actually used; recompute that from each loop's logits and see which loop reproduces the stored value.
Usage: exitgate.py --run 4|8 [--n 200]. Writes results_run{4,8}.json next to this file. Never writes to the SSD.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or str(HERE / "hf_unset")  # caller sets it; never the shared cache
import torch  # noqa: E402

torch.set_num_threads(4)
ROOT = HERE.parents[1]
SSD = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer")
RUNS = {4: ("horizon_cohort2_20261003", "gen_run4"), 8: ("horizon_thinking_20261010", "gen_run8")}
SAMPLE_SEED = 20261018
EARLY = (32, 48)


def load_gen_module(run: int):
    d, name = RUNS[run]
    spec = importlib.util.spec_from_file_location(name, ROOT / "opi/preanswer" / d / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_records(run: int, n: int) -> list[dict]:
    d = SSD / RUNS[run][0]
    pool = []
    for p in sorted(d.glob(f"run{run}_shard_*.pt")):
        for i, r in enumerate(torch.load(p, map_location="cpu", weights_only=True)["records"]):
            if r["gen_ids"]:
                pool.append({"shard": p.name, "idx": i, "task_uid": r["task_uid"], "candidate_idx": r["candidate_idx"],
                             "prompt_ids": r["prompt_ids"], "gen_ids": r["gen_ids"], "gen_logprobs": r["gen_logprobs"]})
    picks = sorted(random.Random(SAMPLE_SEED).sample(range(len(pool)), n))
    return [pool[i] for i in picks], len(pool)


def exit_steps_from_gates(gate_list, exit_threshold):
    """Copied from OuroForCausalLM.forward (identical in the RLTT file and the Thinking shim)."""
    pdf_list = []
    remaining_prob = torch.ones_like(gate_list[0].squeeze(-1))
    for idx, gate_tensor in enumerate(gate_list):
        lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))
        if idx < len(gate_list) - 1:
            p_i = lambda_i * remaining_prob
            remaining_prob = remaining_prob * (1.0 - lambda_i)
        else:
            p_i = remaining_prob
        pdf_list.append(p_i)
    stacked_exit_pdf = torch.stack(pdf_list, dim=2)
    cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
    threshold_mask = cumulative_probs >= exit_threshold
    exit_steps = torch.argmax(threshold_mask.float(), dim=2)
    never_exceeded = ~threshold_mask.any(dim=2)
    exit_steps[never_exceeded] = stacked_exit_pdf.shape[2] - 1
    return exit_steps, cumulative_probs


def build_processors(model, tok, plen, device):
    """The logits processors generate() built in the runs (same kwargs as gen_run4/gen_run8)."""
    from transformers import LogitsProcessorList
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc, mk = model._prepare_generation_config(None, max_new_tokens=448, do_sample=True, temperature=0.7, top_p=0.95,
                                              num_return_sequences=4, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                                              use_cache=True, output_scores=True, return_dict_in_generate=True)
    model._prepare_special_tokens(gc, True, device=device)
    procs = model._get_logits_processor(generation_config=gc, input_ids_seq_length=plen, encoder_input_ids=None,
                                        prefix_allowed_tokens_fn=None, logits_processor=LogitsProcessorList(),
                                        device=device, model_kwargs=mk)
    names = [type(p).__name__ for p in procs]
    # position-independent warpers only, so they can be applied to all positions of one answer at once
    assert names == ["TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"], names
    return procs, {"processors": names, "temperature": gc.temperature, "top_k": gc.top_k, "top_p": gc.top_p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, choices=(4, 8), required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    # self-check of the copied rule in bf16: gate logit 10 at loop 1 -> sigmoid rounds to 1.0 -> exit at loop 1;
    # gate logit 7 at loop 2 (sigmoid 0.9991 -> 1.0 in bf16) -> loop 2; all zeros -> last loop
    chk = torch.tensor([[10., -9, -9, -9], [-9, 7, -9, -9], [0, 0, 0, 0]], dtype=torch.bfloat16)
    assert exit_steps_from_gates([chk[None, :, k, None] for k in range(4)], 1.0)[0].tolist() == [[0, 1, 3]]
    gm = load_gen_module(a.run)
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
    model_sha = None
    if a.run == 8:
        model_sha = gm.ensure_model()
        ext = BGTransformerFeatureExtractor(model_path=gm.MODEL, device="cuda", dtype="auto", force_all_loops=True)
    else:
        ext = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model, tok = ext.model, ext.tokenizer
    thr = getattr(model, "early_exit_threshold", None) if a.run == 8 else model.config.early_exit_threshold
    # shim reads self.early_exit_threshold (set at load from config.json); RLTT reads config.early_exit_threshold
    exit_step = getattr(model, "early_exit_step", None) if a.run == 8 else getattr(model.config, "early_exit_step", None)
    setup = {"run": a.run, "model_path": str(ext.model_path), "model_sha256": model_sha, "dtype": str(ext.dtype),
             "attn_implementation": model.config._attn_implementation, "exit_threshold_used": thr,
             "early_exit_step": exit_step, "total_ut_steps": model.model.total_ut_steps,
             "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    import transformers
    setup["transformers"] = transformers.__version__
    assert thr == 1.0 and exit_step is None and model.model.total_ut_steps == 4, setup
    print(json.dumps(setup), flush=True)

    recs, pool_n = sample_records(a.run, a.n)
    print(f"[exitgate] run {a.run}: sampled {len(recs)} of {pool_n} answers, {time.time() - t0:.0f}s", flush=True)
    procs = None
    cap = {}
    h = model.model.register_forward_hook(lambda _m, _i, out: cap.update(hs=out[1], gates=out[2]))
    per_answer, all_exit, all_fwd, all_lpbest, all_lpdiff_gate, all_amb, all_gates, all_t = [], [], [], [], [], [], [], []
    all_cum = []
    for ai, r in enumerate(recs):
        ids, plen, n = r["prompt_ids"] + r["gen_ids"], len(r["prompt_ids"]), len(r["gen_logprobs"])
        if procs is None:
            procs, proc_info = build_processors(model, tok, plen, ext.device)
        x = torch.tensor([ids], device=ext.device)
        with torch.inference_mode():
            out = model(input_ids=x, use_cache=False)
            hs, gates = cap["hs"], cap["gates"]
            assert len(hs) == 4 and len(gates) == 4 and gates[0].dtype == torch.bfloat16
            exit_steps, cum = exit_steps_from_gates(gates, thr)
            pos = torch.arange(plen - 1, plen - 1 + n, device=ext.device)  # logits at pos produced gen token pos-plen+1
            gate_rule = exit_steps[0, pos]
            # forward's own choice: the loop whose lm_head logits equal the returned logits bit for bit
            loop_logits = [model.lm_head(hh)[0, pos] for hh in hs]  # full-sequence lm_head, as the forward does
            ret = out.logits[0, pos]
            eq = torch.stack([(ll == ret).all(-1) for ll in loop_logits], -1)  # [n, 4]
            fwd = torch.where(eq.any(-1), eq.float().argmax(-1), torch.full_like(gate_rule, -1))
            # stored log-prob check: processed scores from each loop's logits (float32, as generate casts)
            tgt = torch.tensor(r["gen_ids"][:n], device=ext.device)
            lps = []
            for ll in loop_logits:
                sc = procs(x[:, :plen].expand(n, -1), ll.float())
                lps.append(torch.log_softmax(sc, -1).gather(-1, tgt[:, None])[:, 0])
            lps = torch.stack(lps, -1)  # [n, 4]
            stored = torch.tensor(r["gen_logprobs"], device=ext.device)
            diff = (lps - stored[:, None]).abs()  # inf where the token is filtered out under that loop
            best = diff.argmin(-1)
            amb = diff.sort(-1).values[:, :2]  # best and second-best |diff|, for the identifiability rules below
            g = torch.stack([gg[0, pos, 0] for gg in gates], -1)  # [n, 4] bf16 gate logits
        er = gate_rule.cpu()
        early_pos = (er < 3).nonzero().flatten().tolist()
        per_answer.append({"shard": r["shard"], "idx": r["idx"], "task_uid": r["task_uid"], "candidate_idx": r["candidate_idx"],
                           "n_tok": n, "loop_counts": [int((er == k).sum()) for k in range(4)], "early_positions": early_pos,
                           "fwd_matches_gate_rule": bool((fwd.cpu() == er).all()),
                           # per-loop mean |stored - recomputed| log-prob over the answer's tokens, each capped at 5
                           # nats (a token filtered out under a loop gives inf)
                           "lp_mean_absdiff_cap5_per_loop": [float(v) for v in diff.clamp(max=5.0).mean(0).cpu()]})
        all_exit.append(er); all_fwd.append(fwd.cpu()); all_lpbest.append(best.cpu()); all_amb.append(amb.cpu())
        all_lpdiff_gate.append(diff.gather(-1, gate_rule[:, None])[:, 0].cpu()); all_gates.append(g.cpu()); all_t.append(torch.arange(n))
        all_cum.append(cum[0, pos, :3].cpu())
        del out, cap["hs"], cap["gates"], loop_logits, ret, lps
        if (ai + 1) % 20 == 0:
            print(f"[exitgate] {ai + 1}/{len(recs)} answers, {time.time() - t0:.0f}s", flush=True)
    h.remove()

    E, F, B, A, D, G, T, C = (torch.cat(v) for v in (all_exit, all_fwd, all_lpbest, all_amb, all_lpdiff_gate, all_gates, all_t, all_cum))
    lam = torch.sigmoid(G)  # bf16, as in the forward
    one_below = torch.tensor(1.0, dtype=torch.bfloat16).nextafter(torch.tensor(0.0, dtype=torch.bfloat16))
    ntok = len(E)
    # identifiable: the best loop beats the second best by >= 0.05 nats (strict: best within 0.01 and second >= 0.1 away)
    ident = A[:, 0].isfinite() & (A[:, 1] - A[:, 0] >= 0.05)
    strict = (A[:, 0] <= 0.01) & (A[:, 1] >= 0.1)
    gq = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    res = {
        "post_hoc": True, "date": "2026-10-18", "setup": setup, "processing": proc_info, "sample_seed": SAMPLE_SEED,
        "pool_answers": pool_n, "n_answers": len(recs), "n_tokens": ntok,
        "gate_rule_loop_fraction": {f"loop{k + 1}": float((E == k).float().mean()) for k in range(4)},
        "gate_rule_loop_counts": {f"loop{k + 1}": int((E == k).sum()) for k in range(4)},
        "answers_with_any_early_exit": sum(1 for p in per_answer if p["early_positions"]) / len(per_answer),
        "early_exit_by_position": {
            **{f"first{k}_tokens": {"early_tokens": int(((E < 3) & (T < k)).sum()), "tokens": int((T < k).sum()),
                                    "early_fraction": float(((E < 3) & (T < k)).sum() / max(1, (T < k).sum()))} for k in EARLY},
            **{f"after{k}_tokens": {"early_tokens": int(((E < 3) & (T >= k)).sum()), "tokens": int((T >= k).sum()),
                                    "early_fraction": float(((E < 3) & (T >= k)).sum() / max(1, (T >= k).sum()))} for k in EARLY},
            "share_of_early_exits_in_first32": float(((E < 3) & (T < 32)).sum() / max(1, (E < 3).sum())),
            "share_of_early_exits_in_first48": float(((E < 3) & (T < 48)).sum() / max(1, (E < 3).sum())),
        },
        "forward_choice_check": {"tokens_where_returned_logits_match_gate_rule_loop": int((F == E).sum()),
                                 "tokens_with_no_bitwise_loop_match": int((F < 0).sum()), "n_tokens": ntok},
        "logprob_check": {
            "identifiable_tokens": int(ident.sum()),
            "identifiable_note": "a token is identifiable when the best-matching loop beats the second best by >= 0.05 nats",
            "agree_with_gate_rule_on_identifiable": float((B[ident] == E[ident]).float().mean()) if ident.any() else None,
            "lp_best_loop_counts_identifiable": {f"loop{k + 1}": int((B[ident] == k).sum()) for k in range(4)},
            "strict_identifiable_tokens": int(strict.sum()),
            "agree_with_gate_rule_on_strict": float((B[strict] == E[strict]).float().mean()) if strict.any() else None,
            "lp_best_loop_counts_strict": {f"loop{k + 1}": int((B[strict] == k).sum()) for k in range(4)},
            "gate_rule_loop_absdiff_quantiles_q50_q90_q99_maxfinite": [float(v) for v in torch.quantile(D[D.isfinite()].double(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.double))] + [float(D[D.isfinite()].max())],
            "gate_rule_loop_absdiff_le_0.05": float((D <= 0.05).float().mean()),
            "gate_rule_loop_token_filtered_out(inf)": int((~D.isfinite()).sum()),
            "answers_where_loop_with_lowest_mean_absdiff_is": {
                f"loop{k + 1}": sum(1 for p in per_answer if min(range(4), key=lambda j: p["lp_mean_absdiff_cap5_per_loop"][j]) == k)
                for k in range(4)},
            "answers_mean_absdiff_per_loop": {
                f"loop{k + 1}": float(torch.tensor([p["lp_mean_absdiff_cap5_per_loop"][k] for p in per_answer]).mean()) for k in range(4)},
            "early_exit_tokens": {"n": int((E < 3).sum()), "identifiable": int((ident & (E < 3)).sum()),
                                  "lp_best_equals_gate_rule": int((ident & (E < 3) & (B == E)).sum()),
                                  "lp_best_is_loop4": int((ident & (E < 3) & (B == 3)).sum())},
        },
        "gates": {f"loop{k + 1}": {"sigmoid_exactly_1": float((lam[:, k] == 1.0).float().mean()),
                                   "sigmoid_one_ulp_below_1": float((lam[:, k] == one_below).float().mean()),
                                   "gate_logit_q01_q10_q50_q90_q99": [float(v) for v in torch.quantile(G[:, k].float(), gq)],
                                   "gate_logit_max": float(G[:, k].max()), "sigmoid_max": float(lam[:, k].max())}
                  for k in range(4)},
        "cumulative_exit_prob_max_bf16": {f"after_loop{k + 1}": float(C[:, k].max()) for k in range(3)},
        "runtime_seconds": round(time.time() - t0, 1),
        "per_answer": per_answer,
    }
    (HERE / f"results_run{a.run}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_answer"}, indent=1), flush=True)
    ext.cleanup()


if __name__ == "__main__":
    main()
