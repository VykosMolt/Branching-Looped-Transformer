"""M+N S1.4 — the REFERENCE multi-locus inject->carry->prune->loop-back->terminal branch loop.

The never-before-built in-transformer basal-ganglia branch loop, as the CORRECTNESS REFERENCE
(expensive but exact: per-survivor branch-specific re-derivation). All substrate validated:
  - GATE 1 (α=0 re-derive plumbing) bit-exact,
  - GATE 2 (α>0 chaining: splice≡hook, re-derive≡all-hooks-live) bit-exact,
  - GATE 3 (live DualAnchor+CoreContent_v2 prune on branch texts) faithful.

MECHANISM (faithful to architecture + the locked taps):
  LOCI = 24/36/47 across 4 UT loops, schedule order -> 12 injection boundaries; terminal = L4_47.
  At each locus, EVERY current survivor forks into K children (curated per-layer directions, α):
    child = branch_specific_root_pack_v2(lineage) -> build_spliced_branch(locus, dir, α, last-token)
            -> greedy_continue -> candidate continuation text.   [re-derive on the branch's OWN trajectory]
  PRUNE the children (the two-mechanism prune):
    (1) DualAnchor = VALIDITY/survival ALONE (first prune): keep da >= validity_threshold.
    (2) CoreContent_v2 = CONTENT selection + content-threshold (never judges validity), per locus:
        lower threshold @24, higher @36/47 (optionally progressive per loop), then cap at BUDGET by content.
  Survivors carry (lineage = carried KV) and loop back at L*_47 to the next loop's loci.
  TERMINAL @L4_47: confidence-gated pick best-by-content survivor; else flag deferred.
  External verifier LABELS the selected/terminal candidates for analysis ONLY (never used for control).

STRUCTURAL GATES asserted inline:
  GATE 4 (no-original-root-reuse): a survivor's full lineage is ALWAYS passed to the re-derive; its
          length equals the number of loci it has passed (never a clean root after divergence).
  GATE 5 (lineage/splice-stack sanity): on the post-fork lineage, ancestor loci have strictly increasing
          schedule indices and all STRICTLY PRECEDE the fork locus; the just-appended entry EQUALS the
          fork locus; the full lineage is strictly increasing (so at the next locus the whole lineage
          precedes the new fork). Every boundary lies on the branch's actual trajectory.

SCOPE / STANCE (read before interpreting results):
  * This is the FULL STRESS / REFERENCE path: every survivor forks at every one of the 12 loci, so a
    terminal survivor has lineage depth 12 (12 chained α perturbations). It is NOT claimed to be the only
    sensible production policy — later ablations: 24/36-only, 36-only, nonterminal-47-only, skip-terminal-
    fork, fork-only-when-uncertain.
  * PRUNE SCORING IS ON SHORT ROLLOUT TEXT: at each locus a branch's quality is estimated by decoding a
    short continuation and re-encoding that candidate text through the hidden-state taps (gate-3 path).
    The taps remain hidden-state evaluators, but here they score the rollout/candidate representation, NOT
    the carried branch cache in-place. This is a teacher/scaffold reference — NOT a fully internal no-rollout
    selector. Do not conflate the two.
  * FIRST RUN IS RANK-ONLY (reachability question): hard thresholds default OFF (validity/content = -1e9,
    progressive off, terminal-confidence off). The prune is then: DualAnchor (validity, diagnostic only) →
    rank by CoreContent content → cap at BUDGET. The question this run answers is "can the chained branch-
    carry mechanism PRESERVE ORACLE REACHABILITY across the loop?" — NOT "did an arbitrary CC threshold
    prune the correct branch at L2_47?". Thresholds are parameterized for later ablations.
  * TERMINAL IS ANALYSIS-ONLY: locked stance = survivor-set handoff is the safe object; forced top1 is
    diagnostic. Headline metrics report oracle_over_survivors (top-k survivor oracle), selected_correct,
    defer rate — not selected correctness alone. Verifier LABELS only; never used for control.

Run on GPU (backgrounded):
  S1_N_PER=1 S1_K=2 S1_BUDGET=4 venv/bin/python utilities/tests/manual/mpn_s1_4_reference_loop.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v1_common as B1  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
import bg_autoregressive_cache_common_v1 as C  # noqa: E402
import bg_partial_cache_splice_v2_common as S  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402
from mpn_s1_2_multilocus import curated_dirs_by_layer  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")

N_PER = int(os.environ.get("S1_N_PER", "1"))
K = int(os.environ.get("S1_K", "2"))
BUDGET = int(os.environ.get("S1_BUDGET", "4"))
ALPHA = float(os.environ.get("S1_ALPHA", "0.02"))
MNT_INTER = int(os.environ.get("S1_MNT_INTER", "64"))     # continuation length for intermediate scoring
MNT_TERM = int(os.environ.get("S1_MNT_TERM", "160"))      # continuation length at terminal
# FIRST RUN = RANK-ONLY (reachability): hard gates OFF by default; scores are group-relative tournament
# z-scores, NOT calibrated absolute quality, so a hard threshold at small K/BUDGET can accidentally false-
# prune the correct branch. DualAnchor still computed + reported (diagnostic). Thresholds parameterized for
# later ablations: rank-only / DA-threshold-only / DA+CC-threshold / progressive-CC.
VALIDITY_THRESHOLD = float(os.environ.get("S1_VALIDITY_THR", "-1e9"))  # DualAnchor validity gate (OFF default)
PROGRESSIVE = os.environ.get("S1_PROGRESSIVE", "0") == "1"             # raise content threshold per loop?
TERMINAL_CONFIDENCE = float(os.environ.get("S1_TERM_CONF", "-1e9"))    # content z to commit (OFF default; no defer)
# Tap candidate_text format: the locked CoreContent taps were extracted from a MIX of candidate formats,
# but the dominant math/gsm8k tournament sources used PROMPT+ANSWER ("Problem: {q}\n\nSolution:{c}").
# Answer-only under-contextualizes math/logic/reasoning. OFF (answer-only) is acceptable for the first
# MECHANICAL/reachability run (recorded as score_input_mode); paper-grade cross-domain metrics need the
# prompt+answer format to match training. ON uses "Problem: {task_prompt}\n\nSolution: {answer}".
SCORE_WITH_PROMPT = os.environ.get("S1_SCORE_WITH_PROMPT", "0") == "1"
SCORE_INPUT_MODE = "prompt_plus_answer" if SCORE_WITH_PROMPT else "candidate_text_only"

LAYERS = (24, 36, 47)
LOCI = [(loop, layer) for loop in range(4) for layer in LAYERS]   # 12 loci, schedule order
TERMINAL = LOCI[-1]                                              # (3, 47) == L4_47
SCHED = {lc: i for i, lc in enumerate(LOCI)}

# content threshold per layer (later-ablation lean: lower@24, higher@36/47); +loop step if PROGRESSIVE.
# Default OFF (-1e9) for the rank-only first run. Ablation lean values: {24: -1e9, 36: 0.0, 47: 0.25}.
BASE_CONTENT_THR = {24: float(os.environ.get("S1_CC_THR_24", "-1e9")),
                    36: float(os.environ.get("S1_CC_THR_36", "-1e9")),
                    47: float(os.environ.get("S1_CC_THR_47", "-1e9"))}
PROG_STEP = 0.1


def content_threshold(loop: int, layer: int) -> float:
    base = BASE_CONTENT_THR[layer]
    if base <= -1e8:
        return base
    return base + (PROG_STEP * loop if PROGRESSIVE else 0.0)


@dataclass
class Branch:
    bid: str
    lineage: list = field(default_factory=list)   # [(layer, loop, dir_tensor, alpha, token_range)]
    text: str = ""
    da: float = float("nan")
    cc: float = float("nan")
    parent: str = "root"


def lineage_loci(lineage):
    return [(lp, L) for (L, lp, _d, _a, _tr) in lineage]


def assert_lineage_invariants(new_lineage, fork_loop, fork_layer, expected_len):
    """GATE 4 + GATE 5 for ONE fork. `new_lineage` INCLUDES the just-appended current-locus entry.
    Explicitly: full lineage strictly increasing in schedule order; the appended entry EQUALS the fork
    locus; every ancestor STRICTLY PRECEDES the fork locus (so at the next locus the whole lineage
    precedes the new fork)."""
    assert len(new_lineage) == expected_len, f"GATE4 fail: lineage len {len(new_lineage)} != {expected_len}"
    idxs = [SCHED[(lp, L)] for (lp, L) in lineage_loci(new_lineage)]
    fork_idx = SCHED[(fork_loop, fork_layer)]
    assert all(idxs[i] < idxs[i + 1] for i in range(len(idxs) - 1)), f"GATE5 fail: non-monotone lineage {idxs}"
    assert idxs[-1] == fork_idx, f"GATE5 fail: appended {idxs[-1]} != fork locus {fork_idx}"
    assert all(x < fork_idx for x in idxs[:-1]), f"GATE5 fail: ancestor !< fork {idxs[:-1]} vs {fork_idx}"


def parse(task, text):
    return B1.extract_code(text) if task.get("label_type") == "unit_tests" else (B1.parse_final(text) or "")


def verify(task, text, is_code):
    p = parse(task, text)
    if is_code:
        return B1.verify_code_unit_tests(p, task.get("unit_tests", []), task.get("test_setup", "")), p
    return (B1.verify_final(task, p)[1] > 0), p


def score_text_for(task, answer):
    """The text fed to the taps for one branch: answer-only (default) or prompt+answer (training format)."""
    a = answer if str(answer).strip() else "(empty)"
    if SCORE_WITH_PROMPT:
        return f"Problem: {task.get('task_prompt', '')}\n\nSolution: {a}"
    return a


def score_children(ext, children, da_policy, cc_policy, task):
    """Encode each child's score-text -> group (CORRECT domain/task metadata) -> DualAnchor (validity) +
    CoreContent (content) tournament scores."""
    cands = []
    for ch in children:
        feats = ext.encode_text_to_pooled_features(score_text_for(task, ch.text), max_length=512)
        cands.append({"features": feats, "reward": 0.0, "branch_id": ch.bid})
    group = {"group_id": f"s1_4_{task.get('canary_id', 'unknown')}",
             "domain": task.get("domain", "unknown"), "task_id": task.get("canary_id", "ref"),
             "candidates": cands, "kind": "tournament"}
    da = M.score_group(group, da_policy)
    cc = M.score_group(group, cc_policy)
    for i, ch in enumerate(children):
        ch.da = float(da[i]); ch.cc = float(cc[i])


def prune(children, loop, layer):
    """Two-mechanism prune: DualAnchor validity FIRST, then CoreContent content-threshold, then BUDGET."""
    import math
    # (1) DualAnchor validity gate
    valid = [c for c in children if math.isfinite(c.da) and c.da >= VALIDITY_THRESHOLD]
    if not valid:  # never empty: keep most-valid
        valid = sorted([c for c in children if math.isfinite(c.da)], key=lambda c: c.da, reverse=True)[:1] or children[:1]
    # (2) CoreContent content threshold (per locus)
    thr = content_threshold(loop, layer)
    passing = [c for c in valid if math.isfinite(c.cc) and c.cc >= thr]
    if not passing:  # never empty: keep best content among valid
        passing = sorted(valid, key=lambda c: (c.cc if math.isfinite(c.cc) else -1e9), reverse=True)[:1]
    # (3) budget cap by content
    kept = sorted(passing, key=lambda c: (c.cc if math.isfinite(c.cc) else -1e9), reverse=True)[:BUDGET]
    return kept, {"n_children": len(children), "n_valid": len(valid), "n_passing": len(passing),
                  "n_kept": len(kept), "content_threshold": (None if thr <= -1e8 else round(thr, 3))}


def run_task(model, tok, ext, task, dirs, da_policy, cc_policy, device):
    import math
    is_code = task.get("label_type") == "unit_tests"
    prompt = V.build_prompt(tok, task, "direct")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    plen = ids.shape[1]
    tr = (plen - 1, plen)                     # CANONICAL last-token suffix

    # base (clean, no branching) reference
    nl_c, cache_c, _cp, am_c = C.prefill(model, ids, am)
    base_toks, _ = S.greedy_continue(model, cache_c, am_c, nl_c, MNT_TERM, device)
    base_text = tok.decode(base_toks, skip_special_tokens=True)
    base_correct, _ = verify(task, base_text, is_code)
    C.clear_cuda()

    survivors = [Branch(bid="root", lineage=[], text=base_text, parent="-")]
    trace = []
    bcount = 0

    for li, (loop, layer) in enumerate(LOCI):
        is_term = (loop, layer) == TERMINAL
        mnt = MNT_TERM if is_term else MNT_INTER
        children = []
        for s in survivors:
            # GATE 4: the survivor's FULL lineage is what we re-derive on (never a clean root after divergence)
            assert len(s.lineage) == li, f"GATE4: survivor lineage {len(s.lineage)} entering locus {li}"
            # fixed-token_range invariant: every lineage perturbation used the same canonical last-token range
            assert all(e[4] == tr for e in s.lineage), f"token_range drift in lineage vs {tr}"
            rp = branch_specific_root_pack_v2(model, ids, am, s.lineage, loop, layer)
            for k in range(K):
                d = dirs[layer][k % len(dirs[layer])]
                new_lineage = s.lineage + [(layer, loop, d, ALPHA, tr)]
                # GATE 5: every boundary lies on the branch's actual trajectory (monotone; ancestors precede
                # the fork; appended entry == fork locus)
                assert_lineage_invariants(new_lineage, loop, layer, li + 1)
                br = S.build_spliced_branch(model, ids, am, loop, layer, ALPHA, d, tr, root_pack=rp)
                toks, _ = S.greedy_continue(model, br["spliced_cache"], br["am_full"], br["next_logits"], mnt, device)
                bcount += 1
                children.append(Branch(bid=f"b{li}_{bcount}", lineage=new_lineage,
                                       text=tok.decode(toks, skip_special_tokens=True), parent=s.bid))
            del rp
            C.clear_cuda()

        score_children(ext, children, da_policy, cc_policy, task)
        survivors, pst = prune(children, loop, layer)
        trace.append({"locus": f"loop{loop+1}_L{layer}", "is_terminal": is_term, **pst,
                      "da_range": [round(min(c.da for c in children), 3), round(max(c.da for c in children), 3)],
                      "cc_range": [round(min(c.cc for c in children), 3), round(max(c.cc for c in children), 3)],
                      "kept_cc": [round(c.cc, 3) for c in survivors]})
        print(f"    {task['canary_id']} {trace[-1]['locus']:9} children={pst['n_children']} "
              f"valid={pst['n_valid']} pass={pst['n_passing']} kept={pst['n_kept']}", flush=True)

    # TERMINAL selection: confidence-gated best-by-content
    best = max(survivors, key=lambda c: (c.cc if math.isfinite(c.cc) else -1e9))
    deferred = not (math.isfinite(best.cc) and best.cc >= TERMINAL_CONFIDENCE)
    sel_correct, sel_final = verify(task, best.text, is_code)
    # oracle over terminal survivors (analysis only)
    surv_correct = [verify(task, c.text, is_code)[0] for c in survivors]
    return {
        "task": task["canary_id"], "domain": task["domain"], "is_code": is_code,
        "base_correct": bool(base_correct),
        "selected_correct": bool(sel_correct), "selected_final": (sel_final or "")[:80],
        "deferred": bool(deferred), "selected_cc": round(best.cc, 4),
        "n_terminal_survivors": len(survivors),
        "oracle_over_survivors": bool(any(surv_correct)), "n_survivors_correct": int(sum(surv_correct)),
        "selected_lineage_depth": len(best.lineage),
        "trace": trace,
    }


def branch_specific_root_pack_v2(model, ids, am, lineage, b_loop, b_layer):
    """root_prefill_with_boundary with the branch's lineage LayerOutputPerturbHooks ACTIVE, capturing the
    boundary on the branch's OWN α>0 trajectory. Each lineage entry carries the EXACT token_range that
    created it: (layer, loop, direction, alpha, token_range) -> replay reproduces that perturbation
    exactly. (For this reference run the canonical token_range is the fixed last-token suffix of the
    single root prompt, asserted identical across the lineage in run_task.)"""
    handles = []
    for (L, lp, d, a, trange) in lineage:
        hook = C.LayerOutputPerturbHook(d, a, token_range=trange, target_loop=lp)
        handles.append(C.register_perturb_hook(model, L, hook))
    try:
        return S.root_prefill_with_boundary(model, ids, am, b_loop, b_layer)
    finally:
        for h in handles:
            h.remove()


def make_extractor_reusing(model, tok, device):
    """Reuse the ALREADY-LOADED eager Ouro model for BG feature extraction instead of letting
    BGTransformerFeatureExtractor.__init__ load a SECOND ~5GB model (would risk OOM on the 16GB GPU and
    distort runtime). Builds the extractor via __new__ and injects the shared model/tokenizer, replicating
    the force_all_loops config __init__ sets. The RLTT-local model already has total_ut_steps=4 /
    early_exit_threshold=1.0 (what both the splice loop AND the extractor require), so this is a no-op set.
    Extraction registers/removes its own capture hooks during encode; no loop hooks are active at score time."""
    import time as _t
    bm = __import__("src.evaluator.bg_transformer_features",
                    fromlist=["BGTransformerFeatureExtractor", "NUM_LOOPS"])
    Ext, NUM_LOOPS = bm.BGTransformerFeatureExtractor, bm.NUM_LOOPS
    ext = Ext.__new__(Ext)
    ext.model = model
    ext.tokenizer = tok
    ext.device = torch.device(device) if not isinstance(device, torch.device) else device
    ext.dtype = next(model.parameters()).dtype
    ext.force_all_loops = True
    ext.allow_partial = False
    ext.loaded_at = _t.time()
    ext.model_path = None
    ext._original_total_ut_steps = getattr(model.config, "total_ut_steps", None)
    ext._original_early_exit_threshold = getattr(model.config, "early_exit_threshold", None)
    ext._original_inner_total_ut_steps = getattr(getattr(model, "model", None), "total_ut_steps", None)
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = NUM_LOOPS
    if hasattr(model.config, "early_exit_threshold"):
        model.config.early_exit_threshold = 1.0
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "total_ut_steps"):
        inner.total_ut_steps = NUM_LOOPS
    return ext


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]
    dirs = curated_dirs_by_layer(K)

    model, tok, info = C.load_model(attn_implementation="eager")
    device = next(model.parameters()).device
    ext = make_extractor_reusing(model, tok, device)   # REUSE the loaded eager model (no 2nd ~5GB Ouro)
    da_policy = M.baseline_policies(include_diag=True).get("DualAnchor")
    cc_policy = M.crafted_v2_policies().get("CoreContent_v2_blockwise")
    assert da_policy and cc_policy, "missing tap policies"

    rows = []
    for task in tasks:
        print(f"  TASK {task['domain']} {task['canary_id']}", flush=True)
        rows.append(run_task(model, tok, ext, task, dirs, da_policy, cc_policy, device))

    n = len(rows)
    agg = {
        # HEADLINE: reachability of the chained branch-carry mechanism (locked stance: survivor-set handoff)
        "oracle_over_survivors": round(sum(r["oracle_over_survivors"] for r in rows) / n, 3),
        "base_acc": round(sum(r["base_correct"] for r in rows) / n, 3),
        # diagnostic: forced-top1 selection by content
        "selected_acc": round(sum(r["selected_correct"] for r in rows) / n, 3),
        "selected_acc_committed": (round(sum(r["selected_correct"] for r in rows if not r["deferred"]) /
                                         max(1, sum(1 for r in rows if not r["deferred"])), 3)),
        "defer_rate": round(sum(r["deferred"] for r in rows) / n, 3),
        "mean_terminal_survivors": round(sum(r["n_terminal_survivors"] for r in rows) / n, 3),
        "mean_survivors_correct": round(sum(r["n_survivors_correct"] for r in rows) / n, 3),
    }
    rank_only = (VALIDITY_THRESHOLD <= -1e8 and all(v <= -1e8 for v in BASE_CONTENT_THR.values()))
    payload = {"verdict": "S1_4_REFERENCE_LOOP", "structural_gates": "GATE4+GATE5 asserted inline (no AssertionError)",
               "policy": "full_stress_reference_fork_every_locus",
               "prune_mode": ("rank_only_content_plus_budget (validity diagnostic only)" if rank_only
                              else "thresholded (validity/content gates active)"),
               "scoring_note": ("prune scores SHORT ROLLOUT TEXT re-encoded through the hidden-state taps "
                                "(teacher/scaffold), NOT the carried branch cache in-place — not a fully "
                                "internal no-rollout selector"),
               "score_input_mode": SCORE_INPUT_MODE,
               "score_format_caveat": ("locked CoreContent taps were extracted from a MIX incl. PROMPT+ANSWER "
                                       "('Problem: {q}\\n\\nSolution:{c}'); answer-only under-contextualizes — "
                                       "paper-grade cross-domain metrics need S1_SCORE_WITH_PROMPT=1"),
               "terminal_stance": ("survivor-set handoff is the locked safe object; forced top1 (selected_acc) "
                                   "is DIAGNOSTIC. Headline = oracle_over_survivors."),
               "question_answered": "can the chained branch-carry loop PRESERVE ORACLE REACHABILITY across 12 loci?",
               "config": {"N_PER": N_PER, "K": K, "BUDGET": BUDGET, "alpha": ALPHA, "loci": len(LOCI),
                          "terminal": f"loop{TERMINAL[0]+1}_L{TERMINAL[1]}", "mnt_inter": MNT_INTER, "mnt_term": MNT_TERM,
                          "token_range": "fixed last-token suffix (plen-1, plen), stored per lineage entry",
                          "validity_threshold": VALIDITY_THRESHOLD, "progressive": PROGRESSIVE,
                          "terminal_confidence": TERMINAL_CONFIDENCE, "content_thresholds": BASE_CONTENT_THR},
               "n_tasks": n, "aggregate": agg, "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4_reference_loop.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print("\nS1_4_REFERENCE_LOOP — full inject->carry->prune->loop-back->terminal reference (gates 4+5 held)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
