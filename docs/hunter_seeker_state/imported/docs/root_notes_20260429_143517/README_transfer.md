<!-- Imported from `docs/root_notes_20260429_143517/README_transfer.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 89c50a4b26858d5a31ec09da139e677235f289158d3612a2192f6c2ee37b2a50; original line count: 92. -->

# Domain Transfer Experiment

This package tests whether the HH-trained pairwise evaluator transfers beyond HH-RLHF to:
- math
- code
- logic

It has been patched to use the **same evaluator input distribution** as your working best-of-N pipeline: the evaluator always sees **prompt + response** wrapped into HH-style conversation text before hidden states are extracted.

## Files

- `domain_transfer_common.py` — shared loaders, formatting, dataset adapters, oracles
- `arafel_domain.py` — phase 1 zero-shot transfer test
- `disagreement_audit.py` — phase 2 disagreement bucket audit
- `loop_convergence_probe.py` — phase 3 loop-geometry probe
- `cross_domain_matrix.py` — phase 4 feature extraction / training / transfer matrix
- `run_transfer_experiment.py` — convenience orchestrator

## Phase 1

```bash
python arafel_domain.py --domains math code logic --n_prompts 100 --n_candidates 4
```

Outputs:
- `domain_transfer_results/math_results.json`
- `domain_transfer_results/code_results.json`
- `domain_transfer_results/logic_results.json`
- `domain_transfer_results/summary.json`

Baselines:
- first candidate
- random candidate
- majority over extracted answers, where that makes sense
  - enabled for math and logic
  - disabled for code, because answer-majority is not a meaningful oracle-free code baseline

## Phase 2

```bash
python disagreement_audit.py --domains math code logic --mode auto
python disagreement_audit.py --domains math --mode manual
```

Buckets:
- `evaluator_genuinely_wrong`
- `oracle_wrong_or_incomplete`
- `ambiguous_near_tie`
- `style_preference_leak`
- `safety_helpfulness_override`

## Phase 3

```bash
python loop_convergence_probe.py --domains math code logic
```

Outputs:
- `loop_convergence_probe/<domain>_convergence_probe.png`
- `loop_convergence_probe/<domain>_probe_data.json`

## Phase 4

Extract features from phase-1 results:

```bash
python cross_domain_matrix.py --step extract --all_domains --max_pairs_per_prompt 1
```

Train domain-specific evaluators:

```bash
python cross_domain_matrix.py --step train --all_domains --epochs 3 --batch_size 16
```

Evaluate the transfer matrix:

```bash
python cross_domain_matrix.py --step eval
python cross_domain_matrix.py --step eval --include_hh_test --hh_limit 500
python cross_domain_matrix.py --step agreement
```

## Important fixes applied

- fixed the syntax error in `arafel_domain.py`
- fixed evaluator input formatting so scoring uses **prompt + response**, not bare response text
- fixed the HumanEval oracle to call `check(<entry_point>)` instead of `check(candidate)`
- added code-candidate cleanup for fenced output and body-only continuations
- replaced the broken math string oracle with a boxed/final-answer extractor plus symbolic/numeric equivalence checks
- prevented repeated Ouro reloads inside long runs by caching the model/tokenizer/evaluator per process
- made phase 4 use the same formatted texts as phase 1 instead of drifting to bare candidate strings
