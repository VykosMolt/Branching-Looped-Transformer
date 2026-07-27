# Thinking pre-answer replication — pre-registered plan (SEALED)

Sealed: 2026-07-26, before any Thinking generation. Script hashes at seal time:
- bg_v3_thinking_preanswer_generate.py  sha256 29b2f3fa289197445e794db76a6ac28c78a58feded133ae286f3eb7b7f76ba6c
- bg_v3_thinking_preanswer_analysis.py  sha256 d993afd0ebf040f42b347db251eadd61ec59da9ee1c2d56e179e0000969ae76a

## Question
Is the strict pre-answer success-prediction increment RLTT-specific, or present
in the Thinking sibling? Either answer is informative: positive extends the
headline beyond one checkpoint; null separates general recurrent readability
from RLTT-specific prospective-success information.

## Sealed design
- Model: ByteDance/Ouro-2.6B-Thinking @ f1edd81 (pinned snapshot).
- Domain: Horizon Logic, the sealed generator protocol unmodified.
- Task slice: offsets [0,170) of the hash-sorted pool — the SAME tasks as the
  RLTT v2 main run, so checkpoint is the only varied factor.
- k=4, max_new=448 (protocol identity; Thinking's higher truncation/malformed
  rate is itself a reported quantity, with the malformed-sibling shortcut in
  the adversarial baseline), temperature 0.7, top_p 0.95, seed 20260724.
- NO changes to generation settings after observing class balance.

## Sealed endpoints
Primary: within-Thinking heldout increment under the 5-shortcut adversarial
composite. Secondary: the published 4-shortcut composite. Gate: Thinking task
set must equal the RLTT v2 main task set; splits must reproduce split_for.
Verdicts fixed in the analysis script: REPLICATED_AND_ROBUST /
POSITIVE_NOT_ROBUST / NULL / UNDERPOWERED (reported with counts, not massaged).
No cross-checkpoint AUROC comparisons; within-checkpoint increments only.

## Amendment 1 (2026-07-26, before main generation; after a 2-task smoke)
The pinned upstream snapshot's modeling_ouro.py fails inside generate() under
transformers 4.54.1 ("property 'key_cache' of 'UniversalTransformerCache' has
no setter"). Fix: models/ouro_thinking_local/ = symlinks to the pinned
snapshot's weights/config/tokenizer plus a minimal property+setter shim in
modeling_ouro.py (the same patch ouro_rltt_local already carries). The smoke
shard that triggered the error produced 0 usable candidates and was deleted.
No protocol constant changed. Generate script sha256 after the model-path
change is recorded in RUN_MANIFEST at completion.
