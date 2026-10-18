# Exit-gate check for Runs 4 and 8 (post hoc, 2026-10-18)

This analysis was run after the fact and was not part of either run's plan. Question: when Runs 4 (Ouro-RLTT) and 8
(Ouro-2.6B-Thinking) generated with `BGTransformerFeatureExtractor(force_all_loops=True)` (all four loops computed,
exit threshold 1.0), how often did the exit rule pick a loop earlier than the fourth to produce a token?

Method (`exitgate.py`): a fixed-seed sample (seed 20261018) of 200 of the 4,656 answers in each run. Each answer gets one
teacher-forced forward of prompt + generated tokens, using the same extractor, bf16, and model files (RLTT:
`shared/models/ouro_rltt_local`; Thinking: revision f1edd81e, fetched and SHA-256-checked by `gen_run8.ensure_model`,
then deleted). A hook reads the forward's own `hidden_states_list` / `gate_list`. The exit rule copied from the model file
is applied to them in bf16. Two checks: (a) which loop's logits equal the logits the forward returned, bit for bit;
(b) the stored `gen_logprobs` (log-softmax of generate()'s processed scores: T 0.7, top-k 50, top-p 0.95) are recomputed
from each loop's logits.

Run: `HF_HOME=<scratch> /home/moloch/ouro_project/shared/venv/bin/python -B exitgate.py --run 4` (then `--run 8`).
Outputs: `results_run4.json`, `results_run8.json` (summary + per-answer), `run4.log`, `run8.log`.

## Result

| | Run 4 (RLTT) | Run 8 (Thinking) |
|---|---|---|
| generated tokens checked | 70,860 | 76,154 |
| tokens from loop 1 / 2 / 3 / 4 | 0 / 0 / 0 / 100% | 0 / 0 / 0 / 100% |
| answers with any early exit | 0 / 200 | 0 / 200 |
| returned logits equal the gate-rule loop's, bit for bit | 70,860 / 70,860 | 76,154 / 76,154 |
| highest sigmoid(gate) at loop 1 / 2 / 3 | 0.365 / 0.422 / 0.570 | 0.332 / 0.375 / 0.543 |
| highest cumulative exit prob. after loop 3 | 0.828 | 0.777 |
| sigmoid exactly 1.0 (any loop) | 0 | 0 |
| answers whose stored log-probs fit loop 4 best | 200 / 200 | 200 / 200 |

Both gates are far from saturating. In bf16, sigmoid rounds to 1.0 only when the gate logit is at least about 6.2. The
largest logit seen at loops 1-3 was 0.29 (RLTT) and 0.17 (Thinking). So the threshold-1.0 rule never exits early.
Every token comes from loop 4.

Log-prob check: across each answer, loop 4 reproduces the stored values best in every answer. Mean |diff| per answer
(nats, capped at 5) is 0.018 for loop 4 vs 0.074 / 0.17 / 0.42 for loops 3 / 2 / 1 (RLTT); 0.020 vs 0.071 / 0.17 /
0.39 (Thinking). Token by token, the match is noisy: |diff| at loop 4 has q90 about 0.04 and q99 about 0.17 nats. On
tokens where one loop beats the next by at least 0.05 nats, the best-fitting loop is loop 4 87% of the time (91% under a
stricter rule). The values cannot be reproduced bit for bit. Generation used batched (k=4) KV-cached decoding on an RTX
PRO 6000. This check is a single full forward on an RTX 5070 Ti Laptop. The token-level misses are that noise; the
per-answer test is the reliable one.

Notes: attention is `sdpa`. The extractor does not set `attn_implementation`, and transformers 4.54.1 then picks sdpa.
The Thinking shim reads the threshold that `config.json` sets at load (1.0), not the extractor's override. The value
is the same either way.
