<!-- Imported from `claude_sandbox/terminal_predframe_context_8run_ladder_post.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 3442606b24c13bb2cfee2c4015ee370a8e13c315b6c35fb48dfa008e5173fa1f; original line count: 180. -->

# terminal_predframe_context_8run_ladder Post-Ladder Note

Date: 2026-04-29

Status: GPU ladder completed. The terminal-memory collapse from the two bad probes did not reproduce. One specific residual failure mode did appear and was patched: epsilon-random exploration could bypass terminal-memory penalties.

## Run Configuration

Probe:

```text
terminal_predframe_context_8run_ladder
```

Command shape:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARC_API_URL=offline \
./venv/bin/python -m claude_sandbox.train_arc_codex \
  --games ls20 \
  --n_runs 8 \
  --max_steps 160 \
  --agent hunter_seeker \
  --backbone_mode ouro \
  --use_loop_pooler on \
  --loop_pooler_kind gru \
  --self_model_mode off \
  --cortex_monitor_mode off \
  --anchor_train_every 25 \
  --anchor_coefficient 0.1 \
  --anchor_batch_size 1 \
  --unfreeze_encoder_for_anchor \
  --running_checkpoint claude_sandbox/checkpoints_running/terminal_predframe_context_8run_ladder.pt \
  --dump_events_dir claude_sandbox/ablation_event_dumps/terminal_predframe_context_8run_ladder
```

The run was verified on CUDA. `self_model_mode` and `cortex_monitor_mode` were both off intentionally, so this measured terminal memory and predframe-context behavior without self-model/cortex confounds.

## Comparison Summary

| Probe | Levels | Mechanism deltas | Terminal path | Final penalties | Degenerate vector skips | Key diversity |
|---|---:|---|---|---:|---:|---|
| `terminal_memory_probe_8run_post_pending_exactfix` | 0/8 | `[1,1,1,1,1,1,1,0]` | exact only, `prototype_count=0` | `combined=-3032.8034` | `6646` | not emitted |
| `terminal_memory_probe_8run_post_soft_exact_strengthfix` | 0/8 | `[1,1,1,1,1,1,1,1]` | exact only, `prototype_count=0` | `combined=-111.4053` | `6486` | not emitted |
| `terminal_predframe_context_smoke_2run` | 0/2 | `[1,0]` | prototype active, action `2` | `prototype=-0.7371` | `0` | `558/678`, top rep `8` |
| `terminal_predframe_context_8run_ladder` | 0/8 | `[0,1,0,0,0,0,0,1]` | prototype active, action `4` | `prototype=-129.7724` | `0` | `2937/3925`, top rep `9` |

Per-run result for the new 8-run ladder:

```text
levels_completed:       [0,0,0,0,0,0,0,0]
steps:                  [160,130,160,160,160,160,160,130]
mechanism_fail_cum:     [0,1,1,1,1,1,1,2]
mechanism_fail_delta:   [0,1,0,0,0,0,0,1]
terminal_memory_size:   [0,1,1,1,1,1,1,2]
prototype_count:        [0,1,1,1,1,1,1,2]
prototype_calls:        [0,0,134,226,266,347,399,481]
combined_penalty_sum:   [0.0,0.0,-40.9284,-62.0047,-68.0983,-94.8005,-103.5275,-129.7724]
degenerate_vector_skip: [0,0,0,0,0,0,0,0]
key_unique:             [281,662,1029,1377,1784,2175,2510,2937]
key_top_repetition:     [6,6,6,7,7,9,9,9]
```

Final terminal-memory summary for the new ladder:

```json
{
  "size": 2,
  "prototype_count": 2,
  "prototype_count_by_action": {"4": 2},
  "exact_penalty_calls": 0,
  "exact_penalty_sum": 0.0,
  "prototype_penalty_calls": 481,
  "prototype_penalty_sum": -129.7724,
  "combined_penalty_calls": 481,
  "combined_penalty_sum": -129.7724,
  "counterevidence_context_writes": 0,
  "counterevidence_prototype_writes": 48,
  "degenerate_exact_skips": 0,
  "degenerate_vector_skips": 0,
  "key_diversity": {
    "total_count": 3925,
    "unique_count": 2937,
    "top_repetition": 9,
    "zero_delta_count": 3925
  }
}
```

Action/trace summary for the new ladder:

```text
selected actions:                 {"1":595, "2":243, "3":229, "4":153}
selection methods:                {"beam_search":1002, "random":218}
trace modes:                      {"topology":1002, "unknown":218}
selected terminal penalty count:  25
selected terminal penalty sum:    -2.7226
candidate terminal penalty count: 555
candidate terminal penalty sum:   -131.8544
candidate terminal penalties by action: {"4":555}
max prototype similarity:         0.919891
min prototype penalty:            -0.688751
```

## Questions From The Handoff

1. Did terminal memory prevent repeated terminal mechanisms?

Mostly yes. The two bad probes died in 7/8 and 8/8 runs. The predframe 8-run died in 2/8 runs, with no repeated death streak after the first prototype was formed.

2. Did it do so through prototype memory rather than exact action-blacklist behavior?

Yes. Exact penalty calls stayed at `0`. Prototype penalty calls reached `481`, all on action `4`, with `prototype_count=2` and no degenerate-vector skips.

3. Did exploration remain alive?

Yes. The run still took all four directional actions and had 218 random selections. Key diversity was high: `2937` unique keys out of `3925` total, with top repetition only `9`.

4. Did levels completed improve?

No. `levels_completed` stayed `0` for all eight runs. This is now a progress/topology/mechanism competence problem, not the previous terminal-memory structural collapse.

5. Did terminal memory suppress only risky action-contexts or broadly suppress an entire primitive action?

Mostly context-local. Candidate penalties were only observed for action `4`, because both terminal prototypes were action `4`, but the selected action distribution still included action `4` 153 times. Beam search generally avoided penalized action-4 contexts rather than deleting the primitive action.

6. Did prototype similarity/penalty behavior remain sane?

Yes. Prototype similarities reached about `0.92`, penalties stayed bounded, counterevidence accrued through prototype writes (`48`), and degenerate-vector skips stayed at `0`.

## Specific Failure Mode Found

The second new death was run 8, step 130, `last_action=4`, classified as `mechanism`. The action trace showed:

```text
selection_method=random
action=4
trace_mode=unknown
```

Immediately before that, beam-scored candidates were applying prototype terminal penalties to action `4` around `-0.55` to `-0.63`, while selecting safer action `1`. The final random action had no scored candidate context, so epsilon-random exploration could bypass terminal-memory penalties.

Patch applied:

- `HunterSeekerAgent.select_action` now keeps epsilon-random exploration.
- When terminal memory exists, it runs the normal beam scorer as a safety trace.
- It vetoes the random action only if that trace marks the same action/context with `terminal_outcome_penalty < -0.05`.
- The fallback action is the beam-selected action, and the info dict records `selection_method=random_terminal_veto_beam`.
- This does not blacklist action `4`; beam search can still pick action `4` in a safe scored context.

Focused regression:

```text
test_hunter_random_action_uses_terminal_memory_safety_trace
```

Verification after patch:

```text
targeted terminal/topology suite: 45 passed, 97 deselected
full claude_sandbox suite:       248 passed
```

## Next Step

Do not tune terminal penalty strength yet. The structural bug is fixed and the post-patch architecture is test-clean. The next empirical step is a post-veto 8-run ladder, ideally:

```text
terminal_predframe_context_8run_ladder_random_veto
```

Inspect:

- whether `random_terminal_veto_beam` fires;
- whether mechanism deltas fall below 2/8;
- whether levels completed remain 0;
- whether action `4` remains available under beam search;
- whether prototype penalty calls remain bounded;
- whether the remaining blocker is topology/progress rather than terminal memory.
