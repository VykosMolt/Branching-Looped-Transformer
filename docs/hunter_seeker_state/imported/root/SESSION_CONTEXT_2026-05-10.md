<!-- Imported from `SESSION_CONTEXT_2026-05-10.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 077a3d66371be000b8903652290337c79ce8ee167254aec9b40a58d4444f27cf; original line count: 151. -->

# Session context — 2026-05-10

Snapshot written at the end of the first Arch session after the Pop OS → Arch migration. Captures environment state, the Ouro RLTT hidden-states quirk, and a fresh read on `claude_sandbox/` and `~/local_agent/`. Treat as a working note, not a spec.

## Environment

- Host: `arch-legion`, Arch Linux (kernel 7.0.5-arch1-1). Migrated from Pop OS.
- GPU: RTX 5070 Ti Laptop, 12 GB, driver 595.71.05, CUDA 12.8.
- Project venv: `/home/moloch/ouro_project/venv`, Python **3.14.4**.
  - torch `2.12.0.dev20260407+cu128`
  - transformers `4.54.1` (pinned — do not upgrade per `claude_sandbox/README.md` and `local_agent/README.md`)
  - safetensors `0.7.0`, accelerate `1.13.0`
  - All internal symbols modeling_ouro.py imports from transformers resolve.

## Ouro RLTT — loading

The Princeton FSDP shards in `~/Downloads/RLTT/Downloads_RLTT/RLTT/` are already converted to HF safetensors at `models/ouro_rltt_local/` (3 shards, 533 tensors, ~5.3 GB bf16, **2.67B params**, `OuroForCausalLM`). `consolidated_clean.pt` (10.6 GB) under `ouro_rltt/` and the Downloads tree is the older raw consolidation; use the safetensors copy for HF loads.

Load incantation (verified working on this Arch venv, ~1.5 s load + ~1.8 s generate):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok   = AutoTokenizer.from_pretrained("models/ouro_rltt_local", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "models/ouro_rltt_local",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
```

Config highlights: 48 layers, hidden 2048, 16 heads (no GQA), `total_ut_steps=4`, `early_exit_threshold=1.0`, `rope_theta=1e6`, `max_position_embeddings=65536`, vocab 49152, GPT2Fast tokenizer.

## Ouro RLTT — hidden-states (patched 2026-05-10)

**Before the patch:** `model(x, output_hidden_states=True)` returned `out.hidden_states is None`. Cause: `OuroModel.forward` constructed `BaseModelOutputWithPast(last_hidden_state=..., past_key_values=...)` without ever threading `output_hidden_states` through; `OuroForCausalLM.forward` then forwarded the `None` into the final return. The intended workaround was the custom kwarg `return_per_loop_hidden_states=True`, which yields `OuroRLTTCausalLMOutput.per_loop_hidden_states`.

**After the patch:** standard `output_hidden_states=True` works. Returns a tuple of length `total_ut_steps + 1` = 5 elements: `(inputs_embeds, post-loop-0, post-loop-1, post-loop-2, post-loop-3)`. Matches HF convention (each "layer-equivalent" boundary, plus the embedding).

The patch (4 added lines in each of two files):

```python
# In OuroModel.forward, replacing the previous unconditional return:
output_hidden_states = kwargs.get("output_hidden_states", None)
if output_hidden_states is None:
    output_hidden_states = getattr(self.config, "output_hidden_states", False)
return (
    BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=(inputs_embeds, *hidden_states_list) if output_hidden_states else None,
    ),
    hidden_states_list,
    gate_list,
)
```

Applied to both copies (they had diverged):
- `ouro_rltt/modeling_ouro.py` — the Princeton drop (1370 → 1373 lines). Loaded by direct-import + `consolidated_clean.pt`.
- `models/ouro_rltt_local/modeling_ouro.py` — the in-use HF copy (1456 → 1459 lines) with `UniversalTransformerCache(Cache)` property accessors and `adaptive_compute` support. Loaded by `AutoModel.from_pretrained`.

Downstream is unchanged — `OuroForCausalLM.forward` already forwarded `outputs.hidden_states` in both the standard (`CausalLMOutputWithPast`) and RLTT (`OuroRLTTCausalLMOutput`) return paths.

Verified behaviors on this Arch venv after the patch:
- `output_hidden_states=True` → 5-tuple, shapes `(B,T,2048)` bf16.
- `output_hidden_states=True` + `return_per_loop_hidden_states=True` → both populated; `out.hidden_states[-1] == out.per_loop_hidden_states[-1]` (true).
- No flags → `out.hidden_states is None` (no regression for callers who don't ask).
- Generation unaffected.

Sanity numbers (16-token random input, `ouro_rltt/` Princeton copy, `consolidated_clean.pt` weights):

```
[0] embed : mean +0.0003  std 0.0238
[1] loop 0: mean +0.0255  std 0.8732
[2] loop 1: mean +0.0220  std 0.7326
[3] loop 2: mean +0.0110  std 0.8454
[4] loop 3: mean +0.0052  std 0.8484
```

The custom `return_per_loop_hidden_states=True` API is untouched and still the right choice when you also want gate outputs or want to avoid building the embedding entry in the tuple.

## `claude_sandbox/`

- Size: 45 GB. ~130 `checkpoints_*` dirs, ~120 `perf_event_dumps_*` dirs, 20 top-level `.py` modules, two session-summary handoff logs (`SESSION_SUMMARY.md` codex side, `CLAUDE_SESSION_SUMMARY.md` claude side), planning docs in `design/` and at top level.
- Sandbox rules from its README: do not modify user files, keep variants here, do not upgrade transformers past 4.54.1.
- Module structure (synthesized): training entry (`train_arc_codex.py` + `run_ablation_ladder.sh`), agents (`arc_agent_hunter_seeker_codex.py` 1.2 MB monolith; `arc_agent_pairwise_stockfish_codex.py`), encoders/adapters (`grid_encoder_codex.py`, `action_adapters_codex.py`, `observation_adapters_codex.py`, `observation_learning_codex.py`), cognitive components (`anchor_loss.py` Sprint-11a CLT anchor; `self_model.py` Sprint-11b), evaluator (`evaluator_pairwise_codex.py`), diagnostics (`branch_basin_audit.py`, `live_arc_diagnostic.py`, `sandbox_sweep_validate.py`), reporting (`summarize_event_dumps.py`, `compare_ladder_summaries.py`, `online_trace_run_report.py`, `focus_game_*_report.py`, `event_dump_sprint6_ablate.py`).
- Vocabulary: **ladder** = the 8-step ablation ladder defined in `design/ablation_ladder.md`; **anchor** = the CLT-evaluator anchor loss (Sprint-11a fix for Sprint-4 encoder cosine drift); **sprints** = paper-aligned milestones (4 = drifted baseline, 11a/11b = current, 7/8/9/10/12/13 deferred). Target game on wa30; ls20 / ft09 / r11l / tr87 also in rotation.
- Active line (May 1–5 2026): wa30 prestate-gate / post-veto / effective-engram / component-diag trio, then a topology-memory probe (`checkpoints_topology_trio_exact_support*`, `ls20_smoke` cluster).
- Next-move-per-planning: post-veto 8-run ladder → tiny topology memory if topology deaths repeat → cleanup-plan-v4 group 5 (replace guards) → group 2 (RUN_IDs + comparator aggregation) → group 3 (perf). Explicitly **after** the ladder is stable.
- State: thinking is in good order; filesystem is the refactor target. `_cleanup_quarantine/` is genuinely dead. Many `_gpu2/3/4` duplicates. Directory naming has replaced a real changelog.

## DSA harness — post-patch confirmation (2026-05-10)

Ran the standard normal/hard DSA harness (`tools/test_local_agent_dsa_coding.py`, **not** the devils set) to confirm the hidden-states patch did not regress the local-agent wrapper. All 4 tasks passed on first attempt (`direct` mode, no escalation):

```
task                                tier        sec   pass  used_mode
subarray_sum_count                  normal     8.52   True  direct
longest_increasing_subsequence      normal     6.04   True  direct
word_ladder_length                  hard       9.93   True  direct
count_smaller_after_self            hard       5.65   True  direct
TOTAL: 30.15s   ALL PASS: True
```

(Settings: `--mode direct_first --ut-steps 2 --max-tokens 512`, oracle lookups disabled.)

GPU residency check via `local_agent.ouro_backend.DeepThinkModel.load()`:

```
model class : OuroForCausalLM
param device: cuda:0
param dtype : torch.bfloat16
cuda alloc  : 5.34 GB
gpu         : NVIDIA GeForce RTX 5070 Ti Laptop GPU
gen 24 tok  : 2.15s on cuda:0
```

`_assert_gpu_only_model_placement` (`ouro_backend.py:549`) actively enforces this. The wrapper uses its bounded exit-telemetry forward hook (max 256 records) rather than the standard `output_hidden_states` API, so the patch is **neutral for the wrapper** and **additive for any new probe/evaluator code** that wants the HF-standard tuple.

Runs JSON: `runs/local_agent_dsa_coding/dsa_coding_20260510_225817.json`.

## `~/local_agent/`

- Real entry: `python ouro_agent_improved.py` (CLI / REPL / `--headless` / `--server`). Modes: agent | direct | crosscheck | debate | judge | auto | review | classify.
- RLTT path: `DEFAULT_RLTT_MODEL_PATH = ~/ouro_project/models/ouro_rltt_local` (overridable via `LOCAL_AGENT_OURO_MODEL_ID`). Wrapper enforces `transformers==4.54.1`, CUDA-required, `local_files_only=True`, bf16 default (4-bit NF4 only as opt-in fallback via `LOCAL_AGENT_OURO_LOAD_IN_4BIT=1`), `use_cache=False` (works around a `UniversalTransformerCache` setter crash). Adaptive compute defaults off — local calibration showed almost no real short-circuiting at useful thresholds.
- Hidden states: wrapper does NOT call `return_per_loop_hidden_states`. It captures exit/latent telemetry via a bounded forward hook (max 256 records, `OURO_CAPTURE_LATENT_SIGNATURES` on by default). So the kwarg quirk above doesn't bite this code; it only matters for fresh probe/evaluator work.
- GGUF reviewer backends `qwen3-14b` and `deepseek-r1-14b` under `local_agent/models/` are integrated through `ouro_model_managers.py` for crosscheck/judge roles.
- Self-model stack: `LocalSelfModel(SelfModelContextBuilderMixin, SelfModelStoreMixin)` orchestrates; `ouro_self_predictor.py` is the bounded online failure-rate predictor (raises caution only, denom 4.0 after audit); `ouro_vector_memory.py` is the dependency-free hash-feature store.
- Audit state (2026-05-09): six-batch audit closed real bugs across 12 modules; 187 wrapper+self-model tests passing; standard DSA harness solves cleanly; "devil" abstentions (`offline_dynamic_connectivity`, `minimum_xor_paths`) honestly characterized as a route-to-code RLTT limit, not a wrapper bug.
- Stray: `projects/local_self_vectors.json` exists at 0 bytes; vectors actually live inside `local_self_model.json`. Probably a stale path; worth a sweep when convenient.

## Refactor plan (sandbox — when ready, not now)

Per planning docs the ordering matters; do not start before the wa30 ladder is stable.

1. Move `checkpoints_*` + `perf_event_dumps_*` under dated subdirs (`runs/2026-04-*/`, `runs/2026-05-*/`); top level should not be a 250-entry wall.
2. Delete `_cleanup_quarantine/` and obvious `_gpu2/3/4` duplicates. Quarantine once, prune after a week.
3. Adopt timestamped RUN_IDs (already specified in `pre_ladder_audit_backlog_updated.md`).
4. The 1.2 MB `arc_agent_hunter_seeker_codex.py` monolith is the real code refactor — but only after the ladder is stable.
5. Cleanup-plan-v4 group 5 → group 2 → group 3 ordering still applies.

`local_agent` does not need a refactor — keep its audit cadence.

## Open threads

- ~~Decide whether to patch `OuroModel.forward` to honor `output_hidden_states=True`.~~ Done — both copies patched 2026-05-10. Note: `ouro_rltt/` is no longer byte-identical to the Princeton drop; if you ever re-sync from upstream, re-apply this patch.
- Confirm `projects/local_self_vectors.json` path is intentional / dead.
- Wa30 post-veto 8-run ladder still pending.
- Sandbox filesystem refactor pending (see plan above) — not before the wa30 ladder is stable.
