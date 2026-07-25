<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 8032-9001 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: b7a4957181904aaa7e49c1d736bd76930714d1689471650e36195df0ccd3467f -->

## RLTT Weights Located In Downloads (2026-05-04)

### Files Found

- Directory:
  - `/home/moloch/Downloads`
- Total size:
  - about `15G`.
- FSDP config:
  - `/home/moloch/Downloads/fsdp_config.json`;
  - contents: `FSDP_version=1`, `world_size=4`.
- Model shards:
  - `model_world_size_4_rank_0.pt`;
  - `model_world_size_4_rank_1.pt`;
  - `model_world_size_4_rank_2.pt`;
  - `model_world_size_4_rank_3.pt`;
  - each is about `2.5G`;
  - each is a PyTorch zip checkpoint with `540` zip entries and about `2.485 GiB` payload.
- Optimizer shards:
  - `optim_world_size_4_rank_0.pt` through `optim_world_size_4_rank_3.pt`;
  - each is about `1.3G`.
- Extra-state shards:
  - `extra_state_world_size_4_rank_0.pt` through `extra_state_world_size_4_rank_3.pt`;
  - each is about `15K`.
- Hugging Face code/tokenizer zip:
  - `huggingface-20260504T064604Z-3-001.zip`;
  - contains `huggingface/config.json`, `modeling_ouro.py`, `configuration_ouro.py`, tokenizer files, and chat template;
  - does not contain the model weight tensors.

### Architecture Observed

- `huggingface/config.json` describes:
  - `model_type: ouro`;
  - `architectures: ["OuroForCausalLM"]`;
  - `hidden_size: 2048`;
  - `num_hidden_layers: 48`;
  - `num_attention_heads: 16`;
  - `num_key_value_heads: 16`;
  - `intermediate_size: 5632`;
  - `vocab_size: 49152`;
  - `total_ut_steps: 4`;
  - `early_exit_threshold: 1.0`;
  - `rltt_loop_level_checkpointing: true`;
  - `rltt_logprob_chunk_size: 2048`.
- The model shard metadata loads under `FakeTensorMode` as a DTensor/FSDP state dict:
  - 533 model entries;
  - keys include `model.embed_tokens.weight`, transformer layer weights, `model.early_exit_gate.*`, and `lm_head.weight`;
  - tensors report full logical shapes, but values are `torch.distributed.tensor.DTensor` with sharded placements such as `Shard(dim=0)`.

### Compatibility Notes

- The active sandbox currently loads Ouro through `AutoModelForCausalLM.from_pretrained("ByteDance/Ouro-2.6B-Thinking")` in `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`.
- The downloaded RLTT package is not yet in directly loadable Hugging Face weight layout:
  - code/tokenizer are in the HF zip;
  - model weights are separate FSDP/DTensor world-size-4 shards.
- Next step is a conversion/loader pass:
  - extract the HF code/tokenizer into a local model directory;
  - reconstruct or load the world-size-4 DTensor/FSDP model shards;
  - emit a local HF-style checkpoint/safetensors layout or add an explicit local FSDP-shard loader;
  - then point `_load_ouro()` at the local RLTT model path.
- `extra_state_world_size_4_rank_0.pt` did not load with PyTorch 2.6 default `weights_only=True` because it contains numpy objects; do not use `weights_only=False` casually. If needed, inspect it later with an explicit trusted-source/safe-globals path.

## RLTT Local Conversion / Loader Wiring (2026-05-04)

### Constraints

- Transformers must remain pinned to `4.54.1`.
- Verified active venv reports:
  - `transformers.__version__ == "4.54.1"`.
- No package upgrade was performed.

### Implemented

- Added converter:
  - `tools/convert_rltt_fsdp_to_hf.py`.
- Converter behavior:
  - extracts HF code/tokenizer files from `huggingface-20260504T064604Z-3-001.zip`;
  - loads `model_world_size_4_rank_*.pt` with `weights_only=True`;
  - imports `torch.distributed.tensor` so DTensor values can be safely loaded;
  - gathers rank-local DTensor shards along their `Shard(dim=0)` placement;
  - writes HF-style sharded `safetensors`;
  - writes `model.safetensors.index.json`;
  - patches converted config metadata to `transformers_version: 4.54.1`.
- Conversion command used:
  - `/home/moloch/ouro_project/venv/bin/python tools/convert_rltt_fsdp_to_hf.py --downloads-dir /home/moloch/Downloads --output-dir models/ouro_rltt_local --dtype bfloat16 --max-shard-size-gb 1.8 --overwrite`
- Converted output:
  - `models/ouro_rltt_local`;
  - `model-00001.safetensors` through `model-00003.safetensors`;
  - total safetensors payload about `4.969 GiB`;
  - `533` mapped tensors.
- Loader wiring:
  - `PairwiseARCSearchAgent` now accepts `ouro_model_path`;
  - train CLI now accepts `--ouro_model_path`;
  - default remains `OURO_MODEL_PATH` env var or `ByteDance/Ouro-2.6B-Thinking`;
  - local remote-code cache defaults to workspace `.hf_modules_cache` to avoid sandbox/cache write failures.

### Smoke Verification

- Config load succeeded offline under Transformers `4.54.1`:
  - `AutoConfig.from_pretrained("models/ouro_rltt_local", trust_remote_code=True)`;
  - hidden size `2048`, layers `48`, total UT steps `4`.
- Model load succeeded offline on CPU under Transformers `4.54.1`:
  - `AutoModelForCausalLM.from_pretrained("models/ouro_rltt_local", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": "cpu"}, low_cpu_mem_usage=True)`.
- ARC-facing forward path succeeded:
  - `m.model(inputs_embeds=torch.zeros(1, 4, 2048, dtype=torch.bfloat16), use_cache=False)`;
  - returned four loop-state tensors;
  - each loop state had shape `(1, 4, 2048)` and dtype `bfloat16`.
- Evaluator-boundary compatibility check succeeded:
  - converted RLTT model produced four loop-state tensors;
  - tensors were promoted to float32 and passed into `FrozenCLTAnchor`;
  - epoch-2 CLT evaluator returned a valid `(1, 1)` score and anchor loss.
- Agent-path evaluator check succeeded on CPU:
  - `PairwiseARCSearchAgent(backbone_mode="ouro", ouro_model_path="models/ouro_rltt_local")`;
  - `_ouro_loop_states(np.zeros((8, 8)))` returned four tensors of shape `(1, 5, 2048)`;
  - `_ensure_anchor()` loaded the frozen evaluator;
  - evaluator scored the RLTT loop states without shape/device/dtype errors.

### Notes

- The converted checkpoint is BF16. The source shards were FP32, but existing Ouro runtime already loads with `torch_dtype=torch.bfloat16`; BF16 conversion keeps the local disk footprint and load memory aligned with the intended runtime.
- The downloaded remote-code files emit docstring warnings for undocumented custom forward args (`use_weighted_exit`, `exit_at_step`, `exit_threshold`) under 4.54.1, but loading and the ARC-facing model forward both succeed.
- Compatibility status:
  - mechanically compatible with the ARC agent and frozen CLT evaluator API;
  - not yet proven semantically/calibration-compatible with the published CLT evaluator distribution.
- Required before treating RLTT as a drop-in scientific replacement:
  - run a small CLT/text-preference sanity check through RLTT loop states and compare evaluator behavior against the expected epoch-2 evaluator direction;
  - then run an ARC/Ouro smoke with `--ouro_model_path models/ouro_rltt_local` on GPU.

## RLTT Epoch-2 Pairwise Evaluator Result (2026-05-04)

### What Was Run

- Added a manual evaluator runner:
  - `utilities/evaluator/probes/evaluate_pairwise_rltt.py`.
- This is the archived `evaluate_pairwise.py` path with explicit local-model arguments:
  - HH-RLHF chosen/rejected text;
  - local RLTT/Ouro tokenizer and model;
  - four Ouro/RLTT loop-state tensors captured from the model forward;
  - `claude_sandbox.evaluator_pairwise_codex.PairwiseEvaluator`;
  - epoch-2 evaluator checkpoint.
- The runner forces `use_cache=False` during text forwards.
  - This avoids the downloaded RLTT remote-code `UniversalTransformerCache` setter incompatibility under Transformers `4.54.1`.
  - This does not upgrade or change Transformers.
- The runner lives under `utilities/evaluator/probes/` rather than the project root; it is not named `test_*.py`, so pytest will not collect it automatically.

### Verified Model / Checkpoint

- Full run artifact:
  - `runs/eval_pairwise_rltt_epoch2_full.json`.
- Artifact records:
  - `model_path = "models/ouro_rltt_local"`;
  - `checkpoint = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"`;
  - `dataset = "Anthropic/hh-rlhf"`;
  - `split = "test"`;
  - `device = "cuda"`;
  - `examples_evaluated = 8552`.
- Local model config confirms the converted RLTT model:
  - `model_type = "ouro"`;
  - `architectures = ["OuroForCausalLM"]`;
  - `hidden_size = 2048`;
  - `num_hidden_layers = 48`;
  - `total_ut_steps = 4`;
  - `transformers_version = "4.54.1"`;
  - `auto_map` points at local `configuration_ouro` / `modeling_ouro`.

### Results

- 128-example sanity pass:
  - artifact: `runs/eval_pairwise_rltt_epoch2_128.json`;
  - accuracy `0.9609` (`96.1%`);
  - average score `1.4885`.
- Full cached HH-RLHF test split:
  - artifact: `runs/eval_pairwise_rltt_epoch2_full.json`;
  - examples `8552`;
  - accuracy `0.9498362956` (`95.0%`);
  - positive rate `0.9498362956`;
  - average score `1.6523`;
  - score std `1.0109`;
  - min score `-1.5079`;
  - max score `4.5909`.

### Interpretation

- This is the original CLT/text-preference evaluator path, not the ARC GridEncoder path.
- The result is a major compatibility confirmation:
  - RLTT loop states are not merely shape-compatible with the frozen epoch-2 evaluator;
  - the epoch-2 evaluator still gets the expected ~95% HH-RLHF preference accuracy through the converted RLTT model.
- Current compatibility status:
  - converted RLTT weights are mechanically compatible with the active loader;
  - semantically compatible with the published pairwise evaluator distribution at the thinking-loop level;
  - still requires ARC/Ouro smoke and then wa30/topology runs before declaring it a drop-in ARC replacement.

## RLTT Flip / Spatial Probe Follow-Up (2026-05-04)

### Context

- Paper context checked from arXiv `2604.09870`:
  - relational preference encoding is expected to preserve strong pairwise direction while allowing a learned positive offset;
  - the original epoch-2 flip diagnostic expectation is approximately:
    - strong negative normal-vs-flipped correlation near `-0.94`;
    - strict sign flips only around the mid-20% range;
    - mean `normal + flipped` bias near `+2.51`.
- Therefore the flip probe should not be interpreted as a pure antisymmetry test.
  - The important diagnostics are correlation, sign-flip rate, and the learned offset.

### Added Manual Probes

- `utilities/evaluator/probes/flip_pairwise_rltt.py`
  - batched RLTT version of the archived flip diagnostic;
  - uses `models/ouro_rltt_local`;
  - uses `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`;
  - forces `use_cache=False` for RLTT text forwards.
- `utilities/evaluator/probes/probe_spatial_rltt.py`
  - local RLTT version of the archived spatial/grid probe;
  - encodes synthetic grids as text prompts;
  - captures four RLTT/Ouro loop states;
  - compares grid-vs-text loop divergence and probes simple grid-pattern labels from loop states.

### Flip Probe Result

- Command output artifact:
  - `runs/flip_pairwise_rltt_epoch2_100.json`.
- Run details:
  - model `models/ouro_rltt_local`;
  - checkpoint `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`;
  - dataset `Anthropic/hh-rlhf`, split `test`;
  - samples `100`;
  - device `cuda`.
- Result:
  - normal mean `+1.4781`;
  - flipped mean `+1.0213`;
  - sign flips `26/100` (`26.0%`);
  - antisymmetry correlation `-0.9369`;
  - mean `normal + flipped = +2.4993`;
  - delta from expected `+2.51` bias: `-0.0107`.
- Interpretation:
  - RLTT preserves the relational preference signal:
    - correlation is essentially on-target;
    - sign flips are in the expected range and slightly more common than the remembered baseline;
    - learned positive offset is slightly lower than the expected `+2.51`, not higher.
  - This is arguably better than a naive match:
    - same strong ordering relation;
    - slightly lower constant bias;
    - slightly more frequent true sign reversal under pair order swap.

### Spatial/Grid Probe Result

- Command output artifact:
  - `runs/probe_spatial_rltt_100.json`.
- Run details:
  - model `models/ouro_rltt_local`;
  - synthetic grid samples `100`;
  - grid/text comparison samples `20` each;
  - device `cuda`.
- Loop divergence:
  - grid `L1-L4` cosine `0.73397`;
  - text `L1-L4` cosine `0.65459`;
  - grid total refinement delta `25.68`;
  - text total refinement delta `28.87`.
- Pattern probe:
  - simple synthetic pattern classification is saturated at `1.0000` from `L1` through `L4`;
  - `L1 -> L4` gain is `+0.0000`.
- Interpretation:
  - RLTT loop states are active on text-encoded grids; the grid loop trajectory is not flat.
  - This specific probe is too easy:
    - L1 already encodes enough to solve the synthetic pattern labels perfectly;
    - it cannot establish whether later loops add spatial structure.
  - Next spatial probe should be harder and less text-leaky:
    - pairwise spatial transformations rather than single-grid labels;
    - ARC-like before/after relation questions;
    - direct `inputs_embeds` GridEncoder/Ouro loop-signature probe with converted RLTT, not only grid-as-text.

## RLTT Anti-Saturation Spatial Probe (2026-05-04)

### Motivation

- The simple spatial probe saturated:
  - synthetic pattern labels were already `100%` predictable from `L1`;
  - therefore `L4` could not show improvement even if useful refinement existed.
- Next useful test must use harder relational labels and explicitly test loop-delta features.

### Implemented

- Added:
  - `utilities/evaluator/probes/probe_spatial_antisaturation.py`.
- Probe design:
  - deterministic synthetic examples;
  - same examples for RLTT and base Ouro;
  - seven task families:
    - connected component count;
    - symmetry axis;
    - object containment;
    - path existence;
    - occluded continuation;
    - transformation class;
    - collision after action.
- For each task, train the same probe on:
  - `L1`;
  - `L4`;
  - `L4 - L1`;
  - all four loops concatenated.
- This directly asks whether loop evolution itself carries useful information instead of only preserving `L1`.

### Commands / Artifacts

- RLTT:
  - `runs/probe_spatial_antisat_rltt_80.json`;
  - model `models/ouro_rltt_local`;
  - `80` examples per task;
  - device `cuda`.
- Base Ouro:
  - `runs/probe_spatial_antisat_base_80.json`;
  - model `ByteDance/Ouro-2.6B-Thinking`;
  - same deterministic examples;
  - device `cuda`.

### RLTT Results

| task | chance | L1 | L4 | L4-L1 gain | delta probe | all loops |
|---|---:|---:|---:|---:|---:|---:|
| components | 0.25 | 0.775 | 0.787 | +0.012 | 0.688 | 0.762 |
| symmetry | 0.25 | 0.825 | 0.925 | +0.100 | 0.912 | 0.912 |
| containment | 0.50 | 0.963 | 0.938 | -0.025 | 0.925 | 0.950 |
| path | 0.50 | 0.575 | 0.625 | +0.050 | 0.550 | 0.575 |
| occlusion | 0.50 | 0.688 | 0.787 | +0.100 | 0.762 | 0.825 |
| transform | 0.25 | 0.787 | 0.800 | +0.013 | 0.800 | 0.800 |
| collision | 0.50 | 0.912 | 0.850 | -0.062 | 0.887 | 0.925 |

### Base vs RLTT

| task | base L4 | RLTT L4 | RLTT-base L4 | base gain | RLTT gain |
|---|---:|---:|---:|---:|---:|
| components | 0.825 | 0.787 | -0.037 | +0.050 | +0.012 |
| symmetry | 0.925 | 0.925 | +0.000 | +0.100 | +0.100 |
| containment | 0.863 | 0.938 | +0.075 | -0.100 | -0.025 |
| path | 0.600 | 0.625 | +0.025 | +0.012 | +0.050 |
| occlusion | 0.787 | 0.787 | +0.000 | +0.037 | +0.100 |
| transform | 0.863 | 0.800 | -0.062 | +0.062 | +0.013 |
| collision | 0.850 | 0.850 | +0.000 | -0.075 | -0.062 |

Loop-state geometry was nearly identical between base and RLTT:

| task | base L1-L4 cos | RLTT L1-L4 cos | base total delta | RLTT total delta |
|---|---:|---:|---:|---:|
| components | 0.732 | 0.732 | 26.24 | 26.35 |
| symmetry | 0.772 | 0.777 | 23.20 | 23.27 |
| containment | 0.688 | 0.690 | 27.80 | 27.71 |
| path | 0.692 | 0.692 | 28.60 | 29.39 |
| occlusion | 0.693 | 0.700 | 27.09 | 26.77 |
| transform | 0.705 | 0.709 | 27.11 | 26.94 |
| collision | 0.683 | 0.688 | 27.28 | 27.07 |

### Interpretation

- The anti-saturation probe worked:
  - labels no longer all saturate at `L1`;
  - some tasks show meaningful `L4` improvement;
  - `L4 - L1` predicts several labels above chance, especially symmetry, containment, occlusion, transform, and collision.
- RLTT is not worse in loop geometry:
  - L1-L4 cosine and total refinement delta are essentially base-like.
- RLTT's spatial behavior is mixed but promising:
  - better than base on containment and path;
  - tied on symmetry, occlusion, collision L4;
  - worse on component count and transformation-class L4;
  - better L4-over-L1 gain than base on path and occlusion.
- This is the first probe in this session that actually tests loop refinement rather than L1 encoding alone.
- Next spatial tests should be ARC-shaped pairwise/action probes:
  - input/output choice among decoys;
  - "which action changes the state";
  - "which object is movable";
  - "would these objects collide after the proposed move";
  - direct GridEncoder `inputs_embeds` loop-signature comparison for RLTT vs base.

## Multi-Seed ARC-Shaped Spatial Probe (2026-05-04)

### Implemented

- Extended `utilities/evaluator/probes/probe_spatial_antisaturation.py` with ARC-shaped tasks:
  - `arc_output_choice`:
    - one demonstration input/output pair;
    - one test input;
    - four candidate output grids;
    - label is candidate position, balanced across A/B/C/D;
    - tests whether the loop state retains enough relation/candidate information to identify the output matching the inferred rule.
  - `arc_action_choice`:
    - choose which directional action would move the avatar/object into empty space.
  - `arc_movable_choice`:
    - choose which labeled object can move right into an empty cell.
- Ran larger multi-seed comparison:
  - seeds `123`, `456`, `789`;
  - `100` examples per task per seed;
  - ten tasks;
  - same deterministic examples for base and RLTT;
  - device `cuda`.

### Artifacts

- RLTT:
  - `runs/probe_spatial_antisat_rltt_seed123_100.json`
  - `runs/probe_spatial_antisat_rltt_seed456_100.json`
  - `runs/probe_spatial_antisat_rltt_seed789_100.json`
- Base Ouro:
  - `runs/probe_spatial_antisat_base_seed123_100.json`
  - `runs/probe_spatial_antisat_base_seed456_100.json`
  - `runs/probe_spatial_antisat_base_seed789_100.json`

### Aggregate Results

Mean +/- population std across the three seeds:

| task | base L4 | RLTT L4 | RLTT-base L4 | base L4-L1 gain | RLTT L4-L1 gain | RLTT delta probe | RLTT all loops |
|---|---:|---:|---:|---:|---:|---:|---:|
| components | 0.840 +/- 0.014 | 0.847 +/- 0.005 | +0.007 | +0.057 | +0.043 | 0.720 | 0.847 |
| symmetry | 0.900 +/- 0.050 | 0.887 +/- 0.050 | -0.013 | +0.120 | +0.130 | 0.897 | 0.893 |
| containment | 0.913 +/- 0.024 | 0.920 +/- 0.042 | +0.007 | -0.060 | -0.060 | 0.933 | 0.960 |
| path | 0.643 +/- 0.033 | 0.607 +/- 0.025 | -0.037 | +0.093 | +0.043 | 0.583 | 0.607 |
| occlusion | 0.803 +/- 0.031 | 0.817 +/- 0.019 | +0.013 | -0.127 | -0.110 | 0.843 | 0.877 |
| transform | 1.000 +/- 0.000 | 0.997 +/- 0.005 | -0.003 | +0.000 | -0.003 | 1.000 | 1.000 |
| collision | 0.877 +/- 0.005 | 0.883 +/- 0.019 | +0.007 | -0.017 | -0.013 | 0.890 | 0.907 |
| arc_output_choice | 0.273 +/- 0.009 | 0.277 +/- 0.029 | +0.003 | +0.020 | +0.030 | 0.237 | 0.250 |
| arc_action_choice | 0.513 +/- 0.090 | 0.563 +/- 0.079 | +0.050 | -0.170 | -0.117 | 0.577 | 0.657 |
| arc_movable_choice | 0.723 +/- 0.039 | 0.707 +/- 0.009 | -0.017 | -0.050 | -0.087 | 0.707 | 0.713 |

Loop geometry remains base-like:

| task | base L1-L4 cos | RLTT L1-L4 cos | base total delta | RLTT total delta |
|---|---:|---:|---:|---:|
| components | 0.731 | 0.731 | 26.31 | 26.42 |
| symmetry | 0.773 | 0.778 | 23.23 | 23.28 |
| containment | 0.687 | 0.689 | 27.81 | 27.70 |
| path | 0.692 | 0.692 | 28.70 | 29.46 |
| occlusion | 0.693 | 0.701 | 27.09 | 26.75 |
| transform | 0.706 | 0.710 | 27.07 | 26.90 |
| collision | 0.682 | 0.687 | 27.31 | 27.11 |
| arc_output_choice | 0.655 | 0.659 | 28.71 | 28.64 |
| arc_action_choice | 0.694 | 0.698 | 26.12 | 26.14 |
| arc_movable_choice | 0.697 | 0.699 | 26.11 | 26.12 |

### Interpretation

- Components are not a stable RLTT weakness:
  - RLTT slightly beats base at L4 across seeds (`+0.007`);
  - both models show positive L4-over-L1 gain.
- Transformation class is not informative in the current formulation:
  - both base and RLTT saturate at about `1.0`;
  - this task is too easy and should not be used as evidence of a representational gap.
- Real issues exposed:
  - `arc_output_choice` is near chance for both base and RLTT, despite balanced candidate labels;
  - `arc_action_choice` and `arc_movable_choice` are L1-dominant, with L4 often degrading the linearly decodable answer;
  - this suggests the text-encoded ARC-shaped prompts are not yet producing useful iterative spatial reasoning in the loop states.
- RLTT is not worse than base overall:
  - loop geometry is effectively identical;
  - RLTT is better or tied on components, containment, occlusion, collision, and arc_output_choice L4;
  - base is stronger on path and movable-choice L4;
  - action-choice L4 is better for RLTT but still below its own L1.
- Next diagnostic should move away from pure text-encoded grids:
  - direct GridEncoder `inputs_embeds` loop signature through base vs RLTT;
  - ARC-shaped pairwise scoring where the candidate outputs/actions are represented separately, not all flattened into one long text prompt;
  - harder non-saturated transformation tasks with rule families that cannot be solved by a single local object cue.

## Hybrid Engram Memory Notes From Downloads (2026-05-04)

### Source

- Read:
  - `/home/moloch/Downloads/hybrid_engram_memory_notes.md`.

### Summary

- The note argues DeepSeek V4 hybrid attention should not be copied into Ouro.
- The useful architectural lesson is a memory schedule:
  - exact recent context;
  - compressed coarse episode/mechanism memory;
  - sparse fine retrieval of relevant older memories.
- Translation to Hunter Seeker:
  - local exact transition window;
  - existing topology engrams as fine sparse recall;
  - explicit hazard/terminal engrams;
  - mechanism engrams;
  - coarse episode summaries.
- It reinforces existing constraints:
  - do not modify Ouro;
  - do not rewrite Hunter Seeker;
  - keep negative evidence separate from positive support;
  - conflict should suppress optimism rather than hard-block broadly.

### Opinion

- This is directionally correct and fits the current codebase better than another scoring patch.
- It is not a new sprint requirement right now, but it should become the memory architecture target after the current RLTT/base compatibility and direct GridEncoder loop probes are done.
- Highest-value part for current ARC failures:
  - exact recent transition window;
  - separate hazard/terminal negative recall;
  - conflict-aware aggregation;
  - mechanism summaries for delayed effects.
- Lower priority:
  - learned coarse compressors. Start with deterministic/computed summaries first, then learn compression only after diagnostics show which summary fields matter.
- Implementation caution:
  - existing engram penalties are deliberately bounded and conservative;
  - multi-resolution memory must preserve that discipline:
    - no broad action blacklist;
    - no color-global blacklist;
    - cross-action negative recall stays diagnostic unless high-confidence and action-relevant;
    - all memory terms must land in `score_components`.

## Direct GridEncoder Loop Signature Diagnostics (2026-05-04)

### Implemented

- Added:
  - `utilities/evaluator/probes/probe_gridencoder_loop_signature.py`.
- Diagnostic bypasses text prompts:
  - integer grid;
  - `GridEncoder.encode_for_ouro`;
  - `Ouro.model(inputs_embeds=..., attention_mask=...)`;
  - compare base Ouro vs local RLTT on identical encoded grids.
- Metrics are reported for:
  - CLS loop vectors;
  - patch-mean loop vectors;
  - sequence-mean loop vectors;
  - token-level loop movement;
  - cross-model cosine for each loop and for `L4 - L1`.

### Artifacts

- Corrected Sprint 4 encoder baseline:
  - `runs/probe_gridencoder_loop_signature_sprint4_base_rltt_32.json`.
- Newer observation-learning/content-CLS encoder candidate:
  - `runs/probe_gridencoder_loop_signature_contentcls_base_rltt_32.json`.

### Results

Corrected Sprint 4 baseline:

| model | scope | L1-L4 cosine | total delta | L1-L2 | L2-L3 | L3-L4 |
|---|---|---:|---:|---:|---:|---:|
| base | CLS | 0.6705 | 44.60 | 25.41 | 11.45 | 7.74 |
| RLTT | CLS | 0.6740 | 44.28 | 25.35 | 11.39 | 7.55 |
| base | patch mean | 0.1131 | 90.54 | 48.06 | 28.75 | 13.73 |
| RLTT | patch mean | 0.1135 | 91.13 | 48.27 | 29.02 | 13.84 |
| base | token level | 0.1059 | 104.13 | - | - | - |
| RLTT | token level | 0.1064 | 104.92 | - | - | - |

Content-CLS encoder candidate:

| model | scope | L1-L4 cosine | total delta | L1-L2 | L2-L3 | L3-L4 |
|---|---|---:|---:|---:|---:|---:|
| base | CLS | 0.9405 | 20.66 | 17.87 | 1.56 | 1.23 |
| RLTT | CLS | 0.9411 | 20.52 | 17.75 | 1.53 | 1.24 |
| base | patch mean | 0.0759 | 91.28 | 54.87 | 25.39 | 11.03 |
| RLTT | patch mean | 0.0748 | 91.26 | 55.07 | 25.29 | 10.90 |
| base | token level | 0.0771 | 102.12 | - | - | - |
| RLTT | token level | 0.0761 | 102.17 | - | - | - |

Cross-model alignment:

- Sprint 4 encoder:
  - patch-mean L4 cross-model cosine `0.99946`;
  - patch-mean `L4 - L1` cross-model cosine `0.99977`.
- Content-CLS encoder:
  - patch-mean L4 cross-model cosine `0.99947`;
  - patch-mean `L4 - L1` cross-model cosine `0.99976`.

### Interpretation

- Ouro is not inert on GridEncoder embeddings.
- Patch/token scopes show large iterative movement:
  - roughly `90-105` total delta depending on pooling;
  - low L1-L4 cosine around `0.07-0.11` for patch/token scopes.
- Base and RLTT are almost identical on direct grid embeddings.
- The content-CLS encoder changes CLS behavior sharply:
  - CLS becomes much more converged after L2;
  - patch/token dynamics remain large.
- Current evidence points away from "RLTT broke grid embedding loop dynamics".
- More likely issue:
  - downstream policy/ranker/pooler is not extracting the useful candidate/action signal,
  - or action-validity prompts/representations are still too weak.

## Separate-Candidate ARC Probe (2026-05-04)

### Implemented

- Added:
  - `utilities/evaluator/probes/probe_arc_candidate_separate.py`.
- This replaces flattened A/B/C/D prompts with one prompt per candidate:
  - demo/input context;
  - one candidate output/action/object;
  - binary correct/incorrect label.
- Evaluation uses grouped cross-validation by problem id:
  - sibling candidates from the same problem never leak across folds.
- Main metric:
  - held-out choice accuracy: choose candidate with highest binary probe score inside each problem.

### Artifacts

- RLTT:
  - `runs/probe_arc_candidate_separate_rltt_seed123_100.json`.
- Base Ouro:
  - `runs/probe_arc_candidate_separate_base_seed123_100.json`.

### Results

Overall held-out choice accuracy:

| model | L1 | L4 | L4-L1 | all loops |
|---|---:|---:|---:|---:|
| base | 0.257 | 0.310 | 0.333 | 0.280 |
| RLTT | 0.270 | 0.313 | 0.333 | 0.300 |

By task:

| task | model | L1 | L4 | L4-L1 | all loops |
|---|---|---:|---:|---:|---:|
| arc_output_choice | base | 0.390 | 0.650 | 0.610 | 0.570 |
| arc_output_choice | RLTT | 0.390 | 0.610 | 0.590 | 0.550 |
| arc_action_choice | base | 0.140 | 0.170 | 0.170 | 0.180 |
| arc_action_choice | RLTT | 0.130 | 0.190 | 0.220 | 0.200 |
| arc_movable_choice | base | 0.440 | 0.400 | 0.470 | 0.370 |
| arc_movable_choice | RLTT | 0.420 | 0.290 | 0.300 | 0.300 |

Binary AUC / margin notes:

- `arc_output_choice` becomes clearly readable when candidates are scored separately:
  - base L4 AUC `0.728`, margin `2.321`;
  - RLTT L4 AUC `0.735`, margin `2.572`.
- `arc_action_choice` remains weak and often inverted:
  - all AUCs around `0.47-0.49`;
  - negative score margins for both models.
- `arc_movable_choice` is above chance but mostly L1-dominant:
  - base is stronger than RLTT in this formulation;
  - RLTT degrades from L1 to L4.

### Interpretation

- The previous flattened ARC-output probe was hiding useful structure.
- Output candidate correctness is decodable from loop states when each candidate is represented separately.
- L4 is materially better than L1 for output choice.
- Action validity is the current weak point:
  - it is not just a topology runtime issue;
  - even frozen Ouro/RLTT text-loop states do not linearly expose this action-validity relation well in the current prompt format.
- Next diagnostics should focus on action representation:
  - direct GridEncoder candidate/action scoring rather than text-only action descriptions;
  - compare ranker/pooler readouts on chosen vs counterfactual candidate states;
  - inspect whether the policy is reading CLS only while the strongest action signal is in patch tokens or deltas.

## DeepSeek-Inspired Memory Before Ladder Decision (2026-05-04)

Decision:

- Do not enact the full DeepSeek-inspired hybrid memory architecture before the ladder.
- Do enact only low-risk, identity-start, diagnostic-first pieces before the ladder if time permits.

Allowed before ladder:

- recent exact transition window;
- separate hazard/terminal memory namespaces;
- coarse episode summaries as diagnostics;
- conflict-aware support diagnostics:
  - positive support;
  - negative support;
  - conflict flag;
  - best positive/negative similarity;
- score-component logging for every new memory term.

Not allowed before ladder:

- learned hybrid attention;
- learned memory compressors;
- broad action blacklists;
- broad label/color blacklists;
- strong engram steering that changes policy behavior before the ladder baseline is measured;
- any domain/game-specific hardcoding.

Rationale:

- The ladder needs to isolate effects.
- Memory structure that only records and reports evidence is useful and low-risk.
- New learned retrieval/attention or strong memory penalties would make the ladder less interpretable.
- If implemented before ladder, memory terms must remain conservative:
  - diagnostic-only or very low weight;
  - bounded penalties;
  - all terms exposed in `score_components`;
  - no broad generalization from one candidate/action to unrelated candidates/actions.

## Pre-Ladder Diagnostic Memory Patch (2026-05-04)

### Implemented

- `claude_sandbox/observation_learning_codex.py`
  - `TransitionEffectEngram` now carries a `namespace`:
    - `terminal`;
    - `hazard`;
    - `progress`;
    - `change`;
    - `noop`;
    - `unknown`.
  - `TransitionEffectEngramMemory` now:
    - classifies outcome records into namespaces;
    - serializes/deserializes namespaces;
    - reports namespace counts;
    - emits namespace support/best-similarity/match-count fields in recall.
  - Added `RecentExactTransitionWindow`:
    - stores a bounded recent window of exact before/after frame hashes;
    - stores generic effect vectors and state vectors;
    - records action/click/outcome/source/step/confidence;
    - returns diagnostic-only exact-transition fields.

- `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`
  - Instantiates `recent_exact_transition_window`.
  - `_observe_transition_for_learning()` and `ingest_observation_video()` feed the recent exact window.
  - Candidate observation diagnostics now include recent exact transition fields even when neural observation diagnostics are skipped by cadence.
  - Observation summary now reports:
    - `transition_effect_engram_namespaces`;
    - `recent_exact_transition_window`.
  - Checkpoints persist and reload recent exact transition records.

- `claude_sandbox/arc_agent_hunter_seeker_codex.py`
  - `EngramRecord` now carries a `namespace`.
  - Hunter engram collection/retrieval now reports namespace support/best-similarity/match-count fields.
  - `summarize_engram_memory()` reports `by_namespace`.
  - Added coarse episode summaries:
    - terminal/reset summaries;
    - trace counts;
    - frame-change rate;
    - selection-method counts;
    - hazard/engram pressure summaries;
    - engram namespace summary;
    - observation engram namespace summary;
    - recent exact transition window summary.
  - `measurement_summary()` now emits `coarse_episode_memory`.
  - Checkpoints persist/reload coarse episode summaries.
  - Score-component compactors preserve all new namespace and recent-exact fields via prefix handling:
    - `engram_namespace_*`;
    - `obs_engram_namespace_*`;
    - `obs_recent_exact_*`.

### Behavioral Status

- This patch is diagnostic-first.
- It does not add learned memory attention.
- It does not add broad action blacklists.
- It does not add broad label/color blacklists.
- It does not add strong new policy steering.
- Recent exact transition matching is exposed in score components but is not itself a policy score term.
- Namespace aggregation makes positive/negative/hazard/terminal evidence auditable before any later tuning.

### Verification

- `py_compile` passed for:
  - `claude_sandbox/observation_learning_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - focused test files.
- Focused observation-memory tests:
  - `7 passed, 167 deselected`.
- Focused hazard/engram/compaction tests:
  - `9 passed, 130 deselected`.

### RLTT Architecture Validity

- Current evidence says RLTT is valid for the architecture.
- No architecture fork is needed just to use RLTT.
- Evidence:
  - pairwise evaluator still works at roughly the trusted level;
  - RLTT flip tests showed lower bias / more useful sign flips than expected;
  - direct GridEncoder-to-Ouro loop probes showed base/RLTT cross-model cosines around `0.999` on patch/sequence loop dynamics;
  - separate-candidate ARC probes show RLTT exposes output-choice structure similarly to base.
- The likely RLTT work is calibration, not architecture redesign:
  - ranker/evaluator score scale;
  - loop-delta trust thresholds;
  - action-validity readout;
  - patch/token pooling vs CLS-only use;
  - candidate/action representation probes.

### Plan To Keep

Before the full ladder:

1. Keep RLTT/base compatibility diagnostics in the state file.
2. Keep the new memory scaffolding diagnostic-only unless a run proves a specific bounded term is needed.
3. Do not start learned hybrid memory or compressors before ladder baselines.
4. Preserve Transformers `4.54.1`.
5. If RLTT is used for the ladder, compare scalar calibration against base rather than changing architecture.

After the ladder or after an RLTT-weighted encoder acquisition pass:

1. Decide whether any namespace/recent-exact fields should become bounded behavior terms.
2. Expand action-validity probes around direct GridEncoder candidate/action states.
3. Revisit learned coarse memory compression only after deterministic coarse summaries show signal.
4. Continue observation learning toward video/general watching:
   - changed mask;
   - effect summary;
   - inverse action;
   - conservative transition engrams;
   - later self-value/internalized evaluator signal.

## RLTT Pre-Acquisition Hardening (2026-05-04)

### Implemented Before Tonight's RLTT Checkpoint Acquisition

- Added passive calibration logging in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- Added `passive_calibration_summary()` to `measurement_summary()`.
- Added `utilities/evaluator/run_post_rltt_probe_bundle.py`.
- Added/updated focused tests for:
  - calibration score-component preservation;
  - passive calibration raw field computation;
  - recent exact transition diagnostics;
  - engram namespace diagnostics.

### Passive Calibration Fields

The patch is measurement-only. It does not change scoring.

Chosen score components now include raw calibration diagnostics with `calibration_*` keys:

- chosen score and runner-up score;
- chosen-vs-runner-up raw gap;
- candidate-count and score distribution statistics;
- raw ranker logit;
- raw transition/prior normalized scores;
- raw world-model/effective confidence;
- raw loop delta, loop-delta EMA, and loop-delta ratio;
- raw delta-trust;
- raw Ouro confidence / expected exit / trust multiplier when present;
- raw inverse-action probability when observation diagnostics produce it;
- action-validity proxy;
- anchor success rate and anchor loss EMA;
- placeholder normalized values for later calibration.

Compactors now preserve all `calibration_*` fields automatically.

Current policy:

- Do not tune these values before RLTT checkpoint acquisition.
- Use them to fit later model-agnostic scalar calibration:
  - score scale;
  - score gap;
  - loop-delta trust;
  - action-validity proxy;
  - evaluator/ranker margin interpretation.

### Probe Bundle

New script:

`utilities/evaluator/run_post_rltt_probe_bundle.py`

Default bundle:

- `utilities/evaluator/probes/evaluate_pairwise_rltt.py`;
- `utilities/evaluator/probes/flip_pairwise_rltt.py`;
- `utilities/evaluator/probes/probe_spatial_rltt.py`;
- `utilities/evaluator/probes/probe_spatial_antisaturation.py`;
- `utilities/evaluator/probes/probe_arc_candidate_separate.py`;
- `utilities/evaluator/probes/probe_gridencoder_loop_signature.py`.

Dry-run command tested:

```bash
/home/moloch/ouro_project/venv/bin/python utilities/evaluator/run_post_rltt_probe_bundle.py --dry-run --offline --output-dir runs/post_rltt_probe_bundle_dryrun --skip-base-compare --eval-examples 2 --flip-samples 2 --spatial-samples 4 --anti-saturation-samples 4 --arc-samples 4 --grid-samples 4
```

Dry run wrote:

`runs/post_rltt_probe_bundle_dryrun/manifest.json`

Recommended post-acquisition run shape:

```bash
/home/moloch/ouro_project/venv/bin/python utilities/evaluator/run_post_rltt_probe_bundle.py --offline --output-dir runs/post_rltt_probe_bundle_rltt_acquired --model-path models/ouro_rltt_local --model-label rltt --encoder-checkpoint checkpoints_running/sprint4_encoder_reverted.pt --device cuda --hash-model-files
```

Use `--skip-base-compare` if the base model is not locally cached.

### RLTT Source Metadata

Local model path:

`models/ouro_rltt_local`

Size:

- `5.0G`.

Key hashes:

- `config.json`
  - `7d6764dbc8210d023c8d83da4620910808ac5a450532b15550e57d1ef0e4f741`
- `model.safetensors.index.json`
  - `e1842668e1ba1568364a4ae7227a5ab80ab5403c95baba37b92205d2cb22a001`
- `model-00001.safetensors`
  - `9f49e55b0d0ea368f2942cfd3fa256836ea7fa9002c6f860ca2d85ac1de252a1`
- `model-00002.safetensors`
  - `97a0d531a2c624dd98e60c0ebc46a7cc0413642f8cba6df7dac8df01a432df19`
- `model-00003.safetensors`
  - `0bf3d0c111c876ccc7f7ebaeb7cabc8dfccffde30acb4cd780a4c364d3679aad`

Evaluator and encoder hashes:

- `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`
  - `3630c2092eca8db13239f763bc9c212f4b673866e47f811c3095efc57409ec96`
- `checkpoints_running/sprint4_encoder_reverted.pt`
  - `c9675f3e0c49f5856487075de09ee5423b17cc1964c06e08674cf954199157bc`
- `claude_sandbox/checkpoints_encoder_retrain/encoder_content_cls_candidate_20260504.pt`
  - `4bc3c258ba1c3955e5546e7c8a9d0f473de68bb863c529eb111e2ca6e90d8f58`

Disk space before acquisition:

- filesystem size `929G`;
- used `591G`;
- available `292G`.

### Preflight Results

- `transformers` remains `4.54.1`.
- CUDA is visible.
- GPU at preflight:
  - RTX 5070 Ti Laptop GPU;
  - temperature around `44-46 C`;
  - memory around `912-917 MiB / 12227 MiB`.
- RLTT tokenizer loads locally:
  - vocab size `49152`.
- RLTT model loads locally on CUDA:
  - class `OuroForCausalLM`;
  - first parameter device `cuda:0`.
- Epoch-2 pairwise evaluator loads on CUDA:
  - `4,729,985` parameters;
  - first parameter device `cuda:0`.

Important command quirk:

- Prefixing the command itself with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ...` triggered a CUDA lazy-init failure in one preflight.
- Setting those variables inside Python, or passing them via the wrapper subprocess environment, worked.
- For tonight's acquisition/probe runs, prefer:
  - the wrapper script;
  - or set offline variables inside the launched Python process/environment rather than relying on a shell-prefix form.

### Verification

- `py_compile` passed for:
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `utilities/evaluator/run_post_rltt_probe_bundle.py`;
  - updated tests.
- Focused passive-calibration tests:
  - `2 passed, 138 deselected`.
- Full CPU test suite:
  - `430 passed, 1 skipped`.

### Current Decision

- No behavior calibration has been enacted yet.
- RLTT remains architecture-valid.
- The next calibration step should happen after checkpoint acquisition/probe outputs exist.
- Calibration should be fitted from distributions, not hand-tuned around wa30:
  - score gap percentiles;
  - ranker logit scale;
  - loop-delta ratio;
  - evaluator margin/anchor pressure;
  - action-validity proxy.

### End Status For This Handoff

- The pre-acquisition hardening patch is complete.
- The code now logs passive calibration diagnostics but does not use them for behavior.
- The post-RLTT probe bundle runner exists and has been dry-run validated.
- RLTT local weights, epoch-2 evaluator, and relevant encoder checkpoints have hashes recorded above.
- CUDA/RLTT/evaluator preflight passed under `transformers==4.54.1`.
- Full CPU verification passed: `430 passed, 1 skipped`.
- No ladder, topology, or acquisition behavior patch should be added before the next checkpoint acquisition run unless a concrete bug appears.

