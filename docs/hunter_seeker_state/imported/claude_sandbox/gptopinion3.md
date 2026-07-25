<!-- Imported from `claude_sandbox/gptopinion3.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: c5f6de65d7bc6124c7f7a772bc4384587bfe7a244541cfb156fef0ae9890b20d; original line count: 81. -->

1. Dynamic shapes are only half-fixed

The encoder now auto-pads grids to a patch_size multiple, which is good. But downstream code still assumes the original frame/mask dimensions divide cleanly by patch_size. The clearest example is pool_object_features: it computes pH = H // patch_size, pW = W // patch_size, then reshapes the raw object mask into (pH, patch_size, pW, patch_size). That will crash for a 30×30, 31×29, etc. mask.

This same class of bug likely affects next-frame / patch-color supervision too: encoder features may be on the padded patch grid, while target frames/masks may still be original-size. The fix is to centralize padding metadata: every encoder call should return or store original (H, W), padded (Hp, Wp), and crop/pad rules. Then object masks, patch-color targets, next-frame outputs, and click maps should all use the same spatial convention.

Priority: high if you care about non-64×64 / non-divisible domains. ARC-AGI-3 64×64 hides this.

2. SpatialClickPredictor can produce wrong-size maps on odd dimensions

The spatial predictor uses a stride-2 conv and then a stride-2 transpose conv. For even sizes that usually returns the original size; for odd sizes it can return one pixel short. Candidate generation then blends saliency and spatial, expecting both to be the same (H, W) shape.

Easy fix: after self.conv(x), interpolate the output back to frame_tensor.shape[-2:], exactly like the generic click head already interpolates to out_shape.

3. The observation adapter contract is not fully honored

The adapter docs say dense_input(obs) is the canonical GridEncoder input, while current_frame(obs) may be raw/pre-segmentation data. That is clean in theory.

But some core paths still build tensors directly from raw numpy frames inside the agent, especially encode_and_think_batch, which takes frames.astype(np.int64) and feeds them to the encoder. That works for ARC and the mock symbolic adapter because raw frame equals label grid. It will break for RGB/image-like domains where current_frame is an image and dense_input is the quantized label grid.

Fix: separate the concepts hard:

segmented_frame / current_frame for scene parsing and visual dumps.
dense_input for anything going into GridEncoder.
If you need to encode predicted frames during beam search, define a canonical “agent frame” type and make the adapter own conversion both ways.

Priority: medium now, high before real non-ARC experiments.

4. Self-model and cortex-monitor optimizers look dead

The self-model and temporal-context aggregator are constructed with an optimizer. The self-model file also says deviation from zero should become a learned signal through ranker/aux gradients. But the base train_step trains ranker, prior, pooler, spatial predictor, and encoder/world-model heads; I do not see self-model or cortex-monitor optimizer stepping there.

Also, encode_and_think_batch explicitly detaches the context token inside the no-grad Ouro path. So the self-model may affect inference values, but it is probably not learning from those effects.

Same issue for the cortex monitor: it creates cortex_monitor_optimizer, but this looks like another optimizer without a real training path.

Fix options:

Rename current modes honestly: passive_trace, inject_detached.
Or add an actual self-model loss and call self_model_optimizer.step().
Or compute temporal features inside the ranker training step without detaching/storing them as replay constants.

Priority: high if Sprint 11b is supposed to learn, low if it is currently only instrumentation.

5. ActionHead.select_action can crash on empty/invalid action masks

The action head masks unavailable actions by setting all other logits to -inf, then softmaxes and samples. If available_actions=[], or if all provided actions are outside [0, n_actions), then every logit becomes -inf, softmax becomes NaN, and torch.multinomial can fail.

Fix: after applying the mask, check whether at least one valid action was enabled. If not, fall back to a safe default from the action adapter, or to all non-click actions.

Priority: medium, easy fix.

6. Partial checkpoint load silently freezes the encoder

load_partial_checkpoint_for_sandbox loads matching tensors, then sets agent.freeze_encoder = True and freezes encoder params. That is good for diagnostics or “use trusted v17b encoder as frozen retina,” but surprising if someone used partial load expecting continued training.

Also, train_arc_codex.py constructs the agent without passing a freeze_encoder argument, so it inherits the base default.

Fix: expose --freeze_encoder / --unfreeze_encoder, and make partial load print a loud line like:

Partial load forced freeze_encoder=True. Use --unfreeze_encoder_after_load to override.

Priority: medium-high, because silent training configuration bugs waste entire sweeps.

7. online_trace_run_report.py may have a missing dependency

It imports claude_sandbox.sandbox_sweep_validate.MockSweepEnv, but that file was not among the uploaded set. The fallback still imports from claude_sandbox.sandbox_sweep_validate, not from a local unsuffixed module.

So unless that module exists elsewhere in the actual repo, this diagnostic script will fail immediately.

Fix: either include sandbox_sweep_validate.py, or move MockSweepEnv into the diagnostic file / a small shared mock-env module.

Priority: low-medium, but annoying.

8. pad_grids_to_batch is still ARC-defaulted

The utility defaults pad_value=16. That contradicts the domain-generic encoder story, where pad_value should be n_values from the adapter.

If this is only a test helper, not a big deal. If any real path uses it for mock/non-ARC domains, it will pad with an out-of-range token for n_values < 16, causing embedding index errors.

Fix: remove the default or make callers pass adapter.pad_value.
