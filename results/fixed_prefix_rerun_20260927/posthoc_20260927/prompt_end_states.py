"""Post-hoc: hidden state at the last prompt token (layers 24/36/47 x 4 loops), one per Run 1 task, from the
saved prompt IDs. Same capture as Run 3's gen_run3.forward_all (no attention mask)."""
import sys, glob, torch
ROOT = "/home/moloch/ouro_project"
R1 = ROOT + "/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path[:0] = [ROOT, ROOT + "/shared"]
from src.evaluator.bg_transformer_features import (BGTransformerFeatureExtractor, _LoopLayerCapture, TAP_LAYERS,
                                                   LAYER_TO_CAPTURE_INDEX, NUM_LOOPS)
ext = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
prompts = {}
for f in sorted(glob.glob(R1 + "/run1_shard_off*.pt")):
    for r in torch.load(f, map_location="cpu", weights_only=False)["records"]:
        prompts.setdefault(r["task_uid"], r["prompt_ids"])
out = {}
for i, (uid, ids) in enumerate(prompts.items()):
    cap = _LoopLayerCapture()
    with torch.inference_mode(), ext._capture_hooks(cap):
        ext.model(input_ids=torch.tensor([ids], device=ext.device), use_cache=False)
    ext._validate_capture(cap)
    states = {47: cap.boundary, **{l: cap.inter[k] for l, k in LAYER_TO_CAPTURE_INDEX.items()}}
    out[uid] = torch.stack([torch.stack([states[l][u][0, -1].float().cpu() for u in range(NUM_LOOPS)]) for l in TAP_LAYERS])
torch.save(out, R1 + "/posthoc_20260927/prompt_end_states.pt")
print("saved", len(out), "states of shape", tuple(next(iter(out.values())).shape))
