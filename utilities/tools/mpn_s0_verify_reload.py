"""M+N S0 — verify the BACKED-UP assets actually reload (not just checksum-match).

Reads the S0 manifest and proves: (1) the converted HF backbone reloads from the backup on CPU
(trust_remote_code, local_files_only, use_cache=False) and produces finite logits on a tiny
forward; (2) each tap/value .pt torch-loads to a non-empty object; (3) one FSDP model shard
torch-loads (weights_only=True) as a valid checkpoint. Checksums already prove byte-integrity;
this proves loadability. Exits non-zero on any failure.

Run after the backup completes: venv/bin/python utilities/tools/mpn_s0_verify_reload.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

DEST = Path("/home/moloch/ouro_backups/mpn_s0_pre_run_2026-06-13")


def main() -> int:
    man = json.loads((DEST / "S0_BACKUP_MANIFEST.json").read_text())
    if not man.get("all_verified"):
        print("REFUSING: backup manifest says all_verified=False; fix the copy before reload-verify.")
        return 2
    results: dict = {}
    ok = True

    # 1) operational backbone reload + tiny forward (CPU)
    model_dir = DEST / "backbone_operational" / "ouro_rltt_local"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, low_cpu_mem_usage=True, device_map={"": "cpu"}).eval()
        ids = tok("The capital of France is", return_tensors="pt").input_ids
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else out)
        finite = bool(torch.isfinite(logits).all())
        results["backbone_reload"] = {"ok": finite, "hidden_size": cfg.hidden_size,
                                      "layers": cfg.num_hidden_layers,
                                      "total_ut_steps": getattr(cfg, "total_ut_steps", None),
                                      "logits_shape": list(logits.shape)}
        ok = ok and finite
        del model
    except Exception as e:
        results["backbone_reload"] = {"ok": False, "error": repr(e)[:300]}
        ok = False

    # 2) tap/value .pt loadability
    import torch
    tap_labels = [k for k in man["assets"] if k.startswith("tap_") or k.startswith("evaluator_")]
    for label in tap_labels:
        a = man["assets"][label]
        if a.get("status") == "ABSENT_OPTIONAL":
            continue
        try:
            if a["kind"] == "dir":
                pts = list(Path(a["dest"]).rglob("*.pt"))
                loaded = 0
                for p in pts:
                    obj = torch.load(p, map_location="cpu", weights_only=False)
                    loaded += 1 if obj is not None else 0
                results[label] = {"ok": loaded == len(pts) and loaded > 0, "n_pt": len(pts)}
                ok = ok and results[label]["ok"]
            elif str(a["dest"]).endswith(".json"):
                obj = json.loads(Path(a["dest"]).read_text())
                good = obj is not None and (len(obj) > 0 if hasattr(obj, "__len__") else True)
                results[label] = {"ok": bool(good), "kind": "json"}
                ok = ok and bool(good)
            else:
                obj = torch.load(a["dest"], map_location="cpu", weights_only=False)
                good = obj is not None and (len(obj) > 0 if hasattr(obj, "__len__") else True)
                results[label] = {"ok": bool(good)}
                ok = ok and bool(good)
        except Exception as e:
            results[label] = {"ok": False, "error": repr(e)[:200]}
            ok = False

    # 3) one FSDP model shard loads (weights_only=True, after importing dtensor)
    try:
        import torch.distributed.tensor  # noqa: F401  (registers DTensor for safe load)
        shard = DEST / "backbone_rltt_fsdp_source" / "RLTT" / "model_world_size_4_rank_0.pt"
        sd = torch.load(shard, map_location="cpu", weights_only=True)
        results["fsdp_model_shard_rank0"] = {"ok": len(sd) > 0, "n_entries": len(sd)}
        ok = ok and len(sd) > 0
    except Exception as e:
        results["fsdp_model_shard_rank0"] = {"ok": False, "error": repr(e)[:200]}
        ok = False

    results["ALL_RELOAD_OK"] = ok
    (DEST / "S0_RELOAD_VERIFY.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nALL_RELOAD_OK = {ok}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
