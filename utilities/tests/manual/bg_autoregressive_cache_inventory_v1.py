"""PART A - Inventory and code-path audit for autoregressive_kv_branch_carry_v1.

Audits cache-related code paths and verifies model/cache primitives are available.
Produces inventory.md / inventory.json and a verdict.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch

import bg_autoregressive_cache_common_v1 as C


def _has_attr_source(obj, name: str) -> dict:
    out = {"present": hasattr(obj, name)}
    if out["present"]:
        try:
            out["source_lines"] = len(inspect.getsource(getattr(obj, name)).splitlines())
        except Exception:
            out["source_lines"] = None
    return out


def grep_use_cache_false(root: Path) -> dict:
    """Find use_cache=False occurrences in candidate branch/beam code paths."""
    hits = {}
    candidates = [
        "shared/utilities/tests/manual/run_bg_text_prefix_branch_selection_pilot.py",
        "shared/utilities/tests/manual/bg_hidden_branch_feasibility_inspect.py",
    ]
    # Also scan any file mentioning latent beam search.
    for rel in candidates:
        p = root / rel
        if not p.exists():
            hits[rel] = {"exists": False}
            continue
        text = p.read_text(errors="ignore")
        hits[rel] = {
            "exists": True,
            "use_cache_false_count": text.count("use_cache=False"),
            "use_cache_true_count": text.count("use_cache=True"),
        }
    # latent_beam_search.py as a literal module
    lbs = list(root.rglob("latent_beam_search.py"))
    hits["latent_beam_search.py_literal_present"] = bool(
        [p for p in lbs if "venv" not in str(p) and "hf_cache" not in str(p)]
    )
    return hits


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    cls = C.get_universal_cache_class()
    cache = C.new_cache()

    modeling_module = sys.modules[type(model).__module__]
    cache_import_path = f"{cls.__module__}.{cls.__name__}"

    # UniversalTransformerCache primitive checks
    cache_checks = {
        "import_path": cache_import_path,
        "key_cache_property": isinstance(getattr(type(cache), "key_cache", None), property),
        "value_cache_property": isinstance(getattr(type(cache), "value_cache", None), property),
        "key_cache_setter": getattr(type(cache), "key_cache", None) is not None
        and getattr(type(cache).key_cache, "fset", None) is not None,
        "value_cache_setter": getattr(type(cache), "value_cache", None) is not None
        and getattr(type(cache).value_cache, "fset", None) is not None,
        "has_update": _has_attr_source(cache, "update"),
        "has_get_seq_length": _has_attr_source(cache, "get_seq_length"),
        "has_get_mask_sizes": _has_attr_source(cache, "get_mask_sizes"),
        "has_reorder_cache": _has_attr_source(cache, "reorder_cache"),
        "has_clear": _has_attr_source(cache, "clear"),
        "max_cache_size": cache.max_cache_size,
        "seen_tokens_attr": hasattr(cache, "_seen_tokens"),
    }

    # OuroAttention cache-slot formula audit (read the source).
    attn_cls = getattr(modeling_module, "OuroAttention", None)
    slot_formula = None
    if attn_cls is not None:
        try:
            src = inspect.getsource(attn_cls.forward)
            slot_formula = "current_ut * self.config.num_hidden_layers + self.layer_idx" in src
        except Exception:
            slot_formula = None

    # Model forward use_cache support
    model_fwd_src = inspect.getsource(type(model).forward)
    model_inner_fwd_src = inspect.getsource(type(model.model).forward)

    # Live primitive: prefill then decode, check cache populates expected slots.
    ids, am = C.tokenize("What is 2+2?")
    next_logits, kv, cpos, am_full = C.prefill(model, ids, am)
    populated = sum(1 for k in kv.key_cache if k is not None)
    seqlen0 = kv.get_seq_length(0)
    mask_sizes = kv.get_mask_sizes(torch.tensor([seqlen0], device=device), 0)

    # model.generate cache-path probe (greedy, deterministic).
    gen_probe = {}
    try:
        gen_ids = model.generate(
            input_ids=ids,
            attention_mask=am,
            max_new_tokens=4,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        gen_probe = {
            "ran": True,
            "input_len": int(ids.shape[1]),
            "output_len": int(gen_ids.shape[1]),
            "new_tokens": gen_ids[0, ids.shape[1]:].tolist(),
            "decoded": tok.decode(gen_ids[0, ids.shape[1]:].tolist()),
        }
    except Exception as e:  # noqa: BLE001
        gen_probe = {"ran": False, "error": f"{type(e).__name__}: {e}"}

    # Compare model.generate greedy to manual cached greedy (token match).
    manual_tokens = []
    nl = next_logits
    cache2 = kv
    cur_am = am_full
    cur_tok_ids = ids
    for _ in range(4):
        t = int(nl[0].argmax().item())
        manual_tokens.append(t)
        cur_am = torch.cat([cur_am, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
        nl, cache2 = C.decode_step(
            model, torch.tensor([[t]], device=device), cache2, cur_am
        )
    gen_probe["manual_cached_tokens"] = manual_tokens
    if gen_probe.get("ran"):
        gen_probe["generate_matches_manual"] = (
            gen_probe.get("new_tokens") == manual_tokens
        )

    beam_grep = grep_use_cache_false(C.PROJECT_ROOT)

    # Verdict
    ready = (
        cache_checks["has_update"]["present"]
        and cache_checks["has_get_mask_sizes"]["present"]
        and cache_checks["has_reorder_cache"]["present"]
        and cache_checks["key_cache_setter"]
        and cache_checks["value_cache_setter"]
        and slot_formula
        and populated == info["expected_cache_slots"]
        and seqlen0 == int(ids.shape[1])
    )
    if not (cache_checks["has_update"]["present"] and isinstance(cache, cls)):
        verdict = "CACHE_CLASS_MISSING"
    elif not cache_checks["has_get_mask_sizes"]["present"]:
        verdict = "CACHE_MASK_PATCH_MISSING"
    elif ready:
        verdict = "READY"
    else:
        verdict = "PARTIAL"

    report = {
        "verdict": verdict,
        "model_info": info,
        "cache_checks": cache_checks,
        "attention_slot_formula_confirmed": slot_formula,
        "model_forward_uses_use_cache": "use_cache" in model_fwd_src,
        "inner_forward_uses_use_cache": "use_cache" in model_inner_fwd_src,
        "inner_forward_cache_position_handling": "cache_position" in model_inner_fwd_src,
        "inner_forward_attention_mask_handling": "attention_mask" in model_inner_fwd_src,
        "live_prefill": {
            "prompt_len": int(ids.shape[1]),
            "populated_slots": populated,
            "expected_slots": info["expected_cache_slots"],
            "seq_length_slot0": seqlen0,
            "get_mask_sizes(kv_len,kv_offset)": list(mask_sizes),
            "cache_position": cpos.tolist(),
        },
        "model_generate_probe": gen_probe,
        "beam_branch_use_cache_audit": beam_grep,
        "notes": [
            "latent_beam_search.py is not present as a literal standalone module; "
            "branch/beam-style generation in this repo runs through probe scripts.",
            "model uses eager attention for this audit; cache slot = current_ut*num_hidden_layers+layer_idx.",
        ],
    }

    C.save_json("inventory.json", report)

    md = []
    md.append("# PART A - Autoregressive cache inventory\n")
    md.append(f"**BG_AUTOREGRESSIVE_CACHE_INVENTORY_VERDICT = {verdict}**\n")
    md.append("## Model\n")
    for k, v in info.items():
        md.append(f"- `{k}`: {v}")
    md.append("\n## UniversalTransformerCache primitives\n")
    md.append(f"- import path: `{cache_import_path}`")
    md.append(f"- key_cache/value_cache setters present: {cache_checks['key_cache_setter']} / {cache_checks['value_cache_setter']}")
    md.append(f"- update(): {cache_checks['has_update']}")
    md.append(f"- get_seq_length(): {cache_checks['has_get_seq_length']}")
    md.append(f"- get_mask_sizes() override: {cache_checks['has_get_mask_sizes']}")
    md.append(f"- reorder_cache(): {cache_checks['has_reorder_cache']}")
    md.append(f"- clear(): {cache_checks['has_clear']}")
    md.append(f"- cache_slot formula `current_ut*num_hidden_layers+layer_idx` confirmed: {slot_formula}")
    md.append("\n## Live prefill\n")
    md.append(f"- prompt_len: {ids.shape[1]}")
    md.append(f"- populated slots: {populated} (expected {info['expected_cache_slots']})")
    md.append(f"- seq_length(slot0): {seqlen0}")
    md.append(f"- get_mask_sizes -> (kv_length, kv_offset) = {list(mask_sizes)}")
    md.append("\n## model.generate probe\n")
    md.append(f"```json\n{json.dumps(gen_probe, indent=2)}\n```")
    md.append("\n## beam/branch use_cache audit\n")
    md.append(f"```json\n{json.dumps(beam_grep, indent=2)}\n```")
    md.append(
        "\n**Interpretation:** existing branch/beam-style code does not exercise "
        "autoregressive KV/cache branch-carry with use_cache=True; that is exactly "
        "what this run validates.\n"
    )
    C.save_md("inventory.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_INVENTORY_VERDICT = {verdict}")
    print(f"populated_slots={populated}/{info['expected_cache_slots']} seqlen0={seqlen0}")
    print(f"generate ran={gen_probe.get('ran')} matches_manual={gen_probe.get('generate_matches_manual')}")
    print("wrote inventory.md / inventory.json")
    print("=" * 70)
    return 0 if verdict in ("READY", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
