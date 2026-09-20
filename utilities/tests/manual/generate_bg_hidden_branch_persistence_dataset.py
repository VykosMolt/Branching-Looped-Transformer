"""Generate same-prefix hidden-origin branch persistence data."""
from __future__ import annotations

import itertools
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from bg_hidden_branch_suite_common import (
    REPORT_ROOT,
    ensure_report_root,
    finite,
    load_json,
    load_task_subset,
    md_table,
    rel,
    tensor_to_json_stats,
    write_csv,
    write_json,
    write_md,
)
from src.evaluator.bg_controller import BGController
from src.evaluator.bg_hidden_branching import (
    HIDDEN_DIM,
    HiddenDeltaLayerHook,
    cosine,
    delta_rms,
    make_branch_deltas,
    rms_distance,
)


OUT_PT = REPORT_ROOT / "hidden_branch_persistence.pt"
OUT_JSON = REPORT_ROOT / "hidden_branch_persistence.json"
OUT_MD = REPORT_ROOT / "hidden_branch_persistence.md"
OUT_CSV = REPORT_ROOT / "hidden_branch_persistence_rows.csv"
PARTIAL_PT = REPORT_ROOT / "hidden_branch_persistence.partial.pt"
CAPTURE_LAYERS = (24, 30, 36, 42)
BG_LAYERS = (24, 36, 47)
NUM_LOOPS = 4
MAX_TASKS = 8
K = 4
SAFE_SPECS = [
    {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.005, "safety_envelope": True},
    {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.010, "safety_envelope": True},
    {"branch_point": "L36_L1", "target_layer": 36, "target_loop": 1, "alpha": 0.010, "safety_envelope": True},
]
DIAGNOSTIC_SPECS = [
    {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.050, "safety_envelope": False},
    {"branch_point": "L24_L1", "target_layer": 24, "target_loop": 1, "alpha": 0.100, "safety_envelope": False},
]


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).to(device="cpu", dtype=torch.float32)
    m = attention_mask.squeeze(0).to(device="cpu", dtype=torch.float32).unsqueeze(-1)
    denom = m.sum(dim=0).clamp(min=1.0)
    return (h * m).sum(dim=0) / denom


class HiddenCapture:
    def __init__(self) -> None:
        self.layer_states: dict[int, list[torch.Tensor]] = {layer: [] for layer in CAPTURE_LAYERS}
        self.boundary: list[torch.Tensor] = []

    def make_layer_hook(self, layer: int):
        def _hook(_module: Any, _args: Any, output: Any) -> None:
            self.layer_states[layer].append(_output_tensor(output).detach())

        return _hook

    def boundary_hook(self, _module: Any, _args: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            self.boundary = []
            return
        states = output[1]
        self.boundary = [h.detach() for h in states] if states is not None else []


@contextmanager
def capture_hooks(model: Any, capture: HiddenCapture) -> Iterator[None]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if inner is None or layers is None:
        raise RuntimeError("model does not expose model.layers")
    handles = [inner.register_forward_hook(capture.boundary_hook)]
    for layer in CAPTURE_LAYERS:
        idx = layer - 1
        if idx >= len(layers):
            continue
        handles.append(layers[idx].register_forward_hook(capture.make_layer_hook(layer)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def option_distribution(tokenizer: Any, logits: torch.Tensor, options: dict[str, str]) -> dict[str, float]:
    probs = torch.softmax(logits.detach().to(device="cpu", dtype=torch.float32).flatten(), dim=-1)
    out: dict[str, float] = {}
    for letter in sorted(options):
        ids: set[int] = set()
        for text in (letter, " " + letter, letter + ".", " " + letter + "."):
            enc = tokenizer(text, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(int(enc[0]))
        out[letter] = float(sum(float(probs[i].item()) for i in ids if 0 <= i < probs.numel()))
    return out


def capture_one_branch(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    *,
    delta: torch.Tensor,
    branch_id: int,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    enc = tokenizer(task["prompt"], return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    capture = HiddenCapture()
    hook = HiddenDeltaLayerHook(
        model,
        target_layer=int(spec["target_layer"]),
        target_loops=[int(spec["target_loop"])],
        delta=delta,
        position=-1,
        max_rms_fraction=max(delta_rms(delta), 0.02),
    )
    started = time.time()
    with torch.inference_mode():
        try:
            hook.apply()
            with capture_hooks(model, capture):
                out = model(**enc, use_cache=False, logits_to_keep=1)
        finally:
            hook.remove()
    pooled: dict[str, torch.Tensor] = {}
    last: dict[str, torch.Tensor] = {}
    for layer in CAPTURE_LAYERS:
        states = capture.layer_states.get(layer) or []
        for loop_idx, state in enumerate(states[:NUM_LOOPS], start=1):
            key = f"L{layer}_L{loop_idx}"
            pooled[key] = _masked_mean_pool(state, enc["attention_mask"])
            last[key] = state[0, -1, :].detach().to(device="cpu", dtype=torch.float32)
    for loop_idx, state in enumerate(capture.boundary[:NUM_LOOPS], start=1):
        key = f"L47_L{loop_idx}"
        pooled[key] = _masked_mean_pool(state, enc["attention_mask"])
        last[key] = state[0, -1, :].detach().to(device="cpu", dtype=torch.float32)
    bg_features = []
    for layer in BG_LAYERS:
        loop_vecs = []
        for loop in range(1, NUM_LOOPS + 1):
            key = f"L{layer}_L{loop}"
            if key not in pooled:
                raise RuntimeError(f"missing capture key {key}")
            loop_vecs.append(pooled[key])
        bg_features.append(torch.stack(loop_vecs, dim=0))
    features = torch.stack(bg_features, dim=0).to(dtype=torch.float32)
    logits = getattr(out, "logits", None)
    option_probs = {}
    if isinstance(logits, torch.Tensor):
        option_probs = option_distribution(tokenizer, logits[0, -1], dict(task["options"]))
    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "branch_group_id": "",
        "branch_id": int(branch_id),
        "branch_method": "hook_intervention_per_branch",
        "branch_point": spec["branch_point"],
        "target_layer": int(spec["target_layer"]),
        "target_loop": int(spec["target_loop"]),
        "alpha": float(spec["alpha"]),
        "safety_envelope": bool(spec["safety_envelope"]),
        "delta_type": "clean_zero" if delta_rms(delta) == 0.0 else "random_antipodal_or_orthogonal",
        "delta_norm": float(torch.linalg.vector_norm(delta.float()).item()),
        "effective_delta_rms": delta_rms(delta),
        "delta": delta.detach().to(device="cpu", dtype=torch.float32),
        "features": features,
        "pooled_vectors": pooled,
        "last_token_vectors": last,
        "option_distribution": option_probs,
        "hook_diagnostics": hook.diagnostics(),
        "runtime": round(time.time() - started, 3),
        "VRAM": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "cuda_error": "",
        "nan_inf": not all(torch.isfinite(v).all().item() for v in pooled.values()),
        "branch_lineage": f"{task['task_id']}::{spec['branch_point']}::alpha={spec['alpha']}::branch={branch_id}",
    }


def pairwise_geometry(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    vecs = [row["pooled_vectors"][key] for row in records if key in row["pooled_vectors"]]
    if len(vecs) < 2:
        return {"n_pairs": 0, "mean_cosine": 1.0, "mean_rms_distance": 0.0, "max_rms_distance": 0.0}
    cosines = []
    distances = []
    for a, b in itertools.combinations(vecs, 2):
        cosines.append(cosine(a, b))
        distances.append(rms_distance(a, b))
    return {
        "n_pairs": len(distances),
        "mean_cosine": sum(cosines) / len(cosines),
        "mean_rms_distance": sum(distances) / len(distances),
        "max_rms_distance": max(distances),
    }


def score_group(controller: BGController, records: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    features = [row["features"] for row in records]
    details = controller.rank_candidates(features, domain_hint=domain, mode="conservative", return_details=True)
    mat = details["score_matrix"]
    margin = details["margin_sum"]
    for idx, row in enumerate(records):
        row["tap_margin_sum"] = float(margin[idx].item())
        row["tap_wins"] = int(details["wins"][idx].item())
        row["tap_rank"] = int(details["ranking"].index(idx) + 1)
    return {
        "selected_head": details["selected_head"],
        "ranking": [int(x) for x in details["ranking"]],
        "tap_score_spread": float((margin.max() - margin.min()).item()) if len(records) else 0.0,
        "score_matrix": mat.tolist(),
    }


def summarize_group(records: list[dict[str, Any]], spec: dict[str, Any], task: dict[str, Any], controller: BGController) -> dict[str, Any]:
    group_id = f"{task['task_id']}::{spec['branch_point']}::alpha={spec['alpha']:.3f}::safe={int(spec['safety_envelope'])}"
    for row in records:
        row["branch_group_id"] = group_id
    scoring = score_group(controller, records, task["domain"])
    initial_key = f"L{spec['target_layer']}_L{spec['target_loop']}"
    geom: dict[str, dict[str, float]] = {}
    for key in sorted(records[0]["pooled_vectors"]):
        row = pairwise_geometry(records, key)
        init = pairwise_geometry(records, initial_key)
        row["retention_vs_branch_point"] = row["mean_rms_distance"] / max(init["mean_rms_distance"], 1e-8)
        geom[key] = row
    l47 = geom.get("L47_L4") or geom.get("L47_L1") or {"retention_vs_branch_point": 0.0, "mean_rms_distance": 0.0}
    return {
        "branch_group_id": group_id,
        "task_id": task["task_id"],
        "domain": task["domain"],
        "branch_method": "hook_intervention_per_branch",
        "branch_point": spec["branch_point"],
        "target_layer": int(spec["target_layer"]),
        "target_loop": int(spec["target_loop"]),
        "alpha": float(spec["alpha"]),
        "safety_envelope": bool(spec["safety_envelope"]),
        "branch_count": len(records),
        "initial_capture_key": initial_key,
        "geometry": geom,
        "tap": scoring,
        "collapsed_by_l47": bool(l47["retention_vs_branch_point"] < 0.10 or l47["mean_rms_distance"] < 1e-4),
        "max_option_prob_spread": max(
            (
                max(row["option_distribution"].get(letter, 0.0) for row in records)
                - min(row["option_distribution"].get(letter, 0.0) for row in records)
                for letter in task["options"]
            ),
            default=0.0,
        ),
    }


def persistence_verdict(groups: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    safe = [g for g in groups if g.get("safety_envelope")]
    if not safe:
        return "INSUFFICIENT", {}
    def frac(key: str, threshold: float = 0.10) -> float:
        vals = [finite(g.get("geometry", {}).get(key, {}).get("retention_vs_branch_point")) for g in safe if key in g.get("geometry", {})]
        return sum(v >= threshold for v in vals) / max(len(vals), 1)
    rates = {
        "L30_L1": frac("L30_L1"),
        "L36_L1": frac("L36_L1"),
        "L42_L1": frac("L42_L1"),
        "L47_L4": frac("L47_L4"),
    }
    if rates["L47_L4"] >= 0.30:
        verdict = "LATENT_BRANCHES_PERSIST_TO_47"
    elif rates["L42_L1"] >= 0.30:
        verdict = "LATENT_BRANCHES_PERSIST_TO_42"
    elif rates["L36_L1"] >= 0.30:
        verdict = "LATENT_BRANCHES_PERSIST_TO_36"
    elif rates["L30_L1"] < 0.30:
        verdict = "LATENT_BRANCHES_COLLAPSE_BY_30"
    else:
        verdict = "LATENT_BRANCHES_COLLAPSE_IMMEDIATELY"
    return verdict, rates


def main() -> int:
    ensure_report_root()
    started = time.time()
    feasibility = load_json(REPORT_ROOT / "feasibility.json", {})
    if feasibility.get("BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT") == "BLOCKED":
        payload = {"BG_HIDDEN_BRANCH_GENERATION_VERDICT": "BLOCKED", "blocker": "feasibility blocked", "records": []}
        write_json(OUT_JSON, payload)
        print("BG_HIDDEN_BRANCH_GENERATION_VERDICT = BLOCKED")
        return 1
    tasks = load_task_subset()[:MAX_TASKS]
    if len(tasks) < 1:
        payload = {"BG_HIDDEN_BRANCH_GENERATION_VERDICT": "BLOCKED", "blocker": "no task subset", "records": []}
        write_json(OUT_JSON, payload)
        print("BG_HIDDEN_BRANCH_GENERATION_VERDICT = BLOCKED")
        return 1

    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        controller = BGController.from_artifacts(device="cpu")
        device = extractor.device
        model = extractor.model
        tokenizer = extractor.tokenizer
        specs_by_task: dict[str, list[dict[str, Any]]] = {}
        for idx, task in enumerate(tasks):
            specs = list(SAFE_SPECS)
            if idx < 2:
                specs.extend(DIAGNOSTIC_SPECS)
            specs_by_task[task["task_id"]] = specs
        for task_idx, task in enumerate(tasks):
            for spec in specs_by_task[task["task_id"]]:
                seed = 20260518 + task_idx * 101 + int(spec["target_layer"]) * 7 + int(float(spec["alpha"]) * 1000)
                deltas = make_branch_deltas(
                    HIDDEN_DIM,
                    K,
                    float(spec["alpha"]),
                    seed,
                    allow_diagnostic_alpha=not bool(spec["safety_envelope"]),
                )
                group_records: list[dict[str, Any]] = []
                for branch_id, delta in enumerate(deltas):
                    try:
                        row = capture_one_branch(model, tokenizer, task, delta=delta, branch_id=branch_id, spec=spec, device=device)
                        records.append(row)
                        group_records.append(row)
                    except Exception as exc:
                        errors.append(
                            {
                                "task_id": task["task_id"],
                                "branch_point": spec["branch_point"],
                                "alpha": spec["alpha"],
                                "branch_id": branch_id,
                                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                            }
                        )
                if len(group_records) >= 2:
                    groups.append(summarize_group(group_records, spec, task, controller))
                torch.save({"complete": False, "records": records, "groups": groups, "errors": errors}, PARTIAL_PT)
    finally:
        if extractor is not None:
            extractor.cleanup()

    generation_verdict = "HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED" if records else "BLOCKED"
    persistence, rates = persistence_verdict(groups)
    safe_groups = [g for g in groups if g.get("safety_envelope")]
    diag_groups = [g for g in groups if not g.get("safety_envelope")]
    safe_l47 = [finite(g.get("geometry", {}).get("L47_L4", {}).get("retention_vs_branch_point")) for g in safe_groups]
    diag_l47 = [finite(g.get("geometry", {}).get("L47_L4", {}).get("retention_vs_branch_point")) for g in diag_groups]
    high_alpha_note = {
        "safe_mean_l47_retention": sum(safe_l47) / max(len(safe_l47), 1),
        "diagnostic_mean_l47_retention": sum(diag_l47) / max(len(diag_l47), 1),
        "interpretation": (
            "larger diagnostic perturbations persist more, so safe-alpha collapse is plausibly generation-strength-limited"
            if diag_l47 and (sum(diag_l47) / max(len(diag_l47), 1)) > (sum(safe_l47) / max(len(safe_l47), 1)) * 1.5 + 0.05
            else "diagnostic arm did not clearly separate generation-strength from convergence pressure"
        ),
        "diagnostic_scope": "alpha_0.05_0.1_outside_safety_envelope_not_headline",
    }
    json_groups = []
    csv_rows = []
    for group in groups:
        compact = {k: v for k, v in group.items() if k not in {"geometry", "tap"}}
        compact["tap_score_spread"] = group["tap"]["tap_score_spread"]
        compact["selected_head"] = group["tap"]["selected_head"]
        for key, geom in group["geometry"].items():
            compact[f"{key}_retention"] = geom["retention_vs_branch_point"]
            compact[f"{key}_mean_rms_distance"] = geom["mean_rms_distance"]
            compact[f"{key}_mean_cosine"] = geom["mean_cosine"]
        json_groups.append({**group, "tap": {k: v for k, v in group["tap"].items() if k != "score_matrix"}})
        csv_rows.append(compact)
    branch_csv_rows = [
        {
            "branch_group_id": row["branch_group_id"],
            "task_id": row["task_id"],
            "domain": row["domain"],
            "branch_id": row["branch_id"],
            "branch_point": row["branch_point"],
            "alpha": row["alpha"],
            "safety_envelope": row["safety_envelope"],
            "effective_delta_rms": row["effective_delta_rms"],
            "tap_margin_sum": row.get("tap_margin_sum", 0.0),
            "tap_rank": row.get("tap_rank", 0),
            "runtime": row["runtime"],
            "nan_inf": row["nan_inf"],
        }
        for row in records
    ]
    payload = {
        "BG_HIDDEN_BRANCH_GENERATION_VERDICT": generation_verdict,
        "BG_LATENT_BRANCH_PERSISTENCE_VERDICT": persistence,
        "LIVE_BRANCH_METHOD": "hook_intervention_per_branch",
        "branch_count": len(records),
        "branch_group_count": len(groups),
        "task_count": len({r["task_id"] for r in records}),
        "safe_alpha_group_count": len(safe_groups),
        "diagnostic_high_alpha_group_count": len(diag_groups),
        "capture_layers": list(CAPTURE_LAYERS) + [47],
        "persistence_rates": rates,
        "high_alpha_diagnostic": high_alpha_note,
        "counts_by_domain": dict(Counter(r["domain"] for r in records)),
        "groups": json_groups,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save({**payload, "records": records, "groups": groups}, OUT_PT)
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, csv_rows + branch_csv_rows)
    lines = [
        "# BG Hidden-Origin Branch Persistence",
        "",
        f"BG_HIDDEN_BRANCH_GENERATION_VERDICT = {generation_verdict}",
        f"BG_LATENT_BRANCH_PERSISTENCE_VERDICT = {persistence}",
        f"LIVE_BRANCH_METHOD = hook_intervention_per_branch",
        "",
        f"- branch_count: `{len(records)}`",
        f"- branch_group_count: `{len(groups)}`",
        f"- task_count: `{payload['task_count']}`",
        f"- safe_alpha_group_count: `{len(safe_groups)}`",
        f"- diagnostic_high_alpha_group_count: `{len(diag_groups)}`",
        f"- high_alpha_diagnostic: `{high_alpha_note['interpretation']}`",
        "",
        "Persistence verdict is keyed to geometric hidden-feature RMS/cosine, not tap-score spread. Tap spread is secondary because signed +/- deltas can be tap-indifferent under the prior unsigned-effect finding.",
        "",
        "## Group Summary",
        "",
    ]
    lines.extend(md_table(csv_rows[:20], ["branch_group_id", "domain", "alpha", "safety_envelope", "L30_L1_retention", "L36_L1_retention", "L42_L1_retention", "L47_L4_retention", "tap_score_spread"]))
    if errors:
        lines.extend(["", "## Errors", "", *[f"- `{e['task_id']}` {e['branch_point']} alpha {e['alpha']}: {e['error'][:200]}" for e in errors[:10]]])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_BRANCH_GENERATION_VERDICT = {generation_verdict}")
    print(f"BG_LATENT_BRANCH_PERSISTENCE_VERDICT = {persistence}")
    print(f"Wrote {rel(OUT_PT)}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
