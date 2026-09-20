"""PART K — canary checkpoint evaluation of trained arms (resumable, STOP-pausable).

One arm per invocation via ARM=<arm_name>; ARM=selection aggregates all evaluated arms and
selects <=3 adapters for Part L. Cost knob: K_PER_DOMAIN (default 30) picks a deterministic,
seed-fixed subset of generation canary tasks per domain — the SAME subset for every arm and
for the base/prev_sft reference slices, so all deltas are paired. Alignment (logprob, cheap)
keeps the full 110. Reuses the Part D generation/scoring machinery unchanged.
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v2_common as V  # noqa: E402
import evaluate_offline_branch_generator_canary_baselines_v2 as D  # noqa: E402

ARM = os.environ.get("ARM", "selection")
K_PER_DOMAIN = int(os.environ.get("K_PER_DOMAIN", "30"))
SUBSET_SEED = int(os.environ.get("SUBSET_SEED", "1234"))
MAX_SELECT = 3
RESULT_DIR = V.OUT_ROOT / "canary_gen"


def subset_ids() -> set[str]:
    """Deterministic per-domain generation-task subset; full alignment set."""
    rng = random.Random(SUBSET_SEED)
    by_domain: dict[str, list[str]] = {}
    for c in D._canary():
        by_domain.setdefault(c["domain"], []).append(c["canary_id"])
    ids: set[str] = set()
    for domain, cids in sorted(by_domain.items()):
        cids = sorted(cids)
        if domain == "alignment" or len(cids) <= K_PER_DOMAIN:
            ids.update(cids)
        else:
            ids.update(rng.sample(cids, K_PER_DOMAIN))
    return ids


def _adapter_dir(arm: str) -> Path:
    d = V.MODEL_ROOT / arm
    if not (d / "adapter_model.safetensors").exists() and not (d / "adapter_model.bin").exists():
        raise RuntimeError(f"arm {arm} has no final adapter under {d}; train it to completion first")
    return d


def _load_arm(arm: str):
    model, tok = D._load("base")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(_adapter_dir(arm))).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def evaluate_arm(arm: str) -> None:
    ids = subset_ids()
    out_path = RESULT_DIR / f"arm_{arm}.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                done.add(json.loads(l)["canary_id"])
            except Exception:
                pass
    canary = [c for c in D._canary() if c["canary_id"] in ids and c["canary_id"] not in done]
    tasks = [c for c in canary if c["domain"] != "alignment"]
    align = [c for c in canary if c["domain"] == "alignment"]
    job = f"canary_arm_{arm}"
    print(f"[{arm}] pending gen {len(tasks)} + alignment {len(align)} (done {len(done)}, subset {len(ids)})", flush=True)
    if not tasks and not align:
        _finalize_arm(arm, out_path)
        return
    model, tok = _load_arm(arm)
    f = open(out_path, "a")
    t0, n, paused = time.time(), 0, False
    for task in tasks:
        if V.stop_requested(job):
            print(f"[{arm}] STOP requested at {n}", flush=True)
            paused = True
            break
        texts = D._gen_branches(model, tok, task, task.get("K", 4))
        rec = D._score_pool(task, texts)
        f.write(json.dumps(rec, default=V.v2.json_default) + "\n")
        f.flush()
        n += 1
        if n % 10 == 0:
            rate = (time.time() - t0) / n
            print(f"[{arm}] {n}/{len(tasks)} gen | {rate:.0f}s/task | ETA {rate*(len(tasks)-n)/3600:.1f}h", flush=True)
    if not paused:
        for i, task in enumerate(align):
            if V.stop_requested(job):
                paused = True
                break
            lc = D._pref_logprob(model, tok, task.get("task_prompt", ""), task["pref"]["chosen"])
            lr = D._pref_logprob(model, tok, task.get("task_prompt", ""), task["pref"]["rejected"])
            f.write(json.dumps({"canary_id": task["canary_id"], "domain": "alignment",
                                "slice": ["alignment_pref"], "pref_correct": bool(lc > lr),
                                "margin": round(lc - lr, 4)}, default=V.v2.json_default) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{arm}] alignment {i+1}/{len(align)}", flush=True)
    f.close()
    if paused:
        V.set_stage(f"K_canary_{arm}", "CANARY_ARM_PAUSED", {"done": len(done) + n})
        print(V.status_line(f"K_CANARY_{arm.upper()}", "CANARY_ARM_PAUSED"))
        return
    _finalize_arm(arm, out_path)


def _agg_rows(path: Path, ids: set[str]) -> dict:
    """Per-domain aggregate over rows restricted to the paired subset."""
    import collections
    byd = collections.defaultdict(list)
    if not path.exists():
        return {}
    for l in open(path):
        r = json.loads(l)
        if r["canary_id"] in ids:
            byd[r["domain"]].append(r)
    out = {}
    for d, rs in byd.items():
        n = len(rs)
        if d == "alignment":
            k = sum(1 for r in rs if r.get("pref_correct"))
            out[d] = {"n": n, "metric": round(k / n, 4) if n else None, "ci": V.wilson_ci(k, n)}
        else:
            po = sum(1 for r in rs if r["pos_oracle"])
            chars = [len(b.get("text") or "") for r in rs for b in (r.get("branches") or [])]
            out[d] = {"n": n, "metric": round(po / n, 4) if n else None, "ci": V.wilson_ci(po, n),
                      "parse_ok": round(sum(r["parse_ok"] for r in rs) / n, 4) if n else None,
                      "branch_diversity": round(sum(r["distinct_finals"] for r in rs) / n, 3) if n else None,
                      "all_wrong_rate": round(sum(1 for r in rs if r["all_wrong"]) / n, 4) if n else None,
                      "mean_branch_chars": round(sum(chars) / len(chars), 1) if chars else None}
    return out


def _finalize_arm(arm: str, out_path: Path) -> None:
    agg = _agg_rows(out_path, subset_ids())
    V.set_stage(f"K_canary_{arm}", "CANARY_ARM_COMPLETE", {"domains": {k: v.get("metric") for k, v in agg.items()}})
    V.prog(f"K_canary_{arm}", {"verdict": "CANARY_ARM_COMPLETE", "agg": agg})
    print(V.status_line(f"K_CANARY_{arm.upper()}", "CANARY_ARM_COMPLETE"))
    for d, v in sorted(agg.items()):
        print(f"  {d:10} n={v['n']} metric={v['metric']}")


def selection() -> None:
    ids = subset_ids()
    arms_on = (V.read_state().get("stages", {}).get("I_training_matrix", {}) or {}).get("arms_on", [])
    extra = sorted(p.stem for p in (V.MODEL_ROOT / "configs").glob("*_r[0-9].json")
                   if (RESULT_DIR / f"arm_{p.stem}.jsonl").exists())
    arms_on = list(dict.fromkeys(list(arms_on) + extra))
    base = _agg_rows(RESULT_DIR / "base.jsonl", ids)
    prev = _agg_rows(RESULT_DIR / "prev_sft.jsonl", ids)
    doms = list(V.CORE_DOMAINS)
    rows, ranking = [], []
    for arm in arms_on:
        agg = _agg_rows(RESULT_DIR / f"arm_{arm}.jsonl", ids)
        complete = all((agg.get(d, {}).get("n") or 0) > 0 for d in doms)
        deltas = {}
        for d in doms:
            b, a = (base.get(d) or {}).get("metric"), (agg.get(d) or {}).get("metric")
            deltas[d] = round(a - b, 4) if (a is not None and b is not None) else None
        valid = [v for v in deltas.values() if v is not None]
        macro = round(sum(valid) / len(valid), 4) if valid else None
        def _mc(a):
            xs = [v.get("mean_branch_chars") for v in a.values() if v.get("mean_branch_chars")]
            return round(sum(xs) / len(xs), 1) if xs else None
        amc, bmc = _mc(agg), _mc(base)
        chars_ratio = round(amc / bmc, 3) if (amc and bmc) else None
        rows.append({"arm": arm, "complete": complete, "macro_delta": macro,
                     "chars_vs_base": chars_ratio,
                     "degeneration_alarm": bool(chars_ratio is not None and chars_ratio < 0.5),
                     **{f"d_{d}": deltas[d] for d in doms}})
        if complete and macro is not None:
            ranking.append((macro, arm))
    ranking.sort(reverse=True)
    selected = [arm for _, arm in ranking[:MAX_SELECT]]
    improving = [arm for m, arm in ranking if m > 0]
    if not ranking:
        verdict = "K_SELECTION_BLOCKED_NO_COMPLETE_ARMS"
    elif not improving:
        verdict = "NO_ADAPTER_IMPROVES_ON_CANARY"
    else:
        verdict = "ADAPTERS_SELECTED_FOR_HELDOUT"
    payload = {"K_CANARY_SELECTION_VERDICT": verdict, "subset_seed": SUBSET_SEED,
               "k_per_domain": K_PER_DOMAIN, "subset_size": len(ids),
               "selected_for_heldout": selected, "improving_arms": improving,
               "base": base, "prev_sft": prev, "rows": rows}
    V.write_json(V.OUT_ROOT / "canary_checkpoint_selection.json", payload)
    V.write_csv(V.OUT_ROOT / "canary_checkpoint_rows.csv", rows)
    V.write_md(V.OUT_ROOT / "canary_checkpoint_selection.md", [
        "# Canary Checkpoint Selection (Part K)", "",
        V.status_line("K_CANARY_SELECTION_VERDICT", verdict),
        f"Paired subset: seed {SUBSET_SEED}, {K_PER_DOMAIN}/domain generation tasks + full alignment "
        f"({len(ids)} total); identical tasks for base, prev_sft, and every arm. Deltas are arm - base.",
        "", *V.md_table(rows, ["arm", "complete", "macro_delta"] + [f"d_{d}" for d in doms]),
        "", f"Selected for Part L heldout (<= {MAX_SELECT}): {', '.join(selected) if selected else 'none'}.",
        "", "Resumable per-arm; STOP-pausable; selection is idempotent and re-runnable as arms land.",
    ])
    V.set_stage("K_canary_selection", verdict, {"selected": selected})
    V.prog("K_canary_selection", {"verdict": verdict, "rows": rows, "selected": selected})
    print(V.status_line("K_CANARY_SELECTION_VERDICT", verdict))
    for r in rows:
        print(f"  {r['arm']:24} complete={r['complete']} macro_delta={r['macro_delta']}")
    print(f"  selected: {selected}")


def main() -> int:
    V.ensure_dirs()
    if ARM == "selection":
        selection()
    else:
        evaluate_arm(ARM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
