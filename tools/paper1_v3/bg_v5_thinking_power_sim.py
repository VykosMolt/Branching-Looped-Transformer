"""V5 Thinking pre-answer power extension -- sizing simulation (CPU-only).

Sizes a powered, sealed replication of the Thinking-checkpoint strict
pre-answer experiment. Pilot: artifacts/reports/thinking_preanswer_v3_20260726
(170 tasks = offsets [0,170) of the hash-ordered Horizon pool; 680 candidates;
malformed 0.456; 370 scorable; 130 held-out scorable with 20 negatives;
within-Thinking 5-shortcut adversarial increment +0.0273 [-0.1285, +0.2106]).

METHOD -- plug-in task-level resampling from preserved pilot artifacts:
  * The pilot's 59 held-out tasks (56 with >=1 scorable candidate; 3 fully
    malformed) are the plug-in task population. Each task carries its scorable
    candidates' preserved held-out scores (combined / shortcut from the
    adversarial arm_5sc of raw_predictions_heldout.json). The model fit is
    held FIXED at the pilot fit: projected CI width is driven by held-out
    cohort size, which is the quantity the sizing decision needs.
  * A future run with N total tasks draws Binomial(N, 59/170) held-out tasks
    (59/170 = the pilot's empirical deterministic-split fraction), each
    sampled iid with replacement (uniform, or tilted for the sensitivity row)
    from the pilot held-out tasks and relabelled as a fresh task cluster.
    Fully-malformed tasks contribute zero scorable candidates, so scorable
    attrition is carried through exactly at pilot rates.
  * On each simulated cohort the estimator is EXACTLY the sealed analysis
    estimator: incremental AUROC (combined - shortcut), with the 2000-round
    paired task-clustered bootstrap of bg_v2_overnight_horizon_analysis
    .bootstrap_task_clustered at seed 20260727 (the seed the sealed v5
    analysis uses). A vectorized weighted-Mann-Whitney implementation is used
    for speed; at startup it is verified BIT-IDENTICAL (same RNG consumption,
    same skip rules, same quantile convention) to the reference loop on the
    pilot data, and both must reproduce the published pilot bootstrap numbers,
    before any simulation is run.

EFFECT INJECTION (power and TOST cells) -- documented shift procedure:
  The assumed true increment is injected by a calibrated additive shift of
  the combined scores of POSITIVE-class candidates only; shortcut scores and
  negative-class combined scores are untouched:
      s_comb' = s_comb + delta * 1[y == 1]
  delta is calibrated by bisection (monotone in delta, since AUROC is
  rank-based) so that the POOLED pilot held-out population increment
  auroc(comb') - auroc(sc) equals the target: +0.095 for the power cells
  (the RLTT v3 new-only point estimate), 0 for the TOST cells. The achieved
  population increment (within one AUROC step of the target) is reported.

SENSITIVITY ROW: task sampling reweighted by an exponential tilt on the
  per-task (malformed count, negative count), solved by nested grid search,
  to match the C2-d4 rates: malformed 0.378, negatives 0.188 of scorable.

Outputs a compact table per (scenario, N in {600, 900, 1200}): expected
held-out cohort, median/p10/p90 95% CI half-width, power P(lower95 > 0) under
true +0.095, and P(TOST passes at +/-0.05) under true 0. Master seed
20260727, >=500 reps per cell (--reps). Writes sizing_simulation_v5.json into
the v5 report directory. CPU-only; touches no GPU and no model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from proto_introspection_controls_analysis import auroc  # noqa: E402
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered  # noqa: E402

PILOT_ROOT = PROJECT_ROOT / "artifacts/reports/thinking_preanswer_v3_20260726"
PILOT_INDEX = PILOT_ROOT / "horizon_logic/horizon_generation_think_off0_index.json"
PILOT_RAW = PILOT_ROOT / "raw_predictions_heldout.json"
PILOT_RESULTS = PILOT_ROOT / "thinking_preanswer_results.json"
OUT_ROOT = PROJECT_ROOT / "artifacts/reports/thinking_preanswer_power_v5_20260727"

SIM_SEED = 20260727       # master seed for cohort draws
BOOT_SEED = 20260727      # bootstrap seed (must match the sealed v5 analysis)
BOOT_ROUNDS = 2000
N_GRID = (600, 900, 1200)
TRUE_EFFECT_ALT = 0.095   # RLTT v3 new-only point estimate
TOST_MARGIN = 0.05
C2D4_MALFORMED = 0.378    # sensitivity targets (C2-d4 rates)
C2D4_NEG_OF_SCORABLE = 0.188


def _seed(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**31)


# --------------------------------------------------------------------------
# pilot loading
# --------------------------------------------------------------------------

def load_pilot():
    idx = json.loads(PILOT_INDEX.read_text())
    raw = json.loads(PILOT_RAW.read_text())["arm_5sc"]
    by_uid: dict[str, list] = {}
    for t, yy, sc, sco in zip(raw["tasks"], raw["y"],
                              raw["scores"]["shortcut"], raw["scores"]["combined"]):
        by_uid.setdefault(t, []).append((float(yy), float(sco), float(sc)))
    tasks: dict[str, dict] = {}
    for r in idx["records_summary"]:
        t = tasks.setdefault(r["task_uid"], {"split": r["split"], "m": 0, "c": 0, "g": 0, "s": 0})
        if r["malformed"]:
            t["m"] += 1
        else:
            t["c"] += 1
            t["s" if r["success"] else "g"] += 1
    n_total = len(tasks)
    ho = []
    for uid, t in sorted(tasks.items()):
        if t["split"] != "heldout":
            continue
        cands = by_uid.get(uid, [])
        assert len(cands) == t["c"], f"{uid}: preds/index scorable mismatch"
        if cands:
            assert sum(y for y, _, _ in cands) == t["s"], f"{uid}: success count mismatch"
        ho.append({"uid": uid, "m": t["m"], "c": t["c"], "g": t["g"], "cands": cands})
    assert sum(1 for t in ho if t["c"] > 0) == len(by_uid), "held-out preds task set mismatch"
    return ho, n_total, raw


# --------------------------------------------------------------------------
# exact-equivalent vectorized bootstrap (verified against the reference)
# --------------------------------------------------------------------------

def fast_bootstrap_dual(sa, sb, y, task_keys, rounds=BOOT_ROUNDS, seed=BOOT_SEED):
    """Bit-identical to bg_v2_overnight_horizon_analysis.bootstrap_task_clustered
    (same per-round torch.randint RNG consumption, same skip rules, same sorted
    empirical-quantile convention), computed via a weighted Mann-Whitney
    identity: cluster resampling == integer per-candidate weights. Adds a 90%
    interval from the same sorted deltas (for TOST)."""
    uniq = sorted(set(task_keys))
    U = len(uniq)
    pos_of = {t: i for i, t in enumerate(uniq)}
    tidx = torch.tensor([pos_of[t] for t in task_keys])
    rng = torch.Generator().manual_seed(seed)
    picks = torch.empty((rounds, U), dtype=torch.long)
    for r in range(rounds):
        picks[r] = torch.randint(0, U, (U,), generator=rng)
    counts = torch.zeros((rounds, U), dtype=torch.float64)
    counts.scatter_add_(1, picks, torch.ones((rounds, U), dtype=torch.float64))
    w = counts[:, tidx]                                    # (R, n) integer weights
    posm = y > 0.5
    negm = ~posm
    if int(posm.sum()) == 0 or int(negm.sum()) == 0:
        return None
    wp, wn = w[:, posm], w[:, negm]
    out = {}
    wins = {}
    for name, s in (("a", sa), ("b", sb)):
        spos, sneg = s[posm].double(), s[negm].double()
        A = (spos.unsqueeze(1) > sneg.unsqueeze(0)).double() \
            + 0.5 * (spos.unsqueeze(1) == sneg.unsqueeze(0)).double()
        wins[name] = ((wp @ A) * wn).sum(1)
    den = wp.sum(1) * wn.sum(1)
    valid = den > 0
    aa = wins["a"][valid] / den[valid]
    bb = wins["b"][valid] / den[valid]
    deltas = (aa - bb).tolist()
    if not deltas:
        return None
    deltas.sort()
    n = len(deltas)
    out = {
        "n_rounds_used": n,
        "mean_delta": sum(deltas) / n,
        "ci95": [deltas[int(0.025 * n)], deltas[int(0.975 * n)]],
        "ci90": [deltas[int(0.05 * n)], deltas[int(0.95 * n)]],
        "excludes_zero": bool(deltas[int(0.025 * n)] > 0 or deltas[int(0.975 * n)] < 0),
    }
    return out


def gate_fast_bootstrap(raw, published):
    """ABORT unless fast == reference == published pilot JSON on the pilot data."""
    sa = torch.tensor(raw["scores"]["combined"], dtype=torch.float64)
    sb = torch.tensor(raw["scores"]["shortcut"], dtype=torch.float64)
    y = torch.tensor(raw["y"], dtype=torch.float32)
    uids = raw["tasks"]
    pilot_seed = published["seed"]                          # 20260724, as published
    ref = bootstrap_task_clustered(sa, sb, y, uids, rounds=BOOT_ROUNDS, seed=pilot_seed)
    fast = fast_bootstrap_dual(sa, sb, y, uids, rounds=BOOT_ROUNDS, seed=pilot_seed)
    pub = published["arm_5sc"]["paired_task_clustered_bootstrap_combined_minus_shortcut"]
    checks = [
        ("fast_vs_ref_mean", fast["mean_delta"], ref["mean_delta"]),
        ("fast_vs_ref_lo", fast["ci95"][0], ref["ci95"][0]),
        ("fast_vs_ref_hi", fast["ci95"][1], ref["ci95"][1]),
        ("fast_vs_published_mean", fast["mean_delta"], pub["mean_delta"]),
        ("fast_vs_published_lo", fast["ci95"][0], pub["ci95"][0]),
        ("fast_vs_published_hi", fast["ci95"][1], pub["ci95"][1]),
    ]
    for label, got, want in checks:
        d = abs(got - want)
        flag = "OK " if d <= 1e-12 else "FAIL"
        print(f"[sim-gate] [{flag}] {label}: {got!r} vs {want!r} (|d|={d:.3e})", flush=True)
        if d > 1e-12:
            raise SystemExit(f"SIM GATE FAILED: {label} mismatch -- refusing to simulate")
    assert fast["n_rounds_used"] == ref["n_rounds_used"]
    print(f"[sim-gate] PASSED: fast bootstrap bit-identical to reference and to the "
          f"published pilot interval ({fast['n_rounds_used']} rounds)", flush=True)


# --------------------------------------------------------------------------
# effect injection + sensitivity tilt
# --------------------------------------------------------------------------

def pooled_arrays(ho_tasks, shift=0.0):
    ys, sc, sco = [], [], []
    for t in ho_tasks:
        for yy, scomb, ssc in t["cands"]:
            ys.append(yy)
            sco.append(scomb + (shift if yy == 1.0 else 0.0))
            sc.append(ssc)
    return (torch.tensor(sco, dtype=torch.float64), torch.tensor(sc, dtype=torch.float64),
            torch.tensor(ys, dtype=torch.float32))


def calibrate_shift(ho_tasks, target, lo=-1.5, hi=1.5, iters=60):
    """Bisection for delta s.t. pooled pilot held-out increment == target."""
    _, s_sc, y = pooled_arrays(ho_tasks)
    a_sc = auroc(s_sc, y)

    def inc(delta):
        s_comb, _, _ = pooled_arrays(ho_tasks, shift=delta)
        return auroc(s_comb, y) - a_sc

    assert inc(lo) <= target <= inc(hi), "target outside achievable increment range"
    for _ in range(iters):
        mid = (lo + hi) / 2
        if inc(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi, inc(hi)


def tilt_weights(ho_tasks, target_mal, target_neg):
    """Exponential tilt w_j ~ exp(a*m_j + b*g_j) matching the C2-d4 rates:
    E_w[malformed]/4 = target_mal and E_w[neg]/E_w[scorable] = target_neg."""
    mj = torch.tensor([t["m"] for t in ho_tasks], dtype=torch.float64)
    gj = torch.tensor([t["g"] for t in ho_tasks], dtype=torch.float64)
    cj = torch.tensor([t["c"] for t in ho_tasks], dtype=torch.float64)

    def rates(a, b):
        w = torch.exp(a * mj + b * gj)
        w = w / w.sum()
        r_mal = float((w * mj).sum() / 4.0)
        den = float((w * cj).sum())
        r_neg = float((w * gj).sum()) / den if den > 1e-12 else float("nan")
        return w, r_mal, r_neg

    lo_a, hi_a, lo_b, hi_b = -4.0, 4.0, -4.0, 4.0
    best = None
    for _ in range(4):
        for a in torch.linspace(lo_a, hi_a, 41).tolist():
            for b in torch.linspace(lo_b, hi_b, 41).tolist():
                _, r_mal, r_neg = rates(a, b)
                if math.isnan(r_neg):
                    continue
                loss = (r_mal - target_mal) ** 2 + (r_neg - target_neg) ** 2
                if best is None or loss < best[0]:
                    best = (loss, a, b)
        _, a0, b0 = best
        wa, wb = (hi_a - lo_a) / 8, (hi_b - lo_b) / 8
        lo_a, hi_a, lo_b, hi_b = a0 - wa, a0 + wa, b0 - wb, b0 + wb
    _, a0, b0 = best
    w, r_mal, r_neg = rates(a0, b0)
    print(f"[sim] sensitivity tilt: a={a0:+.4f} b={b0:+.4f} -> malformed={r_mal:.4f} "
          f"(target {target_mal}) neg_of_scorable={r_neg:.4f} (target {target_neg})", flush=True)
    return w, {"a": a0, "b": b0, "achieved_malformed": r_mal, "achieved_neg_of_scorable": r_neg}


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def run_cell(tag, ho_tasks, weights, p_ho, n_total, shift, reps, rounds):
    t0 = time.time()
    hws, points = [], []
    power_hits = tost_hits = degenerate = 0
    for rep in range(reps):
        gen = torch.Generator().manual_seed(_seed(SIM_SEED, tag, n_total, rep))
        m = int((torch.rand(n_total, generator=gen) < p_ho).sum())
        sel = torch.multinomial(weights, m, replacement=True, generator=gen).tolist()
        ys, sc, sco, tk = [], [], [], []
        for new_id, j in enumerate(sel):
            for yy, scomb, ssc in ho_tasks[j]["cands"]:
                ys.append(yy)
                sco.append(scomb + (shift if yy == 1.0 else 0.0))
                sc.append(ssc)
                tk.append(new_id)
        y = torch.tensor(ys, dtype=torch.float32)
        if len(ys) == 0 or float(y.mean()) in (0.0, 1.0):
            degenerate += 1
            continue
        s_comb = torch.tensor(sco, dtype=torch.float64)
        s_sc = torch.tensor(sc, dtype=torch.float64)
        boot = fast_bootstrap_dual(s_comb, s_sc, y, tk, rounds=rounds, seed=BOOT_SEED)
        if boot is None:
            degenerate += 1
            continue
        hws.append((boot["ci95"][1] - boot["ci95"][0]) / 2)
        points.append(auroc(s_comb, y) - auroc(s_sc, y))
        if boot["ci95"][0] > 0:
            power_hits += 1
        if boot["ci90"][0] >= -TOST_MARGIN and boot["ci90"][1] <= TOST_MARGIN:
            tost_hits += 1
    hws_t = torch.tensor(hws, dtype=torch.float64)
    q = lambda p: float(torch.quantile(hws_t, p)) if len(hws) else float("nan")  # noqa: E731
    out = {
        "tag": tag, "n_total_tasks": n_total, "reps": reps, "degenerate": degenerate,
        "median_halfwidth95": q(0.5), "p10_halfwidth95": q(0.10), "p90_halfwidth95": q(0.90),
        "mean_point_increment": (sum(points) / len(points)) if points else float("nan"),
        "power_lower95_gt0": power_hits / reps,
        "tost_pass_rate": tost_hits / reps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[sim] cell {tag} N={n_total}: med_hw={out['median_halfwidth95']:.4f} "
          f"power={out['power_lower95_gt0']:.3f} tost={out['tost_pass_rate']:.3f} "
          f"mean_point={out['mean_point_increment']:+.4f} deg={degenerate} "
          f"({out['elapsed_s']}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--boot-rounds", type=int, default=BOOT_ROUNDS)
    args = ap.parse_args()

    ho_tasks, n_total_pilot, raw = load_pilot()
    published = json.loads(PILOT_RESULTS.read_text())
    n_ho = len(ho_tasks)
    p_ho = n_ho / n_total_pilot
    n_scorable_ho = sum(t["c"] for t in ho_tasks)
    n_neg_ho = sum(t["g"] for t in ho_tasks)
    print(f"[sim] pilot: {n_total_pilot} tasks, {n_ho} held-out ({p_ho:.4f}), "
          f"{n_scorable_ho} scorable held-out candidates, {n_neg_ho} negatives, "
          f"{sum(1 for t in ho_tasks if t['c'] == 0)} fully-malformed held-out tasks", flush=True)

    gate_fast_bootstrap(raw, published)

    w_base = torch.ones(n_ho, dtype=torch.float64) / n_ho
    w_tilt, tilt_info = tilt_weights(ho_tasks, C2D4_MALFORMED, C2D4_NEG_OF_SCORABLE)

    d_alt, inc_alt = calibrate_shift(ho_tasks, TRUE_EFFECT_ALT)
    d_null, inc_null = calibrate_shift(ho_tasks, 0.0)
    _, s_sc0, y0 = pooled_arrays(ho_tasks)
    s_comb0, _, _ = pooled_arrays(ho_tasks, shift=0.0)
    inc_plugin = auroc(s_comb0, y0) - auroc(s_sc0, y0)
    print(f"[sim] plug-in population increment (unshifted): {inc_plugin:+.5f}", flush=True)
    print(f"[sim] shift calibration: alt delta={d_alt:+.6f} -> achieved {inc_alt:+.5f} "
          f"(target {TRUE_EFFECT_ALT:+.3f}); null delta={d_null:+.6f} -> achieved "
          f"{inc_null:+.5f} (target 0)", flush=True)

    def exp_counts(w):
        mean_c = float((w * torch.tensor([t["c"] for t in ho_tasks], dtype=torch.float64)).sum()
                       / w.sum())
        mean_g = float((w * torch.tensor([t["g"] for t in ho_tasks], dtype=torch.float64)).sum()
                       / w.sum())
        return mean_c, mean_g

    cells = {}
    for scen, w in (("pilot_rates", w_base), ("c2d4_rates", w_tilt)):
        for n in N_GRID:
            cells[(scen, n, "hw")] = run_cell(f"{scen}-hw", ho_tasks, w, p_ho, n, 0.0,
                                              args.reps, args.boot_rounds)
    for n in N_GRID:
        cells[("pilot_rates", n, "power")] = run_cell(
            "pilot-power", ho_tasks, w_base, p_ho, n, d_alt, args.reps, args.boot_rounds)
        cells[("pilot_rates", n, "tost")] = run_cell(
            "pilot-tost", ho_tasks, w_base, p_ho, n, d_null, args.reps, args.boot_rounds)

    # ---------------- compact table ----------------
    hdr = (f"{'scenario':<12} {'N':>5} {'E[hoT]':>7} {'E[hoC]':>7} {'E[neg]':>7} "
           f"{'med_hw95':>9} {'p10':>7} {'p90':>7} {'pow(+.095)':>11} {'TOST(0)':>8}")
    lines = [hdr, "-" * len(hdr)]
    for scen, w in (("pilot_rates", w_base), ("c2d4_rates", w_tilt)):
        mean_c, mean_g = exp_counts(w)
        for n in N_GRID:
            hw = cells[(scen, n, "hw")]
            pw = cells.get(("pilot_rates", n, "power")) if scen == "pilot_rates" else None
            ts = cells.get(("pilot_rates", n, "tost")) if scen == "pilot_rates" else None
            pw_s = "{:.3f}".format(pw["power_lower95_gt0"]) if pw else "--"
            ts_s = "{:.3f}".format(ts["tost_pass_rate"]) if ts else "--"
            lines.append(
                f"{scen:<12} {n:>5} {n * p_ho:>7.1f} {n * p_ho * mean_c:>7.1f} "
                f"{n * p_ho * mean_g:>7.1f} {hw['median_halfwidth95']:>9.4f} "
                f"{hw['p10_halfwidth95']:>7.4f} {hw['p90_halfwidth95']:>7.4f} "
                f"{pw_s:>11} {ts_s:>8}")
    table = "\n".join(lines)
    print("\n[sim] ================ SIZING TABLE ================", flush=True)
    print(table, flush=True)
    print(f"[sim] wall-clock at 39 s/task: " + "  ".join(
        f"N={n}: {n * 39 / 3600:.1f}h" for n in N_GRID), flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "sim_seed": SIM_SEED, "bootstrap_seed": BOOT_SEED,
        "boot_rounds": args.boot_rounds, "reps": args.reps,
        "n_grid": list(N_GRID), "p_heldout": p_ho,
        "true_effect_alt": TRUE_EFFECT_ALT, "tost_margin": TOST_MARGIN,
        "plugin_population_increment": inc_plugin,
        "shift_alt": {"delta": d_alt, "achieved_increment": inc_alt},
        "shift_null": {"delta": d_null, "achieved_increment": inc_null},
        "sensitivity_tilt": tilt_info,
        "cells": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in cells.items()},
        "table": table,
    }
    out = OUT_ROOT / "sizing_simulation_v5.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[sim] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
