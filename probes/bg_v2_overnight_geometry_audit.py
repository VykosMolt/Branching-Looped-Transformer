"""Paper 1 v2 overnight programme -- Part III: subspace-vs-subspace geometry audit.

Pure linear algebra on FROZEN, already-extracted artifacts. No model loading, no new
extraction, no new intervention campaign. Reads (never writes to):
  - artifacts/reports/cross_loop_early_layer_taps_20260720/features/*.pt
    (readable-feature basis: layers {8,16,24,36,47} x loops {1..4}, per-candidate reward)
  - artifacts/reports/proto_introspection/s1_s3_exact_injection_orthogonality_2026-06-17/
    s1_s3_exact_injection_delta_bundle_2026-06-17.pt (writable/injection deltas)

Per SHARED_ARTIFACT_AUDIT.md: writable deltas exist ONLY at loop{1..4} x layer{24,36,47}
(12 late loci). No early-locus (layer 8/16) writable tensor exists anywhere in the repo.
Per programme instructions this is NOT expanded with a new intervention campaign; the
early-vs-late WRITABLE-alignment comparison is reported as
INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA, while the readable<->outcome overlap and
loop-to-loop rotation analyses (which need only the frozen readable features, present at
all five layers) are run in full at all loci.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

XLOOP_ROOT = PROJECT_ROOT / "artifacts/reports/cross_loop_early_layer_taps_20260720"
INJECTION_BUNDLE = (PROJECT_ROOT / "artifacts/reports/proto_introspection/"
                    "s1_s3_exact_injection_orthogonality_2026-06-17/"
                    "s1_s3_exact_injection_delta_bundle_2026-06-17.pt")
OUT_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724/subspace_geometry"

LAYERS = [8, 16, 24, 36, 47]
SEED = 20260724
RANKS = [1, 3, 5]
N_BOOT_READABLE = 40
N_BOOT_OUTCOME = 100
N_NULL_RANDOM = 2000
N_NULL_PERM = 25  # expensive (refits an ensemble); reduced-round secondary null, documented as such.

LOCI = {
    # name: (loop_1indexed, layer)
    "L1_16": (1, 16),  # optional loop-transition control
    "L2_16": (2, 16),  # optional loop-transition control
    "L3_8": (3, 8),
    "L3_16": (3, 16),
    "L4_8": (4, 8),
    "L4_16": (4, 16),
    "L4_24": (4, 24),
    "L4_36": (4, 36),
    "L4_47": (4, 47),
}
LATE_LOCI = ["L4_24", "L4_36", "L4_47"]
EARLY_LOCI = ["L3_8", "L3_16", "L4_8", "L4_16"]


def log(msg: str) -> None:
    print(f"[geometry] {msg}", flush=True)


# --------------------------------------------------------------------- data loading
def load_readable_pool() -> list[dict]:
    manifest = json.loads((XLOOP_ROOT / "feature_manifest.json").read_text())
    assert manifest["layers"] == LAYERS, f"layer order mismatch: {manifest['layers']}"
    rows = []
    for shard in manifest["per_shard"]:
        payload = torch.load(XLOOP_ROOT / "features" / shard["file"], map_location="cpu", weights_only=False)
        domain = payload.get("domain")
        split = payload.get("split")
        for g in payload["groups"]:
            task_uid = g["task_uid"]
            for c in g["candidates"]:
                rows.append({
                    "task_uid": task_uid, "domain": domain, "split": split,
                    "reward": float(c["reward"]), "features": c["features"],  # [5,4,2048] fp16
                })
    log(f"loaded {len(rows)} candidate rows from {len(manifest['per_shard'])} shards")
    return rows


def locus_index(layer: int, loop_1indexed: int) -> tuple[int, int]:
    return LAYERS.index(layer), loop_1indexed - 1


def build_locus_table(rows: list[dict], layer: int, loop_1indexed: int):
    li, lo = locus_index(layer, loop_1indexed)
    X = torch.stack([r["features"][li, lo, :].float() for r in rows])
    y = torch.tensor([1.0 if r["reward"] >= 0.5 else 0.0 for r in rows])
    task_uids = [r["task_uid"] for r in rows]
    splits = [r["split"] for r in rows]
    return X, y, task_uids, splits


def load_writable_deltas() -> dict:
    bundle = torch.load(INJECTION_BUNDLE, map_location="cpu", weights_only=False)
    rows = bundle["branch_rows"]
    delta = bundle["delta_locus"].float()
    by_locus: dict[str, torch.Tensor] = {}
    for name, (loop_1i, layer) in LOCI.items():
        if name not in LATE_LOCI:
            continue
        idx = [i for i, r in enumerate(rows) if r["loop_1indexed"] == loop_1i and r["layer"] == layer]
        if idx:
            by_locus[name] = delta[idx]
    return by_locus


# --------------------------------------------------------------------- probes / directions
def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0).clamp(min=1e-6)
    return mu, sd


def fit_logreg_torch(X, y, l2=2.0, iters=100):
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + l2 * (w @ w) / n
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def readable_direction(X, y, k_pca=30, l2=2.0):
    """PCA(lowrank) -> logistic; map weight back to original feature-space direction."""
    mu, sd = standardize_fit(X)
    Z = (X - mu) / sd
    Zc = Z - Z.mean(0)
    k = min(k_pca, Zc.shape[0] - 1, Zc.shape[1])
    U, S, V = torch.svd_lowrank(Zc, q=k)
    comps = V.T  # [k, d]
    P = Z @ comps.T  # [n, k]
    w_p, b = fit_logreg_torch(P, y, l2=l2)
    w_z = comps.T @ w_p  # [d] in standardized space
    w_x = w_z / sd  # [d] in original feature space
    n = w_x.norm()
    return (w_x / n) if n > 0 else None


def outcome_direction_meandiff(X, y):
    mu, sd = standardize_fit(X)
    Z = (X - mu) / sd
    pos = Z[y >= 0.5]
    neg = Z[y < 0.5]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        return None
    w_z = pos.mean(0) - neg.mean(0)
    w_x = w_z / sd
    n = w_x.norm()
    return (w_x / n) if n > 0 else None


def bootstrap_task_indices(task_uids: list[str], rng: torch.Generator) -> list[str]:
    uniq = sorted(set(task_uids))
    idx = torch.randint(0, len(uniq), (len(uniq),), generator=rng)
    return [uniq[i] for i in idx.tolist()]


def gather_by_tasks(X, y, task_uids, sampled_tasks):
    from collections import defaultdict
    by_task = defaultdict(list)
    for i, t in enumerate(task_uids):
        by_task[t].append(i)
    idxs = []
    for t in sampled_tasks:
        idxs.extend(by_task.get(t, []))
    if not idxs:
        return None, None
    idx_t = torch.tensor(idxs)
    return X[idx_t], y[idx_t]


def build_ensemble_subspace(X, y, task_uids, splits, direction_fn, n_boot, seed_tag: str):
    train_mask = [s == "train" for s in splits]
    Xtr = X[train_mask]
    ytr = y[train_mask]
    tasks_tr = [t for t, m in zip(task_uids, train_mask) if m]
    rng = torch.Generator().manual_seed(int(hashlib_seed(seed_tag)))
    dirs = []
    for b in range(n_boot):
        sampled = bootstrap_task_indices(tasks_tr, rng)
        Xb, yb = gather_by_tasks(Xtr, ytr, tasks_tr, sampled)
        if Xb is None or yb.float().mean() in (0.0, 1.0) or Xb.shape[0] < 10:
            continue
        try:
            d = direction_fn(Xb, yb)
        except Exception:
            d = None
        if d is not None and torch.isfinite(d).all():
            dirs.append(d)
    if len(dirs) < 5:
        return None, len(dirs)
    D = torch.stack(dirs)  # [B, d]
    return D, len(dirs)


def hashlib_seed(tag: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"{SEED}{tag}".encode()).hexdigest()[:8], 16)


def orthonormal_basis_from_ensemble(D: torch.Tensor, rank: int) -> torch.Tensor:
    """SVD of the stacked [B,d] direction ensemble; top-`rank` right singular vectors."""
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    r = min(rank, Vh.shape[0])
    return Vh[:r].T  # [d, r] orthonormal columns


def writable_basis(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, int]:
    Dc = delta - delta.mean(0)
    U, S, Vh = torch.linalg.svd(Dc, full_matrices=False)
    eff_rank = int((S > 1e-6 * S[0]).sum().item()) if S.numel() and S[0] > 0 else 0
    r = min(rank, Vh.shape[0])
    return Vh[:r].T, eff_rank


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> list[float]:
    """A: [d,rA], B: [d,rB] orthonormal columns. Returns principal angles in degrees."""
    M = A.T @ B
    s = torch.linalg.svdvals(M)
    s = s.clamp(-1.0, 1.0)
    return [math.degrees(math.acos(float(v))) for v in s]


def random_orthonormal(d: int, rank: int, rng: torch.Generator) -> torch.Tensor:
    G = torch.randn(d, rank, generator=rng)
    Q, _ = torch.linalg.qr(G)
    return Q


# --------------------------------------------------------------------- main analysis
def main() -> None:
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_readable_pool()
    writable = load_writable_deltas()
    log(f"writable loci available: {list(writable.keys())} (shapes: "
        f"{ {k: tuple(v.shape) for k, v in writable.items()} })")

    results = {"seed": SEED, "ranks": RANKS, "n_boot_readable": N_BOOT_READABLE,
               "n_boot_outcome": N_BOOT_OUTCOME, "n_null_random": N_NULL_RANDOM,
               "n_null_perm": N_NULL_PERM, "loci": {}}

    subspaces = {}  # locus -> {"readable": {rank: basis}, "outcome": {rank: basis}}
    for name, (loop_1i, layer) in LOCI.items():
        log(f"locus {name} (loop{loop_1i}, layer{layer}): building tables ...")
        X, y, task_uids, splits = build_locus_table(rows, layer, loop_1i)
        n_train = sum(1 for s in splits if s == "train")
        base_rate = float(y.mean())
        log(f"  n={X.shape[0]} n_train={n_train} base_rate={base_rate:.3f}")

        D_read, n_read = build_ensemble_subspace(X, y, task_uids, splits, readable_direction,
                                                  N_BOOT_READABLE, f"{name}:readable")
        D_out, n_out = build_ensemble_subspace(X, y, task_uids, splits, outcome_direction_meandiff,
                                                N_BOOT_OUTCOME, f"{name}:outcome")
        entry = {"n_candidates": int(X.shape[0]), "n_train_candidates": n_train,
                  "base_rate": round(base_rate, 4), "n_readable_dirs": n_read, "n_outcome_dirs": n_out}
        subspaces[name] = {"readable": {}, "outcome": {}, "d": X.shape[1]}
        if D_read is not None:
            for r in RANKS:
                subspaces[name]["readable"][r] = orthonormal_basis_from_ensemble(D_read, r)
        if D_out is not None:
            for r in RANKS:
                subspaces[name]["outcome"][r] = orthonormal_basis_from_ensemble(D_out, r)
        results["loci"][name] = entry

    # --------------------------------------------------------- nulls (random rank-matched)
    rng = torch.Generator().manual_seed(SEED)
    d_ambient = 2048
    null_random = {}
    for r in RANKS:
        vals = []
        for _ in range(N_NULL_RANDOM):
            A = random_orthonormal(d_ambient, r, rng)
            B = random_orthonormal(d_ambient, r, rng)
            angs = principal_angles(A, B)
            vals.append(sum(angs) / len(angs))
        null_random[r] = {"mean_angle_deg": sum(vals) / len(vals),
                           "p05": sorted(vals)[int(0.05 * len(vals))],
                           "p50": sorted(vals)[int(0.50 * len(vals))],
                           "p95": sorted(vals)[int(0.95 * len(vals))]}
    results["null_random_rankmatched"] = null_random
    log(f"random-subspace null (rank-matched, n={N_NULL_RANDOM}): {null_random}")

    # --------------------------------------------------------- readable<->outcome overlap (all loci)
    readable_outcome = {}
    for name in LOCI:
        entry = {}
        for r in RANKS:
            A = subspaces[name]["readable"].get(r)
            B = subspaces[name]["outcome"].get(r)
            if A is None or B is None:
                entry[r] = {"status": "MISSING"}
                continue
            angs = principal_angles(A, B)
            mean_ang = sum(angs) / len(angs)
            null = null_random[r]
            entry[r] = {
                "principal_angles_deg": [round(a, 2) for a in angs],
                "mean_angle_deg": round(mean_ang, 2),
                "null_mean_deg": round(null["mean_angle_deg"], 2),
                "null_p05_deg": round(null["p05"], 2),
                "enriched_vs_null": bool(mean_ang < null["p05"]),
            }
        readable_outcome[name] = entry
    results["readable_outcome_principal_angles"] = readable_outcome

    # --------------------------------------------------------- readable/outcome <-> writable (late loci only)
    writable_geometry = {}
    for name in LATE_LOCI:
        if name not in writable:
            writable_geometry[name] = {"status": "MISSING_WRITABLE"}
            continue
        delta = writable[name]
        entry = {"n_writable_samples": int(delta.shape[0])}
        for r in RANKS:
            if r >= delta.shape[0]:
                entry[r] = {"status": "RANK_EXCEEDS_N"}
                continue
            Wb, eff_rank = writable_basis(delta, r)
            Ar = subspaces[name]["readable"].get(r)
            Or = subspaces[name]["outcome"].get(r)
            row = {"writable_effective_rank": eff_rank}
            if Ar is not None:
                a = principal_angles(Ar, Wb)
                row["readable_vs_writable_deg"] = [round(x, 2) for x in a]
                row["readable_vs_writable_mean_deg"] = round(sum(a) / len(a), 2)
            if Or is not None:
                a2 = principal_angles(Or, Wb)
                row["outcome_vs_writable_deg"] = [round(x, 2) for x in a2]
                row["outcome_vs_writable_mean_deg"] = round(sum(a2) / len(a2), 2)
            null = null_random[r]
            row["null_mean_deg"] = round(null["mean_angle_deg"], 2)
            row["null_p05_deg"] = round(null["p05"], 2)
            if "readable_vs_writable_mean_deg" in row:
                row["readable_enriched_vs_null"] = bool(row["readable_vs_writable_mean_deg"] < null["p05"])
            if "outcome_vs_writable_mean_deg" in row:
                row["outcome_enriched_vs_null"] = bool(row["outcome_vs_writable_mean_deg"] < null["p05"])
            entry[r] = row
        writable_geometry[name] = entry
    results["writable_geometry_late_loci_only"] = writable_geometry
    results["early_locus_writable_status"] = "INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA"
    results["early_locus_writable_reason"] = (
        "No writable/injection/perturbation tensor exists at physical layer 8 or 16 "
        "anywhere in the repo (verified: s1_s3_exact_injection_delta_bundle loci are the "
        "cross product of loop{1..4} x layer{24,36,47} only). Per programme rules, this "
        "was not expanded with a new intervention campaign and is not fabricated or "
        "transported across coordinate systems."
    )

    # --------------------------------------------------------- loop-to-loop readable rotation (layer 16 fixed)
    loop_rotation = {}
    loop_names = ["L1_16", "L2_16", "L3_16", "L4_16"]
    transitions = [("L1_16", "L2_16"), ("L2_16", "L3_16"), ("L3_16", "L4_16")]
    for a_name, b_name in transitions:
        entry = {}
        for r in RANKS:
            A = subspaces[a_name]["readable"].get(r)
            B = subspaces[b_name]["readable"].get(r)
            if A is None or B is None:
                entry[r] = {"status": "MISSING"}
                continue
            angs = principal_angles(A, B)
            mean_ang = sum(angs) / len(angs)
            null = null_random[r]
            entry[r] = {
                "principal_angles_deg": [round(a, 2) for a in angs],
                "mean_angle_deg": round(mean_ang, 2),
                "null_mean_deg": round(null["mean_angle_deg"], 2),
                "rotation_beyond_null": bool(mean_ang > null["p05"]),
            }
        loop_rotation[f"{a_name}_to_{b_name}"] = entry
    results["loop_to_loop_readable_rotation_L16"] = loop_rotation

    # --------------------------------------------------------- reduced-round permutation null (secondary)
    perm_null = {}
    name0 = "L4_47"
    X, y, task_uids, splits = build_locus_table(rows, 47, 4)
    train_mask = [s == "train" for s in splits]
    Xtr, ytr = X[train_mask], y[train_mask]
    rng2 = torch.Generator().manual_seed(hashlib_seed("perm_null"))
    perm_angles = []
    for i in range(N_NULL_PERM):
        perm = torch.randperm(ytr.shape[0], generator=rng2)
        y_perm = ytr[perm]
        d = outcome_direction_meandiff(Xtr, y_perm)
        A = subspaces[name0]["outcome"].get(3)
        if d is None or A is None:
            continue
        ang = principal_angles(A, d.unsqueeze(1))
        perm_angles.append(sum(ang) / len(ang))
    if perm_angles:
        perm_null["locus"] = name0
        perm_null["rank"] = 3
        perm_null["n_rounds"] = len(perm_angles)
        perm_null["mean_angle_deg"] = round(sum(perm_angles) / len(perm_angles), 2)
        perm_null["note"] = ("reduced-round label-permutation null (single label-shuffle "
                             "outcome direction vs the real rank-3 outcome subspace); "
                             "expensive refit-based null kept small on purpose -- see "
                             "RESULTS.md sacrifice log")
    results["secondary_label_permutation_null"] = perm_null

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    out_path = OUT_ROOT / "geometry_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"wrote {out_path}; elapsed={results['elapsed_seconds']}s")

    # save subspace bases (hashes only, to keep the json bounded) + a torch file with full bases
    bases_path = OUT_ROOT / "subspace_bases.pt"
    torch.save({name: {
        "readable": subspaces[name]["readable"],
        "outcome": subspaces[name]["outcome"],
    } for name in LOCI}, bases_path)
    log(f"wrote {bases_path}")


if __name__ == "__main__":
    main()
