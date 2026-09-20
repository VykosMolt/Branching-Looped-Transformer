"""M+N run — S0 backup of all irreplaceable assets, with verified checksums.

Backs up the operational backbone, the pristine RLTT FSDP/optimizer checkpoint, and the live
tap/value artifacts to a separate verified location BEFORE any backbone training (the user's hard
precondition). For every asset: stream-sha256 the source, copy, re-sha256 the copy, assert the
hashes match; write a manifest JSON. Exits non-zero on ANY mismatch or missing required asset.

Run: venv/bin/python utilities/tools/mpn_s0_backup.py   (set OVERWRITE=1 to redo an existing backup)
Caveat recorded in the manifest: the backup lands on the SAME nvme as the sources, so it protects
against accidental overwrite/deletion, NOT disk failure — an offsite/cloud copy is still advised.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HOME = Path("/home/moloch")
PROJECT = HOME / "ouro_project"
DEST = HOME / "ouro_backups" / "mpn_s0_pre_run_2026-06-13"
CHUNK = 1 << 20  # 1 MiB

# (label, source_path, required) — directories are copied whole.
ASSETS = [
    ("backbone_operational", PROJECT / "shared/models/ouro_rltt_local", True),
    ("backbone_rltt_fsdp_source", HOME / "Downloads/RLTT/Downloads_RLTT/RLTT", True),
    ("tap_dualanchor_mix_heads", PROJECT / "opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.pt", True),
    ("tap_corecontent_v2_policy", PROJECT / "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/corecontent_v2_policy.pt", True),
    ("tap_corecontent_v2_linear_pairwise", PROJECT / "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/corecontent_v2_linear_pairwise.pt", True),
    ("tap_corecontent_v2_listwise", PROJECT / "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/corecontent_v2_listwise.pt", True),
    ("tap_corecontent_v2_domain_gated", PROJECT / "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/domain_gated_v2.pt", False),
    ("tap_corecontent_v2_weight_merge", PROJECT / "opi/taps/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/weight_merge_v2.pt", False),
    ("tap_constructed_taps_dir", PROJECT / "opi/taps/probes/bg_core_domain_tap_audit_dualanchor_readiness_v1_2026-06-04/constructed_taps", True),
    ("tap_head_registry", PROJECT / "opi/taps/probes/bg_head_registry_2026-05-17.pt", True),
    ("tap_core_domain_inventory", PROJECT / "opi/taps/probes/bg_core_domain_tap_audit_dualanchor_readiness_v1_2026-06-04/tap_inventory.json", False),
    ("evaluator_pairwise_epoch2_defunct_ref", PROJECT / "rpe/checkpoints/evaluator/pairwise_epoch2.pt", False),
]


def sha256_file(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def hash_tree(root: Path) -> tuple[dict[str, list], int, int]:
    """Per-file sha256 over a directory tree (sorted), returns {relpath: [sha, bytes]}, totals."""
    out: dict[str, list] = {}
    total_bytes = 0
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        sha, n = sha256_file(p)
        out[str(p.relative_to(root))] = [sha, n]
        total_bytes += n
    return out, total_bytes, len(files)


def main() -> int:
    started = time.time()
    overwrite = os.environ.get("OVERWRITE") == "1"
    if DEST.exists() and any(DEST.iterdir()) and not overwrite:
        print(f"REFUSING: backup dir {DEST} exists and is non-empty. Set OVERWRITE=1 to redo.")
        return 2
    DEST.mkdir(parents=True, exist_ok=True)

    # required-asset presence check FIRST (fail before copying anything)
    missing = [str(src) for _, src, req in ASSETS if req and not src.exists()]
    if missing:
        print("REFUSING: required assets missing:\n  " + "\n  ".join(missing))
        return 3

    manifest: dict = {"dest": str(DEST), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "same_disk_caveat": "backup is on the same nvme as sources; protects against "
                      "overwrite/deletion, NOT disk failure — keep an offsite/cloud copy too",
                      "assets": {}}
    all_ok = True
    for label, src, req in ASSETS:
        if not src.exists():
            manifest["assets"][label] = {"source": str(src), "status": "ABSENT_OPTIONAL"}
            print(f"[skip] {label}: absent (optional)")
            continue
        dst = DEST / label / src.name if src.is_dir() else DEST / label / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[copy] {label}: {src}", flush=True)
        if src.is_dir():
            src_hashes, src_bytes, nf = hash_tree(src)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            dst_hashes, dst_bytes, _ = hash_tree(dst)
            ok = src_hashes == dst_hashes
            manifest["assets"][label] = {"source": str(src), "dest": str(dst), "kind": "dir",
                                         "n_files": nf, "bytes": src_bytes, "match": ok,
                                         "per_file_sha256": src_hashes}
        else:
            src_sha, src_bytes = sha256_file(src)
            shutil.copy2(src, dst)
            dst_sha, dst_bytes = sha256_file(dst)
            ok = (src_sha == dst_sha) and (src_bytes == dst_bytes)
            manifest["assets"][label] = {"source": str(src), "dest": str(dst), "kind": "file",
                                         "bytes": src_bytes, "sha256": src_sha, "match": ok}
        all_ok = all_ok and ok
        print(f"       -> {'OK' if ok else 'MISMATCH!!'}  ({manifest['assets'][label].get('bytes',0)/1e9:.3f} GB)", flush=True)

    manifest["all_verified"] = all_ok
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    manifest["env"] = _env()
    (DEST / "S0_BACKUP_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nMANIFEST: {DEST / 'S0_BACKUP_MANIFEST.json'}")
    print(f"ALL_VERIFIED = {all_ok}")
    return 0 if all_ok else 4


def _env() -> dict:
    info = {}
    try:
        import torch, transformers
        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
    except Exception as e:
        info["import_error"] = str(e)
    for m in ("peft", "trl", "bitsandbytes", "datasets"):
        try:
            info[m] = __import__(m).__version__
        except Exception:
            pass
    return info


if __name__ == "__main__":
    raise SystemExit(main())
