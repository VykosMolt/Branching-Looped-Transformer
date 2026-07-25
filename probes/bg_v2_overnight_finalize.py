"""Paper 1 v2 overnight programme -- finalization: SHA256SUMS + FINAL_GIT_STATUS.

Run once, at the very end, after all three experiments have written their outputs.
Computes a SHA256 for every file under the programme root (excluding SHA256SUMS itself)
and records the final git status for the master report.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/moloch/ouro_project")
ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724"


def main() -> None:
    lines = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() or p.name == "SHA256SUMS":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(ROOT)}")
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[finalize] wrote SHA256SUMS with {len(lines)} entries")

    status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True).stdout
    (ROOT / "FINAL_GIT_STATUS.txt").write_text(status, encoding="utf-8")
    print(f"[finalize] wrote FINAL_GIT_STATUS.txt ({len(status.splitlines())} lines)")


if __name__ == "__main__":
    main()
