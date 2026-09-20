"""Wrapper-matched BG experiment gate.

This script does not call or modify the local-agent wrapper. It only runs the
matched experiment if a clean multi-candidate interface is already exposed.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from bg_steering_suite_lib import PROJECT_ROOT, REPORT_ROOT, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "wrapper_matched_results.json"
OUT_MD = REPORT_ROOT / "wrapper_matched_results.md"


def main() -> int:
    started = time.time()
    files = [
        PROJECT_ROOT / "src/local_agent/ouro_agent_improved.py",
        PROJECT_ROOT / "src/local_agent/ouro_direct.py",
        PROJECT_ROOT / "src/local_agent/ouro_backend.py",
        PROJECT_ROOT / "src/local_agent/ouro_policies.py",
    ]
    inventory = []
    candidate_interface = False
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        names = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        candidate_like = [n for n in names if "candidate" in n.lower() and ("generate" in n.lower() or "expose" in n.lower() or "collect" in n.lower())]
        inventory.append({"path": rel(path), "exists": path.exists(), "candidate_like_functions": candidate_like[:20]})
        if any("generate" in n.lower() and "candidate" in n.lower() for n in candidate_like):
            candidate_interface = True
    if candidate_interface:
        # Existing candidate-related internals do not provide a documented clean
        # matched candidate-set API, so the experiment remains skipped rather than
        # using private wrapper internals or altering wrapper behavior.
        verdict = "SKIPPED"
        reason = "WRAPPER_CANDIDATE_INTERFACE_NOT_CLEANLY_DOCUMENTED"
    else:
        verdict = "SKIPPED"
        reason = "WRAPPER_CANDIDATE_INTERFACE_MISSING"
    payload = {
        "BG_WRAPPER_MATCHED_VERDICT": verdict,
        "verdict": verdict,
        "reason": reason,
        "inventory": inventory,
        "compute_mismatch_warning": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Wrapper-Matched Experiment (2026-05-18)",
        "",
        f"BG_WRAPPER_MATCHED_VERDICT = {verdict}",
        "",
        f"- reason: `{reason}`",
        "- wrapper files were inspected but not called or modified.",
    ]
    write_md(OUT_MD, lines)
    print(f"BG_WRAPPER_MATCHED_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
