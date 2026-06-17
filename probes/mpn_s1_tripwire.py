"""M+N S1.0 — live reproduction of the S0 regression tripwire (CoreContent_v2 heldout macro).

Reuses the corecontent_v2 scoring (cached features + locked tap policies) to recompute the
heldout macro top1-oracle for CoreContent_v2_blockwise + DualAnchor + mixedhead_MIX_HH_OBJECTIVE,
and asserts CoreContent ~0.6691 / DualAnchor ~0.3787 within tolerance. Confirms the value system
is consistent on today's stack BEFORE building the S1 loop. Writes ONLY to the new S1 root — does
NOT call heldout_eval_main (which would overwrite the locked artifact).

Run: venv/bin/python utilities/tests/manual/mpn_s1_tripwire.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bg_corecontent_v2_models as M  # noqa: E402

OUT = M.v2.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
REFERENCE = {"CoreContent_v2_blockwise": 0.6691, "DualAnchor": 0.3787, "mixedhead_MIX_HH_OBJECTIVE": 0.5525}
TOL = 0.01


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    policies = {}
    policies.update(M.baseline_policies(include_diag=True))
    policies.update(M.crafted_v2_policies())
    want = [n for n in REFERENCE if n in policies]
    missing = [n for n in REFERENCE if n not in policies]
    keep = {n: policies[n] for n in want}
    rows, by_pol = M.eval_policy_macro(keep, split="heldout", domains=M.CORE_DOMAINS)
    macro = {p: round(M.macro_core(d), 4) for p, d in by_pol.items()}

    checks = {}
    ok = True
    for name, ref in REFERENCE.items():
        got = macro.get(name)
        within = got is not None and abs(got - ref) <= TOL
        checks[name] = {"reference": ref, "reproduced": got, "delta": (round(got - ref, 4) if got is not None else None),
                        "within_tol": within}
        ok = ok and within
    verdict = "S1_TRIPWIRE_REPRODUCED" if ok and not missing else "S1_TRIPWIRE_MISMATCH"
    payload = {"verdict": verdict, "tolerance": TOL, "checks": checks,
               "per_domain": {p: {d: round(v, 4) for d, v in by_pol[p].items()} for p in by_pol},
               "missing_policies": missing, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_0_tripwire.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(checks, indent=2))
    print(f"missing_policies: {missing}")
    print(f"\n{verdict}")
    return 0 if verdict == "S1_TRIPWIRE_REPRODUCED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
