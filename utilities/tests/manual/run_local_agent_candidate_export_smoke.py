"""Tiny optional live smoke for local-agent candidate export."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
LOCAL_AGENT_DIR = ROOT / "shared/src" / "local_agent"
REPORT_DIR = ROOT / "artifacts" / "reports" / "probes"
OUT_JSON = REPORT_DIR / "local_agent_candidate_export_smoke_2026-05-18.json"
OUT_MD = REPORT_DIR / "local_agent_candidate_export_smoke_2026-05-18.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LOCAL_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_AGENT_DIR))


PROMPT = "Write a Python function add_one(x) that returns x + 1."


def write_reports(data: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Local-Agent Candidate Export Smoke",
        "",
        f"WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = {data['WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT']}",
        "",
        f"- Prompt: {PROMPT}",
        f"- Candidates captured: {data.get('candidate_count', 0)}",
        f"- Stages present: {', '.join(data.get('stages_present', [])) or 'none'}",
        f"- Normal output produced: {data.get('normal_output_produced')}",
        f"- Reason: {data.get('reason', '')}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    if os.environ.get("RUN_LIVE_LOCAL_AGENT_CANDIDATE_SMOKE") != "1":
        data = {
            "WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT": "SKIPPED",
            "reason": "live wrapper generation is optional and disabled unless RUN_LIVE_LOCAL_AGENT_CANDIDATE_SMOKE=1",
            "prompt": PROMPT,
            "candidate_count": 0,
            "stages_present": [],
            "normal_output_produced": False,
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_reports(data)
        print("WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = SKIPPED")
        print(f"wrote {OUT_JSON}")
        print(f"wrote {OUT_MD}")
        return

    try:
        import ouro_agent_improved as agent

        model_mgr = agent.LazyPrimaryModelManager()
        trace_result = agent.direct_answer(
            model_mgr,
            PROMPT,
            task_profile=agent.default_task_profile(PROMPT),
            return_candidate_trace=True,
            task_id="smoke/add_one",
        )
        answer, trace = trace_result if isinstance(trace_result, tuple) else (trace_result, None)
        stages = [candidate.stage for candidate in trace.candidates] if trace is not None else []
        verdict = "PASS" if trace is not None and trace.candidates and answer else "PARTIAL"
        data = {
            "WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT": verdict,
            "reason": "direct live smoke completed",
            "prompt": PROMPT,
            "candidate_count": len(stages),
            "stages_present": sorted(set(stages)),
            "normal_output_produced": bool(answer),
            "selected_candidate_uid": trace.selected_candidate_uid if trace is not None else None,
            "final_answer_preview": str(answer)[:500],
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - smoke report
        data = {
            "WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT": "FAIL",
            "reason": repr(exc),
            "prompt": PROMPT,
            "candidate_count": 0,
            "stages_present": [],
            "normal_output_produced": False,
            "elapsed_sec": round(time.time() - started, 3),
        }
    write_reports(data)
    print(f"WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = {data['WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT']}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if data["WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
