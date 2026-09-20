from __future__ import annotations
import os
from bg_core_tap_audit_v1_common import extract_logic_features
if __name__ == "__main__":
    raise SystemExit(extract_logic_features(int(os.environ.get("LOGIC_TASKS_PER_SPLIT", "40"))))
