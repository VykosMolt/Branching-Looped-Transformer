from __future__ import annotations
from bg_corecontent_v2_models import synthesis_main, docs_main
if __name__ == "__main__":
    rc = synthesis_main()
    docs_main()
    raise SystemExit(rc)
