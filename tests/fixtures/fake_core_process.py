from __future__ import annotations

import sys
from pathlib import Path

_LOCAL_ROOT = Path(__file__).resolve().parents[2]
_PROJECTS_ROOT = _LOCAL_ROOT.parent
sys.path.insert(0, str(_LOCAL_ROOT / "src"))
sys.path.insert(
    0,
    str(_PROJECTS_ROOT / "ledgermind-integrations" / "protocol" / "python" / "src"),
)

from ledgermind_local.core_gateway.fake_process import main

if __name__ == "__main__":
    raise SystemExit(main())
