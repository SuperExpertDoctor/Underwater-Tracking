#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run() -> int:
    from underwater_tracking.verification.uuv_tracking_coverage_runner import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run())
