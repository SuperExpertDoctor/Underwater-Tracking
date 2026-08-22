"""Explicit opt-in integration entry point for the default live acceptance."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.live_acceptance


@pytest.mark.skipif(
    os.environ.get("UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE") != "1"
    or os.environ.get("UNDERWATER_TRACKING_RUN_REAL_LLM") != "1",
    reason=(
        "set UNDERWATER_TRACKING_RUN_LIVE_ACCEPTANCE=1 and "
        "UNDERWATER_TRACKING_RUN_REAL_LLM=1 for the owned-process acceptance"
    ),
)
def test_default_main_live_acceptance() -> None:
    """Keep the expensive command discoverable without running it by default."""

    root = Path(__file__).resolve().parents[2]
    output = root / "outputs" / "acceptance-pytest.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_default_live_acceptance.py",
            "--config",
            "configs/scenario/uuv_only_single_target.yaml",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
    )
    assert result.returncode == 0
