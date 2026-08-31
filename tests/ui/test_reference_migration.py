"""Migration-boundary test for the migrated command UI shell.

Controller ruling 1: ``pwsh`` is unavailable on this host, so the plan's
``tests/ui/test_reference_migration.ps1`` is expressed here as an equivalent
pytest test with the SAME assertions:

(a) the required UI files exist (the brief's full required list); and
(b) no file under ``src/underwater_tracking/ui/`` contains the reference
    project's path string — ``Maritime-Surveillance`` — the plan's own
    forbidden-pattern intent.  Paths are anchored from ``__file__``, mirroring
    the PowerShell script's ``$root = Resolve-Path "$PSScriptRoot" + two
    parent segments``.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _ROOT / "src" / "underwater_tracking" / "ui"

# The brief's full required file list (plan file map, Task 1 "Files").
REQUIRED_UI_FILES = (
    "package.json",
    "vite.config.ts",
    "tsconfig.json",
    "index.html",
    "src/App.tsx",
    "src/main.tsx",
    "src/test/setup.ts",
)

# The plan's forbidden-pattern intent: the migrated tree must run entirely
# standalone, with no runtime import or asset path pointing back at the
# reference project.
FORBIDDEN_REFERENCE_PATTERN = "Maritime-Surveillance"


@pytest.mark.parametrize("relative", REQUIRED_UI_FILES)
def test_required_ui_file_exists(relative: str) -> None:
    assert (_UI_ROOT / relative).is_file(), f"missing UI file: {relative}"


def test_no_runtime_reference_to_source_project() -> None:
    if not _UI_ROOT.is_dir():
        return  # required-file assertions above already fail in that case
    offending: list[str] = []
    for path in sorted(_UI_ROOT.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if FORBIDDEN_REFERENCE_PATTERN in text:
            offending.append(str(path.relative_to(_UI_ROOT)))
    assert not offending, f"runtime reference to source project detected in: {offending}"
