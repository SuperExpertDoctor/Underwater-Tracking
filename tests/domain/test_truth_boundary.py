# tests/domain/test_truth_boundary.py
from pathlib import Path


def test_operational_modules_do_not_import_truth():
    roots = [Path("src/underwater_tracking/tracking"), Path("src/underwater_tracking/groups")]
    offenders = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "domain.truth" in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
    assert offenders == []
