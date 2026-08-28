from __future__ import annotations

from pathlib import Path
import re


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_ROOTS = ("src", "tests", "tools", "configs")
_SOURCE_SUFFIXES = frozenset(
    {
        ".css",
        ".html",
        ".js",
        ".json",
        ".jsx",
        ".ps1",
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
)
_REMOVED_SYMBOLS = (
    "OntologyKnowledgeClient",
    "KnowledgeProvider",
    "KnowledgeQueryResult",
    "KnowledgeQueryRun",
    "knowledge_client",
    "KnowledgeConfig",
    "knowledge_queries",
    "source_knowledge_ids",
)
_REMOVED_PATTERN = re.compile("|".join(_REMOVED_SYMBOLS), re.IGNORECASE)


def test_active_source_contains_no_removed_ontology_symbols() -> None:
    violations: list[str] = []
    this_test = Path(__file__).resolve()

    for relative_root in _ACTIVE_ROOTS:
        root = _REPOSITORY_ROOT / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.resolve() == this_test
                or path.suffix not in _SOURCE_SUFFIXES
            ):
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if _REMOVED_PATTERN.search(line):
                    violations.append(f"{path.relative_to(_REPOSITORY_ROOT)}:{line_number}")

    assert not violations, "Removed ontology symbols remain in active source:\n" + "\n".join(
        violations
    )
