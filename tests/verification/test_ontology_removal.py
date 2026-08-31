from __future__ import annotations

from pathlib import Path
import re


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_ROOTS = ("src", "tests", "tools", "configs")
_CACHE_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}
)
_BINARY_SUFFIXES = frozenset(
    {
        ".db",
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mp4",
        ".png",
        ".pyc",
        ".pyo",
        ".shm",
        ".sqlite",
        ".wal",
        ".webp",
        ".woff",
        ".woff2",
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
    "underwater_tracking/knowledge/",
)
_REMOVED_PATTERN = re.compile("|".join(_REMOVED_SYMBOLS), re.IGNORECASE)


def _find_removed_symbol_violations(
    roots: tuple[Path, ...], *, excluded_paths: frozenset[Path]
) -> list[tuple[Path, int]]:
    violations: list[tuple[Path, int]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.resolve() in excluded_paths
                or path.suffix.lower() in _BINARY_SUFFIXES
                or _CACHE_DIRECTORIES.intersection(path.relative_to(root).parts)
            ):
                continue
            contents = path.read_bytes()
            if b"\0" in contents:
                continue
            for line_number, line in enumerate(
                contents.decode("utf-8", errors="ignore").splitlines(), start=1
            ):
                if _REMOVED_PATTERN.search(line):
                    violations.append((path, line_number))
    return violations


def test_active_source_scan_covers_textual_metadata_extensions(tmp_path: Path) -> None:
    active_root = tmp_path / "active"
    active_root.mkdir()
    generated_root = active_root / "dist"
    generated_root.mkdir()
    expected_paths = {
        "generated-metadata.txt",
        "generated-metadata.md",
        "generated-metadata.jsonl",
        "SOURCES.txt",
        "dist/generated-metadata.txt",
    }
    for relative_path in expected_paths:
        (active_root / relative_path).write_text("knowledge_client", encoding="utf-8")

    violations = _find_removed_symbol_violations((active_root,), excluded_paths=frozenset())

    assert {path.relative_to(active_root).as_posix() for path, _ in violations} == expected_paths


def test_active_source_contains_no_removed_ontology_symbols() -> None:
    this_test = Path(__file__).resolve()
    roots = tuple(_REPOSITORY_ROOT / relative_root for relative_root in _ACTIVE_ROOTS)
    violations = _find_removed_symbol_violations(roots, excluded_paths=frozenset({this_test}))

    assert not violations, "Removed ontology symbols remain in active source:\n" + "\n".join(
        f"{path.relative_to(_REPOSITORY_ROOT)}:{line_number}"
        for path, line_number in violations
    )
