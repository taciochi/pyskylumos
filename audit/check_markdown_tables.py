"""Validate the structure and completeness of project Markdown tables."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"audit", "build", "dist"}
PLACEHOLDERS = {"-", "\N{EN DASH}", "\N{EM DASH}"}
SEPARATOR = re.compile(r"^:?-{3,}:?$")


def documentation_files() -> list[Path]:
    """Return live Markdown documentation while excluding generated/internal trees."""
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(part.startswith(".") or part in EXCLUDED_PARTS for part in path.parts)
    )


def table_cells(line: str) -> list[str]:
    """Split a pipe-delimited row while respecting backslash-escaped pipes."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


def validate_document(path: Path) -> list[str]:
    """Return structural and completeness problems for every table in one document."""
    problems: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0

    while index < len(lines):
        if not lines[index].startswith("|"):
            index += 1
            continue

        start = index
        table: list[tuple[int, list[str]]] = []
        while index < len(lines) and lines[index].startswith("|"):
            table.append((index + 1, table_cells(lines[index])))
            index += 1

        relative = path.relative_to(ROOT)
        if len(table) < 3:
            problems.append(f"{relative}:{start + 1}: table needs a header and data row")
            continue

        expected_columns = len(table[0][1])
        separator_cells = table[1][1]
        if len(separator_cells) != expected_columns or not all(
            SEPARATOR.fullmatch(cell) for cell in separator_cells
        ):
            problems.append(f"{relative}:{table[1][0]}: invalid table separator row")

        for line_number, cells in table:
            if len(cells) != expected_columns:
                problems.append(
                    f"{relative}:{line_number}: expected {expected_columns} columns, "
                    f"found {len(cells)}; escape internal pipe characters or use abs(...)"
                )
            for column, cell in enumerate(cells, start=1):
                if not cell:
                    problems.append(f"{relative}:{line_number}: column {column} is empty")
                elif cell in PLACEHOLDERS:
                    problems.append(
                        f"{relative}:{line_number}: column {column} contains only {cell!r}; "
                        "state the intended meaning explicitly"
                    )

    return problems


def main() -> None:
    """Validate every live Markdown document and print a CI-friendly result."""
    documents = documentation_files()
    problems = [problem for path in documents for problem in validate_document(path)]
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"markdown tables: PASS ({len(documents)} documents)")


if __name__ == "__main__":
    main()
