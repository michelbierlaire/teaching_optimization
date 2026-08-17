#!/usr/bin/env python3
"""Convert course scripts to notebooks and report failures per file."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from teaching_optimization.to_jupyter import ConversionError, check_source, convert_file


SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "__pypackages__",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "vendor",
}


@dataclass
class GenerationReport:
    converted: int = 0
    failed: int = 0
    skipped: int = 0


def _iter_python_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*.py")):
        if path.name == "generate_notebooks.py":
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        yield path


def _input_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.directory is not None:
        if not args.directory.is_dir():
            raise ValueError(f"not a directory: {args.directory}")
        paths.extend(_iter_python_files(args.directory))
    paths.extend(args.files)
    for path in args.paths:
        if path.is_dir():
            paths.extend(_iter_python_files(path))
        else:
            paths.append(path)

    unique: dict[Path, None] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved.suffix != ".py":
            continue
        if resolved.name == "generate_notebooks.py":
            continue
        if any(part in SKIP_DIRECTORIES for part in resolved.parts):
            continue
        unique[resolved] = None
    return sorted(unique)


def generate_notebooks(
    files: Iterable[Path],
    *,
    verbose: bool = False,
    docstring_as_code: bool = False,
) -> GenerationReport:
    """Convert all files, continuing after individual failures."""

    report = GenerationReport()
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            check_source(source, source_file=str(path), docstring_as_code=docstring_as_code)
            output = convert_file(path.as_posix(), docstring_as_code=docstring_as_code)
            report.converted += 1
            if verbose:
                print(f"Converted: {path} -> {output}")
        except (OSError, SyntaxError, ConversionError, RuntimeError) as error:
            report.failed += 1
            print(f"Failed: {path}: {error}", file=sys.stderr)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories to process")
    parser.add_argument("--directory", type=Path, help="recursively process only this directory")
    parser.add_argument("--files", nargs="+", type=Path, default=[], help="explicit list of Python files")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--docstring-as-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.directory is None and not args.files and not args.paths:
        args.directory = PROJECT_ROOT / "examples"
    try:
        files = _input_files(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    report = generate_notebooks(
        files,
        verbose=args.verbose,
        docstring_as_code=args.docstring_as_code,
    )
    print(f"Converted: {report.converted}")
    print(f"Failed: {report.failed}")
    print(f"Skipped: {report.skipped}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
