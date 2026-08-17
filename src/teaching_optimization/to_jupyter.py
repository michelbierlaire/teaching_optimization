"""Convert teaching-oriented Python scripts into Jupyter notebooks.

The default convention is deliberately small and predictable:

* consecutive top-level comment lines form one Markdown cell;
* consecutive non-comment lines form one code cell;
* blank physical lines remain in the current cell;
* a module-level docstring becomes Markdown by default; and
* ``# %%`` and ``# %% [markdown]`` explicitly start cells.

Blank lines are not cell separators. For example::

    # This becomes Markdown.
    #
    # It is one Markdown cell.

    import pandas as pd

    data = pd.DataFrame(...)

This produces one Markdown cell followed by one code cell.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import re
import sys
import textwrap
import tokenize
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Optional, Sequence

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


class ConversionError(ValueError):
    """Raised when a source file cannot be converted safely."""


class UnsupportedCellMarkerError(ConversionError):
    """Raised for an unsupported ``# %%`` cell marker."""


@dataclass
class ConversionReport:
    """Summary returned by :func:`check_source`."""

    markdown_cells: int
    code_cells: int
    warnings: list[str] = field(default_factory=list)

    @property
    def total_cells(self) -> int:
        return self.markdown_cells + self.code_cells


class Block(ABC):
    """Compatibility base class for the original block-oriented API."""

    def __init__(self, lines: list[str]):
        self.lines = lines
        self._start: int | None = None
        self._end: int | None = None

    @property
    def start(self) -> int | None:
        return self._start

    @start.setter
    def start(self, value: int | None) -> None:
        self._start = value
        self.end = None
        if value is not None:
            self.identifies_block()

    @property
    def end(self) -> int | None:
        return self._end

    @end.setter
    def end(self, value: int | None) -> None:
        self._end = value
        if value is not None:
            self.clean_block()

    @abstractmethod
    def first_line(self, index: int) -> bool:
        """Return whether ``index`` starts this type of block."""

    def process_block(self, index: int) -> None:
        if not self.first_line(index):
            raise AssertionError(f"Line {index} does not start the block")
        self.start = index

    @abstractmethod
    def identifies_block(self) -> None:
        """Set the end of the block."""

    @abstractmethod
    def clean_block(self) -> None:
        """Normalize the block before it is joined into a cell."""

    def get_block(self) -> list[str]:
        if self.start is None:
            raise ValueError("The beginning of the block has not been identified yet.")
        return self.lines[self.start : self.end]

    @abstractmethod
    def get_cell(self) -> NotebookNode:
        """Return this block as a notebook cell."""

    def is_block_empty(self) -> bool:
        if self.start is None or self.end is None:
            raise ValueError(f"{self.start=} {self.end=}")
        return all(line.strip() == "" for line in self.get_block())


class MarkdownBlock(Block):
    """Compatibility block for a sequence of top-level comment lines."""

    def first_line(self, index: int) -> bool:
        return 0 <= index < len(self.lines) and self.lines[index].lstrip().startswith("#")

    def identifies_block(self) -> None:
        if self.start is None:
            raise ValueError("The beginning of the block has not been identified yet.")
        if self.end is not None:
            raise AssertionError("Should not be called again")
        for index in range(self.start, len(self.lines)):
            if not self.lines[index].lstrip().startswith("#"):
                self.end = index
                return
        self.end = len(self.lines)

    def clean_block(self) -> None:
        if self.start is None or self.end is None:
            raise AssertionError("Both should be set when this function is called")
        for index in range(self.start, self.end):
            line = self.lines[index].lstrip()
            self.lines[index] = line[1:].lstrip().rstrip("\r\n")

    def get_cell(self) -> NotebookNode:
        return new_markdown_cell("\n".join(self.get_block()))


class DocstringBlock(Block):
    """Compatibility block for a triple-quoted string."""

    def __init__(self, lines: list[str], docstring_as_code: bool = False) -> None:
        super().__init__(lines)
        self.as_code = docstring_as_code
        self.closing_sequence = '"""'

    def first_line(self, index: int) -> bool:
        if not 0 <= index < len(self.lines):
            return False
        line = self.lines[index].lstrip()
        if line.startswith('"""'):
            self.closing_sequence = '"""'
            return True
        if line.startswith("'''"):
            self.closing_sequence = "'''"
            return True
        return False

    def identifies_block(self) -> None:
        if self.start is None:
            raise ValueError("The beginning of the block has not been identified yet.")
        if self.end is not None:
            raise AssertionError("Should not be called again")
        for index in range(self.start + 1, len(self.lines)):
            if self.lines[index].lstrip().startswith(self.closing_sequence):
                self.end = index + 1
                return
        self.end = len(self.lines)

    def clean_block(self) -> None:
        if self.start is None or self.end is None:
            raise AssertionError("Both should be set")
        raw = _normalize_newlines("".join(self.lines[self.start : self.end]))
        if self.as_code:
            self.lines[self.start : self.end] = raw.splitlines()
            return
        raw = _strip_docstring_quotes(raw, self.closing_sequence)
        raw = textwrap.dedent(raw).strip("\n")
        cleaned = raw.split("\n") if raw else []
        self.lines[self.start : self.end] = cleaned
        self.end = self.start + len(cleaned)

    def get_cell(self) -> NotebookNode:
        source = "\n".join(self.get_block())
        return new_code_cell(source) if self.as_code else new_markdown_cell(source)


class CodeBlock(Block):
    """Compatibility block for source not claimed by another block."""

    def __init__(self, lines: list[str], other_blocks: list[Block] | None = None) -> None:
        super().__init__(lines)
        self.other_blocks = other_blocks or []

    def first_line(self, index: int) -> bool:
        if not 0 <= index < len(self.lines):
            return False
        return not any(other.first_line(index) for other in self.other_blocks)

    def identifies_block(self) -> None:
        if self.start is None:
            raise ValueError("The beginning of the block has not been identified yet.")
        if self.end is not None:
            raise AssertionError("Should not be called again")
        for index in range(self.start, len(self.lines)):
            if any(other.first_line(index) for other in self.other_blocks):
                self.end = index
                return
        self.end = len(self.lines)

    def clean_block(self) -> None:
        if self.start is None or self.end is None:
            return
        for index in range(self.start, self.end):
            self.lines[index] = self.lines[index].rstrip("\r\n")

    def get_cell(self) -> NotebookNode:
        return new_code_cell("\n".join(self.get_block()))


@dataclass
class _CellSpec:
    kind: str
    source: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _Marker:
    kind: str


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _strip_docstring_quotes(raw: str, default_quote: str = '"""') -> str:
    raw = _normalize_newlines(raw)
    match = re.match(r"(?is)\A(?P<prefix>[rubf]*)(?P<quote>'''|\"\"\"|'|\")", raw)
    quote = match.group("quote") if match else default_quote
    if match:
        raw = raw[match.end() :]
    raw = raw.rstrip("\n")
    if raw.endswith(quote):
        raw = raw[: -len(quote)]
    return raw


def _tokenize_source(source: str) -> tuple[list[Optional[str]], set[int]]:
    lines = source.splitlines(keepends=True)
    comments: list[Optional[str]] = [None] * len(lines)
    string_lines: set[int] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            start_row, start_col = token.start
            end_row, _ = token.end
            if token.type == tokenize.STRING:
                string_lines.update(range(start_row, end_row + 1))
            elif token.type == tokenize.COMMENT and 1 <= start_row <= len(lines):
                prefix = lines[start_row - 1][:start_col]
                if not prefix.strip():
                    comments[start_row - 1] = token.string
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass
    return comments, string_lines


def _marker_from_comment(comment: str, line_number: int) -> Optional[_Marker]:
    body = comment[1:].strip()
    if not body.startswith("%%"):
        return None
    match = re.fullmatch(r"%%(?:\s+\[(?P<kind>[^\]]+)\])?\s*", body)
    if not match:
        raise UnsupportedCellMarkerError(
            f"Unsupported cell marker on line {line_number}: {comment.strip()}"
        )
    requested = (match.group("kind") or "code").strip().lower()
    if requested == "markdown":
        return _Marker("markdown")
    if requested in {"", "code"}:
        return _Marker("code")
    raise UnsupportedCellMarkerError(
        f"Unsupported cell marker type [{requested}] on line {line_number}"
    )


def _find_markers(comments: Sequence[Optional[str]], string_lines: set[int]) -> dict[int, _Marker]:
    markers: dict[int, _Marker] = {}
    for index, comment in enumerate(comments):
        line_number = index + 1
        if comment is not None and line_number not in string_lines:
            marker = _marker_from_comment(comment, line_number)
            if marker is not None:
                markers[index] = marker
    return markers


def _module_docstring_span(tree: ast.Module) -> Optional[tuple[int, int]]:
    if not tree.body:
        return None
    first = tree.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
        return None
    if not isinstance(first.value.value, str):
        return None
    return first.lineno, getattr(first, "end_lineno", first.lineno)


def _top_level_statement_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    return [
        (statement.lineno, getattr(statement, "end_lineno", statement.lineno))
        for statement in tree.body
    ]


def _module_docstring_text(source: str, span: tuple[int, int]) -> str:
    lines = source.splitlines(keepends=True)
    raw = "".join(lines[span[0] - 1 : span[1]])
    return textwrap.dedent(_strip_docstring_quotes(raw)).strip("\n")


def _comment_markdown_line(line: str) -> str:
    stripped = line.rstrip("\r\n").lstrip()
    if stripped.startswith("#"):
        content = stripped[1:]
        return content[1:] if content.startswith(" ") else content
    return line.rstrip("\r\n")


def _append_spec(specs: list[_CellSpec], kind: str, source: str, start: int, end: int) -> None:
    source = _normalize_newlines(source)
    if source.strip():
        specs.append(_CellSpec(kind, source, start, end))


def _parse_default_range(
    lines: Sequence[str],
    start: int,
    end: int,
    comments: Sequence[Optional[str]],
    docstring_span: Optional[tuple[int, int]],
    statement_ranges: Sequence[tuple[int, int]],
    docstring_as_code: bool,
) -> list[_CellSpec]:
    specs: list[_CellSpec] = []
    kind: str | None = None
    current: list[tuple[str, bool]] = []
    current_start = start + 1
    current_end = start

    def flush() -> None:
        nonlocal kind, current
        if kind == "markdown":
            source = "\n".join(
                _comment_markdown_line(line) if is_comment else line.rstrip("\r\n")
                for line, is_comment in current
            )
        elif kind == "code":
            source = _normalize_newlines("".join(line for line, _ in current))
        else:
            source = ""
        if kind is not None:
            _append_spec(specs, kind, source, current_start, current_end)
        kind = None
        current = []

    index = start
    while index < end:
        line_number = index + 1
        is_docstring = docstring_span and docstring_span[0] <= line_number <= docstring_span[1]
        if is_docstring:
            new_kind = "code" if docstring_as_code else "markdown"
            if kind is not None and kind != new_kind:
                flush()
            if kind is None:
                kind = new_kind
                current_start = line_number
            current.append((lines[index], False))
            current_end = line_number
            if new_kind == "markdown" and line_number == docstring_span[0]:
                cleaned = _module_docstring_text(
                    "".join(lines[index : docstring_span[1]]),
                    (1, docstring_span[1] - docstring_span[0] + 1),
                )
                current.pop()
                current.extend((line, False) for line in cleaned.split("\n"))
                current_end = docstring_span[1]
            index = docstring_span[1]
            continue

        comment = comments[index] if index < len(comments) else None
        inside_statement = any(
            statement_start <= line_number <= statement_end
            for statement_start, statement_end in statement_ranges
        )
        is_markdown_comment = comment is not None and not inside_statement
        if is_markdown_comment:
            new_kind = "markdown"
        elif not lines[index].strip():
            if kind is None:
                kind = "code"
                current_start = line_number
            current.append((lines[index], False))
            current_end = line_number
            index += 1
            continue
        else:
            new_kind = "code"

        if kind is not None and kind != new_kind:
            flush()
        if kind is None:
            kind = new_kind
            current_start = line_number
        current.append((lines[index], is_markdown_comment))
        current_end = line_number
        index += 1
    flush()
    return specs


def _parse_explicit_range(
    lines: Sequence[str],
    start: int,
    end: int,
    kind: str,
    comments: Sequence[Optional[str]],
) -> list[_CellSpec]:
    if start >= end:
        return []
    if kind == "markdown":
        source = "\n".join(
            _comment_markdown_line(lines[index])
            if comments[index] is not None
            else lines[index].rstrip("\r\n")
            for index in range(start, end)
        )
    else:
        source = _normalize_newlines("".join(lines[start:end]))
    specs: list[_CellSpec] = []
    _append_spec(specs, kind, source, start + 1, end)
    return specs


def _parse_specs(source: str, *, docstring_as_code: bool = False) -> list[_CellSpec]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    comments, string_lines = _tokenize_source(source)
    markers = _find_markers(comments, string_lines)
    docstring_span = _module_docstring_span(tree)
    statement_ranges = _top_level_statement_ranges(tree)
    if not markers:
        return _parse_default_range(
            lines, 0, len(lines), comments, docstring_span, statement_ranges, docstring_as_code
        )

    specs: list[_CellSpec] = []
    previous = 0
    for marker_index, marker in sorted(markers.items()):
        if previous < marker_index:
            specs.extend(
                _parse_default_range(
                    lines,
                    previous,
                    marker_index,
                    comments,
                    docstring_span,
                    statement_ranges,
                    docstring_as_code,
                )
            )
        next_marker = marker_index + 1
        while next_marker < len(lines) and next_marker not in markers:
            next_marker += 1
        specs.extend(_parse_explicit_range(lines, marker_index + 1, next_marker, marker.kind, comments))
        previous = next_marker
    if previous < len(lines):
        specs.extend(
            _parse_default_range(
                lines, previous, len(lines), comments, docstring_span, statement_ranges, docstring_as_code
            )
        )
    return specs


def source_to_notebook(
    source: str,
    *,
    source_file: str | None = None,
    docstring_as_code: bool = False,
    include_source_metadata: bool = True,
) -> NotebookNode:
    """Convert valid Python source text to a notebook.

    Module-level docstrings become Markdown by default. Function and class
    docstrings remain code. Explicit ``# %%`` markers are removed and control
    cell boundaries. Code cells preserve their source except for line-ending
    normalization.
    """

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    notebook = new_notebook()
    for spec in _parse_specs(source, docstring_as_code=docstring_as_code):
        cell = new_markdown_cell(spec.source) if spec.kind == "markdown" else new_code_cell(spec.source)
        if include_source_metadata:
            if source_file is not None:
                cell.metadata["source_file"] = str(source_file)
            cell.metadata["source_start_line"] = spec.start_line
            cell.metadata["source_end_line"] = spec.end_line
        notebook.cells.append(cell)
    if source_file is not None and include_source_metadata:
        notebook.metadata["source_file"] = str(source_file)
    return notebook


def convert_source(source: str, **kwargs: object) -> NotebookNode:
    """Alias for :func:`source_to_notebook`."""

    return source_to_notebook(source, **kwargs)  # type: ignore[arg-type]


def convert_file(
    input_path: str,
    output_path: str | None = None,
    *,
    docstring_as_code: bool = False,
    include_source_metadata: bool = True,
) -> str:
    """Convert one Python file and return the generated notebook path."""

    output = output_path or os.path.splitext(input_path)[0] + ".ipynb"
    with tokenize.open(input_path) as source_file:
        source = source_file.read()
    notebook = source_to_notebook(
        source,
        source_file=input_path,
        docstring_as_code=docstring_as_code,
        include_source_metadata=include_source_metadata,
    )
    with open(output, "w", encoding="utf-8") as file:
        nbformat.write(notebook, file)
    return output


def check_source(
    source: str,
    *,
    source_file: str | None = None,
    docstring_as_code: bool = False,
) -> ConversionReport:
    """Validate Python, generated cells, and notebook structure."""

    notebook = source_to_notebook(
        source, source_file=source_file, docstring_as_code=docstring_as_code
    )
    for cell in notebook.cells:
        if not cell.source.strip():
            raise ConversionError("empty cell generated")
        if cell.cell_type == "code":
            try:
                compile(cell.source, source_file or "<script>", "exec")
            except SyntaxError as error:
                line = cell.metadata.get("source_start_line", "?")
                raise ConversionError(
                    f"code cell beginning at source line {line} is invalid: {error}"
                ) from error
    nbformat.validate(notebook)
    return ConversionReport(
        markdown_cells=sum(cell.cell_type == "markdown" for cell in notebook.cells),
        code_cells=sum(cell.cell_type == "code" for cell in notebook.cells),
    )


def extract_next_block(lines: list[str], docstring_as_code: bool) -> Iterable[NotebookNode]:
    """Yield converted cells, retaining the original public API."""

    notebook = source_to_notebook("".join(lines), docstring_as_code=docstring_as_code)
    yield from notebook.cells


class SplitTuple(NamedTuple):
    separator: str
    before_file_extension: str
    after_file_extension: str


def split_string(original_line: str, separator: str) -> tuple[str, str]:
    """Split one source line into before/after separator versions."""

    parts = original_line.split(separator)
    before = parts[0].rstrip()
    after = parts[1] if len(parts) > 1 else before
    return before, after


def preprocess(lines: list[str], separator: str) -> tuple[list[str], list[str]]:
    split_lines = [split_string(line, separator) for line in lines]
    return [part[0] for part in split_lines], [part[1] for part in split_lines]


def generate_notebook(lines: list[str], filename: str, docstring_as_code: bool = False) -> None:
    """Write a notebook from source lines, retaining the legacy API."""

    notebook = source_to_notebook(
        "".join(lines), source_file=filename, docstring_as_code=docstring_as_code
    )
    with open(filename, "w", encoding="utf-8") as file:
        nbformat.write(notebook, file)
    print(f"File {filename} created.")


def script_to_notebook(
    input_script: str, splitting: SplitTuple | None, docstring_as_code: bool = False
) -> None:
    """Convert a script, optionally retaining the legacy two-output split."""

    with tokenize.open(input_script) as source_file:
        lines = source_file.readlines()
    base_name = os.path.splitext(input_script)[0]
    if splitting:
        before_lines, after_lines = preprocess(lines, splitting.separator)
        generate_notebook(
            before_lines,
            f"{base_name}_{splitting.before_file_extension}.ipynb",
            docstring_as_code,
        )
        generate_notebook(
            after_lines,
            f"{base_name}_{splitting.after_file_extension}.ipynb",
            docstring_as_code,
        )
        return
    convert_file(input_script, docstring_as_code=docstring_as_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--docstring-as-code", action="store_true")
    parser.add_argument("--no-source-metadata", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with tokenize.open(args.script) as source_file:
            source = source_file.read()
        report = check_source(
            source, source_file=args.script, docstring_as_code=args.docstring_as_code
        )
        output = None
        if not args.check:
            output = convert_file(
                args.script,
                args.output,
                docstring_as_code=args.docstring_as_code,
                include_source_metadata=not args.no_source_metadata,
            )
        if args.verbose or args.check:
            print(f"Input: {args.script}")
            print(f"Markdown cells: {report.markdown_cells}")
            print(f"Code cells: {report.code_cells}")
            print(f"Warnings: {len(report.warnings)}")
            if output:
                print(f"Output: {output}")
        return 0
    except (OSError, SyntaxError, ConversionError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
