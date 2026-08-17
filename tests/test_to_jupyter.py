from __future__ import annotations

import nbformat
import pytest

from teaching_optimization.to_jupyter import (
    ConversionError,
    DocstringBlock,
    UnsupportedCellMarkerError,
    check_source,
    source_to_notebook,
)


def cell_sources(source: str, **kwargs: object) -> list[tuple[str, str]]:
    notebook = source_to_notebook(source, **kwargs)
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def test_module_docstring_does_not_double_newline() -> None:
    source = '"""First line.\n\nSecond line.\n"""\n'
    assert cell_sources(source) == [("markdown", "First line.\n\nSecond line.")]


def test_comment_block_and_blank_comment_lines_are_one_markdown_cell() -> None:
    source = "# Explanation of the model.\n#\n# The next expression defines the likelihood.\n\nloglike = ...\n"
    assert cell_sources(source) == [
        ("markdown", "Explanation of the model.\n\nThe next expression defines the likelihood.\n"),
        ("code", "loglike = ...\n"),
    ]


def test_explicit_markers_control_boundaries_and_are_removed() -> None:
    source = "# %% [markdown]\n# This is a Markdown cell.\n\n# %%\nx = 1\n"
    assert cell_sources(source) == [
        ("markdown", "This is a Markdown cell.\n"),
        ("code", "x = 1\n"),
    ]


def test_marker_inside_string_literal_is_not_a_marker() -> None:
    source = 'value = "# %% [markdown]"\n'
    assert cell_sources(source) == [("code", source)]


def test_module_docstring_is_markdown_but_function_docstring_is_code() -> None:
    source = '"""Module docs."""\n\ndef function():\n    """Function docs."""\n    return 1\n'
    result = cell_sources(source)
    assert result[0] == ("markdown", "Module docs.")
    assert '"""Function docs."""' in result[1][1]


def test_docstring_as_code_preserves_quotes() -> None:
    source = '"""Module docs."""\nvalue = 1\n'
    assert cell_sources(source, docstring_as_code=True)[0] == (
        "code",
        '"""Module docs."""\n',
    )


@pytest.mark.parametrize(
    "source",
    [
        "",
        "# only comments\n#\n",
        "value = 1\n",
        '"""one line"""\n',
        "# final comment\n",
        "value = 1\n# final comment\n",
        "# Windows\r\nvalue = 1\r\n",
    ],
)
def test_end_of_file_cases(source: str) -> None:
    notebook = source_to_notebook(source)
    assert all(cell.source.strip() for cell in notebook.cells)
    check_source(source)


def test_invalid_python_and_unsupported_markers_are_rejected() -> None:
    with pytest.raises(SyntaxError):
        check_source("if:\n")
    with pytest.raises(UnsupportedCellMarkerError):
        source_to_notebook("# %% [raw]\ntext\n")


def test_source_metadata_and_notebook_validation() -> None:
    notebook = source_to_notebook("# docs\nvalue = 1\n", source_file="examples/example.py")
    nbformat.validate(notebook)
    assert notebook.cells[0].metadata.source_file == "examples/example.py"
    assert notebook.cells[0].metadata.source_start_line == 1
    assert notebook.cells[1].metadata.source_end_line == 2


def test_code_cells_compile_and_execute() -> None:
    source = '"""Explanation."""\nresult = 20 + 22\n'
    report = check_source(source, source_file="example.py")
    assert (report.markdown_cells, report.code_cells) == (1, 1)
    namespace: dict[str, object] = {}
    for cell in source_to_notebook(source).cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "example.py", "exec"), namespace)
    assert namespace["result"] == 42


def test_legacy_docstring_block_normalizes_line_endings() -> None:
    lines = ['"""First\r\n', "\r\n", 'Second\r\n', '"""\r\n']
    block = DocstringBlock(lines)
    block.process_block(0)
    assert block.get_cell().source == "First\n\nSecond"
