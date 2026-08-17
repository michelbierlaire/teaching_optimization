# Adapting Python scripts for notebook conversion

Python files are the source of truth. The converter turns them into notebooks
using comments, module docstrings, and optional `# %%` markers.

## Comment blocks

Use consecutive top-level comment lines for teaching prose. A line containing
only `#` creates a blank line in the Markdown cell:

```python
# Explanation of the model.
#
# The next expression defines the likelihood.

loglike = build_likelihood(data)
```

This produces one Markdown cell followed by one code cell. Blank physical
lines are retained in the current cell; they do not create cell boundaries.
Comments inside a function, class, `if`, `try`, or other compound statement
remain Python code comments with that statement.

## Explicit cell markers

Use markers when a boundary must be explicit. The supported forms are:

```python
# %% [markdown]
# # Model setup
#
# We now estimate the parameters.

# %%
import numpy as np
```

The marker lines are removed from the notebook. `# %%` starts a code cell and
`# %% [code]` is also accepted. In a Markdown region, prefix Markdown lines
with `#`. Do not use unsupported forms such as `# %% [raw]`.

Place markers between complete, independently compilable pieces of Python,
not in the middle of a function, loop, parenthesized expression, or
`try`/`except` statement. A `# %%` sequence inside a string literal is not a
marker.

## Docstrings

The first string expression in a module becomes Markdown by default:

```python
"""A short introduction.

This becomes a Markdown cell.
"""
```

Function and class docstrings remain executable Python code. A triple-quoted
string assigned to a variable or used in another expression also remains
code. To keep a module docstring executable, use:

```bash
python -m teaching_optimization.to_jupyter script.py --docstring-as-code
```

## Migrating an existing file

1. Keep explanatory prose as top-level `#` lines.
2. Replace intended blank-line cell separators with `# %%` markers.
3. Put Markdown after `# %% [markdown]` and prefix each line with `#`.
4. Leave function and class docstrings in place.
5. Make every explicit code section independently valid Python.
6. Keep literal marker examples inside strings or reword them when they are
   intended to be displayed rather than used as control syntax.

For example, convert this when three cells are intended:

```python
# %% [markdown]
# Load the data.

# %%
data = load_data()

# %% [markdown]
# Inspect the data.

# %%
data.head()
```

## Validate changes

Check one file without writing a notebook:

```bash
python -m teaching_optimization.to_jupyter examples/example.py --check --verbose
```

The check verifies valid Python, supported markers, non-empty cells, valid
notebook structure, and compilation of every generated code cell. Convert a
directory with:

```bash
python labs/generate_notebooks.py --directory examples --verbose
```

The generator continues after file-level failures and exits nonzero if any
file failed. Generated cells include `source_file`, `source_start_line`, and
`source_end_line` metadata.

### Author checklist

- Blank physical lines are used for readability, not cell boundaries.
- Explicit markers occur only between complete cells.
- Only supported marker forms are used.
- Runtime triple-quoted strings remain code.
- The single-file `--check` command succeeds.
- The generated notebook has the intended cell order and count.
