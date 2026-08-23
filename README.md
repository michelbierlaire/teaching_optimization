# teaching-optimization
[![PyPi](https://img.shields.io/pypi/v/teaching_optimization.svg)](https://pypi.python.org/pypi/teaching_optimization)

Various optimization algorithms used for teaching.

## Script-to-notebook conversion

The project includes a converter for Python teaching scripts. See the
[script conversion guide](https://github.com/michelbierlaire/teaching_optimization/blob/v0.1.0/SCRIPT_CONVERSION_GUIDE.md) for the cell
conventions, explicit markers, docstring behavior, migration examples, and
validation commands.

## Development with uv

Install [uv](https://docs.astral.sh/uv/), then create the locked development
environment and run the tests:

```bash
uv sync --locked
uv run --locked python -m pytest
```

Run the tox matrix when testing the supported Python versions:

```bash
uv run --locked tox
```

Build and validate release artifacts with:

```bash
uv build
uv run --locked twine check dist/*
```
