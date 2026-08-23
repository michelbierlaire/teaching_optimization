# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-08-23

### Added

- Added `teaching_optimization.to_jupyter` for converting teaching-oriented
  Python scripts into Jupyter notebooks.
- Added explicit `# %%` and `# %% [markdown]` cell markers.
- Added support for top-level comment blocks and module docstrings as Markdown
  cells.
- Added validation for supported cell markers, non-empty cells, notebook
  structure, and generated code-cell syntax.
- Added `labs/generate_notebooks.py` for converting multiple files while
  reporting file-level failures.
- Added source-file and source-line metadata to generated notebook cells.
- Added tests and the
  [script conversion guide](SCRIPT_CONVERSION_GUIDE.md).

### Changed

- Blank physical lines are retained within the current cell instead of being
  interpreted as cell boundaries.
- Comments inside functions, classes, and other compound statements remain
  executable Python comments.

## [0.0.2] - 2024-10-25

- Previous published package release.

[0.1.1]: https://github.com/michelbierlaire/teaching_optimization/releases/tag/v0.1.1
[0.0.2]: https://github.com/michelbierlaire/teaching_optimization/releases/tag/v0.0.2
