from __future__ import annotations

from pathlib import Path

from labs.generate_notebooks import generate_notebooks, main


def test_generator_continues_after_a_file_failure(tmp_path: Path, capsys) -> None:
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("# Good\nvalue = 1\n", encoding="utf-8")
    bad.write_text("if:\n", encoding="utf-8")

    report = generate_notebooks([bad, good])

    assert report.converted == 1
    assert report.failed == 1
    assert (tmp_path / "good.ipynb").exists()
    assert not (tmp_path / "bad.ipynb").exists()
    assert "Failed:" in capsys.readouterr().err


def test_generator_directory_excludes_dependencies(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("value = 2\n", encoding="utf-8")

    assert main(["--directory", str(tmp_path)]) == 0
    assert (tmp_path / "good.ipynb").exists()
    assert not (ignored / "ignored.ipynb").exists()
