"""The package version is declared twice; they must agree (0.2.0 shipped with __init__ at 0.1.0)."""
import re
from pathlib import Path

import sprintseq


def test_init_matches_pyproject():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == sprintseq.__version__
