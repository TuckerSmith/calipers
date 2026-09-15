"""Every example must pass its own spec end to end through the CLI — the same path a model uses."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from calipers.cli import app

EXAMPLES = sorted(p for p in (Path(__file__).parent.parent / "examples").iterdir() if (p / "spec.yaml").exists())


@pytest.mark.parametrize("example", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_passes_its_spec(example, tmp_path):
    res = CliRunner().invoke(app, ["run", str(example / "part.py"), "--spec", str(example / "spec.yaml"), "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output[-3000:]
    assert "Contract check: PASS" in res.output and "Provenance lint: PASS" in res.output
