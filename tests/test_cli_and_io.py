from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from calipers import load
from calipers.cli import app
from calipers.export import export
from calipers.render import render_section, render_views

runner = CliRunner()


def test_report_json_schema(bracket_files, tmp_path):
    out = tmp_path / "r.json"
    res = runner.invoke(app, ["report", str(bracket_files["stl"]), "--json", str(out), "--no-symmetry", "-s", "z=0"])
    assert res.exit_code == 0, res.output
    d = json.loads(out.read_text())
    assert d["schema"].startswith("calipers.report/")
    for key in ("source", "summary", "features", "sections", "notes"):
        assert key in d
    assert d["features"]["counts"]["holes"] == 2
    assert "2× through_hole" in res.output


def test_section_cli(bracket_files):
    res = runner.invoke(app, ["section", str(bracket_files["step"]), "-p", "z=0"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.output)
    assert d["n_inner"] == 2 and d["exact"]["loops"]


def test_render_and_export(bracket_files, tmp_path):
    model = load(bracket_files["step"])
    paths = render_views(model, tmp_path / "r", views=["iso", "front"])
    assert all(Path(p).exists() and Path(p).stat().st_size > 1000 for p in paths)
    png, sec = render_section(model, "z=0", tmp_path / "sec.png")
    assert Path(png).exists() and sec["n_inner"] == 2
    for ext in (".stl", ".3mf", ".obj", ".step"):
        p = export(model, tmp_path / f"out{ext}")
        assert Path(p).stat().st_size > 100
    mesh_model = load(tmp_path / "out.stl")
    assert mesh_model.mesh.is_watertight


def test_round_trip_export_preserves_geometry(bracket_files, tmp_path):
    model = load(bracket_files["step"])
    p = export(model, tmp_path / "rt.stl")
    again = load(p)
    assert abs(again.mesh.volume - model.shape.volume) / model.shape.volume < 2e-4
