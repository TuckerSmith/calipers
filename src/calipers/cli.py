"""Command-line interface: ``calipers report|features|section|render|summary|export``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from calipers import __version__
from calipers.model import load

app = typer.Typer(help="Calipers for AI — measure, interrogate and render 3D geometry exactly.", no_args_is_help=True)


def _version(value: bool):
    if value:
        typer.echo(f"calipers {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version.")):
    """Calipers for AI — the model never guesses a dimension; it measures."""


@app.command()
def report(
    file: Path = typer.Argument(..., exists=True, help="STL/OBJ/PLY/3MF or STEP/BREP file"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write the full structured report here"),
    section: list[str] = typer.Option([], "--section", "-s", help="Add a cross-section, e.g. z=12 (repeatable)"),
    symmetry: bool = typer.Option(True, help="Test mirror symmetry (slower on large meshes)"),
    wall: bool = typer.Option(False, help="Sample wall thickness"),
    text: bool = typer.Option(True, help="Print the LLM-facing text digest"),
):
    """Full geometry report: summary, features, optional sections — as text and/or JSON."""
    from calipers.report import build_report

    model = load(file)
    rep = build_report(model, sections=section, symmetry=symmetry, wall=wall)
    if json_out:
        json_out.write_text(rep.to_json())
        typer.echo(f"wrote {json_out}", err=True)
    if text:
        typer.echo(rep.to_text())


@app.command()
def summary(file: Path = typer.Argument(..., exists=True)):
    """Envelope, mass properties, validity and topology counts (JSON)."""
    from calipers import measure

    typer.echo(json.dumps(measure.summary(load(file)), indent=2))


@app.command()
def features(file: Path = typer.Argument(..., exists=True), json_out: Optional[Path] = typer.Option(None, "--json")):
    """Planes and cylindrical features (holes, bosses, pins, fillets) as JSON."""
    from calipers.features import extract_features

    out = json.dumps(extract_features(load(file)), indent=2)
    if json_out:
        json_out.write_text(out)
        typer.echo(f"wrote {json_out}", err=True)
    else:
        typer.echo(out)


@app.command()
def section(
    file: Path = typer.Argument(..., exists=True),
    plane: str = typer.Option(..., "--plane", "-p", help="Plane spec, e.g. z=12, x=-3.5, y=0"),
    png: Optional[Path] = typer.Option(None, help="Also plot the section to this PNG"),
    exact: bool = typer.Option(True, help="For B-reps, add the exact OCCT section loops"),
):
    """Cross-section loops with circle / rectangle fits (JSON)."""
    from calipers import measure
    from calipers.render import render_section

    model = load(file)
    if png:
        _, sec = render_section(model, plane, png)
        typer.echo(f"wrote {png}", err=True)
    else:
        origin, normal = measure.parse_plane(plane)
        sec = measure.section(model, origin, normal)
    if exact and model.is_exact:
        origin, normal = measure.parse_plane(plane)
        ex = measure.section_exact(model, origin, normal)
        if ex:
            sec["exact"] = ex
    typer.echo(json.dumps(sec, indent=2))


@app.command()
def render(
    file: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option(Path("renders"), "--out", "-o"),
    views: str = typer.Option("iso,front,top,right", help="Comma-separated: iso,front,back,top,bottom,right,left"),
    size: int = typer.Option(900, help="Image width in pixels"),
):
    """Shaded orthographic views (PNG) and, for B-reps, hidden-line views (SVG/PNG)."""
    from calipers.render import render_views

    paths = render_views(load(file), out_dir, views=[v.strip() for v in views.split(",") if v.strip()], size_px=size)
    for p in paths:
        typer.echo(p)


@app.command()
def export(file: Path = typer.Argument(..., exists=True), out: Path = typer.Argument(..., help="Target file (.step/.stl/.3mf/.obj/...)")):
    """Convert between formats (B-rep → mesh tessellates; mesh → B-rep is refused)."""
    from calipers.export import export as _export

    typer.echo(_export(load(file), out))


if __name__ == "__main__":  # pragma: no cover
    app()
