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


@app.command()
def verify(file: Path = typer.Argument(..., exists=True), spec: Path = typer.Argument(..., exists=True), json_out: Optional[Path] = typer.Option(None, "--json")):
    """Check a model against a spec (YAML/JSON): pass/fail per requirement with measured values and deviations."""
    from calipers.contracts import verify as _verify
    from calipers.spec import load_spec

    res = _verify(load(file), load_spec(spec))
    if json_out:
        json_out.write_text(json.dumps(res.to_dict(), indent=2))
    typer.echo(res.to_text())
    raise typer.Exit(code=0 if res.passed else 1)


@app.command()
def lint(code: Path = typer.Argument(..., exists=True)):
    """Dimension-provenance lint ('no naked numbers') for a build123d script."""
    from calipers.provenance import lint_file

    res = lint_file(str(code))
    typer.echo(res.to_text())
    raise typer.Exit(code=0 if res.ok else 1)


@app.command()
def run(
    code: Path = typer.Argument(..., exists=True, help="build123d script that assigns the shape to `result`"),
    spec: Optional[Path] = typer.Option(None, "--spec", "-s", exists=True),
    out_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Where result.step/.stl land (temp dir by default)"),
    render: bool = typer.Option(False, help="Also render iso/front/top views"),
    section: list[str] = typer.Option([], "--section", help="Sections to include in the report, e.g. z=0"),
    json_out: Optional[Path] = typer.Option(None, "--json"),
):
    """Execute a build123d script, then report, lint and (with --spec) verify what it produced."""
    from calipers.sandbox import run_code
    from calipers.spec import load_spec

    res = run_code(code.read_text(encoding="utf-8"), workdir=out_dir, spec=load_spec(spec) if spec else None, render=render, sections=tuple(section))
    if json_out:
        json_out.write_text(json.dumps(res.to_dict(), indent=2, default=str))
    typer.echo(res.to_text())
    ok = res.ok and (res.verify is None or res.verify["passed"]) and (res.lint is None or res.lint.get("ok", True))
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def api(name: str = typer.Argument(..., help="build123d name or a fragment to search")):
    """Signature and docstring of a build123d object, or matching names for a fragment."""
    from calipers.sandbox import api_help

    typer.echo(api_help(name))


@app.command()
def redteam(
    n: int = typer.Option(20, help="number of random parts"),
    seed: int = typer.Option(0, help="first seed"),
    out: Optional[Path] = typer.Option(None, help="scoreboard JSON path (a .md is written next to it)"),
    keep: Optional[Path] = typer.Option(None, help="keep the generated STEP/STL files here"),
):
    """Random-part red-team harness: build parts with known ground truth and score both feature paths."""
    from calipers.redteam import run as _run
    from calipers.redteam import to_markdown, write_scoreboard

    res = _run(list(range(seed, seed + n)), keep_dir=keep)
    if out:
        write_scoreboard(res, out, out.with_suffix(".md"))
    typer.echo(to_markdown(res).splitlines()[2])
    raise typer.Exit(code=0 if res["summary"]["failed_seeds"] == [] else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
