"""MCP server: the calipers as tools for Claude Desktop / Claude Code / any MCP client.

Run with ``calipers-mcp`` (stdio). Claude Desktop config (claude_desktop_config.json)::

    {"mcpServers": {"calipers": {"command": "calipers-mcp"}}}

Every tool returns the same text digests the CLI prints, so what the model reads in a chat is
exactly what a human sees on the terminal. Paths are local to the machine running the server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:  # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "calipers",
    instructions=(
        "Calipers for AI: measure 3D geometry exactly instead of guessing. Workflow for generating a part: "
        "1) write the requirements as a spec (see spec_schema); 2) write build123d code whose dimensions are all "
        "declared in a PARAMS dict with sources; 3) run_code with the spec — read the contract failures and the "
        "provenance lint, repair, repeat until everything passes; 4) export. To understand an existing model, "
        "call report first, then section/features for detail. To design around an existing part (an enclosure, "
        "a mount): call reference_summary/report on it, add it to the spec as a reference with keep_out/fit "
        "contracts (see spec_schema), or start from generate(kind=...) and refine. run_candidates ranks several "
        "attempts; diff shows what changed between two revisions."
    ),
)


def _load(path: str):
    from calipers.model import Model

    return Model.load(path)


@mcp.tool()
def report(path: str, sections: Optional[list[str]] = None, symmetry: bool = True, wall_thickness: bool = False) -> str:
    """Geometry report of an STL/OBJ/3MF or STEP file: envelope, volume, planes, cylindrical features
    (holes/bosses with diameters, axes, positions), optional cross-sections (e.g. ["z=12", "x=0"]),
    mirror symmetry and sampled wall thickness. Numbers are exact for STEP, fitted (with residuals) for meshes."""
    from calipers.report import build_report

    rep = build_report(_load(path), sections=sections or [], symmetry=symmetry, wall=wall_thickness)
    return rep.to_text()


@mcp.tool()
def report_json(path: str, sections: Optional[list[str]] = None) -> str:
    """Full structured geometry report as JSON (schema calipers.report/0.1) — use when you need every feature."""
    from calipers.report import build_report

    return build_report(_load(path), sections=sections or [], symmetry=False).to_json()


@mcp.tool()
def section(path: str, plane: str) -> str:
    """Cross-section loops of a model by an axis-aligned plane such as "z=12", "x=-3.5" or "y=0":
    outer/inner loops with areas, circle fits (diameter, centre) and rectangle fits, in world coordinates."""
    from calipers import measure

    model = _load(path)
    origin, normal = measure.parse_plane(plane)
    sec = measure.section(model, origin, normal)
    if model.is_exact:
        ex = measure.section_exact(model, origin, normal)
        if ex:
            sec["exact"] = ex
    return json.dumps(sec, indent=1)


@mcp.tool()
def measure_distance(path: str, points: list[list[float]]) -> str:
    """Closest-surface distance (and signed distance when the mesh is closed) from given 3D points."""
    from calipers import measure

    return json.dumps(measure.closest_surface_points(_load(path), points), indent=1)


@mcp.tool()
def render(path: str, out_dir: str, views: Optional[list[str]] = None) -> str:
    """Render shaded orthographic PNG views (iso, front, top, right, back, bottom, left) and, for STEP,
    hidden-line drawings. Returns the file paths; open them to look at the part."""
    from calipers.render import render_views

    paths = render_views(_load(path), out_dir, views=views or ("iso", "front", "top", "right"))
    return "\n".join(paths)


@mcp.tool()
def spec_schema() -> str:
    """The part-spec (requirements) schema with an example. Write specs in YAML or JSON and pass them to verify/run_code."""
    from calipers import spec as spec_mod

    return spec_mod.__doc__ or ""


@mcp.tool()
def verify(path: str, spec_path: str) -> str:
    """Check a model file against a spec: every requirement becomes a pass/fail line with required vs
    measured and the deviation. Use the failures to repair the design."""
    from calipers.contracts import verify as _verify
    from calipers.spec import load_spec

    return _verify(_load(path), load_spec(spec_path)).to_text()


@mcp.tool()
def lint_provenance(code: str) -> str:
    """'No naked numbers' check for build123d code: every dimension must be declared in a module-level
    PARAMS = {name: (value, "source")} dict with sources spec:/measured:/derived:/standard:/assumption:.
    Reports every literal that is not, plus the assumptions a reviewer should look at."""
    from calipers.provenance import lint_source

    return lint_source(code).to_text()


@mcp.tool()
def run_code(code: str, spec_path: Optional[str] = None, workdir: Optional[str] = None, render_views: bool = False) -> str:
    """Execute build123d code (it must assign the finished shape to `result`), export STEP/STL, then
    return: execution errors (with the failing line), the provenance lint, the geometry report and —
    when a spec is given — the contract check. Iterate until the contract passes and the lint is clean."""
    from calipers.sandbox import run_code as _run
    from calipers.spec import load_spec

    spec = load_spec(spec_path) if spec_path else None
    res = _run(code, workdir=workdir, spec=spec, render=render_views)
    txt = res.to_text()
    if res.renders:
        txt += "\n\nRenders:\n" + "\n".join(res.renders)
    return txt


@mcp.tool()
def run_candidates(codes: list[str], spec_path: Optional[str] = None, workdir: Optional[str] = None, workers: int = 4) -> str:
    """Best-of-N: run several candidate build123d scripts in parallel, verify each against the spec and
    rank them (fewest failed contracts, smallest deviations). Returns the ranking and the best
    candidate's full output. Generate 3–5 different attempts, then repair the best one."""
    from calipers.sandbox import candidates_text
    from calipers.sandbox import run_candidates as _rc
    from calipers.spec import load_spec

    if not codes:
        return "no candidates given: pass at least one build123d script"
    ranked = _rc(codes, spec=load_spec(spec_path) if spec_path else None, workdir=workdir, workers=workers)
    return candidates_text(ranked) + "\n\n" + ranked[0][1].to_text()


@mcp.tool()
def diff(path_a: str, path_b: str) -> str:
    """What changed between two model files (two revisions of a part): extents, volume, planes,
    cylindrical features and slots matched and compared; kernel-exact material added/removed for STEP."""
    from calipers.diff import diff_models, diff_text

    return diff_text(diff_models(_load(path_a), _load(path_b)))


@mcp.tool()
def reference_summary(path: str, rotate: Optional[list[float]] = None, translate: Optional[list[float]] = None) -> str:
    """Envelope of a reference model after optional placement (rotate about x,y,z in degrees, then
    translate): bbox_min/bbox_max/extents/center — the numbers to cite as measured:<id>.bbox_max.z etc."""
    from calipers.reference import load_reference
    from calipers.reference import reference_summary as _rs

    rf = {"id": Path(path).stem, "path": path}
    if rotate or translate:
        rf["place"] = {"rotate": rotate or [0, 0, 0], "translate": translate or [0, 0, 0]}
    return json.dumps(_rs(load_reference(rf)), indent=1)


@mcp.tool()
def generate(
    kind: str,
    reference_path: str,
    out_dir: str,
    ref_id: Optional[str] = None,
    wall: float = 2.0,
    clearance: float = 0.5,
    floor: Optional[float] = None,
    open_face: str = "+z",
    corner_radius: float = 0.0,
    plate_t: float = 3.0,
    standoff_h: float = 5.0,
    standoff_wall: float = 2.0,
    screw_d: Optional[float] = None,
    margin: float = 3.0,
    hole_diameter: Optional[float] = None,
    rotate: Optional[list[float]] = None,
    translate: Optional[list[float]] = None,
    run: bool = True,
) -> str:
    """Generate a sourced build123d script and a spec with fit contracts for a part designed around a
    reference model — kind "enclosure" (open-faced box: wall, clearance, floor, open_face, corner_radius)
    or "mount" (plate with standoffs at the reference's through holes: plate_t, standoff_h, standoff_wall,
    screw_d, margin, hole_diameter). Writes part.py and spec.yaml to out_dir; with run=True also
    executes and verifies them. Edit part.py afterwards and re-run run_code with the spec."""
    from calipers import generators

    place = {"rotate": rotate or [0, 0, 0], "translate": translate or [0, 0, 0]} if (rotate or translate) else None
    if kind == "enclosure":
        res = generators.enclosure(reference_path, ref_id, wall=wall, clearance=clearance, floor=floor, open_face=open_face, corner_radius=corner_radius, place=place, out_dir=out_dir)
    elif kind == "mount":
        res = generators.mount(reference_path, ref_id, plate_t=plate_t, standoff_h=standoff_h, standoff_wall=standoff_wall, screw_d=screw_d, margin=margin, hole_diameter=hole_diameter, place=place, out_dir=out_dir)
    else:
        return "kind must be 'enclosure' or 'mount'"
    txt = f"wrote {res['paths']['code']} and {res['paths']['spec']}\n\n{res['code']}\n\n---- spec.yaml ----\n{res['spec']}"
    if run:
        from calipers.sandbox import run_code as _run
        from calipers.spec import load_spec

        rr = _run(res["code"], workdir=str(Path(out_dir) / "run"), spec=load_spec(res["paths"]["spec"]))
        txt += "\n\n" + rr.to_text()
    return txt


@mcp.tool()
def export(path: str, out: str) -> str:
    """Convert a model file to .step/.stl/.3mf/.obj (mesh → STEP is refused: that would be reconstruction)."""
    from calipers.export import export as _export

    return _export(_load(path), out)


@mcp.tool()
def api_help(name: str) -> str:
    """Signature and docstring of a build123d name (e.g. "Hole", "fillet", "BuildSketch"), or a search
    for names containing a fragment. Use it before guessing an API."""
    from calipers.sandbox import api_help as _help

    return _help(name)


@mcp.tool()
def redteam(n: int = 10, seed: int = 0) -> str:
    """Run the random-part red-team harness for n seeds starting at seed; returns the scoreboard."""
    from calipers.redteam import run, to_markdown

    return to_markdown(run(list(range(seed, seed + n)), verbose=False))


def main() -> None:  # pragma: no cover - entry point
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["mcp", "main", "Path"]
