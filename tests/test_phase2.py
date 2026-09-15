"""Phase 2: spec → contracts, provenance lint, execution sandbox, MCP server."""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml

from calipers import load
from calipers.contracts import verify
from calipers.provenance import lint_source
from calipers.sandbox import api_help, run_code
from calipers.spec import SpecError, normalize

BRACKET_SPEC = """
part: bracket
envelope: {extents: {x: [60, 0.1], y: [40, 0.1], z: [13, 0.1]}}
features:
  - {id: mount_holes, kind: through_hole, diameter: [5.0, 0.05], axis: z, positions: [[-20, 0], [20, 0]], length: [5, 0.1]}
  - {id: boss, kind: boss, diameter: [12, 0.05], height: [8, 0.1], axis: z, positions: [[0, 0]]}
exact_counts: true
planes:
  - {id: top, normal: z, offset: [2.5, 0.05], min_area: 2000}
relations:
  - {type: distance, between: [mount_holes.0, mount_holes.1], value: [40, 0.1]}
  - {type: coaxial, between: [boss.0, boss.0], tol: 0.01}
symmetry: [x, y]
printability: {min_wall: 1.2, bed: [256, 256, 256]}
"""

GOOD_CODE = textwrap.dedent(
    '''
    from build123d import *
    PARAMS = {
        "L": (60.0, "spec:envelope.extents.x"), "W": (40.0, "spec:envelope.extents.y"), "T": (5.0, "spec:mount_holes.length"),
        "hole_d": (5.0, "spec:mount_holes.diameter"), "hole_x": (20.0, "spec:mount_holes.positions"),
        "boss_d": (12.0, "spec:boss.diameter"), "boss_h": (8.0, "spec:boss.height"),
        "fillet_r": (3.0, "assumption:cosmetic corner radius"),
    }
    P = {k: v[0] for k, v in PARAMS.items()}
    with BuildPart() as bp:
        Box(P["L"], P["W"], P["T"])
        fillet(bp.edges().filter_by(Axis.Z), radius=P["fillet_r"])
        with Locations((P["hole_x"], 0), (-P["hole_x"], 0)):
            Hole(radius=P["hole_d"] / 2)
        with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):
            Circle(P["boss_d"] / 2)
        extrude(amount=P["boss_h"])
    result = bp.part
    '''
)


@pytest.fixture(scope="module")
def spec():
    return normalize(yaml.safe_load(BRACKET_SPEC))


def test_spec_normalises_and_validates(spec):
    assert spec["features"][0]["diameter"] == (5.0, 0.05)
    assert spec["features"][0]["count"] == 2 and spec["features"][0]["axis"] == [0.0, 0.0, 1.0]
    assert spec["relations"][0]["value"] == (40.0, 0.1)
    with pytest.raises(SpecError):
        normalize({"features": [{"kind": "hexagon"}]})
    with pytest.raises(SpecError):
        normalize({"relations": [{"type": "distance", "between": ["a"]}]})


def test_contracts_pass_on_the_reference_bracket(bracket_files, spec):
    res = verify(load(bracket_files["step"]), spec)
    assert res.passed, res.to_text()
    ids = {c.id for c in res.checks}
    assert {"envelope.x", "mount_holes.0.diameter", "boss.0.length", "features.exact_counts", "plane.top.offset", "relation.0.distance", "symmetry.y", "print.min_wall"} <= ids
    # the mesh path must pass the same contract (fitted numbers within the spec tolerances)
    res_m = verify(load(bracket_files["stl"]), spec)
    assert res_m.passed, res_m.to_text()


def test_contracts_report_precise_deviations(bracket_files, spec):
    """Tighten the spec so the reference part fails exactly where expected, with the right deviation."""
    s = json.loads(json.dumps(spec))
    s["features"][1]["diameter"] = (12.5, 0.05)  # boss is really Ø12
    s["envelope"]["extents"]["z"] = (14.0, 0.1)  # really 13
    res = verify(load(bracket_files["step"]), s)
    failed = {c.id: c for c in res.checks if not c.passed}
    assert set(failed) == {"boss.0.diameter", "envelope.z"}, list(failed)
    assert failed["boss.0.diameter"].deviation == pytest.approx(-0.5, abs=1e-6)
    assert failed["envelope.z"].deviation == pytest.approx(-1.0, abs=1e-6)


def test_exact_counts_catches_extra_features(bracket_files, spec):
    s = json.loads(json.dumps(spec))
    s["features"] = s["features"][:1]  # spec forgets the boss → it is an extra feature
    res = verify(load(bracket_files["step"]), s)
    extra = next(c for c in res.checks if c.id == "features.exact_counts")
    assert not extra.passed and "boss" in str(extra.measured)


def test_provenance_lint():
    ok = lint_source(GOOD_CODE)
    assert ok.ok and len(ok.params) == 8 and len(ok.assumptions) == 1
    bad = lint_source(GOOD_CODE.replace('Hole(radius=P["hole_d"] / 2)', "Hole(radius=2.4)").replace('extrude(amount=P["boss_h"])', "extrude(amount=7.5)"))
    assert not bad.ok and sorted(n.value for n in bad.naked) == [2.4, 7.5]
    assert all(n.line > 0 and "Hole" in n.context or "extrude" in n.context for n in bad.naked)
    nosrc = lint_source('PARAMS = {"a": (5.0, "made up")}\nx = 1')
    assert not nosrc.ok and "must start with" in nosrc.problems[0]
    none = lint_source("from build123d import *\nresult = Box(10, 20, 30)")
    assert not none.ok and "no module-level PARAMS" in none.problems[0] and len(none.naked) == 3
    structural = lint_source("PARAMS = {}\nfor i in range(4):\n    x = [1, 2][0] * 0.5\n    y = 360 / 2")
    assert structural.ok, structural.to_text()


def test_sandbox_runs_reports_and_verifies(spec, tmp_path):
    res = run_code(GOOD_CODE, workdir=tmp_path, spec=spec)
    assert res.ok, res.to_text()
    assert res.verify["passed"] and res.lint["ok"]
    assert "2× through_hole Ø5.000" in res.report_text
    assert (tmp_path / "result.step").exists() and (tmp_path / "result.stl").exists()


def test_sandbox_structured_errors(tmp_path):
    res = run_code("from build123d import *\nPARAMS = {}\nresult = Box(10, 10, 10).no_such_method()", workdir=tmp_path / "a")
    assert not res.ok and res.error["type"] == "AttributeError" and res.error["line"] == 3
    res = run_code("from build123d import *\nPARAMS = {}\nshape = Box(10, 10, 10)", workdir=tmp_path / "b")
    assert not res.ok and "result" in res.error["message"]
    res = run_code("import time\nPARAMS = {}\ntime.sleep(5)\nresult = None", workdir=tmp_path / "c", timeout=1.5)
    assert not res.ok and res.error["type"] == "Timeout"


def test_api_help():
    assert "radius" in api_help("Hole") and "build123d.Hole(" in api_help("Hole")
    assert "Cylinder" in api_help("cylind")


@pytest.mark.anyio
async def test_mcp_server_tools(bracket_files, tmp_path):
    from mcp.shared.memory import create_connected_server_and_client_session

    from calipers.mcp_server import mcp

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(BRACKET_SPEC)
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert {"report", "section", "verify", "run_code", "lint_provenance", "api_help", "redteam", "spec_schema"} <= tools
        r = await client.call_tool("report", {"path": str(bracket_files["step"]), "symmetry": False})
        assert "2× through_hole Ø5.000" in r.content[0].text
        v = await client.call_tool("verify", {"path": str(bracket_files["stl"]), "spec_path": str(spec_path)})
        assert "Contract check: PASS" in v.content[0].text
        lint = await client.call_tool("lint_provenance", {"code": "PARAMS = {}\nx = Box(10, 20, 30)"})
        assert "3 naked numbers" in lint.content[0].text
        s = await client.call_tool("spec_schema", {})
        assert "features:" in s.content[0].text
