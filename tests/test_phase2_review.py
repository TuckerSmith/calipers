"""Regression cases from the Phase 2 adversarial review (2026-09-15): gaming the verifier and the lint."""

from __future__ import annotations

import textwrap

import pytest
import yaml
from build123d import Box, Cylinder, Location, export_step

from calipers import load
from calipers.contracts import verify
from calipers.provenance import check_sources, lint_source
from calipers.sandbox import run_code
from calipers.spec import SpecError, normalize


def _plate_with_blind_hole(from_top: bool):
    body = Box(60, 40, 10)
    z = 5 - 3 if from_top else -5 + 3  # hole of depth 6 from the top (z ∈ [-1, 5]) or from the bottom
    return body - Cylinder(radius=2.5, height=6).moved(Location((20, 0, z)))


def _spec(**over):
    base = {
        "envelope": {"extents": {"x": [60, 0.1], "y": [40, 0.1], "z": [10, 0.1]}},
        "features": [{"id": "seat", "kind": "blind_hole", "diameter": [5, 0.05], "depth": [6, 0.1], "axis": "z", "positions": [[20, 0]], "entry": "+"}],
        "planes": [{"id": "top", "normal": "z", "offset": [5, 0.05]}],
    }
    base.update(over)
    return normalize(base)


def test_blind_hole_from_the_wrong_face_fails(tmp_path):
    for from_top in (True, False):
        p = tmp_path / f"bh_{from_top}.step"
        export_step(_plate_with_blind_hole(from_top), str(p))
        res = verify(load(p), _spec())
        entry = next(c for c in res.checks if c.id == "seat.0.entry")
        assert entry.passed is from_top, res.to_text()


def test_3d_positions_match_anywhere_on_the_axis_segment(tmp_path):
    p = tmp_path / "bh.step"
    export_step(_plate_with_blind_hole(True), str(p))
    for z in (5.0, 2.0, -1.0):  # mouth, middle, bottom of the hole
        s = _spec(features=[{"id": "seat", "kind": "blind_hole", "diameter": [5, 0.05], "axis": "z", "positions": [[20, 0, z]]}])
        res = verify(load(p), s)
        assert next(c for c in res.checks if c.id == "seat.0.position").passed, (z, res.to_text())
    s = _spec(features=[{"id": "seat", "kind": "blind_hole", "diameter": [5, 0.05], "axis": "z", "positions": [[20, 0, 7.0]]}])
    assert not next(c for c in verify(load(p), s).checks if c.id == "seat.0.position").passed


def test_small_asymmetric_notch_breaks_exact_symmetry(tmp_path):
    body = Box(60, 40, 10) - Cylinder(radius=2.5, height=20).moved(Location((20, 0, 0))) - Cylinder(radius=2.5, height=20).moved(Location((-20, 0, 0)))
    notched = body - Box(3, 3, 1).moved(Location((28.5, 18.5, 4.5)))  # 9 mm³ corner notch
    for shape, expect in ((body, True), (notched, False)):
        p = tmp_path / f"sym_{expect}.step"
        export_step(shape, str(p))
        res = verify(load(p), normalize({"symmetry": ["x", "y"]}))
        got = {c.id: c.passed for c in res.checks if c.id.startswith("symmetry")}
        assert got == {"symmetry.x": expect, "symmetry.y": expect}, res.to_text()


def test_lint_evasions_are_caught():
    evasions = [
        'float("60")', 'int("7")', 'eval("60")', "round(40.0)", "abs(-5.5)", "max(7.3, 0)", "min(9.9)",
        "abs(60j)", "2*2*2*2*2*2", "count=60.5", "n=7.25",
    ]
    for expr in evasions:
        code = f"PARAMS = {{}}\nx = f({expr})" if "=" in expr else f"PARAMS = {{}}\nx = {expr}"
        res = lint_source(code)
        assert not res.ok, f"evasion not caught: {expr}\n{res.to_text()}"
    assert not lint_source("def mk():\n    return {}\nPARAMS = mk()").ok
    assert not lint_source('PARAMS = {"a": (5.0, "spec:a.b")}\nPARAMS = {}').ok
    assert not lint_source('PARAMS = {"a": (5.0, "spec:")}').ok  # empty source body
    assert lint_source("PARAMS = {}\nfor i in range(12):\n    y = 360 / 2\n    z = [0, 1][1]\n    w = f(count=6)").ok


def test_spec_sources_are_checked_against_the_spec():
    spec = _spec()
    res = lint_source('PARAMS = {"d": (5.0, "spec:seat.diameter"), "w": (60.0, "spec:envelope.extents.x"), "bad": (1.0, "spec:no.such.thing"), "off": (7.0, "spec:seat.depth")}')
    problems = check_sources(res, spec)
    assert len(problems) == 2 and any("does not exist" in p for p in problems) and any("7.0 but spec:seat.depth is 6.0" in p for p in problems)


def test_sandbox_rejects_non_solids_and_reports_footer_errors(tmp_path):
    r = run_code("from build123d import *\nPARAMS = {}\nresult = None", workdir=tmp_path / "a")
    assert not r.ok and r.error["type"] == "ContractError" and "NoneType" in r.error["message"]
    sk = textwrap.dedent(
        """
        from build123d import *
        PARAMS = {}
        with BuildSketch() as sk:
            Rectangle(10, 10)
        result = sk.sketch
        """
    )
    r = run_code(sk, workdir=tmp_path / "b")
    assert not r.ok and "no solid" in r.error["message"]
    r = run_code("import sys\nPARAMS = {}\nsys.exit(0)", workdir=tmp_path / "c")
    assert not r.ok and r.error["type"] == "ScriptExited"
    r = run_code("PARAMS = {}\nraise ValueError('first line\\nsecond line')", workdir=tmp_path / "d")
    assert not r.ok and r.error["type"] == "ValueError" and "second line" in r.error["message"] and r.error["line"] == 2


def test_sandbox_cadquery_interop(tmp_path):
    pytest.importorskip("cadquery")
    r = run_code("import cadquery as cq\nPARAMS = {}\nresult = cq.Workplane('XY').box(60, 40, 10)", workdir=tmp_path)
    assert r.ok and "60.000 × 40.000 × 10.000" in r.report_text


def test_plane_offset_hint_and_solid_validity(tmp_path):
    p = tmp_path / "plate.step"
    export_step(Box(60, 40, 10), str(p))
    s = normalize({"planes": [{"id": "bottom", "normal": "z", "offset": [-5, 0.05]}]})
    chk = next(c for c in verify(load(p), s).checks if c.id == "plane.bottom.offset")
    assert not chk.passed and "opposite normal" in chk.note or "sign convention" in chk.note


def test_hungarian_assignment_avoids_greedy_false_fail(tmp_path):
    part = Box(60, 40, 10) - Cylinder(radius=2.5, height=20).moved(Location((-2, 0, 0))) - Cylinder(radius=2.5, height=20).moved(Location((2, 0, 0)))
    p = tmp_path / "greedy.step"
    export_step(part, str(p))
    s = normalize({"features": [{"id": "h", "kind": "through_hole", "diameter": [5, 0.05], "axis": "z", "positions": [[1, 0], [4, 0]], "pos_tol": 3}]})
    res = verify(load(p), s)
    assert all(c.passed for c in res.checks if c.id.endswith(".position")), res.to_text()


def test_spec_validation_catches_bad_refs_and_mixed_positions():
    with pytest.raises(SpecError):
        normalize({"features": [{"id": "h", "kind": "through_hole", "positions": [[1, 0, 5], [2, 0]]}]})
    with pytest.raises(SpecError, match="unknown feature id"):
        normalize({"features": [{"id": "h", "kind": "through_hole", "positions": [[0, 0]]}], "relations": [{"type": "coaxial", "between": ["h.0", "nope.0"]}]})
    with pytest.raises(SpecError, match="out of range"):
        normalize({"features": [{"id": "h", "kind": "through_hole", "positions": [[0, 0]]}], "relations": [{"type": "coaxial", "between": ["h.0", "h.7"]}]})
    with pytest.raises(SpecError, match=r"relations\[0\].value"):
        normalize({"features": [{"id": "h", "kind": "through_hole", "count": 2}], "relations": [{"type": "distance", "between": ["h.0", "h.1"]}]})


def test_spec_yaml_roundtrip_still_verifies(bracket_files):
    doc = yaml.safe_load(
        """
        features:
          - {id: holes, kind: through_hole, diameter: [5, 0.05], axis: z, positions: [[-20, 0, 0], [20, 0, 0]]}
          - {id: boss, kind: boss, diameter: [12, 0.05], axis: z, positions: [[0, 0]], entry: "+"}
        """
    )
    res = verify(load(bracket_files["step"]), normalize(doc))
    assert res.passed, res.to_text()
