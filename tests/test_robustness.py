"""Adversarial cases: arbitrary orientation, coarse tessellation, non-cylindrical curved surfaces."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calipers import build_report, load
from calipers.features import extract_features
from tests.conftest import BRACKET, build_bracket


@pytest.fixture(scope="module")
def rotated_bracket(tmp_path_factory):
    """The bracket rotated by an awkward angle about an awkward axis, exported as STEP and STL."""
    from build123d import Axis, Location, export_step, export_stl

    part = build_bracket()
    axis = Axis((0, 0, 0), (0.3, -0.5, 0.8))
    rotated = part.rotate(axis, 37.0).moved(Location((12.3, -4.5, 6.7)))
    d = tmp_path_factory.mktemp("rot")
    export_step(rotated, str(d / "rot.step"))
    export_stl(rotated, str(d / "rot.stl"), tolerance=0.01, angular_tolerance=0.1)
    return d, rotated


@pytest.mark.parametrize("ext", ["step", "stl"])
def test_rotated_bracket_features_are_orientation_independent(rotated_bracket, ext):
    d, _ = rotated_bracket
    rep = build_report(load(d / f"rot.{ext}"), symmetry=False)
    holes = [c for c in rep.features["cylinders"] if c["kind"] == "through_hole"]
    boss = [c for c in rep.features["cylinders"] if c["kind"] == "boss"]
    assert len(holes) == 2 and len(boss) == 1
    for h in holes:
        assert h["diameter"] == pytest.approx(BRACKET["hole_d"], abs=0.01)
        assert h["height"] == pytest.approx(BRACKET["plate"][2], abs=0.01)
    assert boss[0]["diameter"] == pytest.approx(BRACKET["boss_d"], abs=0.01)
    # all three axes are parallel (the plate normal) and the holes are 40 mm apart
    dirs = [np.asarray(c["axis_dir"]) for c in holes + boss]
    for a in dirs[1:]:
        assert abs(float(dirs[0] @ a)) == pytest.approx(1.0, abs=1e-3)
    dist = np.linalg.norm(np.asarray(holes[0]["axis_point"]) - np.asarray(holes[1]["axis_point"]))
    assert dist == pytest.approx(2 * BRACKET["hole_x"], abs=0.02)
    # the oriented bbox recovers the plate's natural frame
    assert sorted(rep.summary["oriented_bbox"]["extents"], reverse=True) == pytest.approx([60.0, 40.0, 13.0], abs=0.01)
    assert rep.summary["oriented_bbox"]["axis_aligned"] is False
    assert len(rep.features["planes"]) == 7


def test_coarse_tessellation_still_measures(tmp_path):
    """A slicer-grade STL (0.1 mm / 0.5 rad) must still yield the right diameters within its own error."""
    from build123d import export_stl

    export_stl(build_bracket(), str(tmp_path / "coarse.stl"), tolerance=0.1, angular_tolerance=0.5)
    model = load(tmp_path / "coarse.stl")
    f = extract_features(model)
    holes = [c for c in f["cylinders"] if c["kind"] == "through_hole"]
    assert len(holes) == 2
    for h in holes:
        assert h["diameter"] == pytest.approx(BRACKET["hole_d"], abs=0.02)
    boss = [c for c in f["cylinders"] if c["kind"] == "boss"]
    assert len(boss) == 1 and boss[0]["diameter"] == pytest.approx(BRACKET["boss_d"], abs=0.02)


def test_sphere_and_cone_are_not_mistaken_for_cylinders(tmp_path):
    from build123d import Box, BuildPart, Cone, Sphere, export_stl

    with BuildPart() as bp:
        Box(40, 40, 10)
        with Locations_at((10, 0, 5)):
            Sphere(6)
        with Locations_at((-10, 0, 5)):
            Cone(bottom_radius=6, top_radius=2, height=8)
    export_stl(bp.part, str(tmp_path / "sc.stl"), tolerance=0.01, angular_tolerance=0.1)
    f = extract_features(load(tmp_path / "sc.stl"))
    full = [c for c in f["cylinders"] if c["kind"] not in {"partial", "fillet_candidate"}]
    assert full == [], full  # no phantom holes or bosses
    assert f["counts"]["unclassified_regions"] >= 2  # the sphere and the cone are reported, not hidden
    assert any(abs(p["offset"] - 5.0) < 1e-3 and p["normal"][2] > 0.99 for p in f["planes"])  # box top still found


def Locations_at(pt):
    from build123d import Locations

    return Locations(pt)


def test_hole_in_tilted_wall(tmp_path):
    """A hole drilled at 30° to the plate normal: direction, diameter and length must all be right."""
    from build123d import Axis, Box, Cylinder, export_stl

    part = Box(40, 40, 12) - Cylinder(radius=2.0, height=60).rotate(Axis.X, 30.0)
    export_stl(part, str(tmp_path / "tilt.stl"), tolerance=0.01, angular_tolerance=0.1)
    f = extract_features(load(tmp_path / "tilt.stl"))
    holes = [c for c in f["cylinders"] if c["kind"] == "through_hole"]
    assert len(holes) == 1
    h = holes[0]
    assert h["diameter"] == pytest.approx(4.0, abs=0.01)
    assert abs(float(np.asarray(h["axis_dir"]) @ np.array([0.0, -math.sin(math.radians(30)), math.cos(math.radians(30))]))) == pytest.approx(1.0, abs=1e-3)
    # axial extent of the surface: centre-line length plus the spread of the two elliptical ends
    t = math.radians(30)
    assert h["height"] == pytest.approx(12.0 / math.cos(t) + 2 * 2.0 * math.tan(t), abs=0.05)
