"""Regression cases from the independent review of the geometry core (2026-09-15).

Each case reproduced a misclassification before the fix; the assertions state the design intent.
"""

from __future__ import annotations

import numpy as np
import pytest
from build123d import Axis, Box, Cylinder, Locations, Plane, export_step, export_stl

from calipers import build_report, load, measure
from calipers.features import extract_features


def _both(part, tmp_path, name):
    step, stl = tmp_path / f"{name}.step", tmp_path / f"{name}.stl"
    export_step(part, str(step))
    export_stl(part, str(stl), tolerance=0.01, angular_tolerance=0.1)
    return step, stl


def _kinds(f, kind):
    return [c for c in f["cylinders"] if c["kind"] == kind]


def test_box_with_all_edges_filleted_keeps_its_planes(tmp_path):
    from build123d import BuildPart, fillet

    with BuildPart() as bp:
        Box(20, 15, 6)
        fillet(bp.edges(), radius=0.5)
    _, stl = _both(bp.part, tmp_path, "rbox")
    f = extract_features(load(stl))
    assert f["counts"]["planes"] == 6, f["counts"]
    fillets = _kinds(f, "fillet_candidate")
    assert len(fillets) == 12 and all(c["radius"] == pytest.approx(0.5, abs=0.01) for c in fillets)
    assert not [c for c in f["cylinders"] if c["kind"] not in {"fillet_candidate", "partial"}]  # no phantom cylinder


def test_boss_with_base_fillet_is_a_boss(tmp_path):
    from build123d import BuildPart, BuildSketch, Circle, extrude, fillet

    with BuildPart() as bp:
        Box(40, 40, 6)
        with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):
            Circle(6)
        extrude(amount=8)
        fillet(bp.edges().filter_by(Axis.Z, reverse=True).filter_by(lambda e: abs(e.center().Z - 3.0) < 1e-6 and e.length < 40), radius=1.0)
    step, stl = _both(bp.part, tmp_path, "bossfil")
    for path in (step, stl):
        f = extract_features(load(path))
        boss = [c for c in f["cylinders"] if abs(c["diameter"] - 12.0) <= 0.02 and c["coverage_deg"] >= 300]
        assert len(boss) == 1 and boss[0]["kind"] == "boss", (path.suffix, boss)


def test_blind_hole_with_drill_point_is_blind(tmp_path):
    from build123d import Cone, Location

    # body spans z ∈ [-6, 6]; a Ø6 hole from the top face down to z=0 with a 118° drill point below it
    body = Box(30, 30, 12)
    cyl = Cylinder(radius=3, height=6).moved(Location((0, 0, 3)))  # z ∈ [0, 6]
    h_tip = 3 / np.tan(np.radians(59))
    tip = Cone(bottom_radius=0, top_radius=3, height=h_tip).moved(Location((0, 0, -h_tip / 2)))  # apex at z=-h_tip, base at z=0
    part = body - (cyl + tip)
    step, stl = _both(part, tmp_path, "drill")
    for path in (step, stl):
        f = extract_features(load(path))
        holes = [c for c in f["cylinders"] if abs(c["diameter"] - 6.0) <= 0.02 and c["coverage_deg"] >= 300]
        assert len(holes) == 1 and holes[0]["kind"] == "blind_hole", (path.suffix, holes)
        assert not _kinds(f, "fillet_candidate"), (path.suffix, "cone facets must not become fillet candidates")


def test_groove_crossed_by_rib_is_not_a_through_hole(tmp_path):
    body = Box(100, 30, 10)
    groove = Cylinder(radius=3, height=120).rotate(Axis.Y, 90).moved(Plane.XY.offset(5).location)
    rib = Box(0.5, 30, 10)
    part = (body - groove) + rib
    step, stl = _both(part, tmp_path, "groove")
    for path in (step, stl):
        f = extract_features(load(path))
        r3 = [c for c in f["cylinders"] if abs(c["radius"] - 3.0) <= 0.02]
        assert r3 and all(c["coverage_deg"] < 200 for c in r3), (path.suffix, [(c["kind"], c["coverage_deg"]) for c in r3])
        assert not [c for c in r3 if "hole" in c["kind"]]


def test_extra_hole_breaks_symmetry(tmp_path):
    from build123d import BuildPart, Hole

    with BuildPart() as bp:
        Box(60, 40, 5)
        with Locations((20, 0), (-20, 0)):
            Hole(radius=2.5)
        with Locations((0, 6)):
            Hole(radius=1.0)
    _, stl = _both(bp.part, tmp_path, "asym")
    rep = build_report(load(stl), symmetry=True)
    x_plane = next(p for p in rep.symmetry["planes"] if abs(p["plane_normal"][0]) > 0.999)
    assert x_plane["symmetric"]  # the extra hole sits on x=0: still symmetric about the YZ plane
    y_plane = next(p for p in rep.symmetry["planes"] if abs(p["plane_normal"][1]) > 0.999)
    assert not y_plane["symmetric"] and not y_plane["feature_check"]["ok"]


def test_obb_yaw_of_tall_box_and_axis_aligned_hex(tmp_path):
    from build123d import BuildPart, BuildSketch, RegularPolygon, extrude

    _, stl = _both(Box(8, 20, 50).rotate(Axis.Z, 30), tmp_path, "tall")
    obb = measure.summary(load(stl))["oriented_bbox"]
    assert obb["rotation_about_z_deg"] == pytest.approx(30.0, abs=0.05)
    with BuildPart() as bp:
        with BuildSketch():
            RegularPolygon(radius=10, side_count=6)
        extrude(amount=8)
    _, hexstl = _both(bp.part, tmp_path, "hex")
    obb = measure.summary(load(hexstl))["oriented_bbox"]
    assert obb["axis_aligned"] is True and obb["rotation_about_z_deg"] == pytest.approx(0.0, abs=0.05)


def test_dirty_mesh_is_cleaned(tmp_path):
    import trimesh

    _, stl = _both(Box(20, 15, 6), tmp_path, "box")
    m = trimesh.load(str(stl), force="mesh")
    dup = np.vstack([m.faces, m.faces[:5], np.array([[0, 0, 1], [2, 2, 2]])])  # duplicate + degenerate
    dirty = trimesh.Trimesh(vertices=m.vertices, faces=dup, process=False)
    dirty.export(str(tmp_path / "dirty.stl"))
    model = load(tmp_path / "dirty.stl")
    assert model.mesh.is_watertight and any("degenerate/duplicate" in n for n in model.notes)
    s = measure.summary(model)
    assert s["volume"] == pytest.approx(20 * 15 * 6, rel=1e-6)
    f = extract_features(model)
    assert f["counts"]["planes"] == 6 and max(p["area"] for p in f["planes"]) == pytest.approx(300.0, abs=1e-3)


def test_section_coordinates_are_world_coordinates(tmp_path):
    from build123d import BuildPart, Hole

    with BuildPart() as bp:
        Box(60, 40, 5)
        with Locations((20, 10)):
            Hole(radius=2.5)
    _, stl = _both(bp.part, tmp_path, "sec")
    sec = measure.section(load(stl), [0, 0, 0], [0, 0, 1])
    assert sec["plane"]["in_plane_axes"] == ["x", "y"]
    hole = next(L for L in sec["loops"] if L["type"] == "inner")
    assert hole["centroid"] == pytest.approx([20.0, 10.0], abs=0.01)
    outer = next(L for L in sec["loops"] if L["type"] == "outer")
    assert outer["rectangle"]["is_rectangle"] and outer["rectangle"]["angle_deg"] in (pytest.approx(0.0, abs=0.01), pytest.approx(180.0, abs=0.01))
    assert outer["bbox_2d"] == pytest.approx([-30, -20, 30, 20], abs=0.01)


def test_brep_full_cylinder_coverage_is_exactly_360(tmp_path):
    step, _ = _both(Box(30, 30, 10) - Cylinder(radius=4, height=20), tmp_path, "cov")
    f = extract_features(load(step))
    holes = _kinds(f, "through_hole")
    assert len(holes) == 1 and holes[0]["coverage_deg"] == 360.0
