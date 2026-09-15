"""NIST additive-manufacturing test artifact: the exact B-rep and its tessellated STL must agree.

The STL was produced from the STEP by NIST, in a translated frame, so agreement is checked after
aligning the two axis-aligned bounding boxes. This is the STEP-vs-STL acceptance test for Phase 1:
mesh-fitted features must reproduce kernel-exact ones to caliper precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from calipers import build_report, load


@pytest.fixture(scope="module")
def reports(nist_files):
    step = load(nist_files["step"])
    stl = load(nist_files["stl"])
    return step, build_report(step, symmetry=False), stl, build_report(stl, symmetry=False)


def test_brep_is_valid_and_matches_stl_globally(reports):
    step, rs, stl, rm = reports
    assert rs.summary["valid"] and rs.summary["solids"] == 1
    assert rs.summary["extents"] == pytest.approx(rm.summary["extents"], abs=1e-3)
    assert rs.summary["volume"] == pytest.approx(rm.summary["volume"], rel=2e-4)
    assert rs.summary["surface_area"] == pytest.approx(rm.summary["surface_area"], rel=2e-4)


def test_oriented_bbox_recovers_the_100mm_square(reports):
    _, rs, _, rm = reports
    for rep in (rs, rm):
        obb = rep.summary["oriented_bbox"]
        assert obb["extents"] == pytest.approx([100.0, 100.0, 17.0], abs=1e-3)
        assert obb["rotation_about_z_deg"] == pytest.approx(45.0, abs=0.05)
        assert obb["axis_aligned"] is False


def _groups(rep):
    return [(g["kind"], g["diameter"], g["height"], g["count"]) for g in rep.features["cylinder_groups"]]


def _match(groups, kind, d, h, d_tol=0.01, h_tol=0.02):
    return [g for g in groups if g[0] == kind and abs(g[1] - d) <= d_tol and abs(g[2] - h) <= h_tol]


def test_feature_groups_agree(reports):
    _, rs, _, rm = reports
    gs, gm = _groups(rs), _groups(rm)
    assert len(gs) == len(gm), f"exact={gs}\nfitted={gm}"
    for kind, d, h, n in gs:  # every exact group has a fitted twin within caliper tolerance
        twin = _match(gm, kind, d, h)
        assert len(twin) == 1 and twin[0][3] == n, (kind, d, h, n, twin)
    # a few design intents of the artifact, stated independently of the code
    assert _match(gs, "boss", 4.0, 7.0)[0][3] == 16  # the 4 × 4 pin array
    assert _match(gs, "through_hole", 4.0, 10.0)[0][3] == 4  # corner-ish through holes
    assert _match(gs, "through_hole", 10.0, 17.0)[0][3] == 1  # central bore
    for d in (0.25, 0.5, 1.0, 1.5, 2.0):  # fine-feature pin/hole series
        assert _match(gs, "boss", d, 2.0)[0][3] == 1 and _match(gs, "blind_hole", d, 2.0)[0][3] == 1


def test_axis_points_agree_after_alignment(reports):
    step, rs, stl, rm = reports
    shift = np.asarray(rm.summary["bbox_min"]) - np.asarray(rs.summary["bbox_min"])
    exact = rs.features["cylinders"]
    fitted = rm.features["cylinders"]
    assert len(exact) == len(fitted)
    unmatched = []
    for e in exact:
        p = np.asarray(e["axis_point"]) + shift
        best = min(fitted, key=lambda f: np.linalg.norm(np.asarray(f["axis_point"]) - p))
        d = float(np.linalg.norm(np.asarray(best["axis_point"]) - p))
        if d > 0.02 or abs(best["diameter"] - e["diameter"]) > 0.01 or abs(best["height"] - e["height"]) > 0.02:
            unmatched.append((e["id"], e["kind"], e["diameter"], d))
    assert not unmatched, unmatched


def test_mesh_fits_are_tight(reports):
    _, _, _, rm = reports
    for c in rm.features["cylinders"]:
        assert c["fit_rms"] <= 0.01, c


def test_planes_agree(reports):
    """Every exact plane of meaningful size has a fitted twin: same normal, offset (after alignment) and area."""
    _, rs, _, rm = reports
    shift = np.asarray(rm.summary["bbox_min"]) - np.asarray(rs.summary["bbox_min"])
    fitted = rm.features["planes"]
    missing = []
    for p in rs.features["planes"]:
        if p["area"] < 1.0:  # sub-mm² faces (Ø0.25 pin tops) are below the mesh path's noise floor by design
            continue
        n = np.asarray(p["normal"])
        off = p["offset"] + float(n @ shift)
        twins = [q for q in fitted if float(np.asarray(q["normal"]) @ n) > 0.9998 and abs(q["offset"] - off) <= 0.02]
        # the mesh path ignores patches under MIN_REGION_AREA (0.25 mm²) by design, so a group of many
        # tiny faces may legitimately come up short by that much per face
        tol = max(0.5, 0.01 * p["area"], 0.25 * p["regions"])
        if not twins or abs(sum(q["area"] for q in twins) - p["area"]) > tol:
            missing.append((p["id"], p["normal"], round(off, 3), p["area"], [q["area"] for q in twins]))
    assert not missing, missing
