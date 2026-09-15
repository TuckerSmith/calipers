"""#3DBenchy: an organic, non-watertight torture-test mesh with published nominal dimensions.

Nominals from 3dbenchy.com/dimensions: 60.00 × 31.00 × 48.00 mm overall; chimney bore Ø3.00 blind
11.00 deep, chimney outer top Ø7.00; hawsepipe Ø4.00; rear window Ø9.00 inner / Ø12.00 outer with a
0.30 mm flange; cargo box inner 8.00 × 7.00. The model is public domain.
"""

from __future__ import annotations

import pytest

from calipers import build_report, load


@pytest.fixture(scope="module")
def benchy(benchy_file):
    model = load(benchy_file)
    return model, build_report(model, sections=["z=10"], symmetry=True)


def _find(rep, kind, diameter, tol=0.01):
    return [c for c in rep.features["cylinders"] if c["kind"] == kind and abs(c["diameter"] - diameter) <= tol]


def test_overall_dimensions(benchy):
    _, rep = benchy
    assert rep.summary["extents"] == pytest.approx([60.0, 31.0, 48.0], abs=0.01)
    assert rep.summary["watertight"] is False  # known: the published STL has small defects


def test_port_starboard_symmetry(benchy):
    """The hull is mirror-symmetric about y=0; the engraved text on the bottom is the only exception."""
    _, rep = benchy
    y_plane = next(p for p in rep.symmetry["planes"] if abs(p["plane_normal"][1]) > 0.999)
    assert y_plane["surface_symmetric"] and y_plane["p95_deviation"] < 0.02 and y_plane["p99_deviation"] < 0.1
    assert y_plane["offset"] == pytest.approx(0.0, abs=0.01)
    # the feature check is stricter: it flags the one Ø2 recess that belongs to the bottom engraving
    unmatched = y_plane["feature_check"]["unmatched"]
    assert len(unmatched) <= 1
    for cid in unmatched:
        c = next(c for c in rep.features["cylinders"] if c["id"] == cid)
        assert c["diameter"] == pytest.approx(2.0, abs=0.01) and c["height"] < 0.5
    assert not any(p["symmetric"] for p in rep.symmetry["planes"] if abs(p["plane_normal"][1]) < 0.5)  # no other plane


def test_chimney(benchy):
    _, rep = benchy
    bore = _find(rep, "blind_hole", 3.0)
    assert len(bore) == 1
    b = bore[0]
    assert b["height"] == pytest.approx(11.0, abs=0.02)
    assert abs(b["axis_dir"][2]) == pytest.approx(1.0, abs=1e-3)
    assert max(b["start"][2], b["end"][2]) == pytest.approx(48.0, abs=0.02)  # opens at the top
    # the chimney stack: a Ø7.00 outer wall (with a small rounded rim at the very top and a flare
    # below), coaxial with the bore
    outer = [c for c in rep.features["cylinders"] if abs(c["diameter"] - 7.0) <= 0.01 and not c["concave"] and c["coverage_deg"] >= 300]
    assert len(outer) == 1
    assert outer[0]["axis_point"][:2] == pytest.approx([-4.0, 0.0], abs=0.01)
    assert outer[0]["height"] == pytest.approx(2.45, abs=0.1)
    model, _ = benchy
    from calipers import measure

    sec = measure.section(model, [0, 0, 47.0], [0, 0, 1])
    ring = [L for L in sec["loops"] if L["circle"]["is_circle"] and abs(L["centroid"][0] + 4.0) < 0.1]
    assert sorted(round(L["circle"]["diameter"], 2) for L in ring) == pytest.approx([3.0, 7.0], abs=0.01)


def test_hawsepipes(benchy):
    _, rep = benchy
    pipes = _find(rep, "through_hole", 4.0)
    assert len(pipes) == 2
    ys = sorted(p["axis_point"][1] for p in pipes)
    assert ys[0] == pytest.approx(-ys[1], abs=0.02)  # mirrored port / starboard


def test_rear_window(benchy):
    _, rep = benchy
    inner = _find(rep, "through_hole", 9.0)
    assert len(inner) == 1
    flange = _find(rep, "boss", 12.0)
    assert len(flange) == 1
    assert flange[0]["height"] == pytest.approx(0.30, abs=0.01)
    assert flange[0]["axis_point"][2] == pytest.approx(inner[0]["axis_point"][2], abs=0.05)  # coaxial with the bore


def test_cargo_box_section(benchy):
    _, rep = benchy
    rects = [L for L in rep.sections[0]["loops"] if L["rectangle"]["is_rectangle"]]
    sizes = [tuple(round(v, 2) for v in L["rectangle"]["size"]) for L in rects]
    assert any(abs(w - 8.0) <= 0.02 and abs(h - 7.0) <= 0.02 for w, h in sizes), sizes
