"""Ground-truth tests: every number asserted here comes from the bracket's construction parameters."""

from __future__ import annotations

import math

import pytest

from calipers import build_report, load
from calipers import measure
from tests.conftest import BRACKET

L, W, T = BRACKET["plate"]
TOP = T / 2
BOTTOM = -T / 2


@pytest.fixture(scope="module", params=["step", "stl"])
def bracket_report(request, bracket_files):
    model = load(bracket_files[request.param])
    return model, build_report(model, sections=["z=0"], symmetry=True)


def _cyl(rep, kind, diameter, tol=0.01):
    return [c for c in rep.features["cylinders"] if c["kind"] == kind and abs(c["diameter"] - diameter) <= tol]


def test_envelope_and_volume(bracket_report):
    model, rep = bracket_report
    s = rep.summary
    assert s["extents"] == pytest.approx([L, W, T + BRACKET["boss_h"]], abs=1e-3)
    # exact volume: plate with filleted corners, minus two holes, plus boss
    r = BRACKET["fillet_r"]
    plate = (L * W - (4 - math.pi) * r * r) * T
    holes = 2 * math.pi * (BRACKET["hole_d"] / 2) ** 2 * T
    boss = math.pi * (BRACKET["boss_d"] / 2) ** 2 * BRACKET["boss_h"]
    expected = plate - holes + boss
    rel = 1e-6 if model.is_exact else 2e-4  # tessellation at 0.01 mm loses a little volume
    assert s["volume"] == pytest.approx(expected, rel=rel)
    assert s["exact"] is model.is_exact


def test_through_holes(bracket_report):
    _, rep = bracket_report
    holes = _cyl(rep, "through_hole", BRACKET["hole_d"])
    assert len(holes) == 2
    xs = sorted(h["axis_point"][0] for h in holes)
    assert xs == pytest.approx([-BRACKET["hole_x"], BRACKET["hole_x"]], abs=0.01)
    for h in holes:
        assert h["axis_point"][1] == pytest.approx(0.0, abs=0.01)
        assert abs(h["axis_dir"][2]) == pytest.approx(1.0, abs=1e-3)
        assert h["height"] == pytest.approx(T, abs=0.01)
        assert h["concave"] is True
        assert h["open_ends"] == [True, True]


def test_boss(bracket_report):
    _, rep = bracket_report
    boss = _cyl(rep, "boss", BRACKET["boss_d"])
    assert len(boss) == 1
    b = boss[0]
    assert b["height"] == pytest.approx(BRACKET["boss_h"], abs=0.01)
    assert b["concave"] is False
    assert sorted([b["start"][2], b["end"][2]]) == pytest.approx([TOP, TOP + BRACKET["boss_h"]], abs=0.01)


def test_fillets_are_partial_cylinders(bracket_report):
    _, rep = bracket_report
    fillets = [c for c in rep.features["cylinders"] if c["kind"] == "fillet_candidate"]
    assert len(fillets) == 4
    for f in fillets:
        assert f["radius"] == pytest.approx(BRACKET["fillet_r"], abs=0.01)
        assert f["coverage_deg"] == pytest.approx(90.0, abs=6.0)
        assert f["concave"] is False


def test_planes(bracket_report):
    _, rep = bracket_report
    planes = rep.features["planes"]
    assert len(planes) == 7  # top, bottom, 4 sides, boss top
    by_offset = {(tuple(round(v) for v in p["normal"]), round(p["offset"], 2)) for p in planes}
    assert ((0, 0, 1), round(TOP + BRACKET["boss_h"], 2)) in by_offset
    assert ((0, 0, -1), round(-BOTTOM, 2)) in by_offset
    assert ((1, 0, 0), L / 2) in by_offset and ((0, 1, 0), W / 2) in by_offset
    top = next(p for p in planes if p["normal"][2] > 0.99 and abs(p["offset"] - TOP) < 1e-3)
    r = BRACKET["fillet_r"]
    expected_top = L * W - (4 - math.pi) * r * r - 2 * math.pi * (BRACKET["hole_d"] / 2) ** 2 - math.pi * (BRACKET["boss_d"] / 2) ** 2
    assert top["area"] == pytest.approx(expected_top, rel=1e-3)


def test_section_finds_holes(bracket_report):
    _, rep = bracket_report
    sec = rep.sections[0]
    circles = [L_ for L_ in sec["loops"] if L_["type"] == "inner" and L_["circle"]["is_circle"]]
    assert len(circles) == 2
    for c in circles:
        assert c["circle"]["diameter"] == pytest.approx(BRACKET["hole_d"], abs=0.005)
    outer = [L_ for L_ in sec["loops"] if L_["type"] == "outer"]
    assert len(outer) == 1 and outer[0]["rectangle"]["is_rectangle"] is False  # rounded corners


def test_exact_section_on_brep(bracket_files):
    model = load(bracket_files["step"])
    ex = measure.section_exact(model, [0, 0, 0], [0, 0, 1])
    inner = [L_ for L_ in ex["loops"] if L_["type"] == "inner"]
    assert len(inner) == 2 and all(L_["is_circle"] for L_ in inner)
    assert all(L_["diameter"] == pytest.approx(BRACKET["hole_d"], abs=1e-6) for L_ in inner)


def test_symmetry(bracket_report):
    _, rep = bracket_report
    sym = {tuple(abs(round(v)) for v in p["plane_normal"]): p for p in rep.symmetry["planes"]}
    assert sym[(1, 0, 0)]["symmetric"] and sym[(0, 1, 0)]["symmetric"]
    assert not sym[(0, 0, 1)]["symmetric"]  # the boss breaks top/bottom symmetry


def test_oriented_bbox_axis_aligned(bracket_report):
    _, rep = bracket_report
    obb = rep.summary["oriented_bbox"]
    assert obb["axis_aligned"] is True
    assert sorted(obb["extents"], reverse=True) == pytest.approx([L, W, T + BRACKET["boss_h"]], abs=1e-3)


def test_text_digest_mentions_key_facts(bracket_report):
    _, rep = bracket_report
    txt = rep.to_text()
    assert "2× through_hole Ø5.000" in txt
    assert "1× boss Ø12.000" in txt
    assert "Envelope (axis-aligned): 60.000 × 40.000 × 13.000" in txt
