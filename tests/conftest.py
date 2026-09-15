"""Shared fixtures: a bracket with known dimensions (ground truth), plus the standard test models.

Ground truth bracket: 60 × 40 × 5 mm plate, R3 vertical-edge fillets, two Ø5 through-holes at
x = ±20, and a Ø12 × 8 mm boss on top, centred. Every dimension in the tests below is derived from
these numbers, never from the code under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

BRACKET = {
    "plate": (60.0, 40.0, 5.0),
    "fillet_r": 3.0,
    "hole_d": 5.0,
    "hole_x": 20.0,
    "boss_d": 12.0,
    "boss_h": 8.0,
}


def build_bracket():
    from build123d import Axis, Box, BuildPart, BuildSketch, Circle, Hole, Locations, extrude, fillet

    L, W, T = BRACKET["plate"]
    with BuildPart() as bp:
        Box(L, W, T)
        fillet(bp.edges().filter_by(Axis.Z), radius=BRACKET["fillet_r"])
        with Locations((BRACKET["hole_x"], 0), (-BRACKET["hole_x"], 0)):
            Hole(radius=BRACKET["hole_d"] / 2)
        with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):
            Circle(BRACKET["boss_d"] / 2)
        extrude(amount=BRACKET["boss_h"])
    return bp.part


@pytest.fixture(scope="session")
def bracket_files(tmp_path_factory) -> dict[str, Path]:
    from build123d import export_step, export_stl

    d = tmp_path_factory.mktemp("bracket")
    part = build_bracket()
    step, stl = d / "bracket.step", d / "bracket.stl"
    export_step(part, str(step))
    export_stl(part, str(stl), tolerance=0.01, angular_tolerance=0.1)
    return {"step": step, "stl": stl}


@pytest.fixture(scope="session")
def nist_files() -> dict[str, Path]:
    return {"step": FIXTURES / "nist_am_test_artifact.step", "stl": FIXTURES / "nist_am_test_artifact.stl"}


@pytest.fixture(scope="session")
def benchy_file() -> Path:
    return FIXTURES / "3DBenchy.stl"


@pytest.fixture
def anyio_backend():
    return "asyncio"
