"""Phase 3: reference-conditioned design — references, keep-in/keep-out, fit contracts, slots,
patterns, generators, best-of-N, version diff. Every assertion uses construction numbers."""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest
import yaml

from calipers import load
from calipers.contracts import verify
from calipers.features import extract_features
from calipers.spec import SpecError, normalize

DEV = (40.0, 30.0, 10.0)  # the "device": a box centred on the origin, z ∈ [-5, 5]
SLOT_W, SLOT_L, SLOT_D = 4.0, 16.0, 4.0  # a blind slot in its top face, along x


def _build_device():
    from build123d import Box, BuildPart, BuildSketch, Mode, Plane, SlotCenterPoint, extrude

    with BuildPart() as bp:
        Box(*DEV)
        with BuildSketch(Plane.XY.offset(DEV[2] / 2)):
            SlotCenterPoint(center=(0, 0), point=((SLOT_L - SLOT_W) / 2, 0), height=SLOT_W)
        extrude(amount=-SLOT_D, mode=Mode.SUBTRACT)
    return bp.part


def _enclosure(clearance: float, wall: float, floor: float | None = None):
    """Open-top box around the device: cavity = device envelope + clearance."""
    from build123d import Align, Box, Location

    floor = wall if floor is None else floor
    L, W, H = DEV
    outer = Box(L + 2 * clearance + 2 * wall, W + 2 * clearance + 2 * wall, H + clearance + floor, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, -H / 2 - clearance - floor)))
    cav = Box(L + 2 * clearance, W + 2 * clearance, H + clearance + wall, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, -H / 2 - clearance)))
    return outer - cav


@pytest.fixture(scope="module")
def device_files(tmp_path_factory):
    from build123d import export_step, export_stl

    d = tmp_path_factory.mktemp("device")
    part = _build_device()
    export_step(part, str(d / "device.step"))
    export_stl(part, str(d / "device.stl"), tolerance=0.01, angular_tolerance=0.1)
    return {"dir": d, "step": d / "device.step", "stl": d / "device.stl"}


@pytest.fixture(scope="module")
def enclosure_files(tmp_path_factory):
    from build123d import export_step

    d = tmp_path_factory.mktemp("enclosure")
    export_step(_enclosure(0.5, 2.0), str(d / "good.step"))
    export_step(_enclosure(0.0, 2.0), str(d / "tight.step"))  # touches the device: gap 0
    export_step(_enclosure(1.5, 2.0), str(d / "loose.step"))
    return {"good": d / "good.step", "tight": d / "tight.step", "loose": d / "loose.step"}


def _spec(device_path, extra: str = "", clearance: float = 0.5):
    return normalize(yaml.safe_load(textwrap.dedent(f"""
        part: enclosure
        references: [{{id: device, path: {device_path}}}]
        fit:
          - {{type: clearance, ref: device, min: {clearance}}}
          - {{type: gap, ref: device, directions: {{"+x": [{clearance}, {clearance + 0.5}], "-y": [{clearance}, {clearance + 0.5}], "-z": [{clearance}, {clearance + 0.5}]}}}}
          - {{type: enclosed, ref: device, min_fraction: 0.95, open: ["+z"]}}
        {extra}
        """)))


# ----------------------------------------------------------------------------- spec
def test_spec_validation_for_references_regions_and_fit():
    with pytest.raises(SpecError, match="needs a path"):
        normalize({"references": [{"id": "x"}]})
    with pytest.raises(SpecError, match="exactly one of"):
        normalize({"references": [{"id": "d", "path": "d.stl"}], "keep_out": [{"box": {"min": [0, 0, 0], "max": [1, 1, 1]}, "from": "d"}]})
    with pytest.raises(SpecError, match="needs depth"):
        normalize({"references": [{"id": "d", "path": "d.stl"}], "keep_out": [{"from": "d", "face": "+x"}]})
    with pytest.raises(SpecError, match="not a reference id"):
        normalize({"references": [{"id": "d", "path": "d.stl"}], "keep_in": [{"from": "nope"}]})
    with pytest.raises(SpecError, match="must name a reference id"):
        normalize({"references": [{"id": "d", "path": "d.stl"}], "fit": [{"type": "clearance", "ref": "q"}]})
    with pytest.raises(SpecError, match="unknown type"):
        normalize({"references": [{"id": "d", "path": "d.stl"}], "fit": [{"type": "snug", "ref": "d"}]})
    with pytest.raises(SpecError, match="needs directions"):
        normalize({"references": [{"id": "d", "path": "d.stl"}], "fit": [{"type": "gap", "ref": "d"}]})
    with pytest.raises(SpecError, match="a slot needs a width"):
        normalize({"features": [{"id": "s", "kind": "slot"}]})
    with pytest.raises(SpecError, match="perpendicular"):
        normalize({"features": [{"id": "s", "kind": "slot", "width": 4, "direction": "z"}]})
    ok = normalize({"references": [{"id": "d", "path": "d.stl", "place": {"rotate": [90, 0, 0], "translate": [1, 2, 3]}}], "keep_out": [{"id": "k", "from": "d", "face": "-z", "depth": 5, "pad": 1}], "fit": [{"type": "gap", "ref": "d", "directions": {"z": 0.5}}]})
    assert ok["fit"][0]["directions"] == {"z": (0.0, 0.5)} and ok["keep_out"][0]["depth"] == 5.0


# ----------------------------------------------------------------------------- slots & patterns
@pytest.mark.parametrize("kind", ["step", "stl"])
def test_blind_slot_detected_on_both_paths(device_files, kind):
    f = extract_features(load(device_files[kind]))
    assert len(f["slots"]) == 1, f["slots"]
    s = f["slots"][0]
    assert s["kind"] == "blind_slot"
    assert abs(s["width"] - SLOT_W) < 0.02 and abs(s["length"] - SLOT_L) < 0.05 and abs(s["depth"] - SLOT_D) < 0.02
    assert np.allclose(s["center"], [0, 0, DEV[2] / 2 - SLOT_D / 2], atol=0.05)
    assert abs(abs(s["direction"][0]) - 1) < 1e-3 and abs(abs(s["axis_dir"][2]) - 1) < 1e-3
    assert s["exact"] == (kind == "step")
    # the two half-cylinders are no longer loose partials
    assert all(c["kind"] == "slot_end" for c in f["cylinders"] if c["id"] in s["end_ids"])
    assert f["counts"]["slots"] == 1


def test_slot_spec_contract_and_exact_counts(device_files):
    spec = normalize({"features": [{"id": "key", "kind": "slot", "width": [SLOT_W, 0.05], "length": [SLOT_L, 0.1], "depth": [SLOT_D, 0.05], "axis": "z", "direction": "x", "positions": [[0, 0]]}], "exact_counts": True})
    res = verify(load(device_files["step"]), spec)
    assert res.passed, res.to_text()
    wrong = normalize({"features": [{"id": "key", "kind": "slot", "width": [SLOT_W, 0.05], "length": [SLOT_L + 2, 0.1], "axis": "z", "direction": "y", "positions": [[0, 0]]}]})
    res = verify(load(device_files["step"]), wrong)
    failed = {c.id for c in res.checks if not c.passed}
    assert failed == {"key.0.length", "key.0.direction"}, res.to_text()
    # a slot that is not in the spec is an extra feature under exact_counts
    res = verify(load(device_files["step"]), normalize({"exact_counts": True}))
    chk = next(c for c in res.checks if c.id == "features.exact_counts")
    assert not chk.passed and "through_slot" not in str(chk.measured) and "blind_slot" in str(chk.measured)


def test_patterns_grid_and_bolt_circle(tmp_path):
    from build123d import Box, BuildPart, Hole, Locations, PolarLocations, export_step

    with BuildPart() as bp:
        Box(80, 60, 5)
        with Locations((-30, -20), (0, -20), (30, -20), (-30, 20), (0, 20), (30, 20)):
            Hole(radius=2.0)
        with PolarLocations(15.0, 5):
            Hole(radius=1.5)
    export_step(bp.part, str(tmp_path / "pat.step"))
    f = extract_features(load(tmp_path / "pat.step"))
    kinds = {p["type"]: p for p in f["patterns"]}
    assert set(kinds) == {"grid", "circular"}, f["patterns"]
    g, c = kinds["grid"], kinds["circular"]
    assert g["rows"] == 2 and g["cols"] == 3 and sorted(g["pitch"]) == [30.0, 40.0]
    assert c["count"] == 5 and abs(c["pitch_radius"] - 15.0) < 1e-3 and abs(c["angular_pitch_deg"] - 72.0) < 0.01
    assert np.allclose(c["center"][:2], [0, 0], atol=1e-3)


# ----------------------------------------------------------------------------- fit contracts
def test_fit_contracts_pass_for_the_right_enclosure(device_files, enclosure_files):
    spec = _spec(device_files["stl"])
    res = verify(load(enclosure_files["good"]), spec)
    assert res.passed, res.to_text()
    by = {c.id: c for c in res.checks}
    assert abs(by["fit.0.clearance.min_distance"].measured - 0.5) < 1e-3
    for d in ("+x", "-y", "-z"):
        assert abs(by[f"fit.1.gap.{d}"].measured - 0.5) < 1e-3
    assert by["fit.2.enclosed"].measured >= 0.99


def test_fit_contracts_fail_when_tight_or_loose(device_files, enclosure_files):
    spec = _spec(device_files["stl"])
    tight = verify(load(enclosure_files["tight"]), spec)
    failed = {c.id for c in tight.checks if not c.passed}
    assert {"fit.0.clearance.min_distance", "fit.1.gap.+x", "fit.1.gap.-y", "fit.1.gap.-z"} <= failed
    assert "fit.0.clearance.no_overlap" not in failed  # touching is not overlapping
    by = {c.id: c for c in tight.checks}
    assert by["fit.1.gap.-z"].measured == 0.0  # a wall touching the reference reads as gap 0, not as the next wall
    loose = verify(load(enclosure_files["loose"]), spec)
    failed = {c.id for c in loose.checks if not c.passed}
    assert failed == {"fit.1.gap.+x", "fit.1.gap.-y", "fit.1.gap.-z"}, loose.to_text()
    assert all(abs(c.measured - 1.5) < 1e-3 for c in loose.checks if c.id.startswith("fit.1.gap"))


def test_exact_clearance_between_two_breps_and_overlap(device_files, enclosure_files, tmp_path):
    from build123d import Box, export_step

    spec = _spec(device_files["step"])  # STEP reference → kernel-exact distance
    res = verify(load(enclosure_files["good"]), spec)
    c = next(x for x in res.checks if x.id == "fit.0.clearance.min_distance")
    assert c.passed and c.note == "exact" and abs(c.measured - 0.5) < 1e-6
    # a part that intrudes 1 mm into the device from the -x side: overlap volume = 1 × 30 × 10
    export_step(Box(10, 30, 10).moved(__import__("build123d").Location((-24, 0, 0))), str(tmp_path / "intruder.step"))
    res = verify(load(tmp_path / "intruder.step"), spec)
    by = {c.id: c for c in res.checks}
    assert not by["fit.0.clearance.no_overlap"].passed and abs(by["fit.0.clearance.no_overlap"].deviation or 0) == 0
    assert "300.0" in str(by["fit.0.clearance.no_overlap"].measured)
    assert not by["fit.0.clearance.min_distance"].passed and by["fit.0.clearance.min_distance"].measured == 0.0


def test_keep_out_and_keep_in_regions_exact_volumes(device_files, enclosure_files):
    """Keep-out box 1 mm into the +x wall over the full wall height: intrusion = 1 × (30+1+4) × 12.5 mm³."""
    good = load(enclosure_files["good"])
    W, H = DEV[1] + 2 * 0.5 + 2 * 2.0, DEV[2] + 0.5 + 2.0  # outer width, total height
    spec = normalize({
        "keep_out": [
            {"id": "wall", "box": {"min": [DEV[0] / 2 + 0.5 + 1.0, -50, -50], "max": [DEV[0] / 2 + 0.5 + 2.0, 50, 50]}},
            {"id": "screw", "cylinder": {"from": [0, 0, -20], "to": [0, 0, 20], "radius": 1.5}},
            {"id": "above", "box": {"min": [-50, -50, DEV[2] / 2 + 0.5], "max": [50, 50, 50]}},
        ],
        "keep_in": [{"id": "bed", "box": {"center": [0, 0, 0], "size": [100, 100, 100]}}, {"id": "small", "box": {"center": [0, 0, 0], "size": [DEV[0], 100, 100]}}],
    })
    res = verify(good, spec)
    by = {c.id: c for c in res.checks}
    assert not by["keep_out.wall"].passed and abs(by["keep_out.wall"].deviation - 1.0 * W * H) < 1e-3
    assert not by["keep_out.screw"].passed and abs(by["keep_out.screw"].deviation - math.pi * 1.5**2 * 2.0) < 1e-3  # through the 2 mm floor
    assert by["keep_out.above"].passed  # the rim stops at the device top + clearance
    assert by["keep_in.bed"].passed
    # outside |x| <= 20: two full walls (2 × W × H), plus the 0.5 mm of floor and of the two side walls
    # between the cavity edge (x = 20.5) and the box (x = 20)
    outside = 2 * (2.0 * W * H + 0.5 * W * 2.0 + 0.5 * 2 * 2.0 * (H - 2.0))
    assert not by["keep_in.small"].passed and abs(by["keep_in.small"].deviation - outside) < 1e-3
    # the same regions on the STL of the enclosure go through the mesh boolean and agree closely
    res_m = verify(load(device_files["stl"]), normalize({"keep_out": [{"id": "half", "box": {"min": [0, -50, -50], "max": [50, 50, 50]}}]}))
    half = next(c for c in res_m.checks if c.id == "keep_out.half")
    slot_half = (SLOT_L - SLOT_W) / 2 * SLOT_W * SLOT_D + math.pi * (SLOT_W / 2) ** 2 / 2 * SLOT_D
    assert abs(half.deviation - (DEV[0] * DEV[1] * DEV[2] / 2 - slot_half)) < 2.0 and "mesh" in half.note


def test_derived_region_from_reference_face_and_placement(device_files):
    from calipers import reference as refmod

    spec = normalize({"references": [{"id": "device", "path": str(device_files["stl"]), "place": {"rotate": [0, 0, 90], "translate": [10, 0, 0]}}], "keep_out": [{"id": "cable", "from": "device", "face": "+y", "depth": 20, "pad": 1}]})
    ref = refmod.load_reference(spec["references"][0])
    lo, hi = refmod.region_box(spec["keep_out"][0], {"device": ref})
    # rotated 90° about z the 40 × 30 box becomes 30 × 40, shifted +10 in x; the +y face is at y = 20
    assert np.allclose(lo, [10 - 15 - 1, 20, -5 - 1], atol=1e-3) and np.allclose(hi, [10 + 15 + 1, 40, 5 + 1], atol=1e-3)


# ----------------------------------------------------------------------------- generators & sandbox
def test_enclosure_generator_passes_its_own_contracts_unattended(device_files, tmp_path):
    from calipers import generators
    from calipers.sandbox import run_code
    from calipers.spec import load_spec

    res = generators.enclosure(device_files["stl"], "device", wall=2.0, clearance=0.5, open_face="+z", corner_radius=3.0, out_dir=tmp_path / "gen")
    spec = load_spec(res["paths"]["spec"])
    assert spec["references"][0]["path"] == str(device_files["stl"]) or (tmp_path / "gen" / spec["references"][0]["path"]).exists()
    rr = run_code(res["code"], workdir=tmp_path / "gen" / "run", spec=spec)
    assert rr.ok and rr.lint["ok"] and rr.verify["passed"], rr.to_text()
    ext = {c["id"]: c["measured"] for c in rr.verify["checks"] if c["id"].startswith("envelope")}
    assert ext == {"envelope.x": 45.0, "envelope.y": 35.0, "envelope.z": 13.0}
    assert rr.passed and rr.score() == (0, 0, 0, 0, 0.0)
    # sideways-open variant: the floor is opposite the open face and the opening keep-out sits at -x
    res = generators.enclosure(device_files["stl"], "device", wall=1.6, clearance=0.4, floor=3.0, open_face="-x", out_dir=tmp_path / "gen2")
    rr = run_code(res["code"], workdir=tmp_path / "gen2" / "run", spec=load_spec(res["paths"]["spec"]))
    assert rr.passed, rr.to_text()
    ext = {c["id"]: c["measured"] for c in rr.verify["checks"] if c["id"].startswith("envelope")}
    assert ext == {"envelope.x": round(40 + 0.8 + 3.0, 4), "envelope.y": round(30 + 0.8 + 3.2, 4), "envelope.z": round(10 + 0.8 + 3.2, 4)}


def test_mount_generator_measures_holes_and_passes(tmp_path):
    from build123d import Align, Box, BuildPart, Cylinder, Hole, Locations, export_stl

    from calipers import generators
    from calipers.sandbox import run_code
    from calipers.spec import load_spec

    holes = [(-22.0, -12.0), (22.0, -12.0), (-22.0, 12.0), (22.0, 12.0)]
    with BuildPart() as pcb:  # 50 × 30 × 1.6 board, four Ø3.2 holes, a Ø8 component on top
        Box(50, 30, 1.6)
        with Locations(*holes):
            Hole(radius=1.6)
        with Locations((5, 3, 0.8)):
            Cylinder(radius=4, height=6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    export_stl(pcb.part, str(tmp_path / "board.stl"), tolerance=0.01, angular_tolerance=0.1)
    res = generators.mount(tmp_path / "board.stl", "board", plate_t=3.0, standoff_h=5.0, out_dir=tmp_path / "mount")
    assert sorted(h["position"] for h in res["holes"]) == sorted([list(h) for h in holes])
    rr = run_code(res["code"], workdir=tmp_path / "mount" / "run", spec=load_spec(res["paths"]["spec"]))
    assert rr.passed, rr.to_text()
    by = {c["id"]: c for c in rr.verify["checks"]}
    assert by["fit.1.gap.-z"]["measured"] == 0.0 and by["keep_out.device"]["passed"] and by["plane.plate_bottom.offset"]["measured"] == 8.8
    export_stl(_build_device(), str(tmp_path / "noholes.stl"), tolerance=0.01, angular_tolerance=0.1)
    with pytest.raises(ValueError, match="no through holes"):
        generators.mount(tmp_path / "noholes.stl")


def test_measured_sources_are_checked_against_the_reference(device_files):
    from calipers.provenance import check_sources, lint_source

    spec = normalize({"references": [{"id": "device", "path": str(device_files["stl"])}]})
    code = 'PARAMS = {"w": (40.0, "measured:device.extents.x"), "h": (12.0, "measured:device.bbox_max.z"), "q": (1.0, "measured:device.nothing.here"), "bench": (31.0, "measured:calipers on the bench")}'
    lr = lint_source(code)
    problems = check_sources(lr, spec)
    assert len(problems) == 2, problems
    assert any("PARAMS['h'] = 12.0 but measured:device.bbox_max.z measures 5.0" in p for p in problems)
    assert any("names nothing measurable" in p for p in problems)
    # untoleranced spec values (fit.0.min, printability.min_wall) are compared too
    spec2 = normalize({"references": [{"id": "device", "path": str(device_files["stl"])}], "fit": [{"type": "clearance", "ref": "device", "min": 0.5}], "printability": {"min_wall": 1.95}})
    lr = lint_source('PARAMS = {"c": (0.9, "spec:fit.0.min"), "w": (1.95, "spec:printability.min_wall")}')
    assert check_sources(lr, spec2) == ["PARAMS['c'] = 0.9 but spec:fit.0.min is 0.5"]
    # a feature reference resolves through the reference's feature report
    code = 'PARAMS = {"slot_w": (4.0, "measured:device.S01.width"), "slot_c": (0.0, "measured:device.S01.center.x")}'
    assert check_sources(lint_source(code), spec) == []


def test_sandbox_injects_references_and_keeps_error_lines(device_files, tmp_path):
    from calipers.sandbox import run_code

    spec = normalize({"references": [{"id": "device", "path": str(device_files["stl"])}]})
    code = textwrap.dedent(
        """
        from build123d import *
        from calipers import load
        PARAMS = {}
        ref = load(REFERENCES["device"])
        result = Box(ref.mesh.extents[0], ref.mesh.extents[1], ref.mesh.extents[2])
        """
    )
    rr = run_code(code, workdir=tmp_path / "r1", spec=spec, lint=False)
    assert rr.ok and rr.report["summary"]["extents"] == [40.0, 30.0, 10.0]
    rr = run_code("from build123d import *\nPARAMS = {}\nresult = Box(1, 1, 1) + undefined_name\n", workdir=tmp_path / "r2", spec=spec, lint=False)
    assert not rr.ok and rr.error["type"] == "NameError" and rr.error["line"] == 3 and "undefined_name" in rr.error["source_line"]


def test_best_of_n_ranks_by_the_verifier(device_files, tmp_path):
    from calipers import generators
    from calipers.sandbox import candidates_text, run_candidates
    from calipers.spec import load_spec

    res = generators.enclosure(device_files["stl"], "device", wall=2.0, clearance=0.5, out_dir=tmp_path / "gen")
    good = res["code"]
    tight = good.replace("(0.5, 'spec:fit.0.min')", "(0.2, 'spec:fit.0.min')")
    assert tight != good
    broken = good + "\nresult = None\n"
    ranked = run_candidates([tight, broken, good], spec=load_spec(res["paths"]["spec"]), workdir=tmp_path / "best", workers=3)
    assert [i for i, _ in ranked] == [2, 0, 1]
    assert ranked[0][1].passed and not ranked[1][1].passed and not ranked[2][1].ok
    txt = candidates_text(ranked)
    assert txt.startswith("# Best-of-3: candidate 2 is best and passes everything")
    assert "candidate 0:" in txt and "lint problem" in txt


def test_diff_between_two_revisions_is_kernel_exact(bracket_files, tmp_path):
    from build123d import Cylinder, Pos, export_step

    from calipers.diff import diff_models, diff_text
    from calipers.sandbox import run_code
    from tests.conftest import BRACKET, build_bracket

    # a revision: one more Ø6 hole at (0, 12) through the 5 mm plate
    rev = build_bracket() - Pos(0, 12, 0) * Cylinder(3.0, BRACKET["plate"][2])
    export_step(rev, str(tmp_path / "rev.step"))
    d = diff_models(load(bracket_files["step"]), load(tmp_path / "rev.step"))
    assert not d["identical"] and d["exact"]
    assert abs(d["material"]["removed_volume"] - math.pi * 3.0**2 * BRACKET["plate"][2]) < 1e-3 and d["material"]["added_volume"] < 1e-6
    assert d["cylinders"]["unchanged"] == 3 and len(d["cylinders"]["added"]) == 1 and d["cylinders"]["added"][0]["diameter"] == 6.0
    assert d["planes"]["removed"] == [] and d["planes"]["added"] == []
    txt = diff_text(d)
    assert "added through_hole" in txt and "Material removed 141.3717" in txt
    same = diff_models(load(bracket_files["step"]), load(bracket_files["stl"]))
    assert same["cylinders"]["unchanged"] == 3 and not same["exact"]
    # run_code diffs against the previous run in the same workdir
    code = "from build123d import *\nPARAMS = {}\nresult = Box(10, 10, 10)\n"
    run_code(code, workdir=tmp_path / "w", lint=False)
    rr = run_code(code.replace("Box(10, 10, 10)", "Box(10, 10, 12)"), workdir=tmp_path / "w", lint=False)
    assert rr.diff and not rr.diff["identical"] and abs(rr.diff["material"]["added_volume"] - 200.0) < 1e-6 and "previous run → this run" in rr.diff_text


@pytest.mark.timeout(600)
def test_benchy_enclosure_exit_criterion(benchy_file, tmp_path):
    """Phase 3 exit criterion: an enclosure for an imported organic, non-watertight mesh passes its fit
    contracts unattended — generated, executed, linted and verified without a human in the loop."""
    from calipers import generators
    from calipers.sandbox import run_code
    from calipers.spec import load_spec

    res = generators.enclosure(benchy_file, "benchy", wall=2.0, clearance=0.6, open_face="+z", out_dir=tmp_path / "benchy")
    rr = run_code(res["code"], workdir=tmp_path / "benchy" / "run", spec=load_spec(res["paths"]["spec"]))
    assert rr.passed, rr.to_text()
    by = {c["id"]: c for c in rr.verify["checks"]}
    # published Benchy nominals: 60.0 long (x), 31.0 wide (y), 48.0 tall (z)
    assert abs(by["envelope.x"]["measured"] - (60.0 + 1.2 + 4.0)) < 0.3 and abs(by["envelope.y"]["measured"] - (31.0 + 1.2 + 4.0)) < 0.3 and abs(by["envelope.z"]["measured"] - (48.0 + 1.2 + 2.0)) < 0.3
    assert abs(by["fit.0.clearance.min_distance"]["measured"] - 0.6) < 0.01
    assert by["fit.2.enclosed"]["measured"] >= 0.95


def test_render_splits_long_triangles_before_painter_sort():
    """Review finding (Tucker, 15 Sep): the Benchy looked as if it poked through the enclosure wall.
    The geometry was fine (contracts exact); the painter's sort used triangle centroids, and a 60 mm
    wall face is two triangles whose centroids sit behind the boat. Long edges are now split first."""
    import trimesh

    from calipers.render import MAX_EDGE_FRACTION, _painter_ready

    box = trimesh.creation.box(extents=(60, 40, 50))
    ready = _painter_ready(box)
    edges = ready.vertices[ready.edges_unique]
    longest = float(np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max())
    assert longest <= MAX_EDGE_FRACTION * float(np.linalg.norm(box.extents)) + 1e-9
    assert abs(ready.volume - box.volume) < 1e-6 and ready.is_watertight
