"""Reference-conditioned design: keep-in / keep-out volumes and fit contracts against an imported part.

A spec may name *references* — existing models (a scanned device, a PCB, a motor) that the generated
part must fit around, onto or inside. Nothing about the reference is typed by hand: the verifier
loads it, places it, and measures the generated part against it.

Regions (``keep_out`` / ``keep_in``) are solids: an explicit box or cylinder, or a box *derived*
from a reference's envelope (``from: device, face: "+x", depth: 20`` is the volume a cable needs in
front of the device's +x face). Fit checks compare the two bodies:

* ``clearance`` — no overlap, and the closest approach between the surfaces is at least ``min``
  (exact for STEP vs STEP via OCCT, fitted from sampled points otherwise);
* ``gap`` — per direction, rays from the reference surface find the nearest part wall; the smallest
  gap must lie in ``[min, max]`` so the reference is held without rattling (sampled);
* ``enclosed`` — the fraction of the reference surface that sees part material along its outward
  normal, ignoring directions declared ``open`` (sampled).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh

from calipers.measure import r, unit
from calipers.model import Model

DIRECTIONS = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}
N_SURFACE_SAMPLES = 4000
MAX_SAMPLES = 12000
VOLUME_TOL = 1e-6  # mm³: exact booleans of touching solids leave slivers far below this
RAY_EPS = 1e-3  # mm: rays start this far behind the reference surface so a touching wall reads as gap 0


UNIT_SCALE = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4, "in": 25.4, "inches": 25.4}


def direction_vector(name: Any) -> np.ndarray:
    """``"+x"`` / ``"-z"`` / ``"y"`` (meaning +y) / a 3-vector (or its string form) → unit vector."""
    if isinstance(name, str):
        key = name.strip().lower()
        if key in {"x", "y", "z"}:
            key = "+" + key
        if key in DIRECTIONS:
            return np.asarray(DIRECTIONS[key], dtype=float)
        try:  # a vector written as text, e.g. "[1, 0, 0]" (spec keys are strings)
            import ast

            name = ast.literal_eval(name)
        except Exception as exc:
            raise ValueError(f"unknown direction {name!r} (use +x, -x, +y, -y, +z, -z or a 3-vector)") from exc
    v = np.asarray(name, dtype=float)
    if v.shape != (3,) or not np.linalg.norm(v) > 0:
        raise ValueError(f"bad direction {name!r}: need a non-zero 3-vector")
    return unit(v)


# ----------------------------------------------------------------------------- loading & placing
def load_reference(ref: dict, base_dir: Optional[str | Path] = None) -> Model:
    """Load a spec ``references`` entry and apply its placement (rotate about world axes, then translate)."""
    path = Path(ref["path"])
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    units = str(ref.get("units", "mm")).lower()
    if units not in UNIT_SCALE:
        raise ValueError(f"reference {ref['id']!r}: unknown units {units!r} (choose from {sorted(UNIT_SCALE)})")
    model = Model.load(path, units="mm")
    if UNIT_SCALE[units] != 1.0:  # bring the reference into the spec's millimetre frame
        model = scale_model(model, UNIT_SCALE[units])
        model.notes.append(f"reference scaled ×{UNIT_SCALE[units]} from {units} to mm")
    place = ref.get("place") or {}
    rot = place.get("rotate", [0.0, 0.0, 0.0])
    tr = place.get("translate", [0.0, 0.0, 0.0])
    if any(abs(a) > 0 for a in rot) or any(abs(t) > 0 for t in tr):
        model = place_model(model, rot, tr)
        model.notes.append(f"placed: rotate {list(map(float, rot))}° about x, y, z then translate {list(map(float, tr))}")
    model.name = ref["id"]
    return model


def scale_model(model: Model, factor: float) -> Model:
    if model.is_exact:
        out = Model.from_shape(model.shape.scale(float(factor)), name=model.name, source_path=model.source_path)
    else:
        mesh = model.mesh.copy()
        mesh.apply_scale(float(factor))
        out = Model.from_mesh(mesh, name=model.name, source_path=model.source_path)
    out.units = "mm"
    out.notes = list(model.notes)
    return out


def place_model(model: Model, rotate_deg=(0.0, 0.0, 0.0), translate=(0.0, 0.0, 0.0)) -> Model:
    T = np.eye(4)
    for k, ang in enumerate(rotate_deg):
        if ang:
            T = trimesh.transformations.rotation_matrix(math.radians(float(ang)), np.eye(3)[k]) @ T
    T[:3, 3] += np.asarray(translate, dtype=float)
    if model.is_exact:
        from build123d import Axis, Location

        shape = model.shape
        for k, ang in enumerate(rotate_deg):
            if ang:
                shape = shape.rotate(Axis((0, 0, 0), tuple(np.eye(3)[k])), float(ang))
        shape = shape.moved(Location(tuple(map(float, translate))))
        out = Model.from_shape(shape, name=model.name, source_path=model.source_path)
        out.notes = list(model.notes)
        return out
    mesh = model.mesh.copy()
    mesh.apply_transform(T)
    out = Model.from_mesh(mesh, name=model.name, source_path=model.source_path)
    out.units = model.units
    out.notes = list(model.notes)
    return out


# ----------------------------------------------------------------------------- regions
def region_box(region: dict, refs: dict[str, Model]) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned box (min, max) of a region spec, or None when it is a cylinder."""
    if "cylinder" in region:
        return None
    if "box" in region:
        b = region["box"]
        if "min" in b:
            lo, hi = np.asarray(b["min"], float), np.asarray(b["max"], float)
        else:
            c, s = np.asarray(b["center"], float), np.asarray(b["size"], float)
            lo, hi = c - s / 2, c + s / 2
        return lo, hi
    ref = refs[region["from"]]
    lo, hi = np.asarray(ref.mesh.bounds, dtype=float)
    pad = float(region.get("pad", 0.0))
    if "face" in region:
        d = direction_vector(region["face"])
        k = int(np.argmax(np.abs(d)))
        depth = float(region["depth"])
        lo2, hi2 = lo - pad, hi + pad
        if d[k] > 0:
            lo2[k], hi2[k] = hi[k], hi[k] + depth
        else:
            lo2[k], hi2[k] = lo[k] - depth, lo[k]
        return lo2, hi2
    return lo - pad, hi + pad


def region_shape(region: dict, refs: dict[str, Model]):
    """The region as a build123d solid."""
    from build123d import Box, Plane, Pos, Solid, Vector

    if "cylinder" in region:
        c = region["cylinder"]
        a, b = np.asarray(c["from"], float), np.asarray(c["to"], float)
        L = float(np.linalg.norm(b - a))
        return Solid.make_cylinder(float(c["radius"]), L, Plane(origin=Vector(*a), z_dir=Vector(*unit(b - a))))
    lo, hi = region_box(region, refs)
    size = hi - lo
    return Pos(*((lo + hi) / 2)) * Box(float(size[0]), float(size[1]), float(size[2]))


def region_mesh(region: dict, refs: dict[str, Model]) -> trimesh.Trimesh:
    if "cylinder" in region:
        c = region["cylinder"]
        a, b = np.asarray(c["from"], float), np.asarray(c["to"], float)
        L = float(np.linalg.norm(b - a))
        T = trimesh.geometry.align_vectors([0, 0, 1], unit(b - a))
        T[:3, 3] = (a + b) / 2
        return trimesh.creation.cylinder(radius=float(c["radius"]), height=L, sections=128, transform=T)
    lo, hi = region_box(region, refs)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def describe_region(region: dict, refs: dict[str, Model]) -> str:
    if "cylinder" in region:
        c = region["cylinder"]
        return f"cylinder r{c['radius']} from {c['from']} to {c['to']}"
    lo, hi = region_box(region, refs)
    return f"box {r(lo, 3)} → {r(hi, 3)}"


def _mesh_boolean_volume(a: trimesh.Trimesh, b: trimesh.Trimesh, op: str) -> Optional[float]:
    """Volume of a ∩ b or a − b via manifold; None when either mesh is not a closed solid."""
    if not (a.is_watertight and b.is_watertight):
        return None
    try:
        fn = trimesh.boolean.intersection if op == "intersection" else trimesh.boolean.difference
        with np.errstate(invalid="ignore", divide="ignore"):
            res = fn([a, b], engine="manifold")
            if res is None or res.is_empty or len(res.faces) == 0:
                return 0.0
            return float(abs(res.volume))
    except BaseException:  # manifold raises its own error types
        return None


def _shape_volume(shape) -> float:
    """Volume of a build123d result: None (empty boolean), a Shape, or a ShapeList (compound inputs)."""
    if shape is None:
        return 0.0
    if isinstance(shape, (list, tuple)):
        return float(sum(_shape_volume(x) for x in shape))
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, g)
    return float(g.Mass())


def boolean_volume(a, b, op: str) -> float:
    """Kernel-exact volume of ``a ∩ b`` (op "common") or ``a − b`` (op "cut") for build123d shapes,
    computed with OCCT directly so imported compounds behave like solids."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    algo = (BRepAlgoAPI_Common if op == "common" else BRepAlgoAPI_Cut)(a.wrapped, b.wrapped)
    algo.Build()
    if not algo.IsDone():
        raise RuntimeError(f"boolean {op} failed")
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(algo.Shape(), g)
    return abs(float(g.Mass()))


def region_intrusion(model: Model, region: dict, refs: dict[str, Model], mode: str = "keep_out") -> dict:
    """Volume of the model inside a keep-out region, or outside a keep-in region. Exact for B-reps."""
    if model.is_exact:
        shape = region_shape(region, refs)
        try:
            return {"volume": r(boolean_volume(model.shape, shape, "common" if mode == "keep_out" else "cut")), "exact": True}
        except Exception as exc:  # pragma: no cover - kernel edge cases
            return {"volume": None, "exact": True, "error": str(exc)}
    rm = region_mesh(region, refs)
    vol = _mesh_boolean_volume(model.mesh, rm, "intersection" if mode == "keep_out" else "difference")
    if vol is not None:
        return {"volume": r(vol), "exact": False}
    # open mesh: count vertices on the wrong side instead of a volume
    pts = model.mesh.vertices
    inside = rm.contains(pts)
    bad = int(inside.sum()) if mode == "keep_out" else int((~inside).sum())
    return {"volume": None, "exact": False, "vertices_violating": bad, "note": "model mesh is not closed; counted vertices instead of a volume"}


# ----------------------------------------------------------------------------- fit checks
def _surface_samples(model: Model, n: int = N_SURFACE_SAMPLES, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Surface points with outward normals: the mesh vertices plus area-uniform samples (capped)."""
    mesh = model.mesh
    pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    P = np.vstack([mesh.vertices, pts])
    N = np.vstack([mesh.vertex_normals, mesh.face_normals[fidx]])
    if len(P) > MAX_SAMPLES:
        idx = np.random.default_rng(seed).choice(len(P), size=MAX_SAMPLES, replace=False)
        P, N = P[idx], N[idx]
    return np.asarray(P, float), np.asarray(N, float)


def clearance(model: Model, ref: Model) -> dict:
    """Closest approach between the part and the reference, and whether they overlap."""
    out: dict[str, Any] = {}
    if model.is_exact and ref.is_exact:
        try:
            out["overlap_volume"] = r(boolean_volume(model.shape, ref.shape, "common"))
            out["min_distance"] = r(float(model.shape.distance_to(ref.shape)))
            out["exact"] = True
            return out
        except Exception:  # pragma: no cover - fall back to the sampled path
            pass
    pm, rm = model.mesh, ref.mesh
    out["exact"] = False
    vol = _mesh_boolean_volume(pm, rm, "intersection")
    rp, _ = _surface_samples(ref)
    if vol is not None:
        out["overlap_volume"] = r(vol)
    if pm.is_watertight:  # a generated part is always closed; the reference (a scan) may not be
        inside = pm.contains(rp)
        out["reference_points_inside_part"] = int(inside.sum())
        out["reference_points_checked"] = int(len(rp))
    pp, _ = _surface_samples(model)
    if vol is None:  # open reference: a thin intrusion can slip between its surface samples, so also
        # test the part's own surface points against the reference (nearest-face signed distance)
        if rm.is_winding_consistent:
            with np.errstate(invalid="ignore", divide="ignore"):
                sd = trimesh.proximity.signed_distance(rm, pp)
            out["part_points_inside_reference"] = int((sd > 1e-6).sum())
            out["part_points_checked"] = int(len(pp))
        out["note_overlap"] = "reference mesh is not closed: overlap is sampled both ways, not a volume; add keep_out: {from: <ref>} for a hard guarantee"
    with np.errstate(invalid="ignore", divide="ignore"):
        _, d1, _ = trimesh.proximity.closest_point(pm, rp)
        _, d2, _ = trimesh.proximity.closest_point(rm, pp)
    out["min_distance"] = r(float(min(d1.min(), d2.min())))
    out["note"] = "sampled: distances from reference points to the part surface and back"
    return out


def overlaps(c: dict) -> bool:
    if c.get("overlap_volume") is not None and c["overlap_volume"] > VOLUME_TOL:
        return True
    return c.get("reference_points_inside_part", 0) > 0 or c.get("part_points_inside_reference", 0) > 0


def directional_gaps(model: Model, ref: Model, directions: list[np.ndarray], max_distance: float = 1e4) -> list[dict]:
    """For each direction, the nearest part wall seen from the reference surface along that direction."""
    P, N = _surface_samples(ref)
    pm = model.mesh
    out = []
    for d in directions:
        d = unit(d)
        sel = (N @ d) > 0.5  # samples whose outward normal faces the direction
        if sel.sum() < 3:
            sel = (N @ d) > 0.0
        orig = P[sel]
        if len(orig) == 0:
            out.append({"direction": r(d), "n_rays": 0, "hit_fraction": 0.0, "min": None, "p50": None, "max": None})
            continue
        # start a hair behind the surface so a part face touching the reference (gap 0) is still hit
        locs, ridx, _ = pm.ray.intersects_location(orig - RAY_EPS * d, np.tile(d, (len(orig), 1)), multiple_hits=False)
        if len(ridx) == 0:
            out.append({"direction": r(d), "n_rays": int(len(orig)), "hit_fraction": 0.0, "min": None, "p50": None, "max": None})
            continue
        dist = np.maximum(np.linalg.norm(locs - (orig[ridx] - RAY_EPS * d), axis=1) - RAY_EPS, 0.0)
        dist = dist[dist <= max_distance]
        if len(dist) == 0:
            out.append({"direction": r(d), "n_rays": int(len(orig)), "hit_fraction": 0.0, "min": None, "p50": None, "max": None})
            continue
        out.append(
            {
                "direction": r(d),
                "n_rays": int(len(orig)),
                "hit_fraction": r(float(len(dist) / len(orig)), 3),
                "min": r(float(dist.min())),
                "p50": r(float(np.median(dist))),
                "max": r(float(dist.max())),
            }
        )
    return out


def enclosure_coverage(model: Model, ref: Model, open_dirs: list[np.ndarray], max_distance: float = 50.0) -> dict:
    """Fraction of the reference surface that sees part material along its outward normal."""
    P, N = _surface_samples(ref)
    keep = np.ones(len(P), dtype=bool)
    for d in open_dirs:
        keep &= (N @ unit(d)) <= 0.5
    orig, D = P[keep], N[keep]
    if len(orig) == 0:
        return {"fraction": None, "n_rays": 0, "note": "every sample faces an open direction"}
    pm = model.mesh
    locs, ridx, _ = pm.ray.intersects_location(orig - RAY_EPS * D, D, multiple_hits=False)
    hit = np.zeros(len(orig), dtype=bool)
    if len(ridx):
        dist = np.linalg.norm(locs - orig[ridx], axis=1)
        hit[ridx[dist <= max_distance]] = True
    return {"fraction": r(float(hit.mean()), 3), "n_rays": int(len(orig)), "max_distance": max_distance, "exact": False}


def reference_summary(ref: Model) -> dict:
    """The numbers a designer needs from a reference, in its placed frame (for ``measured:`` sources)."""
    from calipers import measure

    s = measure.summary(ref)
    return {
        "id": ref.name,
        "exact": ref.is_exact,
        "bbox_min": s["bbox_min"],
        "bbox_max": s["bbox_max"],
        "extents": s["extents"],
        "center": r((np.asarray(s["bbox_min"]) + np.asarray(s["bbox_max"])) / 2),
        "volume": s.get("volume"),
        "center_of_mass": s.get("center_of_mass"),
        "watertight": s.get("watertight", True),
    }
