"""Measurement primitives — the calipers.

Every function here returns plain dicts of numbers that are safe to hand to an LLM. Each result
carries an ``"exact"`` flag: True when it was evaluated on analytic B-rep geometry, False when it was
estimated from a triangle mesh (in which case a residual or tolerance is reported alongside).

Conventions: units are the model's units (mm by default); vectors are ``[x, y, z]`` lists; angles are
degrees; everything is rounded to ``PRECISION`` decimals so reports are stable and diffable.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import trimesh

from calipers.model import Model

PRECISION = 4  # decimals kept in reports (0.1 µm) — well below any manufacturing tolerance


# ----------------------------------------------------------------------------- small helpers
def r(x: Any, p: int = PRECISION) -> Any:
    """Round floats (recursively through lists/arrays) for stable, readable reports."""
    if isinstance(x, (float, np.floating)):
        v = round(float(x), p)
        return 0.0 if v == 0 else v  # normalise -0.0
    if isinstance(x, (np.ndarray, list, tuple)):
        return [r(v, p) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_between_deg(a: Sequence[float], b: Sequence[float]) -> float:
    ua, ub = unit(a), unit(b)
    return float(math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))


def plane_frame(normal: Sequence[float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """In-plane axes (u, v) for a section plane, using world axes whenever the normal is one.

    For a Z-normal plane u=X, v=Y; for Y: u=X, v=Z; for X: u=Y, v=Z — so 2D section coordinates are
    plain world coordinates. Oblique normals get an arbitrary right-handed frame.
    """
    n = unit(normal)
    names = "xyz"
    for k in range(3):
        if abs(abs(float(n[k])) - 1.0) < 1e-9:
            others = [i for i in range(3) if i != k]
            u = np.eye(3)[others[0]]
            v = np.eye(3)[others[1]]
            if float(np.cross(u, v) @ n) < 0:  # keep (u, v, n) right-handed
                u, v = v, u
                others = others[::-1]
            return u, v, [names[others[0]], names[others[1]]]
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(helper, n))
    v = np.cross(n, u)
    return u, v, ["u", "v"]


def parse_plane(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``"z=12"``, ``"x=-3.5"``, ``"y=0"`` into (origin, normal)."""
    axis, _, val = spec.replace(" ", "").partition("=")
    if axis.lower() not in {"x", "y", "z"} or val == "":
        raise ValueError("plane spec must look like 'z=12', 'x=-3.5' or 'y=0'")
    idx = "xyz".index(axis.lower())
    origin = np.zeros(3)
    origin[idx] = float(val)
    normal = np.zeros(3)
    normal[idx] = 1.0
    return origin, normal


# ----------------------------------------------------------------------------- 2D fits
def fit_circle_2d(pts: np.ndarray) -> dict:
    """Algebraic (Kåsa) least-squares circle fit. Returns centre, radius and RMS radial residual."""
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    cx, cy = -D / 2.0, -E / 2.0
    rad2 = cx**2 + cy**2 - F
    if rad2 <= 0:
        return {"center": [float(cx), float(cy)], "radius": 0.0, "rms": float("inf")}
    rad = math.sqrt(rad2)
    resid = np.hypot(x - cx, y - cy) - rad
    return {"center": [float(cx), float(cy)], "radius": rad, "rms": float(np.sqrt(np.mean(resid**2)))}


def fit_rect_2d(polygon) -> dict:
    """Minimum-area rotated rectangle around a shapely polygon and how well it matches."""
    rect = polygon.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords)[:-1]
    e1 = coords[1] - coords[0]
    e2 = coords[2] - coords[1]
    w, h = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    long_edge = e1 if w >= h else e2  # angle is that of the longer side, in [0, 180)
    ang = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180.0
    if abs(ang - 180.0) < 1e-6:
        ang = 0.0
    fill = float(polygon.area / rect.area) if rect.area > 0 else 0.0
    size = sorted([w, h], reverse=True)
    return {"size": size, "angle_deg": ang, "fill_ratio": fill}


# ----------------------------------------------------------------------------- global properties
def summary(model: Model) -> dict:
    """Envelope, mass properties, validity and topology counts."""
    mesh = model.mesh
    out: dict[str, Any] = {"exact": model.is_exact, "units": model.units}
    if model.is_exact:
        s = model.shape
        bb = s.bounding_box()
        out.update(
            {
                "kind": "brep",
                "valid": bool(s.is_valid),
                "solids": len(s.solids()),
                "bbox_min": r([bb.min.X, bb.min.Y, bb.min.Z]),
                "bbox_max": r([bb.max.X, bb.max.Y, bb.max.Z]),
                "extents": r([bb.size.X, bb.size.Y, bb.size.Z]),
                "volume": r(s.volume),
                "surface_area": r(s.area),
                "center_of_mass": r(list(tuple(s.center()))),
                "faces": len(s.faces()),
                "edges": len(s.edges()),
                "vertices": len(s.vertices()),
                "face_types": _face_type_counts(s),
            }
        )
    else:
        out.update(
            {
                "kind": "mesh",
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "bodies": int(mesh.body_count),
                "bbox_min": r(mesh.bounds[0]),
                "bbox_max": r(mesh.bounds[1]),
                "extents": r(mesh.extents),
                "volume": r(mesh.volume) if mesh.is_watertight else None,
                "volume_note": None if mesh.is_watertight else "not watertight; volume unreliable",
                "surface_area": r(mesh.area),
                "center_of_mass": r(mesh.center_mass if mesh.is_watertight else mesh.centroid),
                "triangles": int(len(mesh.faces)),
                "vertices": int(len(mesh.vertices)),
            }
        )
    out["oriented_bbox"] = oriented_bbox(model)
    out["principal_axes"] = principal_axes(model)
    return out


def _face_type_counts(shape) -> dict:
    counts: dict[str, int] = {}
    for f in shape.faces():
        k = str(f.geom_type).split(".")[-1].lower()
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


MAX_OBB_CANDIDATES = 300


def _min_area_rect(p2: np.ndarray, prefer_axes: bool = False):
    """Rotating-calipers minimum-area rectangle of a 2D point set.

    Returns (w, h, e1, e2, center) with e1/e2 unit edge directions, or None when degenerate. With
    ``prefer_axes`` the axis-aligned rectangle is tried first and only beaten by a real improvement.
    """
    from scipy.spatial import ConvexHull

    if len(p2) < 3:
        return None
    try:
        hull = ConvexHull(p2)
    except Exception:
        return None
    hp = p2[hull.vertices]
    edges = np.roll(hp, -1, axis=0) - hp
    lengths = np.linalg.norm(edges, axis=1)
    edges = edges[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    if len(edges) == 0:
        return None
    # each hull edge direction is a candidate orientation
    angles = list(np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % np.pi, 6)))
    if prefer_axes:
        angles = [0.0] + angles
    best = None
    for a in angles:
        e1 = np.array([math.cos(a), math.sin(a)])
        e2 = np.array([-e1[1], e1[0]])
        x, y = hp @ e1, hp @ e2
        w, h = float(x.max() - x.min()), float(y.max() - y.min())
        area = w * h
        if best is None or area < best[0] * (1.0 - 1e-4):
            center = e1 * 0.5 * (x.max() + x.min()) + e2 * 0.5 * (y.max() + y.min())
            best = (area, w, h, e1, e2, center)
    _, w, h, e1, e2, center = best
    return w, h, e1, e2, center


def oriented_bbox(model: Model) -> dict:
    """Minimum-volume oriented bounding box; exposes the part's natural frame even when it is rotated.

    Candidate frames are the convex hull's face normals (one box axis) combined with the minimum-area
    rectangle of the hull projected along that normal (the other two). This is the classic exact
    approach for polyhedra up to the 2D rectangle step and is robust where heuristic OBBs are not.
    """
    hull = model.mesh.convex_hull
    pts = np.asarray(hull.vertices, dtype=float)
    normals = np.asarray(hull.face_normals, dtype=float)
    areas = np.asarray(hull.area_faces, dtype=float)
    # candidate box normals: world axes first (ties resolve to the authored frame), then the
    # largest distinct hull facets and the principal axes
    _, first = np.unique(np.round(normals, 2), axis=0, return_index=True)
    first = first[np.argsort(-areas[first])][:MAX_OBB_CANDIDATES]
    cands = [np.eye(3)[i] for i in range(3)] + [unit(n) for n in normals[first]]
    cands += [unit(v) for v in np.asarray(principal_axes(model)["vectors"], dtype=float)]

    best: Optional[tuple] = None
    for k, n in enumerate(cands):
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        if k < 3:  # world axis: use the other two world axes as the in-plane frame
            u, v = np.roll(np.eye(3), -k - 1, axis=0)[:2]
        else:
            u = unit(np.cross(n, helper))
            v = np.cross(n, u)
        p2 = np.column_stack([pts @ u, pts @ v])
        rect = _min_area_rect(p2, prefer_axes=(k < 3))
        if rect is None:
            continue
        w, h, e1, e2, center2 = rect
        d = pts @ n
        depth = float(d.max() - d.min())
        vol = w * h * depth
        # switch only for a real improvement (1e-4 relative) so tessellation noise cannot tilt an
        # axis-aligned part into a fake "rotated" frame
        if best is None or vol < best[0] * (1.0 - 1e-4):
            a1 = unit(e1[0] * u + e1[1] * v)
            a2 = unit(e2[0] * u + e2[1] * v)
            center = center2[0] * u + center2[1] * v + n * 0.5 * (d.max() + d.min())
            best = (vol, [w, h, depth], [a1, a2, n], center)
    if best is None:  # degenerate hull; fall back to the axis-aligned box
        lo, hi = model.mesh.bounds
        return {"extents": r(hi - lo), "axes": r(np.eye(3)), "center": r((lo + hi) / 2), "yaw_of_longest_axis_deg": 0.0, "axis_aligned": True}
    _, ext, axes, center = best
    order = np.argsort(-np.asarray(ext))
    ext = [ext[i] for i in order]
    axes = [np.asarray(axes[i]) for i in order]
    # yaw is measured on the box axis that lies flattest in XY (a tall box's longest axis may be Z)
    flat = min(axes, key=lambda a: abs(float(a[2])))
    yaw = math.degrees(math.atan2(flat[1], flat[0])) % 180.0 if abs(float(flat[2])) < 0.999 else 0.0
    aligned = all(np.isclose(np.abs(a).max(), 1.0, atol=1e-3) for a in axes)
    return {
        "extents": r(ext),
        "axes": r(axes),
        "center": r(center),
        "yaw_of_longest_axis_deg": r(yaw, 2),
        # smallest rotation about Z that would align the box with the world axes (0 for aligned parts)
        "rotation_about_z_deg": r(min(yaw % 90.0, 90.0 - (yaw % 90.0)), 2),
        "axis_aligned": bool(aligned),
    }


def principal_axes(model: Model) -> dict:
    mesh = model.mesh
    if mesh.is_watertight:
        vecs = np.asarray(mesh.principal_inertia_vectors)
        vals = np.asarray(mesh.principal_inertia_components)
        return {"method": "inertia", "vectors": r(vecs), "moments": r(vals, 2)}
    pts = mesh.vertices - mesh.vertices.mean(axis=0)
    _, s, vt = np.linalg.svd(pts, full_matrices=False)
    return {"method": "vertex_pca", "vectors": r(vt), "moments": r(s**2 / len(pts), 2)}


# ----------------------------------------------------------------------------- sections
def section(model: Model, origin: Sequence[float], normal: Sequence[float], circle_tol: float = 0.02) -> dict:
    """Cross-section by a plane: closed loops classified as outer/inner with circle and rectangle fits.

    Loop points are taken from the tessellation so they lie *on* the true surface; the circle fit
    residual therefore stays tiny for genuine circles even on coarse meshes (chord sag affects area,
    not vertex placement). ``circle_tol`` is an absolute RMS residual floor in model units; the
    effective threshold is ``max(circle_tol, 0.5% of radius)``.
    """
    mesh = model.mesh
    origin = np.asarray(origin, dtype=float)
    normal = unit(normal)
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    u, v, axes_names = plane_frame(normal)
    result: dict[str, Any] = {
        "plane": {"origin": r(origin), "normal": r(normal), "in_plane_axes": axes_names, "u": r(u), "v": r(v)},
        "exact": False,
        "loops": [],
    }
    if sec is None:
        result["note"] = "plane does not intersect the model"
        return result
    # 2D coordinates are (p·u, p·v) — world coordinates along the in-plane axes, not an arbitrary frame
    T = np.eye(4)
    T[:3, :3] = np.vstack([u, v, normal])
    T[:3, 3] = -T[:3, :3] @ origin
    path2d, to3d = sec.to_2D(to_2D=T)
    loops: list[dict] = []
    for poly in path2d.polygons_full:
        loops.append(_loop_info(poly.exterior, "outer", poly, circle_tol, to3d))
        for hole in poly.interiors:
            from shapely.geometry import Polygon

            loops.append(_loop_info(hole, "inner", Polygon(hole.coords), circle_tol, to3d))
    # stable order: largest area first
    loops.sort(key=lambda L: -L["area"])
    for i, L in enumerate(loops):
        L["id"] = f"loop_{i:02d}"
    result["loops"] = loops
    result["n_outer"] = sum(1 for L in loops if L["type"] == "outer")
    result["n_inner"] = sum(1 for L in loops if L["type"] == "inner")
    return result


def _loop_info(ring, kind: str, poly, circle_tol: float, to3d: np.ndarray) -> dict:
    pts = np.asarray(ring.coords)[:-1]
    fit = fit_circle_2d(pts)
    thr = max(circle_tol, 0.005 * fit["radius"])
    is_circle = math.isfinite(fit["rms"]) and fit["rms"] <= thr and len(pts) >= 8
    rect = fit_rect_2d(poly)
    is_rect = rect["fill_ratio"] >= 0.985 and not is_circle
    c2 = np.array([fit["center"][0], fit["center"][1], 0.0, 1.0])
    c3 = (to3d @ c2)[:3]
    info = {
        "type": kind,
        "area": r(float(poly.area)),
        "perimeter": r(float(ring.length)),
        "centroid": r([float(poly.centroid.x), float(poly.centroid.y)]),
        "bbox_2d": r(list(poly.bounds)),
        "n_points": int(len(pts)),
        "circle": {
            "is_circle": bool(is_circle),
            "diameter": r(2 * fit["radius"]),
            "center_3d": r(c3),
            "rms_residual": r(fit["rms"]),
        },
        "rectangle": {"is_rectangle": bool(is_rect), **{k: r(v) for k, v in rect.items()}},
    }
    return info


def section_exact(model: Model, origin: Sequence[float], normal: Sequence[float]) -> Optional[dict]:
    """Exact planar section of a B-rep: edge types (line/circle/...) and circle radii read from OCCT.

    Returns None for mesh models. Loops are reported per section face with their wires' edge types;
    a wire made of a single closed CIRCLE edge (or several arcs with one radius) is an exact circle.
    """
    if not model.is_exact:
        return None
    from build123d import Face, Plane, Vector

    bb = model.shape.bounding_box()
    size = 4.0 * max(bb.size.X, bb.size.Y, bb.size.Z) + 10.0
    plane = Plane(origin=Vector(*origin), z_dir=Vector(*unit(normal)))
    cutter = Face.make_rect(size, size, plane)
    try:
        sec = model.shape.intersect(cutter)
    except Exception as exc:  # pragma: no cover - kernel edge cases
        return {"exact": True, "error": f"intersection failed: {exc}"}
    faces = sec.faces() if sec is not None else []
    loops = []
    for f in faces:
        wires = [f.outer_wire()] + list(f.inner_wires())
        for wi, w in enumerate(wires):
            edges = w.edges()
            kinds = [str(e.geom_type).split(".")[-1].lower() for e in edges]
            radii = sorted({round(e.radius, PRECISION) for e in edges if str(e.geom_type).endswith("CIRCLE")})
            all_circ = all(k == "circle" for k in kinds)
            entry = {
                "type": "outer" if wi == 0 else "inner",
                "n_edges": len(edges),
                "edge_types": sorted(set(kinds)),
                "is_circle": bool(all_circ and len(radii) == 1),
                "length": r(w.length),
            }
            if radii:
                entry["circle_radii"] = radii
            if entry["is_circle"]:
                c = edges[0].arc_center
                entry["diameter"] = r(2 * radii[0])
                entry["center_3d"] = r([c.X, c.Y, c.Z])
            loops.append(entry)
    return {"plane": {"origin": r(origin), "normal": r(unit(normal))}, "exact": True, "loops": loops}


# ----------------------------------------------------------------------------- distances & thickness
def closest_surface_points(model: Model, points: Iterable[Sequence[float]]) -> dict:
    pts = np.asarray(list(points), dtype=float).reshape(-1, 3)
    closest, dist, tri = trimesh.proximity.closest_point(model.mesh, pts)
    out = {"exact": False, "points": r(pts), "closest": r(closest), "distance": r(dist)}
    if model.mesh.is_watertight:
        inside = model.mesh.contains(pts)
        out["signed_distance"] = r(np.where(inside, -dist, dist))
    return out


def wall_thickness(model: Model, n_samples: int = 400, seed: int = 0) -> dict:
    """Sample the surface, shoot rays inward, report the distance to the opposite wall.

    Approximate by construction (mesh ray casting); useful for min-wall checks before printing.
    """
    mesh = model.mesh
    rng = np.random.default_rng(seed)
    pts, fidx = trimesh.sample.sample_surface(mesh, n_samples, seed=int(rng.integers(0, 2**31)))
    normals = mesh.face_normals[fidx]
    eps = 1e-3 * float(np.max(mesh.extents))
    origins = pts - normals * eps
    locs, ray_ids, _ = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    if len(ray_ids) == 0:
        return {"exact": False, "n_samples": n_samples, "note": "no inward hits (open mesh?)"}
    d = np.linalg.norm(locs - pts[ray_ids], axis=1)
    return {
        "exact": False,
        "n_samples": int(n_samples),
        "n_hits": int(len(d)),
        "min": r(float(d.min())),
        "p05": r(float(np.percentile(d, 5))),
        "median": r(float(np.median(d))),
        "max": r(float(d.max())),
    }


def mirror_symmetry(model: Model, n_samples: int = 3000, tol: float = 0.05, seed: int = 0) -> dict:
    """Test mirror symmetry about the world axes and the principal planes.

    For each candidate plane, surface samples are mirrored and their distance to the surface
    measured. A plane is reported symmetric when p95 ≤ tol, p99 ≤ 3·tol and RMS ≤ tol — the tail
    conditions catch a single small asymmetric feature (an extra hole, a tab) that p95 alone
    would miss. Feature-level confirmation is added by :func:`calipers.report.build_report`.
    """
    mesh = model.mesh
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    com = mesh.center_mass if mesh.is_watertight else mesh.centroid
    # candidates: world axes first (models are usually authored axis-aligned), then principal axes
    cands = [np.eye(3)[i] for i in range(3)] + [unit(v) for v in np.asarray(principal_axes(model)["vectors"], dtype=float)]
    uniq: list[np.ndarray] = []
    for n in cands:
        if not any(abs(float(n @ u)) > math.cos(math.radians(2.0)) for u in uniq):
            uniq.append(n)

    search = pts[: max(500, n_samples // 3)]  # a cheaper subset for the offset search

    def deviation(n: np.ndarray, offset: float, P: np.ndarray = pts) -> tuple[float, float, float]:
        d = (P @ n) - offset
        mirrored = P - 2.0 * d[:, None] * n[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            _, dist, _ = trimesh.proximity.closest_point(mesh, mirrored)
        return float(np.percentile(dist, 95)), float(np.percentile(dist, 99)), float(np.sqrt(np.mean(dist**2)))

    planes = []
    for n in uniq:
        off0 = float(com @ n)
        p95, p99, rms = deviation(n, off0)
        best = (p95, p99, rms, off0)
        if p95 <= 20 * tol:  # close: refine the plane offset by a short golden-section search (±1 mm)
            lo, hi = off0 - 1.0, off0 + 1.0
            g = (math.sqrt(5) - 1) / 2
            a, b = hi - g * (hi - lo), lo + g * (hi - lo)
            fa, fb = deviation(n, a, search)[0], deviation(n, b, search)[0]
            for _ in range(8):
                if fa < fb:
                    hi, b, fb = b, a, fa
                    a = hi - g * (hi - lo)
                    fa = deviation(n, a, search)[0]
                else:
                    lo, a, fa = a, b, fb
                    b = lo + g * (hi - lo)
                    fb = deviation(n, b, search)[0]
            off_ref = a if fa < fb else b
            cand = deviation(n, off_ref) + (off_ref,)  # final evaluation on the full sample
            if cand[0] < best[0]:
                best = cand
        p95, p99, rms, off = best
        planes.append(
            {
                "plane_normal": r(n),
                "offset": r(off),
                "through": r(n * off),
                "p95_deviation": r(p95),
                "p99_deviation": r(p99),
                "rms_deviation": r(rms),
                "symmetric": bool(p95 <= tol and p99 <= 3 * tol and rms <= tol),
            }
        )
    planes.sort(key=lambda p: p["p95_deviation"])
    return {"exact": False, "tolerance": tol, "n_samples": n_samples, "planes": planes}


def feature_symmetry(features: dict, plane_normal: Sequence[float], offset: float, tol: float = 0.05) -> dict:
    """Confirm a mirror plane at feature level: every full cylinder must have a mirrored twin.

    Point sampling can miss a single small asymmetric feature; mirroring the *features* cannot.
    """
    n = unit(plane_normal)
    cyls = [c for c in features.get("cylinders", []) if c["kind"] not in {"partial", "fillet_candidate"}]
    unmatched = []
    for c in cyls:
        p = np.asarray(c["axis_point"])
        pm = p - 2.0 * (float(p @ n) - offset) * n
        d = np.asarray(c["axis_dir"])
        dm = d - 2.0 * float(d @ n) * n
        ok = False
        for o in cyls:
            if o["kind"] != c["kind"] or abs(o["diameter"] - c["diameter"]) > 0.02:
                continue
            if np.linalg.norm(np.asarray(o["axis_point"]) - pm) <= max(tol, 0.002 * c["diameter"]) and abs(float(np.asarray(o["axis_dir"]) @ dm)) > 0.9995:
                ok = True
                break
        if not ok:
            unmatched.append(c["id"])
    return {"checked": len(cyls), "unmatched": unmatched, "ok": not unmatched}
