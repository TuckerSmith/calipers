"""Feature extraction: planes and cylinders (holes, bosses, pins, shafts, fillets) from B-reps and meshes.

Two code paths, one output schema:

* **B-rep** — feature parameters are *read* from OCCT surfaces (``exact: True``). Faces that OCCT split
  at a seam (a hole is often two 180° faces) are merged back into one feature.
* **Mesh** — triangles are grouped into smooth regions; each region is *peeled* into primitives:
  first a whole-region cylinder test, then planar patches, then RANSAC cylinders (which also catches
  fillets as partial cylinders). Results carry ``fit_rms`` and ``coverage_deg`` so a reader knows how
  good each fit is (``exact: False``).

Orientation semantics: a cylinder whose surface normals point *toward* its axis is concave (a hole
or bore); pointing *away* is convex (a boss, pin or shaft). Through/blind is decided by probing the
solid just past each end of the cylinder (next to the wall and on the axis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

from calipers.measure import fit_circle_2d, r, unit
from calipers.model import Model

# Angular tolerances (degrees) for grouping mesh triangles.
COPLANAR_DEG = 0.5  # triangles of one tessellated plane are coplanar to float precision
SMOOTH_DEG = 32.0  # adjacent triangles of a curved surface, even at slicer-grade tessellation

MIN_REGION_FACES = 4
MIN_REGION_AREA = 0.25  # mm² — below this a region is tessellation noise
MAX_REGIONS_PEELED = 400  # bound runtime on very large organic meshes
RANSAC_SAMPLES = 300
RANSAC_SCORE_FACES = 20_000  # hypotheses are scored on at most this many (area-weighted) faces
RNG_SEED = 7
MAX_SAMPLE_POINTS = 600  # points kept per cylinder for coverage recomputation after merges
NORMAL_AXIS_TOL = math.sin(math.radians(1.0))  # cones/tori have normals tilted along the axis
INLIER_AXIAL_TOL = math.sin(math.radians(2.0))  # per-face: keeps tangent fillet strips out of a cylinder


@dataclass
class Cylinder:
    radius: float
    axis_point: np.ndarray  # mid-height point on the axis
    axis_dir: np.ndarray  # unit vector, canonical sign
    half_height: float
    coverage_deg: float
    concave: Optional[bool]
    area: float
    fit_rms: float
    exact: bool
    source_faces: list[int] = field(default_factory=list)
    kind: str = "partial"
    open_ends: list = field(default_factory=list)
    sample_points: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def height(self) -> float:
        return 2.0 * self.half_height

    @property
    def start(self) -> np.ndarray:
        return self.axis_point - self.axis_dir * self.half_height

    @property
    def end(self) -> np.ndarray:
        return self.axis_point + self.axis_dir * self.half_height


@dataclass
class PlaneFeature:
    normal: np.ndarray
    offset: float  # normal · point
    area: float
    center: np.ndarray
    regions: int
    faces: int
    exact: bool
    bbox_min: np.ndarray
    bbox_max: np.ndarray


# ----------------------------------------------------------------------------- geometry helpers
def canonical_dir(d: np.ndarray) -> np.ndarray:
    """Flip a direction so its largest-magnitude component is positive (stable, diffable reports)."""
    d = unit(d)
    i = int(np.argmax(np.abs(d)))
    return -d if d[i] < 0 else d


def _basis_perp(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = unit(d)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(d, helper))
    v = np.cross(d, u)
    return u, v


def _angular_coverage_deg(pts: np.ndarray, axis_point: np.ndarray, d: np.ndarray) -> float:
    """Angular extent of sample points around an axis, robust to coarse tessellation.

    Distinct azimuths are sorted and the largest gap between neighbours is found. If that gap is
    clearly larger than the typical spacing there is a real opening and the coverage is 360° minus
    the gap; otherwise the surface wraps all the way round.
    """
    u, v = _basis_perp(d)
    rel = pts - axis_point
    ang = np.unique(np.round(np.degrees(np.arctan2(rel @ v, rel @ u)), 4))
    if len(ang) < 3:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    max_gap = float(gaps.max())
    typical = float(np.median(gaps))
    if max_gap <= max(3.0 * typical, 2.0):
        return 360.0
    return float(360.0 - max_gap)


def _subsample(pts: np.ndarray, n: int = MAX_SAMPLE_POINTS) -> np.ndarray:
    if len(pts) <= n:
        return np.array(pts, dtype=float)
    idx = np.linspace(0, len(pts) - 1, n).astype(int)
    return np.array(pts[idx], dtype=float)


def _make_cylinder(points: np.ndarray, d: np.ndarray, radius: float, center3: np.ndarray, rms: float) -> Cylinder:
    """Build a canonical Cylinder from an axis direction, a point on the axis and the sample points."""
    d = canonical_dir(d)
    t = (points - center3) @ d
    t_min, t_max = float(t.min()), float(t.max())
    axis_point = center3 + d * 0.5 * (t_min + t_max)
    return Cylinder(
        radius=float(radius),
        axis_point=axis_point,
        axis_dir=d,
        half_height=0.5 * (t_max - t_min),
        coverage_deg=_angular_coverage_deg(points, axis_point, d),
        concave=None,
        area=0.0,
        fit_rms=float(rms),
        exact=False,
        sample_points=_subsample(points),
    )


def fit_cylinder_lsq(points: np.ndarray, normals: np.ndarray, weights: np.ndarray) -> Optional[Cylinder]:
    """Least-squares cylinder: axis from the normal covariance, then a 2D circle fit in the axis plane."""
    if len(points) < 6 or len(normals) < 3:
        return None
    w = weights / max(float(weights.sum()), 1e-12)
    M = (normals * w[:, None]).T @ normals
    evals, evecs = np.linalg.eigh(M)  # ascending
    if evals[0] > 0.02 * max(evals[2], 1e-12) and evals[0] > 1e-4:
        return None  # normals are not all perpendicular to one axis
    if evals[1] < 0.01 * max(evals[2], 1e-12):
        return None  # normals barely rotate: a plane (or a sliver), not a cylinder
    d = unit(evecs[:, 0])
    u, v = _basis_perp(d)
    p2 = np.column_stack([points @ u, points @ v])
    fit = fit_circle_2d(p2)
    if not math.isfinite(fit["rms"]) or fit["radius"] <= 0:
        return None
    center3 = fit["center"][0] * u + fit["center"][1] * v
    return _make_cylinder(points, d, fit["radius"], center3, fit["rms"])


def _radial_residuals(points: np.ndarray, cyl: Cylinder) -> np.ndarray:
    rel = points - cyl.axis_point
    radial = rel - np.outer(rel @ cyl.axis_dir, cyl.axis_dir)
    return np.linalg.norm(radial, axis=1) - cyl.radius


def _concavity(face_centroids: np.ndarray, face_normals: np.ndarray, areas: np.ndarray, cyl: Cylinder) -> bool:
    rel = face_centroids - cyl.axis_point
    radial = rel - np.outer(rel @ cyl.axis_dir, cyl.axis_dir)
    s = np.einsum("ij,ij->i", face_normals, radial)
    return bool((s * areas).sum() < 0)  # normals point toward the axis → concave


def cylinder_tolerance(radius: float) -> float:
    """Radial RMS tolerance for accepting a cylinder fit: 0.5% of the radius, clamped to [0.03, 0.12] mm."""
    return float(np.clip(0.005 * radius, 0.03, 0.12))


# ----------------------------------------------------------------------------- mesh segmentation
def _components(n_faces: int, adjacency: np.ndarray, keep: np.ndarray, restrict: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """Connected components of faces over adjacency pairs flagged by ``keep``; optionally restricted to a subset."""
    if restrict is not None:
        mask = np.zeros(n_faces, dtype=bool)
        mask[restrict] = True
        keep = keep & mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
        nodes = np.asarray(restrict)
    else:
        nodes = np.arange(n_faces)
    edges = adjacency[keep]
    if len(edges) == 0:
        return [np.array([i]) for i in nodes]
    comps = trimesh.graph.connected_components(edges, nodes=nodes, min_len=1)
    return [np.asarray(c) for c in comps]


class _MeshCtx:
    """Precomputed per-mesh arrays shared by the peeling steps."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.adj = mesh.face_adjacency
        self.adj_ang = np.degrees(mesh.face_adjacency_angles)
        self.adj_len = np.linalg.norm(
            mesh.vertices[mesh.face_adjacency_edges[:, 0]] - mesh.vertices[mesh.face_adjacency_edges[:, 1]], axis=1
        )
        self.areas = mesh.area_faces
        self.normals = mesh.face_normals
        self.centers = np.asarray(mesh.triangles_center)
        self.tri = np.asarray(mesh.triangles)  # cached once: trimesh re-hashes big arrays on every access
        self.n = len(mesh.faces)
        self.diag = float(np.linalg.norm(mesh.extents))

    def verts_of(self, faces: np.ndarray) -> np.ndarray:
        return self.mesh.vertices[np.unique(self.mesh.faces[faces])]

    def boundary_smooth_fraction(self, faces: np.ndarray) -> float:
        """Length fraction of a face set's boundary whose dihedral angle is 'smooth' (tangent-ish)."""
        mask = np.zeros(self.n, dtype=bool)
        mask[faces] = True
        on_boundary = mask[self.adj[:, 0]] != mask[self.adj[:, 1]]
        if not on_boundary.any():
            return 0.0
        L = self.adj_len[on_boundary]
        smooth = self.adj_ang[on_boundary] < SMOOTH_DEG
        return float(L[smooth].sum() / max(L.sum(), 1e-12))


def _face_vertex_residuals(ctx: _MeshCtx, faces: np.ndarray, cyl: Cylinder) -> np.ndarray:
    """Radial residual of each corner of each face, shape (n, 3)."""
    tri = ctx.tri[faces].reshape(-1, 3)
    return _radial_residuals(tri, cyl).reshape(-1, 3)


def _centroid_allowance(ctx: _MeshCtx, faces: np.ndarray, cyl: Cylinder) -> np.ndarray:
    """How far inside the true cylinder a face centroid may legitimately sit because of chord sag.

    A triangle spanning an azimuth Δ has its centroid at radius r·sqrt(5/9 + 4/9·cos Δ) (two corners at
    one azimuth, one at the other). Faces spanning more than 45° are not treated as cylinder facets.
    """
    u, v = _basis_perp(cyl.axis_dir)
    tri = ctx.tri[faces] - cyl.axis_point  # (n, 3, 3)
    rad = np.stack([tri @ u, tri @ v], axis=-1)  # (n, 3, 2)
    rad /= np.maximum(np.linalg.norm(rad, axis=-1, keepdims=True), 1e-12)
    cos01 = np.einsum("ij,ij->i", rad[:, 0], rad[:, 1])
    cos02 = np.einsum("ij,ij->i", rad[:, 0], rad[:, 2])
    cos12 = np.einsum("ij,ij->i", rad[:, 1], rad[:, 2])
    cmin = np.clip(np.minimum(np.minimum(cos01, cos02), cos12), -1.0, 1.0)
    spread = np.arccos(cmin)
    sag = (1.0 - np.sqrt(5.0 / 9.0 + 4.0 / 9.0 * np.cos(spread))) * cyl.radius
    sag[spread > math.radians(45.0)] = 0.0
    return cylinder_tolerance(cyl.radius) + sag


def _cylinder_quality(ctx: _MeshCtx, faces: np.ndarray, cyl: Cylinder) -> Optional[float]:
    """Return the vertex radial RMS if ``faces`` are well described by ``cyl``; None otherwise.

    Checks: vertex radial residual, area-weighted face-centroid residual with a chord-sag allowance
    (so a rectangle's circumcircle cannot pass on vertices alone, yet slicer-grade tessellations
    still pass), normals perpendicular to the axis (so cones and tori are rejected), meaningful
    angular coverage and a sane radius.
    """
    tol = cylinder_tolerance(cyl.radius)
    pts = ctx.verts_of(faces)
    vres = _radial_residuals(pts, cyl)
    v_rms = float(math.sqrt(np.mean(vres**2)))
    if v_rms > tol or np.mean(np.abs(vres) <= 3 * tol) < 0.97:
        return None
    A = ctx.areas[faces]
    w = A / max(float(A.sum()), 1e-12)
    allow = _centroid_allowance(ctx, faces, cyl)
    cres = _radial_residuals(ctx.centers[faces], cyl)
    excess = np.maximum(np.abs(cres) - allow, 0.0)  # residual beyond what sag explains
    if math.sqrt(float((w * excess**2).sum())) > tol or float(w[excess <= 2 * tol].sum()) < 0.97:
        return None
    axial = np.abs(ctx.normals[faces] @ cyl.axis_dir)
    if math.sqrt(float((w * axial**2).sum())) > NORMAL_AXIS_TOL:
        return None  # not a cylinder: normals tilt along the axis (cone, torus, sphere)
    if cyl.coverage_deg < 20.0 or cyl.radius > 50.0 * ctx.diag:
        return None
    return v_rms


def _try_whole_cylinder(ctx: _MeshCtx, faces: np.ndarray) -> Optional[Cylinder]:
    pts = ctx.verts_of(faces)
    cyl = fit_cylinder_lsq(pts, ctx.normals[faces], ctx.areas[faces])
    if cyl is None:
        return None
    q = _cylinder_quality(ctx, faces, cyl)
    if q is None:
        return None
    cyl.fit_rms = q
    return cyl


def _inlier_faces(ctx: _MeshCtx, faces: np.ndarray, cyl: Cylinder) -> np.ndarray:
    """Faces of ``faces`` whose normals are ⟂ to the axis and whose corners all lie on the cylinder."""
    tol = cylinder_tolerance(cyl.radius)
    # a cylinder facet's normal is exactly ⟂ to the axis at any tessellation; the first strip of a
    # tangent fillet is tilted by half its angular step (≈ 3°), so a tight test keeps it out
    perp = np.abs(ctx.normals[faces] @ cyl.axis_dir) < INLIER_AXIAL_TOL
    vres = np.abs(_face_vertex_residuals(ctx, faces, cyl)).max(axis=1)
    return faces[perp & (vres <= tol)]


def _largest_component(ctx: _MeshCtx, faces: np.ndarray) -> Optional[np.ndarray]:
    if len(faces) == 0:
        return None
    comps = _components(ctx.n, ctx.adj, ctx.adj_ang < SMOOTH_DEG, restrict=faces)
    comp = max(comps, key=lambda c: ctx.areas[c].sum())
    if len(comp) < MIN_REGION_FACES or ctx.areas[comp].sum() < MIN_REGION_AREA:
        return None
    return comp


def _ransac_cylinder(ctx: _MeshCtx, faces: np.ndarray, rng: np.random.Generator) -> Optional[tuple[Cylinder, np.ndarray]]:
    """Find one cylinder inside a face set. Returns the cylinder and the inlier face indices (connected).

    Pairs are drawn with probability proportional to face area, so a wall of a few hundred large
    triangles is not starved by thousands of tiny fillet triangles next to it.
    """
    n = len(faces)
    if n < 2 * MIN_REGION_FACES:
        return None
    N = ctx.normals[faces]
    C = ctx.centers[faces]
    A = ctx.areas[faces]
    V = ctx.tri[faces]  # (n, 3, 3) corners: on the true surface regardless of tessellation
    p = A / A.sum()
    # score hypotheses on an area-weighted subset when the face set is huge (organic meshes)
    if n > RANSAC_SCORE_FACES:
        sub = np.sort(rng.choice(n, size=RANSAC_SCORE_FACES, replace=False, p=p))
    else:
        sub = np.arange(n)
    Ns, As, Vs = N[sub], A[sub], V[sub]
    candidates: list[tuple] = []  # (score, d, rad, center3)
    tries = min(RANSAC_SAMPLES, n * (n - 1) // 2)
    for _ in range(tries):
        i, j = rng.choice(n, size=2, replace=False, p=p)
        d = np.cross(N[i], N[j])
        nd = np.linalg.norm(d)
        if nd < math.sin(math.radians(3.0)):
            continue  # nearly parallel normals: axis ill-defined
        d /= nd
        u, v = _basis_perp(d)
        ci = np.array([C[i] @ u, C[i] @ v])
        cj = np.array([C[j] @ u, C[j] @ v])
        ni = np.array([N[i] @ u, N[i] @ v])
        nj = np.array([N[j] @ u, N[j] @ v])
        M = np.column_stack([-ni, nj])  # ci - s*ni = cj - t*nj
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        s, t = np.linalg.solve(M, cj - ci)
        if abs(s) < 1e-6 or (s > 0) != (t > 0):
            continue
        rad = abs(s)
        center2 = ci - s * ni
        # the seed pair sits at face centroids (inside the true surface by the chord sag); take the
        # candidate radius from the seed faces' own corners, which lie on the true surface
        Vij = np.stack([V[[i, j]] @ u, V[[i, j]] @ v], axis=-1) - center2  # seed corners (2, 3, 2)
        rad = float(np.median(np.linalg.norm(Vij, axis=-1)))
        perp = np.abs(Ns @ d) < INLIER_AXIAL_TOL
        V2 = np.stack([Vs @ u, Vs @ v], axis=-1) - center2  # (m, 3, 2)
        vres = np.abs(np.linalg.norm(V2, axis=-1) - rad).max(axis=1)
        inl = perp & (vres <= cylinder_tolerance(rad))
        score = float(As[inl].sum()) * (float(A.sum()) / max(float(As.sum()), 1e-12))  # scaled to the full set
        if score >= max(MIN_REGION_AREA, 0.01 * float(A.sum())) and inl.sum() >= min(MIN_REGION_FACES, 3):
            candidates.append((score, d, rad, center2 @ np.column_stack([u, v]).T))
    if not candidates:
        return None
    # try the best few seeds: the top scorer can be a bogus large circle grazing two parallel
    # fillets, which then fails refinement, while the next seed is the genuine cylinder
    candidates.sort(key=lambda c: -c[0])
    seen: list[tuple[float, np.ndarray]] = []
    for score, d, rad, center3 in candidates:
        if any(abs(rad - r0) < 0.02 * max(rad, 1.0) and abs(float(d @ d0)) > 0.999 for r0, d0 in seen):
            continue  # same hypothesis as one already tried
        seen.append((rad, d))
        if len(seen) > 6:
            break
        seed = _make_cylinder(ctx.verts_of(faces[:1]), d, rad, center3, 0.0)  # geometry only; extent irrelevant here
        comp = _largest_component(ctx, _inlier_faces(ctx, faces, seed))
        if comp is None:
            continue
        # refine by least squares on the component, re-collect inliers against the refined cylinder,
        # and refine once more (the RANSAC seed pair is exact only for two faces)
        cyl = fit_cylinder_lsq(ctx.verts_of(comp), ctx.normals[comp], ctx.areas[comp])
        if cyl is None:
            cyl = _make_cylinder(ctx.verts_of(comp), d, rad, center3, 0.0)
        for _ in range(2):
            comp2 = _largest_component(ctx, _inlier_faces(ctx, faces, cyl))
            if comp2 is None:
                break
            cyl2 = fit_cylinder_lsq(ctx.verts_of(comp2), ctx.normals[comp2], ctx.areas[comp2])
            if cyl2 is None:
                break
            comp, cyl = comp2, cyl2
        q = _cylinder_quality(ctx, comp, cyl)
        if q is None:
            continue
        cyl.fit_rms = q
        return cyl, comp
    return None


def _strip_patches(ctx: _MeshCtx, subs: list[np.ndarray], region_faces: np.ndarray) -> set[int]:
    """Indices of coplanar patches that are facet strips of a curved surface, not real planes.

    A strip has a mostly 'smooth' boundary *and* either a smooth neighbour patch of similar area
    (tessellation strips come in chains of siblings) or a negligible share of its region (a facet
    of a free-form surface). A real plane fully surrounded by fillets has a smooth boundary too,
    but its neighbours are tiny fillet strips and it carries real area, so it is kept.
    """
    comp_of = np.full(ctx.n, -1, dtype=np.int64)
    for k, sub in enumerate(subs):
        comp_of[sub] = k
    areas = np.array([ctx.areas[sub].sum() for sub in subs])
    region_area = float(ctx.areas[region_faces].sum())
    a, b = ctx.adj[:, 0], ctx.adj[:, 1]
    ca, cb = comp_of[a], comp_of[b]
    smooth_pairs = (ca >= 0) & (cb >= 0) & (ca != cb) & (ctx.adj_ang < SMOOTH_DEG)
    neighbours: dict[int, set[int]] = {}
    for i, j in zip(ca[smooth_pairs], cb[smooth_pairs]):
        neighbours.setdefault(int(i), set()).add(int(j))
        neighbours.setdefault(int(j), set()).add(int(i))
    strips: set[int] = set()
    for k, sub in enumerate(subs):
        sib = [j for j in neighbours.get(k, ()) if 0.4 <= areas[j] / max(areas[k], 1e-12) <= 2.5]
        if len(sib) >= 2:
            strips.add(k)  # a link in a chain of similar facets (cylinder strips, chamfer/cone facets)
            continue
        if ctx.boundary_smooth_fraction(sub) < 0.75:
            continue
        if sib or areas[k] < 0.02 * region_area:
            strips.add(k)
    return strips


def _peel_region(ctx: _MeshCtx, faces: np.ndarray, rng: np.random.Generator, planes: list, cyls: list, unclassified: list) -> None:
    # 1. whole region is a cylinder?
    cyl = _try_whole_cylinder(ctx, faces)
    if cyl is not None:
        _accept_cylinder(ctx, cyl, faces, cyls)
        return
    # 2. peel planar patches (a planar rectangle is only 2 triangles, so no face-count floor)
    remaining = faces
    subs = [s for s in _components(ctx.n, ctx.adj, ctx.adj_ang < COPLANAR_DEG, restrict=faces) if ctx.areas[s].sum() >= MIN_REGION_AREA]
    strips = _strip_patches(ctx, subs, faces)
    for k, sub in enumerate(subs):
        if k in strips:
            continue
        planes.append((sub, float(ctx.areas[sub].sum())))
        remaining = np.setdiff1d(remaining, sub, assume_unique=True)
    # 3. RANSAC cylinders on what is left (fillets, partial bores, ...)
    for _ in range(12):
        if len(remaining) < 2 * MIN_REGION_FACES or ctx.areas[remaining].sum() < MIN_REGION_AREA:
            break
        found = _ransac_cylinder(ctx, remaining, rng)
        if found is None:
            break
        cyl, comp = found
        _accept_cylinder(ctx, cyl, comp, cyls)
        remaining = np.setdiff1d(remaining, comp, assume_unique=True)
    # 4. leftovers
    if len(remaining) >= MIN_REGION_FACES and ctx.areas[remaining].sum() >= MIN_REGION_AREA:
        for comp in _components(ctx.n, ctx.adj, ctx.adj_ang < SMOOTH_DEG, restrict=remaining):
            a = float(ctx.areas[comp].sum())
            if len(comp) < MIN_REGION_FACES or a < MIN_REGION_AREA:
                continue
            pts = ctx.verts_of(comp)
            w = ctx.areas[comp]
            unclassified.append(
                {
                    "area": a,
                    "n_faces": int(len(comp)),
                    "bbox_min": pts.min(axis=0),
                    "bbox_max": pts.max(axis=0),
                    "mean_normal": unit((ctx.normals[comp] * w[:, None]).sum(axis=0)),
                }
            )


def _accept_cylinder(ctx: _MeshCtx, cyl: Cylinder, faces: np.ndarray, cyls: list) -> None:
    cyl.concave = _concavity(ctx.centers[faces], ctx.normals[faces], ctx.areas[faces], cyl)
    cyl.area = float(ctx.areas[faces].sum())
    cyl.source_faces = faces.tolist()
    cyls.append(cyl)


def mesh_features(model: Model) -> dict:
    mesh = model.mesh
    ctx = _MeshCtx(mesh)
    rng = np.random.default_rng(RNG_SEED)
    regions = _components(ctx.n, ctx.adj, ctx.adj_ang < SMOOTH_DEG)
    # filter by area only: a planar rectangle is a legitimate 2-triangle region
    regions = [c for c in regions if ctx.areas[c].sum() >= MIN_REGION_AREA]
    regions.sort(key=lambda c: -ctx.areas[c].sum())
    plane_patches: list[tuple[np.ndarray, float]] = []
    cylinders: list[Cylinder] = []
    unclassified: list[dict] = []
    for comp in regions[:MAX_REGIONS_PEELED]:
        _peel_region(ctx, comp, rng, plane_patches, cylinders, unclassified)
    for comp in regions[MAX_REGIONS_PEELED:]:
        pts = ctx.verts_of(comp)
        unclassified.append(
            {"area": float(ctx.areas[comp].sum()), "n_faces": int(len(comp)), "bbox_min": pts.min(axis=0),
             "bbox_max": pts.max(axis=0), "mean_normal": unit(ctx.normals[comp].mean(axis=0)), "note": "not analysed (region cap)"}
        )
    groups: list[dict] = []
    for sub, a in plane_patches:
        n = unit((ctx.normals[sub] * ctx.areas[sub][:, None]).sum(axis=0))
        verts = ctx.verts_of(sub)
        off = float(np.median(verts @ n))
        _add_to_plane_group(groups, n, off, a, verts.mean(axis=0), verts.min(axis=0), verts.max(axis=0), len(sub))
    planes = [_finish_plane_group(g, exact=False) for g in groups]
    cylinders = merge_coaxial(cylinders)
    classify_cylinders(model, cylinders)
    return _assemble(planes, cylinders, unclassified, exact=False, method="mesh_region_peel", model=model)


# ----------------------------------------------------------------------------- plane grouping
def _add_to_plane_group(groups: list[dict], n: np.ndarray, off: float, area: float, center: np.ndarray, bbmin: np.ndarray,
                        bbmax: np.ndarray, n_faces: int, ang_tol_deg: float = 1.0, off_tol: float = 0.02) -> None:
    cos_tol = math.cos(math.radians(ang_tol_deg))
    for g in groups:
        gn = unit(g["normal"])
        if float(gn @ n) >= cos_tol and abs(g["offset"] / g["area"] - off) <= off_tol:
            g["normal"] += n * area
            g["offset"] += off * area
            g["center"] += center * area
            g["area"] += area
            g["regions"] += 1
            g["faces"] += n_faces
            g["bbmin"] = np.minimum(g["bbmin"], bbmin)
            g["bbmax"] = np.maximum(g["bbmax"], bbmax)
            return
    groups.append(
        {"normal": n * area, "offset": off * area, "center": center * area, "area": area, "regions": 1, "faces": n_faces,
         "bbmin": np.array(bbmin, dtype=float), "bbmax": np.array(bbmax, dtype=float)}
    )


def _finish_plane_group(g: dict, exact: bool) -> PlaneFeature:
    return PlaneFeature(
        normal=unit(g["normal"]),
        offset=g["offset"] / g["area"],
        area=g["area"],
        center=g["center"] / g["area"],
        regions=g["regions"],
        faces=g["faces"],
        exact=exact,
        bbox_min=g["bbmin"],
        bbox_max=g["bbmax"],
    )


# ----------------------------------------------------------------------------- B-rep path
def _uv_coverage_deg(face) -> Optional[float]:
    """Angular extent of a cylindrical face read from its U parameter range (exact, no tessellation)."""
    try:
        from OCP.BRepTools import BRepTools

        umin, umax, _, _ = BRepTools.UVBounds_s(face.wrapped)
        return float(min(360.0, math.degrees(umax - umin)))
    except Exception:
        return None


def brep_features(model: Model) -> dict:
    from build123d import GeomType

    shape = model.shape
    groups: list[dict] = []
    cylinders: list[Cylinder] = []
    unclassified: list[dict] = []
    for i, f in enumerate(shape.faces()):
        gt = str(f.geom_type).split(".")[-1].lower()
        bb = f.bounding_box()
        bbmin = np.array([bb.min.X, bb.min.Y, bb.min.Z])
        bbmax = np.array([bb.max.X, bb.max.Y, bb.max.Z])
        if f.geom_type == GeomType.PLANE:
            c = f.center()
            nv = f.normal_at(c)
            n = unit([nv.X, nv.Y, nv.Z])
            cen = np.array([c.X, c.Y, c.Z])
            _add_to_plane_group(groups, n, float(cen @ n), float(f.area), cen, bbmin, bbmax, 1)
        elif f.geom_type == GeomType.CYLINDER:
            ax = f.axis_of_rotation
            d = unit([ax.direction.X, ax.direction.Y, ax.direction.Z])
            p0 = np.array([ax.position.X, ax.position.Y, ax.position.Z])
            verts, _ = f.tessellate(0.01, 0.1)
            pts = np.array([[p.X, p.Y, p.Z] for p in verts])
            cyl = _make_cylinder(pts, d, float(f.radius), p0, 0.0)
            cyl.exact = True
            cov = _uv_coverage_deg(f)
            if cov is not None:
                cyl.coverage_deg = cov
            c = f.center()
            nv = f.normal_at(c)
            cen = np.array([c.X, c.Y, c.Z])
            nrm = np.array([nv.X, nv.Y, nv.Z])
            rel = cen - cyl.axis_point
            radial = rel - (rel @ cyl.axis_dir) * cyl.axis_dir
            cyl.concave = bool(nrm @ radial < 0)
            cyl.area = float(f.area)
            cyl.source_faces = [i]
            cylinders.append(cyl)
        else:
            unclassified.append(
                {"surface_type": gt, "area": float(f.area), "n_faces": 1, "bbox_min": bbmin, "bbox_max": bbmax, "face_index": i}
            )
    planes = [_finish_plane_group(g, exact=True) for g in groups]
    cylinders = merge_coaxial(cylinders)
    classify_cylinders(model, cylinders)
    return _assemble(planes, cylinders, unclassified, exact=True, method="brep_faces", model=model)


# ----------------------------------------------------------------------------- shared post-processing
def merge_coaxial(cyls: list[Cylinder], rad_tol: float = 0.01, axis_deg: float = 0.5, dist_tol: float = 0.02) -> list[Cylinder]:
    """Merge cylinder pieces that share an axis, radius and concavity and overlap/abut along the axis."""
    used = [False] * len(cyls)
    merged: list[Cylinder] = []
    for i, a in enumerate(cyls):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(cyls):
                if used[j]:
                    continue
                if any(_coaxial(g, b, rad_tol, axis_deg, dist_tol) for g in group):
                    group.append(b)
                    used[j] = True
                    changed = True
        merged.append(_merge_group(group))
    return merged


def _coaxial(a: Cylinder, b: Cylinder, rad_tol: float, axis_deg: float, dist_tol: float) -> bool:
    if abs(a.radius - b.radius) > max(rad_tol, 0.002 * a.radius):
        return False
    if a.concave is not None and b.concave is not None and a.concave != b.concave:
        return False
    if abs(float(a.axis_dir @ b.axis_dir)) < math.cos(math.radians(axis_deg)):
        return False
    rel = b.axis_point - a.axis_point
    perp = rel - (rel @ a.axis_dir) * a.axis_dir
    if np.linalg.norm(perp) > max(dist_tol, 0.005 * a.radius):
        return False
    lo_a, hi_a = sorted([a.start @ a.axis_dir, a.end @ a.axis_dir])
    lo_b, hi_b = sorted([b.start @ a.axis_dir, b.end @ a.axis_dir])
    gap = max(lo_a, lo_b) - min(hi_a, hi_b)
    return gap <= max(0.05, 0.02 * max(hi_a - lo_a, hi_b - lo_b, 1.0))


def _merge_group(group: list[Cylinder]) -> Cylinder:
    if len(group) == 1:
        return group[0]
    ref = group[0]
    d = ref.axis_dir
    total_area = sum(g.area for g in group)
    radius = sum(g.radius * g.area for g in group) / total_area
    ts = []
    for g in group:
        ts.extend([(g.start - ref.axis_point) @ d, (g.end - ref.axis_point) @ d])
    t_min, t_max = min(ts), max(ts)
    axis_point = ref.axis_point + d * 0.5 * (t_min + t_max)
    votes = [g.concave for g in group if g.concave is not None]
    pts = [g.sample_points for g in group if g.sample_points is not None]
    if pts:
        allpts = np.vstack(pts)
        coverage = _angular_coverage_deg(allpts, axis_point, d)  # union of azimuths, never a sum
        samples = _subsample(allpts)
    else:
        coverage = max(g.coverage_deg for g in group)
        samples = None
    return Cylinder(
        radius=radius,
        axis_point=axis_point,
        axis_dir=d,
        half_height=0.5 * (t_max - t_min),
        coverage_deg=coverage,
        concave=(sum(votes) * 2 > len(votes)) if votes else None,
        area=total_area,
        fit_rms=max(g.fit_rms for g in group),
        exact=all(g.exact for g in group),
        source_faces=sum((g.source_faces for g in group), []),
        sample_points=samples,
    )


def _probe_inside(model: Model, points: np.ndarray) -> list[Optional[bool]]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if model.is_exact:
        from build123d import Vector

        return [bool(model.shape.is_inside(Vector(*p), tolerance=1e-4)) for p in points]
    mesh = model.mesh
    if not (mesh.is_watertight or mesh.is_winding_consistent):
        return [None] * len(points)
    # nearest-face signed distance (positive inside): orders of magnitude faster than ray parity on
    # large meshes, and reliable here because probes sit a deliberate 0.1–0.2 mm away from surfaces
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = trimesh.proximity.signed_distance(mesh, points)
    return [bool(x > 0) for x in sd]


def _end_probe_points(c: Cylinder, azimuths: int = 4) -> list[np.ndarray]:
    """Probe points past each end of a cylinder (see :func:`classify_cylinders`): per end, a near
    ring, a deeper ring and one deep point on the axis, i.e. ``2*azimuths + 1`` points."""
    delta_ax = max(0.1, 0.02 * c.height)
    delta_r = min(0.2, 0.25 * c.radius)
    rad = c.radius - delta_r if c.concave else c.radius + delta_r
    u, v = _basis_perp(c.axis_dir)
    out = []
    for end, sign in ((c.start, -1.0), (c.end, 1.0)):
        pts = []
        for depth in (delta_ax, max(delta_ax, 0.5 * c.radius)):  # a near ring and a deeper one (flared bases)
            for k in range(azimuths):
                a = 2 * math.pi * (k + 0.5) / azimuths
                pts.append(end + c.axis_dir * sign * depth + rad * (math.cos(a) * u + math.sin(a) * v))
        pts.append(end + c.axis_dir * sign * max(delta_ax, c.radius))
        out.append(np.array(pts))
    return out


def classify_cylinders(model: Model, cyls: list[Cylinder], azimuths: int = 4) -> None:
    """Attach a ``kind`` to each full cylinder by probing the solid past both ends.

    Two probe sets per end: a ring just past the end plane and just *inside* the wall for concave
    features (holes) or just *outside* it for convex ones (bosses), a deeper ring, and one point on
    the axis one radius past the end. Material at any set counts as 'closed'. The rings make an
    annular boss around a hole read as attached; the deeper probes see past a base flare, a bottom
    fillet or a drill point. All probes go to the kernel in one batch (mesh ray casting is slow).
    """
    full = [c for c in cyls if c.coverage_deg >= 300.0]
    for c in cyls:
        if c.coverage_deg < 300.0:
            c.kind = "fillet_candidate" if 45.0 <= c.coverage_deg <= 135.0 else "partial"
    if not full:
        return
    batches = [_end_probe_points(c, azimuths) for c in full]
    allpts = np.vstack([p for b in batches for p in b])
    ins_all = _probe_inside(model, allpts)
    per_end = 2 * azimuths + 1
    k = 0
    for c in full:
        ins: list[Optional[bool]] = []
        for _ in range(2):
            chunk = ins_all[k : k + per_end]
            k += per_end
            if any(x is None for x in chunk):
                ins.append(None)
                continue
            near_in = sum(chunk[:azimuths]) * 2 > azimuths
            deep_ring_in = sum(chunk[azimuths : 2 * azimuths]) * 2 > azimuths
            ins.append(bool(near_in or deep_ring_in or chunk[-1]))
        c.open_ends = [None if x is None else (not x) for x in ins]
        if c.concave:
            if ins[0] is None:
                c.kind = "hole"
            elif not ins[0] and not ins[1]:
                c.kind = "through_hole"
            elif ins[0] != ins[1]:
                c.kind = "blind_hole"
            else:
                c.kind = "internal_bore"
        else:
            if ins[0] is None:
                c.kind = "boss"
            elif ins[0] and ins[1]:
                c.kind = "shaft"
            elif ins[0] or ins[1]:
                c.kind = "boss"
            else:
                c.kind = "cylinder"


def detect_slots(model: Model, cyls: list[Cylinder], ids: list[str]) -> list[dict]:
    """Pair concave half-cylinders (≈180° coverage, same radius, parallel axes, facing away from each
    other) into slots: width = end diameter, length = axis distance + width. Through/blind is decided by
    probing the solid past each end at the slot centre, like holes."""
    halves = [(i, c) for i, c in enumerate(cyls) if c.concave and 150.0 <= c.coverage_deg <= 210.0 and c.sample_points is not None]
    used: set[int] = set()
    slots: list[dict] = []

    def outward(c: Cylinder) -> np.ndarray:  # in-plane direction from the axis towards the surface (the round end)
        rel = c.sample_points.mean(axis=0) - c.axis_point
        return unit(rel - (rel @ c.axis_dir) * c.axis_dir)

    for i, a in halves:
        if i in used:
            continue
        best = None
        for j, b in halves:
            if j <= i or j in used:
                continue
            if abs(a.radius - b.radius) > max(0.02, 0.01 * a.radius) or abs(float(a.axis_dir @ b.axis_dir)) < math.cos(math.radians(1.0)):
                continue
            rel = b.axis_point - a.axis_point
            perp = rel - (rel @ a.axis_dir) * a.axis_dir
            dist = float(np.linalg.norm(perp))
            if dist < 0.5 * a.radius:
                continue  # coaxial pieces, not two slot ends
            lo_a, hi_a = sorted([float(a.start @ a.axis_dir), float(a.end @ a.axis_dir)])
            lo_b, hi_b = sorted([float(b.start @ a.axis_dir), float(b.end @ a.axis_dir)])
            overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
            if overlap < 0.5 * min(a.height, b.height):
                continue
            pd = perp / dist
            if float(outward(a) @ pd) > -0.7 or float(outward(b) @ pd) < 0.7:
                continue  # the round ends must face away from each other
            if best is None or dist < best[0]:
                best = (dist, j, b, pd, lo_a, hi_a, lo_b, hi_b)
        if best is None:
            continue
        dist, j, b, pd, lo_a, hi_a, lo_b, hi_b = best
        used.update({i, j})
        d = a.axis_dir
        lo, hi = min(lo_a, lo_b), max(hi_a, hi_b)
        t_mid = 0.5 * (lo + hi)
        base = a.axis_point - (float(a.axis_point @ d) - t_mid) * d
        center = base + 0.5 * dist * pd
        depth = hi - lo
        delta = max(0.1, 0.02 * depth)
        probes = np.array([center - d * (0.5 * depth + delta), center + d * (0.5 * depth + delta)])
        ins = _probe_inside(model, probes)
        if any(x is None for x in ins):
            kind = "slot"
        elif not ins[0] and not ins[1]:
            kind = "through_slot"
        elif ins[0] != ins[1]:
            kind = "blind_slot"
        else:
            kind = "internal_slot"
        slots.append(
            {
                "kind": kind,
                "width": r(2 * a.radius),
                "length": r(dist + 2 * a.radius),
                "center": r(center),
                "direction": r(canonical_dir(pd)),
                "axis_dir": r(d),
                "depth": r(depth),
                "open_ends": [None if x is None else (not x) for x in ins],
                "end_ids": [ids[i], ids[j]],
                "exact": bool(a.exact and b.exact),
                "fit_rms": r(max(a.fit_rms, b.fit_rms)),
            }
        )
    slots.sort(key=lambda s: (s["width"], *s["center"]))
    return [{"id": f"S{k + 1:02d}", **s} for k, s in enumerate(slots)]


def _assemble(planes: list[PlaneFeature], cyls: list[Cylinder], unclassified: list[dict], exact: bool, method: str, model: Optional[Model] = None) -> dict:
    plane_out = []
    for i, p in enumerate(sorted(planes, key=lambda p: -p.area)):
        plane_out.append(
            {
                "id": f"P{i + 1:02d}",
                "normal": r(p.normal),
                "offset": r(p.offset),
                "area": r(p.area),
                "center": r(p.center),
                "bbox_min": r(p.bbox_min),
                "bbox_max": r(p.bbox_max),
                "regions": int(p.regions),
                "faces": int(p.faces),
                "exact": bool(p.exact),
            }
        )
    full_first = sorted(cyls, key=lambda c: (c.coverage_deg < 300.0, c.radius, *np.round(c.axis_point, 3)))
    cyl_out = []
    for i, c in enumerate(full_first):
        cyl_out.append(
            {
                "id": f"C{i + 1:02d}",
                "kind": c.kind,
                "diameter": r(2 * c.radius),
                "radius": r(c.radius),
                "axis_point": r(c.axis_point),
                "axis_dir": r(c.axis_dir),
                "start": r(c.start),
                "end": r(c.end),
                "height": r(c.height),
                "coverage_deg": r(c.coverage_deg, 1),
                "concave": c.concave,
                "open_ends": c.open_ends,
                "area": r(c.area),
                "exact": bool(c.exact),
                "fit_rms": r(c.fit_rms),
                "n_faces": len(c.source_faces),
            }
        )
    groups: dict[tuple, dict] = {}
    for e in cyl_out:
        if e["kind"] in {"partial", "fillet_candidate", "slot_end"}:
            continue
        key = (e["kind"], round(e["diameter"], 2), round(e["height"], 1))
        g = groups.setdefault(
            key, {"kind": e["kind"], "diameter": e["diameter"], "height": e["height"], "count": 0, "ids": [], "axis_points": []}
        )
        g["count"] += 1
        g["ids"].append(e["id"])
        g["axis_points"].append(e["axis_point"])
    unc_out = [{"id": f"R{i + 1:02d}", **{k: r(v) for k, v in u.items()}} for i, u in enumerate(sorted(unclassified, key=lambda u: -u["area"]))]
    slots = detect_slots(model, full_first, [e["id"] for e in cyl_out]) if model is not None else []
    for e in cyl_out:  # slot ends are accounted for: no longer loose partial cylinders
        if any(e["id"] in s["end_ids"] for s in slots):
            e["kind"] = "slot_end"
    return {
        "exact": exact,
        "method": method,
        "planes": plane_out,
        "cylinders": cyl_out,
        "cylinder_groups": sorted(groups.values(), key=lambda g: (-g["count"], g["diameter"])),
        "slots": slots,
        "patterns": detect_patterns(cyl_out),
        "unclassified_regions": unc_out,
        "counts": {
            "planes": len(plane_out),
            "cylinders": len(cyl_out),
            "holes": sum(1 for e in cyl_out if "hole" in e["kind"] or e["kind"] == "internal_bore"),
            "bosses_pins_shafts": sum(1 for e in cyl_out if e["kind"] in {"boss", "shaft", "cylinder"}),
            "partial_cylinders": sum(1 for e in cyl_out if e["kind"] in {"partial", "fillet_candidate", "slot_end"}),
            "slots": len(slots),
            "unclassified_regions": len(unc_out),
        },
    }


def detect_patterns(cyl_out: list[dict], tol: float = 0.05) -> list[dict]:
    """Group same-kind, same-diameter, parallel features into circular / linear / grid patterns."""
    out: list[dict] = []
    full = [e for e in cyl_out if e["kind"] not in {"partial", "fillet_candidate", "slot_end"}]
    by_key: dict[tuple, list[dict]] = {}
    for e in full:
        by_key.setdefault((e["kind"], round(e["diameter"], 2)), []).append(e)
    for (kind, dia), members in by_key.items():
        if len(members) < 3:
            continue
        d0 = np.asarray(members[0]["axis_dir"])
        members = [m for m in members if abs(float(np.asarray(m["axis_dir"]) @ d0)) > 0.9995]
        if len(members) < 3:
            continue
        from calipers.spec import inplane_basis

        u, v = inplane_basis(d0)  # world axes when the feature axis is one, so grid rows/cols read naturally
        P = np.array([[np.asarray(m["axis_point"]) @ u, np.asarray(m["axis_point"]) @ v] for m in members])
        ids = [m["id"] for m in members]
        ctr = P.mean(axis=0)
        rad = np.linalg.norm(P - ctr, axis=1)
        entry = {"kind": kind, "diameter": dia, "count": len(ids), "ids": ids, "axis_dir": members[0]["axis_dir"]}
        if rad.max() - rad.min() <= tol and rad.mean() > tol:  # bolt circle: equal radius, equal angular pitch
            ang = np.sort(np.degrees(np.arctan2(*(P - ctr)[:, ::-1].T)) % 360.0)
            pitches = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
            if pitches.max() - pitches.min() <= 0.5:
                c3 = ctr[0] * u + ctr[1] * v + float(np.mean([np.asarray(m["axis_point"]) @ d0 for m in members])) * d0
                out.append({**entry, "type": "circular", "center": r(c3), "pitch_radius": r(float(rad.mean())), "angular_pitch_deg": r(float(pitches.mean()), 2)})
                continue
        # linear: collinear with equal spacing
        rel = P - P[0]
        _, sv, vt = np.linalg.svd(rel, full_matrices=False)
        if len(sv) > 1 and sv[1] <= tol * math.sqrt(len(P)):
            t = np.sort(rel @ vt[0])
            steps = np.diff(t)
            if len(steps) and steps.max() - steps.min() <= tol and steps.min() > tol:
                out.append({**entry, "type": "linear", "direction": r(canonical_dir(vt[0][0] * u + vt[0][1] * v)), "pitch": r(float(steps.mean()))})
                continue
        # grid: rows × cols on the two in-plane axes
        xs, ys = np.unique(np.round(P[:, 0] / tol)) * tol, np.unique(np.round(P[:, 1] / tol)) * tol
        if len(P) >= 4 and len(xs) * len(ys) == len(P) and len(xs) > 1 and len(ys) > 1:
            px, py = np.diff(xs), np.diff(ys)
            if (px.max() - px.min() <= tol) and (py.max() - py.min() <= tol):
                out.append({**entry, "type": "grid", "rows": int(len(ys)), "cols": int(len(xs)), "pitch": [r(float(px.mean())), r(float(py.mean()))], "u": r(u), "v": r(v)})
    return out


def extract_features(model: Model) -> dict:
    """Entry point: exact features for B-reps, fitted features for meshes."""
    return brep_features(model) if model.is_exact else mesh_features(model)
