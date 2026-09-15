"""Requirements as data: the part spec schema.

A spec is a small YAML/JSON document that states what a part must satisfy. Every requirement in it
is *checkable* by :mod:`calipers.contracts` against real geometry — nothing here is prose. A spec is
also the source of truth for dimension provenance: generated code cites ``spec:<feature>.<field>``.

Conventions: ``positions`` are either 2-D in-plane coordinates perpendicular to the feature axis
(for axis z: ``[x, y]``; the position along the axis is free) or 3-D points that must lie on the
feature's axis *segment* (anywhere between its two ends). A plane's ``offset`` is ``normal · point``,
so the bottom face of a plate centred at the origin with thickness 5 is ``normal [0,0,-1], offset 2.5``.

Minimal example::

    part: bracket
    units: mm
    envelope:
      extents: {x: [60, 0.1], y: [40, 0.1], z: [13, 0.1]}      # nominal, ± tolerance
    solid: {valid: true, single_body: true}
    features:
      - id: mount_holes
        kind: through_hole
        diameter: [5.0, 0.05]
        axis: z
        positions: [[-20, 0], [20, 0]]      # in-plane coordinates ⟂ axis
        pos_tol: 0.1
        length: [5.0, 0.1]
      - id: boss
        kind: boss
        diameter: [12, 0.05]
        height: [8, 0.1]
        axis: z
        positions: [[0, 0]]
      - id: seat
        kind: blind_hole
        diameter: [8, 0.05]
        depth: [6, 0.1]
        axis: z
        positions: [[0, 0]]
        entry: "+"                # the open end faces +axis (drilled from the top); "-" from the bottom
    exact_counts: true          # no cylindrical features beyond those listed (fillets excluded)
    planes:
      - {id: top, normal: [0, 0, 1], offset: [2.5, 0.05], min_area: 2000}   # offset = normal · point
      - {id: bottom, normal: [0, 0, -1], offset: [2.5, 0.05]}              # the face z = -2.5
    relations:
      - {type: distance, between: [mount_holes.0, mount_holes.1], value: [40, 0.1]}
      - {type: coaxial, between: [boss.0, center_hole.0], tol: 0.05}
    symmetry: [x, y]            # mirror planes (world axes) the part must have
    printability: {min_wall: 1.2, bed: [256, 256, 256]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

AXES = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
FEATURE_KINDS = {"through_hole", "blind_hole", "boss", "shaft", "cylinder", "internal_bore", "hole"}


class SpecError(ValueError):
    pass


def load_spec(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        spec = yaml.safe_load(text)
    else:
        spec = json.loads(text)
    return normalize(spec)


def toleranced(v: Any, default_tol: float = 0.05, where: str = "") -> tuple[float, float]:
    """Accept ``5.0``, ``[5.0, 0.05]``, ``{nominal: 5, tol: 0.05}`` → (nominal, tol)."""
    if isinstance(v, bool):
        v = None
    if isinstance(v, (int, float)):
        return float(v), default_tol
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    if isinstance(v, dict) and "nominal" in v:
        return float(v["nominal"]), float(v.get("tol", default_tol))
    raise SpecError(f"{where + ': ' if where else ''}bad toleranced value: {v!r} (use 5.0, [5.0, 0.05] or {{nominal: 5, tol: 0.05}})")


def axis_vector(a: Any) -> np.ndarray:
    if isinstance(a, str):
        if a.lower() not in AXES:
            raise SpecError(f"unknown axis {a!r}")
        return np.array(AXES[a.lower()])
    v = np.asarray(a, dtype=float)
    n = np.linalg.norm(v)
    if v.shape != (3,) or n == 0:
        raise SpecError(f"bad axis {a!r}")
    return v / n


def normalize(spec: dict) -> dict:
    """Validate and fill defaults. Raises SpecError with a precise message on problems."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping")
    out: dict[str, Any] = {"part": spec.get("part", "part"), "units": spec.get("units", "mm")}
    env = spec.get("envelope", {}) or {}
    out["envelope"] = {}
    if "extents" in env:
        ext = {}
        for k in "xyz":
            if k in env["extents"]:
                ext[k] = toleranced(env["extents"][k], where=f"envelope.extents.{k}")
        out["envelope"]["extents"] = ext
    for k in ("max", "min"):
        if k in env:
            v = list(map(float, env[k]))
            if len(v) != 3:
                raise SpecError(f"envelope.{k} needs 3 values")
            out["envelope"][k] = v
    if "volume" in spec:
        vol = spec["volume"]
        out["volume"] = {k: float(vol[k]) for k in ("min", "max") if k in vol} if isinstance(vol, dict) else {"nominal": toleranced(vol, where="volume")}
    solid = spec.get("solid", {}) or {}
    out["solid"] = {"valid": bool(solid.get("valid", True)), "single_body": bool(solid.get("single_body", True))}
    feats = []
    ids = set()
    for i, f in enumerate(spec.get("features", []) or []):
        if "kind" not in f:
            raise SpecError(f"features[{i}] needs a kind")
        if f["kind"] not in FEATURE_KINDS:
            raise SpecError(f"features[{i}]: unknown kind {f['kind']!r} (choose from {sorted(FEATURE_KINDS)})")
        fid = str(f.get("id", f"feature_{i}"))
        if fid in ids:
            raise SpecError(f"duplicate feature id {fid!r}")
        ids.add(fid)
        nf: dict[str, Any] = {"id": fid, "kind": f["kind"], "axis": axis_vector(f.get("axis", "z")).tolist()}
        if "diameter" in f:
            nf["diameter"] = toleranced(f["diameter"], where=f"{fid}.diameter")
        for key in ("length", "height", "depth"):
            if key in f:
                nf["length"] = toleranced(f[key], where=f"{fid}.{key}")
        if "entry" in f:
            if str(f["entry"]) not in {"+", "-"}:
                raise SpecError(f"feature {fid}: entry must be '+' or '-' (the open end relative to the axis direction)")
            nf["entry"] = str(f["entry"])
        positions = f.get("positions")
        if positions is not None:
            pos = [list(map(float, p)) for p in positions]
            if any(len(p) not in (2, 3) for p in pos) or len({len(p) for p in pos}) > 1:
                raise SpecError(f"feature {fid}: positions must all be [u, v] in-plane or all [x, y, z] (no mixing)")
            nf["positions"] = pos
            nf["count"] = len(pos)
        elif "count" in f:
            nf["count"] = int(f["count"])
        else:
            nf["count"] = 1
        nf["pos_tol"] = float(f.get("pos_tol", 0.1))
        feats.append(nf)
    out["features"] = feats
    out["exact_counts"] = bool(spec.get("exact_counts", False))
    planes = []
    for i, p in enumerate(spec.get("planes", []) or []):
        np_ = {"id": str(p.get("id", f"plane_{i}")), "normal": axis_vector(p.get("normal", "z")).tolist()}
        if "offset" in p:
            np_["offset"] = toleranced(p["offset"], where=f"planes.{np_['id']}.offset")
        if "min_area" in p:
            np_["min_area"] = float(p["min_area"])
        planes.append(np_)
    out["planes"] = planes
    rels = []
    for i, r in enumerate(spec.get("relations", []) or []):
        t = r.get("type")
        if t not in {"distance", "coaxial", "parallel", "perpendicular"}:
            raise SpecError(f"relations[{i}]: unknown type {t!r}")
        between = r.get("between")
        if not (isinstance(between, list) and len(between) == 2):
            raise SpecError(f"relations[{i}]: 'between' needs two feature refs like boss.0")
        nr: dict[str, Any] = {"type": t, "between": [str(b) for b in between]}
        for b in nr["between"]:
            fid, _, idx = str(b).partition(".")
            match = [f for f in feats if f["id"] == fid]
            if not match:
                raise SpecError(f"relations[{i}]: unknown feature id {fid!r} in {b!r}")
            if idx and not (idx.isdigit() and int(idx) < match[0]["count"]):
                raise SpecError(f"relations[{i}]: {b!r} index out of range (feature has {match[0]['count']} instance(s))")
        if t == "distance":
            nr["value"] = toleranced(r.get("value"), where=f"relations[{i}].value")
        else:
            nr["tol"] = float(r.get("tol", 0.05))
        rels.append(nr)
    out["relations"] = rels
    sym = spec.get("symmetry", []) or []
    out["symmetry"] = [axis_vector(s).tolist() for s in sym]
    pr = spec.get("printability", {}) or {}
    out["printability"] = {}
    if "min_wall" in pr:
        out["printability"]["min_wall"] = float(pr["min_wall"])
    if "bed" in pr:
        out["printability"]["bed"] = list(map(float, pr["bed"]))
    return out


def feature_ref(spec: dict, ref: str) -> tuple[dict, int]:
    """Resolve ``"boss.0"`` → (feature dict, index)."""
    fid, _, idx = ref.partition(".")
    for f in spec["features"]:
        if f["id"] == fid:
            return f, int(idx or 0)
    raise SpecError(f"unknown feature reference {ref!r}")


def positions_3d(f: dict) -> list[np.ndarray] | None:
    """Spec positions as 3D points that must lie on the feature's axis segment, when given as [x, y, z]."""
    if "positions" not in f:
        return None
    if all(len(p) == 3 for p in f["positions"]):
        return [np.asarray(p, dtype=float) for p in f["positions"]]
    return None


def inplane_basis(axis: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """In-plane (u, v) axes for 2D positions ⟂ a feature axis: world axes when the axis is one."""
    a = np.asarray(axis, dtype=float)
    for k, name in enumerate("xyz"):
        if abs(abs(a[k]) - 1.0) < 1e-9:
            others = [i for i in range(3) if i != k]
            return np.eye(3)[others[0]], np.eye(3)[others[1]]
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, a)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)
