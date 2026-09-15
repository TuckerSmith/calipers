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

Slots are features too: ``kind: slot`` with ``width`` (the diameter of the rounded ends), ``length``
(overall, end to end), ``direction`` (the long side, ⟂ axis) and ``depth``; ``exact_counts`` covers
cylindrical features *and* slots.

Reference-conditioned design (Phase 3) — design around an imported part::

    references:
      - {id: device, path: device.stl, place: {rotate: [0, 0, 0], translate: [0, 0, 0]}}
    keep_out:                       # the part must have no material here
      - {id: usb, from: device, face: "+x", depth: 20, pad: 1}    # box in front of the device's +x face
      - {id: lid, box: {min: [-40, -30, 20], max: [40, 30, 30]}}
      - {id: screw, cylinder: {from: [0, 0, -5], to: [0, 0, 5], radius: 1.6}}
    keep_in:                        # all material must be inside
      - {id: bed, box: {center: [0, 0, 50], size: [250, 250, 100]}}
    fit:
      - {type: clearance, ref: device, min: 0.3}              # no overlap; closest approach >= 0.3
      - {type: gap, ref: device, directions: {"+x": [0.3, 1.0], "-x": [0.3, 1.0], "-z": [0, 0.5]}}
      - {type: enclosed, ref: device, min_fraction: 0.85, open: ["+z"]}

Reference paths are relative to the spec file. Generated code may cite ``measured:device.bbox_max.z``
(and ``extents.x``, ``center.y``, ``C03.diameter`` …) — those values are checked against the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

AXES = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
FEATURE_KINDS = {"through_hole", "blind_hole", "boss", "shaft", "cylinder", "internal_bore", "hole", "slot"}
FIT_TYPES = {"clearance", "gap", "enclosed"}


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
    return normalize(spec, base_dir=p.parent)


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


def normalize(spec: dict, base_dir: str | Path | None = None) -> dict:
    """Validate and fill defaults. Raises SpecError with a precise message on problems.

    ``base_dir`` is where relative reference paths resolve (the spec file's folder)."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping")
    out: dict[str, Any] = {"part": spec.get("part", "part"), "units": spec.get("units", "mm")}
    out["_base_dir"] = str(base_dir) if base_dir is not None else spec.get("_base_dir")
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
        if f["kind"] == "slot":
            if "width" not in f:
                raise SpecError(f"feature {fid}: a slot needs a width (the diameter of its rounded ends)")
            nf["width"] = toleranced(f["width"], where=f"{fid}.width")
            if "length" in f:
                nf["slot_length"] = toleranced(f["length"], where=f"{fid}.length")  # overall, end to end
            if "depth" in f:
                nf["length"] = toleranced(f["depth"], where=f"{fid}.depth")
            nf["direction"] = axis_vector(f.get("direction", "x")).tolist()  # along the slot's long side
            if abs(float(np.dot(nf["direction"], nf["axis"]))) > 1e-6:
                raise SpecError(f"feature {fid}: slot direction must be perpendicular to its axis")
        else:
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
    # ---- references, regions, fit (Phase 3)
    refs = []
    for i, rf in enumerate(spec.get("references", []) or []):
        if not isinstance(rf, dict) or "path" not in rf:
            raise SpecError(f"references[{i}] needs a path (STL/OBJ/3MF/STEP)")
        rid = str(rf.get("id", Path(rf["path"]).stem))
        if rid in ids or any(x["id"] == rid for x in refs):
            raise SpecError(f"duplicate id {rid!r} (reference ids share the namespace with features)")
        nr: dict[str, Any] = {"id": rid, "path": str(rf["path"])}
        if "units" in rf:
            from calipers.reference import UNIT_SCALE

            if str(rf["units"]).lower() not in UNIT_SCALE:
                raise SpecError(f"references[{rid}].units: unknown units {rf['units']!r} (choose from {sorted(UNIT_SCALE)})")
            nr["units"] = str(rf["units"]).lower()
        place = rf.get("place") or {}
        if place:
            rot, tr = place.get("rotate", [0, 0, 0]), place.get("translate", [0, 0, 0])
            if len(rot) != 3 or len(tr) != 3:
                raise SpecError(f"references[{rid}].place: rotate and translate need 3 values each")
            nr["place"] = {"rotate": list(map(float, rot)), "translate": list(map(float, tr))}
        refs.append(nr)
    out["references"] = refs
    ref_ids = {x["id"] for x in refs}
    for mode in ("keep_out", "keep_in"):
        regions = []
        for i, reg in enumerate(spec.get(mode, []) or []):
            regions.append(_normalize_region(reg, ref_ids, f"{mode}[{i}]", i))
        out[mode] = regions
    fits = []
    for i, f in enumerate(spec.get("fit", []) or []):
        t = f.get("type")
        if t not in FIT_TYPES:
            raise SpecError(f"fit[{i}]: unknown type {t!r} (choose from {sorted(FIT_TYPES)})")
        if f.get("ref") not in ref_ids:
            raise SpecError(f"fit[{i}]: 'ref' must name a reference id (known: {sorted(ref_ids)})")
        nf = {"type": t, "ref": str(f["ref"])}
        if t == "clearance":
            nf["min"] = float(f.get("min", 0.0))
            if "max" in f:
                nf["max"] = float(f["max"])
        elif t == "gap":
            dirs = f.get("directions")
            if not isinstance(dirs, dict) or not dirs:
                raise SpecError(f"fit[{i}]: gap needs directions like {{'-z': [0, 0.5], '+x': [0.3, 1.0]}}")
            nd = {}
            for name, rng in dirs.items():
                from calipers.reference import direction_vector

                try:
                    direction_vector(name)  # validates; vectors are kept in their string form as keys
                except ValueError as exc:
                    raise SpecError(f"fit[{i}].directions: {exc}") from exc
                name = name if isinstance(name, str) else str(list(map(float, name)))
                if isinstance(rng, (int, float)):
                    rng = [0.0, float(rng)]
                if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
                    raise SpecError(f"fit[{i}].directions[{name!r}]: give [min, max] gap in mm")
                nd[str(name)] = (float(rng[0]), float(rng[1]))
            nf["directions"] = nd
        else:
            nf["min_fraction"] = float(f.get("min_fraction", 0.9))
            nf["open"] = []
            for o in f.get("open", []) or []:
                from calipers.reference import direction_vector

                try:
                    direction_vector(o)
                except ValueError as exc:
                    raise SpecError(f"fit[{i}].open: {exc}") from exc
                nf["open"].append(o if isinstance(o, str) else list(map(float, o)))
            nf["max_distance"] = float(f.get("max_distance", 50.0))
        fits.append(nf)
    out["fit"] = fits
    return out


def _normalize_region(reg: Any, ref_ids: set[str], where: str, index: int) -> dict:
    if not isinstance(reg, dict):
        raise SpecError(f"{where}: must be a mapping")
    kinds = [k for k in ("box", "cylinder", "from") if k in reg]
    if len(kinds) != 1:
        raise SpecError(f"{where}: give exactly one of box:, cylinder: or from: <reference id>")
    out: dict[str, Any] = {"id": str(reg.get("id", f"region_{index}"))}
    if "box" in reg:
        b = reg["box"]
        if isinstance(b, dict) and "min" in b and "max" in b:
            lo, hi = list(map(float, b["min"])), list(map(float, b["max"]))
            if len(lo) != 3 or len(hi) != 3 or any(h < lo_ for h, lo_ in zip(hi, lo)):
                raise SpecError(f"{where}.box: min/max need 3 values with max >= min")
            out["box"] = {"min": lo, "max": hi}
        elif isinstance(b, dict) and "center" in b and "size" in b:
            out["box"] = {"center": list(map(float, b["center"])), "size": list(map(float, b["size"]))}
        else:
            raise SpecError(f"{where}.box: use {{min: [..], max: [..]}} or {{center: [..], size: [..]}}")
    elif "cylinder" in reg:
        c = reg["cylinder"]
        try:
            out["cylinder"] = {"from": list(map(float, c["from"])), "to": list(map(float, c["to"])), "radius": float(c["radius"])}
        except (KeyError, TypeError) as exc:
            raise SpecError(f"{where}.cylinder: needs from: [x,y,z], to: [x,y,z], radius") from exc
    else:
        if reg["from"] not in ref_ids:
            raise SpecError(f"{where}: from: {reg['from']!r} is not a reference id (known: {sorted(ref_ids)})")
        out["from"] = str(reg["from"])
        out["pad"] = float(reg.get("pad", 0.0))
        if "face" in reg:
            from calipers.reference import direction_vector

            direction_vector(reg["face"])
            if "depth" not in reg:
                raise SpecError(f"{where}: a face: region needs depth: (how far out from that face)")
            out["face"] = str(reg["face"])
            out["depth"] = float(reg["depth"])
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
