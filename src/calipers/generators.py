"""Generators for reference-conditioned parts: an enclosure around, and a mount under, an imported model.

A generator does not produce geometry directly. It measures the reference and writes two files a
model (or a person) would otherwise have to write: a build123d script whose every dimension is
sourced (``measured:<ref>...`` for numbers read off the reference, ``spec:`` for requirements,
``assumption:`` for the rest) and the spec with the fit contracts that prove the result. The pair
then goes through the ordinary ``calipers run --spec`` loop, so a generated part is verified
exactly like a hand-written one — the generator is a strong first draft, never a trusted answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from calipers.model import Model
from calipers.reference import DIRECTIONS, direction_vector, load_reference, reference_summary

AXES = "xyz"
BED = [256.0, 256.0, 256.0]


def _ref_entry(ref_path: str | Path, ref_id: Optional[str], place: Optional[dict], out_dir: Optional[Path]) -> dict:
    p = Path(ref_path).resolve()
    entry: dict[str, Any] = {"id": ref_id or p.stem.replace("-", "_").replace(" ", "_"), "path": str(p)}
    if out_dir is not None:
        try:
            entry["path"] = str(p.relative_to(Path(out_dir).resolve()))
        except ValueError:
            pass
    if place:
        entry["place"] = {"rotate": list(map(float, place.get("rotate", [0, 0, 0]))), "translate": list(map(float, place.get("translate", [0, 0, 0])))}
    return entry


def _box_helper() -> str:
    return (
        "def box_between(lo, hi):\n"
        "    size = [hi[i] - lo[i] for i in range(3)]\n"
        "    center = [(lo[i] + hi[i]) / 2 for i in range(3)]\n"
        "    return Pos(*center) * Box(*size)\n"
    )


def _params_block(params: dict[str, tuple[float, str]]) -> str:
    width = max(len(k) for k in params) + 3
    lines = ["PARAMS = {"]
    for k, (v, src) in params.items():
        lines.append(f"    {(repr(k) + ':').ljust(width)} ({float(v)!r}, {src!r}),")
    lines.append("}")
    lines.append("P = {k: v[0] for k, v in PARAMS.items()}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- enclosure
def enclosure(
    ref_path: str | Path,
    ref_id: Optional[str] = None,
    wall: float = 2.0,
    clearance: float = 0.5,
    floor: Optional[float] = None,
    open_face: str = "+z",
    corner_radius: float = 0.0,
    gap_slack: float = 1.0,
    place: Optional[dict] = None,
    out_dir: str | Path | None = None,
    bed: Optional[list[float]] = None,
) -> dict:
    """An open-faced box around the reference: cavity = envelope + clearance, walls of ``wall``, a
    ``floor`` (defaults to ``wall``) opposite the open face. Returns ``{"code", "spec", "spec_dict",
    "summary"}`` and writes ``part.py`` / ``spec.yaml`` into ``out_dir`` when given."""
    floor = wall if floor is None else float(floor)
    out_dir = Path(out_dir) if out_dir is not None else None
    ref = _ref_entry(ref_path, ref_id, place, out_dir)
    rid = ref["id"]
    model = load_reference({**ref, "path": str(Path(ref_path).resolve())}, None)
    summ = reference_summary(model)
    lo, hi = np.asarray(summ["bbox_min"], float), np.asarray(summ["bbox_max"], float)
    d = direction_vector(open_face)
    k = int(np.argmax(np.abs(d)))
    positive = d[k] > 0
    open_name = ("+" if positive else "-") + AXES[k]
    min_wall = round(min(wall, floor) - 0.05, 3)

    params: dict[str, tuple[float, str]] = {}
    for i, ax in enumerate(AXES):
        params[f"ref_min_{ax}"] = (float(lo[i]), f"measured:{rid}.bbox_min.{ax}")
        params[f"ref_max_{ax}"] = (float(hi[i]), f"measured:{rid}.bbox_max.{ax}")
    params["clearance"] = (float(clearance), "spec:fit.0.min")
    params["wall"] = (float(wall), "derived:printability.min_wall + 0.05 (spec margin)" if wall <= floor else "assumption:side wall thickness (generator option)")
    params["floor"] = (float(floor), "derived:printability.min_wall + 0.05 (spec margin)" if floor < wall else ("derived:wall" if floor == wall else "assumption:floor thickness (generator option)"))
    if corner_radius > 0:
        params["corner_r"] = (float(corner_radius), "assumption:outer corner radius (generator option)")

    # per-side thickness expressions: wall everywhere, the floor opposite the open face, none on it
    t_lo = ['P["wall"]'] * 3
    t_hi = ['P["wall"]'] * 3
    if positive:
        t_hi[k], t_lo[k] = "0", 'P["floor"]'
    else:
        t_lo[k], t_hi[k] = "0", 'P["floor"]'
    cav_lo = ", ".join(f'P["ref_min_{ax}"] - P["clearance"]' for ax in AXES)
    cav_hi = ", ".join(f'P["ref_max_{ax}"] + P["clearance"]' for ax in AXES)
    out_lo = ", ".join(f"cav_lo[{i}] - {t_lo[i]}" if t_lo[i] != "0" else f"cav_lo[{i}]" for i in range(3))
    out_hi = ", ".join(f"cav_hi[{i}] + {t_hi[i]}" if t_hi[i] != "0" else f"cav_hi[{i}]" for i in range(3))
    extend = f'cav_hi[{k}] = cav_hi[{k}] + P["wall"]' if positive else f'cav_lo[{k}] = cav_lo[{k}] - P["wall"]'
    code = [
        f'"""{rid}_enclosure — generated by `calipers generate enclosure`; every dimension is sourced."""',
        "from build123d import *",
        "",
        _params_block(params),
        "",
        _box_helper(),
        "# cavity: the reference envelope grown by the clearance on every side",
        f"cav_lo = [{cav_lo}]",
        f"cav_hi = [{cav_hi}]",
        f"# outer shell: walls around the cavity, a floor opposite the open face ({open_name}), nothing on it",
        f"out_lo = [{out_lo}]",
        f"out_hi = [{out_hi}]",
        "outer = box_between(out_lo, out_hi)",
    ]
    if corner_radius > 0:
        code.append(f'outer = fillet(outer.edges().filter_by(Axis.{AXES[k].upper()}), radius=P["corner_r"])')
    code += [
        "# the open side: run the cavity out through the shell so that face is open",
        extend,
        "cavity = box_between(cav_lo, cav_hi)",
        "result = outer - cavity",
        "",
    ]
    ext = hi - lo
    extents = {}
    for i, ax in enumerate(AXES):
        L = float(ext[i] + 2 * clearance + (2 * wall if i != k else floor))
        extents[ax] = [round(L, 4), 0.05]
    other_dirs = [n for n in DIRECTIONS if n != open_name]
    spec = {
        "part": f"{rid}_enclosure",
        "units": "mm",
        "references": [ref],
        "envelope": {"extents": extents},
        "solid": {"valid": True, "single_body": True},
        "keep_out": [{"id": "opening", "from": rid, "face": open_name, "depth": round(clearance + wall + 1.0, 3), "pad": round(0.9 * clearance, 4)}],
        "fit": [
            {"type": "clearance", "ref": rid, "min": float(clearance)},
            {"type": "gap", "ref": rid, "directions": {n: [float(clearance), round(clearance + gap_slack, 4)] for n in other_dirs}},
            {"type": "enclosed", "ref": rid, "min_fraction": 0.85, "open": [open_name]},
        ],
        "printability": {"min_wall": min_wall, "bed": list(bed or BED)},
    }
    return _finish("\n".join(code), spec, summ, out_dir)


# ----------------------------------------------------------------------------- mount
def mount(
    ref_path: str | Path,
    ref_id: Optional[str] = None,
    plate_t: float = 3.0,
    standoff_h: float = 5.0,
    standoff_wall: float = 2.0,
    screw_d: Optional[float] = None,
    margin: float = 3.0,
    hole_diameter: Optional[float] = None,
    place: Optional[dict] = None,
    out_dir: str | Path | None = None,
    bed: Optional[list[float]] = None,
) -> dict:
    """A plate under the reference with standoffs at its through holes (axis z) and screw holes
    through them. Hole positions and diameters are *measured* on the reference; ``hole_diameter``
    selects which hole size to use when the reference has several (default: the most common)."""
    from calipers.features import extract_features

    out_dir = Path(out_dir) if out_dir is not None else None
    ref = _ref_entry(ref_path, ref_id, place, out_dir)
    rid = ref["id"]
    model: Model = load_reference({**ref, "path": str(Path(ref_path).resolve())}, None)
    summ = reference_summary(model)
    feats = extract_features(model)
    holes = [c for c in feats["cylinders"] if c["kind"] == "through_hole" and abs(float(c["axis_dir"][2])) > 0.9995]
    if not holes:
        raise ValueError(f"reference {rid!r} has no through holes along z to mount by (found kinds: {sorted({c['kind'] for c in feats['cylinders']})})")
    if hole_diameter is None:
        dias = [round(c["diameter"], 1) for c in holes]
        hole_diameter = max(set(dias), key=dias.count)
    chosen = [c for c in holes if abs(c["diameter"] - hole_diameter) <= 0.15]
    if not chosen:
        raise ValueError(f"no through hole of diameter ≈ {hole_diameter} on {rid!r} (available: {sorted({round(c['diameter'], 2) for c in holes})})")
    chosen.sort(key=lambda c: (round(c["axis_point"][1], 3), round(c["axis_point"][0], 3)))
    hole_d = float(np.mean([c["diameter"] for c in chosen]))
    screw = float(screw_d) if screw_d is not None else round(hole_d * 0.85, 1)
    standoff_d = round(hole_d + 2 * standoff_wall, 3)
    lo, hi = np.asarray(summ["bbox_min"], float), np.asarray(summ["bbox_max"], float)
    plate_top = float(lo[2] - standoff_h)
    plate_bottom = float(plate_top - plate_t)

    params: dict[str, tuple[float, str]] = {}
    for ax, i in (("x", 0), ("y", 1)):
        params[f"ref_min_{ax}"] = (float(lo[i]), f"measured:{rid}.bbox_min.{ax}")
        params[f"ref_max_{ax}"] = (float(hi[i]), f"measured:{rid}.bbox_max.{ax}")
    params["ref_min_z"] = (float(lo[2]), f"measured:{rid}.bbox_min.z")
    for n, c in enumerate(chosen):
        params[f"hole_{n}_x"] = (float(c["axis_point"][0]), f"measured:{rid}.{c['id']}.axis_point.x")
        params[f"hole_{n}_y"] = (float(c["axis_point"][1]), f"measured:{rid}.{c['id']}.axis_point.y")
    params["margin"] = (float(margin), "assumption:plate margin around the reference footprint (generator option)")
    params["standoff_h"] = (float(standoff_h), "spec:standoffs.height")
    params["standoff_d"] = (standoff_d, "spec:standoffs.diameter")
    params["screw_d"] = (screw, "spec:screw_holes.diameter")
    params["plate_t"] = (float(plate_t), "derived:screw_holes.length - standoffs.height")
    positions = [[round(float(c["axis_point"][0]), 4), round(float(c["axis_point"][1]), 4)] for c in chosen]
    code = [
        f'"""{rid}_mount — generated by `calipers generate mount`; hole positions were measured on the reference."""',
        "from build123d import *",
        "",
        _params_block(params),
        "",
        _box_helper(),
        'plate_lo = [P["ref_min_x"] - P["margin"], P["ref_min_y"] - P["margin"], P["ref_min_z"] - P["standoff_h"] - P["plate_t"]]',
        'plate_hi = [P["ref_max_x"] + P["margin"], P["ref_max_y"] + P["margin"], P["ref_min_z"] - P["standoff_h"]]',
        "body = box_between(plate_lo, plate_hi)",
        "holes = [" + ", ".join(f'(P["hole_{n}_x"], P["hole_{n}_y"])' for n in range(len(chosen))) + "]",
        "for hx, hy in holes:  # standoffs rise from the plate top to the underside of the reference",
        '    body += Pos(hx, hy, plate_hi[2]) * Cylinder(P["standoff_d"] / 2, P["standoff_h"], align=(Align.CENTER, Align.CENTER, Align.MIN))',
        "for hx, hy in holes:  # screw holes through standoff and plate",
        '    body -= Pos(hx, hy, plate_lo[2]) * Cylinder(P["screw_d"] / 2, P["standoff_h"] + P["plate_t"], align=(Align.CENTER, Align.CENTER, Align.MIN))',
        "result = body",
        "",
    ]
    spec = {
        "part": f"{rid}_mount",
        "units": "mm",
        "references": [ref],
        "envelope": {"extents": {"x": [round(float(hi[0] - lo[0] + 2 * margin), 4), 0.05], "y": [round(float(hi[1] - lo[1] + 2 * margin), 4), 0.05], "z": [round(standoff_h + plate_t, 4), 0.05]}},
        "solid": {"valid": True, "single_body": True},
        "features": [
            {"id": "standoffs", "kind": "boss", "diameter": [standoff_d, 0.05], "height": [float(standoff_h), 0.05], "axis": "z", "positions": positions, "pos_tol": 0.05},
            {"id": "screw_holes", "kind": "through_hole", "diameter": [screw, 0.05], "length": [round(standoff_h + plate_t, 4), 0.05], "axis": "z", "positions": positions, "pos_tol": 0.05},
        ],
        "exact_counts": True,
        "relations": [{"type": "coaxial", "between": [f"standoffs.{n}", f"screw_holes.{n}"], "tol": 0.02} for n in range(len(chosen))],
        "planes": [{"id": "plate_bottom", "normal": [0, 0, -1], "offset": [round(-plate_bottom, 4), 0.05]}, {"id": "plate_top", "normal": [0, 0, 1], "offset": [round(plate_top, 4), 0.05]}],
        "keep_out": [{"id": "device", "from": rid, "pad": 0.0}],
        "fit": [{"type": "clearance", "ref": rid, "min": 0.0}, {"type": "gap", "ref": rid, "directions": {"-z": [0.0, 0.05]}}],
        "printability": {"min_wall": round(min(plate_t, standoff_wall) - 0.05, 3), "bed": list(bed or BED)},
    }
    return _finish("\n".join(code), spec, summ, out_dir, extra={"holes": [{"id": c["id"], "diameter": c["diameter"], "position": p} for c, p in zip(chosen, positions)]})


def _finish(code: str, spec: dict, summ: dict, out_dir: Optional[Path], extra: Optional[dict] = None) -> dict:
    spec_yaml = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    out: dict[str, Any] = {"code": code, "spec": spec_yaml, "spec_dict": spec, "reference": summ, **(extra or {})}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "part.py").write_text(code, encoding="utf-8")
        (out_dir / "spec.yaml").write_text(spec_yaml, encoding="utf-8")
        out["paths"] = {"code": str(out_dir / "part.py"), "spec": str(out_dir / "spec.yaml")}
    return out
