"""Version diffing: what changed between two models (two runs of a script, or a part and its revision).

Matches planes by normal and offset, cylindrical features by kind / axis / position (Hungarian
assignment), slots by centre, and reports parameter deltas, additions and removals. For two B-reps
the symmetric-difference volume is computed exactly by the kernel, so an edit that "silently"
moved unrelated material shows up as a number.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from calipers import measure
from calipers.features import extract_features
from calipers.measure import r
from calipers.model import Model

FULL_KINDS = {"through_hole", "blind_hole", "boss", "shaft", "cylinder", "internal_bore", "hole"}


def _match(a: list[dict], b: list[dict], cost_fn, max_cost: float) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    if not a or not b:
        return [], list(a), list(b)
    from scipy.optimize import linear_sum_assignment

    C = np.full((len(a), len(b)), 1e9)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            C[i, j] = cost_fn(x, y)
    rows, cols = linear_sum_assignment(C)
    pairs, ua, ub = [], set(range(len(a))), set(range(len(b)))
    for i, j in zip(rows, cols):
        if C[i, j] <= max_cost:
            pairs.append((a[i], b[j]))
            ua.discard(i)
            ub.discard(j)
    return pairs, [a[i] for i in sorted(ua)], [b[j] for j in sorted(ub)]


def _cyl_cost(x: dict, y: dict) -> float:
    if x["kind"] != y["kind"] or abs(float(np.asarray(x["axis_dir"]) @ np.asarray(y["axis_dir"]))) < 0.9995:
        return 1e9
    rel = np.asarray(y["axis_point"]) - np.asarray(x["axis_point"])
    d = np.asarray(x["axis_dir"])
    return float(np.linalg.norm(rel - (rel @ d) * d)) + 0.1 * abs(x["diameter"] - y["diameter"])


def _plane_cost(x: dict, y: dict) -> float:
    if float(np.asarray(x["normal"]) @ np.asarray(y["normal"])) < 0.9995:
        return 1e9
    return abs(x["offset"] - y["offset"])


def diff_models(a: Model, b: Model, features_a: Optional[dict] = None, features_b: Optional[dict] = None, match_tol: float = 2.0) -> dict:
    sa, sb = measure.summary(a), measure.summary(b)
    fa = features_a if features_a is not None else extract_features(a)
    fb = features_b if features_b is not None else extract_features(b)
    out: dict[str, Any] = {"a": a.name, "b": b.name, "exact": a.is_exact and b.is_exact}
    out["summary"] = {
        "extents": {"a": sa["extents"], "b": sb["extents"], "delta": r(np.asarray(sb["extents"]) - np.asarray(sa["extents"]))},
        "volume": {"a": sa.get("volume"), "b": sb.get("volume"), "delta": None if sa.get("volume") is None or sb.get("volume") is None else r(sb["volume"] - sa["volume"])},
        "surface_area": {"a": sa["surface_area"], "b": sb["surface_area"], "delta": r(sb["surface_area"] - sa["surface_area"])},
        "center_of_mass_shift": r(float(np.linalg.norm(np.asarray(sb["center_of_mass"]) - np.asarray(sa["center_of_mass"])))),
    }
    if a.is_exact and b.is_exact:
        try:
            from calipers.reference import boolean_volume

            removed = boolean_volume(a.shape, b.shape, "cut")
            added = boolean_volume(b.shape, a.shape, "cut")
            out["material"] = {"removed_volume": r(removed), "added_volume": r(added), "exact": True}
        except Exception as exc:  # pragma: no cover - kernel edge cases
            out["material"] = {"error": str(exc)}
    # planes
    pairs, gone, new = _match(fa["planes"], fb["planes"], _plane_cost, match_tol)
    out["planes"] = {
        "changed": [{"a": x["id"], "b": y["id"], "normal": x["normal"], "offset": {"a": x["offset"], "b": y["offset"], "delta": r(y["offset"] - x["offset"])}, "area_delta": r(y["area"] - x["area"])} for x, y in pairs if abs(y["offset"] - x["offset"]) > 1e-4 or abs(y["area"] - x["area"]) > 0.01 * max(x["area"], 1e-9)],
        "removed": [{"id": x["id"], "normal": x["normal"], "offset": x["offset"], "area": x["area"]} for x in gone],
        "added": [{"id": y["id"], "normal": y["normal"], "offset": y["offset"], "area": y["area"]} for y in new],
        "unchanged": sum(1 for x, y in pairs if abs(y["offset"] - x["offset"]) <= 1e-4 and abs(y["area"] - x["area"]) <= 0.01 * max(x["area"], 1e-9)),
    }
    # cylinders
    ca = [c for c in fa["cylinders"] if c["kind"] in FULL_KINDS]
    cb = [c for c in fb["cylinders"] if c["kind"] in FULL_KINDS]
    pairs, gone, new = _match(ca, cb, _cyl_cost, match_tol)
    changed = []
    same = 0
    for x, y in pairs:
        dd = y["diameter"] - x["diameter"]
        dh = y["height"] - x["height"]
        shift = float(np.linalg.norm(np.asarray(y["axis_point"]) - np.asarray(x["axis_point"])))
        if abs(dd) > 1e-4 or abs(dh) > 1e-4 or shift > 1e-4:
            changed.append({"a": x["id"], "b": y["id"], "kind": x["kind"], "diameter": {"a": x["diameter"], "b": y["diameter"], "delta": r(dd)}, "height": {"a": x["height"], "b": y["height"], "delta": r(dh)}, "axis_point_shift": r(shift)})
        else:
            same += 1
    fmt = lambda c: {"id": c["id"], "kind": c["kind"], "diameter": c["diameter"], "height": c["height"], "axis_point": c["axis_point"], "axis_dir": c["axis_dir"]}  # noqa: E731
    out["cylinders"] = {"changed": changed, "removed": [fmt(x) for x in gone], "added": [fmt(y) for y in new], "unchanged": same}
    # slots
    sa_, sb_ = fa.get("slots", []), fb.get("slots", [])
    pairs, gone, new = _match(sa_, sb_, lambda x, y: float(np.linalg.norm(np.asarray(x["center"]) - np.asarray(y["center"]))) if x["kind"] == y["kind"] else 1e9, match_tol)
    out["slots"] = {
        "changed": [{"a": x["id"], "b": y["id"], "width_delta": r(y["width"] - x["width"]), "length_delta": r(y["length"] - x["length"]), "depth_delta": r(y["depth"] - x["depth"])} for x, y in pairs if any(abs(y[k] - x[k]) > 1e-4 for k in ("width", "length", "depth"))],
        "removed": [{"id": x["id"], "kind": x["kind"], "center": x["center"]} for x in gone],
        "added": [{"id": y["id"], "kind": y["kind"], "center": y["center"]} for y in new],
    }
    out["identical"] = (
        not out["planes"]["changed"] and not out["planes"]["removed"] and not out["planes"]["added"]
        and not changed and not out["cylinders"]["removed"] and not out["cylinders"]["added"]
        and not out["slots"]["changed"] and not out["slots"]["removed"] and not out["slots"]["added"]
        and (out.get("material", {}).get("removed_volume", 0.0) or 0.0) <= 1e-6 and (out.get("material", {}).get("added_volume", 0.0) or 0.0) <= 1e-6
    )
    return out


def diff_text(d: dict) -> str:
    s = d["summary"]
    lines = [f"# Diff: {d['a']} → {d['b']} ({'no geometric change' if d['identical'] else 'changed'}; {'exact' if d['exact'] else 'fitted'})"]
    ext = s["extents"]
    lines.append(f"Extents {ext['a']} → {ext['b']} (Δ {ext['delta']}); volume Δ {s['volume']['delta']} mm³; centre of mass moved {s['center_of_mass_shift']} mm.")
    if "material" in d and "removed_volume" in d["material"]:
        lines.append(f"Material removed {d['material']['removed_volume']} mm³, added {d['material']['added_volume']} mm³ (kernel-exact).")
    p, c, sl = d["planes"], d["cylinders"], d["slots"]
    lines.append(f"Planes: {p['unchanged']} unchanged, {len(p['changed'])} moved/resized, {len(p['removed'])} removed, {len(p['added'])} added.")
    for x in p["changed"]:
        lines.append(f"  - plane {x['a']}→{x['b']} normal {x['normal']}: offset {x['offset']['a']} → {x['offset']['b']} (Δ {x['offset']['delta']}), area Δ {x['area_delta']}")
    for x in p["removed"]:
        lines.append(f"  - removed plane {x['id']} normal {x['normal']} offset {x['offset']} area {x['area']}")
    for x in p["added"]:
        lines.append(f"  - added plane {x['id']} normal {x['normal']} offset {x['offset']} area {x['area']}")
    lines.append(f"Cylindrical features: {c['unchanged']} unchanged, {len(c['changed'])} changed, {len(c['removed'])} removed, {len(c['added'])} added.")
    for x in c["changed"]:
        lines.append(f"  - {x['kind']} {x['a']}→{x['b']}: Ø {x['diameter']['a']} → {x['diameter']['b']} (Δ {x['diameter']['delta']}), length {x['height']['a']} → {x['height']['b']} (Δ {x['height']['delta']}), axis moved {x['axis_point_shift']}")
    for x in c["removed"]:
        lines.append(f"  - removed {x['kind']} {x['id']} Ø{x['diameter']} × {x['height']} at {x['axis_point']}")
    for x in c["added"]:
        lines.append(f"  - added {x['kind']} {x['id']} Ø{x['diameter']} × {x['height']} at {x['axis_point']}")
    if sl["changed"] or sl["removed"] or sl["added"]:
        lines.append(f"Slots: {len(sl['changed'])} changed, {len(sl['removed'])} removed, {len(sl['added'])} added.")
    return "\n".join(lines)
