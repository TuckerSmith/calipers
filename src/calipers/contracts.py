"""Executable contracts: check a model against a spec and report every deviation.

The verifier is deterministic and cheap; the generator (an LLM) is expensive and fallible. Every
requirement becomes one ``Check`` with what was required, what was measured, the deviation and a
verdict, so a model can repair precisely instead of guessing what went wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from calipers import measure
from calipers.features import extract_features
from calipers.model import Model
from calipers.spec import inplane_basis, positions_3d

FULL_KINDS = {"through_hole", "blind_hole", "boss", "shaft", "cylinder", "internal_bore", "hole"}


@dataclass
class Check:
    id: str
    description: str
    passed: bool
    required: Any = None
    measured: Any = None
    deviation: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "passed": self.passed,
            "required": self.required,
            "measured": self.measured,
            "deviation": None if self.deviation is None else round(float(self.deviation), 4),
            "note": self.note,
        }


@dataclass
class VerifyResult:
    checks: list[Check] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "n_checks": len(self.checks), "n_failed": self.n_failed, "checks": [c.to_dict() for c in self.checks]}

    def to_text(self) -> str:
        lines = [f"# Contract check: {'PASS' if self.passed else 'FAIL'} ({len(self.checks) - self.n_failed}/{len(self.checks)} checks passed)"]
        for c in self.checks:
            mark = "ok  " if c.passed else "FAIL"
            dev = "" if c.deviation is None else f" (dev {c.deviation:+.4f})"
            lines.append(f"[{mark}] {c.id}: {c.description} — required {c.required}, measured {c.measured}{dev}{' — ' + c.note if c.note else ''}")
        return "\n".join(lines)


def _tol_check(cid: str, desc: str, nominal: float, tol: float, measured: Optional[float], note: str = "") -> Check:
    if measured is None:
        return Check(cid, desc, False, required=f"{nominal} ± {tol}", measured=None, note=note or "not found")
    dev = float(measured) - nominal
    return Check(cid, desc, abs(dev) <= tol + 1e-9, required=f"{nominal} ± {tol}", measured=round(float(measured), 4), deviation=dev, note=note)


def _target_distance(c: dict, target: np.ndarray, axis: np.ndarray, three_d: bool) -> float:
    """Distance from a spec target to a detected cylinder: to its axis line (2-D targets, along-axis
    component ignored) or to its axis *segment* between start and end (3-D targets)."""
    if not three_d:
        rel = np.asarray(c["axis_point"]) - target
        rel = rel - (rel @ axis) * axis
        return float(np.linalg.norm(rel))
    a, b = np.asarray(c["start"]), np.asarray(c["end"])
    ab = b - a
    t = float(np.clip((target - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(target - (a + t * ab)))


def _match_features(spec_f: dict, cyls: list[dict], used: set[str]) -> list[tuple[Optional[dict], Optional[np.ndarray]]]:
    """For each required instance, the detected cylinder assigned to it (kind, axis, position) — or None.

    Instances of one feature are assigned jointly (Hungarian algorithm on position distance, diameter
    deviation as tiebreaker) so a greedy first choice cannot leave a later instance without its twin.
    """
    axis = np.asarray(spec_f["axis"], dtype=float)
    kind = spec_f["kind"]
    cands = [c for c in cyls if c["id"] not in used and (c["kind"] == kind or (kind == "hole" and "hole" in c["kind"]))]
    cands = [c for c in cands if abs(float(np.asarray(c["axis_dir"]) @ axis)) > 0.9995]
    p3 = positions_3d(spec_f)
    if p3 is not None:
        targets: list[Optional[np.ndarray]] = list(p3)
    elif "positions" in spec_f:
        u, v = inplane_basis(axis)
        targets = [np.asarray(p[0]) * u + np.asarray(p[1]) * v for p in spec_f["positions"]]
    else:
        targets = [None] * spec_f["count"]
    if not cands:
        return [(None, t) for t in targets]
    nom_d = spec_f.get("diameter", (None, None))[0]
    cost = np.zeros((len(targets), len(cands)))
    for i, t in enumerate(targets):
        for j, c in enumerate(cands):
            d = 0.0 if t is None else _target_distance(c, t, axis, p3 is not None)
            tie = 0.0 if nom_d is None else 1e-3 * abs(c["diameter"] - nom_d)
            cost[i, j] = d + tie
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    assigned = dict(zip(rows.tolist(), cols.tolist()))
    out = []
    for i, t in enumerate(targets):
        if i in assigned:
            c = cands[assigned[i]]
            used.add(c["id"])
            out.append((c, t))
        else:
            out.append((None, t))
    return out


def verify(model: Model, spec: dict, features: Optional[dict] = None) -> VerifyResult:
    res = VerifyResult()
    summ = measure.summary(model)
    feats = features if features is not None else extract_features(model)
    res.features, res.summary = feats, summ
    checks = res.checks

    # ---- solid
    if model.is_exact:
        if spec["solid"]["valid"]:
            ok = bool(summ["valid"]) and summ["solids"] >= 1
            checks.append(Check("solid.valid", "B-rep is a valid solid", ok, True, ok, note="" if summ["solids"] else "no solid body (a sketch or face?)"))
        if spec["solid"]["single_body"]:
            checks.append(Check("solid.single_body", "exactly one solid body", summ["solids"] == 1, 1, summ["solids"]))
    else:
        checks.append(Check("solid.watertight", "mesh is watertight", bool(summ["watertight"]), True, bool(summ["watertight"])))
        if spec["solid"]["single_body"]:
            checks.append(Check("solid.single_body", "exactly one body", summ["bodies"] == 1, 1, summ["bodies"]))

    # ---- envelope
    env = spec["envelope"]
    ext = summ["extents"]
    for k, (nom, tol) in env.get("extents", {}).items():
        checks.append(_tol_check(f"envelope.{k}", f"axis-aligned extent along {k}", nom, tol, ext["xyz".index(k)]))
    if "max" in env:
        ok = all(e <= m + 1e-6 for e, m in zip(ext, env["max"]))
        checks.append(Check("envelope.max", "extents within the maximum envelope", ok, env["max"], ext))
    if "min" in env:
        ok = all(e >= m - 1e-6 for e, m in zip(ext, env["min"]))
        checks.append(Check("envelope.min", "extents at least the minimum envelope", ok, env["min"], ext))
    if "volume" in spec:
        v = summ.get("volume")
        vol = spec["volume"]
        if "nominal" in vol:
            nom, tol = vol["nominal"]
            checks.append(_tol_check("volume", "volume", nom, tol, v))
        else:
            ok = v is not None and vol.get("min", -1) <= v <= vol.get("max", float("inf"))
            checks.append(Check("volume", "volume within range", bool(ok), vol, v))

    # ---- features
    cyls = [c for c in feats["cylinders"] if c["kind"] in FULL_KINDS]
    used: set[str] = set()
    matched_by_ref: dict[str, dict] = {}
    for f in spec["features"]:
        pairs = _match_features(f, cyls, used)
        for i, (c, target) in enumerate(pairs):
            ref = f"{f['id']}.{i}"
            if c is None:
                where = "" if target is None else f" near {np.round(target, 3).tolist()}"
                checks.append(Check(f"{ref}.exists", f"{f['kind']} '{f['id']}' #{i}{where}", False, f["kind"], None, note="no matching feature (kind/axis)"))
                continue
            matched_by_ref[ref] = c
            checks.append(Check(f"{ref}.exists", f"{f['kind']} '{f['id']}' #{i} found as {c['id']}", True, f["kind"], c["kind"]))
            if "diameter" in f:
                nom, tol = f["diameter"]
                checks.append(_tol_check(f"{ref}.diameter", "diameter", nom, tol, c["diameter"], note="" if c["exact"] else f"fitted, rms {c['fit_rms']}"))
            if "length" in f:
                nom, tol = f["length"]
                checks.append(_tol_check(f"{ref}.length", "length / depth along the axis", nom, tol, c["height"]))
            if target is not None:
                axis = np.asarray(f["axis"])
                three_d = positions_3d(f) is not None
                d = _target_distance(c, target, axis, three_d)
                what = "distance from the target point to the axis segment" if three_d else "in-plane axis position"
                checks.append(Check(f"{ref}.position", what, d <= f["pos_tol"] + 1e-9, f"within {f['pos_tol']} of {np.round(target, 3).tolist()}", np.round(c["axis_point"], 4).tolist(), deviation=d))
            if "entry" in f and c["kind"] in {"blind_hole", "boss"}:
                axis = np.asarray(f["axis"])
                ends = c.get("open_ends") or [None, None]
                # c["start"] → c["end"] runs along c["axis_dir"]; map to the spec's axis direction
                same = float(np.asarray(c["axis_dir"]) @ axis) > 0
                open_plus, open_minus = (ends[1], ends[0]) if same else (ends[0], ends[1])
                want_plus = f["entry"] == "+"
                observed = "+" if open_plus and not open_minus else "-" if open_minus and not open_plus else "?"
                ok = (open_plus if want_plus else open_minus) is True and (open_minus if want_plus else open_plus) is False
                desc = "open end (mouth) faces" if c["kind"] == "blind_hole" else "free end faces"
                checks.append(Check(f"{ref}.entry", f"{desc} {f['entry']}axis", bool(ok), f["entry"], observed, note="" if ok else "feature enters from the wrong face"))
    if spec["exact_counts"]:
        extra = [f"{c['id']} {c['kind']} Ø{c['diameter']}" for c in cyls if c["id"] not in used]
        checks.append(Check("features.exact_counts", "no cylindrical features beyond the spec", not extra, "none extra", extra or "none"))

    # ---- planes
    for p in spec["planes"]:
        n = np.asarray(p["normal"])
        cands = [q for q in feats["planes"] if float(np.asarray(q["normal"]) @ n) > 0.9995]
        if "offset" in p:
            nom, tol = p["offset"]
            cands.sort(key=lambda q: abs(q["offset"] - nom))
            best = cands[0] if cands else None
            chk = _tol_check(f"plane.{p['id']}.offset", f"plane with normal {p['normal']} at offset (offset = normal · point)", nom, tol, None if best is None else best["offset"])
            if not chk.passed:
                hints = []
                if any(abs(q["offset"] + nom) <= tol for q in cands):
                    hints.append(f"a plane with this normal exists at offset {-nom} (sign convention: offset = normal · point)")
                opp = [q for q in feats["planes"] if float(np.asarray(q["normal"]) @ n) < -0.9995 and abs(q["offset"] - nom) <= tol]
                if opp:
                    hints.append(f"a plane with the opposite normal exists at offset {nom}")
                chk.note = "; ".join(hints)
            checks.append(chk)
        else:
            best = max(cands, key=lambda q: q["area"]) if cands else None
            checks.append(Check(f"plane.{p['id']}.exists", f"plane with normal {p['normal']}", best is not None, "exists", None if best is None else best["id"]))
        if "min_area" in p:
            area = None if best is None else best["area"]
            checks.append(Check(f"plane.{p['id']}.area", "plane area at least", area is not None and area >= p["min_area"], f">= {p['min_area']}", area))

    # ---- relations
    for i, rel in enumerate(spec["relations"]):
        a, b = rel["between"]
        ca, cb = matched_by_ref.get(a), matched_by_ref.get(b)
        rid = f"relation.{i}.{rel['type']}"
        if ca is None or cb is None:
            checks.append(Check(rid, f"{rel['type']} between {a} and {b}", False, note="one of the features was not found"))
            continue
        pa, pb = np.asarray(ca["axis_point"]), np.asarray(cb["axis_point"])
        da, db = np.asarray(ca["axis_dir"]), np.asarray(cb["axis_dir"])
        if rel["type"] == "distance":
            nom, tol = rel["value"]
            # distance between axes (perpendicular to the first axis)
            r_ = pb - pa
            d = float(np.linalg.norm(r_ - (r_ @ da) * da))
            checks.append(_tol_check(rid, f"axis distance between {a} and {b}", nom, tol, d))
        elif rel["type"] == "coaxial":
            r_ = pb - pa
            perp = float(np.linalg.norm(r_ - (r_ @ da) * da))
            parallel = abs(float(da @ db)) > 0.9995
            checks.append(Check(rid, f"{a} and {b} coaxial", parallel and perp <= rel["tol"], f"axis offset <= {rel['tol']}", round(perp, 4), deviation=perp))
        elif rel["type"] == "parallel":
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(da @ db))))))
            checks.append(Check(rid, f"{a} and {b} parallel", ang <= rel["tol"], f"angle <= {rel['tol']}°", round(ang, 4), deviation=ang))
        elif rel["type"] == "perpendicular":
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(da @ db))))))
            checks.append(Check(rid, f"{a} and {b} perpendicular", abs(ang - 90.0) <= rel["tol"], f"angle within {rel['tol']}° of 90", round(ang, 4), deviation=ang - 90.0))

    # ---- symmetry
    if spec["symmetry"]:
        sym = measure.mirror_symmetry(model)
        for n in spec["symmetry"]:
            nv = np.asarray(n)
            plane = next((p for p in sym["planes"] if abs(float(np.asarray(p["plane_normal"]) @ nv)) > 0.9995), None)
            cid = f"symmetry.{'xyz'[int(np.argmax(np.abs(nv)))]}"
            if plane is None:
                checks.append(Check(cid, f"mirror symmetry about the plane with normal {n}", False, "symmetric", None))
                continue
            ok = plane["symmetric"] and measure.feature_symmetry(feats, plane["plane_normal"], plane["offset"])["ok"]
            measured = f"p95 {plane['p95_deviation']}, p99 {plane['p99_deviation']}"
            if model.is_exact:  # kernel-exact confirmation: mirror the solid and compare volumes
                ex = measure.exact_symmetry(model, plane["plane_normal"], plane["offset"])
                if ex and "symmetric" in ex:
                    ok = ok and ex["symmetric"]
                    measured += f", mismatched volume {ex['mismatch_volume']} mm³ ({ex['mismatch_fraction']:.2e} of the part)"
            checks.append(Check(cid, f"mirror symmetry about the plane with normal {n}", bool(ok), "symmetric", measured))

    # ---- printability
    pr = spec["printability"]
    if "bed" in pr:
        ok = all(e <= b + 1e-6 for e, b in zip(sorted(ext), sorted(pr["bed"])))
        checks.append(Check("print.bed", "fits the print bed (any orientation of axes)", ok, pr["bed"], ext))
    if "min_wall" in pr:
        wt = measure.wall_thickness(model)
        mn = wt.get("p05")
        checks.append(Check("print.min_wall", "5th-percentile sampled wall thickness at least", mn is not None and mn >= pr["min_wall"], f">= {pr['min_wall']}", mn, note="sampled ray casting"))
    return res
