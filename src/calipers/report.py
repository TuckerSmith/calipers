"""The geometry report: a hierarchical, LLM-facing description of a model.

Three levels of detail, so a model can read the cheap summary first and drill down only when needed:

1. ``summary`` — envelope, oriented bbox, mass properties, validity, topology counts.
2. ``features`` — planes and cylindrical features (holes / bosses / pins / fillets) with parameters.
3. ``sections`` — cross-section loops (with circle/rectangle fits) on request.

``GeometryReport.to_text()`` renders a compact digest meant to be pasted into a prompt; ``to_dict()``
/ ``to_json()`` give the full structured data. Every block states whether it is exact (B-rep) or
fitted (mesh) — a reader should never have to guess how much to trust a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from calipers import measure
from calipers.features import extract_features
from calipers.model import Model

SCHEMA = "calipers.report/0.1"


@dataclass
class GeometryReport:
    source: dict
    summary: dict
    features: dict
    sections: list[dict] = field(default_factory=list)
    symmetry: Optional[dict] = None
    wall_thickness: Optional[dict] = None
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "summary": self.summary,
            "features": self.features,
            "sections": self.sections,
            "symmetry": self.symmetry,
            "wall_thickness": self.wall_thickness,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    # ------------------------------------------------------------------ LLM digest
    def to_text(self, max_planes: int = 8, max_groups: int = 12, max_cyls: int = 16) -> str:
        s, f = self.summary, self.features
        exact = "B-rep — exact" if s["exact"] else "mesh — fitted estimates"
        u = s.get("units", "mm")
        out: list[str] = []
        out.append(f"# Geometry report: {self.source.get('name')} ({exact})")
        ext = s["extents"]
        obb = s["oriented_bbox"]
        line = f"Envelope (axis-aligned): {_fmt3(ext)} {u}"
        if not obb["axis_aligned"]:
            axes = np.asarray(obb["axes"], dtype=float)
            z_like = np.abs(axes @ np.array([0.0, 0.0, 1.0])).max() > 0.995
            vol_ratio = float(np.prod(obb["extents"]) / max(float(np.prod(ext)), 1e-9))
            if z_like:  # a flat part simply rotated about Z: report its natural frame
                line += f"; rotated about Z by {obb['rotation_about_z_deg']:.1f}°, natural frame {_fmt3(obb['extents'])} {u}"
            elif vol_ratio < 0.9:
                line += f"; a tilted minimum box would be {_fmt3(obb['extents'])} {u} ({vol_ratio:.0%} of the envelope volume)"
        out.append(line + ".")
        vol = s.get("volume")
        vol_s = f"{vol:.1f} {u}³" if vol is not None else f"n/a ({s.get('volume_note')})"
        com = s.get("center_of_mass")
        out.append(f"Volume {vol_s} · surface {s['surface_area']:.1f} {u}² · centre of mass ({_fmt3(com, ', ')}).")
        if s["kind"] == "brep":
            ft = ", ".join(f"{v} {k}" for k, v in s["face_types"].items())
            out.append(f"Topology: {'valid' if s['valid'] else 'INVALID'} solid, {s['solids']} body/bodies, {s['faces']} faces ({ft}), {s['edges']} edges.")
        else:
            wt = "watertight" if s["watertight"] else "NOT watertight"
            out.append(f"Mesh: {s['triangles']} triangles, {s['vertices']} vertices, {wt}, {s['bodies']} body/bodies.")
        if self.symmetry:
            sym = [p for p in self.symmetry["planes"] if p["symmetric"]]
            near = [p for p in self.symmetry["planes"] if p.get("surface_symmetric") and not p["symmetric"]]
            if sym:
                out.append("Mirror symmetry: " + "; ".join(
                    f"plane normal ({_fmt3(p['plane_normal'], ', ')}) at offset {p['offset']:.3f} (p95 dev {p['p95_deviation']:.3f})" for p in sym) + ".")
            for p in near:
                chk = p.get("feature_check", {})
                out.append(
                    f"Mirror symmetry (surface only): plane normal ({_fmt3(p['plane_normal'], ', ')}) at offset {p['offset']:.3f} "
                    f"is symmetric to p99 {p['p99_deviation']:.3f} {u}, but {len(chk.get('unmatched', []))} feature(s) have no mirror twin: "
                    f"{', '.join(chk.get('unmatched', []))}."
                )
            if not sym and not near:
                b = self.symmetry["planes"][0]
                out.append(
                    f"Mirror symmetry: none within tol {self.symmetry['tolerance']:.2f}; closest candidate normal "
                    f"({_fmt3(b['plane_normal'], ', ')}) at offset {b['offset']:.3f} with p95 dev {b['p95_deviation']:.3f} {u}."
                )
        # planes
        planes = f["planes"]
        out.append("")
        out.append(f"## Planes ({len(planes)} distinct{' , exact' if f['exact'] else ', fitted'})")
        for p in planes[:max_planes]:
            out.append(f"- {p['id']}: normal ({_fmt3(p['normal'], ', ')}) at offset {p['offset']:.3f}, area {p['area']:.1f} {u}², centre ({_fmt3(p['center'], ', ')}), {p['regions']} region(s)")
        if len(planes) > max_planes:
            out.append(f"- … {len(planes) - max_planes} more (see JSON)")
        # cylinders
        cyls = f["cylinders"]
        groups = f["cylinder_groups"]
        c = f["counts"]
        out.append("")
        out.append(f"## Cylindrical features ({c['holes']} holes/bores, {c['bosses_pins_shafts']} bosses/pins/shafts, {c['partial_cylinders']} partial/fillet-like)")
        for g in groups[:max_groups]:
            pts = "; ".join(f"({_fmt3(a, ', ')})" for a in g["axis_points"][:6])
            more = f" … +{g['count'] - 6}" if g["count"] > 6 else ""
            out.append(f"- {g['count']}× {g['kind']} Ø{g['diameter']:.3f} × {g['height']:.3f} long — axis mid-points: {pts}{more}")
        if len(groups) > max_groups:
            out.append(f"- … {len(groups) - max_groups} more groups (see JSON)")
        for pat in f.get("patterns", []):
            if pat["type"] == "circular":
                out.append(f"- pattern: {pat['count']}× {pat['kind']} Ø{pat['diameter']:.3f} on a circle of radius {pat['pitch_radius']:.3f} centred ({_fmt3(pat['center'], ', ')}), pitch {pat['angular_pitch_deg']}° ({', '.join(pat['ids'])})")
            elif pat["type"] == "linear":
                out.append(f"- pattern: {pat['count']}× {pat['kind']} Ø{pat['diameter']:.3f} in a line along ({_fmt3(pat['direction'], ', ')}), pitch {pat['pitch']:.3f} ({', '.join(pat['ids'])})")
            else:
                out.append(f"- pattern: {pat['rows']}×{pat['cols']} grid of {pat['kind']} Ø{pat['diameter']:.3f}, pitch {pat['pitch'][0]:.3f} × {pat['pitch'][1]:.3f} ({', '.join(pat['ids'])})")
        for sl in f.get("slots", []):
            fit = "" if sl["exact"] else f", fit rms {sl['fit_rms']:.4f}"
            out.append(f"- {sl['id']} {sl['kind']}: width {sl['width']:.3f} × length {sl['length']:.3f}, depth {sl['depth']:.3f}, centre ({_fmt3(sl['center'], ', ')}), along ({_fmt3(sl['direction'], ', ')}), axis ({_fmt3(sl['axis_dir'], ', ')}){fit}")
        partial = [x for x in cyls if x["kind"] in {"partial", "fillet_candidate", "slot_end"}]
        if partial:
            radii = sorted({round(x["radius"], 2) for x in partial})
            out.append(f"- partial cylinders / fillet candidates: {len(partial)} with radii {radii[:10]}")
        listed = [x for x in cyls if x["kind"] not in {"partial", "fillet_candidate", "slot_end"}][:max_cyls]
        if listed:
            out.append("  Detail (first %d): " % len(listed))
            for x in listed:
                fit = "" if x["exact"] else f", fit rms {x['fit_rms']:.4f}"
                out.append(f"  - {x['id']} {x['kind']} Ø{x['diameter']:.3f}, axis dir ({_fmt3(x['axis_dir'], ', ')}) through ({_fmt3(x['axis_point'], ', ')}), from ({_fmt3(x['start'], ', ')}) to ({_fmt3(x['end'], ', ')}), length {x['height']:.3f}{fit}")
        unc = f["unclassified_regions"]
        if unc:
            out.append("")
            out.append(f"## Unclassified curved surfaces: {len(unc)} (largest {unc[0]['area']:.1f} {u}²)" + (
                "" if f["exact"] else " — free-form or non-cylindrical geometry; use sections/renders to inspect"))
        # sections
        for sec in self.sections:
            out.append("")
            pl = sec["plane"]
            tag = "exact" if sec.get("exact") else "fitted"
            out.append(f"## Section by plane origin ({_fmt3(pl['origin'], ', ')}) normal ({_fmt3(pl['normal'], ', ')}) [{tag}]")
            for L in sec.get("loops", []):
                if "circle" in L:  # mesh section
                    desc = f"{L['type']} loop, area {L['area']:.2f}"
                    if L["circle"]["is_circle"]:
                        desc += f", CIRCLE Ø{L['circle']['diameter']:.3f} centred ({_fmt3(L['circle']['center_3d'], ', ')})"
                    elif L["rectangle"]["is_rectangle"]:
                        desc += f", RECTANGLE {L['rectangle']['size'][0]:.3f} × {L['rectangle']['size'][1]:.3f} at {L['rectangle']['angle_deg']:.1f}°"
                    else:
                        desc += f", free-form (bbox {_fmt3(L['bbox_2d'], ', ')})"
                    out.append(f"- {L.get('id','')}: {desc}")
                else:  # exact section
                    desc = f"{L['type']} loop of {L['n_edges']} edges {L['edge_types']}"
                    if L.get("is_circle"):
                        desc += f", CIRCLE Ø{L['diameter']:.3f} centred ({_fmt3(L['center_3d'], ', ')})"
                    elif L.get("circle_radii"):
                        desc += f", contains arcs of radii {L['circle_radii']}"
                    out.append(f"- {desc}")
        if self.wall_thickness:
            w = self.wall_thickness
            if "min" in w:
                out.append("")
                out.append(f"## Wall thickness (sampled, {w['n_hits']} rays): min {w['min']:.3f}, p05 {w['p05']:.3f}, median {w['median']:.3f} {u}")
        if self.notes:
            out.append("")
            out.append("Notes: " + " ".join(self.notes))
        return "\n".join(out)


def _fmt3(v: Sequence[float], sep: str = " × ") -> str:
    return sep.join(f"{float(x):.3f}" for x in v)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def build_report(
    model: Model,
    sections: Sequence[str] = (),
    symmetry: bool = True,
    wall: bool = False,
    exact_sections: bool = True,
) -> GeometryReport:
    """Run the standard interrogation and package it as a :class:`GeometryReport`."""
    summ = measure.summary(model)
    feats = extract_features(model)
    secs = []
    for spec in sections:
        origin, normal = measure.parse_plane(spec)
        sec = measure.section(model, origin, normal)
        sec["spec"] = spec
        if exact_sections and model.is_exact:
            ex = measure.section_exact(model, origin, normal)
            if ex and "loops" in ex:
                sec["exact_loops"] = ex["loops"]
        secs.append(sec)
    sym = measure.mirror_symmetry(model) if symmetry else None
    if sym:  # surface-level symmetry from sampling; feature-level confirmation from the extracted features
        for p in sym["planes"]:
            chk = measure.feature_symmetry(feats, p["plane_normal"], p["offset"])
            p["surface_symmetric"] = bool(p["symmetric"])
            p["feature_check"] = chk
            p["symmetric"] = bool(p["symmetric"] and chk["ok"])
    wt = measure.wall_thickness(model) if wall else None
    src = {"name": model.name, "path": model.source_path, "kind": model.kind, "units": model.units}
    return GeometryReport(source=src, summary=summ, features=feats, sections=secs, symmetry=sym, wall_thickness=wt, notes=list(model.notes))
