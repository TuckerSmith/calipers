"""Feasibility spike: what does each representation give an LLM to reason about?

1. Build a known part with build123d (ground truth dimensions are known).
2. Export STEP (B-rep) and STL (mesh).
3. Re-import both and interrogate them headlessly:
   - mesh: bbox, volume, watertight, planar facet segmentation, hole detection
   - B-rep: exact face types, cylinder radii/axes, planar face areas
4. Emit a structured JSON "geometry report" — the kind of thing an LLM can actually reason over.
5. Produce an orthographic SVG render headlessly (no GPU) to confirm VLM-style feedback is possible.
"""
import json, math, time
import numpy as np
from build123d import *  # noqa

t0 = time.time()
# ---- 1. Ground truth part: 60x40x5 plate, two Ø5 through-holes at ±20mm, 3mm corner fillets, Ø12 boss 8mm tall
with BuildPart() as bp:
    Box(60, 40, 5)
    fillet(bp.edges().filter_by(Axis.Z), radius=3)
    with Locations((20, 0), (-20, 0)):
        Hole(radius=2.5)
    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):
        Circle(6)
    extrude(amount=8)
part = bp.part
export_step(part, "spike_part.step")
export_stl(part, "spike_part.stl", tolerance=0.01, angular_tolerance=0.1)
print(f"built + exported in {time.time()-t0:.2f}s")

# ---- 2. Mesh-side interrogation (what a scan / STL gives you)
import trimesh
m = trimesh.load("spike_part.stl", force="mesh")
report = {"source": "spike_part.stl", "mesh": {}}
mm = report["mesh"]
mm["faces"] = int(len(m.faces))
mm["vertices"] = int(len(m.vertices))
mm["watertight"] = bool(m.is_watertight)
mm["bbox_min"] = [round(float(v), 3) for v in m.bounds[0]]
mm["bbox_max"] = [round(float(v), 3) for v in m.bounds[1]]
mm["extents"] = [round(float(v), 3) for v in m.extents]
mm["volume_mm3"] = round(float(m.volume), 2)
mm["surface_area_mm2"] = round(float(m.area), 2)
mm["center_of_mass"] = [round(float(v), 3) for v in m.center_mass]
# planar facet segmentation: groups of coplanar adjacent triangles
facets = m.facets
facet_areas = m.facets_area
order = np.argsort(-facet_areas)
planes = []
for i in order[:6]:
    n = m.facets_normal[i]
    tri = facets[i]
    pts = m.vertices[m.faces[tri]].reshape(-1, 3)
    planes.append({
        "normal": [round(float(v), 3) for v in n],
        "area_mm2": round(float(facet_areas[i]), 2),
        "offset_along_normal": round(float(np.dot(pts.mean(axis=0), n)), 3),
    })
mm["largest_planar_faces"] = planes
# hole detection via 2D cross-section at mid-plate height: count closed loops, fit circles
section = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
loops = []
if section is not None:
    path2d, _ = section.to_2D()
    for poly in path2d.polygons_full:
        outer = poly.exterior
        loops.append({"type": "outer", "area_mm2": round(poly.area, 2),
                      "bbox": [round(v, 2) for v in poly.bounds]})
        for hole in poly.interiors:
            xs, ys = np.array(hole.coords).T
            cx, cy = xs.mean(), ys.mean()
            r = float(np.mean(np.hypot(xs - cx, ys - cy)))
            loops.append({"type": "hole", "center": [round(cx, 2), round(cy, 2)],
                          "fit_diameter_mm": round(2 * r, 3)})
mm["section_z0"] = loops

# ---- 3. B-rep-side interrogation (what STEP gives you: exact semantics)
imported = import_step("spike_part.step")
bb = imported.bounding_box()
br = report["brep"] = {}
br["bbox_min"] = [round(v, 3) for v in (bb.min.X, bb.min.Y, bb.min.Z)]
br["bbox_max"] = [round(v, 3) for v in (bb.max.X, bb.max.Y, bb.max.Z)]
br["volume_mm3"] = round(imported.volume, 3)
br["is_valid"] = bool(imported.is_valid)
faces = imported.faces()
by_type = {}
for f in faces:
    by_type[str(f.geom_type)] = by_type.get(str(f.geom_type), 0) + 1
br["face_count_by_type"] = by_type
cyls = []
for f in faces.filter_by(GeomType.CYLINDER):
    ax = f.axis_of_rotation
    cyls.append({
        "radius_mm": round(f.radius, 4),
        "axis_dir": [round(v, 3) for v in (ax.direction.X, ax.direction.Y, ax.direction.Z)],
        "axis_point": [round(v, 3) for v in (ax.position.X, ax.position.Y, ax.position.Z)],
        "area_mm2": round(f.area, 3),
    })
br["cylindrical_faces"] = sorted(cyls, key=lambda c: (c["radius_mm"], c["axis_point"]))
br["planar_faces_top5_by_area"] = [
    {"normal": [round(v, 3) for v in (f.normal_at().X, f.normal_at().Y, f.normal_at().Z)],
     "area_mm2": round(f.area, 3), "center": [round(v, 3) for v in (f.center().X, f.center().Y, f.center().Z)]}
    for f in sorted(faces.filter_by(GeomType.PLANE), key=lambda f: -f.area)[:5]
]

# ---- 4. Headless orthographic render to SVG (VLM-style feedback without a GPU)
exp = ExportSVG(scale=2)
exp.add_layer("vis", line_color=(0, 0, 0), line_weight=0.5)
exp.add_layer("hid", line_color=(120, 120, 120), line_weight=0.25, line_type=LineType.ISO_DASH)
vis, hid = imported.project_to_viewport((100, -100, 80))
exp.add_shape(vis, layer="vis")
exp.add_shape(hid, layer="hid")
exp.write("spike_iso.svg")
report["render"] = "spike_iso.svg written"
report["elapsed_s"] = round(time.time() - t0, 2)

json.dump(report, open("spike_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
