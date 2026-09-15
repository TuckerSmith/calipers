# Changelog

## 0.1.0 — 2026-09-15 — Phase 1: the digital calipers

- `Model`: unified loader for STL/OBJ/PLY/3MF/OFF/GLB (trimesh) and STEP/BREP (build123d/OCCT);
  lazy fine tessellation of B-reps; exact-vs-fitted provenance everywhere.
- `measure`: summary (envelope, minimum-volume oriented box via hull-normal + rotating calipers,
  volume, area, centre of mass, principal axes, validity/topology), cross-sections with circle and
  rectangle fits (plus exact OCCT sections for B-reps), closest-surface distances, sampled wall
  thickness, mirror-symmetry detection with offset refinement.
- `features`: planes and cylindrical features. B-rep: read from OCCT faces, seam-split faces merged.
  Mesh: smooth-region segmentation → peel (whole-region cylinder, coplanar patches with a strip
  guard, RANSAC cylinders for fillets/partial bores) → merge coaxial pieces → classify by probing
  the solid past each end (through/blind hole, boss, shaft, fillet candidate).
- `report`: `GeometryReport` with JSON schema `calipers.report/0.1` and an LLM text digest.
- `render`: shaded orthographic views (numpy painter + matplotlib, no GPU), hidden-line SVG/PNG
  for B-reps, annotated section plots.
- `export`: STEP/BREP/STL/3MF/OBJ/PLY (with an OCCT fallback STEP writer for imported compounds).
- CLI: `calipers report|summary|features|section|render|export`.
- Tests (49): ground-truth bracket in STEP and STL, NIST AM test artifact STEP-vs-STL agreement
  (35 cylindrical features, all planes ≥ 1 mm²), 3DBenchy published nominals, arbitrary rotation,
  coarse tessellation, sphere/cone rejection, oblique hole, CLI/IO round trips, plus regression
  cases from an independent adversarial review (below).
- Review fixes (same day): whole-region cylinder test now also checks face centroids (with a
  chord-sag allowance) and normal alignment, so a rectangle's circumcircle or a cone can no longer
  pass as a cylinder; planar patches surrounded by fillets are kept (strip guard uses sibling
  patches instead of an area fraction); RANSAC samples by area, tries the best few seeds and
  re-collects inliers after refinement; end probing adds a deeper ring and an on-axis point (base
  flares, bottom fillets, drill points); merged coverage is a union of azimuths; B-rep coverage is
  read from the face's U range; symmetry requires p95/p99/RMS *and* feature-level twins (an extra
  hole now breaks symmetry) and reports surface-only symmetry separately; oriented bbox prefers
  the world frame on ties and measures yaw on the flattest axis; section loops are reported in
  world in-plane coordinates; degenerate/duplicate triangles are cleaned on load.
