# calipers — project context for Claude sessions

Read this first. It carries the decisions and state that a fresh session needs; the long-form
reasoning is in `docs/approach-v0.1.md`.

## What this is

**Calipers for AI**: kernel-exact interrogation, verification and (from Phase 2) generation of 3D
geometry for LLM-driven CAD. Owner: Tucker. Working principle: *the model never guesses a
dimension; it measures.* The LLM reads/writes parametric CAD code (build123d on OCCT), the kernel
owns the truth, and every number handed to the model is tagged exact (B-rep) or fitted (mesh).

Four goals, in priority order for the first use case ("design around a reference object"):
1. Import and *understand* an existing model (STL/OBJ/3MF/STEP) — Phase 1, done.
2. Store models in an LLM-legible way — layered: STEP as truth, code as canonical editable form,
   `geometry_report.json` / text digest as the model-facing view. Done in principle (Phase 1).
3. Create parts from requirements and/or reference models with verified accuracy — Phase 2/3.
4. An effective interface — MCP server + CLI. CLI done; MCP is Phase 2.

## Decisions (don't relitigate without a reason)

- Kernel language: **build123d** primary, CadQuery interop (same OCCT kernel). Not OpenSCAD (no B-rep).
- Representation: never voxels/point clouds/raw mesh text as the LLM-facing form. Reports + code.
- Reconstruction scope: **feature-level** (planes, cylinders, holes, bosses, fillets, sections), not
  full parametric reconstruction of arbitrary scans. Backflip AI does that commercially; not needed.
- Dimension provenance ("no naked numbers") becomes a core principle in Phase 2: every literal in
  generated geometry must trace to a requirement, a measurement taken by this tool, or a rule.
- Physical print-and-caliper loop: **out of scope** (Tucker's call, 15 Sep 2026). Digital accuracy only.
- Zoo/KCL: not on the critical path; optional alternate generator for benchmarking only.
- Success is measured: CADGenBench composite + our own STEP-vs-STL / nominal-dimension tests.
- Interface: MCP server + CLI. No GUI. FreeCAD handoff = drop STEP/STL into Tucker's folder.

## Layout

```
src/calipers/
  model.py     Model: mesh and/or B-rep, lazy tessellation, units (mm)
  measure.py   summary, oriented bbox, sections (+exact OCCT sections), symmetry, wall thickness
  features.py  planes + cylinders (holes/bosses/pins/shafts/fillets) from B-rep (exact) or mesh (fitted)
  report.py    GeometryReport: JSON (schema calipers.report/0.1) and LLM text digest
  render.py    headless shaded views, hidden-line SVG for B-reps, section plots
  export.py    STEP/BREP/STL/3MF/OBJ/PLY export
  cli.py       `calipers report|summary|features|section|render|export`
tests/         pytest; fixtures: NIST AM test artifact (STEP+STL, public domain), 3DBenchy (public domain)
docs/          approach-v0.1.md (the plan), this file
spikes/        the original feasibility spike
```

## How the mesh feature path works (the non-obvious part)

1. Triangles → smooth regions (adjacent dihedral < 32°, so slicer-grade tessellations stay connected).
2. Each region is *peeled*: whole-region cylinder test → coplanar patches (with a guard so that
   facet strips of curved surfaces are not taken as planes) → RANSAC cylinders on the remainder
   (catches fillets as partial cylinders) → leftovers reported as unclassified curved regions.
3. Cylinder pieces sharing axis/radius/concavity are merged; kind (through/blind hole, boss, shaft,
   fillet_candidate) is decided by probing the solid just past each end, next to the wall.
4. Everything carries `fit_rms` and `coverage_deg`. Tolerance: 0.5 % of radius, clamped 0.03–0.12 mm.

Known limits: spheres/cones/tori are reported as unclassified (by design, no phantom cylinders);
a coplanar patch is dropped as a "facet strip" when its boundary is ≥ 75 % tangent-smooth *and* it
either has a similar-area smooth sibling or is < 2 % of its region; faces < 0.25 mm² are ignored;
non-watertight meshes get approximate volume and inside tests (nearest-face signed distance).
`height` of a cylinder is the axial extent of its *surface* (an oblique hole reads longer than its
centre-line length). Symmetry has two levels: `surface_symmetric` (sampled p95/p99/RMS) and
`symmetric` (also every full cylinder has a mirrored twin).

Performance: typical parts < 2 s; a 225k-triangle organic mesh (Benchy) takes ~40 s in feature
extraction (RANSAC on the free-form hull) and ~10 s in symmetry. Candidates if that matters:
embree ray casting, a compiled RANSAC scorer, skipping RANSAC on regions with no ⟂-normal pairs.

## Working agreement

- Tests must stay green: `python -m pytest -q` (≈90 s; Benchy tests dominate).
- Every measurement path must state exact vs fitted. Never round away below 1 µm in code; reports
  round to 4 decimals.
- Prefer adding a ground-truth test (build the part with build123d, assert on the construction
  numbers) over trusting output by eye.
- Session cadence: finish with green tests, update `CHANGELOG.md`, note the next decision for Tucker.
- Tucker prefers explain-then-he-executes for things he wants to learn, autonomous execution
  otherwise; keep decisions that shape the product (schemas, tool surface) explicit and small.

## Roadmap

- Phase 1 (done): geometry core + CLI + tests against the bracket, NIST artifact, Benchy.
- Phase 2: requirements schema → executable contracts; build123d execution sandbox with error
  capture and docs lookup; MCP server (`report`, `measure`, `section`, `render`, `execute`,
  `verify`, `export`); dimension provenance linter; version diffing.
- Phase 3: reference-conditioned design (keep-in/keep-out volumes, clearance/fit tests, enclosure &
  mount generators), coaxial/pattern grouping, counterbores, fillet rings, sphere/cone/torus fits.
- Phase 4: evaluation harness (CADGenBench, CADTestBench, own parts), docs, FreeCAD handoff polish.
