# Changelog

## 0.2.0 — 2026-09-15 — Phase 2: generate & verify

- `spec`: requirements schema (YAML/JSON) — envelope, volume, solid validity, cylindrical features
  with diameter/length/positions (2-D in-plane or 3-D on-axis)/entry face, exact counts, planes
  (offset = normal · point), relations (distance, coaxial, parallel, perpendicular), symmetry,
  printability — validated with precise error messages.
- `contracts`: `verify(model, spec)` turns every requirement into a pass/fail check with required vs
  measured and the deviation; instances assigned by the Hungarian algorithm; kernel-exact mirror
  test (symmetric-difference volume) for B-reps.
- `provenance`: "no naked numbers" lint — every dimension must be declared in `PARAMS` with a
  `spec:/measured:/derived:/standard:/assumption:` source; constant folding, string-smuggling and
  PARAMS re-definition are caught; `spec:` sources are resolved against the spec and their values
  compared to the nominal.
- `sandbox`: `run_code()` executes build123d (or CadQuery) code in a subprocess with a timeout,
  refuses non-solid results, reports structured errors (type, message, failing line), then
  reports/verifies/lints; `api_help()` for build123d signatures.
- `mcp_server` (`calipers-mcp`): report, report_json, section, measure_distance, render,
  spec_schema, verify, lint_provenance, run_code, export, api_help, redteam.
- `redteam`: random parametric parts with ground truth (plates, discs, tubes, L-brackets with wall
  holes; through/blind/counterbored/chamfered holes; bosses with base fillets; slots; corner
  fillets; rigid transforms; fine/medium/coarse tessellation) scored against both feature paths —
  CLI + scoreboard + pinned CI seeds. First 128-seed sweep found two algorithm bugs (chamfer facets
  taken as planes at coarse tessellation; tangent fillet strips absorbed into cylinder fits) — fixed.
- CLI: `verify`, `lint`, `run`, `api`, `redteam`. Tests: 70 (+ pinned red-team seeds).
- Independent adversarial review of Phase 2 found 12 issues (blind hole from the wrong face
  passing, 3-D positions meaning mid-height, small asymmetric notch passing symmetry, twelve lint
  evasions, unchecked `spec:` sources, broken CadQuery interop, phantom footer line numbers,
  scripts exiting early, sketches passing `solid.valid`, plane-offset sign confusion, greedy
  instance matching, spec validation gaps) — all fixed with regression tests.
- End-to-end demo: a fresh agent given only the CLI produced a 4-feature sensor mount from a spec
  with 43/43 contracts passing and a clean provenance lint on its first geometry attempt.

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
