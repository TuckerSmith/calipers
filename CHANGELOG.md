# Changelog

## 0.3.0 — 2026-09-15 — Phase 3: design around a reference

- `reference`: a spec may name **references** (STL/OBJ/3MF/STEP, optionally placed by rotate/translate).
  **keep_out / keep_in** regions (explicit boxes and cylinders, or boxes derived from a reference's
  envelope: `from: device, face: "+x", depth: 20`) are checked by kernel-exact booleans for B-reps
  (manifold mesh booleans otherwise). **fit** contracts: `clearance` (no overlap + closest approach,
  exact for STEP vs STEP), `gap` (per direction, the nearest part wall seen from the reference
  surface must lie in [min, max] — held without rattling), `enclosed` (fraction of the reference
  surface covered, minus declared open directions).
- `generators`: `calipers generate enclosure|mount --ref FILE` writes a *sourced* build123d script
  (`measured:<ref>.bbox_max.z`, `measured:<ref>.C03.axis_point.x` …) plus the spec with its fit
  contracts, then runs the ordinary verify loop. Enclosure: cavity = envelope + clearance, walls,
  floor opposite the open face, optional corner radius. Mount: plate + standoffs at the reference's
  measured through holes, screw holes through them, coaxial relations, "rests on the standoffs" gap.
- Provenance: `measured:<reference id>.<path>` sources are resolved against the loaded reference
  (envelope fields and feature ids) and their values compared (0.05 mm); the sandbox injects
  `REFERENCES = {id: path}` so scripts can load the reference themselves.
- Slots: `kind: slot` in specs (width, length, direction, depth); both feature paths pair
  half-cylinders into `through_slot` / `blind_slot` with centre, direction and depth; `exact_counts`
  now covers slots; the red team scores slots on both paths. Patterns: grid / linear / circular
  groups of identical features in the report.
- Best-of-N: `calipers best a.py b.py c.py --spec` / MCP `run_candidates` runs candidates in parallel
  and ranks them by the verifier (executed < lint clean < contracts passed < fewest failures <
  smallest total deviation).
- Version diff: `calipers diff a.step b.step` / MCP `diff`; `run_code` diffs against the previous
  `result.step` in the same work dir. Kernel-exact material added/removed for STEP pairs.
- Sandbox: address-space and file-size limits on the script subprocess (a resource limit, not a
  security boundary — documented as such). MCP server ported to `mcp` 2.x (`MCPServer`) with 1.x kept.
- Exit criterion met: `examples/benchy_enclosure` — an enclosure for the 3DBenchy (an organic,
  non-watertight mesh) is generated, executed, linted and passes 16/16 contracts unattended (≈55 s).
  Tests: 88 (+ pinned red-team seeds, now including slots).
- Findings fixed on the way: build123d's `intersect` on an imported STEP compound returns a
  `ShapeList` (a naive `.volume` read 0 → booleans now go to OCCT directly); rays cast from a
  reference surface missed a part face touching it (gap read the *next* wall) → rays start 1 µm
  behind the surface; `Shape.intersect` returns `None` for an empty result.
- Self-review pass (8 adversarial constructions: mesh part vs STEP reference, placed references,
  swallowed device, tilted part, 45° slots on both paths, two references + bolt keep-out, cwd
  independence, one-sided tight cavity): all correct. One gap found and fixed: `spec:` sources
  pointing at untoleranced values (`fit.0.min`, `printability.min_wall`) were not value-compared.
- CI: the GitHub Actions lint step had been red since the Phase 1 push (ruff's default rule set
  drifted; 210 import-order findings), so the test jobs never ran. Rule set pinned in
  `pyproject.toml` (`E4 E7 E9 F I`), imports sorted. `docs/images/` holds renders of the two
  generated examples (also on the status page for this release).
- Not done (carried to Phase 4): pockets and chamfers as spec features, sphere/cone/torus fits on
  meshes, fillet rings, reference features in `relations`, an independent adversarial review of
  Phase 3 (budget), CADGenBench harness.

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
