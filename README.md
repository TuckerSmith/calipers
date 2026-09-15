# calipers

**Calipers for AI** — kernel-exact interrogation, verification and generation of 3D geometry for
LLM-driven CAD.

An LLM cannot read a triangle mesh, and it should not have to. `calipers` turns any STL/OBJ/3MF or
STEP file into a structured, dimensionally exact description a model can reason about, and gives it
instruments (sections, distances, symmetry, renders) to measure anything it is unsure of. Every number
is tagged **exact** (read from analytic B-rep surfaces) or **fitted** (estimated from a mesh, with the
fit residual), so the model always knows how much to trust it.

```
$ calipers report bracket.stl -s z=0

# Geometry report: bracket (mesh — fitted estimates)
Envelope (axis-aligned): 60.000 × 40.000 × 13.000 mm.
Volume 12669.5 mm³ · surface 6138.9 mm² · centre of mass (0.000, 0.000, 0.464).
Mesh: 2044 triangles, 1020 vertices, watertight, 1 body/bodies.
Mirror symmetry: plane normal (1.000, 0.000, 0.000) at offset 0.000 (p95 dev 0.000); plane normal (0.000, 1.000, 0.000) ...

## Planes (7 distinct, fitted)
- P01: normal (0.000, 0.000, -1.000) at offset 2.500, area 2353.0 mm², centre (0.000, 0.000, -2.500), 1 region(s)
...
## Cylindrical features (2 holes/bores, 1 bosses/pins/shafts, 4 partial/fillet-like)
- 2× through_hole Ø5.000 × 5.000 long — axis mid-points: (-20.000, 0.000, 0.000); (20.000, 0.000, 0.000)
- 1× boss Ø12.000 × 8.000 long — axis mid-points: (0.000, 0.000, 6.500)
- partial cylinders / fillet candidates: 4 with radii [3.0]
## Section by plane origin (0.000, 0.000, 0.000) normal (0.000, 0.000, 1.000) [fitted]
- loop_01: inner loop, area 19.63, CIRCLE Ø4.999 centred (-20.000, 0.000, 0.000)
```

## Install

```
pip install -e ".[dev,png]"        # build123d (OCCT), trimesh, shapely, matplotlib, typer, pytest, cairosvg
python -m pytest -q                # ≈4 min; NIST AM test artifact, 3DBenchy, pinned red-team seeds
```

## Commands

| command | what it gives you |
|---|---|
| `calipers report FILE [-s z=12] [--json out.json] [--wall]` | the full report: summary, planes, cylindrical features, sections, symmetry |
| `calipers summary FILE` | envelope, oriented bbox, volume, area, centre of mass, validity, topology counts |
| `calipers features FILE` | planes and cylinders (holes / bosses / pins / shafts / fillet candidates) as JSON |
| `calipers section FILE -p z=12 [--png sec.png]` | cross-section loops with circle / rectangle fits (exact OCCT loops for STEP) |
| `calipers render FILE -o dir [--views iso,front,top,right]` | shaded orthographic PNGs; hidden-line SVG/PNG for STEP |
| `calipers export FILE out.step\|.stl\|.3mf\|.obj` | format conversion (mesh → B-rep is refused: that is reconstruction) |
| `calipers verify FILE spec.yaml` / `calipers run part.py --spec spec.yaml` | contracts against a model / execute + lint + report + verify a script |
| `calipers generate enclosure\|mount --ref device.stl -o out/` | sourced script + spec around a reference, then the verify loop |
| `calipers best a.py b.py c.py --spec spec.yaml` | best-of-N: rank candidate scripts by the verifier |
| `calipers diff a.step b.step` | what changed between two revisions (kernel-exact for STEP) |

Python: `from calipers import load, build_report`; `rep = build_report(load("part.step"), sections=["z=0"])`;
`rep.to_text()` for the digest, `rep.to_dict()` / `rep.to_json()` for the data.

## What it measures today (Phase 1)

- **Summary**: axis-aligned envelope, minimum-volume oriented box (recovers a part's natural frame when
  it is rotated), volume, surface area, centre of mass, principal axes, watertightness / validity,
  triangle or face counts (by surface type for B-reps).
- **Planes**: grouped coplanar surfaces with normal, offset, area, centre and extent.
- **Cylindrical features**: diameter, axis, start/end, length, angular coverage, concave/convex and a
  *kind*: `through_hole`, `blind_hole`, `internal_bore`, `boss`, `shaft`, `cylinder`,
  `fillet_candidate`, `partial`. Identical features are grouped ("16× boss Ø4.000 × 7.000").
- **Sections**: closed loops classified outer/inner, with circle and rectangle fits; exact edge types
  and radii from OCCT for STEP input.
- **Symmetry**: mirror planes (world and principal axes, offset refined) with 95th-percentile deviation.
- **Wall thickness**: sampled inward ray casting (min / p05 / median).

Validated against a ground-truth bracket (mesh path reproduces the B-rep path exactly), the NIST
additive-manufacturing test artifact (all 35 cylindrical features and every plane ≥ 1 mm² agree
between STEP and STL within 0.01 mm) and the 3DBenchy (published nominals recovered: 60 × 31 × 48
envelope, chimney bore Ø3.00 × 11.00, hawsepipes Ø4.00, rear window Ø9.00 / Ø12.00 with 0.30 flange,
cargo box 8.00 × 7.00). See `tests/`.

## Generate and verify (Phase 2)

Write the requirements as a spec (`calipers-mcp` → `spec_schema`, or `src/calipers/spec.py`), then
let a model write build123d code whose every dimension is declared with a source:

```python
PARAMS = {"hole_d": (5.0, "spec:mount_holes.diameter"), "wall": (2.4, "assumption:3 perimeters")}
P = {k: v[0] for k, v in PARAMS.items()}
...
result = bp.part
```

```
$ calipers run part.py --spec spec.yaml --out out/
# Execution OK in 3.1s → out/result.step
# Provenance lint: PASS — 18 parameters, 0 naked numbers, 0 problems
# Geometry report: result (B-rep — exact) ...
# Contract check: PASS (43/43 checks passed)
[ok  ] mount_holes.0.diameter: diameter — required 4.3 ± 0.05, measured 4.3 (dev +0.0000)
...
```

Failures come back as exact deviations, execution errors with the failing line, and every unsourced
number — so the model repairs instead of guessing. `calipers verify part.step spec.yaml` checks an
existing file; `calipers lint part.py` runs the provenance check alone; `calipers api Hole` prints
a build123d signature.

**MCP:** `pip install -e .` then add `{"mcpServers": {"calipers": {"command": "calipers-mcp"}}}` to
Claude Desktop's config (or `claude mcp add calipers -- calipers-mcp` for Claude Code). Tools:
`report`, `report_json`, `section`, `measure_distance`, `render`, `spec_schema`, `verify`,
`lint_provenance`, `run_code`, `run_candidates`, `diff`, `reference_summary`, `generate`, `export`,
`api_help`, `redteam`.

**Red team:** `calipers redteam --n 50 --seed 0 --out scoreboard.json` builds random parts with
known ground truth and scores both feature paths; every failing seed is reproducible with `--keep`.
See `docs/testing-and-review.md` for the whole process.

## Design around a reference (Phase 3)

Put the existing part in the spec and state how the new one must relate to it. Nothing about the
reference is typed by hand — the verifier loads it, places it and measures:

```yaml
references: [{id: device, path: device.stl, place: {rotate: [0, 0, 0], translate: [0, 0, 0]}}]
keep_out:  [{id: usb, from: device, face: "+x", depth: 20, pad: 1}]     # a cable needs this volume
fit:
  - {type: clearance, ref: device, min: 0.3}                              # no overlap, ≥ 0.3 everywhere
  - {type: gap, ref: device, directions: {"+x": [0.3, 1.0], "-z": [0, 0.5]}}   # held, not rattling
  - {type: enclosed, ref: device, min_fraction: 0.85, open: ["+z"]}
```

`calipers generate enclosure --ref device.stl --wall 2 --clearance 0.5 --open +z -o out/` writes a
build123d script whose numbers cite the reference (`measured:device.bbox_max.z`) and the spec above,
then runs it; `generate mount` puts standoffs at the reference's measured through holes.
`examples/benchy_enclosure` is the 3DBenchy (an organic, non-watertight mesh) in a printed enclosure:
generated and verified with no human in the loop, 16/16 contracts. Specs also take `kind: slot`
features, and the report groups grids / lines / bolt circles of identical features.

## Roadmap

Phase 4: evaluation harness (CADGenBench, CADTestBench, own parts), pockets/chamfers as spec
features, sphere/cone/torus fits, FreeCAD handoff polish. `docs/approach-v0.1.md` has the reasoning
and the alternatives considered.

## Test fixtures

- `tests/fixtures/nist_am_test_artifact.{step,stl}` — NIST Additive Manufacturing Test Artifact
  (Moylan et al.), a US Government work in the public domain. https://www.nist.gov/el/intelligent-systems-division-73500/production-systems-group/nist-additive-manufacturing-test
- `tests/fixtures/3DBenchy.stl` — #3DBenchy by Creative Tools, released to the public domain. https://www.3dbenchy.com/

## License

MIT.
