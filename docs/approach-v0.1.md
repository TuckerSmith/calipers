# AI-Native 3D Understanding & Design — Draft Approach v0.1

*Draft for decision, 15 Sep 2026. Not a spec; a recommendation with the reasoning shown so you can push back on it.*

## 1. The problem, reframed

Your starting hypothesis is that an LLM can't understand a triangle mesh, so we should convert the mesh into some matrix or multidimensional array that "contains the shape." The first half is right and well documented. NVIDIA's LLaMA-Mesh, the most serious attempt to feed raw meshes to an LLM as text, has to quantize every coordinate to 64 bins per axis (about 1 mm on a 60 mm part) and caps meshes at 500 faces to fit an 8k-token context. That is toy precision; the test bracket in the spike below tessellates to 2,044 faces and it is a trivial part. Raw meshes are the wrong input.

The second half — a voxel/occupancy matrix — I'd argue against as the *primary* representation, for three reasons. Token cost scales as O(n³): 0.1 mm resolution over a 100 mm part is 10⁹ cells, and even a coarse 64³ grid is 262k values that an LLM reasons over poorly. It's lossy in exactly the dimension you care about (a Ø5.00 hole becomes "somewhere between 4.9 and 5.1 depending on grid alignment"). And it has no semantics: nothing in the array says "this is a through-hole on a 40 mm bolt pattern." Voxels and signed-distance grids are the right tool for some *computations* (collision, fit, coarse topology, ML training signals), just not for what the LLM reads.

The reframe that makes this tractable: the LLM does not need to *see* the shape. It needs three things a human engineer also uses, because no engineer reasons over triangles either.

1. **A semantic, exact, editable description** of geometry — what a CAD feature tree or a parametric script is.
2. **Instruments** — the ability to measure anything it is unsure about instead of guessing (calipers, section views, feature detection).
3. **A kernel that owns the truth** — the LLM states intent as code, an exact geometry engine executes it, and the results are measured back to the LLM.

Dimensional accuracy then stops being a property of the model's perception and becomes a property of the loop: intent → exact geometry → measurement → comparison against the requirement → repair. The LLM never emits a number that hasn't been checked.

## 2. What already exists (so we build on it, not beside it)

This area moved fast in 2025–26, and it's worth being clear-eyed about what's solved:

- **CAD-as-code is the settled research representation.** Three May-2026 benchmarks — Text2CAD-Bench (600 human-verified parts), BenchCAD (17,900 execution-verified programs across 106 industrial part families), and CADTests — all target **CadQuery** Python. Hugging Face's CADGenBench is tool-agnostic (submit a STEP) but its reference baseline uses **build123d**. Both run on the same OCCT kernel and interoperate.
- **Tool-augmented frontier models beat trained specialists without training.** CAD-Assistant (ICCV 2025) wrapped GPT-4o around FreeCAD's Python API plus renderers and JSON parameter extractors and beat supervised baselines zero-shot. CADTests turned requirements into executable geometric assertions and used them as feedback: Claude 4.6 with tests reached a 0.625 requirement pass-rate versus 0.48 for a ReAct agent and 0.025 for the trained Text2CAD model. Same idea, in product form: pzfreo's build123d-mcp raised its CADGenBench score from 0.36 to 0.46 and validity from 88% to 100% purely by adding measure/render/validate tools; agentcad returns metrics plus a render on every execution.
- **Frontier models still fail in specific, measurable ways.** BenchCAD: vision-based QA trails code-based QA by 15–20 points (models read code better than pictures), models substitute simple sketch-extrude for lofts/sweeps/twists, and 64% of "successful" edits silently corrupt an unrelated feature. Text2CAD-Bench: invalidity rises from ~15% on simple parts to 70–90% once sweeps and shells appear.
- **Mesh → parametric ("understand an existing model") is the least-solved part in open source.** CAD-Recode (ICCV 2025) fine-tuned a 1.5B model to emit CadQuery from point clouds and is state of the art on DeepCAD/Fusion360/CC3D, but on normalized, dataset-style shapes. Commercially, Backflip AI launched a scan/STL → editable parametric CAD copilot on 3 Aug 2026 ($20/mo, Fusion add-in, "moderate-complexity machined and turned parts"). The open MCP servers mostly measure imports at bounding-box level.

**Honest read:** "LLM writes CAD code, tools give it feedback" is no longer novel — it is the 2026 baseline. The gap, and the part that matches your four goals, is on the *understanding* side: a rich, LLM-oriented interrogation layer for arbitrary STL/STEP inputs, requirements captured as executable contracts, and design-around-a-reference (enclosures, mounts, mating parts) with fit verified rather than eyeballed. That is also exactly the 3D-printing use case your earlier scan → Zoo → FreeCAD pipeline was reaching for.

## 3. Representation options considered

| Representation | Exactness | LLM legibility | Editable | Role in the proposed system |
|---|---|---|---|---|
| Triangle mesh (STL/OBJ/3MF) | tessellated approx. | very poor — thousands of triangles, no semantics | poor | **input/output only** (scans, slicer) |
| Voxel / occupancy / SDF grid | resolution-bound, O(n³) | poor as text | poor | internal: collision, fit, coarse topology |
| Point cloud | sampled | poor as text; good for ML encoders | none | scan intake; optional CAD-Recode initializer |
| Neural implicit / latent | n/a | opaque | none | not used for precision work |
| B-rep (STEP, OCCT) | **exact analytic surfaces** | medium — verbose but semantic (face types, radii, axes) | medium | **ground truth artifact**; interrogated via kernel |
| Feature history (DeepCAD JSON, Fusion Gallery) | exact | good | good | dataset formats only; too tied to one CAD |
| **CAD-as-code (build123d / CadQuery)** | exact (OCCT) | **best — models are trained on code; diffable, testable** | **best** | **canonical LLM-facing representation** |
| **Structured geometry report (JSON)** | exact, derived | very good | read-only | **the "understanding" layer for imports** |
| Multi-view renders / dimensioned drawings | visual | good for sanity, poor for exact numbers | n/a | secondary channel for VLM checks |

Kernel language choice: **build123d primary, CadQuery interop.** build123d is the modern API on the same OCCT kernel (CADGenBench's baseline, agentcad, build123d-mcp all use it), and CadQuery keeps us compatible with the research benchmarks. OpenSCAD deserves a mention because one practitioner bake-off found it far less error-prone for LLMs (tiny declarative API, Python libraries invite hallucinated methods), but it has no B-rep, no STEP export, and no real fillets — a bad trade for precision work. The mitigation for API hallucination is the loop itself: execute, read the error, repair, plus a docs-lookup tool.

## 4. Recommended architecture: *code + instruments + contracts*

```
                 ┌──────────────────────────────────────────────┐
  STL/STEP/scan  │  PERCEPTION  (geometry interrogation)         │
  ─────────────▶ │  import → repair → segment → feature-detect   │──▶ geometry_report.json
                 │  queries: bbox, axes, symmetry, planes,       │    (summary ▸ features ▸ raw, on demand)
                 │  cylinders/holes/bosses, sections, silhouettes│
                 └──────────────────────────────────────────────┘
  requirements   ┌──────────────────────────────────────────────┐
  (text/YAML)    │  CONTRACTS  (requirements as executable tests) │
  ─────────────▶ │  dimensional asserts, topology, keep-in/out,  │──▶ pass/fail + deviations
                 │  fit vs reference, printability (P2S bed etc.)│
                 └──────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────┐
  LLM (Claude)   │  GENERATION LOOP                              │
  ◀────────────▶ │  spec → build123d code → execute (OCCT) →     │──▶ STEP (exact) + STL/3MF (print)
                 │  measure → compare to contracts → repair      │    + render + diff vs last version
                 └──────────────────────────────────────────────┘
                 ┌──────────────────────────────────────────────┐
                 │  INTERFACE: MCP server (Claude Desktop/Code/  │
                 │  Cowork) + CLI + folder drop → FreeCAD/slicer │
                 └──────────────────────────────────────────────┘
```

**Perception.** Converts any mesh or STEP into a hierarchical report the LLM can read at the level of detail it needs: a one-paragraph summary (extents, volume, watertightness, symmetry, principal axes), then a feature list (planar regions with normals and offsets, cylindrical regions with radius/axis/through-or-blind, fillets, patterns), then raw geometry only on request. From B-rep this is exact (OCCT gives face types and parameters directly); from meshes it uses region growing and RANSAC primitive fitting, cross-sections, and silhouettes. The LLM can also *ask*: "section at z=12", "distance between hole A and face B", "what's the wall thickness here." This is the layer that makes the system's understanding dimensionally accurate rather than impressionistic, and it's the least-served gap in open tooling.

**Contracts.** Requirements live in a small schema (dimensions with tolerances, features, relationships, fit against a reference, printability constraints) and compile to executable assertions run against the produced B-rep — the CADTests idea, but as a first-class workflow artifact rather than a benchmark. Every generated part ships with its test results and a deviation table. This is where your "high degree of fidelity, precision, and accuracy" goal is actually enforced.

**Generation loop.** Claude writes build123d, the kernel executes, the perception layer measures the result, the contracts grade it, and the model repairs. Reference-conditioned design (an enclosure for a scanned device, a bracket to mate with an existing part) works the same way, with the reference's report as input and fit tests (clearance, keep-in/keep-out, IoU against the reference where reconstruction is the goal) as the contract.

**Interface.** An MCP server is the natural fit for how you already work with Claude; it makes the whole thing available in Claude Desktop, Claude Code, and Cowork with no UI to build. A CLI covers scripting and CI. Output lands as STEP (exact, editable in FreeCAD) plus STL/3MF for the P2S, dropped into your project folder like the earlier pipeline.

## 5. Feasibility spike (done today)

To make sure this isn't hand-waving, I built a known part (60×40×5 plate, two Ø5 holes at ±20 mm, R3 corner fillets, Ø12 boss) with build123d, exported STEP and STL, then re-imported both cold and interrogated them headlessly in the cloud sandbox. Total runtime 0.41 s.

- From the **STL alone**: extents 60.000 × 40.000 × 13.000; watertight; a z=0 cross-section found the outer loop plus two holes at (±20.00, 0.01) with fitted diameter **4.999 mm**; the six largest planar facets came back with correct normals and offsets.
- From the **STEP**: valid solid; 7 planar + 7 cylindrical faces; cylinders reported with exact radii 2.5 / 3.0 / 6.0 mm and axes; volume 12,669.80 mm³ (mesh: 12,669.45 — 0.003 % tessellation error).
- A hidden-line isometric SVG rendered with no GPU, suitable for VLM sanity checks.

Everything needed (CadQuery 2.8, build123d 0.11, OCCT 7.9 via OCP, trimesh 5.1, manifold3d, shapely) installed cleanly with pip, so the harness can run in the cloud sandbox, in CI, or on your laptop.

## 6. Phased roadmap

**Phase 1 — Foundation (target: 1–2 working sessions).** Repo skeleton with tests. Geometry core: import STL/OBJ/3MF/STEP, mesh repair, `geometry_report.json` (summary → features → raw), measurement primitives (bbox, sections, distances, face/cylinder detection), headless render, STEP/STL export. CLI. Validated on the spike part and a handful of your own STLs. *Exit criterion: reports on real parts are correct to caliper precision and readable by Claude without further explanation.*

**Phase 2 — Generate & verify.** Requirements schema and contract runner; build123d execution sandbox with error capture and docs lookup; the generation loop as an MCP server; version diffing. *Exit: Claude produces a simple bracket/enclosure from a YAML spec with all contracts passing, unattended.*

**Phase 3 — Reference-conditioned design & reconstruction.** Mesh primitive segmentation (RANSAC planes/cylinders/spheres/cones), hole/boss/pocket/fillet detection, keep-in/keep-out volumes, fit tests, enclosure and mount generators; parametric reconstruction of simple scanned parts with IoU/Chamfer reporting; optional CAD-Recode as an initializer. *Exit: a printed enclosure for a scanned object fits on the first or second try, and the deviation report predicted it.*

**Phase 4 — Interface polish & evaluation.** FreeCAD handoff, docs, and a benchmark harness reusing CADGenBench and CADTestBench plus your own parts so progress is a number, not a feeling. Only after this would fine-tuning a small model be worth discussing; the evidence says tool-augmented frontier models win first.

## 7. How to employ me (working model)

- **Effort level.** Keep max effort for this decision, the Phase 1 geometry core, and any OCCT/kernel debugging — those are numerically subtle and a wrong foundation is expensive. Scale to the default effort for scaffolding, docs, test boilerplate, and routine iteration; the harness catches mistakes there and the 3× spend isn't buying much. Practically: max for Phase 1 design and core, default from Phase 2 on, bump back up when something is stuck.
- **Division of labor.** Given your two working modes, I'd suggest I build the harness autonomously in the cloud sandbox (it's headless and reproducible there), commit it to a git repo and your connected "3D Printing" folder, and bring you the decisions that shape the product: the report schema, the requirements schema, and the MCP tool surface. You own the physical loop — print, measure with calipers, feed back — and design reviews. For pieces you'd rather learn by doing (e.g., the mesh segmentation math), I switch to explain-then-you-execute.
- **Sub-agents.** Not needed for Phase 1. Useful from Phase 2 for independent tracks (mesh segmentation vs MCP scaffolding in parallel) and for an independent review pass over geometry math before it ships.
- **Cadence.** Each session ends with tests green, a short changelog in the project, and the next decision queued for you.

## 8. Risks and how we handle them

Frontier models hallucinate API calls in rich Python libraries — mitigated by execute-and-repair, a docs-lookup tool, and a curated set of build123d idioms in the system prompt. Complex operations (lofts, sweeps, shells) degrade sharply per the benchmarks — Phase 2 starts with sketch/extrude/revolve/fillet/pattern parts, which cover most 3D-printing work, and grows the vocabulary deliberately. Edits silently corrupting unrelated features — every edit reruns the full contract suite and diffs against the previous version. Scan noise breaking primitive fitting — repair pass, tolerance-aware RANSAC, and a "confidence" field in the report so the LLM knows which numbers are exact and which are fitted. Scope creep — the exit criteria above are the gate for each phase.

## 9. Decisions I need from you

1. **First use case to optimize:** (a) new parts from requirements, (b) understanding/reconstructing existing STLs, or (c) designing around a reference object (enclosures, mounts). My recommendation is **(c)** — it exercises all four goals and matches your printing workflow — with (a) falling out of it almost for free.
2. **Kernel language:** build123d primary with CadQuery interop (recommended), or CadQuery primary for closer alignment with the research benchmarks.
3. **Interface:** MCP server + CLI first (recommended); FreeCAD workbench or web viewer later, if at all.
4. **Where the code lives:** a GitHub repo (recommended) mirrored into your "3D Printing" folder, or the folder alone.
5. **Inputs to test with:** any STL/STEP files you'd like Phase 1 validated against, and whether you still have a Zoo token (optional — not on the critical path).

Once you answer, I'll turn Phase 1 into a task list and start building.

## Sources

Text2CAD-Bench — https://arxiv.org/html/2605.18430v1 · BenchCAD — https://arxiv.org/html/2605.10865v1 · CADTests — https://arxiv.org/html/2605.07807v1 · CADGenBench — https://github.com/huggingface/cadgenbench · CAD-Recode — https://arxiv.org/abs/2412.14042 · CAD-Assistant — https://arxiv.org/html/2412.13810v2 · LLaMA-Mesh — https://arxiv.org/html/2411.09595v1 · build123d-mcp — https://github.com/pzfreo/build123d-mcp · agentcad — https://agentcad.dev/ · Backflip AI launch — https://www.businesswire.com/news/home/20260803007022/en/ · AI CAD tool comparison — https://blog.neural4d.com/comparisons/best-ai-cad-generator/ · OpenSCAD vs CadQuery vs build123d bake-off — https://grandpacad.com/en/blog/openscad-vs-cadquery-vs-build123d
