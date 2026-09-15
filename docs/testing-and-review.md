# Testing and adversarial review — how calipers stays honest

The value of this project is in whether its numbers can be trusted. So verification is not a
phase; it is the architecture. This document is the process, in the order the evidence gets
stronger. Every layer is automated except the last, and the last has a script.

## 1. Ground-truth unit tests (`tests/test_bracket.py`, `test_robustness.py`, `test_review_cases.py`)

Parts are *built in code* with build123d from known numbers, exported to STEP and STL, and the
assertions use the construction numbers — never the tool's own output. Regression cases from every
review live here with a docstring saying what went wrong before the fix.

## 2. Standard artefacts with published nominals (`test_nist_artifact.py`, `test_benchy.py`)

The NIST additive-manufacturing test artifact (STEP and STL must agree feature for feature) and
the 3DBenchy (published nominal dimensions must be recovered from a non-watertight organic mesh).
These catch drift on real-world tessellation and mesh defects that synthetic parts never show.

## 3. Teacher / student red team (`calipers redteam`, `test_redteam.py`)

`calipers/redteam.py` builds random parts from recipes — plates, discs, tubes, L-brackets; through,
blind and counterbored holes; bosses with base fillets; chamfered hole mouths; slots; corner
fillets; holes through upright walls; arbitrary rigid transforms; fine/medium/coarse tessellation —
each with its ground truth, then scores:

* the **exact B-rep path** against the recipe (kinds, diameters, positions, heights), and
* the **mesh path** against the recipe *and* against the B-rep path (plane counts).

The B-rep path is the *teacher*: it reads analytic surfaces, so where the fitted *student* path
disagrees with it the student is wrong (or the recipe is — both are findings). A fixed seed range
runs in CI and must be 100 % correct; larger sweeps (`calipers redteam --n 200 --seed 1000`) are run
before a release and every failing seed is reproducible with `--keep`. On 15 Sep 2026 a 128-seed
sweep surfaced two algorithm bugs (chamfer facets accepted as planes at coarse tessellation;
tangent fillet strips absorbed into cylinder fits) and four recipe bugs within an hour — the harness
paid for itself immediately.

## 4. Contracts for generated parts (`calipers verify`, `calipers run --spec`)

Generated geometry is never judged by eye. A spec turns requirements into executable checks
(`calipers/contracts.py`): every check reports required vs measured and the deviation, and the
provenance lint (`calipers/provenance.py`) rejects any dimension the author did not source. The
*generator* is an LLM and is expected to be wrong at first; the *verifier* is deterministic, cheap
and never lies. This is the teacher/student split applied to design: the kernel teaches, the model
learns per iteration.

### 4b. Fit contracts against a reference (Phase 3)

When the spec names a reference, the checks are between two bodies. For STEP vs STEP the overlap
volume and the closest approach come from the kernel (exact). For meshes — scans are meshes — the
verifier samples the reference surface (vertices + area-uniform points), tests them against the
closed generated part (any reference point inside the part is an overlap, whatever the scan's
topology), and casts rays per direction to find the nearest wall; the *minimum* over samples is the
gap, so a single high spot is enough to fail. Keep-out / keep-in regions are solids intersected
with the part (OCCT for B-reps, manifold for meshes). `tests/test_phase3.py` builds a device, an
enclosure with a known clearance and a keep-out box that overlaps a wall by exactly 1 mm, and
asserts the volumes and gaps from the construction numbers; `examples/benchy_enclosure` is the
unattended end-to-end case.

## 5. Independent adversarial review (a script, not a vibe)

After each phase, a separate agent — with no stake in the code — is given the modules and this
instruction: *"Construct parts that break it. Report only what you reproduced, with the
construction, observed vs expected, and a proposed fix."* Its findings become regression tests
(section 1) before the fixes are considered done. Phase 1's review found 12 real issues; the rate at
which a fresh reviewer finds new ones is the project's real quality metric. Suggested cadence: one
review per phase, plus one whenever the geometry math changes.

Reviewer prompt template: `docs/review-prompt.md`.

## 6. Architectural notes: harnesses, teacher/student, distribution

*Why not train a model?* Everything above works because the kernel is an oracle. A learned
model in the loop would inherit the oracle's checks, not replace them; training is worth
revisiting only when the benchmark harness (Phase 4) shows tool-augmented frontier models hitting
a wall on a class of parts.

*Distribution that pays:* generation is expensive and stochastic, verification is cheap and
deterministic, so the right parallelism is **best-of-N with the verifier as judge** — several
generator samples (or agents) for one spec, verified in parallel, the best repaired further.
The MCP tools are stateless and side-effect free per call, so any orchestrator (Claude Code
sub-agents, a workflow, a queue) can fan out without coordination.

*Distribution that does not pay:* splitting one part's geometry across agents. A part is one
consistent B-rep; contracts are global (symmetry, exact counts). Keep one owner per part.
