# Adversarial review prompt (copy into a fresh agent)

You are reviewing `calipers` (installed in editable mode at <repo>; `python -m pytest -q` runs the
tests). Read `CLAUDE.md` first. Review these modules for numerical and logic bugs: <list>.

Be adversarial and empirical: do not just read — *construct* parts with build123d, export STL/STEP,
run `from calipers import load, build_report` (and `calipers.contracts.verify`, `calipers.sandbox.run_code`
for Phase 2) and compare the output with the numbers you built the part from. Try the cases the
author probably did not: coaxial features from opposite faces, features that break out of walls,
tangent blends, very small and very large parts, slicer-grade tessellation, dirty meshes, specs with
contradictory or borderline tolerances, generated code that games the provenance lint.

Report concisely: a numbered list of confirmed bugs, each with (a) the reproducing construction in
1–3 lines, (b) observed vs expected, (c) a specific proposed fix naming the function. Then the cases
that worked. Do not edit files under the repo — report only. Skip anything you cannot reproduce.
