"""Red-team harness in CI: a fixed seed range must stay fully correct.

If this fails, run ``calipers redteam --seed <s> --n 1 --keep out/`` for the listed seed, look at the
recipe and the report, and decide whether the bug is in the calipers or in the recipe (both count).
"""

from __future__ import annotations

import pytest

from calipers.redteam import run

SEEDS = list(range(0, 24))


@pytest.mark.timeout(1500)
def test_redteam_fixed_seeds():
    res = run(SEEDS, verbose=False)
    s = res["summary"]
    failed = [c for c in res["cases"] if not c.get("ok")]
    detail = "\n".join(f"seed {c['seed']}: " + (c.get("error") or "; ".join(f"{k}: missed {v['missed']} spurious {v['spurious']} planes {v['planes']}" for k, v in c["paths"].items())) for c in failed)
    assert s["failed_seeds"] == [], detail
    assert s["brep_recall"] == 1.0 and s["mesh_recall"] == 1.0 and s["mesh_precision"] == 1.0
    assert s["max_mesh_diameter_error"] <= 0.01
