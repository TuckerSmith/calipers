"""Red-team harness: random parametric parts with known ground truth, scored against the calipers.

Why this exists: hand-written tests only cover the cases someone thought of. This module builds
parts from random recipes (plates, discs, L-brackets; through/blind/counterbored holes; bosses;
slots; corner fillets; arbitrary rigid transforms; several tessellation qualities), exports them as
STEP and STL, runs both feature paths and scores them against the recipe:

* **teacher / student**: the exact B-rep path is checked against the recipe (kinds, diameters,
  positions), and the mesh path is checked against the recipe *and* against the B-rep path
  (planes, counts). A disagreement is a bug in one of them, or a bug in the recipe — all three
  are worth finding.
* every failure is reproducible from its seed: ``calipers redteam --seed N --n 1 --keep DIR``.

The CLI writes a scoreboard (JSON + Markdown); ``tests/test_redteam.py`` pins a fixed seed range
with thresholds so regressions fail CI.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from calipers.features import extract_features
from calipers.model import Model

TESSELLATION = {"fine": (0.01, 0.1), "medium": (0.05, 0.3), "coarse": (0.1, 0.5)}


@dataclass
class GTCylinder:
    kind: str  # through_hole | blind_hole | boss
    diameter: float
    axis_point: np.ndarray  # mid-height point on the axis (world)
    axis_dir: np.ndarray  # unit
    height: float


@dataclass
class Recipe:
    seed: int
    base: str
    dims: dict
    cylinders: list[GTCylinder] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)
    fillet_r: float = 0.0
    transform: dict = field(default_factory=dict)
    tessellation: str = "fine"
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        c = ", ".join(f"{g.kind} Ø{g.diameter:.2f}×{g.height:.2f}" for g in self.cylinders)
        c += "".join(f", slot {sl['width']:.1f}×{sl['length']:.2f}" for sl in self.slots)
        return f"seed {self.seed}: {self.base} {self.dims}, fillet R{self.fillet_r}, [{c}], slots {len(self.slots)}, {self.tessellation}"


# ----------------------------------------------------------------------------- recipe → part
def _rand_free_spot(rng, taken: list[tuple[float, float, float]], L: float, W: float, margin: float, radius: float, tries: int = 60):
    """A random (x, y) inside ±(L/2 - margin), ±(W/2 - margin) that keeps ≥ 1.5 mm from other features."""
    if L / 2 - margin - radius <= 0 or W / 2 - margin - radius <= 0:
        return None  # no room for this feature on this base
    for _ in range(tries):
        x = rng.uniform(-L / 2 + margin + radius, L / 2 - margin - radius)
        y = rng.uniform(-W / 2 + margin + radius, W / 2 - margin - radius)
        if all(math.hypot(x - tx, y - ty) >= radius + tr + 1.5 for tx, ty, tr in taken):
            taken.append((x, y, radius))
            return x, y
    return None


def build_recipe(seed: int) -> tuple[object, Recipe]:
    """Build a random part and its ground truth. Returns (build123d Part, Recipe)."""
    from build123d import Axis, BuildPart, BuildSketch, Box, Cylinder, Location, Locations, Mode, SlotCenterPoint, extrude, fillet

    rng = np.random.default_rng(seed)
    base = rng.choice(["plate", "plate", "disc", "lbracket", "tube"])
    T = float(rng.uniform(3.0, 14.0))
    L = float(rng.uniform(30.0, 110.0))
    W = float(rng.uniform(20.0, 80.0))
    recipe = Recipe(seed=int(seed), base=str(base), dims={}, tessellation=str(rng.choice(["fine", "fine", "medium", "coarse"])))
    taken: list[tuple[float, float, float]] = []
    top_z = T / 2  # base is centred at the origin: z ∈ [-T/2, T/2]

    with BuildPart() as bp:
        if base == "disc":
            D = float(rng.uniform(30.0, 90.0))
            Cylinder(radius=D / 2, height=T)
            recipe.dims = {"D": round(D, 3), "T": round(T, 3)}
            # the disc's own wall is a convex cylinder with both ends free: kind "cylinder"
            recipe.cylinders.append(GTCylinder("cylinder", D, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), T))
            L = W = D * 0.7071  # inscribed square for feature placement
        elif base == "tube":
            D = float(rng.uniform(20.0, 60.0))
            wall = float(rng.uniform(1.5, 6.0))
            T = float(rng.uniform(10.0, 60.0))
            Cylinder(radius=D / 2, height=T)
            Cylinder(radius=D / 2 - wall, height=T + 2, mode=Mode.SUBTRACT)
            recipe.dims = {"D": round(D, 3), "wall": round(wall, 3), "T": round(T, 3)}
            recipe.cylinders.append(GTCylinder("cylinder", D, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), T))
            recipe.cylinders.append(GTCylinder("through_hole", D - 2 * wall, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), T))
            L = W = 0.0  # no further features on a tube
        elif base == "lbracket":
            Box(L, W, T)
            wall_t = float(rng.uniform(3.0, 8.0))
            wall_h = float(rng.uniform(15.0, 40.0))
            with Locations((0, W / 2 - wall_t / 2, T / 2 + wall_h / 2)):
                Box(L, wall_t, wall_h)
            recipe.dims = {"L": round(L, 3), "W": round(W, 3), "T": round(T, 3), "wall_t": round(wall_t, 3), "wall_h": round(wall_h, 3)}
            # holes through the upright wall (axis along Y)
            wall_holes: list[tuple[float, float, float]] = []
            for _ in range(int(rng.integers(0, 3))):
                d = float(rng.choice([3.0, 4.0, 5.0, 6.0]))
                if d > 0.5 * wall_h - 2 or d > 0.3 * L:
                    continue
                x = float(rng.uniform(-L / 2 + d + 2, L / 2 - d - 2))
                z = float(rng.uniform(T / 2 + d / 2 + 2, T / 2 + wall_h - d / 2 - 2))
                if any(math.hypot(x - px, z - pz) < d / 2 + pd / 2 + 1.5 for px, pz, pd in wall_holes):
                    continue
                wall_holes.append((x, z, d))
                cy = W / 2 - wall_t / 2
                with Locations((x, cy, z)):
                    Cylinder(radius=d / 2, height=wall_t + 2, rotation=(90, 0, 0), mode=Mode.SUBTRACT)
                recipe.cylinders.append(GTCylinder("through_hole", d, np.array([x, cy, z]), np.array([0.0, 1.0, 0.0]), wall_t))
            W = W - wall_t  # keep features off the wall footprint (plate region y < W/2 - wall_t)
        else:
            Box(L, W, T)
            recipe.dims = {"L": round(L, 3), "W": round(W, 3), "T": round(T, 3)}
        if base == "plate" and rng.random() < 0.6:
            r = float(rng.uniform(0.5, min(5.0, 0.2 * min(L, W))))
            fillet(bp.edges().filter_by(Axis.Z), radius=r)
            recipe.fillet_r = round(r, 3)
        margin = max(2.0, recipe.fillet_r + 1.0)
        # for the L-bracket the usable plate is shifted: centre features on the plate part only
        y_off = -recipe.dims.get("wall_t", 0.0) / 2

        # through holes
        for _ in range(int(rng.integers(0, 5)) if L > 0 else 0):
            d = float(rng.choice([1.5, 2.0, 2.5, 3.0, 3.2, 4.0, 5.0, 6.0, 6.5, 8.0, 10.0, 12.0]))
            if d > 0.4 * min(L, W):
                continue
            spot = _rand_free_spot(rng, taken, L, W, margin, d / 2)
            if spot is None:
                continue
            x, y = spot
            y += y_off
            with Locations((x, y, 0)):
                Cylinder(radius=d / 2, height=T + 2, mode=Mode.SUBTRACT)
            recipe.cylinders.append(GTCylinder("through_hole", d, np.array([x, y, 0.0]), np.array([0.0, 0.0, 1.0]), T))
        # counterbore one through hole (if any)
        cb_candidates = [g for g in recipe.cylinders if g.kind == "through_hole" and g.diameter <= 6.5 and T >= 6.0 and abs(g.axis_dir[2]) > 0.99]
        if cb_candidates and rng.random() < 0.5:
            g = cb_candidates[int(rng.integers(0, len(cb_candidates)))]
            D = round(g.diameter * float(rng.uniform(1.6, 2.2)), 2)
            depth = round(float(rng.uniform(0.25, 0.5)) * T, 2)
            with Locations((g.axis_point[0], g.axis_point[1], top_z - depth / 2 + 0.5)):
                Cylinder(radius=D / 2, height=depth + 1.0, mode=Mode.SUBTRACT)
            # the hole below the counterbore becomes shorter; the counterbore is a blind hole
            g.height = round(T - depth, 3)
            g.axis_point = np.array([g.axis_point[0], g.axis_point[1], -T / 2 + g.height / 2])
            recipe.cylinders.append(GTCylinder("blind_hole", D, np.array([g.axis_point[0], g.axis_point[1], top_z - depth / 2]), np.array([0.0, 0.0, 1.0]), depth))
            taken.append((g.axis_point[0], g.axis_point[1] - y_off, D / 2))
        # blind holes from the top
        for _ in range(int(rng.integers(0, 3)) if L > 0 else 0):
            d = float(rng.choice([2.0, 3.0, 4.0, 5.0, 6.0]))
            depth = round(float(rng.uniform(0.3, 0.8)) * T, 2)
            spot = _rand_free_spot(rng, taken, L, W, margin, d / 2)
            if spot is None:
                continue
            x, y = spot
            y += y_off
            with Locations((x, y, top_z - depth / 2 + 0.5)):
                Cylinder(radius=d / 2, height=depth + 1.0, mode=Mode.SUBTRACT)
            recipe.cylinders.append(GTCylinder("blind_hole", d, np.array([x, y, top_z - depth / 2]), np.array([0.0, 0.0, 1.0]), depth))
        # bosses on top
        for _ in range(int(rng.integers(0, 3)) if L > 0 else 0):
            d = float(rng.choice([4.0, 6.0, 8.0, 10.0, 12.0, 15.0]))
            h = round(float(rng.uniform(2.0, 15.0)), 2)
            spot = _rand_free_spot(rng, taken, L, W, margin, d / 2)
            if spot is None:
                continue
            x, y = spot
            y += y_off
            with Locations((x, y, top_z + h / 2)):
                Cylinder(radius=d / 2, height=h)
            recipe.cylinders.append(GTCylinder("boss", d, np.array([x, y, top_z + h / 2]), np.array([0.0, 0.0, 1.0]), h))
        # fillet the base of one boss (a torus that must not become a phantom cylinder)
        bosses = [g for g in recipe.cylinders if g.kind == "boss"]
        if bosses and rng.random() < 0.4:
            g = bosses[0]
            rf = float(rng.choice([0.5, 1.0, 1.5]))
            try:
                base_edges = bp.edges().filter_by(lambda e: str(e.geom_type).endswith("CIRCLE") and abs(e.center().Z - top_z) < 1e-6 and abs(e.radius - g.diameter / 2) < 1e-6 and abs(e.center().X - g.axis_point[0]) < 1e-3 and abs(e.center().Y - g.axis_point[1]) < 1e-3)
                if base_edges:
                    fillet(base_edges, radius=rf)
                    g.height = round(g.height - rf, 3)  # the cylindrical face now starts above the fillet
                    g.axis_point = np.array([g.axis_point[0], g.axis_point[1], top_z + rf + g.height / 2])
                    recipe.notes.append(f"boss base fillet R{rf}")
            except Exception:
                pass
        # chamfer the mouth of one through hole (a cone: hole gets shorter, no phantom cylinder)
        holes = [g for g in recipe.cylinders if g.kind == "through_hole" and abs(g.axis_dir[2]) > 0.99 and g.height >= T - 1e-6]
        if holes and rng.random() < 0.4:
            from build123d import chamfer

            g = holes[-1]
            ch = float(rng.choice([0.5, 1.0]))
            try:
                mouth = bp.edges().filter_by(lambda e: str(e.geom_type).endswith("CIRCLE") and abs(e.center().Z - top_z) < 1e-6 and abs(e.radius - g.diameter / 2) < 1e-6 and abs(e.center().X - g.axis_point[0]) < 1e-3 and abs(e.center().Y - g.axis_point[1]) < 1e-3)
                if mouth:
                    chamfer(mouth, length=ch)
                    g.height = round(g.height - ch, 3)
                    g.axis_point = np.array([g.axis_point[0], g.axis_point[1], -T / 2 + g.height / 2])
                    recipe.notes.append(f"hole chamfer {ch}")
            except Exception:
                pass
        # a slot (two half-cylinders joined by planes) — must NOT be reported as a hole
        if base not in ("disc", "tube") and rng.random() < 0.4:
            w = float(rng.choice([3.0, 4.0, 5.0, 6.0]))
            length = float(rng.uniform(2.0 * w, 4.0 * w))
            spot = _rand_free_spot(rng, taken, L, W, margin, length / 2)
            if spot is not None:
                x, y = spot
                y += y_off
                with BuildSketch(Location((0, 0, top_z))) as sk:
                    with Locations((x, y)):
                        SlotCenterPoint(center=(0, 0), point=(length / 2 - w / 2, 0), height=w)
                extrude(sk.sketch, amount=-(T + 1.0), mode=Mode.SUBTRACT)
                recipe.slots.append({"width": w, "length": length, "center": np.array([x, y, 0.0]), "direction": np.array([1.0, 0.0, 0.0]), "axis": np.array([0.0, 0.0, 1.0]), "depth": T})
    part = bp.part

    # random rigid transform
    if rng.random() < 0.6:
        axis_dir = rng.normal(size=3)
        axis_dir /= np.linalg.norm(axis_dir)
        angle = float(rng.uniform(5.0, 175.0))
        shift = rng.uniform(-40.0, 40.0, size=3)
        part = part.rotate(Axis((0, 0, 0), tuple(axis_dir)), angle).moved(Location(tuple(shift)))
        R = _rotation_matrix(axis_dir, math.radians(angle))
        for g in recipe.cylinders:
            g.axis_point = R @ g.axis_point + shift
            g.axis_dir = R @ g.axis_dir
        for sl in recipe.slots:
            sl["center"] = R @ sl["center"] + shift
            sl["direction"] = R @ sl["direction"]
            sl["axis"] = R @ sl["axis"]
        recipe.transform = {"axis": [round(float(v), 4) for v in axis_dir], "angle_deg": round(angle, 3), "shift": [round(float(v), 3) for v in shift]}
    return part, recipe


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


# ----------------------------------------------------------------------------- scoring
def _match(gt: GTCylinder, found: list[dict], d_tol: float, pos_tol: float, h_tol: float) -> Optional[dict]:
    best = None
    for c in found:
        if c["kind"] != gt.kind or abs(c["diameter"] - gt.diameter) > d_tol:
            continue
        if abs(float(np.asarray(c["axis_dir"]) @ gt.axis_dir)) < 0.9995:
            continue
        rel = np.asarray(c["axis_point"]) - gt.axis_point
        perp = rel - (rel @ gt.axis_dir) * gt.axis_dir
        if np.linalg.norm(perp) > pos_tol or abs(float(rel @ gt.axis_dir)) > h_tol + 0.5 * abs(c["height"] - gt.height):
            continue
        if abs(c["height"] - gt.height) > h_tol:
            continue
        err = abs(c["diameter"] - gt.diameter)
        if best is None or err < best[0]:
            best = (err, c)
    return None if best is None else best[1]


def score_case(seed: int, keep_dir: Optional[Path] = None) -> dict:
    from build123d import export_step, export_stl

    t0 = time.time()
    part, recipe = build_recipe(seed)
    workdir = keep_dir or Path(os.environ.get("CALIPERS_REDTEAM_TMP", "/tmp/calipers_redteam"))
    workdir.mkdir(parents=True, exist_ok=True)
    step = workdir / f"rt_{seed}.step"
    stl = workdir / f"rt_{seed}.stl"
    export_step(part, str(step))
    lin, ang = TESSELLATION[recipe.tessellation]
    export_stl(part, str(stl), tolerance=lin, angular_tolerance=ang)
    res = {"seed": seed, "recipe": recipe.describe(), "tessellation": recipe.tessellation, "n_gt": len(recipe.cylinders), "paths": {}}
    tol_by_tess = {"fine": (0.02, 0.05, 0.05), "medium": (0.05, 0.1, 0.1), "coarse": (0.1, 0.2, 0.2)}
    for label, path in (("brep", step), ("mesh", stl)):
        model = Model.load(path)
        feats = extract_features(model)
        full = [c for c in feats["cylinders"] if c["kind"] not in {"partial", "fillet_candidate", "slot_end"}]
        d_tol, pos_tol, h_tol = (0.005, 0.01, 0.01) if label == "brep" else tol_by_tess[recipe.tessellation]
        matched, misses, d_errs = 0, [], []
        used = set()
        for g in recipe.cylinders:
            m = _match(g, [c for c in full if c["id"] not in used], d_tol, pos_tol, h_tol)
            if m is None:
                misses.append(f"{g.kind} Ø{g.diameter:.2f}×{g.height:.2f} at {np.round(g.axis_point, 2).tolist()}")
            else:
                used.add(m["id"])
                matched += 1
                d_errs.append(abs(m["diameter"] - g.diameter))
        spurious = [f"{c['kind']} Ø{c['diameter']:.3f}×{c['height']:.2f}" for c in full if c["id"] not in used]
        slot_hits, slot_miss = 0, []
        used_slots: set[str] = set()
        for sl in recipe.slots:
            hit = None
            for s_ in feats.get("slots", []):
                if s_["id"] in used_slots or s_["kind"] != "through_slot":
                    continue
                if abs(s_["width"] - sl["width"]) > d_tol or abs(s_["length"] - sl["length"]) > 2 * d_tol or abs(s_["depth"] - sl["depth"]) > h_tol:
                    continue
                if abs(float(np.asarray(s_["direction"]) @ sl["direction"])) < 0.9995 or abs(float(np.asarray(s_["axis_dir"]) @ sl["axis"])) < 0.9995:
                    continue
                rel = np.asarray(s_["center"]) - sl["center"]
                if np.linalg.norm(rel - (rel @ sl["axis"]) * sl["axis"]) > pos_tol:
                    continue
                hit = s_
                break
            if hit is None:
                slot_miss.append(f"slot {sl['width']:.1f}×{sl['length']:.2f} at {np.round(sl['center'], 2).tolist()}")
            else:
                used_slots.add(hit["id"])
                slot_hits += 1
        spurious_slots = [f"{s_['kind']} {s_['width']}×{s_['length']}" for s_ in feats.get("slots", []) if s_["id"] not in used_slots]
        res["paths"][label] = {
            "slot_recall": (slot_hits / len(recipe.slots)) if recipe.slots else 1.0,
            "missed_slots": slot_miss,
            "spurious_slots": spurious_slots,
            "recall": matched / max(len(recipe.cylinders), 1),
            "precision": (len(full) - len(spurious)) / max(len(full), 1),
            "missed": misses,
            "spurious": spurious,
            "max_diameter_error": max(d_errs) if d_errs else 0.0,
            "planes": feats["counts"]["planes"],
            "partials": feats["counts"]["partial_cylinders"],
            "unclassified": feats["counts"]["unclassified_regions"],
        }
    b, m = res["paths"]["brep"], res["paths"]["mesh"]
    res["plane_count_agrees"] = abs(b["planes"] - m["planes"]) <= 0  # teacher vs student
    res["ok"] = (
        b["recall"] == 1.0 and b["precision"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0 and res["plane_count_agrees"]
        and b["slot_recall"] == 1.0 and m["slot_recall"] == 1.0 and not b["spurious_slots"] and not m["spurious_slots"]
    )
    res["elapsed_s"] = round(time.time() - t0, 2)
    if keep_dir is None:
        for p in (step, stl):
            try:
                p.unlink()
            except OSError:
                pass
    return res


def run(seeds: list[int], keep_dir: Optional[Path] = None, verbose: bool = True) -> dict:
    cases = []
    for s in seeds:
        try:
            r = score_case(s, keep_dir)
        except Exception as exc:  # a crash is a finding too
            r = {"seed": s, "ok": False, "error": f"{type(exc).__name__}: {exc}", "paths": {}}
        cases.append(r)
        if verbose:
            flag = "ok " if r.get("ok") else "FAIL"
            print(f"[{flag}] seed {s}: {r.get('recipe', r.get('error', ''))}", flush=True)
            if not r.get("ok"):
                for label, p in r.get("paths", {}).items():
                    if p["missed"] or p["spurious"] or p.get("missed_slots") or p.get("spurious_slots"):
                        print(f"       {label}: missed {p['missed'] + p.get('missed_slots', [])} spurious {p['spurious'] + p.get('spurious_slots', [])}", flush=True)
                if r.get("paths") and not r.get("plane_count_agrees", True):
                    print(f"       planes: brep {r['paths']['brep']['planes']} vs mesh {r['paths']['mesh']['planes']}", flush=True)
    n_ok = sum(1 for c in cases if c.get("ok"))
    summary = {
        "n": len(cases),
        "ok": n_ok,
        "pass_rate": n_ok / max(len(cases), 1),
        "brep_recall": float(np.mean([c["paths"]["brep"]["recall"] for c in cases if c.get("paths")])) if cases else 0,
        "mesh_recall": float(np.mean([c["paths"]["mesh"]["recall"] for c in cases if c.get("paths")])) if cases else 0,
        "mesh_precision": float(np.mean([c["paths"]["mesh"]["precision"] for c in cases if c.get("paths")])) if cases else 0,
        "max_mesh_diameter_error": max([c["paths"]["mesh"]["max_diameter_error"] for c in cases if c.get("paths")] or [0]),
        "failed_seeds": [c["seed"] for c in cases if not c.get("ok")],
    }
    return {"summary": summary, "cases": cases}


def to_markdown(result: dict) -> str:
    s = result["summary"]
    lines = [
        "# calipers red-team scoreboard",
        "",
        f"{s['ok']}/{s['n']} cases fully correct ({s['pass_rate']:.0%}). B-rep recall {s['brep_recall']:.3f}; mesh recall {s['mesh_recall']:.3f}, precision {s['mesh_precision']:.3f}; max mesh diameter error {s['max_mesh_diameter_error']:.4f} mm.",
        "",
        "| seed | tessellation | GT | brep R/P | mesh R/P | planes brep/mesh | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in result["cases"]:
        if "error" in c:
            lines.append(f"| {c['seed']} | – | – | – | – | – | crash: {c['error']} |")
            continue
        b, m = c["paths"]["brep"], c["paths"]["mesh"]
        missed = m["missed"] + b["missed"] + m.get("missed_slots", []) + b.get("missed_slots", [])
        spurious = m["spurious"] + b["spurious"] + m.get("spurious_slots", []) + b.get("spurious_slots", [])
        status = "ok" if c["ok"] else "; ".join(filter(None, ["missed: " + ", ".join(missed) if missed else "", "spurious: " + ", ".join(spurious) if spurious else "", "" if c["plane_count_agrees"] else "plane count differs"]))
        lines.append(f"| {c['seed']} | {c['tessellation']} | {c['n_gt']} | {b['recall']:.2f}/{b['precision']:.2f} | {m['recall']:.2f}/{m['precision']:.2f} | {b['planes']}/{m['planes']} | {status} |")
    return "\n".join(lines)


def write_scoreboard(result: dict, out_json: Path, out_md: Optional[Path] = None) -> None:
    out_json.write_text(json.dumps(result, indent=1, default=str))
    if out_md:
        out_md.write_text(to_markdown(result))
