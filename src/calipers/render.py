"""Headless rendering: shaded orthographic views, hidden-line views for B-reps, and 2D section plots.

No GPU or display is needed. Shaded views use a small numpy painter's-algorithm rasteriser on top of
matplotlib; hidden-line views come from OCCT's HLR through build123d's SVG exporter (B-reps only).
Renders are the *secondary* channel for an LLM — numbers come from :mod:`calipers.measure` — but a
picture catches gross mistakes (wrong orientation, missing feature) that a table of numbers hides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from calipers.measure import parse_plane, section, unit  # noqa: E402
from calipers.model import Model  # noqa: E402

# camera direction (object → camera) and the up vector, per named view
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], str]] = {
    "iso": ((1.0, -1.0, 0.8), (0.0, 0.0, 1.0), "isometric from +X −Y +Z"),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), "front: X → right, Z → up (viewed from −Y)"),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "back: −X → right, Z → up (viewed from +Y)"),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "top: X → right, Y → up (viewed from +Z)"),
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), "bottom: −X → right, Y → up (viewed from −Z)"),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "right: Y → right, Z → up (viewed from +X)"),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "left: −Y → right, Z → up (viewed from −X)"),
}
DEFAULT_VIEWS = ("iso", "front", "top", "right")
MAX_RENDER_FACES = 120_000


def _camera(view: Sequence[float], up: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = unit(view)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:  # up parallel to view: pick another up
        x = np.cross([0.0, 1.0, 0.0], z)
    x = unit(x)
    y = np.cross(z, x)
    return x, y, z


def _decimated(mesh):
    if len(mesh.faces) <= MAX_RENDER_FACES:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=MAX_RENDER_FACES)
    except Exception:
        return mesh


MAX_EDGE_FRACTION = 0.03  # of the bbox diagonal: longer triangles are split before depth sorting


def _painter_ready(mesh):
    """Split long triangles so a painter's sort by centroid depth cannot draw a far object over a
    near wall (a 60 mm wall face is two triangles whose centroids sit *behind* a part inside the box)."""
    import trimesh

    max_edge = MAX_EDGE_FRACTION * float(np.linalg.norm(mesh.extents))
    edges = mesh.vertices[mesh.edges_unique]
    if max_edge <= 0 or np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max() <= max_edge:
        return mesh
    try:
        v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=6)
    except Exception:
        return mesh
    if len(f) > 4 * MAX_RENDER_FACES:
        return mesh
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def render_view(model: Model, view: str, out_path: str | os.PathLike, size_px: int = 900, title: Optional[str] = None) -> str:
    """Shaded orthographic view of the model's mesh, saved as PNG. Returns the path."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {sorted(VIEWS)}")
    cam, up, label = VIEWS[view]
    mesh = _painter_ready(_decimated(model.mesh))
    x, y, z = _camera(cam, up)
    tri = np.asarray(mesh.triangles)  # (n, 3, 3)
    n = np.asarray(mesh.face_normals)
    facing = n @ z
    keep = facing > -0.05 if mesh.is_winding_consistent else np.ones(len(tri), dtype=bool)
    tri, n, facing = tri[keep], n[keep], facing[keep]
    P = tri.reshape(-1, 3)
    px = (P @ x).reshape(-1, 3)
    py = (P @ y).reshape(-1, 3)
    depth = (P @ z).reshape(-1, 3).mean(axis=1)
    order = np.argsort(depth)  # far first
    light = unit(z + 0.45 * x + 0.65 * y)
    shade = 0.28 + 0.72 * np.clip(n @ light, 0.0, 1.0)
    polys = np.stack([px, py], axis=-1)[order]
    colors = np.column_stack([shade, shade, shade * 0.98, np.ones_like(shade)])[order]

    w = float(px.max() - px.min()) if len(px) else 1.0
    h = float(py.max() - py.min()) if len(py) else 1.0
    fig_w = size_px / 100.0
    fig_h = max(2.5, fig_w * (h / max(w, 1e-9)) * 0.9 + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    pc = PolyCollection(polys, facecolors=colors, edgecolors="none", antialiased=False)
    ax.add_collection(pc)
    pad = 0.04 * max(w, h)
    ax.set_xlim(px.min() - pad, px.max() + pad)
    ax.set_ylim(py.min() - pad, py.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{w:.2f} {model.units} across")
    ax.set_ylabel(f"{h:.2f} {model.units} tall")
    ax.set_title(title or f"{model.name} — {label}", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = str(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_views(model: Model, out_dir: str | os.PathLike, views: Sequence[str] = DEFAULT_VIEWS, size_px: int = 900) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for v in views:
        paths.append(render_view(model, v, out_dir / f"{model.name}_{v}.png", size_px=size_px))
    if model.is_exact:
        for v in views:
            p = render_hidden_line(model, v, out_dir / f"{model.name}_{v}_hlr.svg")
            if p:
                paths.append(p)
    return paths


def render_hidden_line(model: Model, view: str, out_path: str | os.PathLike) -> Optional[str]:
    """Hidden-line SVG of a B-rep (OCCT HLR). Returns None for mesh models or on failure."""
    if not model.is_exact:
        return None
    try:
        from build123d import ExportSVG, LineType

        cam, up, _ = VIEWS[view]
        exp = ExportSVG(scale=2)
        exp.add_layer("visible", line_color=(0, 0, 0), line_weight=0.5)
        exp.add_layer("hidden", line_color=(140, 140, 140), line_weight=0.25, line_type=LineType.ISO_DASH)
        vis, hid = model.shape.project_to_viewport(tuple(float(c) for c in cam), viewport_up=tuple(float(c) for c in up))
        exp.add_shape(vis, layer="visible")
        exp.add_shape(hid, layer="hidden")
        exp.write(str(out_path))
        png = _svg_to_png(str(out_path))
        return png or str(out_path)
    except Exception:
        return None


def _svg_to_png(svg_path: str, width: int = 900) -> Optional[str]:
    try:
        import cairosvg

        png = str(Path(svg_path).with_suffix(".png"))
        cairosvg.svg2png(url=svg_path, write_to=png, output_width=width, background_color="white")
        return png
    except Exception:
        return None


def render_section(model: Model, plane: str, out_path: str | os.PathLike, size_px: int = 800) -> tuple[str, dict]:
    """Plot the cross-section loops with circle/rectangle annotations. Returns (path, section dict)."""
    origin, normal = parse_plane(plane)
    sec = section(model, origin, normal)
    mesh = model.mesh
    fig, ax = plt.subplots(figsize=(size_px / 100.0, size_px / 100.0), dpi=100)
    cut = mesh.section(plane_origin=origin, plane_normal=normal)
    if cut is not None:
        path2d, _ = cut.to_2D()
        for ent in path2d.entities:
            pts = path2d.vertices[ent.points]
            ax.plot(pts[:, 0], pts[:, 1], color="black", lw=1.0)
        for L in sec["loops"]:
            cx, cy = L["centroid"]
            tag = L["id"]
            if L["circle"]["is_circle"]:
                tag += f" Ø{L['circle']['diameter']:.2f}"
            elif L["rectangle"]["is_rectangle"]:
                tag += f" {L['rectangle']['size'][0]:.2f}×{L['rectangle']['size'][1]:.2f}"
            ax.annotate(tag, (cx, cy), fontsize=8, ha="center", color="tab:blue")
    axis = "xyz"[int(np.argmax(np.abs(normal)))]
    inplane = [a for a in "xyz" if a != axis]
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{model.name} — section {plane} (in-plane axes: {inplane[0]}, {inplane[1]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return str(out_path), sec
