"""Unified in-memory model: a mesh, a B-rep, or both.

Two sources of truth exist in this domain and they are not interchangeable:

* **B-rep** (STEP/BREP via OCCT): exact analytic surfaces. Radii, axes and planes are *read*, not fitted.
* **Mesh** (STL/OBJ/PLY/3MF/OFF/GLB via trimesh): a tessellated approximation. Everything geometric
  about it beyond vertex positions is *estimated*.

`Model` carries whichever was loaded and can tessellate a B-rep into a mesh on demand (never the
reverse — mesh→B-rep is reconstruction, which is a later phase and always approximate).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import trimesh

Kind = Literal["mesh", "brep"]

MESH_EXTENSIONS = {".stl", ".obj", ".ply", ".3mf", ".off", ".glb", ".gltf"}
BREP_EXTENSIONS = {".step", ".stp", ".brep", ".brp"}

# Tessellation quality used when a B-rep must be sampled as a mesh (mm, radians).
DEFAULT_LINEAR_DEFLECTION = 0.01
DEFAULT_ANGULAR_DEFLECTION = 0.1


@dataclass
class Model:
    """A loaded 3D model with optional exact (B-rep) and approximate (mesh) representations."""

    name: str
    kind: Kind
    source_path: Optional[str] = None
    units: str = "mm"
    shape: Any = None  # build123d.Shape when kind == "brep"
    _mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, name: str = "mesh", source_path: str | None = None) -> "Model":
        m = cls(name=name, kind="mesh", source_path=source_path, _mesh=mesh)
        m._annotate_mesh()
        return m

    @classmethod
    def from_shape(cls, shape: Any, name: str = "shape", source_path: str | None = None) -> "Model":
        return cls(name=name, kind="brep", source_path=source_path, shape=shape)

    @classmethod
    def load(cls, path: str | os.PathLike, units: str = "mm") -> "Model":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        ext = p.suffix.lower()
        if ext in BREP_EXTENSIONS:
            from build123d import import_brep, import_step

            shape = import_step(str(p)) if ext in {".step", ".stp"} else import_brep(str(p))
            m = cls.from_shape(shape, name=p.stem, source_path=str(p))
            m.units = "mm"  # OCCT's STEP reader normalises to mm
            return m
        if ext in MESH_EXTENSIONS:
            mesh = trimesh.load(str(p), force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"{p} did not load as a single mesh")
            mesh, cleaned = clean_mesh(mesh)
            m = cls.from_mesh(mesh, name=p.stem, source_path=str(p))
            if cleaned:
                m.notes.append(cleaned)
            m.units = units
            m.notes.append(f"Mesh formats carry no units; values assumed to be {units}.")
            return m
        raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------ properties
    @property
    def is_exact(self) -> bool:
        """True when analytic geometry is available (B-rep). Mesh-derived numbers are estimates."""
        return self.kind == "brep"

    @property
    def mesh(self) -> trimesh.Trimesh:
        """A triangle mesh view. For a B-rep this is a fine tessellation (cached)."""
        if self._mesh is None:
            if self.shape is None:
                raise ValueError("Model has neither mesh nor shape")
            self._mesh = tessellate(self.shape)
            self._annotate_mesh()
        return self._mesh

    def _annotate_mesh(self) -> None:
        m = self._mesh
        if m is None:
            return
        if not m.is_watertight:
            self.notes.append(
                "Mesh is not watertight: volume, centre of mass and inside/outside tests are approximate."
            )
        if m.is_watertight and m.volume < 0:
            m.invert()
            self.notes.append("Mesh had inward-facing normals; flipped.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Model(name={self.name!r}, kind={self.kind!r}, source={self.source_path!r})"


def clean_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, Optional[str]]:
    """Drop degenerate and duplicate triangles and merge vertices; returns (mesh, note or None).

    Duplicated faces double plane areas and break watertightness; zero-area 'pole' triangles from
    tessellators do the same. Cleaning is lossless for a valid mesh.
    """
    n0 = len(mesh.faces)
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    removed = n0 - len(mesh.faces)
    note = f"Removed {removed} degenerate/duplicate triangle(s) before measuring." if removed else None
    return mesh, note


def tessellate(
    shape: Any,
    linear_deflection: float = DEFAULT_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_ANGULAR_DEFLECTION,
) -> trimesh.Trimesh:
    """Tessellate a build123d shape into a trimesh at the requested quality."""
    verts, tris = shape.tessellate(linear_deflection, angular_deflection)
    v = np.array([[p.X, p.Y, p.Z] for p in verts], dtype=float)
    f = np.array(tris, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    mesh, _ = clean_mesh(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh


def load(path: str | os.PathLike, units: str = "mm") -> Model:
    """Convenience wrapper around :meth:`Model.load`."""
    return Model.load(path, units=units)
