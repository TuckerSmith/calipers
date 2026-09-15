"""Export a model to exchange formats. B-rep formats need a B-rep; mesh formats tessellate on demand."""

from __future__ import annotations

import os
from pathlib import Path

from calipers.model import BREP_EXTENSIONS, MESH_EXTENSIONS, Model


def _write_step(shape, path: str) -> None:
    """STEP writer that accepts any OCCT shape, including compounds that came from ``import_step``."""
    try:
        from build123d import export_step

        export_step(shape, path)
        return
    except Exception:
        pass  # fall through to the raw OCCT writer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)
    writer.Transfer(shape.wrapped, STEPControl_AsIs)
    if writer.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")


def export(model: Model, path: str | os.PathLike, linear_deflection: float = 0.01, angular_deflection: float = 0.1) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in BREP_EXTENSIONS:
        if not model.is_exact:
            raise ValueError("cannot write a B-rep format from a mesh-only model (that would be reconstruction)")
        if ext in {".step", ".stp"}:
            _write_step(model.shape, str(p))
        else:
            from build123d import export_brep

            export_brep(model.shape, str(p))
        return str(p)
    if ext in MESH_EXTENSIONS:
        if model.is_exact:
            from calipers.model import tessellate

            mesh = tessellate(model.shape, linear_deflection, angular_deflection)
        else:
            mesh = model.mesh
        mesh.export(str(p))
        return str(p)
    raise ValueError(f"unsupported export format: {ext}")
