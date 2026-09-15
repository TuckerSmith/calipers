"""calipers — kernel-exact interrogation, verification and generation of 3D geometry for LLM-driven CAD.

The guiding principle: an AI never guesses a dimension; it measures. Every number that leaves this
package is tagged with where it came from (exact B-rep evaluation vs. a fit against a mesh) so a
downstream model can reason about how much to trust it.
"""

from calipers.model import Model, load
from calipers.report import GeometryReport, build_report

__all__ = ["Model", "load", "GeometryReport", "build_report", "__version__"]
__version__ = "0.2.0"
