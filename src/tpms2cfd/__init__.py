"""tpms2cfd: Dual-track analytic B-Rep and watertight Sheet-TPMS CAD-to-CFD toolkit."""

from .brep_engine import BREP_TOPOLOGIES, build_brep_case, readback_inspect_step
from .sheet_engine import SHEET_TOPOLOGIES, build_sheet_case, export_fluent_surface_msh
from .qa_audit import audit_fluid_mesh, gmsh_mesh_fluid, voxel_connectivity_and_los

__all__ = [
    "BREP_TOPOLOGIES",
    "SHEET_TOPOLOGIES",
    "build_brep_case",
    "readback_inspect_step",
    "build_sheet_case",
    "export_fluent_surface_msh",
    "audit_fluid_mesh",
    "gmsh_mesh_fluid",
    "voxel_connectivity_and_los",
]
__version__ = "1.0.0"
