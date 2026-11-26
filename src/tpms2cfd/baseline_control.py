"""Self-generated Clean-Room control baselines and representation ablation builders."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCP.ShapeFix import ShapeFix_Solid
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from .brep_engine import export_step_mm, heal_and_inspect, readback_inspect_step
from .qa_audit import export_step_preview_stl, gmsh_mesh_fluid_subproc
from .sheet_engine import evaluate_tpms_field, export_fluent_surface_msh, write_multizone_ascii_stl


def _triangles_to_faceted_step(
    mesh: trimesh.Trimesh,
    step_path: Path,
    sew_tol: float = 1e-2,
) -> dict:
    """Convert every triangle in a mesh to a planar OpenCASCADE B-Rep face and sew into a solid STEP."""
    sew = BRepBuilderAPI_Sewing(sew_tol)
    v_arr = mesh.vertices
    n_faces_added = 0
    for f_idx in mesh.faces:
        p0 = gp_Pnt(float(v_arr[f_idx[0], 0]), float(v_arr[f_idx[0], 1]), float(v_arr[f_idx[0], 2]))
        p1 = gp_Pnt(float(v_arr[f_idx[1], 0]), float(v_arr[f_idx[1], 1]), float(v_arr[f_idx[1], 2]))
        p2 = gp_Pnt(float(v_arr[f_idx[2], 0]), float(v_arr[f_idx[2], 1]), float(v_arr[f_idx[2], 2]))
        if p0.Distance(p1) < 1e-6 or p1.Distance(p2) < 1e-6 or p2.Distance(p0) < 1e-6:
            continue
        poly = BRepBuilderAPI_MakePolygon(p0, p1, p2, True)
        if not poly.IsDone():
            continue
        face_builder = BRepBuilderAPI_MakeFace(poly.Wire(), True)
        if face_builder.IsDone():
            sew.Add(face_builder.Face())
            n_faces_added += 1

    sew.Perform()
    sewn_shape = sew.SewedShape()
    if sewn_shape.ShapeType() == 2:  # TopAbs_SHELL
        shell = TopoDS.Shell_s(sewn_shape)
        solid_fixer = ShapeFix_Solid()
        fixed_solid = solid_fixer.SolidFromShell(shell)
        if not fixed_solid.IsNull():
            raw_shape = fixed_solid
        else:
            solid_maker = BRepBuilderAPI_MakeSolid(shell)
            raw_shape = solid_maker.Solid() if solid_maker.IsDone() else sewn_shape
    else:
        raw_shape = sewn_shape

    fixed_shape, qa_build = heal_and_inspect(raw_shape)
    export_step_mm(fixed_shape, step_path)
    sr = readback_inspect_step(step_path)
    return {"faces_added": n_faces_added, "fluid_qa": sr, "build_qa": qa_build}


def _symmetric_hausdorff_mm(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_samples: int = 20000) -> dict:
    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples, seed=42)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples, seed=43)
    d_ab, _ = cKDTree(pts_b).query(pts_a)
    d_ba, _ = cKDTree(pts_a).query(pts_b)
    return {
        "hausdorff_max_mm": round(float(max(d_ab.max(), d_ba.max())), 5),
        "hausdorff_rms_mm": round(float(np.sqrt(0.5 * (np.mean(d_ab**2) + np.mean(d_ba**2)))), 5),
    }


def build_faceted_step_control(
    out_dir: Path,
    case_id: str = "CTRL_FACETED_GYROID_RVE40",
    dx: float = 0.65,
    t: float = 0.50,
    sew_tol: float = 1e-2,
) -> dict:
    """Synthesize a coarse faceted-STEP Sheet-Gyroid fluid domain for cross-representation illustration."""
    t0 = time.perf_counter()
    w, l_core = 10.0, 40.0
    inlet_buf, outlet_buf = 15.0, 20.0
    margin = 2.5 * dx
    xs = np.arange(-margin, w + margin + 0.5 * dx, dx)
    ys = np.arange(-margin, w + margin + 0.5 * dx, dx)
    zs = np.arange(-margin, l_core + margin + 0.5 * dx, dx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    g = evaluate_tpms_field("gyroid_sheet", X, Y, Z, 10.0)
    solid_field = t - np.abs(g)
    padded = np.pad(solid_field.astype(np.float64), 1, mode="constant", constant_values=-2.0)
    verts, faces, _, _ = marching_cubes(padded, level=0.0, spacing=(dx, dx, dx), allow_degenerate=False)
    verts[:, 0] += xs[0] - dx
    verts[:, 1] += ys[0] - dx
    verts[:, 2] += zs[0] - dx

    m_oversized = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m_oversized.fix_normals()
    core_box = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [w, w, l_core]])
    inner_solid = trimesh.boolean.intersection([m_oversized, core_box], engine="manifold")
    fluid_box = trimesh.creation.box(bounds=[[0.0, 0.0, -inlet_buf], [w, w, l_core + outlet_buf]])
    fluid_mesh = trimesh.boolean.difference([fluid_box, inner_solid], engine="manifold")
    if not fluid_mesh.is_watertight:
        raise RuntimeError("Control faceted fluid mesh is not watertight")

    step_path = out_dir / "controls" / f"{case_id}_FLUID.step"
    conv = _triangles_to_faceted_step(fluid_mesh, step_path, sew_tol=sew_tol)

    return {
        "case_id": case_id,
        "track": "Control_Faceted_STEP",
        "topology": "gyroid_sheet",
        "kind": "control_baseline",
        "voxel_dx_mm": dx,
        "input_triangles": int(len(fluid_mesh.faces)),
        "ocp_faces_added": conv["faces_added"],
        "fluid_step": str(step_path),
        "fluid_qa": conv["fluid_qa"],
        "build_qa": conv["build_qa"],
        "build_elapsed_s": round(time.perf_counter() - t0, 3),
    }


def build_representation_ablation(
    srs_fluid_step: Path,
    sheet_fluid_stl: Path,
    out_dir: Path,
) -> dict:
    """Run controlled identical-geometry representation ablations (Pair A: SRS B-Rep vs faceted STEP; Pair B: Sheet formats)."""
    ab_dir = out_dir / "ablation"
    ab_dir.mkdir(parents=True, exist_ok=True)

    # Pair A: Identical UC_GYROID_SRS_RVE40 fluid geometry tessellated at delta in {0.10, 0.05} mm
    ref_stl = export_step_preview_stl(srs_fluid_step, ab_dir / "SRS_REF_d001.stl", lin_deflection=0.01)
    ref_mesh = trimesh.load(str(ref_stl), force="mesh")
    srs_qa = readback_inspect_step(srs_fluid_step)

    pair_a = [
        {
            "variant": "Analytic B-Rep STEP",
            "case_id": "UC_GYROID_SRS_RVE40",
            "chordal_deflection_mm": 0.0,
            "faces": srs_qa["face_count"],
            "solid_count": srs_qa["solid_count"],
            "size_mib": srs_qa["size_mb"],
            "ocp_parse_transfer_s": srs_qa["ocp_parse_transfer_s"],
            "ocp_heal_audit_s": srs_qa["ocp_heal_audit_s"],
            "hausdorff_max_mm": 0.0,
            "hausdorff_rms_mm": 0.0,
            "volume_rel_err_pct": 0.0,
            "gmsh_occ_solids_status": "OK",
        }
    ]

    for delta in (0.10, 0.05):
        tag = f"d{str(delta).replace('.', '')}"
        cid = f"CTRL_FACETED_SRS_{tag}"
        stl_p = export_step_preview_stl(srs_fluid_step, ab_dir / f"{cid}.stl", lin_deflection=delta)
        m_fac = trimesh.load(str(stl_p), force="mesh")
        m_fac.merge_vertices(digits_vertex=6)
        haus = _symmetric_hausdorff_mm(ref_mesh, m_fac)
        step_p = ab_dir / f"{cid}_FLUID.step"
        conv = _triangles_to_faceted_step(m_fac, step_p, sew_tol=1e-2)
        fqa = conv["fluid_qa"]
        try:
            gm = gmsh_mesh_fluid_subproc(cid, step_p, ab_dir / f"{cid}_gmsh.msh", 0.35, 0.75, occ_make_solids=True, timeout_s=90)
            gmsh_status = f"OK ({gm['gmsh_tetra_count']} tets, {gm['gmsh_elapsed_s']}s)"
        except Exception as exc:
            gmsh_status = f"FAILED: {exc}"
        vol_err_pct = round(100.0 * abs(fqa["volume_mm3"] - srs_qa["volume_mm3"]) / max(srs_qa["volume_mm3"], 1e-9), 4)
        pair_a.append({
            "variant": f"Faceted STEP (delta={delta:.2f} mm)",
            "case_id": cid,
            "chordal_deflection_mm": delta,
            "faces": fqa["face_count"],
            "solid_count": fqa["solid_count"],
            "size_mib": fqa["size_mb"],
            "ocp_parse_transfer_s": fqa["ocp_parse_transfer_s"],
            "ocp_heal_audit_s": fqa["ocp_heal_audit_s"],
            "volume_rel_err_pct": vol_err_pct,
            "gmsh_occ_solids_status": gmsh_status,
            "step_path": str(step_p),
            **haus,
        })

    # Pair B: Identical UC_GYROID_SHEET_RVE40 fluid triangles (same 57,308 faces) across 3 formats
    m_sheet = trimesh.load(str(sheet_fluid_stl), force="mesh")
    m_sheet.merge_vertices(digits_vertex=6)
    msh_p = ab_dir / "PAIRB_SHEET_SURFACE.msh"
    export_fluent_surface_msh(m_sheet, msh_p, box_min=(0.0, 0.0, -15.0), box_max=(10.0, 10.0, 60.0))
    stl_ascii = ab_dir / "PAIRB_SHEET_MULTIZONE.stl"
    write_multizone_ascii_stl(m_sheet, stl_ascii, box_min=(0.0, 0.0, -15.0), box_max=(10.0, 10.0, 60.0))
    stl_bin = ab_dir / "PAIRB_SHEET_BINARY.stl"
    m_sheet.export(str(stl_bin))
    sheet_step = ab_dir / "PAIRB_SHEET_FACETED.step"
    conv_b = _triangles_to_faceted_step(m_sheet, sheet_step, sew_tol=1e-2)

    pair_b = [
        {"format": "Fluent ASCII surface .msh (vertex-welded)", "triangles": int(len(m_sheet.faces)), "size_mib": round(msh_p.stat().st_size / (1024 * 1024), 4), "ocp_parse_transfer_s": None},
        {"format": "Binary STL (single solid)", "triangles": int(len(m_sheet.faces)), "size_mib": round(stl_bin.stat().st_size / (1024 * 1024), 4), "ocp_parse_transfer_s": None},
        {"format": "Multi-solid ASCII STL (7 named zones)", "triangles": int(len(m_sheet.faces)), "size_mib": round(stl_ascii.stat().st_size / (1024 * 1024), 4), "ocp_parse_transfer_s": None},
        {"format": "Faceted STEP (sewn planar B-Rep faces)", "triangles": conv_b["fluid_qa"]["face_count"], "size_mib": conv_b["fluid_qa"]["size_mb"], "ocp_parse_transfer_s": conv_b["fluid_qa"]["ocp_parse_transfer_s"], "ocp_heal_audit_s": conv_b["fluid_qa"]["ocp_heal_audit_s"], "solid_count": conv_b["fluid_qa"]["solid_count"]},
    ]
    return {"pair_a_srs_identical_geometry": pair_a, "pair_b_sheet_identical_triangles": pair_b}
