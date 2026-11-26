"""Topological, DfAM printability, voxel connectivity, and Gmsh 3D quality audit utilities."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import gmsh
import numpy as np
import trimesh
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI
from scipy import ndimage


def export_step_preview_stl(step_path: Path, stl_path: Path, lin_deflection: float = 0.10) -> Path:
    """Tessellate a STEP file to STL purely for visualization and voxel ray-casting."""
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed STEP read: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, lin_deflection, False, 0.30, True).Perform()
    StlAPI.Write_s(shape, str(stl_path), False)
    return stl_path


def voxel_connectivity_and_los(fluid_mesh: trimesh.Trimesh, pitch: float = 0.25) -> dict:
    """Compute inlet-to-outlet fluid connectivity, trapped-void fraction, and axial line-of-sight."""
    bmin, bmax = fluid_mesh.bounds
    xs = np.arange(bmin[0] + 0.5 * pitch, bmax[0], pitch) + 0.00137
    ys = np.arange(bmin[1] + 0.5 * pitch, bmax[1], pitch) + 0.00193
    zs = np.arange(bmin[2] + 0.5 * pitch, bmax[2], pitch)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    origins = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, bmin[2] - 1.0)])
    dirs = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    locs, idx_ray, _ = fluid_mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    filled = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)

    odd_parity_rays = 0
    if len(idx_ray) > 0:
        order = np.argsort(idx_ray)
        idx_s, z_s = idx_ray[order], locs[order, 2]
        splits = np.flatnonzero(np.diff(idx_s)) + 1
        groups_idx = idx_s[np.r_[0, splits]]
        groups_z = np.split(z_s, splits)
        ny = len(ys)
        for r_i, zlist in zip(groups_idx, groups_z):
            zlist = np.sort(zlist)
            if len(zlist) > 1:
                zlist = zlist[np.r_[True, np.diff(zlist) > 1e-4]]
            if len(zlist) % 2 != 0:
                odd_parity_rays += 1
                continue
            ix, iy = divmod(int(r_i), ny)
            for k in range(0, len(zlist), 2):
                filled[ix, iy] |= (zs >= zlist[k]) & (zs <= zlist[k + 1])

    labeled, n_cc = ndimage.label(filled)
    if n_cc == 0:
        return {
            "fluid_connected_fraction": 0.0,
            "trapped_void_fraction": 1.0,
            "axial_los_pct": 0.0,
            "voxel_components": 0,
            "odd_parity_rays": int(odd_parity_rays),
            "total_rays": int(len(origins)),
        }

    inlet_labels = set(np.unique(labeled[:, :, :3])) - {0}
    outlet_labels = set(np.unique(labeled[:, :, -3:])) - {0}
    through_labels = inlet_labels & outlet_labels
    total_fluid = int(filled.sum())
    through_fluid = int(np.isin(labeled, list(through_labels)).sum()) if through_labels else 0
    trapped_fluid = total_fluid - through_fluid
    col_open = np.all(filled, axis=2)
    return {
        "fluid_connected_fraction": round(through_fluid / max(total_fluid, 1), 6),
        "trapped_void_fraction": round(trapped_fluid / max(total_fluid, 1), 6),
        "axial_los_pct": round(100.0 * float(col_open.sum()) / float(col_open.size), 3),
        "voxel_components": int(n_cc),
        "odd_parity_rays": int(odd_parity_rays),
        "total_rays": int(len(origins)),
    }


def audit_mesh_topology(mesh_path: Path, merge_digits: int = 8) -> dict:
    """Measure real manifold boundary, non-manifold edge, and degenerate-face defects on an STL mesh."""
    m = trimesh.load(str(mesh_path), force="mesh")
    m.merge_vertices(digits_vertex=merge_digits)
    edges_sorted = np.sort(m.edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))
    degenerate_faces = int(np.sum(m.area_faces < 1e-12))
    return {
        "watertight": bool(m.is_watertight),
        "is_volume": bool(m.is_volume),
        "winding_consistent": bool(m.is_winding_consistent),
        "euler_number": int(m.euler_number),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "degenerate_faces": degenerate_faces,
        "min_face_area_mm2": float(m.area_faces.min()),
    }


def _collect_tetra_quality(msh_path: Path, occ_volumes: int, elapsed_s: float) -> dict:
    _, tet_tags, _ = gmsh.model.mesh.getElements(3)
    all_tets = np.concatenate(tet_tags) if tet_tags else np.array([], dtype=np.int64)
    if len(all_tets) == 0:
        raise RuntimeError(f"Gmsh produced 0 tetrahedra for {msh_path.name}")
    sicn = gmsh.model.mesh.getElementQualities(all_tets, "minSICN")
    jac = gmsh.model.mesh.getElementQualities(all_tets, "minDetJac")
    return {
        "gmsh_occ_volumes": occ_volumes,
        "gmsh_tetra_count": int(len(all_tets)),
        "gmsh_min_sicn": round(float(np.min(sicn)), 6),
        "gmsh_p01_sicn": round(float(np.percentile(sicn, 1)), 5),
        "gmsh_median_sicn": round(float(np.median(sicn)), 5),
        "gmsh_pct_sicn_lt_0p1": round(100.0 * float(np.mean(sicn < 0.10)), 4),
        "gmsh_nonpositive_sicn": int(np.sum(sicn <= 0.0)),
        "gmsh_negative_jacobian": int(np.sum(jac <= 0.0)),
        "gmsh_mesh_mb": round(msh_path.stat().st_size / (1024 * 1024), 3),
        "gmsh_elapsed_s": round(elapsed_s, 2),
        "msh_path": str(msh_path),
    }


def gmsh_mesh_fluid(
    case_id: str,
    geom_path: Path,
    msh_path: Path,
    lc_min: float,
    lc_max: float,
    monolith_outer_wall: bool = False,
    occ_make_solids: bool = False,
) -> dict:
    """Generate a 3D tetrahedral volume mesh from either an analytic STEP or multi-zone STL."""
    t0 = time.perf_counter()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    try:
        gmsh.model.add(case_id)
        if geom_path.suffix.lower() == ".step":
            if occ_make_solids:
                gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)
            gmsh.model.occ.importShapes(str(geom_path))
            gmsh.model.occ.synchronize()
            vols = gmsh.model.getEntities(3)
            if len(vols) != 1:
                raise RuntimeError(f"Expected 1 OCC volume for {case_id}, got {len(vols)}")

            face_bbs = [(tag, gmsh.model.getBoundingBox(2, tag)) for _, tag in gmsh.model.getEntities(2)]
            zmin, zmax = min(bb[2] for _, bb in face_bbs), max(bb[5] for _, bb in face_bbs)
            tol = 1e-3
            inlet_bbs = [bb for _, bb in face_bbs if abs(bb[2] - zmin) < tol and abs(bb[5] - zmin) < tol]
            xmin, ymin = min(bb[0] for bb in inlet_bbs), min(bb[1] for bb in inlet_bbs)
            xmax, ymax = max(bb[3] for bb in inlet_bbs), max(bb[4] for bb in inlet_bbs)

            inlet_s, outlet_s, side_s, wall_s = [], [], [], []
            for tag, (fx0, fy0, fz0, fx1, fy1, fz1) in face_bbs:
                if abs(fz0 - zmin) < tol and abs(fz1 - zmin) < tol:
                    inlet_s.append(tag)
                elif abs(fz0 - zmax) < tol and abs(fz1 - zmax) < tol:
                    outlet_s.append(tag)
                elif (
                    abs(fx0 - xmin) < tol and abs(fx1 - xmin) < tol
                    or abs(fx0 - xmax) < tol and abs(fx1 - xmax) < tol
                    or abs(fy0 - ymin) < tol and abs(fy1 - ymin) < tol
                    or abs(fy0 - ymax) < tol and abs(fy1 - ymax) < tol
                ):
                    side_s.append(tag)
                else:
                    wall_s.append(tag)

            gmsh.model.addPhysicalGroup(3, [vols[0][1]], name="fluid")
            if inlet_s:
                gmsh.model.addPhysicalGroup(2, inlet_s, name="inlet")
            if outlet_s:
                gmsh.model.addPhysicalGroup(2, outlet_s, name="outlet")
            if side_s:
                gmsh.model.addPhysicalGroup(2, side_s, name="outer_wall" if monolith_outer_wall else "sides")
            if wall_s:
                gmsh.model.addPhysicalGroup(2, wall_s, name="internal_wall")

            gmsh.option.setNumber("Mesh.MeshSizeMin", lc_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", lc_max)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            occ_vols = len(vols)
        else:
            import manifold3d

            tmp_dir = msh_path.parent / "_tmp_zones" / case_id
            tmp_dir.mkdir(parents=True, exist_ok=True)
            m = trimesh.load(str(geom_path), force="mesh")
            m.merge_vertices(digits_vertex=8)
            if not m.is_watertight:
                raise RuntimeError(f"Merged STL mesh for {case_id} is not watertight")

            def _load_stl_into_gmsh(mesh_obj: trimesh.Trimesh, tag_name: str, eps_tag: str) -> None:
                tmp_stl = tmp_dir / f"{case_id}_{eps_tag}_merged.stl"
                mesh_obj.export(str(tmp_stl))
                gmsh.model.add(tag_name)
                gmsh.merge(str(tmp_stl))
                gmsh.model.mesh.removeDuplicateNodes()
                all_surfs = [tag for _, tag in gmsh.model.getEntities(2)]
                sl = gmsh.model.geo.addSurfaceLoop(all_surfs)
                vol = gmsh.model.geo.addVolume([sl])
                gmsh.model.geo.synchronize()

                gmsh.model.addPhysicalGroup(3, [vol], name="fluid")
                if all_surfs:
                    gmsh.model.addPhysicalGroup(2, all_surfs, name="boundary")

                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeMin", lc_min)
                gmsh.option.setNumber("Mesh.MeshSizeMax", lc_max)
                gmsh.option.setNumber("Mesh.Algorithm3D", 1)
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

            occ_vols = 1
            _load_stl_into_gmsh(m, case_id, "orig")
            try:
                gmsh.model.mesh.generate(3)
            except Exception:
                gmsh.clear()
                man = manifold3d.Manifold(
                    manifold3d.Mesh(
                        vert_properties=np.asarray(m.vertices, dtype=np.float32),
                        tri_verts=np.asarray(m.faces, dtype=np.uint32),
                    )
                )
                mo = man.simplify(1e-4).to_mesh()
                m_simp = trimesh.Trimesh(vertices=mo.vert_properties[:, :3], faces=mo.tri_verts, process=False)
                if not m_simp.is_watertight:
                    raise RuntimeError(f"Simplified STL mesh for {case_id} is not watertight")
                _load_stl_into_gmsh(m_simp, f"{case_id}_simp", "simp1e4")
                gmsh.model.mesh.generate(3)

        if geom_path.suffix.lower() == ".step":
            gmsh.model.mesh.generate(3)
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()
        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.ScalingFactor", 1.0)
        gmsh.write(str(msh_path))
        return _collect_tetra_quality(msh_path, occ_vols, time.perf_counter() - t0)
    finally:
        gmsh.finalize()


def gmsh_mesh_fluid_subproc(
    case_id: str,
    geom_path: Path,
    msh_path: Path,
    lc_min: float,
    lc_max: float,
    monolith_outer_wall: bool = False,
    occ_make_solids: bool = False,
    timeout_s: int = 90,
) -> dict:
    """Run gmsh_mesh_fluid in an isolated subprocess bounded by a wall-clock timeout."""
    src_root = str(Path(__file__).resolve().parents[1])
    code = (
        "import json, sys; from pathlib import Path; "
        f"sys.path.insert(0, {src_root!r}); "
        "from tpms2cfd.qa_audit import gmsh_mesh_fluid; "
        f"res = gmsh_mesh_fluid({case_id!r}, Path({str(geom_path)!r}), Path({str(msh_path)!r}), "
        f"{lc_min!r}, {lc_max!r}, monolith_outer_wall={monolith_outer_wall!r}, occ_make_solids={occ_make_solids!r}); "
        "print(json.dumps(res))"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Gmsh exceeded {timeout_s}s timeout") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()[-1]
        raise RuntimeError(err)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def audit_fluid_mesh(
    case_id: str,
    fluid_geom_path: Path,
    preview_stl_path: Path | None = None,
    voxel_pitch: float = 0.25,
) -> dict:
    """Compute ray-cast voxel connectivity and line-of-sight metrics for any fluid domain."""
    if fluid_geom_path.suffix.lower() == ".step":
        if preview_stl_path is None:
            raise ValueError("preview_stl_path required when auditing STEP geometry")
        stl = export_step_preview_stl(fluid_geom_path, preview_stl_path)
    else:
        stl = fluid_geom_path
    fm = trimesh.load(str(stl), force="mesh")
    fm.merge_vertices(digits_vertex=8)
    res = voxel_connectivity_and_los(fm, pitch=voxel_pitch)
    res["watertight"] = bool(fm.is_watertight)
    return res
