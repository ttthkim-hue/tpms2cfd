"""Track B: Watertight implicit Sheet-TPMS builder and single-volume Fluent ASCII surface mesh exporter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes


SHEET_TOPOLOGIES = (
    "gyroid_sheet",
    "diamond_sheet",
    "schwarz_p_sheet",
    "iwp_sheet",
    "graded_gyroid_sheet",
)


@dataclass(frozen=True)
class SheetCaseSpec:
    """Configuration for an implicit periodic or axially graded Sheet-TPMS geometry."""

    case_id: str
    topology: str
    kind: str  # "unit_cell" or "monolith"
    width_mm: float
    core_length_mm: float
    pitch_mm: float
    level_t: float
    voxel_dx_mm: float = 0.40
    grade_delta_t: float = 0.0
    inlet_buffer_mm: float = 15.0
    outlet_buffer_mm: float = 20.0
    outer_wall_mm: float = 0.0


DEFAULT_SHEET_SPECS: tuple[SheetCaseSpec, ...] = (
    SheetCaseSpec("UC_GYROID_SHEET_RVE40", "gyroid_sheet", "unit_cell", 10.0, 40.0, 10.0, 0.50, voxel_dx_mm=0.40),
    SheetCaseSpec("UC_DIAMOND_SHEET_RVE40", "diamond_sheet", "unit_cell", 10.0, 40.0, 10.0, 0.40, voxel_dx_mm=0.40),
    SheetCaseSpec("UC_SCHWARZ_P_SHEET_RVE40", "schwarz_p_sheet", "unit_cell", 10.0, 40.0, 10.0, 0.45, voxel_dx_mm=0.40),
    SheetCaseSpec("UC_IWP_SHEET_RVE40", "iwp_sheet", "unit_cell", 10.0, 40.0, 10.0, 0.70, voxel_dx_mm=0.40),
    SheetCaseSpec("UC_GRADED_GYROID_RVE40", "graded_gyroid_sheet", "unit_cell", 10.0, 40.0, 10.0, 0.35, voxel_dx_mm=0.40, grade_delta_t=0.25),
)


def evaluate_tpms_field(topology: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, pitch_a: float) -> np.ndarray:
    """Evaluate zero-level periodic TPMS implicit field F(x, y, z)."""
    k = 2.0 * math.pi / pitch_a
    kx, ky, kz = k * x, k * y, k * z
    if topology in ("gyroid_sheet", "graded_gyroid_sheet"):
        return np.sin(kx) * np.cos(ky) + np.sin(ky) * np.cos(kz) + np.sin(kz) * np.cos(kx)
    if topology == "diamond_sheet":
        sx, cx = np.sin(kx), np.cos(kx)
        sy, cy = np.sin(ky), np.cos(ky)
        sz, cz = np.sin(kz), np.cos(kz)
        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
    if topology == "schwarz_p_sheet":
        return np.cos(kx) + np.cos(ky) + np.cos(kz)
    if topology == "iwp_sheet":
        cx, cy, cz = np.cos(kx), np.cos(ky), np.cos(kz)
        c2x, c2y, c2z = np.cos(2.0 * kx), np.cos(2.0 * ky), np.cos(2.0 * kz)
        return 2.0 * (cx * cy + cy * cz + cz * cx) - (c2x + c2y + c2z)
    raise ValueError(f"Unsupported Sheet-TPMS topology: {topology}")


def taubin_smooth(verts: np.ndarray, faces: np.ndarray, iterations: int = 8, lam: float = 0.45, mu: float = -0.48) -> np.ndarray:
    """Apply volume-preserving Taubin shrinkage/inflation smoothing to a closed mesh."""
    v = verts.copy()
    n_verts = len(v)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sym = np.vstack([edges, edges[:, ::-1]])
    order = np.lexsort((edges_sym[:, 1], edges_sym[:, 0]))
    edges_sorted = edges_sym[order]
    keep = np.ones(len(edges_sorted), dtype=bool)
    keep[1:] = np.any(edges_sorted[1:] != edges_sorted[:-1], axis=1)
    edges_u = edges_sorted[keep]
    src, dst = edges_u[:, 0], edges_u[:, 1]
    deg = np.maximum(np.bincount(src, minlength=n_verts).astype(np.float64), 1.0)

    def step(arr: np.ndarray, factor: float) -> np.ndarray:
        nbr_sum = np.zeros_like(arr)
        np.add.at(nbr_sum, src, arr[dst])
        return arr + factor * (nbr_sum / deg[:, None] - arr)

    for _ in range(iterations):
        v = step(v, lam)
        v = step(v, mu)
    return v


def measure_edt_thickness_mm(solid_mask: np.ndarray, dx: float) -> dict[str, float]:
    """Estimate minimum, 5th-percentile, and median ridge wall thickness via 3D EDT."""
    dist = ndimage.distance_transform_edt(solid_mask) * dx
    if all(s > 6 for s in solid_mask.shape):
        sl = (slice(2, -2), slice(2, -2), slice(2, -2))
        d_sub, m_sub = dist[sl], solid_mask[sl]
    else:
        d_sub, m_sub = dist, solid_mask
    local_max = ndimage.maximum_filter(d_sub, size=3)
    ridge = d_sub[m_sub & (d_sub >= local_max - 1e-6) & (d_sub > 0.5 * dx)]
    if len(ridge) == 0:
        return {
            "min_ridge_thickness_mm": 0.0,
            "p05_ridge_thickness_mm": 0.0,
            "median_ridge_thickness_mm": 0.0,
            "voxel_dx_mm": dx,
            "quantization_step_mm": round(float(2.0 * dx), 3),
        }
    return {
        "min_ridge_thickness_mm": round(float(2.0 * np.min(ridge)), 3),
        "p05_ridge_thickness_mm": round(float(2.0 * np.percentile(ridge, 5)), 3),
        "median_ridge_thickness_mm": round(float(2.0 * np.median(ridge)), 3),
        "voxel_dx_mm": dx,
        "quantization_step_mm": round(float(2.0 * dx), 3),
    }


def _classify_fluid_faces(
    mesh: trimesh.Trimesh,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    monolith_outer_wall: bool = False,
    tol: float = 1e-4,
) -> list[tuple[str, int, str, np.ndarray]]:
    tri_v = mesh.vertices[mesh.faces]

    def on_plane(axis: int, val: float) -> np.ndarray:
        return np.all(np.abs(tri_v[:, :, axis] - val) < tol, axis=1)

    m_in = on_plane(2, box_min[2])
    m_out = on_plane(2, box_max[2])
    m_x0 = on_plane(0, box_min[0]) & (~m_in) & (~m_out)
    m_x1 = on_plane(0, box_max[0]) & (~m_in) & (~m_out)
    m_y0 = on_plane(1, box_min[1]) & (~m_in) & (~m_out) & (~m_x0) & (~m_x1)
    m_y1 = on_plane(1, box_max[1]) & (~m_in) & (~m_out) & (~m_x0) & (~m_x1)
    m_wall = ~(m_in | m_out | m_x0 | m_x1 | m_y0 | m_y1)

    if monolith_outer_wall:
        return [
            ("inlet", 20, "mass-flow-inlet", m_in),
            ("outlet", 5, "pressure-outlet", m_out),
            ("outer_wall", 3, "wall", m_x0 | m_x1 | m_y0 | m_y1),
            ("internal_wall", 3, "wall", m_wall),
        ]
    return [
        ("inlet", 20, "mass-flow-inlet", m_in),
        ("outlet", 5, "pressure-outlet", m_out),
        ("symmetry_xmin", 7, "symmetry", m_x0),
        ("symmetry_xmax", 7, "symmetry", m_x1),
        ("symmetry_ymin", 7, "symmetry", m_y0),
        ("symmetry_ymax", 7, "symmetry", m_y1),
        ("internal_wall", 3, "wall", m_wall),
    ]


def write_multizone_ascii_stl(
    mesh: trimesh.Trimesh,
    out_path: Path,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    monolith_outer_wall: bool = False,
) -> dict[str, int]:
    """Export an unmerged multi-solid ASCII STL with named CFD boundary patches."""
    zones = _classify_fluid_faces(mesh, box_min, box_max, monolith_outer_wall=monolith_outer_wall)
    tri_v = mesh.vertices[mesh.faces]
    norms = mesh.face_normals
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with out_path.open("w", encoding="ascii") as fp:
        for zname, _, _, mask in zones:
            idx = np.where(mask)[0]
            counts[zname] = int(len(idx))
            if len(idx) == 0:
                continue
            fp.write(f"solid {zname}\n")
            for fi in idx:
                nx, ny, nz = norms[fi]
                p0, p1, p2 = tri_v[fi]
                fp.write(f"  facet normal {nx:.7e} {ny:.7e} {nz:.7e}\n")
                fp.write("    outer loop\n")
                fp.write(f"      vertex {p0[0]:.8e} {p0[1]:.8e} {p0[2]:.8e}\n")
                fp.write(f"      vertex {p1[0]:.8e} {p1[1]:.8e} {p1[2]:.8e}\n")
                fp.write(f"      vertex {p2[0]:.8e} {p2[1]:.8e} {p2[2]:.8e}\n")
                fp.write("    endloop\n")
                fp.write("  endfacet\n")
            fp.write(f"endsolid {zname}\n")
    return counts


def export_fluent_surface_msh(
    mesh_or_path: trimesh.Trimesh | Path,
    out_path: Path,
    box_min: tuple[float, float, float] | None = None,
    box_max: tuple[float, float, float] | None = None,
    monolith_outer_wall: bool = False,
) -> Path:
    """Convert a fluid surface mesh to a single closed vertex-welded Fluent ASCII surface .msh."""
    m = trimesh.load(str(mesh_or_path), force="mesh") if isinstance(mesh_or_path, Path) else mesh_or_path.copy()
    if not m.is_watertight:
        m.merge_vertices(digits_vertex=8)
        m.update_faces(m.nondegenerate_faces())
    if not m.is_watertight:
        raise RuntimeError(f"Merged surface mesh is not watertight: {out_path}")

    v, f = m.vertices, m.faces
    bmin = tuple(float(x) for x in v.min(axis=0)) if box_min is None else box_min
    bmax = tuple(float(x) for x in v.max(axis=0)) if box_max is None else box_max
    zones = _classify_fluid_faces(m, bmin, bmax, monolith_outer_wall=monolith_outer_wall)

    lines = [
        '(0 "Watertight single-object surface mesh generated by tpms2cfd")',
        "(2 3)",
        f"(10 (0 1 {len(v):x} 0 3))",
        f"(10 (1 1 {len(v):x} 1 3)(",
    ]
    lines.extend(f"{x:.8e} {y:.8e} {z:.8e}" for x, y, z in v)
    lines.append("))")
    lines.append(f"(13 (0 1 {len(f):x} 0))")
    cur = 1
    zmeta = []
    for zid, (zn, bc_id, bc_str, mask) in enumerate(zones, start=2):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise RuntimeError(f"Empty boundary zone '{zn}' in {out_path}")
        end = cur + len(idx) - 1
        lines.append(f"(13 ({zid} {cur:x} {end:x} {bc_id} 3)(")
        for fi in idx:
            a, b, c = f[fi] + 1
            lines.append(f"{a:x} {b:x} {c:x} 0 0")
        lines.append("))")
        zmeta.append((zid, bc_str, zn))
        cur = end + 1
    for zid, bc_str, zn in zmeta:
        lines.append(f"(45 ({zid} {bc_str} {zn})())")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out_path


def _voxelize_mesh_mask(mesh: trimesh.Trimesh, bmin: tuple[float, float, float], bmax: tuple[float, float, float], pitch: float) -> np.ndarray:
    xs = np.arange(bmin[0] + 0.5 * pitch, bmax[0], pitch) + 0.00073
    ys = np.arange(bmin[1] + 0.5 * pitch, bmax[1], pitch) + 0.00117
    zs = np.arange(bmin[2] + 0.5 * pitch, bmax[2], pitch)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    origins = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, bmin[2] - 1.0)])
    dirs = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    locs, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    filled = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
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
                continue
            ix, iy = divmod(int(r_i), ny)
            for k in range(0, len(zlist), 2):
                filled[ix, iy] |= (zs >= zlist[k]) & (zs <= zlist[k + 1])
    return filled


def build_sheet_case(spec: SheetCaseSpec, out_dir: Path, edt_dx_mm: float = 0.15) -> dict:
    """Build solid STL, named fluid STL, and vertex-welded Fluent surface .msh for a Sheet-TPMS case."""
    t0 = time.perf_counter()
    w, l_core, dx = spec.width_mm, spec.core_length_mm, spec.voxel_dx_mm
    ix_lo, ix_hi = spec.outer_wall_mm, w - spec.outer_wall_mm
    margin = 2.5 * dx
    xs = np.arange(ix_lo - margin, ix_hi + margin + 0.5 * dx, dx)
    ys = np.arange(ix_lo - margin, ix_hi + margin + 0.5 * dx, dx)
    zs = np.arange(-margin, l_core + margin + 0.5 * dx, dx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    g = evaluate_tpms_field(spec.topology, X, Y, Z, spec.pitch_mm)
    if spec.grade_delta_t != 0.0:
        t_field = spec.level_t + spec.grade_delta_t * np.clip(Z / l_core, 0.0, 1.0)
    else:
        t_field = spec.level_t
    solid_field = t_field - np.abs(g)
    padded = np.pad(solid_field.astype(np.float64), 1, mode="constant", constant_values=-2.0)
    verts, faces, _, _ = marching_cubes(padded, level=0.0, spacing=(dx, dx, dx), allow_degenerate=False)
    verts[:, 0] += xs[0] - dx
    verts[:, 1] += ys[0] - dx
    verts[:, 2] += zs[0] - dx

    m_pre = trimesh.Trimesh(vertices=verts.copy(), faces=faces, process=True)
    m_pre.fix_normals()
    vol_pre = abs(float(m_pre.volume))

    verts = taubin_smooth(verts, faces, iterations=8)
    m_oversized = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m_oversized.fix_normals()
    if not m_oversized.is_watertight:
        raise RuntimeError(f"Oversized sheet mesh not watertight for {spec.case_id}")
    vol_post = abs(float(m_oversized.volume))
    taubin_vol_delta_pct = round(100.0 * (vol_post - vol_pre) / max(vol_pre, 1e-12), 4)

    inner_box = trimesh.creation.box(bounds=[[ix_lo, ix_lo, 0.0], [ix_hi, ix_hi, l_core]])
    inner_solid = trimesh.boolean.intersection([m_oversized, inner_box], engine="manifold")

    if spec.outer_wall_mm > 0.0:
        outer_full = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [w, w, l_core]])
        outer_cut = trimesh.creation.box(bounds=[[ix_lo, ix_lo, -1.0], [ix_hi, ix_hi, l_core + 1.0]])
        wall_shell = trimesh.boolean.difference([outer_full, outer_cut], engine="manifold")
        solid_mesh = trimesh.boolean.union([wall_shell, inner_solid], engine="manifold")
    else:
        solid_mesh = inner_solid

    if not solid_mesh.is_watertight:
        raise RuntimeError(f"Clipped solid STL not watertight for {spec.case_id}")

    # Measure 3D EDT wall thickness on the ACTUAL exported post-Taubin solid mesh at fine pitch
    solid_vox = _voxelize_mesh_mask(solid_mesh, (ix_lo, ix_lo, 0.0), (ix_hi, ix_hi, l_core), edt_dx_mm)
    thick_info = measure_edt_thickness_mm(solid_vox, edt_dx_mm)

    z_start, z_end = -spec.inlet_buffer_mm, l_core + spec.outlet_buffer_mm
    fluid_box = trimesh.creation.box(bounds=[[ix_lo, ix_lo, z_start], [ix_hi, ix_hi, z_end]])
    fluid_mesh = trimesh.boolean.difference([fluid_box, inner_solid], engine="manifold")
    if not fluid_mesh.is_watertight:
        raise RuntimeError(f"Clipped fluid STL not watertight for {spec.case_id}")

    s_path = out_dir / "sheet" / "solid" / f"{spec.case_id}_SOLID.stl"
    f_stl_path = out_dir / "sheet" / "fluid" / f"{spec.case_id}_FLUID_NAMED.stl"
    f_msh_path = out_dir / "sheet" / "fluid" / f"{spec.case_id}_FLUID_SURFACE.msh"
    s_path.parent.mkdir(parents=True, exist_ok=True)
    solid_mesh.export(str(s_path))

    monolith_mode = spec.outer_wall_mm > 0.0
    zone_counts = write_multizone_ascii_stl(
        fluid_mesh,
        f_stl_path,
        box_min=(ix_lo, ix_lo, z_start),
        box_max=(ix_hi, ix_hi, z_end),
        monolith_outer_wall=monolith_mode,
    )
    export_fluent_surface_msh(
        fluid_mesh,
        f_msh_path,
        box_min=(ix_lo, ix_lo, z_start),
        box_max=(ix_hi, ix_hi, z_end),
        monolith_outer_wall=monolith_mode,
    )

    env_vol_mm3 = w * w * l_core
    solid_vol_mm3 = abs(float(solid_mesh.volume))
    porosity_pct = 100.0 * (1.0 - solid_vol_mm3 / env_vol_mm3)

    zones = _classify_fluid_faces(
        fluid_mesh, (ix_lo, ix_lo, z_start), (ix_hi, ix_hi, z_end), monolith_outer_wall=monolith_mode
    )
    wall_mask = next(mask for zname, _, _, mask in zones if zname == "internal_wall")
    core_wetted_cm2 = float(fluid_mesh.area_faces[wall_mask].sum()) / 100.0

    return {
        "case_id": spec.case_id,
        "track": "TrackB_Sheet_TPMS",
        "topology": spec.topology,
        "kind": spec.kind,
        "pitch_a_mm": spec.pitch_mm,
        "level_t": spec.level_t,
        "grade_delta_t": spec.grade_delta_t,
        "voxel_dx_mm": dx,
        "cross_section_mm": [w, w],
        "core_length_mm": l_core,
        "inlet_buffer_mm": spec.inlet_buffer_mm,
        "outlet_buffer_mm": spec.outlet_buffer_mm,
        "outer_wall_mm": spec.outer_wall_mm,
        "min_thickness_mm": thick_info["median_ridge_thickness_mm"],
        "porosity_pct": round(porosity_pct, 3),
        "solid_volume_cm3": round(solid_vol_mm3 / 1000.0, 4),
        "fluid_volume_cm3": round(abs(float(fluid_mesh.volume)) / 1000.0, 4),
        "core_wetted_area_cm2": round(core_wetted_cm2, 3),
        "specific_area_m_inv": round((core_wetted_cm2 * 1e-4) / (env_vol_mm3 * 1e-9), 1),
        "solid_triangles": int(len(solid_mesh.faces)),
        "fluid_triangles": int(len(fluid_mesh.faces)),
        "solid_watertight": bool(solid_mesh.is_watertight),
        "fluid_watertight": bool(fluid_mesh.is_watertight),
        "fluid_zone_triangles": zone_counts,
        "taubin_volume_delta_pct": taubin_vol_delta_pct,
        "thickness_edt": thick_info,
        "solid_stl": str(s_path),
        "fluid_stl": str(f_stl_path),
        "fluid_surface_msh": str(f_msh_path),
        "solid_size_mb": round(s_path.stat().st_size / (1024 * 1024), 4),
        "fluid_stl_size_mb": round(f_stl_path.stat().st_size / (1024 * 1024), 4),
        "fluid_msh_size_mb": round(f_msh_path.stat().st_size / (1024 * 1024), 4),
        "build_elapsed_s": round(time.perf_counter() - t0, 3),
    }
