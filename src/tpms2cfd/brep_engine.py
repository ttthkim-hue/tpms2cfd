"""Track A: Analytic OpenCASCADE B-Rep engine for cellular POCS/SRS solids and fluid domains."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt


BREP_TOPOLOGIES = (
    "square_channel",
    "gyroid_srs",
    "diamond_pocs",
    "helical_diamond",
    "kelvin_pocs",
    "gyroid_srs_monolith",
)


@dataclass(frozen=True)
class BrepCaseSpec:
    """Configuration for a clean-room analytic B-Rep geometry case."""

    case_id: str
    topology: str
    kind: str  # "unit_cell" or "monolith"
    width_mm: float
    core_length_mm: float
    pitch_mm: float
    diameter_mm: float
    inlet_buffer_mm: float = 15.0
    outlet_buffer_mm: float = 20.0
    outer_wall_mm: float = 0.0
    phase: tuple[float, float, float] = (0.125, 0.125, 0.125)
    swirl_amp_mm: float = 0.0
    helix_period_mm: float = 40.0


DEFAULT_BREP_SPECS: tuple[BrepCaseSpec, ...] = (
    BrepCaseSpec("UC_SQ_CHANNEL_RVE40", "square_channel", "unit_cell", 5.0, 40.0, 5.0, 1.0, phase=(0.0, 0.0, 0.0)),
    BrepCaseSpec("UC_GYROID_SRS_RVE40", "gyroid_srs", "unit_cell", 10.0, 40.0, 10.0, 2.0, phase=(0.125, 0.125, 0.125)),
    BrepCaseSpec("UC_DIAMOND_POCS_RVE40", "diamond_pocs", "unit_cell", 10.0, 40.0, 10.0, 1.8, phase=(0.25, 0.25, 0.25)),
    BrepCaseSpec("UC_HELIX_POCS_RVE40", "helical_diamond", "unit_cell", 10.0, 40.0, 10.0, 1.7, phase=(0.25, 0.25, 0.25), swirl_amp_mm=2.0, helix_period_mm=40.0),
    BrepCaseSpec("UC_KELVIN_POCS_RVE40", "kelvin_pocs", "unit_cell", 10.0, 40.0, 10.0, 1.6, phase=(0.25, 0.25, 0.25)),
    BrepCaseSpec("MONO_GYROID_SRS_30x30x60", "gyroid_srs_monolith", "monolith", 30.0, 60.0, 10.0, 2.0, outer_wall_mm=1.0, phase=(0.125, 0.125, 0.125)),
)


def make_box(xmin: float, ymin: float, zmin: float, xmax: float, ymax: float, zmax: float):
    """Construct an axis-aligned OpenCASCADE box primitive."""
    return BRepPrimAPI_MakeBox(gp_Pnt(xmin, ymin, zmin), gp_Pnt(xmax, ymax, zmax)).Shape()


def make_cylinder_between(p1: tuple[float, float, float], p2: tuple[float, float, float], radius: float):
    """Construct a circular cylinder aligned along vector p2 - p1."""
    vx, vy, vz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    length = math.sqrt(vx * vx + vy * vy + vz * vz)
    if length < 1e-9:
        return None
    ax = gp_Ax2(gp_Pnt(p1[0], p1[1], p1[2]), gp_Dir(vx / length, vy / length, vz / length))
    return BRepPrimAPI_MakeCylinder(ax, radius, length).Shape()


def make_sphere(p: tuple[float, float, float], radius: float):
    """Construct a sphere primitive centered at point p."""
    return BRepPrimAPI_MakeSphere(gp_Pnt(p[0], p[1], p[2]), radius).Shape()


def bool_op(op_cls, s1, s2, fuzzy: float = 1e-4):
    """Execute a parallel OBB-accelerated OpenCASCADE Boolean operation."""
    op = op_cls(s1, s2)
    op.SetRunParallel(True)
    op.SetUseOBB(True)
    op.SetFuzzyValue(fuzzy)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"Boolean operation {op_cls.__name__} failed")
    return op.Shape()


def binary_tree_fuse(shapes: list, fuzzy: float = 1e-4):
    """Fuse shapes recursively in a balanced binary tree to control Boolean depth."""
    if not shapes:
        raise ValueError("Empty shape list for fusion")
    cur = list(shapes)
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                nxt.append(bool_op(BRepAlgoAPI_Fuse, cur[i], cur[i + 1], fuzzy=fuzzy))
            else:
                nxt.append(cur[i])
        cur = nxt
    return cur[0]


def _shape_z_mid(shape) -> float:
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    _, _, zmin, _, _, zmax = b.Get()
    return 0.5 * (zmin + zmax)


def heal_and_inspect(shape) -> tuple[object, dict]:
    """Run ShapeFix + SameParameter and audit topological integrity."""
    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    fixed = fixer.Shape()
    BRepLib.SameParameter_s(fixed)

    analyzer = BRepCheck_Analyzer(fixed)
    brep_valid = bool(analyzer.IsValid())

    def count_type(ttype) -> int:
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(fixed, ttype, m)
        return m.Extent()

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(fixed, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    free_edges = 0
    nonmanifold_edges = 0
    degenerated_edges = 0
    for i in range(1, edge_face_map.Extent() + 1):
        edge = TopoDS.Edge_s(edge_face_map.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            degenerated_edges += 1
            continue
        nf = edge_face_map.FindFromIndex(i).Extent()
        if nf == 1:
            free_edges += 1
        elif nf > 2:
            nonmanifold_edges += 1

    bbox = Bnd_Box()
    BRepBndLib.AddOptimal_s(fixed, bbox, False, False)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    vp = GProp_GProps()
    BRepGProp.VolumeProperties_s(fixed, vp)
    sp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(fixed, sp)

    info = {
        "brep_valid": brep_valid,
        "solid_count": count_type(TopAbs_SOLID),
        "shell_count": count_type(TopAbs_SHELL),
        "face_count": count_type(TopAbs_FACE),
        "free_edges": free_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "degenerated_edges": degenerated_edges,
        "bbox_mm": [round(xmin, 5), round(ymin, 5), round(zmin, 5), round(xmax, 5), round(ymax, 5), round(zmax, 5)],
        "volume_mm3": round(float(vp.Mass()), 5),
        "area_mm2": round(float(sp.Mass()), 5),
    }
    return fixed, info


def export_step_mm(shape, path: Path) -> None:
    """Write a B-Rep shape to STEP AP214/AP203 using millimeter units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP: {path}")


def readback_inspect_step(path: Path) -> dict:
    """Read back a STEP file into OpenCASCADE and record pure parse time vs heal+audit time."""
    t0 = time.perf_counter()
    reader = STEPControl_Reader()
    st = reader.ReadFile(str(path))
    if st != IFSelect_RetDone:
        raise RuntimeError(f"Failed STEP readback: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    t_parse = time.perf_counter() - t0
    t_h0 = time.perf_counter()
    _, info = heal_and_inspect(shape)
    t_heal = time.perf_counter() - t_h0
    info["size_mb"] = round(path.stat().st_size / (1024 * 1024), 4)
    info["ocp_parse_transfer_s"] = round(t_parse, 3)
    info["ocp_heal_audit_s"] = round(t_heal, 3)
    info["ocp_readback_s"] = round(t_parse + t_heal, 3)
    return info


def srs_unit_cell_graph() -> tuple[list[tuple[float, float, float]], list[tuple[tuple[float, float, float], tuple[float, float, float]]]]:
    """Generate the I4_132 single-Gyroid skeletal graph in [0, 1)^3."""
    verts = []
    for i in (1, 3, 5, 7):
        for j in (1, 3, 5, 7):
            for k in (1, 3, 5, 7):
                x, y, z = i / 8.0, j / 8.0, k / 8.0
                g = (
                    math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)
                    + math.sin(2 * math.pi * y) * math.cos(2 * math.pi * z)
                    + math.sin(2 * math.pi * z) * math.cos(2 * math.pi * x)
                )
                if g > 1.49:
                    verts.append((x, y, z))
    target_d2 = (math.sqrt(2.0) / 4.0) ** 2
    return verts, _periodic_edges_by_dist2(verts, target_d2)


def diamond_unit_cell_graph(phase: tuple[float, float, float] = (0.125, 0.125, 0.125)):
    """Generate the Fd-3m diamond skeletal graph in [0, 1)^3 shifted by phase."""
    fcc = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    verts_a = [tuple((c[idx] + phase[idx]) for idx in range(3)) for c in fcc]
    verts_b = [tuple((c[idx] + 0.25 + phase[idx]) for idx in range(3)) for c in fcc]
    verts = verts_a + verts_b
    offsets = [
        (0.25, 0.25, 0.25),
        (0.25, -0.25, -0.25),
        (-0.25, 0.25, -0.25),
        (-0.25, -0.25, 0.25),
    ]
    edges = [(va, (va[0] + off[0], va[1] + off[1], va[2] + off[2])) for va in verts_a for off in offsets]
    return verts, edges


def kelvin_unit_cell_graph(phase: tuple[float, float, float] = (0.125, 0.125, 0.125)):
    """Generate the BCC Kelvin tetrakaidecahedron skeletal graph in [0, 1)^3 shifted by phase."""
    raw_verts = set()
    vals = (0.5, 0.25, 0.0)
    from itertools import permutations

    for p in permutations(vals):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    vx = round((0.5 + sx * p[0]) % 1.0, 6)
                    vy = round((0.5 + sy * p[1]) % 1.0, 6)
                    vz = round((0.5 + sz * p[2]) % 1.0, 6)
                    raw_verts.add((vx, vy, vz))
    verts = sorted(raw_verts)
    target_d2 = 0.25**2 + 0.25**2  # sqrt(2)/4 edge length
    edges = _periodic_edges_by_dist2(verts, target_d2)
    if phase != (0.0, 0.0, 0.0):
        verts = [(v[0] + phase[0], v[1] + phase[1], v[2] + phase[2]) for v in verts]
        edges = [
            (
                (a[0] + phase[0], a[1] + phase[1], a[2] + phase[2]),
                (b[0] + phase[0], b[1] + phase[1], b[2] + phase[2]),
            )
            for a, b in edges
        ]
    return verts, edges


def _periodic_edges_by_dist2(verts: list[tuple[float, float, float]], target_d2: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    edges = []
    seen = set()
    for v in verts:
        for u in verts:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        u_shift = (u[0] + dx, u[1] + dy, u[2] + dz)
                        d2 = (u_shift[0] - v[0]) ** 2 + (u_shift[1] - v[1]) ** 2 + (u_shift[2] - v[2]) ** 2
                        if abs(d2 - target_d2) < 1e-6:
                            key = tuple(sorted([v, u_shift]))
                            if key not in seen:
                                seen.add(key)
                                edges.append((v, u_shift))
    return edges


def build_lattice_solids(
    base_edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    pitch: float,
    diameter: float,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    swirl_amp: float = 0.0,
    helix_period: float = 40.0,
) -> list:
    """Instantiate analytic cylindrical struts and spherical joint caps covering a bounding box."""
    radius = diameter / 2.0
    margin = radius + 0.05
    nx_min, nx_max = math.floor((box_min[0] - margin) / pitch) - 1, math.ceil((box_max[0] + margin) / pitch) + 1
    ny_min, ny_max = math.floor((box_min[1] - margin) / pitch) - 1, math.ceil((box_max[1] + margin) / pitch) + 1
    nz_min, nz_max = math.floor((box_min[2] - margin) / pitch) - 1, math.ceil((box_max[2] + margin) / pitch) + 1
    wx, wy = box_max[0] - box_min[0], box_max[1] - box_min[1]

    def shift_pt(pt: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = pt
        if swirl_amp != 0.0 and box_min[0] <= x <= box_max[0] and box_min[1] <= y <= box_max[1]:
            env = math.sin(math.pi * (x - box_min[0]) / wx) * math.sin(math.pi * (y - box_min[1]) / wy)
            ang = 2.0 * math.pi * z / helix_period
            x += swirl_amp * env * math.sin(ang)
            y += swirl_amp * env * math.cos(ang)
        return (x, y, z)

    def seg_hits_box(p1, p2) -> bool:
        for i in range(9):
            t = i * 0.125
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            pz = p1[2] + t * (p2[2] - p1[2])
            if (
                box_min[0] - margin <= px <= box_max[0] + margin
                and box_min[1] - margin <= py <= box_max[1] + margin
                and box_min[2] - margin <= pz <= box_max[2] + margin
            ):
                return True
        return False

    def pt_in_aabb(p) -> bool:
        return all(box_min[idx] - margin <= p[idx] <= box_max[idx] + margin for idx in range(3))

    parts = []
    seen_nodes = set()
    seen_edges = set()
    for ix in range(nx_min, nx_max + 1):
        for iy in range(ny_min, ny_max + 1):
            for iz in range(nz_min, nz_max + 1):
                for p1_u, p2_u in base_edges:
                    p1 = shift_pt(((p1_u[0] + ix) * pitch, (p1_u[1] + iy) * pitch, (p1_u[2] + iz) * pitch))
                    p2 = shift_pt(((p2_u[0] + ix) * pitch, (p2_u[1] + iy) * pitch, (p2_u[2] + iz) * pitch))
                    if not seg_hits_box(p1, p2):
                        continue
                    k1, k2 = tuple(round(v, 5) for v in p1), tuple(round(v, 5) for v in p2)
                    ekey = tuple(sorted([k1, k2]))
                    if ekey not in seen_edges:
                        seen_edges.add(ekey)
                        cyl = make_cylinder_between(p1, p2, radius)
                        if cyl is not None:
                            parts.append(cyl)
                    for k, pt in ((k1, p1), (k2, p2)):
                        if k not in seen_nodes and pt_in_aabb(pt):
                            seen_nodes.add(k)
                            parts.append(make_sphere(pt, radius))
    return parts


def _fuse_chunked(parts: list, chunk_size: int = 60, fuzzy: float = 1e-4):
    if len(parts) <= 250:
        return binary_tree_fuse(parts, fuzzy=fuzzy)
    parts_s = sorted(parts, key=_shape_z_mid)
    chunks = [binary_tree_fuse(parts_s[i : i + chunk_size], fuzzy=fuzzy) for i in range(0, len(parts_s), chunk_size)]
    return binary_tree_fuse(chunks, fuzzy=fuzzy)


def build_brep_case(spec: BrepCaseSpec, out_dir: Path, force_rebuild: bool = False) -> dict:
    """Build solid and buffered fluid STEP files for an analytic B-Rep topology."""
    t0 = time.perf_counter()
    solid_dir = out_dir / "brep" / "solid"
    fluid_dir = out_dir / "brep" / "fluid"
    s_path = solid_dir / f"{spec.case_id}_SOLID.step"
    f_path = fluid_dir / f"{spec.case_id}_FLUID.step"

    if not force_rebuild and s_path.is_file() and f_path.is_file():
        sr = readback_inspect_step(s_path)
        fr = readback_inspect_step(f_path)
        if _is_clean_brep(sr) and _is_clean_brep(fr):
            return _format_brep_record(spec, s_path, f_path, sr, fr, time.perf_counter() - t0)

    w = spec.width_mm
    l_core = spec.core_length_mm

    if spec.topology == "square_channel":
        ch = 4.0
        wall_half = (w - ch) / 2.0
        outer = make_box(0.0, 0.0, 0.0, w, w, l_core)
        chan = make_box(wall_half, wall_half, -1.0, w - wall_half, w - wall_half, l_core + 1.0)
        solid_raw = bool_op(BRepAlgoAPI_Cut, outer, chan)
        fluid_box = make_box(0.0, 0.0, -spec.inlet_buffer_mm, w, w, l_core + spec.outlet_buffer_mm)
        fluid_raw = bool_op(BRepAlgoAPI_Cut, fluid_box, solid_raw)

    elif spec.topology in ("gyroid_srs", "diamond_pocs", "helical_diamond", "kelvin_pocs"):
        edges = _get_graph_edges(spec.topology, spec.phase)
        parts = build_lattice_solids(
            edges,
            spec.pitch_mm,
            spec.diameter_mm,
            (0.0, 0.0, 0.0),
            (w, w, l_core),
            swirl_amp=spec.swirl_amp_mm,
            helix_period=spec.helix_period_mm,
        )
        fused = _fuse_chunked(parts)
        core_box = make_box(0.0, 0.0, 0.0, w, w, l_core)
        solid_raw = bool_op(BRepAlgoAPI_Common, fused, core_box, fuzzy=1e-4)
        fluid_box = make_box(0.0, 0.0, -spec.inlet_buffer_mm, w, w, l_core + spec.outlet_buffer_mm)
        fluid_raw = bool_op(BRepAlgoAPI_Cut, fluid_box, solid_raw, fuzzy=1e-4)

    elif spec.topology == "gyroid_srs_monolith":
        ow = spec.outer_wall_mm
        edges = _get_graph_edges("gyroid_srs", spec.phase)
        parts = build_lattice_solids(
            edges,
            spec.pitch_mm,
            spec.diameter_mm,
            (ow - 0.3, ow - 0.3, 0.0),
            (w - ow + 0.3, w - ow + 0.3, l_core),
        )
        inner_lattice = _fuse_chunked(parts)
        inner_box = make_box(ow, ow, 0.0, w - ow, w - ow, l_core)
        inner_clipped = bool_op(BRepAlgoAPI_Common, inner_lattice, inner_box, fuzzy=1e-4)
        outer_wall = bool_op(
            BRepAlgoAPI_Cut,
            make_box(0.0, 0.0, 0.0, w, w, l_core),
            make_box(ow, ow, -1.0, w - ow, w - ow, l_core + 1.0),
            fuzzy=1e-4,
        )
        solid_raw = bool_op(BRepAlgoAPI_Fuse, outer_wall, inner_clipped, fuzzy=1e-4)
        fluid_box = make_box(ow, ow, -spec.inlet_buffer_mm, w - ow, w - ow, l_core + spec.outlet_buffer_mm)
        fluid_raw = bool_op(BRepAlgoAPI_Cut, fluid_box, inner_clipped, fuzzy=1e-4)
    else:
        raise ValueError(f"Unsupported Track A topology: {spec.topology}")

    solid, _ = heal_and_inspect(solid_raw)
    fluid, _ = heal_and_inspect(fluid_raw)
    export_step_mm(solid, s_path)
    export_step_mm(fluid, f_path)
    sr = readback_inspect_step(s_path)
    fr = readback_inspect_step(f_path)
    if not _is_clean_brep(sr):
        raise RuntimeError(f"Solid B-Rep audit failed for {spec.case_id}: {sr}")
    if not _is_clean_brep(fr):
        raise RuntimeError(f"Fluid B-Rep audit failed for {spec.case_id}: {fr}")
    return _format_brep_record(spec, s_path, f_path, sr, fr, time.perf_counter() - t0)


def _get_graph_edges(topology: str, phase: tuple[float, float, float]):
    if topology == "gyroid_srs":
        verts, edges = srs_unit_cell_graph()
        if phase != (0.0, 0.0, 0.0):
            edges = [
                (
                    (a[0] + phase[0], a[1] + phase[1], a[2] + phase[2]),
                    (b[0] + phase[0], b[1] + phase[1], b[2] + phase[2]),
                )
                for a, b in edges
            ]
        return edges
    if topology in ("diamond_pocs", "helical_diamond"):
        _, edges = diamond_unit_cell_graph(phase=phase)
        return edges
    if topology == "kelvin_pocs":
        _, edges = kelvin_unit_cell_graph(phase=phase)
        return edges
    raise ValueError(f"Unknown graph topology: {topology}")


def _is_clean_brep(qa: dict) -> bool:
    return (
        qa.get("brep_valid") is True
        and qa.get("solid_count") == 1
        and qa.get("free_edges") == 0
        and qa.get("nonmanifold_edges") == 0
        and qa.get("degenerated_edges") == 0
    )


def _format_brep_record(spec: BrepCaseSpec, s_path: Path, f_path: Path, sr: dict, fr: dict, elapsed_s: float) -> dict:
    w, l_core = spec.width_mm, spec.core_length_mm
    env_vol = w * w * l_core
    porosity_pct = 100.0 * (1.0 - sr["volume_mm3"] / env_vol)
    if spec.topology == "square_channel":
        wetted_cm2 = (4.0 * 4.0 * l_core) / 100.0
    elif spec.kind == "monolith":
        wetted_cm2 = (sr["area_mm2"] - 4.0 * w * l_core) / 100.0
    else:
        ext_area = 2.0 * w * w + 4.0 * w * (l_core + spec.inlet_buffer_mm + spec.outlet_buffer_mm)
        wetted_cm2 = (sr["area_mm2"] + fr["area_mm2"] - ext_area) / 200.0

    return {
        "case_id": spec.case_id,
        "track": "TrackA_BREP",
        "topology": spec.topology,
        "kind": spec.kind,
        "pitch_a_mm": spec.pitch_mm,
        "cross_section_mm": [w, w],
        "core_length_mm": l_core,
        "inlet_buffer_mm": spec.inlet_buffer_mm,
        "outlet_buffer_mm": spec.outlet_buffer_mm,
        "outer_wall_mm": spec.outer_wall_mm,
        "min_thickness_mm": spec.diameter_mm,
        "porosity_pct": round(porosity_pct, 3),
        "core_wetted_area_cm2": round(wetted_cm2, 3),
        "specific_area_m_inv": round((wetted_cm2 * 1e-4) / (env_vol * 1e-9), 1),
        "solid_step": str(s_path),
        "fluid_step": str(f_path),
        "solid_qa": sr,
        "fluid_qa": fr,
        "build_elapsed_s": round(elapsed_s, 3),
    }
