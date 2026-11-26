#!/usr/bin/env python3
"""Step 2: Generate all 11 positive configurations + representation ablations and run real topological/DfAM/Gmsh QA."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

from tpms2cfd.baseline_control import build_faceted_step_control, build_representation_ablation
from tpms2cfd.brep_engine import DEFAULT_BREP_SPECS, build_brep_case
from tpms2cfd.qa_audit import audit_fluid_mesh, audit_mesh_topology, gmsh_mesh_fluid
from tpms2cfd.sheet_engine import DEFAULT_SHEET_SPECS, build_sheet_case

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _rel(path_str: str | None, base: Path) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return p.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local tpms2cfd geometry, ablation, and Gmsh benchmark")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "geometries")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    geom_out = args.out_dir.resolve()
    res_dir = args.results_dir.resolve()
    geom_out.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    out_json = res_dir / "local_geometry_gmsh_benchmark.json"
    out_csv = res_dir / "local_geometry_gmsh_benchmark.csv"
    abl_json = res_dir / "representation_ablation.json"

    env_meta = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gmsh_threads": 1,
        "packages": {p: _pkg_version(p) for p in ("numpy", "scipy", "scikit-image", "trimesh", "manifold3d", "gmsh", "cadquery-ocp")},
    }

    rows: list[dict] = []

    print("=== Building & Auditing Track A (Analytic B-Rep) ===", flush=True)
    srs_fluid_step = None
    for spec in DEFAULT_BREP_SPECS:
        rec = build_brep_case(spec, geom_out)
        cid = spec.case_id
        f_step = Path(rec["fluid_step"])
        if cid == "UC_GYROID_SRS_RVE40":
            srs_fluid_step = f_step
        pitch = 0.40 if spec.kind == "monolith" else 0.25
        vinfo = audit_fluid_mesh(cid, f_step, geom_out / "preview_stl" / f"{cid}_FLUID.stl", voxel_pitch=pitch)
        assert vinfo["odd_parity_rays"] == 0, f"Odd-parity rays in {cid}: {vinfo}"

        if spec.case_id == "UC_SQ_CHANNEL_RVE40":
            lc_min, lc_max = 0.35, 0.70
        elif spec.kind == "monolith":
            lc_min, lc_max = 0.55, 1.25
        else:
            lc_min, lc_max = 0.35, 0.75

        gstats = gmsh_mesh_fluid(
            cid,
            f_step,
            geom_out / "gmsh" / f"{cid}.msh",
            lc_min=lc_min,
            lc_max=lc_max,
            monolith_outer_wall=(spec.kind == "monolith"),
        )
        gstats["msh_path"] = _rel(gstats.get("msh_path"), REPO_ROOT)
        row = {
            "case_id": cid,
            "track": rec["track"],
            "topology": rec["topology"],
            "kind": rec["kind"],
            "cross_section_mm": f"{spec.width_mm}x{spec.width_mm}",
            "core_length_mm": spec.core_length_mm,
            "porosity_pct": rec["porosity_pct"],
            "core_wetted_area_cm2": rec["core_wetted_area_cm2"],
            "specific_area_m_inv": rec["specific_area_m_inv"],
            "nominal_thickness_mm": rec["min_thickness_mm"],
            "min_thickness_mm": 0.5 if spec.topology == "square_channel" else rec["min_thickness_mm"],
            "solid_size_mib": rec["solid_qa"]["size_mb"],
            "solid_faces": rec["solid_qa"]["face_count"],
            "fluid_size_mib": rec["fluid_qa"]["size_mb"],
            "fluid_faces": rec["fluid_qa"]["face_count"],
            "ocp_parse_transfer_s": rec["fluid_qa"]["ocp_parse_transfer_s"],
            "ocp_heal_audit_s": rec["fluid_qa"]["ocp_heal_audit_s"],
            "ocp_readback_s": rec["fluid_qa"]["ocp_readback_s"],
            "build_elapsed_s": rec["build_elapsed_s"],
            "brep_valid": rec["solid_qa"]["brep_valid"] and rec["fluid_qa"]["brep_valid"],
            "free_edges": rec["solid_qa"]["free_edges"] + rec["fluid_qa"]["free_edges"],
            "nonmanifold_edges": rec["solid_qa"]["nonmanifold_edges"] + rec["fluid_qa"]["nonmanifold_edges"],
            "degenerated_edges": rec["solid_qa"]["degenerated_edges"] + rec["fluid_qa"]["degenerated_edges"],
            **vinfo,
            **gstats,
        }
        assert row["free_edges"] == 0 and row["nonmanifold_edges"] == 0 and row["degenerated_edges"] == 0
        assert row["fluid_connected_fraction"] == 1.0 and row["trapped_void_fraction"] == 0.0
        assert row["gmsh_negative_jacobian"] == 0
        rows.append(row)
        print(
            f"[TRACK-A] {cid}: faces={row['fluid_faces']}, size={row['fluid_size_mib']}MiB, "
            f"conn={row['fluid_connected_fraction']}, LOS={row['axial_los_pct']}%, "
            f"tets={row['gmsh_tetra_count']:,}, p01SICN={row['gmsh_p01_sicn']:.4f}, gmsh={row['gmsh_elapsed_s']}s",
            flush=True,
        )

    print("=== Building & Auditing Track B (Watertight Sheet-TPMS) ===", flush=True)
    gyroid_sheet_stl = None
    for spec in DEFAULT_SHEET_SPECS:
        rec = build_sheet_case(spec, geom_out, edt_dx_mm=0.15)
        cid = spec.case_id
        f_stl = Path(rec["fluid_stl"])
        s_stl = Path(rec["solid_stl"])
        if cid == "UC_GYROID_SHEET_RVE40":
            gyroid_sheet_stl = f_stl
        topo_solid = audit_mesh_topology(s_stl)
        topo_fluid = audit_mesh_topology(f_stl)
        vinfo = audit_fluid_mesh(cid, f_stl, voxel_pitch=0.25)
        assert vinfo["odd_parity_rays"] == 0, f"Odd-parity rays in {cid}: {vinfo}"
        gstats = gmsh_mesh_fluid(cid, f_stl, geom_out / "gmsh" / f"{cid}.msh", lc_min=0.45, lc_max=1.10)
        gstats["msh_path"] = _rel(gstats.get("msh_path"), REPO_ROOT)
        row = {
            "case_id": cid,
            "track": rec["track"],
            "topology": rec["topology"],
            "kind": rec["kind"],
            "cross_section_mm": f"{spec.width_mm}x{spec.width_mm}",
            "core_length_mm": spec.core_length_mm,
            "porosity_pct": rec["porosity_pct"],
            "core_wetted_area_cm2": rec["core_wetted_area_cm2"],
            "specific_area_m_inv": rec["specific_area_m_inv"],
            "nominal_thickness_mm": None,
            "min_thickness_mm": rec["thickness_edt"]["min_ridge_thickness_mm"],
            "p05_thickness_mm": rec["thickness_edt"]["p05_ridge_thickness_mm"],
            "median_thickness_mm": rec["thickness_edt"]["median_ridge_thickness_mm"],
            "edt_voxel_dx_mm": rec["thickness_edt"]["voxel_dx_mm"],
            "taubin_volume_delta_pct": rec["taubin_volume_delta_pct"],
            "solid_size_mib": rec["solid_size_mb"],
            "solid_faces": rec["solid_triangles"],
            "fluid_size_mib": rec["fluid_msh_size_mb"],
            "fluid_faces": rec["fluid_triangles"],
            "fluid_stl_size_mib": rec["fluid_stl_size_mb"],
            "ocp_parse_transfer_s": None,
            "ocp_heal_audit_s": None,
            "ocp_readback_s": None,
            "build_elapsed_s": rec["build_elapsed_s"],
            "brep_valid": bool(topo_solid["watertight"] and topo_fluid["watertight"] and topo_solid["is_volume"] and topo_fluid["is_volume"]),
            "free_edges": topo_solid["boundary_edges"] + topo_fluid["boundary_edges"],
            "nonmanifold_edges": topo_solid["nonmanifold_edges"] + topo_fluid["nonmanifold_edges"],
            "degenerated_edges": topo_solid["degenerate_faces"] + topo_fluid["degenerate_faces"],
            "solid_euler_number": topo_solid["euler_number"],
            "fluid_euler_number": topo_fluid["euler_number"],
            **vinfo,
            **gstats,
        }
        assert row["free_edges"] == 0 and row["nonmanifold_edges"] == 0
        assert row["fluid_connected_fraction"] == 1.0 and row["trapped_void_fraction"] == 0.0
        assert row["gmsh_negative_jacobian"] == 0
        rows.append(row)
        print(
            f"[TRACK-B] {cid}: tris={row['fluid_faces']:,}, msh={row['fluid_size_mib']}MiB, "
            f"dV_taubin={row['taubin_volume_delta_pct']}%, thick(min/p05/med)={row['min_thickness_mm']}/{row['p05_thickness_mm']}/{row['median_thickness_mm']}mm, "
            f"tets={row['gmsh_tetra_count']:,}, sicn<0.1={row['gmsh_pct_sicn_lt_0p1']}%, gmsh={row['gmsh_elapsed_s']}s",
            flush=True,
        )

    print("=== Building & Auditing Sheet-Gyroid Faceted STEP Control ===", flush=True)
    ctrl = build_faceted_step_control(geom_out, case_id="CTRL_FACETED_GYROID_RVE40", dx=0.65, t=0.50)
    ctrl_step = Path(ctrl["fluid_step"])
    t_g0 = time.perf_counter()
    try:
        from tpms2cfd.qa_audit import gmsh_mesh_fluid_subproc

        ctrl_gmsh = gmsh_mesh_fluid_subproc(
            "CTRL_FACETED_GYROID_RVE40",
            ctrl_step,
            geom_out / "gmsh" / "CTRL_FACETED_GYROID_RVE40.msh",
            0.45,
            1.10,
            occ_make_solids=True,
            timeout_s=90,
        )
        ctrl_gmsh["msh_path"] = _rel(ctrl_gmsh.get("msh_path"), REPO_ROOT)
        ctrl_gmsh["gmsh_status"] = "OK"
    except Exception as exc:
        ctrl_gmsh = {"gmsh_status": f"FAILED: {exc}", "gmsh_elapsed_s": round(time.perf_counter() - t_g0, 2)}

    ctrl_row = {
        "case_id": ctrl["case_id"],
        "track": ctrl["track"],
        "topology": ctrl["topology"],
        "kind": ctrl["kind"],
        "cross_section_mm": "10.0x10.0",
        "core_length_mm": 40.0,
        "porosity_pct": None,
        "core_wetted_area_cm2": round(ctrl["fluid_qa"]["area_mm2"] / 100.0, 3),
        "specific_area_m_inv": None,
        "nominal_thickness_mm": None,
        "min_thickness_mm": None,
        "solid_size_mib": None,
        "solid_faces": None,
        "fluid_size_mib": ctrl["fluid_qa"]["size_mb"],
        "fluid_faces": ctrl["fluid_qa"]["face_count"],
        "solid_count": ctrl["fluid_qa"]["solid_count"],
        "ocp_parse_transfer_s": ctrl["fluid_qa"]["ocp_parse_transfer_s"],
        "ocp_heal_audit_s": ctrl["fluid_qa"]["ocp_heal_audit_s"],
        "ocp_readback_s": ctrl["fluid_qa"]["ocp_readback_s"],
        "build_elapsed_s": ctrl["build_elapsed_s"],
        "brep_valid": ctrl["fluid_qa"]["brep_valid"],
        "free_edges": ctrl["fluid_qa"]["free_edges"],
        "nonmanifold_edges": ctrl["fluid_qa"]["nonmanifold_edges"],
        "degenerated_edges": ctrl["fluid_qa"]["degenerated_edges"],
        **ctrl_gmsh,
    }
    rows.append(ctrl_row)
    print(
        f"[CONTROL] {ctrl_row['case_id']}: solids={ctrl_row['solid_count']}, faces={ctrl_row['fluid_faces']:,}, "
        f"size={ctrl_row['fluid_size_mib']}MiB, parse={ctrl_row['ocp_parse_transfer_s']}s, heal={ctrl_row['ocp_heal_audit_s']}s, "
        f"gmsh={ctrl_gmsh.get('gmsh_elapsed_s')}s ({ctrl_gmsh.get('gmsh_status')})",
        flush=True,
    )

    print("=== Running Identical-Geometry Representation Ablation (Pair A + Pair B) ===", flush=True)
    assert srs_fluid_step is not None and gyroid_sheet_stl is not None
    ablation = build_representation_ablation(srs_fluid_step, gyroid_sheet_stl, geom_out)
    for item in ablation.get("pair_a_srs_identical_geometry", []):
        if "step_path" in item:
            item["step_path"] = _rel(item["step_path"], REPO_ROOT)
    abl_json.write_text(json.dumps({"environment": env_meta, **ablation}, indent=2, ensure_ascii=False), encoding="utf-8")

    out_json.write_text(json.dumps({"environment": env_meta, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved local benchmark -> {out_json} & {abl_json}", flush=True)


if __name__ == "__main__":
    main()
