"""Command-line interface for tpms2cfd."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .brep_engine import DEFAULT_BREP_SPECS, build_brep_case
from .qa_audit import audit_fluid_mesh, gmsh_mesh_fluid
from .sheet_engine import DEFAULT_SHEET_SPECS, build_sheet_case


def cmd_build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if args.track in ("all", "brep"):
        for spec in DEFAULT_BREP_SPECS:
            if args.case and spec.case_id not in args.case:
                continue
            records.append(build_brep_case(spec, out_dir, force_rebuild=args.force))
    if args.track in ("all", "sheet"):
        for spec in DEFAULT_SHEET_SPECS:
            if args.case and spec.case_id not in args.case:
                continue
            records.append(build_sheet_case(spec, out_dir))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(records)} geometries -> {manifest_path}")


def cmd_audit(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.manifest).resolve().parent
    audited = []
    for c in manifest["cases"]:
        cid = c["case_id"]
        if c["track"] == "TrackA_BREP":
            f_path = Path(c["fluid_step"])
            vinfo = audit_fluid_mesh(
                cid,
                f_path,
                out_dir / "preview_stl" / f"{cid}_FLUID.stl",
                voxel_pitch=0.50 if c["kind"] == "monolith" else 0.25,
            )
            gstats = {}
            if args.gmsh and c["kind"] == "unit_cell":
                gstats = gmsh_mesh_fluid(cid, f_path, out_dir / "gmsh" / f"{cid}.msh", 0.35, 0.75)
        else:
            f_path = Path(c["fluid_stl"])
            vinfo = audit_fluid_mesh(cid, f_path, voxel_pitch=0.25)
            gstats = {}
            if args.gmsh:
                gstats = gmsh_mesh_fluid(cid, f_path, out_dir / "gmsh" / f"{cid}.msh", 0.45, 1.10)
        audited.append({"case_id": cid, "voxel_qa": vinfo, "gmsh_qa": gstats})
    audit_path = out_dir / "audit_report.json"
    audit_path.write_text(json.dumps({"rows": audited}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audited {len(audited)} geometries -> {audit_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tpms2cfd", description="Dual-track B-Rep & Sheet-TPMS CAD-to-CFD CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Synthesize Track A B-Rep STEP and Track B Sheet-TPMS STL/MSH geometries")
    p_build.add_argument("--track", choices=["all", "brep", "sheet"], default="all")
    p_build.add_argument("--out-dir", default="./tpms2cfd_build")
    p_build.add_argument("--case", nargs="*", default=None)
    p_build.add_argument("--force", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_audit = sub.add_parser("audit", help="Run topological/voxel/Gmsh QA on a build manifest")
    p_audit.add_argument("--manifest", required=True)
    p_audit.add_argument("--gmsh", action="store_true")
    p_audit.set_defaults(func=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
