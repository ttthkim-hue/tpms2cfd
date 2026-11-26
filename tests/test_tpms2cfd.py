"""Unit and integration tests for the tpms2cfd package."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import trimesh

from tpms2cfd.baseline_control import build_faceted_step_control
from tpms2cfd.brep_engine import BrepCaseSpec, build_brep_case
from tpms2cfd.cli import build_parser
from tpms2cfd.qa_audit import audit_fluid_mesh, audit_mesh_topology, gmsh_mesh_fluid
from tpms2cfd.sheet_engine import SheetCaseSpec, build_sheet_case


class Tpms2CfdTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_brep_kelvin_and_channel_integrity(self) -> None:
        for spec in (
            BrepCaseSpec("TEST_SQ", "square_channel", "unit_cell", 5.0, 20.0, 5.0, 1.0, phase=(0.0, 0.0, 0.0)),
            BrepCaseSpec("TEST_KELVIN", "kelvin_pocs", "unit_cell", 10.0, 20.0, 10.0, 1.6, phase=(0.25, 0.25, 0.25)),
        ):
            rec = build_brep_case(spec, self.out_dir)
            self.assertEqual(rec["solid_qa"]["solid_count"], 1)
            self.assertEqual(rec["fluid_qa"]["solid_count"], 1)
            self.assertEqual(rec["solid_qa"]["degenerated_edges"], 0)
            self.assertEqual(rec["fluid_qa"]["free_edges"], 0)
            self.assertEqual(rec["fluid_qa"]["nonmanifold_edges"], 0)
            vinfo = audit_fluid_mesh(
                spec.case_id,
                Path(rec["fluid_step"]),
                self.out_dir / "preview" / f"{spec.case_id}.stl",
                voxel_pitch=0.50,
            )
            self.assertEqual(vinfo["fluid_connected_fraction"], 1.0)
            self.assertEqual(vinfo["trapped_void_fraction"], 0.0)

    def test_sheet_topologies_watertight_grading_and_msh(self) -> None:
        specs = (
            SheetCaseSpec("TEST_P", "schwarz_p_sheet", "unit_cell", 10.0, 20.0, 10.0, 0.45, voxel_dx_mm=0.50),
            SheetCaseSpec("TEST_IWP", "iwp_sheet", "unit_cell", 10.0, 20.0, 10.0, 0.70, voxel_dx_mm=0.50),
            SheetCaseSpec("TEST_GRADED", "graded_gyroid_sheet", "unit_cell", 10.0, 20.0, 10.0, 0.35, voxel_dx_mm=0.50, grade_delta_t=0.25),
        )
        for spec in specs:
            rec = build_sheet_case(spec, self.out_dir, edt_dx_mm=0.20)
            self.assertTrue(rec["solid_watertight"])
            self.assertTrue(rec["fluid_watertight"])
            self.assertGreater(rec["thickness_edt"]["median_ridge_thickness_mm"], 0.5)
            topo = audit_mesh_topology(Path(rec["fluid_stl"]))
            self.assertEqual(topo["boundary_edges"], 0)
            self.assertEqual(topo["nonmanifold_edges"], 0)
            vinfo = audit_fluid_mesh(spec.case_id, Path(rec["fluid_stl"]), voxel_pitch=0.50)
            self.assertEqual(vinfo["odd_parity_rays"], 0)
            self.assertEqual(vinfo["fluid_connected_fraction"], 1.0, msg=f"Failed connectivity on {spec.case_id}")
            self.assertEqual(vinfo["trapped_void_fraction"], 0.0, msg=f"Trapped voids on {spec.case_id}")

            # Verify Fluent .msh format contains 7 expected boundary zones, bc_id 20 on inlet, and valid hex indices
            msh_text = Path(rec["fluid_surface_msh"]).read_text(encoding="ascii")
            for zname in ("inlet", "outlet", "symmetry_xmin", "symmetry_xmax", "symmetry_ymin", "symmetry_ymax", "internal_wall"):
                self.assertIn(f" {zname})())", msh_text)
            self.assertIn(" 20 3)(", msh_text)
            self.assertIn("(45 (2 mass-flow-inlet inlet)())", msh_text)
            m_verts = re.search(r"\(10 \(1 1 ([0-9a-fA-F]+) 1 3\)\(", msh_text)
            self.assertIsNotNone(m_verts)

            # Verify Gmsh meshing works on short Sheet fluid STL with 0 negative Jacobians
            if spec.case_id == "TEST_GRADED":
                gstats = gmsh_mesh_fluid(spec.case_id, Path(rec["fluid_stl"]), self.out_dir / "gmsh" / "graded.msh", 0.70, 1.40)
                self.assertEqual(gstats["gmsh_negative_jacobian"], 0)
                self.assertGreater(gstats["gmsh_tetra_count"], 500)

                # Verify axial thickness grading: outlet cross-section solid volume > inlet solid volume
                sm = trimesh.load(rec["solid_stl"], force="mesh")
                inlet_slice = trimesh.boolean.intersection([sm, trimesh.creation.box(bounds=[[0, 0, 0], [10, 10, 8]])], engine="manifold")
                outlet_slice = trimesh.boolean.intersection([sm, trimesh.creation.box(bounds=[[0, 0, 12], [10, 10, 20]])], engine="manifold")
                self.assertGreater(abs(float(outlet_slice.volume)), abs(float(inlet_slice.volume)))

    def test_cli_and_control_faceted_step(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["build", "--track", "brep", "--out-dir", str(self.out_dir)])
        self.assertEqual(args.track, "brep")
        ctrl = build_faceted_step_control(self.out_dir, case_id="TEST_CTRL", dx=0.80, t=0.50)
        self.assertGreater(ctrl["fluid_qa"]["face_count"], 1000)
        self.assertTrue(Path(ctrl["fluid_step"]).is_file())


if __name__ == "__main__":
    unittest.main()
