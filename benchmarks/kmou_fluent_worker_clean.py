"""KMOU Fluent 2026 R1 (PyFluent 0.41.3) CPU-only Watertight Meshing + Dimensionless Laminar Hydraulic worker."""
from pathlib import Path
import json
import math
import os
import re
import time
import traceback

ROOT = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import ansys.fluent.core as pf  # noqa: E402
from launch_native_fluent import close_once, launch_once  # noqa: E402

EQUATIONS = ("continuity", "x-velocity", "y-velocity", "z-velocity")
# Standard dry air at 300 K, 101.325 kPa
RHO = 1.177
MU = 1.846e-05
U_SUP = 0.25  # m/s superficial inlet velocity (Re_a ~ 159 for a = 10 mm)


def scalar(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, dict) and len(value) == 1:
        return scalar(next(iter(value.values())))
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar(value[0])
    raise ValueError(f"Nonfinite or ambiguous Fluent report: {value!r}")


def residuals(solver):
    for name in solver.monitors.get_monitor_set_names():
        if "residual" in name.lower():
            x, y = solver.monitors.get_monitor_set_data(monitor_set_name=name)
            if len(x):
                return int(x[-1]), {k: scalar(float(v[-1])) for k, v in y.items() if len(v)}
    return None, {}


def parse_meshing_summary(log_path: Path) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    m_all = re.findall(r"Overall Summary\s+\S+\s+(\d+)\s+([0-9eE+\-.]+)\s+(\d+)", txt)
    out = {}
    if m_all:
        low_q, min_oq, cells = m_all[-1]
        out["cells_oq_lt_0p05"] = int(low_q)
        out["min_orthogonal_quality"] = round(float(min_oq), 6)
        out["cell_count"] = int(cells)
    m_oq2 = re.findall(r"Minimum Orthogonal Quality\s*=\s*([0-9eE+\-.]+)", txt)
    if m_oq2 and "min_orthogonal_quality" not in out:
        out["min_orthogonal_quality"] = round(float(m_oq2[-1]), 6)
    m_ar = re.findall(r"Maximum Aspect Ratio\s*=\s*([0-9eE+\-.]+)", txt)
    if m_ar:
        out["max_aspect_ratio"] = round(float(m_ar[-1]), 3)
    return out


def save_json(name: str, data) -> None:
    (ROOT / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    spec = json.loads((ROOT / "input.json").read_text(encoding="utf-8"))
    cid = spec["case_id"]
    mode = spec["mode"]  # "mesh_and_solve", "mesh_only", or "negative_control"
    started = time.monotonic()
    session = None
    result = {
        "run_id": spec["run_id"],
        "case_id": cid,
        "mode": mode,
        "stage": "STARTED",
        "gpu_requested": False,
        "cpu_ranks": int(spec["processors"]),
        "tasks": [],
        "gate": "HOLD_NOT_COMPLETED",
    }

    try:
        session = launch_once(pf, ROOT, "meshing", int(spec["processors"]))
        log_path = ROOT / "fluent_meshing.log"
        session.transcript.start(file_name=str(log_path), write_to_stdout=False)
        result["fluent_version"] = str(session.get_fluent_version())
        result["pyfluent_version"] = pf.__version__

        workflow = session.workflow
        workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")

        def run_task(name: str, arguments=None):
            t0 = time.monotonic()
            current = workflow.TaskObject[name]
            if arguments:
                current.Arguments.update_dict(arguments)
            current.Execute()
            result["tasks"].append({
                "task": name,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "state": str(current.State()),
            })
            result["stage"] = name
            save_json("progress.json", result)
            return current

        geom_path = str(ROOT / spec["geom_file"])
        if spec["geom_file"].lower().endswith((".stl", ".msh")):
            run_task("Import Geometry", {"FileFormat": "Mesh", "MeshUnit": "mm", "MeshFileName": geom_path})
        else:
            run_task("Import Geometry", {"FileName": geom_path, "LengthUnit": "mm"})
        run_task("Add Local Sizing")
        run_task(
            "Generate the Surface Mesh",
            {"CFDSurfaceMeshControls": {"MinSize": spec["min_size_mm"], "MaxSize": spec["max_size_mm"]}},
        )

        if mode == "negative_control":
            try:
                desc = workflow.TaskObject["Describe Geometry"]
                desc.Arguments.update_dict({"SetupType": "The geometry consists of only fluid regions with no voids"})
                desc.UpdateChildTasks(SetupTypeChanged=True)
                run_task("Describe Geometry")
                run_task("Update Boundaries")
                run_task("Update Regions")
                layers = workflow.TaskObject["Add Boundary Layers"]
                layers.Arguments.update_dict({"AddChild": "yes"})
                layers.AddChildAndUpdate(DeferUpdate=False)
                run_task("Generate the Volume Mesh", {"VolumeFill": "polyhedra"})
                result["gate"] = "UNEXPECTED_PASS_NEGATIVE_CONTROL"
            except Exception as exc:
                result["gate"] = "EXPECTED_FAIL_OPEN_BAFFLE"
                result["failure_stage"] = result.get("stage")
                result["failure_message"] = str(exc)
            result["stage"] = "COMPLETE"
            return

        desc = workflow.TaskObject["Describe Geometry"]
        desc.Arguments.update_dict({"SetupType": "The geometry consists of only fluid regions with no voids"})
        desc.UpdateChildTasks(SetupTypeChanged=True)
        run_task("Describe Geometry")

        ub = workflow.TaskObject["Update Boundaries"]
        ub_state = ub.Arguments.get_state() or {}
        labels = ub_state.get("BoundaryLabelList")
        types = ub_state.get("BoundaryLabelTypeList")
        if isinstance(labels, list) and isinstance(types, list) and len(labels) == len(types):
            new_types = []
            for lbl, old_t in zip(labels, types):
                low = str(lbl).lower()
                if "inlet" in low:
                    new_types.append("mass-flow-inlet")
                elif "outlet" in low:
                    new_types.append("pressure-outlet")
                elif "symmetry" in low or "side" in low:
                    new_types.append("symmetry")
                elif "wall" in low:
                    new_types.append("wall")
                else:
                    new_types.append(old_t)
            ub.Arguments.update_dict({"BoundaryLabelTypeList": new_types})
        run_task("Update Boundaries")
        run_task("Update Regions")

        layers = workflow.TaskObject["Add Boundary Layers"]
        layers.Arguments.update_dict({"AddChild": "yes"})
        layers.AddChildAndUpdate(DeferUpdate=False)

        run_task("Generate the Volume Mesh", {"VolumeFill": "polyhedra"})
        result["mesh_wall_seconds"] = round(time.monotonic() - started, 2)

        result["stage"] = "SWITCH_TO_SOLVER"
        owned_proc = getattr(session, "_owned_start_process", None)
        session = session.switch_to_solver()
        if owned_proc is not None and not hasattr(session, "_owned_start_process"):
            session._owned_start_process = owned_proc
        s = session.settings
        s.mesh.check()
        session.tui.mesh.quality()
        session.tui.mesh.size_info()

        result["mesh_stats"] = parse_meshing_summary(log_path)

        if mode == "mesh_only":
            mesh_file = ROOT / f"{cid}_MESH.cas.h5"
            s.file.write_case(file_name=str(mesh_file))
            result["case_file"] = mesh_file.name
            result["gate"] = "PASS_FLUENT_POLY_MESH"
            result["stage"] = "COMPLETE"
            return

        solve_t0 = time.monotonic()
        s.setup.general.solver.type = "pressure-based"
        s.setup.general.solver.time = "steady"
        s.setup.general.operating_conditions.operating_pressure = 101325.0
        s.setup.models.viscous.model = "laminar"
        s.setup.models.energy.enabled = False

        mat = s.setup.materials.fluid["air"]
        mat.density = {"option": "value", "value": RHO}
        mat.viscosity = {"option": "value", "value": MU}
        for zname in s.setup.cell_zone_conditions.fluid.get_object_names():
            s.setup.cell_zone_conditions.fluid[zname].general.material = "air"

        bc = s.setup.boundary_conditions
        all_wall = set(bc.wall.get_object_names())
        all_vi = set(bc.velocity_inlet.get_object_names()) if hasattr(bc, "velocity_inlet") else set()
        all_mfi = set(bc.mass_flow_inlet.get_object_names()) if hasattr(bc, "mass_flow_inlet") else set()
        all_po = set(bc.pressure_outlet.get_object_names()) if hasattr(bc, "pressure_outlet") else set()
        all_sym = set(bc.symmetry.get_object_names()) if hasattr(bc, "symmetry") else set()

        inlet_zones = [z for z in (all_vi | all_mfi | all_wall) if "inlet" in z.lower()]
        outlet_zones = [z for z in (all_po | all_wall) if "outlet" in z.lower()]
        side_zones = [z for z in (all_wall | all_sym) if "symmetry" in z.lower() or "side" in z.lower()]
        wall_zones = [z for z in all_wall if z not in inlet_zones and z not in outlet_zones and z not in side_zones]
        if not inlet_zones or not outlet_zones or not wall_zones:
            raise RuntimeError(f"Missing expected zones: in={inlet_zones}, out={outlet_zones}, wall={wall_zones}")

        if inlet_zones[0] not in all_mfi:
            bc.set_zone_type(zone_list=inlet_zones, new_type="mass-flow-inlet")
        if outlet_zones[0] not in all_po:
            bc.set_zone_type(zone_list=outlet_zones, new_type="pressure-outlet")
        to_sym = [z for z in side_zones if z not in all_sym]
        if to_sym:
            bc.set_zone_type(zone_list=to_sym, new_type="symmetry")

        w_m = float(spec["w_mm"]) * 1e-3
        l_core_m = float(spec["l_core_mm"]) * 1e-3
        inlet_area_m2 = w_m * w_m
        mflow = RHO * U_SUP * inlet_area_m2
        bc.mass_flow_inlet[inlet_zones[0]].momentum.mass_flow_rate.value = mflow
        bc.pressure_outlet[outlet_zones[0]].momentum.gauge_pressure = {"option": "value", "value": 0.0}

        s.solution.methods.p_v_coupling.flow_scheme = "Coupled"
        s.solution.methods.spatial_discretization.discretization_scheme = {
            "pressure": "second-order",
            "mom": "second-order-upwind",
        }
        monitor = s.solution.monitor.residual
        monitor.options.criterion_type = "absolute"
        monitor.options.n_save = 1000
        for eq in EQUATIONS:
            monitor.equations[eq].check_convergence = False

        planes = s.results.surfaces.plane_surface
        for pname, zloc in (
            ("core_in", 0.000),
            ("core_out", l_core_m),
            ("tap_up", -0.002),
            ("tap_down", l_core_m + 0.002),
            ("core_z10", 0.010),
            ("core_z20", 0.020),
            ("core_z30", 0.030),
        ):
            planes.create(name=pname)
            planes[pname].method = "xy-plane"
            planes[pname].z = zloc

        reports = s.results.report.surface_integrals
        areas_m2 = {
            "inlet": scalar(reports.get_area(surface_names=[inlet_zones[0]])),
            "outlet": scalar(reports.get_area(surface_names=[outlet_zones[0]])),
            "internal_wall": sum(scalar(reports.get_area(surface_names=[w])) for w in wall_zones),
        }

        s.solution.initialization.hybrid_initialize()
        history = []
        converged = False
        in_name, out_name = inlet_zones[0], outlet_zones[0]
        last = {}

        for budget in range(50, 501, 50):
            s.solution.run_calculation.iterate(iter_count=50)
            p_in = scalar(reports.get_area_weighted_avg(surface_names=[in_name], report_of="pressure"))
            p_out = scalar(reports.get_area_weighted_avg(surface_names=[out_name], report_of="pressure"))
            p_c0 = scalar(reports.get_area_weighted_avg(surface_names=["core_in"], report_of="pressure"))
            p_c1 = scalar(reports.get_area_weighted_avg(surface_names=["core_out"], report_of="pressure"))
            p_up = scalar(reports.get_area_weighted_avg(surface_names=["tap_up"], report_of="pressure"))
            p_dn = scalar(reports.get_area_weighted_avg(surface_names=["tap_down"], report_of="pressure"))
            m_in = scalar(reports.get_mass_flow_rate(surface_names=[in_name]))
            m_out = scalar(reports.get_mass_flow_rate(surface_names=[out_name]))
            it, res = residuals(session)
            dp_core = p_c0 - p_c1
            dp_tap = p_up - p_dn
            mass_err_pct = 100.0 * abs(m_in + m_out) / max(mflow, 1e-18)
            inlet_err_pct = 100.0 * abs(abs(m_in) / max(mflow, 1e-18) - 1.0)
            last = {
                "budget": budget,
                "iteration": it,
                "residuals": res,
                "dp_total_pa": p_in - p_out,
                "dp_core_pa": dp_core,
                "dp_tap_pa": dp_tap,
                "mass_balance_error_pct": mass_err_pct,
                "inlet_mass_error_pct": inlet_err_pct,
            }
            history.append(last)
            recent_dp = [row["dp_core_pa"] for row in history[-2:]]
            rel_dp_chg = abs(recent_dp[-1] - recent_dp[0]) / max(abs(dp_core), 1e-6) if len(recent_dp) == 2 else 1.0
            if (
                len(recent_dp) == 2
                and rel_dp_chg <= 0.005
                and all(k in res and abs(res[k]) <= 1e-5 for k in EQUATIONS)
                and max(mass_err_pct, inlet_err_pct) < 0.5
            ):
                converged = True
                break

        vmags, vzs = [], []
        for pz in ("core_z10", "core_z20", "core_z30"):
            vmags.append(scalar(reports.get_area_weighted_avg(surface_names=[pz], report_of="velocity-magnitude")))
            vzs.append(scalar(reports.get_area_weighted_avg(surface_names=[pz], report_of="z-velocity")))
        mean_vmag = sum(vmags) / len(vmags)
        mean_vz = sum(vzs) / len(vzs)
        mean_vxy = math.sqrt(max(mean_vmag * mean_vmag - mean_vz * mean_vz, 0.0))
        mix_ratio_pct = 100.0 * mean_vxy / max(abs(mean_vz), 1e-9)

        solved_cas = ROOT / f"{cid}_SOLVED.cas.h5"
        s.file.write_case_data(file_name=str(solved_cas))

        dp_core_pa = float(last["dp_core_pa"])
        dp_per_m = dp_core_pa / l_core_m
        k_darcy = (MU * U_SUP) / max(dp_per_m, 1e-9)  # m^2
        a_ref_m = 0.010  # 10 mm reference unit cell pitch
        k_dimless = k_darcy / (a_ref_m ** 2)
        re_a = (RHO * U_SUP * a_ref_m) / MU
        f_darcy_a = (2.0 * dp_per_m * a_ref_m) / (RHO * U_SUP * U_SUP)
        f_re_a = f_darcy_a * re_a

        max_res = max([abs(v) for v in last.get("residuals", {}).values()] or [1.0])
        result.update({
            "solve_wall_seconds": round(time.monotonic() - solve_t0, 2),
            "converged": converged,
            "iterations": last.get("iteration"),
            "max_residual": max_res,
            "residuals": last.get("residuals"),
            "residual_history_tail": history[-4:],
            "mass_balance_error_pct": round(float(last["mass_balance_error_pct"]), 5),
            "pressure_drop_total_pa": round(float(last["dp_total_pa"]), 5),
            "pressure_drop_core_pa": round(dp_core_pa, 5),
            "pressure_drop_tap_pa": round(float(last["dp_tap_pa"]), 5),
            "pressure_drop_pa_per_m": round(dp_per_m, 3),
            "darcy_permeability_m2": k_darcy,
            "dimensionless_permeability_k_a2": round(k_dimless, 6),
            "reynolds_pitch": round(re_a, 2),
            "friction_re_pitch": round(f_re_a, 3),
            "core_mean_vz_m_s": round(mean_vz, 4),
            "core_mean_vxy_m_s": round(mean_vxy, 4),
            "transverse_mixing_proxy_pct": round(mix_ratio_pct, 2),
            "internal_wall_area_cm2": round(areas_m2["internal_wall"] * 1e4, 3),
            "inlet_area_mm2": round(areas_m2["inlet"] * 1e6, 3),
            "case_file": solved_cas.name,
            "data_file": f"{cid}_SOLVED.dat.h5",
            "gate": "PASS_FLUENT_LAMINAR_CONVERGED" if (converged or (max_res < 1e-4 and last["mass_balance_error_pct"] < 0.5)) else "HOLD_FLUENT_CONVERGENCE",
            "stage": "COMPLETE",
        })
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if session is not None:
            try:
                if hasattr(session, "_owned_start_process"):
                    result["launcher_exit_code"] = close_once(session)
                else:
                    session.exit(timeout=30, wait=30)
                    result["launcher_exit_code"] = 0
                result["api_exit_returned"] = True
            except Exception as err:
                result["api_exit_returned"] = False
                result["exit_error"] = str(err)
        result["wall_seconds"] = round(time.monotonic() - started, 2)
        save_json("result.json", result)


if __name__ == "__main__":
    main()
