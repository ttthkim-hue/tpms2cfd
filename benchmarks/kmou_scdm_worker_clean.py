# SpaceClaim 2026 R1 (/ScriptAPI=261) headless worker for CAD import, Named Selection tagging, and .scdocx benchmark.
import json
import os
import time
import traceback
import clr

clr.AddReference("SpaceClaim.Api.V261")
from System.Collections.Generic import List  # noqa: E402
from SpaceClaim.Api.V261 import Document, ImportOptions, Window, IDocObject, Group  # noqa: E402
from SpaceClaim.Api.V261.Geometry import Matrix  # noqa: E402

work = os.path.dirname(__file__)
with open(os.path.join(work, "input.json"), "r") as stream:
    settings = json.load(stream)

case_id = settings["case_id"]
kind = settings["kind"]  # "brep_fluid", "control_brep_fluid", or "stl_solid"
is_monolith = bool(settings.get("is_monolith", False))
result = {
    "run_id": settings["run_id"],
    "case_id": case_id,
    "kind": kind,
    "stage": "STARTED",
    "source_sha256": settings["source_sha256"],
    "gpu_requested": False,
    "headless_requested": True,
}
start = time.time()


def collect_entities(part, out_bodies, out_meshes):
    for b in part.Bodies:
        out_bodies.append((b, part))
    if hasattr(part, "Meshes"):
        for m in part.Meshes:
            out_meshes.append((m, part))
    for comp in part.Components:
        collect_entities(comp.Content, out_bodies, out_meshes)


try:
    options = ImportOptions.Create()
    options.CleanUpBodies = False
    options.ImproveImportedData = False
    options.HealBodies = False
    options.StitchSurfaces = False
    options.ImportNames = True

    src_path = os.path.join(work, settings["filename"])
    document = Document.Open(src_path, options)
    result["stage"] = "IMPORTED"
    result["import_seconds"] = round(time.time() - start, 3)

    bodies = []
    meshes = []
    collect_entities(document.MainPart, bodies, meshes)
    result["body_count"] = len(bodies)
    result["mesh_count"] = len(meshes)

    if kind in ("brep_fluid", "control_brep_fluid"):
        if len(bodies) < 1:
            raise RuntimeError("No B-Rep bodies imported from STEP")
        # For faceted controls that fail sewing into 1 solid, inspect aggregate metrics safely
        total_faces = sum(int(b.Shape.Faces.Count) for b, _ in bodies)
        total_edges = sum(int(b.Shape.Edges.Count) for b, _ in bodies)
        all_closed = all(bool(b.Shape.IsClosed) for b, _ in bodies)
        all_manifold = all(bool(b.Shape.IsManifold) for b, _ in bodies)
        total_pieces = sum(int(b.Shape.PieceCount) for b, _ in bodies)
        total_vol_cm3 = sum(float(b.Shape.Volume) * 1e6 for b, _ in bodies)
        total_area_cm2 = sum(float(b.Shape.SurfaceArea) * 1e4 for b, _ in bodies)
        result["body_metrics"] = {
            "closed": all_closed,
            "manifold": all_manifold,
            "pieces": total_pieces,
            "faces": total_faces,
            "edges": total_edges,
            "volume_cm3": round(total_vol_cm3, 5),
            "total_surface_area_cm2": round(total_area_cm2, 4),
        }

        if len(bodies) == 1 and all_closed and all_manifold and total_pieces == 1:
            body, _ = bodies[0]
            face_bbs = []
            for f in body.Faces:
                fb = f.Shape.GetBoundingBox(Matrix.Identity, True)
                face_bbs.append((f, fb.MinCorner.X, fb.MinCorner.Y, fb.MinCorner.Z, fb.MaxCorner.X, fb.MaxCorner.Y, fb.MaxCorner.Z))

            zmin = min(item[3] for item in face_bbs)
            zmax = max(item[6] for item in face_bbs)
            tol = 1e-6
            inlet_items = [it for it in face_bbs if abs(it[3] - zmin) < tol and abs(it[6] - zmin) < tol]
            if not inlet_items:
                raise RuntimeError("No planar inlet faces found at zmin=%g" % zmin)
            xmin = min(it[1] for it in inlet_items)
            ymin = min(it[2] for it in inlet_items)
            xmax = max(it[4] for it in inlet_items)
            ymax = max(it[5] for it in inlet_items)

            if is_monolith:
                gnames = ("inlet", "outlet", "outer_wall", "internal_wall")
            else:
                gnames = ("inlet", "outlet", "symmetry_xmin", "symmetry_xmax", "symmetry_ymin", "symmetry_ymax", "internal_wall")

            groups = dict((name, List[IDocObject]()) for name in gnames)
            areas_cm2 = dict((name, 0.0) for name in gnames)
            for f, fx0, fy0, fz0, fx1, fy1, fz1 in face_bbs:
                if abs(fz0 - zmin) < tol and abs(fz1 - zmin) < tol:
                    gkey = "inlet"
                elif abs(fz0 - zmax) < tol and abs(fz1 - zmax) < tol:
                    gkey = "outlet"
                elif not is_monolith:
                    if abs(fx0 - xmin) < tol and abs(fx1 - xmin) < tol:
                        gkey = "symmetry_xmin"
                    elif abs(fx0 - xmax) < tol and abs(fx1 - xmax) < tol:
                        gkey = "symmetry_xmax"
                    elif abs(fy0 - ymin) < tol and abs(fy1 - ymin) < tol:
                        gkey = "symmetry_ymin"
                    elif abs(fy0 - ymax) < tol and abs(fy1 - ymax) < tol:
                        gkey = "symmetry_ymax"
                    else:
                        gkey = "internal_wall"
                else:
                    if (
                        (abs(fx0 - xmin) < tol and abs(fx1 - xmin) < tol)
                        or (abs(fx0 - xmax) < tol and abs(fx1 - xmax) < tol)
                        or (abs(fy0 - ymin) < tol and abs(fy1 - ymin) < tol)
                        or (abs(fy0 - ymax) < tol and abs(fy1 - ymax) < tol)
                    ):
                        gkey = "outer_wall"
                    else:
                        gkey = "internal_wall"
                groups[gkey].Add(f)
                areas_cm2[gkey] += float(f.Shape.Area) * 1e4

            for gkey in gnames:
                if groups[gkey].Count > 0:
                    Group.Create(document.MainPart, gkey, groups[gkey])
                elif kind == "brep_fluid":
                    raise RuntimeError("Empty Named Selection '%s' for %s" % (gkey, case_id))

            result["boundary_faces"] = dict((k, int(groups[k].Count)) for k in gnames)
            result["boundary_area_cm2"] = dict((k, round(areas_cm2[k], 4)) for k in gnames)
        elif kind == "brep_fluid":
            raise RuntimeError(
                "B-Rep is not single closed manifold (bodies=%d, closed=%s, manifold=%s, pieces=%d)"
                % (len(bodies), all_closed, all_manifold, total_pieces)
            )
    else:
        if len(bodies) + len(meshes) < 1:
            raise RuntimeError("No body or facet mesh found in STL")

    hidden_window = Window.Create(document.MainPart, False)
    document.IsModified = True
    out_name = case_id + ".scdocx"
    out_path = os.path.join(work, out_name)
    DocumentSave.Execute(out_path)  # noqa: F821
    result["native_file_exists"] = os.path.isfile(out_path)
    result["native_size_bytes"] = os.path.getsize(out_path) if result["native_file_exists"] else 0
    result["saved_scdocx"] = out_name
    result["gate"] = "PASS_SPACECLAIM" if result["native_file_exists"] and result["native_size_bytes"] >= 1024 else "HOLD_SPACECLAIM"
    result["stage"] = "COMPLETE"
except Exception:
    result["gate"] = "HOLD_SPACECLAIM"
    result["error"] = traceback.format_exc()
finally:
    result["wall_seconds"] = round(time.time() - start, 3)
    with open(os.path.join(work, "result.json"), "w") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True)
