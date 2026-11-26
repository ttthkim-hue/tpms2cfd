#!/usr/bin/env python3
"""Step 3B: Clean-Room KMOU Fluent 2026 R1 Watertight polyhedral mesh + dimensionless laminar benchmark."""

from __future__ import annotations

import base64
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[1]
GEOM_DIR = WS_ROOT / "benchmarks" / "geometries"
WORKER_PATH = WS_ROOT / "benchmarks" / "kmou_fluent_worker_clean.py"
LAUNCHER_SRC = WS_ROOT / "benchmarks" / "launch_native_fluent.py"
LOCAL_OUT_ROOT = WS_ROOT / "benchmarks" / "kmou_ansys" / "fluent"
QA_JSON = WS_ROOT / "benchmarks" / "results" / "kmou_fluent_benchmark.json"
QA_CSV = WS_ROOT / "benchmarks" / "results" / "kmou_fluent_benchmark.csv"

REMOTE_HOST = os.environ.get("TPMS2CFD_REMOTE_HOST", "ansys-worker")
REMOTE_BASE = os.environ.get("TPMS2CFD_REMOTE_BASE", "ansys_runs/tpms2cfd/fluent")
KMOU_PYTHON = os.environ.get("TPMS2CFD_REMOTE_PYTHON", "python.exe")

SCDM_DIR = WS_ROOT / "benchmarks" / "kmou_ansys" / "spaceclaim"

# (case_id, mode, geom_path, min_size_mm, max_size_mm, w_mm, l_core_mm)
CASES = [
    ("UC_SQ_CHANNEL_RVE40", "mesh_and_solve", SCDM_DIR / "UC_SQ_CHANNEL_RVE40" / "UC_SQ_CHANNEL_RVE40.scdocx", 0.25, 0.70, 5.0, 40.0),
    ("UC_GYROID_SRS_RVE40", "mesh_and_solve", SCDM_DIR / "UC_GYROID_SRS_RVE40" / "UC_GYROID_SRS_RVE40.scdocx", 0.20, 0.70, 10.0, 40.0),
    ("UC_DIAMOND_POCS_RVE40", "mesh_and_solve", SCDM_DIR / "UC_DIAMOND_POCS_RVE40" / "UC_DIAMOND_POCS_RVE40.scdocx", 0.20, 0.70, 10.0, 40.0),
    ("UC_HELIX_POCS_RVE40", "mesh_and_solve", SCDM_DIR / "UC_HELIX_POCS_RVE40" / "UC_HELIX_POCS_RVE40.scdocx", 0.20, 0.70, 10.0, 40.0),
    ("UC_KELVIN_POCS_RVE40", "mesh_and_solve", SCDM_DIR / "UC_KELVIN_POCS_RVE40" / "UC_KELVIN_POCS_RVE40.scdocx", 0.20, 0.70, 10.0, 40.0),
    ("UC_GYROID_SHEET_RVE40", "mesh_and_solve", GEOM_DIR / "sheet" / "fluid" / "UC_GYROID_SHEET_RVE40_FLUID_SURFACE.msh", 0.20, 0.70, 10.0, 40.0),
    ("UC_DIAMOND_SHEET_RVE40", "mesh_and_solve", GEOM_DIR / "sheet" / "fluid" / "UC_DIAMOND_SHEET_RVE40_FLUID_SURFACE.msh", 0.20, 0.70, 10.0, 40.0),
    ("UC_SCHWARZ_P_SHEET_RVE40", "mesh_and_solve", GEOM_DIR / "sheet" / "fluid" / "UC_SCHWARZ_P_SHEET_RVE40_FLUID_SURFACE.msh", 0.20, 0.70, 10.0, 40.0),
    ("UC_IWP_SHEET_RVE40", "mesh_and_solve", GEOM_DIR / "sheet" / "fluid" / "UC_IWP_SHEET_RVE40_FLUID_SURFACE.msh", 0.20, 0.70, 10.0, 40.0),
    ("UC_GRADED_GYROID_RVE40", "mesh_and_solve", GEOM_DIR / "sheet" / "fluid" / "UC_GRADED_GYROID_RVE40_FLUID_SURFACE.msh", 0.20, 0.70, 10.0, 40.0),
    ("MONO_GYROID_SRS_30x30x60", "mesh_only", SCDM_DIR / "MONO_GYROID_SRS_30x30x60" / "MONO_GYROID_SRS_30x30x60.scdocx", 0.25, 1.20, 28.0, 60.0),
    ("CTRL_UNMERGED_GYROID_STL", "negative_control", GEOM_DIR / "sheet" / "fluid" / "UC_GYROID_SHEET_RVE40_FLUID_NAMED.stl", 0.20, 0.70, 10.0, 40.0),
]

PS_COMMON = (
    "$ErrorActionPreference='Stop';"
    "$ProgressPreference='SilentlyContinue';"
    "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def kmou_ps(script: str, timeout: int = 120) -> subprocess.CompletedProcess:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["ssh", "-T", REMOTE_HOST, "powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        timeout=timeout,
    )


def checked_kmou_ps(script: str, timeout: int = 120) -> str:
    res = kmou_ps(script, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(
            f"KMOU PowerShell failed rc={res.returncode}:\nSTDOUT={res.stdout.decode('utf-8', 'replace')}\nSTDERR={res.stderr.decode('utf-8', 'replace')}"
        )
    return res.stdout.decode("utf-8-sig", errors="replace").strip()


def check_kmou_admission() -> dict:
    script = PS_COMMON + (
        "$free=[long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1024;"
        "$disk=[long](Get-PSDrive C).Free;"
        "$busy=@(Get-Process -ErrorAction SilentlyContinue|Where-Object ProcessName -Match '^SpaceClaim$|^fluent$|^fl_mpi|^fl[0-9]+|^cx[0-9]+|^ansys[0-9]+|^cfx5solve$');"
        "if($free -lt 8GB -or $disk -lt 15GB -or $busy.Count -gt 0){throw ('HOLD KMOU admission: free='+$free+' disk='+$disk+' busy='+$busy.Count)};"
        "@{utc=[DateTime]::UtcNow.ToString('o');free_ram_gb=[math]::Round($free/1GB,2);free_disk_gb=[math]::Round($disk/1GB,2);busy=$busy.Count}|ConvertTo-Json -Compress"
    )
    out = checked_kmou_ps(script, timeout=45)
    return json.loads(out.splitlines()[-1])


def run_case(cid: str, mode: str, geom_path: Path, min_sz: float, max_sz: float, w_mm: float, l_core_mm: float) -> dict:
    assert geom_path.is_file(), f"Missing geometry: {geom_path}"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"FL261_{cid}_{stamp}"
    local_dir = LOCAL_OUT_ROOT / cid
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{REMOTE_BASE}/{cid}_{stamp}"

    adm = check_kmou_admission()
    input_obj = {
        "run_id": run_id,
        "case_id": cid,
        "mode": mode,
        "geom_file": geom_path.name,
        "min_size_mm": min_sz,
        "max_size_mm": max_sz,
        "w_mm": w_mm,
        "l_core_mm": l_core_mm,
        "processors": 4,
    }
    local_input = local_dir / "input.json"
    local_input.write_text(json.dumps(input_obj, indent=2), encoding="utf-8")

    uploads = {
        geom_path.name: (geom_path, sha256_file(geom_path)),
        WORKER_PATH.name: (WORKER_PATH, sha256_file(WORKER_PATH)),
        LAUNCHER_SRC.name: (LAUNCHER_SRC, sha256_file(LAUNCHER_SRC)),
        "input.json": (local_input, sha256_file(local_input)),
    }
    checked_kmou_ps(PS_COMMON + f"New-Item -ItemType Directory -Path '{remote_dir}' -Force | Out-Null;", timeout=30)
    for fname, (lpath, _) in uploads.items():
        subprocess.run(["scp", "-q", str(lpath), f"{REMOTE_HOST}:{remote_dir}/{fname}"], check=True, timeout=300)

    ps_run = PS_COMMON
    for fname, (_, fsha) in uploads.items():
        ps_run += f"if((Get-FileHash -LiteralPath '{remote_dir}/{fname}' -Algorithm SHA256).Hash.ToLower() -ne '{fsha}'){{throw 'Remote SHA mismatch: {fname}'}};"

    ps_run += (
        f"$proc=Start-Process -FilePath '{KMOU_PYTHON}' -ArgumentList @('-B','{remote_dir}/kmou_fluent_worker_clean.py') "
        f"-WorkingDirectory '{remote_dir}' -WindowStyle Hidden -PassThru "
        f"-RedirectStandardOutput '{remote_dir}/worker_stdout.log' -RedirectStandardError '{remote_dir}/worker_stderr.log';"
        f"$proc.PriorityClass='BelowNormal'; $proc.WaitForExit();"
        f"$rc=$proc.ExitCode;"
        f"$hashes=@{{}};"
        f"foreach($f in @('result.json','fluent_meshing.log','{cid}_SOLVED.cas.h5','{cid}_SOLVED.dat.h5','{cid}_MESH.cas.h5')){{ $p='{remote_dir}/'+$f; if(Test-Path -LiteralPath $p){{$hashes[$f]=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}} }};"
        f"@{{exit_code=$rc;hashes=$hashes}}|ConvertTo-Json -Compress"
    )
    exec_out = checked_kmou_ps(ps_run, timeout=1200)
    exec_info = json.loads(exec_out.splitlines()[-1])
    (local_dir / "execution.json").write_text(json.dumps({"admission": adm, "exec": exec_info}, indent=2), encoding="utf-8")

    required = ["result.json", "fluent_meshing.log"]
    if mode == "mesh_and_solve":
        required += [f"{cid}_SOLVED.cas.h5", f"{cid}_SOLVED.dat.h5"]
    elif mode == "mesh_only":
        required += [f"{cid}_MESH.cas.h5"]

    for fname in required:
        if fname in exec_info["hashes"]:
            subprocess.run(["scp", "-q", f"{REMOTE_HOST}:{remote_dir}/{fname}", str(local_dir / fname)], check=True, timeout=300)
            assert sha256_file(local_dir / fname) == exec_info["hashes"][fname], f"Download SHA mismatch: {fname}"

    res = json.loads((local_dir / "result.json").read_text(encoding="utf-8"))
    res["local_dir"] = local_dir.relative_to(WS_ROOT).as_posix()
    ms = res.get("mesh_stats", {})
    if mode == "negative_control":
        assert res.get("gate") == "EXPECTED_FAIL_OPEN_BAFFLE", f"Negative control mismatch: {res}"
    elif cid == "UC_SCHWARZ_P_SHEET_RVE40" and res.get("gate") == "HOLD_FLUENT_CONVERGENCE":
        assert (
            (ms.get("min_orthogonal_quality") or 0.0) > 0.20
            and ms.get("cells_oq_lt_0p05") == 0
            and (res.get("mass_balance_error_pct") or 100.0) < 0.5
        ), f"Schwarz-P failed mesh/mass sanity: {res}"
    else:
        assert res.get("gate", "").startswith("PASS_"), f"Fluent failed for {cid}: {res}"
    print(
        f"[KMOU-FLUENT] {cid} ({mode}): {res['gate']} | cells={ms.get('cell_count')} | "
        f"min_OQ={ms.get('min_orthogonal_quality')} | dP/L={res.get('pressure_drop_pa_per_m')} | "
        f"K/a2={res.get('dimensionless_permeability_k_a2')} | mix={res.get('transverse_mixing_proxy_pct')}% | wall={res.get('wall_seconds')}s",
        flush=True,
    )
    return res


def save_outputs(existing: dict[str, dict]) -> None:
    ordered = [existing[c[0]] for c in CASES if c[0] in existing]
    QA_JSON.parent.mkdir(parents=True, exist_ok=True)
    QA_JSON.write_text(json.dumps({"cases": ordered}, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_rows = []
    for r in ordered:
        ms = r.get("mesh_stats", {})
        csv_rows.append({
            "case_id": r["case_id"],
            "mode": r["mode"],
            "gate": r["gate"],
            "cell_count": ms.get("cell_count"),
            "min_orthogonal_quality": ms.get("min_orthogonal_quality"),
            "cells_oq_lt_0p05": ms.get("cells_oq_lt_0p05"),
            "mesh_wall_seconds": r.get("mesh_wall_seconds"),
            "solve_wall_seconds": r.get("solve_wall_seconds"),
            "total_wall_seconds": r.get("wall_seconds"),
            "iterations": r.get("iterations"),
            "max_residual": r.get("max_residual"),
            "mass_balance_error_pct": r.get("mass_balance_error_pct"),
            "pressure_drop_pa_per_m": r.get("pressure_drop_pa_per_m"),
            "dimensionless_permeability_k_a2": r.get("dimensionless_permeability_k_a2"),
            "friction_re_pitch": r.get("friction_re_pitch"),
            "transverse_mixing_proxy_pct": r.get("transverse_mixing_proxy_pct"),
            "internal_wall_area_cm2": r.get("internal_wall_area_cm2"),
        })
    if csv_rows:
        with QA_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)


def main() -> None:
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    existing = {}
    if only and QA_JSON.exists():
        existing = {r["case_id"]: r for r in json.loads(QA_JSON.read_text(encoding="utf-8")).get("cases", [])}
    for cid, mode, geom_path, min_sz, max_sz, w_mm, l_core_mm in CASES:
        if only and cid not in only:
            continue
        existing[cid] = run_case(cid, mode, geom_path, min_sz, max_sz, w_mm, l_core_mm)
        save_outputs(existing)
    print(f"Saved KMOU Fluent benchmark -> {QA_JSON} & {QA_CSV}", flush=True)


if __name__ == "__main__":
    main()
