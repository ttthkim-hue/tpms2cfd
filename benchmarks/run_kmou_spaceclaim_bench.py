#!/usr/bin/env python3
"""Step 3A: Clean-Room KMOU SpaceClaim 2026 R1 CAD import, Named Selection, and compression benchmark."""

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
WORKER_PATH = WS_ROOT / "benchmarks" / "kmou_scdm_worker_clean.py"
LOCAL_OUT_ROOT = WS_ROOT / "benchmarks" / "kmou_ansys" / "spaceclaim"
QA_JSON = WS_ROOT / "benchmarks" / "results" / "kmou_spaceclaim_benchmark.json"
QA_CSV = WS_ROOT / "benchmarks" / "results" / "kmou_spaceclaim_benchmark.csv"

REMOTE_HOST = os.environ.get("TPMS2CFD_REMOTE_HOST", "ansys-worker")
REMOTE_BASE = os.environ.get("TPMS2CFD_REMOTE_BASE", "ansys_runs/tpms2cfd/spaceclaim")
SCDM_EXE = os.environ.get(
    "TPMS2CFD_SCDM_EXE",
    str(Path(os.environ.get("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\v261")) / "scdm" / "SpaceClaim.exe"),
)

CASES = [
    ("UC_SQ_CHANNEL_RVE40", "brep_fluid", False, GEOM_DIR / "brep" / "fluid" / "UC_SQ_CHANNEL_RVE40_FLUID.step"),
    ("UC_GYROID_SRS_RVE40", "brep_fluid", False, GEOM_DIR / "brep" / "fluid" / "UC_GYROID_SRS_RVE40_FLUID.step"),
    ("UC_DIAMOND_POCS_RVE40", "brep_fluid", False, GEOM_DIR / "brep" / "fluid" / "UC_DIAMOND_POCS_RVE40_FLUID.step"),
    ("UC_HELIX_POCS_RVE40", "brep_fluid", False, GEOM_DIR / "brep" / "fluid" / "UC_HELIX_POCS_RVE40_FLUID.step"),
    ("UC_KELVIN_POCS_RVE40", "brep_fluid", False, GEOM_DIR / "brep" / "fluid" / "UC_KELVIN_POCS_RVE40_FLUID.step"),
    ("MONO_GYROID_SRS_30x30x60", "brep_fluid", True, GEOM_DIR / "brep" / "fluid" / "MONO_GYROID_SRS_30x30x60_FLUID.step"),
    ("UC_GYROID_SHEET_RVE40", "stl_solid", False, GEOM_DIR / "sheet" / "solid" / "UC_GYROID_SHEET_RVE40_SOLID.stl"),
    ("UC_DIAMOND_SHEET_RVE40", "stl_solid", False, GEOM_DIR / "sheet" / "solid" / "UC_DIAMOND_SHEET_RVE40_SOLID.stl"),
    ("UC_SCHWARZ_P_SHEET_RVE40", "stl_solid", False, GEOM_DIR / "sheet" / "solid" / "UC_SCHWARZ_P_SHEET_RVE40_SOLID.stl"),
    ("UC_IWP_SHEET_RVE40", "stl_solid", False, GEOM_DIR / "sheet" / "solid" / "UC_IWP_SHEET_RVE40_SOLID.stl"),
    ("UC_GRADED_GYROID_RVE40", "stl_solid", False, GEOM_DIR / "sheet" / "solid" / "UC_GRADED_GYROID_RVE40_SOLID.stl"),
    ("CTRL_FACETED_GYROID_RVE40", "control_brep_fluid", False, GEOM_DIR / "controls" / "CTRL_FACETED_GYROID_RVE40_FLUID.step"),
    ("CTRL_FACETED_SRS_d010", "control_brep_fluid", False, GEOM_DIR / "ablation" / "CTRL_FACETED_SRS_d01_FLUID.step"),
    ("CTRL_FACETED_SRS_d005", "control_brep_fluid", False, GEOM_DIR / "ablation" / "CTRL_FACETED_SRS_d005_FLUID.step"),
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
        "$busy=@(Get-Process -ErrorAction SilentlyContinue|Where-Object ProcessName -Match '^SpaceClaim$|^fluent$|^fl_mpi|^fl[0-9]+|^cx[0-9]+|^ansys[0-9]+|^cfx5solve$');"
        "if($free -lt 8GB -or $busy.Count -gt 0){throw ('HOLD KMOU admission: free='+$free+' busy='+$busy.Count)};"
        "@{utc=[DateTime]::UtcNow.ToString('o');free_ram_gb=[math]::Round($free/1GB,2);busy=$busy.Count}|ConvertTo-Json -Compress"
    )
    out = checked_kmou_ps(script, timeout=45)
    return json.loads(out.splitlines()[-1])


def run_case(cid: str, kind: str, is_monolith: bool, src_path: Path) -> dict:
    assert src_path.is_file(), f"Missing CAD source: {src_path}"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"SC261_{cid}_{stamp}"
    local_dir = LOCAL_OUT_ROOT / cid
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{REMOTE_BASE}/{cid}_{stamp}"

    adm = check_kmou_admission()
    src_sha = sha256_file(src_path)
    worker_sha = sha256_file(WORKER_PATH)
    input_obj = {
        "run_id": run_id,
        "case_id": cid,
        "kind": kind,
        "is_monolith": is_monolith,
        "filename": src_path.name,
        "source_sha256": src_sha,
    }
    local_input = local_dir / "input.json"
    local_input.write_text(json.dumps(input_obj, indent=2), encoding="utf-8")
    input_sha = sha256_file(local_input)

    checked_kmou_ps(PS_COMMON + f"New-Item -ItemType Directory -Path '{remote_dir}' -Force | Out-Null;", timeout=30)
    uploads = {
        src_path.name: (src_path, src_sha),
        WORKER_PATH.name: (WORKER_PATH, worker_sha),
        "input.json": (local_input, input_sha),
    }
    for fname, (lpath, _) in uploads.items():
        subprocess.run(["scp", "-q", str(lpath), f"{REMOTE_HOST}:{remote_dir}/{fname}"], check=True, timeout=240)

    verify_sha_ps = PS_COMMON
    for fname, (_, fsha) in uploads.items():
        verify_sha_ps += f"if((Get-FileHash -LiteralPath '{remote_dir}/{fname}' -Algorithm SHA256).Hash.ToLower() -ne '{fsha}'){{throw 'Remote SHA mismatch: {fname}'}};"

    run_sc_ps = (
        verify_sha_ps
        + f"$work='{remote_dir}'; $exe='{SCDM_EXE}';"
        + "$args=@('/Headless=True','/Splash=False','/Welcome=False','/ExitAfterScript=True','/ScriptAPI=261',('/RunScript='+$work+'/kmou_scdm_worker_clean.py'),('/ScriptOutput='+$work+'/script_output.log'));"
        + "$proc=Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $work -WindowStyle Hidden -PassThru -RedirectStandardOutput ($work+'/stdout.log') -RedirectStandardError ($work+'/stderr.log');"
        + "$proc.PriorityClass='BelowNormal'; $pidOwned=$proc.Id; $start=[DateTime]::UtcNow; $maxMem=0L; $reason='EXITED';"
        + "while(-not $proc.WaitForExit(1000)){$proc.Refresh();$maxMem=[Math]::Max($maxMem,$proc.WorkingSet64);if(([DateTime]::UtcNow-$start).TotalSeconds -gt 480 -or $proc.WorkingSet64 -gt 14GB){$reason='HOLD_TIMEOUT_OR_MEM';$proc.Kill();$proc.WaitForExit();break}};"
        + "$hashes=@{}; foreach($f in @('result.json','" + cid + ".scdocx')){$p=$work+'/'+$f; if(Test-Path -LiteralPath $p){$hashes[$f]=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}};"
        + "@{run_id='" + run_id + "';pid=$pidOwned;exit_code=$proc.ExitCode;reason=$reason;wall_seconds=[math]::Round(([DateTime]::UtcNow-$start).TotalSeconds,2);peak_ram_mb=[math]::Round($maxMem/1MB,1);hashes=$hashes}|ConvertTo-Json -Compress"
    )
    exec_json_str = checked_kmou_ps(run_sc_ps, timeout=540)
    exec_info = json.loads(exec_json_str.splitlines()[-1])
    (local_dir / "execution.json").write_text(json.dumps({"admission": adm, "exec": exec_info}, indent=2), encoding="utf-8")

    scdocx_name = f"{cid}.scdocx"
    for fname in ("result.json", scdocx_name):
        if fname in exec_info["hashes"]:
            subprocess.run(["scp", "-q", f"{REMOTE_HOST}:{remote_dir}/{fname}", str(local_dir / fname)], check=True, timeout=240)
            assert sha256_file(local_dir / fname) == exec_info["hashes"][fname], f"Download SHA mismatch for {fname}"

    res = json.loads((local_dir / "result.json").read_text(encoding="utf-8-sig"))
    res["source_size_mb"] = round(src_path.stat().st_size / (1024 * 1024), 4)
    res["local_scdocx"] = (local_dir / scdocx_name).relative_to(WS_ROOT).as_posix()
    res["scdocx_size_mb"] = round((local_dir / scdocx_name).stat().st_size / (1024 * 1024), 4) if (local_dir / scdocx_name).is_file() else None
    res["kmou_peak_ram_mb"] = exec_info["peak_ram_mb"]
    if kind == "brep_fluid":
        assert res.get("gate") == "PASS_SPACECLAIM", f"SpaceClaim failed for {cid}: {res}"
    print(
        f"[KMOU-SC] {cid}: {res['gate']} | import={res.get('import_seconds')}s | wall={res['wall_seconds']}s | "
        f"src={res['source_size_mb']}MB -> scdocx={res['scdocx_size_mb']}MB | faces={res.get('body_metrics', {}).get('faces')} | RAM={res['kmou_peak_ram_mb']}MB",
        flush=True,
    )
    return res


def main() -> None:
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    existing = {}
    if only and QA_JSON.exists():
        existing = {r["case_id"]: r for r in json.loads(QA_JSON.read_text(encoding="utf-8")).get("cases", [])}

    for cid, kind, is_mono, src in CASES:
        if only and cid not in only:
            continue
        existing[cid] = run_case(cid, kind, is_mono, src)
        ordered = [existing[c[0]] for c in CASES if c[0] in existing]
        QA_JSON.parent.mkdir(parents=True, exist_ok=True)
        QA_JSON.write_text(json.dumps({"cases": ordered}, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_rows = [
            {
                "case_id": r["case_id"],
                "kind": r["kind"],
                "gate": r.get("gate"),
                "source_size_mb": r.get("source_size_mb"),
                "scdocx_size_mb": r.get("scdocx_size_mb"),
                "import_seconds": r.get("import_seconds"),
                "wall_seconds": r.get("wall_seconds"),
                "faces": (r.get("body_metrics") or {}).get("faces"),
                "closed": (r.get("body_metrics") or {}).get("closed"),
                "manifold": (r.get("body_metrics") or {}).get("manifold"),
                "kmou_peak_ram_mb": r.get("kmou_peak_ram_mb"),
            }
            for r in ordered
        ]
        with QA_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"Saved KMOU SpaceClaim benchmark -> {QA_JSON}", flush=True)


if __name__ == "__main__":
    main()
