"""Clean-room single-launch CPU-only Fluent session manager for Windows."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

from ansys.fluent.core.launcher.launch_options import UIMode
from ansys.fluent.core.launcher.standalone_launcher import StandaloneLauncher


def launch_once(pf, directory, mode: str, processors: int):
    """Launch a single owned CPU-only Fluent process without retry loops."""
    directory = Path(directory)
    awp_root = Path(os.environ.get("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\v261"))
    fluent_exe = os.environ.get("TPMS2CFD_FLUENT_EXE", str(awp_root / "fluent" / "ntbin" / "win64" / "fluent.exe"))
    launcher = StandaloneLauncher(
        product_version=261,
        dimension=3,
        precision="double",
        processor_count=processors,
        gpu=False,
        mode=mode,
        fluent_path=fluent_exe,
        ui_mode=UIMode.NO_GUI_OR_GRAPHICS,
        start_watchdog=False,
        cleanup_on_exit=True,
        start_transcript=False,
        cwd=str(directory),
        additional_arguments="-wait" + (f" -tm{processors}" if mode == "meshing" else ""),
        dry_run=True,
    )
    command, server_file = launcher()
    argument_count = ctypes.c_int()
    parse = ctypes.windll.shell32.CommandLineToArgvW
    parse.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(ctypes.c_wchar_p)
    parsed = parse(command, ctypes.byref(argument_count))
    if not parsed:
        raise ctypes.WinError()
    try:
        arguments = [parsed[index] for index in range(argument_count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(parsed)

    environment = os.environ.copy()
    environment["REMOTING_THROW_LAST_TUI_ERROR"] = "1"
    environment["REMOTING_THROW_LAST_SETTINGS_ERROR"] = "1"
    previous_mtime = Path(server_file).stat().st_mtime_ns
    start = time.monotonic()
    with (directory / "fluent_startup.log").open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            arguments,
            cwd=str(directory),
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
        )
        record = {
            "argv": arguments,
            "pid": process.pid,
            "gpu_requested": False,
            "processor_count": processors,
            "launch_count": 1,
            "shell": False,
        }
        (directory / "fluent_launch.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        try:
            while Path(server_file).stat().st_mtime_ns <= previous_mtime:
                if process.poll() is not None:
                    record["exit_code"] = process.returncode
                    (directory / "fluent_launch_exit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    raise RuntimeError(
                        f"Fluent launcher exited {process.returncode} before creating server info; read fluent_startup.log"
                    )
                if time.monotonic() - start > 120:
                    raise TimeoutError("Single Fluent launch did not create server info in 120 seconds")
                time.sleep(1)
            time.sleep(1)
            session = pf.connect_to_fluent(
                server_info_file_name=server_file,
                cleanup_on_exit=True,
                start_watchdog=False,
                start_transcript=False,
            )
            idle_minutes = session.preferences.General.IdleTimeout()
            session.application_runtime.set_idle_timeout(idle_minutes * 60)
            session._owned_start_process = process
            Path(server_file).unlink(missing_ok=True)
            return session
        except Exception:
            if process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=20,
                    check=False,
                )
            Path(server_file).unlink(missing_ok=True)
            raise


def close_once(session) -> int:
    """Close the PyFluent session and reap its owned launcher process."""
    process = session._owned_start_process
    session.exit(timeout=30, wait=30)
    return process.wait(timeout=10)
