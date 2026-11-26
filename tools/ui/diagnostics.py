"""Collect local hardware initialization failures without starting UI or streams."""

import json
import os
from pathlib import Path
import subprocess

from h264 import ffmpeg_path, probe_encoder


def video_controllers():
    if os.name != "nt":
        return {"error": "Windows GPU inventory is unavailable on this platform"}
    powershell = (
        Path(os.environ.get("SystemRoot", "C:/Windows"))
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    try:
        result = subprocess.run(
            [
                str(powershell), "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); "
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, DriverVersion, Status | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return {"error": result.stderr.decode("utf8", errors="replace")[-2000:]}
        return json.loads(result.stdout.decode("utf-8-sig"))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def diagnose(config_path, report):
    # Reading diagnostics must not migrate or save the user's parameter file.
    result = {"video_controllers": video_controllers()}
    try:
        path = config_path if config_path.is_file() else Path(__file__).parent / "ui_config.json"
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        config.setdefault("h264_ffmpeg", "auto")
        config.setdefault("h264_qp", 28)
        result["ffmpeg"] = ffmpeg_path(config["h264_ffmpeg"])
        result["encoders"] = {
            encoder: probe_encoder(config, encoder)
            for encoder in ("h264_qsv", "h264_nvenc", "h264_amf")
        }
        result["hardware_encoder_available"] = any(
            value["available"] for value in result["encoders"].values()
        )
    except (OSError, TypeError, ValueError) as error:
        result["error"] = str(error)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return 0 if "error" not in result else 1
