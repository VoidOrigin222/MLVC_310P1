"""Build the same portable V0.5 monitor for Thor and Ascend."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/ui-exe/dist")
    parser.add_argument("--ffmpeg", type=Path, help="指定随程序打包的 FFmpeg")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("需要在 Windows 上构建")
    ffmpeg = (args.ffmpeg or Path(shutil.which("ffmpeg") or "ffmpeg.exe")).resolve()
    if not ffmpeg.is_file():
        parser.error("FFmpeg 不存在，无法打包 H.264 对照功能")
    output = args.output.resolve()
    if (output / "ULBVCMonitor.exe").exists():
        parser.error("输出目录已有程序，请选择新的 --output 目录")
    work = REPO / "artifacts/ui-exe/build"
    spec = REPO / "artifacts/ui-exe/spec"
    env = os.environ.copy()
    conda = Path(sys.base_prefix) / "Library/bin"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--noupx",
        "--name",
        "ULBVCMonitor",
        "--distpath",
        str(output),
        "--workpath",
        str(work),
        "--specpath",
        str(spec),
    ]
    if conda.is_dir():
        env["PATH"] = str(conda) + os.pathsep + env.get("PATH", "")
        for name in ("tcl86t.dll", "tk86t.dll"):
            if (conda / name).exists():
                command += ["--add-binary", str(conda / name) + ";."]
    for name in ("index.html", "app.js", "styles.css", "ui_config.json", "ui_config.310p.json", "ui_config.thor.json"):
        command += ["--add-data", str(ROOT / name) + ";."]
    dependencies = [ffmpeg] + sorted(ffmpeg.parent.glob("*.dll"))
    for file in dependencies:
        command += ["--add-binary", str(file) + ";."]
    license_file = ffmpeg.parent.parent / "LICENSE"
    if license_file.is_file():
        command += ["--add-data", str(license_file) + ";licenses/ffmpeg"]
    command.append(str(ROOT / "launch_ui.py"))
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    for name in ("ui_config.json", "ui_config.310p.json", "ui_config.thor.json"):
        shutil.copy2(ROOT / name, output / name)
    shutil.copy2(ROOT / "README.md", output / "README.md")
    exe = output / "ULBVCMonitor.exe"
    metadata = {
        "version": "0.5",
        "platforms": ["Thor", "Ascend 310P"],
        "executable": exe.name,
        "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "bytes": exe.stat().st_size,
        "ffmpeg_files": [p.name for p in dependencies],
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.iterdir()
            if p.suffix in (".py", ".js", ".css", ".html", ".json")
        },
    }
    (output / "BUILD_INFO.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf8"
    )
    (output / "ULBVCMonitor.exe.sha256").write_text(metadata["sha256"] + "  ULBVCMonitor.exe\n", encoding="utf-8")
    print(exe)


if __name__ == "__main__":
    main()
