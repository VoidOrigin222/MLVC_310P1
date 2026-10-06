"""Build a Windows x64 one-file UI, with FFmpeg and its DLLs embedded."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/ui-exe/dist")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Build this executable on Windows x64")
    if importlib.util.find_spec("PyInstaller") is None:
        parser.error("Install the build dependencies: python -m pip install -r requirements-build.txt")
    ffmpeg = (args.ffmpeg or Path(shutil.which("ffmpeg") or "ffmpeg.exe")).resolve()
    if not ffmpeg.is_file():
        parser.error("FFmpeg not found; pass --ffmpeg path/to/ffmpeg.exe")
    output = args.output.resolve()
    exe = output / "SemanticVideoUI.exe"
    if exe.exists():
        parser.error("Output already exists; choose a new --output directory")
    work = REPO / "artifacts/ui-exe/build"
    spec = REPO / "artifacts/ui-exe/spec"
    command = [sys.executable, "-m", "PyInstaller", "--onefile", "--windowed", "--noupx",
               "--name", "SemanticVideoUI", "--distpath", str(output),
               "--workpath", str(work), "--specpath", str(spec)]
    environment = os.environ.copy()
    conda_bin = Path(sys.base_prefix) / "Library/bin"
    if conda_bin.is_dir():
        environment["PATH"] = str(conda_bin) + os.pathsep + environment.get("PATH", "")
        # A venv built from Conda does not expose conda-meta to PyInstaller.
        # Add Tk's native libraries explicitly for the launcher window.
        for library in ("tcl86t.dll", "tk86t.dll"):
            file = conda_bin / library
            if file.is_file():
                command += ["--add-binary", str(file) + ";."]
    for filename in ("index.html", "styles.css", "app.js"):
        command += ["--add-data", str(ROOT / filename) + ";."]
    dependencies = [ffmpeg] + sorted(ffmpeg.parent.glob("*.dll"))
    for file in dependencies:
        # Keep all FFmpeg binaries together at the bundle root. PyInstaller
        # can then deduplicate the same DLLs found during dependency analysis.
        command += ["--add-binary", str(file) + ";."]
    license_file = ffmpeg.parent.parent / "LICENSE"
    if license_file.is_file():
        command += ["--add-data", str(license_file) + ";licenses/ffmpeg"]
    command += [str(ROOT / "launch_ui.py")]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    (output / "SemanticVideoUI.exe.sha256").write_text(digest + "  SemanticVideoUI.exe\n", encoding="utf-8")
    metadata = {"executable": exe.name, "sha256": digest, "bytes": exe.stat().st_size,
                "python": sys.version, "ffmpeg_files": [p.name for p in dependencies],
                "ffmpeg_version": subprocess.check_output([str(ffmpeg), "-version"], text=True),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "source_files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("web_ui.py", "launch_ui.py", "index.html", "app.js", "styles.css")}}
    (output / "BUILD_INFO.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(exe))


if __name__ == "__main__":
    main()
