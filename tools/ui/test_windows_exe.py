"""Integration check of the built one-file UI, without machine Python/FFmpeg PATH."""
import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    exe = args.executable.resolve()
    environment = os.environ.copy()
    systemroot = environment.get("SystemRoot", "C:\\Windows")
    environment["PATH"] = systemroot + "\\System32;" + systemroot
    environment["MLVC_UI_STATS_PORT"] = "0"
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    processes = []
    with tempfile.TemporaryDirectory(prefix="semantic-ui-exe-test-") as work:
        work = Path(work)
        result = {"executable": str(exe), "path_contains_python_or_ffmpeg": False}
        try:
            self_report = work / "self-test.json"
            process = subprocess.Popen([str(exe), "--self-test", str(self_report)],
                                       env=environment, cwd=work, startupinfo=startup)
            processes.append(process)
            assert process.wait(timeout=90) == 0, "Frozen self-test failed"
            result["frozen_self_test"] = json.loads(self_report.read_text(encoding="utf-8"))
            assert result["frozen_self_test"]["passed"]
            assert Path(result["frozen_self_test"]["ffmpeg"]).is_absolute(), "FFmpeg must come from the bundle"
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            process = subprocess.Popen([str(exe), "--port", str(port), "--no-browser", "--auto-close", "15"],
                                       env=environment, cwd=work, startupinfo=startup)
            processes.append(process)
            address = f"http://127.0.0.1:{port}/api/health"
            deadline = time.monotonic() + 45
            while True:
                try:
                    with urlopen(address, timeout=1) as response:
                        assert json.load(response)["application"] == "SemanticVideoUI"
                    break
                except (OSError, URLError):
                    assert process.poll() is None, "Launcher exited before becoming ready"
                    assert time.monotonic() < deadline, "Launcher did not become ready"
                    time.sleep(0.15)
            duplicate = subprocess.Popen([str(exe), "--port", str(port), "--no-browser", "--auto-close", "5"],
                                         env=environment, cwd=work, startupinfo=startup)
            processes.append(duplicate)
            assert duplicate.wait(timeout=30) == 0
            assert process.poll() is None, "Existing launcher must survive the second launch"
            with urlopen(address, timeout=2) as response:
                assert response.status == 200
            base = address.removesuffix('/api/health')
            with urlopen(base + '/api/config', timeout=2) as response:
                config = json.load(response)
            expected_urls = {
                'original': 'rtsp://127.0.0.1:8554/camera-original',
                'h264': 'rtsp://127.0.0.1:8554/camera-h264',
                'mlvc': 'rtsp://127.0.0.1:8554/mlvc',
                'webrtc': 'http://127.0.0.1:8889',
            }
            for key, url in expected_urls.items():
                assert config[key] == url, (key, config[key])
            with urlopen(base + '/', timeout=2) as response:
                page = response.read().decode('utf-8')
            assert 'value="rtsp://127.0.0.1:8554/mlvc"' in page
            result['default_urls'] = expected_urls
            result["single_instance_reuse"] = True
            assert process.wait(timeout=30) == 0, "GUI shutdown failed"
            try:
                urlopen(address, timeout=1)
            except (OSError, URLError):
                result["service_stopped_on_window_close"] = True
            else:
                raise AssertionError("Service remained after closing its window")
            result["gui_startup"] = True
            result["passed"] = True
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            for process in processes:
                if process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
