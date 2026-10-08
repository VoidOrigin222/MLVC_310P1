"""Integration check of the built one-file UI, without machine Python/FFmpeg PATH."""
import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


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
    environment.pop("MLVC_UI_CONFIG", None)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    processes = []
    with tempfile.TemporaryDirectory(prefix="semantic-ui-exe-test-") as work:
        work = Path(work).resolve()
        result = {"executable": str(exe), "path_contains_python_or_ffmpeg": False}
        # Exercise the default path beside a relocated EXE, from a different cwd.
        runtime = work / 'portable'
        runtime.mkdir()
        executable = runtime / exe.name
        shutil.copyfile(exe, executable)
        config_file = runtime / 'ui_config.json'
        try:
            self_report = work / "self-test.json"
            process = subprocess.Popen([str(executable), "--self-test", str(self_report)],
                                       env=environment, cwd=work, startupinfo=startup)
            processes.append(process)
            assert process.wait(timeout=90) == 0, "Frozen self-test failed"
            result["frozen_self_test"] = json.loads(self_report.read_text(encoding="utf-8"))
            assert result["frozen_self_test"]["passed"]
            assert Path(result["frozen_self_test"]["ffmpeg"]).is_absolute(), "FFmpeg must come from the bundle"
            assert Path(result['frozen_self_test']['config_file']) == config_file
            defaults = json.loads(config_file.read_text(encoding='utf-8'))
            assert defaults['semantic'] == 'rtsp://127.0.0.1:8554/ulbvc'
            assert defaults['ffmpeg'] == 'auto'
            # A file edited before startup must supply the actual runtime settings.
            edited = {**defaults, 'h264_qp': 37, 'transport': 'udp',
                      'semantic': 'rtsp://192.168.10.20:8554/ulbvc?token=sample'}
            config_file.write_text(json.dumps(edited), encoding='utf-8')
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            process = subprocess.Popen([str(executable), "--port", str(port), "--no-browser", "--auto-close", "15"],
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
            duplicate = subprocess.Popen([str(executable), "--port", str(port), "--no-browser", "--auto-close", "5"],
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
                'mlvc': edited['semantic'],
                'webrtc': 'http://127.0.0.1:8889',
            }
            for key, url in expected_urls.items():
                assert config[key] == url, (key, config[key])
            assert config['h264_qp'] == 37 and config['transport'] == 'udp'
            assert config['ffmpeg'] == 'auto'
            with urlopen(base + '/', timeout=2) as response:
                page = response.read().decode('utf-8')
            assert 'value="rtsp://127.0.0.1:8554/ulbvc"' in page
            result['loaded_config_file'] = True
            saved = {**config, 'h264_qp': 38, 'mlvc_stats_port': 40341,
                     'mlvc': 'rtsp://127.0.0.1:8554/ulbvc-updated'}
            request = Request(base + '/api/config', data=json.dumps(saved).encode('utf-8'),
                              headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=3) as response:
                assert response.status == 200
            disk = json.loads(config_file.read_text(encoding='utf-8'))
            assert disk['semantic'] == saved['mlvc'] and disk['h264_qp'] == 38
            assert disk['stats_port'] == 40341 and disk['ffmpeg'] == 'auto'
            result['api_saved_config_file'] = True
            result["single_instance_reuse"] = True
            assert process.wait(timeout=30) == 0, "GUI shutdown failed"
            try:
                urlopen(address, timeout=1)
            except (OSError, URLError):
                result["service_stopped_on_window_close"] = True
            else:
                raise AssertionError("Service remained after closing its window")
            process = subprocess.Popen([str(executable), "--port", str(port), "--no-browser", "--auto-close", "5"],
                                       env=environment, cwd=work, startupinfo=startup)
            processes.append(process)
            deadline = time.monotonic() + 45
            while True:
                try:
                    with urlopen(base + '/api/config', timeout=1) as response:
                        restored = json.load(response)
                    break
                except (OSError, URLError):
                    assert process.poll() is None, 'Restart exited before becoming ready'
                    assert time.monotonic() < deadline, 'Restart did not become ready'
                    time.sleep(0.15)
            for key in ('original', 'h264', 'mlvc', 'webrtc', 'transport', 'h264_qp', 'ffmpeg', 'mlvc_stats_port'):
                assert restored[key] == saved[key], (key, restored[key], saved[key])
            assert process.wait(timeout=30) == 0
            result['settings_survive_restart'] = True
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
