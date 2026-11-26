"""Desktop launcher with an explicit lifetime and an external parameter file."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
import webbrowser
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from server import APP_NAME, APP_VERSION, ConfigStore, Monitor, ROOT, make_handler, validate_config
from h264 import ffmpeg_path
from diagnostics import diagnose


def self_test(report):
    result = {"passed": False}
    try:
        ffmpeg = ffmpeg_path()
        if getattr(sys, "frozen", False):
            assert Path(ffmpeg).is_file(), "Bundled FFmpeg missing"
        subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            check=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result["ffmpeg"] = "passed"
        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads((ROOT / "ui_config.json").read_text(encoding="utf8"))
            cfg["h264_auto_start"] = False
            cfg = validate_config(cfg)
            sockets = [
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in cfg["channels"]
            ]
            for channel, stream in zip(cfg["channels"], sockets):
                stream.bind(("127.0.0.1", 0))
                channel["stats_port"] = stream.getsockname()[1]
            for stream in sockets:
                stream.close()
            path = Path(directory) / "parameters.json"
            path.write_text(json.dumps(cfg), encoding="utf8")
            monitor = Monitor(ConfigStore(path))
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(monitor))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            address = f"http://127.0.0.1:{server.server_port}"
            try:
                for route, marker in (
                    ("/", b"channel-grid"),
                    ("/app.js", b"RTCPeerConnection"),
                    ("/styles.css", b"channel-grid"),
                    ("/api/health", APP_NAME.encode()),
                ):
                    with urlopen(address + route, timeout=5) as response:
                        assert marker in response.read(), route
                with urlopen(address + "/api/config", timeout=5) as response:
                    assert json.load(response) == cfg
                result.update(
                    passed=True, assets="passed", config="passed", http="passed"
                )
            finally:
                server.shutdown()
                server.server_close()
                worker.join(2)
                monitor.close()
    except Exception:
        result["error"] = traceback.format_exc()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="多路语义压缩视频监控")
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else ROOT
    parser.add_argument("--config", type=Path, default=base / "ui_config.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--self-test", type=Path)
    parser.add_argument("--diagnostics", type=Path, help="保存本机显卡和硬件编码诊断后退出")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto-close", type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("端口必须为 0–65535")
    if args.self_test:
        return self_test(args.self_test)
    if args.diagnostics:
        return diagnose(args.config, args.diagnostics)
    import tkinter as tk
    from tkinter import messagebox, ttk

    window = tk.Tk()
    window.withdraw()
    try:
        store = ConfigStore(args.config)
        # Reuse only an instance serving the same file. Reload that file so a
        # second double-click never silently keeps parameters edited on disk.
        try:
            address = f"http://127.0.0.1:{args.port}"
            with urlopen(address + "/api/health", timeout=0.7) as response:
                health = json.load(response)
            same_file = os.path.normcase(str(store.path)) == os.path.normcase(
                health.get("config_path", "")
            )
            if health.get("app") == APP_NAME and same_file:
                request = Request(
                    address + "/api/config",
                    data=json.dumps(store.snapshot()).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=3):
                    pass
                if not args.no_browser:
                    webbrowser.open(address)
                window.destroy()
                return 0
        except (OSError, ValueError):
            pass
        monitor = Monitor(store)
        server = None
        for port in range(args.port, min(args.port + 10, 65536)):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(monitor))
                break
            except OSError:
                continue
        if server is None:
            monitor.close()
            raise OSError("页面端口已占用，请使用 --port 指定其他端口")
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
    except (OSError, ValueError) as error:
        messagebox.showerror("启动失败", str(error), parent=window)
        window.destroy()
        return 1
    address = f"http://127.0.0.1:{server.server_port}"
    window.title(f"ulbvc 视频监控 V{APP_VERSION}")
    window.geometry("420x210")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=23)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="视频监控", font=("Microsoft YaHei UI", 15)).pack(
        anchor="w"
    )
    ttk.Label(frame, text=address, padding=(0, 12)).pack(anchor="w")
    ttk.Label(frame, text="关闭启动器会停止页面，板端程序独立运行。").pack(
        anchor="w", pady=(0, 18)
    )

    def close():
        window.withdraw()
        server.shutdown()
        server.server_close()
        monitor.close()
        worker.join(2)
        window.destroy()

    actions = ttk.Frame(frame)
    actions.pack(fill="x")
    ttk.Button(actions, text="打开页面", command=lambda: webbrowser.open(address)).pack(
        side="left"
    )
    ttk.Button(actions, text="停止并退出", command=close).pack(side="right")
    window.protocol("WM_DELETE_WINDOW", close)
    window.deiconify()
    if not args.no_browser:
        window.after(100, lambda: webbrowser.open(address))
    if args.auto_close:
        window.after(int(args.auto_close * 1000), close)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
