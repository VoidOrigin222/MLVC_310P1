"""Windows GUI entry point for the self-contained UI executable."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import traceback
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def existing_ui(port: int) -> bool:
    """Recognize this UI, including the previous Python-only launcher."""
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/config", timeout=1) as response:
            config = json.load(response)
        return isinstance(config, dict) and all(
            key in config for key in ("original", "h264", "mlvc", "webrtc", "h264_qp")
        )
    except (OSError, ValueError, URLError):
        return False


def self_test(report: Path) -> int:
    """Exercise the frozen resources, FFmpeg/libx264, HTTP API and shutdown."""
    os.environ["MLVC_UI_STATS_PORT"] = "0"
    import web_ui

    result = {"passed": False}
    server = None
    thread = None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        for route, expected in (("/", b"video-grid"), ("/app.js", b"DelayedVideo"),
                                ("/styles.css", b"video-grid"), ("/api/health", b"SemanticVideoUI")):
            with urlopen(address + route, timeout=5) as response:
                assert response.status == 200 and expected in response.read(), route
        ffmpeg = web_ui.controller.cfg["ffmpeg"]
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                   "-i", "color=c=black:s=64x64:r=5", "-frames:v", "2",
                   "-c:v", "libx264", "-qp", "40", "-f", "null", "-"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert run.returncode == 0, run.stderr
        result.update(passed=True, assets="passed", http_api="passed", ffmpeg_libx264="passed",
                      ffmpeg=ffmpeg, stats_port=web_ui.controller.cfg["mlvc_stats_port"])
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        web_ui.controller.close()
        if thread is not None:
            thread.join(timeout=3)
            result["shutdown"] = not thread.is_alive()
            result["passed"] = result["passed"] and result["shutdown"]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="语义压缩三路视频控制台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--self-test", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto-close", type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return self_test(args.self_test)
    if args.port and existing_ui(args.port):
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        return 0

    import tkinter as tk
    from tkinter import messagebox, ttk
    import web_ui

    window = tk.Tk()
    window.title("语义压缩视频控制台")
    window.geometry("390x190")
    window.resizable(False, False)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), web_ui.Handler)
    except OSError as error:
        window.withdraw()
        messagebox.showerror("启动失败", f"无法启动页面服务，请检查端口 {args.port}。\n{error}", parent=window)
        web_ui.controller.close()
        window.destroy()
        return 1

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    frame = ttk.Frame(window, padding=22)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="三路视频控制台", font=("Microsoft YaHei UI", 14)).pack(anchor="w")
    ttk.Label(frame, text=address, padding=(0, 8)).pack(anchor="w")
    ttk.Label(frame, text="关闭此窗口会停止 UI 服务及其 H.264 推流。").pack(anchor="w", pady=(0, 14))

    def close() -> None:
        window.withdraw()
        server.shutdown()
        web_ui.controller.close()
        server.server_close()
        thread.join(timeout=3)
        window.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="打开页面", command=lambda: webbrowser.open(address)).pack(side="left")
    ttk.Button(buttons, text="停止并退出", command=close).pack(side="right")
    window.protocol("WM_DELETE_WINDOW", close)
    if not args.no_browser:
        window.after(200, lambda: webbrowser.open(address))
    if args.auto_close > 0:
        window.after(int(args.auto_close * 1000), close)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
