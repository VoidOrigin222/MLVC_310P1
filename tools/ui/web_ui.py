# -*- coding: utf-8 -*-
"""Browser video wall for the MLVC comparison pipeline.

The page uses MediaMTX WebRTC/WHEP for all three videos.  RTSP is kept on
MediaMTX and is never sent to the browser.  The local H.264 comparison feed is
created by ffmpeg and published back to MediaMTX as an RTSP path.
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen



METRIC_WINDOW_SECONDS = 12.0
BITRATE_ADJUST_SECONDS = 30.0


UI_ROOT = Path(__file__).resolve().parent
LOG_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "SemanticVideoUI" / "logs"
    if getattr(sys, "frozen", False) else UI_ROOT
)


def default_ffmpeg() -> str:
    for bundled in (UI_ROOT / "ffmpeg.exe", UI_ROOT / "ffmpeg" / "ffmpeg.exe"):
        if bundled.is_file():
            return str(bundled)
    return "ffmpeg"


class Controller:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.cfg = {
            "original": "rtsp://127.0.0.1:8554/camera-original",
            "h264": "rtsp://127.0.0.1:8554/camera-h264",
            "mlvc": "rtsp://127.0.0.1:8554/ulbvc",
            "webrtc": "http://127.0.0.1:8889",
            "transport": "tcp",
            "ffmpeg": default_ffmpeg(),
            "h264_kbps": 8000,
            "h264_qp": 40,
            "mlvc_kbps": 2000,
            "h264_bpp": 0.129,
            "mlvc_bpp": 0.032,
            "mlvc_stats_port": int(os.environ.get("MLVC_UI_STATS_PORT", "39341")),
            "mode": "same_quality",
            "fps": 30,
        }
        self.h264_encoder: subprocess.Popen | None = None
        self.ffmpeg_log = None
        self.last_error = ""
        self._h264_samples: deque[tuple[float, int]] = deque()
        self._mlvc_stats_socket: socket.socket | None = None
        self._mlvc_stats_bound_port: int | None = None
        self._mlvc_stats_error = ""
        self._mlvc_stats_latest: dict[str, float | int] | None = None
        self._mlvc_media_samples: deque[tuple[float, int]] = deque()
        self._mlvc_wire_samples: deque[tuple[float, int]] = deque()
        self._mlvc_fps = float(self.cfg.get("fps", 30))
        self._mlvc_metrics_ready = False
        self._mlvc_metrics = {"kbps": 0.0, "bpp": 0.0, "kbs": 0.0, "ratio": 0.0}
        self._h264_requested = False
        self._h264_rate_kbps: int | None = None
        self._h264_target_kbps: float | None = None
        self._h264_last_adjustment = 0.0
        self._h264_match_state = "stopped"
        self._h264_qp: int | None = None
        self._control_stop = threading.Event()
        self._control_thread: threading.Thread | None = None
        self._ensure_mlvc_stats_socket()

    @staticmethod
    def _record_counter_sample(history: deque[tuple[float, int]], now: float, value: int) -> bool:
        """Append a cumulative byte counter and retain a sliding-window base."""
        reset = bool(history and value < history[-1][1])
        if reset:
            history.clear()
        history.append((now, value))
        cutoff = now - METRIC_WINDOW_SECONDS
        # Keep one sample before the window so the first in-window sample has
        # a real baseline and the rate does not jump when the window moves.
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()
        return reset

    @staticmethod
    def _window_rate(history: deque[tuple[float, int]], now: float) -> float:
        if len(history) < 2:
            return 0.0
        latest_time, latest_value = history[-1]
        if now - latest_time > METRIC_WINDOW_SECONDS:
            return 0.0
        base_time, base_value = history[0]
        elapsed = latest_time - base_time
        if elapsed <= 0.0 or latest_value < base_value:
            return 0.0
        return (latest_value - base_value) / elapsed

    def _refresh_mlvc_metrics(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        media_bytes_per_second = self._window_rate(self._mlvc_media_samples, now)
        wire_bytes_per_second = self._window_rate(self._mlvc_wire_samples, now)
        if media_bytes_per_second <= 0.0 and wire_bytes_per_second <= 0.0:
            latest = self._mlvc_stats_latest
            if latest is not None and now - float(latest["received_at"]) > METRIC_WINDOW_SECONDS:
                self._mlvc_metrics = {"kbps": 0.0, "bpp": 0.0, "kbs": 0.0, "ratio": 0.0}
                self._mlvc_metrics_ready = False
            return
        pixels = 1920.0 * 1080.0 * max(1.0, self._mlvc_fps)
        media_kbps = media_bytes_per_second * 8.0 / 1000.0
        bpp = media_kbps * 1000.0 / pixels if pixels else 0.0
        self._mlvc_metrics = {
            "kbps": media_kbps,
            "bpp": bpp,
            "kbs": wire_bytes_per_second / 1024.0,
            "ratio": 24.0 / bpp if bpp else 0.0,
        }
        self._mlvc_metrics_ready = True

    def _ensure_mlvc_stats_socket(self) -> None:
        port = int(self.cfg.get("mlvc_stats_port", 39341) or 0)
        if self._mlvc_stats_socket is not None and self._mlvc_stats_bound_port == port:
            return
        if self._mlvc_stats_socket is not None:
            self._mlvc_stats_socket.close()
            self._mlvc_stats_socket = None
            self._mlvc_stats_bound_port = None
        self._mlvc_stats_error = ""
        if port <= 0:
            return
        try:
            stats_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            stats_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            stats_socket.bind(("0.0.0.0", port))
            stats_socket.setblocking(False)
            self._mlvc_stats_socket = stats_socket
            self._mlvc_stats_bound_port = port
        except OSError as exc:
            self._mlvc_stats_error = str(exc)
            try:
                stats_socket.close()
            except UnboundLocalError:
                pass

    def _drain_mlvc_stats(self) -> None:
        self._ensure_mlvc_stats_socket()
        stats_socket = self._mlvc_stats_socket
        if stats_socket is None:
            return
        while True:
            try:
                raw, _address = stats_socket.recvfrom(8192)
            except BlockingIOError:
                return
            except OSError:
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("kind") != "mlvc_rtp_stats":
                    continue
                media_bytes = int(payload["media_unit_bytes"])
                wire_bytes = int(payload.get("wire_bytes", media_bytes))
                stream_fps = max(1.0, float(payload.get("fps", self.cfg.get("fps", 30))))
                now = time.monotonic()
            except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            media_reset = self._record_counter_sample(self._mlvc_media_samples, now, media_bytes)
            wire_reset = self._record_counter_sample(self._mlvc_wire_samples, now, wire_bytes)
            if media_reset or wire_reset:
                # Encoder restart: establish a new sliding-window base.
                self._mlvc_metrics = {"kbps": 0.0, "bpp": 0.0, "kbs": 0.0, "ratio": 0.0}
                self._mlvc_metrics_ready = False
            self._mlvc_fps = stream_fps
            self._mlvc_stats_latest = {
                "frames": int(payload.get("frames", -1)),
                "media_unit_bytes": media_bytes,
                "wire_bytes": wire_bytes,
                "fps": stream_fps,
                "received_at": now,
            }
            self._refresh_mlvc_metrics(now)

    def stop(self) -> None:
        with self.lock:
            self._h264_requested = False
            self._stop_encoder()
            self._h264_match_state = "stopped"
            self._h264_target_kbps = None

    def _stop_encoder(self) -> None:
        process = self.h264_encoder
        self.h264_encoder = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self.ffmpeg_log is not None:
            self.ffmpeg_log.close()
            self.ffmpeg_log = None

    def close(self) -> None:
        self._control_stop.set()
        self.stop()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2)
        if self._mlvc_stats_socket is not None:
            self._mlvc_stats_socket.close()
            self._mlvc_stats_socket = None
            self._mlvc_stats_bound_port = None

    def start(self, cfg: dict) -> None:
        with self.lock:
            self._start_locked(cfg)

    def _validated_config(self, cfg: dict) -> dict:
        updated = {**self.cfg, **{k: v for k, v in cfg.items() if k in self.cfg}}
        if updated["mode"] not in ("same_bandwidth", "same_quality"):
            raise ValueError("Unsupported comparison mode")
        qp = updated["h264_qp"]
        if isinstance(qp, bool) or not isinstance(qp, int) or not 0 <= qp <= 51:
            raise ValueError("H.264 QP must be an integer from 0 to 51")
        return updated

    def configure(self, cfg: dict) -> None:
        self.cfg.update(self._validated_config(cfg))
        self._ensure_mlvc_stats_socket()

    def _start_locked(self, cfg: dict) -> None:
        self._validated_config(cfg)
        self.stop()
        self.configure(cfg)
        if not self.cfg["original"]:
            raise ValueError("原图 RTSP 地址为空")
        if not self.cfg["h264"]:
            raise ValueError("H.264 输出 RTSP 地址为空")
        self.last_error = ""
        self._h264_qp = None
        self._h264_requested = True
        self._h264_rate_kbps = None
        self._h264_last_adjustment = 0.0
        self._h264_match_state = "waiting"
        self._status_locked()
        if self._control_thread is None or not self._control_thread.is_alive():
            self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self._control_thread.start()

    def _control_loop(self) -> None:
        # Keep tracking even when the browser is closed. Serialize UDP draining,
        # counter samples, starts, and stops with the HTTP control requests.
        while not self._control_stop.wait(1.0):
            try:
                with self.lock:
                    if self._h264_requested:
                        self._status_locked()
            except (OSError, ValueError) as exc:
                with self.lock:
                    self.last_error = str(exc)

    def _bitrate_target(self, now: float) -> float | None:
        latest = self._mlvc_stats_latest
        samples = self._mlvc_media_samples
        if (latest is None or now - float(latest["received_at"]) > METRIC_WINDOW_SECONDS
                or len(samples) < 2 or samples[-1][0] - samples[0][0] < METRIC_WINDOW_SECONDS):
            return None
        # Use actual MLVC codec bytes; the RTP pacing ceiling is not a bitrate.
        # The comparison allows normal variation and H.264 packet overhead.
        target = self._window_rate(samples, now) * 8.0 / 1000.0
        return target if math.isfinite(target) and target > 0.0 else None

    def _sync_bitrate(self, now: float, source_ready: bool) -> None:
        if not self._h264_requested:
            return
        if self.cfg["mode"] == "same_quality":
            self._sync_quality(now, source_ready)
            return
        # A comparison run keeps its startup budget. Fluctuating MLVC GOP
        # sizes must not continually restart the H.264 preview.
        if self.h264_encoder is not None:
            self._h264_match_state = "running"
            return
        target = self._h264_target_kbps or self._bitrate_target(now)
        if target is None or not source_ready:
            self._h264_match_state = "waiting"
            return
        if (self._h264_rate_kbps is not None and
                now - self._h264_last_adjustment < BITRATE_ADJUST_SECONDS):
            return
        self._h264_target_kbps = target
        self._launch_h264(max(1, round(target)))

    def _sync_quality(self, now: float, source_ready: bool) -> None:
        if self.h264_encoder is not None:
            self._h264_match_state = "running"
            return
        if not source_ready:
            self._h264_match_state = "waiting"
            return
        if (self._h264_qp is not None and
                now - self._h264_last_adjustment < BITRATE_ADJUST_SECONDS):
            return
        self._launch_h264(qp=self._h264_qp if self._h264_qp is not None else self.cfg["h264_qp"])

    def _launch_h264(self, rate: int | None = None, *, qp: int | None = None) -> None:
        self._stop_encoder()
        self._h264_samples.clear()
        self._h264_rate_kbps = rate
        self._h264_qp = qp
        if rate is not None:
            self.cfg["h264_kbps"] = rate
        self._h264_last_adjustment = time.monotonic()
        self._h264_match_state = "settling"
        output = self.cfg["h264"]
        args = [
            self.cfg["ffmpeg"], "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", self.cfg["transport"], "-fflags", "nobuffer",
            # Probe enough frames to distinguish the 90 kHz RTP clock from FPS.
            "-flags", "low_delay", "-probesize", "5000000", "-analyzeduration", "1000000",
            "-i", self.cfg["original"], "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-threads", "2", "-tune", "zerolatency",
        ]
        if qp is not None:
            args += ["-qp", str(qp)]
        else:
            args += ["-b:v", f"{rate}k", "-minrate", f"{rate}k", "-maxrate", f"{rate}k",
                     "-bufsize", f"{rate * 2}k", "-x264-params", "nal-hrd=cbr:filler=1"]
        args += [
            "-g", "96", "-keyint_min", "96", "-sc_threshold", "0", "-bf", "0",
            # Preserve input timestamps without CFR duplication or frame dropping.
            "-fps_mode:v", "passthrough", "-enc_time_base:v", "demux",
            "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", output,
        ]
        try:
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            self.ffmpeg_log = open(LOG_ROOT / "ffmpeg_ui.log", "ab")
            self.h264_encoder = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=self.ffmpeg_log,
                creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                               | getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        except OSError as exc:
            if self.ffmpeg_log is not None:
                self.ffmpeg_log.close()
                self.ffmpeg_log = None
            self.last_error = str(exc)
            raise
        self.last_error = ""

    def status(self) -> dict:
        with self.lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        now = time.monotonic()
        self._drain_mlvc_stats()
        self._refresh_mlvc_metrics(now)
        fps = max(1.0, float(self.cfg.get("fps", 30)))
        pixels = 1920.0 * 1080.0 * fps
        original_kbs = pixels * 3.0 / 1024.0
        metrics = {"original": {"kbps": pixels * 24.0 / 1000.0, "bpp": 24.0, "kbs": original_kbs, "ratio": 1.0, "window_seconds": METRIC_WINDOW_SECONDS}}
        for key in ("h264",):
            kbps = max(0.0, float(self.cfg.get(key + "_kbps", 0)))
            bpp = kbps * 1000.0 / pixels if pixels else 0.0
            metrics[key] = {"kbps": kbps, "bpp": bpp, "kbs": kbps / 8.0, "ratio": 24.0 / bpp if bpp else 0.0, "window_seconds": METRIC_WINDOW_SECONDS}
        latest_stats = self._mlvc_stats_latest
        stats_age = None if latest_stats is None else max(0.0, now - float(latest_stats["received_at"]))
        mlvc_actual = self._mlvc_metrics_ready and stats_age is not None and stats_age <= METRIC_WINDOW_SECONDS * 2.0
        metrics["mlvc"] = dict(self._mlvc_metrics)
        metrics["mlvc"]["window_seconds"] = METRIC_WINDOW_SECONDS
        process = self.h264_encoder
        if process is not None and process.poll() is not None:
            self.last_error = "H.264 FFmpeg 已退出，请检查原图 RTSP 路径和 MediaMTX。"
            self.h264_encoder = None
        paths = None
        try:
            with urlopen("http://127.0.0.1:9997/v3/paths/list", timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            paths = set()
            for item in payload.get("items", []):
                name = item.get("name") or item.get("path")
                if isinstance(name, str):
                    if item.get("ready", False):
                        paths.add(name.strip("/"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        def published(url: str) -> bool:
            if paths is None:
                return False
            try:
                name = urlparse(url).path.strip("/")
            except (TypeError, ValueError):
                return False
            return name in paths
        visible_paths = sorted(paths) if paths is not None else []
        h264_path_name = urlparse(self.cfg["h264"]).path.strip("/")
        measured_kbs = 0.0
        h264_actual = False
        if paths is not None:
            for item in payload.get("items", []):
                name = item.get("name") or item.get("path")
                received = item.get("bytesReceived", item.get("inboundBytes"))
                if item.get("ready", False) and isinstance(name, str) and isinstance(received, (int, float)) and name.strip("/") == h264_path_name:
                    self._record_counter_sample(self._h264_samples, now, int(received))
                    measured_kbs = self._window_rate(self._h264_samples, now) / 1024.0
                    h264_actual = len(self._h264_samples) >= 2
        if not published(self.cfg["h264"]):
            self._h264_samples.clear()
        for key in ("h264",):
            if self._h264_samples:
                metrics[key]["kbs"] = max(0.0, measured_kbs)
                metrics[key]["kbps"] = max(0.0, measured_kbs * 1024.0 * 8.0 / 1000.0)
                metrics[key]["bpp"] = metrics[key]["kbps"] * 1000.0 / pixels if pixels else 0.0
                metrics[key]["ratio"] = 24.0 / metrics[key]["bpp"] if metrics[key]["bpp"] else 0.0
            if not h264_actual:
                metrics[key].update(kbps=0.0, bpp=0.0, kbs=0.0, ratio=0.0)
        self._sync_bitrate(now, published(self.cfg["original"]))
        # Starting a publisher invalidates its previous sliding window.
        if not self._h264_samples:
            h264_actual = False
            metrics["h264"].update(kbps=0.0, bpp=0.0, kbs=0.0, ratio=0.0)
        return {
            "feeds": {"original": published(self.cfg["original"]), "h264": published(self.cfg["h264"]), "mlvc": published(self.cfg["mlvc"])},
            "metrics": metrics,
            "error": self.last_error,
            "mlvc_actual": mlvc_actual,
            "h264_actual": h264_actual,
            "mlvc_stats_age": stats_age,
            "mlvc_stats_source": "encoder RTP media units",
            "mlvc_stats_error": self._mlvc_stats_error,
            "media_server": paths is not None,
            "paths": visible_paths,
            "metric_window_seconds": METRIC_WINDOW_SECONDS,
            "bitrate_match": {
                "state": self._h264_match_state,
                "target_kbps": self._h264_target_kbps,
                "encoder_kbps": self._h264_rate_kbps,
            },
            "mode": self.cfg["mode"],
            "quality_control": {"qp": self._h264_qp},
        }


controller = Controller()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def send_json(self, data: object, code: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_asset(self, filename: str, content_type: str) -> None:
        raw = (UI_ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_asset("index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self.send_asset("styles.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self.send_asset("app.js", "text/javascript; charset=utf-8")
        elif path == "/api/config":
            self.send_json(controller.cfg)
        elif path == "/api/health":
            self.send_json({"application": "SemanticVideoUI", "version": 1})
        elif path == "/api/status":
            self.send_json(controller.status())
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/config":
                with controller.lock:
                    controller.configure(body)
                self.send_json({"message": "配置已更新"})
            elif path == "/api/start":
                controller.start(body)
                self.send_json({"message": "H.264 推流已启动"})
            elif path == "/api/stop":
                controller.stop()
                self.send_json({"message": "已停止 H.264 发布"})
            else:
                self.send_json({"error": "not found"}, 404)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("MLVC WebRTC UI: http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        server.server_close()


if __name__ == "__main__":
    main()
