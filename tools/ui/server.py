"""Configurable multichannel WebRTC monitor; Python standard library only."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, urlopen

from h264 import Publishers

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/styles.css": ("styles.css", "text/css"),
}
APP_NAME = "ULBVCMonitor"
APP_VERSION = "0.5"


def migrate_config(value):
    """Convert the previous single-channel parameter file into the shared schema."""
    if not isinstance(value, dict) or "channels" in value or "original" not in value:
        return value
    allowed = {"original", "h264", "semantic", "ulbvc", "webrtc", "transport",
               "h264_qp", "ffmpeg", "stats_port", "ulbvc_stats_port"}
    if set(value) - allowed:
        raise ValueError("旧参数文件包含未知字段")
    if "semantic" in value and "ulbvc" in value and value["semantic"] != value["ulbvc"]:
        raise ValueError("旧参数文件中的解码流地址冲突")
    result = json.loads((ROOT / "ui_config.310p.json").read_text(encoding="utf-8"))
    first = result["channels"][0]
    first.update(original=value["original"], h264=value.get("h264", first["h264"]),
                 semantic=value.get("semantic", value.get("ulbvc", first["semantic"])),
                 stats_port=value.get("stats_port", value.get("ulbvc_stats_port", 39341)))
    result.update(webrtc=value.get("webrtc", result["webrtc"]),
                  h264_qp=value.get("h264_qp", 40), h264_ffmpeg=value.get("ffmpeg", "auto"),
                  h264_transport=value.get("transport", "tcp"))
    used = {first["stats_port"]}
    for channel in result["channels"][1:]:
        while channel["stats_port"] in used:
            channel["stats_port"] += 1
        used.add(channel["stats_port"])
    return result


def validate_config(value):
    required = {
        "webrtc",
        "media_api",
        "original_delay_ms",
        "channels",
    }
    h264_defaults = {
        "h264_qp": 40,
        "h264_delay_ms": 200,
        "h264_ffmpeg": "auto",
        "h264_encoder": "auto",
        "h264_auto_start": True,
        "h264_transport": "tcp",
    }
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - set(h264_defaults)
    ):
        raise ValueError("参数文件字段不完整，或包含未知字段")
    result = copy.deepcopy(value)
    for key, default in h264_defaults.items():
        result.setdefault(key, default)
    if type(result["h264_qp"]) is not int or not 0 <= result["h264_qp"] <= 51:
        raise ValueError("H.264 QP 必须为 0–51 的整数")
    if (
        type(result["h264_delay_ms"]) is not int
        or not 0 <= result["h264_delay_ms"] <= 2000
    ):
        raise ValueError("H.264 显示补偿必须为 0–2000 ms 的整数")
    if type(result["h264_auto_start"]) is not bool:
        raise ValueError("H.264 自动启动必须为布尔值")
    if result["h264_encoder"] not in ("auto", "h264_qsv", "h264_nvenc", "h264_amf", "libx264"):
        raise ValueError("请选择有效的 H.264 编码器")
    if result["h264_transport"] not in ("tcp", "udp"):
        raise ValueError("RTSP 传输必须为 TCP 或 UDP")
    if not isinstance(result["h264_ffmpeg"], str) or not result["h264_ffmpeg"].strip():
        raise ValueError("FFmpeg 路径不能为空")

    def address(text, schemes, path=False):
        if not isinstance(text, str) or any(c.isspace() for c in text):
            raise ValueError("地址不能包含空白字符")
        parsed = urlsplit(text)
        if parsed.scheme not in schemes or not parsed.hostname:
            raise ValueError("请填写完整的视频源或服务地址")
        _ = parsed.port
        if parsed.fragment or parsed.query or (path and not parsed.path.strip("/")):
            raise ValueError("视频地址必须包含完整路径，不能包含查询参数或锚点")
        if any(part in (".", "..") for part in parsed.path.split("/")):
            raise ValueError("地址路径无效")
        return text.rstrip("/")

    result["webrtc"] = address(result["webrtc"], ("http", "https"))
    result["media_api"] = address(result["media_api"], ("http", "https"))
    delay = result["original_delay_ms"]
    if type(delay) is not int or not 0 <= delay <= 2000:
        raise ValueError("原图显示补偿必须为 0–2000 ms 的整数")
    channels = result["channels"]
    if not isinstance(channels, list) or not channels:
        raise ValueError("至少需要配置一个通道")
    ids, ports = set(), set()
    for index, channel in enumerate(channels):
        fields = {
            "id",
            "name",
            "original",
            "semantic",
            "width",
            "height",
            "fps",
            "stats_port",
        }
        if (
            not isinstance(channel, dict)
            or not fields <= set(channel)
            or set(channel) - fields - {"h264"}
        ):
            raise ValueError("通道参数不完整，或包含未知字段")
        if not isinstance(channel["id"], str) or not re.fullmatch(
            r"[a-z0-9_-]{1,32}", channel["id"]
        ):
            raise ValueError("通道 ID 只能使用小写字母、数字、下划线和连字符")
        if channel["id"] in ids:
            raise ValueError("通道 ID 不可重复")
        ids.add(channel["id"])
        if (
            not isinstance(channel["name"], str)
            or not 1 <= len(channel["name"].strip()) <= 32
        ):
            raise ValueError("通道名称须为 1–32 个字符")
        channel["name"] = channel["name"].strip()
        for key in ("original", "semantic"):
            channel[key] = address(channel[key], ("rtsp", "rtsps"), True)
        if "h264" not in channel:
            origin = urlsplit(channel["original"])
            channel["h264"] = f"{origin.scheme}://{origin.netloc}/camera-h264{index}"
        channel["h264"] = address(channel["h264"], ("rtsp", "rtsps"), True)
        if channel["h264"] in (channel["original"], channel["semantic"]):
            raise ValueError("H.264 对照地址必须与原图和语义压缩地址不同")
        for key in ("width", "height"):
            if type(channel[key]) is not int or not 16 <= channel[key] <= 8192:
                raise ValueError("图像尺寸须为 16–8192 的整数")
        if (
            type(channel["fps"]) not in (int, float)
            or not math.isfinite(channel["fps"])
            or not 1 <= channel["fps"] <= 240
        ):
            raise ValueError("目标帧率须为 1–240")
        port = channel["stats_port"]
        if type(port) is not int or not 1024 <= port <= 65535 or port in ports:
            raise ValueError("统计端口须为 1024–65535 且不可重复")
        ports.add(port)
    return result


class ConfigStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        if self.path.exists():
            previous = json.loads(self.path.read_text(encoding="utf-8-sig"))
            self.value = validate_config(migrate_config(previous))
            if self.value != previous:
                if "channels" not in previous:
                    backup = self.path.with_name(self.path.name + ".v0.5-backup")
                    if not backup.exists():
                        backup.write_bytes(self.path.read_bytes())
                self.save(self.value)
        else:
            self.value = validate_config(
                json.loads((ROOT / "ui_config.json").read_text(encoding="utf-8"))
            )
            self.save(self.value)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.value)

    def save(self, value):
        validated = validate_config(value)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self.path.parent, delete=False
                ) as out:
                    temporary = Path(out.name)
                    json.dump(validated, out, ensure_ascii=False, indent=2)
                    out.write("\n")
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, self.path)
                self.value = validated
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
        return self.snapshot()


class Counters:
    """Cumulative codec counters, never the decoded H.264 playback bitrate."""

    def __init__(self):
        self.lock = threading.Lock()
        self.samples = {}

    def record(self, channel, value, now=None):
        if not isinstance(value, dict) or value.get("kind") not in (
            "semantic_codec_stats",
            "ulbvc_rtp_stats",
        ):
            return False
        count = value.get("payload_bytes")
        frames = value.get("frames")
        if (
            type(count) is not int
            or type(frames) is not int
            or min(count, frames) < 0
            or max(count, frames) > 2**63 - 1
        ):
            return False
        now = time.monotonic() if now is None else now
        wire = value.get("wire_bytes")
        if wire is not None and (type(wire) is not int or not 0 <= wire <= 2**63 - 1):
            return False
        with self.lock:
            samples = self.samples.setdefault(channel, deque(maxlen=120))
            if samples and (count < samples[-1][1] or frames < samples[-1][2] or
                            value["kind"] != samples[-1][4] or
                            (wire is not None and samples[-1][3] is not None and wire < samples[-1][3])):
                samples.clear()
            samples.append((now, count, frames, wire, value["kind"]))
            while len(samples) > 2 and samples[1][0] < now - 3:
                samples.popleft()
        return True

    def clear(self):
        with self.lock:
            self.samples.clear()

    def metrics(self, channel, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            samples = list(self.samples.get(channel["id"], ()))
        if len(samples) < 2 or now - samples[-1][0] > 5:
            return None
        elapsed = samples[-1][0] - samples[0][0]
        frames = samples[-1][2] - samples[0][2]
        count = samples[-1][1] - samples[0][1]
        if elapsed <= 0 or frames <= 0:
            return None
        bpp = count * 8 / (frames * channel["width"] * channel["height"])
        return {
            "kbs": count / elapsed / 1000,
            "bpp": bpp,
            "ratio": 24 / bpp if bpp else None,
            "encode_fps": frames / elapsed,
            "wire_kbs": ((samples[-1][3] - samples[0][3]) / elapsed / 1000
                         if samples[-1][3] is not None and samples[0][3] is not None else None),
            "source": samples[-1][4],
        }


class Monitor:
    def __init__(self, store):
        self.store = store
        self.counters = Counters()
        self.closed = threading.Event()
        self.lock = threading.RLock()
        self.sessions = {}
        self.sockets = {}
        self.bind_error = ""
        self.h264 = Publishers()
        self.ready_paths = None
        self.h264_thread = threading.Thread(target=self.maintain_h264, daemon=True)
        self.h264_thread.start()
        self.thread = threading.Thread(target=self.receive_stats, daemon=True)
        self.thread.start()

    def maintain_h264(self):
        while not self.closed.is_set():
            self.h264.update(self.store.snapshot(), self.ready_paths)
            self.closed.wait(1)

    def receive_stats(self):
        while not self.closed.is_set():
            cfg = self.store.snapshot()
            wanted = {c["stats_port"]: c["id"] for c in cfg["channels"]}
            for port in list(self.sockets):
                if port not in wanted:
                    self.sockets.pop(port).close()
            errors = []
            for port in wanted:
                if port in self.sockets:
                    continue
                stream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    stream.bind(("0.0.0.0", port))
                    stream.setblocking(False)
                    self.sockets[port] = stream
                except OSError as error:
                    stream.close()
                    errors.append(f"{port}: {error}")
            self.bind_error = "; ".join(errors)
            sockets = list(self.sockets.values())
            if not sockets:
                self.closed.wait(1)
                continue
            ready, _, _ = select.select(sockets, [], [], 0.25)
            for stream in ready:
                try:
                    data, _ = stream.recvfrom(8192)
                    channel = wanted.get(stream.getsockname()[1])
                    if channel:
                        self.counters.record(channel, json.loads(data))
                except (OSError, ValueError, UnicodeError):
                    pass
        for stream in self.sockets.values():
            stream.close()

    def status(self):
        cfg = self.store.snapshot()
        paths = None
        try:
            with urlopen(
                cfg["media_api"] + "/v3/paths/list?itemsPerPage=100",
                timeout=0.7,
            ) as response:
                paths = {
                    p["name"]
                    for p in json.load(response).get("items", [])
                    if p.get("ready")
                }
        except (OSError, ValueError, KeyError):
            pass
        self.ready_paths = paths
        return {
            "channels": [
                {
                    "id": c["id"],
                    "metrics": self.counters.metrics(c),
                    "feeds": {
                        kind: (
                            None
                            if paths is None
                            else urlsplit(c[kind]).path.strip("/") in paths
                        )
                        for kind in ("original", "h264", "semantic")
                    },
                }
                for c in cfg["channels"]
            ],
            "stats_error": self.bind_error,
            "h264": self.h264.status(),
        }

    def connect(self, channel_id, kind, offer):
        cfg = self.store.snapshot()
        channel = next((c for c in cfg["channels"] if c["id"] == channel_id), None)
        if channel is None or kind not in ("original", "h264", "semantic"):
            raise ValueError("视频通道不存在")
        path = urlsplit(channel[kind]).path.strip("/")
        endpoint = cfg["webrtc"] + "/" + quote(path, safe="/") + "/whep"
        request = Request(
            endpoint,
            data=offer,
            headers={"Content-Type": "application/sdp"},
            method="POST",
        )
        with urlopen(request, timeout=8) as response:
            answer = response.read(1024 * 1024)
            location = response.headers.get("Location")
        if not location:
            raise ValueError("播放服务未返回会话地址")
        session = urljoin(endpoint, location)
        if (
            urlsplit(session).netloc != urlsplit(endpoint).netloc
            or urlsplit(session).scheme != urlsplit(endpoint).scheme
        ):
            raise ValueError("播放服务返回了无效会话地址")
        token = uuid.uuid4().hex
        with self.lock:
            self.sessions[token] = session
        return answer, token

    def disconnect(self, token):
        with self.lock:
            session = self.sessions.pop(token, None)
        if session:
            try:
                with urlopen(Request(session, method="DELETE"), timeout=2):
                    pass
            except OSError:
                pass

    def close(self):
        self.closed.set()
        self.h264.close()
        self.h264_thread.join(timeout=2)
        self.thread.join(timeout=2)
        from concurrent.futures import ThreadPoolExecutor

        with self.lock:
            sessions = list(self.sessions)
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(self.disconnect, sessions))


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, body, content_type="application/json", headers=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ASSETS:
                name, mime = ASSETS[path]
                self.send(200, (ROOT / name).read_bytes(), mime)
            elif path == "/api/config":
                self.send(200, monitor.store.snapshot())
            elif path == "/api/status":
                self.send(200, monitor.status())
            elif path == "/api/health":
                self.send(
                    200,
                    {
                        "app": APP_NAME,
                        "version": APP_VERSION,
                        "config_path": str(monitor.store.path),
                    },
                )
            else:
                self.send(404, {"error": "地址不存在"})

        def do_POST(self):
            token = None
            try:
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    self.send(403, {"error": "请求来源无效"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("请求内容长度无效")
                body = self.rfile.read(length)
                path = urlsplit(self.path).path
                if path == "/api/config":
                    config = monitor.store.save(json.loads(body))
                    monitor.counters.clear()
                    monitor.h264.enabled = None
                    self.send(200, config)
                elif path in ("/api/h264/start", "/api/h264/stop"):
                    monitor.h264.enabled = path.endswith("/start")
                    if not monitor.h264.enabled:
                        monitor.h264.stop()
                    self.send(200, {"ok": True})
                elif re.fullmatch(
                    r"/api/whep/[a-z0-9_-]+/(original|h264|semantic)", path
                ):
                    _, _, _, channel, kind = path.split("/")
                    answer, token = monitor.connect(channel, kind, body)
                    self.send(
                        201,
                        answer,
                        "application/sdp",
                        {"Location": "/api/session/" + token},
                    )
                else:
                    self.send(404, {"error": "地址不存在"})
            except HTTPError as error:
                self.send(502, {"error": f"视频源暂不可用（{error.code}）"})
            except (URLError, TimeoutError, ConnectionError):
                if token:
                    monitor.disconnect(token)
                try:
                    self.send(502, {"error": "无法连接视频播放服务"})
                except OSError:
                    pass
            except (ValueError, UnicodeError) as error:
                self.send(400, {"error": str(error)})
            except OSError:
                if token:
                    monitor.disconnect(token)
                try:
                    self.send(
                        500,
                        {"error": "保存或连接失败，请检查参数文件和服务地址"},
                    )
                except OSError:
                    pass

        def do_DELETE(self):
            path = urlsplit(self.path).path
            if re.fullmatch(r"/api/session/[0-9a-f]{32}", path):
                monitor.disconnect(path.rsplit("/", 1)[1])
                self.send(200, {"ok": True})
            else:
                self.send(404, {"error": "会话不存在"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="多路语义压缩视频监控")
    directory = Path(sys.executable).parent if getattr(sys, "frozen", False) else ROOT
    parser.add_argument("--config", type=Path, default=directory / "ui_config.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    monitor = Monitor(ConfigStore(args.config))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(monitor))
    server.daemon_threads = True
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        monitor.close()


if __name__ == "__main__":
    main()
