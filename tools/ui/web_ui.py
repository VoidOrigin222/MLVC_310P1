"""Local HTML control panel for the MLVC low-latency comparison pipeline.

The browser owns controls and statistics. Video stays in ffplay windows because
normal browsers cannot consume RTSP with the same latency. The H.264 path can
be encoded from the original RTSP through an ffmpeg stdout pipe.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLVC 三路低延时对比</title>
<style>
:root{font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#e7edf5;background:#0d1420}*{box-sizing:border-box}body{margin:0}header{padding:20px 28px;background:#121d2d;border-bottom:1px solid #24334a}h1{font-size:22px;margin:0 0 6px}p{color:#9caec3;margin:4px 0}.layout{max-width:1250px;margin:auto;padding:20px;display:grid;gap:16px}.panel{background:#121d2d;border:1px solid #263851;border-radius:10px;padding:16px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.form{display:grid;grid-template-columns:130px 1fr 130px 1fr;gap:9px;align-items:center}.full{grid-column:1/-1}label{color:#adbed2}input,select{width:100%;padding:9px;border-radius:6px;border:1px solid #3b506b;background:#0b1320;color:#e7edf5}button{padding:9px 15px;border:0;border-radius:6px;background:#2779d4;color:#fff;cursor:pointer;margin-right:8px}button.stop{background:#9c3b4b}button.secondary{background:#405675}.card{padding:15px;background:#0b1320;border:1px solid #263851;border-radius:8px;min-height:150px}.card h3{margin:0 0 10px}.state{font-size:13px;color:#78d49a}.metric{display:flex;justify-content:space-between;border-top:1px solid #263851;padding:7px 0;color:#b7c6d8}.metric b{color:#fff}.note{font-size:13px;color:#9caec3}.error{color:#ff9e9e}@media(max-width:800px){.grid{grid-template-columns:1fr}.form{grid-template-columns:1fr}.full{grid-column:auto}}
</style></head><body><header><h1>MLVC 三路低延时对比</h1><p>浏览器负责控制和统计，视频在本机 ffplay 窗口中播放，以保持 RTSP 低延时。</p></header>
<main class="layout"><section class="panel"><div class="form">
<label>原图 RTSP</label><input id="original"><label>MLVC RTSP</label><input id="mlvc">
<label>H.264 RTSP</label><input id="h264"><label>RTSP传输</label><select id="transport"><option>tcp</option><option>udp</option></select>
<label>ffmpeg路径</label><input id="ffmpeg" value="ffmpeg"><label>ffplay路径</label><input id="ffplay" value="ffplay">
<label>H.264 pipe</label><input id="pipe" type="checkbox" checked><span></span><span></span>
<div class="full"><button onclick="start()">启动三路</button><button class="stop" onclick="stop()">停止</button><span id="message" class="note"></span></div>
</div></section>
<section class="panel"><h2>三路状态</h2><div class="grid"><div class="card"><h3>原图</h3><div id="original-state" class="state">未启动</div><p class="note">ffplay 窗口</p><div class="metric">带宽 <b id="original-kbs">—</b></div><div class="metric">压缩率 <b>1.00 倍</b></div></div>
<div class="card"><h3>H.264</h3><div id="h264-state" class="state">未启动</div><p class="note">ffplay 窗口</p><div class="metric">bpp <b id="h264-bpp">—</b></div><div class="metric">带宽 <b id="h264-kbs">—</b></div><div class="metric">压缩率 <b id="h264-ratio">—</b></div></div>
<div class="card"><h3>MLVC</h3><div id="mlvc-state" class="state">未启动</div><p class="note">ffplay 窗口</p><div class="metric">bpp <b id="mlvc-bpp">—</b></div><div class="metric">带宽 <b id="mlvc-kbs">—</b></div><div class="metric">压缩率 <b id="mlvc-ratio">—</b></div></div></div></section>
<section class="panel"><h2>对比调参</h2><div class="form"><label>目标模式</label><select id="mode" onchange="refresh()"><option value="same_bandwidth">同带宽</option><option value="same_quality">同质量（按 bpp）</option></select><label>帧率</label><input id="fps" type="number" value="30" min="1"><label>H.264 码率 kbps</label><input id="h264_kbps" type="number" value="8000"><label>H.264 bpp</label><input id="h264_bpp_in" type="number" value="0.129" step="0.001"><label>MLVC 码率 kbps</label><input id="mlvc_kbps" type="number" value="2000"><label>MLVC bpp</label><input id="mlvc_bpp_in" type="number" value="0.032" step="0.001"></div><p class="note">统计按 1920×1080、RGB 原图 24 bpp 计算；播放窗口的 OSD 同步显示这些数值。</p></section></main>
<script>
const $=id=>document.getElementById(id);let timer;
async function api(path,body){let r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});return r.json()}
function values(){return {original:$('original').value,mlvc:$('mlvc').value,h264:$('h264').value,transport:$('transport').value,ffmpeg:$('ffmpeg').value,ffplay:$('ffplay').value,pipe:$('pipe').checked,mode:$('mode').value,fps:+$('fps').value,h264_kbps:+$('h264_kbps').value,mlvc_kbps:+$('mlvc_kbps').value,h264_bpp:+$('h264_bpp_in').value,mlvc_bpp:+$('mlvc_bpp_in').value}}
async function start(){let x=await api('/api/start',values());$('message').textContent=x.message||x.error||'';refresh()}
async function stop(){let x=await api('/api/stop',{});$('message').textContent=x.message||x.error||'';refresh()}
async function refresh(){await api('/api/config',values());let x=await api('/api/status');for(let k of ['original','h264','mlvc']){$(k+'-state').textContent=x.feeds[k]?'运行中':'未启动'}for(let k of ['h264','mlvc']){$(k+'-bpp').textContent=x.metrics[k].bpp.toFixed(4);$(k+'-kbs').textContent=x.metrics[k].kbs.toFixed(1)+' KB/s';$(k+'-ratio').textContent=x.metrics[k].ratio.toFixed(2)+' 倍'}$('original-kbs').textContent=x.metrics.original.kbs.toFixed(1)+' KB/s';if(x.error)$('message').textContent=x.error}
fetch('/api/config').then(r=>r.json()).then(x=>{for(let k of ['original','mlvc','h264'])$(k).value=x[k];refresh()});timer=setInterval(refresh,1000);
</script></body></html>'''


class Controller:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tmp = Path(tempfile.mkdtemp(prefix="mlvc_web_"))
        self.cfg = {"original": "rtsp://192.168.5.3:8554/camera-original", "mlvc": "rtsp://192.168.5.11:8554/mlvc", "h264": "rtsp://127.0.0.1:8554/camera-h264", "transport": "tcp", "ffmpeg": "ffmpeg", "ffplay": "ffplay", "pipe": True, "mode": "same_bandwidth", "fps": 30, "h264_kbps": 8000, "mlvc_kbps": 2000, "h264_bpp": 0.129, "mlvc_bpp": 0.032}
        self.players = {"original": None, "h264": None, "mlvc": None}
        self.encoders = {"h264": None}

    def _osd(self, key: str, text: str) -> Path:
        path = self.tmp / f"{key}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def _player(self, key: str, source: list[str]) -> list[str]:
        osd = self._osd(key, "waiting for statistics").as_posix().replace(":", r"\:").replace("'", r"\'")
        vf = f"drawtext=fontcolor=white:fontsize=24:box=1:boxcolor=black@0.55:x=12:y=12:textfile='{osd}':reload=1"
        return [self.cfg["ffplay"], "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer", "-flags", "low_delay", "-framedrop", "-sync", "video", "-probesize", "32", "-analyzeduration", "0", "-vf", vf, "-window_title", key, *source]

    def stop(self) -> None:
        with self.lock:
            for p in list(self.players.values()) + list(self.encoders.values()):
                if p and p.poll() is None:
                    p.terminate()
            self.players = {k: None for k in self.players}
            self.encoders = {"h264": None}

    def start(self, cfg: dict) -> None:
        self.stop()
        self.cfg.update(cfg)
        self._start_url("original", self.cfg["original"])
        if self.cfg.get("pipe", True):
            rate = max(100, int(float(self.cfg["h264_kbps"])))
            enc = [self.cfg["ffmpeg"], "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer", "-rtsp_transport", self.cfg["transport"], "-i", self.cfg["original"], "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-b:v", f"{rate}k", "-maxrate", f"{rate}k", "-bufsize", f"{max(100, rate // 2)}k", "-g", "30", "-bf", "0", "-f", "h264", "pipe:1"]
            self.encoders["h264"] = subprocess.Popen(enc, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.players["h264"] = subprocess.Popen(self._player("h264", ["-f", "h264", "-i", "-"]), stdin=self.encoders["h264"].stdout)
        else:
            self._start_url("h264", self.cfg["h264"])
        self._start_url("mlvc", self.cfg["mlvc"])

    def _start_url(self, key: str, url: str) -> None:
        if not url:
            raise ValueError(key + " RTSP 地址为空")
        self.players[key] = subprocess.Popen(self._player(key, ["-rtsp_transport", self.cfg["transport"], url]))

    def status(self) -> dict:
        fps = max(1.0, float(self.cfg.get("fps", 30)))
        pixels = 1920 * 1080 * fps
        metrics = {"original": {"bpp": 24.0, "kbs": 1920 * 1080 * fps * 3 / 1024, "ratio": 1.0}}
        for key in ("h264", "mlvc"):
            if self.cfg.get("mode") == "same_quality":
                kbps = float(self.cfg.get(key + "_bpp", 0)) * pixels / 1000
            else:
                kbps = float(self.cfg.get(key + "_kbps", 0))
            bpp = kbps * 1000 / pixels
            metrics[key] = {"bpp": bpp, "kbs": kbps / 8, "ratio": 24 / bpp if bpp else 0}
            self._osd(key, f"{key}   压缩率 {metrics[key]['ratio']:.2f}x   bpp {bpp:.4f}   带宽 {kbps / 8:.1f} KB/s")
        self._osd("original", f"原图   压缩率 1.00x   bpp 24.0000   带宽 {metrics['original']['kbs']:.1f} KB/s")
        return {"feeds": {k: bool(p and p.poll() is None) for k, p in self.players.items()}, "metrics": metrics}


controller = Controller()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def _send(self, data: object, code: int = 200, content_type: str = "application/json") -> None:
        raw = data.encode() if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTML, content_type="text/html")
        elif path == "/api/config":
            self._send(controller.cfg)
        elif path == "/api/status":
            self._send(controller.status())
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/config":
                controller.cfg.update(body)
                self._send({"message": "配置已更新"})
            elif path == "/api/start":
                controller.start(body)
                self._send({"message": "三路 ffplay 已启动"})
            elif path == "/api/stop":
                controller.stop()
                self._send({"message": "已停止"})
            else:
                self._send({"error": "not found"}, 404)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send({"error": str(exc)}, 400)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("MLVC UI: http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.server_close()


if __name__ == "__main__":
    main()
