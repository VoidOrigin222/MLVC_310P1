"""Windows low latency three stream comparison UI.

Playback stays in ffplay windows so the live path does not gain browser
buffering.  H.264 can be generated from the original RTSP stream through an
ffmpeg stdout pipe.  Statistics are written to drawtext files and refreshed
once per second.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


@dataclass
class Feed:
    name: str
    url: str = ""
    player: subprocess.Popen | None = None
    encoder: subprocess.Popen | None = None
    osd: Path | None = None


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MLVC 三路低延时对比")
        self.geometry("980x620")
        self.minsize(820, 520)
        self.tmp = Path(tempfile.mkdtemp(prefix="mlvc_ui_"))
        self.feeds = {k: Feed(n) for k, n in (("original", "原图"), ("h264", "H.264"), ("mlvc", "MLVC"))}
        self.vars = {
            "original": tk.StringVar(value="rtsp://192.168.5.3:8554/camera-original"),
            "h264": tk.StringVar(value="rtsp://127.0.0.1:8554/camera-h264"),
            "mlvc": tk.StringVar(value="rtsp://192.168.5.11:8554/mlvc"),
            "ffmpeg": tk.StringVar(value="ffmpeg"),
            "ffplay": tk.StringVar(value="ffplay"),
            "transport": tk.StringVar(value="tcp"),
            "pipe": tk.BooleanVar(value=True),
            "mode": tk.StringVar(value="same_bandwidth"),
            "h264_kbps": tk.DoubleVar(value=8000.0),
            "mlvc_kbps": tk.DoubleVar(value=2000.0),
            "h264_bpp": tk.DoubleVar(value=0.129),
            "mlvc_bpp": tk.DoubleVar(value=0.032),
            "h264_kbs": tk.DoubleVar(value=1000.0),
            "mlvc_kbs": tk.DoubleVar(value=250.0),
        }
        self.ratios = {k: tk.StringVar(value="0.00") for k in ("h264", "mlvc")}
        self.status = tk.StringVar(value="就绪。确认地址后点击启动三路。")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(250, self._poll)
        self.after(1000, self._stats)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(12, weight=1)
        ttk.Label(root, text="输入与播放", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        fields = (("原图 RTSP", "original"), ("H.264 RTSP", "h264"), ("MLVC RTSP", "mlvc"), ("ffmpeg", "ffmpeg"), ("ffplay", "ffplay"))
        for row, (label, key) in enumerate(fields, 1):
            ttk.Label(root, text=label, width=12).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(root, textvariable=self.vars[key]).grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
        bar = ttk.Frame(root)
        bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(bar, text="RTSP传输").pack(side="left")
        ttk.Combobox(bar, textvariable=self.vars["transport"], values=("tcp", "udp"), width=7, state="readonly").pack(side="left", padx=5)
        ttk.Checkbutton(bar, text="H.264 使用原图→ffmpeg pipe", variable=self.vars["pipe"]).pack(side="left", padx=12)
        ttk.Button(bar, text="启动三路", command=self.start).pack(side="right", padx=3)
        ttk.Button(bar, text="停止", command=self.stop).pack(side="right", padx=3)
        ttk.Separator(root).grid(row=7, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(root, text="对比参数（OSD 每秒刷新）", font=("Segoe UI", 12, "bold")).grid(row=8, column=0, columnspan=3, sticky="w")
        table = ttk.Frame(root)
        table.grid(row=9, column=0, columnspan=3, sticky="ew", pady=5)
        for col, heading in enumerate(("流", "码率 kbps", "bpp", "带宽 KB/s", "压缩率（倍）")):
            ttk.Label(table, text=heading).grid(row=0, column=col, padx=5, sticky="w")
        self._metric(table, 1, "H.264", "h264")
        self._metric(table, 2, "MLVC", "mlvc")
        ttk.Label(table, text="原图").grid(row=3, column=0, padx=5, sticky="w")
        ttk.Label(table, text="—").grid(row=3, column=1, padx=5, sticky="w")
        ttk.Label(table, text="24.000").grid(row=3, column=2, padx=5, sticky="w")
        self.raw_kbs = tk.StringVar(value="182250.0")
        ttk.Label(table, textvariable=self.raw_kbs).grid(row=3, column=3, padx=5, sticky="w")
        ttk.Label(table, text="1.00").grid(row=3, column=4, padx=5, sticky="w")
        modes = ttk.Frame(root)
        modes.grid(row=10, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(modes, text="调参目标").pack(side="left")
        ttk.Radiobutton(modes, text="同带宽", variable=self.vars["mode"], value="same_bandwidth").pack(side="left", padx=8)
        ttk.Radiobutton(modes, text="同质量（手动填 bpp）", variable=self.vars["mode"], value="same_quality").pack(side="left", padx=8)
        ttk.Label(modes, text="1920×1080，30 FPS；原图按 RGB 24 bpp 估算").pack(side="right")
        ttk.Label(root, text="运行日志", font=("Segoe UI", 11, "bold")).grid(row=11, column=0, columnspan=3, sticky="w")
        self.log = tk.Text(root, height=7, state="disabled", wrap="word")
        self.log.grid(row=12, column=0, columnspan=3, sticky="nsew")
        ttk.Label(root, textvariable=self.status, foreground="#245b8a").grid(row=13, column=0, columnspan=3, sticky="w", pady=5)

    def _metric(self, parent: ttk.Frame, row: int, name: str, key: str) -> None:
        ttk.Label(parent, text=name).grid(row=row, column=0, padx=5, sticky="w")
        ttk.Entry(parent, textvariable=self.vars[f"{key}_kbps"], width=12).grid(row=row, column=1, padx=5)
        ttk.Entry(parent, textvariable=self.vars[f"{key}_bpp"], width=12).grid(row=row, column=2, padx=5)
        ttk.Entry(parent, textvariable=self.vars[f"{key}_kbs"], width=12).grid(row=row, column=3, padx=5)
        ttk.Label(parent, textvariable=self.ratios[key]).grid(row=row, column=4, padx=5, sticky="w")

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("[%H:%M:%S] ") + text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _num(self, key: str, default: float) -> float:
        try:
            return max(0.0, float(self.vars[key].get()))
        except (tk.TclError, ValueError):
            return default

    def _stats(self) -> None:
        pixels = 1920.0 * 1080.0 * 30.0
        for key in ("h264", "mlvc"):
            if self.vars["mode"].get() == "same_bandwidth":
                bpp = self._num(f"{key}_kbps", 0) * 1000.0 / pixels
                self.vars[f"{key}_bpp"].set(bpp)
            else:
                kbps = self._num(f"{key}_bpp", 0) * pixels / 1000.0
                self.vars[f"{key}_kbps"].set(kbps)
            kbps = self._num(f"{key}_kbps", 0)
            self.vars[f"{key}_kbs"].set(kbps / 8.0)
            bpp = self._num(f"{key}_bpp", 0)
            self.ratios[key].set(f"{24.0 / bpp:.2f}" if bpp else "0.00")
            self._write_osd(key, bpp, kbps / 8.0, self.ratios[key].get())
        self.raw_kbs.set(f"{1920 * 1080 * 30 * 3 / 1024:.1f}")
        self._write_osd("original", 24.0, float(self.raw_kbs.get()), "1.00")
        self.after(1000, self._stats)

    def _write_osd(self, key: str, bpp: float, kbs: float, ratio: str) -> None:
        feed = self.feeds[key]
        feed.osd = feed.osd or self.tmp / f"{key}.txt"
        feed.osd.write_text(f"{feed.name}   压缩率 {ratio}x   bpp {bpp:.4f}   带宽 {kbs:.1f} KB/s", encoding="utf-8")

    def _player_args(self, feed: Feed, source: list[str]) -> list[str]:
        feed.osd = feed.osd or self.tmp / f"{feed.name}.txt"
        if not feed.osd.exists():
            feed.osd.write_text(f"{feed.name}   waiting for statistics", encoding="utf-8")
        osd = (feed.osd or self.tmp / f"{feed.name}.txt").as_posix().replace(":", r"\:").replace("'", r"\'")
        vf = f"drawtext=fontcolor=white:fontsize=24:box=1:boxcolor=black@0.55:x=12:y=12:textfile='{osd}':reload=1"
        return [self.vars["ffplay"].get(), "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer", "-flags", "low_delay", "-framedrop", "-sync", "video", "-probesize", "32", "-analyzeduration", "0", "-vf", vf, "-window_title", feed.name, *source]

    def start(self) -> None:
        self.stop()
        try:
            self._url("original", self.vars["original"].get())
            if self.vars["pipe"].get():
                self._pipe_h264()
            else:
                self._url("h264", self.vars["h264"].get())
            self._url("mlvc", self.vars["mlvc"].get())
            self.status.set("三路播放已启动；统计信息正在叠加到 OSD。")
        except (OSError, ValueError) as exc:
            self._log(f"启动失败：{exc}")
            messagebox.showerror("启动失败", str(exc))
            self.stop()

    def _url(self, key: str, url: str) -> None:
        if not url.strip():
            raise ValueError(f"{self.feeds[key].name} RTSP 地址为空")
        feed = self.feeds[key]
        feed.player = subprocess.Popen(self._player_args(feed, ["-rtsp_transport", self.vars["transport"].get(), url]), creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        self._log(f"已启动 {feed.name}：{url}")

    def _pipe_h264(self) -> None:
        feed = self.feeds["h264"]
        kbps = max(100, int(self._num("h264_kbps", 8000)))
        enc = [self.vars["ffmpeg"].get(), "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer", "-rtsp_transport", self.vars["transport"].get(), "-i", self.vars["original"].get(), "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{max(100, kbps // 2)}k", "-g", "30", "-bf", "0", "-f", "h264", "pipe:1"]
        feed.encoder = subprocess.Popen(enc, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        feed.player = subprocess.Popen(self._player_args(feed, ["-f", "h264", "-i", "-"]), stdin=feed.encoder.stdout, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if feed.encoder.stdout:
            feed.encoder.stdout.close()
        self._log(f"已启动 H.264 pipe：{kbps} kbps")

    def stop(self) -> None:
        for feed in self.feeds.values():
            for process in (feed.player, feed.encoder):
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            feed.player = feed.encoder = None
        if hasattr(self, "status"):
            self.status.set("已停止。")

    def _poll(self) -> None:
        for feed in self.feeds.values():
            if feed.player and feed.player.poll() is not None:
                self._log(f"{feed.name} 已退出，返回码 {feed.player.returncode}")
                feed.player = None
        self.after(250, self._poll)

    def close(self) -> None:
        self.stop()
        for path in self.tmp.glob("*.txt"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            self.tmp.rmdir()
        except OSError:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
