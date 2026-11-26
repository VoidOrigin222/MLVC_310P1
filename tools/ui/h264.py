"""Owned H.264 comparison publishers, shared by Thor and Ascend UI profiles."""

from pathlib import Path
import os
import subprocess
import tempfile
import threading
import time


def ffmpeg_path(configured="auto"):
    if configured != "auto":
        return configured
    bundled = Path(__file__).resolve().parent / "ffmpeg.exe"
    return str(bundled) if bundled.is_file() else "ffmpeg"


def encoder_options(config, encoder):
    """Use the same constant-QP, low-latency settings for probing and streaming."""
    args = ["-c:v", encoder]
    if encoder == "h264_qsv":
        # -q:v applies FFmpeg's QP-to-lambda conversion required by QSV CQP.
        args += [
            "-preset",
            "veryfast",
            "-flags:v",
            "+qscale",
            "-q:v",
            str(config["h264_qp"]),
            "-i_qfactor",
            "1",
            "-i_qoffset",
            "0",
            "-look_ahead",
            "0",
            "-async_depth",
            "1",
        ]
    elif encoder == "h264_amf":
        args += [
            "-usage",
            "ultralowlatency",
            "-quality",
            "speed",
            "-rc",
            "cqp",
            "-qp_i",
            str(config["h264_qp"]),
            "-qp_p",
            str(config["h264_qp"]),
            "-qp_b",
            str(config["h264_qp"]),
            "-preanalysis",
            "0",
            "-preencode",
            "0",
            "-frame_skipping",
            "0",
            # AMF defaults to 16 concurrent frames in the bundled FFmpeg.
            "-async_depth",
            "1",
        ]
    elif encoder == "h264_nvenc":
        args += [
            "-preset",
            "p1",
            "-tune",
            "ull",
            "-rc",
            "constqp",
            "-qp",
            str(config["h264_qp"]),
            "-rc-lookahead",
            "0",
        ]
    elif encoder == "libx264":
        args += ["-preset", "ultrafast", "-tune", "zerolatency", "-qp", str(config["h264_qp"])]
    else:
        raise ValueError(f"Unsupported hardware H.264 encoder: {encoder}")
    return args + ["-bf", "0", "-g", "96"]


def command(config, channel, encoder):
    args = [
        ffmpeg_path(config["h264_ffmpeg"]),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-xerror",
    ]
    if encoder == "h264_qsv":
        args += ["-hwaccel", "d3d11va"]
    elif encoder == "h264_amf":
        # Keep decoded surfaces on the GPU for AMF; no CPU frame download.
        args += ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"]
    elif encoder == "h264_nvenc":
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif encoder == "libx264":
        pass
    else:
        raise ValueError(f"Unsupported hardware H.264 encoder: {encoder}")
    args += [
        "-rtsp_transport",
        config.get("h264_transport", "tcp"),
        "-timeout",
        "5000000",
        "-analyzeduration",
        "1000000",
        "-probesize",
        "5000000",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        # FFmpeg frame threading can retain 15 frames even with D3D11VA.
        # At 12 FPS that adds 1.25 s before the hardware encoder sees a frame.
        "-threads:v",
        "1",
        "-i",
        channel["original"],
        "-map",
        "0:v:0",
        "-an",
    ]
    return args + encoder_options(config, encoder) + [
        # Copy-mode DVPP streams may expose the 90 kHz RTP clock as FPS.
        # Keep input timestamps and never manufacture CFR filler frames.
        "-fps_mode:v",
        "passthrough",
        "-enc_time_base:v",
        "demux",
        "-rtsp_transport",
        config.get("h264_transport", "tcp"),
        "-f",
        "rtsp",
        channel["h264"],
    ]


def probe_encoder(config, encoder):
    args = [
        ffmpeg_path(config["h264_ffmpeg"]),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x480:r=30,format=nv12",
        "-frames:v",
        "1",
    ] + encoder_options(config, encoder) + ["-f", "null", "-"]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {
            "available": result.returncode == 0,
            "returncode": result.returncode,
            "detail": result.stderr.decode("utf8", errors="replace")[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"available": False, "returncode": None, "detail": "初始化超过 8 秒"}
    except OSError as error:
        return {"available": False, "returncode": None, "detail": str(error)}


class Publishers:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = {}
        self.encoder = None
        self.probe_key = None
        self.probe_after = 0
        self.error = ""
        self.probe_results = {}
        self.enabled = None
        self.sequence = 0
        self.closed = False
        self.log_root = Path(tempfile.gettempdir()) / "ULBVCMonitor-h264"

    def stop(self, channel=None):
        with self.lock:
            ids = [channel] if channel else list(self.jobs)
            for key in ids:
                job = self.jobs.pop(key, None)
                if not job:
                    continue
                process = job["process"]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                job["log"].close()

    def probe(self, config):
        key = (config["h264_ffmpeg"], config["h264_encoder"], config["h264_qp"])
        if key == self.probe_key and (
            self.encoder or time.monotonic() < self.probe_after
        ):
            return
        self.probe_key = key
        self.encoder = None
        self.probe_results = {}
        choices = (
            ["h264_qsv", "h264_nvenc", "h264_amf"]
            if os.name == "nt"
            else ["h264_nvenc"]
        )
        if config["h264_encoder"] != "auto":
            choices = [config["h264_encoder"]]
        for encoder in choices:
            result = probe_encoder(config, encoder)
            self.probe_results[encoder] = result
            if result["available"]:
                self.encoder = encoder
                self.error = ""
                return
        self.probe_after = time.monotonic() + 30
        self.error = "H.264 编码器不可用，请检查 FFmpeg、所选编码器及驱动"

    def update(self, config, ready_paths):
        with self.lock:
            if self.closed:
                return
            enabled = (
                config["h264_auto_start"] if self.enabled is None else self.enabled
            )
            if not enabled:
                self.stop()
                return
            valid_ids = {c["id"] for c in config["channels"]}
            for key in list(self.jobs):
                if key not in valid_ids:
                    self.stop(key)
            # An unavailable MediaMTX API is not evidence that video vanished.
            if ready_paths is None:
                return
            for channel in config["channels"]:
                if (channel["id"] in self.jobs and
                        channel["original"].split("/", 3)[-1] not in ready_paths):
                    self.stop(channel["id"])
            candidates = [
                c
                for c in config["channels"]
                if c["original"].split("/", 3)[-1] in ready_paths
            ]
            if not candidates:
                return
            self.probe(config)
            if not self.encoder or self.closed:
                return
            for channel in candidates:
                args = command(config, channel, self.encoder)
                job = self.jobs.get(channel["id"])
                if job and job["args"] != args:
                    self.stop(channel["id"])
                    job = None
                if job and job["process"].poll() is None:
                    continue
                if job and time.monotonic() < job["started"] + 10:
                    continue
                if job:
                    self.stop(channel["id"])
                # A ready external comparison stream belongs to its publisher.
                if channel["h264"].split("/", 3)[-1] in ready_paths:
                    continue
                self.log_root.mkdir(parents=True, exist_ok=True)
                self.sequence += 1
                path = (
                    self.log_root / f"{os.getpid()}-{channel['id']}-{self.sequence}.log"
                )
                log = path.open("wb")
                try:
                    process = subprocess.Popen(
                        args,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except OSError:
                    log.close()
                    self.error = "无法启动 H.264 推流，请检查 FFmpeg 路径"
                    continue
                self.jobs[channel["id"]] = {
                    "process": process,
                    "log": log,
                    "args": args,
                    "started": time.monotonic(),
                    "path": str(path),
                }

    def status(self):
        with self.lock:
            return {
                "encoder": self.encoder,
                "error": self.error,
                "probes": dict(self.probe_results),
                "channels": {
                    key: job["process"].poll() is None for key, job in self.jobs.items()
                },
            }

    def close(self):
        self.closed = True
        self.stop()
