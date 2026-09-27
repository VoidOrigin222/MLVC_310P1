"""Attach to a short-lived RTSP publisher and report received H.264 frames."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg was not found on PATH")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "info",
        "-stats",
        "-rtsp_transport",
        args.transport,
        "-i",
        args.url,
        "-t",
        str(args.duration),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-f",
        "null",
        os.devnull,
    ]

    deadline = time.monotonic() + args.timeout
    last_output = ""
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=args.duration + 15,
                check=False,
            )
            last_output = result.stdout
        except subprocess.TimeoutExpired as error:
            last_output = error.stdout or ""
            if isinstance(last_output, bytes):
                last_output = last_output.decode(errors="replace")

        counts = [int(value) for value in re.findall(r"frame=\s*(\d+)", last_output)]
        frames = max(counts, default=0)
        if frames > 0:
            print(last_output, end="" if last_output.endswith("\n") else "\n")
            print(f"rtsp_frames_received={frames}")
            return 0
        time.sleep(0.1)

    print(last_output, end="" if last_output.endswith("\n") else "\n")
    print("RTSP publisher did not deliver any H.264 frames", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
