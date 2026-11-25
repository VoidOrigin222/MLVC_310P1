#!/usr/bin/env python3
"""Run a finite dual-board test, capture the decoded image and monitor temperature."""

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from urllib.request import urlopen


ROOT = "/root/ULBVC_310P1"
REPO = Path(__file__).resolve().parents[1]


def remote(host, command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], **kwargs)


def start(host, role, config, name, profile_npu=False):
    runner = f"bash scripts/run_{role}.sh {shlex.quote(config)}"
    if profile_npu and role == "encode":
        application = f"{ROOT}/build/ulbvc_encode --config {shlex.quote(config)}"
        runner = ("/usr/local/Ascend/cann/tools/profiler/bin/msprof "
                  f"--output={ROOT}/logs/{name}-msprof --ascendcl=on --runtime-api=on "
                  "--task-time=on --ai-core=off "
                  f"--application={shlex.quote(application)}")
    command = (
        f"cd {ROOT} || exit 1; source scripts/env.sh >/dev/null; {runner} "
        f"> {ROOT}/logs/{name}-{role}.log 2>&1 & child=$!; "
        f"echo $child > {ROOT}/logs/{name}-{role}.pid; wait $child"
    )
    return subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, command],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop(host, role, name):
    path = f"{ROOT}/logs/{name}-{role}.pid"
    # PID files belong to this test's unique prefix; never match unrelated jobs.
    script = f"""import os,signal
from pathlib import Path
p=Path({path!r})
def stop_tree(pid):
    try:
        children=Path('/proc/%s/task/%s/children'%(pid,pid)).read_text().split()
        for child in children: stop_tree(int(child))
        os.kill(pid,signal.SIGTERM)
    except (FileNotFoundError,ProcessLookupError): pass
if p.is_file(): stop_tree(int(p.read_text()))
"""
    remote(host, "python3 -c " + shlex.quote(script),
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)


def temperature(host):
    result = remote(host, "npu-smi info", capture_output=True, timeout=10)
    for line in result.stdout.decode(errors="replace").splitlines():
        columns = line.split("|")
        if len(columns) >= 5 and "310P1" in columns[1]:
            return {"health": columns[2].strip(), "temperature_c": int(columns[3].split()[1])}
    raise RuntimeError("could not read device health from " + host)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=180,
                        help="maximum session seconds; 0 keeps a monitored live session running")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--profile-npu", action="store_true",
                        help="collect encoder CANN API/task timeline (adds profiling overhead)")
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("timeout must be nonnegative")
    if not re.fullmatch(r"[a-z0-9_-]+", args.name):
        parser.error("name must contain only lowercase letters, digits, underscores or hyphens")
    output = REPO / "artifacts/dual-board" / args.name
    if output.exists():
        parser.error("test output already exists; choose a new name")
    output.mkdir(parents=True)
    decoder = start("310PDecoder", "decode", args.decoder_config, args.name)
    time.sleep(0.5)
    encoder = start("310PEncoder", "encode", args.encoder_config, args.name, args.profile_npu)
    started = time.monotonic()
    samples, capture, capture_log = [], None, None
    last_temperature = -10
    failure = None
    try:
        while encoder.poll() is None or decoder.poll() is None:
            elapsed = time.monotonic() - started
            if args.timeout and elapsed > args.timeout:
                raise RuntimeError("test timeout")
            if encoder.poll() not in (None, 0) or decoder.poll() not in (None, 0):
                raise RuntimeError("a board process failed")
            if elapsed - last_temperature >= 5:
                sample = {"elapsed_s": elapsed, "encoder": temperature("310PEncoder"),
                          "decoder": temperature("310PDecoder")}
                samples.append(sample)
                last_temperature = elapsed
                if any(value["temperature_c"] >= 90 or value["health"] != "OK"
                       for value in (sample["encoder"], sample["decoder"])):
                    raise RuntimeError("temperature or device health limit reached")
            if args.capture and capture is None:
                try:
                    with urlopen("http://127.0.0.1:9997/v3/paths/list", timeout=1) as response:
                        paths = json.load(response)["items"]
                    ready = any(path.get("name") == "ulbvc" and path.get("ready") for path in paths)
                except (OSError, ValueError):
                    ready = False
                if ready:
                    capture_log = (output / "capture.log").open("wb")
                    capture = subprocess.Popen([
                        shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
                        "-rtsp_transport", "tcp", "-analyzeduration", "100000", "-probesize", "1000000",
                        "-i", "rtsp://127.0.0.1:8554/ulbvc", "-frames:v", "1", "-update", "1",
                        str(output / "decoded.png")], stdout=capture_log, stderr=subprocess.STDOUT)
            time.sleep(0.2)
    except Exception as error:
        failure = str(error)
        stop("310PEncoder", "encode", args.name)
        stop("310PDecoder", "decode", args.name)
    finally:
        for process in (encoder, decoder):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        if capture is not None:
            try:
                capture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                capture.kill()
                capture.wait()
        if capture_log is not None:
            capture_log.close()
        for host, role in (("310PEncoder", "encode"), ("310PDecoder", "decode")):
            subprocess.run(["scp", f"{host}:{ROOT}/logs/{args.name}-{role}.log", str(output / f"{role}.log")],
                           check=True, capture_output=True)
        metrics = {}
        for role in ("encode", "decode"):
            values = {}
            for line in (output / f"{role}.log").read_text(errors="replace").splitlines():
                match = re.fullmatch(r"([a-z_][a-z0-9_]*)=([0-9]+(?:\.[0-9]+)?)", line)
                if match:
                    values[match[1]] = float(match[2]) if "." in match[2] else int(match[2])
            metrics[role] = values
        sent, received = metrics["encode"], metrics["decode"]
        if not failure and not encoder.returncode and not decoder.returncode:
            if not sent.get("frames") or sent.get("frames") != received.get("frames"):
                failure = "encoded/decoded frame count mismatch"
            elif received.get("forward_dropped_frames", 0):
                failure = "display forwarding dropped frames"
            elif sent.get("detection_frames", 0) and any((
                sent["detection_frames"] != sent["frames"],
                received.get("detection_frames") != sent["frames"],
                received.get("detection_boxes") != sent.get("detection_boxes"),
                "detection_frame_sync=ok" not in (output / "encode.log").read_text(),
            )):
                failure = "detection frame association/count mismatch"
            elif args.capture and not (output / "decoded.png").is_file():
                failure = "decoded screenshot was not captured"
        report = {"elapsed_s": time.monotonic() - started, "failure": failure,
                  "encoder_returncode": encoder.returncode, "decoder_returncode": decoder.returncode,
                  "metrics": metrics, "temperature_samples": samples,
                  "image_captured": (output / "decoded.png").is_file()}
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    if failure or encoder.returncode or decoder.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
