#!/usr/bin/env python3
"""Receive MLVC raw FP16 YUV frames and forward them as VideoTrans JPEG frames.

The raw stream is produced by mlvc_acl_cpp_udp with
forward_mode=raw_fp16_yuv444. Conversion and JPEG encoding happen on this
remote process, so the decoder machine does not pay that CPU cost.
"""

import argparse
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MAGIC = b"\xeb\x90"
TAIL = b"\xcd\xde"
HEADER_BYTES = 14
TAIL_BYTES = 2
PAYLOAD_BYTES = 1024
RAW_KIND = 4


def packetize(payload: bytes):
    total_packets = (len(payload) + PAYLOAD_BYTES - 1) // PAYLOAD_BYTES
    if total_packets > 0xFFFF:
        raise ValueError("VideoTrans payload is too large")
    total_size = len(payload)
    for index in range(total_packets):
        chunk = payload[index * PAYLOAD_BYTES : (index + 1) * PAYLOAD_BYTES]
        header = struct.pack(
            "<2sHHHI BB",
            MAGIC,
            index,
            total_packets,
            len(chunk),
            total_size,
            0,
            0,
        )
        yield header + chunk + TAIL


def yuv444_fp16_to_bgr(raw: bytes, width: int, height: int, padded_width: int, padded_height: int):
    expected = 3 * padded_width * padded_height
    array = np.frombuffer(raw, dtype=np.float16)
    if array.size != expected:
        raise ValueError(f"raw tensor has {array.size} elements, expected {expected}")
    planes = array.reshape(3, padded_height, padded_width).astype(np.float32, copy=False)
    y = planes[0, :height, :width]
    cb = planes[1, :height, :width]
    cr = planes[2, :height, :width]
    # Matches mlvc::io::ConvertTensorToBgr (BT.709, values in [0, 1]).
    r = y + (2.0 - 2.0 * 0.2126) * (cr - 0.5)
    b = y + (2.0 - 2.0 * 0.0722) * (cb - 0.5)
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.round(np.stack((b, g, r), axis=-1) * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--listen-port", "--port", type=int, default=50000)
    parser.add_argument("--forward-host", required=True, help="VideoTrans destination host")
    parser.add_argument("--forward-port", type=int, default=50000)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--receive-buffer-mb", type=int, default=16)
    parser.add_argument("--save-raw-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.queue_size < 1 or args.receive_buffer_mb < 1:
        parser.error("queue and receive buffer sizes must be positive")
    if args.save_raw_dir:
        args.save_raw_dir.mkdir(parents=True, exist_ok=True)

    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_socket.setsockopt(
        socket.SOL_SOCKET, socket.SO_RCVBUF, args.receive_buffer_mb * 1024 * 1024
    )
    receive_socket.bind((args.bind_host, args.listen_port))
    forward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    forward_address = (args.forward_host, args.forward_port)
    frames = queue.Queue(maxsize=args.queue_size)
    dropped = 0
    forwarded = 0
    started = time.monotonic()

    def worker() -> None:
        nonlocal forwarded
        while True:
            item = frames.get()
            if item is None:
                frames.task_done()
                return
            frame_index, width, height, padded_width, padded_height, raw = item
            try:
                if args.save_raw_dir:
                    path = args.save_raw_dir / (
                        f"frame_{frame_index:06d}_{width}x{height}_"
                        f"{padded_width}x{padded_height}.fp16"
                    )
                    path.write_bytes(raw)
                bgr = yuv444_fp16_to_bgr(raw, width, height, padded_width, padded_height)
                ok, encoded = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if not ok:
                    raise RuntimeError("cv2.imencode failed")
                payload = struct.pack("<I", len(encoded)) + encoded.tobytes() + struct.pack("<I", 0)
                for packet in packetize(payload):
                    forward_socket.sendto(packet, forward_address)
                forwarded += 1
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"forward frame={frame_index} jpeg={len(encoded)}B "
                    f"fps={forwarded / elapsed:.2f}",
                    flush=True,
                )
            except Exception as exc:  # keep receiving later frames
                print(f"forward frame={frame_index} failed: {exc}", flush=True)
            finally:
                frames.task_done()

    thread = threading.Thread(target=worker, name="raw-yuv-forward", daemon=True)
    thread.start()
    print(
        f"Listening on {args.bind_host}:{args.listen_port}; "
        f"forwarding VideoTrans to {args.forward_host}:{args.forward_port}",
        flush=True,
    )

    chunks = {}
    expected_packets = 0
    expected_size = 0
    try:
        while True:
            packet, _source = receive_socket.recvfrom(HEADER_BYTES + PAYLOAD_BYTES + TAIL_BYTES)
            if (
                len(packet) < HEADER_BYTES + TAIL_BYTES
                or packet[:2] != MAGIC
                or packet[-2:] != TAIL
            ):
                continue
            packet_index, total_packets, payload_size = struct.unpack_from("<HHH", packet, 2)
            total_size = struct.unpack_from("<I", packet, 8)[0]
            if total_packets == 0 or packet_index >= total_packets:
                continue
            if len(packet) != HEADER_BYTES + payload_size + TAIL_BYTES:
                continue
            if packet_index == 0:
                chunks = {}
                expected_packets = total_packets
                expected_size = total_size
            if total_packets != expected_packets or total_size != expected_size:
                continue
            chunks.setdefault(packet_index, packet[HEADER_BYTES : HEADER_BYTES + payload_size])
            if len(chunks) != expected_packets:
                continue
            message = b"".join(chunks[index] for index in range(expected_packets))[:expected_size]
            chunks = {}
            if len(message) < 26 or message[0] != RAW_KIND:
                continue
            frame_index = struct.unpack_from("<i", message, 1)[0]
            width, height, padded_width, padded_height = struct.unpack_from("<IIII", message, 5)
            if message[21] != 1:
                continue
            raw_size = struct.unpack_from("<I", message, 22)[0]
            raw = message[26 : 26 + raw_size]
            if len(raw) != raw_size:
                continue
            try:
                frames.put_nowait((frame_index, width, height, padded_width, padded_height, raw))
            except queue.Full:
                dropped += 1
                print(f"drop frame={frame_index} queue_full total_dropped={dropped}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        receive_socket.close()
        frames.put(None)
        frames.join()
        forward_socket.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
