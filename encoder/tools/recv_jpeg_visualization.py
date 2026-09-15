#!/usr/bin/env python3
"""Receive JPEG VideoTrans packets sent by mlvc_decode in UDP mode."""

import argparse
import os
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MAGIC = b"\xeb\x90"
TAIL = b"\xcd\xde"
HEADER_BYTES = 14
TAIL_BYTES = 2
PAYLOAD_BYTES = 1024


@dataclass
class FrameBuffer:
    total_packets: int
    total_size: int
    channel: int
    created: float = field(default_factory=time.monotonic)
    chunks: dict = field(default_factory=dict)

    def add(self, index: int, payload: bytes) -> bytes | None:
        self.chunks.setdefault(index, payload)
        if len(self.chunks) != self.total_packets:
            return None
        return b"".join(self.chunks[i] for i in range(self.total_packets))[: self.total_size]


def parse_packet(packet: bytes):
    if len(packet) < HEADER_BYTES + TAIL_BYTES:
        return None
    if packet[:2] != MAGIC or packet[-2:] != TAIL:
        return None
    index, total_packets, payload_size = struct.unpack_from("<HHH", packet, 2)
    total_size = struct.unpack_from("<I", packet, 8)[0]
    channel = packet[12]
    if total_packets == 0 or index >= total_packets:
        return None
    if payload_size > PAYLOAD_BYTES or len(packet) != HEADER_BYTES + payload_size + TAIL_BYTES:
        return None
    return index, total_packets, total_size, channel, packet[
        HEADER_BYTES : HEADER_BYTES + payload_size
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--save-dir", type=Path, default=Path())
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--window-name", default="MLVC VideoTrans")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--recv-buffer-mb", type=int, default=16)
    args = parser.parse_args()
    if args.timeout <= 0 or args.recv_buffer_mb <= 0:
        parser.error("timeout and recv buffer must be positive")
    if not args.no_display and os.name != "nt" and os.environ.get("DISPLAY", "") == "":
        parser.error("no DISPLAY; use --no-display or configure a graphical session")
    if args.save_dir != Path():
        args.save_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_RCVBUF, args.recv_buffer_mb * 1024 * 1024
    )
    sock.bind((args.bind_host, args.port))
    print(f"Listening for VideoTrans JPEG on {args.bind_host}:{args.port}", flush=True)

    frames = {}
    displayed = 0
    dropped = 0
    try:
        while True:
            now = time.monotonic()
            expired = [key for key, frame in frames.items() if now - frame.created > args.timeout]
            for key in expired:
                del frames[key]
                dropped += 1

            packet, source = sock.recvfrom(HEADER_BYTES + PAYLOAD_BYTES + TAIL_BYTES)
            parsed = parse_packet(packet)
            if parsed is None:
                continue
            index, total_packets, total_size, channel, payload = parsed
            key = (source[0], channel)
            frame = frames.get(key)
            if (
                frame is None
                or index == 0
                or frame.total_packets != total_packets
                or frame.total_size != total_size
            ):
                frame = FrameBuffer(total_packets, total_size, channel)
                frames[key] = frame
            message = frame.add(index, payload)
            if message is None:
                continue
            del frames[key]
            if len(message) < 8:
                dropped += 1
                continue
            jpeg_size = struct.unpack_from("<I", message, 0)[0]
            jpeg_end = 4 + jpeg_size
            if jpeg_end + 4 > len(message):
                dropped += 1
                continue
            image = cv2.imdecode(
                np.frombuffer(message[4:jpeg_end], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                dropped += 1
                continue
            frame_index = displayed
            if args.save_dir != Path():
                cv2.imwrite(str(args.save_dir / f"frame_{frame_index:06d}.jpg"), image)
            displayed += 1
            print(
                f"frame={frame_index} source={source[0]}:{source[1]} "
                f"jpeg={jpeg_size}B size={image.shape[1]}x{image.shape[0]} dropped={dropped}",
                flush=True,
            )
            if not args.no_display:
                cv2.imshow(args.window_name, image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
