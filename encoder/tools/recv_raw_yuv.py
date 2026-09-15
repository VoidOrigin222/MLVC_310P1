#!/usr/bin/env python3
"""Receive MLVC raw FP16 YUV444 frames using the VideoTrans fragment envelope."""

import argparse
import queue
import socket
import struct
import threading
from pathlib import Path

MAGIC = b"\xeb\x90"
TAIL = b"\xcd\xde"
HEADER_BYTES = 14
TAIL_BYTES = 2
PAYLOAD_BYTES = 1024
RAW_KIND = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--save-dir", type=Path, required=True)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # A raw 1080p FP16 YUV444 frame is about 12.5 MB. Keep enough kernel
    # buffering for at least one frame while the writer persists the previous.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    sock.bind((args.bind_host, args.port))
    print(f"Listening on {args.bind_host}:{args.port}")

    write_queue = queue.Queue(maxsize=8)

    def write_frames() -> None:
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                return
            frame_index, source, width, height, padded_width, padded_height, raw = item
            output = args.save_dir / (
                f"frame_{frame_index:06d}_{width}x{height}_"
                f"{padded_width}x{padded_height}.fp16"
            )
            output.write_bytes(raw)
            print(
                f"frame={frame_index} source={source[0]}:{source[1]} bytes={len(raw)}",
                flush=True,
            )
            write_queue.task_done()

    writer = threading.Thread(target=write_frames, daemon=True)
    writer.start()

    chunks = {}
    expected_packets = 0
    expected_size = 0
    while True:
        packet, source = sock.recvfrom(HEADER_BYTES + PAYLOAD_BYTES + TAIL_BYTES)
        if len(packet) < HEADER_BYTES + TAIL_BYTES:
            continue
        if packet[:2] != MAGIC or packet[-2:] != TAIL:
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
        chunks.setdefault(packet_index, packet[HEADER_BYTES:HEADER_BYTES + payload_size])
        if len(chunks) != expected_packets:
            continue
        message = b"".join(chunks[index] for index in range(expected_packets))
        message = message[:expected_size]
        if len(message) < 26 or message[0] != RAW_KIND:
            chunks = {}
            continue
        frame_index = struct.unpack_from("<i", message, 1)[0]
        width, height, padded_width, padded_height = struct.unpack_from("<IIII", message, 5)
        pixel_format = message[21]
        raw_size = struct.unpack_from("<I", message, 22)[0]
        raw = message[26:26 + raw_size]
        if pixel_format != 1 or len(raw) != raw_size:
            chunks = {}
            continue
        write_queue.put(
            (frame_index, source, width, height, padded_width, padded_height, raw)
        )
        chunks = {}


if __name__ == "__main__":
    raise SystemExit(main())
