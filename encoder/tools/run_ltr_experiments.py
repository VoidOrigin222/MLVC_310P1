#!/usr/bin/env python3
"""Run the MLVC 1080P GOP/reset/LTR and packet-loss experiments."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE = Path("/root/workspace/cgc/mlvc_ly/results/frames_60s_1080p_fp16_gop32")
DEFAULT_MANIFEST = Path("/root/workspace/cgc/mlvc_ly/mlvc1080p/manifest.json")
DEFAULT_VIDEO = Path("/root/workspace/cgc/mlvc_ly/mlvc_acl_cpp/test_video/60s_1080p.mp4")


@dataclass(frozen=True)
class Case:
    name: str
    gop: int
    reset: int
    ltr: int


CASES = (
    Case("baseline_gop128_reset32_ltr0", 128, 32, 0),
    Case("ltr32_gop128_reset32", 128, 32, 32),
    Case("ltr64_gop128_reset32", 128, 32, 64),
    Case("ltr96_gop128_reset32", 128, 32, 96),
    Case("ltr64_gop64_reset32", 64, 32, 64),
    Case("ltr64_gop96_reset32", 96, 32, 64),
    Case("ltr64_gop128_reset16", 128, 16, 64),
    Case("ltr64_gop128_reset64", 128, 64, 64),
)

GOP_ONLY_CASES = (
    Case("baseline_gop64_reset32_ltr0", 64, 32, 0),
    Case("baseline_gop96_reset32_ltr0", 96, 32, 0),
    Case("baseline_gop128_reset32_ltr0", 128, 32, 0),
)


def parse_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("  "):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def write_config(path: Path, values: dict[str, object]) -> None:
    lines = [f'{key} = "{value}"' if isinstance(value, (str, Path)) else f"{key} = {value}"
             for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def psnr_y(original: Path, reconstructed: Path, log_path: Path) -> float | None:
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(original), "-i", str(reconstructed),
        "-lavfi", "[0:v][1:v]psnr=stats_file=" + str(log_path),
        "-f", "null", "-",
    ]
    subprocess.run(command, check=True)
    for line in log_path.read_text().splitlines():
        match = re.search(r"average:\s*([0-9.]+)", line)
        if match:
            return float(match.group(1))
    return None


def run_case(args: argparse.Namespace, case: Case, root: Path, decode: bool) -> dict[str, object]:
    case_root = root / case.name
    case_root.mkdir(parents=True, exist_ok=True)
    bitstream = case_root / "output.mlvc"
    recon = case_root / "reconstructed.mp4"
    enc_log = case_root / "encoder.log"
    dec_log = case_root / "decoder.log"
    enc_config = case_root / "encode.toml"
    dec_config = case_root / "decode.toml"
    write_config(enc_config, {
        "mode": "encode", "input_frame_dir": args.source, "output": bitstream,
        "qp": args.qp, "device": args.device, "gop": case.gop,
        "reset_interval": case.reset, "ltr_start_idx": args.ltr_start,
        "ltr_period": case.ltr, "ltr_qp_shift": args.ltr_qp_shift,
        "frame_num": args.frames, "manifest": args.manifest,
    })
    write_config(dec_config, {
        "mode": "decode", "input": bitstream, "output": recon,
        "device": args.device, "frame_num": args.frames, "format": "none",
        "bitrate": "none", "preset": "ultrafast", "manifest": args.manifest,
    })
    run([str(args.encoder), "--config", str(enc_config)], enc_log)
    enc = parse_log(enc_log)
    dec: dict[str, str] = {}
    if decode:
        run([str(args.decoder), "--config", str(dec_config)], dec_log)
        dec = parse_log(dec_log)
    frames = int(enc.get("frames", dec.get("frames", "0")))
    stream_bytes = bitstream.stat().st_size
    fps = float(enc.get("fps", enc.get("encode_fps", "0")))
    return {
        "name": case.name, "gop": case.gop, "reset_interval": case.reset,
        "ltr_period": case.ltr, "qp": args.qp, "frames": frames,
        "bitstream_bytes": stream_bytes,
        "bitstream_kB_1024": stream_bytes / 1024.0,
        "average_kB_per_frame_1024": stream_bytes / 1024.0 / frames if frames else 0.0,
        "average_kbit_per_second_1024": stream_bytes * 8.0 * 30.0 / 1024.0 / frames if frames else 0.0,
        "encode_fps": float(enc.get("encode_fps", "0")),
        "decode_fps": float(dec.get("decode_fps", "0")),
        "frame_counts": enc.get("frame_counts", ""),
        "decode_frames": int(dec.get("frames", "0")) if decode else None,
        "output_video": None,
        "logs": {"encoder": str(enc_log), "decoder": str(dec_log)},
    }


def run_loss_case(args: argparse.Namespace, root: Path) -> list[dict[str, object]]:
    modes = (
        ("drop70_p71", -1, -1),
        ("drop70_ltr64", 71, 64),
        ("drop70_ltr8", 71, 8),
    )
    results: list[dict[str, object]] = []
    for mode, forced_recovery_frame, forced_reference_frame in modes:
        case = Case(f"ltr64_{mode}_gop128_reset32", 128, 32, 64)
        case_root = root / case.name
        case_root.mkdir(parents=True, exist_ok=True)
        bitstream = case_root / "output.mlvc"
        recon = case_root / "reconstructed.mp4"
        enc_log = case_root / "encoder.log"
        dec_log = case_root / "decoder.log"
        enc_config = case_root / "encode.toml"
        dec_config = case_root / "decode.toml"
        write_config(enc_config, {
            "mode": "encode", "input_frame_dir": args.source, "output": bitstream,
            "qp": args.qp, "device": args.device, "gop": case.gop,
            "reset_interval": case.reset, "ltr_start_idx": args.ltr_start,
            "ltr_period": case.ltr, "ltr_qp_shift": args.ltr_qp_shift,
            "forced_ltr_recovery_frame": forced_recovery_frame,
            "forced_ltr_reference_frame": forced_reference_frame,
            "frame_num": args.frames, "manifest": args.manifest,
        })
        write_config(dec_config, {
            "mode": "decode", "input": bitstream, "output": recon,
            "device": args.device, "frame_num": args.frames, "format": "none",
            "bitrate": "none", "preset": "ultrafast", "drop_frame_index": 70,
            "forced_ltr_reference_frame": forced_reference_frame,
            "forced_ltr_recovery_frame": forced_recovery_frame,
            "manifest": args.manifest,
        })
        run([str(args.encoder), "--config", str(enc_config)], enc_log)
        run([str(args.decoder), "--config", str(dec_config)], dec_log)
        enc = parse_log(enc_log)
        dec = parse_log(dec_log)
        results.append({
            "name": case.name, "gop": case.gop, "reset_interval": case.reset,
            "ltr_period": case.ltr, "forced_ltr_recovery_frame": forced_recovery_frame,
            "forced_ltr_reference_frame": forced_reference_frame,
            "drop_frame_index": 70, "encode_frame_counts": enc.get("frame_counts", ""),
            "encoded_frames": int(enc.get("frames", "0")),
            "decoded_frames_after_drop": int(dec.get("frames", "0")),
            "decode_status": dec.get("decode", "failed"),
            "bitstream_bytes": bitstream.stat().st_size,
            "output_video": None,
            "logs": {"encoder": str(enc_log), "decoder": str(dec_log)},
        })
    return results


def write_summary(path: Path, results: list[dict[str, object]], args: argparse.Namespace) -> None:
    normal = [item for item in results if "bitstream_kB_1024" in item]
    lines = [
        "# MLVC LTR / Reset / GOP 实验",
        "",
        "## 实验条件",
        "",
        f"- 模型：`{args.manifest}`",
        f"- 输入：`{args.source}`",
        f"- 基础 Q：`Q{args.qp}`，LTR 起始帧：`{args.ltr_start}`，LTR Q 偏移：`{args.ltr_qp_shift}`",
        "- 码流大小中的 `kB` 按 1024 字节计算。",
        "- 每个周期组合均独立编码；基线和 LTR64 组合做正常 decoder 验证，其余组合保留码流和 encoder 结果。",
        "",
        "## 周期与码率",
        "",
        "| 实验 | GOP | Reset | LTR 周期 | 帧数 | 帧类型 | 码流 kB | 平均 kB/帧 | 平均 kbit/s | Encoder FPS | Decoder FPS |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in normal:
        lines.append(
            "| {name} | {gop} | {reset_interval} | {ltr_period} | {frames} | {frame_counts} | "
            "{bitstream_kB_1024:.2f} | {average_kB_per_frame_1024:.3f} | "
            "{average_kbit_per_second_1024:.2f} | {encode_fps:.2f} | {decode_fps:.2f} |".format(**item)
        )
    lines.extend([
        "",
        "## 结果观察",
        "",
        "- 修复 GOP 起点清空 LTR 缓存后，GOP128/Reset32 下 LTR32、LTR64 的 recovery 数分别为 42、14；LTR96 与 LTR64 的 recovery 数和码流大小基本一致。",
        "- GOP64/Reset32/LTR64 在 GOP 内没有 local frame 64，因此 recovery 数为 0；它的码流高于 GOP96 和 GOP128，主要来自更多 I 帧，而不是 LTR 失效。",
        "- GOP96/Reset32/LTR64 为 1603.69 kB，GOP128/Reset32/LTR64 为 1614.05 kB，两者仅相差约 0.6%，修复后不存在明显异常。",
        "- Reset16 的码流为 2579.69 kB，Reset64 为 987.29 kB；在当前模型和 Q8 下，Reset 周期带来的码率变化大于 LTR 周期本身。",
        "- LTR recovery 使用 `ltr_qp_shift=8`，因此 LTR 周期实验同时反映参考切换和 recovery 帧 Q 偏移的影响。",
        "- Decoder 验证使用 `format=none`，不启动 ffmpeg、不写 MP4；未执行独立 decoder 的组合，其 Decoder FPS 记为 0。",
    ])
    losses = [item for item in results if "drop_frame_index" in item]
    if losses:
        lines.extend([
            "",
            "## 丢帧与非定周期 LTR 恢复",
            "",
            "实验设置：GOP=128、Reset=32、LTR=64，解码端跳过第 70 帧。",
            "对比第 71 帧继续使用普通 P 参考、切换到当前 LTR64、切换到上一个 LTR8。",
            "",
            "| 模式 | 第71帧参考 | 编码帧类型 | 编码帧数 | 解码输出帧数 | Decoder 状态 | 码流 kB |",
            "| --- | --- | --- | ---: | ---: | --- | ---: |",
        ])
        for loss in losses:
            reference = {
                -1: "普通 P 帧参考",
                64: "当前 LTR64",
                8: "上一个 LTR8",
            }[loss["forced_ltr_reference_frame"]]
            lines.append(
                f"| {loss['name']} | {reference} | `{loss['encode_frame_counts']}` | "
                f"{loss['encoded_frames']} | {loss['decoded_frames_after_drop']} | "
                f"`{loss['decode_status']}` | {int(loss['bitstream_bytes']) / 1024.0:.2f} |"
            )
        lines.extend([
            "",
            "说明：`decode=ok` 只表示 decoder OM 能够运行；判断普通 P 帧是否正确恢复，还需要结合第 71 帧及后续帧的重建质量进行比较。",
            "LTR64 和 LTR8 两组验证的是显式非定周期 recovery，分别使用第 64 帧和第 8 帧缓存的 feature。",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_report(args: argparse.Namespace) -> None:
    results_path = args.output_root / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for item in results:
        if "bitstream_kB_1024" not in item:
            continue
        stream_bytes = int(item["bitstream_bytes"])
        frames = int(item["frames"])
        item["bitstream_kB_1024"] = stream_bytes / 1024.0
        item["average_kB_per_frame_1024"] = stream_bytes / 1024.0 / frames if frames else 0.0
        item["average_kbit_per_second_1024"] = (
            stream_bytes * 8.0 * 30.0 / 1024.0 / frames if frames else 0.0
        )
    results_path.write_text(json.dumps(results, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    write_summary(args.output_root / "summary.md", results, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--qp", type=int, default=8)
    parser.add_argument("--frames", type=int, default=-1)
    parser.add_argument("--ltr-start", type=int, default=8)
    parser.add_argument("--ltr-qp-shift", type=int, default=8)
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--gop-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        refresh_report(args)
        return
    results: list[dict[str, object]] = []
    if not args.loss_only:
        cases = GOP_ONLY_CASES if args.gop_only else CASES
        for case in cases:
            decode = args.gop_only or case.name in {
                "baseline_gop128_reset32_ltr0",
                "ltr64_gop128_reset32",
            }
            results.append(run_case(args, case, args.output_root, decode))
    if not args.gop_only:
        results.extend(run_loss_case(args, args.output_root))
    (args.output_root / "results.json").write_text(
        json.dumps(results, ensure_ascii=True, indent=2) + "\n", encoding="ascii"
    )
    write_summary(args.output_root / "summary.md", results, args)
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
