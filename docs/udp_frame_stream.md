# UDP Frame Stream

The copied `mlvc_acl_cpp_udp` project keeps the original MLVC model, reference
state, and official rANS implementation. It changes only the transport path:

```text
encoder: read one input frame -> Encoder OM -> rANS -> UDP frame datagram
                                                         |
decoder: write one PNG frame <- Decoder OM <- rANS <-----+
```

The encoder sends one logical header message, one logical message for each
completed frame, and one end message. Each message is fragmented into 1024-byte
UDP payloads using the VideoTrans packet layout:

```text
[EB 90] [packet_index:u16 LE] [total_packets:u16 LE]
        [payload_size:u16 LE] [total_message_size:u32 LE]
        [channel:u8] [reserved:u8] [payload <= 1024] [CD DE]
```

The reassembled MLVC message starts with a kind byte (`1` header, `2` frame,
`3` end). A frame message then contains frame index, frame type, Q index,
payload length, and the complete official MLVC rANS payload. The receiver
validates magic, tail, fragment bounds, and all reassembly lengths before
passing a frame to the decoder. The receiver socket uses an 8 MiB kernel
receive buffer, matching the VideoTrans receiver's burst-oriented design.
UDP itself does not guarantee delivery; a missing fragment causes the current
logical message to be discarded when the next message starts. MLVC receive
uses a dedicated `recvfrom`/reassembly thread and a condition-variable queue;
the decoder thread only consumes complete messages. VideoTrans forwarding uses
the same pattern with a dedicated packet-send thread.

## VideoTrans 转发

解码器可将 `MLVCDecoder` 的 `x_hat` 转成 JPEG，并按 VideoTrans
`to_visualization` 格式转发。设置 `format = "none"` 可禁用本地 PNG/MP4
输出，再设置 `forward_host` 和 `forward_port`（VideoTrans viewer 默认端口为
`50000`）。转发负载包含 JPEG 和空目标框列表，接收端可直接使用
`VideoTrans/tools/recv_visualization.py` 接收。

## 原始 FP16 YUV 转发

设置 `forward_mode = "raw_fp16_yuv444"` 可绕过 FP16 YUV 到 BGR 的转换和
JPEG 编码。解码主线程只复制 `x_hat` 到有界队列，独立发送线程负责 UDP
分片；接收端示例为 `tools/recv_raw_yuv.py`，输出原始 FP16 NCHW 三平面文件。

1920x1088 的 YUV444 FP16 帧约 12.53 MB，30 FPS 约需 3.0 Gbit/s。发送队列
最多缓存 4 帧，网络跟不上时仅丢弃远端待发送帧；`forward_dropped_frames`
报告丢弃数量，本地解码不受影响。

若远端机器需要继续接入 VideoTrans 可视化端，运行
`tools/recv_forward_raw_yuv.py`。它在远端独立线程中完成 YUV→BGR、JPEG 和
VideoTrans UDP 转发：

```bash
python3 tools/recv_forward_raw_yuv.py \
  --listen-port 50000 \
  --forward-host 127.0.0.1 \
  --forward-port 50001
```

如果链路带宽有限，推荐直接使用解码端的 `forward_mode = "jpeg_async"`，由
解码端独立转发线程完成 YUV→BGR 和 JPEG，远端 PC 运行
`tools/recv_jpeg_visualization.py`：

```bash
python3 tools/recv_jpeg_visualization.py --port 50000
```

无图形界面时加 `--no-display --save-dir /path/to/jpg`。

Run `mlvc_decode` first so it is blocked on the UDP port, then run
`mlvc_encode`. The example configs use Q8, GOP128, Reset32, LTR start 8,
LTR period 64, and LTR QP shift 8. `frame_num` can be reduced for a smoke test
or set to `-1` for all available frames.
