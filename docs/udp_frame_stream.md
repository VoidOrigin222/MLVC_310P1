# UDP Frame Stream

The copied `mlvc_acl_cppv1` project keeps the original MLVC model, reference
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
the decoder thread only consumes complete messages. Raw YUV forwarding uses the
same pattern with a dedicated packet-send thread.

## 输出转发

JPEG/VideoTrans 转发方案已从 v1 代码中删除。当前解码器支持本地视频输出，以及 `raw_fp16_yuv444` 原始 FP16 YUV UDP 转发；RTSP 输出由 Issue #11 单独实现。

Run `mlvc_decode` first so it is blocked on the UDP port, then run
`mlvc_encode`. The example configs use Q8, GOP128, Reset32, LTR start 8,
LTR period 64, and LTR QP shift 8. `frame_num` can be reduced for a smoke test
or set to `-1` for all available frames.
