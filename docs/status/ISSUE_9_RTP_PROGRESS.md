# Issue #9 RTP 进度

更新日期：2026-09-26

## 已完成

- 新增 `RtpMessageReceiver`，负责 UDP socket、RTP v2 头解析、SSRC 校验和 MLVC payload 转交。
- 增加首包初始化、连续包、跳号丢包、重复包和乱序包统计。
- 乱序分片仍交给 `RtpMlvcReassembler`，只有完整媒体单元才返回。
- 新增 `test_rtp_message_receiver_sequence.cc`，覆盖首包初始化、连续序列号、丢包、重复、乱序重组和 SSRC 变化拒绝。
- 已将测试注册到 CMake。

## P1 验证

在 Ascend 310P1 编码端（Release）上通过：

```text
rtp_mlvc ......................... Passed
rtp_reassembler .................. Passed
rtp_message_receiver_sequence .... Passed
rtp_sender ....................... Passed
100% tests passed, 0 tests failed
```

## 实现状态

RTP sender、编码器输出、解码器 `input_transport_mode = "rtp"` 接入以及双机验收均已完成。
## 发送端和运行时接入（2026-09-12）

- 新增 `RtpMlvcSender`，支持 RTP v2、payload type 96、16 位序列号、90 kHz 时间戳、稳定 SSRC 和 1200 字节分片。
- 新增 MLVC RTP sender/receiver 适配器，header、frame、end 消息继续使用现有 MLVC 字节格式。
- 编码器支持 `output_transport_mode = "rtp"`，解码器支持 `input_transport_mode = "rtp"`，默认仍为旧 UDP。
- 新增 `test_rtp_sender`，P1 上四个 RTP 测试全部通过。

## 双机端到端验收（2026-09-12）

- 120 帧：编码端 `frames=120`、解码端 `frames=120`，双方均正常退出并报告 `encode=ok` / `decode=ok`。
- 537 帧：编码端 `frames=537`、解码端 `frames=537`，双方均正常退出并报告 `encode=ok` / `decode=ok`。
- 验收使用 Ascend 310P1 编码端和解码端，RTP 端口分别为 39191 和 39192。

Issue #9 的实现和双机验收已完成。

## Grifcc 对抗性评论回归（2026-09-26）

- 重组器按每个待处理单元的最近活动时间淘汰，不再因为全局清理周期误删刚开始的帧。
- 已完成单元有界去重；接收端按 frame ID 有界重排，完整重复帧不会再次进入 decoder。
- 编码时间戳由 64 位 PTS/当前 FPS 计算，再按 RTP 32 位时间戳自然回绕；25 FPS 回归为每帧 3600 ticks。
- RTP EOS 携带独占 `next_frame_id` 边界。若 EOS 先于前序帧完成，接收端继续等待该边界；缺帧超时会报错而不是提前报告正常结束。
- 新增对应乱序/EOS/重复帧测试；310P1 RTP 相关 5 项及全量 22 项 CTest 通过。
