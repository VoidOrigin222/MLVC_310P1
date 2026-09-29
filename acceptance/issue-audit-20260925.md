# GitHub Issue 审查（更新于 2026-09-30）

范围：读取 `VoidOrigin222/MLVC_310P1` 的全部 16 个 issue 及开放 issue 评论。本轮以 Grifcc 明确报告的问题和复现条件为验收范围，不把后续扩展建议一律当作当前阻塞。当前 #1–#8、#10、#12 为 closed；#9、#11、#13–#16 仍为 open。已在 #13–#16 和 #11 回复本轮修复与验证；已关闭 issue 保持原状。

工作树在审查前已有多处未提交改动；本记录只把与 issue 直接相关的修改列为本次审查结论，不把其他既有改动归为本轮新增。

## #4 本地码流未限制 payload，且未校验帧号和 header

GitHub 状态：open。结论：Grifcc 指出的码流尺寸与已加载 decoder 模型输出 shape 不匹配未拒绝问题已修复；校验接入解码入口，反例测试和板端全量测试通过。

代码位置：

- `mlvc/src/io/mlvc_bitstream.cc:55-115`：header 范围、有限 FPS/码率、QP/GOP/LTR、frame type 与连续帧号验证。
- `mlvc/src/io/mlvc_bitstream.cc:219-252`：legacy reader 限制 payload 大小、检查剩余文件字节后才分配。
- `mlvc/src/io/mlvc_bitstream.cc:361-410`：新版 `.mlvc` reader 同样先检查 payload 上限和剩余长度，再读入；frame index 必须连续。
- `mlvc/src/io/udp_frame_transport.cc:67-123,135-210,219-389`：UDP/RTP 消息头与帧序校验、帧长度限制、首帧类型和 QP 校验。
- `common/src/transport/udp_message_transport.cc:53-63,409-480`：UDP fragment 长度/数量/保留字段、重复 fragment 内容及完整消息长度校验。
- `tools/cpp/tests/test_mlvc_bitstream_validation.cc`、`tools/cpp/tests/test_udp_fragment_validation.cc`：非法头、帧号、payload 和 UDP 分片负向用例。

对抗性审查：payload 长度在分配前受到分辨率上限和文件剩余长度双重约束；短读会失败，不会将不完整 payload 交给解码器。`ValidateMlvcDecoderOutputShape` 会比较 visible/coded dimensions；测试覆盖模型 shape 过小和通道错误，均在解码前拒绝。

## #7 配置项接受后不生效，部分数值范围未验证

GitHub 状态：open。结论：Issue 正文中的配置校验已覆盖；Grifcc 评论提到的 RAW YUV 发送功能已从当前仓库移除。rate-control 整数预算也在配置解析和控制器构造处设置了安全上限。

代码位置：

- `application/src/cli/encode_config.cc:27-91,96-216`：严格 typed getter、profile/type 检查、有限 FPS/target bitrate、QP/LTR/队列范围检查。
- `application/src/cli/decode_config.cc:27-81,87-200`：解码配置 typed getter、数值 FPS、RTSP/transport/profile/队列范围检查。
- `application/src/stream/mlvc/encode.cc:124-205`：warmup 帧实际排除在性能测量之外。
- `application/src/stream/mlvc/decode.cc:621-625`：解码流水线总容量受 `frame_buffer_slots` 与 queue capacity 共同限制。
- `application/src/stream/mlvc/encode/encode_frame.cc:46-52`：LTR QP shift 后最终 QP 被限制在配置的 `[min_qp, max_qp]` 中。
- `tools/cpp/tests/test_encode_config.cc`、`tools/cpp/tests/test_decode_config.cc`：未知 profile、错误 TOML 类型、FPS 和负范围用例。

对抗性审查：普通 NaN/Inf、错误类型和区间反转已拒绝；“有限且极大”的 `target_bitrate_bps` 在进入整数 leaky-bucket 预算前也会被上限检查拒绝。仓库内已无 `RawYuvUdpSender`/raw-YUV UDP 发送路径。

## #9 MLVC 码流与 RTP 载荷格式设计（版本 1）

GitHub 状态：open。结论：Grifcc 评论中的三个复现问题已处理：逐单元超时、完整媒体单元/帧去重与重排、按 FPS/PTS 生成 RTP 时间戳且避免有符号乘法溢出。Issue 描述中的未来扩展不作为本轮阻塞。

代码位置：

- `common/src/transport/rtp_mlvc.cc:39-72`：当前 `ML` 自定义 fragment descriptor 编解码。
- `mlvc/src/io/udp_frame_transport.cc:13-16,67-123,219-389`：现有 header/frame/end 消息格式。
- `common/src/transport/rtp_mlvc.cc:85-171`：RTP fragment 重组；当前按 descriptor 的 legacy 单元语义处理。
- `common/src/transport/rtp_mlvc.cc:445-500`：RTP header parser 与现有私有 RTCP config 请求格式。

对抗性审查：`test_rtp_reassembler` 覆盖清理扫描边界前后仍活跃的单元；`test_rtp_receiver_recovery` 覆盖后帧/EOS 先到、前帧迟到及完整单元重复；`test_rtp_sender` 验证 25 FPS 对应 3600 个 90 kHz ticks。RTP EOS 以独占帧数为边界等待，超时则明确报缺帧，不提前报告正常 EOF。

## #10 UDP 发送没有 packet pacing，会形成瞬时带宽突发

GitHub 状态：open。结论：Grifcc 复现的 pacer 重复入账已通过负 token 债务修正；UDP/RTP 队列计入待加入和正在发送的数据，pacer/wire 统计计入 IPv4+UDP 开销。板端速率回归通过。

代码位置：

- `common/src/transport/udp_pacer.cc:15-53`：token bucket 与单调时钟发送间隔。
- `common/src/transport/udp_message_transport.cc:131-178,218-272`：按包含 UDP/IP 开销的字节数执行 pacing，限制完整消息队列字节数/排队时延；队列错误向调用方传播。
- `common/src/transport/rtp_mlvc.cc:233-281,326-383`：RTP 完整 unit 预留队列额度，不拆散丢弃 fragment，按 packet pacing 并统计 wire bytes/queue delay/socket block。
- `tools/cpp/tests/test_rtp_sender.cc`、`tools/cpp/tests/test_udp_sender_flush_error.cc`：pacing、完整发送和错误传播测试。

对抗性审查：`test_udp_pacer_rate` 以 80 kbit/s、21 个 1000-byte 消费量检查累计等待预算；UDP/RTP sender 保留发送中额度，排队时延判断计入当前待加入消息。RTCP 拥塞反馈属于后续能力，不计入 Grifcc 此条 pacing 问题的完成条件。

## #11 功能逻辑问题：RTSP demo 替代 JPEG/UDP 显示

GitHub 状态：open。结论：功能实现与 1080p 全链路验收已记录；GitHub issue 没有关闭。

代码位置：

- `application/src/cli/decode_config.cc:129-151`：RTSP 输出与 `dvpp`/`libx264` 后端配置。
- `application/src/stream/mlvc/decode.cc:72-185,500-537,771-809`：异步 NV12/VENC 发布、显式关闭和丢帧统计。
- `mlvc/src/io/dvpp_h264_encoder.cc`：DVPP H.264 VENC 生命周期和编码路径。
- `mlvc/src/io/video_io.cc`：板端检测到 FFmpeg 开发库时，`RtspVideoPublisher` 通过
  libavformat/libavcodec 直接完成 RTSP 发布；没有开发库的主机才使用兼容子进程路径。
- `tools/cpp/tests/test_rtsp_video_publisher.cc`：本地 RTSP 握手服务器验证直通 H.264、
  NV12/libx264 编码和交错 RTP 数据写出。
- `acceptance/issue11-venc-20260925/fullchain-537-39220-20260925/实测说明.md`：537/537 解码帧、29.7511 FPS、VENC 发布 537 帧、`forward_dropped_frames=0`。

对抗性审查：PC RTSP 客户端日志只有 486 帧，但客户端在流已开始后才完成握手；设备日志证明发布端 537 帧且转发丢帧为 0，因此不能据此认定 PC 侧或 decoder 丢了 51 帧。该数据来自归档实测，不是本轮重跑。针对后续“不能使用 pipe、应直接使用 libavformat”的反馈，板端已安装 `ffmpeg-devel`，构建显示 `RTSP publisher backend: libavformat`，直连测试和 23 项 CTest 全部通过。

## #12 流水线重排表绕过配置的有界队列

GitHub 状态：open。结论：总在途额度包含执行中、重排表已完成和待消费结果；板端对应测试通过；未 push/关闭。

代码位置：

- `common/src/framework/streaming_pipeline.cc:82-100`：输入提交先申请总在途额度，消费输出后释放。
- `common/src/framework/streaming_pipeline.cc:112-166`：worker/reorder/output 共用额度；有序发布和异常传播。
- `tools/cpp/tests/test_streaming_pipeline_order.cc:50-102`：阻塞首项、填满后续完成结果、断言下一次输入阻塞并在释放额度后继续。

对抗性审查：测试覆盖首序号 worker 长时间阻塞的关键场景，确认 completed-results 不能绕过 capacity；本轮 310P1 全量 CMake/CTest 包含该用例并通过。

## 本轮验证

- `git diff --check` 通过。
- 编码端 310P1 Release 全量构建成功；本轮的全量 CTest 共 22 项。
- 新增的 #9 重排/EOS/重复帧针对性回归在板端通过；全量 RTP 相关用例 5/5 通过。
- #11 的 537 帧全链路记录为既有实测（编码 33.5355 FPS，解码 29.7511 FPS，VENC 发布 537 帧、转发丢帧 0）；本轮未重跑真实视频链路。
- 未向 GitHub 写评论、未 push、未关闭任何 issue。
