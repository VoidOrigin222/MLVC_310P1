# Issue #11 RTSP 输出进度

更新时间：2026-09-26

## 实现

解码器支持 `output_transport_mode = "rtsp"`。`RtspVideoPublisher` 将 FP16 YUV444 转为 BGR 后交给后台线程，在线程内写入 FFmpeg/libx264 管道。发布队列有界，队列满时累计 `dropped_frames`；工作线程和 FFmpeg 错误通过 `Close()` 传播。

RTSP 发布默认使用 UDP 媒体传输（RTSP 控制连接仍使用 TCP 8554，RTP/RTCP 使用 MediaMTX 的 UDP 8000/8001）。

当前实现保持单后台线程：颜色转换和管道写入在同一线程串行执行。此前尝试的两级异步队列已回退，因为实测没有带来端到端收益。

示例配置：

- `configs/decoder.toml`
- `configs/decoder.toml`

## 自动化验证

在两台 Ascend 310P1 设备上执行：

```text
cmake -S . -B build -DMLVC_BUILD_APPS=ON -DBUILD_TESTING=ON
cmake --build build -j2
ctest --output-on-failure
```

两端均成功构建；`rtsp_video_publisher` 测试通过。解码端回退构建还伴随 Ascend 头文件的既有 `-Wpedantic` 零长度数组警告和时钟偏移提示，不影响构建结果。

## MediaMTX 实测（2026-09-13）

本机 MediaMTX v1.21.0 监听 TCP 8554（RTSP 控制连接）；媒体数据改用 UDP RTP/RTCP 端口 8000/8001，通过 SSH 反向隧道转发到 P1：

- 720P / 120 帧：解码 60.16 FPS，发布器丢帧 58，FFmpeg 正常退出。
- 1080P / 120 帧：解码 24.42 FPS，发布器丢帧 65，FFmpeg 正常退出。
- MediaMTX 路径 `mlvc720` 和 `mlvc1080` 均返回 `RTSP/1.0 200 OK`。

UDP 媒体传输复测（1080P / 120 帧）：编码 31.78 FPS，解码 24.52 FPS，发布器丢帧 65，FFmpeg 正常退出。与 TCP 模式的 24.42 FPS 基本一致，说明传输协议切换没有解决当前性能瓶颈。

720P / 120 帧 UDP 复测：编码 79.19 FPS，解码 60.15 FPS，发布器丢帧 59，FFmpeg 正常退出。

当前瓶颈在 Decoder OM 的 CPU 镜像/D2H 拷贝、YUV 转换和软件 `libx264`；单纯拆分发布线程未能恢复 30 FPS。后续可评估设备驻留输出配合异步 D2H 或 DVPP VENC。

历史状态（2026-09-13）：**in_progress**。当时 1080P 约 24.52 FPS，尚未达到 30 FPS 验收目标；
以下为后续 DVPP VENC 路径的最新验收结果。

## DVPP VENC 全链路复测（2026-09-25）

在解码端启用 DVPP H.264 VENC，并通过 RTP 完成编码板到解码板的全链路验证：

- 编码端：537/537 帧，33.5355 FPS，`encode=ok`。
- 解码端：537/537 帧，29.7511 FPS，`decode=ok`。
- DVPP VENC/RTSP：发布 537 帧，1920×1080 H.264，30 tbr。
- 解码端 `forward_dropped_frames=0`，`drop_frame_index=-1`。
- 编码端和解码端 `bitstream_bytes=644207` 一致。

RTSP 输出使用解码板已有的精简 FFmpeg runtime，并由 `output_transport_rtsp_encoder = "dvpp"`
选择 DVPP 编码路径；FFmpeg 只负责将 H.264 Annex-B 码流发布到 MediaMTX，不再进行软件
`libx264` 重编码。PC 客户端接入时机晚于发布起点，因此客户端统计帧数可以少于发布器计数，
不能据此判定解码或 DVPP 丢帧。验收摘要见
[`acceptance/issue11-venc-20260925/fullchain-537-39220-20260925/实测说明.md`](../../acceptance/issue11-venc-20260925/fullchain-537-39220-20260925/实测说明.md)；原始日志和配置仅保留在本地验收目录。

Issue #11 的 RTSP 显示功能和 30 FPS 级别的 1080P 实测已完成；GitHub issue 仍保持开放，
本轮未发布验收评论或关闭 issue。RTSP demo 不作为核心编解码路径的性能指标。
