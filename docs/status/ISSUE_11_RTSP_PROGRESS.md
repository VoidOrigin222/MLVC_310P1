# Issue #11 RTSP 输出进度

更新时间：2026-09-30

## 实现

解码器支持 `output_transport_mode = "rtsp"`。`RtspVideoPublisher` 在板端通过
`libavformat` 直接完成 RTSP 控制、SDP、RTP/RTCP 封装和网络写入，不再启动 FFmpeg
子进程或通过 `pipe` 传输。`dvpp` 模式直接发布 DVPP 生成的 H.264 Annex-B 帧；
`libx264` 模式通过 libavcodec 编码 NV12 帧后交给 libavformat。发布队列有界，工作线程
和 libavformat/libavcodec 错误通过 `Close()` 传播。

没有安装 FFmpeg 开发库的主机仍保留旧的 FFmpeg 子进程兼容路径；310P1 板端安装
`ffmpeg-devel` 后构建时会显示 `RTSP publisher backend: libavformat`。

RTSP 发布默认使用 UDP 媒体传输（RTSP 控制连接仍使用 TCP 8554，RTP/RTCP 使用 MediaMTX 的 UDP 8000/8001）。

当前实现保持单后台线程：颜色转换/编码和网络写入在同一线程串行执行，队列仍受容量限制。

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

板端 `build-direct` 成功构建；`rtsp_video_publisher` 通过本地 RTSP 握手服务器，实际验证
OPTIONS/ANNOUNCE/SETUP/RECORD、TCP 交错 RTP 数据、H.264 直通和 NV12/libx264 两条路径。
全量 CTest 23/23 通过。

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

Issue #11 的 RTSP 显示功能和 30 FPS 级别的 1080P 实测已完成。针对后续“应直接使用
libavformat”的反馈，当前板端版本已改为直接 API 后端；旧 pipe 仅作为未安装开发库主机的
兼容路径保留。RTSP demo 不作为核心编解码路径的性能指标。
