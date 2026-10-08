# 语义压缩三路视频控制台

现有 RTSP / MediaMTX 链路上的本地 WebRTC 控制台，无需安装前端依赖。

- 固定三路并排预览：原图、H.264、语义压缩；支持单路全屏。
- H.264 推流启动 / 停止、三路重新连接、参数抽屉与错误反馈。
- 实际接收 FPS、分辨率、bpp、带宽（kB/s）与压缩比。
- H.264 与语义压缩的码率对比曲线，显示最近 30 秒。
- 窄窗口保留三路并排。
- H.264 使用固定 QP；原图与 H.264 页面播放均额外缓冲 200 ms 解码帧。

页面会自动连接已有视频源。点击“启动 H.264 推流”才启动本地 FFmpeg；
关闭或刷新浏览器只释放播放会话，不会停止正在运行的推流进程。
停止按钮会断开本页预览并停止 UI 管理的 H.264 进程，板端原图和 MLVC 进程仍由板端控制。
参数通过“参数设置 → 保存并连接”提交，在本次服务运行期间生效；服务重启恢复默认配置。

文件：`web_ui.py` 提供控制接口，`index.html`、`styles.css`、`app.js` 提供界面。
验证命令：在本目录运行 `python -m unittest -v test_web_ui.py`。

运行需要 Python 3.10 或更新版本、包含 libx264 的 FFmpeg，以及已配置的 MediaMTX
RTSP / WebRTC / API 服务。Python 后端只使用标准库，前端不需要安装 npm 依赖。
板端配套使用 `motion` 分支的 `e32a134` 或更新版本（包含编码端 UDP 带宽统计）。
Node.js 仅用于运行前端回归测试。

## 同质量模式

页面不显示模式标识，H.264 使用 libx264 固定 QP（`-qp`），不设置
CBR/VBV 码率限制，码率随画面复杂度变化。不再采样、传输原始帧或计算 PSNR。
语义压缩的板端参数由 `configs/encoder.toml` 控制，页面不修改它。

H.264 QP 初始值为 40，可在参数设置中调整（0–51 整数）。它是人工调整画质的起始值，
不是从语义压缩 QP 或历史 CRF 直接换算的对应值。QP 调小提高画质，调大降低画质。
在同质量模式保存新 QP 会重启 H.264 发布器使其生效。“同质量”表示主观近似画质对照，
不保证两路具有相同 PSNR。

页面对原图与 H.264 解码帧各加入 200 ms 显示缓冲，以 canvas 绘制到期帧，额外延时约 200 ms（受刷新周期影响）。语义压缩直接播放；重连和停止时释放缓存帧，每路队列最多 12 帧，不累积延时。这是固定延时补偿，三路各自的编码与网络延时仍可能波动，不保证逐帧同步。

测试页面延时队列：`node test_delayed_video.cjs`。

## 地址和路径

| 用途 | 地址 |
| --- | --- |
| 原图 RTSP（编码板发布） | `rtsp://127.0.0.1:8554/camera-original` |
| H.264 RTSP（Windows FFmpeg 发布） | `rtsp://127.0.0.1:8554/camera-h264` |
| 语义压缩 RTSP（解码板发布） | `rtsp://127.0.0.1:8554/ulbvc` |
| 网页 | `http://127.0.0.1:8765` |
| MediaMTX WebRTC | `http://127.0.0.1:8889` |
| MediaMTX API | `http://127.0.0.1:9997` |

UI 与 MediaMTX 默认在同一台电脑，播放与发布地址使用 `127.0.0.1`。
参数设置显示并保存完整 RTSP 地址，包括语义压缩的 `/ulbvc` 流路径；可直接修改完整路径。
板端配置中的 RTSP 服务器 IP 和统计接收 IP 仍需填写该电脑的局域网 IP。

## 启动网页服务

### Windows EXE

双击 `SemanticVideoUI.exe`，启动器会启动本地 UI 服务并打开浏览器。
EXE 内置 Python、网页资源、FFmpeg 及共享库，不需要另外安装 Python 或 FFmpeg。
启动器窗口中的“停止并退出”或关闭按钮会停止 UI 服务及它管理的 H.264 发布器。
再次双击时，如果已有 UI 在运行，会打开现有页面。
MediaMTX 和板端编解码仍按原部署方式提前启动。

EXE 的 FFmpeg 日志保存到 `%LOCALAPPDATA%\SemanticVideoUI\logs\ffmpeg_ui.log`。

在 Windows x64 上重建 EXE（使用含 Tkinter 的 Python 3.10+）：

```powershell
cd tools/ui
python -m pip install -r requirements-build.txt
python build_windows_exe.py
```

构建时需要 FFmpeg 可在 PATH 中找到，或通过 `--ffmpeg` 指定 `ffmpeg.exe`。
默认输出为仓库根目录的 `artifacts/ui-exe/dist/SemanticVideoUI.exe`。

验证生成的 EXE（在移除 Python / FFmpeg PATH 的子进程中测试页面、编码、启动及退出）：

```powershell
python test_windows_exe.py ../../artifacts/ui-exe/dist/SemanticVideoUI.exe --report ../../artifacts/ui-exe/dist/VALIDATION.json
```

### Python 方式

```powershell
# 从仓库根目录执行
cd tools/ui
python web_ui.py
```

打开 `http://127.0.0.1:8765`。如果网页服务已经在运行，不要再次启动第二个实例。

## 播放方式

网页使用 MediaMTX 的 WebRTC/WHEP 接口显示三路 `<video>`：

- 原图：编码板将 DVPP JPEGD 输出的设备 NV12 交给 DVPP H.264 编码，再通过 libavformat RTSP 发布到 `camera-original`；原图发布使用 VENC 通道 0，运动估计使用通道 1。
- H.264：Windows 本地 FFmpeg 从原图 RTSP 读取，使用 `libx264` 编码后发布到 `camera-h264`
- 语义压缩：解码板发布的 `ulbvc`

视频不经过网页转码，也不使用 HLS 的多段缓冲。MediaMTX 需要开启 WebRTC，默认
控制端口为 `8889`；当前 `mediamtx.yml` 已开启 `webrtc`、CORS 和低延时 UDP ICE。

语义压缩统计由编码板通过 UDP `39341` 发布：`wire_bytes` 是 RTP/UDP/IP
线速字节，网页用它计算占用带宽，不使用解码端的 H.264 预览流统计。

编码板配置需要设置：

```toml
mlvc_stats_host = "192.168.5.3"
mlvc_stats_port = 39341
mlvc_stats_interval_frames = 30
```

## 正确启动顺序

### 1. 确认 MediaMTX 已运行

由部署方启动 MediaMTX，开放 RTSP、WebRTC/WHEP 和 HTTP API 服务。
本 UI 不自动启动 MediaMTX。

### 2. 启动板端

先启动解码板，再启动编码板。编码板必须先发布 `camera-original`，Windows FFmpeg 才能启动。

解码板 `192.168.5.11`：

```bash
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run.sh configs/decoder.toml
```

编码板 `192.168.5.13`：

```bash
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run_encode.sh configs/encoder.toml
```

两条命令都在各自板子的前台终端运行；需要停止时在对应终端按 `Ctrl+C`。

启动后可在 Windows 检查路径：

```powershell
(Invoke-RestMethod http://127.0.0.1:9997/v3/paths/list).items |
  Select-Object name, ready, tracks
```

确认列表中出现 `camera-original` 后，再启动 Windows FFmpeg。

### 3. 启动 Windows FFmpeg（推荐方式）

网页服务已经运行时，通过本地控制接口启动 FFmpeg：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8765/api/start `
  -ContentType "application/json" `
  -Body "{}"
```

该接口会启动一个由 UI 管理的 FFmpeg 子进程：

```text
camera-original (RTSP) → FFmpeg/libx264 → camera-h264 (RTSP)
```

H.264 默认使用同质量的固定 QP 模式，使用 veryfast、2 个编码线程、GOP 96。
原图源就绪后启动，不等待语义压缩的码率统计，不设置 CBR/VBV 限制。
停止推流后不会自动启动；正常运行不依赖浏览器轮询。

FFmpeg 启动时最多分析 1 秒媒体数据（探测字节上限 5 MB），避免把 RTP 的
90 kHz 时钟误判为视频帧率。输出使用 `-fps_mode:v passthrough` 和
`-enc_time_base:v demux`，保留原图时间戳，每个输入帧只编码一次。
不要再使用 `-probesize 32 -analyzeduration 0`，也不要添加 CFR 补帧或 `fps` 滤镜。

然后确认 `camera-h264` 已发布：

```powershell
(Invoke-RestMethod http://127.0.0.1:9997/v3/paths/list).items |
  Select-Object name, ready, tracks
```

列表应至少包含：`camera-original`、`camera-h264`、`ulbvc`。

### 4. 打开网页并连接

打开 `http://127.0.0.1:8765`，点击“重新连接视频”（或刷新页面）。三路视频分别对应
`camera-original`、`camera-h264` 和 `ulbvc`。

不要同时使用网页的 `/api/start` 和另一个手工 FFmpeg 进程发布 `camera-h264`，否则会发生
发布者争用。同一时间只保留一个 H.264 发布进程。

## H.264 发布失败时

如果 FFmpeg 在原图流尚未就绪时启动，会在 `ffmpeg_ui.log` 中看到 `not enough frames`、
`Failed reading RTSP data` 或 `Broken pipe`，随后 `camera-h264` 不会出现。处理方法是：

1. 等 `camera-original` 在 MediaMTX API 中显示 `ready=true`；
2. 再执行上面的 `/api/start` 命令；
3. 确认 `camera-h264` 出现后刷新网页。

FFmpeg 异常退出后，UI 会在 30 秒重试间隔后尝试恢复当前编码参数；明确停止后不会自动恢复。
日志位置：`tools/ui/ffmpeg_ui.log`（相对于仓库根目录）。

浏览器显示播放状态、分辨率、实际接收 FPS、bpp、带宽（kB/s）与压缩比。页面参数面板默认收起，
打开页面后自动连接三路视频。未收到实际数据时显示“—”。
界面的 kB/s 沿用原项目每秒字节 / 1024 口径。
MediaMTX API 中路径必须 `ready=true` 才判定为已发布；“播放中”由浏览器实际播放事件确认。

原图带宽使用固定理论值：1920 × 1080 × 配置 FPS × 3 / 1024，默认 30 FPS 时为 182,250.0 kB/s，不随相机画面变化。原图 bpp 为 24，压缩比为 1，作为未压缩图像基准。H.264 沿用 MediaMTX 接收统计；语义压缩带宽沿用编码板 RTP/UDP/IP 线速统计，bpp 和压缩比按媒体单元字节计算。压缩比为 24 / bpp。原图不显示理论码率。

码率曲线每秒记录后端实际 kbs，与画面下方的带宽数值一致，单位统一显示 kB/s（沿用每秒字节 / 1024 口径），保留最近 30 秒；无有效统计时断开曲线，不以零值代替。曲线沿用后端已有的 12 秒平均速率统计。画面与曲线在桌面布局下保持 32 px 间距。

## 回归测试

在 `tools/ui` 目录执行，不需要连接相机、板子或启动推流：

```powershell
python -m unittest -v test_web_ui.py
node test_delayed_video.cjs
node test_bitrate_history.cjs
node test_settings.cjs
```
