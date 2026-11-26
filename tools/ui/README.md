# ulbvc 统一五路 UI · V0.5

Thor 与 Ascend 310P 使用同一套网页、Python 后端和 Windows 程序。通过参数文件适配视频地址和统计端口，不再维护两种界面。

默认五组通道，每组并排显示原图、H.264 对照、ulbvc 解码，共 15 个画面。支持五路总览、单通道放大、单画面全屏、暂停预览、重新连接和独立通道设置。无视频时显示“无信号”，无统计时显示 `—`。五组展示容量不意味着要求板端同时运行五路模型。

## 运行

双击 `ULBVCMonitor.exe`；同目录保留 `ui_config.json`。源码启动需要 Python 3.10+：

```powershell
python tools/ui/launch_ui.py
# 只启动后台 HTTP 服务：
python tools/ui/server.py --port 8765
```

`run_ui.bat`、原有 `run_web_ui.bat` 均调用同一个启动器；`web_ui.py` 仅为旧命令的兼容入口。
默认页面为 `http://127.0.0.1:8765/`。关闭启动器结束本地服务和自建 H.264 对照进程；板端编解码与 MediaMTX 独立管理。启动 UI 不会启动板端模型。

## 一套程序，两种配置模板

| 参数文件 | 使用方式 |
| --- | --- |
| `ui_config.json` | 日常参数，默认与 310P 现有单路链路对应，保留五组位置 |
| `ui_config.310p.json` | 310P 初始模板：第一组使用 `/camera-original`、`/camera-h264`、`/ulbvc`，统计端口 39341，1080P/30 FPS |
| `ui_config.thor.json` | Thor 初始模板：`/input0`～`/input4`、`/camera-h2640`～`/camera-h2644`、`/ulbvc0`～`/ulbvc4`，统计端口 39600～39604 |

同一个程序可直接指定模板或独立参数文件：

```powershell
.\ULBVCMonitor.exe --config .\ui_config.thor.json
# 源码方式相同：
python tools/ui/launch_ui.py --config tools/ui/ui_config.310p.json
```

模板是接入起点，视频地址、源尺寸与统计端口必须与实际设备匹配。可以在同一个五路配置中混合 Thor 与 310P 通道；每路使用唯一统计端口。界面填写的 FPS/尺寸用于统计计算，不修改板端采集、检测或编码分辨率。

“参数设置 → 保存并连接”会校验并原子保存参数。旧 310P 单路参数文件首次加载时自动迁移为五组，保留原有视频地址、H.264 QP、FFmpeg 路径、TCP/UDP 和统计端口，原文件备份为 `ui_config.json.v0.5-backup`。新参数文件与 Thor 已有多路格式兼容。

## 视频与 H.264 对照

视频继续通过 MediaMTX 的 WHEP/WebRTC 接口播放。解码板已叠加的检测框与 label 随原画面显示，UI 不重新运行检测或按浏览器到达时间拼接框。

H.264 对照支持 `auto`、`h264_qsv`、`h264_nvenc`、`h264_amf`、`libx264`。自动模式保持硬件优先，不偷偷回退软件；需要软件编码时明确选择 `libx264`。

310P 模板保持原有软件 x264、固定 QP 40、手动启动；Thor 模板保持硬件自动、QP 28、原图就绪后自动启动。两种模板调用同一套实现。H.264 QP 越小画质越高；它和板端 ulbvc 的 q_index 方向及标度不同。

原图和 H.264 默认分别补偿显示 200 ms，可在参数中调整。这是播放补偿，与编码板发送队列的 200 ms 预算独立，也不是严格逐帧对齐或端到端延迟测量。视频使用各自源分辨率播放，允许单通道放大或全屏。

UI 只管理它自己创建的对照推流进程，已有外部 H.264 流不会被替换；暂停预览不停止板端编解码。

## 统一统计口径

后端自动识别两种现有报文，无需在界面切换设备类型：

- Thor：`kind = "semantic_codec_stats"`，使用 `payload_bytes` / `frames`。
- 310P：`kind = "ulbvc_rtp_stats"`，使用 `payload_bytes` / `frames`，另读取 `wire_bytes`。

ulbvc 主指标与曲线统一为编码净载荷码率，十进制 kB/s；bpp 用实际编码帧数和配置的源尺寸计算，压缩比以 RGB24 的 24 bpp 为基准。310P 如有 `wire_bytes`，附加显示 RTP/UDP/IP 发送带宽，包含 RTP 重传；不将该值冒充 codec 净码率，也不宣称包含独立 RTCP 流量。

H.264 对照统计为浏览器收到的 RTP 净载荷，采样位置与编码器不同。原图指标为配置源尺寸/帧率的 RGB24 理论值，仅在原图播放时显示。标题上的 FPS 是浏览器实际收到/解码的 FPS。

累计计数回退、协议来源改变、保存新配置或超过 5 秒未更新时清除旧统计基准，避免把旧设备数据继续显示为当前值。没有有效采样时不显示虚构的 0。曲线保留最近 30 秒。

PC 的展示与本地 H.264 对照独立于板间的 RTP/RTCP 码率闭环。本次统一不增加 PC 调节板端 qp 或带宽的入口；RTCP 的完整状态面板仍可后续扩展。

310P 当前板端统计地址来自 `configs/encoder.toml` 的 `ulbvc_stats_host` / `ulbvc_stats_port`；Thor 来自其编码配置的 `ui_stats_host` / `ui_stats_port`。地址应填写 UI 电脑的局域网 IP，统计端口不要与 RTP/RTCP 端口混用。

## 维护与同步

统一维护入口为 310P 仓库的 `tools/ui`。Thor 仓库保留同版本镜像，使用同步脚本更新，保留对端自己的 `ui_config.json`：

```powershell
python tools/sync_ui.py path/to/other/checkout/tools/ui
python tools/sync_ui.py path/to/other/checkout/tools/ui --check
```

脚本只同步显式列出的共用源码、测试及模板，不覆盖运行参数。两端交付同一个 `ULBVCMonitor.exe`，仅外置配置不同。

## 验证和打包

```powershell
python -m unittest discover -s tools/ui -p 'test_*.py' -v
node --check tools/ui/app.js
node tools/ui/test_frame_delay.cjs
node tools/ui/test_player_recovery.cjs
node tools/ui/test_connection_recovery.cjs
python tools/ui/launch_ui.py --self-test artifacts/ui-self-test.json
python tools/ui/build_windows_exe.py --output artifacts/unified-ui/dist
python tools/ui/test_windows_exe.py artifacts/unified-ui/dist/ULBVCMonitor.exe --report artifacts/unified-ui/exe-test.json
```

构建需要 `requirements-build.txt` 中的 PyInstaller、可用的 FFmpeg 和 Tkinter。构建产物包含 FFmpeg、网页、两种配置模板、说明、文件哈希和构建信息。`--self-test` 只在后台检查资源、HTTP、FFmpeg 及清理，不打开窗口或启动板端模型。

本轮后台回归覆盖参数迁移/备份、Thor/310P 混合 UDP 统计、净载荷与 RTP/IP 分离、软件/硬件对照命令、配置保存、WHEP 会话归属、视频帧补偿和连接恢复。板端保持停止，实时视频和不同电脑的硬件编码可用性需要在对应环境验证。


## 2026-11-25 时间戳与卡帧修复

统一 UI 时曾漏带旧 310P 界面的时间戳直通设置，短探测窗口下可能将 90 kHz RTP 时钟当成帧率，造成大量 CFR 补帧、堆积和 H.264 卡住。本次恢复 `-fps_mode:v passthrough`、`-enc_time_base:v demux`、`nobuffer` / `low_delay`，并恢复足够的输入探测窗口；不使用与 passthrough 冲突的 `-r` / `-fpsmax`。

实际相机流对照中，旧参数重复帧计数为 14103，修复参数为 0，修复后的对照推流约 29.6 FPS。输入确定离线时停止本 UI 自建的对应 H.264 进程，避免继续保留卡死/失去输入的进程；只有媒体 API 暂不可达时保留现有推流，不误判为源离线。

原图/H.264 的 200 ms 固定播放补偿保持原值。它不是本次新增的延迟，也不等于板间发送队列预算。编码端另已修复“开启码控时完成帧仍等待两帧”的调度，编译及相关 CTest 通过；编码板 SSH 暂时断连，尚未获得修复后等待时间的板上测量，不能据此承诺具体端到端延迟。

现场解码板曾达到 103°C / Warning，现已停机，用户要求散热未处理前保持停止。本修复包不会启动板端模型。更换程序时请手动退出旧 UI，再打开新包中的 `ULBVCMonitor.exe`，避免两套后端争用统计端口。

FFmpeg 参数依据：[官方文档](https://ffmpeg.org/ffmpeg.html)。
