# 全分辨率相机原生 NV12 运动输入验收

2026-10-02 实时相机 → 原生 NV12 运动代理 → MLVC → 双机 RTP 3000 帧长测，
编码 **29.8352 FPS**，接收解码 **29.857 FPS**，输出全部 3000 帧。
接近本机相机约 29.85 FPS 的实际采集速度，跟上 nominal 30 FPS 的相机负载；
**不宣称严格持续 30 FPS、零采集缺口或端到端两帧延迟**。采集序号缺口为 1，
网络 forwarding drop 为 0；旧颜色往返路径为 26.7962 FPS、采集序号缺口 333。

运动代理仍为 1920×1080，主 MLVC 为 1920×1080 画面、1920×1088 模型 padding。
此次优化减少运动输入转换；本交付没有降低运动分辨率，也没有保留降采样功能。
旧默认 `motion_camera_nv12=false` 保留，全分辨率原生 NV12 由独立配置显式开启。

## 测量条件与长测结果

- V4L2 `/dev/video0`、MJPG、1920×1080、nominal 30 FPS。
- 实时采集，`camera_preload_frames=0`，不启用 RTSP 或 trace。
- QP 11、GOP 96、feature reset 32、LTR 关闭。
- DVPP 全分辨率运动代理、`motion_prefetch_frames=2`、`motion_camera_nv12=true`、`motion_skip_loop_filter=true`。
- 排除 96 帧预热，编码 FPS 测量 2904 帧；接收解码包含等待到达，不是独立解码极限。

| 指标 | 3000 帧实拍结果 |
| --- | --- |
| 编码 / 接收解码 FPS | **29.8352 / 29.857** |
| 三个 900 帧窗口 FPS | 29.6361 / 29.8419 / 29.8505 |
| 编码 / 接收 / 输出帧数 | 3000 / 3000 / 3000 |
| 相机读取帧数 / V4L2 sequence gaps | 3001 / 1 |
| JPEG 错误 / 网络 forwarding drop | 0 / 0 |
| camera queue 最大深度 / 满等待次数 | 1 / 0 |
| tensor queue 最大深度 / 满等待次数 | 2 / 0 |
| 运动阶段平均 ms | 26.0258 |
| 输入队列运动准备回调平均 ms | 0，使用相机线程已生成的 sidecar |
| 相机运动 NV12 拷贝阶段平均 ms | 3.00596 |
| source-ready → encode 平均 / 最大 ms | 69.0823 / 144.965 |
| source-ready → RTP output 平均 / 最大 ms | 103.243 / 176.654 |
| 非零量化平移帧 | 0，不作为动态正确性证据 |

首次 900 帧统计时 sequence gaps 已为 1，随后 1800、2700 帧及结束仍为 1，
未持续累积采集缺口；但不是完全零掉帧。全部 3000 帧输出证明已编码发送帧收发完整，
不等于完整拍摄链路无漏取。本轮量化平移均为零，未验证持续显著全局平移实拍场景；
正确性由离线视频及 synthetic 测试验证，相机验收用于吞吐。

source-ready 在同帧 tensor 与运动 sidecar 准备完成、成对出队后记录，不包含
camera/tensor/driver 前置队列、采集或网络显示。运动预取两帧只限制 arena 三帧，
不能据此认定总延迟两帧。运动准备回调为零不代表无准备成本；相机 NV12 拷贝仍计时，
各并行阶段墙时不能相加，也不是纯硬件核耗时。

## 独立容量测试与兼容性

全分辨率原生 NV12 实拍 300 帧的完整 MLVC 文件输出编码为 29.9337 FPS、采集序号缺口 1。
最终纯全分辨率源码预加载 120 帧、运行 900 帧的完整处理吞吐为 **31.0808 FPS**，
运动阶段平均 28.82 ms、相机 NV12 拷贝阶段平均 1.37133 ms；
它排除实时采集，只说明处理容量，不代替在线长测。

最终精简源码解码板完整 CTest **33/33 通过**（84.59 秒），编码板针对本次变更的
相关测试 **6/6 通过**（2.57 秒）。旧默认 libx264、DVPP 和 warp-off 三项
本轮初版回归 SHA256 与此前对应样本一致；后续去镜像和清理未改变该后端行为。
测试保留配置默认及 sidecar gate、借用 CPU/pinned
FP16 与 owned 输入逐字节一致、外部 owner 生命周期、真实 1080p 转换、同帧槽 lease
及 EOF 清理。

最终全分辨率离线 fixture100 来自 `614lab.mp4` 的前 100 帧，包含非零运动。
两板各 102 份 feature/reset trace 全部一致，SHA 清单摘要与旧验收一致：
`c163ddf86387810a598b619e8bdf55e8f9c4653fee97727c3e06f74221a8b4e6`。
文件码流 SHA256 同样与旧 full-DVPP100 一致：
`af4f2d7dc9a33abcaf298df4b17a10143d0ff3f8c669baddd88e7b3243aa310e`。
该验证检查动态离线输入的两板参考/重构状态同步，不把实拍全零平移作为准确性证据。

原生 NV12 即使保持全分辨率，也可能与旧颜色往返得到的代理像素不同，因而可能改变
MV、量化平移及最终码流；主 MLVC tensor 保持原转换路径，不承诺新 opt-in 代理与
旧代理逐位一致。旧源码实拍结果见 [原在线验收](../motion-online-20261002/README.md)，
无节拍重复文件吞吐见 [文件吞吐验收](../motion-throughput-20261002/README.md)。

## 复现与资产

推荐 [camera_motion_rtp_nv12.toml](../../configs/camera_motion_rtp_nv12.toml)，
配套 [decoder_camera_motion_rtp.toml](../../configs/decoder_camera_motion_rtp.toml)，
使用原 1080p manifest、UDP 39340、payload type 96。两板均在独立仓库运行，
不修改原部署目录或全局依赖。先接收后发送，运行脚本加载 CANN/custom-op 环境：

```bash
# mlvc_decoder：先启动 3000 帧接收。
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run.sh configs/decoder_camera_motion_rtp.toml
```

```bash
# mlvc_encoder：实时相机，不预加载、无 RTSP/trace。
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run_encode.sh configs/camera_motion_rtp_nv12.toml
```

相机和 VENC 通道应由同一验收进程独占；接收 `format="none"` 执行完整解码，
但不输出可播放视频。完整测试入口为 `ctest --test-dir build --output-on-failure`。
metadata 30 FPS 不等于限速或吞吐保证。

本轮配置及日志根目录为两板 `/root/workplace/grifcc/motion-scale-validation`。
目录名沿用本轮验证目录，不代表交付保留降采样功能。

| 资产 | 文件 |
| --- | --- |
| 最终预加载 120、运行 900 帧容量 | `camera-preload-final900.toml` / `.log` |
| 最终相机 3000 帧编码日志 | `camera-final3000-encode.log` |
| 最终相机 3000 帧接收日志 | `camera-final3000-decode.log` |
| 全分辨率 100 帧 trace SHA 清单 | `full-trace-encoder.sha256` / `full-trace-decoder.sha256` |

两板各 102 份 trace 的逐文件 SHA 清单及运行日志作为最终证据保留；
原始大型临时 FP16 trace 目录不保留，以节省板端空间。
最终实时与预加载复测、离线同步及本次相关测试均已完成。
本轮验收基于 `947a280` 加最终纯全分辨率开发 diff。
实际部署版本使用 `git rev-parse HEAD` 读取，最终提交 SHA 由主流程提交后交付。
