# 全分辨率相机原生 NV12 运动输入

推荐相机配置为 [camera_motion_rtp_nv12.toml](../configs/camera_motion_rtp_nv12.toml)。
接收端配套 [decoder_camera_motion_rtp.toml](../configs/decoder_camera_motion_rtp.toml)，
先运行解码接收脚本，再运行编码发送脚本；两端默认处理 3000 帧。
运动代理保持 1920×1080，主 MLVC 输入仍为 1920×1080 画面、底边 padding 到 1088；
没有降低运动分辨率，也没有保留降采样配置或实现。

`motion_camera_nv12` 默认为 false。开启时要求 camera 输入、在线 translation warp 和
`motion_prefetch_frames>0`，不能用于 CSV 重放；DVPP 和 libx264 均支持这一输入路径。
推荐配置采用 DVPP、两帧预取和跳过运动解码器环路滤波，旧默认配置保持原行为。

相机 JPEG 解码的 NV12 按可见行复制成 packed sidecar，正确使用 row stride、height stride
和 UV 起始偏移，不把 padding 当作画面。它避开运动输入的 NV12 → BGR → FP16 → NV12
往返；主 MLVC tensor 仍走原转换路径。原生 NV12 与原往返像素可能不同，因此可能
改变 MV、量化平移及最终码流，不承诺新代理与旧代理逐位一致。

同一 camera packet 携带 tensor 与 motion sidecar，在同一队列锁内成对移动。
sidecar 借用同帧槽直到 MLVC Release 后才复用；EOF、取消和异常清理释放 tensor 的
external owner 与 sidecar。FP16 转换路径直接读取合法 CPU/pinned view，省去整帧 mirror，
owned 与 borrowed 输入须逐字节一致；设备地址或短 buffer 明确拒绝。

最终实时相机双机 RTP 3000 帧测得编码 29.8352 FPS、接收解码 29.857 FPS，
全部 3000 帧输出；采集序号缺口 1、网络 forwarding drop 0。
结果接近本机相机约 29.85 FPS 的实际速度，不宣称严格持续 30 FPS、零采集缺口或
两帧端到端延迟。最终预加载 120 帧、运行 900 帧的 31.0808 FPS 只说明处理容量。
条件、计时边界与日志见 [原生 NV12 验收](../acceptance/motion-nv12-20261002/README.md)。
