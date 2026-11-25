# JPEGD 等待耗时调查：2026-11-24

后续已完成在线实现与编码板部署，900 帧测试为 29.82 FPS，详见
[在线部署记录](JPEGD_DEPLOYMENT_20261124.md)。以下保留离线定位阶段
的证据和当时尚未完成的事项。

已在两张稳定相机样本上定位到具体触发因素：**相机将 JPEG 熵编码段
最后一个字节的剩余位填成 0；只将这些填充位改为 1，独立 JPEGD 的
单帧耗时就从约 30.80 ms 降到 4.45 ms。** 图像系数、量化表、Huffman
表、分辨率、重启间隔均保留，Pillow 解出的 RGB 像素逐字节一致。

这是离线修正样本后的隔离对照结果。在线修复的 CPU 开销尚未测量，
没有将修复接入生产相机路径，也没有恢复实时编解码。尚不能据此宣称
整链路已经达到 30 FPS，或确认底层具体使用了哪一种硬件/软件回退机制。

## 稳定相机样本与主要证据

相机使用 /dev/video0、MJPG 1920×1080，驱动协商帧间隔 1/30。
每次启动后丢弃前 30 张有效帧，保存 sequence=31 的图像后立即关闭相机。
两张样本分别为 227664、240048 字节，较启动首帧明显更大。因此，前期
首帧约 18.8 ms 的结果不能替代稳定出图后的结果。

以下均在编码板测量，每组 48 帧、默认 stream、每帧同步；每条命令由
timeout 15s 限时。解码板和生产编解码应用保持停止。

| 相同稳定画面的处理方式 | JPEGD 单帧 ms | 说明 |
| --- | ---: | --- |
| 相机原始 JPEG | 30.8047 | 原始码流，隔离解码 |
| 原始 JPEG + YOLO11s 720P 每秒 30 次推理 | 31.2760 | 同 context、独立 stream，未启动 codec/VENC |
| 保留 DCT 系数的无损 JPEG 整理 | 4.4480 | Pillow RGB 完全一致，会重写熵编码并去掉重启标记 |
| 仅修正原始 JPEG 的熵编码填充位 | **4.4454** | 保留重启标记；Pillow RGB 完全一致 |
| RGB 解码后质量 95 重新压成 JPEG | 4.4730 | 有损对照，不作为生产修复方案 |

原始 JPEG 加入 YOLO 后仅增加约 0.47 ms，未复现此前整链路约 35.72 ms
的耗时。这支持优先优化相机 JPEG 输入，但不排除 codec、VENC 等其他
任务的争用；它们没有参加本轮受控实验。表中的 JPEGD 时间不包含离线
码流整理/修正的耗时，隔离解码吞吐也不是整链路 FPS。

## 具体码流差异

两张稳定相机样本均为 SOF0、8 bit、1920×1080、4:2:2、单 SOS，
每 120 个 MCU 使用一个重启标记，共 134 个重启标记，编号顺序正确。
没有发现渐进 JPEG 或三 SOS 的情况。DRI 为 120 本身不是充分触发条件：
另用 Pillow 12.3 生成相同 DRI 的受控样本仍只需约 4.45 ms。

离线工具解析 Huffman 数据，按每个重启段的 MCU 数逐块走完 DC/AC
符号，只将段尾剩余的 0～7 个填充位改为 1；如产生 0xFF，补入规定的
0x00 byte stuffing。它不修改任何有效系数位，不重新量化，也不删除
重启标记。原始头部和 EOI 后已有字节均保留。

- 样本 A：135 个熵编码段，122 段存在 0 填充并被修正。
- 样本 B：135 个熵编码段，119 段存在 0 填充并被修正。
- 两张图修正前后，Pillow 解码 RGB 均逐字节一致。
- 板上四种同步组合的修正后耗时均约 4.40～4.46 ms。

这与 JPEG 标准对 Huffman 段尾使用 1-bit padding 的规定相符，参见
[ITU-T T.81，B.1.1.5 与 F.1.2.3](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)。
当前证据说明填充位修正可以消除这些样本上的大部分额外耗时；尚未获取
足够驱动内部证据，不能直接把原始路径称为纯软件解码。

板上每组测试均检查解码返回码，并将有效 NV12 像素与生产 JPEGD 包装器
在**同一份输入**上的输出比较，全部通过。这个检查验证同步、缓冲复用和
帧顺序；它没有证明原始慢路径与修正后快路径的 NV12 输出逐字节相同。
接入前仍需验证两条路径的像素差异及检测输出。

## 已排除的简单开关

先前使用启动首帧做了以下受控检查：

| 改动 | 默认单帧 ms | 结论 |
| --- | ---: | --- |
| 原始启动首帧 | 18.7787 | 首帧基线 |
| 增加 JFIF APP0 头 | 18.8046 | 未改善 |
| 仅去掉 EOI 后的 0 字节 | 18.8153 | 未改善；与熵编码填充位是不同位置 |
| 拆分/整理 DQT、DHT 与头部顺序 | 18.7824 | 未改善 |
| 压缩输入预先放入 acldvppMalloc 内存 | 18.8021 | 未改善，预复制不计入 JPEGD 时间 |

默认 stream / ACL_STREAM_FAST_SYNC × 每批 1 帧 / 2 帧的组合，
在稳定原始样本上均约 30.80～30.98 ms，没有显示明显收益。暂未将
快同步、双帧提交或输入内存切换加入生产路径。

最初公交车/灰图两个静态 JPEG 测得约 4.5 ms，只用于验证工具；真实
相机样本、稳定出图样本及填充位对照才构成上述定位过程。

## 工具与复现

tools/cpp/benchmarks/benchmark_dvpp_jpeg_decode.cc 对两张 JPEG 交替
解码，输出槽独占内存和描述信息，同步后才复用，销毁前完成未结束工作。
测量前后检查有效 NV12，验证用 D2H 不计入测量。可选输入内存模式及
同 context 的 YOLO 受控负载。

~~~bash
cmake --build build --target benchmark_dvpp_jpeg_decode --parallel 4
source scripts/env.sh
timeout 15s build/benchmark_dvpp_jpeg_decode \
  logs/camera_steady_a.jpg logs/camera_steady_b.jpg 48 direct
timeout 15s build/benchmark_dvpp_jpeg_decode \
  logs/camera_padding_fixed_a.jpg logs/camera_padding_fixed_b.jpg 48 direct
# 可选：在 direct 后添加 720P OM 路径，产生每秒最多 30 次同 context 推理。
~~~

tools/cpp/validation/camera_v4l2_probe.cc 新增 FPS 和丢弃帧数参数，
输出协商格式、帧间隔、sequence、丢弃数，最长等待 3 秒。

~~~bash
timeout 5s build/camera_v4l2_probe \
  /dev/video0 1920 1080 MJPG logs/camera_steady_a.jpg 30 30
~~~

本地证据位于 artifacts/jpegd-investigation/：

- results.json：14 组实验、原始日志关联、26 张 JPEG 的 SHA256。
- jpegd-camera-steady-v1.log、jpegd-camera-steady-yolo-v1.log。
- jpegd-camera-padding-fixed-v1.log、entropy-padding-validation.json。
- check_entropy_padding.py：仅用于此次固定 SOF0/4:2:2 样本的离线
  熵编码检查与填充位修正，不是可直接上线的通用 JPEG 解析器。
- lossless-control-validation.json、jpegd-camera-lossless-v1.log。

jpegd-camera-restart-v1.log 是无效的早期控制：Pillow 9.5 忽略了
restart 参数，文件没有 DRI；只使用已检查 DRI 的 v2 控制作为依据。
另一次 TurboJPEG 保留重启间隔的变换尝试也因旧版接口忽略参数而无效，
未纳入结论。填充位修正实验不依赖这两次尝试。

## 接入前仍需完成

优先实现并测量轻量的在线填充位规范化，避免在实时路径中先完整解码再
重新压缩 JPEG。需要健全的格式边界、异常输入回退、修复耗时统计，以及
原始/修正后像素和检测结果对照。再用相机 + 检测 + ulbvc 编码验证实际
吞吐；实时双板验证仍须遵守用户的停机要求及解码板散热约束。

API 依据：

- [CANN stream 快同步说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0069.html)
- [JPEGD 异步接口说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/appdevgapi/aclcppdevg_03_0357.html)
