# 检测并行链路优化结果：2026-11-24

后续已定位相机 JPEG 的熵编码填充问题并部署在线修正，720P 检测 +
1080P 编码的 900 帧复测为 29.82 FPS，最终版本 300 帧为 29.91 FPS；
原始 1080P 检测约 24.37 FPS。详见
[在线部署记录](JPEGD_DEPLOYMENT_20261124.md)。以下保留上一轮优化记录。

实机摄像头链路已接近 28 FPS，尚未稳定达到 30 FPS。默认保持 720P
检测、1920×1080 相机与 ulbvc 编码。每个编码帧都有对应的检测结果，
编码 QP 仍为 11，原图预览保持开启。

## 性能结果

| 摄像头检测配置 | 优化前 FPS | 优化后 FPS | 帧数 |
| --- | ---: | ---: | ---: |
| 原始 1080P 检测 + 1080P 编码 | 17.95 | 24.11 | 各 300 |
| 720P 检测 + 1080P 编码 | 19.84 | 27.42 | 各 300 |
| 720P 检测 + 1080P 编码，较长复测 | — | 27.90 | 900，前 30 帧 warmup |

300 帧同参数对照中，720P 检测链路吞吐提高约 38%。现场画面和设备温度
未完全固定，因此这些数字表示本次实机观测。900 帧测试中，编码、检测、
解码、RTSP 输出均为 900 帧，转发丢帧为 0，设备健康状态正常。
摄像头 V4L2 仍出现 60 个 sequence gaps；完整的编码帧对应关系不代表
处理了相机的每次曝光。

900 帧运行约 32 秒，采样到的编码板最高 42°C、解码板最高 85°C；
本轮是有限帧验证，不代表持续数小时的热稳定性测试。

## 实现

1. 在 NV12 的 Y、UV 两个平面分别缩放，再融合颜色转换、RGB uint8
   舍入、归一化和 FP16 NCHW 排布。减少整帧 RGB 中间图和重复遍历。
2. 检测分成预处理线程与模型/后处理线程。三个独立输入缓冲区和一个
   有界的已准备队列允许下一帧预处理与当前帧模型调用重叠。缓冲区在
   模型调用完成后才归还；结果使用原始 frame ID 的 promise 交回发送端。
   退出时排空任务，失败时传播异常。
3. 相机 BGR→FP16 YUV444 增加 ARM NEON 转换并保留多核行处理；
   两个转换源文件在 ARM 上单独使用 `-O3`，没有开启 fast-math。
4. 新增 detection_prepare 时间线轨道，保留原有检测、模型和发送等待计时。

900 帧实测中，相机 BGR→FP16 转换平均约 2.69 ms，原先约 28–29 ms。
检测预处理平均约 12.45 ms，原先约 31–33 ms。新的
`detection_pack_fp16_avg_ms` 包含融合的 NV12 颜色转换与 FP16 排布；
独立 `detection_color_convert_avg_ms` 为 0 表示已融合。

预处理与模型调用已经有时间重叠，不能再把两者的平均耗时直接相加
来推算吞吐。受控样图时间线中，120 帧累计约 256.79 ms 的预处理
区间与模型调用区间重叠；这不是 AI Core 利用率。

曾尝试锁页检测缓冲区，摄像头实测约 27.72 FPS，没有可辨认的吞吐收益，
最终使用可复用的普通主机缓冲区。

## 正确性验证

- 36 项板端 CTest 全部通过。
- 新增 NV12→RGB FP16 对照测试：覆盖向量块、标量尾部、非紧密 stride
  和 letterbox 填补，逐元素与 OpenCV RGB 参考一致。
- 新增相机 BGR→FP16 对照测试：覆盖 stride、尾部和边缘复制填补；
  本次数值样本的最大绝对差为 0.00048828125。
- 120 帧静态公交车样图：发送/接收 600 个框，解码 RTSP 叠框、标签和
  置信度显示正常。
- 120 帧交替有目标/空白样图：偶数帧 1 辆公交车、4 个人，奇数帧 0 个
  目标；逐帧检查 frame ID、类别与数量全部通过，发送/接收共 300 个框。
  该测试针对输入缓冲区复用和结果串帧。
- 与优化前样图框比较，最小匹配 IoU 为 0.98534，最大置信度差为 0.00635。
  720P 路径改变了缩放与颜色转换的顺序，不能据此代替 COCO 精度评测。
  1080P/720P 的 letterbox 和框还原几何保持一致。

交替样图测试测得 30.46 FPS，静态样图约 29.48 FPS；它们使用文件输入，
不是摄像头链路结果，未替代上面的实机 FPS。

## 剩余限制

900 帧摄像头运行的 JPEGD 入口调用平均约 35.72 ms，其中包含同步等待；
相机队列、tensor 队列均未出现满队列等待。原先 CANN 采集中，JPEGD
线程的 `aclrtSynchronizeStream` 平均约 33.09 ms，而预测尺寸与异步
提交 API 分别约 0.04 ms、0.19 ms。这些数据不足以将等待时间全部
归因于 JPEG 硬件计算。随后隔离调查发现，稳定相机 JPEG 单独解码
约 30.80 ms，仅修正其熵编码段尾的 0 填充为 1 后降到约 4.45 ms；
Pillow 解码像素完全一致。在线修复开销与整链路收益尚未测量，详见
[JPEGD 调查记录](JPEGD_INVESTIGATION_20261124.md)。

原始 1080P 检测的模型调用仍明显更重，当前优先使用帧率更高的 720P
检测配置。720P 检测逐帧执行，检测帧数与编码帧数一一对应。

## 配置与证据

生产配置 `configs/encoder.toml`、`configs/decoder.toml` 保持最新。
900 帧复测配置：`configs/validation/encoder_performance_720p.toml` 和
`decoder_performance.toml`。交替序列配置：
`configs/validation/encoder_detection_alternating.toml`。

本地证据目录：

- `artifacts/dual-board/opt-parallel-pre-720p-v1/`：300 帧 720P 摄像头。
- `artifacts/dual-board/opt-native-1080p-final/`：300 帧原始 1080P 检测。
- `artifacts/dual-board/opt-720p-long-final/`：900 帧摄像头复测。
- `artifacts/dual-board/opt-alternating-final/`：交替序列及 `check.json`。
- `artifacts/dual-board/opt-fixture-final/`：静态样图、trace 和解码截图。
- `artifacts/detection-profiling/ctest-optimization-final.log`：36 项测试日志。

`tools/prepare_codec_test_frames.py --alternate-blank` 生成交替序列，
`tools/check_detection_sequence.py` 验证逐帧类别、数量及参考框位置。
本轮遵循本地修改、打包、部署的流程，两块板使用 `/root/ULBVC_310P1`。
