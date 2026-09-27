# 解码端算子级 Profiling

日期：2026-09-26。设备：Ascend 310P1 解码端（CANN 9.1）。测试使用 priority-7 解码实验副本、537 帧 RTP 输入及原 DVPP VENC/RTSP 输出配置。启用 `msprof --task-time=l1 --model-execution=on --ge-api=l1 --runtime-api=on --ai-core=on --aic-metrics=PipeUtilization` 后，设备端完成 537/537 解码及 537 帧 VENC 发布，`forward_dropped_frames=0`。带 profiler 的 `decode_fps=29.9577` 只记录运行完整性，不作为性能结论。

## 采样范围

Profiler 的有效算子汇总包含 **96 次** `MLVCDecoder_ascend310p1` 执行（Model ID 2），以及 96 次 `MlvcFp16Yuv444ToNv12` 自定义算子执行；应用本身处理了 537 帧。算子时长是这些已采样硬件任务的汇总，不等于全链路墙钟耗时，也不要把不同 stream 的任务时长直接相加来推算 FPS。

首次采集未启用 `--model-execution=on`，只得到模型 API 总时长且硬件任务汇总异常；本报告只使用第二次、打开该开关后生成的 `op_summary` / `op_statistic` 数据。

## 热点结果

| 算子类别 | 96 次采样总时长 | 折算每次解码执行 | 占汇总 AI Core 任务时长 |
|---|---:|---:|---:|
| Conv2D（MLVCDecoder） | 2,035.2 ms | 21.20 ms | 48.45% |
| FP16 YUV444→NV12 自定义算子 | 1,214.6 ms | 12.65 ms | 28.92% |
| StridedSliceD（MLVCDecoder） | 398.9 ms | 4.16 ms | 9.50% |
| TransData（MLVCDecoder） | 335.0 ms | 3.49 ms | 7.98% |
| DepthToSpace（MLVCDecoder） | 174.0 ms | 1.81 ms | 4.14% |

模型内具体热点（`op_summary` 的 Task Duration）：

- `/Slice`：96 次平均约 **2.15 ms**；`/Slice_1`：96 次平均约 **2.01 ms**。两者对形状 `1×6×136×240×16` 的 FP16 `NC1HWC0` 特征做通道切分，各输出一半通道；合计约 **4.16 ms/解码执行**。
- `/hyper_decoder/conv/conv/Conv`：约 **1.28 ms/次**。
- `trans_TransData_2` 和 `trans_TransData_4`：分别约 **1.66 ms** 和 **1.10 ms/次**，涉及 NCHW 与 `NC1HWC0` 数据布局转换。
- Conv2D 汇总中有大量重复的 FFN/卷积节点。若干 FFN 卷积节点单次约 **0.5 ms**，Profiler 报告的 Cube 利用率约 **92–94%**，说明 Conv2D 是模型侧最大的整体成本。

自定义 FP16→NV12 算子在 96 次采样中平均 **12.65 ms**，与应用 trace 中约 12.8 ms/帧吻合，证明本次全链路使用的是 ACL 自定义算子，而非 CPU fallback；之前单算子测试的更低时延没有在全链路里复现。

交叉检查本轮 537 帧应用 trace：`MLVCDecoder` 阶段平均 **31.75 ms/帧**，FP16→NV12 device 事件平均 **12.89 ms/帧**。Profiler 汇总的 Decoder 节点任务时长折算约 **31.10 ms/次**，量级吻合。两者测量范围不同，不能当作完全相同的计时口径。

## 优化优先级

1. **先分析 Conv2D 汇总热点**：它约占模型及输出转换合计 AI Core 任务时长的 48.45%。用具体高耗时 FFN/卷积节点评估 OM 图布局、卷积融合及 ATC 编译结果；精度相关改动必须额外验证重建画质。
2. **评估消除两次大特征切片及布局转换**：`/Slice`、`/Slice_1`、`TransData` 合计约 7.65 ms/解码执行。优先看 ONNX/ATC 图能否把通道拆分并入生产者/消费者，或减少 NCHW↔`NC1HWC0` 往返。
3. **单独优化 NV12 自定义算子**：它仍占约 12.65 ms/次，是显著成本；以本轮相同全链路负载作为验收条件，避免只按隔离 microbenchmark 判断收益。

## 数据文件

本地验收目录保留了 `op_statistic`、`op_summary`、`task_time`、`fusion_op` CSV、CANN timeline 和原始 profiler 采集。为避免将大型、设备相关的原始 profiling 数据提交到 GitHub，这些文件由仓库忽略；本报告保留摘要结果。
