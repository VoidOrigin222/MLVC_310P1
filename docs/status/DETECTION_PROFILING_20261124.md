# 检测与编码并行性能定位：2026-11-24

本页记录优化前的实现与测量。当前实现与复测结果见
[优化结果](DETECTION_OPTIMIZATION_20261124.md)。

已定位到检测工作线程的预处理是当前主要瓶颈之一。720P 检测每帧的
预处理、模型调用、后处理在同一个线程内依次执行，共约 49.39 ms，
与整链路约 19.60 FPS 的结果一致。30 FPS 对应每帧 33.33 ms。

## 并行方式

同一张源图像进入两条分支：YOLO11s 检测和 ulbvc 编码。检测有独立的
工作线程和 ACL stream，使用异步模型调用后等待自己的 stream 完成；
编码分支同步执行 codec 模型，并与运动估计、熵编码组成流水线。
两者共享同一块编码板的 NPU、CPU 和内存资源。

发送端在相同 frame ID 的编码和检测结果都完成后，才将码流及检测 TLV
入队到 RTP。软件分支并行不等于两个模型的全部算子都同时占用硬件。

## 对照测试

每组 120 帧，源相机/编码均为 1920×1080，QP 11，原图预览开启，运动估计、
RTP 和下位机 RTSP 输出开启。三组都成功编码/解码 120 帧；开启检测的
两组也都有 120 份匹配的检测结果。解码端转发丢帧数均为 0。

| 检测配置 | 整链路编码吞吐 FPS | ulbvc 编码模型调用均值 ms |
| --- | ---: | ---: |
| 关闭检测 | 26.67 | 25.73 |
| 原始 1080P 检测 | 17.50 | 29.25 |
| 缩放至 720P 检测 | 19.60 | 26.80 |

这些短测包含启动后的前几帧，未设置 warmup。与前一轮 300 帧测量的
28.75 / 17.95 / 19.84 FPS 有差异，不能将两轮测试混作同一次对照。

| 检测线程每帧耗时 | 1080P ms | 720P ms |
| --- | ---: | ---: |
| NV12 → RGB | 7.91 | 9.65 |
| RGB 缩放 | 0.00 | 14.63 |
| RGB → FP16 NCHW、归一化和排布 | 16.18 | 7.17 |
| 预处理合计 | 24.09 | 31.44 |
| 模型调用，包括输入/输出拷贝及同步等待 | 27.64 | 15.87 |
| 后处理、还原框和 NMS | 3.82 | 2.08 |
| 检测线程串行合计 | 55.55 | 49.39 |

720P 模型调用比 1080P 少约 11.77 ms，但增加了缩放，预处理多约
7.35 ms，因此降低检测分辨率后，整体吞吐提升有限。

编码器相机分支还在 CPU 上执行 BGR → FP16 YUV444，均值约 27–29 ms；
检测又独立做颜色转换和排布，存在重复处理同一源图像的开销。

## 等待与输出

| 每帧主机侧等待/输出 | 1080P ms | 720P ms |
| --- | ---: | ---: |
| 提交检测时等待有界队列空间 | 13.12 | 10.25 |
| 已提交的检测任务排队延迟 | 103.39 | 85.02 |
| 发送端额外等待检测 future | 21.27 | 17.64 |
| RTP SendFrame 入发送队列 | 0.09 | 0.21 |

检测线程吞吐不足时，上游排队，编码输出也会等检测。增大队列不能消除
每帧约 49 ms 的工作量，还会增加排队延迟。RTP 入队在这次低码率场景
不是主要瓶颈；此项不代表网络端到端传播时延。

以上不同线程的时间及嵌套计时不能全部相加。例如发送端等待和检测执行
可能属于同一段墙钟时间。

## CANN 核查

另外采集了 120 帧 720P 检测并行运行的 CANN trace，吞吐 19.70 FPS。
这组开启底层采集，有额外开销，单独保存，未替换上表普通计时结果。

检测线程每帧的 API 主机耗时：输入 H2D 约 2.30 ms，输出 D2H 约
1.42 ms，异步提交约 0.27 ms，等待检测 stream 约 11.04 ms。
这说明原有 `detection_inference_avg_ms` 是模型调用边界，包含拷贝和
同步等待，不能作为纯计算时间解释。

模型 ID 4 由 YOLO 算子名及独立异步调用线程核对，模型 ID 1 是 ulbvc
编码模型。CANN step trace 中两者各有 120 次执行记录；设备模型执行
区间均值约为 10.43 ms 和 18.51 ms。两者执行区间存在部分重叠，120 帧
累计交集约 305.97 ms。这个区间包括调度及算子间隙，不等于 AI Core
同时计算的时间，也不能据此推算 AI Core 利用率。

原始任务表中有等待/通知任务，其计时不能当作算子计算时间累加。
字段解释参考[华为 CANN 任务调度数据说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0062.html)。

## 优化顺序

1. 优先降低检测的 CPU 预处理耗时，重点是 720P 缩放以及重复颜色转换；
   对共享源图像设计可复用的预处理结果。
2. 分离/流水化预处理与模型执行，测量它们之间的重叠收益，同时保持
   frame ID、输入缓冲区生命周期和框坐标映射正确。
3. 继续减少同步拷贝/等待，并检查实际设备算子时间线中的资源争用。
4. 优化后再做摄像头 300 帧对照和有目标样图的同步/叠框回归，验证
   30 FPS 是否达到。当前测量还不能承诺 30 FPS。

## 复现与证据

本轮增加细分计时、独立 detection trace 轨道、发送端等待计时，以及
`tools/analyze_detection_profile.py`。诊断配置在
`configs/validation/encoder_profile_{baseline,1080p,720p}.toml` 和
`decoder_profile.toml`。默认工作配置仍使用 720P 检测和 1080P 编码。

```bash
python tools/run_dual_board_validation.py \
  --encoder-config configs/validation/encoder_profile_720p.toml \
  --decoder-config configs/validation/decoder_profile.toml \
  --name unique-profile-name --timeout 180 --profile-npu
```

设备侧 CANN 未自动解析时，可加载 `scripts/env.sh` 后，运行
`${CANN_HOME}/tools/profiler/profiler_tool/analysis/msprof/msprof.py`
的 `export summary -dir <PROF目录>` 和 `export timeline -dir <PROF目录>`。

本地证据：`artifacts/dual-board/profile-{baseline,1080p,720p}-v1/`、
`profile-720p-npu-v1/`，以及 `artifacts/detection-profiling/summary.json`
和 `cann/` 中的 CANN 原始导出 CSV/JSON。9 项相关板端测试通过，
覆盖检测元数据、配置、流水线顺序和 RTP。
