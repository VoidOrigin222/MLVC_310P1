# MLVC ACL C++

本工程用于在 Ascend 310P 上运行 MLVC 1080p 推理。它把视频帧送入 Encoder OM，
经过官方 rANS 熵编码生成 MLVC 码流，再由 Decoder OM 解码并输出重建结果。

## 代码组织

代码按“底层库、应用编排、可执行程序、开发工具”分层：

~~~text
common/                     通用底层库（公共接口和实现）
mlvc/                       MLVC 项目底层库（公共接口和实现）
application/                完整应用层（include/ 接口、src/ 实现）
apps/                      正式入口源码（编码、解码）
tools/cpp/benchmarks/      C++ 性能 benchmark
tools/cpp/validation/      ACL、stage、Tensor 验证程序
tools/data/                视频预处理和帧提取工具
tools/*.py                 模型转换、评测、远端接收等脚本
configs/                   运行配置
custom_ops/                Ascend 自定义算子
~~~

`common/include/mlvc/transport` 只处理无业务语义的 UDP Socket、分片、线程和队列；
`mlvc/include/mlvc/io` 负责 MLVC/VideoTrans 消息的编码和解码。MLVC 编解码的 detail
头文件统一位于 `mlvc/include/mlvc/codec/detail`，应用层不再依赖 `mlvc/src` 源码目录。

`common/`、`mlvc/` 中的代码不直接解析命令行；`apps` 中的入口也不实现编解码算法。
底层目标生成 `libmlvc_common.a` 和 `libmlvc_codec.a`，应用编排目标生成
`libmlvc_application.a`；正式应用和
需要应用层能力的 benchmark 再链接后者。文件模式和 UDP 模式共用
`apps/encode_main.cc` 与 `apps/decode_main.cc`，由配置选择传输方式。

详细的文件职责和依赖关系见 [docs/code_architecture.md](docs/code_architecture.md)。

当前测试采用两个独立 app：

~~~text
预处理帧目录 -> mlvc_encode -> .mlvc 码流 -> mlvc_decode -> PNG/MP4
~~~

默认文件模式仍通过文件传递码流；本复制工程另外提供 UDP 逐帧模式，见下文。

## 环境与编译

要求：Ascend 310P、CANN、CMake 3.22+、C++17 和 OpenCV。

~~~bash
cd /root/workspace/cgc/mlvc_ly/mlvc_acl_cppv1
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j2
~~~

`scripts/configure.sh` 会按 Release 配置工程，当前正式构建使用 `-O3`。

## 模型与输入

模型包位于：

~~~text
/root/workspace/cgc/mlvc_ly/mlvc1080p/
├── manifest.json
├── om/MLVCEncoder_ascend310p3.om
├── om/MLVCDecoder_ascend310p3.om
├── gaussian_pmf.json
├── bit_estimator_pmf.json
├── metadata.json
└── sidecars.mlvc.ulbvcsc
~~~

模型输入为 1920x1080 视频预处理后的 FP16 YUV444 数据，OM 固定尺寸为
1x3x1088x1920。底部补 8 行，使用右下 edge padding，输入范围为 [0, 1]，不是 RGB。

速度测试使用预处理帧目录：

~~~text
frame_0.fp16
frame_1.fp16
...
source_info.txt
~~~

从 MP4 生成帧目录：

~~~bash
./build/extract_acl_video_frames \
  --manifest ../mlvc1080p/manifest.json \
  --video /path/to/input.mp4 \
  --output-frame-dir /path/to/frames
~~~

## 运行

编码：

~~~bash
./build/mlvc_encode --config configs/encode_1080p.toml
~~~

解码：

~~~bash
./build/mlvc_decode --config configs/decode_1080p.toml
~~~

720p MLVC 使用独立的 MLVC 模型包 `../mlvc720p/1280x720/`，不能使用
`../../ulbvc_acl_cpp/models/720p_acl/` 的多 stage UL-BVC 模型。720p 的测试视频为
`../../ulbvc_acl_cpp/test_video/60s.mp4`，可直接运行：

~~~bash
./build/mlvc_encode --config configs/encode_720p.toml
./build/mlvc_decode --config configs/decode_720p.toml
~~~

编码配置至少需要指定模型 manifest、输入帧目录、输出码流、设备和帧数：

~~~toml
mode = "encode"
input_frame_dir = "/path/to/frames"
output = "output/1080p.mlvc"
qp = 8
device = 1
gop = 96
reset_interval = 32
frame_num = -1
execution_profile = "pipeline-v1"

[model]
manifest = "../mlvc1080p/manifest.json"

[pipeline]
stream_workers = 1
queue_capacity = 3
frame_buffer_slots = 3
entropy_workers = 1
graph_packet_capacity = 2
~~~

可选的逐帧率控配置：`target_bitrate_bps` 为正时启用，`min_qp` 和 `max_qp` 约束每帧
基础 Q；不设置时保持固定 `qp`。

`device`、`frame_num`、输入/输出路径、`qp`、`gop`、`reset_interval`、LTR 参数和输出格式都从
配置文件读取；缺省值只在对应键缺失时生效。`[pipeline]` 控制执行资源：
`stream_workers` 是有界帧/码流管道的 worker 数，`queue_capacity` 是管道队列上限，
`frame_buffer_slots` 是编码输入帧缓冲槽位数，`entropy_workers` 是 rANS 线程池大小，
`graph_packet_capacity` 是 graph packet 池容量。输入预取和 MLVC 的 DPB/LTR 编解码仍按帧序运行，
因此增加 worker 不会改变有状态 NPU 主循环的串行约束。

## UDP 逐帧模式

UDP 模式使用现有的 `mlvc_encode` 和 `mlvc_decode`，编码器完成一帧的
Encoder OM 与官方 rANS 后立即发送一个 UDP datagram；解码器收到该帧后立即
执行 rANS 解码和 Decoder OM，双方各自维护参考状态，不需要先生成完整码流。

当前版本已移除 JPEG/VideoTrans 转发方案。解码器仅支持本地文件输出，或通过 `output_transport_mode = "raw_fp16_yuv444"` 转发原始 FP16 YUV。RTSP 输出将在 Issue #11 中单独设计。

~~~toml
target_bitrate_bps = 253952.0
min_qp = 4
max_qp = 10
~~~

解码配置示例：

~~~toml
mode = "decode"
input = "output/1080p.mlvc"
output_video = "output/1080p.mp4"
format = "mp4"
device = 1
frame_num = -1
fps = 30

[model]
manifest = "../mlvc1080p/manifest.json"
~~~

## 数据流

Encoder 输出 feature、z_raw、y_raw_0 和 y_raw_1。CPU 按官方顺序执行 rANS 编码：

~~~text
Encoder OM -> feature 回写
          -> y_raw_1 -> y_raw_0 -> z_raw
          -> 写入 .mlvc
~~~

Decoder 按官方顺序执行 rANS 解码，再调用 Decoder OM：

~~~text
.mlvc -> z_raw -> y_raw_0 -> y_raw_1
      -> Decoder OM -> x_hat、feature
      -> MP4 或 PNG
~~~

当前使用异步流水线，使 CPU rANS 与相邻帧的 NPU 推理重叠。Encoder 和 Decoder
内部仍按帧顺序更新 feature 参考状态。

## e1d1 实测结果

### 当前正式 Release 复测

正式 `build` 使用 Release `-O3` 编译，测试输入为 1080p FP16 YUV444 预处理帧目录，
Q8、GOP=96、reset interval=32、LTR 关闭、Ascend 310P Device 1，使用官方 rANS。

#### README 默认命令

按上面的两个命令依次执行 Encoder 和 Decoder，完整处理 1804 帧，Decoder 输出 MP4：

| 项目 | 结果 |
| --- | ---: |
| 码流文件大小 | 1705.71 kB |
| 码流 payload | 1675.69 kB |
| 平均码率 | 28.37 kB/s |
| Encoder FPS（按默认命令复测） | 36.7451 |
| Decoder FPS（包含 MP4 输出） | 18.6945 |
| 解码帧数 | 1804 |
| 输出视频 | 1920x1080、30 FPS、1804 帧 |

输出文件为 `output/videos/1080p.mp4`，编码格式为 H.264、像素格式为 `yuv420p`。

#### 纯推理吞吐

将 Decoder 配置中的 `format` 设为 `none`、不写 PNG/MP4 时，最新完整复测为
`40.0894 FPS`。该结果不包含视频输出开销，不能与默认 MP4 输出结果直接比较。

多次完整运行约为 `36～40 FPS`，FPS 会受 CPU 调度和 NPU 状态影响。

本次纯推理吞吐测试的 Release NPU 全局采样循环间隔为 1 秒，统计覆盖 Decoder 进程完整生命周期，包含启动
和退出阶段的 0% 采样；活跃平均值只统计大于 0% 的采样点。

| 阶段 | 采样点数 | 全生命周期平均 | 活跃采样平均 | 最小值 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decoder | 32 | 66.9688% | 71.4333% | 0% | 75% |

### 历史 Q8 PNG/PSNR 记录

以下结果是旧的测评记录，使用 warmup 16 帧并统计 1788 帧；Decoder 开启 PNG 输出，
且不是当前 Release 纯吞吐结果，仅用于保留重建质量数据。

| 指标 | Encoder | Decoder |
| --- | ---: | ---: |
| 平均耗时 | 24.8871 ms/帧 | 54.5620 ms/帧 |
| FPS | 40.1814 | 18.3278 |
| 码流大小 | 1689.79 kB | 1689.79 kB |
| 平均码率 | 28.3522 kB/s | 28.3522 kB/s |
| 重建结果 | - | 1804 张 PNG |
| 平均 PSNR | - | 29.755249 dB |

历史模块耗时：

| 模块 | 平均耗时 |
| --- | ---: |
| 输入准备 | 5.2327 ms/帧 |
| Encoder OM | 24.0160 ms/帧 |
| Encoder feature 回写 | 0.6368 ms/帧 |
| rANS 编码 | 16.7128 ms/帧 |
| 码流写入 | 0.0053 ms/帧 |
| 码流读取 | 0.0030 ms/帧 |
| rANS 解码 | 54.4155 ms/帧 |
| Decoder OM | 23.1463 ms/帧 |
| Decoder feature 回写 | 0.6218 ms/帧 |
| 输出队列提交 | 2.8416 ms/帧 |

异步流水线中的模块耗时存在重叠，不能直接相加得到 FPS。

结果文件：

- 码流：../results/e1d1_official_q8/60s_1080p_q8.mlvc
- 重建 PNG：../results/e1d1_official_q8/recon_png_full/
- 编码日志：../results/e1d1_official_q8/encoder_q8.log
- 解码日志：../results/e1d1_official_q8/decoder_q8.log

### 单 OM 基准（历史测量）

单 OM 测试不包含输入准备、rANS、码流 I/O 和输出写入：

| OM | 平均耗时 |
| --- | ---: |
| MLVCEncoder | 25.7729 ms/帧 |
| MLVCDecoder | 28.0249 ms/帧 |

## 性能观测

运行期间查看 NPU：

~~~bash
npu-smi info
~~~

生成 Chrome trace：

~~~toml
profile_output = "output/profile.json"
~~~

结果建议统一放在 /root/workspace/cgc/mlvc_ly/output/ 下。
