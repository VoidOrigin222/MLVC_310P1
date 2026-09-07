# C++ Code Architecture

本工程分为四层。依赖方向从应用向底层单向流动：

```text
apps/ and tools/
        |
application/src/
        |
mlvc/src/codec, mlvc/src/io, mlvc/src/entropy
        |
common/src/runtime, common/src/core, common/src/framework
```

## 目录职责

| 目录 | 职责 | 不应该放什么 |
| --- | --- | --- |
| `common/include/mlvc/core` | Tensor、Buffer、Status、数据类型 | 业务流程 |
| `common/include/mlvc/runtime` | ACL context、OM stage、manifest | 命令行解析 |
| `common/include/mlvc/framework` | 异步任务、graph executor、熵 worker、profiling | 编解码业务决策 |
| `common/include/mlvc/entropy` | 基础 byte-wise rANS | MLVC 码流格式 |
| `common/include/mlvc/transport` | UDP Socket、分片、收发线程和消息队列 | MLVC header/frame 语义 |
| `mlvc/include/mlvc/codec` | MLVC 编解码算法接口、帧类型、码率控制 | `main()` |
| `mlvc/include/mlvc/codec/detail/frame` | I/P 编解码、参考帧状态 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/stage` | stage 类型、执行、workspace、输出策略 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/quantization` | QScale 量化步长缓存 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/rate_control` | QP 和目标码率控制 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/tensor` | FP16、float、int8 Tensor 转换 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/prior` | prior 分块、重排和恢复 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/entropy` | 异步熵输入和 pinned buffer | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/profile` | 编解码和内存分配 profiling | MLVC 项目外部调用 |
| `mlvc/include/mlvc/codec/detail/memory` | 编解码 scratch 工作区 | MLVC 项目外部调用 |
| `mlvc/include/mlvc/entropy` | MLVC sidecar 和官方熵编码接口 | UDP 或视频写出 |
| `mlvc/include/mlvc/io` | 视频帧、`.mlvc` 文件、MLVC UDP 消息序列化 | NPU stage 调用 |
| `mlvc/include/mlvc/runtime` | MLVC 专用 ACL prior/index 算子接口 | 通用 ACL runtime |
| `mlvc/include/mlvc/bitstream` | MLVC SPS/NAL 结构 | 通用文件 IO |
| `application/include/mlvc/application/input` | 输入队列接口 | 底层 Tensor 实现 |
| `application/include/mlvc/application/pipeline` | MLVC 帧/码流 DataObject 与生产者适配 | 通用任务调度实现 |
| `application/include/mlvc/application` | 进度显示接口 | 底层 Tensor 实现 |
| `application/include/mlvc/application/cli` | 配置到运行参数的适配接口 | 具体 OM 推理 |
| `application/include/mlvc/application/stream` | 高层 stream 编排接口 | 底层 Tensor 和 ACL 细节 |
| `common/src/` | 通用底层接口的实现 | MLVC 业务流程 |
| `mlvc/src/` | MLVC 项目底层接口的实现 | 可执行程序入口 |
| `application/src/` | 配置、输入、stream 级编排 | 独立底层算法 |
| `apps/` | 用户直接启动的编解码程序入口源码 | benchmark/验证逻辑 |
| `tools/cpp/benchmarks` | 独立性能测试程序 | 正式运行入口 |
| `tools/cpp/validation` | 一次性检查和 ACL 验证程序 | 核心库实现 |
| `tools/data` | 输入视频和帧目录处理工具 | 编解码主循环 |

## 应用层

正式入口位于 `apps/`：

```text
apps/encode_main.cc       -> ParseConfigPath -> LoadEncoderConfig
                          -> create MlvcCodecRuntime + Profiler/Graph/EntropyWorker
                          -> RunEncodeStream(config.stream, services)
apps/decode_main.cc       -> ParseConfigPath -> LoadDecoderConfig
                          -> create MlvcCodecRuntime + Profiler/Graph/EntropyWorker
                          -> RunDecodeStream(config, services)
```

两个入口按顺序完成运行时环境准备、命令行解析、配置加载和应用执行，不在 `main()` 中
直接创建 OM、熵编码或视频 IO 对象。配置加载和编解码执行保持为两个明确步骤。
配置解析位于 `application/src/cli/`，输入预取位于 `application/src/input/`，进度显示位于
`application/src/progress.cc`。

分辨率不是 C++ 分支条件：模型 manifest 中的 tensor shape 决定输入、latent 和参考特征尺寸。
`mlvc1080p/` 与 `mlvc720p/1280x720/` 分别是 1080p 和 720p 的 MLVC 运行包；每个包包含
`MLVCEncoder`/`MLVCDecoder` OM、熵 PMF 生成的 sidecar 和 manifest。UL-BVC 的 720p 多 stage
模型不满足 MLVC 的两 stage runtime 契约，不能作为 MLVC manifest 使用。

高层 stream 编排位于 `application/src/stream/`：

- `stream/mlvc/common.cc`：MLVC 编解码共享的模型检查、源视频尺寸、运行时状态和参考特征辅助函数。
- `stream/mlvc/encode.cc`：`RunEncodeStream` 编码入口，只负责运行时组装、生产者/消费者管道生命周期和结果统计。
- `stream/mlvc/encode/encode_setup.cc`：编码输入校验和 MLVC 码流 header 构造。
- `stream/mlvc/encode/encode_state.cc`：GOP、Reset、DPB 和 LTR 参考状态。
- `stream/mlvc/encode/encode_frame.cc`：单帧 QP 决策、Encoder OM 执行、latent 提取和 rANS 任务提交。
- `stream/mlvc/encode/encode_output.cc`：rANS future 的顺序刷新、码流文件/UDP 输出和码率统计。
- `stream/mlvc/decode.cc`：解码侧码流接收、rANS 解码、DPB/LTR 恢复、视频写出和转发。

## 底层库

`common/src/runtime/` 只关心通用 Ascend 运行时和模型执行。`AclRuntime` 管理 device/context/stream，
`AclStage` 管理单个 OM，`AclModelSet` 按 stage 名称管理模型集合。

`mlvc/src/codec/` 负责 MLVC 算法本身；`mlvc/include/mlvc/codec/detail/` 保存实现所需的
detail 类型和调度接口。它们不是稳定的外部 API，但通过 MLVC 库的 include 根统一引用，
不再依赖 `mlvc/src` 的私有源码路径。应用和工具应优先使用 `mlvc/include/mlvc/codec`
中的公开接口；当前 stream 编排因需要维护 DPB、LTR 和 stage workspace，仍显式使用
`detail` 接口，后续可再收敛为更高层的 codec session。

`common/src/entropy/` 提供基础 rANS；`mlvc/src/entropy/` 实现 sidecar 驱动的官方 MLVC
熵编码；`common/src/transport/` 负责无业务语义的 UDP 分片、Socket、线程和队列，
`mlvc/src/io/` 只负责 MLVC/VideoTrans 消息序列化；`common/src/framework/` 提供线程池、
生产者/消费者管道和 profiling 基础设施。

### 任务与内存接口

`common/include/mlvc/framework/task_executor.h` 定义 `Task`、`FunctionTask` 和
`TaskExecutor`；`common/include/mlvc/framework/thread_pool.h` 提供真正的 worker 线程池、
任务队列、future 和停止语义。任务带有名称和帧号，执行器只负责排队和线程生命周期；应用层可以把
编码、熵处理或发送工作注册为任务，而不把线程创建散落在 `main()` 和帧循环中。当前
`EntropyWorker` 通过可配置线程数的 `ThreadPool` 实现该接口，并继续提供 MLVC 所需的带返回值
提交函数；编码和解码配置中的 `[pipeline].entropy_workers` 控制线程数。

运行资源配置在 `[pipeline]` 中统一读取：`stream_workers`、`queue_capacity`、
`frame_buffer_slots`、`entropy_workers` 和 `graph_packet_capacity` 分别控制管道 worker、
有界队列、编码输入缓冲槽位、rANS 线程池和 graph packet 池。设备号、帧数、路径、QP/GOP/LTR
参数及解码输出格式继续由配置文件的对应键控制，并保留代码默认值以兼容旧配置。

`application/include/mlvc/application/input/frame_buffer_pool.h` 定义 `FrameBufferPool`，
`FrameInputArena` 是当前固定槽位实现。`AsyncFrameInputQueue` 只依赖该接口，不再依赖具体
的 `std::vector<TensorData>` 布局；后续可替换为 pinned、device 或共享内存池而不改输入队列。

`application/include/mlvc/application/runtime/mlvc_codec_runtime.h` 的 `MlvcCodecRuntime` 统一
拥有一个 pipeline 所需的 `StageRuntime`、`StageModelSet`、`RuntimeSidecar` 和
`StageOutputWorkspace`。`main()` 创建它并通过 `EncodePipelineServices`/
`DecodePipelineServices` 注入，同时注入 profiler、graph executor 和熵线程；因此模型、ACL
上下文和 workspace 的生命周期覆盖整个应用执行。独立调用 stream 接口时仍允许传入空指针，
由 stream 创建自包含的运行时服务，便于小型测试直接复用。

通用生产者/消费者框架位于 `common/include/mlvc/framework/`：`DataObject` 是跨层传递的
类型擦除基类，`DataProducer`、`DataConsumer` 管理线程，`StreamingPipeline` 提供有界队列和
背压，`ThreadPool` 提供可复用的后台任务线程。应用层只在
`application/pipeline/codec_frame_pipeline.*` 定义 MLVC 适配对象：编码使用
`PreparedFramePacket`，解码使用 `BitstreamPacket`，并通过 `CallbackDataConsumer` 消费。
MLVC 路径由独立生产者线程读取输入，经管道转发后由消费者按帧序执行 NPU、DPB/LTR
和输出；没有引入 DAG，避免把有状态编解码强行拆成无序任务图。
NPU/DPB 仍保持串行，避免有状态参考帧被无序并行访问；输入预取、任务调度和 rANS worker
可以独立并发。`stream_workers` 当前作用于无状态的管道转发/背压层，不能把 MLVC 的有状态
编码或解码循环变成无序并行执行。

UDP transport 对发送和接收队列设置固定上限；分片接收使用 socket 超时，首片丢失或中途
丢片会报告错误并退出接收线程，不会无限等待。stream 的 `ScopedRuntimeState` 会在返回时
恢复全局 ACL 状态，避免把已销毁的 workspace、stream 或 graph 指针留在全局变量中。

`mlvc/src/codec/` 下的实现目录与 detail 头文件一一对应：`frame` 处理 I/P 帧和参考状态，
`stage` 处理 OM stage 调度，`rate_control` 处理 QP 和目标码率，`quantization` 只处理
QScale，`tensor` 处理 Tensor 格式转换，`prior` 处理 prior 排布和恢复，`entropy` 处理异步
熵输入，`profile` 处理性能和分配统计，`memory` 处理工作区。新增实现应放入对应模块，
不要直接堆在 `mlvc/src/codec/` 根目录。

## 两条运行路径

文件模式：

```text
mlvc_encode
  -> application/src/cli
  -> application/src/stream
  -> Encoder OM + rANS
  -> io::MlvcBitstreamWriter
```

UDP 模式：

```text
mlvc_encode（UDP 配置）
  -> Encoder OM + rANS
  -> io::UdpMlvcSender
  -> io::UdpMlvcReceiver
  -> Decoder OM + rANS
  -> io::AsyncVideoTransUdpSender (可选)
```

UDP、熵编码、输入预取、任务管道和 JPEG 转发分别使用自己的队列/线程；DPB 和 LTR 状态仍由
编解码主线程按帧序维护。`ScopedRuntimeState` 还用进程内互斥保护全局 ACL 状态，避免两个
pipeline 同时改写全局 workspace/stream 指针。

## 构建边界

CMake 将通用底层编译为 `mlvc_common`，MLVC 底层编译为 `mlvc_codec`，应用源文件编译为
`mlvc_application`，依赖关系为 `mlvc_common -> mlvc_codec -> mlvc_application`。
新增用户功能应放在 `apps/`；新增通用能力应放在 `common/`；新增 MLVC 能力应放在
`mlvc/`；新增应用层接口应放在 `application/include/mlvc/application`，实现放在对应的
`application/src/`；
一次性测试放在 `tools/cpp/validation`，性能测量放在 `tools/cpp/benchmarks`。
