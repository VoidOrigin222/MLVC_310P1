# 平移 warp 部署

新功能由编码配置 `translation_warp = true` 开启，默认关闭。关闭时沿用原 MLVC
参考更新、码流和模型流程。解码器从码流识别新模式。开发与运行目录为两台设备上的
`/root/workplace/grifcc/MLVC_310P1`；Git 分支为 `motion`，提交身份只配置在本仓库。

## 算法与模型边界

运动估计采用 H.264 代理视频的过去参考运动向量，以块面积加权分别求水平、垂直中位数。
向量符号转换为参考帧到当前帧的方向，除以特征步长 8，半格向零取整。平移使用整数切片复制，
未覆盖边界保留原位置。每个 P 帧携带两个有符号字节，GOP 首帧为零。几何字节计入码率控制和
`geometry_bytes`、`codec_total_bytes`，文件总大小还包含 MLVC-ES 封装。

已审计的 1080p FP16 主图将 96 通道参考按 `[0:48]` 特征、`[48:96]` memory 分开。
特征 adaptor 是逐位置 1×1 仿射变换，memory 直接进入解码器。对整个 96 通道参考作同一位置
选择，与 adaptor 后同步平移两部分可交换。`tools/audit_translation_warp_models.py` 检查
消费者、切片、adaptor 属性、权重一致性并执行 FP16 逐位对照。运行时仅接受已审计的主图
SHA256 和参考尺寸，其他模型需重新审计；当前模式不支持 AIPP 模型。

为对齐原 Python 实验，新模式在 0、96、192… 使用 0.5 常量参考帧；在 32、64、128…
使用上一重构帧。额外模型 `MLVCReferenceFromFrame` 从帧运行 `shift_input`、
pixel-unshuffle 和 `feature_adaptor_i`，输出完整 96 通道，保留 memory；它不运行
`feature_adaptor_p`，后者仍在主图内部。编码端仅在这些非 GOP reset 前一帧额外运行
解码器保存重构帧。P32 仍依赖 P31，因此码流保留短参考依赖；它不成为独立随机访问帧。

主图 ONNX 指纹：

- Encoder：`ecaeb1cbff5bb688b1ac69cb293c9ab92463c33bca573ed0be87b9486d1e3581`
- Decoder：`905bbdf1e24a10bd706cb217b80f14f1435393baee8d18c9693889406ef3a91c`
- ReferenceFromFrame ONNX：`27cc437d1537424bb5647407f600b24a25f2cf0a0b04f51db84c218ef8c7eb68`
- ReferenceFromFrame OM：`91f68c4cf52286ff20f2ca1b80d453fa70d275925d9c80680b56c90874ab962f`

## 配置与验证

当前推荐文件编码配置 `configs/encoder_motion_dvpp_prefetch2.toml`，RTP 编码配置
`configs/encoder_motion_rtp_prefetch2.toml`，分别配合 `configs/decoder_motion.toml` 和
`configs/decoder_motion_rtp.toml`。根据设备上的输入帧目录和输出路径调整。
原 `configs/encoder_motion.toml` 保留在线 libx264 medium/1 串行入口及 CSV 重放示例。
模型 manifest 必须同时包含两个主 OM 和新增参考帧 adaptor OM。
配置要求 GOP 96、reset 32、LTR 关闭。支持 MLVC-ES 文件和 RTP，旧 UDP 消息不携带位移。
当前模式拒绝改变 GOP 或强制 I 帧的控制命令，随机访问请求等待自然 GOP。

在线估计默认 `motion_backend = "libx264"`，CRF18、medium、GOP96、无 B 帧、单参考，并关闭 lookahead
以逐帧返回运动。这与原 Python 离线代理的 lookahead 行为可能不同。需要复现实验指令时，
设置 `motion_shifts_file` 重放完整无表头 CSV，每行 `frame_index,kx,ky`，从 0 连续编号，
包含 GOP 首帧零位移；行数必须与实际处理帧数一致。缺失、越界及多余指令会报错。
`export_motion_translation` 提供保留离线 x264 lookahead 的原 YUV 视频代理导出工具。

测试覆盖正负双向与对角平移、全部未覆盖边界、源数据不变、越界拒绝、加权中位数、
真实 x264 运动、P 帧两字节传输及旧/新 reset 调度对照。两板各 30/30 测试已通过；
100 帧文件回归及两端参考/重构同步结果见
[2026-10-02 验收记录](../acceptance/translation-warp-20261002/README.md)。
DVPP 双机 RTP 已完成 100 帧收发，全部 feature 和 reset 重构 trace 一致，
具体统计和性能限制以该记录为准。

仅在新模式下设置环境变量 `MLVC_WARP_TRACE_DIR` 可开启同步诊断。编码和解码进程各使用
独立目录，每帧写 `frame_N.feature.fp16`；非 GOP reset 前一帧另写
`frame_N.reset_frame.fp16`。比较两端对应文件即可验证全部 96 通道及 reset 重构同步。
默认不产生这些文件。

DVPP 代理需要持续的 H.264 编码通道、无 B 帧及单参考，以保留真实 P 帧运动向量。
设置 `motion_backend = "dvpp"` 可使用持续通道的 8 Mbps VBR 硬件编码代理，运动提取仍通过
软件解码器的公开运动向量接口完成。它使用 SPS/PPS/slice 校验确保实际引用上一帧；遇到
多参考、重排或不支持的引用策略会报错。硬件 VBR 与 x264 CRF 的运动决策可能不同。
CSV 指令优先于默认 libx264 设置，CSV 与显式 DVPP 设置同时出现会报配置冲突。
硬件代理的输出须通过软件 H.264 运动提取器审计；DVPP 实验验证结果由部署验收记录说明。

## 板端运行入口

以下命令使用已部署的新仓库、私有依赖和 `fixture100`，配置默认处理前 100 帧。
板端 OpenCV 构建不含 FFmpeg，因此本次入口使用预生成的 FP16 帧目录。
`format = "none"` 会执行完整 MLVC 解码并更新参考，但不输出可播放视频。
运行脚本负责加载 CANN 环境，解码脚本同时加载 custom-op 环境。
运行前在对应板的终端进入新仓库并创建输出目录：

```bash
cd /root/workplace/grifcc/MLVC_310P1
mkdir -p output
```

文件链路先在编码板运行在线 libx264，然后将码流拷贝到解码板同一相对路径：

```bash
# mlvc_encoder
bash scripts/bash_run_encode.sh configs/encoder_motion.toml
```

```powershell
# 本机终端：通过 SSH 别名转存码流，目标为新部署目录。
scp -3 mlvc_encoder:/root/workplace/grifcc/MLVC_310P1/output/motion.mlvc mlvc_decoder:/root/workplace/grifcc/MLVC_310P1/output/motion.mlvc
```

```bash
# mlvc_decoder
bash scripts/bash_run.sh configs/decoder_motion.toml
```

DVPP 文件链路使用已准备的独立配置，输出为 `output/motion_dvpp.mlvc`：

```bash
# mlvc_encoder
bash scripts/bash_run_encode.sh configs/encoder_motion_dvpp.toml
```

```powershell
# 本机终端：拷贝为解码配置使用的 motion.mlvc。
scp -3 mlvc_encoder:/root/workplace/grifcc/MLVC_310P1/output/motion_dvpp.mlvc mlvc_decoder:/root/workplace/grifcc/MLVC_310P1/output/motion.mlvc
```

```bash
# mlvc_decoder
bash scripts/bash_run.sh configs/decoder_motion.toml
```

精确重放保存的 Python 运动时，使用 `encoder_motion.toml` 中说明的
`motion_shifts_file` 项；保持 `motion_backend = "libx264"`，该项会优先选择 CSV 重放。
不应同时设置 CSV 和 DVPP。

RTP 配置为 [encoder_motion_rtp.toml](../configs/encoder_motion_rtp.toml) 和
[decoder_motion_rtp.toml](../configs/decoder_motion_rtp.toml)，默认使用 DVPP 代理、
解码板地址 `192.168.5.11`、专用 UDP 端口 `39340`、payload type 96。
先在解码板启动接收，再在编码板启动发送：

```bash
# mlvc_decoder，保持该进程运行
bash scripts/bash_run.sh configs/decoder_motion_rtp.toml
```

```bash
# mlvc_encoder
bash scripts/bash_run_encode.sh configs/encoder_motion_rtp.toml
```

需要同步诊断时，在两板各自运行命令前设置不同的 `MLVC_WARP_TRACE_DIR`。
目录应为空或全新，例如分别使用 `output/rtp_encoder_trace` 和
`output/rtp_decoder_trace`；不得将不同运行的 trace 混合比较。
本配置已用于 100 帧 DVPP 双机 RTP 验收，结果及资产索引见验收记录。
原串行版本的 100 帧功能验收中，关闭 trace 的 DVPP 文件编码为 16.8469 FPS、
解码为 33.4945 FPS；RTP trace 启用时编码为 14.7898 FPS、接收解码为 15.4712 FPS，
后者包含实时接收等待。这些是优化前的历史结果。
这些结果限定于本次 100 帧样本，不承诺长期稳定 30 FPS。

## 有界预取与运动后端调优

为满足允许延迟 1～2 帧的运行约束，新增实验配置
`configs/encoder_motion_dvpp_prefetch2.toml`（文件）和
`configs/encoder_motion_rtp_prefetch2.toml`（RTP）。它们使用
`motion_prefetch_frames = 2`，单线程按帧序预先计算运动，与主 MLVC 编码重叠。
该选项的默认值为 0，保留原串行路径；可选值为 0、1、2。
运动代理的帧间参考状态不能由多个 worker 并发更新。
帧槽覆盖输入准备、运动队列和正在编码的帧，预取 2 时总共最多持有 3 帧。
EOF 会排空已接受的帧；运动、输入或消费错误会取消输入等待并保留原始异常。
“最多两帧预取”是缓冲上限，实际 ready-to-output 时延还包含运动、编码、熵和传输等待，
不能仅凭帧槽数量认定已满足两帧时间的端到端延迟。

在线预取采用三个阶段：输入线程准备每槽独立的 NV12 缓存，单运动线程按序进行代理
编码和软件 MV 提取，MLVC 消费线程按序编码。NV12 缓存与源 tensor 共用帧槽 lease，
完成 MLVC 编码后才复用；camera 的 external owner 也保留至该时刻。
DVPP 代理将缓存直接 H2D 上传到编码器自有的 DVPP 输入池，省去中间 device-to-device
拷贝。原 `EncodeDevice` 的 device 输入、ready event 和 opt-in zero-copy 行为保留。
`ready_at` 在源 tensor 读完、NV12 转换开始前记录，转换时间仍计入时延。

固定 QP 且开启预取时，消费线程在处理帧前后按序回收已经 ready 的熵任务并发送，
减少已完成码流继续等待后续编码的时间。尚未 ready 的任务不被提前等待；超过原有
两任务窗口时仍使用原兜底回收。码率控制启用时及默认串行模式保留原回收时序，
不改变 rate-control 更新和求解顺序。

`motion_x264_preset` 默认为 `medium`，允许 `medium`、`veryfast`、`superfast`、
`ultrafast`；`motion_x264_threads` 默认为 1，范围 1～16。非默认值只适用于开启 warp 的
在线 libx264，不能与 DVPP 或 CSV 重放组合。preset 和线程数可能改变代理编码决策和
运动向量；不能沿用原 medium/1 的逐位一致性结论。
`configs/encoder_motion_libx264_prefetch2.toml` 使用 ultrafast/8，仅供 A/B 对照，
后端选择以目标负载上的速度和正确性验收为准。

`motion_skip_loop_filter` 默认为 `false`。设为 `true` 时，仅跳过运动提取用软件 H.264
解码器的环路滤波；该解码器的像素输出会被丢弃，运动向量继续从公开 side data 提取。
它不跳过 MLVC 解码、参考更新或输出重构的处理。此开关适用于在线 libx264/DVPP warp，
与关闭 warp 或 CSV 重放组合会报错。两份 DVPP 预取示例已开启此选项；
旧配置和默认行为保持不变。
运动提取器会拒绝引用策略不支持的语法，以及 **FFmpeg 报告的解码错误或损坏帧**。
它不能保证检测任意损坏、截断或错误掩盖；部分截断输入可能不被 FFmpeg 标记为错误。
skip 路径处理本机 libx264/VENC 产生且经过引用策略校验的受限 H.264 代理码流，
不扩大为支持外部任意 H.264 输入。

正式 300 帧文件 A/B 已确认 DVPP 预取 2 达到 30.3582 FPS，同轮 libx264 ultrafast/8
为 27.2604 FPS，当前推荐 `encoder_motion_dvpp_prefetch2.toml` 文件入口及
`encoder_motion_rtp_prefetch2.toml` RTP 配置。优化前的原串行验收保留为历史。
3000 帧重复负载长测排除 96 帧预热后为 30.5254 FPS，三个 900 帧窗口均超过 30 FPS。
输入是 fixture300 硬链接重复 10 遍，fps metadata 没有限速；文件输出平均时延
126.930 ms、最大 176.942 ms，未证明总延迟不超过两帧或 camera 持续无掉帧。
推荐组合的 RTP 100 帧已完成，两板 102 份 trace 全部一致，且与优化前对应 trace 一致。
该运行开启 trace 写盘，只用于一致性验证。无 trace 双机 RTP 3000 帧发送端为
30.543 FPS，三个 900 帧窗口均超过 30 FPS，接收端解码并输出全部 3000 帧。
接收解码 30.5856 FPS 包含等待到达；发送端 source-ready 到 RTP 入队平均 126.813 ms、
最大 179.052 ms，不是网络或显示时延。本次受控重复文件链路未缺帧，不延伸为 camera
实测或端到端两帧延迟承诺。
最新矩阵、同输入码流指纹、重复负载长测及 RTP 验收见
[吞吐验收记录](../acceptance/motion-throughput-20261002/README.md)。
本次 A/B 使用无节拍文件输入；即使文件吞吐达到 30 FPS，也不等同于实时时钟下的
camera 采集、完整网络链路或端到端两帧延迟保证。

已部署源码 `afd4a0f` 的 1080p 实拍在线 3000 帧长测中，编码为 26.7962 FPS，
接收解码为 26.809 FPS（包含等待帧到达），未达到 30 FPS。
本轮量化平移均为零，未验证持续显著全局平移场景；
采集序号缺口 333；全部 3000 帧解码不能代表完整采集链路无掉帧。
相机采集、转换成本、前置队列和完整短/长测结果见
[实时相机在线速度验收](../acceptance/motion-online-20261002/README.md)。
此结果与无节拍文件输入的吞吐验收分别记录。

完整测试入口为 `ctest --test-dir build --output-on-failure`；已完成运行的两板日志均为
`/root/workplace/grifcc/prefetch-validation/tests-final.log`，各 32/32 通过。
增强 motion proxy 负例复跑日志为同目录 `test-parity-fixed.log`，两板通过。
原串行版本的日志保留在 `/root/workplace/grifcc/final-ctest.log`。
新增参考模型的独立数值验证入口是
`build/check_translation_reference_adaptor --manifest <manifest> --input <input.fp16> --expected <oracle.fp16>`。
