# 运动预取吞吐验收记录

验收日期：2026-10-02。本记录补充后续性能优化；
[原 100 帧功能验收](../translation-warp-20261002/README.md) 保留为历史证据。
开发、依赖和运行均位于两板 `/root/workplace/grifcc` 下的新部署，原 MLVC 部署未覆盖。
部署版本通过新仓库内 `git rev-parse HEAD` 读取，最终提交 SHA 另行交付。

## 已确认结论与适用范围

正式生产代码的 300 帧无 trace 文件测试中，DVPP、两帧预取、跳过运动解码环路滤波
组合达到 **30.3582 FPS**；随后 3000 帧重复输入长测排除 96 帧预热后达到
**30.5254 FPS**，三个 900 帧窗口均超过 30 FPS。在线 libx264 ultrafast/8 的正式
300 帧对照为 27.2604 FPS。本轮选择 DVPP 两帧预取作为推荐入口。
默认回归、MV 对照及 RTP 100 帧同步已通过；无 trace 双机 RTP 3000 帧发送端为
**30.543 FPS**，接收解码输出全部 3000 帧，未缺帧。
这不是实时时钟下 camera 持续采集无掉帧的结论。

输入是 `fixture300`，1920×1080 的归一化 FP16 tensor，底边 replicate 至 1088，
帧率 metadata 为 30；QP 11、GOP 96、feature reset 32、禁用 LTR。
输入来自文件、没有按 30 FPS 节拍投喂，未开启 trace。
吞吐包含当前文件编码流程的实际工作，不用 metadata 的 30 代替测量。

两帧预取表示在当前编码帧之外最多持有两帧，不表示输出总延迟只有两帧。
`ready_at` 在源 tensor 读完、NV12 转换前记录；ready-to-output 包含转换、运动、
MLVC 编码、熵任务和文件写入或 RTP 入队调用，终点不是接收或显示。
300 帧推荐组合的文件输出平均值为 126.896 ms、最大值为 174.464 ms，
超过 30 FPS 下的两帧时间约 66.7 ms，因此本轮没有证明“总延迟不超过两帧”。

## 正式 300 帧 A/B 矩阵

下表均开启 `motion_skip_loop_filter`。时延单位为 ms；“未来帧上限”为 lease 计数，
其中 DVPP 串行入口沿用旧输入缓冲设置，不由预取参数限制已有准备队列。

| 运动后端 | 预取 | 编码 FPS | ready-to-encode 平均 | ready-to-output 平均 | ready-to-output 最大 | 未来帧上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DVPP | 0 | 18.9548 | 153.602 | 258.460 | 322.257 | 不以此参数约束 |
| DVPP | 1 | 26.4530 | 71.1936 | 93.8197 | 149.245 | 1 |
| DVPP | 2 | **30.3582** | 94.1647 | 126.896 | 174.464 | 2 |
| libx264 ultrafast / 8 threads | 2 | 27.2604 | — | 138.885 | 191.766 | 2 |

DVPP 三种预取设置的 MLVC 文件 SHA256 完全一致：
`4f2605b39802b2322919ad9868c7ba0d0539e51caddd84000d9c8b22287ffda1`。
它与优化前 DVPP 串行 300 帧结果也完全一致；优化前该次串行吞吐为 16.338 FPS。
此证据证明该 sample 的最终码流保持一致，不扩展为其他输入的通用一致性结论。

在线 libx264 默认 medium/1 的 100 帧文件仍与原结果逐位一致，SHA256 为
`a917a3efbd5b31b02d9c714ca80146a7e93ec2d83a636aa36000512689907aeb`。
其他 preset/线程数可能改变运动向量和 MLVC 码流，不能继承这个结论。
先前 8 线程参考测量约为 medium 12.20、veryfast 16.84、superfast 17.93 FPS，
本轮未选用；正式后端对比以表中重测的 ultrafast/8 为准。

## 正式 3000 帧重复输入长测

长测配置为 `final-dvpp-3000.toml`。输入将 `fixture300` 用硬链接重复 10 遍，
因此 3000 帧是重复负载，不是 3000 个独立真实帧。未按 30 FPS 节拍投喂，也未开启
trace；fps metadata 不执行限速。编码 FPS 排除前 96 帧预热，测量 2904 帧；
时延统计包含全部 3000 帧。

| 指标 | 结果 |
| --- | --- |
| 测量编码 FPS（排除预热） | **30.5254** |
| 连续三个 900 帧窗口 FPS | 30.4825 / 30.4919 / 30.5566 |
| 编码帧数 | 3000：I=32、P=2968 |
| 非零运动帧 | 2631 |
| 最大持有帧槽 / 未来帧上限 | 3 / 2 |
| 提前回收 ready 熵任务数量 | 2999 |
| source-ready → codec 平均 / 最大 | 94.1052 / 143.641 ms |
| source-ready → 文件 output 平均 / 最大 | 126.930 / 176.942 ms |
| 熵载荷 / 几何 / codec total | 3678792 / 5936 / 3684728 字节 |
| 文件大小 | 3840900 字节 |

文件 SHA256：`d8bcaba9c2fdac6daa913e44d4dc860c5558a4bf86b53a54ca463bcd2a3ebd66`。
编码板日志：`/root/workplace/grifcc/prefetch-validation/final-dvpp-3000.log`。
本结果验证了该重复负载下持续文件编码吞吐；输出延迟仍高于两帧时间，
不证明完整 camera/网络链路的实时吞吐、持续无掉帧或端到端两帧延迟。

## 运行配置与架构

推荐文件入口为 [encoder_motion_dvpp_prefetch2.toml](../../configs/encoder_motion_dvpp_prefetch2.toml)，
RTP 入口为 [encoder_motion_rtp_prefetch2.toml](../../configs/encoder_motion_rtp_prefetch2.toml)。
两者使用 DVPP、`motion_prefetch_frames = 2`、`motion_skip_loop_filter = true`。
示例默认使用 `fixture100`、处理 100 帧；复现本矩阵时应改为实际 `fixture300` 路径及
`frame_num = 300`。libx264 对照入口为
[encoder_motion_libx264_prefetch2.toml](../../configs/encoder_motion_libx264_prefetch2.toml)，
使用 ultrafast/8，供 A/B 对照。

```bash
# 编码板，在新部署目录运行。
cd /root/workplace/grifcc/MLVC_310P1
mkdir -p output
bash scripts/bash_run_encode.sh configs/encoder_motion_dvpp_prefetch2.toml
```

文件编码完成后，在本机终端转存到解码配置的输入路径：

```powershell
scp -3 mlvc_encoder:/root/workplace/grifcc/MLVC_310P1/output/motion_dvpp_prefetch2.mlvc mlvc_decoder:/root/workplace/grifcc/MLVC_310P1/output/motion.mlvc
```

```bash
# 解码板
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run.sh configs/decoder_motion.toml
```

RTP 先启动解码板接收，再启动编码板；地址 `192.168.5.11`、端口 `39340`：

```bash
# 解码板
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run.sh configs/decoder_motion_rtp.toml
```

```bash
# 编码板
cd /root/workplace/grifcc/MLVC_310P1
bash scripts/bash_run_encode.sh configs/encoder_motion_rtp_prefetch2.toml
```

完整文件传输、解码和诊断操作见 [部署说明](../../docs/translation_warp.md)。
输入准备、运动代理和 MLVC 编码分别执行；NV12 缓存与源 tensor 共用 slot lease，
三帧槽限制当前帧加两帧未来帧。DVPP 直接 H2D 到编码器自有池，省去中间 D2D。
固定 QP 且开启预取时按帧序提前发送已 ready 的熵任务，保留未 ready 和窗口满时的
兜底回收；有码率控制及默认串行模式保持旧时序。

DVPP 仅代理 H.264 编码由 VENC 硬件完成，MV 仍由软件解码器导出，不声称 VDEC 硬解。
skip 仅作用于运动软件解码器的环路滤波，其像素输出会丢弃，不跳过 MLVC 重构。
该路径处理本机受限代理码流，不扩大支持外部任意 H.264；拒绝 FFmpeg 报告的解码
错误或损坏帧，不保证检测任意损坏、截断或错误掩盖。

## 最终功能回归与运动向量对照

两板正式代码各 **32/32 CTest 通过**；增强的 motion proxy 错误负例随后在两板复跑
通过，分别耗时 0.60 / 0.59 秒。
已有预取测试覆盖顺序、EOF drain、两帧上限、slot 独立缓存、取消、原异常保留及
ready_at 在转换前的计时边界。默认行为保持预取 0、x264 medium/1、skip=false。

| 回归 / 对照 | 结果 |
| --- | --- |
| warp 关闭，100 帧 synthetic | 与旧 baseline 文件 SHA256 相同：`7aeae47e1db1658717e2c16a741db61dc511a5693481a9d74a9d7908853ec204` |
| DVPP 推荐组合，100 帧文件 | 30.4066 FPS，文件 SHA256 与优化前相同：`af4f2d7dc9a33abcaf298df4b17a10143d0ff3f8c669baddd88e7b3243aa310e` |
| 在线 libx264 默认 medium/1，100 帧 | SHA256 与原默认路径相同，见前文 `a917…07aeb` 完整指纹 |
| DVPP host upload，300 帧六 MV 字段逐项对照 | 同一代理 packet 分别用 baseline/fast 解码；全部通过，2380339 个向量，field hash `8653638112290388609` |
| 原 DVPP device API，14 帧 | 640×384、GOP 4、强制 I2；8508 个向量，field hash `985445601634293040`，通过 |

host 对照命令为 `check_dvpp_motion_proxy 300 96 1920 1080 32 1`，覆盖正向、静止、
负向运动、GOP 和强制 I 帧。专门构造的 damaged P 帧触发了真实 FFmpeg 错误并被拒绝；
这个负例不意味着能够检测任意损坏。

## 推荐组合的 RTP 100 帧同步

两板各生成 102 份 FP16 trace（100 份 feature、31/63 帧 reset 重构），对应文件全部
SHA256 一致，清单 SHA256 为
`c163ddf86387810a598b619e8bdf55e8f9c4653fee97727c3e06f74221a8b4e6`，
也与优化前 RTP trace 一致。

编码与接收均 100 帧，I=2、P=98，70 帧非零运动，熵字节 138599、几何字节 196，
codec total 为 138795 字节。解码 `frames=100`、`output_frames=100`、`forward_drop=0`。
trace 写盘使编码为 25.3046 FPS、接收解码为 25.7899 FPS；本次运行验证一致性，
不用这些值作无 trace 性能结论，接收解码速度也包含实时接收等待。

## 正式无 trace 双机 RTP 3000 帧长测

此测量使用同一 `fixture300` 重复十遍的 3000 帧输入，不按 metadata 的 30 FPS
限速，未写 trace。发送端排除前 96 帧预热，与文件长测的测量范围相同。

| 指标 | 结果 |
| --- | --- |
| 发送端编码 FPS | **30.543** |
| 连续三个 900 帧发送窗口 FPS | 30.4945 / 30.5505 / 30.5558 |
| 接收解码 FPS | 30.5856，包含等待帧到达，不是独立解码极限 |
| 编码 / 接收帧统计 | 3000，I=32、P=2968，非零运动 2631 |
| 解码 / output / forward_dropped | 3000 / 3000 / 0 |
| 两端熵载荷 / 几何 / codec total | 3678792 / 5936 / 3684728 字节 |
| source-ready → codec 平均 / 最大 | 93.9944 / 146.803 ms |
| source-ready → RTP enqueue 平均 / 最大 | 126.813 / 179.052 ms |

RTP enqueue 时延不包含后续网络传输、接收和显示，不能当作完整网络或显示时延。
100 帧 trace 验证了两端参考状态逐位同步；3000 帧无 trace 验证了本次受控重复文件输入
在两板 RTP 链路上持续约 30.5 FPS 且全部帧输出。两种运行共同覆盖一致性与吞吐，
不延伸为 camera 实时采集、独立真实 3000 帧内容或端到端两帧延迟承诺。

## 验收资产索引

日志根目录：两板 `/root/workplace/grifcc/prefetch-validation/`。
推荐配置位于新仓库 `configs/`；版本通过新仓库 `git rev-parse HEAD` 读取。

| 资产 | 文件 / 路径 |
| --- | --- |
| 两板完整测试 | `tests-final.log` |
| 两板增强负例复跑 | `test-parity-fixed.log` |
| 编码板 host MV 对照 | `dvpp-mv-parity300.log` |
| 编码板原 device API | `dvpp-device14.log` |
| 编码板 warp-off 回归 | `legacy-final100.log` |
| 编码板默认 libx264 回归 | `default-x264100.log` |
| 编码板 DVPP 最终 100 帧文件 | `final-dvpp100.log` |
| 编码板重复负载长测 | `final-dvpp-3000.log` |
| RTP 编码 / 接收解码 | `rtp100-encode.log` / `rtp100-decode.log` |
| RTP trace 清单 | `rtp100-trace.sha256` |
| 两板 RTP trace 目录 | `/root/workplace/grifcc/trace-prefetch-rtp100` |
| RTP 3000 帧编码 / 接收解码日志 | `rtp3000-encode.log` / `rtp3000-decode.log` |
| RTP 3000 帧实际配置 | `rtp3000-encode.toml` / `rtp3000-decode.toml` |

最终独立构建程序的 SHA256（不同板和不同程序不要求哈希相等）：

| 程序 | SHA256 |
| --- | --- |
| 编码板 `build/mlvc_encode` | `7986728cebe5dc5db262e06677e07f41e19b6c869607c1d0d91a1219550b16d6` |
| 解码板 `build/mlvc_decode` | `cc859df4b5d6583bba69f515b209f28b9cd358ce150effeee40554dfd48840b8` |

本记录内的测试、回归、MV 对照、文件和双机 RTP 验收已完成。
主流程提交、推送后另行交付最终 SHA，并核对两板新仓库 HEAD 一致。
