# 整数平移 warp 板端验收记录

验收日期：2026-10-02。开发分支：`motion`。部署版本在新仓库内通过
`git rev-parse HEAD` 读取；提交后另行交付最终 SHA，不在本文件内写入自引用提交号。

## 部署范围

编码端 `mlvc_encoder`、解码端 `mlvc_decoder` 均使用
`/root/workplace/grifcc/MLVC_310P1`，新增依赖位于 `/root/workplace/grifcc/deps`。
未覆盖原 `/root/workplace/MLVC_c`、`/root/workplace/MLVC_decode` 部署及系统库。
Git 提交身份仅在新仓库配置为 `Grifcc <grifcc@foxmail.com>`。

`translation_warp` 默认关闭。开启后使用 GOP 96、feature reset 32、禁用 LTR；
文件和 RTP 通过版本化能力扩展识别新模式，每个 P 帧新增两个 signed int8 几何字节。
I 帧位移为零。缺失、冲突或不支持的扩展明确拒绝，旧 raw UDP 不支持新模式。
算法、模型边界和配置见 [部署说明](../../docs/translation_warp.md)，
运动后端及依赖构建见 [后端说明](../../docs/motion_backends.md)。

## 输入与判定范围

实际视频为 `614lab.mp4` 前 100 帧，原尺寸 1920×1080、30 FPS，按原 Python inference
色彩转换和归一化生成 FP16 输入，底边 replicate 至模型尺寸 1920×1088。
首帧 fixture 已与原 Python `prepare_frame` 输出逐元素核对，误差为零。
Python 保存的运动 JSON 转为连续全帧 CSV；0、96 帧位移为零。

新模式在 0、96 帧从 0.5 常量参考初始化，在 32、64 帧从上一重构帧初始化。
编码端在 31、63 帧额外解码相同 latent，保存 reset 所需重构帧；P32/P64 仍依赖上一帧。
两端同步对照覆盖全部 100 帧参考 feature，以及 31、63 帧 reset 重构输入。

以下吞吐为本次 100 帧运行的实测值，不能据此承诺长期稳定 30 FPS。
未作画质收益、其他视频或长时间运行的普遍结论。

## 已完成的验证

| 项目 | 结果与证据 |
| --- | --- |
| 原功能基线 | 编码、解码两板各 24/24 基线测试通过 |
| 新功能关闭回归 | 100 帧 synthetic 码流与原模式 SHA256 完全一致，见下方指纹 |
| Python CSV 重放 | 100 帧：I=2、P=98，70 帧非零位移；两端 102 份 trace 的 SHA 列表完全一致 |
| 在线 libx264 | 同一 100 帧 sample 的输出文件与 CSV 重放 SHA256 完全一致；此结论仅适用于该 sample |
| DVPP 文件最终测量 | 无 trace，100 帧完成，70 帧非零位移；两板文件 SHA256 一致；与 libx264 的编码决策、熵字节数量不同 |
| DVPP forced IDR probe | 6 帧、GOP 4、强制 I2；I0/I2/I4 位移零，P1/P3/P5 均为 `(2,-1)` |
| 新参考模型 oracle | 两板常量 0.5 case 误差为零；random 重构范围 case 全部元素通过容差判定 |
| DVPP RTP 两板同步 | 编码、接收解码均 100 帧，I=2、P=98，70 帧非零位移；两端各 102 份 trace 的 SHA 列表完全一致 |
| 最终完整测试 | 两板各 30/30 CTest 通过；编码板 83.03 秒、解码板 82.88 秒；日志均为 `/root/workplace/grifcc/final-ctest.log` |

100 帧实际文件统计如下。`codec_total` 包含熵载荷与几何字节；文件大小另含封装。
98 个 P 帧的几何开销均为 `98 × 2 = 196` 字节，一次性能力头单独属于封装。

| 后端 | 熵字节 | 几何字节 | codec_total 字节 | 文件字节 | 编码 FPS | 解码 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python CSV 重放 | 138818 | 196 | 139014 | 144386 | 29.9168 | 33.2769 |
| 在线 libx264 | 138818 | 196 | 139014 | 144386 | 3.18455 | — |
| DVPP 文件最终测量（无 trace） | 138599 | 196 | 138795 | 144167 | 16.8469 | 33.4945 |
| DVPP 双机 RTP（启用 trace） | 138599 | 196 | 138795 | 不适用 | 14.7898 | 15.4712 |

RTP 行的接收解码 FPS 包含等待实时编码和网络接收，不是纯解码 benchmark。
两端各 102 份 trace 包含全部 100 帧 feature 和 31、63 帧 reset 重构输入；
不仅帧数和字节统计一致，对应 FP16 内容的 SHA256 也完全相同。

DVPP 路径使用持续的 **VENC 硬件 H.264 编码**，随后由软件 H.264 解码器导出运动向量，
并解析 SPS/PPS/slice 校验上一帧参考策略。没有实现或声称 VDEC 硬解运动向量导出。
软件提取和输入转换仍有开销；本结果不构成完整硬件编解码或长期实时性能验收。
forced IDR probe 验证的是硬件代理通道的重置和过去参考运动向量：P1/P3/P5 的向量数量
分别为 926、963、947。它没有改变新 MLVC 模式拒绝动态强制 I 帧的配置约束。

## 参考模型校验

额外 `MLVCReferenceFromFrame` 输出形状为 `[1,96,136,240]`，
包含 48 通道 feature 和 48 通道真实 memory，保留原 Python `feature_adaptor_i` 行为。
输出不运行 `feature_adaptor_p`；主编解码模型继续执行该 adaptor。
原 checkpoint 的 p-adaptor FP16 权重、偏置与两个主 ONNX initializer 严格一致。

两板独立运行 `check_translation_reference_adaptor`，均比较完整 3,133,440 个 FP16 元素：

| case | 最大绝对误差 | 平均绝对误差 | feature 最大误差 | memory 最大误差 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 常量 0.5 GOP 参考 | 0 | 0 | 0 | 0 | exact 通过 |
| random 重构帧范围 [0,1] | 0.009765625 | 0.000300827255572 | 0.0048828125 | 0.009765625 | atol=0.02、rtol=0.02，零超限 |

oracle 来自原 checkpoint 的真实 Python 路径。板端日志位于两板
`/root/workplace/grifcc/reference-{configure,build,constant,random}.log`。

## 关键指纹

| 资产 | SHA256 |
| --- | --- |
| 关闭功能的 100 帧 synthetic 码流 | `7aeae47e1db1658717e2c16a741db61dc511a5693481a9d74a9d7908853ec204` |
| Python CSV / 在线 libx264 的 100 帧文件 | `a917a3efbd5b31b02d9c714ca80146a7e93ec2d83a636aa36000512689907aeb` |
| 两板 DVPP 最终 100 帧文件 | `af4f2d7dc9a33abcaf298df4b17a10143d0ff3f8c669baddd88e7b3243aa310e` |
| 两板 Python CSV trace SHA 列表 | `f38fcb2d725e899c9c5b45eff37c84d320d68ec03a032e58545928091fc2144b` |
| 两板 DVPP RTP trace SHA 列表 | `c163ddf86387810a598b619e8bdf55e8f9c4653fee97727c3e06f74221a8b4e6` |
| Encoder 主 ONNX | `ecaeb1cbff5bb688b1ac69cb293c9ab92463c33bca573ed0be87b9486d1e3581` |
| Decoder 主 ONNX | `905bbdf1e24a10bd706cb217b80f14f1435393baee8d18c9693889406ef3a91c` |
| 原 checkpoint | `cfe12eed07b00cdb98c0cd07136797f3b49e8d52f7b6bd3b81cf0a5312a4aa7c` |
| ReferenceFromFrame ONNX | `27cc437d1537424bb5647407f600b24a25f2cf0a0b04f51db84c218ef8c7eb68` |
| ReferenceFromFrame 310P1 OM（334172 字节） | `91f68c4cf52286ff20f2ca1b80d453fa70d275925d9c80680b56c90874ab962f` |
| 独立 motion manifest | `83e51edf7d9c9f0392b741436c5266319a2b179e3a0b5e9eaa26282e984eaeb0` |

manifest 文件为 `mlvc1080p/manifest_motion_ascend310p1.json`，保留原主模型及 sidecar，
另加参考帧 adaptor 的形状、字节数和哈希；原 manifest 不覆盖。

## 板端验收资产索引

下列路径在编码、解码板的独立部署根 `/root/workplace/grifcc` 下；
配置中的接收地址为 `192.168.5.11`，端口为 `39340`，payload type 为 96。

| 资产 | 路径 |
| --- | --- |
| 编码日志 | `/root/workplace/grifcc/dvpp-rtp100-encode.log`（编码板） |
| 接收解码日志 | `/root/workplace/grifcc/dvpp-rtp100-decode.log`（解码板） |
| DVPP 无 trace 编码日志 | `/root/workplace/grifcc/dvpp-final-encode.log`（编码板） |
| DVPP 无 trace 解码日志 | `/root/workplace/grifcc/dvpp-final-decode.log`（解码板） |
| DVPP forced IDR probe 日志 | `/root/workplace/grifcc/dvpp-forced-idr.log`（编码板） |
| DVPP 最终编码文件 | `/root/workplace/grifcc/MLVC_310P1/output/motion_dvpp.mlvc`（编码板） |
| DVPP 最终解码输入文件 | `/root/workplace/grifcc/MLVC_310P1/output/motion.mlvc`（解码板） |
| 两板 trace 目录 | `/root/workplace/grifcc/trace-dvpp-rtp100` |
| 两板 `sha256sum *.fp16` 清单 | `/root/workplace/grifcc/trace-dvpp-rtp100.sha256` |
| 实际编码配置 | `/root/workplace/grifcc/configs/encoder-motion-rtp.toml` |
| 实际解码配置 | `/root/workplace/grifcc/configs/decoder-motion-rtp.toml` |
| 仓库运行示例 | `configs/encoder_motion_rtp.toml`、`configs/decoder_motion_rtp.toml` |

## 版本交付

测试、文件链路和 RTP 同步验收已完成。主流程提交、推送后另行交付最终 SHA，
并在两板新仓库执行 `git rev-parse HEAD` 核对部署版本一致。

本记录证明列出的 100 帧文件/RTP 链路、两端同步、基线回归和模型数值校验，
不扩展为其他输入或长期性能承诺。
