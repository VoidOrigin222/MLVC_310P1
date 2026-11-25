# 在线 JPEG 填充位修正部署与实测：2026-11-24

已在本地实现、编译并部署到编码板，默认配置开启
`camera_jpeg_normalize_padding = true`。1080P 相机输入不变，720P 检测
与 ulbvc 编码逐帧运行，900 帧复测为 **29.8185 FPS**。原始 1080P
检测配置为 **24.3651 FPS**，这一档尚未达到 30 FPS。

本轮只运行编码板。原图 DVPP H.264/RTSP 预览、DVPP 运动估计、检测、
ulbvc 编码和 RTP 发送均启用；RTP 目标改为工作站的验证接收端。
解码板保持停止，未验证本版本双板解码显示或长期散热。

## 同机对照

固定相机 MJPG 1920×1080、30 FPS，QP 11、GOP 96、reset 32，
运动预取 2，检测阈值 0.25。每轮前 30 帧预热，FPS 使用预热后的计时。
前四轮关闭/开启修正的测试使用相同构建版本及同一相机场景。最后一轮
在完成 JPEG 尾部缓冲及可选检测报告判断调整的最终构建上复测。

| 检测分辨率 | 在线修正 | 编码/检测帧数 | 编码 FPS | 相机 JPEGD 路径 ms |
| --- | --- | ---: | ---: | ---: |
| 720P | 关 | 300 / 300 | 27.9343 | 35.7165 |
| 720P | 开 | 300 / 300 | 29.7944 | 22.6186 |
| 720P | 开 | 900 / 900 | **29.8185** | **22.8579** |
| 原始 1080P | 开 | 300 / 300 | 24.3651 | 见原始日志 |
| 720P，最终部署版本 | 开 | 300 / 300 | **29.9147** | 22.5903 |

900 帧开启修正测试中，在线规范化平均 16.9098 ms，已计入表中 JPEGD
路径总耗时；它不是“整条相机路径只要 4.45 ms”。隔离静态样本上的
在线修正 + JPEGD 为 19.2402 ms，其中修正约 14.6860 ms。

基线 300 帧的 V4L2 sequence gaps 为 21；开启修正的 300 帧及 900
帧各为 1。900 帧中检测报告 frame_id 连续为 0～899，检测与编码的
帧号关联检查通过。相机预取会多解一张图，故相机统计为 901 次。
900 帧的原图 RTSP 发布 901 帧、丢弃 0，JPEG 解码错误 0，规范化
异常/不支持格式回退均为 0。检测逐帧执行，没有通过检测抽帧提速。

五轮工作站收到的 UDP 字节数加每包 28 字节 UDP/IP 开销，均等于
编码端记录的 RTP wire bytes。900 帧收到 963 包、305128 字节，对应
332092 wire bytes；这验证包数/字节收齐，没有进行媒体单元重组或解码。
900 帧测试期间编码板采样温度最高 42°C，Health 均为 OK。

相机场景未检出目标，不把空场景吞吐视为任意复杂场景的性能保证。
另用带公交车/行人的正样本及空白负样本检查修正前后的检测结果。

## 正确性检查

- 新增 CPU 测试覆盖重启边界、FF/00 stuffing、幂等性、原输入不变、
  截断和字节变异输入回退。真实相机两张样本的 C++ 输出与独立 Python
  离线修正结果逐字节一致。
- `jpeg_padding`、`encode_config` 两项 CTest 通过；CPU UBSan trap
  构建及测试通过。环境缺少 ASan 运行库，未完成 ASan 检查。
- 在原始慢路径和规范化快路径间比较板上有效 NV12：两张真实相机
  样本的最大绝对像素差为 1，平均绝对差 0.0126907。图像系数没有改变，
  但两条 DVPP 解码路径的像素结果并非逐字节一致。
- 使用 JPEG 重启段尾 0 填充的公交车正样本运行生产 `YoloWorker`，
  原始/修正后均检出 5 框，类别与 label 一致，最小匹配 IoU 为 1，
  最大置信度差 0.000976562。空白负样本前后均为 0 框。
- 检测对照使用空报告路径，顺带修正了可选报告未打开时仍尝试写入的
  判断，改用 `is_open()`；生产非空报告行为保持一致。

这些是定向回归和样本对照，不代替完整 COCO 精度评测。

## 实现与开关

`ulbvc/src/io/jpeg_padding.cc` 支持单扫描、8-bit、SOF0 Huffman JPEG。
它解析 DC/AC 符号来确定每段的真实有效位数，仅规范化末尾 0～7 位，
必要时插入 FF/00 stuffing；不改变 DCT 系数、量化表或重启间隔。
全部边界验证完成后才发布修正后的副本，原始 V4L2 缓冲区不被改写。
不支持或异常输入使用原始码流回退到原解码路径，并累计计数。

`DvppJpegDecoder` 在提交 JPEGD 前执行规范化，副本持续保留至 stream
同步完成，并提供尾部 scratch 空间。新增日志字段
`jpeg_padding_ms`、`jpeg_padding_normalized_frames`、
`jpeg_padding_changed_segments`、`jpeg_padding_unsupported_frames`、
`jpeg_padding_invalid_frames`，便于判断是否真的命中修复。

默认配置在本地 `configs/encoder.toml`；设
`camera_jpeg_normalize_padding = false` 可恢复原 JPEG 输入行为。
性能对照配置为 `configs/validation/encoder_jpeg_padding_*.toml`。
`tools/run_encoder_validation.py` 提供有限帧测试、工作站 UDP 接收、
编码板温度监测及自身进程清理；它不会启动解码板。

本地证据：

- `artifacts/jpegd-online/results.json`：五轮计数、温度、RTP 字节核对、
  像素与检测框对照。
- `artifacts/jpegd-online/profile-summary.json`：CPU trace 汇总。
- `artifacts/encoder-only/jpeg-padding-baseline-720p-v1/`。
- `artifacts/encoder-only/jpeg-padding-720p-v1/`。
- `artifacts/encoder-only/jpeg-padding-720p-long-v1/`。
- `artifacts/encoder-only/jpeg-padding-1080p-v1/`。
- `artifacts/encoder-only/jpeg-padding-720p-final/`：最终部署构建的复测。
- `artifacts/jpegd-online/jpeg-padding-detection-v2.log`。

部署遵循本地修改、打包、编译、同步到设备的流程，两板源码清单与本地
校验一致。生产编解码程序在短测结束后均保持停止，配置和二进制已更新。

## 仍可优化的部分

720P 检测已接近 30 FPS 的相机输入速率。在线 Huffman 解析仍占约
16.9 ms，可以进一步降低 CPU 开销；运动代理平均工作约 29.55 ms。
原始 1080P 检测的 host inference 约 36.08 ms、后处理约 4.26 ms，
仍是这一配置的明显限制。host inference 包含拷贝和调度，不能当作纯
NPU 核函数耗时；各流水阶段存在重叠，耗时不能简单相加。
