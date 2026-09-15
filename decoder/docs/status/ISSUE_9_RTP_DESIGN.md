# Issue #9：MLVC v1 RTP 载荷设计

## 目标

定义传输无关的 MLVC 媒体单元，并用 RTP 承载同一媒体单元；文件和网络路径共享帧语义。

## 媒体单元

- `SCU`：流配置单元，携带 `config_id` 和解码所需配置。
- `EFU`：编码帧单元，携带 `frame_id`、帧类型、QP、参考帧信息和熵编码数据。
- `EOS`：结束单元。

## RTP 载荷

RTP 使用标准 12 字节头，payload type 固定为  dynamic range `96`，时钟频率 90 kHz；每个会话使用随机 SSRC。
载荷采用网络字节序，格式如下：

| 字段 | 长度 |
|---|---:|
| magic `ML` | 2 |
| version | 1 |
| unit_type | 1 |
| flags（首/末片、随机访问） | 1 |
| header_length | 1 |
| config_id | 4 |
| unit_id | 4 |
| fragment_offset | 4 |
| unit_length | 4 |
| fragment_length | 2 |
| payload | N |

## 接收规则

接收端按 SSRC、`config_id`、`unit_id` 建立有界重组上下文；拒绝越界、长度不一致、未知关键版本和超过资源上限的单元。RTP 序列号用于检测丢包、重复和乱序。缺片超时后丢弃整个媒体单元，不得交付部分 EFU。

## 分阶段实现

1. 实现 `RtpMlvcPacketizer` 和 `RtpMlvcReassembler`，加入字节序、边界、乱序和丢包测试。
2. 在编码端增加 RTP 输出模式，保留旧 UDP 模式作为独立适配器。
3. 在解码端增加 RTP 输入模式，验证 SCU、EFU、EOS 端到端重组。
4. 在两台 310P1 上完成 720p/1080p 长测。

## 验收标准

- 正常流 537 帧完整解码。
- 人为丢失、重复、乱序和截断分片均被检测。
- 不发生跨帧混包或部分 EFU 交付。
- 文件路径与 RTP 路径产生一致的帧语义。
