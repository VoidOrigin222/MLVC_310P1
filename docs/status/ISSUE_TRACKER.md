# MLVC Issue 台账

更新时间：2026-09-30（GitHub 状态已核对；已关闭 issue 保持原状）

| 编号 | GitHub 状态 | 本地审查结论 |
|---|---|---|
| #1 | closed | 已关闭 |
| #2 | closed | 已关闭 |
| #3 | closed | 已关闭 |
| #4 | closed | 解码入口拒绝 stream/model 输出 shape 不匹配 |
| #5 | closed | 已关闭 |
| #6 | closed | 已关闭 |
| #7 | closed | 配置校验通过；RAW YUV UDP 功能已移除，rate-control 整数预算有上限 |
| #8 | closed | 已关闭 |
| #9 | open | Grifcc 评论中的 timeout、完整单元去重/跨帧重排、FPS 时间戳与 EOS 边界问题已修复并做板端回归 |
| #10 | closed | pacer 重复计账及队列/wire 字节计量已修正；由 issue 作者关闭 |
| #11 | open | 1080p 链路已实测；RTSP 已改为直接 libavformat 发布，板端 CTest 23/23 通过 |
| #12 | closed | 在途额度包含重排结果；由 issue 作者关闭 |
| #13 | open | manifest 摘要覆盖 metadata、Gaussian PMF 与 bit-estimator PMF；已回复修复和验证结果 |
| #14 | open | EFU 校验失败按损坏帧丢弃，接收会话继续；已回复修复和验证结果 |
| #15 | open | 丢帧后在有界重排/恢复窗口中继续接收；已回复修复和验证结果 |
| #16 | open | 短暂跨帧乱序在有界窗口中等待/重排；已回复修复和验证结果 |

## 审查说明

GitHub 当前共有 16 个 issue。#1–#8、#10、#12 已关闭；#9、#11、#13–#16 仍开放。按用户要求，已关闭 issue 保持原状。逐项代码位置、现状、修复与对抗性审查见 [issue 审查记录](../../acceptance/issue-audit-20260925.md)。

Issue #11 的 537 帧实测摘要见 [验收报告](../../acceptance/issue11-venc-20260925/fullchain-537-39220-20260925/实测说明.md)；本轮另完成直接 libavformat RTSP 板端构建、端到端握手测试和 1800 帧相机编码验证。原始日志、配置和 trace 保留在本地验收目录，不纳入 GitHub。

历史性能迭代记录仍保留在仓库中：早期 1080P 软件编码/CPU 转换路径低于 30 FPS；ACL NV12 路径及独立 conversion stream 的数据见旧验收记录。2026-09-25 的 VENC 端到端验收是后续结果，PC 客户端因晚加入只收到了 486 帧；板端日志记录发布 537 帧且 `forward_dropped_frames=0`。
