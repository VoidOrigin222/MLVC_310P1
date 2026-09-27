# MLVC Issue 台账

更新时间：2026-09-26（与 GitHub 状态分开记录；未 push、未关闭 issue）

| 编号 | GitHub 状态 | 本地审查结论 |
|---|---|---|
| #1 | closed | 已关闭 |
| #2 | closed | 已关闭 |
| #3 | closed | 已关闭 |
| #4 | open | Grifcc 指出的 stream/model 输出 shape 不匹配已在解码入口拒绝；310P1 测试通过 |
| #5 | closed | 已关闭 |
| #6 | closed | 已关闭 |
| #7 | open | 配置校验通过；Grifcc 提到的 RAW YUV UDP 功能已移除，rate-control 整数预算有上限 |
| #8 | closed | 已关闭 |
| #9 | open | Grifcc 评论中的 timeout、完整单元去重/跨帧重排、FPS 时间戳与 EOS 边界问题已修复并做板端回归 |
| #10 | open | pacer 重复计账及队列/wire 字节计量已修正；板端 pacing 回归通过 |
| #11 | open | 1080p RTP→解码→DVPP VENC→RTSP 全链路有 537 帧实测记录；本轮未重跑视频链路 |
| #12 | open | 在途额度包含重排结果；310P1 全量 CTest 覆盖通过 |

## 审查说明

GitHub 当前共有 12 个 issue：#1、#2、#3、#5、#6、#8 已关闭；#4、#7、#9、#10、#11、#12 仍开放。逐项代码位置、现状、修复与对抗性审查见 [2026-09-25 issue 审查记录](../../acceptance/issue-audit-20260925.md)。

Issue #11 的 537 帧实测摘要见 [验收报告](../../acceptance/issue11-venc-20260925/fullchain-537-39220-20260925/实测说明.md)；其结果为编码 33.5355 FPS、解码 29.7511 FPS、VENC 发布 537 帧且转发丢帧 0。本轮未重新跑 RTSP 视频链路。原始日志、配置和 trace 保留在本地验收目录，不纳入 GitHub。

历史性能迭代记录仍保留在仓库中：早期 1080P 软件编码/CPU 转换路径低于 30 FPS；ACL NV12 路径及独立 conversion stream 的数据见旧验收记录。2026-09-25 的 VENC 端到端验收是后续结果，PC 客户端因晚加入只收到了 486 帧；板端日志记录发布 537 帧且 `forward_dropped_frames=0`。
