# MLVC Issue 台账

更新时间：2026-09-15

| 编号 | 标题 | 状态 |
|---|---|---|
| #1 | sidecar 一致性校验 | completed |
| #2 | 多 worker 帧顺序 | completed |
| #3 | manifest 与 OM 兼容性检查 | completed |
| #4 | 码流 header、帧号和 payload 校验 | completed |
| #5 | LTR 引用状态一致性 | completed |
| #6 | UDP 异步发送错误传播 | completed |
| #7 | 配置生效和值范围校验 | completed |
| #8 | 命名与接口一致性 | completed |
| #9 | MLVC over RTP v1 | completed |
| #10 | UDP packet pacing | completed |
| #11 | RTSP 输出（MediaMTX） | in_progress |

## #11 当前状态

RTSP 发布器、配置接入、单元测试和 MediaMTX 推流链路已经实现。发布器当前采用单后台线程；RTSP 控制连接使用 TCP 8554，媒体数据使用 UDP RTP/RTCP 端口 8000/8001。UDP 模式 720P/120 帧为 60.15 FPS。针对 1080P 低于 30 FPS 的问题依次验证了设备驻留镜像、CPU NV12 转换和 ACL/NPU NV12 转换三种方案，解码分别达到 26.60、26.47 和 26.95 FPS；方案 3 未丢帧但仍未达到验收线。因此 Issue #11 保持 in_progress。

1080P 优化实测明细见 [issue-11-rtsp.md](../issues/issue-11-rtsp.md)。

2026-09-14 已完成方法 1 + 方法 3 的双缓冲异步 D2H：120 帧解码 26.91 FPS、0 丢帧；537 帧解码 28.42 FPS、0 丢帧，均正常退出。Issue #11 仍未达到 30 FPS 验收线。

同日生成 537 帧 Chrome trace：rANS 熵解码平均 29.70 ms/帧（累计 34.4%）、解码 OM 28.84 ms/帧（33.4%）、CPU NV12 转换 15.92 ms/帧（18.4%）、libx264 写入 11.84 ms/帧（13.7%）。两台 P1 的 ACL NV12 自定义算子当前不可用，NV12 阶段走 CPU 回退。

## ACL NV12 算子修复（2026-09-14）

Issue #11 的 ACL/NV12 子问题已完成。根因是两台设备未把实际 `mlvc_prior_ops/vendors/mlvc` 加入 `ASCEND_CUSTOM_OPP_PATH`，且 `ascend-lab` 使用了不含 `MlvcFp16Yuv444ToNv12` 的旧算子包。运行时现在会从候选库路径自动发现 vendor 根目录并补齐环境变量；两台设备也已同步包含 NV12 kernel、元数据和 ACLN API 的算子包。

清除全部算子相关环境变量后，两台 P1 的 `test_fp16_yuv444_to_nv12_acl` 均通过，1080P ACL 转换事件耗时约 12.26 ms/帧。Issue #11 整体仍为 `in_progress`，因为 RTSP 1080P 端到端 FPS 还需在 ACL 路径启用后重新完成 120/537 帧验收。


ACL 路径双机复测（2026-09-14）：1080P RTP→RTSP-UDP，120 帧编码/解码 32.01/22.83 FPS，537 帧 32.49/25.22 FPS，均完整解码且 0 丢帧。trace 确认 `acl_video.fp16_yuv444_to_nv12.device` 537 次、`copy.rtsp.nv12_d2h.device` 537 次，CPU NV12 回退事件为 0。ACL 算子子问题已关闭；Issue #11 整体仍因 1080P 低于 30 FPS 保持 `in_progress`。

独立 conversion stream 复测：ACL NV12 转换从 decode stream 移出，并以 timeline event 建立依赖；537 帧解码 24.98 FPS、0 丢帧。相较 decode stream 版本 25.22 FPS 无显著提升，当前主要瓶颈为 ACL kernel 与 Decoder OM 的 NPU 资源争用。

## DVPP H.264 VENC 评估（2026-09-15）

两台 P1 已完成 ACL VENC 和低层 MPI 通道探测。ACL `aclvencCreateChannel` 返回 `507018`（AICPU exception），独立 `aclrtProcessReport` 线程不能改变结果；低层 MPI 探测返回 `HI_ERR_VENC_NO_MEM`。设备 `libdvpp_op_base.so` 的 `hi_mpi_venc_create_chn` 实际由 `dvpp::JpegeManager` 导出，当前 runtime 没有可用的 H.264 VENC manager。详细设计和证据见 [ISSUE_11_DVPP_VENC_DESIGN.md](ISSUE_11_DVPP_VENC_DESIGN.md)。Issue #11 继续保持 `in_progress`，现阶段保留 `libx264` 回退。
