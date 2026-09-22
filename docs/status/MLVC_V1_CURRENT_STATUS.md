# MLVC v1 鍙屾満閮ㄧ讲鐘舵€侊紙2026-09-12锛?

## 椤圭洰鐩爣

鍦ㄤ袱鍙?Ascend 310P1 璁惧涓婅繍琛?MLVC v1 缂栫爜鍒拌В鐮佺殑绔埌绔摼璺€?

## Issue 鐘舵€?

- **Issue #3锛氬凡瀹屾垚**銆侫CL 妯″瀷 manifest 涓?OM 鐨勬暟閲忋€佸悕绉般€乨type 鍜屽畬鏁?shape 鏍￠獙宸茶ˉ榻愩€?
- **Issue #10锛氬凡瀹屾垚**銆俇DP token bucket pacing 宸叉帴鍏ュ彂閫侀摼璺紝骞跺畬鎴?720p/1080p 鍙屾満楠岃瘉銆?
- **Issue #9锛氬凡瀹屾垚**銆俁TP receiver銆乻ender銆佸垎鐗囬噸缁勩€佸簭鍒楀彿涓撻」娴嬭瘯銆佺紪鐮佸櫒/瑙ｇ爜鍣ㄩ厤缃帴鍏ヤ互鍙?120/537 甯у弻鏈洪獙鏀跺潎宸插畬鎴愩€?

## Upload Baseline (2026-09-22)

The formal source baseline includes completed Issues #1 through #10. Issue #9
also includes bounded RTP reassembly contexts, fragment timeout cleanup,
random non-zero default SSRC, SCU session-config resend, and an MLVC RTCP
configuration-request codec. The formal-device `rtp_reassembler` and
`rtp_sender` tests pass on both Ascend 310P1 devices.

The pure RTP codec path was measured with 537 frames: 1080p encode/decode at
33.54/31.89 FPS and 720p encode/decode at 84.89/80.83 FPS. The RTP-to-RTSP
1080p output path remains below 30 FPS and is tracked as Issue #11.

## 宸查獙璇佺粨鏋?

- 妯″瀷鍔犺浇娴嬭瘯锛氶€氳繃銆?
- 720p 鍙屾満锛?37/537 甯э紝缂栫爜杩涚▼鍜岃В鐮佽繘绋嬪潎姝ｅ父閫€鍑恒€?
- 1080p 鍙屾満锛?37/537 甯э紝缂栫爜杩涚▼鍜岃В鐮佽繘绋嬪潎姝ｅ父閫€鍑恒€?
- 1080p PNG 杈撳嚭锛?20 甯у叏閮ㄥ彲璇伙紝鍒嗚鲸鐜?1920脳1080銆?
- P1 RTP 鍗曞厓娴嬭瘯锛歚rtp_mlvc`銆乣rtp_reassembler`銆乣rtp_message_receiver_sequence`銆乣rtp_sender` 鍏ㄩ儴閫氳繃銆?

## 閮ㄧ讲鍩虹嚎

褰撳墠寮€鍙戝熀绾夸负 v1銆傝澶囦笂鐨勫師濮?v2銆乿3 鐩綍宸插垹闄わ紱淇楠岃瘉浣跨敤鐙珛鐩綍锛岄伩鍏嶈鐩栨棦鏈?v1 閮ㄧ讲銆?

## 鍚庣画宸ヤ綔

1. 缁х画澶勭悊 Issue #1銆?2銆?4銆?5銆?6銆?7銆?8銆?11銆?


## Issue #4 and #7 completion (2026-09-12)

Issue #4 is complete. All three input paths use the same MLVC header/frame validation, including payload bounds and frame sequence checks; UDP and RTP senders validate before transmission.

Issue #7 is complete. Configuration parsing now preserves numeric TOML values such as `fps = 25.0`, rejects unsupported profiles and invalid ranges, applies encoder warmup profiling, and uses decoder `frame_buffer_slots` to bound the bitstream queue.

P1 evidence: `cmake --build build -j2` passed with application targets enabled. The targeted CTest set (`rtp_mlvc`, `rtp_reassembler`, `rtp_message_receiver_sequence`, `rtp_sender`, `mlvc_bitstream_validation`, `decode_config`) passed.

## Issue #1/#2/#5/#6/#8 completion (2026-09-12)

The remaining implementation issues requested for this pass are complete:

- **#1 sidecar safety:** exact shape/dtype byte validation, file-size bounds, duplicate-name rejection, and trailing-byte rejection are implemented with malformed-input tests.
- **#2 frame ordering:** `StreamingPipeline` now assigns sequence numbers and publishes results in input order even with multiple stream workers. Encoder and decoder preserve the configured worker count.
- **#5 LTR consistency:** forced LTR recovery/reference values are serialized in MLVC bitstream/header version 4. Decoder uses the stream values and rejects conflicting overrides; I-frame cache reset is symmetric.
- **#6 UDP error propagation:** asynchronous forwarders have idempotent `Close()` methods that join, flush, and rethrow the final send error before success is reported.
- **#8 naming consistency:** decoder transport options use `output_transport_*` names throughout the public options and runtime implementation.

P1 Release validation: `ascend-lab` full build and CTest passed (17/17); `decode-310p1` Release build and CTest excluding the 70-second initial-wait test passed (16/16). Issue #11 remains open.

## Post-change RTP verification (2026-09-12)

After the Issue #1/#2/#5/#6/#8 changes and MLVC header v4 update, dual-device RTP smoke runs completed successfully:

| Resolution | Frames | Encode FPS | Decode FPS | Bytes match |
|---|---:|---:|---:|---|
| 720p | 120 | 79.63 | 76.30 | yes |
| 1080p | 120 | 31.70 | 29.64 | yes |

Both runs exited with `encode=ok`, `decode=ok`, and exactly 120 decoded frames.
- 720p, 537 frames: encode 81.30 FPS, decode 80.50 FPS, 225,495 bytes on both ends, success.


## 1080p decode tuning (2026-09-12)

The 1080p P1 decoder profile now uses `pipeline.entropy_workers = 4` in `configs/decoder.toml`. In a controlled 120-frame RTP run this raised decode throughput from 29.64 FPS (2 workers) to 30.11 FPS (4 workers), while preserving all 120 frames and byte-for-byte payload agreement. Eight workers reduced throughput to 29.27 FPS, so 4 is the selected setting.

## 1080p decode data path (2026-09-12)

For `format = "none"` without an output transport, the decoder now uses ACL-mirror
binding and keeps `x_hat`/reference features device-resident. CPU materialization
is retained for entropy consumers and for file and raw output paths. The JPEG and
DVPP-JPEG forwarding branches have been removed. The entropy implementation also avoids the previous full y-scale
temporary plane and branch-heavy symbol narrowing loop.  The changes were
validated on P1 with 120 and 537 frame RTP runs; payloads and frame counts match.

Measured decoder throughput is 30.34 FPS (120 frames) and 31.81 FPS (537 frames)
at 1080P.  The remaining limit is the scalar rANS decode plus the 1080P decoder
OM, so larger gains require an optimized entropy kernel or a model/bitstream
change that enables independent partitions or batching.

## Issue #11 optimization comparison (2026-09-13)

Three 1080P/120-frame dual-P1 experiments were run with RTP input and RTSP-UDP output. Device-resident x_hat mirror reached 26.60 FPS (76 publisher drops); CPU NV12 conversion reached 26.47 FPS (76 drops); ACL/NPU NV12 conversion followed by a smaller D2H reached 26.95 FPS (0 drops). All runs decoded 120 frames and exited successfully. None exceeded the 30 FPS target; the ACL/NPU path is the current best and still synchronizes after each D2H, leaving double-buffered overlap as the next optimization candidate.

The combined device-resident plus ACL/NPU path now uses two-buffer asynchronous D2H with a dedicated copy stream. Dual-P1 RTSP-UDP validation reached 26.91 FPS for 120 frames and 28.42 FPS for 537 frames; both runs decoded every frame with zero publisher drops and exited successfully. The 30 FPS target remains open.

A 537-frame Chrome trace measured average busy times of 29.70 ms for rANS entropy decode, 28.84 ms for the MLVCDecoder OM, 15.92 ms for CPU NV12 conversion, and 11.84 ms for libx264 writes. The ACL NV12 custom operator was unavailable on both P1 devices during this run, so the NV12 stage used its CPU fallback. The measured decode rate was 28.70 FPS with zero drops.

## ACL NV12 operator availability fix (2026-09-14)

The ACL NV12 fallback was caused by deployment configuration and an out-of-date custom operator package. `ASCEND_CUSTOM_OPP_PATH` did not include the `mlvc_prior_ops/vendors/mlvc` vendor root, and the `ascend-lab` package lacked the `MlvcFp16Yuv444ToNv12` kernel registration in `binary_info_config.json` and the corresponding ACLN symbols. The runtime loader now discovers the vendor root from candidate libraries, prepends it to `ASCEND_CUSTOM_OPP_PATH` when needed, and supports the sibling `mlvc_acl_cppv1` repository used on P1.

The complete NV12 package was synchronized to both P1 devices. With `ASCEND_CUSTOM_OPP_PATH`, `MLVC_VIDEO_OPAPI_LIB`, and `MLVC_PRIOR_OPAPI_LIB` removed from the environment, `test_fp16_yuv444_to_nv12_acl` passed on both devices. The 1080P ACL conversion measured about 12.26 ms per frame. This confirms that 310P1 supports the operator; the previous CPU fallback was a path/package deployment issue. RTSP 120/537-frame FPS measurements must be rerun with the ACL path enabled.

## ACL RTSP 双机复测（2026-09-14）

修复后的 ACL NV12 路径已接入实际 RTSP 输出并完成 1080P 双机验证：120 帧编码/解码 32.01/22.83 FPS，537 帧 32.49/25.22 FPS，均完整解码、0 丢帧、正常退出。537 帧 trace 出现 537 次 `acl_video.fp16_yuv444_to_nv12.device` 和 `copy.rtsp.nv12_d2h.device`，`rtsp.cpu_nv12_convert` 为 0；ACL 转换平均约 12.82 ms/帧，NV12 D2H 平均约 0.29 ms/帧。ACL 算子可用性问题已解决，Issue #11 仍需继续优化 1080P 端到端 FPS 以达到 30 FPS 目标。
