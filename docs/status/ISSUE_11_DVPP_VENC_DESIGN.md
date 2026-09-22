# Issue #11：DVPP VENC 设计与可用性验证

更新时间：2026-09-15

## 目标

将 RTSP 输出链路中的软件 `libx264` 替换为设备侧 H.264 编码：

```text
Decoder OM → ACL/NV12 device buffer → DVPP VENC H.264
           → H.264 码流 D2H → RTSP publisher（`-c:v copy`）
```

该设计可以消除 NV12 到 CPU 后再执行 `libx264` 的开销，并保留现有 MediaMTX UDP 传输配置。

## 已完成的验证

1. 在统一工程中增加了 `DvppH264Encoder` 实验封装和单帧 benchmark。
2. 两台 310P1 均可编译该封装。
3. 两台设备上的 ACL VENC 调用均在 `aclvencCreateChannel` 返回 `507018`。错误详情为：

   ```text
   ACL_ERROR_RT_AICPU_EXCEPTION
   An error occurred in the kernel task, retCode=0x2a
   [Sync][Stream] ... sendFrameStream ... errorCode = 507018
   ```

4. 补充独立 `aclrtProcessReport` 线程并使用 Linux TID 后，结果不变。
5. 直接调用低层 MPI 接口进行通道探测时，两台设备均返回 `0xa008800c`（`HI_ERR_VENC_NO_MEM`）。
6. 当前 `libdvpp_op_base.so` 的导出实现为：

   ```text
   dvpp::JpegeManager::hi_mpi_venc_create_chn(...)
   ```

   未发现 H.264 专用 VENC manager；该 runtime 暴露的是 JPEG 相关 VENC 兼容接口。已有 DVPP JPEG 编码验证成功，不能据此推断 H.264 VENC 可用。

## 最小验证程序

新增程序：

```text
tools/cpp/benchmarks/benchmark_dvpp_venc_minimal.cc
```

它只执行 ACL 初始化、创建 report 线程、设置 H.264/NV12/分辨率参数和 `aclvencCreateChannel`，不分配输入帧、不调用 RTSP，也不接触主编码器。支持三档参数级别：

```bash
./build/benchmark_dvpp_venc_minimal --level 0 --width 1920 --height 1088
./build/benchmark_dvpp_venc_minimal --level 1 --width 1920 --height 1088
./build/benchmark_dvpp_venc_minimal --level 2 --width 1920 --height 1088
```

2026-09-16 双机结果：两台 P1 的 level 0/1/2 均在通道创建阶段返回 `507018` 和 AICPU `retCode=0x2a`。另外，level 0 在 1280×720 和 1920×1080 下结果相同，排除了分辨率和码控参数导致失败的可能性。

## 当前结论

在现有 310P1 镜像（CANN 9.1.0 host library、设备 runtime 25.5.t7）上，ACL VENC H.264 通道无法创建。继续把该接口接入主 RTSP 路径会触发 AICPU 异常，因此暂不替换稳定的 `libx264` 路径，也不上传未验证的 VENC 接入代码。

## 后续实现条件

只有满足以下条件之一，才继续实现硬件 H.264 输出：

- 更换为提供 H.264 VENC manager 的匹配 CANN/NNRT runtime，并重新完成单帧 SPS/PPS/IDR 验证；或
- 使用设备厂商提供的完整 Media/MPI VENC 运行时，完成通道创建、启动、送帧、取流、释放和码流线程封装。

硬件路径准备好后，RTSP publisher 应增加 H.264 码流入口，并使用 FFmpeg `-f h264 -c:v copy` 直接复用码流。当前 publisher 和 `libx264` 回退路径保持不变。
