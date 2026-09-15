# MLVC v1：两台 Ascend 310P1 的已验证部署

源码基线来自编码设备原有 `mlvc_acl_cppv1`。本目录是隔离修复副本，不是 MLVCv2 重构版。`_v3` 仅沿用工作区要求的设备测试产物命名。

两端部署目录：`/root/workplace/mlvc_20260903/mlvc_v3/p1-v1-repair-20260911`。

先在解码端启动：

```sh
ssh decode-310p1
cd /root/workplace/mlvc_20260903/mlvc_v3/p1-v1-repair-20260911
bash scripts/run_p1.sh decode 720p
```

再在编码端启动：

```sh
ssh ascend-lab
cd /root/workplace/mlvc_20260903/mlvc_v3/p1-v1-repair-20260911
bash scripts/run_p1.sh encode 720p
```

两端都改为 `1080p` 可测试另一分辨率。配置位于 `configs/p1/`，默认传输 537 帧、QP 2、GOP 96、reset interval 32、LTR period 64，UDP 端口 39191。解码端等待码流结束，默认 `format=none`，仍会实际执行 Decoder OM，仅不生成预览或图片。不要混用分辨率、旧部署二进制或 RTP 版本。

本机一键自动验收（需要 SSH 别名）：

```powershell
python tools/run_p1_pair.py --resolution 720p --frames 537 --ltr-period 64
python tools/run_p1_pair.py --resolution 1080p --frames 537 --ltr-period 64
```

脚本等待解码器实际绑定 UDP 端口再启动编码器，保留配置、stdout/stderr 和 JSON 指标，检查退出码、成功标记、帧数和 payload 总字节数。设备任务有超时；失败路径等待本次任务退出，最多可能等待 180 秒。

从源码重新构建（在编码 P1）：

```sh
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DOpenCV_DIR=/opt/opencv/lib64/cmake/opencv4 \
  -DTOMLPLUSPLUS_INCLUDE_DIR=/root/workplace/mlvc_20260903/third_party/tomlplusplus/include \
  -DMLVC_MSRTC_RANS_DIR=/root/workplace/mlvc_20260903/mlvc-main/mlvc-main/packages/msrtc_rans \
  -DMLVC_TEST_MANIFEST=/root/workplace/mlvc_20260903/mlvc720p/1280x720/manifest_310P1.json \
  -DMLVC_APP_SUFFIX=_v3
cmake --build build -j4
ctest --test-dir build --output-on-failure --timeout 90
```

修复：恢复损坏的 ACL 张量校验代码；接受 ATC 的 `<operator>:<numeric output index>:<logical output name>` 输出别名，同时保留逻辑名称、输入名称精确匹配，以及输入输出的数量、dtype、shape 检查。模型文件不变。

测试覆盖和性能边界见跟踪目录 `docs/testing/2026-09-11-p1-v1-repair-report.md`。本版本保持旧 UDP 协议，不提供 RTCP、重传或中途加入保障；成功实测不能外推为任意网络条件下无丢包。
