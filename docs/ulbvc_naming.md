# ulbvc 命名与部署

310P1 源码、配置、脚本、注释、UI 和文档统一使用 ulbvc。
C++ 命名空间和头文件前缀为 `ulbvc`，类型使用 `Ulbvc`，宏和环境变量使用 `ULBVC`。
第三方许可证和实现保留原文。

## 构建与运行

从新的源码目录执行完整构建。旧 CMake 缓存和二进制不能直接复用：

```bash
ULBVC_BUILD_DIR="$PWD/build-ulbvc" bash scripts/bash_build_encode.sh
ULBVC_BUILD_DIR="$PWD/build-ulbvc" bash scripts/bash_build.sh
```

产物为 `build-ulbvc/ulbvc_encode` 和 `build-ulbvc/ulbvc_decode`。
构建及运行时的环境变量均使用 `ULBVC_` 前缀，例如 `ULBVC_CANN_HOME`、
`ULBVC_BUILD_DIR`、`ULBVC_ACL_REPO_ROOT` 和 `ULBVC_UI_CONFIG`。

自定义算子的注册名称、导出符号和 vendor 也已更新，必须重新构建、安装仓库内
两套算子包：

```bash
bash scripts/build_custom_opp.sh
```

新目录分别为 `output/custom_opp/wsiluchunkadd/vendors/ulbvc` 和
`output/custom_opp/ulbvc_prior_ops/vendors/ulbvc`。
`scripts/acl_env.sh` 加载这些隔离安装目录，不要求覆盖系统算子。
新增符号包括 `aclnnUlbvcSinglePart`、`aclnnUlbvcBuildIndexEncode` 和
`aclnnUlbvcFp16Yuv444ToNv12`；旧算子库不提供这些符号。

## 外部模型资产

模型二进制不在源码仓库中。部署时应准备一份独立的 ulbvc 模型目录，并核对：

- 配置中的 `manifest` 指向实际存在的目录，例如 `ulbvc1080p`。
- manifest 中两个模型记录的 `name` 分别为 `ULBVCEncoder` 和 `ULBVCDecoder`。
- manifest、metadata 与模型文件路径使用一致的新名称，相关数组键也应与读取代码一致。
- 文件改名只更新对应路径；文件内容发生变化时，重新生成该文件的 `bytes`、
  `sha256`，保留完整性验证。不要直接修改 OM 的二进制字节。

模型包身份摘要使用 `ULBVC-MODEL-BUNDLE-V2` 域标识；编解码端必须使用匹配的
manifest、metadata、PMF 和 sidecar。网络会话中的摘要随配套模型资产重新计算。

## 接口与码流

- RTP 统计消息的 `kind` 为 `ulbvc_rtp_stats`；板端配置使用 `ulbvc_stats_host`、
  `ulbvc_stats_port`、`ulbvc_stats_interval_frames`。
- UI 内部接口使用 `ulbvc` 视频键；磁盘参数文件继续使用 `semantic` 字段。
- 默认语义视频 RTSP 路径为 `/ulbvc`，现场参数中的地址应与发布端一致。
- 文件扩展名为 `.ulbvc`，文件签名为 `ULBVCBS1` 或 `ULBVCES1`。
- RTCP APP 名称受四字节长度限制，使用 `ULBV` 标识 ulbvc 控制消息。

这是整套接口的命名切换。编码端、解码端、算子包和 UI 必须配套更新；历史文件与
旧程序不属于新命名版本的兼容范围。现有板端部署及本地历史交付包不会由源码修改
自动更新。
