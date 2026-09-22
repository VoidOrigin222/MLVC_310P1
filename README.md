# MLVC 310P1

MLVC 是面向 Ascend 310P1 的编解码部署项目。编码器和解码器共用一套源码，通过同一个 CMake 工程构建为两个可执行文件。

## 目录结构

```text
common/       公共运行时、传输层和流水线框架
mlvc/         MLVC 编解码、熵编码、I/O 和 ACL 运行时
application/  应用流水线和命令行配置
apps/         encode_main.cc 和 decode_main.cc
configs/      编码器和解码器 TOML 配置
custom_ops/   Ascend 自定义算子
tools/        测试、性能基准和辅助工具
```

## 构建

```bash
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j4
```

项目已将官方 rANS 实现和 toml++ 头文件集成在仓库的 `third_party/`
目录中，CMake 会自动使用仓库内版本，不再依赖源码目录之外的
`mlvc-main/` 或外部 `third_party/` 目录。OpenCV、Ascend CANN 和 DVPP
属于设备系统依赖，仍需在 Ascend 设备环境中安装。

## 运行

```bash
./build/mlvc_encode --config configs/encoder.toml
./build/mlvc_decode --config configs/decoder.toml
```

仓库中的标准配置使用 RTP 输入/输出和无输出解码模式。切换传输方式或设备时，请修改 `configs/encoder.toml` 和 `configs/decoder.toml` 中的传输、模型和输入参数。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

项目使用 `.clang-format` 中的 Google C++ 代码格式规范。

## 当前验证状态

Issue #1 至 Issue #10 已在当前源码中完成。Issue #11 尚未完成，原因是 1080P RTP 到 RTSP 的输出全链路尚未稳定达到 30 FPS。

Ascend 310P1 上已完成 537 帧纯 RTP 编解码实测：

- 1080P：编码 33.54 FPS，解码 31.89 FPS；
- 720P：编码 84.89 FPS，解码 80.83 FPS。

## 模型资产

仓库不包含设备构建目录、生成文件、本地日志和模型二进制文件。模型资产应通过 manifest 和外部部署目录提供，具体部署方式见 [P1_DEPLOYMENT.md](P1_DEPLOYMENT.md)。
