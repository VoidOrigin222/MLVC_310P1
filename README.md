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
bash scripts/bash_build.sh          # 构建解码端
bash scripts/bash_build_encode.sh  # 构建编码端
```

脚本默认使用 `/usr/local/Ascend/cann-9.1.0` 和 `/opt/opencv/lib64/cmake/opencv4`；可分别通过 `MLVC_CANN_HOME`、`OpenCV_DIR`、`MLVC_BUILD_DIR` 和 `BUILD_JOBS` 覆盖。构建需要目标机已安装 CANN、OpenCV 和 DVPP。

项目已将官方 rANS 实现和 toml++ 头文件集成在仓库的 `third_party/`
目录中，CMake 会自动使用仓库内版本，不再依赖源码目录之外的
`mlvc-main/` 或外部 `third_party/` 目录。OpenCV、Ascend CANN 和 DVPP
属于设备系统依赖，仍需在 Ascend 设备环境中安装。

## 运行

```bash
bash scripts/bash_run_encode.sh [configs/encoder.toml]
bash scripts/bash_run.sh [configs/decoder.toml]
```

运行脚本会加载 CANN 环境。仓库中的配置是示例；运行前请修改输入帧目录、模型 manifest、设备地址、端口和输出方式。若要使用隔离安装的向量化算子包，可显式设置 `MLVC_VECTOR_OPAPI_LIB=/path/to/libcust_opapi.so`；需要非系统默认 FFmpeg 时，设置 `MLVC_FFMPEG_BIN=/path/to/ffmpeg/bin`。脚本不会自动继承这些变量或依赖板端特定目录。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

项目使用 `.clang-format` 中的 Google C++ 代码格式规范。

## 项目状态

各 issue 的 GitHub 状态、代码审查结论和设备验证记录见 [`docs/status/ISSUE_TRACKER.md`](docs/status/ISSUE_TRACKER.md)。设备实测数据见 [`docs/status/`](docs/status/) 下的记录。Issue 在 GitHub 上的开闭状态以 GitHub 页面为准。

## 模型资产

仓库不包含设备构建目录、生成文件、本地日志和模型二进制文件。模型资产应通过 manifest 和外部部署目录提供，具体部署方式见 [P1_DEPLOYMENT.md](P1_DEPLOYMENT.md)。
