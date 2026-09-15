# MLVC 310P1

MLVC deployment sources for Ascend 310P1, split into two independent directories.

```text
encoder/   Encoder application and its sources
decoder/   Decoder application and its sources
```

Each directory is self-contained at the source level and can be copied to its target P1 device. Shared codec, transport, runtime, tests, and custom operators are kept in both directories so the two deployments can be built independently. The official rANS package and toml++ remain external dependencies; pass their parent directory with `-DMLVC_EXTERNAL_ROOT`.

## Build the encoder

```bash
cd encoder
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j4
./build/mlvc_encode --config configs/p1/encoder_1080p.toml
```

## Build the decoder

```bash
cd decoder
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j4
./build/mlvc_decode --config configs/p1/decoder_1080p.toml
```

When the source directory is placed beside `third_party/` and `mlvc-main/`, the default CMake paths work automatically. Otherwise configure with, for example:

```bash
cmake -S . -B build -DMLVC_EXTERNAL_ROOT=/root/workplace/mlvc_20260903
```

RTP and RTSP examples are included in the corresponding endpoint configuration directories.

The project uses the repository `.clang-format` (Google C++ style, C++17).
