# MLVC 310P1

MLVC deployment sources for Ascend 310P1. Encoder and decoder share one unified source tree and are built as two executables from the same CMake project.

## Layout

```text
common/       Shared runtime, transport, and framework code
mlvc/         MLVC codec, entropy, I/O, and ACL runtime
application/  Pipeline and command-line configuration code
apps/         encode_main.cc and decode_main.cc
configs/      Encoder and decoder TOML configurations
custom_ops/   Ascend custom operators
tools/        Tests, benchmarks, and utilities
```

## Build

```bash
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j4
```

The official rANS package and toml++ are external dependencies. By default CMake looks for `third_party/` and `mlvc-main/` beside this source directory. Override the location when needed:

```bash
cmake -S . -B build \
  -DMLVC_EXTERNAL_ROOT=/root/workplace/mlvc_20260903
```

## Run

```bash
./build/mlvc_encode --config configs/encoder.toml
./build/mlvc_decode --config configs/decoder.toml
```

The canonical configurations use RTP input/output and the no-output decode path. Change the transport fields in these two files when another mode is required.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

The project uses Google C++ style through `.clang-format`.
