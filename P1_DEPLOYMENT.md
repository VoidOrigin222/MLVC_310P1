# P1 deployment

Use one unified source directory on either device. The same build produces both `mlvc_encode` and `mlvc_decode`; select the executable and configuration appropriate for the device role.

```bash
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j4
```

If the source directory is placed at `/root/workplace/mlvc_20260903/mlvc_acl_cpp`, the default external dependency paths resolve automatically. Otherwise set the dependency root explicitly:

```bash
cmake -S . -B build -DMLVC_EXTERNAL_ROOT=/root/workplace/mlvc_20260903
```

## Encoder

```bash
./build/mlvc_encode --config configs/encoder.toml
```

## Decoder

```bash
./build/mlvc_decode --config configs/decoder.toml
```

The configuration files contain deployment-specific model, input, output, address, and port values. Update those values for a different P1 pair.
