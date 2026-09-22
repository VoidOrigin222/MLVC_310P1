# P1 deployment

Use one unified source directory on either device. The same build produces both `mlvc_encode` and `mlvc_decode`; select the executable and configuration appropriate for the device role.

```bash
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
./scripts/configure.sh
cmake --build build -j4
```

The rANS implementation and toml++ headers are included under `third_party/`
and are resolved from the repository automatically. OpenCV, Ascend CANN and
DVPP remain device-system dependencies and must be installed on the target
Ascend environment.

## Encoder

```bash
./build/mlvc_encode --config configs/encoder.toml
```

## Decoder

```bash
./build/mlvc_decode --config configs/decoder.toml
```

The configuration files contain deployment-specific model, input, output, address, and port values. Update those values for a different P1 pair.
