# P1 deployment

Use one unified source directory on either device. The same build produces both `mlvc_encode` and `mlvc_decode`; select the executable and configuration appropriate for the device role.

```bash
bash scripts/bash_build.sh          # decoder
bash scripts/bash_build_encode.sh  # encoder
```

The rANS implementation and toml++ headers are included under `third_party/`
and are resolved from the repository automatically. OpenCV, Ascend CANN and
DVPP remain device-system dependencies and must be installed on the target
Ascend environment.

## Encoder

```bash
bash scripts/bash_run_encode.sh [configs/encoder.toml]
```

## Decoder

```bash
bash scripts/bash_run.sh [configs/decoder.toml]
```

The configuration files are examples, not a ready-to-run device pair. Before running, update the input frame directory, model manifest, device addresses, ports, and output mode for the target setup. To load an isolated vectorized operator package in the decoder process, set `MLVC_VECTOR_OPAPI_LIB` explicitly. Set `MLVC_FFMPEG_BIN` when FFmpeg is not on the system `PATH`.
