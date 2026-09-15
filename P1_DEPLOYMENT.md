# P1 deployment

Deploy the two endpoints separately. Copy `encoder/` to the encoding device and `decoder/` to the decoding device; do not build both applications from one shared endpoint directory.

## Encoder device

```bash
cd encoder
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DOpenCV_DIR=/opt/opencv/lib64/cmake/opencv4 \
  -DMLVC_EXTERNAL_ROOT=/root/workplace/mlvc_20260903
cmake --build build -j4
./build/mlvc_encode --config configs/p1/encoder_1080p.toml
```

## Decoder device

```bash
cd decoder
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DOpenCV_DIR=/opt/opencv/lib64/cmake/opencv4 \
  -DMLVC_EXTERNAL_ROOT=/root/workplace/mlvc_20260903
cmake --build build -j4
./build/mlvc_decode --config configs/p1/decoder_1080p.toml
```

The endpoint configuration files contain device addresses, ports, model manifests, and frame paths used during P1 validation. Adjust those deployment-specific values before running on another pair of devices.
