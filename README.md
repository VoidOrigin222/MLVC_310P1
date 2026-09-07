# MLVC 310P1

本项目用于在 Ascend 310P1 上运行 MLVC 编码和解码。编码端读取 FP16
YUV444 帧，执行 Encoder OM 和 rANS 熵编码后，通过 UDP 发送 MLVC 码流；
解码端接收码流，执行 rANS 解码和 Decoder OM，并将 JPEG 帧转发到 PC。

仓库只包含源码和示例配置，不包含 CANN、OM、sidecar、FP16 输入帧、构建产物
和运行结果。

## 运行环境

- Ascend 310P1，驱动、固件、CANN 9.1 与 310P OPP 版本需匹配。
- OpenCV 4.10，安装路径为 `/opt/opencv`。
- CMake 3.22 或更新版本，以及支持 C++17 的编译器。
- 以下目录需与本仓库位于同一个工作目录：

```text
workspace/
|- MLVC_310P1/                       # 本仓库
|- third_party/tomlplusplus/include/
|- mlvc-main/mlvc-main/packages/msrtc_rans/
|- mlvc1080p/
|  |- manifest_310P1.json
|  |- sidecars.mlvc.ulbvcsc
|  `- MLVCEncoder_310P1.om, MLVCDecoder_310P1.om
|- mlvc720p/1280x720/
|  |- manifest_310P1.json
|  |- sidecars.mlvc.ulbvcsc
|  `- om/MLVCEncoder_ascend310p1.om, om/MLVCDecoder_ascend310p1.om
`- data/preprocessed_frames/
   |- 614lab_1080p_fp16/
   `- 614lab_720p_fp16/
```

每个 FP16 帧目录需要包含 `frame_0.fp16`、`frame_1.fp16` 等帧文件，以及记录
`width`、`height`、`fps` 的 `source_info.txt`。

## 编译

在 Ascend 设备上执行。请按实际位置调整工程路径：

```bash
cd /root/workplace/mlvc_20260903/mlvc_acl_cpp_udp
source /usr/local/Ascend/cann-9.1.0/set_env.sh

cmake -S . -B build \
  -DMLVC_BUILD_APPS=ON \
  -DBUILD_TESTING=ON \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DOpenCV_DIR=/opt/opencv/lib64/cmake/opencv4
cmake --build build -j"$(nproc)"
```

运行关键回归测试：

```bash
ctest --test-dir build --output-on-failure \
  -R 'udp_initial_wait|reference_reset_schedule|decode_config'
```

## UDP 配置

运行时只使用两份配置文件：

```text
configs/encoder.toml
configs/decoder.toml
```

运行前，按实际网络拓扑修改以下字段：

```toml
# encoder.toml：解码 P1 的地址和监听端口
output_transport_host = "<decoder-p1-ip>"
output_transport_port = 39089

# decoder.toml：PC 接收端的地址和监听端口
input_transport_port = 39089
output_transport_host = "<pc-receiver-ip>"
output_transport_port = 50000
```

先启动 PC JPEG 接收程序，再启动解码端。解码端会无限等待第一条 UDP 消息；收到
配置的 `frame_num` 帧后自动正常退出。

## 运行 1080p

1080p 输入、P1 OM 和 manifest 必须匹配。614lab 序列使用 `frame_num = 537`；
连续接收 UDP 码流时可设为 `-1`。

```toml
# configs/encoder.toml
input_frame_dir = "../data/preprocessed_frames/614lab_1080p_fp16"
frame_num = 537
qp = 2                         # 有效范围：0 到 63
[model]
manifest = "../mlvc1080p/manifest_310P1.json"
```

```toml
# configs/decoder.toml
frame_num = 537
[model]
manifest = "../mlvc1080p/manifest_decoder_310P1.json"
```

先在解码 P1 上启动解码端：

```bash
cd /root/workplace/mlvc_20260903/mlvc_acl_cpp_udp
source /usr/local/Ascend/cann-9.1.0/set_env.sh
./build/mlvc_decode --config configs/decoder.toml
```

再在编码 P1 上启动编码端：

```bash
cd /root/workplace/mlvc_20260903/mlvc_acl_cpp_udp
source /usr/local/Ascend/cann-9.1.0/set_env.sh
./build/mlvc_encode --config configs/encoder.toml
```

## 切换到 720p

启动命令不变，只需在两份 TOML 中将模型和编码端输入替换为对应的 720p P1 模型包：

```toml
# configs/encoder.toml
input_frame_dir = "../data/preprocessed_frames/614lab_720p_fp16"
frame_num = 537
[model]
manifest = "../mlvc720p/1280x720/manifest_310P1.json"
```

```toml
# configs/decoder.toml
frame_num = 537
[model]
manifest = "../mlvc720p/1280x720/manifest_310P1.json"
```

不能用 1080p manifest 处理 720p 输入，也不能反向使用。OM 的输入 shape 和特征
tensor shape 是静态的，必须与所选模型包匹配。
