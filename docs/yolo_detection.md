# YOLO11s on Ascend 310P1

Latest optimized camera results: 24.11 FPS with native 1080P detection,
27.42 FPS with 720P detection (300 frames), and 27.90 FPS for the 900-frame
720P detection run with 30 warmup frames. Source and codec remain 1080P;
the default detector remains 720P. Stable camera throughput of 30 FPS has
not been established. See the [optimization report](status/DETECTION_OPTIMIZATION_20261124.md).

Detailed CPU preprocessing, queue wait and CANN concurrency measurements:
[profiling report](status/DETECTION_PROFILING_20261124.md).

The camera remains 1920×1080. Detection and ulbvc encoding process the same
source frame on separate workers and ACL streams. The detector uses the
camera's visible NV12 sidecar, before codec padding. Its bounded queue applies
backpressure; it does not silently sample or drop detection frames.
Preparation and inference use separate workers and three owned input buffers.
The 720P runtime resizes the NV12 luma/chroma planes before the fused RGB/FP16
conversion; the exported ONNX reference uses RGB preprocessing. The functional
box comparison and numerical tolerances are recorded in the optimization report.

## Model profiles

Both models use the official YOLO11s COCO 80-class weights, static batch 1,
FP16 input/output, ONNX opset 17, and application-side NMS. Both compiled
successfully with CANN 9.1.0 ATC for Ascend310P1; opset 13 was not needed.

| Profile | Source | Detector input NCHW | Preprocessing |
| --- | --- | --- | --- |
| 1080P | 1920×1080 | 1×3×1088×1920 | No resizing; pad 4 rows above/below |
| 720P | 1920×1080 | 1×3×736×1280 | Resize to 1280×720; pad 8 rows above/below |

Padding is RGB 114. Coordinates are restored with `(coordinate - padding) /
gain`, then clipped to the 1920×1080 source. The 720P gain is 2/3. Codec
bottom padding is a separate operation and must not be applied to detections.

Assets are local deployment files under
`models/detection/yolo11s/fp16_opset17/{1080p,720p}/`: ONNX, OM, labels,
metadata, checksums and validation tensors. Model binaries are excluded from
Git. `tools/export_yolo11.py` exports a native half precision PyTorch model;
it checks actual ONNX input/output types, since a CPU export can otherwise
silently retain FP32. Reproducible dependencies are in
`tools/requirements-yolo-export.txt`.

```bash
python tools/export_yolo11.py --help
bash scripts/convert_yolo_om.sh path/to/model.onnx path/to/output_prefix 1088 1920
```

Standalone hardware timing on the centered bus/person fixture, 5 warmups and
30 measured executions, was 16.744 ms for 1080P and 8.877 ms for 720P.
These are model execution times, excluding preprocessing, postprocessing,
encoding, transport and display. Both OM outputs produced the same five
detections as the corresponding ONNX reference (one bus and four people).
Minimum matched box IoU was 0.9912 and 0.9859 respectively. This is a
functional comparison, not a COCO accuracy evaluation.

Initial integration camera tests before optimization used 300 frames, QP 11, 1080P MJPEG camera
input, concurrent original preview, full motion processing, RTP and DVPP
RTSP output. The measured encoding-throughput timer includes waiting for
frame preparation and detection; it excludes initial model/camera setup.

| Configuration | Measured encoding FPS | Frames encoded / detected / decoded |
| --- | --- | --- |
| Without detector (baseline) | 28.75 | 300 / 0 / 300 |
| 1080P detector + 1080P ulbvc | 17.95 | 300 / 300 / 300 |
| 720P detector + 1080P ulbvc | 19.84 | 300 / 300 / 300 |

These tests did not reach 30 FPS. Camera V4L2 sequence gaps were observed
under backpressure, so matching encoded/detected/decoded counts do not mean
every camera exposure was processed. All transmitted detections matched their
encoded frame IDs; decoder display forwarding dropped zero frames. The
default encoder config enables the tested 720P detector, with the source and
codec still at 1080P. The native 1080P validation profile remains available.

A separate repeated bus/person fixture passed 120 frames and 600 boxes across
both boards, with visible boxes, labels and confidence in the decoded RTSP
screenshot. This controlled fixture achieved 27.57 encoding FPS; it is not a
replacement for the live-camera result. Evidence is local under
`artifacts/dual-board/{camera-1080p-final,camera-720p-final,detection-fixture-final}/`.
All 34 board CTest cases passed; the metadata/geometry and expanded detector
configuration tests passed again after the final detector changes.

## Frame association and display

An optional versioned detection TLV (type `0x0010`) travels inside the same
ULBVC RTP media unit as the encoded frame. It carries frame ID, source width
and height, and up to 100 boxes. Each box carries class ID, track ID,
confidence, source-space `cx,cy,w,h`, and label. YOLO11s is a detector;
`track_id=-1` means no object tracker is attached. The class ID identifies the
COCO category, not a persistent object identity.

The sender waits for both encoding and detection before releasing that frame.
The receiver checks the frame ID and dimensions, then overlays rectangles,
labels and confidence onto the DVPP RTSP display buffer. The codec reference
tensor is not changed by the display overlay. File bitstreams preserve the
same optional TLV. Legacy UDP output is rejected when detection is enabled.

Configuration keys are `detection_enabled`, `detection_model`,
`detection_labels`, `detection_report`, `detection_confidence`,
`detection_nms_iou` and `detection_queue_capacity`. The report is JSONL with
one record per source frame. Validation profiles are in
`configs/validation/encoder_detect_1080p.toml` and
`configs/validation/encoder_detect_720p.toml`.

## Local source first deployment

Camera JPEG padding normalization is enabled by default with
`camera_jpeg_normalize_padding = true`. It preserves JPEG coefficients and
falls back to the original input on unsupported or malformed formats.
See [online deployment measurements](status/JPEGD_DEPLOYMENT_20261124.md)
for encoder-only throughput, pixel/detection comparisons, and test limits.

Edit local sources/configuration, package them with
`tools/package_device.py`, transfer the archive, extract it under
`/root/ULBVC_310P1`, and run `tools/verify_device_source.py` there.
`scripts/build_device.sh` builds against the isolated SDK/runtime libraries.
Both custom operator packages must also use the same ulbvc naming.

```bash
bash scripts/run_encode.sh configs/encoder.toml
bash scripts/run_decode.sh configs/decoder.toml
ctest --test-dir build --output-on-failure
```

`tools/run_dual_board_validation.py` runs finite tests, saves logs and a
decoded screenshot under local `artifacts/dual-board/`, and monitors NPU
temperature/health. It stops its own processes if temperature reaches 90°C.
The RTSP display uses the Windows MediaMTX server at 192.168.5.2; RTP travels
from the encoder at 192.168.5.13 to the decoder at 192.168.5.11:39340.
