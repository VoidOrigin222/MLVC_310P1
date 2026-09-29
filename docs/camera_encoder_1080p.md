# Encoder board: native 1080p V4L2 camera into MLVC

Verified on the Ascend 310P1 encoder board on 2026-09-29. `/dev/video0`
negotiates MJPG 1920×1080 at 30 FPS; its 1080p mode cannot deliver more than
30 distinct live frames per second. The requested **processing throughput** is
measured separately by preloading real V4L2 frames before the timed run.

## Selected path

`scripts/bash_run_encode_camera_1080p.sh` selects the isolated camera binary
and `configs/encoder_camera_1080p.toml`. It reads V4L2 MMAP MJPEG, decodes
JPEG with DVPP, pads NV12 to 1088 lines, feeds the static AIPP encoder OM, and
runs the existing MLVC encoder, entropy coding, and RTP output. The production
config uses the existing encoder board RTP target `192.168.5.7:39220` and has
camera RTSP output disabled. This task checked only the encoder board; RTP
reception and decoding were not measured.

```bash
cd /root/workplace/MLVC_c
bash scripts/bash_run_encode_camera_1080p.sh
```

The isolated binary and AIPP bundle are under
`/root/workplace/MLVC_c/camera_1080p_aipp/`. The original
`build/mlvc_encode` and `configs/encoder.toml` were not replaced. The AIPP OM
SHA-256 is `9bd673907a7c85e8f054823c74c9dbf56d34e4d4d63c1e42961a266b4d245a18`.
A complete local copy of the model bundle and build inputs is in the ignored
`artifacts/camera_1080p_aipp/` directory. The board also has a source archive
at `camera_1080p_aipp/source_20260929.tar.gz`.

The Git version is `release/encoder-1080p-aipp-20260929` with tag
`encoder-1080p-aipp-20260929`. The board's `camera_1080p_aipp/VERSION` records
the matching commit and checksums. Generated OM, ONNX, and local acceptance
logs remain outside Git; `configs/aipp_nv12_709_invert.cfg` records the AIPP
build settings. A fresh checkout needs the matching model bundle placed under
`camera_1080p_aipp/model_bundle/` before starting the camera launcher.

## CPU and AIPP comparison

Both paths use the same V4L2 1080p MJPEG capture and DVPP JPEG decoder. CPU
mode copies NV12 to the host, converts to BGR with OpenCV, then converts to the
existing FP16 Y/Cb/Cr network input. AIPP mode keeps NV12 on the device and
uses a separately compiled static AIPP encoder OM. Selecting the manifest in
the TOML selects the path; the CPU benchmark uses the original P1 manifest.

For each independent processing run, the program first captures 120 real
1080p MJPEG frames through V4L2, then repeatedly processes those frames. The
preload is outside the `encode_fps` timer. Each run encodes 400 frames with 20
warmup frames; the measured 380 frames include JPEGD, input conversion or
AIPP, MLVC execution, entropy coding, and RTP loopback sending. Repetition
isolates processing capacity from the camera's 30 FPS delivery limit.

| Path | Isolated run 1 | Isolated run 2 | Deployed binary |
| --- | ---: | ---: | ---: |
| CPU NV12→FP16 | 32.3440 | 32.3660 | 32.4908 FPS |
| NV12→AIPP | 34.4054 | 34.1326 | **34.3057 FPS** |

The deployed AIPP binary completed a separate live 180 frame test (20 warmup)
at 29.7905 FPS and `encode=ok`; this reflects the camera's 30 FPS limit.
After building the release branch from Git on the encoder board, the new binary
also completed a 60-frame native live smoke run with `encode=ok`. Its separate
400-frame, 20-warmup, 120-frame-preload AIPP run measured **34.2227 FPS** and
`encode=ok`.
The one-off benchmark configurations were removed from the board after the
results were recorded. The local acceptance directory retains their logs.

## Five-minute live run: input backpressure found

A bounded run with the production camera/RTP settings encoded 9,300 native
1080p frames from 12:53:03 to 12:58:38 (+08:00), with 30 warmup frames.
It completed with `encode=ok`, but measured **28.3033 FPS**. V4L2 sequence
gaps totaled **491** camera frames. Both input queues reached their fixed
capacity of three: the camera queue had 2,030 full waits totaling 8.525 s
(17.085 ms maximum), and the converted tensor queue had 2,112 full waits
totaling 84.892 s (52.523 ms maximum). At 7,200 captured frames the sequence
gap count was still two and camera queue full waits were zero; the backlog
appeared in the later part of the run. This is bounded queue backpressure and
camera frame loss, so the live path **did not sustain 30 FPS without drops**.

During the slowdown, `npu-smi` showed 101°C and health `Warning`. The error
information observed in the follow-up run was `lp tmonitor error`. Raising
fan duty from 30% to 80% for a second run did not fix the problem: by 8,100
captured frames, V4L2 gaps were 1,177 and camera queue full waits were 3,772;
the NPU reached 103°C. That run was stopped after more than five minutes to
avoid further high-temperature operation. Fan duty was restored to 30%; the
device returned to health `OK` after the process exited. These observations
strongly associate the throughput loss with high temperature, but they do
not prove a specific throttling mechanism. Cooling and NPU performance under
sustained load need investigation before treating this as a no-drop live
service.

## Six-minute synthetic-input control

The historical control used `input_synthetic = true` to generate 1080p FP16
frames in memory and feed the
original P1 MLVC encoder model, with entropy coding and RTP loopback retained.
There is no camera device or frame file input. The 11,000-frame run on the
same encoder board lasted from 13:24:45 to 13:30:45 (+08:00), with 30 warmup
frames, `encode=ok`, and **30.7940 FPS** overall.

The 900-frame interval rate started at 33.40 FPS, stayed near 33.3 through
6,300 frames, then fell to 32.19 at 8,100, 28.48 at 9,000, 25.81 at 9,900,
and **23.82 FPS** at 10,800. The synthetic input's ready queue peaked at
one frame; no camera input can accumulate or be dropped in this mode. The
producer did wait for a free slot 10,997 times (319.2 s total), which is
bounded backpressure from the encoder to an on-demand generator, not an
unbounded queue. The NPU was at 102°C but `OK` around 3 min 45 s and later
showed `Warning` with `lp tmonitor error` as interval throughput fell. This
control shows that long-run slowdown occurs even without the camera pipeline.
It supports a board/encoder sustained-load problem; it does not identify the
exact throttling point.

The one-off synthetic run configuration was removed from the board after
recording the result.

The AIPP input is an approximation of the original FP16 conversion. A
single-camera-frame encoder model comparison against the original P1 OM gave
mean absolute differences of 0.02668 on `feature`, 0.00413 on `z_raw`,
0.00235 on `y_raw_0`, and 0.00124 on `y_raw_1`. The bitstreams are therefore
not bit-identical. Decoder picture quality was outside this encoder-only
check. Raw logs are saved under
`acceptance/camera-encoder-1080p-aipp-20260929/`; the concise result is in
its `README.md`.

The earlier 720p/60 camera result was a temporary experiment and does not
satisfy the native 1080p requirement; its board deployment was removed.
