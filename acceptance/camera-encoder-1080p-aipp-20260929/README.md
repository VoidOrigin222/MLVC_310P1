# 2026-09-29 1080p camera encoder acceptance

Encoder board only. `/dev/video0` was verified as MJPG 1920×1080/30 by V4L2.
The deployed isolated AIPP build at
`/root/workplace/MLVC_c/camera_1080p_aipp/build/mlvc_encode` completed:

- Native live V4L2 input: 180 frames, 20 warmup, `encode_fps=29.7905`,
  `encode=ok`, RTP sent to the configured encoder target.
- Independent processing: 120 native V4L2 frames preloaded, 400 encoded,
  20 warmup, `encode_fps=34.3057`, `encode=ok`, RTP loopback.
- Same deployed binary with the original FP16 model and CPU input conversion:
  120 preloaded frames, 400 encoded, 20 warmup, `encode_fps=32.4908`,
  `encode=ok`, RTP loopback.

The preload happens before timed processing. `encode_fps` counts completed
encoded frames over wall time, including DVPP JPEG decode, input preparation,
MLVC, entropy coding, and RTP send. It does not claim more than 30 distinct
live 1080p frames per second from the sensor. AIPP was selected because it
was faster in all three measured independent runs. CPU and AIPP use different
preprocessing numerics; the AIPP OM is not bit-identical to the original model.
No decoder-side quality or frame receipt was tested.

The source committed on `release/encoder-1080p-aipp-20260929` was rebuilt on
the encoder board. Its AIPP binary completed a new 60-frame native camera
smoke run with `encode=ok` and a 400-frame independent processing run with
120 preloaded frames, 20 warmup frames, **34.2227 FPS**, and `encode=ok`.

## Sustained live test

With the production 1080p camera/RTP settings, 9,300 frames ran for 5 min
35 s and completed with `encode=ok`. The measured 9,270 frames averaged
**28.3033 FPS**. The input did accumulate: V4L2 sequence gaps reached 491;
the three-slot camera and tensor queues both filled. The camera queue waited
2,030 times for 8.525 s total, and the tensor queue waited 2,112 times for
84.892 s total. At 7,200 frames, gaps were still two and the camera queue
had not waited; accumulation appeared late. NPU temperature reached 101°C
with health `Warning` during the slowdown.

A second run with fan duty changed from 30% to 80% still reached 103°C and
`Warning`; by 8,100 frames, gaps were 1,177 and camera queue full waits
were 3,772. It was stopped after 5 min 35 s due to temperature. Fan duty
was restored to 30%, and the NPU returned to health `OK`. The short
independent processing result above does not establish sustained no-drop
live operation.

## No-camera control

The same isolated binary was run with `input_synthetic = true`, the original
1080p FP16 P1 encoder model, and RTP loopback. It generated 11,000 frames in
memory without opening V4L2 or reading frame files. From 13:24:45 to
13:30:45 (+08:00), all frames encoded successfully; the 10,970 measured
frames averaged **30.7940 FPS**. Per 900-frame throughput was about 33.3 FPS
initially but fell to **23.8151 FPS** by frame 10,800. The input ready queue
never exceeded one frame. Its producer waited for free slots as the encoder
slowed, but there was no unbounded input buildup. NPU health changed to
`Warning` during the late slowdown. This isolates a sustained encoder/board
performance decline from camera capture, JPEGD and AIPP.

Raw logs in this directory are local acceptance evidence and intentionally
ignored by Git. The code, configurations and explanation are in
`docs/camera_encoder_1080p.md`.
