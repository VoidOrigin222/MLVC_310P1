# Translation motion backends

The warp consumes two signed feature-cell offsets per P frame. All backends use the same C++
area-weighted median of `-motion_x / motion_scale` and `-motion_y / motion_scale`, followed by
`copysign(floor(abs(displacement) / 8 + 0.5 - 1e-9), displacement)`. Half-cell ties round toward
zero; offsets outside the signed 8-bit range fail. GOP starts and forced I frames use `(0, 0)`.
The geometry overhead is **2 bytes/P frame** and must be included in rate calculations.

## Backend selection and comparison

| Backend | Input and proxy policy | Intended use |
| --- | --- | --- |
| `replay` | Python motion JSON converted to frame-indexed CSV | Reproduce a previously validated experiment and compare the MLVC path independently of proxy encoding |
| `libx264` | Host FP16 YUV444 converted through the existing MLVC 8-bit NV12 conversion; medium, CRF18, B=0, refs=1, fixed GOP, scene-cut disabled, zerolatency | Synchronous file/camera operation and CPU reference for the live pipeline |
| `dvpp` | Host FP16 converted to NV12 and uploaded, or an existing device NV12 surface; persistent MPI H.264 VENC, Normal P, VBR 8 Mbps by default | Hardware proxy encoding, followed by software H.264 decoding to export motion vectors |
| `export_motion_translation` tool | Original video pixels and bit depth; medium, CRF18, B=0, refs=1, fixed GOP, scene-cut disabled, default x264 lookahead | Offline comparison with the original Python proxy command |

The original Python script is `/root/workspace/mlvc/video/run_sample_inference.py` on
`dcvc_lqy`. It transcodes the original video with FFmpeg `libx264`, `medium`, CRF18, `-bf 0`,
`-refs 1`, `-g 96`, `-keyint_min 96`, `-sc_threshold 0`. It does not request zerolatency or
an 8-bit pixel format. The source `614lab.mp4` is HEVC **YUV420P10LE**, 1920x1080.
Its software toolchain is FFmpeg 6.1.1, while this deployment pins FFmpeg 4.4.8 and x264
stable source revision `b35605ace3` (libx264 ABI 165, all bit depths enabled).

The live FP16/NV12 proxy and the offline source-video proxy have different input pixels and
latency settings. DVPP VBR also makes different coding decisions from libx264 CRF18. Their
motion vectors are not claimed to match the original Python vectors exactly. Validate each
backend with its own motion trace and the resulting MLVC reconstruction/bitrate. When exact
replay of the Python experiment is required, use the saved Python motion commands.

## Why the DVPP proxy keeps its channel alive

The existing output encoder contains a workaround that rebuilds the MPI VENC channel for
every submitted image. This loses inter-frame references and is unsuitable for motion
extraction. New `DvppH264EncoderConfig` fields are opt-in:

- `persistent_channel=false` preserves the original output behavior.
- `single_reference=false` preserves the original reference configuration.
- `channel=0` preserves the original output channel.

`DvppTranslationEstimator` enables persistence and the reference policy on channel 1. It uses
increasing microsecond `pts = frame_index * 1000000 / fps` and `time_ref = 2 * frame_index`.
The legacy restart mode retains its previous zero metadata. On the tested 310P1/CANN 9.1
environment, the new mode successfully submits consecutive I/P frames without the channel
restart workaround. This observation establishes the working combination, rather than
attributing the old failure to a single field in isolation.

The reference configuration is `base=1`, `enhance=0`, **`pred_en=HI_TRUE`**. Huawei documents
this as the 1x reference mode. `HI_FALSE` makes base-layer frames reference the GOP IDR,
which would estimate displacement against the GOP anchor instead of the previous frame.
See the official [hi_mpi_venc_set_ref_param reference](https://www.hiascend.com/doc_center/source/en/canncommercial/700/inferapplicationdev/aclcppdevg/aclcppdevg_03_0655.html).
The implementation checks the configuration returned by `hi_mpi_venc_get_ref_param`.

The DVPP SPS may declare `max_num_ref_frames=2`; this is decoded-picture-buffer capacity,
and does not by itself prove that a P slice uses two references. `H264ReferencePolicy`
parses every Annex B access unit and verifies:

- Progressive IDR/I and P pictures, with consecutive `frame_num` values and no B slices.
- Each P slice has exactly **one active list-0 reference**, including any slice override.
- No reference-list reordering, long-term IDR reference, or adaptive reference marking.
- Every picture is retained as a reference (`nal_ref_idc != 0`).

Together these conditions make default list 0 select the immediately preceding short-term
reference. Unsupported syntax fails explicitly. A real 640x384 synthetic sequence translated
16 pixels right and 8 pixels up passed on the encoder: each P produced `(kx, ky)=(2,-1)`,
and GOP starts produced zero. Both regular GOP resets and a forced I frame are tested by
the validation tools. The software decoder is used because the DVPP VDEC interface does not
provide FFmpeg `AVMotionVector` side data; the hardware path does not claim direct VDEC MV export.

This backend therefore implements **DVPP hardware proxy encoding with CPU motion-vector
extraction**. It is not a complete hardware encode/decode pipeline. Running an additional
VDEC pass would not supply the required motion-vector metadata, so the estimator avoids it.

## Isolated FFmpeg build

Both boards use `/root/workplace/grifcc/deps/ffmpeg`; the existing MLVC directories and
system libraries are not replaced. Source packages are under
`/root/workplace/grifcc/deps/src`. Build x264 first:

```bash
cd /root/workplace/grifcc/deps/src/x264
./configure --prefix=/root/workplace/grifcc/deps/ffmpeg --enable-shared --disable-cli
make -j4
make install
```

The FFmpeg source is the official
[FFmpeg 4.4.8 release](https://ffmpeg.org/releases/ffmpeg-4.4.8.tar.xz). The deployed configure
command is:

```bash
cd /root/workplace/grifcc/deps/src/ffmpeg-4.4.8
export PKG_CONFIG_PATH=/root/workplace/grifcc/deps/ffmpeg/lib/pkgconfig
./configure \
  --prefix=/root/workplace/grifcc/deps/ffmpeg \
  --enable-shared --disable-static --disable-debug --disable-doc \
  --disable-autodetect --disable-everything --enable-gpl --enable-libx264 \
  --enable-encoder=libx264,rawvideo \
  --enable-decoder=h264,hevc,rawvideo,mpeg2video \
  --enable-parser=h264,hevc \
  --enable-muxer=h264,rtsp,rtp,mpegts,rawvideo,mp4 \
  --enable-demuxer=h264,hevc,rtsp,rtp,mpegts,mov,rawvideo \
  --enable-protocol=file,tcp,udp,rtp,pipe \
  --enable-filter=scale,format,null --enable-network \
  --disable-avdevice --disable-postproc \
  --extra-ldflags=-Wl,-rpath,/root/workplace/grifcc/deps/ffmpeg/lib
make -j4
make install
```

**`CONFIG_MPEGVIDEO` must be enabled.** In FFmpeg 4.4.8,
`libavcodec/h264dec.c` only calls `ff_print_debug_info2` to produce motion-vector side data
when this component is enabled. A minimal H.264-only `--disable-everything` build decodes
valid P frames but exports no motion vectors. Enabling `mpeg2video` pulls this dependency
in; verify `#define CONFIG_MPEGVIDEO 1` in generated `config.h`, and run the real proxy test.
The HEVC decoder and MOV demuxer are required for the original 10-bit MP4 input.

Point CMake at the private include/library paths and retain their runtime library path.
Run commands with the private library directory first in `LD_LIBRARY_PATH` when using
standalone probes; the system libavcodec has the same major SONAME.

## Validation tools

```bash
ctest --test-dir build --output-on-failure -R 'motion_translation|motion_proxy|h264_reference_policy'
LD_LIBRARY_PATH=/root/workplace/grifcc/deps/ffmpeg/lib:/usr/local/Ascend/cann/lib64 \
  build/check_dvpp_motion_proxy 110 96 640 384 2
LD_LIBRARY_PATH=/root/workplace/grifcc/deps/ffmpeg/lib:/usr/local/Ascend/cann/lib64 \
  build/export_motion_translation testdata/motion/614lab.mp4 537 motion.json motion.shifts 96
```

`check_motion_translation` covers vector sign, quarter-pixel scales, area weights, separate
axis medians, median ties, all half-cell boundaries, overflow, and invalid samples.
`check_h264_reference_policy` includes negative cases for extra active references, reordered
lists, adaptive marking, frame gaps, B/non-reference slices, malformed or truncated syntax,
and unsupported slice partitions. `check_motion_proxy` performs actual libx264 encode/decode
on a known translated image, including stride padding, periodic GOP resets, forced IDR and
out-of-order input rejection. The hardware probe is run explicitly rather than by ordinary
CTest, because it owns a real VENC channel.

The offline exporter writes JSON for all frames and a binary sidecar containing two bytes
for each non-GOP-start frame. It keeps x264 default lookahead, flushes both source and proxy
at EOF, and rejects source formats that libx264 cannot encode directly. Library versions,
threading and bit depth can affect coding decisions; compare exported shifts with the saved
Python JSON before describing an export as equivalent.

The completed 537-frame offline comparison on `614lab.mp4` produced 1062 sidecar bytes.
Compared with the saved Python JSON, 15 frames had different quantized `(kx, ky)`, 203 had
different `(tx, ty)`, and 529 had different motion-vector counts. This comparison establishes
that the pinned deployment proxy is not an exact substitute for the saved Python trace.

## Resource ownership and error behavior

The proxy encoder, decoder, frame, packet, host staging buffer and device upload buffer use
RAII. The upload buffer is reused when dimensions are unchanged. Every VENC packet is copied
to owned host bytes and the MPI stream is released before software decoding. Any policy or
codec failure stops the motion operation; there is no fallback to zero vectors or per-frame
I encoding. A valid P picture containing only intra blocks has no displacement evidence
and uses zero motion. Configuration/decode errors do not take this path.

The estimator requires consecutive frame indices starting at zero and is owned by one
encoding stream. It uses an existing ACL context, whose lifetime must exceed the estimator's.
MPI VENC system initialization follows the existing encoder's ownership model. Multiple MPI
encoder instances in one process would need coordinated `hi_mpi_sys_init/exit` ownership;
the deployed encoder motion proxy and decoder output encoder run in separate processes.
