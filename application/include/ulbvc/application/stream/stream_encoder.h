#ifndef ULBVC_APPLICATION_STREAM_STREAM_ENCODER_H_
#define ULBVC_APPLICATION_STREAM_STREAM_ENCODER_H_

#include <ulbvc/detection/yolo_detector.h>
#include <ulbvc/io/video_io.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace ulbvc::codec {

struct PipelineOptions {
  std::size_t stream_workers = 1;
  std::size_t queue_capacity = 3;
  std::size_t frame_buffer_slots = 3;
  std::size_t entropy_workers = 1;
  std::size_t graph_packet_capacity = 2;
};

struct EncodeStreamOptions {
  std::filesystem::path manifest_path;
  std::filesystem::path input_frame_dir;
  std::filesystem::path input_video_path;
  std::filesystem::path input_camera_device;
  bool input_synthetic = false;
  std::optional<ulbvc::io::CameraCaptureOptions> camera_options;
  std::filesystem::path output_bitstream_path;
  std::filesystem::path profile_output_path;
  std::string execution_profile;
  bool enable_stage_fusion = false;
  bool translation_warp = false;
  std::string motion_backend = "libx264";
  int motion_prefetch_frames = 0;
  std::string motion_x264_preset = "medium";
  int motion_x264_threads = 1;
  bool motion_skip_loop_filter = false;
  bool motion_camera_nv12 = false;
  // Optional full-frame CSV: frame_index,kx,ky, including zero GOP starts.
  std::filesystem::path motion_shifts_file;
  double fps = 30.0;
  int qp = 18;
  int frame_num = -1;
  int profile_warmup_frames = 0;
  int gop = 96;
  int reset_interval = 32;
  int ltr_start_idx = 8;
  int ltr_period = 64;
  int ltr_qp_shift = 8;
  double target_bitrate_bps = 0.0;
  std::string rate_control_mode = "fixed_q";
  uint64_t adaptive_max_bitrate_bps = 2000000;
  uint32_t adaptive_feedback_timeout_ms = 3000;
  uint32_t rtcp_feedback_interval_ms = 500;
  int min_qp = 0;
  int max_qp = 63;
  int forced_ltr_recovery_frame = -1;
  int forced_ltr_reference_frame = -1;
  int device = 0;
  std::string output_transport_host;
  std::string output_transport_mode = "udp";
  int output_transport_port = 0;
  uint8_t output_transport_payload_type = 96;
  uint64_t output_transport_pacing_rate_bps = 0;
  std::size_t output_transport_max_burst_bytes = 4096;
  std::size_t output_transport_max_queue_bytes = 4u * 1024u * 1024u;
  uint64_t output_transport_max_queue_delay_ms = 200;
  // Optional UDP JSON telemetry for the UI.  The counters describe the
  // encoded ULBVC RTP stream, not the decoder's H.264 preview stream.
  std::string ulbvc_stats_host;
  int ulbvc_stats_port = 0;
  int ulbvc_stats_interval_frames = 30;
  PipelineOptions pipeline;
  ulbvc::detection::Options detection;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_APPLICATION_STREAM_STREAM_ENCODER_H
