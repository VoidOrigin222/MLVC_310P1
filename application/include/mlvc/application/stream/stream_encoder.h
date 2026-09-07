#ifndef MLVC_APPLICATION_STREAM_STREAM_ENCODER_H_
#define MLVC_APPLICATION_STREAM_STREAM_ENCODER_H_

#include <filesystem>
#include <cstddef>
#include <string>

namespace mlvc::codec {

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
  std::filesystem::path output_bitstream_path;
  std::filesystem::path profile_output_path;
  std::string execution_profile;
  bool enable_stage_fusion = false;
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
  int min_qp = 0;
  int max_qp = 63;
  int forced_ltr_recovery_frame = -1;
  int forced_ltr_reference_frame = -1;
  int device = 0;
  std::string udp_host;
  int udp_port = 0;
  PipelineOptions pipeline;
};

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_STREAM_ENCODER_H
